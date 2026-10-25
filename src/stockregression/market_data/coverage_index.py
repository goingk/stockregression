from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from ..price_validation import anomalies_from_attrs
from .models import MarketDataError
from .package_layout import PackageLayout
from .timeframes import normalize_timeframe


COVERAGE_ARCHIVE_SCHEMA_VERSION = 1


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _day_text(value: Any) -> str:
    return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")


def _frame_hash(frame: pd.DataFrame | None) -> str | None:
    if frame is None or frame.empty:
        return None
    fields = [
        field for field in (
            "timestamp", "open", "high", "low", "close", "volume", "turnover",
            "preclose", "tradestatus", "is_st", "source", "source_verified",
            "is_synthetic", "is_auction_placeholder", "opening_auction_volume",
            "adj_factor", "metadata",
        ) if field in frame
    ]
    canonical = frame[fields].sort_values("timestamp").reset_index(drop=True)
    return hashlib.sha256(
        canonical.to_json(
            orient="records", date_format="iso", double_precision=15,
            force_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _group_timestamps(values: Iterable[Any], interval_minutes: int) -> list[dict[str, Any]]:
    stamps = pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce")).dropna().sort_values()
    if not len(stamps):
        return []
    groups: list[dict[str, Any]] = []
    start = previous = stamps[0]
    count = 1
    for value in stamps[1:]:
        if value - previous == pd.Timedelta(minutes=interval_minutes):
            previous, count = value, count + 1
            continue
        groups.append({"start": start.isoformat(), "end": previous.isoformat(), "missing": count})
        start = previous = value
        count = 1
    groups.append({"start": start.isoformat(), "end": previous.isoformat(), "missing": count})
    return groups


def _month_key(day: Any) -> str:
    return pd.Timestamp(day).strftime("%Y-%m")


class CoverageArchive:
    """Persistent per-day coverage and gap records stored inside the package.

    SQLite mirrors these records for range queries.  The package JSON files
    are the recoverable source of truth, so copying a package to another
    catalog does not force another full market scan.
    """

    def __init__(self, repository):
        self.repository = repository
        self.package: PackageLayout = repository.package
        # Cache immutable payloads across helpers in one repository. A file
        # state change invalidates the entry even if its reference is unchanged.
        self._month_cache = repository.__dict__.setdefault("_quality_file_cache", {})

    def _read_payload(self, relative: str, digest: str) -> dict[str, Any] | None:
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            return None
        path = self.package.absolute(relative)
        try:
            state = path.stat()
            key = (str(self.package.root), relative, digest,
                   state.st_size, state.st_mtime_ns, state.st_ctime_ns)
            if key in self._month_cache:
                return self._month_cache[key]
            raw = path.read_bytes()
            if not digest or hashlib.sha256(raw).hexdigest() != digest:
                return None
            value = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(value, dict):
            return None
        if len(self._month_cache) >= 512:
            self._month_cache.pop(next(iter(self._month_cache)))
        self._month_cache[key] = value
        return value

    def _remember_written_payload(self, relative, digest, raw):
        """Reuse bytes just written atomically; later file changes invalidate it."""
        state = self.package.absolute(relative).stat()
        key = (str(self.package.root), relative, digest,
               state.st_size, state.st_mtime_ns, state.st_ctime_ns)
        if len(self._month_cache) >= 512:
            self._month_cache.pop(next(iter(self._month_cache)))
        self._month_cache[key] = json.loads(raw.decode("utf-8"))

    def _key(self, symbol: str, timeframe: str, adjustment: str) -> str:
        return PackageLayout.dataset_key(symbol, timeframe, adjustment)

    def head_reference(
        self, symbol: str, timeframe: str, adjustment: str = "none",
    ) -> dict[str, Any] | None:
        key = self._key(symbol, normalize_timeframe(timeframe), adjustment)
        head = self.package.manifest().get("datasets", {}).get(key) or {}
        reference = head.get("quality_archive")
        return dict(reference) if isinstance(reference, dict) else None

    def load_head(
        self, symbol: str, timeframe: str, adjustment: str = "none",
    ) -> dict[str, Any] | None:
        reference = self.head_reference(symbol, timeframe, adjustment)
        if not reference:
            return None
        relative = str(reference.get("path") or "")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            return None
        expected = str(reference.get("sha256") or "")
        value = self._read_payload(relative, expected)
        if value is None:
            return None
        package = self.package.manifest()
        if (
            int(value.get("schema_version") or 0) != COVERAGE_ARCHIVE_SCHEMA_VERSION
            or int(reference.get("schema_version") or 0) != COVERAGE_ARCHIVE_SCHEMA_VERSION
            or value.get("package_id") != package.get("package_id")
            or value.get("generation") != package.get("generation")
            or value.get("symbol") != symbol
            or normalize_timeframe(value.get("timeframe")) != normalize_timeframe(timeframe)
            or value.get("adjustment", "none") != adjustment
            or int(value.get("data_version") or 0) != int(reference.get("data_version") or 0)
            or int(value.get("revision") or 0) != int(reference.get("revision") or 0)
            or int(value.get("rules_version") or 0) != int(reference.get("rules_version") or 0)
        ):
            return None
        value = dict(value)
        value["_reference"] = dict(reference)
        value["_pending_updates"] = list(reference.get("pending_updates") or [])
        return value

    def _month_payload(self, item: dict[str, Any]) -> dict[str, Any] | None:
        relative = str(item.get("path") or "")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            return None
        value = self._read_payload(relative, str(item.get("sha256") or ""))
        if value is None:
            return None
        package = self.package.manifest()
        if (
            int(value.get("schema_version") or 0) != COVERAGE_ARCHIVE_SCHEMA_VERSION
            or value.get("package_id") != package.get("package_id")
            or value.get("generation") != package.get("generation")
            or value.get("month") != item.get("month")
            or not value.get("symbol")
            or not value.get("timeframe")
        ):
            return None
        payload = dict(value)
        return payload

    def trusted_head(
        self, symbol: str, timeframe: str, adjustment: str = "none",
        *, start=None, end=None, verify_files: bool = True,
    ) -> dict[str, Any] | None:
        """Require all requested months to be usable for a coverage claim."""
        head = self.validation_head(
            symbol, timeframe, adjustment, start=start, end=end,
            verify_files=verify_files,
        )
        return head if head and not head.get("_invalid_months") else None

    def validation_head(
        self, symbol: str, timeframe: str, adjustment: str = "none",
        *, start=None, end=None, verify_files: bool = True,
    ) -> dict[str, Any] | None:
        """Keep good months available when an individual month is damaged."""
        head = self.load_head(symbol, timeframe, adjustment)
        if not head or not verify_files:
            return head
        invalid: set[str] = set()
        requested_start = pd.Timestamp(start).to_period("M") if start is not None else None
        requested_end = pd.Timestamp(end).to_period("M") if end is not None else None
        for item in head.get("months") or []:
            try:
                month = pd.Period(str(item.get("month")), freq="M")
            except (TypeError, ValueError):
                return None
            if requested_start is not None and month < requested_start:
                continue
            if requested_end is not None and month > requested_end:
                continue
            payload = self._month_payload(item)
            if payload is None:
                invalid.add(str(month))
                continue
            if (
                payload.get("symbol") != head.get("symbol")
                or normalize_timeframe(payload.get("timeframe")) != normalize_timeframe(head.get("timeframe"))
                or payload.get("adjustment", "none") != head.get("adjustment", "none")
            ):
                invalid.add(str(month))
                continue
            days = payload.get("days")
            if not isinstance(days, list) or len(days) != int(item.get("days") or 0):
                invalid.add(str(month))
                continue
            if days and (
                str(days[0].get("trade_date"))[:10] != str(item.get("first_day"))[:10]
                or str(days[-1].get("trade_date"))[:10] != str(item.get("last_day"))[:10]
            ):
                invalid.add(str(month))
        head["_invalid_months"] = sorted(invalid)
        return head

    @staticmethod
    def pending_dates(head: dict[str, Any] | None, start=None, end=None) -> set[pd.Timestamp]:
        """Dates invalidated by a publication that has not been revalidated."""
        if not head:
            return set()
        lower = pd.Timestamp(start).normalize() if start is not None else None
        upper = pd.Timestamp(end).normalize() if end is not None else None
        output: set[pd.Timestamp] = set()
        for update in head.get("_pending_updates") or []:
            for value in (update or {}).get("dates") or []:
                try:
                    day = pd.Timestamp(value).normalize()
                except (TypeError, ValueError):
                    continue
                if lower is not None and day < lower:
                    continue
                if upper is not None and day > upper:
                    continue
                output.add(day)
        return output

    def current_data_version(self, symbol: str, timeframe: str, adjustment: str = "none") -> int | None:
        dataset = self._dataset(symbol, timeframe, adjustment)
        return int(dataset["current_version"]) if dataset else None

    def _dataset(self, symbol: str, timeframe: str, adjustment: str):
        # dataset() also projects historical quality state. Publication only
        # needs the small data head and must not scan the full day mirror.
        with self.repository._connect() as db:
            row = self.repository._dataset_row(db, symbol, normalize_timeframe(timeframe), adjustment)
        return dict(row) if row else None

    def is_current(
        self, symbol: str, timeframe: str, adjustment: str = "none",
    ) -> bool:
        head = self.load_head(symbol, timeframe, adjustment)
        if not head:
            return False
        return int(head.get("data_version") or 0) == int(
            self.current_data_version(symbol, timeframe, adjustment) or 0
        )

    def rows(
        self, symbol: str, timeframe: str, adjustment: str = "none",
        *, start=None, end=None,
    ) -> list[dict[str, Any]]:
        timeframe = normalize_timeframe(timeframe)
        if getattr(self.repository, "_read_only", False):
            reference = self.head_reference(symbol, timeframe, adjustment)
            if reference:
                head = self.load_head(symbol, timeframe, adjustment)
                if head is None:
                    raise MarketDataError(
                        f"只读覆盖档案 head 无法验证：{symbol} {timeframe}"
                    )
                lower = pd.Timestamp(start).normalize() if start is not None else None
                upper = pd.Timestamp(end).normalize() if end is not None else None
                selected_months = {
                    str(item.get("month")) for item in head.get("months") or []
                    if (lower is None or pd.Period(str(item.get("month")), freq="M") >= lower.to_period("M"))
                    and (upper is None or pd.Period(str(item.get("month")), freq="M") <= upper.to_period("M"))
                }
                month_rows = self._published_month_rows(head, selected_months)
                return sorted(
                    [
                        row for month in month_rows.values() for row in month
                        if (lower is None or pd.Timestamp(row["trade_date"]).normalize() >= lower)
                        and (upper is None or pd.Timestamp(row["trade_date"]).normalize() <= upper)
                    ],
                    key=lambda row: str(row["trade_date"]),
                )
            # Packages without an archive head can still have legacy mirror
            # rows. Reading that mirror is safe; read-only repositories never
            # repair or claim it as current.
            return self._mirror_rows(symbol, timeframe, adjustment, start=start, end=end)

        self._ensure_mirror(symbol, timeframe, adjustment)
        return self._mirror_rows(symbol, timeframe, adjustment, start=start, end=end)

    def _mirror_rows(
        self, symbol: str, timeframe: str, adjustment: str = "none",
        *, start=None, end=None,
    ) -> list[dict[str, Any]]:
        clauses = ["symbol=?", "timeframe=?", "adjustment=?"]
        params: list[Any] = [symbol, normalize_timeframe(timeframe), adjustment]
        if start is not None:
            clauses.append("trade_date>=?")
            params.append(str(pd.Timestamp(start).date()))
        if end is not None:
            clauses.append("trade_date<=?")
            params.append(str(pd.Timestamp(end).date()))
        with self.repository._connect() as db:
            records = db.execute(
                "SELECT * FROM parquet_coverage_days WHERE " + " AND ".join(clauses)
                + " ORDER BY trade_date",
                params,
            ).fetchall()
        result = []
        for row in records:
            item = dict(row)
            for field, default in (("gaps", []), ("pending", []), ("report", {})):
                try:
                    item[field] = json.loads(item.get(field) or json.dumps(default))
                except (TypeError, ValueError):
                    item[field] = default
            item["is_trading_day"] = bool(item.get("is_trading_day"))
            item["scanned"] = bool(item.get("scanned"))
            result.append(item)
        return result

    def _ensure_mirror(self, symbol: str, timeframe: str, adjustment: str) -> None:
        if getattr(self.repository, "_read_only", False):
            # Read-only download contexts read the package head directly via
            # rows(); they must not acquire the package publication lock or
            # attempt to repair a lagging SQLite projection.
            return
        reference = self.head_reference(symbol, timeframe, adjustment)
        if not reference:
            return
        with self.repository._connect() as db:
            indexed = db.execute(
                "SELECT path,sha256,revision FROM parquet_coverage_archives "
                "WHERE symbol=? AND timeframe=? AND adjustment=?",
                (symbol, normalize_timeframe(timeframe), adjustment),
            ).fetchone()
        if indexed and all(str(indexed[key]) == str(reference.get(key))
                           for key in ("path", "sha256", "revision")):
            return
        # A publisher may have committed the package but not its SQLite mirror.
        # Serialize with it before readers use any day result.
        with self.package.transaction_guard():
            # Restoring every coverage archive can be very expensive and is
            # unnecessary here: this reference identifies one authoritative
            # archive head. Restore only this dataset from its immutable files.
            current = self.head_reference(symbol, timeframe, adjustment)
            if current and not (indexed and all(
                str(indexed[key]) == str(current.get(key))
                for key in ("path", "sha256", "revision")
            )):
                head = self.load_head(symbol, timeframe, adjustment)
                if head is None:
                    raise MarketDataError(
                        f"覆盖档案 head 无法读取，拒绝从旧 SQLite 镜像继续：{symbol} {timeframe}"
                    )
                self._restore_mirror_from_head(head)

    def _published_month_rows(
        self, head: dict[str, Any], months: set[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Read verified month payloads from the published head.

        SQLite is a rebuildable mirror. In particular, it can lag a head that
        was published immediately before a process or mirror transaction
        failed, so it must not be used as the merge base for that head.
        """
        symbol = str(head.get("symbol") or "")
        timeframe = normalize_timeframe(str(head.get("timeframe") or "1day"))
        adjustment = str(head.get("adjustment") or "none")
        selected = None if months is None else {str(value)[:7] for value in months}
        result: dict[str, list[dict[str, Any]]] = {}
        pending = {
            _day_text(day): str(update.get("reason") or "行情修订")
            for update in head.get("_pending_updates") or []
            for day in update.get("dates") or []
        }
        for item in head.get("months") or []:
            month = str(item.get("month") or "")
            if selected is not None and month not in selected:
                continue
            payload = self._month_payload(item)
            if (
                payload is None
                or payload.get("symbol") != symbol
                or normalize_timeframe(str(payload.get("timeframe") or "1day")) != timeframe
                or payload.get("adjustment", "none") != adjustment
                or not isinstance(payload.get("days"), list)
                or len(payload["days"]) != int(item.get("days") or 0)
            ):
                if not self._mirror_matches_head(head):
                    raise MarketDataError(
                        f"覆盖档案月份不可验证，SQLite 镜像也未确认属于当前 head：{month or item.get('path')}"
                    )
                period = pd.Period(month, freq="M")
                # A matching archive reference proves this mirror was written
                # for this exact head. It is safe to use it to repair only the
                # damaged month, while all valid months still come from the
                # immutable published head.
                rows = self._mirror_rows(
                    symbol, timeframe, adjustment,
                    start=period.start_time, end=period.end_time,
                )
                result[month] = [{
                    **row, "dirty_reason": pending.get(
                        _day_text(row["trade_date"]), row.get("dirty_reason"),
                    ),
                } for row in rows]
                continue
            rows = []
            for row in payload["days"]:
                if not isinstance(row, dict) or _month_key(row.get("trade_date")) != month:
                    raise MarketDataError(f"覆盖档案月份日期无效：{month}")
                day = _day_text(row["trade_date"])
                rows.append({
                    **row,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "adjustment": adjustment,
                    "dirty_reason": pending.get(day, row.get("dirty_reason")),
                })
            result[month] = rows
        if selected is not None:
            for month in selected:
                result.setdefault(month, [])
        return result

    def _mirror_matches_head(self, head: dict[str, Any]) -> bool:
        reference = dict(head.get("_reference") or {})
        if not reference:
            return False
        return self._mirror_matches_reference(
            reference, str(head.get("symbol") or ""),
            normalize_timeframe(str(head.get("timeframe") or "1day")),
            str(head.get("adjustment") or "none"),
        )

    def _mirror_matches_reference(
        self, reference: dict[str, Any], symbol: str,
        timeframe: str, adjustment: str,
    ) -> bool:
        with self.repository._connect() as db:
            indexed = db.execute(
                "SELECT path,sha256,revision FROM parquet_coverage_archives "
                "WHERE symbol=? AND timeframe=? AND adjustment=?",
                (symbol, normalize_timeframe(timeframe), adjustment),
            ).fetchone()
        return bool(indexed) and all(
            str(indexed[key]) == str(reference.get(key))
            for key in ("path", "sha256", "revision")
        )

    def _mirror_dirty_reasons(
        self, symbol: str, timeframe: str, adjustment: str,
        months: set[str] | None,
    ) -> dict[str, str]:
        clauses = ["symbol=?", "timeframe=?", "adjustment=?", "dirty_reason IS NOT NULL"]
        arguments: list[Any] = [symbol, normalize_timeframe(timeframe), adjustment]
        if months is not None:
            selected = sorted(months)
            if not selected:
                return {}
            clauses.append("substr(trade_date,1,7) IN (" + ",".join("?" for _ in selected) + ")")
            arguments.extend(selected)
        with self.repository._connect() as db:
            rows = db.execute(
                "SELECT trade_date,dirty_reason FROM parquet_coverage_days WHERE "
                + " AND ".join(clauses),
                arguments,
            ).fetchall()
        return {_day_text(row["trade_date"]): str(row["dirty_reason"]) for row in rows}

    def _restore_mirror_from_head(self, head: dict[str, Any]) -> int:
        """Restore just this archive's SQLite projection in one transaction."""
        symbol = str(head["symbol"])
        timeframe = normalize_timeframe(str(head["timeframe"]))
        adjustment = str(head.get("adjustment") or "none")
        month_rows = self._published_month_rows(head)
        entries = [row for month in sorted(month_rows) for row in month_rows[month]]
        reference = dict(head.get("_reference") or {})
        if not reference:
            raise MarketDataError("覆盖档案 head 缺少已发布引用")
        with self.repository._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "DELETE FROM parquet_coverage_days WHERE symbol=? AND timeframe=? AND adjustment=?",
                (symbol, timeframe, adjustment),
            )
            db.execute(
                "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? AND adjustment=?",
                (symbol, timeframe, adjustment),
            )
            restored = self.save_rows(entries, db=db)
            self._write_validations([row for row in entries if not row.get("dirty_reason")], db)
            self.record_archive(head, reference, db=db)
        return restored

    def save_rows(self, rows: Sequence[dict[str, Any]], *, db=None) -> int:
        """Write a mirror, only after publication or while restoring a head."""
        if not rows:
            return 0
        values = []
        for row in rows:
            symbol = str(row["symbol"])
            timeframe = normalize_timeframe(str(row["timeframe"]))
            adjustment = str(row.get("adjustment") or "none")
            trade_date = _day_text(row["trade_date"])
            gaps = row.get("gaps") or []
            pending = row.get("pending") or []
            values.append((
                symbol, timeframe, adjustment, trade_date,
                int(bool(row.get("is_trading_day", True))),
                int(bool(row.get("scanned", True))), str(row.get("status") or "unknown"),
                int(row.get("row_count") or 0), row.get("first_time"), row.get("last_time"),
                int(row.get("synthetic_rows") or 0), int(row.get("auction_placeholders") or 0),
                int(row.get("blocked_rows") or 0), int(row.get("gap_count") or len(gaps)),
                _json(gaps), _json(pending), row.get("source_hash"),
                row.get("dependency_hash"), int(row.get("rules_version") or 0),
                row.get("dirty_reason"), _json(row.get("report") or {}),
                row.get("checked_at"),
            ))
        with (nullcontext(db) if db is not None else self.repository._connect()) as connection:
            connection.executemany(
                """INSERT OR REPLACE INTO parquet_coverage_days(
                   symbol,timeframe,adjustment,trade_date,is_trading_day,scanned,
                   status,row_count,first_time,last_time,synthetic_rows,
                   auction_placeholders,blocked_rows,gap_count,gaps,pending,
                   source_hash,dependency_hash,rules_version,dirty_reason,report,checked_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
        return len(values)

    def mark_dirty(
        self, symbol: str, timeframe: str, dates: Iterable[Any], *, reason: str,
        adjustment: str = "none",
    ) -> int:
        values = sorted({_day_text(day) for day in dates if day is not None})
        if not values:
            return 0
        changed = 0
        with self.repository._connect() as db:
            for offset in range(0, len(values), 400):
                chunk = values[offset:offset + 400]
                placeholders = ",".join("?" for _ in chunk)
                before = db.total_changes
                db.execute(
                    "UPDATE parquet_coverage_days SET dirty_reason=? "
                    "WHERE symbol=? AND timeframe=? AND adjustment=? AND trade_date IN ("
                    + placeholders + ")",
                    (reason, symbol, normalize_timeframe(timeframe), adjustment, *chunk),
                )
                changed += int(db.total_changes - before)
        return changed

    def delete_dates(
        self, symbol: str, timeframe: str, dates: Iterable[Any],
        adjustment: str = "none",
    ) -> int:
        values = sorted({_day_text(day) for day in dates if day is not None})
        if not values:
            return 0
        removed = 0
        with self.repository._connect() as db:
            for offset in range(0, len(values), 400):
                chunk = values[offset:offset + 400]
                placeholders = ",".join("?" for _ in chunk)
                before = db.total_changes
                db.execute(
                    "DELETE FROM parquet_coverage_days WHERE symbol=? AND timeframe=? "
                    "AND adjustment=? AND trade_date IN (" + placeholders + ")",
                    (symbol, normalize_timeframe(timeframe), adjustment, *chunk),
                )
                removed += int(db.total_changes - before)
        return removed

    def mark_all_dirty(
        self, symbol: str, timeframe: str, reason: str, adjustment: str = "none",
    ) -> int:
        with self.repository._connect() as db:
            before = db.total_changes
            db.execute(
                "UPDATE parquet_coverage_days SET dirty_reason=? "
                "WHERE symbol=? AND timeframe=? AND adjustment=?",
                (reason, symbol, normalize_timeframe(timeframe), adjustment),
            )
            return int(db.total_changes - before)

    def observed_range(
        self, symbol: str, timeframe: str, adjustment: str = "none",
    ) -> tuple[Any, Any]:
        with self.repository._connect() as db:
            row = db.execute(
                "SELECT MIN(trade_date) first_day,MAX(trade_date) last_day "
                "FROM parquet_coverage_days WHERE symbol=? AND timeframe=? "
                "AND adjustment=? AND scanned=1",
                (symbol, normalize_timeframe(timeframe), adjustment),
            ).fetchone()
        return (row["first_day"], row["last_day"]) if row else (None, None)

    def coverage_gaps(
        self, symbol: str, timeframe: str, start, end, adjustment: str = "none",
    ) -> dict[str, Any]:
        head = self.trusted_head(symbol, timeframe, adjustment, start=start, end=end)
        rows = self.rows(symbol, timeframe, adjustment, start=start, end=end)
        scanned = {pd.Timestamp(row["trade_date"]).normalize() for row in rows if row["scanned"]}
        requested = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
        unknown = [day for day in requested if day.normalize() not in scanned]
        calendar = self.repository.auxiliary(symbol, "calendar")
        calendar_days = self.repository._calendar_dates(
            (calendar or {}).get("payload") or []
        )
        if len(calendar_days):
            unknown = [day for day in unknown if day.normalize() in calendar_days]
        gaps: list[dict[str, Any]] = []
        pending = self.pending_dates(head, start, end)
        for row in rows:
            if row.get("dirty_reason") or pd.Timestamp(row["trade_date"]).normalize() in pending:
                continue
            for gap in row.get("gaps") or []:
                gaps.append({**gap, "trade_date": row["trade_date"]})
        return {
            "rows": rows, "unknown_dates": [day.date().isoformat() for day in unknown],
            "gaps": gaps, "archive": head, "archive_valid": head is not None,
        }

    def _archive_base(self, symbol: str, timeframe: str, adjustment: str) -> str:
        safe_symbol = "".join(c for c in str(symbol) if c.isalnum() or c in ".-_")
        safe_adjustment = "".join(c for c in str(adjustment) if c.isalnum() or c in "-_")
        return f".stockregression/quality/archives/{normalize_timeframe(timeframe)}/{safe_symbol}/{safe_adjustment}"

    def _summarize_month(self, dataset, timeframe, month, entries, calendar_dates) -> dict[str, Any]:
        from .repository import minute_grid_report_stale, validation_rules_version

        rules_version = validation_rules_version(timeframe)
        scanned = [row for row in entries if row.get("scanned")]
        clean = [row for row in scanned if not row.get("dirty_reason")
                 and not minute_grid_report_stale(timeframe, row)
                 and int(row.get("rules_version") or 0) >= rules_version]
        pending = {_day_text(row["trade_date"]): str(row.get("dirty_reason") or "待重新检查")
                   for row in entries if row not in clean
                   or row.get("status") not in {"passed", "failed", "suspended"}}
        period = pd.Period(month, freq="M")
        lower = max(pd.Timestamp(dataset["first_time"]).normalize(), period.start_time.normalize())
        last = pd.Timestamp(dataset["last_time"]).normalize()
        if len(calendar_dates):
            last = max(last, calendar_dates.max())
        upper = min(last, period.end_time.normalize())
        state = self.repository._validation_from_parts(
            dataset, entries, calendar_dates, start=lower, end=upper, include_days=False,
        )
        def boundary(field, operation):
            values = [row[field] for row in scanned if row.get(field)]
            return operation(values) if values else None
        return {
            "rules_version": rules_version,
            "scanned_days": sum(1 << (pd.Timestamp(row["trade_date"]).day - 1) for row in scanned),
            "first_time": boundary("first_time", min), "last_time": boundary("last_time", max),
            "scanned_start": boundary("trade_date", min), "scanned_end": boundary("trade_date", max),
            "gap_count": sum(int(row.get("gap_count") or 0) for row in clean),
            "failed_count": sum(row.get("status") == "failed" for row in clean),
            "pending_dates": pending,
            "verified_through": state.get("verified_through"),
            "complete": bool(state.get("verified_through") and not state.get("pending_count")
                             and not state.get("failed_count")),
        }

    def commit_rows(
        self, rows: Sequence[dict[str, Any]], *, validation_rows=(),
        expected_data_version: int | None = None, changed_months=None,
    ) -> dict[str, Any]:
        import time
        from ..performance import record_seconds

        started = time.perf_counter()
        try:
            result = self._commit_rows_impl(
                rows, validation_rows=validation_rows,
                expected_data_version=expected_data_version,
                changed_months=changed_months,
            )
            result["_publication_seconds"] = time.perf_counter() - started
            return result
        finally:
            record_seconds("ledger_publication", time.perf_counter() - started)

    def commit_rows_many(
        self, chunks: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Publish independent symbol ledgers with one package head switch.

        Each chunk accepts ``rows``, ``validation_rows``,
        ``expected_data_version`` and ``changed_months``. All immutable month
        pages and heads are written first, then all package references advance
        atomically, and finally every SQLite mirror is updated in one
        transaction. A conflict therefore publishes none of the batch.
        """
        import time
        from ..performance import record_seconds

        if not chunks:
            return []
        started = time.perf_counter()
        try:
            return self._commit_rows_many_impl(chunks)
        finally:
            record_seconds("ledger_publication", time.perf_counter() - started)

    def _commit_rows_impl(
        self, rows: Sequence[dict[str, Any]], *, validation_rows=(),
        expected_data_version: int | None = None, changed_months=None,
    ) -> dict[str, Any]:
        """Publish immutable evidence before making its SQLite mirror visible."""
        if not rows:
            return {"_stale_dates": [], "_accepted_dates": []}
        symbol, timeframe = str(rows[0]["symbol"]), str(rows[0]["timeframe"])
        adjustment = str(rows[0].get("adjustment") or "none")
        with self.package.transaction_guard():
            accepted, stale, evidence_reads = self._current_rows(rows, expected_data_version)
            if not accepted:
                return {**(self.load_head(symbol, timeframe, adjustment) or {}),
                        "_stale_dates": sorted(stale), "_accepted_dates": [],
                        "_evidence_read_calls": evidence_reads}
            current = self.current_data_version(symbol, timeframe, adjustment)
            months = {_month_key(row["trade_date"]) for row in accepted}
            head, reference = self._prepare_publication(
                symbol, timeframe, adjustment, expected_data_version=current,
                changed_months=months, _updates=accepted,
            )
            self.package.publish_quality_archive(
                symbol, timeframe, adjustment, reference,
                expected_data_version=int(head["data_version"]),
                expected_revision=int(head["revision"]) - 1,
            )
            # The package head is published before its SQLite projection. Make
            # the projection match every touched month from that exact head,
            # including previously published rows from a failed earlier
            # mirror transaction, in the same SQLite transaction as this
            # chunk's accepted evidence.
            published = self.load_head(symbol, timeframe, adjustment)
            if published is None:
                raise MarketDataError("覆盖档案 head 发布后无法重新读取")
            month_rows = self._published_month_rows(published, months)
            reference = dict(published.get("_reference") or {})
            accepted_dates = {_day_text(row["trade_date"]) for row in accepted}
            with self.repository._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                for month in sorted(months):
                    start = pd.Period(month, freq="M").start_time.strftime("%Y-%m-%d")
                    end = pd.Period(month, freq="M").end_time.strftime("%Y-%m-%d")
                    db.execute(
                        "DELETE FROM parquet_coverage_days WHERE symbol=? AND timeframe=? "
                        "AND adjustment=? AND trade_date>=? AND trade_date<=?",
                        (symbol, timeframe, adjustment, start, end),
                    )
                    db.execute(
                        "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? "
                        "AND adjustment=? AND trade_date>=? AND trade_date<=?",
                        (symbol, timeframe, adjustment, start, end),
                    )
                    month_entries = month_rows.get(month, [])
                    self.save_rows(month_entries, db=db)
                    self._write_validations(
                        [row for row in month_entries if not row.get("dirty_reason")], db,
                    )
                self.record_archive(head, reference, db=db)
            return {**head, "_stale_dates": sorted(stale),
                    "_accepted_dates": sorted(accepted_dates),
                    "_evidence_read_calls": evidence_reads}

    def _commit_rows_many_impl(self, chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = []
        with self.package.transaction_guard():
            seen: set[tuple[str, str, str]] = set()
            for chunk in chunks:
                rows = list(chunk.get("rows") or [])
                if not rows:
                    prepared.append({"rows": [], "accepted": [], "stale": set(),
                                     "evidence_reads": 0})
                    continue
                symbol = str(rows[0]["symbol"])
                timeframe = str(rows[0]["timeframe"])
                adjustment = str(rows[0].get("adjustment") or "none")
                key = (symbol, timeframe, adjustment)
                if key in seen:
                    raise ValueError(f"覆盖档案批次包含重复数据集：{key}")
                seen.add(key)
                accepted, stale, evidence_reads = self._current_rows(
                    rows, chunk.get("expected_data_version")
                )
                item = {
                    "rows": rows, "accepted": accepted, "stale": stale,
                    "evidence_reads": evidence_reads, "symbol": symbol,
                    "timeframe": timeframe, "adjustment": adjustment,
                }
                if accepted:
                    months = {_month_key(row["trade_date"]) for row in accepted}
                    current = self.current_data_version(symbol, timeframe, adjustment)
                    head, reference = self._prepare_publication(
                        symbol, timeframe, adjustment, expected_data_version=current,
                        changed_months=months, _updates=accepted,
                    )
                    item.update(
                        months=months, head=head, reference=reference,
                        month_rows=self._published_month_rows(head, months),
                    )
                prepared.append(item)

            publishable = [item for item in prepared if item.get("head")]
            if publishable:
                self.package.publish_quality_archives([{
                    "symbol": item["symbol"],
                    "timeframe": item["timeframe"],
                    "adjustment": item["adjustment"],
                    "reference": item["reference"],
                    "expected_data_version": int(item["head"]["data_version"]),
                    "expected_revision": int(item["head"]["revision"]) - 1,
                } for item in publishable])

                with self.repository._connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    for item in publishable:
                        symbol = item["symbol"]
                        timeframe = item["timeframe"]
                        adjustment = item["adjustment"]
                        for month in sorted(item["months"]):
                            start = pd.Period(month, freq="M").start_time.strftime("%Y-%m-%d")
                            end = pd.Period(month, freq="M").end_time.strftime("%Y-%m-%d")
                            db.execute(
                                "DELETE FROM parquet_coverage_days WHERE symbol=? AND timeframe=? "
                                "AND adjustment=? AND trade_date>=? AND trade_date<=?",
                                (symbol, timeframe, adjustment, start, end),
                            )
                            db.execute(
                                "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? "
                                "AND adjustment=? AND trade_date>=? AND trade_date<=?",
                                (symbol, timeframe, adjustment, start, end),
                            )
                            month_entries = item["month_rows"].get(month, [])
                            self.save_rows(month_entries, db=db)
                            self._write_validations(
                                [row for row in month_entries if not row.get("dirty_reason")], db,
                            )
                        self.record_archive(item["head"], item["reference"], db=db)

            output = []
            for item in prepared:
                head = item.get("head")
                if head is None:
                    rows = item["rows"]
                    if rows:
                        head = self.load_head(
                            item["symbol"], item["timeframe"], item["adjustment"],
                        ) or {}
                    else:
                        head = {}
                accepted_dates = {
                    _day_text(row["trade_date"]) for row in item["accepted"]
                }
                if item.get("reference"):
                    head = dict(head)
                    head["_reference"] = dict(item["reference"])
                    head["_pending_updates"] = list(
                        item["reference"].get("pending_updates") or []
                    )
                output.append({
                    **head, "_stale_dates": sorted(item["stale"]),
                    "_accepted_dates": sorted(accepted_dates),
                    "_evidence_read_calls": int(item["evidence_reads"]),
                })
            return output

    @staticmethod
    def _write_validations(rows, db) -> None:
        if not rows:
            return
        db.executemany(
            """INSERT OR REPLACE INTO parquet_day_validation(
               symbol,timeframe,adjustment,trade_date,status,row_count,
               source_hash,dependency_hash,rules_version,report,checked_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            [(row["symbol"], row["timeframe"], row.get("adjustment", "none"),
              _day_text(row["trade_date"]), row["status"], int(row.get("row_count") or 0),
              row.get("source_hash"), row.get("dependency_hash"), int(row.get("rules_version") or 0),
              _json(row.get("report") or {}), row.get("checked_at")) for row in rows],
        )

    def _current_rows(self, rows, expected_version):
        """Check worker evidence against current data while holding the package lock."""
        from .validation import _dependency_hash
        from .timeframes import MINUTE_TIMEFRAMES

        symbol, timeframe = rows[0]["symbol"], rows[0]["timeframe"]
        adjustment = rows[0].get("adjustment", "none")
        current_version = self.current_data_version(symbol, timeframe, adjustment)
        daily_version = self.current_data_version(symbol, "1day", adjustment)
        calendar = self.repository.auxiliary(symbol, "calendar")
        calendar_days = self.repository._calendar_dates((calendar or {}).get("payload") or [])
        actions = self.repository.auxiliary(symbol, "actions")
        manifests = {}
        frames = {}
        day_hashes: dict[tuple[str, str], str | None] = {}
        dependency_hashes: dict[str, str] = {}

        def shard_signature(period, version, day):
            if not version:
                return None
            key = (period, int(version))
            if key not in manifests:
                manifests[key] = self.package.load_dataset_manifest(symbol, period, adjustment, int(version))
            manifest = manifests[key]
            if manifest is None:
                return None
            storage = self.repository._storage_period(period, day)[0]
            return tuple(sorted((str(item.get("path")), str(item.get("sha256")))
                                for item in manifest.get("shards") or []
                                if str(item.get("partition_key") or item.get("period_key")) == storage))

        def current_hash(period, day):
            cache_key = (str(period), _day_text(day))
            if cache_key in day_hashes:
                return day_hashes[cache_key]
            storage, start, end = self.repository._storage_period(period, day)
            key = (period, storage)
            if key not in frames:
                frames[key] = self.repository.active_rows(
                    symbol, period, adjustment=adjustment, start=start,
                    end=pd.Timestamp(end) - pd.Timedelta(days=1),
                )
            frame = frames[key]
            if frame.empty or "timestamp" not in frame:
                day_hashes[cache_key] = None
                return None
            part = frame.loc[pd.to_datetime(frame.timestamp).dt.normalize() == day]
            day_hashes[cache_key] = _frame_hash(part)
            return day_hashes[cache_key]

        accepted, stale = [], set()
        for row in rows:
            day = pd.Timestamp(row["trade_date"]).normalize()
            day_text = _day_text(day)
            current_dependency = dependency_hashes.get(day_text)
            if current_dependency is None:
                current_dependency = _dependency_hash(
                    calendar_days, actions, day, timeframe=timeframe,
                )
                dependency_hashes[day_text] = current_dependency
            if row.get("dependency_hash") != current_dependency:
                stale.add(day_text)
                continue
            base = row.get("_data_version", expected_version)
            changed = base is None or int(base) != int(current_version or 0)
            report = row.get("report") or {}
            skipped = bool(report.get("non_trading") or report.get("no_trades")
                           or row.get("status") == "suspended")
            if changed and not skipped:
                old = shard_signature(timeframe, base, day)
                new = shard_signature(timeframe, current_version, day)
                if old is None or old != new:
                    if row.get("source_hash") != current_hash(timeframe, day):
                        stale.add(day_text)
                        continue
            if timeframe in MINUTE_TIMEFRAMES:
                daily_base = row.get("_daily_version")
                if daily_base is None or int(daily_base or 0) != int(daily_version or 0):
                    old = shard_signature("1day", daily_base, day)
                    new = shard_signature("1day", daily_version, day)
                    if old is None or old != new:
                        if row.get("_daily_source_hash") != current_hash("1day", day):
                            stale.add(day_text)
                            continue
            accepted.append(row)
        return accepted, stale, len(frames)

    def persist(
        self, symbol: str, timeframe: str, adjustment: str = "none", *,
        expected_data_version: int | None = None,
        changed_months: Iterable[str] | None = None,
        _updates: Sequence[dict[str, Any]] = (),
        _record_mirror: bool = True,
    ) -> dict[str, Any]:
        with self.package.transaction_guard():
            head, reference = self._prepare_publication(
                symbol, timeframe, adjustment, expected_data_version=expected_data_version,
                changed_months=changed_months, _updates=_updates,
            )
            self.package.publish_quality_archive(
                symbol, timeframe, adjustment, reference,
                expected_data_version=int(head["data_version"]),
                expected_revision=int(head["revision"]) - 1,
            )
            if _record_mirror:
                self.record_archive(head, reference)
            return head

    def _prepare_publication(
        self, symbol: str, timeframe: str, adjustment: str = "none", *,
        expected_data_version: int | None = None,
        changed_months: Iterable[str] | None = None,
        _updates: Sequence[dict[str, Any]] = (),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Write immutable month pages and a head without advancing its reference."""
        from .repository import validation_rules_version, utc_now

        timeframe = normalize_timeframe(timeframe)
        rules_version = validation_rules_version(timeframe)
        dataset = self._dataset(symbol, timeframe, adjustment)
        if not dataset:
            raise MarketDataError("没有可归档的本地行情")
        changed = None if changed_months is None else sorted({
            str(value)[:7] for value in changed_months if value
        })
        package = self.package.manifest()
        base = self._archive_base(symbol, timeframe, adjustment)
        reference = self.head_reference(symbol, timeframe, adjustment)
        previous_head = self.load_head(symbol, timeframe, adjustment)
        if reference and previous_head is None:
            if not self._mirror_matches_reference(
                reference, symbol, timeframe, adjustment,
            ):
                # A corrupt or missing published head must never be replaced
                # from a potentially stale SQLite mirror: that could make
                # published dates disappear permanently.
                raise MarketDataError(
                    f"已发布覆盖档案 head 无法验证，拒绝以 SQLite 镜像覆盖：{symbol} {timeframe}"
                )
            # The mirror explicitly records this exact published head. It can
            # therefore recover a damaged head, but rebuild all month payloads
            # from the mirror because the old head's unaffected month refs are
            # no longer readable.
            previous_head = None
            changed = None
        if previous_head is not None:
            grouped = defaultdict(
                list, self._published_month_rows(
                    previous_head, None if changed is None else set(changed),
                ),
            )
            if self._mirror_matches_head(previous_head):
                # Local invalidation marks are written to SQLite before their
                # replacement head. Carry only the dirty flag from a mirror
                # proven to represent this exact head; never use a lagging
                # mirror as the content merge base.
                mirror_months = set(grouped) if changed is None else set(changed)
                dirty_by_day = self._mirror_dirty_reasons(
                    symbol, timeframe, adjustment,
                    None if changed is None else mirror_months,
                )
                for month in mirror_months:
                    if any(day[:7] == month for day in dirty_by_day):
                        grouped[month] = [{
                            **row,
                            "dirty_reason": dirty_by_day.get(
                                _day_text(row["trade_date"]), row.get("dirty_reason"),
                            ),
                        } for row in grouped.get(month, [])]
        else:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            # No archive head exists yet. The SQLite projection is the only
            # seed available for the first publication.
            for row in self.rows(symbol, timeframe, adjustment):
                grouped[_month_key(row["trade_date"])].append(row)
            if changed is not None:
                grouped = defaultdict(list, {
                    month: grouped.get(month, []) for month in changed
                })
        updates_by_month: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in _updates:
            month = _month_key(row["trade_date"])
            # Never serialize transient worker evidence as package metadata.
            entry = {key: value for key, value in row.items() if not key.startswith("_")}
            updates_by_month[month][_day_text(row["trade_date"])] = entry
        for month, updates in updates_by_month.items():
            entries = {_day_text(item["trade_date"]): item for item in grouped.get(month, [])}
            entries.update(updates)
            grouped[month] = [entries[key] for key in sorted(entries)]
        calendar = self.repository.auxiliary(symbol, "calendar")
        calendar_dates = self.repository._calendar_dates((calendar or {}).get("payload") or [])
        months: list[dict[str, Any]] = []
        for month, entries in sorted(grouped.items()):
            if not entries:
                continue
            payload = {
                "schema_version": COVERAGE_ARCHIVE_SCHEMA_VERSION,
                "package_id": package["package_id"],
                "generation": package["generation"],
                "symbol": symbol,
                "timeframe": timeframe,
                "adjustment": adjustment,
                "month": month,
                "days": entries,
            }
            raw = _json(payload).encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            relative = f"{base}/months/{month}-{digest[:20]}.json"
            path = self.package.absolute(relative)
            valid_existing = bool(
                path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest
            )
            if not valid_existing:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".writing")
                temporary.write_bytes(raw)
                temporary.replace(path)
                self._remember_written_payload(relative, digest, raw)
            months.append({
                "month": month, "path": relative, "sha256": digest,
                "size": len(raw),
                "days": len(entries),
                "first_day": entries[0]["trade_date"], "last_day": entries[-1]["trade_date"],
                "summary": self._summarize_month(dataset, timeframe, month, entries, calendar_dates),
            })
        if changed is not None:
            previous_months = {
                str(item.get("month")): dict(item)
                for item in ((previous_head or {}).get("months") or [])
            }
            replaced = set(changed)
            months = [
                item for month, item in previous_months.items()
                if month not in replaced
            ] + months
            months.sort(key=lambda item: str(item["month"]))
        # Summaries are carried in the small head directory. Unchanged months
        # need no day-row query or payload decode when publishing one month.
        for item in months:
            if (not isinstance(item.get("summary"), dict)
                    or "scanned_days" not in item["summary"]
                    or int(item["summary"].get("rules_version") or 0) != rules_version):
                month = str(item["month"])
                payload = self._month_payload(item)
                entries = (payload or {}).get("days")
                if entries is None:
                    period = pd.Period(month, freq="M")
                    entries = self.rows(symbol, timeframe, adjustment,
                                        start=period.start_time, end=period.end_time)
                item["summary"] = self._summarize_month(dataset, timeframe, month, entries, calendar_dates)
        summaries = [item["summary"] for item in months]
        unresolved: dict[str, str] = {}
        for update in (previous_head or {}).get("_pending_updates") or []:
            for value in update.get("dates") or []:
                unresolved[_day_text(value)] = str(update.get("reason") or "待重新检查")
        for summary in summaries:
            unresolved.update(summary.get("pending_dates") or {})
        month_summaries = {str(item["month"]): item["summary"] for item in months}
        for day in calendar_dates:
            if day < pd.Timestamp(dataset["first_time"]).normalize():
                continue
            summary = month_summaries.get(str(day.to_period("M"))) or {}
            if not int(summary.get("scanned_days") or 0) & (1 << (day.day - 1)):
                unresolved.setdefault(_day_text(day), "尚未扫描")
        for row in _updates:
            day = _day_text(row["trade_date"])
            if row.get("scanned") and not row.get("dirty_reason"):
                unresolved.pop(day, None)
        def minimum(field):
            values = [item[field] for item in summaries if item.get(field)]
            return min(values) if values else None
        def maximum(field):
            values = [item[field] for item in summaries if item.get(field)]
            return max(values) if values else None
        aggregate = {
            "first_time": minimum("first_time"), "last_time": maximum("last_time"),
            "scanned_start": minimum("scanned_start"), "scanned_end": maximum("scanned_end"),
        }
        gaps = sum(int(item.get("gap_count") or 0) for item in summaries)
        failed = sum(int(item.get("failed_count") or 0) for item in summaries)
        pending = len(unresolved)
        verified_through = None
        expected_months = {str(day.to_period("M")) for day in calendar_dates
                           if pd.Timestamp(dataset["first_time"]).normalize() <= day}
        expected_months.update(str(item["month"]) for item in months)
        by_month = {str(item["month"]): item["summary"] for item in months}
        for month in sorted(expected_months):
            summary = by_month.get(month)
            if summary is None or not summary.get("verified_through"):
                break
            if any(day[:7] == month and day not in (summary.get("pending_dates") or {})
                   for day in unresolved):
                # Recompute only the affected month's prefix; pending dates
                # override older clean summaries carried across publications.
                period = pd.Period(month, freq="M")
                entries = self.rows(symbol, timeframe, adjustment,
                                    start=period.start_time, end=period.end_time)
                overlay = {_day_text(row["trade_date"]): row for row in _updates}
                entries = [overlay.get(_day_text(row["trade_date"]), row) for row in entries]
                entries = [{**row, "dirty_reason": unresolved.get(_day_text(row["trade_date"]),
                           row.get("dirty_reason"))} for row in entries]
                summary = self._summarize_month(dataset, timeframe, month, entries, calendar_dates)
            if summary.get("verified_through"):
                verified_through = summary["verified_through"]
            if not summary.get("complete"):
                break
        verification = {"verified_through": verified_through}
        revision = int((reference or {}).get("revision") or 0) + 1
        data_version = int(dataset["current_version"])
        if (
            expected_data_version is not None
            and data_version != int(expected_data_version)
        ):
            raise MarketDataError(
                f"行情版本已从 {expected_data_version} 变为 {data_version}；"
                "本次档案结果已丢弃，请按新版本重新校验"
            )
        head = {
            "schema_version": COVERAGE_ARCHIVE_SCHEMA_VERSION,
            "package_id": package["package_id"],
            "generation": package["generation"],
            "symbol": symbol, "timeframe": timeframe, "adjustment": adjustment,
            "data_version": data_version, "revision": revision,
            "rules_version": int(rules_version),
            "data_start": dataset.get("first_time"),
            "data_end": dataset.get("last_time"),
            "first_time": aggregate["first_time"],
            "last_time": aggregate["last_time"],
            "scanned_start": aggregate["scanned_start"],
            "scanned_end": aggregate["scanned_end"],
            "gap_count": gaps, "pending_count": pending, "failed_count": failed,
            "verified_through": verification.get("verified_through"),
            "months": months,
        }
        raw = _json(head).encode("utf-8")
        head_digest = hashlib.sha256(raw).hexdigest()
        relative = f"{base}/head-v{data_version}-q{revision}-{head_digest[:12]}.json"
        path = self.package.absolute(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".writing")
        if path.is_file():
            temporary.unlink(missing_ok=True)
            if hashlib.sha256(path.read_bytes()).hexdigest() != head_digest:
                temporary.write_bytes(raw)
                temporary.replace(path)
        else:
            temporary.write_bytes(raw)
            temporary.replace(path)
            self._remember_written_payload(relative, head_digest, raw)
        pending_groups: dict[str, list[str]] = defaultdict(list)
        for day, reason in unresolved.items():
            pending_groups[reason].append(day)
        published_reference = {
            "schema_version": COVERAGE_ARCHIVE_SCHEMA_VERSION,
            "path": relative, "sha256": head_digest,
            "data_version": data_version, "revision": revision,
            "rules_version": int(rules_version),
            "gap_count": gaps, "pending_count": pending, "failed_count": failed,
            "pending_updates": [
                {"dates": sorted(days), "reason": reason}
                for reason, days in sorted(pending_groups.items())
            ],
        }
        return head, published_reference

    def record_archive(self, head: dict[str, Any], reference: dict[str, Any], *, db=None) -> None:
        from .repository import utc_now

        with (nullcontext(db) if db is not None else self.repository._connect()) as connection:
            connection.execute(
                """INSERT OR REPLACE INTO parquet_coverage_archives(
                   symbol,timeframe,adjustment,data_version,revision,path,sha256,
                   gap_count,pending_count,failed_count,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    head["symbol"], head["timeframe"], head.get("adjustment", "none"),
                    int(head["data_version"]), int(head["revision"]),
                    str(reference["path"]), str(reference["sha256"]),
                    int(head.get("gap_count") or 0), int(head.get("pending_count") or 0),
                    int(head.get("failed_count") or 0), utc_now(),
                ),
            )

    def restore(self, reference: dict[str, Any]) -> int:
        """Restore SQLite coverage rows from an already verified head payload."""
        payload = dict(reference.get("payload") or {})
        if not payload:
            return 0
        entries: list[dict[str, Any]] = []
        for item in payload.get("months") or []:
            month = self._month_payload(item)
            if month is None:
                continue
            days = month.get("days")
            if (
                not isinstance(days, list)
                or month.get("symbol") != payload.get("symbol")
                or normalize_timeframe(month.get("timeframe"))
                != normalize_timeframe(payload.get("timeframe"))
                or month.get("adjustment", "none") != payload.get("adjustment", "none")
                or len(days) != int(item.get("days") or 0)
            ):
                return 0
            entries.extend(days)
        with self.repository._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "DELETE FROM parquet_coverage_days WHERE symbol=? AND timeframe=? AND adjustment=?",
                (payload["symbol"], payload["timeframe"], payload["adjustment"]),
            )
            pending = {str(day)[:10]: str(update.get("reason") or "行情修订")
                       for update in payload.get("_pending_updates") or []
                       for day in update.get("dates") or []}
            entries = [{**row, "dirty_reason": pending.get(str(row["trade_date"])[:10],
                       row.get("dirty_reason"))} for row in entries]
            restored = self.save_rows(entries, db=db)
            db.execute(
                "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? AND adjustment=?",
                (payload["symbol"], payload["timeframe"], payload.get("adjustment", "none")),
            )
            self._write_validations([row for row in entries if not row.get("dirty_reason")], db)
        return restored

    @staticmethod
    def gap_windows(gaps: Sequence[dict[str, Any]], interval_minutes: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for gap in gaps:
            try:
                start = pd.Timestamp(gap["start"])
                end = pd.Timestamp(gap["end"])
            except (KeyError, TypeError, ValueError):
                continue
            windows.append((start - pd.Timedelta(days=10), end + pd.Timedelta(minutes=interval_minutes)))
        return windows


def anomalies_for_day(frame: pd.DataFrame, day: Any) -> pd.DataFrame:
    anomalies = anomalies_from_attrs(frame)
    if anomalies.empty:
        return anomalies
    stamps = pd.to_datetime(anomalies.get("timestamp"), errors="coerce")
    target = pd.Timestamp(day).normalize()
    return anomalies.loc[stamps.dt.normalize() == target].copy()


def coverage_frame_hash(frame: pd.DataFrame | None) -> str | None:
    return _frame_hash(frame)


def coverage_gap_groups(values: Iterable[Any], interval_minutes: int) -> list[dict[str, Any]]:
    return _group_timestamps(values, interval_minutes)


def coverage_timestamp_groups(values: Iterable[Any], interval_minutes: int) -> list[dict[str, Any]]:
    """Compact, lossless run map of the timestamps actually present on a day."""
    return [
        {"start": item["start"], "end": item["end"], "count": item["missing"]}
        for item in _group_timestamps(values, interval_minutes)
    ]
