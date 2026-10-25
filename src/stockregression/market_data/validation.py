from __future__ import annotations

import hashlib
import time
from typing import Any, Iterable

import pandas as pd

from ..price_validation import anomalies_in_range
from .coverage_index import (
    CoverageArchive, coverage_frame_hash, coverage_gap_groups, coverage_timestamp_groups,
)
from .models import MarketDataError
from .quality import validate_and_fill
from .repository import VALIDATION_RULES_VERSION, validation_rules_version, utc_now
from .timeframes import MINUTE_TIMEFRAMES, normalize_timeframe, timeframe_spec


_DAILY_COMMIT_BATCH_SIZE = 512


def _frame_day_records(
    frame: pd.DataFrame,
) -> dict[pd.Timestamp, list[tuple[int, dict[str, Any]]]]:
    """Index daily source rows without materializing one DataFrame per day."""
    if frame.empty or "timestamp" not in frame:
        return {}
    dates = pd.to_datetime(frame.timestamp, errors="coerce").dt.normalize()
    records = frame.to_dict(orient="records")
    result: dict[pd.Timestamp, list[tuple[int, dict[str, Any]]]] = {}
    for position, (day, row) in enumerate(zip(dates, records)):
        if pd.notna(day):
            result.setdefault(pd.Timestamp(day).normalize(), []).append((position, row))
    return result


def _daily_single_row_hashes(frame: pd.DataFrame) -> dict[pd.Timestamp, str]:
    """Hash ordinary one-row days from one byte-compatible yearly JSON pass."""
    if frame.empty or "timestamp" not in frame:
        return {}
    fields = [field for field in (
        "timestamp", "open", "high", "low", "close", "volume", "turnover",
        "preclose", "tradestatus", "is_st", "source", "source_verified",
        "is_synthetic", "is_auction_placeholder", "opening_auction_volume",
        "adj_factor", "metadata",
    ) if field in frame]
    if not fields:
        return {}
    ordered = frame[fields].sort_values("timestamp")
    serialized = ordered.to_json(
        orient="records", date_format="iso", double_precision=15, force_ascii=False,
    )
    body = serialized[1:-1]
    records = []
    start = None
    depth = 0
    quoted = escaped = False
    for position, character in enumerate(body):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                records.append(body[start:position + 1])
    if len(records) != len(ordered):
        return {}
    days = pd.to_datetime(ordered.timestamp, errors="coerce").dt.normalize()
    counts = days.value_counts()
    result = {}
    for day, record in zip(days, records):
        if pd.notna(day) and int(counts.loc[day]) == 1:
            result[pd.Timestamp(day).normalize()] = hashlib.sha256(
                f"[{record}]".encode("utf-8")
            ).hexdigest()
    return result


def _daily_preloaded_frames(repository, preloaded_months, adjustment):
    """Merge overlapping preloaded month views once into their storage years."""
    grouped: dict[str, list[tuple[int, pd.DataFrame]]] = {}
    source_attrs = {}
    for source_id, frame in enumerate((preloaded_months or {}).values()):
        if frame.empty or "timestamp" not in frame:
            continue
        source_attrs[source_id] = dict(frame.attrs)
        stamps = pd.to_datetime(frame.timestamp)
        for year, positions in frame.groupby(stamps.dt.year, sort=False).indices.items():
            key = str(repository._storage_period("1day", pd.Timestamp(year=int(year), month=1, day=1))[0])
            piece = frame.iloc[positions].copy()
            piece.attrs = {}
            grouped.setdefault(key, []).append((source_id, piece))
    result = {}
    for key, sources in grouped.items():
        pieces = []
        for source_id, piece in sources:
            piece = piece.copy()
            piece["__daily_preload_source"] = source_id
            pieces.append(piece)
        merged = pd.concat(pieces, ignore_index=True, sort=False)
        # Resolve overlaps between month views, while preserving true duplicate
        # timestamps that already existed inside the selected source view.
        latest_source = merged.groupby("timestamp", sort=False)["__daily_preload_source"].transform("max")
        merged = merged.loc[merged["__daily_preload_source"] == latest_source]
        merged = merged.drop(columns="__daily_preload_source").reset_index(drop=True)
        attrs = {}
        for source_id, _piece in sources:
            for name, values in source_attrs[source_id].items():
                if isinstance(values, list):
                    attrs.setdefault(name, []).extend(values)
                else:
                    attrs[name] = values
        merged.attrs.update(attrs)
        for name in ("blocked_timestamps", "price_anomaly_rows"):
            if name in merged.attrs:
                values = merged.attrs[name]
                if name == "blocked_timestamps":
                    merged.attrs[name] = list(dict.fromkeys(values))
                else:
                    merged.attrs[name] = list({str(row): row for row in values}.values())
        result[key] = merged
    return result


def _daily_fast_report(part: pd.DataFrame, day: pd.Timestamp, pending: list[str]) -> tuple[dict, dict]:
    """Produce the daily branch of validate_and_fill without re-entering its generic pipeline."""
    if len(part) == 1:
        stamp = pd.Timestamp(part.timestamp.iloc[0])
        if stamp.normalize() == day:
            blocked_values = part.attrs.get("blocked_timestamps", [])
            blocked = (
                pd.DatetimeIndex(pd.to_datetime(blocked_values, errors="coerce").dropna())
                if blocked_values else pd.DatetimeIndex([])
            )
            pending = list(pending)
            if blocked.size:
                pending.append(f"存在 {blocked.size} 条无效价格记录，相关时间点保持缺口且不自动填充")
            report = {
                "expected_rows": 1, "observed_rows": 1, "expected_days": 1,
                "observed_days": 1, "calendar_available": True,
                "missing_rows": 0, "synthetic_rows": 0, "gap_count": 0,
                "gaps": [], "unfillable": 0,
                "status": "pending_validation" if pending else "complete",
                "blocked_price_rows": int(len(blocked)),
            }
            anomaly_rows = part.attrs.get("price_anomaly_rows") or []
            if anomaly_rows:
                anomalies = pd.DataFrame(anomaly_rows)
                if "timestamp" in anomalies:
                    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"], errors="coerce")
                columns = [column for column in ("timestamp", "date", "source", "quality_reason") if column in anomalies]
                report["price_anomalies"] = anomalies.head(200)[columns].astype(str).to_dict("records")
            report["pending_validation"] = list(dict.fromkeys(pending))
            report["first_time"] = stamp.isoformat()
            report["last_time"] = stamp.isoformat()
            report["coverage_status"] = "partial" if pending else "complete"
            return report, {
                "missing_timestamps": [], "missing_days": [],
                "blocked_timestamps": [str(value) for value in blocked],
                "auction_placeholders": 0,
            }
    result = part.copy().sort_values("timestamp", kind="stable").reset_index(drop=True)
    blocked = pd.DatetimeIndex(pd.to_datetime(
        result.attrs.get("blocked_timestamps", []), errors="coerce",
    ).dropna())
    if blocked.size:
        pending = list(pending)
        pending.append(f"存在 {blocked.size} 条无效价格记录，相关时间点保持缺口且不自动填充")
    observed_labels = pd.DatetimeIndex(result.timestamp.dt.normalize())
    missing_labels = pd.DatetimeIndex([day]).difference(observed_labels)
    gaps = [{"start": str(value.date()), "end": str(value.date()), "missing": 1} for value in missing_labels]
    status = "complete" if not len(missing_labels) and not pending else (
        "pending_validation" if not len(missing_labels) else "partial"
    )
    report = {
        "expected_rows": 1, "observed_rows": len(observed_labels),
        "expected_days": 1, "observed_days": int(observed_labels.nunique()),
        "calendar_available": True, "missing_rows": len(missing_labels),
        "synthetic_rows": 0, "gap_count": len(gaps), "gaps": gaps[:500],
        "unfillable": len(missing_labels), "status": status,
        "blocked_price_rows": int(len(blocked)),
    }
    anomalies = pd.DataFrame(result.attrs.get("price_anomaly_rows") or [])
    if not anomalies.empty:
        if "timestamp" in anomalies:
            anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"], errors="coerce")
        columns = [column for column in ("timestamp", "date", "source", "quality_reason") if column in anomalies]
        report["price_anomalies"] = anomalies.head(200)[columns].astype(str).to_dict("records")
    report["pending_validation"] = list(dict.fromkeys(pending))
    report["first_time"] = result.timestamp.min().isoformat() if len(result) else None
    report["last_time"] = result.timestamp.max().isoformat() if len(result) else None
    details = {
        "missing_timestamps": [],
        "missing_days": [str(value.date()) for value in missing_labels],
        "blocked_timestamps": [str(value) for value in blocked],
        "auction_placeholders": 0,
    }
    report["coverage_status"] = "complete" if status == "complete" else "partial"
    return report, details


def _daily_fast_single_report(
    stamp_value, pending: list[str], blocked_values: list[str],
) -> tuple[dict, dict]:
    """Build the ordinary one-row daily report without another DataFrame pass."""
    stamp = pd.Timestamp(stamp_value)
    blocked = pd.DatetimeIndex(pd.to_datetime(blocked_values, errors="coerce").dropna())
    pending = list(pending)
    if blocked.size:
        pending.append(f"存在 {blocked.size} 条无效价格记录，相关时间点保持缺口且不自动填充")
    unique_pending = list(dict.fromkeys(pending))
    report = {
        "expected_rows": 1, "observed_rows": 1, "expected_days": 1,
        "observed_days": 1, "calendar_available": True,
        "missing_rows": 0, "synthetic_rows": 0, "gap_count": 0,
        "gaps": [], "unfillable": 0,
        "status": "pending_validation" if unique_pending else "complete",
        "blocked_price_rows": int(len(blocked)),
        "pending_validation": unique_pending,
        "first_time": stamp.isoformat(), "last_time": stamp.isoformat(),
        "coverage_status": "partial" if unique_pending else "complete",
    }
    return report, {
        "missing_timestamps": [], "missing_days": [],
        "blocked_timestamps": [str(value) for value in blocked],
        "auction_placeholders": 0,
    }


def _day_hash(frame: pd.DataFrame) -> str | None:
    return coverage_frame_hash(frame)


def _calendar_hash(calendar_days: pd.DatetimeIndex, day: pd.Timestamp) -> str:
    # A tail extension must not invalidate earlier sessions in the same month.
    # The day's own trading status is the calendar fact used by its checks.
    target = pd.Timestamp(day).normalize()
    if not len(calendar_days) or target < calendar_days.min() or target > calendar_days.max():
        fact = "unknown"
    else:
        fact = "trading" if target in calendar_days else "non-trading"
    return hashlib.sha256(fact.encode()).hexdigest()


def _actions_dependency(actions: dict | None, day: pd.Timestamp) -> list[dict[str, Any]]:
    payload = list((actions or {}).get("payload") or [])
    if not payload:
        return []
    try:
        exact = []
        for row in payload:
            value = row.get("ex_date")
            if value in {None, ""}:
                continue
            stamp = pd.Timestamp(value).normalize()
            if stamp <= pd.Timestamp(day).normalize() <= stamp + pd.Timedelta(days=10):
                exact.append(row)
        return exact
    except (TypeError, ValueError):
        return payload


def _dependency_hash(
    calendar_days: pd.DatetimeIndex, actions: dict | None, day: pd.Timestamp,
    daily_row: pd.Series | None = None,
    *, timeframe: str = "1day", rules_version: int | None = None,
) -> str:
    payload = {
        "rules": rules_version if rules_version is not None else validation_rules_version(timeframe),
        "calendar": _calendar_hash(calendar_days, day),
        "actions": _actions_dependency(actions, day),
    }
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def _target_months(values: Iterable[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    months = sorted({pd.Timestamp(value).to_period("M") for value in values})
    return [
        (period.start_time.normalize(), period.end_time.normalize())
        for period in months
    ]


def _read_daily_dependencies(
    repository, symbol: str, targets: list[pd.Timestamp], adjustment: str,
    *, read_metrics: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    if not targets:
        return pd.DataFrame()
    pieces = []
    periods = {}
    for day in targets:
        key, start, end = repository._storage_period("1day", day)
        periods[str(key)] = (pd.Timestamp(start), pd.Timestamp(end))
    for period, (start, end) in periods.items():
        if read_metrics is not None:
            read_metrics["calls"] = int(read_metrics.get("calls") or 0) + 1
            read_metrics.setdefault("periods", set()).add(period)
        part = repository.active_rows(
            symbol, "1day", adjustment=adjustment,
            start=start, end=end - pd.Timedelta(days=1),
            **({"use_cache": False} if not use_cache else {}),
        )
        if not part.empty:
            pieces.append(part)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).drop_duplicates("timestamp", keep="last")


def _expected_days(
    service, symbol: str, timeframe: str, lower, upper, adjustment: str,
    *, prefetched_dates: pd.DatetimeIndex | None = None, archive=None,
) -> tuple[pd.DatetimeIndex, bool, pd.DatetimeIndex]:
    repository = service.repository
    archive = archive or CoverageArchive(repository)
    rows = archive.rows(
        symbol, timeframe, adjustment=adjustment, start=lower, end=upper
    )
    calendar = repository.auxiliary(symbol, "calendar")
    calendar_days = repository._calendar_dates(
        (calendar or {}).get("payload") or []
    )
    if rows or archive.load_head(symbol, timeframe, adjustment):
        prefetched = (
            pd.DatetimeIndex(prefetched_dates).normalize().unique()
            if prefetched_dates is not None else pd.DatetimeIndex([])
        )
        observed = pd.DatetimeIndex([pd.Timestamp(row["trade_date"]) for row in rows])
        values = calendar_days.append(observed).append(prefetched).normalize().unique().sort_values()
        return (
            values[(values >= pd.Timestamp(lower)) & (values <= pd.Timestamp(upper))],
            bool(len(calendar_days)), calendar_days,
        )
    prefetched = (
        pd.DatetimeIndex(prefetched_dates).normalize().unique().sort_values()
        if prefetched_dates is not None else pd.DatetimeIndex([])
    )
    if len(prefetched):
        values = calendar_days.append(prefetched).normalize().unique().sort_values()
        return (
            values[
                (values >= pd.Timestamp(lower))
                & (values <= pd.Timestamp(upper))
            ],
            bool(len(calendar_days)), calendar_days,
        )
    if prefetched_dates is not None:
        return (
            calendar_days[
                (calendar_days >= pd.Timestamp(lower))
                & (calendar_days <= pd.Timestamp(upper))
            ],
            bool(len(calendar_days)), calendar_days,
        )
    values = repository.trading_days(symbol, lower, upper)
    if len(calendar_days):
        values = calendar_days.append(values).normalize().unique().sort_values()
        values = values[
            (values >= pd.Timestamp(lower)) & (values <= pd.Timestamp(upper))
        ]
    return values, bool(len(calendar_days)), calendar_days


def validation_targets(
    service, symbol: str, timeframe: str, *, mode: str = "incremental",
    start=None, end=None, dates: Iterable[Any] | None = None,
    force_dates: bool = False, adjustment: str = "none",
    prefetched_dates: pd.DatetimeIndex | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return only days whose coverage record is absent or explicitly dirty."""
    repository = service.repository
    timeframe = normalize_timeframe(timeframe)
    adjustment = str(adjustment or "none")
    dataset = repository.dataset(symbol, timeframe, adjustment)
    if not dataset:
        raise MarketDataError("没有可校验的本地行情")
    if selection is not None:
        return selection
    lower = pd.Timestamp(start or dataset["first_time"]).normalize()
    upper = pd.Timestamp(end or dataset["last_time"]).normalize()
    if end is None:
        saved_calendar = repository.auxiliary(symbol, "calendar")
        calendar_dates = repository._calendar_dates(
            (saved_calendar or {}).get("payload") or []
        )
        if len(calendar_dates):
            upper = max(upper, calendar_dates.max().normalize())
    if upper < lower:
        raise MarketDataError("校验开始日期不能晚于结束日期")
    archive = CoverageArchive(repository)
    calendar_days, calendar_available, trading_days = _expected_days(
        service, symbol, timeframe, lower, upper, adjustment,
        prefetched_dates=prefetched_dates,
        archive=archive,
    )
    requested_dates = None
    if dates is not None:
        requested_dates = {
            pd.Timestamp(value).normalize() for value in dates
            if value not in {None, ""}
        }
    head_loader = getattr(archive, "validation_head", None) or archive.trusted_head
    head = head_loader(
        symbol, timeframe, adjustment=str(dataset.get("adjustment", "none")),
        start=lower, end=upper,
    )
    if not calendar_available and head:
        invalid_months = set(head.get("_invalid_months") or [])
        missing_archive_days = []
        for item in head.get("months") or []:
            month = str(item.get("month") or "")[:7]
            if month not in invalid_months:
                continue
            try:
                first_day = pd.Timestamp(item["first_day"]).normalize()
                last_day = pd.Timestamp(item["last_day"]).normalize()
            except (KeyError, TypeError, ValueError):
                continue
            first_day = max(first_day, lower)
            last_day = min(last_day, upper)
            if first_day <= last_day:
                missing_archive_days.extend(pd.date_range(first_day, last_day, freq="D"))
        if missing_archive_days:
            calendar_days = calendar_days.append(
                pd.DatetimeIndex(missing_archive_days)
            ).normalize().unique().sort_values()
    loaded_rows = archive.load_for_validation(
        symbol, timeframe, adjustment=str(dataset.get("adjustment", "none")),
        start=lower, end=upper, head=head,
    ) if hasattr(archive, "load_for_validation") else archive.rows(
        symbol, timeframe, start=lower, end=upper,
        adjustment=str(dataset.get("adjustment", "none")),
    )
    rows = {
        pd.Timestamp(row["trade_date"]).normalize(): row
        for row in loaded_rows
    }
    pending = archive.pending_dates(head, lower, upper)
    has_pending_scope = bool(archive.pending_dates(head))
    actions = repository.auxiliary(symbol, "actions")
    if requested_dates is not None:
        requested_in_range = {
            day for day in requested_dates if lower <= day <= upper
        }
        calendar_days = calendar_days.append(
            pd.DatetimeIndex(sorted(requested_in_range))
        ).normalize().unique().sort_values()
    current_version = int(dataset.get("current_version") or 0)
    file_state = repository.market_file_state(
        symbol, timeframe, str(dataset.get("adjustment", "none")),
    )
    files_valid = str(file_state.get("status") or "readable") == "readable"
    invalid_ranges = []
    for start_value, end_value in file_state.get("invalid_ranges") or []:
        try:
            invalid_ranges.append((
                pd.Timestamp(start_value).normalize(),
                pd.Timestamp(end_value).normalize(),
            ))
        except (TypeError, ValueError):
            continue
    archive_current = bool(
        head and int(head.get("data_version") or 0) == current_version
    )
    version_changed_without_scope = bool(
        head and int(head.get("data_version") or 0) != current_version
        and not has_pending_scope
    )
    targets = []
    reused = 0
    dependency_stale: list[pd.Timestamp] = []
    rules_stale: list[pd.Timestamp] = []
    for day in calendar_days:
        day = pd.Timestamp(day).normalize()
        if requested_dates is not None and day not in requested_dates:
            continue
        row = rows.get(day)
        row_rules_version = int(row.get("rules_version") or 0) if row else 0
        current_rules_version = validation_rules_version(timeframe)
        expected_dependency = _dependency_hash(
            trading_days, actions, day, timeframe=timeframe,
            rules_version=max(row_rules_version, current_rules_version),
        )
        explicitly_requested = bool(requested_dates is not None and force_dates and day in requested_dates)
        dependency_mismatch = bool(
            row and row.get("dependency_hash")
            and str(row.get("dependency_hash")) != expected_dependency
        )
        rule_mismatch = bool(
            row and row_rules_version < current_rules_version
        )
        invalid_months = set((head or {}).get("_invalid_months") or [])
        stale = bool(
            mode == "full" or explicitly_requested or head is None
            or day.to_period("M").strftime("%Y-%m") in invalid_months
            or (not files_valid and (
                not invalid_ranges
                or any(start <= day <= end for start, end in invalid_ranges)
            ))
            or version_changed_without_scope or day in pending
            or row is None or not row.get("scanned")
            or row.get("dirty_reason")
            or dependency_mismatch or rule_mismatch
        )
        if stale:
            targets.append(day)
            if dependency_mismatch:
                dependency_stale.append(day)
            if rule_mismatch:
                rules_stale.append(day)
        else:
            reused += 1
    return {
        "targets": targets, "calendar_days": calendar_days,
        "trading_days": trading_days,
        "reused_days": reused, "calendar_available": calendar_available,
        "lower": lower, "upper": upper, "dataset": dataset,
        "archive": head, "archive_hit": bool(head and (reused or archive_current)),
        "archive_current": archive_current,
        "pending_dates": sorted(pending),
        "has_pending_scope": has_pending_scope,
        "file_state": file_state,
        "dependency_stale": dependency_stale,
        "rules_stale": rules_stale,
    }


def _empty_day_gaps(timeframe: str, day: pd.Timestamp) -> list[dict[str, Any]]:
    spec = timeframe_spec(timeframe)
    if spec.kind == "minute":
        from ..quality import minute_slots

        slots = minute_slots(day)
        size = spec.minutes or 1
        if size != 1:
            slots = slots[(size - 1)::size]
        return coverage_gap_groups(slots, size)
    return [{"start": day.date().isoformat(), "end": day.date().isoformat(), "missing": 1}]


def finalize_validation_publication(
    service, symbol: str, timeframe: str, *, adjustment: str = "none",
    lower, upper, ledger: dict[str, Any], archive_head: dict[str, Any] | None,
) -> dict[str, Any]:
    """Commit one already-published read-only validation result to SQLite.

    The package head is authoritative. This projection runs only after every
    evidence row in a publication batch has passed its version/dependency
    checks, and it never changes the archive reference itself.
    """
    repository = service.repository
    head = dict(archive_head or {})
    pending_by_day: dict[str, str] = {}
    for item in head.get("months") or []:
        summary = item.get("summary") if isinstance(item, dict) else None
        if isinstance(summary, dict):
            pending_by_day.update({
                str(day): str(reason)
                for day, reason in (summary.get("pending_dates") or {}).items()
            })
    for update in head.get("_pending_updates") or []:
        for day in update.get("dates") or []:
            pending_by_day.setdefault(str(day), str(update.get("reason") or "待重新检查"))
    pending = sorted(pending_by_day)
    stale = list((archive_head or {}).get("_stale_dates") or [])
    pending.extend(value for value in stale if value not in pending)
    failed = int(head.get("failed_count") or 0)
    pending_count = int(head.get("pending_count") or 0)
    verified_through = head.get("verified_through")
    final_status = "complete" if (
        not failed and not pending and verified_through
        and not pending_count
    ) else "partial"
    if final_status == "complete":
        reason = None
    elif failed:
        reason = f"{failed} 个交易日未通过校验"
    else:
        reason = f"{max(len(pending), pending_count)} 个交易日待复核"
    checked_days = int(ledger.get("checked_days") or 0)
    if checked_days:
        repository.save_coverage(
            symbol, timeframe, adjustment, lower, upper,
            {
                "status": final_status,
                "expected_rows": int(ledger.get("expected_rows") or 0),
                "observed_rows": int(ledger.get("observed_rows") or 0),
                "synthetic_rows": int(ledger.get("synthetic_rows") or 0),
                "gap_count": int(ledger.get("gap_count") or 0),
                "checked_days": checked_days,
                "failed_days": failed,
                "calendar_available": bool(ledger.get("calendar_available")),
                "coverage_status": final_status,
                "reason": reason,
                "archive_hit": bool(ledger.get("archive_hit")),
                "reused_days": int(ledger.get("reused_days") or 0),
            },
        )
        repository.update_dataset_status(
            symbol, timeframe, adjustment,
            "ready" if final_status == "complete" else "partial",
            reason,
        )
    return {
        "status": final_status,
        "pending_validation": pending,
        "failed_days": failed,
        "verified_from": head.get("scanned_start") or head.get("first_time"),
        "verified_through": verified_through,
        "archive_revision": int((archive_head or {}).get("revision") or 0),
    }


def validate_incremental(
    service,
    symbol: str,
    timeframe: str,
    *,
    mode: str = "incremental",
    start=None,
    end=None,
    dates: Iterable[Any] | None = None,
    force_dates: bool = False,
    adjustment: str = "none",
    preloaded_months: dict[str, pd.DataFrame] | None = None,
    preloaded_daily: pd.DataFrame | None = None,
    selection: dict[str, Any] | None = None,
    persist_results: bool = True,
    persist_archive: bool = True,
    _stale_retry: int = 0,
    _daily_fast_path: bool = True,
    progress=lambda *_: None,
    cancelled=lambda: False,
) -> dict:
    """Validate only absent, changed or explicitly re-opened days."""
    repository = service.repository
    symbol, timeframe = symbol, normalize_timeframe(timeframe)
    if mode not in {"incremental", "full"}:
        raise MarketDataError("校验模式应为 incremental/full")
    prefetched_dates = None
    if preloaded_months is not None:
        pieces = [
            frame[["timestamp"]] for frame in preloaded_months.values()
            if not frame.empty and "timestamp" in frame
        ]
        prefetched_dates = pd.DatetimeIndex(
            pd.concat(pieces, ignore_index=True).timestamp if pieces else []
        )
    selection = validation_targets(
        service, symbol, timeframe, mode=mode, start=start, end=end,
        dates=dates, force_dates=force_dates, adjustment=adjustment,
        prefetched_dates=prefetched_dates,
        selection=selection,
    )
    dataset = selection["dataset"]
    adjustment = str(dataset.get("adjustment", "none"))
    base_data_version = int(dataset.get("current_version") or 0)
    daily_dataset = (
        repository.dataset(symbol, "1day", adjustment)
        if timeframe in MINUTE_TIMEFRAMES else None
    )
    daily_data_version = int((daily_dataset or {}).get("current_version") or 0)
    if preloaded_daily is not None:
        daily_data_version = int(preloaded_daily.attrs.get("_coverage_data_version", daily_data_version))
    lower, upper = selection["lower"], selection["upper"]
    calendar_days = selection["calendar_days"]
    trading_days = selection["trading_days"]
    targets: list[pd.Timestamp] = selection["targets"]
    scanned_months = {pd.Timestamp(day).to_period("M") for day in targets}
    target_storage_map = {
        pd.Timestamp(day).normalize(): repository._storage_period(timeframe, day)
        for day in targets
    }
    scanned_shards = {value[0] for value in target_storage_map.values()}
    calendar_available = selection["calendar_available"]
    head = selection.get("archive")
    file_state = selection.get("file_state") or {}
    files_valid = str(file_state.get("status") or "readable") == "readable"
    invalid_ranges = []
    for start_value, end_value in file_state.get("invalid_ranges") or []:
        try:
            invalid_ranges.append((
                pd.Timestamp(start_value).normalize(),
                pd.Timestamp(end_value).normalize(),
            ))
        except (TypeError, ValueError):
            continue
    needs_untrusted_rebuild = bool(
        head is None or (
            int(head.get("data_version") or 0)
            != int(dataset.get("current_version") or 0)
            and not selection.get("has_pending_scope")
        )
    )
    if persist_results and needs_untrusted_rebuild and not (head or {}).get("_invalid_months"):
        CoverageArchive(repository).mark_all_dirty(
            symbol, timeframe, "行情版本或档案文件待复核",
            adjustment=str(dataset.get("adjustment", "none")),
        )
    elif persist_results and not files_valid:
        archive_index = CoverageArchive(repository)
        invalid_days = [
            row["trade_date"]
            for row in archive_index.rows(
                symbol, timeframe, adjustment=adjustment, start=lower, end=upper,
            )
            if any(
                start <= pd.Timestamp(row["trade_date"]).normalize() <= end
                for start, end in invalid_ranges
            )
        ]
        archive_index.mark_dirty(
            symbol, timeframe, invalid_days, reason="行情分片文件异常",
            adjustment=str(dataset.get("adjustment", "none")),
        )
    elif persist_results:
        archive_index = CoverageArchive(repository)
        if selection.get("dependency_stale"):
            archive_index.mark_dirty(
                symbol, timeframe, selection["dependency_stale"],
                reason="日历或公司行动依赖变化",
                adjustment=str(dataset.get("adjustment", "none")),
            )
        if selection.get("rules_stale"):
            archive_index.mark_dirty(
                symbol, timeframe, selection["rules_stale"],
                reason="校验规则升级",
                adjustment=str(dataset.get("adjustment", "none")),
            )
    if not targets:
        state = repository.validation_state(
            symbol, timeframe, adjustment=adjustment, start=lower, end=upper,
        )
        archive_head = selection.get("archive")
        return {
            "symbol": symbol, "timeframe": timeframe,
            "status": "complete" if (
                state.get("verified_through") and not state.get("failed")
                and not state.get("pending")
            ) else "partial",
            "mode": mode, "checked_days": 0,
            "skipped_days": len(calendar_days), "reused_days": selection["reused_days"],
            "verified_from": state.get("verified_from"),
            "verified_through": state.get("verified_through"),
            "pending_validation": state.get("pending", []),
            "failed_days": len(state.get("failed") or []),
            "calendar_available": calendar_available,
            "archive_hit": selection["archive_hit"],
            "ledger_publication_seconds": 0.0,
            "archive_revision": int((archive_head or {}).get("revision") or 0),
            "_archive_head": archive_head,
            "scanned_months": 0, "scanned_shards": 0,
            "market_data_read_calls": 0, "market_data_shard_reads": 0,
            "evidence_read_calls": 0,
            "daily_dependency_read_calls": 0, "daily_dependency_shard_reads": 0,
        }

    actions = repository.auxiliary(symbol, "actions")
    daily_calendar_hashes = None
    daily_calendar_span = None
    daily_trading_day_set = None
    if timeframe == "1day" and _daily_fast_path and not (actions or {}).get("payload"):
        daily_calendar_hashes = {}
        for calendar_fact in ("unknown", "trading", "non-trading"):
            calendar_hash = hashlib.sha256(calendar_fact.encode()).hexdigest()
            payload = {"rules": validation_rules_version(timeframe), "calendar": calendar_hash, "actions": []}
            daily_calendar_hashes[calendar_fact] = hashlib.sha256(str(payload).encode()).hexdigest()
        if len(trading_days):
            daily_calendar_span = (trading_days.min(), trading_days.max())
            daily_trading_day_set = set(trading_days)
    target_read_metrics = {"calls": 0}
    daily_read_metrics = {"calls": 0}
    daily_by_day: dict[pd.Timestamp, pd.Series] = {}
    if timeframe in MINUTE_TIMEFRAMES:
        daily = preloaded_daily if preloaded_daily is not None else _read_daily_dependencies(
            repository, symbol, targets, adjustment, read_metrics=daily_read_metrics,
            use_cache=mode != "full",
        )
        if not daily.empty:
            daily = daily.assign(day=pd.to_datetime(daily.timestamp).dt.normalize())
            daily_by_day = {
                pd.Timestamp(day).normalize(): part.iloc[-1]
                for day, part in daily.groupby("day", sort=True)
            }
    checked: list[dict] = []
    coverage_days: list[dict[str, Any]] = []
    checked_days = 0
    passed_days = 0
    suspended_days = 0
    failed_days = 0
    expected_rows = observed_rows = synthetic_rows = gap_count = 0
    month_key = None
    batch_month = None
    daily_empty_frame = pd.DataFrame()
    changed_archive_months: set[str] = set()
    archive_head: dict[str, Any] | None = None
    month_frame = pd.DataFrame()
    month_days = pd.DatetimeIndex([])
    target_storage_periods: set[str] = set()
    additional_read_calls = 0
    additional_shard_reads = 0
    evidence_read_calls = 0
    publication_seconds = 0.0
    preloaded = {
        str(month): frame for month, frame in (preloaded_months or {}).items()
    }
    daily_fast = timeframe == "1day" and _daily_fast_path
    daily_period_frames = _daily_preloaded_frames(repository, preloaded_months, adjustment) if daily_fast and preloaded_months else {}
    daily_period_rows: dict[str, dict[pd.Timestamp, list[tuple[int, dict[str, Any]]]]] = {
        key: _frame_day_records(frame) for key, frame in daily_period_frames.items()
    }
    daily_hashes: dict[str, dict[pd.Timestamp, str]] = {
        key: _daily_single_row_hashes(frame) for key, frame in daily_period_frames.items()
    }
    daily_anomalies: dict[str, dict[pd.Timestamp, list[dict[str, Any]]]] = {}
    daily_blocked: dict[str, dict[pd.Timestamp, list[pd.Timestamp]]] = {}
    instrument = service.store.instrument(symbol) if timeframe == "1day" else None
    asset_type = str((instrument or {}).get("asset_type") or "")
    from ..performance import current_trace
    trace = current_trace()
    daily_started = time.perf_counter() if daily_fast else 0.0
    if daily_fast:
        trace.add_count("daily_validation_fast_path_calls")

    def flush_batch() -> None:
        nonlocal checked, coverage_days, batch_month, archive_head
        nonlocal additional_read_calls, additional_shard_reads, evidence_read_calls, publication_seconds
        if not coverage_days:
            return
        changed_archive_months.update(
            str(pd.Timestamp(row["trade_date"]).to_period("M"))
            for row in coverage_days
        )
        archive = CoverageArchive(repository)
        if persist_results:
            if persist_archive:
                archive_head = archive.commit_rows(
                    coverage_days, validation_rows=checked,
                    expected_data_version=base_data_version,
                    changed_months=changed_archive_months,
                )
                publication_seconds += float((archive_head or {}).get("_publication_seconds") or 0)
                evidence_reads = int((archive_head or {}).get("_evidence_read_calls") or 0)
                evidence_read_calls += evidence_reads
                additional_read_calls += evidence_reads
                additional_shard_reads += evidence_reads
                stale_dates = list((archive_head or {}).get("_stale_dates") or [])
                if stale_dates and _stale_retry < 3:
                    retry_result = validate_incremental(
                        service, symbol, timeframe, mode="incremental",
                        start=min(pd.Timestamp(value) for value in stale_dates),
                        end=max(pd.Timestamp(value) for value in stale_dates),
                        dates=stale_dates, force_dates=True, adjustment=adjustment,
                        persist_results=True, persist_archive=True,
                        progress=progress, cancelled=cancelled,
                        _stale_retry=_stale_retry + 1,
                        _daily_fast_path=_daily_fast_path,
                    )
                    archive_head = retry_result.get("_archive_head", archive_head)
                    publication_seconds += float(retry_result.get("ledger_publication_seconds") or 0)
                    additional_read_calls += int(retry_result.get("market_data_read_calls") or 0)
                    additional_shard_reads += int(retry_result.get("market_data_shard_reads") or 0)
                    evidence_read_calls += int(retry_result.get("evidence_read_calls") or 0)
            checked = []
            coverage_days = []
            batch_month = None
            changed_archive_months.clear()

    for index, day in enumerate(targets, start=1):
        if cancelled():
            raise InterruptedError("任务已取消")
        day = pd.Timestamp(day).normalize()
        target_month = day.to_period("M")
        if persist_results and batch_month is not None and (
            (daily_fast and len(coverage_days) >= _DAILY_COMMIT_BATCH_SIZE)
            or (not daily_fast and target_month != batch_month)
        ):
            flush_batch()
        if batch_month is None:
            batch_month = target_month
        if not daily_fast or index == 1 or index % 32 == 0 or index == len(targets):
            progress((index - 1) / max(1, len(targets)), f"校验 {timeframe}：{index - 1}/{len(targets)} 日 · {day.date()}")
        checked_at = utc_now()
        report: dict[str, Any] = {}
        status = "failed"
        reason = None
        skip_minute = False
        details: dict[str, Any] = {}
        calendar_covers_day = bool(
            calendar_available and len(trading_days)
            and trading_days.min() <= day <= trading_days.max()
        )
        is_trading_day = (not calendar_covers_day) or day in trading_days
        daily_row = daily_by_day.get(day) if timeframe in MINUTE_TIMEFRAMES else None
        dependency_failed = timeframe in MINUTE_TIMEFRAMES and daily_row is None
        if calendar_covers_day and not is_trading_day:
            status, reason = "passed", "非交易日，不要求行情记录"
            report = {"status": "complete", "expected_rows": 0, "observed_rows": 0,
                      "non_trading": True, "reason": reason}
            skip_minute = True
            dependency_failed = False
        if not skip_minute and timeframe in MINUTE_TIMEFRAMES and daily_row is not None:
            daily_status = pd.to_numeric(pd.Series([daily_row.get("tradestatus")]), errors="coerce").iloc[0]
            daily_volume = pd.to_numeric(pd.Series([daily_row.get("volume")]), errors="coerce").iloc[0]
            if pd.notna(daily_status) and int(daily_status) == 0:
                status, reason = "suspended", "确认停牌"
                report = {"status": "complete", "expected_rows": 0, "observed_rows": 0, "reason": reason}
                skip_minute = True
            elif pd.notna(daily_volume) and float(daily_volume) == 0:
                status, reason = "passed", "交易状态正常但全天无成交量"
                report = {"status": "complete", "expected_rows": 0, "observed_rows": 0,
                          "no_trades": True, "reason": reason}
                skip_minute = True
        if dependency_failed:
            reason = "缺少对应日线，无法确认交易状态与分钟依赖"
        part = None if daily_fast else pd.DataFrame()
        daily_values = None
        daily_position = None
        fast_daily_day = False
        if not skip_minute and status != "suspended":
            storage_key, shard_start, shard_end = target_storage_map[day]
            key = str(storage_key)
            monthly_preloaded = preloaded.get(str(day.to_period("M")))
            period_preloaded = preloaded.get(key)
            if daily_fast:
                if key not in daily_period_frames:
                    if monthly_preloaded is not None or period_preloaded is not None:
                        candidate = monthly_preloaded if monthly_preloaded is not None else period_preloaded
                        daily_period_frames[key] = candidate
                    else:
                        if key != month_key:
                            target_read_metrics["calls"] += 1
                            target_storage_periods.add(key)
                            month_frame = repository.active_rows(
                                symbol, timeframe, adjustment=adjustment,
                                start=shard_start,
                                end=pd.Timestamp(shard_end) - pd.Timedelta(days=1),
                                **({"use_cache": False} if mode == "full" else {}),
                            )
                            month_key = key
                        daily_period_frames[key] = month_frame
                    daily_period_rows[key] = _frame_day_records(daily_period_frames[key])
                    daily_hashes[key] = _daily_single_row_hashes(daily_period_frames[key])
                source_frame = daily_period_frames[key]
                day_records = daily_period_rows[key].get(day, [])
                if key not in daily_anomalies:
                    anomaly_rows = source_frame.attrs.get("price_anomaly_rows") or []
                    grouped_anomalies: dict[pd.Timestamp, list[dict[str, Any]]] = {}
                    if anomaly_rows:
                        anomaly_frame = pd.DataFrame(anomaly_rows)
                        stamps = pd.to_datetime(anomaly_frame.get("timestamp"), errors="coerce")
                        for anomaly_day, positions in anomaly_frame.groupby(stamps.dt.normalize(), sort=False).indices.items():
                            grouped_anomalies[pd.Timestamp(anomaly_day).normalize()] = anomaly_frame.iloc[positions].to_dict("records")
                    daily_anomalies[key] = grouped_anomalies
                if key not in daily_blocked:
                    blocked_values = pd.DatetimeIndex(pd.to_datetime(
                        source_frame.attrs.get("blocked_timestamps", []), errors="coerce",
                    ).dropna())
                    daily_blocked[key] = {
                        pd.Timestamp(blocked_day).normalize(): list(values)
                        for blocked_day, values in pd.Series(blocked_values).groupby(blocked_values.normalize())
                    }
                anomaly_rows = daily_anomalies[key].get(day, [])
                day_blocked = [stamp.isoformat() for stamp in daily_blocked[key].get(day, [])]
                details["blocked_timestamps"] = day_blocked
                price_anomalies = (
                    anomalies_in_range(source_frame, day, day)
                    if anomaly_rows else None
                )
                if len(day_records) > 1:
                    source_days = pd.DatetimeIndex(source_frame.timestamp).normalize()
                    part = source_frame.loc[source_days == day].copy()
                    price_anomalies = anomalies_in_range(source_frame, day, day)
                    part.attrs["price_anomaly_rows"] = price_anomalies.to_dict("records")
                    part.attrs["blocked_timestamps"] = day_blocked
                else:
                    if day_records:
                        daily_position, daily_values = day_records[0]
                        fast_daily_day = True
            elif monthly_preloaded is not None or period_preloaded is not None:
                source_frame = monthly_preloaded if monthly_preloaded is not None else period_preloaded
                source_days = pd.DatetimeIndex(source_frame.timestamp).normalize()
                part = source_frame.loc[source_days == day].copy()
                price_anomalies = anomalies_in_range(source_frame, day, day)
                part.attrs["price_anomaly_rows"] = price_anomalies.to_dict("records")
                blocked = pd.DatetimeIndex(pd.to_datetime(
                    source_frame.attrs.get("blocked_timestamps", []), errors="coerce",
                ).dropna())
                day_blocked = [stamp.isoformat() for stamp in blocked if stamp.normalize() == day]
                part.attrs["blocked_timestamps"] = day_blocked
                details["blocked_timestamps"] = day_blocked
            else:
                if key != month_key:
                    target_read_metrics["calls"] += 1
                    target_storage_periods.add(key)
                    month_frame = repository.active_rows(
                        symbol, timeframe, adjustment=adjustment,
                        start=shard_start,
                        end=pd.Timestamp(shard_end) - pd.Timedelta(days=1),
                        **({"use_cache": False} if mode == "full" else {}),
                    )
                    month_days = pd.DatetimeIndex(month_frame.timestamp).normalize()
                    month_key = key
                source_frame = month_frame
                source_days = pd.DatetimeIndex(source_frame.timestamp).normalize()
                part = source_frame.loc[source_days == day].copy()
                price_anomalies = anomalies_in_range(source_frame, day, day)
                part.attrs["price_anomaly_rows"] = price_anomalies.to_dict("records")
                blocked = pd.DatetimeIndex(pd.to_datetime(
                    source_frame.attrs.get("blocked_timestamps", []), errors="coerce",
                ).dropna())
                day_blocked = [stamp.isoformat() for stamp in blocked if stamp.normalize() == day]
                part.attrs["blocked_timestamps"] = day_blocked
                details["blocked_timestamps"] = day_blocked
            if price_anomalies is not None and not price_anomalies.empty:
                status = "failed"
                reason = f"存在 {len(price_anomalies)} 条无效价格，已隔离且禁止补造"
                blocked_values = pd.to_datetime(price_anomalies.get("timestamp"), errors="coerce")
                details["missing_timestamps"] = [str(value) for value in blocked_values.dropna()]
                report = {
                    "status": "failed", "expected_rows": 1, "observed_rows": 0,
                    "gap_count": len(price_anomalies), "blocked_price_rows": len(price_anomalies),
                    "price_anomalies": price_anomalies.head(50).astype(str).to_dict("records"),
                    "reason": reason,
                }
            elif daily_values is None and (part is None or part.empty):
                reason = reason or "交易日没有行情记录"
            elif timeframe in {"1week", "1month"}:
                if part.source_verified.all():
                    status = "passed"
                    report = {"status": "complete", "expected_rows": 1, "observed_rows": 1, "gap_count": 0}
                else:
                    reason = "周期来源尚未核验"
            else:
                pending: list[str] = []
                if timeframe == "1day":
                    required = ["open", "high", "low", "close", "volume", "preclose", "tradestatus"]
                    if daily_values is not None:
                        missing = [field for field in required
                                   if field not in daily_values or pd.isna(daily_values[field])]
                    else:
                        missing = [field for field in required if field not in part or part[field].isna().all()]
                    if asset_type == "stock" and (
                        ("is_st" not in daily_values if daily_values is not None else "is_st" not in part)
                        or (pd.isna(daily_values["is_st"]) if daily_values is not None else part.is_st.isna().any())
                    ):
                        missing.append("is_st")
                    if missing:
                        pending.append("缺少严格回测字段：" + ", ".join(missing))
                if fast_daily_day:
                    if daily_values is not None:
                        report, details = _daily_fast_single_report(
                            daily_values["timestamp"], pending, day_blocked,
                        )
                    else:
                        report, details = _daily_fast_report(part, day, pending)
                    validation_pending = report.get("pending_validation") or []
                else:
                    calendar_part = pd.DataFrame({"date": [day]})
                    validation = validate_and_fill(
                        part, timeframe, calendar=calendar_part, fill=True,
                        pending=pending, coverage_start=day, coverage_end=day, symbol=symbol,
                    )
                    report = validation.report
                    details = validation.details or details
                    validation_pending = validation.pending
                if (
                    report.get("status") == "complete"
                    and not validation_pending and not dependency_failed
                ):
                    status = "passed"
                else:
                    reason = "; ".join(validation_pending) or reason or "所选区间存在覆盖缺口"
        part_empty = daily_values is None and (part is None or part.empty)
        missing_values = details.get("missing_timestamps") or []
        missing_days = details.get("missing_days") or []
        if timeframe in MINUTE_TIMEFRAMES:
            gaps = coverage_gap_groups(missing_values, timeframe_spec(timeframe).minutes or 1)
        elif missing_days:
            gaps = [{"start": str(value), "end": str(value), "missing": 1} for value in missing_days]
        elif status == "failed" and part_empty and not skip_minute:
            gaps = _empty_day_gaps(timeframe, day)
        else:
            gaps = []
        if daily_calendar_hashes is not None and daily_row is None:
            if daily_calendar_span is None or day < daily_calendar_span[0] or day > daily_calendar_span[1]:
                calendar_fact = "unknown"
            else:
                calendar_fact = "trading" if day in daily_trading_day_set else "non-trading"
            dependency = daily_calendar_hashes[calendar_fact]
        else:
            dependency = _dependency_hash(
                trading_days, actions, day, daily_row, timeframe=timeframe,
            )
        source_hash = (
            daily_hashes.get(key, {}).get(day)
            if fast_daily_day and day in daily_hashes.get(key, {})
            else _day_hash(source_frame.iloc[[daily_position]])
            if fast_daily_day and daily_position is not None
            else _day_hash(part if part is not None else daily_empty_frame)
        )
        if daily_values is not None:
            stamp_text = pd.Timestamp(daily_values["timestamp"]).isoformat()
            observed_timestamps = [{"start": stamp_text, "end": stamp_text, "count": 1}]
        else:
            observed_timestamps = coverage_timestamp_groups(
                part["timestamp"] if part is not None and "timestamp" in part else [],
                timeframe_spec(timeframe).minutes or (1 if timeframe in MINUTE_TIMEFRAMES else 1440),
            )
        row_count = 1 if daily_values is not None else len(part) if part is not None else 0
        report = {
            **report, "reason": reason, "gaps": gaps, "gap_count": len(gaps),
            "observed_timestamps": observed_timestamps,
        }
        expected_rows += int(report.get("expected_rows") or 0)
        observed_rows += int(report.get("observed_rows") or 0)
        synthetic_rows += int(report.get("synthetic_rows") or 0)
        gap_count += len(gaps)
        checked.append({
            "symbol": symbol, "timeframe": timeframe, "adjustment": adjustment,
            "trade_date": str(day.date()), "status": status, "row_count": row_count,
            "source_hash": source_hash, "dependency_hash": dependency,
            "rules_version": validation_rules_version(timeframe), "report": report,
            "checked_at": checked_at,
        })
        coverage_days.append({
            "symbol": symbol, "timeframe": timeframe, "adjustment": adjustment,
            "trade_date": str(day.date()), "is_trading_day": is_trading_day,
            "scanned": True, "status": status, "row_count": row_count,
            "first_time": str(daily_values["timestamp"] if daily_values is not None else part.timestamp.min()) if not part_empty else None,
            "last_time": str(daily_values["timestamp"] if daily_values is not None else part.timestamp.max()) if not part_empty else None,
            "synthetic_rows": int(bool(daily_values.get("is_synthetic"))) if daily_values is not None else int(part.is_synthetic.sum()) if not part_empty and "is_synthetic" in part else 0,
            "auction_placeholders": int(details.get("auction_placeholders") or 0),
            "blocked_rows": len(details.get("blocked_timestamps") or []),
            "gaps": gaps, "gap_count": len(gaps),
            "pending": [] if status in {"passed", "suspended"} else [reason or "未通过校验"],
            "source_hash": source_hash, "dependency_hash": dependency,
            "rules_version": validation_rules_version(timeframe), "dirty_reason": None,
            "report": report, "checked_at": checked_at,
            "_data_version": int(
                source_frame.attrs.get("_coverage_data_version", base_data_version)
                if daily_values is not None else (part.attrs.get("_coverage_data_version", base_data_version) if part is not None else base_data_version)
            ),
            "_daily_version": daily_data_version if timeframe in MINUTE_TIMEFRAMES else None,
            "_daily_source_hash": (
                _day_hash(pd.DataFrame([daily_row]))
                if timeframe in MINUTE_TIMEFRAMES and daily_row is not None else None
            ),
        })
        checked_days += 1
        if status == "passed":
            passed_days += 1
        elif status == "suspended":
            suspended_days += 1
        else:
            failed_days += 1
        if persist_results and daily_fast and len(coverage_days) >= _DAILY_COMMIT_BATCH_SIZE:
            flush_batch()
    if daily_fast:
        trace.add_seconds("daily_validation_evaluation", time.perf_counter() - daily_started)
        trace.add_count("daily_validation_days", checked_days)
        trace.add_count("daily_validation_period_groups", len(daily_period_rows))
    if cancelled():
        raise InterruptedError("任务已取消")
    flush_batch()
    progress(1.0, f"校验 {timeframe}：{checked_days}/{len(targets)} 日完成")
    if not persist_results:
        return {
            "symbol": symbol, "timeframe": timeframe, "mode": mode,
            "checked_days": checked_days, "passed_days": passed_days,
            "suspended_days": suspended_days, "failed_days": failed_days,
            "skipped_days": max(0, len(calendar_days) - checked_days),
            "reused_days": selection["reused_days"],
            "calendar_available": calendar_available,
            "archive_hit": selection["reused_days"] > 0,
            "_coverage_rows": coverage_days, "_validation_rows": checked,
            "scanned_months": len(scanned_months), "scanned_shards": len(scanned_shards),
            "market_data_read_calls": target_read_metrics["calls"] + additional_read_calls,
            "market_data_shard_reads": len(target_storage_periods) + additional_shard_reads,
            "evidence_read_calls": evidence_read_calls,
            "daily_dependency_read_calls": daily_read_metrics["calls"],
            "daily_dependency_shard_reads": len(daily_read_metrics.get("periods", set())),
            "expected_rows": expected_rows, "observed_rows": observed_rows,
            "synthetic_rows": synthetic_rows, "gap_count": gap_count,
        }
    state = repository.validation_state(
        symbol, timeframe, adjustment=adjustment, start=lower, end=upper,
    )
    remaining_stale = list((archive_head or {}).get("_stale_dates") or [])
    pending_validation = list(state.get("pending") or [])
    pending_validation.extend(
        value for value in remaining_stale if value not in pending_validation
    )
    passed = passed_days
    suspended = suspended_days
    # Discarded worker attempts must not override the published verdict after
    # a retry has succeeded against newer data.
    failed = int(state.get("failed_count") or len(state.get("failed") or []))
    final_status = "complete" if (
        not failed and not state.get("failed") and not state.get("pending")
        and not remaining_stale and state.get("verified_through")
    ) else "partial"
    repository.save_coverage(
        symbol, timeframe, adjustment, lower, upper,
        {
            "status": final_status,
            "expected_rows": expected_rows,
            "observed_rows": observed_rows,
            "synthetic_rows": synthetic_rows,
            "gap_count": gap_count,
            "checked_days": checked_days, "failed_days": failed,
            "calendar_available": calendar_available,
            "coverage_status": final_status,
            "reason": None if final_status == "complete" else f"{failed or len(state.get('failed') or [])} 个交易日未通过校验",
            "archive_hit": selection["reused_days"] > 0,
            "reused_days": selection["reused_days"],
        },
    )
    repository.update_dataset_status(
        symbol, timeframe, adjustment,
        "ready" if final_status == "complete" else "partial",
        None if final_status == "complete" else f"{failed or len(state.get('failed') or [])} 个交易日未通过校验",
    )
    return {
        "symbol": symbol, "timeframe": timeframe,
        "status": final_status, "mode": mode,
        "checked_days": checked_days, "passed_days": passed,
        "suspended_days": suspended, "failed_days": failed,
        "skipped_days": max(0, len(calendar_days) - checked_days),
        "reused_days": selection["reused_days"],
        "verified_from": state.get("verified_from"),
        "verified_through": state.get("verified_through"),
        "pending_validation": pending_validation,
        "calendar_available": calendar_available,
        "archive_hit": selection["reused_days"] > 0,
        "ledger_publication_seconds": publication_seconds,
        "archive_revision": int((archive_head or {}).get("revision") or 0),
        "_archive_head": archive_head,
        "scanned_months": len(scanned_months), "scanned_shards": len(scanned_shards),
        "market_data_read_calls": target_read_metrics["calls"] + additional_read_calls,
        "market_data_shard_reads": len(target_storage_periods) + additional_shard_reads,
        "evidence_read_calls": evidence_read_calls,
        "daily_dependency_read_calls": daily_read_metrics["calls"],
        "daily_dependency_shard_reads": len(daily_read_metrics.get("periods", set())),
    }
