from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from .cache import cached
from .models import DataVersionNotFound, MarketDataError
from .package_layout import (
    PackageLayout, asset_type_for_symbol, auxiliary_relative_path,
    package_file_relative_path, resolve_package_root,
)
from .timeframes import MINUTE_TIMEFRAMES, normalize_timeframe
from ..price_validation import details_to_rows, price_issue_details, split_invalid_prices


PARSER_VERSION = "market-data-v3-price-validation"
COMPACTION_THRESHOLD = 8
VALIDATION_RULES_VERSION = 5
PRICE_FIELDS = ("open", "high", "low", "close", "volume", "turnover")
TIMEFRAMES = ("1min", "5min", "15min", "30min", "60min", "1day", "1week", "1month")
PARQUET_SCHEMA_VERSION = "market-bar-v1"
PARQUET_FIELDS = (
    "symbol", "timeframe", "adjustment", "timestamp",
    "open", "high", "low", "close", "volume", "turnover",
    "preclose", "tradestatus", "is_st", "source", "source_verified",
    "is_synthetic", "is_auction_placeholder", "opening_auction_volume",
    "adj_factor", "metadata",
)
PARQUET_SCHEMA = pa.schema([
    pa.field("symbol", pa.string()),
    pa.field("timeframe", pa.string()),
    pa.field("adjustment", pa.string()),
    pa.field("timestamp", pa.timestamp("ns")),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("volume", pa.float64()),
    pa.field("turnover", pa.float64()),
    pa.field("preclose", pa.float64()),
    pa.field("tradestatus", pa.float64()),
    pa.field("is_st", pa.bool_()),
    pa.field("source", pa.string()),
    pa.field("source_verified", pa.bool_()),
    pa.field("is_synthetic", pa.bool_()),
    pa.field("is_auction_placeholder", pa.bool_()),
    pa.field("opening_auction_volume", pa.float64()),
    pa.field("adj_factor", pa.float64()),
    pa.field("metadata", pa.string()),
])
BOOLEAN_FIELDS = ("source_verified", "is_synthetic", "is_auction_placeholder")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, (pd.Timestamp, datetime)):
            return item.isoformat()
        if item is pd.NA:
            return None
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    return json.dumps(clean(value), ensure_ascii=False, default=str, allow_nan=False)


def _load(value: str | None, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


def _time_text(value: Any) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tz is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


class MarketDataRepository:
    """Immutable Parquet partitions with transactional version manifests."""

    def __init__(
        self,
        root: str | Path,
        package_dir: str | Path | None = None,
        *,
        market_settings=None,
    ):
        from ..settings import load_settings

        self.root = Path(root).resolve()
        self.db_path = self.root / "catalog.sqlite"
        self.market_settings = market_settings or load_settings().market_data
        self.sqlite_timeout = int(self.market_settings.sqlite_timeout_seconds)
        for timeframe in TIMEFRAMES:
            (self.root / "imports" / timeframe).mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            self._initialize(db)
        previous_package = self.meta("package_dir") or self.meta("source_package_dir")
        selected_package = package_dir or previous_package
        self.package = PackageLayout(
            resolve_package_root(selected_package or self.root / "package"),
            market_settings=self.market_settings,
        )
        self.package_dir = self.package.root
        self.market_dir = self.package_dir
        self.staging_dir = self.package.staging_dir
        if previous_package and Path(previous_package).resolve() != self.package_dir:
            self._clear_market_index()
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('package_dir',?)",
                (str(self.package_dir),),
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('source_package_dir',?)",
                (str(self.package_dir),),
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('package_id',?)",
                (self.package.manifest()["package_id"],),
            )
        self._seed_manifest_from_index()
        self._recover_package_transactions()

    def attach_package(self, package_dir: str | Path) -> None:
        previous = self.package_dir
        self.package = PackageLayout(resolve_package_root(package_dir))
        self.package_dir = self.package.root
        self.market_dir = self.package_dir
        self.staging_dir = self.package.staging_dir
        if previous != self.package_dir:
            self._clear_market_index()
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('package_dir',?)",
                (str(self.package_dir),),
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('source_package_dir',?)",
                (str(self.package_dir),),
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('package_id',?)",
                (self.package.manifest()["package_id"],),
            )

    def _clear_market_index(self) -> None:
        """Detach the old package index while preserving saved results and configs."""
        with self._connect() as db:
            tables = {
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for table in (
                "parquet_day_validation", "parquet_increments", "parquet_source_archives",
                "parquet_derived", "parquet_coverage", "parquet_imports",
                "parquet_auxiliary_history", "parquet_files", "parquet_versions",
                "parquet_datasets",
            ):
                if table in tables:
                    db.execute("DELETE FROM " + table)
            if "active_snapshots" in tables:
                db.execute("DELETE FROM active_snapshots")
            if "state" in tables:
                db.execute("DELETE FROM state WHERE key LIKE 'market-source-package%'")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=self.sqlite_timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(f"PRAGMA busy_timeout={self.sqlite_timeout * 1000}")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    connection = _connect

    @staticmethod
    def _initialize(db: sqlite3.Connection) -> None:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS parquet_datasets(
                symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
                adjustment TEXT NOT NULL DEFAULT 'none',
                current_version INTEGER NOT NULL DEFAULT 0,
                first_time TEXT, last_time TEXT, completed_through TEXT,
                row_count INTEGER NOT NULL DEFAULT 0,
                raw_rows INTEGER NOT NULL DEFAULT 0,
                synthetic_rows INTEGER NOT NULL DEFAULT 0,
                pending_rows INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending', reason TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(symbol,timeframe,adjustment)
            );
            CREATE INDEX IF NOT EXISTS parquet_datasets_timeframe
                ON parquet_datasets(timeframe,updated_at);
            CREATE TABLE IF NOT EXISTS parquet_versions(
                symbol TEXT NOT NULL, timeframe TEXT NOT NULL, adjustment TEXT NOT NULL,
                version INTEGER NOT NULL, created_at TEXT NOT NULL,
                row_count INTEGER NOT NULL, raw_rows INTEGER NOT NULL DEFAULT 0,
                synthetic_rows INTEGER NOT NULL DEFAULT 0,
                pending_rows INTEGER NOT NULL DEFAULT 0,
                first_time TEXT, last_time TEXT, status TEXT NOT NULL, source TEXT,
                file_count INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0,
                report TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(symbol,timeframe,adjustment,version)
            );
            CREATE TABLE IF NOT EXISTS parquet_files(
                symbol TEXT NOT NULL, timeframe TEXT NOT NULL, adjustment TEXT NOT NULL,
                version INTEGER NOT NULL, partition_key TEXT NOT NULL,
                path TEXT NOT NULL, sha256 TEXT NOT NULL, row_count INTEGER NOT NULL,
                first_time TEXT, last_time TEXT, size INTEGER NOT NULL,
                schema_version TEXT NOT NULL,
                PRIMARY KEY(symbol,timeframe,adjustment,version,partition_key)
            );
            CREATE INDEX IF NOT EXISTS parquet_files_version
                ON parquet_files(symbol,timeframe,adjustment,version);
            CREATE TABLE IF NOT EXISTS parquet_increments(
                symbol TEXT NOT NULL, timeframe TEXT NOT NULL, adjustment TEXT NOT NULL,
                version INTEGER NOT NULL, sequence INTEGER NOT NULL,
                period_key TEXT NOT NULL, period_start TEXT NOT NULL, period_end TEXT NOT NULL,
                path TEXT NOT NULL, sha256 TEXT NOT NULL, row_count INTEGER NOT NULL,
                first_time TEXT, last_time TEXT, size INTEGER NOT NULL,
                schema_version TEXT NOT NULL,
                PRIMARY KEY(symbol,timeframe,adjustment,version,period_key)
            );
            CREATE INDEX IF NOT EXISTS parquet_increments_version
                ON parquet_increments(symbol,timeframe,adjustment,version,sequence);
            CREATE TABLE IF NOT EXISTS parquet_day_validation(
                symbol TEXT NOT NULL, timeframe TEXT NOT NULL, adjustment TEXT NOT NULL,
                trade_date TEXT NOT NULL, status TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                source_hash TEXT, dependency_hash TEXT, rules_version INTEGER NOT NULL,
                report TEXT NOT NULL DEFAULT '{}', checked_at TEXT NOT NULL,
                PRIMARY KEY(symbol,timeframe,adjustment,trade_date)
            );
            CREATE INDEX IF NOT EXISTS parquet_day_validation_range
                ON parquet_day_validation(symbol,timeframe,adjustment,trade_date,status);
            CREATE TABLE IF NOT EXISTS parquet_source_archives(
                source_path TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL,
                archive_path TEXT NOT NULL, size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL DEFAULT 0,
                copied_bytes INTEGER NOT NULL, archived_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS parquet_imports(
                symbol TEXT NOT NULL, timeframe TEXT NOT NULL, adjustment TEXT NOT NULL,
                fingerprint TEXT NOT NULL, params_hash TEXT NOT NULL,
                parser_version TEXT NOT NULL, file_name TEXT, status TEXT NOT NULL,
                rows_read INTEGER NOT NULL DEFAULT 0,
                inserted INTEGER NOT NULL DEFAULT 0,
                updated INTEGER NOT NULL DEFAULT 0,
                unchanged INTEGER NOT NULL DEFAULT 0,
                synthetic_rows INTEGER NOT NULL DEFAULT 0, version INTEGER,
                report TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL,
                PRIMARY KEY(symbol,timeframe,adjustment,fingerprint,params_hash,parser_version)
            );
            CREATE TABLE IF NOT EXISTS parquet_coverage(
                symbol TEXT NOT NULL, timeframe TEXT NOT NULL, adjustment TEXT NOT NULL,
                start_time TEXT NOT NULL, end_time TEXT NOT NULL, status TEXT NOT NULL,
                expected_rows INTEGER, observed_rows INTEGER, synthetic_rows INTEGER,
                gap_count INTEGER NOT NULL DEFAULT 0, report TEXT NOT NULL DEFAULT '{}',
                checked_at TEXT NOT NULL,
                PRIMARY KEY(symbol,timeframe,adjustment,start_time,end_time)
            );
            CREATE TABLE IF NOT EXISTS parquet_derived(
                symbol TEXT NOT NULL, timeframe TEXT NOT NULL, adjustment TEXT NOT NULL,
                source_timeframe TEXT NOT NULL, target_timeframe TEXT NOT NULL,
                source_version INTEGER NOT NULL, start_time TEXT, end_time TEXT,
                stale INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
                PRIMARY KEY(symbol,timeframe,adjustment,target_timeframe)
            );
            CREATE TABLE IF NOT EXISTS parquet_auxiliary_history(
                symbol TEXT NOT NULL, kind TEXT NOT NULL, version TEXT NOT NULL,
                payload TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(symbol,kind,version)
            );
            CREATE TABLE IF NOT EXISTS parquet_meta(
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
        """)
        archive_columns = {
            row[1] for row in db.execute("PRAGMA table_info(parquet_source_archives)")
        }
        if "source_mtime_ns" not in archive_columns:
            db.execute(
                "ALTER TABLE parquet_source_archives "
                "ADD COLUMN source_mtime_ns INTEGER NOT NULL DEFAULT 0"
            )
        now = utc_now()
        db.execute(
            "INSERT OR IGNORE INTO parquet_meta(key,value) VALUES('generation',?)",
            (hashlib.sha256(f"generation|{now}|{uuid.uuid4()}".encode()).hexdigest()[:24],),
        )

    @property
    def generation(self) -> str:
        with self._connect() as db:
            row = db.execute("SELECT value FROM parquet_meta WHERE key='generation'").fetchone()
        return row[0] if row else "unknown"

    def path(self, timeframe: str) -> Path:
        return self.package_dir

    def meta(self, key: str) -> str | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT value FROM parquet_meta WHERE key=?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES(?,?)",
                (key, value),
            )

    @property
    def source_package(self) -> Path | None:
        return self.package_dir if self.package_dir.is_dir() else None

    @property
    def package_id(self) -> str:
        return str(self.package.manifest()["package_id"])

    def package_info(self) -> dict:
        manifest = self.package.manifest()
        files = manifest.get("files", {})
        sizes = {
            str(path): int(entry.get("size") or 0)
            for path, entry in files.items()
            if entry.get("size") is not None
        }
        current = 0
        with self._connect() as db:
            rows = db.execute(
                """SELECT DISTINCT path,size FROM (
                       SELECT path,size FROM parquet_files
                       UNION ALL SELECT path,size FROM parquet_increments
                   )"""
            ).fetchall()
        for row in rows:
            relative = self.package.relative(row["path"])
            sizes.setdefault(relative, int(row["size"] or 0))
        current = sum(size for size in sizes.values() if size > 0)
        readable = self.package_dir.is_dir()
        pending = self.package.pending_transactions()
        manifest_files = bool(manifest.get("files"))
        index_only = bool(
            manifest.get("index_only")
            or any(entry.get("index_only") for entry in manifest.get("files", {}).values())
        )
        return {
            "package_id": manifest.get("package_id"),
            "package_version": manifest.get("package_version", 0),
            "schema_version": manifest.get("schema_version"),
            "path": str(self.package_dir),
            "status": (
                "recovery_required" if pending else
                "index_only" if readable and index_only else
                "empty" if readable and not rows and not manifest_files else
                "ready" if readable else "missing"
            ),
            "readable": readable,
            "manifest_ready": manifest_files,
            "bytes": current,
            "file_count": len(set(sizes) | {str(path) for path in files}),
            "updated_at": manifest.get("updated_at"),
            "pending_recovery": len(pending),
        }

    def _recover_package_transactions(self) -> None:
        for transaction in self.package.pending_transactions():
            try:
                payload = json.loads(transaction.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            target = self.package.absolute(str(payload.get("target") or ""))
            version = int(payload.get("version") or 0)
            manifest_entry = self.package.manifest().get("files", {}).get(
                str(payload.get("target") or "")
            ) or {}
            published = int(manifest_entry.get("version") or 0) >= version > 0
            if published and target.is_file():
                with self._connect() as db:
                    dataset = self._dataset_row(
                        db, payload.get("symbol"), payload.get("timeframe"), "none"
                    )
                    files = self._current_files(
                        db, payload.get("symbol"), payload.get("timeframe"), "none",
                        int(dataset["current_version"]) if dataset else 0,
                    )
                indexed = any(
                    self.package.relative(row.get("path", "")) == payload.get("target")
                    and int(row.get("size") or -1) == target.stat().st_size
                    for row in files
                )
                if indexed:
                    self.package.finish_transaction(transaction)
                else:
                    self.package.update_transaction(transaction, {"status": "needs_index"})
                continue
            backup_value = payload.get("backup")
            if not published and backup_value:
                backup = self.package.absolute(backup_value)
                if backup.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".recovering")
                    shutil.copy2(backup, temporary)
                    os.replace(temporary, target)
                    with self._connect() as db:
                        db.execute(
                            """UPDATE parquet_files SET path=?,sha256=?,size=?
                               WHERE symbol=? AND timeframe=? AND adjustment=? AND path=?""",
                            (
                                payload.get("target"),
                                f"stat:{target.stat().st_size}:{target.stat().st_mtime_ns}",
                                target.stat().st_size,
                                payload.get("symbol"), payload.get("timeframe"), "none",
                                backup_value,
                            ),
                        )
                    self.package.finish_transaction(transaction)
                    continue
            if payload.get("status") == "installed" and not target.exists() and backup_value:
                self.package.update_transaction(transaction, {"status": "blocked_missing_file"})

    def _seed_manifest_from_index(self) -> None:
        """Adopt a pre-existing index without copying or rewriting package files."""
        manifest = self.package.manifest()
        if manifest.get("files"):
            return
        with self._connect() as db:
            rows = db.execute(
                """SELECT f.path,f.schema_version,MAX(f.version) version
                   FROM parquet_files f JOIN parquet_datasets d
                   ON d.symbol=f.symbol AND d.timeframe=f.timeframe
                   AND d.adjustment=f.adjustment AND d.current_version=f.version
                   GROUP BY f.path,f.schema_version"""
            ).fetchall()
        files = {}
        for row in rows:
            path = self._absolute_source_path(row["path"])
            relative = self.package.relative(path)
            if relative.startswith("../") or relative == ".":
                continue
            files[relative] = {
                "version": int(row["version"] or 0),
                "schema_version": row["schema_version"],
                "adopted_from_index": True,
                "index_only": True,
                "updated_at": utc_now(),
            }
        if files:
            manifest["files"] = files
            manifest["index_only"] = True
            manifest["package_version"] = int(manifest.get("package_version", 0)) + 1
            self.package.publish(manifest)

    def save_basic_info(self, symbol: str, metadata: dict) -> Path:
        asset = asset_type_for_symbol(symbol)
        path = self.package_dir / f"{asset}_basic_data.parquet"
        scalar_metadata = {
            key: value for key, value in metadata.items()
            if value is None or isinstance(value, (str, int, float, bool, np.generic))
        }
        payload = {
            "ts_code": symbol,
            **{key: value for key, value in scalar_metadata.items() if key not in {"symbol", "ts_code"}},
        }
        incoming = pd.DataFrame([payload])
        # The basic-info file is shared by every symbol of an asset type, so the
        # whole read/merge/replace cycle has to run under one exclusive lock.
        # Without it two writers merge against the same stale snapshot and one
        # of their rows is silently lost, and the fixed temporary name makes
        # concurrent writers fight over the same ".writing" file.
        with self.file_lock(path):
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".writing")
            try:
                if path.is_file():
                    with duckdb.connect() as connection:
                        connection.register("incoming_basic", incoming)
                        connection.execute(
                            "COPY (SELECT * FROM read_parquet(?) WHERE ts_code<>? "
                            "UNION ALL BY NAME SELECT * FROM incoming_basic) TO '" +
                            str(temporary).replace("'", "''") +
                            "' (FORMAT PARQUET, COMPRESSION ZSTD)",
                            [str(path), symbol],
                        )
                else:
                    incoming.to_parquet(temporary, compression="zstd", index=False)
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
        self.package.record_file(
            self.package.relative(path), row_count=None, sha256=None,
            metadata={"kind": "basic"},
        )
        return path

    def _absolute_source_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        package_candidate = self.package_dir / candidate
        if package_candidate.exists():
            return package_candidate
        return self.root / candidate

    def lock(self, symbol: str, timeframe: str, adjustment: str = "none") -> FileLock:
        # A shared daily file is the real contention boundary. Locking only by
        # security allowed two writers to merge different symbols into stale
        # copies and lose one another's rows.
        return self.package.lock_for(
            "file:" + package_file_relative_path(symbol, normalize_timeframe(timeframe))
        )

    def file_lock(self, path: str | Path) -> FileLock:
        relative = self.package.relative(path)
        return self.package.lock_for("file:" + relative)

    def package_file_path(self, symbol: str, timeframe: str) -> Path:
        return self.package_dir / package_file_relative_path(symbol, timeframe)

    @staticmethod
    def _dataset_row(db, symbol, timeframe, adjustment):
        return db.execute(
            "SELECT * FROM parquet_datasets WHERE symbol=? AND timeframe=? AND adjustment=?",
            (symbol, timeframe, adjustment),
        ).fetchone()

    def _current_files(self, db, symbol, timeframe, adjustment, version: int) -> list[dict]:
        if version <= 0:
            return []
        rows = db.execute(
            """SELECT * FROM parquet_files
               WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?
               ORDER BY first_time,partition_key""",
            (symbol, timeframe, adjustment, version),
        ).fetchall()
        return [dict(row) for row in rows]

    def _current_increments(
        self, db, symbol, timeframe, adjustment, version: int,
    ) -> list[dict]:
        if version <= 0:
            return []
        rows = db.execute(
            """SELECT * FROM parquet_increments
               WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?
               ORDER BY sequence,period_start""",
            (symbol, timeframe, adjustment, version),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _storage_period(timeframe: str, stamp) -> tuple[str, str, str]:
        value = pd.Timestamp(stamp)
        if timeframe in MINUTE_TIMEFRAMES:
            start = value.normalize().replace(day=1)
            end = (start + pd.offsets.MonthBegin(1)).normalize()
            key = f"month={start:%Y-%m}"
        else:
            start = pd.Timestamp(year=value.year, month=1, day=1)
            end = pd.Timestamp(year=value.year + 1, month=1, day=1)
            key = f"year={value.year:04d}"
        return key, start.isoformat(), end.isoformat()

    @staticmethod
    def _period_windows(increments: Sequence[dict]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        return [
            (pd.Timestamp(row["period_start"]), pd.Timestamp(row["period_end"]))
            for row in increments
        ]

    def _copy_version_files(
        self, db, symbol: str, timeframe: str, adjustment: str,
        source_version: int, target_version: int,
    ) -> None:
        db.execute(
            """INSERT INTO parquet_files(
               symbol,timeframe,adjustment,version,partition_key,path,sha256,
               row_count,first_time,last_time,size,schema_version)
               SELECT symbol,timeframe,adjustment,?,partition_key,path,sha256,
                      row_count,first_time,last_time,size,schema_version
               FROM parquet_files
               WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?""",
            (target_version, symbol, timeframe, adjustment, source_version),
        )

    def _copy_current_increments(
        self, db, symbol: str, timeframe: str, adjustment: str,
        source_version: int, target_version: int, skip_periods: set[str] | None = None,
    ) -> None:
        rows = self._current_increments(db, symbol, timeframe, adjustment, source_version)
        skip_periods = skip_periods or set()
        for row in rows:
            if row["period_key"] in skip_periods:
                continue
            db.execute(
                """INSERT INTO parquet_increments(
                   symbol,timeframe,adjustment,version,sequence,period_key,
                   period_start,period_end,path,sha256,row_count,first_time,
                   last_time,size,schema_version)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, timeframe, adjustment, target_version,
                    int(row["sequence"]), row["period_key"], row["period_start"],
                    row["period_end"], row["path"], row["sha256"],
                    int(row["row_count"]), row["first_time"], row["last_time"],
                    int(row["size"]), row["schema_version"],
                ),
            )

    @staticmethod
    def _merge_market_frames(existing: pd.DataFrame, incoming: pd.DataFrame):
        inserted: list[dict] = []
        updated: list[dict] = []
        unchanged = 0
        conflicts: list[dict] = []
        existing_by_ts = {
            pd.Timestamp(row["timestamp"]): row
            for row in existing.to_dict("records")
        }
        for row in incoming.to_dict("records"):
            stamp = pd.Timestamp(row["timestamp"])
            old = existing_by_ts.get(stamp)
            if old is None:
                inserted.append(row)
                existing_by_ts[stamp] = row
                continue
            if MarketDataRepository._same_market_row(old, row):
                unchanged += 1
                continue
            old_issues = price_issue_details(pd.DataFrame([old]))
            new_issues = price_issue_details(pd.DataFrame([row]))
            if not new_issues.empty:
                unchanged += 1
                conflicts.append({
                    "timestamp": _time_text(stamp),
                    "reason": "无效价格记录不会覆盖现有行情：" + str(new_issues.iloc[0]["quality_reason"]),
                })
                continue
            if not old_issues.empty:
                updated.append(row)
                existing_by_ts[stamp] = row
                continue
            old_rank = MarketDataRepository._quality_rank(
                old.get("source"), old.get("source_verified"), old.get("is_synthetic")
            )
            new_rank = MarketDataRepository._quality_rank(
                row.get("source"), row.get("source_verified"), row.get("is_synthetic")
            )
            if new_rank < old_rank:
                unchanged += 1
                conflicts.append({
                    "timestamp": _time_text(stamp),
                    "reason": "较低优先级记录不能覆盖已核验或原始行情",
                })
                continue
            updated.append(row)
            existing_by_ts[stamp] = row
        combined = pd.DataFrame(existing_by_ts.values())
        if not combined.empty:
            combined = MarketDataRepository._prepare_frame(combined)
        return combined, inserted, updated, unchanged, conflicts

    def _write_increment_file(
        self, symbol: str, timeframe: str, adjustment: str,
        period_key: str, frame: pd.DataFrame,
    ) -> dict:
        period = "month" if period_key.startswith("month=") else "year"
        bucket = period_key.split("=", 1)[1]
        return self._write_partition(
            symbol, timeframe, adjustment, bucket, frame,
            period=period, base_dir=self.package.increments_dir,
        )

    def dataset(self, symbol: str, timeframe: str, adjustment: str = "none") -> dict | None:
        timeframe = normalize_timeframe(timeframe)
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
            if not dataset:
                return None
            item = dict(dataset)
            derived = db.execute(
                """SELECT * FROM parquet_derived WHERE symbol=? AND timeframe=?
                   AND adjustment=? AND target_timeframe=?""",
                (symbol, timeframe, adjustment, timeframe),
            ).fetchone()
            files = self._current_files(db, symbol, timeframe, adjustment, int(item["current_version"]))
            increments = self._current_increments(
                db, symbol, timeframe, adjustment, int(item["current_version"])
            )
            coverage_rows = db.execute(
                """SELECT * FROM parquet_coverage
                   WHERE symbol=? AND timeframe=? AND adjustment=?
                   ORDER BY start_time,end_time""",
                (symbol, timeframe, adjustment),
            ).fetchall()
            history = db.execute(
                """SELECT COALESCE(SUM(size),0) FROM (
                       SELECT path,MAX(size) size FROM (
                           SELECT path,size FROM parquet_files
                           WHERE symbol=? AND timeframe=? AND adjustment=?
                           UNION ALL
                           SELECT path,size FROM parquet_increments
                           WHERE symbol=? AND timeframe=? AND adjustment=?
                       ) GROUP BY path
                   )""",
                (symbol, timeframe, adjustment, symbol, timeframe, adjustment),
            ).fetchone()[0]
        item["metadata"] = _load(item.get("metadata"), {})
        item["timeframe"] = timeframe
        item["version"] = int(item["current_version"])
        file_errors = []
        missing_files = 0
        changed_files = 0
        manifest_files = self.package.manifest().get("files", {})
        for row in files:
            path = self._absolute_source_path(str(row.get("path", "")))
            if not path.is_file():
                file_errors.append(str(path))
                missing_files += 1
                continue
            metadata = manifest_files.get(self.package.relative(path), {})
            expected_size = int(row.get("size") or 0)
            if metadata.get("size") is not None and expected_size and path.stat().st_size != expected_size:
                file_errors.append(f"文件内容摘要已变化：{path}")
                changed_files += 1
            elif metadata.get("mtime_ns") and int(metadata["mtime_ns"]) != path.stat().st_mtime_ns:
                file_errors.append(f"文件修改时间已变化：{path}")
                changed_files += 1
        for row in increments:
            path = self._absolute_source_path(str(row.get("path", "")))
            if not path.is_file():
                file_errors.append(str(path))
                missing_files += 1
            elif int(row.get("size") or 0) and path.stat().st_size != int(row["size"]):
                file_errors.append(f"增量文件大小已变化：{path}")
                changed_files += 1
        item["file_status"] = "missing" if missing_files else "changed" if changed_files else "readable"
        latest_coverage = coverage_rows[-1] if coverage_rows else None
        coverage_report = _load(latest_coverage["report"], {}) if latest_coverage else {}
        coverage_value = str(
            (dict(latest_coverage).get("status") if latest_coverage else None)
            or coverage_report.get("status") or "unknown"
        )
        intervals = self._coverage_intervals(coverage_rows)
        first_day = pd.Timestamp(item["first_time"]).normalize() if item.get("first_time") else None
        last_day = pd.Timestamp(item["last_time"]).normalize() if item.get("last_time") else None
        fully_covered = bool(
            first_day is not None and last_day is not None
            and any(left <= first_day and right >= last_day for left, right in intervals)
        )
        item["coverage_status"] = (
            "complete" if fully_covered else "partial" if intervals else "unknown"
        )
        stored_status = str(item.get("status") or "pending_validation")
        item["quality_status"] = (
            "failed" if stored_status == "failed" else
            "passed" if stored_status == "ready" and item["coverage_status"] == "complete" else
            "pending"
        )
        item["fully_covered"] = fully_covered
        if item["file_status"] == "missing":
            effective_status = "missing"
        elif item["file_status"] == "changed":
            effective_status = "partial"
        elif stored_status == "failed":
            effective_status = "failed"
        elif item["quality_status"] == "passed" and item["fully_covered"]:
            effective_status = "ready"
        elif item["coverage_status"] == "partial" or stored_status == "partial":
            effective_status = "partial"
        else:
            effective_status = "pending_validation"
        item["stored_status"] = stored_status
        item["latest_record_time"] = item.get("last_time")
        verified_through = None
        if first_day is not None:
            for left, right in intervals:
                if left <= first_day <= right:
                    verified_through = right
                    for next_left, next_right in intervals:
                        if next_left <= pd.Timestamp(verified_through) + pd.Timedelta(days=1):
                            verified_through = max(pd.Timestamp(verified_through), next_right)
                if first_day <= pd.Timestamp(verified_through or first_day):
                    break
        compliance = self.validation_state(symbol, timeframe, adjustment=adjustment)
        item["verified_from"] = compliance.get("verified_from")
        item["verified_through"] = compliance.get("verified_through")
        item["validation"] = {
            "rules_version": compliance.get("rules_version"),
            "validated_days": compliance.get("validated_days", 0),
            "pending": compliance.get("pending", []),
            "failed": compliance.get("failed", []),
            "calendar_available": compliance.get("calendar_available", False),
        }
        if compliance.get("failed"):
            item["quality_status"] = "failed"
        elif (
            last_day is not None and compliance.get("verified_through")
            and pd.Timestamp(compliance["verified_through"]).normalize() >= last_day
        ):
            item["quality_status"] = "passed"
        elif compliance.get("verified_through"):
            item["quality_status"] = "partial"
        else:
            item["quality_status"] = "pending"
        if item["quality_status"] == "passed" and item["file_status"] == "readable":
            effective_status = "ready"
        elif item["quality_status"] == "failed":
            effective_status = "failed"
        elif item["file_status"] == "missing" or item["file_status"] == "changed":
            effective_status = item["file_status"] if item["file_status"] == "missing" else "partial"
        elif item["quality_status"] == "partial" or item["coverage_status"] == "partial":
            effective_status = "partial"
        else:
            effective_status = "pending_validation"
        item["status"] = effective_status
        item["ready"] = effective_status == "ready"
        source_rows = [row for row in files if self._is_package_source(row)]
        item["storage"] = "parquet-source" if source_rows else "parquet"
        item["storage_path"] = (
            str(self._absolute_source_path(source_rows[0]["path"]))
            if source_rows else str(self.market_dir / timeframe / f"symbol={symbol}")
        )
        item["file_count"] = len(files) + len(increments)
        item["base_file_count"] = len(files)
        item["increment_count"] = len(increments)
        item["current_bytes"] = sum(int(row["size"]) for row in [*files, *increments])
        item["history_bytes"] = int(history or 0)
        item["derived"] = dict(derived) if derived else None
        item["derived_stale"] = bool(derived["stale"]) if derived else False
        item["coverage_unknown"] = item["coverage_status"] == "unknown"
        item["file_errors"] = file_errors
        item["archive"] = self.archive_status(symbol, timeframe, adjustment)
        return item

    @staticmethod
    def _is_package_source(row: dict) -> bool:
        schema = str(row.get("schema_version", ""))
        return schema.startswith(("etf-source", "stock-source", "package-source"))

    def datasets(
        self, timeframe: str | None = None, query: str = "", status: str | None = None,
        limit: int = 200, offset: int = 0,
    ) -> dict:
        where, args = [], []
        if timeframe:
            where.append("timeframe=?")
            args.append(normalize_timeframe(timeframe))
        if query:
            where.append("(symbol LIKE ? OR metadata LIKE ?)")
            args.extend([f"%{query}%", f"%{query}%"])
        if status:
            if status in {"ready", "partial"} and status == "ready":
                where.append(
                    "status='ready' AND EXISTS (SELECT 1 FROM parquet_coverage c "
                    "WHERE c.symbol=parquet_datasets.symbol AND c.timeframe=parquet_datasets.timeframe "
                    "AND c.adjustment=parquet_datasets.adjustment AND c.status='complete')"
                )
            elif status == "pending_validation":
                where.append("status IN ('pending_validation','pending')")
            else:
                where.append("status=?")
                args.append(status)
        suffix = (" WHERE " + " AND ".join(where)) if where else ""
        with self._connect() as db:
            total = db.execute("SELECT COUNT(*) FROM parquet_datasets" + suffix, args).fetchone()[0]
            rows = db.execute(
                "SELECT * FROM parquet_datasets" + suffix +
                " ORDER BY updated_at DESC,symbol,timeframe LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        items = []
        for row in rows:
            item = self.dataset(row["symbol"], row["timeframe"], row["adjustment"])
            if item:
                items.append(item)
        return {"items": items, "count": int(total), "timeframes": list(TIMEFRAMES)}

    def symbols_for_timeframe(self, timeframe: str) -> list[str]:
        timeframe = normalize_timeframe(timeframe)
        with self._connect() as db:
            rows = db.execute(
                "SELECT symbol FROM parquet_datasets WHERE timeframe=? "
                "AND row_count>0 ORDER BY symbol",
                (timeframe,),
            ).fetchall()
        return [row[0] for row in rows]

    def inventory(
        self, *, query: str = "", asset_type: str | None = None,
        timeframe: str | None = None, status: str | None = None,
        limit: int = 50, offset: int = 0,
    ) -> dict:
        """Return one server-paginated inventory row per security."""
        timeframe = normalize_timeframe(timeframe) if timeframe else None
        where, args = [], []
        if query:
            where.append(
                "(d.symbol LIKE ? OR i.name LIKE ? OR d.metadata LIKE ?)"
            )
            args.extend([f"%{query}%"] * 3)
        if asset_type in {"stock", "etf"}:
            prefix = "etf" if asset_type == "etf" else "stock"
            where.append(
                "(i.asset_type=? OR (d.metadata LIKE ? AND i.asset_type IS NULL))"
            )
            args.extend([asset_type, f'%"asset_type": "{asset_type}"%'])
        if timeframe:
            where.append(
                "d.symbol IN (SELECT symbol FROM parquet_datasets WHERE timeframe=?)"
            )
            args.append(timeframe)
        suffix = (" WHERE " + " AND ".join(where)) if where else ""
        with self._connect() as db:
            rows = db.execute(
                "SELECT d.* FROM parquet_datasets d "
                "LEFT JOIN instruments i ON i.symbol=d.symbol" + suffix +
                " ORDER BY d.symbol,d.timeframe",
                args,
            ).fetchall()
            coverage_rows = db.execute(
                """SELECT symbol,timeframe,adjustment,status,report,start_time,end_time
                   FROM parquet_coverage ORDER BY end_time DESC,checked_at DESC"""
            ).fetchall()
            file_rows = db.execute(
                """SELECT symbol,timeframe,adjustment,version,path,size FROM parquet_files
                   ORDER BY version"""
            ).fetchall()
            increment_rows = db.execute(
                """SELECT symbol,timeframe,adjustment,version,path,size FROM parquet_increments
                   ORDER BY version"""
            ).fetchall()
            instrument_rows = db.execute("SELECT * FROM instruments").fetchall()
            validation_rows = db.execute(
                """SELECT symbol,timeframe,adjustment,trade_date,status,rules_version
                   FROM parquet_day_validation ORDER BY trade_date"""
            ).fetchall()
            calendar_rows = db.execute(
                """SELECT symbol,payload,updated_at FROM parquet_auxiliary_history
                   WHERE kind='calendar' ORDER BY updated_at"""
            ).fetchall()
        instruments = {row["symbol"]: dict(row) for row in instrument_rows}
        coverage_map: dict[tuple[str, str, str], list[dict]] = {}
        for row in coverage_rows:
            coverage_map.setdefault(
                (row["symbol"], row["timeframe"], row["adjustment"]), []
            ).append(dict(row))
        file_map: dict[tuple[str, str, str], list[dict]] = {}
        for row in file_rows:
            file_map.setdefault(
                (row["symbol"], row["timeframe"], row["adjustment"]), []
            ).append(dict(row))
        increment_map: dict[tuple[str, str, str], list[dict]] = {}
        for row in increment_rows:
            increment_map.setdefault(
                (row["symbol"], row["timeframe"], row["adjustment"]), []
            ).append(dict(row))
        validation_map: dict[tuple[str, str, str], list[dict]] = {}
        for row in validation_rows:
            validation_map.setdefault(
                (row["symbol"], row["timeframe"], row["adjustment"]), []
            ).append(dict(row))
        calendar_by_symbol = {}
        for row in calendar_rows:
            payload = _load(row["payload"], [])
            calendar_by_symbol[row["symbol"]] = (self._calendar_dates(payload), bool(payload))
        calendar_by_market = self._calendar_dates_by_market()
        manifest_files = self.package.manifest().get("files", {})
        selected_rows = [dict(row) for row in rows if not timeframe or row["timeframe"] == timeframe]
        symbols_all = sorted({row["symbol"] for row in selected_rows})
        dataset_summaries = []
        trading_dates_by_symbol: dict[str, pd.DatetimeIndex] = {}
        for row in selected_rows:
            key = (row["symbol"], row["timeframe"], row["adjustment"])
            coverage_rows_for_key = coverage_map.get(key, [])
            coverage = coverage_rows_for_key[-1] if coverage_rows_for_key else None
            coverage_status = "unknown"
            fully_covered = False
            if coverage:
                report = _load(coverage["report"], {})
                intervals = self._coverage_intervals(coverage_rows_for_key)
                first_day = pd.Timestamp(row["first_time"]).normalize()
                last_day = pd.Timestamp(row["last_time"]).normalize()
                fully_covered = any(left <= first_day and right >= last_day for left, right in intervals)
                coverage_status = "complete" if fully_covered else "partial" if intervals else "unknown"
            current_files = [item for item in file_map.get(key, []) if int(item.get("version") or 0) == int(row["current_version"] or 0)]
            current_increments = [
                item for item in increment_map.get(key, [])
                if int(item.get("version") or 0) == int(row["current_version"] or 0)
            ]
            file_status = "readable"
            for item in current_files:
                path = self._absolute_source_path(item["path"])
                if not path.is_file():
                    file_status = "missing"
                    break
                metadata = manifest_files.get(self.package.relative(path), {})
                if metadata.get("size") is not None and int(item.get("size") or 0) and path.stat().st_size != int(item["size"]):
                    file_status = "changed"
                elif metadata.get("mtime_ns") and int(metadata["mtime_ns"]) != path.stat().st_mtime_ns:
                    file_status = "changed"
            for item in current_increments:
                path = self._absolute_source_path(item["path"])
                if not path.is_file():
                    file_status = "missing"
                    break
                if int(item.get("size") or 0) and path.stat().st_size != int(item["size"]):
                    file_status = "changed"
            stored = str(row.get("status") or "pending_validation")
            stored_calendar = calendar_by_symbol.get(row["symbol"])
            if stored_calendar is not None:
                calendar_dates, calendar_available = stored_calendar
            else:
                calendar_dates = calendar_by_market.get(row["symbol"][-2:], pd.DatetimeIndex([]))
                calendar_available = bool(len(calendar_dates))
            records = validation_map.get(key, [])
            expected_dates = None
            if records:
                if row["symbol"] not in trading_dates_by_symbol:
                    same_symbol = [item for item in selected_rows if item["symbol"] == row["symbol"]]
                    trading_dates_by_symbol[row["symbol"]] = self.trading_days(
                        row["symbol"], min(item["first_time"] for item in same_symbol),
                        max(item["last_time"] for item in same_symbol),
                    )
                expected_dates = trading_dates_by_symbol[row["symbol"]]
            compliance = self._validation_from_parts(
                row, records, calendar_dates, calendar_available=calendar_available,
                expected_dates=expected_dates,
            )
            quality_status = (
                "failed" if compliance.get("failed") else
                "passed" if compliance.get("verified_through") and pd.Timestamp(
                    compliance["verified_through"]
                ).normalize() >= pd.Timestamp(row["last_time"]).normalize() else
                "partial" if compliance.get("verified_through") else "pending"
            )
            effective = (
                "missing" if file_status == "missing" else
                "failed" if quality_status == "failed" else
                "partial" if file_status == "changed" else
                "ready" if quality_status == "passed" else
                "partial" if quality_status == "partial" or coverage_status == "partial" else
                "pending_validation"
            )
            dataset_summaries.append({
                **row, "file_status": file_status, "quality_status": quality_status,
                "coverage_status": coverage_status, "status": effective,
                "ready": effective == "ready",
                "fully_covered": fully_covered,
                "verified_from": compliance.get("verified_from"),
                "verified_through": compliance.get("verified_through"),
                "validation": compliance,
                "increment_count": len(current_increments),
            })
        if status:
            dataset_summaries = [row for row in dataset_summaries if row["status"] == status]
        symbols = sorted({row["symbol"] for row in dataset_summaries})
        page_symbols = symbols[int(offset): int(offset) + int(limit)]
        page_set = set(page_symbols)
        grouped = {symbol: [] for symbol in page_symbols}
        for row in dataset_summaries:
            if row["symbol"] in page_set:
                grouped[row["symbol"]].append(row)
        items = []
        for symbol in page_symbols:
            periods = {}
            for row in grouped[symbol]:
                stored = row.get("metadata")
                metadata = _load(stored, {}) if isinstance(stored, str) else (stored or {})
                periods[row["timeframe"]] = {
                    "symbol": symbol, "timeframe": row["timeframe"],
                    "current_version": row["current_version"],
                    "first_time": row["first_time"], "last_time": row["last_time"],
                    "completed_through": row["completed_through"],
                    "latest_record_time": row["last_time"],
                    "verified_from": row.get("verified_from"),
                    "verified_through": row.get("verified_through"),
                    "row_count": row["row_count"], "synthetic_rows": row["synthetic_rows"],
                    "status": row["status"], "quality_status": row["quality_status"],
                    "coverage_status": row["coverage_status"], "file_status": row["file_status"],
                    "ready": row["ready"], "reason": row.get("reason"),
                    "updated_at": row.get("updated_at"), "metadata": metadata,
                    "validation": row.get("validation"),
                    "increment_count": row.get("increment_count", 0),
                }
            base = instruments.get(symbol, {})
            metadata = next((item["metadata"] for item in periods.values()), {})
            items.append({
                "symbol": symbol,
                "name": base.get("name") or metadata.get("name") or symbol,
                "asset_type": base.get("asset_type") or metadata.get("asset_type") or asset_type_for_symbol(symbol),
                "periods": periods,
                "updated_at": max((str(item.get("updated_at") or "") for item in periods.values()), default=None),
            })
        bytes_by_path = {}
        for row in dataset_summaries:
            for item in file_map.get((row["symbol"], row["timeframe"], row["adjustment"]), []):
                if int(item.get("version") or 0) == int(row["current_version"] or 0):
                    bytes_by_path[self.package.relative(item["path"])] = int(item.get("size") or 0)
            for item in increment_map.get((row["symbol"], row["timeframe"], row["adjustment"]), []):
                if int(item.get("version") or 0) == int(row["current_version"] or 0):
                    bytes_by_path[self.package.relative(item["path"])] = int(item.get("size") or 0)
        return {
            "items": items,
            "count": len(symbols),
            "dataset_count": len(dataset_summaries),
            "total_symbols": len(symbols_all),
            "stats": {
                "symbols": len(symbols),
                "datasets": len(dataset_summaries),
                "ready": sum(1 for row in dataset_summaries if row["status"] == "ready"),
                "pending": sum(1 for row in dataset_summaries if row["status"] != "ready"),
                "rows": sum(int(row["row_count"] or 0) for row in dataset_summaries),
                "bytes": sum(bytes_by_path.values()),
            },
            "limit": int(limit), "offset": int(offset),
        }

    def version_info(
        self, symbol: str, timeframe: str, version: int, adjustment: str = "none",
    ) -> dict | None:
        timeframe = normalize_timeframe(timeframe)
        with self._connect() as db:
            row = db.execute(
                """SELECT * FROM parquet_versions
                   WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?""",
                (symbol, timeframe, adjustment, int(version)),
            ).fetchone()
            files = self._current_files(db, symbol, timeframe, adjustment, int(version)) if row else []
        if not row:
            return None
        item = dict(row)
        item["report"] = _load(item.get("report"), {})
        item["storage"] = "parquet-source"
        manifest_files = self.package.manifest().get("files", {})
        file_status = "readable"
        for file_row in files:
            path = self._absolute_source_path(file_row["path"])
            if not path.is_file():
                file_status = "missing"
                break
            metadata = manifest_files.get(self.package.relative(path), {})
            if metadata and (
                int(file_row.get("size") or 0) != path.stat().st_size
                or (metadata.get("mtime_ns") and int(metadata["mtime_ns"]) != path.stat().st_mtime_ns)
            ):
                file_status = "changed"
        item["file_status"] = file_status
        return item

    @staticmethod
    def _canonical_fields(fields: Sequence[str] | None) -> tuple[str, ...]:
        if not fields:
            return PARQUET_FIELDS
        selected = [field for field in dict.fromkeys(fields) if field in PARQUET_FIELDS]
        if "timestamp" not in selected:
            selected.insert(0, "timestamp")
        return tuple(selected)

    def _read_paths(
        self, paths: Sequence[str], *, start=None, end=None, before=None,
        fields: Sequence[str] | None = None, limit: int | None = None,
        descending: bool = False,
        exclude_periods: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = (),
    ) -> pd.DataFrame:
        columns = self._canonical_fields(fields)
        if not paths:
            return pd.DataFrame(columns=list(columns))
        absolute = [str(self._absolute_source_path(path)) for path in paths]
        with duckdb.connect() as connection:
            relation = connection.read_parquet(absolute, union_by_name=True)
            if start is not None:
                relation = relation.filter(f"timestamp >= TIMESTAMP '{_time_text(start)}'")
            if before is not None:
                relation = relation.filter(f"timestamp < TIMESTAMP '{_time_text(before)}'")
            if end is not None:
                relation = relation.filter(f"timestamp < TIMESTAMP '{_time_text(end)}'")
            for period_start, period_end in exclude_periods:
                relation = relation.filter(
                    "NOT (timestamp >= TIMESTAMP '" + _time_text(period_start)
                    + "' AND timestamp < TIMESTAMP '" + _time_text(period_end) + "')"
                )
            relation = relation.order("timestamp DESC" if descending else "timestamp ASC")
            if limit is not None:
                cushion = 64 if descending else 0
                relation = relation.limit(int(limit) + cushion)
            relation = relation.project(", ".join(columns))
            return self._frame_for_consumer(relation.df())

    def _read_source_files(
        self,
        files: Sequence[dict],
        *,
        symbol: str,
        timeframe: str,
        start=None,
        end=None,
        before=None,
        fields: Sequence[str] | None = None,
        limit: int | None = None,
        descending: bool = False,
        exclude_periods: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = (),
    ) -> pd.DataFrame:
        """Read package files in place and expose canonical bar columns."""
        from .reader import normalize_frame

        columns = self._canonical_fields(fields)
        source_kind = str(files[0].get("schema_version", ""))
        path = self._absolute_source_path(files[0]["path"])
        if not path.is_file():
            raise MarketDataError("行情源文件不存在：" + str(path))
        is_daily = timeframe == "1day" or source_kind.endswith("source-daily")
        time_column = "trade_date" if is_daily else "trade_time"
        relative = self.package.relative(path)
        file_metadata = self.package.file_metadata(relative)
        with duckdb.connect() as connection:
            relation = connection.read_parquet(str(path))
            if start is not None:
                relation = relation.filter(
                    f"{time_column} >= TIMESTAMP '{_time_text(start)}'"
                )
            if before is not None:
                relation = relation.filter(
                    f"{time_column} < TIMESTAMP '{_time_text(before)}'"
                )
            if end is not None:
                relation = relation.filter(
                    f"{time_column} < TIMESTAMP '{_time_text(end)}'"
                )
            if is_daily:
                relation = relation.filter(f"ts_code = '{symbol}'")
            for period_start, period_end in exclude_periods:
                relation = relation.filter(
                    "NOT (" + time_column + " >= TIMESTAMP '" + _time_text(period_start)
                    + "' AND " + time_column + " < TIMESTAMP '" + _time_text(period_end) + "')"
                )
            relation = relation.order(
                time_column + (" DESC" if descending and limit is not None else " ASC")
            )
            if limit is not None and descending:
                relation = relation.limit(int(limit) + 64)
            raw = relation.df()
        if raw.empty:
            return pd.DataFrame(columns=list(columns))
        if "preclose" not in raw and "pre_close" in raw:
            raw["preclose"] = raw["pre_close"]
        raw["timestamp"] = pd.to_datetime(raw[time_column])
        raw_flags = raw.get("source_verified")
        raw_synthetic = raw.get("is_synthetic")
        raw_auction = raw.get("is_auction_placeholder")
        raw_stamps = pd.to_datetime(raw[time_column], errors="coerce")

        def flag_lookup(values):
            if values is None:
                return {}
            series = pd.Series(values, index=raw.index)
            return {
                pd.Timestamp(stamp): value
                for stamp, value in zip(raw_stamps, series)
                if pd.notna(stamp)
            }

        flag_by_stamp = flag_lookup(raw_flags)
        synthetic_by_stamp = flag_lookup(raw_synthetic)
        auction_by_stamp = flag_lookup(raw_auction)
        label_convention = str(file_metadata.get("label_convention") or "end")
        normalized, _ = normalize_frame(
            raw, timeframe, label_convention=label_convention,
            source=str(file_metadata.get("source") or "package"),
            skip_invalid_rows=True,
        )
        normalize_anomalies = normalized.attrs.get("price_anomaly_rows") or []
        if is_daily:
            volume_multiplier = float(file_metadata.get("volume_multiplier", 100.0))
            turnover_multiplier = float(file_metadata.get("turnover_multiplier", 1000.0))
            normalized["volume"] = pd.to_numeric(
                normalized.volume, errors="coerce"
            ) * volume_multiplier
            normalized["turnover"] = pd.to_numeric(
                normalized.turnover, errors="coerce"
            ) * turnover_multiplier
            if "tradestatus" in raw:
                normalized["tradestatus"] = pd.to_numeric(
                    raw["tradestatus"], errors="coerce"
                ).fillna(1).to_numpy()
            else:
                normalized["tradestatus"] = (
                    pd.to_numeric(normalized.volume, errors="coerce") > 0
                ).astype(int)
        if raw_flags is not None:
            normalized["source_verified"] = normalized.timestamp.map(
                flag_by_stamp
            ).map(lambda value: bool(value) if pd.notna(value) else False).astype(bool)
        else:
            normalized["source_verified"] = bool(
                file_metadata.get("source_verified", False)
            )
        if raw_synthetic is not None:
            normalized["is_synthetic"] = normalized.timestamp.map(
                synthetic_by_stamp
            ).map(lambda value: bool(value) if pd.notna(value) else False).astype(bool)
        if raw_auction is not None:
            normalized["is_auction_placeholder"] = normalized.timestamp.map(
                auction_by_stamp
            ).map(lambda value: bool(value) if pd.notna(value) else False).astype(bool)
        if start is not None:
            normalized = normalized[normalized.timestamp >= pd.Timestamp(start)]
        if before is not None:
            normalized = normalized[normalized.timestamp < pd.Timestamp(before)]
        if end is not None:
            normalized = normalized[normalized.timestamp < pd.Timestamp(end)]
        normalized = normalized.sort_values("timestamp", kind="stable")
        if limit is not None:
            normalized = (
                normalized.tail(int(limit))
                if descending else normalized.head(int(limit))
            )
        for field, default in (
            ("symbol", symbol), ("timeframe", timeframe), ("adjustment", "none"),
            ("metadata", "{}"),
        ):
            if field not in normalized:
                normalized[field] = default
        for field in columns:
            if field in normalized:
                continue
            if field in BOOLEAN_FIELDS:
                normalized[field] = False
            elif field == "is_st":
                normalized[field] = pd.Series(pd.NA, index=normalized.index, dtype="boolean")
            elif field in {"source", "metadata"}:
                normalized[field] = "package" if field == "source" else "{}"
            else:
                normalized[field] = np.nan
        result = self._frame_for_consumer(normalized[list(columns)])
        if normalize_anomalies:
            existing = result.attrs.get("price_anomaly_rows") or []
            merged_rows = {str(row.get("timestamp")): row for row in [*existing, *normalize_anomalies]}
            result.attrs["price_anomaly_rows"] = sorted(
                merged_rows.values(), key=lambda row: str(row.get("timestamp") or "")
            )
            result.attrs["blocked_timestamps"] = sorted(
                str(row["timestamp"]) for row in merged_rows.values() if row.get("timestamp")
            )
        return result

    def _read_version(
        self, symbol: str, timeframe: str, adjustment: str, version: int, *,
        start=None, end=None, before=None, fields: Sequence[str] | None = None,
        limit: int | None = None, descending: bool = False, file_lock: bool = True,
        years: Iterable[int] | None = None, use_cache: bool = True,
    ) -> pd.DataFrame:
        with self._connect() as db:
            files = self._current_files(db, symbol, timeframe, adjustment, version)
            increments = self._current_increments(
                db, symbol, timeframe, adjustment, version
            )
        if not files and not increments:
            return pd.DataFrame(columns=list(self._canonical_fields(fields)))
        if years is not None:
            wanted = {str(int(year)) for year in years}
            files = [
                row for row in files
                if self._is_package_source(row)
                or str(row["partition_key"]).split("=")[-1] in wanted
            ]
            increments = [
                row for row in increments
                if any(
                    pd.Timestamp(row["period_start"]).year <= int(year) <= pd.Timestamp(row["period_end"]).year
                    for year in wanted
                )
            ]
        if not files and not increments:
            return pd.DataFrame(columns=list(self._canonical_fields(fields)))
        columns = self._canonical_fields(fields)
        source_files = [row for row in files if self._is_package_source(row)]
        managed_files = [row for row in files if not self._is_package_source(row)]
        excluded = self._period_windows(increments)
        file_key = []
        for row in [*files, *increments]:
            if self._is_package_source(row):
                path = self._absolute_source_path(row["path"])
                try:
                    stat = path.stat()
                    file_key.append((str(path), stat.st_size, stat.st_mtime_ns))
                except OSError:
                    file_key.append((str(path), None, None))
            else:
                file_key.append(row["path"])
        key = (
            str(self.root), self.generation, symbol, timeframe, adjustment, int(version),
            _time_text(start) if start is not None else None,
            _time_text(end) if end is not None else None,
            _time_text(before) if before is not None else None,
            tuple(columns), limit, bool(descending),
            tuple(file_key),
        )

        def load():
            frames: list[pd.DataFrame] = []
            if source_files:
                path = self._absolute_source_path(source_files[0]["path"])
                if file_lock:
                    guard = self.file_lock(path)
                else:
                    class _NoLock:
                        def __enter__(self): return self
                        def __exit__(self, *_): return False
                    guard = _NoLock()
                with guard:
                    frames.append(self._read_source_files(
                        source_files, symbol=symbol, timeframe=timeframe,
                        start=start, end=end, before=before, fields=columns,
                        limit=limit, descending=descending,
                        exclude_periods=excluded,
                    ))
            if managed_files:
                frames.append(self._read_paths(
                    [row["path"] for row in managed_files],
                    start=start, end=end, before=before, fields=columns,
                    limit=limit, descending=descending,
                    exclude_periods=excluded,
                ))
            for increment in increments:
                frames.append(self._read_paths(
                    [increment["path"]], start=start, end=end, before=before,
                    fields=columns, limit=limit, descending=descending,
                ))
            anomaly_parts = [
                row
                for part in frames
                for row in (part.attrs.get("price_anomaly_rows") or [])
            ]
            parts = [frame for frame in frames if frame is not None and not frame.empty]
            if not parts:
                empty = pd.DataFrame(columns=list(columns))
                if anomaly_parts:
                    unique = {str(row.get("timestamp")): row for row in anomaly_parts}
                    empty.attrs["price_anomaly_rows"] = sorted(
                        unique.values(), key=lambda row: str(row.get("timestamp") or "")
                    )
                return empty
            merged = self._prepare_frame(pd.concat(parts, ignore_index=True, sort=False))
            merged = self._frame_for_consumer(merged[list(columns)])
            if limit is not None:
                merged = (
                    merged.tail(int(limit)) if descending else merged.head(int(limit))
                )
            if anomaly_parts:
                unique = {str(row.get("timestamp")): row for row in anomaly_parts}
                merged.attrs["price_anomaly_rows"] = sorted(
                    unique.values(), key=lambda row: str(row.get("timestamp") or "")
                )
                merged.attrs["blocked_timestamps"] = sorted(
                    str(row["timestamp"]) for row in unique.values() if row.get("timestamp")
                )
            return merged

        return cached(key, load) if use_cache else load()

    def active_rows(
        self, symbol: str, timeframe: str, adjustment: str = "none",
        start=None, end=None, limit: int | None = None,
    ) -> pd.DataFrame:
        timeframe = normalize_timeframe(timeframe)
        dataset = self.dataset(symbol, timeframe, adjustment)
        if not dataset:
            return pd.DataFrame()
        exclusive_end = (
            pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
            if end is not None else None
        )
        return self._read_version(
            symbol, timeframe, adjustment, int(dataset["current_version"]),
            start=start, end=exclusive_end, limit=limit,
        )

    def bars(
        self, symbol: str, timeframe: str, adjustment: str = "none", start=None,
        end=None, version: int | None = None, limit: int | None = None,
    ) -> pd.DataFrame:
        timeframe = normalize_timeframe(timeframe)
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
            exists = True
            if dataset and version is not None and int(version) != int(dataset["current_version"]):
                exists = bool(db.execute(
                    """SELECT 1 FROM parquet_versions
                       WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?""",
                    (symbol, timeframe, adjustment, int(version)),
                ).fetchone())
        if not dataset:
            if version is not None:
                raise DataVersionNotFound(
                    f"DATA_VERSION_NOT_FOUND: {symbol} {timeframe} 版本 {version} 不存在"
                )
            return pd.DataFrame()
        selected = int(version) if version is not None else int(dataset["current_version"])
        if version is not None and not exists:
            raise DataVersionNotFound(
                f"DATA_VERSION_NOT_FOUND: {symbol} {timeframe} 版本 {version} 已不存在"
            )
        exclusive_end = (
            pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
            if end is not None else None
        )
        return self._read_version(
            symbol, timeframe, adjustment, selected,
            start=start, end=exclusive_end, limit=limit,
        )

    def last_bars(
        self, symbol: str, timeframe: str, *, before=None, limit: int = 600,
        adjustment: str = "none", fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        dataset = self.dataset(symbol, normalize_timeframe(timeframe), adjustment)
        if not dataset:
            return pd.DataFrame()
        frame = self._read_version(
            symbol, normalize_timeframe(timeframe), adjustment, int(dataset["current_version"]),
            before=before, fields=fields, limit=limit, descending=True,
        )
        return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

    def read_frame(
        self, symbol: str, timeframe: str, *, adjustment: str = "none",
        version: int | None = None, start=None, end=None, before=None,
        fields: Sequence[str] | None = None, limit: int | None = None,
        descending: bool = False, use_cache: bool = True,
    ) -> pd.DataFrame:
        timeframe = normalize_timeframe(timeframe)
        dataset = self.dataset(symbol, timeframe, adjustment)
        if not dataset:
            if version is not None:
                raise DataVersionNotFound(
                    f"DATA_VERSION_NOT_FOUND: {symbol} {timeframe} 版本 {version} 不存在"
                )
            return pd.DataFrame(columns=list(self._canonical_fields(fields)))
        selected = int(version) if version is not None else int(dataset["current_version"])
        exclusive_end = (
            pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
            if end is not None else None
        )
        return self._read_version(
            symbol, timeframe, adjustment, selected, start=start, end=exclusive_end, before=before,
            fields=fields, limit=limit, descending=descending, use_cache=use_cache,
        )

    @staticmethod
    def _frame_for_consumer(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame.copy()
        inherited = frame.attrs.get("price_anomaly_rows") or []
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        for field in BOOLEAN_FIELDS:
            if field in frame:
                frame[field] = frame[field].fillna(False).astype(bool)
        if "is_st" in frame:
            frame["is_st"] = frame["is_st"].astype("boolean")
        if "metadata" in frame:
            frame["metadata"] = frame["metadata"].map(
                lambda value: _load(value, {}) if isinstance(value, str) else (value or {})
            )
        valid, current = split_invalid_prices(frame)
        anomaly_rows = [*inherited, *details_to_rows(current)]
        if anomaly_rows:
            unique = {str(row.get("timestamp")): row for row in anomaly_rows}
            valid.attrs["price_anomaly_rows"] = sorted(
                unique.values(), key=lambda row: str(row.get("timestamp") or "")
            )
            valid.attrs["blocked_timestamps"] = sorted(
                str(row["timestamp"]) for row in unique.values() if row.get("timestamp")
            )
        return valid.sort_values("timestamp", kind="stable").reset_index(drop=True)

    def fingerprint_seen(
        self, symbol: str, timeframe: str, adjustment: str,
        fingerprint: str, params_hash: str,
    ) -> dict | None:
        timeframe = normalize_timeframe(timeframe)
        with self._connect() as db:
            row = db.execute(
                """SELECT * FROM parquet_imports
                   WHERE symbol=? AND timeframe=? AND adjustment=? AND fingerprint=?
                     AND params_hash=? AND parser_version=?""",
                (symbol, timeframe, adjustment, fingerprint, params_hash, PARSER_VERSION),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["report"] = _load(item.get("report"), {})
        return item

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        result = frame.copy()
        if "timestamp" not in result and "ts" in result:
            result["timestamp"] = result.ts
        if "timestamp" not in result:
            raise MarketDataError("行情缺少 timestamp")
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
        if getattr(result["timestamp"].dt, "tz", None) is not None:
            result["timestamp"] = result["timestamp"].dt.tz_convert(
                "Asia/Shanghai"
            ).dt.tz_localize(None)
        for field in PRICE_FIELDS:
            if field not in result:
                result[field] = 0.0 if field == "volume" else np.nan
            result[field] = pd.to_numeric(result[field], errors="coerce")
        for field in ("preclose", "tradestatus", "opening_auction_volume", "adj_factor"):
            if field not in result:
                result[field] = np.nan
            result[field] = pd.to_numeric(result[field], errors="coerce")
        for field, default in (
            ("source", "import"), ("source_verified", False),
            ("is_synthetic", False), ("is_auction_placeholder", False),
        ):
            if field not in result:
                result[field] = default
        result["source"] = result["source"].fillna("import").astype(str)
        for field in BOOLEAN_FIELDS:
            result[field] = result[field].fillna(False).astype(bool)
        if "is_st" not in result:
            result["is_st"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        return result.sort_values("timestamp", kind="stable").drop_duplicates(
            "timestamp", keep="last"
        ).reset_index(drop=True)

    def _write_partition(
        self, symbol: str, timeframe: str, adjustment: str, year: int | str, frame: pd.DataFrame,
        *, prepared: bool = False, period: str = "year", base_dir: str | Path | None = None,
    ) -> dict:
        result = frame if prepared else self._prepare_frame(frame)
        if period == "all":
            period_key = f"snapshot={uuid.uuid4().hex[:12]}"
        elif period == "month":
            month_text = str(year)
            result = result[
                result.timestamp.dt.strftime("%Y-%m") == month_text
            ].copy()
            period_key = f"month={month_text}"
        else:
            result = result[result.timestamp.dt.year == int(year)].copy()
            period_key = f"year={int(year)}"
        if result.empty:
            raise MarketDataError("不能写入空 Parquet 分区")
        result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
        result["symbol"] = symbol
        result["timeframe"] = timeframe
        result["adjustment"] = adjustment
        if "metadata" not in result:
            metadata_values = ["{}"] * len(result)
        else:
            raw_metadata = result["metadata"].tolist()
            if not any(value for value in raw_metadata):
                metadata_values = ["{}"] * len(result)
            else:
                metadata_values = [
                    value if isinstance(value, str) else _json(value or {})
                    for value in raw_metadata
                ]
        st_values = [None if pd.isna(value) else bool(value) for value in result["is_st"]]
        arrays = {
            "symbol": pa.array(result["symbol"].tolist(), type=pa.string()),
            "timeframe": pa.array(result["timeframe"].tolist(), type=pa.string()),
            "adjustment": pa.array(result["adjustment"].tolist(), type=pa.string()),
            "timestamp": pa.array(result["timestamp"], type=pa.timestamp("ns")),
            "open": pa.array(result["open"], type=pa.float64()),
            "high": pa.array(result["high"], type=pa.float64()),
            "low": pa.array(result["low"], type=pa.float64()),
            "close": pa.array(result["close"], type=pa.float64()),
            "volume": pa.array(result["volume"], type=pa.float64()),
            "turnover": pa.array(result["turnover"], type=pa.float64()),
            "preclose": pa.array(result["preclose"], type=pa.float64()),
            "tradestatus": pa.array(result["tradestatus"], type=pa.float64()),
            "is_st": pa.array(st_values, type=pa.bool_()),
            "source": pa.array(result["source"].tolist(), type=pa.string()),
            "source_verified": pa.array(result["source_verified"].tolist(), type=pa.bool_()),
            "is_synthetic": pa.array(result["is_synthetic"].tolist(), type=pa.bool_()),
            "is_auction_placeholder": pa.array(result["is_auction_placeholder"].tolist(), type=pa.bool_()),
            "opening_auction_volume": pa.array(result["opening_auction_volume"], type=pa.float64()),
            "adj_factor": pa.array(result["adj_factor"], type=pa.float64()),
            "metadata": pa.array(metadata_values, type=pa.string()),
        }
        table = pa.Table.from_arrays(
            [arrays[field.name] for field in PARQUET_SCHEMA], schema=PARQUET_SCHEMA
        )
        temporary = self.staging_dir / f"{uuid.uuid4().hex}.parquet.writing"
        try:
            pq.write_table(
                table, temporary, compression="zstd", use_dictionary=True,
                write_statistics=True, row_group_size=16384, version="2.6",
            )
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            root = Path(base_dir) if base_dir is not None else self.market_dir
            folder = (
                root / timeframe / f"symbol={symbol}" /
                f"adjustment={adjustment}" / period_key
            )
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{digest[:24]}.parquet"
            if target.exists():
                temporary.unlink(missing_ok=True)
            else:
                temporary.replace(target)
            size = target.stat().st_size
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return {
            "partition_key": period_key, "path": self.package.relative(target),
            "sha256": digest, "row_count": len(result),
            "first_time": _time_text(result.timestamp.min()),
            "last_time": _time_text(result.timestamp.max()),
            "size": size, "schema_version": PARQUET_SCHEMA_VERSION,
        }

    def _backup_and_point_old_version(
        self, symbol: str, timeframe: str, adjustment: str,
        version: int, files: Sequence[dict],
    ) -> Path | None:
        if not files:
            return None
        source = self._absolute_source_path(files[0]["path"])
        if not source.is_file():
            raise MarketDataError("待更新的行情源文件不存在：" + str(source))
        backup_folder = self.package.history_dir / "files"
        backup_folder.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            f"{source}|{source.stat().st_size}|{source.stat().st_mtime_ns}".encode()
        ).hexdigest()[:20]
        backup = backup_folder / f"{digest}.parquet"
        if not backup.exists():
            temporary = backup.with_name(backup.name + ".writing")
            shutil.copy2(source, temporary)
            os.replace(temporary, backup)
        with self._connect() as db:
            db.execute(
                """UPDATE parquet_files SET path=?,sha256=?,size=?
                   WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?""",
                (
                    self.package.relative(backup),
                    f"stat:{backup.stat().st_size}:{backup.stat().st_mtime_ns}",
                    backup.stat().st_size,
                    symbol, timeframe, adjustment, version,
                ),
            )
        return backup

    def _write_source_data(
        self, symbol: str, timeframe: str, adjustment: str, version: int,
        frame: pd.DataFrame, old_files: Sequence[dict],
    ) -> dict:
        """Atomically merge one security into its package Parquet file."""
        source = (
            self._absolute_source_path(old_files[0]["path"])
            if old_files and self._is_package_source(old_files[0])
            else self.package_file_path(symbol, timeframe)
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        transaction_id, transaction_path = self.package.begin_transaction({
            "target": self.package.relative(source),
            "version": int(version),
            "symbol": symbol,
            "timeframe": timeframe,
        })
        backup = None
        if old_files and self._is_package_source(old_files[0]):
            backup = self._backup_and_point_old_version(
                symbol, timeframe, adjustment, max(0, int(version) - 1), old_files,
            )
            if backup is not None:
                self.package.update_transaction(transaction_path, {
                    "status": "backed_up",
                    "backup": self.package.relative(backup),
                })
        result = self._prepare_frame(frame).sort_values(
            "timestamp", kind="stable"
        ).reset_index(drop=True)
        temporary = source.with_name(source.name + ".writing")
        completed = False
        try:
            if timeframe == "1day":
                new_rows = pd.DataFrame({
                    "ts_code": symbol,
                    "trade_date": result.timestamp.dt.normalize(),
                    "open": result.open, "high": result.high, "low": result.low,
                    "close": result.close,
                    "pre_close": result.preclose,
                    "change": result.close - result.preclose,
                    "pct_chg": (
                        (result.close - result.preclose) /
                        result.preclose.replace(0, np.nan) * 100
                    ),
                    "vol": result.volume / 100.0,
                    "amount": result.turnover / 1000.0,
                    "adj_factor": result.adj_factor,
                    "exchange": symbol.rsplit(".", 1)[-1],
                    "tradestatus": result.tradestatus,
                    "is_st": result.is_st,
                    "source": result.source,
                    "source_verified": result.source_verified,
                    "is_synthetic": result.is_synthetic,
                    "is_auction_placeholder": result.is_auction_placeholder,
                    "opening_auction_volume": result.opening_auction_volume,
                })
                with duckdb.connect() as connection:
                    connection.register("incoming_daily", new_rows)
                    source_sql = str(source).replace("'", "''") if source.exists() else None
                    if source_sql:
                        query = (
                            "SELECT * FROM read_parquet(?) WHERE ts_code <> ? "
                            "UNION ALL BY NAME SELECT * FROM incoming_daily "
                            "ORDER BY ts_code, trade_date"
                        )
                        connection.execute(
                            "COPY (" + query + ") TO '" + str(temporary).replace("'", "''") +
                            "' (FORMAT PARQUET, COMPRESSION ZSTD)",
                            [str(source), symbol],
                        )
                    else:
                        connection.execute(
                            "COPY (SELECT * FROM incoming_daily ORDER BY trade_date) TO '" +
                            str(temporary).replace("'", "''") +
                            "' (FORMAT PARQUET, COMPRESSION ZSTD)"
                        )
            else:
                output = pd.DataFrame({
                    "ts_code": symbol,
                    "trade_time": result.timestamp,
                    "trade_date": result.timestamp.dt.normalize(),
                    "open": result.open, "high": result.high, "low": result.low,
                    "close": result.close, "vol": result.volume,
                    "amount": result.turnover, "adj_factor": result.adj_factor,
                    "preclose": result.preclose,
                    "tradestatus": result.tradestatus,
                    "is_st": result.is_st,
                    "source": result.source,
                    "source_verified": result.source_verified,
                    "is_synthetic": result.is_synthetic,
                    "is_auction_placeholder": result.is_auction_placeholder,
                    "opening_auction_volume": result.opening_auction_volume,
                })
                output.to_parquet(temporary, compression="zstd", index=False)
            last_error = None
            for attempt in range(6):
                try:
                    os.replace(temporary, source)
                    last_error = None
                    break
                except PermissionError as exc:
                    last_error = exc
                    time.sleep(0.15 * (attempt + 1))
            if last_error is not None:
                raise last_error
            relative = self.package.relative(source)
            self.package.update_transaction(transaction_path, {
                "status": "installed",
                "new_sha256": f"stat:{source.stat().st_size}:{source.stat().st_mtime_ns}",
            })
            if backup is not None and timeframe == "1day":
                self._propagate_shared_file_version(
                    symbol, timeframe, relative, backup, int(version), source
                )
            asset = asset_type_for_symbol(symbol)
            self.package.set_file_metadata(relative, {
                "schema_version": f"{asset}-source-daily" if timeframe == "1day" else f"{asset}-source-minute",
                "label_convention": "end" if bool(result.source_verified.all()) else "auto",
                "volume_multiplier": 100.0 if timeframe == "1day" else 1.0,
                "turnover_multiplier": 1000.0 if timeframe == "1day" else 1.0,
                "source_verified": bool(result.source_verified.all()),
                "parser_version": PARSER_VERSION,
                "version": int(version),
                "row_count": len(result) if not source.exists() else None,
            })
            self.package.record_file(
                relative,
                version=int(version),
                row_count=int(len(result)),
                sha256=f"stat:{source.stat().st_size}:{source.stat().st_mtime_ns}",
            )
            completed = True
        finally:
            temporary.unlink(missing_ok=True)
        signature = f"stat:{source.stat().st_size}:{source.stat().st_mtime_ns}"
        asset = asset_type_for_symbol(symbol)
        return {
            "partition_key": f"source={asset}_{timeframe}",
            "path": self.package.relative(source),
            "sha256": signature,
            "row_count": len(result),
            "first_time": _time_text(result.timestamp.min()),
            "last_time": _time_text(result.timestamp.max()),
            "size": source.stat().st_size,
            "schema_version": (
                f"{asset}-source-daily" if timeframe == "1day" else f"{asset}-source-minute"
            ),
            "_transaction_path": str(transaction_path) if completed else None,
        }

    def _propagate_shared_file_version(
        self, symbol: str, timeframe: str, relative: str, backup: Path,
        incoming_version: int, source: Path,
    ) -> None:
        """Freeze every old dataset reference and point current ones at a new shared file."""
        backup_relative = self.package.relative(backup)
        size = source.stat().st_size
        signature = f"stat:{size}:{source.stat().st_mtime_ns}"
        now = utc_now()
        with self._connect() as db:
            datasets = db.execute(
                "SELECT * FROM parquet_datasets WHERE timeframe=? AND symbol<>?",
                (timeframe, symbol),
            ).fetchall()
            for dataset in datasets:
                current = int(dataset["current_version"] or 0)
                files = self._current_files(
                    db, dataset["symbol"], timeframe, dataset["adjustment"], current
                )
                if not files or not any(str(row.get("path")) == relative for row in files):
                    continue
                db.execute(
                    """UPDATE parquet_files SET path=?,sha256=?,size=?
                       WHERE symbol=? AND timeframe=? AND adjustment=? AND path=? AND version<=?""",
                    (
                        backup_relative,
                        f"stat:{backup.stat().st_size}:{backup.stat().st_mtime_ns}",
                        backup.stat().st_size,
                        dataset["symbol"], timeframe, dataset["adjustment"],
                        relative, current,
                    ),
                )
                next_version = current + 1
                exists = db.execute(
                    """SELECT 1 FROM parquet_versions WHERE symbol=? AND timeframe=?
                       AND adjustment=? AND version=?""",
                    (dataset["symbol"], timeframe, dataset["adjustment"], next_version),
                ).fetchone()
                if exists:
                    continue
                original = db.execute(
                    """SELECT * FROM parquet_versions WHERE symbol=? AND timeframe=?
                       AND adjustment=? AND version=?""",
                    (dataset["symbol"], timeframe, dataset["adjustment"], current),
                ).fetchone()
                if original:
                    values = dict(original)
                    values["version"] = next_version
                    values["created_at"] = now
                    db.execute(
                        """INSERT INTO parquet_versions(
                           symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                           synthetic_rows,pending_rows,first_time,last_time,status,source,
                           file_count,bytes,report
                           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            values["symbol"], values["timeframe"], values["adjustment"],
                            next_version, values["created_at"], values["row_count"],
                            values["raw_rows"], values["synthetic_rows"], values["pending_rows"],
                            values["first_time"], values["last_time"], values["status"],
                            values["source"], values["file_count"], size, values["report"],
                        ),
                    )
                for item in files:
                    db.execute(
                        """INSERT OR REPLACE INTO parquet_files(
                           symbol,timeframe,adjustment,version,partition_key,path,sha256,
                           row_count,first_time,last_time,size,schema_version
                           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            dataset["symbol"], timeframe, dataset["adjustment"], next_version,
                            item["partition_key"], relative, signature, item["row_count"],
                            item["first_time"], item["last_time"], size, item["schema_version"],
                        ),
                    )
                db.execute(
                    "UPDATE parquet_datasets SET current_version=? WHERE symbol=? AND timeframe=? AND adjustment=?",
                    (next_version, dataset["symbol"], timeframe, dataset["adjustment"]),
                )

    def commit_daily_batch(self, prepared: Sequence[dict]) -> list[dict]:
        """Merge one daily download batch into each shared daily file once."""
        grouped: dict[str, list[dict]] = {}
        for item in prepared:
            symbol = item["symbol"]
            existing = self.dataset(symbol, "1day")
            path = (
                self._absolute_source_path(existing["storage_path"])
                if existing and existing.get("storage") == "parquet-source"
                else self.package_file_path(symbol, "1day")
            )
            grouped.setdefault(self.package.relative(path), []).append(item)
        results: list[dict] = []
        for relative, items in grouped.items():
            results.extend(self._commit_daily_file_batch(relative, items))
        return results

    def _commit_daily_file_batch(self, relative: str, prepared: Sequence[dict]) -> list[dict]:
        source = self.package.absolute(relative)
        source.parent.mkdir(parents=True, exist_ok=True)
        with self.file_lock(source):
            changes = []
            for item in prepared:
                symbol = item["symbol"]
                with self._connect() as db:
                    row = self._dataset_row(db, symbol, "1day", "none")
                    dataset = dict(row) if row else None
                    current_version = int(dataset["current_version"]) if dataset else 0
                    old_files = self._current_files(
                        db, symbol, "1day", "none", current_version
                    )
                existing = (
                    self._read_version(
                        symbol, "1day", "none", current_version,
                        use_cache=False, file_lock=False,
                    ) if old_files else pd.DataFrame()
                )
                existing_by_ts = {
                    pd.Timestamp(row["timestamp"]): row
                    for row in existing.to_dict("records")
                }
                frame = self._prepare_frame(item["frame"])
                inserted, updated, unchanged, conflicts = [], [], 0, []
                if not existing_by_ts:
                    combined = frame.copy()
                    inserted = frame.to_dict("records")
                else:
                    for incoming in frame.to_dict("records"):
                        stamp = pd.Timestamp(incoming["timestamp"])
                        old = existing_by_ts.get(stamp)
                        if old is None:
                            inserted.append(incoming)
                            continue
                        if self._same_market_row(old, incoming):
                            unchanged += 1
                            continue
                        if self._quality_rank(
                            incoming.get("source"), incoming.get("source_verified"),
                            incoming.get("is_synthetic"),
                        ) < self._quality_rank(
                            old.get("source"), old.get("source_verified"), old.get("is_synthetic")
                        ):
                            unchanged += 1
                            conflicts.append({"timestamp": _time_text(stamp)})
                            continue
                        updated.append(incoming)
                    for incoming in [*inserted, *updated]:
                        existing_by_ts[pd.Timestamp(incoming["timestamp"])] = incoming
                    combined = self._prepare_frame(pd.DataFrame(existing_by_ts.values()))
                changed = len(inserted) + len(updated)
                pending_after = int(combined.source_verified.eq(False).sum()) if "source_verified" in combined else 0
                pending_flags = bool(item["report"].get("pending_validation"))
                status = "pending_validation" if pending_after or pending_flags else "ready"
                if item["report"].get("gaps"):
                    status = "partial"
                old_status = dataset.get("status") if dataset else None
                semantic = changed > 0 or dataset is None or status != old_status
                version = current_version + 1 if semantic else current_version
                changes.append({
                    **item, "dataset": dataset, "old_files": old_files,
                    "current_version": current_version, "version": version,
                    "combined": combined, "inserted": len(inserted),
                    "updated": len(updated), "unchanged": unchanged,
                    "conflicts": conflicts, "status": status,
                    "pending": pending_after, "semantic": semantic,
                })
            changing = [item for item in changes if item["semantic"] and item["combined"].shape[0]]
            if changing:
                incoming = pd.concat(
                    [self._daily_source_frame(item["combined"], item["symbol"]) for item in changing],
                    ignore_index=True,
                )
                backup = None
                if source.is_file():
                    backup = self._backup_daily_file(source)
                temporary = source.with_name(source.name + ".writing")
                symbols = [item["symbol"] for item in changing]
                if source.is_file():
                    placeholders = ",".join("?" for _ in symbols)
                    with duckdb.connect() as connection:
                        connection.register("incoming_daily_batch", incoming)
                        connection.execute(
                            "COPY (SELECT * FROM read_parquet(?) WHERE ts_code NOT IN (" + placeholders +
                            ") UNION ALL BY NAME SELECT * FROM incoming_daily_batch ORDER BY ts_code,trade_date) TO '" +
                            str(temporary).replace("'", "''") +
                            "' (FORMAT PARQUET, COMPRESSION ZSTD)",
                            [str(source), *symbols],
                        )
                else:
                    incoming.sort_values(["ts_code", "trade_date"], kind="stable").to_parquet(
                        temporary, compression="zstd", index=False
                    )
                os.replace(temporary, source)
                self.package.record_file(
                    relative, sha256=f"stat:{source.stat().st_size}:{source.stat().st_mtime_ns}",
                    metadata={"schema_version": source.name.split("_", 1)[0] + "-source-daily",
                              "label_convention": "end", "volume_multiplier": 100.0,
                              "turnover_multiplier": 1000.0, "batch_size": len(changing)},
                )
                self._publish_daily_batch_db(relative, source, changes, changing, backup)
            else:
                self._refresh_unchanged_daily_results(changes)
        return [
            {
                "symbol": item["symbol"], "timeframe": "1day", "status": item["status"],
                "version": item["version"], "rows_read": int(item["combined"].shape[0]),
                "inserted": item["inserted"], "updated": item["updated"],
                "unchanged": item["unchanged"], "conflicts": item.get("conflicts", []),
                "first_time": _time_text(item["combined"].timestamp.min()),
                "last_time": _time_text(item["combined"].timestamp.max()),
                "message": f"新增 {item['inserted']} 行，修正 {item['updated']} 行，未变化 {item['unchanged']} 行",
            }
            for item in changes
        ]

    def _backup_daily_file(self, source: Path) -> Path:
        digest = hashlib.sha256(
            f"{source}|{source.stat().st_size}|{source.stat().st_mtime_ns}".encode()
        ).hexdigest()[:20]
        backup = self.package.history_dir / "files" / f"{digest}.parquet"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            temporary = backup.with_name(backup.name + ".writing")
            shutil.copy2(source, temporary)
            os.replace(temporary, backup)
        return backup

    @staticmethod
    def _daily_source_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        result = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        return pd.DataFrame({
            "ts_code": symbol, "trade_date": result.timestamp.dt.normalize(),
            "open": result.open, "high": result.high, "low": result.low,
            "close": result.close, "pre_close": result.preclose,
            "change": result.close - result.preclose,
            "pct_chg": (result.close - result.preclose) / result.preclose.replace(0, np.nan) * 100,
            "vol": result.volume / 100.0, "amount": result.turnover / 1000.0,
            "adj_factor": result.adj_factor, "exchange": symbol.rsplit(".", 1)[-1],
            "tradestatus": result.tradestatus, "is_st": result.is_st,
            "source": result.source, "source_verified": result.source_verified,
            "is_synthetic": result.is_synthetic,
            "is_auction_placeholder": result.is_auction_placeholder,
            "opening_auction_volume": result.opening_auction_volume,
        })

    def _publish_daily_batch_db(self, relative, source, changes, changing, backup):
        signature = f"stat:{source.stat().st_size}:{source.stat().st_mtime_ns}"
        now = utc_now()
        changing_symbols = {item["symbol"] for item in changing}
        with self._connect() as db:
            users = db.execute(
                """SELECT d.* FROM parquet_datasets d JOIN parquet_files f
                   ON f.symbol=d.symbol AND f.timeframe=d.timeframe
                   AND f.adjustment=d.adjustment AND f.version=d.current_version
                   WHERE d.timeframe='1day' AND f.path=?""",
                (relative,),
            ).fetchall()
            file_rows = db.execute(
                """SELECT f.* FROM parquet_files f JOIN parquet_datasets d
                   ON d.symbol=f.symbol AND d.timeframe=f.timeframe
                   AND d.adjustment=f.adjustment AND d.current_version=f.version
                   WHERE d.timeframe='1day' AND f.path=?""",
                (relative,),
            ).fetchall()
            user_files: dict[tuple[str, str, int], list[dict]] = {}
            for file_row in file_rows:
                user_files.setdefault(
                    (file_row["symbol"], file_row["adjustment"], int(file_row["version"])),
                    [],
                ).append(dict(file_row))
            db.execute("BEGIN IMMEDIATE")
            if backup is not None:
                db.execute(
                    "UPDATE parquet_files SET path=?,sha256=?,size=? WHERE path=?",
                    (self.package.relative(backup),
                     f"stat:{backup.stat().st_size}:{backup.stat().st_mtime_ns}",
                     backup.stat().st_size, relative),
                )
            for user in users:
                symbol = user["symbol"]
                if symbol in changing_symbols:
                    continue
                current_version = int(user["current_version"] or 0)
                old_files = user_files.get((symbol, user["adjustment"], current_version), [])
                if not old_files:
                    continue
                next_version = current_version + 1
                original = db.execute(
                    "SELECT * FROM parquet_versions WHERE symbol=? AND timeframe='1day' AND adjustment='none' AND version=?",
                    (symbol, current_version),
                ).fetchone()
                if original and not db.execute(
                    "SELECT 1 FROM parquet_versions WHERE symbol=? AND timeframe='1day' AND adjustment='none' AND version=?",
                    (symbol, next_version),
                ).fetchone():
                    values = dict(original)
                    db.execute(
                        """INSERT INTO parquet_versions(
                           symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                           synthetic_rows,pending_rows,first_time,last_time,status,source,
                           file_count,bytes,report) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (values["symbol"], "1day", "none", next_version, now, values["row_count"],
                         values["raw_rows"], values["synthetic_rows"], values["pending_rows"],
                         values["first_time"], values["last_time"], values["status"],
                         values["source"], 1, source.stat().st_size, values["report"]),
                    )
                    for old in old_files:
                        db.execute(
                            """INSERT OR REPLACE INTO parquet_files(
                               symbol,timeframe,adjustment,version,partition_key,path,sha256,
                               row_count,first_time,last_time,size,schema_version
                               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (symbol, "1day", "none", next_version,
                             old["partition_key"], relative, signature, old["row_count"],
                             old["first_time"], old["last_time"], source.stat().st_size,
                             old["schema_version"]),
                        )
                    db.execute(
                        "UPDATE parquet_datasets SET current_version=? WHERE symbol=? AND timeframe='1day' AND adjustment='none'",
                        (next_version, symbol),
                    )
                    for item in changes:
                        if item["symbol"] == symbol:
                            item["version"] = next_version
            for item in changing:
                combined = item["combined"]
                first_time = _time_text(combined.timestamp.min())
                last_time = _time_text(combined.timestamp.max())
                row_count = len(combined)
                synthetic = int(combined.is_synthetic.sum()) if "is_synthetic" in combined else 0
                item_metadata = {**(_load(item["dataset"]["metadata"], {}) if item["dataset"] else {}), **item["metadata"]}
                db.execute(
                    """INSERT OR REPLACE INTO parquet_versions(
                       symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                       synthetic_rows,pending_rows,first_time,last_time,status,source,
                       file_count,bytes,report) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (item["symbol"], "1day", "none", item["version"], now, row_count,
                     row_count - synthetic, synthetic, item["pending"], first_time, last_time,
                     item["status"], item["source"], 1, source.stat().st_size,
                     _json(item["report"])),
                )
                db.execute(
                    """INSERT OR REPLACE INTO parquet_files(
                       symbol,timeframe,adjustment,version,partition_key,path,sha256,
                       row_count,first_time,last_time,size,schema_version
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (item["symbol"], "1day", "none", item["version"],
                     f"source={asset_type_for_symbol(item['symbol'])}_1day", relative, signature,
                     row_count, first_time, last_time, source.stat().st_size,
                     f"{asset_type_for_symbol(item['symbol'])}-source-daily"),
                )
                if item["dataset"]:
                    db.execute(
                        """UPDATE parquet_datasets SET current_version=?,first_time=?,last_time=?,
                           completed_through=?,row_count=?,raw_rows=?,synthetic_rows=?,pending_rows=?,
                           status=?,reason=?,metadata=?,updated_at=?
                           WHERE symbol=? AND timeframe='1day' AND adjustment='none'""",
                        (item["version"], first_time, last_time, last_time, row_count,
                         row_count - synthetic, synthetic, item["pending"], item["status"],
                         item["report"].get("reason"), _json(item_metadata), now, item["symbol"]),
                    )
                else:
                    db.execute(
                        """INSERT INTO parquet_datasets(
                           symbol,timeframe,adjustment,current_version,first_time,last_time,
                           completed_through,row_count,raw_rows,synthetic_rows,pending_rows,
                           status,reason,metadata,created_at,updated_at
                           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (item["symbol"], "1day", "none", item["version"], first_time,
                         last_time, last_time, row_count, row_count - synthetic, synthetic,
                         item["pending"], item["status"], item["report"].get("reason"),
                         _json(item_metadata), now, now),
                    )
                db.execute(
                    """INSERT OR REPLACE INTO parquet_imports(
                       symbol,timeframe,adjustment,fingerprint,params_hash,parser_version,
                       file_name,status,rows_read,inserted,updated,unchanged,synthetic_rows,
                       version,report,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (item["symbol"], "1day", "none", item["fingerprint"],
                     item["params_hash"], PARSER_VERSION, f"sync:{item['source']}",
                     "completed", len(item["frame"]), item["inserted"], item["updated"],
                     item["unchanged"], synthetic, item["version"],
                     _json({**item["report"], "conflicts": item["conflicts"]}), now),
                )
            db.commit()

    def _refresh_unchanged_daily_results(self, changes):
        for item in changes:
            active = self.dataset(item["symbol"], "1day")
            if active:
                item["version"] = active["current_version"]

    @staticmethod
    def _quality_rank(source_name, verified, synthetic) -> int:
        if bool(synthetic):
            return 0
        if str(source_name or "").lower() in {"aggregate", "derived"}:
            return 1
        return 3 if bool(verified) else 2

    @staticmethod
    def _same_market_row(old: dict, new: dict) -> bool:
        for field in (
            "open", "high", "low", "close", "volume", "turnover", "preclose",
            "tradestatus", "is_st", "source_verified", "is_synthetic",
            "is_auction_placeholder",
        ):
            left, right = old.get(field), new.get(field)
            left_null = left is None or (not isinstance(left, (bool, str)) and pd.isna(left))
            right_null = right is None or (not isinstance(right, (bool, str)) and pd.isna(right))
            if left_null and right_null:
                continue
            if left_null != right_null:
                return False
            if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
                if bool(left) != bool(right):
                    return False
                continue
            if isinstance(left, (float, int, np.number)) or isinstance(right, (float, int, np.number)):
                try:
                    if not np.isclose(float(left), float(right), rtol=0, atol=1e-9):
                        return False
                except (TypeError, ValueError):
                    return False
            elif left != right:
                return False
        return True

    def _commit_incremental_bars(
        self, symbol: str, timeframe: str, frame: pd.DataFrame, **kwargs,
    ) -> dict:
        with self.lock(symbol, timeframe, kwargs.get("adjustment", "none")):
            return self._commit_incremental_bars_unlocked(
                symbol, timeframe, frame, **kwargs
            )

    def _commit_incremental_bars_unlocked(
        self, symbol: str, timeframe: str, frame: pd.DataFrame, *,
        adjustment: str, source: str, file_name: str | None,
        fingerprint: str | None, params_hash: str, report: dict,
        metadata: dict, completed_through: str | None,
        advance_completed: bool, derived: dict | None,
    ) -> dict:
        """Publish additions and corrections without rewriting historical base files."""
        with self._connect() as db:
            dataset_row = self._dataset_row(db, symbol, timeframe, adjustment)
            if not dataset_row:
                raise MarketDataError("增量写入需要已有行情底座")
            dataset = dict(dataset_row)
            current_version = int(dataset["current_version"])
            base_files = self._current_files(
                db, symbol, timeframe, adjustment, current_version
            )
            increments = self._current_increments(db, symbol, timeframe, adjustment, current_version)

        periods: dict[str, tuple[str, str]] = {}
        for stamp in frame.timestamp:
            key, start, end = self._storage_period(timeframe, stamp)
            periods[key] = (start, end)
        ordered = sorted(periods.items())
        read_start = pd.Timestamp(min(value[0] for _, value in ordered))
        read_end = pd.Timestamp(max(value[1] for _, value in ordered)) - pd.Timedelta(nanoseconds=1)
        existing = self._read_version(
            symbol, timeframe, adjustment, current_version,
            start=read_start, end=read_end + pd.Timedelta(nanoseconds=1),
            use_cache=False, file_lock=False,
        )
        existing_by_period: dict[str, pd.DataFrame] = {}
        if not existing.empty:
            for key, (start, end) in periods.items():
                part = existing[
                    (existing.timestamp >= pd.Timestamp(start))
                    & (existing.timestamp < pd.Timestamp(end))
                ]
                existing_by_period[key] = part

        changed_periods: dict[str, pd.DataFrame] = {}
        all_inserted = all_updated = 0
        unchanged = 0
        conflicts: list[dict] = []
        changed_dates: set[str] = set()
        row_delta = 0
        synthetic_delta = 0
        pending_delta = 0
        new_first = pd.Timestamp(dataset["first_time"]) if dataset.get("first_time") else None
        new_last = pd.Timestamp(dataset["last_time"]) if dataset.get("last_time") else None
        for key, (start, end) in ordered:
            incoming = frame[
                (frame.timestamp >= pd.Timestamp(start))
                & (frame.timestamp < pd.Timestamp(end))
            ]
            if incoming.empty:
                continue
            current = existing_by_period.get(key, pd.DataFrame())
            if timeframe in {"1week", "1month"} and not current.empty:
                period_name = "W-FRI" if timeframe == "1week" else "M"
                incoming_periods = set(incoming.timestamp.dt.to_period(period_name))
                current = current[
                    ~current.timestamp.dt.to_period(period_name).isin(incoming_periods)
                ]
            combined, inserted, updated, same, period_conflicts = self._merge_market_frames(
                current, incoming
            )
            unchanged += same
            conflicts.extend(period_conflicts)
            if not inserted and not updated:
                continue
            all_inserted += len(inserted)
            all_updated += len(updated)
            changed_periods[key] = combined
            row_delta += len(combined) - len(current)
            old_synthetic = int(current.is_synthetic.sum()) if "is_synthetic" in current else 0
            new_synthetic = int(combined.is_synthetic.sum()) if "is_synthetic" in combined else 0
            synthetic_delta += new_synthetic - old_synthetic
            old_pending = int(current.source_verified.eq(False).sum()) if "source_verified" in current else 0
            new_pending = int(combined.source_verified.eq(False).sum()) if "source_verified" in combined else 0
            pending_delta += new_pending - old_pending
            for row in [*inserted, *updated]:
                changed_dates.add(str(pd.Timestamp(row["timestamp"]).date()))
            new_first = combined.timestamp.min() if new_first is None else min(new_first, combined.timestamp.min())
            new_last = combined.timestamp.max() if new_last is None else max(new_last, combined.timestamp.max())

        if not changed_periods:
            return {
                "symbol": symbol, "timeframe": timeframe, "version": current_version,
                "rows_read": len(frame), "inserted": 0, "updated": 0,
                "unchanged": unchanged, "removed": 0, "synthetic": 0,
                "status": dataset.get("status"), "conflicts": conflicts,
                "file_count": len(increments), "bytes": 0, "changed_dates": [],
            }

        version = current_version + 1
        now = utc_now()
        written: list[dict] = []
        skip_periods = set(changed_periods)
        for index, (key, combined) in enumerate(sorted(changed_periods.items()), start=1):
            written_file = self._write_increment_file(symbol, timeframe, adjustment, key, combined)
            _, start, end = self._storage_period(timeframe, combined.timestamp.iloc[0])
            written.append({
                **written_file, "period_key": key, "period_start": start,
                "period_end": end, "sequence": int(index),
            })

        kept_increments = [
            row for row in increments if row["period_key"] not in skip_periods
        ]
        total_bytes = (
            sum(int(row.get("size") or 0) for row in [*base_files, *kept_increments])
            + sum(int(row["size"]) for row in written)
        )

        row_count = max(0, int(dataset.get("row_count") or 0) + row_delta)
        synthetic_rows = max(0, int(dataset.get("synthetic_rows") or 0) + synthetic_delta)
        pending_after = max(0, int(dataset.get("pending_rows") or 0) + pending_delta)
        status = "pending_validation" if pending_after or report.get("pending_validation") else "ready"
        if report.get("gaps"):
            status = "partial"
        completed_value = (
            completed_through if completed_through is not None
            else _time_text(new_last) if advance_completed and new_last is not None
            else dataset.get("completed_through")
        )
        item_metadata = {**_load(dataset.get("metadata"), {}), **(metadata or {}),
                         "layout": "base+increments"}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT OR REPLACE INTO parquet_versions(
                   symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                   synthetic_rows,pending_rows,first_time,last_time,status,source,
                   file_count,bytes,report)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, timeframe, adjustment, version, now, row_count,
                    max(0, row_count - synthetic_rows), synthetic_rows,
                    pending_after, _time_text(new_first), _time_text(new_last), status,
                    source, len(base_files) + len(kept_increments) + len(written),
                    total_bytes,
                    _json(report),
                ),
            )
            self._copy_version_files(
                db, symbol, timeframe, adjustment, current_version, version
            )
            self._copy_current_increments(
                db, symbol, timeframe, adjustment, current_version, version,
                skip_periods=skip_periods,
            )
            for row in written:
                db.execute(
                    """INSERT INTO parquet_increments(
                       symbol,timeframe,adjustment,version,sequence,period_key,
                       period_start,period_end,path,sha256,row_count,first_time,
                       last_time,size,schema_version)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, timeframe, adjustment, version, row["sequence"],
                        row["period_key"], row["period_start"], row["period_end"],
                        row["path"], row["sha256"], row["row_count"],
                        row["first_time"], row["last_time"], row["size"],
                        row["schema_version"],
                    ),
                )
            db.execute(
                """UPDATE parquet_datasets SET current_version=?,first_time=?,last_time=?,
                   completed_through=?,row_count=?,raw_rows=?,synthetic_rows=?,pending_rows=?,
                   status=?,reason=?,metadata=?,updated_at=?
                   WHERE symbol=? AND timeframe=? AND adjustment=?""",
                (
                    version, _time_text(new_first), _time_text(new_last),
                    completed_value, row_count, max(0, row_count - synthetic_rows),
                    synthetic_rows, pending_after, status, report.get("reason"),
                    _json(item_metadata), now, symbol, timeframe, adjustment,
                ),
            )
            if fingerprint:
                db.execute(
                    """INSERT OR REPLACE INTO parquet_imports(
                       symbol,timeframe,adjustment,fingerprint,params_hash,parser_version,
                       file_name,status,rows_read,inserted,updated,unchanged,synthetic_rows,
                       version,report,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, timeframe, adjustment, fingerprint, params_hash,
                        PARSER_VERSION, file_name, "completed", len(frame), all_inserted,
                        all_updated, unchanged, int(frame.is_synthetic.sum()),
                        version, _json({**report, "conflicts": conflicts}), now,
                    ),
                )
            if derived:
                db.execute(
                    """INSERT OR REPLACE INTO parquet_derived(
                       symbol,timeframe,adjustment,source_timeframe,target_timeframe,
                       source_version,start_time,end_time,stale,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,0,?)""",
                    (
                        symbol, timeframe, adjustment, derived["source_timeframe"],
                        timeframe, int(derived["source_version"]),
                        derived.get("start_time"), derived.get("end_time"), now,
                    ),
                )
            db.commit()
        self.invalidate_day_validations(
            symbol, timeframe, sorted(changed_dates), adjustment=adjustment,
        )
        self._mark_derived_stale(symbol, adjustment, timeframe, version)
        compaction = self._compact_dataset_if_needed(
            symbol, timeframe, adjustment, version
        )
        return {
            "symbol": symbol, "timeframe": timeframe, "version": version,
            "rows_read": len(frame), "inserted": all_inserted, "updated": all_updated,
            "unchanged": unchanged, "removed": 0, "synthetic": int(frame.is_synthetic.sum()),
            "status": status, "conflicts": conflicts, "file_count": len(written),
            "bytes": sum(int(row["size"]) for row in written),
            "changed_dates": sorted(changed_dates), "increment_count": len(written),
            "compaction": compaction,
        }

    def _commit_snapshot_bars(
        self, symbol: str, timeframe: str, frame: pd.DataFrame, **kwargs,
    ) -> dict:
        with self.lock(symbol, timeframe, kwargs.get("adjustment", "none")):
            return self._commit_snapshot_bars_unlocked(symbol, timeframe, frame, **kwargs)

    def _commit_snapshot_bars_unlocked(
        self, symbol: str, timeframe: str, frame: pd.DataFrame, *,
        adjustment: str, source: str, file_name: str | None,
        fingerprint: str | None, params_hash: str, report: dict,
        metadata: dict, completed_through: str | None,
        advance_completed: bool, derived: dict | None,
    ) -> dict:
        """Replace the visible base with an immutable snapshot, dropping overlays."""
        with self._connect() as db:
            dataset_row = self._dataset_row(db, symbol, timeframe, adjustment)
            dataset = dict(dataset_row) if dataset_row else None
            current_version = int(dataset["current_version"]) if dataset else 0
            current_files = self._current_files(
                db, symbol, timeframe, adjustment, current_version
            ) if dataset else []
            old_days = {
                pd.Timestamp(row["trade_date"]).normalize()
                for row in db.execute(
                    """SELECT trade_date FROM parquet_day_validation
                       WHERE symbol=? AND timeframe=? AND adjustment=?""",
                    (symbol, timeframe, adjustment),
                ).fetchall()
            }
        incoming_days = set(frame.timestamp.dt.normalize().unique())
        new_hashes = self._frame_day_hashes(frame)
        changed_days = (
            {pd.Timestamp(day).normalize() for day in incoming_days}
            if dataset is None and not old_days else
            {pd.Timestamp(day).normalize() for day in old_days - incoming_days}
        )
        with self._connect() as db:
            records = db.execute(
                """SELECT trade_date,source_hash FROM parquet_day_validation
                   WHERE symbol=? AND timeframe=? AND adjustment=?""",
                (symbol, timeframe, adjustment),
            ).fetchall()
        for record in records:
            day = pd.Timestamp(record["trade_date"]).normalize()
            if new_hashes.get(day) != record["source_hash"]:
                changed_days.add(day)
        validated_days = {
            pd.Timestamp(record["trade_date"]).normalize() for record in records
        }
        changed_days.update(
            pd.Timestamp(day).normalize() for day in incoming_days - validated_days
        )

        if any(self._is_package_source(row) for row in current_files):
            # The old version must keep its own bytes before an external file is
            # replaced by the new snapshot.
            self._backup_and_point_old_version(
                symbol, timeframe, adjustment, current_version, current_files,
            )

        version = current_version + 1
        now = utc_now()
        base = self._write_partition(
            symbol, timeframe, adjustment, 0, frame,
            period="all", base_dir=self.package.bases_dir,
        )
        synthetic_rows = int(frame.is_synthetic.sum())
        pending_after = int(frame.source_verified.eq(False).sum())
        status = "pending_validation" if pending_after or report.get("pending_validation") else "ready"
        if report.get("gaps"):
            status = "partial"
        first_time = frame.timestamp.min()
        last_time = frame.timestamp.max()
        completed_value = completed_through if completed_through is not None else (
            _time_text(last_time) if advance_completed else (dataset or {}).get("completed_through")
        )
        item_metadata = {**((dataset or {}).get("metadata") and _load(dataset["metadata"], {}) or {}),
                         **(metadata or {}), "layout": "immutable-base"}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT OR REPLACE INTO parquet_versions(
                   symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                   synthetic_rows,pending_rows,first_time,last_time,status,source,
                   file_count,bytes,report) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, timeframe, adjustment, version, now, len(frame),
                    len(frame) - synthetic_rows, synthetic_rows, pending_after,
                    _time_text(first_time), _time_text(last_time), status, source,
                    1, int(base["size"]), _json(report),
                ),
            )
            db.execute(
                """INSERT INTO parquet_files(
                   symbol,timeframe,adjustment,version,partition_key,path,sha256,
                   row_count,first_time,last_time,size,schema_version)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, timeframe, adjustment, version, base["partition_key"],
                    base["path"], base["sha256"], base["row_count"],
                    base["first_time"], base["last_time"], base["size"],
                    base["schema_version"],
                ),
            )
            if dataset:
                db.execute(
                    """UPDATE parquet_datasets SET current_version=?,first_time=?,last_time=?,
                       completed_through=?,row_count=?,raw_rows=?,synthetic_rows=?,pending_rows=?,
                       status=?,reason=?,metadata=?,updated_at=?
                       WHERE symbol=? AND timeframe=? AND adjustment=?""",
                    (
                        version, _time_text(first_time), _time_text(last_time),
                        completed_value, len(frame), len(frame) - synthetic_rows,
                        synthetic_rows, pending_after, status, report.get("reason"),
                        _json(item_metadata), now, symbol, timeframe, adjustment,
                    ),
                )
            else:
                db.execute(
                    """INSERT INTO parquet_datasets(
                       symbol,timeframe,adjustment,current_version,first_time,last_time,
                       completed_through,row_count,raw_rows,synthetic_rows,pending_rows,
                       status,reason,metadata,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, timeframe, adjustment, version, _time_text(first_time),
                        _time_text(last_time), completed_value, len(frame),
                        len(frame) - synthetic_rows, synthetic_rows, pending_after, status,
                        report.get("reason"), _json(item_metadata), now, now,
                    ),
                )
            if fingerprint:
                db.execute(
                    """INSERT OR REPLACE INTO parquet_imports(
                       symbol,timeframe,adjustment,fingerprint,params_hash,parser_version,
                       file_name,status,rows_read,inserted,updated,unchanged,synthetic_rows,
                       version,report,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, timeframe, adjustment, fingerprint, params_hash,
                        PARSER_VERSION, file_name, "completed", len(frame), len(frame),
                        0, 0, synthetic_rows, version, _json(report), now,
                    ),
                )
            db.commit()
        self.invalidate_day_validations(
            symbol, timeframe,
            sorted(str(day.date()) for day in changed_days), adjustment=adjustment,
        )
        self._mark_derived_stale(symbol, adjustment, timeframe, version)
        return {
            "symbol": symbol, "timeframe": timeframe, "version": version,
            "rows_read": len(frame), "inserted": len(frame), "updated": 0,
            "unchanged": 0, "removed": 0, "synthetic": synthetic_rows,
            "status": status, "conflicts": [], "file_count": 1,
            "bytes": int(base["size"]), "changed_dates": sorted(
                str(day.date()) for day in changed_days
            ), "layout": "immutable-base",
        }

    @staticmethod
    def _frame_day_hashes(frame: pd.DataFrame) -> dict[pd.Timestamp, str]:
        fields = [
            field for field in (
                "timestamp", "open", "high", "low", "close", "volume", "turnover",
                "preclose", "tradestatus", "is_st", "source", "source_verified",
                "is_synthetic", "is_auction_placeholder",
            ) if field in frame
        ]
        result = {}
        for day, part in frame.groupby(frame.timestamp.dt.normalize(), sort=True):
            canonical = part[fields].sort_values("timestamp").reset_index(drop=True)
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(canonical, index=False).values.tobytes()
            ).hexdigest()
            result[pd.Timestamp(day).normalize()] = digest
        return result

    def commit_bars(
        self, symbol: str, timeframe: str, frame: pd.DataFrame, *,
        adjustment: str = "none", source: str = "import", file_name: str | None = None,
        fingerprint: str | None = None, params_hash: str | None = None,
        report: dict | None = None, metadata: dict | None = None,
        completed_through: str | None = None, derived: dict | None = None,
        replace_range: bool = False, advance_completed: bool = True,
        import_mode: str = "merge",
    ) -> dict:
        timeframe = normalize_timeframe(timeframe)
        frame = self._prepare_frame(frame)
        if frame.empty:
            raise MarketDataError("没有可提交的行情记录")
        params_hash = params_hash or hashlib.sha256(f"{source}|{adjustment}".encode()).hexdigest()
        report = report or {}
        metadata = metadata or {}
        with self._connect() as db:
            existing_dataset = self._dataset_row(db, symbol, timeframe, adjustment)
        if import_mode not in {"merge", "snapshot"}:
            raise MarketDataError("import_mode 应为 merge 或 snapshot")
        if import_mode == "snapshot":
            return self._commit_snapshot_bars(
                symbol, timeframe, frame, adjustment=adjustment, source=source,
                file_name=file_name, fingerprint=fingerprint, params_hash=params_hash,
                report=report, metadata=metadata, completed_through=completed_through,
                advance_completed=advance_completed, derived=derived,
            )
        if existing_dataset is not None and not replace_range:
            return self._commit_incremental_bars(
                symbol, timeframe, frame, adjustment=adjustment, source=source,
                file_name=file_name, fingerprint=fingerprint, params_hash=params_hash,
                report=report, metadata=metadata, completed_through=completed_through,
                advance_completed=advance_completed, derived=derived,
            )
        with self.lock(symbol, timeframe, adjustment):
            with self._connect() as db:
                dataset_row = self._dataset_row(db, symbol, timeframe, adjustment)
                dataset = dict(dataset_row) if dataset_row else None
                current_version = int(dataset["current_version"]) if dataset else 0
                old_files = self._current_files(db, symbol, timeframe, adjustment, current_version)
            incoming_years = {int(year) for year in frame.timestamp.dt.year.unique()}
            affected_years = set(incoming_years)
            if timeframe in {"1week", "1month"}:
                for year in incoming_years:
                    affected_years.update((year - 1, year, year + 1))
            existing = pd.DataFrame()
            if old_files and affected_years:
                existing = self._read_version(
                    symbol, timeframe, adjustment, current_version,
                    years=affected_years, use_cache=False, file_lock=False,
                )
            existing_by_ts: dict[pd.Timestamp, dict] = {}
            if not existing.empty:
                for row in existing.to_dict("records"):
                    existing_by_ts[pd.Timestamp(row["timestamp"])] = row
            removals: set[pd.Timestamp] = set()
            incoming_days = set(frame.timestamp.dt.normalize())
            if replace_range:
                removals.update(
                    stamp for stamp in existing_by_ts
                    if pd.Timestamp(stamp).normalize() in incoming_days
                )
            if timeframe in {"1week", "1month"} and not replace_range:
                period_name = "W-FRI" if timeframe == "1week" else "M"
                incoming_periods = set(frame.timestamp.dt.to_period(period_name))
                incoming_timestamps = set(frame.timestamp)
                for stamp in existing_by_ts:
                    if stamp in incoming_timestamps:
                        continue
                    if pd.Timestamp(stamp).to_period(period_name) in incoming_periods:
                        removals.add(stamp)
            inserted: list[dict] = []
            updated: list[dict] = []
            unchanged = 0
            conflicts: list[dict] = []
            new_dataset = dataset is None
            if new_dataset and not existing_by_ts:
                # First import needs no per-row Python dictionary copies. This
                # path is used by the million-row ETF minute files.
                combined = frame.copy()
            else:
                frame_records = frame.to_dict("records")
                for row in frame_records:
                    stamp = pd.Timestamp(row["timestamp"])
                    old = existing_by_ts.get(stamp)
                    if old is None:
                        inserted.append(row)
                        continue
                    if self._same_market_row(old, row):
                        unchanged += 1
                        continue
                    old_rank = self._quality_rank(
                        old.get("source"), old.get("source_verified"), old.get("is_synthetic")
                    )
                    new_rank = self._quality_rank(
                        row.get("source"), row.get("source_verified"), row.get("is_synthetic")
                    )
                    if new_rank < old_rank:
                        unchanged += 1
                        conflicts.append({
                            "timestamp": _time_text(stamp),
                            "reason": "较低优先级记录不能覆盖已核验或原始行情",
                        })
                        continue
                    updated.append(row)
                for stamp in removals:
                    existing_by_ts.pop(stamp, None)
                for row in [*inserted, *updated]:
                    existing_by_ts[pd.Timestamp(row["timestamp"])] = row
                combined = pd.DataFrame(existing_by_ts.values())
                if not combined.empty:
                    combined = self._prepare_frame(combined)
            removed_count = len(removals)
            changed = len(frame) if new_dataset else len(inserted) + len(updated) + removed_count
            old_status = dataset.get("status") if dataset else None
            pending_flags = bool(report.get("pending_validation"))
            gaps = bool(report.get("gaps"))
            if new_dataset:
                pending_after = (
                    int(combined["source_verified"].eq(False).sum())
                    if not combined.empty and "source_verified" in combined else 0
                )
                synthetic_rows = (
                    int(combined["is_synthetic"].sum()) if not combined.empty else 0
                )
            elif changed > 0:
                previous_pending = int(dataset.get("pending_rows") or 0)
                previous_synthetic = int(dataset.get("synthetic_rows") or 0)
                old_pending = (
                    int(existing["source_verified"].eq(False).sum())
                    if not existing.empty and "source_verified" in existing else 0
                )
                old_synthetic = (
                    int(existing["is_synthetic"].sum()) if not existing.empty else 0
                )
                new_pending = (
                    int(combined["source_verified"].eq(False).sum())
                    if not combined.empty and "source_verified" in combined else 0
                )
                new_synthetic = (
                    int(combined["is_synthetic"].sum()) if not combined.empty else 0
                )
                pending_after = max(0, previous_pending - old_pending + new_pending)
                synthetic_rows = max(0, previous_synthetic - old_synthetic + new_synthetic)
            else:
                pending_after = int(dataset.get("pending_rows") or 0)
                synthetic_rows = int(dataset.get("synthetic_rows") or 0)
            status = "pending_validation" if pending_after or pending_flags else "ready"
            if gaps:
                status = "partial"
            now = utc_now()
            semantic_change = changed > 0 or new_dataset or status != old_status
            version = current_version + 1 if semantic_change else current_version
            files = list(old_files)
            if semantic_change and changed > 0:
                files = [
                    self._write_source_data(
                        symbol, timeframe, adjustment, version,
                        combined, old_files,
                    )
                ]
            if not files and semantic_change:
                raise MarketDataError("行情提交后没有可发布的分区")
            first_time = min(
                (row["first_time"] for row in files if row.get("first_time")), default=None
            )
            last_time = max(
                (row["last_time"] for row in files if row.get("last_time")), default=None
            )
            row_count = sum(int(row["row_count"]) for row in files)
            raw_rows = row_count - synthetic_rows
            completed_value = (
                completed_through if completed_through is not None
                else last_time if advance_completed
                else (dataset.get("completed_through") if dataset else None)
            )
            item_metadata = {**(_load(dataset["metadata"], {}) if dataset else {}), **metadata}
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if semantic_change:
                    db.execute(
                        """INSERT OR REPLACE INTO parquet_versions(
                           symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                           synthetic_rows,pending_rows,first_time,last_time,status,source,
                           file_count,bytes,report
                           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            symbol, timeframe, adjustment, version, now, row_count, raw_rows,
                            synthetic_rows, pending_after, first_time, last_time, status, source,
                            len(files), sum(int(row["size"]) for row in files), _json(report),
                        ),
                    )
                    db.execute(
                        """DELETE FROM parquet_files WHERE symbol=? AND timeframe=?
                           AND adjustment=? AND version=?""",
                        (symbol, timeframe, adjustment, version),
                    )
                    for row in files:
                        db.execute(
                            """INSERT INTO parquet_files(
                               symbol,timeframe,adjustment,version,partition_key,path,sha256,
                               row_count,first_time,last_time,size,schema_version
                               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                symbol, timeframe, adjustment, version, row["partition_key"],
                                row["path"], row["sha256"], row["row_count"], row["first_time"],
                                row["last_time"], row["size"], row["schema_version"],
                            ),
                        )
                if dataset:
                    db.execute(
                        """UPDATE parquet_datasets SET current_version=?,first_time=?,last_time=?,
                           completed_through=?,row_count=?,raw_rows=?,synthetic_rows=?,pending_rows=?,
                           status=?,reason=?,metadata=?,updated_at=?
                           WHERE symbol=? AND timeframe=? AND adjustment=?""",
                        (
                            version, first_time, last_time, completed_value, row_count, raw_rows,
                            synthetic_rows, pending_after, status, report.get("reason"),
                            _json(item_metadata), now, symbol, timeframe, adjustment,
                        ),
                    )
                else:
                    db.execute(
                        """INSERT INTO parquet_datasets(
                           symbol,timeframe,adjustment,current_version,first_time,last_time,
                           completed_through,row_count,raw_rows,synthetic_rows,pending_rows,
                           status,reason,metadata,created_at,updated_at
                           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            symbol, timeframe, adjustment, version, first_time, last_time,
                            completed_value, row_count, raw_rows, synthetic_rows, pending_after,
                            status, report.get("reason"), _json(item_metadata), now, now,
                        ),
                    )
                if fingerprint:
                    db.execute(
                        """INSERT OR REPLACE INTO parquet_imports(
                           symbol,timeframe,adjustment,fingerprint,params_hash,parser_version,
                           file_name,status,rows_read,inserted,updated,unchanged,synthetic_rows,
                           version,report,updated_at
                           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            symbol, timeframe, adjustment, fingerprint, params_hash, PARSER_VERSION,
                            file_name, "completed", len(frame),
                            len(frame) if new_dataset else len(inserted),
                            len(updated), unchanged,
                            int(frame["is_synthetic"].sum()) if "is_synthetic" in frame else 0,
                            version, _json({**report, "conflicts": conflicts}), now,
                        ),
                    )
                if derived:
                    db.execute(
                        """INSERT OR REPLACE INTO parquet_derived(
                           symbol,timeframe,adjustment,source_timeframe,target_timeframe,
                           source_version,start_time,end_time,stale,updated_at
                           ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (
                            symbol, timeframe, adjustment, derived["source_timeframe"], timeframe,
                            int(derived["source_version"]), derived.get("start_time"),
                            derived.get("end_time"), 0, now,
                        ),
                    )
                if changed:
                    db.execute(
                        """UPDATE parquet_derived SET stale=1 WHERE symbol=? AND adjustment=?
                           AND source_timeframe=? AND source_version<?""",
                        (symbol, adjustment, timeframe, version),
                    )
                db.commit()
            if changed:
                self._mark_derived_stale(symbol, adjustment, timeframe, version)
            transaction_path = files[0].get("_transaction_path") if len(files) == 1 else None
            if transaction_path:
                self.package.finish_transaction(Path(transaction_path))
        return {
            "symbol": symbol, "timeframe": timeframe, "version": version,
            "rows_read": len(frame),
            "inserted": len(frame) if new_dataset else len(inserted),
            "updated": len(updated), "unchanged": unchanged, "removed": removed_count,
            "synthetic": int(frame["is_synthetic"].sum()) if "is_synthetic" in frame else 0,
            "status": status, "conflicts": conflicts, "file_count": len(files),
            "bytes": sum(int(row["size"]) for row in files),
        }

    def _mark_derived_stale(self, symbol, adjustment, source_timeframe, source_version):
        with self._connect() as db:
            db.execute(
                """UPDATE parquet_derived SET stale=1 WHERE symbol=? AND adjustment=?
                   AND source_timeframe=? AND source_version<?""",
                (symbol, adjustment, source_timeframe, source_version),
            )

    def mark_derived_target_stale(
        self, symbol: str, timeframe: str, adjustment: str = "none",
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE parquet_derived SET stale=1,updated_at=? "
                "WHERE symbol=? AND timeframe=? AND adjustment=? AND target_timeframe=?",
                (utc_now(), symbol, timeframe, adjustment, timeframe),
            )

    def invalidate_day_validations(
        self, symbol: str, timeframe: str, days: Iterable[str], *,
        adjustment: str = "none",
    ) -> int:
        values = sorted({str(day)[:10] for day in days if str(day)})
        if not values:
            return 0
        removed = 0
        with self._connect() as db:
            for start in range(0, len(values), 400):
                chunk = values[start:start + 400]
                placeholders = ",".join("?" for _ in chunk)
                before = db.total_changes
                db.execute(
                    "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? "
                    "AND adjustment=? AND trade_date IN (" + placeholders + ")",
                    (symbol, timeframe, adjustment, *chunk),
                )
                removed += int(db.total_changes - before)
        return removed

    def record_day_validations(self, rows: Iterable[dict]) -> int:
        values = list(rows)
        if not values:
            return 0
        with self._connect() as db:
            db.executemany(
                """INSERT OR REPLACE INTO parquet_day_validation(
                   symbol,timeframe,adjustment,trade_date,status,row_count,
                   source_hash,dependency_hash,rules_version,report,checked_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        row["symbol"], row["timeframe"], row.get("adjustment", "none"),
                        str(row["trade_date"])[:10], row["status"],
                        int(row.get("row_count", 0)), row.get("source_hash"),
                        row.get("dependency_hash"), int(row.get(
                            "rules_version", VALIDATION_RULES_VERSION
                        )), _json(row.get("report", {})), row.get("checked_at") or utc_now(),
                    )
                    for row in values
                ],
            )
        return len(values)

    def invalidate_all_validations(
        self, symbol: str, adjustment: str = "none",
    ) -> int:
        with self._connect() as db:
            before = db.total_changes
            db.execute(
                "DELETE FROM parquet_day_validation WHERE symbol=? AND adjustment=?",
                (symbol, adjustment),
            )
            return int(db.total_changes - before)

    def day_validations(
        self, symbol: str, timeframe: str, *,
        adjustment: str = "none", start=None, end=None,
    ) -> list[dict]:
        clauses = ["symbol=?", "timeframe=?", "adjustment=?"]
        params: list[Any] = [symbol, timeframe, adjustment]
        if start is not None:
            clauses.append("trade_date>=?")
            params.append(str(pd.Timestamp(start).date()))
        if end is not None:
            clauses.append("trade_date<=?")
            params.append(str(pd.Timestamp(end).date()))
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM parquet_day_validation WHERE " + " AND ".join(clauses)
                + " ORDER BY trade_date",
                params,
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["report"] = _load(item.get("report"), {})
            output.append(item)
        return output

    def changed_source_validation_days(
        self, symbol: str, timeframe: str, file_row: dict,
        adjustment: str = "none",
    ) -> list[str]:
        """Compare a replaced source file with persisted per-day hashes."""
        records = self.day_validations(symbol, timeframe, adjustment=adjustment)
        if not records:
            return []
        try:
            frame = self._read_source_files(
                [file_row], symbol=symbol, timeframe=timeframe,
            )
        except Exception:
            return sorted({str(row["trade_date"])[:10] for row in records})
        new_hashes = self._frame_day_hashes(frame)
        changed = []
        for row in records:
            day = pd.Timestamp(row["trade_date"]).normalize()
            if new_hashes.get(day) != row.get("source_hash"):
                changed.append(str(day.date()))
        return changed

    def trading_days(self, symbol: str, start, end) -> pd.DatetimeIndex:
        """Combine auxiliary dates with the package's actual daily trading series.

        Read the daily version directly: dataset() itself asks validation_state()
        for compliance, so going through dataset()/read_frame() would recurse.
        """
        first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        calendar = self.auxiliary(symbol, "calendar")
        values = pd.DatetimeIndex(pd.to_datetime(
            [row.get("date") for row in (calendar or {}).get("payload", [])],
            errors="coerce",
        ).dropna())
        with self._connect() as db:
            daily_row = self._dataset_row(db, symbol, "1day", "none")
        if daily_row:
            daily = self._read_version(
                symbol, "1day", "none", int(daily_row["current_version"]),
                start=first, end=last + pd.Timedelta(days=1),
            )
            if not daily.empty:
                values = values.append(pd.DatetimeIndex(daily.timestamp))
            blocked = daily.attrs.get("price_anomaly_rows") or []
            if blocked:
                values = values.append(pd.DatetimeIndex(pd.to_datetime(
                    [row.get("timestamp") for row in blocked], errors="coerce",
                ).dropna()))
        values = values.normalize().unique().sort_values()
        return values[(values >= first) & (values <= last)]

    @staticmethod
    def _calendar_dates(payload) -> pd.DatetimeIndex:
        if not payload:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(pd.to_datetime(
            [row.get("date") for row in payload], errors="coerce"
        ).dropna()).normalize().unique().sort_values()

    def _calendar_dates_by_market(self) -> dict[str, pd.DatetimeIndex]:
        output = {}
        for suffix in ("SH", "SZ"):
            path = self.package.absolute(auxiliary_relative_path(f"000001.{suffix}", "calendar"))
            if not path.is_file():
                continue
            try:
                output[suffix] = self._calendar_dates(pd.read_parquet(path).to_dict("records"))
            except (OSError, ValueError):
                continue
        return output

    def _validation_from_parts(
        self, dataset: dict, records: list[dict], calendar_dates: pd.DatetimeIndex,
        *, calendar_available: bool | None = None, start=None, end=None,
        expected_dates: pd.DatetimeIndex | None = None,
    ) -> dict:
        symbol, timeframe = dataset["symbol"], dataset["timeframe"]
        first = pd.Timestamp(start or dataset["first_time"]).normalize()
        last = pd.Timestamp(end or dataset["last_time"]).normalize()
        by_day = {pd.Timestamp(row["trade_date"]).normalize(): row for row in records}
        if calendar_available is None:
            calendar_available = bool(len(calendar_dates))
        expected = pd.DatetimeIndex([])
        if expected_dates is not None:
            expected = expected_dates[(expected_dates >= first) & (expected_dates <= last)]
        elif calendar_available:
            expected = calendar_dates[(calendar_dates >= first) & (calendar_dates <= last)]
        if records:
            # Keep historical failures visible even with an incremental calendar.
            observed = pd.DatetimeIndex(sorted(by_day)).normalize().sort_values()
            expected = expected.append(observed).unique().sort_values()
        if timeframe in {"1week", "1month"} and records:
            expected = pd.DatetimeIndex(sorted(by_day)).normalize()
        if not len(expected):
            expected = pd.DatetimeIndex(sorted(by_day)).normalize()
        failed = []
        pending = []
        for day in expected:
            row = by_day.get(pd.Timestamp(day).normalize())
            if row is None:
                pending.append(str(day.date()))
            elif row["status"] == "failed":
                failed.append(str(day.date()))
            elif int(row.get("rules_version") or 0) < VALIDATION_RULES_VERSION:
                pending.append(str(day.date()))
        verified_from = None
        verified_through = None
        started = False
        for index, day in enumerate(expected):
            row = by_day.get(pd.Timestamp(day).normalize())
            passed = bool(
                row and row["status"] in {"passed", "suspended"}
                and int(row.get("rules_version") or 0) >= VALIDATION_RULES_VERSION
            )
            if not started:
                if index == 0 and not passed:
                    break
                started = True
                verified_from = pd.Timestamp(day).normalize()
                verified_through = verified_from
                continue
            if not passed:
                break
            verified_through = pd.Timestamp(day).normalize()
        return {
            "symbol": symbol, "timeframe": timeframe,
            "rules_version": VALIDATION_RULES_VERSION,
            "verified_from": verified_from.date().isoformat() if verified_from is not None else None,
            "verified_through": verified_through.date().isoformat() if verified_through is not None else None,
            "validated_days": sum(
                1 for row in records if row["status"] in {"passed", "suspended"}
                and int(row.get("rules_version") or 0) >= VALIDATION_RULES_VERSION
            ),
            "failed": failed, "pending": pending,
            "calendar_available": calendar_available,
            "range_start": first.date().isoformat(), "range_end": last.date().isoformat(),
            "expected_start": expected[0].date().isoformat() if len(expected) else None,
            "expected_end": expected[-1].date().isoformat() if len(expected) else None,
            "calendar_start": calendar_dates[0].date().isoformat() if len(calendar_dates) else None,
            "calendar_end": calendar_dates[-1].date().isoformat() if len(calendar_dates) else None,
        }

    def validation_state(
        self, symbol: str, timeframe: str, *, adjustment: str = "none",
        start=None, end=None,
    ) -> dict:
        with self._connect() as db:
            dataset_row = self._dataset_row(db, symbol, timeframe, adjustment)
        dataset = dict(dataset_row) if dataset_row else None
        if not dataset:
            return {
                "symbol": symbol, "timeframe": timeframe, "rules_version": VALIDATION_RULES_VERSION,
                "verified_from": None, "verified_through": None, "validated_days": 0,
                "failed": [], "pending": [], "calendar_available": False,
            }
        first = pd.Timestamp(start or dataset["first_time"]).normalize()
        last = pd.Timestamp(end or dataset["last_time"]).normalize()
        records = self.day_validations(
            symbol, timeframe, adjustment=adjustment, start=first, end=last
        )
        calendar = self.auxiliary(symbol, "calendar")
        payload = calendar.get("payload") if calendar else None
        return self._validation_from_parts(
            dataset, records, self._calendar_dates(payload),
            calendar_available=bool(payload), start=start, end=end,
            expected_dates=self.trading_days(symbol, first, last) if records else None,
        )

    def archive_status(
        self, symbol: str, timeframe: str, adjustment: str = "none",
    ) -> dict:
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
            files = self._current_files(
                db, symbol, timeframe, adjustment,
                int(dataset["current_version"]) if dataset else 0,
            )
            archives = db.execute(
                "SELECT * FROM parquet_source_archives"
            ).fetchall()
        lookup = {Path(row["source_path"]).resolve(): dict(row) for row in archives}
        sources = []
        for row in files:
            if not self._is_package_source(row):
                continue
            path = self._absolute_source_path(row["path"]).resolve()
            record = lookup.get(path)
            sources.append({
                "path": str(path),
                "size": path.stat().st_size if path.is_file() else int(row.get("size") or 0),
                "archived": bool(record and self.package.absolute(record["archive_path"]).is_file()),
                "archive_path": record.get("archive_path") if record else None,
                "archived_at": record.get("archived_at") if record else None,
            })
        return {
            "symbol": symbol, "timeframe": timeframe,
            "protected": bool(files) and (not sources or all(item["archived"] for item in sources)),
            "sources": sources,
        }

    def archive_dataset(
        self, symbol: str, timeframe: str, adjustment: str = "none",
        progress=lambda *_: None, cancelled=lambda: False,
    ) -> dict:
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
            files = self._current_files(
                db, symbol, timeframe, adjustment,
                int(dataset["current_version"]) if dataset else 0,
            )
        copied = reused = 0
        total_bytes = 0
        for index, row in enumerate(files, start=1):
            if cancelled():
                raise InterruptedError("任务已取消")
            if not self._is_package_source(row):
                continue
            source = self._absolute_source_path(row["path"]).resolve()
            if not source.is_file():
                raise MarketDataError("行情源文件不存在：" + str(source))
            with self._connect() as db:
                existing = db.execute(
                    "SELECT * FROM parquet_source_archives WHERE source_path=?",
                    (str(source),),
                ).fetchone()
            source_stat = source.stat()
            if (
                existing and int(existing["size"]) == source_stat.st_size
                and int(existing["source_mtime_ns"] or 0) == source_stat.st_mtime_ns
            ):
                archive_path = self.package.absolute(existing["archive_path"])
                if archive_path.is_file():
                    reused += 1
                    total_bytes += int(existing["copied_bytes"] or 0)
                    progress(index / max(1, len(files)), f"复用历史快照 {source.name}")
                    continue
            digest = hashlib.sha256()
            temporary = self.staging_dir / f"archive-{uuid.uuid4().hex}.writing"
            copied_bytes = 0
            try:
                with source.open("rb") as reader, temporary.open("wb") as writer:
                    while chunk := reader.read(4 * 1024 * 1024):
                        digest.update(chunk)
                        writer.write(chunk)
                        copied_bytes += len(chunk)
                sha256 = digest.hexdigest()
                folder = self.package.archives_dir / sha256[:2]
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{sha256}{source.suffix.lower() or '.parquet'}"
                if target.exists():
                    temporary.unlink(missing_ok=True)
                else:
                    os.replace(temporary, target)
                relative = self.package.relative(target)
                with self._connect() as db:
                    db.execute(
                        """INSERT OR REPLACE INTO parquet_source_archives(
                           source_path,source_sha256,archive_path,size,source_mtime_ns,
                           copied_bytes,archived_at)
                           VALUES(?,?,?,?,?,?,?)""",
                        (str(source), sha256, relative, source_stat.st_size,
                         source_stat.st_mtime_ns, copied_bytes, utc_now()),
                    )
                copied += 1
                total_bytes += copied_bytes
                progress(index / max(1, len(files)), f"已保护历史快照 {source.name}")
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "symbol": symbol, "timeframe": timeframe,
            "archived": bool(files), "copied": copied, "reused": reused,
            "bytes": total_bytes, "status": "completed",
        }

    def compact_dataset(
        self, symbol: str, timeframe: str, adjustment: str = "none",
    ) -> dict:
        timeframe = normalize_timeframe(timeframe)
        # Compaction runs inside the dataset writer lock. ``dataset()`` asks
        # validation_state() to read the shared source file and would try to
        # acquire that same non-reentrant file lock.
        with self._connect() as db:
            dataset_row = self._dataset_row(db, symbol, timeframe, adjustment)
        if not dataset_row:
            raise MarketDataError("没有可整理的行情")
        dataset = dict(dataset_row)
        version = int(dataset["current_version"])
        with self._connect() as db:
            increments = self._current_increments(db, symbol, timeframe, adjustment, version)
            current_files = self._current_files(
                db, symbol, timeframe, adjustment, version
            )
        if not increments:
            return {"status": "skipped", "reason": "没有增量分片", "version": version}
        frames = {
            row["period_key"]: self._read_paths(
                [row["path"]], fields=self._canonical_fields(None)
            )
            for row in increments
        }
        groups: list[list[dict]] = []
        for row in increments:
            if not groups:
                groups.append([row])
                continue
            previous = groups[-1][-1]
            if pd.Timestamp(row["period_start"]) <= pd.Timestamp(previous["period_end"]):
                groups[-1].append(row)
            else:
                groups.append([row])
        compacted: list[dict] = []
        for sequence, group in enumerate(groups, start=1):
            if len(group) == 1:
                row = group[0]
                compacted.append({**row, "sequence": sequence})
                continue
            parts = [frames[row["period_key"]] for row in group if not frames[row["period_key"]].empty]
            if not parts:
                continue
            frame = self._prepare_frame(pd.concat(parts, ignore_index=True, sort=False))
            written = self._write_partition(
                symbol, timeframe, adjustment, 0, frame,
                period="all", base_dir=self.package.increments_dir,
            )
            compacted.append({
                **written, "period_key": f"span={sequence}",
                "period_start": group[0]["period_start"],
                "period_end": group[-1]["period_end"], "sequence": sequence,
            })
        new_version = version + 1
        now = utc_now()
        total_bytes = (
            sum(int(row.get("size") or 0) for row in current_files)
            + sum(int(row["size"]) for row in compacted)
        )
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO parquet_versions(
                   symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                   synthetic_rows,pending_rows,first_time,last_time,status,source,
                   file_count,bytes,report) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, timeframe, adjustment, new_version, now,
                    int(dataset.get("row_count") or 0),
                    int(dataset.get("raw_rows") or 0),
                    int(dataset.get("synthetic_rows") or 0),
                    int(dataset.get("pending_rows") or 0),
                    dataset.get("first_time"), dataset.get("last_time"),
                    dataset.get("status", "partial"), "compact",
                    len(current_files) + len(compacted), total_bytes, "{}",
                ),
            )
            self._copy_version_files(
                db, symbol, timeframe, adjustment, version, new_version
            )
            for row in compacted:
                db.execute(
                    """INSERT INTO parquet_increments(
                       symbol,timeframe,adjustment,version,sequence,period_key,
                       period_start,period_end,path,sha256,row_count,first_time,
                       last_time,size,schema_version)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, timeframe, adjustment, new_version, row["sequence"],
                        row["period_key"], row["period_start"], row["period_end"],
                        row["path"], row["sha256"], row["row_count"], row["first_time"],
                        row["last_time"], row["size"], row["schema_version"],
                    ),
                )
            db.execute(
                """UPDATE parquet_datasets SET current_version=?,metadata=?,updated_at=?
                   WHERE symbol=? AND timeframe=? AND adjustment=?""",
                (
                    new_version,
                    _json({**_load(dataset.get("metadata"), {}), "layout": "compacted"}),
                    now, symbol, timeframe, adjustment,
                ),
            )
            db.commit()
        return {
            "status": "completed", "version": new_version, "files": len(compacted),
            "bytes": sum(int(row["size"]) for row in compacted),
            "increment_count": len(increments),
        }

    def _compact_dataset_if_needed(
        self, symbol: str, timeframe: str, adjustment: str, version: int,
    ) -> dict | None:
        with self._connect() as db:
            count = db.execute(
                """SELECT count(*) FROM parquet_increments
                   WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?""",
                (symbol, timeframe, adjustment, version),
            ).fetchone()[0]
        if int(count) < COMPACTION_THRESHOLD:
            return None
        return self.compact_dataset(symbol, timeframe, adjustment)

    def save_coverage(
        self, symbol: str, timeframe: str, adjustment: str, start, end, report: dict,
    ) -> None:
        timeframe = normalize_timeframe(timeframe)
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
        version = int(dataset["current_version"]) if dataset else None
        report = {
            **report,
            "package_id": self.package_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "adjustment": adjustment,
            "version": version,
            "checked_start": _time_text(start),
            "checked_end": _time_text(end),
        }
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO parquet_coverage(
                   symbol,timeframe,adjustment,start_time,end_time,status,expected_rows,
                   observed_rows,synthetic_rows,gap_count,report,checked_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, timeframe, adjustment, _time_text(start), _time_text(end),
                    report.get("status", "unknown"), report.get("expected_rows"),
                    report.get("observed_rows"), report.get("synthetic_rows"),
                    report.get("gap_count", 0), _json(report), utc_now(),
                ),
            )
        self.package.write_quality(symbol, timeframe, report)

    def update_dataset_status(
        self, symbol: str, timeframe: str, adjustment: str,
        status: str, reason: str | None,
    ) -> None:
        timeframe = normalize_timeframe(timeframe)
        with self._connect() as db:
            db.execute(
                """UPDATE parquet_datasets SET status=?,reason=?,updated_at=?
                   WHERE symbol=? AND timeframe=? AND adjustment=?""",
                (status, reason, utc_now(), symbol, timeframe, adjustment),
            )

    def coverage(self, symbol: str, timeframe: str, adjustment: str = "none") -> dict | None:
        timeframe = normalize_timeframe(timeframe)
        item = self.dataset(symbol, timeframe, adjustment)
        if not item:
            return None
        with self._connect() as db:
            versions = db.execute(
                """SELECT version,created_at,status,source,report,row_count,file_count,bytes
                   FROM parquet_versions WHERE symbol=? AND timeframe=? AND adjustment=?
                   ORDER BY version DESC LIMIT 20""",
                (symbol, timeframe, adjustment),
            ).fetchall()
            segments = db.execute(
                """SELECT * FROM parquet_coverage WHERE symbol=? AND timeframe=?
                   AND adjustment=? ORDER BY end_time DESC LIMIT 200""",
                (symbol, timeframe, adjustment),
            ).fetchall()
        item["versions"] = [
            {**dict(row), "report": _load(row["report"], {})} for row in versions
        ]
        item["segments"] = [
            {**dict(row), "report": _load(row["report"], {})} for row in segments
        ]
        return item

    def save_auxiliary(
        self, symbol: str, kind: str, payload: Any, *,
        version: str = "local", verified: bool = False,
    ) -> str:
        # Calendars are shared by exchange, so different securities must use
        # the same lock for the entire read/merge/replace operation.
        path = self.package.absolute(auxiliary_relative_path(symbol, kind))
        with self.file_lock(path):
            return self._save_auxiliary_unlocked(
                symbol, kind, payload, version=version, verified=verified,
            )

    def _save_auxiliary_unlocked(
        self, symbol: str, kind: str, payload: Any, *,
        version: str = "local", verified: bool = False,
    ) -> str:
        frame = pd.DataFrame(payload if isinstance(payload, list) else payload or [])
        relative = auxiliary_relative_path(symbol, kind)
        destination = self.package.absolute(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if kind == "calendar" and destination.is_file() and "date" in frame:
            previous = pd.read_parquet(destination)
            if not previous.empty and "date" in previous:
                frame = pd.concat([previous[["date"]], frame[["date"]]], ignore_index=True)
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame = frame.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        if destination.exists():
            backup_dir = self.package.history_dir / "auxiliary" / symbol / kind
            backup_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(
                f"{destination.stat().st_size}|{destination.stat().st_mtime_ns}".encode()
            ).hexdigest()[:16]
            backup = backup_dir / f"{str(version)}-{digest}.parquet"
            if not backup.exists():
                shutil.copy2(destination, backup)
        temporary = destination.with_name(destination.name + ".writing")
        frame.to_parquet(temporary, compression="zstd", index=False)
        os.replace(temporary, destination)
        self.package.record_auxiliary(relative, version=str(version), verified=verified)
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO parquet_auxiliary_history(
                   symbol,kind,version,payload,verified,updated_at
                   ) VALUES(?,?,?,?,?,?)""",
                (symbol, kind, str(version), _json(payload), int(verified), utc_now()),
            )
        from ..storage import Store
        Store(self.root).save_market_auxiliary(
            symbol, kind, payload, version=str(version), verified=verified
        )
        return str(version)

    def auxiliary(self, symbol: str, kind: str, version: str | None = None) -> dict | None:
        relative = auxiliary_relative_path(symbol, kind)
        path = self.package.absolute(relative)
        if version is None and path.is_file():
            # Windows readers also need this lock while a writer replaces it.
            with self.file_lock(path):
                manifest = self.package.manifest().get("auxiliary", {}).get(relative, {})
                frame = pd.read_parquet(path)
            payload = frame.to_dict("records") if not frame.empty else []
            return {
                "symbol": symbol, "kind": kind,
                "version": str(manifest.get("version") or "local"),
                "payload": payload,
                "verified": bool(manifest.get("verified", False)),
                "updated_at": manifest.get("updated_at"),
                "storage": "parquet-source",
            }
        with self._connect() as db:
            if version is None:
                row = db.execute(
                    """SELECT * FROM parquet_auxiliary_history WHERE symbol=? AND kind=?
                       ORDER BY updated_at DESC LIMIT 1""",
                    (symbol, kind),
                ).fetchone()
            else:
                row = db.execute(
                    """SELECT * FROM parquet_auxiliary_history
                       WHERE symbol=? AND kind=? AND version=?""",
                    (symbol, kind, str(version)),
                ).fetchone()
        return {**dict(row), "payload": _load(row["payload"], [])} if row else None

    def auxiliary_versions(self, symbol: str) -> dict[str, str]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT h.kind,h.version FROM parquet_auxiliary_history h
                   JOIN (
                       SELECT kind,MAX(updated_at) updated_at
                       FROM parquet_auxiliary_history WHERE symbol=? GROUP BY kind
                   ) latest ON latest.kind=h.kind AND latest.updated_at=h.updated_at
                   WHERE h.symbol=?""",
                (symbol, symbol),
            ).fetchall()
        return {row["kind"]: row["version"] for row in rows}

    def active_manifest(self, symbol: str, *, metadata: dict | None = None) -> dict | None:
        market: dict[str, dict] = {}
        for timeframe in TIMEFRAMES:
            dataset = self.dataset(symbol, timeframe)
            if dataset and dataset["row_count"]:
                market[timeframe] = {
                    "version": dataset["current_version"], "first": dataset["first_time"],
                    "last": dataset["last_time"], "rows": dataset["row_count"],
                    "status": dataset["status"], "metadata": dataset["metadata"],
                    "files": dataset["file_count"], "bytes": dataset["current_bytes"],
                }
        if not market:
            return None
        content = {
            "symbol": symbol, "market_data": True, "market_versions": market,
            "metadata": metadata or {}, "generation": self.generation,
            "auxiliary_versions": self.auxiliary_versions(symbol),
            "package_id": self.package_id,
            "package_version": int(self.package.manifest().get("package_version", 0)),
        }
        version = "mkt-" + hashlib.sha256(_json(content).encode()).hexdigest()[:24]
        minute = next(
            (name for name in ("1min", "5min", "15min", "30min", "60min") if name in market),
            None,
        )
        files = {
            "minute": [f"package:{minute}:{symbol}"] if minute else [],
            "daily": [f"package:1day:{symbol}"] if "1day" in market else [],
            "actions": [], "calendar": [],
        }
        return {
            **content, "version": version, "created_at": utc_now(),
            "files": files, "profiles": {},
        }

    def storage_stats(self, symbol: str, timeframe: str, adjustment: str = "none") -> dict:
        dataset = self.dataset(symbol, timeframe, adjustment)
        if not dataset:
            return {"storage": "parquet", "current_bytes": 0, "history_bytes": 0}
        return {
            "storage": "parquet", "current_bytes": dataset["current_bytes"],
            "history_bytes": dataset["history_bytes"], "file_count": dataset["file_count"],
        }

    def observed_days(self, symbol: str, timeframe: str, adjustment: str = "none") -> int:
        timeframe = normalize_timeframe(timeframe)
        dataset = self.dataset(symbol, timeframe, adjustment)
        if not dataset:
            return 0
        if timeframe == "1day":
            return int(dataset.get("row_count") or 0)
        with self._connect() as db:
            files = self._current_files(
                db, symbol, timeframe, adjustment, int(dataset["current_version"])
            )
            increments = self._current_increments(
                db, symbol, timeframe, adjustment, int(dataset["current_version"])
            )
        source = next((row for row in files if self._is_package_source(row)), None)
        if not source or increments:
            frame = self._read_version(
                symbol, timeframe, adjustment, int(dataset["current_version"]),
                fields=("timestamp",), use_cache=False,
            )
            return int(frame.timestamp.dt.normalize().nunique()) if not frame.empty else 0
        path = self._absolute_source_path(source["path"])
        if not path.is_file():
            return 0
        with duckdb.connect() as connection:
            row = connection.execute(
                "SELECT count(DISTINCT CAST(trade_date AS DATE)) FROM read_parquet(?)",
                [str(path)],
            ).fetchone()
        return int(row[0] or 0) if row else 0

    def coverage_covers(
        self, symbol: str, timeframe: str, start, end, adjustment: str = "none",
    ) -> bool:
        start_text = _time_text(pd.Timestamp(start).normalize())
        end_text = _time_text(pd.Timestamp(end).normalize())
        with self._connect() as db:
            rows = db.execute(
                """SELECT start_time,end_time,status,report FROM parquet_coverage
                   WHERE symbol=? AND timeframe=? AND adjustment=?""",
                (symbol, normalize_timeframe(timeframe), adjustment),
            ).fetchall()
        intervals = self._coverage_intervals(rows)
        start_day = pd.Timestamp(start_text)
        end_day = pd.Timestamp(end_text)
        return any(left <= start_day and right >= end_day for left, right in intervals)

    @staticmethod
    def _coverage_intervals(rows) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        intervals = []
        for row in rows:
            report = _load(row["report"], {})
            value = str(report.get("coverage_status") or row["status"] or "")
            if value != "complete":
                continue
            try:
                left = pd.Timestamp(row["start_time"]).normalize()
                right = pd.Timestamp(row["end_time"]).normalize()
            except (TypeError, ValueError):
                continue
            intervals.append((left, right))
        intervals.sort()
        merged = []
        for left, right in intervals:
            if not merged or left > merged[-1][1] + pd.Timedelta(days=1):
                merged.append([left, right])
            else:
                merged[-1][1] = max(merged[-1][1], right)
        return [(left, right) for left, right in merged]
