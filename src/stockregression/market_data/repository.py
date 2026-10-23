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
from .package_layout import PackageLayout, asset_type_for_symbol, auxiliary_relative_path, resolve_package_root
from .timeframes import MINUTE_TIMEFRAMES, normalize_timeframe
from ..price_validation import details_to_rows, price_issue_details, split_invalid_prices


PARSER_VERSION = "market-data-v3-price-validation"
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
        # Validate the on-disk manifest before changing any index state. In
        # particular, schema-1 packages must never trigger the old path-change
        # index reset before the migration command has frozen the active refs.
        package_manifest = self.package.manifest()
        self.package_dir = self.package.root
        self.market_dir = self.package_dir
        self.staging_dir = self.package.staging_dir
        previous_id = self.meta("package_id")
        previous_generation = self.meta("generation")
        if previous_id and (
            previous_id != package_manifest["package_id"]
            or (previous_generation and previous_generation != package_manifest["generation"])
        ):
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
                (package_manifest["package_id"],),
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('generation',?)",
                (package_manifest["generation"],),
            )
        self._seed_manifest_from_index()
        self._recover_package_transactions()

    def attach_package(self, package_dir: str | Path) -> None:
        previous_id = self.meta("package_id")
        previous_generation = self.meta("generation")
        package = PackageLayout(resolve_package_root(package_dir))
        manifest = package.manifest()
        self.package = package
        self.package_dir = self.package.root
        self.market_dir = self.package_dir
        self.staging_dir = self.package.staging_dir
        if previous_id and (
            previous_id != manifest["package_id"]
            or (previous_generation and previous_generation != manifest["generation"])
        ):
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
                (manifest["package_id"],),
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('generation',?)",
                (manifest["generation"],),
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
                "parquet_day_validation", "parquet_shards", "parquet_source_archives",
                "parquet_derived", "parquet_coverage", "parquet_imports",
                "parquet_auxiliary_history", "parquet_versions",
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
        from .shard_index import initialize_shard_index

        initialize_shard_index(db)
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

    def _assert_writable(self) -> None:
        if (self.root / ".market-data-migration.lock").exists():
            raise MarketDataError("行情数据迁移期间暂停写入，请稍后重试")
        if self.package.pending_transactions():
            raise MarketDataError("数据包有待恢复事务，请重启服务恢复索引后再写入")

    @property
    def source_package(self) -> Path | None:
        return self.package_dir if self.package_dir.is_dir() else None

    @property
    def package_id(self) -> str:
        return str(self.package.manifest()["package_id"])

    def package_info(self) -> dict:
        manifest = self.package.manifest()
        heads = manifest.get("datasets") or {}
        with self._connect() as db:
            indexed = db.execute(
                "SELECT symbol,timeframe,adjustment,current_version FROM parquet_datasets"
            ).fetchall()
            raw_rows = db.execute(
                """SELECT s.path,MAX(s.size) size FROM parquet_shards s
                   JOIN parquet_datasets d ON d.symbol=s.symbol AND d.timeframe=s.timeframe
                    AND d.adjustment=s.adjustment AND d.current_version=s.version
                   WHERE s.timeframe IN ('1day','1min') GROUP BY s.path"""
            ).fetchall()
            package_id = db.execute(
                "SELECT value FROM parquet_meta WHERE key='package_id'"
            ).fetchone()
            generation = db.execute(
                "SELECT value FROM parquet_meta WHERE key='generation'"
            ).fetchone()
        indexed_heads = {
            self.package.dataset_key(row["symbol"], row["timeframe"], row["adjustment"]):
                int(row["current_version"])
            for row in indexed
        }
        expected_heads = {
            key: int(value.get("version") or 0) for key, value in heads.items()
        }
        index_ready = (
            str(package_id[0] if package_id else "") == str(manifest.get("package_id"))
            and str(generation[0] if generation else "") == str(manifest.get("generation"))
            and all(indexed_heads.get(key) == version for key, version in expected_heads.items())
            and all(key in expected_heads for key in indexed_heads)
        )
        unique_sizes = {str(row["path"]): int(row["size"] or 0) for row in raw_rows}
        shard_count = len(unique_sizes)
        shard_bytes = sum(unique_sizes.values())
        readable = self.package_dir.is_dir()
        pending = self.package.pending_transactions()
        manifest_ready = bool(heads)
        status = (
            "recovery_required" if pending else
            "ready" if readable and index_ready and manifest_ready else
            "index_stale" if readable and manifest_ready else
            "empty" if readable else "missing"
        )
        return {
            "package_id": manifest.get("package_id"),
            "generation": manifest.get("generation"),
            "package_version": manifest.get("package_version", 0),
            "schema_version": manifest.get("schema_version"),
            "storage_layout": manifest.get("storage_layout", "time_partitioned"),
            "index_status": "ready" if index_ready else "stale",
            "path": str(self.package_dir),
            "status": status,
            "readable": readable,
            "manifest_ready": manifest_ready,
            "shard_count": shard_count,
            "shard_bytes": shard_bytes,
            "bytes": shard_bytes,
            "file_count": shard_count,
            "updated_at": manifest.get("updated_at"),
            "pending_recovery": len(pending),
        }

    def _recover_package_transactions(self) -> None:
        for transaction in self.package.pending_transactions():
            try:
                payload = json.loads(transaction.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if payload.get("kind") == "dataset":
                symbol = str(payload.get("symbol") or "")
                timeframe = normalize_timeframe(str(payload.get("timeframe") or "1day"))
                adjustment = str(payload.get("adjustment") or "none")
                expected_version = int(payload.get("version") or 0)
                with self._connect() as db:
                    dataset = self._dataset_row(db, symbol, timeframe, adjustment)
                    indexed_version = int(dataset["current_version"] or 0) if dataset else 0
                head = self.package.manifest().get("datasets", {}).get(
                    self.package.dataset_key(symbol, timeframe, adjustment), {}
                )
                published_version = int(head.get("version") or 0)
                if indexed_version >= expected_version and published_version >= expected_version:
                    self.package.finish_transaction(transaction)
                elif indexed_version < expected_version and published_version < expected_version:
                    # The SQLite commit did not happen. Installed content
                    # addressed shards remain harmless orphans until explicit
                    # garbage collection can prove they are unreferenced.
                    self.package.finish_transaction(transaction)
                else:
                    self.package.update_transaction(transaction, {"status": "needs_rebuild"})
                continue
            raise MarketDataError(f"不支持的旧文件事务：{transaction.name}")

    def _seed_manifest_from_index(self) -> None:
        """Reconcile the published package heads and the rebuildable SQLite index."""
        manifest = self.package.manifest()
        with self._connect() as db:
            datasets = db.execute(
                "SELECT * FROM parquet_datasets WHERE current_version>0 "
                "ORDER BY timeframe,symbol,adjustment"
            ).fetchall()
        database_versions = {
            self.package.dataset_key(row["symbol"], row["timeframe"], row["adjustment"]):
                int(row["current_version"])
            for row in datasets
        }
        package_ahead = []
        for row in datasets:
            key = self.package.dataset_key(
                str(row["symbol"]), str(row["timeframe"]), str(row["adjustment"])
            )
            head = manifest.get("datasets", {}).get(key) or {}
            version = int(row["current_version"])
            if int(head.get("version") or 0) > version:
                package_ahead.append((str(row["symbol"]), str(row["timeframe"])))
                continue
            manifest_path = self.package.absolute(str(head.get("manifest") or ""))
            if int(head.get("version") or 0) != version or not manifest_path.is_file():
                self._publish_dataset_manifest(
                    str(row["symbol"]), str(row["timeframe"]),
                    str(row["adjustment"]), version,
                )
        for key, head in manifest.get("datasets", {}).items():
            if int(head.get("version") or 0) <= database_versions.get(str(key), 0):
                continue
            try:
                symbol, timeframe, _adjustment = str(key).split("|", 2)
            except ValueError as exc:
                raise MarketDataError(f"包清单中的数据集键无效：{key}") from exc
            package_ahead.append((symbol, timeframe))
        if package_ahead:
            symbols = sorted({symbol for symbol, _ in package_ahead})
            timeframes = sorted({timeframe for _, timeframe in package_ahead})
            self.rebuild_index(symbols=symbols, timeframes=timeframes)

    def _manifest_shards(self, shards: Sequence[dict]) -> list[dict]:
        normalized = []
        for row in shards:
            path = self._absolute_source_path(str(row["path"])).resolve()
            relative = self.package.relative(path)
            if Path(relative).is_absolute() or relative.startswith("../"):
                raise MarketDataError(f"活跃行情分片不属于当前数据包：{path}")
            if not path.is_file():
                raise MarketDataError(f"活跃行情分片缺失：{path}")
            normalized.append({
                "partition_key": row.get("period_key", row.get("partition_key")),
                "period_start": row.get("period_start"),
                "period_end": row.get("period_end"),
                "path": relative,
                "sha256": row.get("sha256"),
                "size": int(row.get("size") or path.stat().st_size),
                "row_count": int(row.get("row_count") or 0),
                "first_time": row.get("first_time"),
                "last_time": row.get("last_time"),
                "schema_version": row.get("schema_version"),
            })
        normalized.sort(key=lambda item: (
            str(item.get("period_start") or item.get("first_time") or ""),
            str(item["path"]),
        ))
        return normalized

    def _publish_dataset_manifest(
        self, symbol: str, timeframe: str, adjustment: str, version: int,
    ) -> dict:
        """Publish the active SQLite view as immutable package metadata."""
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
            if not dataset or int(dataset["current_version"]) != int(version):
                raise MarketDataError(
                    f"索引版本已变化，不能发布 {symbol} {timeframe} v{version}"
                )
            version_row = db.execute(
                "SELECT * FROM parquet_versions WHERE symbol=? AND timeframe=? "
                "AND adjustment=? AND version=?",
                (symbol, timeframe, adjustment, int(version)),
            ).fetchone()
            if not version_row:
                raise MarketDataError(f"行情版本记录缺失：{symbol} {timeframe} v{version}")
            shards = self._current_shards(db, symbol, timeframe, adjustment, int(version))
            derived = db.execute(
                "SELECT * FROM parquet_derived WHERE symbol=? AND timeframe=? "
                "AND adjustment=? AND target_timeframe=?",
                (symbol, timeframe, adjustment, timeframe),
            ).fetchone()
        payload = {
            "symbol": symbol,
            "timeframe": normalize_timeframe(timeframe),
            "adjustment": adjustment,
            "version": int(version),
            "row_count": int(version_row["row_count"]),
            "first_time": version_row["first_time"],
            "last_time": version_row["last_time"],
            "status": version_row["status"],
            "source": version_row["source"],
            "created_at": version_row["created_at"],
            "report": _load(version_row["report"], {}),
            "completed_through": dataset["completed_through"],
            "raw_rows": int(dataset["raw_rows"] or 0),
            "synthetic_rows": int(dataset["synthetic_rows"] or 0),
            "pending_rows": int(dataset["pending_rows"] or 0),
            "reason": dataset["reason"],
            "metadata": _load(dataset["metadata"], {}),
            "shards": self._manifest_shards(shards),
            "derived": dict(derived) if derived else None,
        }
        head = self.package.manifest().get("datasets", {}).get(
            self.package.dataset_key(symbol, timeframe, adjustment), {}
        )
        return self.package.publish_dataset(
            payload, expected_version=int(head.get("version") or 0),
        )

    def rebuild_index(
        self, *, symbols: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
    ) -> dict:
        """Rebuild market tables from published schema-2 dataset manifests."""
        package = self.package.manifest()
        if int(package.get("schema_version") or 0) != 2:
            raise MarketDataError("只支持从 schema 2 分片包重建索引")
        wanted_symbols = set(symbols or ())
        wanted_timeframes = {normalize_timeframe(value) for value in timeframes or ()}
        heads = package.get("datasets") or {}
        selected = []
        for key, head in sorted(heads.items()):
            try:
                symbol, timeframe, adjustment = str(key).split("|", 2)
                timeframe = normalize_timeframe(timeframe)
            except (ValueError, TypeError) as exc:
                raise MarketDataError(f"包清单中的数据集键无效：{key}") from exc
            if wanted_symbols and symbol not in wanted_symbols:
                continue
            if wanted_timeframes and timeframe not in wanted_timeframes:
                continue
            selected.append((key, symbol, timeframe, adjustment, dict(head)))
        if not selected:
            return {"status": "completed", "datasets": 0, "versions": 0, "shards": 0}

        prepared: list[tuple[str, str, str, int, dict, list[dict]]] = []
        for key, symbol, timeframe, adjustment, head in selected:
            relative = str(head.get("manifest") or "")
            if not relative:
                raise MarketDataError(f"包清单缺少数据集版本清单引用：{key}")
            current_path = self.package.absolute(relative)
            if not current_path.is_file():
                raise MarketDataError(f"数据集版本清单缺失：{current_path}")
            history_paths = sorted(
                current_path.parent.glob("v*.json"),
                key=lambda path: int(path.stem[1:]) if path.stem[1:].isdigit() else -1,
            )
            versions = []
            for path in history_paths:
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise MarketDataError(f"数据集版本清单不可读：{path}: {exc}") from exc
                if (
                    value.get("schema_version") != 2
                    or value.get("package_id") != package.get("package_id")
                    or value.get("generation") != package.get("generation")
                    or value.get("symbol") != symbol
                    or normalize_timeframe(value.get("timeframe")) != timeframe
                    or value.get("adjustment", "none") != adjustment
                ):
                    continue
                version = int(value.get("version") or 0)
                shards = list(value.get("shards") or [])
                for shard in shards:
                    shard_path = str(shard.get("path") or "").replace("\\", "/")
                    if not shard_path.startswith("partitions/"):
                        raise MarketDataError(f"分片路径不在 partitions 下：{shard_path}")
                    absolute = self.package.absolute(shard_path)
                    if not absolute.is_file():
                        raise MarketDataError(f"分片文件缺失：{absolute}")
                    if int(shard.get("size") or -1) != absolute.stat().st_size:
                        raise MarketDataError(f"分片大小与清单不符：{absolute}")
                versions.append((version, value, shards))
            if not versions:
                raise MarketDataError(f"没有可用的数据集版本：{key}")
            if max(item[0] for item in versions) != int(head.get("version") or 0):
                raise MarketDataError(f"包级清单版本与数据集清单不一致：{key}")
            prepared.extend(
                (symbol, timeframe, adjustment, version, value, shards)
                for version, value, shards in versions
            )

        current_keys = {(item[1], item[2], item[3]) for item in selected}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous_versions = {}
            for symbol, timeframe, adjustment in current_keys:
                row = self._dataset_row(db, symbol, timeframe, adjustment)
                previous_versions[(symbol, timeframe, adjustment)] = (
                    int(row["current_version"]) if row else 0
                )
            if not wanted_symbols and not wanted_timeframes:
                for table in ("parquet_shards", "parquet_versions",
                              "parquet_datasets", "parquet_derived"):
                    db.execute(f"DELETE FROM {table}")
            else:
                for symbol, timeframe, adjustment in current_keys:
                    db.execute(
                        "DELETE FROM parquet_shards WHERE symbol=? AND timeframe=? AND adjustment=?",
                        (symbol, timeframe, adjustment),
                    )
                    db.execute(
                        "DELETE FROM parquet_versions WHERE symbol=? AND timeframe=? AND adjustment=?",
                        (symbol, timeframe, adjustment),
                    )
                    db.execute(
                        "DELETE FROM parquet_datasets WHERE symbol=? AND timeframe=? AND adjustment=?",
                        (symbol, timeframe, adjustment),
                    )
                    db.execute(
                        "DELETE FROM parquet_derived WHERE symbol=? AND timeframe=? AND adjustment=?",
                        (symbol, timeframe, adjustment),
                    )
            version_count = 0
            shard_count = 0
            latest_by_key = {}
            for symbol, timeframe, adjustment, version, value, shards in prepared:
                created_at = str(value.get("created_at") or utc_now())
                row_count = int(value.get("row_count") or 0)
                synthetic = int(value.get("synthetic_rows") or 0)
                pending = int(value.get("pending_rows") or 0)
                status = str(value.get("status") or "pending_validation")
                byte_count = sum(int(shard.get("size") or 0) for shard in shards)
                db.execute(
                    """INSERT INTO parquet_versions(
                       symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                       synthetic_rows,pending_rows,first_time,last_time,status,source,
                       file_count,bytes,report) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, timeframe, adjustment, version, created_at, row_count,
                        int(value.get("raw_rows") or max(0, row_count - synthetic)),
                        synthetic, pending, value.get("first_time"), value.get("last_time"),
                        status, value.get("source"), len(shards), byte_count,
                        _json(value.get("report") or {}),
                    ),
                )
                for sequence, shard in enumerate(shards, start=1):
                    key = str(shard.get("partition_key") or "")
                    start = shard.get("period_start") or shard.get("first_time")
                    end = shard.get("period_end") or shard.get("last_time")
                    if not key or not start or not end:
                        raise MarketDataError(
                            f"分片缺少周期边界：{shard.get('path')}"
                        )
                    db.execute(
                        """INSERT INTO parquet_shards(
                           symbol,timeframe,adjustment,version,sequence,period_key,
                           period_start,period_end,path,sha256,row_count,first_time,
                           last_time,size,schema_version)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            symbol, timeframe, adjustment, version, sequence, key,
                            str(start), str(end), shard["path"], shard.get("sha256") or "",
                            int(shard.get("row_count") or 0), shard.get("first_time"),
                            shard.get("last_time"), int(shard.get("size") or 0),
                            shard.get("schema_version") or PARQUET_SCHEMA_VERSION,
                        ),
                    )
                derived = value.get("derived")
                if derived:
                    db.execute(
                        """INSERT OR REPLACE INTO parquet_derived(
                           symbol,timeframe,adjustment,source_timeframe,target_timeframe,
                           source_version,start_time,end_time,stale,updated_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (
                            symbol, timeframe, adjustment,
                            derived.get("source_timeframe", "1min"), timeframe,
                            int(derived.get("source_version") or 0),
                            derived.get("start_time"), derived.get("end_time"),
                            int(derived.get("stale") or 0), utc_now(),
                        ),
                    )
                latest_by_key[(symbol, timeframe, adjustment)] = value
                version_count += 1
                shard_count += len(shards)
            for (symbol, timeframe, adjustment), value in latest_by_key.items():
                row_count = int(value.get("row_count") or 0)
                synthetic = int(value.get("synthetic_rows") or 0)
                new_version = int(value["version"])
                if previous_versions.get((symbol, timeframe, adjustment), 0) != new_version:
                    # The package manifest is authoritative during recovery. If
                    # it advanced while SQLite did not, old quality conclusions
                    # cannot be carried onto the newly restored market view.
                    db.execute(
                        "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? AND adjustment=?",
                        (symbol, timeframe, adjustment),
                    )
                    db.execute(
                        "DELETE FROM parquet_coverage WHERE symbol=? AND timeframe=? AND adjustment=?",
                        (symbol, timeframe, adjustment),
                    )
                db.execute(
                    """INSERT INTO parquet_datasets(
                       symbol,timeframe,adjustment,current_version,first_time,last_time,
                       completed_through,row_count,raw_rows,synthetic_rows,pending_rows,
                       status,reason,metadata,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, timeframe, adjustment, new_version,
                        value.get("first_time"), value.get("last_time"),
                        value.get("completed_through") or value.get("last_time"),
                        row_count, int(value.get("raw_rows") or max(0, row_count - synthetic)),
                        synthetic, int(value.get("pending_rows") or 0),
                        value.get("status") or "pending_validation", value.get("reason"),
                        _json(value.get("metadata") or {}),
                        value.get("created_at") or utc_now(), utc_now(),
                    ),
                )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('package_id',?)",
                (str(package["package_id"]),),
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('generation',?)",
                (str(package["generation"]),),
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('package_dir',?)",
                (str(self.package_dir),),
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('source_package_dir',?)",
                (str(self.package_dir),),
            )
            db.commit()
        return {
            "status": "completed", "datasets": len(latest_by_key),
            "versions": version_count, "shards": shard_count,
            "package_id": package["package_id"], "generation": package["generation"],
        }

    def save_basic_info(self, symbol: str, metadata: dict) -> Path:
        self._assert_writable()
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
        resolved = (candidate if candidate.is_absolute() else self.package_dir / candidate).resolve()
        try:
            resolved.relative_to(self.package_dir.resolve())
        except ValueError as exc:
            raise MarketDataError(f"行情分片引用逃出当前数据包：{path}") from exc
        return resolved

    def lock(self, symbol: str, timeframe: str, adjustment: str = "none") -> FileLock:
        return self.package.lock_for(
            "dataset:" + PackageLayout.dataset_key(
                symbol, normalize_timeframe(timeframe), adjustment
            )
        )

    def file_lock(self, path: str | Path) -> FileLock:
        relative = self.package.relative(path)
        return self.package.lock_for("file:" + relative)

    @staticmethod
    def _dataset_row(db, symbol, timeframe, adjustment):
        return db.execute(
            "SELECT * FROM parquet_datasets WHERE symbol=? AND timeframe=? AND adjustment=?",
            (symbol, timeframe, adjustment),
        ).fetchone()

    def _current_shards(
        self, db, symbol, timeframe, adjustment, version: int,
        *, start=None, end=None, before=None,
    ) -> list[dict]:
        if version <= 0:
            return []
        clauses = ["symbol=?", "timeframe=?", "adjustment=?", "version=?"]
        params: list[Any] = [symbol, timeframe, adjustment, version]
        upper = end if end is not None else before
        if start is not None:
            clauses.append("datetime(period_end)>datetime(?)")
            params.append(_time_text(start))
        if upper is not None:
            clauses.append("datetime(period_start)<datetime(?)")
            params.append(_time_text(upper))
        rows = db.execute(
            "SELECT * FROM parquet_shards WHERE " + " AND ".join(clauses)
            + " ORDER BY sequence,period_start",
            params,
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

    def _copy_version_shards(
        self, db, symbol: str, timeframe: str, adjustment: str,
        source_version: int, target_version: int, skip_periods: set[str] | None = None,
    ) -> None:
        rows = self._current_shards(db, symbol, timeframe, adjustment, source_version)
        skip_periods = skip_periods or set()
        for row in rows:
            if row["period_key"] in skip_periods:
                continue
            db.execute(
                """INSERT INTO parquet_shards(
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

    def _write_period_shard(
        self, symbol: str, timeframe: str, adjustment: str,
        period_key: str, frame: pd.DataFrame,
    ) -> dict:
        bucket = period_key.split("=", 1)[1]
        return self._write_partition(
            symbol, timeframe, adjustment, bucket, frame,
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
            shards = self._current_shards(
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
                       SELECT path,MAX(size) size FROM parquet_shards
                       WHERE symbol=? AND timeframe=? AND adjustment=? GROUP BY path
                   )""",
                (symbol, timeframe, adjustment),
            ).fetchone()[0]
        item["metadata"] = _load(item.get("metadata"), {})
        item["timeframe"] = timeframe
        item["version"] = int(item["current_version"])
        file_errors: list[str] = []
        manifest = self.package.manifest()
        head = manifest.get("datasets", {}).get(
            self.package.dataset_key(symbol, timeframe, adjustment), {}
        )
        if int(head.get("version") or 0) != int(item["current_version"]):
            item["file_status"] = "changed"
            file_errors.append("SQLite 索引与包级清单版本不一致")
        else:
            # File existence and content are checked when a query opens the
            # selected shard; listing relies on the manifest-backed index.
            item["file_status"] = "readable"
        item["file_errors"] = file_errors
        item["missing_files"] = 0
        item["changed_files"] = len(file_errors)
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
        item["storage"] = "parquet"
        item["storage_path"] = str(
            self.package.partitions_dir / timeframe / f"symbol={symbol}"
        )
        item["file_count"] = len(shards)
        item["shard_count"] = len(shards)
        item["current_bytes"] = sum(int(row["size"]) for row in shards)
        item["history_bytes"] = int(history or 0)
        item["derived"] = dict(derived) if derived else None
        item["derived_stale"] = bool(derived["stale"]) if derived else False
        item["coverage_unknown"] = item["coverage_status"] == "unknown"
        item["file_errors"] = file_errors
        item["archive"] = self.archive_status(symbol, timeframe, adjustment)
        return item

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
            shard_rows = db.execute(
                """SELECT symbol,timeframe,adjustment,version,path,size FROM parquet_shards
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
        shard_map: dict[tuple[str, str, str], list[dict]] = {}
        for row in shard_rows:
            shard_map.setdefault(
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
        manifest_heads = self.package.manifest().get("datasets", {})
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
            current_shards = [
                item for item in shard_map.get(key, [])
                if int(item.get("version") or 0) == int(row["current_version"] or 0)
            ]
            head = manifest_heads.get(self.package.dataset_key(*key)) or {}
            file_status = (
                "readable"
                if int(head.get("version") or 0) == int(row["current_version"] or 0)
                else "changed"
            )
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
                "shard_count": len(current_shards),
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
                    "shard_count": row.get("shard_count", 0),
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
            for item in shard_map.get((row["symbol"], row["timeframe"], row["adjustment"]), []):
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
        if not row:
            return None
        item = dict(row)
        item["report"] = _load(item.get("report"), {})
        item["storage"] = "parquet"
        file_status = "readable"
        dataset_manifest = self.package.load_dataset_manifest(
            symbol, timeframe, adjustment, int(version)
        )
        if dataset_manifest is None:
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
            relation = relation.order("timestamp DESC" if descending else "timestamp ASC")
            if limit is not None:
                relation = relation.limit(int(limit))
            relation = relation.project(", ".join(columns))
            return self._frame_for_consumer(relation.df())

    def _read_version(
        self, symbol: str, timeframe: str, adjustment: str, version: int, *,
        start=None, end=None, before=None, fields: Sequence[str] | None = None,
        limit: int | None = None, descending: bool = False,
        years: Iterable[int] | None = None, use_cache: bool = True,
    ) -> pd.DataFrame:
        with self._connect() as db:
            shards = self._current_shards(
                db, symbol, timeframe, adjustment, version,
                start=start, end=end, before=before,
            )
        try:
            dataset_manifest = self.package.load_dataset_manifest(
                symbol, timeframe, adjustment, int(version)
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise MarketDataError(
                f"数据集版本清单不可用：{symbol} {timeframe} v{version}: {exc}"
            ) from exc
        if dataset_manifest is None:
            raise DataVersionNotFound(
                f"DATA_VERSION_NOT_FOUND: {symbol} {timeframe} 版本 {version} 不在当前数据代际中"
            )
        published_paths = {
            str(item.get("path") or "").replace("\\", "/")
            for item in dataset_manifest.get("shards", [])
        }
        index_paths = {
            self.package.relative(self._absolute_source_path(row["path"]).resolve())
            for row in shards
        }
        if not index_paths.issubset(published_paths):
            raise MarketDataError(
                f"行情索引引用未发布的分片：{symbol} {timeframe} v{version}"
            )
        if (start is None and end is None and before is None and years is None
                and index_paths != published_paths):
            raise MarketDataError(
                f"行情索引与数据集清单不一致：{symbol} {timeframe} v{version}"
            )
        if years is not None:
            wanted = {int(year) for year in years}
            shards = [
                row for row in shards
                if any(
                    pd.Timestamp(row["period_start"]) < pd.Timestamp(year + 1, 1, 1)
                    and pd.Timestamp(row["period_end"]) > pd.Timestamp(year, 1, 1)
                    for year in wanted
                )
            ]
        if limit is not None and descending and not start and not end:
            recent = sorted(shards, key=lambda row: str(row["period_start"]), reverse=True)
            shards = []
            selected_rows = 0
            for row in recent:
                shards.append(row)
                selected_rows += int(row.get("row_count") or 0)
                if selected_rows >= int(limit):
                    break
        if not shards:
            return pd.DataFrame(columns=list(self._canonical_fields(fields)))
        columns = self._canonical_fields(fields)
        partition_files = shards
        file_key = tuple(str(row["path"]) for row in partition_files)
        key = (
            str(self.root), self.generation, symbol, timeframe, adjustment, int(version),
            _time_text(start) if start is not None else None,
            _time_text(end) if end is not None else None,
            _time_text(before) if before is not None else None,
            tuple(columns), limit, bool(descending), file_key,
        )

        def load():
            for row in partition_files:
                path = self._absolute_source_path(row["path"])
                if not path.is_file():
                    raise MarketDataError("当前行情分片不存在：" + str(path))
            result = self._read_paths(
                [row["path"] for row in partition_files], start=start, end=end,
                before=before, fields=columns, limit=limit, descending=descending,
            )
            return result.sort_values("timestamp", kind="stable").reset_index(drop=True)

        return cached(key, load) if use_cache else load()

    def active_rows(
        self, symbol: str, timeframe: str, adjustment: str = "none",
        start=None, end=None, limit: int | None = None,
    ) -> pd.DataFrame:
        timeframe = normalize_timeframe(timeframe)
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
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
        timeframe = normalize_timeframe(timeframe)
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
        if not dataset:
            return pd.DataFrame()
        frame = self._read_version(
            symbol, timeframe, adjustment, int(dataset["current_version"]),
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
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
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
        *, prepared: bool = False,
    ) -> dict:
        result = frame if prepared else self._prepare_frame(frame)
        if timeframe in MINUTE_TIMEFRAMES:
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
            folder = self.package.partitions_dir / timeframe / f"symbol={symbol}" / f"adjustment={adjustment}" / period_key
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

    def commit_daily_batch(self, prepared: Sequence[dict]) -> list[dict]:
        """Commit daily rows per symbol through the canonical shard publisher."""
        results = []
        for item in prepared:
            results.append(self.commit_bars(
                item["symbol"], "1day", item["frame"],
                source=item.get("source", "import"),
                file_name=item.get("file_name"),
                fingerprint=item.get("fingerprint"),
                params_hash=item.get("params_hash"),
                report=item.get("report") or {},
                metadata=item.get("metadata") or {},
                advance_completed=True,
                import_mode="merge",
            ))
        return results

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
            "is_auction_placeholder", "opening_auction_volume", "adj_factor", "source",
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

    def _commit_merge_bars(
        self, symbol: str, timeframe: str, frame: pd.DataFrame, **kwargs,
    ) -> dict:
        with self.lock(symbol, timeframe, kwargs.get("adjustment", "none")):
            return self._commit_merge_bars_unlocked(
                symbol, timeframe, frame, **kwargs
            )

    def _commit_merge_bars_unlocked(
        self, symbol: str, timeframe: str, frame: pd.DataFrame, *,
        adjustment: str, source: str, file_name: str | None,
        fingerprint: str | None, params_hash: str, report: dict,
        metadata: dict, completed_through: str | None,
        advance_completed: bool, derived: dict | None, replace_range: bool = False,
    ) -> dict:
        """Publish additions and corrections as complete time-bucket shards."""
        with self._connect() as db:
            dataset_row = self._dataset_row(db, symbol, timeframe, adjustment)
            dataset = dict(dataset_row) if dataset_row else None
            current_version = int(dataset["current_version"]) if dataset else 0
            current_shards = self._current_shards(db, symbol, timeframe, adjustment, current_version)

        periods: dict[str, tuple[str, str]] = {}
        for stamp in frame.timestamp:
            key, start, end = self._storage_period(timeframe, stamp)
            periods[key] = (start, end)
        ordered = sorted(periods.items())
        read_start = pd.Timestamp(min(value[0] for _, value in ordered))
        read_end = pd.Timestamp(max(value[1] for _, value in ordered)) - pd.Timedelta(nanoseconds=1)
        # A first import has no published version yet.  Version zero is an
        # empty baseline, not a readable historical snapshot.
        existing = (
            self._read_version(
                symbol, timeframe, adjustment, current_version,
                start=read_start, end=read_end + pd.Timedelta(nanoseconds=1),
                use_cache=False,
            )
            if current_version > 0 else pd.DataFrame()
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
        all_removed = 0
        synthetic_delta = 0
        pending_delta = 0
        new_first = (
            pd.Timestamp(dataset["first_time"])
            if dataset and dataset.get("first_time") else None
        )
        new_last = (
            pd.Timestamp(dataset["last_time"])
            if dataset and dataset.get("last_time") else None
        )
        for key, (start, end) in ordered:
            incoming = frame[
                (frame.timestamp >= pd.Timestamp(start))
                & (frame.timestamp < pd.Timestamp(end))
            ]
            if incoming.empty:
                continue
            current = existing_by_period.get(key, pd.DataFrame())
            original_current = current
            if replace_range and not current.empty:
                incoming_days = set(incoming.timestamp.dt.normalize())
                current = current[
                    ~current.timestamp.dt.normalize().isin(incoming_days)
                ].copy()
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
            row_delta += len(combined) - len(original_current)
            all_removed += max(0, len(original_current) - len(current))
            old_synthetic = int(original_current.is_synthetic.sum()) if "is_synthetic" in original_current else 0
            new_synthetic = int(combined.is_synthetic.sum()) if "is_synthetic" in combined else 0
            synthetic_delta += new_synthetic - old_synthetic
            old_pending = int(original_current.source_verified.eq(False).sum()) if "source_verified" in original_current else 0
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
                "status": (dataset or {}).get("status"), "conflicts": conflicts,
                "file_count": len(current_shards), "shard_count": 0, "bytes": 0, "changed_dates": [],
            }

        version = current_version + 1
        now = utc_now()
        written: list[dict] = []
        skip_periods = set(changed_periods)
        transaction_path = self.package.begin_transaction({
            "kind": "dataset", "status": "writing_shards",
            "symbol": symbol, "timeframe": timeframe, "adjustment": adjustment,
            "base_version": current_version, "version": version,
        })[1]
        try:
            for index, (key, combined) in enumerate(sorted(changed_periods.items()), start=1):
                written_file = self._write_period_shard(symbol, timeframe, adjustment, key, combined)
                _, start, end = self._storage_period(timeframe, combined.timestamp.iloc[0])
                written.append({
                    **written_file, "period_key": key, "period_start": start,
                    "period_end": end, "sequence": int(index),
                })
        except Exception:
            self.package.finish_transaction(transaction_path)
            raise
        self.package.update_transaction(transaction_path, {
            "status": "shards_installed", "shards": [row["path"] for row in written],
        })

        kept_shards = [
            row for row in current_shards if row["period_key"] not in skip_periods
        ]
        total_bytes = (
            sum(int(row.get("size") or 0) for row in kept_shards)
            + sum(int(row["size"]) for row in written)
        )

        row_count = max(0, int((dataset or {}).get("row_count") or 0) + row_delta)
        synthetic_rows = max(0, int((dataset or {}).get("synthetic_rows") or 0) + synthetic_delta)
        pending_after = max(0, int((dataset or {}).get("pending_rows") or 0) + pending_delta)
        status = "pending_validation" if pending_after or report.get("pending_validation") else "ready"
        if report.get("gaps"):
            status = "partial"
        completed_value = (
            completed_through if completed_through is not None
            else _time_text(new_last) if advance_completed and new_last is not None
            else (dataset or {}).get("completed_through")
        )
        item_metadata = {
            **_load((dataset or {}).get("metadata"), {}), **(metadata or {}),
            "layout": "time_partitioned",
        }
        published_payload = {
            "symbol": symbol, "timeframe": timeframe, "adjustment": adjustment,
            "version": version, "row_count": row_count,
            "first_time": _time_text(new_first), "last_time": _time_text(new_last),
            "completed_through": completed_value, "status": status,
            "reason": report.get("reason"), "source": source, "created_at": now,
            "report": report, "raw_rows": max(0, row_count - synthetic_rows),
            "synthetic_rows": synthetic_rows, "pending_rows": pending_after,
            "metadata": item_metadata,
            "shards": self._manifest_shards([*kept_shards, *written]),
            "derived": derived,
        }
        self.package.publish_dataset(published_payload, expected_version=current_version)
        self.package.update_transaction(transaction_path, {"status": "manifest_published"})
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
                    source, len(kept_shards) + len(written),
                    total_bytes,
                    _json(report),
                ),
            )
            self._copy_version_shards(
                db, symbol, timeframe, adjustment, current_version, version,
                skip_periods=skip_periods,
            )
            for row in written:
                db.execute(
                    """INSERT INTO parquet_shards(
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
            if dataset:
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
            else:
                db.execute(
                    """INSERT INTO parquet_datasets(
                       symbol,timeframe,adjustment,current_version,first_time,last_time,
                       completed_through,row_count,raw_rows,synthetic_rows,pending_rows,
                       status,reason,metadata,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, timeframe, adjustment, version, _time_text(new_first),
                        _time_text(new_last), completed_value, row_count,
                        max(0, row_count - synthetic_rows), synthetic_rows, pending_after,
                        status, report.get("reason"), _json(item_metadata), now, now,
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
            for changed_day in sorted(changed_dates):
                db.execute(
                    "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? "
                    "AND adjustment=? AND trade_date=?",
                    (symbol, timeframe, adjustment, changed_day),
                )
            db.execute(
                "UPDATE parquet_derived SET stale=1,updated_at=? WHERE symbol=? "
                "AND adjustment=? AND source_timeframe=? AND source_version<?",
                (now, symbol, adjustment, timeframe, version),
            )
            db.commit()
        self.package.update_transaction(transaction_path, {"status": "database_committed"})
        self.package.finish_transaction(transaction_path)
        return {
            "symbol": symbol, "timeframe": timeframe, "version": version,
            "rows_read": len(frame), "inserted": all_inserted, "updated": all_updated,
            "unchanged": unchanged, "removed": all_removed, "synthetic": int(frame.is_synthetic.sum()),
            "status": status, "conflicts": conflicts, "file_count": len(written),
            "bytes": sum(int(row["size"]) for row in written),
            "changed_dates": sorted(changed_dates), "shard_count": len(written),
            "layout": "time_partitioned",
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
        """Publish a complete input snapshot and withdraw omitted time buckets."""
        with self._connect() as db:
            dataset_row = self._dataset_row(db, symbol, timeframe, adjustment)
            dataset = dict(dataset_row) if dataset_row else None
            current_version = int(dataset["current_version"]) if dataset else 0
            current_file_count = len(self._current_shards(
                db, symbol, timeframe, adjustment, current_version
            ))
        old_rows: dict[pd.Timestamp, dict] = {}
        if current_version > 0:
            current = self._read_version(
                symbol, timeframe, adjustment, current_version,
                use_cache=False,
            )
            old_rows = {
                pd.Timestamp(row["timestamp"]): row
                for row in current.to_dict("records")
            }
            new_rows = {
                pd.Timestamp(row["timestamp"]): row
                for row in frame.to_dict("records")
            }
            identical = (
                len(old_rows) == len(new_rows)
                and old_rows.keys() == new_rows.keys()
                and all(self._same_market_row(old_rows[key], new_rows[key]) for key in old_rows)
            )
            if identical:
                if fingerprint:
                    with self._connect() as db:
                        db.execute(
                            """INSERT OR REPLACE INTO parquet_imports(
                               symbol,timeframe,adjustment,fingerprint,params_hash,parser_version,
                               file_name,status,rows_read,inserted,updated,unchanged,synthetic_rows,
                               version,report,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                symbol, timeframe, adjustment, fingerprint, params_hash,
                                PARSER_VERSION, file_name, "completed", len(frame), 0, 0,
                                len(frame), int(frame.is_synthetic.sum()), current_version,
                                _json(report), utc_now(),
                            ),
                        )
                return {
                    "symbol": symbol, "timeframe": timeframe, "version": current_version,
                    "rows_read": len(frame), "inserted": 0, "updated": 0,
                    "unchanged": len(frame), "removed": 0,
                    "synthetic": int(frame.is_synthetic.sum()),
                    "status": dataset.get("status"), "conflicts": [],
                    "file_count": current_file_count, "shard_count": 0, "bytes": 0,
                    "changed_dates": [], "layout": "time_partitioned",
                }
        new_rows = {
            pd.Timestamp(row["timestamp"]): row
            for row in frame.to_dict("records")
        }
        changed_timestamps = {
            stamp for stamp in old_rows.keys() | new_rows.keys()
            if stamp not in old_rows or stamp not in new_rows
            or not self._same_market_row(old_rows[stamp], new_rows[stamp])
        }
        changed_days = {stamp.normalize() for stamp in changed_timestamps}
        changed_day_values = sorted(str(day.date()) for day in changed_days)
        version = current_version + 1
        now = utc_now()
        bounds: dict[str, tuple[str, str]] = {}
        for stamp in frame.timestamp:
            key, start, end = self._storage_period(timeframe, stamp)
            bounds[key] = (start, end)
        transaction_path = self.package.begin_transaction({
            "kind": "dataset", "status": "writing_shards",
            "symbol": symbol, "timeframe": timeframe, "adjustment": adjustment,
            "base_version": current_version, "version": version,
            "changed_dates": changed_day_values,
        })[1]
        written = []
        try:
            for sequence, (key, (start, end)) in enumerate(sorted(bounds.items()), start=1):
                lower, upper = pd.Timestamp(start), pd.Timestamp(end)
                part = frame[(frame.timestamp >= lower) & (frame.timestamp < upper)]
                if part.empty:
                    continue
                item = self._write_period_shard(symbol, timeframe, adjustment, key, part)
                written.append({
                    **item, "period_key": key, "period_start": start,
                    "period_end": end, "sequence": sequence,
                })
        except Exception:
            self.package.finish_transaction(transaction_path)
            raise
        if not written:
            self.package.finish_transaction(transaction_path)
            raise MarketDataError("完整快照没有可发布的时间分片")
        self.package.update_transaction(transaction_path, {
            "status": "shards_installed", "shards": [row["path"] for row in written],
        })
        synthetic_rows = int(frame.is_synthetic.sum())
        pending_rows = int(frame.source_verified.eq(False).sum())
        status = "pending_validation" if pending_rows or report.get("pending_validation") else "ready"
        if report.get("gaps"):
            status = "partial"
        first_time = frame.timestamp.min()
        last_time = frame.timestamp.max()
        completed_value = completed_through if completed_through is not None else (
            _time_text(last_time) if advance_completed else (dataset or {}).get("completed_through")
        )
        item_metadata = {
            **_load((dataset or {}).get("metadata"), {}), **(metadata or {}),
            "layout": "time_partitioned", "import_mode": "snapshot",
        }
        self.package.publish_dataset({
            "symbol": symbol, "timeframe": timeframe, "adjustment": adjustment,
            "version": version, "row_count": len(frame),
            "first_time": _time_text(first_time), "last_time": _time_text(last_time),
            "completed_through": completed_value, "status": status,
            "reason": report.get("reason"), "source": source, "created_at": now,
            "report": report, "raw_rows": len(frame) - synthetic_rows,
            "synthetic_rows": synthetic_rows, "pending_rows": pending_rows,
            "metadata": item_metadata, "shards": self._manifest_shards(written),
            "derived": derived,
        }, expected_version=current_version)
        self.package.update_transaction(transaction_path, {"status": "manifest_published"})
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO parquet_versions(
                   symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                   synthetic_rows,pending_rows,first_time,last_time,status,source,
                   file_count,bytes,report) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, timeframe, adjustment, version, now, len(frame),
                    len(frame) - synthetic_rows, synthetic_rows, pending_rows,
                    _time_text(first_time), _time_text(last_time), status, source,
                    len(written), sum(int(row["size"]) for row in written), _json(report),
                ),
            )
            for row in written:
                db.execute(
                    """INSERT INTO parquet_shards(
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
            dataset_values = (
                _time_text(first_time), _time_text(last_time), completed_value,
                len(frame), len(frame) - synthetic_rows, synthetic_rows, pending_rows,
                status, report.get("reason"), _json(item_metadata), now,
            )
            if dataset:
                db.execute(
                    """UPDATE parquet_datasets SET current_version=?,first_time=?,last_time=?,
                       completed_through=?,row_count=?,raw_rows=?,synthetic_rows=?,pending_rows=?,
                       status=?,reason=?,metadata=?,updated_at=?
                       WHERE symbol=? AND timeframe=? AND adjustment=?""",
                    (version, *dataset_values, symbol, timeframe, adjustment),
                )
            else:
                db.execute(
                    """INSERT INTO parquet_datasets(
                       symbol,timeframe,adjustment,current_version,first_time,last_time,
                       completed_through,row_count,raw_rows,synthetic_rows,pending_rows,
                       status,reason,metadata,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, timeframe, adjustment, version,
                        _time_text(first_time), _time_text(last_time), completed_value,
                        len(frame), len(frame) - synthetic_rows, synthetic_rows,
                        pending_rows, status, report.get("reason"), _json(item_metadata),
                        now, now,
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
            if derived:
                db.execute(
                    """INSERT OR REPLACE INTO parquet_derived(
                       symbol,timeframe,adjustment,source_timeframe,target_timeframe,
                       source_version,start_time,end_time,stale,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,0,?)""",
                    (
                        symbol, timeframe, adjustment, derived["source_timeframe"],
                        timeframe, int(derived["source_version"]), derived.get("start_time"),
                        derived.get("end_time"), now,
                    ),
                )
            for start in range(0, len(changed_day_values), 400):
                changed_chunk = changed_day_values[start:start + 400]
                if not changed_chunk:
                    continue
                placeholders = ",".join("?" for _ in changed_chunk)
                db.execute(
                    "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? "
                    "AND adjustment=? AND trade_date IN (" + placeholders + ")",
                    (symbol, timeframe, adjustment, *changed_chunk),
                )
            for changed_day in changed_day_values:
                db.execute(
                    "DELETE FROM parquet_coverage WHERE symbol=? AND timeframe=? "
                    "AND adjustment=? AND date(start_time)<=? AND date(end_time)>=?",
                    (symbol, timeframe, adjustment, changed_day, changed_day),
                )
            db.execute(
                "UPDATE parquet_derived SET stale=1,updated_at=? WHERE symbol=? "
                "AND adjustment=? AND source_timeframe=? AND source_version<?",
                (now, symbol, adjustment, timeframe, version),
            )
            db.commit()
        self.package.update_transaction(transaction_path, {"status": "database_committed"})
        self.package.finish_transaction(transaction_path)
        return {
            "symbol": symbol, "timeframe": timeframe, "version": version,
            "rows_read": len(frame), "inserted": len(frame), "updated": 0,
            "unchanged": 0, "removed": 0, "synthetic": synthetic_rows,
            "status": status, "conflicts": [], "file_count": len(written), "shard_count": len(written),
            "bytes": sum(int(row["size"]) for row in written),
            "changed_dates": sorted(str(day.date()) for day in changed_days),
            "layout": "time_partitioned",
        }

    def commit_bars(
        self, symbol: str, timeframe: str, frame: pd.DataFrame, *,
        adjustment: str = "none", source: str = "import", file_name: str | None = None,
        fingerprint: str | None = None, params_hash: str | None = None,
        report: dict | None = None, metadata: dict | None = None,
        completed_through: str | None = None, derived: dict | None = None,
        replace_range: bool = False, advance_completed: bool = True,
        import_mode: str = "merge",
    ) -> dict:
        """Publish merged or replacement input as immutable time-bucket shards."""
        timeframe = normalize_timeframe(timeframe)
        self._assert_writable()
        frame = self._prepare_frame(frame)
        if frame.empty:
            raise MarketDataError("没有可提交的行情记录")
        if import_mode not in {"merge", "snapshot"}:
            raise MarketDataError("import_mode 应为 merge 或 snapshot")
        params_hash = params_hash or hashlib.sha256(
            f"{source}|{adjustment}".encode()
        ).hexdigest()
        report = report or {}
        metadata = metadata or {}
        if import_mode == "snapshot":
            return self._commit_snapshot_bars(
                symbol, timeframe, frame, adjustment=adjustment, source=source,
                file_name=file_name, fingerprint=fingerprint, params_hash=params_hash,
                report=report, metadata=metadata, completed_through=completed_through,
                advance_completed=advance_completed, derived=derived,
            )
        return self._commit_merge_bars(
            symbol, timeframe, frame, adjustment=adjustment, source=source,
            file_name=file_name, fingerprint=fingerprint, params_hash=params_hash,
            report=report, metadata=metadata, completed_through=completed_through,
            advance_completed=advance_completed, derived=derived,
            replace_range=replace_range,
        )

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
            version = int(dataset["current_version"]) if dataset else 0
            shards = self._current_shards(db, symbol, timeframe, adjustment, version)
        return {
            "symbol": symbol, "timeframe": timeframe,
            "protected": bool(shards), "sources": [],
        }

    def archive_dataset(
        self, symbol: str, timeframe: str, adjustment: str = "none",
        progress=lambda *_: None, cancelled=lambda: False,
    ) -> dict:
        with self._connect() as db:
            dataset = self._dataset_row(db, symbol, timeframe, adjustment)
            version = int(dataset["current_version"]) if dataset else 0
            shards = self._current_shards(db, symbol, timeframe, adjustment, version)
        if cancelled():
            raise InterruptedError("任务已取消")
        return {
            "symbol": symbol, "timeframe": timeframe,
            "archived": bool(shards), "copied": 0, "reused": 0,
            "bytes": 0, "status": "completed",
            "reason": "内容摘要分片不可变，无需另存副本",
        }

    def compact_dataset(
        self, symbol: str, timeframe: str, adjustment: str = "none",
    ) -> dict:
        """Canonical time buckets are complete shards and are not span-compacted."""
        dataset = self.dataset(symbol, normalize_timeframe(timeframe), adjustment)
        if not dataset:
            raise MarketDataError("没有可整理的行情")
        return {
            "status": "skipped",
            "reason": "规范年/月分片不跨周期合并",
            "version": int(dataset["current_version"]),
        }

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
        self._assert_writable()
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
        frame = self._read_version(
            symbol, timeframe, adjustment, int(dataset["current_version"]),
            fields=("timestamp",), use_cache=False,
        )
        return int(frame.timestamp.dt.normalize().nunique()) if not frame.empty else 0

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
