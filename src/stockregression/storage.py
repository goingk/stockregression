from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from .models import normalize_symbol


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat() if not pd.isna(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dumps(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "catalog.sqlite"
        self.lock = FileLock(str(self.root / "sync.lock"), timeout=5)
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS instruments(symbol TEXT PRIMARY KEY,name TEXT,exchange TEXT,board TEXT,listing_date TEXT,delisting_date TEXT);
                CREATE TABLE IF NOT EXISTS watchlist(symbol TEXT PRIMARY KEY,created_at TEXT);
                CREATE TABLE IF NOT EXISTS snapshots(version TEXT PRIMARY KEY,symbol TEXT,created_at TEXT,manifest TEXT);
                CREATE TABLE IF NOT EXISTS active_snapshots(symbol TEXT PRIMARY KEY,version TEXT);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT,status TEXT,progress REAL,message TEXT,error TEXT,result_id TEXT,payload TEXT,created_at TEXT,updated_at TEXT,cancel_requested INTEGER DEFAULT 0,pid INTEGER);
                CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,kind TEXT,symbol TEXT,created_at TEXT,title TEXT,path TEXT);
                CREATE TABLE IF NOT EXISTS configs(id TEXT PRIMARY KEY,name TEXT,config TEXT,created_at TEXT);
                CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT);
                CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status,created_at);
            """)
            if "metadata" not in {row[1] for row in db.execute("PRAGMA table_info(instruments)")}:
                db.execute("ALTER TABLE instruments ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sync_items(run_key TEXT,symbol TEXT,status TEXT,mode TEXT,daily_through TEXT,minute_through TEXT,error TEXT,updated_at TEXT,PRIMARY KEY(run_key,symbol));
                CREATE INDEX IF NOT EXISTS sync_items_status ON sync_items(run_key,status);
                CREATE TABLE IF NOT EXISTS market_auxiliary(
                    symbol TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    version TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL, PRIMARY KEY(symbol,kind)
                );
            """)
            db.executescript("""
                CREATE TABLE IF NOT EXISTS research_registry(
                    registry_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    input_start TEXT,
                    input_end TEXT,
                    data_version TEXT,
                    validation_intervals TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    overlapping_prior_studies INTEGER NOT NULL DEFAULT 0,
                    exact_repeat_prior_studies INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS research_registry_symbol ON research_registry(symbol,created_at);
            """)
            registry_columns = {row[1] for row in db.execute("PRAGMA table_info(research_registry)")}
            for column in ("overlapping_prior_studies", "exact_repeat_prior_studies"):
                if column not in registry_columns:
                    db.execute(f"ALTER TABLE research_registry ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            sync_columns = {row[1] for row in db.execute("PRAGMA table_info(sync_items)")}
            for column in ("daily_status", "minute_status"):
                if column not in sync_columns:
                    db.execute(f"ALTER TABLE sync_items ADD COLUMN {column} TEXT NOT NULL DEFAULT 'pending'")

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def put_instrument(self, symbol: str, name: str = "", **metadata):
        from .rules import instrument_terms
        symbol = normalize_symbol(symbol)
        exchange = symbol.split(".")[1]
        board = "star" if symbol.startswith(("688", "689")) else "chinext" if symbol.startswith("3") else "main"
        with self.db() as db:
            old = db.execute("SELECT * FROM instruments WHERE symbol=?", (symbol,)).fetchone()
            merged = {**(json.loads(old["metadata"] or "{}") if old else {}), **{key: value for key, value in metadata.items() if value is not None}}
            merged = instrument_terms(symbol, {**merged, "name": name or (old["name"] if old else symbol)})
            db.execute("INSERT OR REPLACE INTO instruments(symbol,name,exchange,board,listing_date,delisting_date,metadata) VALUES(?,?,?,?,?,?,?)", (
                symbol, merged["name"], exchange, merged["board"],
                metadata.get("listing_date") or (old["listing_date"] if old else None),
                metadata.get("delisting_date") or (old["delisting_date"] if old else None),
                dumps(merged),
            ))

    def instruments(self, query: str = "", limit: int | None = 100, asset_type: str | None = None, offset: int = 0) -> list[dict]:
        with self.db() as db:
            rows = db.execute("SELECT * FROM instruments WHERE symbol LIKE ? OR name LIKE ? ORDER BY symbol", (f"%{query}%", f"%{query}%")).fetchall()
        items = [self._instrument(row) for row in rows]
        if asset_type:
            items = [item for item in items if item["asset_type"] == asset_type]
        return items[offset:] if limit is None else items[offset:offset + limit]

    @staticmethod
    def _instrument(row):
        from .rules import instrument_terms
        item = dict(row)
        metadata = json.loads(item.pop("metadata", "{}") or "{}")
        return instrument_terms(item["symbol"], {**item, **metadata})

    def instrument(self, symbol):
        with self.db() as db:
            row = db.execute("SELECT * FROM instruments WHERE symbol=?", (normalize_symbol(symbol),)).fetchone()
        return self._instrument(row) if row else None

    def add_watch(self, symbol: str, name: str = ""):
        self.put_instrument(symbol, name)
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO watchlist VALUES(?,?)", (normalize_symbol(symbol), now_iso()))

    def watches(self):
        with self.db() as db:
            return [self._instrument(r) for r in db.execute("SELECT i.* FROM watchlist w JOIN instruments i USING(symbol) ORDER BY w.created_at").fetchall()]

    def sync_item(self, run_key, symbol, **values):
        with self.db() as db:
            row = db.execute("SELECT * FROM sync_items WHERE run_key=? AND symbol=?", (run_key, symbol)).fetchone()
            if not values:
                return dict(row) if row else None
            item = {"status": "pending", "mode": "history", "daily_through": None, "minute_through": None, "error": None, "daily_status": "pending", "minute_status": "pending", **(dict(row) if row else {}), **values}
            db.execute("INSERT OR REPLACE INTO sync_items(run_key,symbol,status,mode,daily_through,minute_through,error,updated_at,daily_status,minute_status) VALUES(?,?,?,?,?,?,?,?,?,?)", (run_key, symbol, item["status"], item["mode"], item["daily_through"], item["minute_through"], item["error"], now_iso(), item["daily_status"], item["minute_status"]))

    def sync_status(self, run_key=None, status=None, limit=100, offset=0):
        conditions, arguments = [], []
        for column, value in (("run_key", run_key), ("status", status)):
            if value:
                conditions.append(column + "=?")
                arguments.append(value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.db() as db:
            counts = {row[0]: row[1] for row in db.execute("SELECT status,COUNT(*) FROM sync_items" + where + " GROUP BY status", arguments)}
            rows = db.execute("SELECT * FROM sync_items" + where + " ORDER BY updated_at DESC,symbol LIMIT ? OFFSET ?", [*arguments, limit, offset]).fetchall()
        return {"items": [dict(row) for row in rows], "counts": counts, "total": sum(counts.values())}

    def state(self, key: str, value: Any = ...):
        with self.db() as db:
            if value is not ...:
                db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, dumps(value)))
                return value
            row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def save_market_auxiliary(self, symbol: str, kind: str, payload: Any, *, version="local", verified=False):
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO market_auxiliary(symbol,kind,payload,version,verified,updated_at) VALUES(?,?,?,?,?,?)",
                (normalize_symbol(symbol), kind, dumps(payload), version, int(verified), now_iso()),
            )

    def market_auxiliary(self, symbol: str, kind: str):
        with self.db() as db:
            row = db.execute(
                "SELECT * FROM market_auxiliary WHERE symbol=? AND kind=?",
                (normalize_symbol(symbol), kind),
            ).fetchone()
        return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    def write_frame(self, symbol: str, kind: str, frame: pd.DataFrame, date_column: str) -> list[str]:
        if frame.empty:
            return []
        frame = frame.copy().sort_values(date_column, kind="stable").reset_index(drop=True)
        paths = []
        years = pd.to_datetime(frame[date_column]).dt.year.fillna(0).astype(int)
        for year, part in frame.groupby(years, sort=True):
            folder = self.root / "parquet" / symbol / kind / str(year)
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / f"{uuid.uuid4().hex}.writing"
            table = pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False)
            pq.write_table(table, temporary, compression="zstd")
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            target = folder / f"{digest}.parquet"
            if target.exists():
                temporary.unlink()
            else:
                temporary.replace(target)
            paths.append(target.relative_to(self.root).as_posix())
        return paths

    def read_files(self, files: list[str], column: str | None = None, start=None, end=None) -> pd.DataFrame:
        if not files:
            return pd.DataFrame()
        virtual = [name for name in files if str(name).startswith("market:")]
        if virtual:
            parts = []
            for name in virtual:
                pieces = str(name).split(":")
                if len(pieces) != 3:
                    raise ValueError("无效的市场数据库引用：" + str(name))
                timeframe, symbol = pieces[1], pieces[2]
                from .market_data.repository import MarketDataRepository
                part = MarketDataRepository(self.root).bars(
                    symbol, timeframe, start=start, end=end
                )
                if not part.empty:
                    if column == "date" or timeframe in {"1day", "1week", "1month"}:
                        part["date"] = pd.to_datetime(part.timestamp).dt.normalize()
                    parts.append(part)
            return pd.concat(parts, ignore_index=True).sort_values(column or "timestamp").reset_index(drop=True) if parts else pd.DataFrame()
        paths = [str(self.root / name) for name in files]
        with duckdb.connect() as connection:
            relation = connection.read_parquet(paths, union_by_name=True)
            if column and start is not None:
                relation = relation.filter(f'"{column}" >= TIMESTAMP \'{pd.Timestamp(start).isoformat()}\'')
            if column and end is not None:
                relation = relation.filter(f'"{column}" < TIMESTAMP \'{(pd.Timestamp(end).normalize() + pd.Timedelta(days=1)).isoformat()}\'')
            return relation.df()

    def publish(self, symbol: str, minute: pd.DataFrame, daily: pd.DataFrame, actions: pd.DataFrame, calendar: pd.DataFrame, metadata: dict, profiles: dict, incremental: bool = False) -> dict:
        symbol = normalize_symbol(symbol)
        previous = self.snapshot(symbol) if incremental else None
        files = {}
        for key, kind, frame, column in [("minute", "1m", minute, "timestamp"), ("daily", "1d", daily, "date"), ("actions", "actions", actions, "ex_date"), ("calendar", "calendar", calendar, "date")]:
            old_paths = previous["files"].get(key, []) if previous else []
            if previous:
                if frame.empty:
                    files[key] = old_paths
                    continue
                years = {str(year) for year in pd.to_datetime(frame[column]).dt.year.unique()}
                touched = [path for path in old_paths if path.split("/")[3] in years]
                frame = pd.concat([self.read_files(touched), frame], ignore_index=True).sort_values(column, kind="stable").drop_duplicates(column, keep="last")
                untouched = [path for path in old_paths if path not in touched]
            else:
                untouched = []
            files[key] = sorted(untouched + self.write_frame(symbol, kind, frame, column))
        content = {"symbol": symbol, "files": files, "metadata": metadata, "profiles": profiles}
        version = hashlib.sha256(dumps(content).encode()).hexdigest()[:24]
        manifest = {**content, "version": version, "created_at": now_iso()}
        folder = self.root / "manifests"
        folder.mkdir(exist_ok=True)
        target = folder / f"{version}.json"
        if not target.exists():
            temporary = target.with_suffix(".writing")
            temporary.write_text(dumps(manifest), encoding="utf-8")
            temporary.replace(target)
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?)", (version, symbol, manifest["created_at"], dumps(manifest)))
            db.execute("INSERT OR REPLACE INTO active_snapshots VALUES(?,?)", (symbol, version))
        return self.snapshot(symbol, version)

    def put_market_snapshot(self, manifest: dict, symbol: str):
        """Register a snapshot whose bar payload lives in frequency databases."""
        symbol = normalize_symbol(symbol)
        version = manifest["version"]
        created = manifest.get("created_at") or now_iso()
        manifest = {**manifest, "created_at": created}
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?)", (version, symbol, created, dumps(manifest)))
            db.execute("INSERT OR REPLACE INTO active_snapshots VALUES(?,?)", (symbol, version))
        return self.snapshot(symbol, version)

    def snapshot(self, symbol: str, version: str | None = None) -> dict | None:
        symbol = normalize_symbol(symbol)
        with self.db() as db:
            if version:
                row = db.execute("SELECT manifest FROM snapshots WHERE symbol=? AND version=?", (symbol, version)).fetchone()
            else:
                row = db.execute("SELECT s.manifest FROM snapshots s JOIN active_snapshots a USING(version) WHERE a.symbol=?", (symbol,)).fetchone()
        if row:
            manifest = json.loads(row[0])
            if version is None and manifest.get("market_data"):
                from .market_data.repository import MarketDataRepository
                live = MarketDataRepository(self.root).active_manifest(symbol)
                if live and live.get("version") != manifest.get("version"):
                    instrument = self.instrument(symbol) or {"symbol": symbol, "name": symbol}
                    live["metadata"] = {**instrument, "is_demo": False, "warnings": []}
                    self.put_market_snapshot(live, symbol)
                    return live
            return manifest
        if version is None:
            from .market_data.repository import MarketDataRepository
            live = MarketDataRepository(self.root).active_manifest(symbol)
            if live:
                # A live manifest can be returned to an API caller and then
                # stored in a job payload. Persist it before handing it out so
                # ``snapshot(symbol, version)`` can resolve the same version in
                # the worker process.
                instrument = self.instrument(symbol) or {"symbol": symbol, "name": symbol}
                live["metadata"] = {**instrument, "is_demo": False, "warnings": []}
                self.put_market_snapshot(live, symbol)
            return live
        return None

    def new_job(self, kind: str, payload: dict) -> str:
        job_id = uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT INTO jobs(id,kind,status,progress,message,payload,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (job_id, kind, "queued", 0, "等待执行", dumps(payload), now_iso(), now_iso()))
        return job_id

    def update_job(self, job_id: str, **values):
        allowed = {"status", "progress", "message", "error", "result_id", "cancel_requested", "pid", "payload"}
        values = {k: v for k, v in values.items() if k in allowed}
        if "payload" in values:
            values["payload"] = dumps(values["payload"])
        values["updated_at"] = now_iso()
        with self.db() as db:
            db.execute(f"UPDATE jobs SET {','.join(k+'=?' for k in values)} WHERE id=?", (*values.values(), job_id))

    def job(self, job_id: str) -> dict | None:
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["cancel_requested"] = bool(result["cancel_requested"])
        return result

    def jobs(self, limit=100):
        with self.db() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))]
        return [self.job(job_id) for job_id in ids]

    def recent_submitted_symbols(self, limit: int = 10) -> list[dict[str, str]]:
        """Return the most recently submitted symbols from formal backtest jobs."""
        if limit <= 0:
            return []
        accepted_kinds = {"backtest", "optimization", "optimize", "batch", "batch_backtest"}
        with self.db() as db:
            rows = db.execute(
                "SELECT kind,payload,created_at FROM jobs ORDER BY created_at DESC,rowid DESC"
            ).fetchall()

        recent: dict[str, str] = {}

        def add(symbol: Any, submitted_at: Any) -> None:
            if not isinstance(symbol, str) or not isinstance(submitted_at, str) or symbol in ("",):
                return
            try:
                normalized = normalize_symbol(symbol)
            except (TypeError, ValueError):
                return
            recent.setdefault(normalized, submitted_at)

        def config_symbol(value: Any) -> str | None:
            if not isinstance(value, dict):
                return None
            return value.get("symbol") if isinstance(value.get("symbol"), str) else None

        for row in rows:
            if row["kind"] not in accepted_kinds:
                continue
            try:
                payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue

            # Older batch records kept the original submitted configs together.
            configs = payload.get("configs")
            if isinstance(configs, list):
                for config in configs:
                    if isinstance(config, dict) and isinstance(config.get("config"), dict):
                        config = config["config"]
                    add(config_symbol(config), row["created_at"])

            config = payload.get("config")
            if not isinstance(config, dict):
                config = payload
            base = config.get("base")
            symbol = config_symbol(base) if isinstance(base, dict) else config_symbol(config)
            add(symbol, row["created_at"])

            if len(recent) >= limit:
                break

        return [
            {"symbol": symbol, "last_submitted_at": submitted_at}
            for symbol, submitted_at in list(recent.items())[:limit]
        ]

    def save_result(self, kind: str, result: dict, symbol: str, title: str) -> str:
        result_id = uuid.uuid4().hex
        folder = self.root / "results"
        folder.mkdir(exist_ok=True)
        target = folder / f"{result_id}.json"
        target.write_text(dumps({**result, "id": result_id}), encoding="utf-8")
        with self.db() as db:
            db.execute("INSERT INTO results VALUES(?,?,?,?,?,?)", (result_id, kind, symbol, now_iso(), title, str(target.relative_to(self.root))))
        return result_id

    def result(self, result_id: str) -> dict | None:
        with self.db() as db:
            row = db.execute("SELECT path FROM results WHERE id=?", (result_id,)).fetchone()
        return json.loads((self.root / row[0]).read_text(encoding="utf-8")) if row else None

    def results(self):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT id,kind,symbol,created_at,title FROM results ORDER BY created_at DESC LIMIT 200")]

    def register_research(
        self,
        registry_id: str,
        protocol_hash: str,
        symbol: str,
        input_start: str | None,
        input_end: str | None,
        data_version: str | None,
        validation_intervals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Record a v2 study and count previously used validation dates.

        The job id is the registry key so a process retry cannot create a second
        audit event. Interval overlap intentionally ignores protocol hash and
        snapshot version: reusing the same dates with a changed seed or snapshot
        is still prior exposure to those dates.
        """
        symbol = normalize_symbol(symbol)
        intervals = []
        seen = set()
        for item in validation_intervals:
            try:
                start = str(pd.Timestamp(item["start"]).date())
                end = str(pd.Timestamp(item["end"]).date())
                if end < start:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            intervals.append({"start": start, "end": end, **({"kind": item["kind"]} if item.get("kind") else {})})
        intervals.sort(key=lambda item: (item["start"], item["end"], item.get("kind", "")))
        encoded_intervals = dumps(intervals)
        with self.db() as db:
            # Serialize the read/overlap/insert sequence across concurrent jobs.
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM research_registry WHERE registry_id=?", (registry_id,)).fetchone()
            if existing:
                intervals = json.loads(existing["validation_intervals"] or "[]")
                overlap_count = int(existing["overlapping_prior_studies"])
                exact_count = int(existing["exact_repeat_prior_studies"])
                return {
                    "registry_id": existing["registry_id"],
                    "protocol_hash": existing["protocol_hash"],
                    "symbol": existing["symbol"],
                    "input_start": existing["input_start"],
                    "input_end": existing["input_end"],
                    "data_version": existing["data_version"],
                    "validation_intervals": intervals,
                    "overlapping_prior_studies": overlap_count,
                    "exact_repeat_prior_studies": exact_count,
                    "recorded_overlap_status": "overlap_detected" if overlap_count else "no_recorded_overlap",
                    "human_untouched_verified": False,
                    "idempotent_replay": True,
                }
            prior = db.execute(
                "SELECT registry_id,validation_intervals FROM research_registry WHERE symbol=?",
                (symbol,),
            ).fetchall()
            current_pairs = {(item["start"], item["end"]) for item in intervals}
            overlapping = set()
            exact = set()
            for row in prior:
                prior_intervals = json.loads(row["validation_intervals"] or "[]")
                if any(
                    pd.Timestamp(old["start"]) <= pd.Timestamp(new["end"])
                    and pd.Timestamp(new["start"]) <= pd.Timestamp(old["end"])
                    for old in prior_intervals
                    for new in intervals
                ):
                    overlapping.add(row["registry_id"])
                if current_pairs.intersection({(item["start"], item["end"]) for item in prior_intervals}):
                    exact.add(row["registry_id"])
            db.execute(
                "INSERT INTO research_registry VALUES(?,?,?,?,?,?,?,?,?,?)",
                (registry_id, protocol_hash, symbol, input_start, input_end, data_version, encoded_intervals, now_iso(), len(overlapping), len(exact)),
            )
        return {
            "registry_id": registry_id,
            "protocol_hash": protocol_hash,
            "symbol": symbol,
            "input_start": input_start,
            "input_end": input_end,
            "data_version": data_version,
            "validation_intervals": intervals,
            "overlapping_prior_studies": len(overlapping),
            "exact_repeat_prior_studies": len(exact),
            "recorded_overlap_status": "overlap_detected" if overlapping else "no_recorded_overlap",
            "human_untouched_verified": False,
            "idempotent_replay": False,
        }
