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
        symbol = normalize_symbol(symbol)
        exchange = symbol.split(".")[1]
        board = "star" if symbol.startswith(("688", "689")) else "chinext" if symbol.startswith("3") else "main"
        with self.db() as db:
            old = db.execute("SELECT * FROM instruments WHERE symbol=?", (symbol,)).fetchone()
            db.execute("INSERT OR REPLACE INTO instruments VALUES(?,?,?,?,?,?)", (
                symbol, name or (old["name"] if old else symbol), exchange, board,
                metadata.get("listing_date") or (old["listing_date"] if old else None),
                metadata.get("delisting_date") or (old["delisting_date"] if old else None),
            ))

    def instruments(self, query: str = "", limit: int = 100) -> list[dict]:
        with self.db() as db:
            rows = db.execute("SELECT * FROM instruments WHERE symbol LIKE ? OR name LIKE ? ORDER BY symbol LIMIT ?", (f"%{query}%", f"%{query}%", limit)).fetchall()
        return [dict(row) for row in rows]

    def add_watch(self, symbol: str, name: str = ""):
        self.put_instrument(symbol, name)
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO watchlist VALUES(?,?)", (normalize_symbol(symbol), now_iso()))

    def watches(self):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT i.* FROM watchlist w JOIN instruments i USING(symbol) ORDER BY w.created_at").fetchall()]

    def state(self, key: str, value: Any = ...):
        with self.db() as db:
            if value is not ...:
                db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, dumps(value)))
                return value
            row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

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
        paths = [str(self.root / name) for name in files]
        with duckdb.connect() as connection:
            relation = connection.read_parquet(paths, union_by_name=True)
            if column and start is not None:
                relation = relation.filter(f'"{column}" >= TIMESTAMP \'{pd.Timestamp(start).isoformat()}\'')
            if column and end is not None:
                relation = relation.filter(f'"{column}" < TIMESTAMP \'{(pd.Timestamp(end).normalize() + pd.Timedelta(days=1)).isoformat()}\'')
            return relation.df()

    def publish(self, symbol: str, minute: pd.DataFrame, daily: pd.DataFrame, actions: pd.DataFrame, calendar: pd.DataFrame, metadata: dict, profiles: dict) -> dict:
        symbol = normalize_symbol(symbol)
        files = {
            "minute": self.write_frame(symbol, "1m", minute, "timestamp"),
            "daily": self.write_frame(symbol, "1d", daily, "date"),
            "actions": self.write_frame(symbol, "actions", actions, "ex_date"),
            "calendar": self.write_frame(symbol, "calendar", calendar, "date"),
        }
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

    def snapshot(self, symbol: str, version: str | None = None) -> dict | None:
        with self.db() as db:
            if version:
                row = db.execute("SELECT manifest FROM snapshots WHERE symbol=? AND version=?", (normalize_symbol(symbol), version)).fetchone()
            else:
                row = db.execute("SELECT s.manifest FROM snapshots s JOIN active_snapshots a USING(version) WHERE a.symbol=?", (normalize_symbol(symbol),)).fetchone()
        return json.loads(row[0]) if row else None

    def new_job(self, kind: str, payload: dict) -> str:
        job_id = uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT INTO jobs(id,kind,status,progress,message,payload,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (job_id, kind, "queued", 0, "等待执行", dumps(payload), now_iso(), now_iso()))
        return job_id

    def update_job(self, job_id: str, **values):
        allowed = {"status", "progress", "message", "error", "result_id", "cancel_requested", "pid"}
        values = {k: v for k, v in values.items() if k in allowed}
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
