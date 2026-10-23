"""Transactional upgrade of the rebuildable shard index; no market files move."""
from __future__ import annotations

import sqlite3

from .models import MarketDataError


SHARD_TABLE_SQL = """CREATE TABLE IF NOT EXISTS parquet_shards(
    symbol TEXT NOT NULL, timeframe TEXT NOT NULL, adjustment TEXT NOT NULL,
    version INTEGER NOT NULL, sequence INTEGER NOT NULL,
    period_key TEXT NOT NULL, period_start TEXT NOT NULL, period_end TEXT NOT NULL,
    path TEXT NOT NULL, sha256 TEXT NOT NULL, row_count INTEGER NOT NULL,
    first_time TEXT, last_time TEXT, size INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    PRIMARY KEY(symbol,timeframe,adjustment,version,period_key)
)"""


def initialize_shard_index(db: sqlite3.Connection) -> None:
    """Rename the former shard-reference table atomically, retaining every version.

    A populated legacy base-file table cannot represent canonical complete
    buckets and is rejected rather than silently discarded or overlaid.
    """
    db.execute("SAVEPOINT shard_index_upgrade")
    try:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "parquet_files" in tables:
            if db.execute("SELECT 1 FROM parquet_files LIMIT 1").fetchone():
                raise MarketDataError("旧底座索引非空，无法自动统一为规范分片；原索引未更改")
            db.execute("DROP TABLE parquet_files")
        if "parquet_increments" in tables:
            old_nonempty = db.execute("SELECT 1 FROM parquet_increments LIMIT 1").fetchone()
            if "parquet_shards" in tables:
                new_nonempty = db.execute("SELECT 1 FROM parquet_shards LIMIT 1").fetchone()
                if old_nonempty and new_nonempty:
                    raise MarketDataError("新旧分片索引同时含有记录，拒绝猜测或合并；原索引未更改")
                if old_nonempty:
                    db.execute("DROP TABLE parquet_shards")
                else:
                    db.execute("DROP TABLE parquet_increments")
            if "parquet_shards" not in tables or old_nonempty:
                db.execute("ALTER TABLE parquet_increments RENAME TO parquet_shards")
        db.execute(SHARD_TABLE_SQL)
        db.execute("DROP INDEX IF EXISTS parquet_increments_version")
        db.execute("CREATE INDEX IF NOT EXISTS parquet_shards_version "
                   "ON parquet_shards(symbol,timeframe,adjustment,version,sequence)")
        db.execute("RELEASE SAVEPOINT shard_index_upgrade")
    except Exception:
        db.execute("ROLLBACK TO SAVEPOINT shard_index_upgrade")
        db.execute("RELEASE SAVEPOINT shard_index_upgrade")
        raise
