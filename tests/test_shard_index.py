from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from stockregression.market_data.models import MarketDataError
from stockregression.market_data.repository import MarketDataRepository
from stockregression.market_data.shard_index import SHARD_TABLE_SQL, initialize_shard_index
from stockregression.storage import Store


def _tables(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _legacy_index(db):
    db.execute(SHARD_TABLE_SQL.replace("parquet_shards", "parquet_increments"))
    db.execute("CREATE TABLE parquet_files(path TEXT)")
    db.execute("CREATE TABLE business(id INTEGER PRIMARY KEY, payload TEXT)")
    db.execute("INSERT INTO business VALUES(7,'keep me')")
    refs = [
        ("510300.SH", "1min", "none", version, 1, "month=2026-01",
         "2026-01-01", "2026-02-01", f"partitions/{version}.parquet",
         str(version) * 64, 240, "2026-01-05 09:31:00", "2026-01-05 15:00:00",
         1024, "market-bar-v1")
        for version in (1, 2)
    ]
    db.executemany("INSERT INTO parquet_increments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", refs)
    db.commit()
    return refs


@pytest.mark.parametrize("existing_unified", [False, True])
def test_index_upgrade_preserves_all_versions_and_business_rows(existing_unified):
    with sqlite3.connect(":memory:") as db:
        refs = _legacy_index(db)
        if existing_unified:
            db.execute(SHARD_TABLE_SQL)
        initialize_shard_index(db)
        initialize_shard_index(db)
        assert _tables(db) == {"parquet_shards", "business"}
        assert db.execute("SELECT * FROM parquet_shards ORDER BY version").fetchall() == refs
        assert db.execute("SELECT * FROM business").fetchall() == [(7, "keep me")]


@pytest.mark.parametrize("conflict", ["base", "both_indexes", "index_creation"])
def test_failed_index_upgrade_rolls_back_every_schema_change(conflict):
    with sqlite3.connect(":memory:") as db:
        refs = _legacy_index(db)
        if conflict == "base":
            db.execute("INSERT INTO parquet_files VALUES('old-base.parquet')")
        elif conflict == "both_indexes":
            db.execute(SHARD_TABLE_SQL)
            db.execute("INSERT INTO parquet_shards SELECT * FROM parquet_increments")
        db.commit()
        before = db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall()
        if conflict == "index_creation":
            db.set_authorizer(lambda action, *_args:
                              sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_INDEX
                              else sqlite3.SQLITE_OK)
        with pytest.raises((MarketDataError, sqlite3.DatabaseError)):
            initialize_shard_index(db)
        db.set_authorizer(None)
        assert db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall() == before
        assert db.execute("SELECT * FROM parquet_increments ORDER BY version").fetchall() == refs
        assert db.execute("SELECT * FROM business").fetchall() == [(7, "keep me")]


def test_empty_legacy_index_does_not_replace_populated_unified_index():
    with sqlite3.connect(":memory:") as db:
        refs = _legacy_index(db)
        db.execute(SHARD_TABLE_SQL)
        db.execute("INSERT INTO parquet_shards SELECT * FROM parquet_increments")
        db.execute("DELETE FROM parquet_increments")
        initialize_shard_index(db)
        assert _tables(db) == {"parquet_shards", "business"}
        assert db.execute("SELECT * FROM parquet_shards ORDER BY version").fetchall() == refs


def test_repository_upgrade_keeps_manifest_files_and_historical_reads(tmp_path):
    root = tmp_path / "data"
    store = Store(root)
    store.add_watch("510300.SH", "ETF")
    repository = MarketDataRepository(root)
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-12-31", "2026-01-05"]),
        "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
        "volume": 100.0, "turnover": 100.0, "source_verified": True,
    })
    first = repository.commit_bars("510300.SH", "1day", frame, source="fixture")
    second = repository.commit_bars(
        "510300.SH", "1day", frame.iloc[1:], source="fixture", import_mode="snapshot",
    )
    identity = (repository.package_info()["package_id"], repository.generation)
    # Recreate the formerly deployed SQLite schema without changing the package.
    with repository.connection() as db:
        refs = [tuple(row) for row in db.execute("SELECT * FROM parquet_shards ORDER BY version,period_key")]
        db.execute("ALTER TABLE parquet_shards RENAME TO parquet_increments")
        db.execute("CREATE TABLE parquet_files(path TEXT)")
    package_files = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in repository.package_dir.rglob("*")
        if path.is_file() and (path.suffix in {".json", ".parquet"})
    }
    watches = store.watches()

    upgraded = MarketDataRepository(root)
    assert (upgraded.package_info()["package_id"], upgraded.generation) == identity
    with upgraded.connection() as db:
        assert not {"parquet_files", "parquet_increments"} & _tables(db)
        assert [tuple(row) for row in db.execute(
            "SELECT * FROM parquet_shards ORDER BY version,period_key"
        )] == refs
    assert store.watches() == watches
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in package_files} == package_files
    assert len(upgraded.bars("510300.SH", "1day", version=first["version"])) == 2
    assert len(upgraded.bars("510300.SH", "1day", version=second["version"])) == 1
    duplicate = upgraded.commit_bars("510300.SH", "1day", frame.iloc[1:], source="fixture")
    assert duplicate["version"] == second["version"]
    assert duplicate["shard_count"] == 0
    assert upgraded.dataset("510300.SH", "1day")["shard_count"] == 1


@pytest.mark.parametrize("reader", ["read_frame", "active_rows", "last_bars"])
def test_shard_readers_do_not_load_compliance_or_unrelated_history(tmp_path, monkeypatch, reader):
    root = tmp_path / "data"
    Store(root)
    repository = MarketDataRepository(root)
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-05 09:31", "2026-02-02 09:31"]),
        "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
        "volume": 100.0, "turnover": 100.0, "source_verified": True,
    })
    repository.commit_bars("510300.SH", "1min", frame, source="fixture")
    monkeypatch.setattr(repository, "dataset", lambda *_args, **_kwargs:
                        pytest.fail("A shard read must not load all-history compliance"))
    opened = []
    original = repository._read_paths

    def capture(paths, **kwargs):
        opened.extend(paths)
        return original(paths, **kwargs)

    monkeypatch.setattr(repository, "_read_paths", capture)
    if reader == "last_bars":
        actual = repository.last_bars("510300.SH", "1min", limit=1)
        month = "month=2026-02"
    else:
        actual = getattr(repository, reader)(
            "510300.SH", "1min", start="2026-01-01", end="2026-01-31",
        )
        month = "month=2026-01"
    assert len(actual) == 1
    assert len(opened) == 1 and month in opened[0]
