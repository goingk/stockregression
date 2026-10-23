from __future__ import annotations

import json
import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression import api, sqlite_catalog
from stockregression.market_data import package_layout
from stockregression.market_data.models import MarketDataError
from stockregression.market_data.package_layout import PackageLayout
from stockregression.market_data.reader import normalize_frame
from stockregression.market_data.repository import MarketDataRepository
from stockregression.source_health import SourceFailure


def _source_failure():
    raise SourceFailure("baostock", "etf_daily", "empty", "未返回行情")


def _bars():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-10-22"]),
        "open": [1.0], "high": [1.2], "low": [0.9], "close": [1.1],
        "volume": [100.0], "turnover": [110.0],
    })


def test_source_failure_crosses_spawn_pool_without_breaking_other_symbols():
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        with pytest.raises(SourceFailure) as raised:
            pool.submit(_source_failure).result(timeout=30)
        assert (raised.value.source, raised.value.capability, raised.value.category) == (
            "baostock", "etf_daily", "empty",
        )
        assert str(raised.value) == "未返回行情"
        assert pool.submit(abs, -3).result(timeout=10) == 3


def test_recovery_waits_for_inflight_publication(tmp_path, monkeypatch):
    repository = MarketDataRepository(tmp_path)
    writing, release, opening, opened = Event(), Event(), Event(), Event()
    original = repository._write_period_shard

    def pause(*args, **kwargs):
        writing.set()
        assert release.wait(15)
        return original(*args, **kwargs)

    monkeypatch.setattr(repository, "_write_period_shard", pause)

    def reopen():
        opening.set()
        result = MarketDataRepository(tmp_path)
        opened.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(repository.commit_bars, "562820.SH", "1day", _bars())
        try:
            assert writing.wait(10)
            reader = pool.submit(reopen)
            assert opening.wait(5)
            assert not opened.wait(0.3), "recovery must wait, not delete a live transaction"
            pending = repository.package.pending_transactions()
            assert len(pending) == 1
            assert json.loads(pending[0].read_text())['kind'] == 'dataset'
        finally:
            release.set()
        assert writer.result(timeout=15)["version"] == 1
        recovered = reader.result(timeout=15)
    assert recovered.package.pending_transactions() == []
    assert len(recovered.active_rows("562820.SH", "1day")) == 1


def test_missing_transaction_is_never_resurrected(tmp_path):
    package = PackageLayout(tmp_path)
    _, path = package.begin_transaction({"kind": "dataset", "symbol": "562820.SH"})
    package.finish_transaction(path)
    with pytest.raises(FileNotFoundError):
        package.update_transaction(path, {"status": "needs_rebuild"})
    assert not path.exists()


def test_damaged_transaction_is_quarantined_only_after_published_shards_validate(tmp_path):
    repository = MarketDataRepository(tmp_path)
    repository.commit_bars("562820.SH", "1day", _bars())
    later_metadata = {"validation": {"pending_validation": []}, "proof": "checked after publication"}
    with repository._connect() as db:
        db.execute(
            "UPDATE parquet_datasets SET status='ready',reason=NULL,metadata=?,updated_at=?",
            (json.dumps(later_metadata), "2026-10-23T09:00:00+00:00"),
        )
    path = repository.package.transactions_dir / "damaged.json"
    original = '{"id":"damaged","status":"needs_rebuild"}'
    path.write_text(original, encoding="utf-8")
    recovered = MarketDataRepository(tmp_path)
    retained = recovered.package.control / "quarantine" / "transactions" / path.name
    assert retained.read_text(encoding="utf-8") == original
    assert recovered.package.pending_transactions() == []
    assert recovered.package_info()["index_status"] == "ready"
    assert len(recovered.active_rows("562820.SH", "1day")) == 1
    with recovered._connect() as db:
        row = db.execute("SELECT status,metadata,updated_at FROM parquet_datasets").fetchone()
    assert row["status"] == "ready"
    assert json.loads(row["metadata"]) == later_metadata
    assert row["updated_at"] == "2026-10-23T09:00:00+00:00"


def test_failed_validation_preserves_damaged_transaction(tmp_path):
    repository = MarketDataRepository(tmp_path)
    repository.commit_bars("562820.SH", "1day", _bars())
    path = repository.package.transactions_dir / "damaged.json"
    path.write_text('{"id":"damaged","status":"needs_rebuild"}', encoding="utf-8")
    next(repository.package_dir.rglob("*.parquet")).unlink()
    with pytest.raises(MarketDataError, match="分片文件缺失"):
        MarketDataRepository(tmp_path)
    assert path.is_file()


def test_atomic_manifest_replace_retries_windows_sharing_error(tmp_path, monkeypatch):
    destination = tmp_path / "manifest.json"
    destination.write_text('{"version":1}')
    replace = package_layout.os.replace
    calls = []

    def transient(source, target):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("sharing violation")
        replace(source, target)

    monkeypatch.setattr(package_layout.os, "replace", transient)
    monkeypatch.setattr(package_layout.time, "sleep", lambda _: None)
    package_layout._atomic_json(destination, {"version": 2})
    assert json.loads(destination.read_text()) == {"version": 2}
    assert len(calls) == 3
    assert not list(tmp_path.glob("*.writing"))


def test_permanent_manifest_replace_failure_preserves_published_file(tmp_path, monkeypatch):
    destination = tmp_path / "manifest.json"
    destination.write_text('{"version":1}')
    replace = Mock(side_effect=PermissionError("sharing violation"))
    monkeypatch.setattr(package_layout.os, "replace", replace)
    monkeypatch.setattr(package_layout.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        package_layout._atomic_json(destination, {"version": 2})
    assert json.loads(destination.read_text()) == {"version": 1}
    assert replace.call_count == 5
    assert not list(tmp_path.glob("*.writing"))


def test_catalog_setup_closes_failed_connection_and_retries_without_replaying_transaction(tmp_path, monkeypatch):
    error = sqlite3.OperationalError("disk I/O error")
    error.sqlite_errorcode = sqlite3.SQLITE_IOERR
    failed = Mock()
    failed.execute.side_effect = error
    real_connect = sqlite3.connect
    good = real_connect(tmp_path / "catalog.sqlite")
    connect = Mock(side_effect=[failed, good])
    monkeypatch.setattr(sqlite_catalog.sqlite3, "connect", connect)
    monkeypatch.setattr(sqlite_catalog.time, "sleep", lambda _: None)
    with sqlite_catalog.catalog_connection(tmp_path / "catalog.sqlite", 1) as db:
        db.execute("CREATE TABLE evidence(value INTEGER)")
        db.execute("INSERT INTO evidence VALUES(1)")
    failed.close.assert_called_once()
    assert connect.call_count == 2
    with real_connect(tmp_path / "catalog.sqlite") as db:
        assert db.execute("SELECT value FROM evidence").fetchall() == [(1,)]


def test_catalog_does_not_retry_a_failed_write_transaction(tmp_path, monkeypatch):
    real_connect = sqlite3.connect
    connect = Mock(side_effect=real_connect)
    monkeypatch.setattr(sqlite_catalog.sqlite3, "connect", connect)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        with sqlite_catalog.catalog_connection(tmp_path / "catalog.sqlite", 1) as db:
            db.execute("CREATE TABLE evidence(value INTEGER)")
            db.execute("INSERT INTO evidence VALUES(1)")
            db.execute("INSERT INTO missing_table VALUES(2)")
    assert connect.call_count == 1
    with real_connect(tmp_path / "catalog.sqlite") as db:
        assert db.execute("SELECT value FROM evidence").fetchall() == []


@pytest.mark.parametrize("name", [None, "trade_date", "bar_clock"])
def test_parquet_named_datetime_index_imports_as_timestamp(tmp_path, name):
    frame = _bars().set_index("timestamp")
    frame.index.name = name
    path = tmp_path / "562820.SH.parquet"
    frame.to_parquet(path)
    normalized, _ = normalize_frame(pd.read_parquet(path), "1day")
    assert normalized.timestamp.tolist() == [pd.Timestamp("2026-10-22")]


def test_package_and_chart_errors_are_actionable_not_empty_success(tmp_path, monkeypatch):
    app = api.create_app(str(tmp_path), start_jobs=False)

    def unavailable(*args, **kwargs):
        raise MarketDataError("事务记录不可读：damaged.json")

    monkeypatch.setattr(api, "MarketDataService", unavailable)
    with TestClient(app) as client:
        for endpoint, code in (
            ("/api/v1/data/package", "PACKAGE_NOT_READY"),
            ("/api/v1/market/bars/562820.SH?frequency=1d", "MARKET_DATA_UNAVAILABLE"),
        ):
            response = client.get(endpoint)
            assert response.status_code == 409
            assert response.json()["detail"] == {"code": code, "message": "事务记录不可读：damaged.json"}
