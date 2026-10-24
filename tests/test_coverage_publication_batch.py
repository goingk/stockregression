from __future__ import annotations

import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.coverage_index import CoverageArchive
from stockregression.market_data.validation import validate_incremental
from stockregression.storage import Store


SYMBOLS = ("600000.SH", "600001.SH")
DAYS = pd.bdate_range("2026-01-05", periods=5)


def _service(tmp_path) -> MarketDataService:
    service = MarketDataService(Store(tmp_path / "data"))
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": DAYS}).to_csv(calendar, index=False)
    for index, symbol in enumerate(SYMBOLS, start=1):
        path = tmp_path / f"{symbol}.csv"
        pd.DataFrame({
            "date": DAYS,
            "open": 10.0 + index,
            "high": 11.0 + index,
            "low": 9.0 + index,
            "close": 10.2 + index,
            "volume": 24_000.0,
            "turnover": 240_000.0,
            "preclose": 10.0 + index,
            "tradestatus": 1,
            "is_st": 0,
        }).to_csv(path, index=False)
        service.import_file(symbol, "1day", path, calendar_path=calendar)
    return service


def _changed_evidence(service: MarketDataService):
    evidence = {}
    for index, symbol in enumerate(SYMBOLS, start=1):
        frame = service.repository.active_rows(symbol, "1day")
        frame["close"] = frame["close"] + 0.01 * index
        commit = service.repository.commit_bars(
            symbol, "1day", frame, source="fixture",
            report={"status": "complete", "pending_validation": []},
            metadata={"asset_type": "stock"},
        )
        result = validate_incremental(
            service, symbol, "1day", mode="full",
            persist_results=False, persist_archive=False,
        )
        evidence[symbol] = {
            "rows": result["_coverage_rows"],
            "validation_rows": result["_validation_rows"],
            "expected_data_version": commit["version"],
        }
    return evidence


def test_batch_publication_advances_manifest_once_for_all_symbols(tmp_path, monkeypatch):
    service = _service(tmp_path)
    evidence = _changed_evidence(service)
    archive = CoverageArchive(service.repository)
    before = service.repository.package.manifest()
    calls = 0
    original = service.repository.package.publish_quality_archives

    def counted(entries):
        nonlocal calls
        calls += 1
        return original(entries)

    monkeypatch.setattr(service.repository.package, "publish_quality_archives", counted)
    results = archive.commit_rows_many([
        {**evidence[symbol], "changed_months": {"2026-01"}} for symbol in SYMBOLS
    ])

    after = service.repository.package.manifest()
    assert calls == 1
    assert after["package_version"] == before["package_version"] + 1
    assert all(result["_accepted_dates"] for result in results)
    assert all(not result["_stale_dates"] for result in results)
    for symbol, result in zip(SYMBOLS, results, strict=True):
        reference = archive.head_reference(symbol, "1day")
        assert reference["revision"] == result["revision"]
        assert service.repository.package.absolute(reference["path"]).is_file()
        assert len(archive.rows(symbol, "1day")) == len(DAYS)


def test_batch_reference_conflict_does_not_advance_any_symbol(tmp_path):
    service = _service(tmp_path)
    archive = CoverageArchive(service.repository)
    package = service.repository.package
    before_manifest = package.manifest_path.read_bytes()
    before_version = package.manifest()["package_version"]
    before_refs = {
        symbol: archive.head_reference(symbol, "1day") for symbol in SYMBOLS
    }
    entries = []
    for symbol in SYMBOLS:
        head = archive.load_head(symbol, "1day")
        reference = {**before_refs[symbol], "revision": int(before_refs[symbol]["revision"]) + 1}
        entries.append({
            "symbol": symbol,
            "timeframe": "1day",
            "adjustment": "none",
            "reference": reference,
            "expected_data_version": int(head["data_version"]),
            "expected_revision": int(before_refs[symbol]["revision"]),
        })
    entries[1]["expected_revision"] += 7

    with pytest.raises(ValueError, match="修订冲突"):
        package.publish_quality_archives(entries)

    assert package.manifest_path.read_bytes() == before_manifest
    assert package.manifest()["package_version"] == before_version
    assert {
        symbol: archive.head_reference(symbol, "1day") for symbol in SYMBOLS
    } == before_refs


def test_batch_manifest_survives_sqlite_mirror_failure(tmp_path, monkeypatch):
    service = _service(tmp_path)
    evidence = _changed_evidence(service)
    archive = CoverageArchive(service.repository)
    original = archive.record_archive

    def fail_mirror(*args, **kwargs):
        raise RuntimeError("injected batch mirror failure")

    monkeypatch.setattr(archive, "record_archive", fail_mirror)
    with pytest.raises(RuntimeError, match="batch mirror"):
        archive.commit_rows_many([
            {**evidence[symbol], "changed_months": {"2026-01"}}
            for symbol in SYMBOLS
        ])
    monkeypatch.setattr(archive, "record_archive", original)

    # Both archive heads are authoritative even though the SQLite transaction
    # rolled back. A fresh repository must restore both projections from them.
    reopened = MarketDataService(Store(tmp_path / "data"))
    recovered = CoverageArchive(reopened.repository)
    for symbol in SYMBOLS:
        assert recovered.head_reference(symbol, "1day") is not None
        rows = recovered.rows(symbol, "1day")
        assert len(rows) == len(DAYS)
        assert all(row.get("scanned") for row in rows)


def test_batch_commit_uses_one_sqlite_mirror_transaction(tmp_path, monkeypatch):
    service = _service(tmp_path)
    evidence = _changed_evidence(service)
    archive = CoverageArchive(service.repository)
    original_connect = service.repository._connect
    begins = 0

    class ConnectionProxy:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, *args, **kwargs):
            nonlocal begins
            if str(sql).strip().upper() == "BEGIN IMMEDIATE":
                begins += 1
            return self.connection.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    class ConnectionContext:
        def __enter__(self):
            self.context = original_connect()
            return ConnectionProxy(self.context.__enter__())

        def __exit__(self, exc_type, exc, traceback):
            return self.context.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(service.repository, "_connect", ConnectionContext)
    archive.commit_rows_many([
        {**evidence[symbol], "changed_months": {"2026-01"}} for symbol in SYMBOLS
    ])

    assert begins == 1
