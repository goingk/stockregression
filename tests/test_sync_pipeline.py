from __future__ import annotations

import time
import threading

import pandas as pd
import pytest
from concurrent.futures import ThreadPoolExecutor

from stockregression.market_data import MarketDataService
from stockregression.market_data import sync as sync_module
from stockregression.market_data.coverage_index import CoverageArchive
from stockregression.market_data.sync import MarketDataSyncService
from stockregression.settings import Settings
from stockregression.storage import Store


SYMBOLS = ["000001.SZ", "000002.SZ"]
RUN_KEY = "parallel-pipeline"


def _prepared(symbol):
    return {
        "symbol": symbol,
        "frame": pd.DataFrame(),
        "source": "fixture",
        "metadata": {},
        "request_start": pd.Timestamp("2026-01-05"),
        "request_end": pd.Timestamp("2026-01-06"),
        "coverage_start": pd.Timestamp("2026-01-05"),
        "coverage_end": pd.Timestamp("2026-01-06"),
    }


def test_next_download_starts_while_previous_validation_is_running(tmp_path, monkeypatch):
    configured = Settings(
        data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"),
        performance={"cpu_budget_mode": "custom", "cpu_budget": 2},
        market_data={"daily_batch_size": 1},
    )
    service = MarketDataService(
        Store(configured.data_dir, settings=configured),
        package_dir=configured.package_dir, settings=configured,
    )
    sync = MarketDataSyncService(service)
    events = []
    publication_batches = []
    first_validation_started = threading.Event()
    second_download_started = threading.Event()
    monkeypatch.setattr(sync, "_worker_count", lambda _: 2)

    def prepare(symbol, *_args):
        if symbol == SYMBOLS[1]:
            assert first_validation_started.wait(2)
            second_download_started.set()
        events.append(("download", symbol))
        return _prepared(symbol)

    def commit(symbol, _timeframe, _frame, **_kwargs):
        return {
            "symbol": symbol,
            "version": 2,
            "_deferred_validation": {
                "symbol": symbol, "timeframe": "1day", "adjustment": "none",
                "start": "2026-01-05", "end": "2026-01-06",
                "expected_data_version": 2, "had_archive": False,
                "merge_seconds": 0.0, "started_at": time.perf_counter(),
                "result": {}, "report": {}, "pending": [], "metadata": {},
                "changed_dates": [], "import_mode": "merge", "fingerprint": None,
            },
        }

    def validate(_root, _package, symbol, *_args):
        events.append(("validation_started", symbol))
        if symbol == SYMBOLS[0]:
            first_validation_started.set()
            assert second_download_started.wait(2)
        else:
            time.sleep(0.01)
        events.append(("validation_finished", symbol))
        return {
            "symbol": symbol, "timeframe": "1day", "checked_days": 0,
            "reused_days": 0, "calendar_available": False, "archive_hit": False,
            "_coverage_rows": [], "_validation_rows": [],
            "_worker_validation_seconds": 0.15,
        }

    def publish(_archive, chunks):
        publication_batches.append([len(chunk["rows"]) for chunk in chunks])
        return [
            {"data_version": chunk["expected_data_version"], "revision": 1,
             "verified_through": "2026-01-06", "_stale_dates": [],
             "_accepted_dates": []}
            for chunk in chunks
        ]

    def finalize(_self, context, _ledger, publication, **_kwargs):
        return {
            "symbol": context["symbol"], "version": 2, "last_time": "2026-01-06",
            "quality_status": "complete", "quality_revision": publication["revision"],
        }

    monkeypatch.setattr(sync, "_prepare_daily", prepare)
    monkeypatch.setattr(service, "_commit_frame", commit)
    monkeypatch.setattr(sync_module, "_validate_daily_worker", validate)
    monkeypatch.setattr(CoverageArchive, "commit_rows_many", publish)
    monkeypatch.setattr(MarketDataService, "_finalize_deferred_commit", finalize)

    result = sync._sync_daily_batches(
        SYMBOLS, "history", start="2026-01-05", end="2026-01-06",
        resume=False, retry_failed=False, workers=2,
        progress=lambda *_: None, cancelled=lambda: False, run_key=RUN_KEY,
    )

    assert result["status"] == "completed", result
    assert len(result["symbols"]) == 2
    assert publication_batches == [[0, 0]]
    first_validation = next(
        index for index, event in enumerate(events)
        if event == ("validation_started", SYMBOLS[0])
    )
    second_download = next(
        index for index, event in enumerate(events)
        if event == ("download", SYMBOLS[1])
    )
    first_validation_end = next(
        index for index, event in enumerate(events)
        if event == ("validation_finished", SYMBOLS[0])
    )
    assert first_validation < second_download < first_validation_end


def test_parallel_local_validation_publishes_all_heads_once(tmp_path, monkeypatch):
    from test_sync_resume_integration import run_key, seed

    first_symbol, second_symbol = SYMBOLS
    first_service, days, _ = seed(tmp_path, symbol=first_symbol)
    service, _, _ = seed(tmp_path, symbol=second_symbol)
    cutoff = str(days[-1].date())
    key = run_key(service, cutoff)
    for symbol in SYMBOLS:
        service.store.sync_item(key, symbol, status="completed")
    sync = MarketDataSyncService(service, cutoff_date=cutoff)
    monkeypatch.setattr(sync, "_worker_count", lambda _: 4)
    monkeypatch.setattr(sync, "_fetch", lambda *_: pytest.fail("local ledger must not fetch"))
    monkeypatch.setattr(sync_module, "ProcessPoolExecutor", _ThreadedProcessPool)
    calls = []
    original = CoverageArchive.commit_rows_many

    def counted(archive, chunks):
        calls.append(len(chunks))
        return original(archive, chunks)

    monkeypatch.setattr(CoverageArchive, "commit_rows_many", counted)
    before = service.repository.package.manifest()["package_version"]
    result = sync.sync(SYMBOLS, "1day", "incremental", resume=True)

    assert result["status"] == "completed", result
    assert len(result["symbols"]) == 2
    assert calls == [2]
    assert service.repository.package.manifest()["package_version"] == before + 1
    archive = CoverageArchive(service.repository)
    assert all(archive.load_head(symbol, "1day") is not None for symbol in SYMBOLS)
    assert all(len(archive.rows(symbol, "1day")) == len(days) for symbol in SYMBOLS)


class _ThreadedProcessPool(ThreadPoolExecutor):
    def __init__(self, max_workers, **_kwargs):
        super().__init__(max_workers=max_workers)
