from itertools import count
from types import SimpleNamespace

import pandas as pd
import pytest

from stockregression import jobs, settings as settings_module
from stockregression.market_data import MarketDataService
from stockregression.market_data.sync import MarketDataSyncService
from stockregression.settings import Settings
from stockregression.storage import Store

SYMBOLS = ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]
RUN_KEY = "progress-regression"


@pytest.fixture
def service(tmp_path, monkeypatch):
    configured = Settings(
        data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"),
        performance={"cpu_budget_mode": "custom", "cpu_budget": 2},
        market_data={"daily_batch_size": 2},
    )
    monkeypatch.setattr(settings_module, "load_settings", lambda: configured)
    monkeypatch.setattr(jobs, "load_settings", lambda: configured)
    return MarketDataService(Store(configured.data_dir, settings=configured),
                             package_dir=configured.package_dir, settings=configured)


def prepared(symbol):
    return {
        "symbol": symbol, "frame": pd.DataFrame(), "source": "fixture",
        "request_start": pd.Timestamp("2026-01-05"),
        "request_end": pd.Timestamp("2026-01-06"),
    }


def run_daily(sync, symbols, progress, **kwargs):
    return sync._sync_daily_batches(
        symbols, "history", start="2026-01-05", end="2026-01-06",
        resume=kwargs.get("resume", False), retry_failed=kwargs.get("retry_failed", False),
        workers=2, progress=progress, cancelled=lambda: False, run_key=RUN_KEY,
    )


def assert_progress(values):
    assert values
    assert all(0 <= value <= 1 for value in values)
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1)


def test_broken_process_pool_retries_unfinished_downloads_and_uses_threads(service, monkeypatch):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool
    from stockregression.market_data import sync as sync_module

    sync = MarketDataSyncService(service)
    downloads, commits, pools, values = [], [], [], []

    class BrokenPool:
        def __init__(self, **kwargs):
            pools.append(self)

        def submit(self, *args, **kwargs):
            future = Future()
            future.set_exception(BrokenProcessPool("worker exited"))
            return future

        def shutdown(self, **kwargs):
            pass

    def prepare(symbol, *_):
        downloads.append(symbol)
        return prepared(symbol)

    def commit(symbol, *_args, **_kwargs):
        commits.append(symbol)
        return {"symbol": symbol, "last_time": "2026-01-06"}

    monkeypatch.setattr(sync_module, "ProcessPoolExecutor", BrokenPool)
    monkeypatch.setattr(sync, "_worker_count", lambda _: 4)
    monkeypatch.setattr(sync, "_prepare_daily", prepare)
    monkeypatch.setattr(service, "_commit_frame", commit)
    result = run_daily(sync, SYMBOLS, lambda value, _: values.append(value))
    assert result["status"] == "completed"
    assert result["failures"] == []
    assert len(pools) == 1, "remaining batches must not create another broken process pool"
    assert sorted(downloads) == sorted(commits) == sorted(SYMBOLS)
    assert_progress(values)


def test_daily_download_reports_progress_before_first_commit(service, monkeypatch):
    sync = MarketDataSyncService(service)
    values, commit_progress = [], []
    monkeypatch.setattr(sync, "_prepare_daily", lambda symbol, *_: prepared(symbol))

    def commit(symbol, *_args, **_kwargs):
        # Downloads must be visible before the batch starts committing rows.
        commit_progress.append(list(values))
        return {"symbol": symbol, "last_time": "2026-01-06"}

    monkeypatch.setattr(service, "_commit_frame", commit)
    result = run_daily(sync, SYMBOLS, lambda fraction, _: values.append(fraction))
    assert result["status"] == "completed"
    assert len(result["symbols"]) == 4
    assert sorted(set(commit_progress[0])) == pytest.approx([1 / 8, 2 / 8])
    assert commit_progress[-1][-1] < 1
    assert_progress(values)


def test_healthy_process_pool_is_reused_across_daily_batches(service, monkeypatch):
    from concurrent.futures import Future
    from stockregression.market_data import sync as sync_module

    sync = MarketDataSyncService(service)
    pools = []

    class HealthyPool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2
            assert kwargs["initializer"] is sync_module._initialize_daily_worker
            pools.append(self)

        def submit(self, function, root, package, symbol, *_args):
            future = Future()
            future.set_result(prepared(symbol))
            return future

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(sync_module, "ProcessPoolExecutor", HealthyPool)
    monkeypatch.setattr(sync, "_worker_count", lambda _: 4)
    monkeypatch.setattr(service, "_commit_frame", lambda symbol, *_a, **_kw: {
        "symbol": symbol, "last_time": "2026-01-06", "quality_status": "complete",
    })
    result = run_daily(sync, SYMBOLS, lambda *_: None)
    assert result["status"] == "completed"
    assert len(result["symbols"]) == 4
    assert len(pools) == 1


@pytest.mark.parametrize("outcome", [
    "download_failure", "write_failure", "skip", "checkpoint_failure", "resume", "retry_failed",
])
def test_daily_outcomes_settle_each_symbol_once(service, monkeypatch, outcome):
    sync = MarketDataSyncService(service)
    values, downloads = [], []
    target = SYMBOLS[0]
    checkpoint_skip = outcome in {"resume", "retry_failed"}
    if outcome == "resume":
        monkeypatch.setattr(sync, "_daily_checkpoint_action", lambda symbol, state, *_: "reuse" if symbol in SYMBOLS[:2] else "download")
        monkeypatch.setattr(sync, "_local_daily_validation", lambda symbol, *_: {"symbol": symbol, "last_time": "2026-01-06", "quality_status": "complete"})
    if checkpoint_skip:
        for symbol in SYMBOLS[:2] if outcome == "resume" else SYMBOLS[2:]:
            service.store.sync_item(RUN_KEY, symbol, status="completed" if outcome == "resume" else "failed")

    def prepare(symbol, *_):
        downloads.append(symbol)
        if symbol == target:
            if outcome == "download_failure":
                raise RuntimeError("fixture download failed")
            if outcome in {"skip", "checkpoint_failure"}:
                return {"symbol": symbol, "skipped": True, "message": "already current"}
        return prepared(symbol)

    def commit(symbol, *_args, **_kwargs):
        if symbol == target and outcome == "write_failure":
            raise RuntimeError("fixture write failed")
        return {"symbol": symbol, "last_time": "2026-01-06"}

    original_sync_item = service.store.sync_item
    def sync_item(run_key, symbol, **values):
        if symbol == target and outcome == "checkpoint_failure" and values.get("status") == "completed":
            raise RuntimeError("fixture checkpoint failed")
        return original_sync_item(run_key, symbol, **values)

    monkeypatch.setattr(sync, "_prepare_daily", prepare)
    monkeypatch.setattr(service, "_commit_frame", commit)
    monkeypatch.setattr(service.store, "sync_item", sync_item)
    result = run_daily(sync, SYMBOLS, lambda fraction, _: values.append(fraction),
                       resume=outcome == "resume", retry_failed=outcome == "retry_failed")
    failed = outcome.endswith("failure")
    assert len(result["symbols"]) == (2 if checkpoint_skip else 3)
    assert len(result["failures"]) == int(failed)
    assert result["status"] == ("partial" if failed else "completed")
    if checkpoint_skip:
        assert set(downloads) == set(SYMBOLS[2:])
        assert len(result["skipped"]) == 2
        assert 0.5 in values
    assert_progress(values)


@pytest.mark.parametrize("initialize", [False, True], ids=["multiple_timeframes", "market_initialize"])
def test_sync_job_progress_is_monotonic_across_stages(service, monkeypatch, initialize):
    # A local clock exposes all rapid callbacks without sleeps or global patches.
    monkeypatch.setattr(jobs, "time", SimpleNamespace(monotonic=count(1).__next__))
    calls, observed = [], []

    def refresh(self, progress, cancelled):
        progress(0.1, "catalog start")
        self.store.put_instrument(symbol=SYMBOLS[0], name="fixture")
        progress(1.0, "catalog complete")
        return {"count": 1}

    def sync(self, symbols, timeframe, mode, **kwargs):
        calls.append((timeframe, list(symbols)))
        for fraction in (0, 0.25, 1):
            kwargs["progress"](fraction, f"{timeframe}: {fraction}")
        return {"status": "completed", "symbols": [{"symbol": symbols[0]}], "failures": [], "skipped": []}

    monkeypatch.setattr(jobs.DataService, "refresh_instruments", refresh)
    monkeypatch.setattr(MarketDataSyncService, "sync", sync)
    payload = {
        "timeframes": ["1day"] if initialize else ["1min", "1day"],
        "symbols": [] if initialize else [SYMBOLS[0]],
        "scope": "market" if initialize else "selected",
        "initialize": initialize, "mode": "history" if initialize else "repair",
    }
    store = service.store
    job_id = store.new_job("market_sync", payload)
    assert store.claim_job(job_id, "progress-test")
    jobs.execute_job(str(store.root), job_id, observer=lambda fraction, _: observed.append(fraction))
    job = store.job(job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["progress"] == 1
    assert_progress(observed)
    if initialize:
        assert calls == [("1day", [SYMBOLS[0]])]
        assert observed == pytest.approx([0.01, 0.1, 0.1, 0.325, 1])
    else:
        assert [timeframe for timeframe, _ in calls] == ["1day", "1min"]
        assert observed == pytest.approx([0, 0.125, 0.5, 0.5, 0.625, 1])
