from __future__ import annotations

import pytest
from typer.testing import CliRunner

from stockregression import cli, jobs
from stockregression.settings import Settings
from stockregression.storage import Store


def test_legacy_cli_defaults_to_watchlist_without_widening_to_market(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(data_dir=str(tmp_path)))
    monkeypatch.setattr(cli, "_run", lambda kind, payload: captured.update(kind=kind, payload=payload))

    result = CliRunner().invoke(cli.app, ["sync", "--mode", "daily"])

    assert result.exit_code == 0, result.output
    assert captured["kind"] == "market_sync"
    assert captured["payload"]["scope"] == "watchlist"
    assert captured["payload"]["symbols"] == []


def test_data_cli_defaults_to_watchlist_without_widening_to_market(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(data_dir=str(tmp_path)))
    monkeypatch.setattr(cli, "_run", lambda kind, payload: captured.update(kind=kind, payload=payload))

    result = CliRunner().invoke(cli.app, ["data", "sync", "--mode", "incremental"])

    assert result.exit_code == 0, result.output
    assert captured["kind"] == "market_sync"
    assert captured["payload"]["scope"] == "watchlist"
    assert captured["payload"]["symbols"] == []


def test_legacy_cli_defaults_to_selected_when_symbols_are_present(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(data_dir=str(tmp_path)))
    monkeypatch.setattr(cli, "_run", lambda kind, payload: captured.update(kind=kind, payload=payload))

    result = CliRunner().invoke(cli.app, ["sync", "--mode", "daily", "--symbol", "600000.SH"])

    assert result.exit_code == 0, result.output
    assert captured["payload"]["scope"] == "selected"
    assert captured["payload"]["symbols"] == ["600000.SH"]


def test_explicit_empty_selected_scope_is_rejected_before_job_submission(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(data_dir=str(tmp_path)))
    monkeypatch.setattr(cli, "_run", lambda *args: called.append(args))

    result = CliRunner().invoke(cli.app, ["sync", "--mode", "daily", "--scope", "selected"])

    assert result.exit_code != 0
    assert "selected" in result.output
    assert called == []


def _execute(store: Store, payload: dict) -> dict:
    job_id = store.new_job("market_sync", payload)
    assert store.claim_job(job_id, "scope-audit")
    jobs.execute_job(str(store.root), job_id)
    return store.job(job_id)


def test_worker_defaults_missing_scope_to_watchlist_and_completes_empty_scope(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"))
    store = Store(settings.data_dir, settings=settings)
    job = _execute(store, {
        "mode": "incremental", "symbols": [], "scope": None,
        "timeframe": "1day", "timeframes": ["1day", "1min"],
        "configuration_snapshot": {"settings": settings.model_dump(mode="json")},
    })

    assert job["payload"]["scope"] == "watchlist"
    assert job["status"] == "completed", job.get("error")
    report = store.result(job["result_id"])
    assert report["status"] == "completed"
    assert report["failures"] == []


def test_worker_explicit_empty_watchlist_completes_noop(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"))
    store = Store(settings.data_dir, settings=settings)
    job = _execute(store, {
        "mode": "incremental", "symbols": [], "scope": "watchlist",
        "timeframe": "1day", "timeframes": ["1day", "1min"],
        "configuration_snapshot": {"settings": settings.model_dump(mode="json")},
    })

    assert job["payload"]["scope"] == "watchlist"
    assert job["status"] == "completed", job.get("error")
    report = store.result(job["result_id"])
    assert report["status"] == "completed"
    assert report["failures"] == []


@pytest.mark.parametrize("scope,symbols,frozen", [
    (None, [], None),
    (None, [], {"1day": ["600000.SH"]}),
    ("market", ["600000.SH"], {"1day": ["600000.SH"]}),
])
def test_worker_watchlist_and_frozen_scope_do_not_include_other_local_symbols(
    tmp_path, monkeypatch, scope, symbols, frozen,
):
    settings = Settings(data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"))
    store = Store(settings.data_dir, settings=settings)
    store.put_instrument("000001.SZ", "本地但不在自选")
    store.add_watch("600000.SH", "自选")
    if frozen is not None:
        store.add_watch("000001.SZ", "新增自选")
    selected_by_timeframe = []

    class FakeSync:
        def __init__(self, service, **kwargs):
            self.service = service
            self.repository = service.repository

        def sync(self, symbols, timeframe, *args, **kwargs):
            selected_by_timeframe.append((timeframe, list(symbols)))
            return {
                "status": "completed", "timeframe": timeframe,
                "symbols": [{"symbol": symbol} for symbol in symbols],
                "failures": [], "skipped": [],
            }

    monkeypatch.setattr(jobs, "MarketDataSyncService", FakeSync)
    payload = {
        "mode": "incremental", "symbols": symbols, "scope": scope,
        "timeframe": "1day", "timeframes": ["1day"],
        "configuration_snapshot": {"settings": settings.model_dump(mode="json")},
    }
    if frozen is not None:
        payload["sync_symbols_by_timeframe"] = frozen
    job = _execute(store, payload)

    assert job["status"] == "completed", job.get("error")
    assert selected_by_timeframe == [("1day", ["600000.SH"])]
    assert job["payload"]["sync_symbols_by_timeframe"] == {"1day": ["600000.SH"]}


@pytest.mark.parametrize("frozen", [None, {}, [], {"1day": "600000.SH"}, {"1min": []}])
def test_invalid_frozen_scope_fails_without_selecting_current_watchlist(tmp_path, monkeypatch, frozen):
    settings = Settings(data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"))
    store = Store(settings.data_dir, settings=settings)
    store.add_watch("600000.SH", "当前自选")
    calls = []
    monkeypatch.setattr(jobs.MarketDataSyncService, "sync", lambda *args, **kwargs: calls.append(args))
    job = _execute(store, {
        "mode": "incremental", "symbols": [], "scope": "watchlist",
        "timeframe": "1day", "timeframes": ["1day"],
        "sync_symbols_by_timeframe": frozen,
        "configuration_snapshot": {"settings": settings.model_dump(mode="json")},
    })
    assert job["status"] == "failed"
    assert "冻结" in job["error"]
    assert calls == []


def test_empty_frozen_period_does_not_select_current_watchlist(tmp_path, monkeypatch):
    settings = Settings(data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"))
    store = Store(settings.data_dir, settings=settings)
    store.add_watch("600000.SH", "新增自选")
    calls = []
    monkeypatch.setattr(jobs.MarketDataSyncService, "sync", lambda *args, **kwargs: calls.append(args))
    job = _execute(store, {
        "mode": "incremental", "symbols": [], "scope": "watchlist",
        "timeframe": "1day", "timeframes": ["1day"],
        "sync_symbols_by_timeframe": {"1day": []},
        "configuration_snapshot": {"settings": settings.model_dump(mode="json")},
    })
    assert job["status"] == "completed", job.get("error")
    assert job["payload"]["sync_symbols_by_timeframe"] == {"1day": []}
    assert calls == []


def test_worker_rejects_explicit_empty_selected_scope(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"))
    store = Store(settings.data_dir, settings=settings)
    job = _execute(store, {
        "mode": "incremental", "symbols": [], "scope": "selected",
        "timeframe": "1day", "timeframes": ["1day"],
        "configuration_snapshot": {"settings": settings.model_dump(mode="json")},
    })

    assert job["status"] == "failed"
    assert "selected" in (job["error"] or "")


def test_market_timeframe_without_targets_is_a_completed_noop(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"))
    store = Store(settings.data_dir, settings=settings)
    job = _execute(store, {
        "mode": "incremental", "symbols": [], "scope": "market",
        "timeframe": "1day", "timeframes": ["1day", "1min"],
        "configuration_snapshot": {"settings": settings.model_dump(mode="json")},
    })

    assert job["status"] == "completed", job.get("error")
    report = store.result(job["result_id"])
    assert report["status"] == "completed"
    assert all(run["symbols"] == [] for run in report["runs"])
