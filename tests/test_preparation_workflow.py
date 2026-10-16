from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.data import DataService
from stockregression.jobs import execute_job
from stockregression.models import BacktestConfig
from stockregression.providers import normalize_actions
from stockregression.rules import instrument_terms
from stockregression.settings import load_settings
from stockregression.storage import Store


def market(symbol="600000.SH", days=5):
    calendar = pd.bdate_range("2024-01-02", periods=days)
    minute_rows, daily_rows = [], []
    previous_close = None
    for offset, day in enumerate(calendar):
        slots = pd.DatetimeIndex([
            *pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min"),
            *pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min"),
        ])
        close = 10.0 + offset * 0.08 + np.arange(240) * 0.0002
        opening = np.r_[close[0], close[:-1]]
        high = np.maximum(opening, close) + 0.001
        low = np.minimum(opening, close) - 0.001
        volume = np.full(240, 100 + offset, dtype="int64")
        for index, timestamp in enumerate(slots):
            minute_rows.append({
                "timestamp": timestamp,
                "open": float(opening[index]),
                "high": float(high[index]),
                "low": float(low[index]),
                "close": float(close[index]),
                "volume": int(volume[index]),
                "turnover": float(volume[index] * close[index]),
                "source": "tdx",
            })
        daily_rows.append({
            "date": day,
            "open": float(opening[0]),
            "high": float(high.max()),
            "low": float(low.min()),
            "close": float(close[-1]),
            "preclose": float(previous_close if previous_close is not None else opening[0]),
            "volume": int(volume.sum()),
            "turnover": float((volume * close).sum()),
            "tradestatus": 1,
            "is_st": False,
            "limit_up": 100.0,
            "limit_down": 0.01,
        })
        previous_close = float(close[-1])
    daily = pd.DataFrame(daily_rows)
    minute = pd.DataFrame(minute_rows)
    actions = normalize_actions(pd.DataFrame())
    calendar_frame = pd.DataFrame({"date": calendar})
    metadata = instrument_terms(symbol, {
        "name": "固定测试夹具",
        "listing_date": str(calendar[0].date()),
        "actions_verified": True,
        "hf_revision": load_settings().hf_revision,
    })
    profile = {"tdx": {"source": "tdx", "verified": True, "convention": "end", "volume_multiplier": 1}}
    return calendar, daily, minute, actions, calendar_frame, metadata, profile


def config_for(calendar, symbol="600000.SH", *, frequency="1m", schema_version=2, start_offset=1):
    return BacktestConfig(
        schema_version=schema_version,
        symbol=symbol,
        start_date=calendar[start_offset].date(),
        end_date=calendar[-1].date(),
        frequency=frequency,
    )


def publish(store, symbol, daily, minute, actions, calendar_frame, metadata, profiles, *, incremental=False):
    return store.publish(symbol, minute, daily, actions, calendar_frame, metadata, profiles, incremental=incremental)


def install_mock_providers(monkeypatch, end_date, source):
    import stockregression.data as module

    calendar, daily, minute, actions, calendar_frame, metadata, _ = source
    monkeypatch.setattr(module, "latest_closed_date", lambda: pd.Timestamp(end_date))
    monkeypatch.setattr(module, "fetch_hf", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(module, "fetch_auxiliary", lambda *args, **kwargs: (
        daily.copy(), actions.copy(), calendar_frame.copy(), metadata.copy()
    ))
    monkeypatch.setattr(module, "fetch_tdx", lambda *args, **kwargs: minute.copy())
    monkeypatch.setattr(module, "fetch_sina", lambda *args, **kwargs: pytest.fail("successful TDX fixture should not request Sina"))
    monkeypatch.setattr(module, "infer_profile", lambda frame, *_args, **_kwargs: {
        "source": "tdx", "verified": True, "convention": "end", "volume_multiplier": 1,
    })


def test_empty_store_prepares_data_for_backtest_worker_and_preview(tmp_path, monkeypatch):
    source = market(days=5)
    calendar, *_ = source
    install_mock_providers(monkeypatch, calendar[-1], source)
    sync_calls = []
    original_sync = DataService.sync

    def track_sync(self, symbols, mode, *args, **kwargs):
        sync_calls.append(mode)
        return original_sync(self, symbols, mode, *args, **kwargs)

    monkeypatch.setattr(DataService, "sync", track_sync)

    worker_root = tmp_path / "worker"
    worker_store = Store(worker_root)
    config = config_for(calendar)
    job_id = worker_store.new_job("backtest", {"config": config.model_dump(mode="json"), "version": None})
    execute_job(str(worker_root), job_id)
    worker_job = worker_store.job(job_id)
    active = worker_store.snapshot(config.symbol)
    assert worker_job["status"] == "completed", worker_job["error"]
    assert worker_job["payload"]["version"] == active["version"]
    assert worker_store.result(worker_job["result_id"])["data_version"] == active["version"]
    assert sync_calls == ["history"]

    preview_root = tmp_path / "preview"
    with TestClient(create_app(str(preview_root), start_jobs=False)) as client:
        response = client.post("/api/v1/backtests/preview", json={
            "session_id": "empty-store-preview",
            "revision": 1,
            "config": config.model_dump(mode="json"),
        })
        assert response.status_code == 200
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = client.get("/api/v1/backtests/preview/empty-store-preview").json()
            if state["status"] in {"completed", "failed"}:
                break
            time.sleep(.02)
        assert state["status"] == "completed", state
        assert state["version"] == Store(preview_root).snapshot(config.symbol)["version"]
        assert sync_calls == ["history", "history"]
        changed = config.model_dump(mode="json")
        changed["buy_shares"] = 200
        again = client.post("/api/v1/backtests/preview", json={
            "session_id": "empty-store-preview",
            "revision": 2,
            "config": changed,
        })
        assert again.status_code == 200
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = client.get("/api/v1/backtests/preview/empty-store-preview").json()
            if state["revision"] == 2 and state["status"] in {"completed", "failed"}:
                break
            time.sleep(.02)
        assert state["revision"] == 2 and state["status"] == "completed", state
        assert sync_calls == ["history", "history"]
        assert client.get("/api/v1/jobs").json() == []
        assert client.get("/api/v1/results").json() == []


def test_tail_daily_gap_selects_daily_increment(tmp_path, monkeypatch):
    source = market(days=5)
    calendar, daily, _minute, actions, calendar_frame, metadata, _profiles = source
    store = Store(tmp_path)
    old_manifest = publish(
        store,
        "600000.SH",
        daily.iloc[:-1].copy(),
        pd.DataFrame(),
        actions,
        calendar_frame.iloc[:-1].copy(),
        metadata,
        {},
    )
    config = config_for(calendar, frequency="1d", schema_version=1, start_offset=0)
    calls = []

    def sync(self, symbols, mode, *args, **kwargs):
        calls.append(mode)
        active = self.store.snapshot(symbols[0])
        manifest = self.store.publish(
            symbols[0],
            pd.DataFrame(),
            daily.iloc[[-1]].copy(),
            actions,
            calendar_frame.iloc[[-1]].copy(),
            active["metadata"],
            active["profiles"],
            incremental=True,
        )
        return {"status": "completed", "symbols": [{"symbol": symbols[0], "version": manifest["version"]}], "failures": []}

    monkeypatch.setattr(DataService, "sync", sync)
    import stockregression.data as module
    monkeypatch.setattr(module, "latest_closed_date", lambda: calendar[-1])

    bundle = DataService(store).ensure_bundle(config)
    assert calls == ["daily"]
    assert bundle.daily.date.max().normalize() == calendar[-1]
    assert store.snapshot(config.symbol)["version"] != old_manifest["version"]


def test_known_minute_quality_gap_is_attempted_once_then_reused(tmp_path, monkeypatch):
    source = market(days=5)
    calendar, daily, _minute, actions, calendar_frame, metadata, _profiles = source
    store = Store(tmp_path)
    manifest = publish(store, "600000.SH", daily, pd.DataFrame(), actions, calendar_frame, metadata, {})
    config = config_for(calendar, schema_version=1, start_offset=0)
    calls = []

    def known_gap(self, symbols, mode, *args, **kwargs):
        calls.append(mode)
        return {"status": "completed", "symbols": [{"symbol": symbols[0], "version": manifest["version"]}], "failures": []}

    monkeypatch.setattr(DataService, "sync", known_gap)
    first = DataService(store).ensure_bundle(config)
    second = DataService(store).ensure_bundle(config)
    assert first.version == second.version == manifest["version"]
    assert first.quality["daily_days"] == len(calendar)
    assert any("分钟数据源补数后仍存在缺口" in warning for warning in second.warnings)
    assert calls == ["repair"]


def test_invalid_sync_candidate_does_not_replace_active_snapshot(tmp_path, monkeypatch):
    import stockregression.data as module

    source = market(days=5)
    calendar, daily, minute, actions, calendar_frame, metadata, profiles = source
    store = Store(tmp_path)
    active = publish(store, "600000.SH", daily.iloc[:-1].copy(), minute[minute.timestamp.dt.normalize() != calendar[-1]].copy(), actions, calendar_frame.iloc[:-1].copy(), metadata, profiles)
    invalid_daily = daily.iloc[[-1]].copy()
    invalid_daily.loc[invalid_daily.index[0], "close"] = 0.0
    monkeypatch.setattr(module, "latest_closed_date", lambda: calendar[-1])
    monkeypatch.setattr(module, "fetch_auxiliary", lambda *args, **kwargs: (
        invalid_daily.copy(), actions.copy(), calendar_frame.iloc[[-1]].copy(), metadata.copy()
    ))
    monkeypatch.setattr(module, "fetch_tdx", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mock TDX unavailable")))
    monkeypatch.setattr(module, "fetch_sina", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mock Sina unavailable")))
    publish_calls = []
    original_publish = store.publish

    def track_publish(*args, **kwargs):
        publish_calls.append(True)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(store, "publish", track_publish)
    service = DataService(store)
    with pytest.raises(ValueError, match="日线价格无效"):
        service._sync_one("600000.SH", "daily", lambda *_: None, lambda: False)
    assert publish_calls == []
    assert store.snapshot("600000.SH")["version"] == active["version"]


def test_concurrent_ensure_from_independent_services_downloads_once(tmp_path, monkeypatch):
    source = market(days=5)
    calendar, *_ = source
    install_mock_providers(monkeypatch, calendar[-1], source)
    sync_calls = []
    sync_lock = threading.Lock()
    started, release = threading.Event(), threading.Event()

    def sync(self, symbols, mode, progress=lambda *_: None, cancelled=lambda: False, **_kwargs):
        with sync_lock:
            sync_calls.append(mode)
        started.set()
        assert release.wait(5)
        manifest = self.store.publish(
            symbols[0], source[2].copy(), source[1].copy(), source[3].copy(), source[4].copy(), source[5].copy(), source[6].copy()
        )
        return {"status": "completed", "symbols": [{"symbol": symbols[0], "version": manifest["version"]}], "failures": []}

    monkeypatch.setattr(DataService, "sync", sync)
    root = tmp_path / "shared"
    config = config_for(calendar)
    services = [DataService(Store(root)), DataService(Store(root))]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(services[0].ensure_bundle, config)
            assert started.wait(3)
            second = pool.submit(services[1].ensure_bundle, config)
            time.sleep(.1)
            release.set()
            bundles = [first.result(timeout=10), second.result(timeout=10)]
    finally:
        release.set()
    assert sync_calls == ["history"]
    assert bundles[0].version == bundles[1].version == Store(root).snapshot(config.symbol)["version"]
