from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression import source_health
from stockregression.api import create_app
from stockregression.jobs import execute_job
from stockregression.settings import MarketDataSettings, Settings
from stockregression.source_health import (
    SourceFailure,
    SourceRouter,
    fetch_auxiliary_routed,
    probe_source,
    record_success,
    source_state,
    validate_source_settings,
)
from stockregression.storage import Store


def _minute_frame(source="sina"):
    return pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-10-16 14:59"), pd.Timestamp("2026-10-16 15:00")],
        "open": [10.0, 10.1], "high": [10.1, 10.2], "low": [9.9, 10.0],
        "close": [10.1, 10.2], "volume": [100, 120], "turnover": [1010.0, 1224.0],
        "source": [source, source],
    })


def test_optional_protocol_failure_falls_back_once_and_stays_suspended(tmp_path, monkeypatch):
    store = Store(
        tmp_path / "data",
        settings=Settings(data_dir=str(tmp_path / "data"), recent_minute_source="tdx"),
    )
    for market in ("stock", "etf"):
        record_success(store, "tdx", "recent_minute", market, rows=10)
    calls = {"tdx": 0, "sina": 0}

    def tdx():
        calls["tdx"] += 1
        raise RuntimeError("unpack requires a buffer of 4 bytes")

    def sina():
        calls["sina"] += 1
        return _minute_frame()

    frame, selected, route = SourceRouter(store).call(
        "recent_minute", "stock", {"tdx": tdx, "sina": sina}
    )
    assert len(frame) == 2 and selected == "sina" and route["fallback"] is True
    assert calls == {"tdx": 1, "sina": 1}
    failed = source_state(store, "tdx", "recent_minute", "stock")
    assert failed["status"] == "failed" and failed["error_category"] == "protocol"
    assert failed["cooldown_until"]

    calls.clear()
    calls.update(tdx=0, sina=0)
    _, selected, route = SourceRouter(store).call(
        "recent_minute", "stock", {"tdx": tdx, "sina": sina}
    )
    assert selected == "sina" and calls == {"tdx": 0, "sina": 1}


def test_transient_optional_failure_retries_once_before_fallback(tmp_path, monkeypatch):
    store = Store(
        tmp_path / "data",
        settings=Settings(data_dir=str(tmp_path / "data"), recent_minute_source="tdx"),
    )
    record_success(store, "tdx", "recent_minute", "stock", rows=1)
    calls = []

    def tdx():
        calls.append("tdx")
        raise TimeoutError("request timed out")

    _, selected, _ = SourceRouter(store).call(
        "recent_minute", "stock", {"tdx": tdx, "sina": lambda: _minute_frame()}
    )
    assert selected == "sina" and calls == ["tdx", "tdx"]
    assert source_state(store, "tdx", "recent_minute", "stock")["error_category"] == "timeout"


def test_http_5xx_is_treated_as_transient():
    class Response:
        status_code = 502

    error = RuntimeError("Bad Gateway")
    error.response = Response()

    assert source_health.classify_source_error(error) == "network"


def test_recent_minute_falls_back_to_ths_after_default_failure(tmp_path):
    store = Store(tmp_path / "data")
    calls = []

    def unavailable_sina():
        calls.append("sina")
        raise ConnectionError("sina unavailable")

    def available_ths():
        calls.append("ths")
        return _minute_frame("ths")

    frame, selected, route = SourceRouter(store).call(
        "recent_minute", "stock",
        {"sina": unavailable_sina, "ths": available_ths},
        fallbacks=("ths",),
    )

    assert selected == "ths" and len(frame) == 2
    assert route["fallback"] is True and route["effective"] == "ths"
    assert "sina unavailable" in route["reason"]
    assert calls == ["sina", "sina", "ths"]
    assert source_state(store, "sina", "recent_minute", "stock")["status"] == "failed"
    assert source_state(store, "ths", "recent_minute", "stock")["status"] == "passed"


def test_recent_minute_tries_ths_after_optional_and_default_fail(tmp_path):
    store = Store(
        tmp_path / "data",
        settings=Settings(data_dir=str(tmp_path / "data"), recent_minute_source="tdx"),
    )
    record_success(store, "tdx", "recent_minute", "stock", rows=1)
    calls = []

    def unavailable(name):
        def call():
            calls.append(name)
            raise RuntimeError(name + " unavailable")
        return call

    _, selected, route = SourceRouter(store).call(
        "recent_minute", "stock",
        {
            "tdx": unavailable("tdx"), "sina": unavailable("sina"),
            "ths": lambda: _minute_frame("ths"),
        },
        fallbacks=("ths",),
    )

    assert selected == "ths" and route["fallback"] is True
    assert calls == ["tdx", "sina"]
    assert "tdx unavailable" in route["reason"] and "sina unavailable" in route["reason"]


def test_partial_primary_response_is_rejected_and_falls_back_to_ths(tmp_path):
    store = Store(tmp_path / "data")
    calls = []

    def sina():
        calls.append("sina")
        return _minute_frame("sina")

    def ths():
        calls.append("ths")
        return _minute_frame("ths")

    def require_complete(value, source):
        if source == "sina":
            raise SourceFailure(source, "recent_minute", "empty", "分钟窗口不完整")

    frame, selected, route = SourceRouter(store).call(
        "recent_minute", "stock", {"sina": sina, "ths": ths},
        fallbacks=("ths",), accept=require_complete,
    )

    assert selected == "ths" and len(frame) == 2
    assert calls == ["sina", "ths"]
    assert route["fallback"] is True and "不完整" in route["reason"]
    assert source_state(store, "sina", "recent_minute", "stock")["status"] == "unknown"


def test_one_symbol_empty_response_does_not_suspend_the_source(tmp_path, monkeypatch):
    store = Store(
        tmp_path / "data",
        settings=Settings(data_dir=str(tmp_path / "data"), recent_minute_source="tdx"),
    )
    record_success(store, "tdx", "recent_minute", "stock", rows=10)
    frame, selected, route = SourceRouter(store).call(
        "recent_minute", "stock",
        {
            "tdx": lambda: (_ for _ in ()).throw(RuntimeError("所选标的未返回数据")),
            "sina": lambda: _minute_frame(),
        },
    )
    assert selected == "sina" and route["fallback"] is True and len(frame) == 2
    assert source_state(store, "tdx", "recent_minute", "stock")["status"] == "passed"


def test_etf_actions_use_akshare_independently_of_daily_failure(tmp_path, monkeypatch):
    from stockregression import providers

    frozen = Settings(
        data_dir=str(tmp_path / "data"),
        etf_daily_source="eastmoney",
        market_data=MarketDataSettings(
            etf_daily_source="eastmoney", network_timeout_seconds=17,
        ),
    )
    store = Store(tmp_path / "data", settings=frozen)
    store.put_instrument("159611.SZ", "测试ETF", asset_type="etf", etf_type="ETF")
    daily = pd.DataFrame({
        "date": [pd.Timestamp("2026-10-16")], "open": [1.0], "high": [1.1],
        "low": [0.9], "close": [1.05], "preclose": [1.0], "volume": [120000],
        "turnover": [126000.0], "tradestatus": [1],
    })
    calendar = pd.DataFrame({"date": [pd.Timestamp("2026-10-16")]})
    actions = providers.normalize_actions(pd.DataFrame())

    monkeypatch.setattr(providers, "latest_closed_date", lambda: pd.Timestamp("2026-10-16"))
    monkeypatch.setattr(
        providers, "fetch_etf_daily",
        lambda *_args, **_kwargs: daily.copy(),
    )
    probe = probe_source(store, "eastmoney", "etf_daily", "etf")
    assert probe["status"] == "completed"
    assert probe["results"][0]["qualification"]["product_type"] == "etf"

    timeouts = []

    def daily_source(symbol, start, end, source, cancelled, network_timeout=None):
        timeouts.append(network_timeout)
        if source == "eastmoney":
            raise ConnectionError("eastmoney unavailable")
        return daily.copy()

    monkeypatch.setattr("stockregression.providers.fetch_etf_daily", daily_source)
    monkeypatch.setattr("stockregression.providers.fetch_market_calendar", lambda *args: calendar.copy())
    monkeypatch.setattr(
        "stockregression.providers.fetch_market_actions",
        lambda *args, **kwargs: (actions.copy(), {"actions_verified": True, "warnings": []}),
    )
    result, result_actions, result_calendar, metadata = fetch_auxiliary_routed(
        store, "159611.SZ", "2026-10-16", "2026-10-16"
    )
    assert len(result) == 1 and result_actions.empty and len(result_calendar) == 1
    assert metadata["daily_source"] == "baostock"
    assert metadata["actions_source"] == "akshare"
    assert timeouts == [17, 17, 17]
    assert source_state(store, "eastmoney", "etf_daily", "etf")["status"] == "failed"
    assert source_state(store, "akshare", "etf_actions", "etf")["status"] == "passed"


def test_source_router_uses_the_task_market_policy(tmp_path, monkeypatch):
    frozen = Settings(
        data_dir=str(tmp_path / "data"),
        recent_minute_source="tdx",
        market_data=MarketDataSettings(
            recent_minute_source="tdx", request_retries=2,
        ),
    )
    store = Store(tmp_path / "data", settings=frozen)
    record_success(store, "tdx", "recent_minute", "stock", rows=1)
    monkeypatch.setattr(
        source_health, "load_settings",
        lambda: Settings(recent_minute_source="sina", max_workers=1),
    )
    calls = []

    def unavailable_tdx():
        calls.append("tdx")
        raise TimeoutError("request timed out")

    _, selected, route = SourceRouter(store).call(
        "recent_minute", "stock",
        {"tdx": unavailable_tdx, "sina": lambda: _minute_frame()},
    )
    assert selected == "sina" and route["configured"] == "tdx"
    assert calls == ["tdx", "tdx", "tdx"]


def test_source_settings_do_not_require_manual_probe(tmp_path):
    store = Store(tmp_path / "data")
    validate_source_settings(store, {"recent_minute_source": "tdx"})
    record_success(store, "tdx", "recent_minute", "stock", rows=1)
    validate_source_settings(store, {"recent_minute_source": "tdx"})
    record_success(store, "tdx", "recent_minute", "etf", rows=1)
    validate_source_settings(store, {"recent_minute_source": "tdx"})


def test_legacy_hf_settings_are_ignored_and_default_sources_apply(tmp_path):
    settings = Settings.model_validate({
        "data_dir": str(tmp_path),
        "hf_revision": "legacy-stock-revision",
        "etf_hf_revision": "legacy-etf-revision",
    })
    assert settings.recent_minute_source == "sina"
    assert settings.etf_daily_source == "baostock"


def test_source_api_reads_cache_and_accepts_optional_source_without_probe(tmp_path, monkeypatch):
    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        response = client.get("/api/v1/data/sources")
        assert response.status_code == 200, response.text
        capabilities = {item["capability"]: item for item in response.json()["capabilities"]}
        assert capabilities["recent_minute"]["configured"] in {"sina", "tdx"}
        assert capabilities["etf_daily"]["configured"] in {"baostock", "eastmoney"}
        ths = next(
            option for option in capabilities["recent_minute"]["sources"]
            if option["source"] == "ths"
        )
        assert ths["role"] == "fallback" and ths["selectable"] is False
        optional = next(
            option for option in capabilities["recent_minute"]["sources"]
            if option["source"] == "tdx"
        )
        assert optional["selectable"] is True
        revision = client.get("/api/v1/configuration").json()["revision"]
        rejected = client.patch("/api/v1/configuration", json={
            "base_revision": revision,
            "patch": {"market_data": {"recent_minute_source": "tdx"}},
        })
        assert rejected.status_code == 200
        invalid = client.post("/api/v1/data/sources/probe", json={
            "source": "tdx", "capability": "etf_daily", "market": "etf",
        })
        assert invalid.status_code == 422
        accepted = client.post("/api/v1/data/sources/probe", json={
            "source": "tdx", "capability": "recent_minute", "market": "all",
        })
        assert accepted.status_code == 200, accepted.text
        job_id = accepted.json()["job_id"]
        assert client.get(f"/api/v1/jobs/{job_id}").json()["kind"] == "market_source_probe"
        monkeypatch.setattr(
            "stockregression.providers.fetch_sina",
            lambda symbol, **kwargs: _minute_frame(),
        )
        # The submitted probe names TDX; exercise the job path with a temporary
        # source substitution so this API test remains offline.
        probe_payload = app.state.store.job(job_id)["payload"]
        probe_payload["source"] = "sina"
        app.state.store.update_job(job_id, payload=probe_payload)
        assert app.state.store.claim_job(job_id, "test-worker")
        execute_job(str(tmp_path), job_id)
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"] == "completed", job.get("error")
        refreshed = client.get("/api/v1/data/sources").json()
        sina = next(
            option for item in refreshed["capabilities"] if item["capability"] == "recent_minute"
            for option in item["sources"] if option["source"] == "sina"
        )
        assert all(state["status"] == "passed" for state in sina["states"])


def test_minute_history_sync_is_rejected_with_import_guidance(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        response = client.post("/api/v1/data/sync", json={
            "symbols": ["600000.SH"], "timeframe": "1min", "mode": "history",
        })
        assert response.status_code == 422
        assert "数据包" in response.json()["detail"] and "导入" in response.json()["detail"]


def test_probe_records_market_specific_health_without_market_write(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    monkeypatch.setattr(
        "stockregression.providers.fetch_sina",
        lambda symbol, **kwargs: _minute_frame(),
    )
    result = probe_source(store, "sina", "recent_minute", "all")
    assert result["status"] == "completed" and len(result["results"]) == 2
    assert all(item["state"]["status"] == "passed" for item in result["results"])
    assert source_state(store, "sina", "recent_minute", "stock")["rows"] == 2
    assert source_state(store, "sina", "recent_minute", "etf")["rows"] == 2
    assert store.state("sync:preflight") is None
    assert source_state(store, "tdx", "recent_minute", "stock")["status"] == "unknown"
