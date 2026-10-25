import pandas as pd
import time
from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.market_data import MarketDataService
from stockregression.storage import Store
from stockregression.models import OptimizationConfig


def _landscape(store):
    rows = []
    for a in range(6):
        for b in range(6):
            for buy in (100, 200):
                for sell in (100, 200):
                    index = len(rows) + 1
                    rows.append({"candidate_id": index,
                        "params": {"buy_step": .01 + a * .001, "sell_step": .01 + b * .001,
                                   "buy_shares": buy, "sell_shares": sell},
                        "evaluation_status": "completed", "training_excess_return": index / 10000,
                        "training_return": index / 5000, "j_score": (145 - index) / 10000,
                        "score_complete": True, "trade_count": 30, "buy_count": 15, "sell_count": 15,
                        "max_principal_drawdown": .02})
    result = {"protocol_version": 5, "result_schema_version": 2,
        "config": {"protocol_version": 5, "base": {"parameter_links": {"spacing": False, "shares": False}},
                   "strategy": {"strategy_version": 5, "recommendation_count": 2,
                                "granularity": {"buy_step": .001, "sell_step": .001, "buy_shares": 100, "sell_shares": 100}},
                   "ranges": {"buy_step": {"low": .01, "high": .015}, "sell_step": {"low": .01, "high": .015},
                              "buy_shares": {"low": 100, "high": 200}, "sell_shares": {"low": 100, "high": 200}}},
        "fixed": {"current": {"as_of": "2024-01-02", "recommendation_regions": []}},
        "evidence": {"search_candidate_archives": {"current": rows}}}
    return store.save_result("optimization", result, "600000.SH", "space"), rows


def test_space_queries_complete_slice_before_sampling_and_preserves_global_scale(tmp_path, monkeypatch):
    store = Store(tmp_path)
    result_id, rows = _landscape(store)
    def forbidden(*args, **kwargs):
        raise AssertionError("landscape must only read candidate archives")
    monkeypatch.setattr("stockregression.data.DataService.load_bundle", forbidden)
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        path = f"/api/v1/results/{result_id}/landscape?view=space&fixed_axis=sell_shares&fixed_tick=2&x=buy_step&y=sell_step&z=buy_shares&limit=7"
        first = client.get(path)
        assert first.status_code == 200, first.text
        payload = first.json()
        assert payload["total"] == 72
        assert payload["returned_count"] == 7
        assert payload["sampled"] is True
        assert all(p["params"]["sell_shares"] == 200 for p in payload["points"])
        assert 144 in [p["candidate_id"] for p in payload["points"]]
        assert payload["points"] == client.get(path).json()["points"]
        other = client.get(path.replace("fixed_tick=2", "fixed_tick=1")).json()
        assert payload["metric_bounds"] == other["metric_bounds"]
        assert payload["axis_bounds"] == other["axis_bounds"]
        assert payload["metric_bounds"]["excess"]["max"] == rows[-1]["training_excess_return"]
        assert payload["parameter_space"]["active_axes"] == ["buy_step", "sell_step", "buy_shares", "sell_shares"]


def test_two_axis_fallback_is_an_exact_plane_and_bad_ticks_are_rejected(tmp_path):
    result_id, _ = _landscape(Store(tmp_path))
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        path = f"/api/v1/results/{result_id}/landscape?view=space"
        payload = client.get(path + "&x=buy_step&y=sell_step&limit=10000").json()
        assert payload["total"] == 36
        assert all(p["params"]["buy_shares"] == 200 and p["params"]["sell_shares"] == 200 for p in payload["points"])
        assert client.get(path + "&fixed_axis=sell_shares&fixed_tick=99").status_code == 422
        assert client.get(path + "&x=buy_step&y=buy_step").status_code == 422


def test_calendar_uses_current_period_bounds_and_never_reads_minute_bodies(tmp_path, monkeypatch):
    store = Store(tmp_path)
    service = MarketDataService(store)
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
    daily = pd.DataFrame({"date": days, "open": 10., "close": 10., "high": 11., "low": 9.,
                          "volume": 1000., "turnover": 10000., "preclose": 10., "tradestatus": 1, "is_st": False})
    daily_path = tmp_path / "daily.csv"
    daily.to_csv(daily_path, index=False)
    service.import_file("159995.SZ", "1day", daily_path)
    minute = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02 09:31", "2024-01-03 09:31"]),
                           "open": 10., "close": 10., "high": 11., "low": 9., "volume": 1000., "turnover": 10000.})
    minute_path = tmp_path / "minute.csv"
    minute.to_csv(minute_path, index=False)
    service.import_file("159995.SZ", "1min", minute_path)
    service.repository.save_auxiliary("159995.SZ", "calendar", [{"date": str(day.date())} for day in days], version="calendar-test", verified=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("date controls must not read price bodies")
    monkeypatch.setattr(type(service.repository), "bars", forbidden)
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        daily_result = client.get("/api/v1/data/calendar/159995.SZ?frequency=1d")
        minute_result = client.get("/api/v1/data/calendar/159995.SZ?frequency=15m")
        assert daily_result.status_code == minute_result.status_code == 200
        assert daily_result.json()["last_date"] == "2024-01-05"
        assert minute_result.json()["last_date"] == "2024-01-03"
        assert minute_result.json()["trading_dates"] == ["2024-01-02", "2024-01-03"]
        assert client.get("/api/v1/data/calendar/159995.SZ?data_version=missing").status_code == 409
        empty = client.get("/api/v1/data/calendar/600001.SH").json()
        assert empty["last_date"] is None and not empty["calendar_available"]


def test_preview_sessions_are_ephemeral_revision_scoped_and_do_not_authorize_formal_runs(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    app.state.research_previews.callback = lambda config, cancelled: {"status": "ready", "data_version": "preview-only"}
    config = OptimizationConfig(base={"symbol": "600000.SH", "start_date": "2024-01-02", "end_date": "2024-12-31"})
    frozen = config.model_dump(mode="json")
    formal_id = app.state.store.new_job("backtest", {"config": frozen["base"]})
    with TestClient(app) as client:
        assert client.delete("/api/v1/optimizations/preview-sessions/pane?through_revision=2").status_code == 200
        stale = client.post("/api/v1/optimizations/preview-sessions", json={"session_id": "pane", "revision": 1, "config": frozen})
        assert stale.status_code == 202 and stale.json()["status"] == "cancelled"
        current = client.post("/api/v1/optimizations/preview-sessions", json={"session_id": "pane", "revision": 3, "config": frozen, "estimate_key": "shape"})
        assert current.status_code == 202
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = client.get("/api/v1/optimizations/preview-sessions/pane").json()
            if state["status"] == "completed":
                break
            time.sleep(.01)
        assert state["result"]["data_version"] == "preview-only"
        client.delete("/api/v1/optimizations/preview-sessions/pane?through_revision=2")
        assert client.get("/api/v1/optimizations/preview-sessions/pane").json()["status"] == "completed"
        assert app.state.store.job(formal_id)["status"] == "queued"
        assert len(client.get("/api/v1/jobs").json()) == 1
        # Formal submission performs its own actual data-readiness check.
        denied = client.post("/api/v1/optimizations", json=frozen)
        assert denied.status_code == 409 and denied.json()["detail"]["code"] == "DATA_NOT_READY"
        frozen["base"]["parameter_links"] = {"spacing": True, "shares": False}
        frozen["base"]["sell_step"] = .04
        assert client.post("/api/v1/optimizations/preview-sessions", json={"session_id": "pane", "revision": 4, "config": frozen}).status_code == 422


def test_legacy_context_inherits_base_links_and_round_trips_without_silent_override():
    config = OptimizationConfig.model_validate({
        "base": {"symbol": "600000.SH", "start_date": "2024-01-02",
                 "parameter_links": {"spacing": True, "shares": True}},
        "research_context": {"symbol": "600000.SH", "start_date": "2024-01-02"},
    })
    assert config.base.parameter_links.spacing and config.research_context.parameter_links.spacing
    restored = OptimizationConfig.model_validate(config.model_dump(mode="json"))
    assert restored.base.parameter_links == config.base.parameter_links
    assert restored.protocol_description()["parameter_space"]["active_axes"] == ["buy_step", "buy_shares"]
