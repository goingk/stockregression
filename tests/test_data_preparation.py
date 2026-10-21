from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from stockregression.api import create_app
from stockregression.market_data import MarketDataService
from stockregression.market_data.preparation import MarketPreparationService
from stockregression.models import BacktestConfig
from stockregression.quality import minute_slots
from stockregression.jobs import execute_job
from stockregression.storage import Store


def _config():
    return BacktestConfig(
        symbol="600000.SH", start_date=pd.Timestamp("2026-01-05").date(),
        end_date=pd.Timestamp("2026-01-06").date(), frequency="1d",
    )


def _gapped_minute_fixture(tmp_path, count=40):
    days = pd.bdate_range("2026-01-05", periods=count)
    prior = pd.Timestamp("2026-01-02")
    all_days = pd.DatetimeIndex([prior]).append(days)
    minute_rows = []
    for day in all_days:
        for timestamp in minute_slots(day):
            minute_rows.append({
                "timestamp": timestamp, "open": 10.0, "high": 10.5, "low": 9.5,
                "close": 10.2, "volume": 100.0, "turnover": 1000.0,
            })
    minute = pd.DataFrame(minute_rows)
    gap_day = days[1]
    gap_slots = minute_slots(gap_day)
    minute = minute[~minute.timestamp.between(gap_slots[0], gap_slots[49])].copy()
    daily = pd.DataFrame({
        "date": all_days, "open": 10.0, "high": 10.5, "low": 9.5,
        "close": 10.2, "preclose": 10.2, "volume": 24_000.0,
        "turnover": 240_000.0, "tradestatus": 1, "is_st": 0,
    })
    daily.loc[daily.date == gap_day, ["volume", "turnover"]] = [19_000.0, 190_000.0]
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": all_days}).to_csv(calendar, index=False)
    daily_path, minute_path = tmp_path / "daily.csv", tmp_path / "minute.csv"
    daily.to_csv(daily_path, index=False)
    minute.to_csv(minute_path, index=False)
    service = MarketDataService(Store(tmp_path / "data"))
    service.import_file("600000.SH", "1day", daily_path, calendar_path=calendar)
    service.import_file("600000.SH", "1min", minute_path, calendar_path=calendar, label_convention="end")
    config = BacktestConfig(
        symbol="600000.SH", start_date=pd.Timestamp("2026-01-04").date(),
        end_date=days[-1].date(), frequency="1m",
    )
    return service, config, gap_day


def test_readiness_allows_a_bounded_minute_gap_to_degrade_to_daily(tmp_path):
    service, config, gap_day = _gapped_minute_fixture(tmp_path, count=40)
    readiness = service.readiness(config)

    assert readiness["ready"] is True
    assert [row["date"] for row in readiness["degraded_days"]] == [str(gap_day.date())]
    bundle = service.load_backtest_bundle(config)
    assert bundle.quality["daily_days"] == 1
    assert bundle.quality["minute_ratio"] >= 0.95
    assert bundle.quality["days"][1]["mode"] == "daily"


def test_readiness_rejects_a_minute_gap_above_the_fallback_budget(tmp_path):
    service, config, _ = _gapped_minute_fixture(tmp_path, count=10)
    readiness = service.readiness(config)

    assert readiness["ready"] is False
    assert any(gap["kind"] == "quality" for gap in readiness["gaps"])


def test_preparation_plan_lists_dependencies_without_network(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    plan = MarketPreparationService(service).plan(_config())

    assert plan["ready"] is False
    assert plan["package_id"]
    assert [step["kind"] for step in plan["steps"]] == ["auxiliary", "bars", "validate"]
    assert plan["steps"][1]["timeframe"] == "1day"
    assert not list((tmp_path / "data" / "imports").rglob("*.job"))


def test_preparation_api_previews_and_submits_parent_job(tmp_path):
    root = tmp_path / "data"
    app = create_app(str(root), start_jobs=False)
    body = {"config": _config().model_dump(mode="json")}
    with TestClient(app) as client:
        plan = client.post("/api/v1/data/preparation/plan", json=body)
        assert plan.status_code == 200, plan.text
        assert plan.json()["steps"]
        submitted = client.post("/api/v1/data/preparation", json=body)
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job_id"]
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["kind"] == "market_prepare"
        assert job["status"] == "queued"


def test_minute_preparation_without_local_history_requires_import(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    config = _config().model_copy(update={"frequency": "1m"})
    preparation = MarketPreparationService(service)
    plan = preparation.plan(config)
    assert any(step["kind"] == "import" for step in plan["steps"])
    with pytest.raises(Exception, match="本地数据包"):
        preparation.execute(config)


def test_inventory_api_reports_package_identity_and_server_pagination(tmp_path):
    root = tmp_path / "data"
    app = create_app(str(root), start_jobs=False)
    with TestClient(app) as client:
        package = client.get("/api/v1/data/package")
        inventory = client.get("/api/v1/data/inventory?limit=50&offset=0")
        assert package.status_code == 200, package.text
        assert package.json()["package_id"].startswith("pkg-")
        assert package.json()["readable"] is True
        assert inventory.status_code == 200, inventory.text
        assert inventory.json()["count"] == 0
        assert inventory.json()["limit"] == 50


def test_market_scope_update_does_not_refresh_the_catalog(tmp_path, monkeypatch):
    from stockregression.data import DataService
    from stockregression.market_data.sync import MarketDataSyncService
    from stockregression.market_data.repository import MarketDataRepository

    root = tmp_path / "data"
    app = create_app(str(root), start_jobs=False)
    monkeypatch.setattr(MarketDataRepository, "symbols_for_timeframe", lambda self, timeframe: ["600000.SH"])
    monkeypatch.setattr(DataService, "refresh_instruments", lambda *args, **kwargs: pytest.fail("package update must not refresh catalog"))
    calls = []

    def sync(self, symbols, timeframe, mode, **kwargs):
        calls.append((symbols, timeframe, mode))
        return {"status": "completed", "symbols": [], "failures": [], "skipped": [], "message": "fixture"}

    monkeypatch.setattr(MarketDataSyncService, "sync", sync)
    job_id = app.state.store.new_job("market_sync", {
        "symbols": [], "timeframe": "1day", "mode": "incremental", "scope": "market",
        "initialize": False, "workers": 1,
    })
    assert app.state.store.claim_job(job_id, "test-worker")
    execute_job(str(root), job_id)
    job = app.state.store.job(job_id)
    assert job["status"] == "completed", job.get("error")
    assert calls == [(["600000.SH"], "1day", "incremental")]
