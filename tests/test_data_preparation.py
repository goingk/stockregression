from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from stockregression.api import create_app
from stockregression.market_data import MarketDataService
from stockregression.market_data.preparation import MarketPreparationService
from stockregression.models import BacktestConfig
from stockregression.jobs import execute_job
from stockregression.storage import Store


def _config():
    return BacktestConfig(
        symbol="600000.SH", start_date=pd.Timestamp("2026-01-05").date(),
        end_date=pd.Timestamp("2026-01-06").date(), frequency="1d",
    )


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
    execute_job(str(root), job_id)
    job = app.state.store.job(job_id)
    assert job["status"] == "completed", job.get("error")
    assert calls == [(["600000.SH"], "1day", "incremental")]
