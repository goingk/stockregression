from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from stockregression import api, jobs, settings as settings_module
from stockregression.market_data import MarketDataService
from stockregression.market_data.models import MarketSyncCancelled
from stockregression.market_data.sync import SOURCE_VERSION
from stockregression.settings import Settings
from stockregression.storage import Store


def configure_tmp(monkeypatch, root):
    configured = Settings(
        data_dir=str(root / "data"), package_dir=str(root / "package"),
        performance={"cpu_budget_mode": "custom", "cpu_budget": 1, "download_workers": 1},
    )
    monkeypatch.setattr(settings_module, "load_settings", lambda: configured)
    monkeypatch.setattr(jobs, "load_settings", lambda: configured)
    return configured


def test_sync_submit_freezes_cutoff_package_identity_and_probe_options(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "latest_closed_date", lambda: api.pd.Timestamp("2026-10-23"))
    app = api.create_app(str(tmp_path / "app"), start_jobs=False)
    with TestClient(app) as client:
        sync = client.post("/api/v1/data/sync", json={
            "symbols": ["600000.SH"], "timeframe": "1day", "mode": "incremental",
        })
        assert sync.status_code == 200
        payload = app.state.store.job(sync.json()["job_id"])["payload"]
        assert payload["cutoff_date"] == "2026-10-23"
        assert payload["package_id"]
        assert payload["package_generation"]

        probe = client.post("/api/v1/data/sources/probe", json={
            "source": "eastmoney", "capability": "etf_daily", "product_type": "reit",
            "sample_symbol": "508000.SH",
        })
        assert probe.status_code == 200
        probe_payload = app.state.store.job(probe.json()["job_id"])["payload"]
        assert probe_payload["product_type"] == "reit"
        assert probe_payload["sample_symbol"] == "508000.SH"


def test_legacy_market_sync_resume_restores_checkpoint_symbols_and_frozen_snapshot(tmp_path):
    app = api.create_app(str(tmp_path / "app"), start_jobs=False)
    store = app.state.store
    package_identity = api.MarketDataService(store).repository
    package_id = package_identity.package_id
    package_generation = package_identity.generation
    legacy_id = "aedd20b8d0d84f00807d76a022d3f402"
    snapshot = {"revision": "legacy-revision", "settings": {"frozen": True}}
    created_id = store.new_job("market_sync", {
        "timeframe": "1day", "timeframes": ["1day"], "mode": "incremental",
        "scope": "market", "symbols": [], "package_id": package_id,
        "package_generation": package_generation, "configuration_snapshot": snapshot,
    })
    with store.db() as db:
        db.execute("UPDATE jobs SET id=? WHERE id=?", (legacy_id, created_id))
    store.update_job(legacy_id, status="failed", message="interrupted")
    run_key = f"market-sync:{package_id}:1day:incremental:2026-10-23:{SOURCE_VERSION}"
    symbols = [f"{index:06d}.SZ" for index in range(2762)]
    with store.db() as db:
        db.executemany(
            "INSERT INTO sync_items(run_key,symbol,status,mode,updated_at) VALUES(?,?,?,?,?)",
            [(run_key, symbol, "completed", "incremental", "2026-10-23T00:00:00+00:00")
             for symbol in symbols],
        )

    with TestClient(app) as client:
        resumed = client.post(f"/api/v1/jobs/{legacy_id}/resume")
        assert resumed.status_code == 202
        new_id = resumed.json()["job_id"]
        assert new_id != legacy_id
        original = store.job(legacy_id)
        created = store.job(new_id)
        assert original["status"] == "failed"
        payload = created["payload"]
        assert len(payload["symbols"]) == 2762
        assert payload["cutoff_date"] == "2026-10-23"
        assert payload["package_id"] == package_id
        assert payload["package_generation"] == package_generation
        assert payload["configuration_snapshot"] == snapshot
        assert payload["resume"] is True


def test_resume_uses_matching_checkpoint_cutoff_and_rejects_changed_package_generation(tmp_path):
    app = api.create_app(str(tmp_path / "app"), start_jobs=False)
    store = app.state.store
    package = MarketDataService(store).repository
    payload = {
        "timeframe": "1day", "timeframes": ["1day"], "mode": "incremental",
        "scope": "market", "symbols": [], "package_id": package.package_id,
        "package_generation": package.generation,
    }
    job_id = store.new_job("market_sync", payload)
    with store.db() as db:
        db.execute("UPDATE jobs SET created_at=? WHERE id=?", ("2026-10-24T17:00:00+08:00", job_id))
    store.update_job(job_id, status="failed", message="interrupted")
    run_key = f"market-sync:{package.package_id}:1day:incremental:2026-10-23:{SOURCE_VERSION}"
    with store.db() as db:
        db.execute(
            "INSERT INTO sync_items(run_key,symbol,status,mode,updated_at) VALUES(?,?,?,?,?)",
            (run_key, "600000.SH", "completed", "incremental", "2026-10-23T00:00:00+00:00"),
        )

    with TestClient(app) as client:
        resumed = client.post(f"/api/v1/jobs/{job_id}/resume")
        assert resumed.status_code == 202
        resumed_payload = store.job(resumed.json()["job_id"])["payload"]
        assert resumed_payload["cutoff_date"] == "2026-10-23"
        assert resumed_payload["symbols"] == ["600000.SH"]

        changed = dict(payload, symbols=["600000.SH"], package_generation="older-generation")
        changed_id = store.new_job("market_sync", changed)
        store.update_job(changed_id, status="failed", message="interrupted")
        rejected = client.post(f"/api/v1/jobs/{changed_id}/resume")
        assert rejected.status_code == 409
        assert "代际" in rejected.json()["detail"]


def test_resume_rejects_unknown_cutoff_instead_of_using_current_date(tmp_path):
    app = api.create_app(str(tmp_path / "app"), start_jobs=False)
    store = app.state.store
    package = MarketDataService(store).repository
    job_id = store.new_job("market_sync", {
        "timeframe": "1day", "timeframes": ["1day"], "mode": "incremental",
        "scope": "selected", "symbols": ["600000.SH"],
        "package_id": package.package_id, "package_generation": package.generation,
    })
    with store.db() as db:
        db.execute("UPDATE jobs SET created_at=? WHERE id=?", ("not-a-timestamp", job_id))
    store.update_job(job_id, status="failed", message="interrupted")

    with TestClient(app) as client:
        response = client.post(f"/api/v1/jobs/{job_id}/resume")
    assert response.status_code == 409
    assert "截止日期" in response.json()["detail"]


def test_cancel_report_aggregates_failure_categories_from_completed_and_current_runs():
    report = jobs._market_sync_cancel_report(
        [{"timeframe": "1day", "failure_categories": {"network": 2}, "failures": []}],
        {
            "timeframe": "1min",
            "failures": [{"symbol": "600000.SH", "category": "source_failure"}],
            "failure_categories": {"source_failure": 1},
        },
        ["1day", "1min"],
    )
    assert report["failure_categories"] == {"network": 2, "source_failure": 1}


def test_job_openapi_declares_typed_progress_detail_for_job_endpoints(tmp_path):
    app = api.create_app(str(tmp_path / "app"), start_jobs=False)
    schema = app.openapi()
    schemas = schema["components"]["schemas"]
    job_schema = schemas["JobResponse"]
    detail_schema = schemas["JobProgressDetail"]
    assert "progress_detail" in job_schema["properties"]
    progress_schema = job_schema["properties"]["progress_detail"]
    assert any(item.get("$ref", "").endswith("/JobProgressDetail")
               for item in progress_schema.get("anyOf", [progress_schema]))
    assert {"stage", "symbol", "checked_days", "total_days", "counts", "resources", "cutoff_date"} <= set(detail_schema["properties"])
    assert detail_schema["additionalProperties"] is True

    endpoints = {
        ("/api/v1/jobs", "get"),
        ("/api/v1/jobs/history", "get"),
        ("/api/v1/jobs/active", "get"),
        ("/api/v1/jobs/{job_id}", "get"),
        ("/api/v1/jobs/{job_id}/cancel", "post"),
    }
    for path, method in endpoints:
        response_schema = schema["paths"][path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert response_schema


def test_old_job_response_keeps_extensions_without_inventing_optional_fields(tmp_path, monkeypatch):
    app = api.create_app(str(tmp_path / "app"), start_jobs=False)
    store = app.state.store
    job_id = store.new_job("legacy_kind", {"retained": True})
    original_job = store.job

    def legacy_job(target_id):
        value = original_job(target_id)
        value.pop("progress_detail", None)
        value["legacy_extension"] = "kept"
        return value

    monkeypatch.setattr(store, "job", legacy_job)
    with TestClient(app) as client:
        response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == job_id
    assert body["kind"] == "legacy_kind"
    assert body["payload"] == {"retained": True}
    assert body["legacy_extension"] == "kept"
    assert "progress_detail" not in body


def test_execute_job_persists_claim_progress_and_idempotent_cancel_report(tmp_path, monkeypatch):
    configured = configure_tmp(monkeypatch, tmp_path)
    root = tmp_path / "data"
    store = Store(root, settings=configured)
    market = MarketDataService(store, package_dir=configured.package_dir, settings=configured)
    job_id = store.new_job("market_sync", {
        "symbols": ["600000.SH"], "timeframes": ["1day", "1min"], "timeframe": "1day",
        "scope": "selected", "mode": "incremental", "cutoff_date": "2026-10-23",
        "package_id": market.repository.package_id,
        "package_generation": market.repository.generation,
        "configuration_snapshot": {"settings": configured.model_dump(mode="json")},
    })
    assert store.claim_job(job_id, "job-claim")
    assert store.job(job_id)["status"] == "running"
    assert store.job(job_id)["claim_token"] == "job-claim"
    calls = []
    progress_detail_writes = []
    original_update_job = Store.update_job

    def count_detail_writes(self, target_id, **values):
        if "progress_detail" in values:
            progress_detail_writes.append(target_id)
        return original_update_job(self, target_id, **values)

    monkeypatch.setattr(Store, "update_job", count_detail_writes)

    class FakeSync:
        def __init__(self, service, *, job_id=None, job_claim_token=None, detail=None, cutoff_date=None):
            self.service = service
            self.repository = service.repository
            self.job_id = job_id
            self.job_claim_token = job_claim_token
            self.detail = detail
            self.cutoff_date = cutoff_date
            if job_id:
                calls.append((job_id, job_claim_token, cutoff_date))

        def sync(self, symbols, timeframe, mode, **kwargs):
            assert symbols == ["600000.SH"]
            if timeframe == "1day":
                return {"status": "completed", "timeframe": timeframe,
                        "symbols": [{"symbol": "600000.SH", "quality_status": "complete"}],
                        "failures": [], "skipped": []}
            assert store.update_job(job_id, cancel_requested=True)
            for day in range(1250):
                self.detail({
                    "stage": "history_validation", "symbol": "600000.SH", "checked_days": day,
                    "total_days": 1250,
                    "counts": {"total": 1, "processed": day, "succeeded": 0, "failed": 0,
                               "reused": 0, "skipped": 0, "unprocessed": 1},
                    "resources": {"download_workers": 1, "baostock_sessions": 1, "write_workers": 1},
                    "cutoff_date": "2026-10-23",
                })
                kwargs["progress"](min(.99, day / 1250), f"{day}/1250 日")
            raise MarketSyncCancelled({
                "status": "cancelled", "timeframe": "1min", "symbols": [], "failures": [],
                "downloaded_uncommitted": ["600000.SH"],
                "published_quality_pending": ["600000.SH"], "unprocessed": ["000001.SZ"],
            })

    monkeypatch.setattr(jobs, "MarketDataSyncService", FakeSync)
    for replay in range(2):
        writes_before = len(progress_detail_writes)
        jobs.execute_job(str(root), job_id)
        result = store.job(job_id)
        assert result["status"] == "cancelled"
        assert result["progress"] < 1
        assert result["result_id"] == f"market-sync-cancel-{job_id}", (calls, result)
        detail = result["progress_detail"]
        assert detail["stage"] == "history_validation"
        assert detail["checked_days"] == 1249
        assert detail["counts"]["unprocessed"] == 1
        assert detail["resources"]["baostock_sessions"] == 1
        report = store.result(result["result_id"])
        assert report["status"] == "cancelled"
        assert [run["timeframe"] for run in report["runs"]] == ["1day", "1min"]
        assert report["downloaded_uncommitted"] == ["600000.SH"]
        assert report["published_quality_pending"] == ["600000.SH"]
        assert report["unprocessed"] == ["000001.SZ"]
        assert report.get("quality_status") != "complete"
        assert report["failure_categories"] == {}
        assert len(progress_detail_writes) - writes_before <= 5
        # Re-enter the same cancelled report under the same worker claim to model retry/replay.
        if replay == 0:
            assert store.update_job(job_id, status="running", claim_token="job-claim", claimed_at=datetime.now(timezone.utc).isoformat(), cancel_requested=False)

    assert calls == [(job_id, "job-claim", "2026-10-23")] * 4
    assert len(store.results(kind="market_sync")) == 1
    assert [path.name for path in (root / "results").glob("*.json")] == [f"market-sync-cancel-{job_id}.json"]
