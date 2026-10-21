from fastapi.testclient import TestClient

from stockregression.api import create_app


def test_removed_demo_and_legacy_migration_endpoints_are_absent(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"
        sync = client.post("/api/v1/data/sync", json={"symbols": ["600000.SH"], "timeframe": "1day", "mode": "incremental"})
        assert sync.status_code == 200
        assert client.get(f"/api/v1/jobs/{sync.json()['job_id']}").json()["kind"] == "market_sync"
        assert client.get("/openapi.json").status_code == 200
        assert client.post("/api/v1/data/demo", json={"days": 25}).status_code in {404, 405}
        assert client.post("/api/v1/data/migrate-legacy", json={}).status_code in {404, 405}


def test_missing_data_requests_are_rejected_without_demo_fallback(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        assert client.post("/api/v1/watchlist", json={"symbol": "920001.BJ"}).status_code == 422
        missing_data = client.post("/api/v1/backtests", json={"symbol": "600000.SH", "start_date": "2026-01-01"})
        assert missing_data.status_code == 409
        assert missing_data.json()["detail"]["code"] == "DATA_NOT_READY"
        batch = client.post("/api/v1/backtests/batch", json={"configs": [{"symbol": "600000.SH", "start_date": "2026-01-01"}]})
        assert batch.status_code == 409
        optimization = client.post("/api/v1/optimizations", json={
            "base": {"schema_version": 3, "symbol": "600000.SH", "start_date": "2026-01-01"},
        })
        assert optimization.status_code == 409
        assert optimization.json()["detail"]["code"] == "DATA_NOT_READY"
        legacy_optimization = client.post("/api/v1/optimizations", json={
            "base": {"symbol": "600000.SH", "start_date": "2026-01-01"}, "n_trials": 1,
        })
        assert legacy_optimization.status_code == 422
        old_protocol = client.post("/api/v1/optimizations", json={
            "protocol_version": 4,
            "base": {"schema_version": 3, "symbol": "600000.SH", "start_date": "2026-01-01"},
        })
        assert old_protocol.status_code == 422
        assert client.post("/api/v1/data/demo", json={"days": 20}).status_code in {404, 405}
        assert client.post("/api/v1/data/migrate-legacy", json={}).status_code in {404, 405}


def test_restart_endpoint_is_guarded_and_requires_explicit_job_confirmation(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    called = []
    with TestClient(app) as client:
        health = client.get("/api/v1/health").json()
        assert health["instance_id"]
        assert health["restart_supported"] is False
        unavailable = client.post("/api/v1/system/restart", json={"force": True})
        assert unavailable.status_code == 409
        assert unavailable.json()["detail"]["code"] == "RESTART_UNAVAILABLE"

        app.state.restart_callback = lambda active: called.append(active)
        assert client.get("/api/v1/health").json()["restart_supported"] is True
        app.state.store.new_job("backtest", {"fixture": True})
        protected = client.post("/api/v1/system/restart", json={"force": False})
        assert protected.status_code == 409
        assert protected.json()["detail"]["code"] == "RESTART_JOBS_ACTIVE"
        assert called == []

        accepted = client.post("/api/v1/system/restart", json={"force": True})
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["previous_instance_id"] == health["instance_id"]
        assert called == [1]
