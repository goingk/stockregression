from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.data import DataService
from stockregression.jobs import execute_job


def test_complete_api_workflow_and_version_lock(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"
        sync = client.post("/api/v1/data/sync", json={"symbols": ["600000.SH"], "timeframe": "1day", "mode": "incremental"})
        assert sync.status_code == 200
        assert client.get(f"/api/v1/jobs/{sync.json()['job_id']}").json()["kind"] == "market_sync"
        assert client.get("/openapi.json").status_code == 200
        job_id = client.post("/api/v1/data/demo", json={"days": 25}).json()["job_id"]
        execute_job(str(tmp_path), job_id)
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
        coverage = client.get("/api/v1/data/coverage/000001.SZ").json()
        assert coverage["is_demo"] and coverage["minute_ratio"] == 1
        config = {"symbol": "000001", "start_date": coverage["first_date"], "end_date": coverage["last_date"]}
        saved = client.post("/api/v1/configs", json={"name": "账户测试", "config": config})
        assert saved.status_code == 200
        assert client.get("/api/v1/configs").json()[0]["config"]["symbol"] == "000001.SZ"
        run_id = client.post("/api/v1/backtests", json=config).json()["job_id"]
        batch = client.post("/api/v1/backtests/batch", json={"configs": [config, config]})
        assert batch.status_code == 200 and len(batch.json()["job_ids"]) == 2
        captured_version = client.get(f"/api/v1/jobs/{run_id}").json()["payload"]["version"]
        # Updating the active snapshot after submission cannot change queued inputs.
        DataService(app.state.store).demo(days=30)
        execute_job(str(tmp_path), run_id)
        job = client.get(f"/api/v1/jobs/{run_id}").json()
        assert job["status"] == "completed", job.get("error")
        result = client.get(f"/api/v1/results/{job['result_id']}").json()
        assert result["data_version"] == captured_version
        assert set(result["strategies"]) == {"grid", "dca", "buy_hold"}
        assert len(result["strategies"]["grid"]["equity"]) == 25
        equity = client.get(f"/api/v1/results/{job['result_id']}/export?kind=equity")
        assert equity.status_code == 200 and "attachment" in equity.headers["content-disposition"]
        assert b"strategy" in equity.content
        assert len(equity.content.splitlines()) > 25 * 3
        exported_json = client.get(f"/api/v1/results/{job['result_id']}/export?kind=json")
        assert exported_json.status_code == 200
        assert exported_json.json()["data_version"] == captured_version
        exported_trades = client.get(f"/api/v1/results/{job['result_id']}/export?kind=trades")
        assert exported_trades.status_code == 200
        assert "attachment" in exported_trades.headers["content-disposition"]
        assert b"strategy" in exported_trades.content
        bars = client.get("/api/v1/data/bars/000001.SZ?limit=40").json()
        assert len(bars) <= 40 and all(row["high"] >= row["low"] for row in bars)
        locked_bars = client.get(f"/api/v1/data/bars/000001.SZ?limit=40&version={captured_version}").json()
        assert locked_bars[0]["timestamp"] > bars[0]["timestamp"]


def test_validation_cancel_and_short_optimization_api(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        assert client.post("/api/v1/watchlist", json={"symbol": "920001.BJ"}).status_code == 422
        missing_data = client.post("/api/v1/backtests", json={"symbol": "600000.SH", "start_date": "2026-01-01"})
        assert missing_data.status_code == 409
        assert missing_data.json()["detail"]["code"] == "DATA_NOT_READY"
        batch = client.post("/api/v1/backtests/batch", json={"configs": [{"symbol": "600000.SH", "start_date": "2026-01-01"}]})
        assert batch.status_code == 409
        optimization = client.post("/api/v1/optimizations", json={"base": {"symbol": "600000.SH", "start_date": "2026-01-01"}, "n_trials": 1})
        assert optimization.status_code == 409
        queued = client.post("/api/v1/data/demo", json={"days": 20}).json()["job_id"]
        client.post(f"/api/v1/jobs/{queued}/cancel")
        execute_job(str(tmp_path), queued)
        assert client.get(f"/api/v1/jobs/{queued}").json()["status"] == "cancelled"
        assert app.state.store.snapshot("000001.SZ") is None
        DataService(app.state.store).demo(days=20)
        coverage = DataService(app.state.store).coverage("000001.SZ")
        config = {"base": {"symbol": "000001.SZ", "start_date": coverage["first_date"]}, "n_trials": 1, "mode": "both", "ranges": {"initial_position": {"low": 0.5, "high": 0.5}, "min_position": {"low": 0, "high": 0}, "max_position": {"low": 1, "high": 1}}}
        opt_id = client.post("/api/v1/optimizations", json=config).json()["job_id"]
        execute_job(str(tmp_path), opt_id)
        result = client.get(f"/api/v1/jobs/{opt_id}/result").json()
        assert result["historical"]["best_result"]["strategies"]["grid"]["metrics"]
        assert result["robust"]["status"] == "unavailable"


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
