from fastapi.testclient import TestClient

from stockregression import settings
from stockregression.api import create_app
from stockregression.performance import effective_cpu_budget
from stockregression.resources import ResourceLease
from stockregression.settings import PerformanceSettings
from stockregression.storage import Store


def test_performance_settings_merge_partial_updates(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", settings_file)

    saved = settings.save_settings({"performance": {"download_workers": 6}})

    assert saved.performance.download_workers == 6
    assert saved.performance.validation_workers == 4
    assert saved.performance.workspace_mb == 8192
    assert saved.performance.cpu_budget_mode == "auto"
    assert effective_cpu_budget(saved.performance) >= 1


def test_resource_lease_rebalances_two_running_research_jobs(tmp_path):
    store = Store(tmp_path)
    first = store.new_job("optimization", {"fixture": 1})
    store.update_job(first, status="running")
    lease = ResourceLease(store, first, PerformanceSettings(workspace_mb=1024))
    budget = effective_cpu_budget(lease.settings)

    assert lease.acquire() == budget

    second = store.new_job("optimization", {"fixture": 2})
    store.update_job(second, status="running")
    assert lease.rebalance() == max(1, budget // 2)

    lease.release()
    with store.db() as db:
        assert db.execute("SELECT COUNT(*) FROM resource_leases").fetchone()[0] == 0


def test_performance_api_reports_effective_budget_and_leases(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    root = tmp_path / "data"
    app = create_app(str(root), start_jobs=False)
    with TestClient(app) as client:
        response = client.get("/api/v1/performance")
        initial = client.get("/api/v1/settings").json()
        saved = client.put("/api/v1/settings", json={
            "base_revision": initial["config_revision"],
            "performance": {"cpu_budget_mode": "custom", "cpu_budget": 6},
        })

    assert response.status_code == 200
    payload = response.json()
    assert payload["effective_cpu_budget"] >= 1
    assert payload["logical_cpus"] >= 1
    assert payload["settings"]["download_workers"] == 4
    assert payload["active_leases"] == []
    assert saved.status_code == 200
    assert saved.json()["performance"]["cpu_budget_mode"] == "custom"
    assert saved.json()["performance"]["cpu_budget"] == 6
    assert saved.json()["performance"]["validation_workers"] == 4
