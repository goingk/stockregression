from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stockregression import settings as settings_module
from stockregression.api import create_app
from stockregression.configuration import configuration_schema
from stockregression.market_data import MarketDataService
from stockregression.storage import Store


def _provenance() -> dict:
    return {
        "engine_version": "0.1.0",
        "source_fingerprint": "src-review",
        "dependency_lock_hash": "lock-review",
        "configuration_fingerprint": "config-review",
    }


def _v5_result() -> dict:
    return {
        "protocol_version": 5,
        "result_schema_version": 2,
        "symbol": "600000.SH",
        "data_version": "v-review",
        "config": {"protocol_version": 5, "base": {"symbol": "600000.SH"}},
        "computation_provenance": {
            "reproducible": True,
            "completion_allowed": True,
            "finished": _provenance(),
        },
    }


def test_result_metadata_migration_reads_legacy_files_only_once(tmp_path):
    root = tmp_path / "data"
    store = Store(root)
    result_id = store.save_result(
        "backtest", {"schema_version": 3, "config": {}, "strategies": {}},
        "600000.SH", "普通回测",
    )
    target = root / "results" / f"{result_id}.json"
    with store.db() as db:
        db.execute("DELETE FROM state WHERE key='results_metadata_v1'")

    reads: list[str] = []
    original = Path.read_text

    def counted_read(path: Path, *args, **kwargs):
        if path == target:
            reads.append(str(path))
        return original(path, *args, **kwargs)

    with patch.object(Path, "read_text", counted_read):
        Store(root)
        after_first = len(reads)
        Store(root)
        Store(root)

    assert after_first == 1
    assert len(reads) == 1


def test_job_page_uses_one_database_connection(tmp_path):
    store = Store(tmp_path)
    for index in range(100):
        store.new_job("backtest", {"index": index})

    connections = 0
    original = store.db

    @contextmanager
    def counted_db():
        nonlocal connections
        connections += 1
        with original() as db:
            yield db

    store.db = counted_db  # type: ignore[method-assign]
    jobs = store.jobs(limit=100)

    assert len(jobs) == 100
    assert connections == 1
    assert all(job["cancel_requested"] is False for job in jobs)


def test_jobs_keep_fifo_order_when_their_clock_timestamp_is_equal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "stockregression.storage.now_iso",
        lambda: "2026-01-02T03:04:05+00:00",
    )
    store = Store(tmp_path)
    created = [store.new_job("backtest", {"index": index}) for index in range(4)]

    assert store.claim_jobs("worker", 4) == created
    stored = sorted(store.jobs(limit=4), key=lambda item: item["created_at"])
    assert [item["id"] for item in stored] == created
    assert len({item["created_at"] for item in stored}) == len(created)


def test_results_filter_kind_before_pagination(tmp_path):
    store = Store(tmp_path)
    oldest = store.save_result("backtest", {}, "600000.SH", "旧回测")
    for index in range(205):
        store.save_result("market_sync", {}, "600000.SH", f"同步 {index}")

    first_page = store.results(limit=50, kind="backtest")

    assert [item["id"] for item in first_page] == [oldest]
    assert all(item["kind"] == "backtest" for item in first_page)


def test_completion_creation_keeps_snapshot_and_resume_keeps_root_checkpoint(tmp_path, monkeypatch):
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    result_id = store.save_result("optimization", _v5_result(), "600000.SH", "补算测试")
    monkeypatch.setattr(
        "stockregression.reproducibility.computation_provenance",
        lambda *args, **kwargs: _provenance(),
    )
    monkeypatch.setattr(store, "snapshot", lambda *args, **kwargs: {"version": "v-review"})

    with TestClient(app) as client:
        payload = {"decision": "current", "decision_index": 0, "reporting_budget": 50}
        first_response = client.post(f"/api/v1/results/{result_id}/completion", json=payload)
        assert first_response.status_code == 202, first_response.text
        first = first_response.json()
        assert first["deduplicated"] is False
        first_job = first["job_id"]
        first_payload = store.job(first_job)["payload"]
        assert first_payload["checkpoint_id"] == first_job
        assert first_payload["configuration_snapshot"]
        assert first_payload["configuration_revision"]

        duplicate = client.post(f"/api/v1/results/{result_id}/completion", json=payload).json()
        assert duplicate["job_id"] == first_job
        assert duplicate["deduplicated"] is True
        assert store.job(first_job)["payload"] == first_payload

        store.update_job(first_job, status="failed")
        second_job = client.post(f"/api/v1/jobs/{first_job}/resume").json()["job_id"]
        assert store.job(second_job)["payload"]["checkpoint_id"] == first_job
        store.update_job(second_job, status="failed")
        third_job = client.post(f"/api/v1/jobs/{second_job}/resume").json()["job_id"]
        assert store.job(third_job)["payload"]["checkpoint_id"] == first_job


def test_cancelling_queued_upload_removes_its_staging_directories(tmp_path):
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/data/upload",
            data={"symbol": "600000.SH", "timeframe": "1day", "label_convention": "end"},
            files={
                "market_file": (
                    "600000.csv",
                    "date,open,high,low,close,volume\n2026-01-05,10,11,9,10,100\n",
                    "text/csv",
                ),
            },
        )
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        payload = app.state.store.job(job_id)["payload"]
        paths = [Path(path) for path in payload["upload_dirs"]]
        assert any(path.exists() for path in paths)
        staging_uploads = (
            MarketDataService(app.state.store).repository.package.staging_dir / "uploads"
        )
        staging_uploads.mkdir(parents=True, exist_ok=True)
        sentinel = staging_uploads / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")

        cancelled = client.post(f"/api/v1/jobs/{job_id}/cancel")

    assert cancelled.json()["status"] == "cancelled"
    assert not any(path.exists() for path in paths)
    assert staging_uploads.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_restart_only_scheduler_settings_are_reported_and_documented(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", tmp_path / "config" / "application.json")
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    with TestClient(app) as client:
        current = client.get("/api/v1/configuration").json()
        saved = client.patch("/api/v1/configuration", json={
            "base_revision": current["revision"],
            "patch": {
                "application": {"max_workers": 1},
                "market_data": {"scheduler_poll_seconds": 3},
            },
        })
        assert saved.status_code == 200, saved.text
        pending = saved.json()["pending_restart"]

    assert "application.max_workers" in pending
    assert "market_data.scheduler_poll_seconds" in pending
    fields = {field["path"]: field for field in configuration_schema()["fields"]}
    assert fields["application.max_workers"]["effect"] == "需重启"
    assert fields["market_data.scheduler_poll_seconds"]["effect"] == "需重启"


def test_data_bars_latest_returns_the_tail_in_ascending_order_and_before_is_exclusive(tmp_path):
    days = pd.bdate_range("2026-01-05", periods=3)
    service = MarketDataService(Store(tmp_path / "data"))
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": days}).to_csv(calendar, index=False)
    source = tmp_path / "daily.csv"
    pd.DataFrame({
        "date": days,
        "open": [10, 11, 12],
        "high": [10.5, 11.5, 12.5],
        "low": [9.5, 10.5, 11.5],
        "close": [10.2, 11.2, 12.2],
        "volume": [100, 100, 100],
        "turnover": [1000, 1000, 1000],
        "preclose": [10, 10.2, 11.2],
        "tradestatus": 1,
        "is_st": 0,
    }).to_csv(source, index=False)
    imported = service.import_file(
        "600000.SH", "1day", source, calendar_path=calendar, label_convention="end",
    )

    app = create_app(str(tmp_path / "data"), start_jobs=False)
    with TestClient(app) as client:
        tail = client.get(
            "/api/v1/data/bars/600000.SH",
            params={"frequency": "1day", "latest": "true", "limit": 2},
        )
        before = client.get(
            "/api/v1/data/bars/600000.SH",
            params={
                "frequency": "1day", "latest": "true", "limit": 5,
                "before": days[-1].date().isoformat(),
            },
        )

    assert tail.status_code == 200, tail.text
    tail_dates = [str(row["timestamp"])[:10] for row in tail.json()]
    assert tail_dates == [days[1].date().isoformat(), days[2].date().isoformat()]
    assert tail_dates == sorted(tail_dates)
    before_dates = [str(row["timestamp"])[:10] for row in before.json()]
    assert before_dates == [days[0].date().isoformat(), days[1].date().isoformat()]
