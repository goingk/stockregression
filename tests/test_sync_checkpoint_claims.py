from __future__ import annotations

import sqlite3

from stockregression.settings import Settings
from stockregression.storage import Store


def make_store(root):
    settings = Settings(data_dir=str(root), package_dir=str(root / "package"))
    return Store(root, settings=settings)


def running_job(store, job_id, token):
    created = store.new_job("market_sync", {})
    assert created != job_id
    # Give the row a stable fixture ID while keeping normal job creation's schema.
    with store.db() as db:
        db.execute("UPDATE jobs SET id=? WHERE id=?", (job_id, created))
    assert store.claim_job(job_id, token)


def test_competing_claims_and_stale_owner_cannot_finish_new_attempt(tmp_path):
    store = make_store(tmp_path / "catalog")
    running_job(store, "job-a", "claim-a")
    running_job(store, "job-b", "claim-b")

    first = store.claim_sync_item(
        "run", "600000.SH", job_id="job-a", job_claim_token="claim-a", attempt_token="attempt-a",
    )
    assert first["attempt_token"] == "attempt-a"
    assert store.claim_sync_item(
        "run", "600000.SH", job_id="job-b", job_claim_token="claim-b", attempt_token="attempt-b",
    ) is None
    assert store.sync_attempt_current(
        "run", "600000.SH", job_id="job-a", job_claim_token="claim-a", attempt_token="attempt-a",
    )

    # A terminal owner no longer protects its row, so another valid job can reclaim it.
    assert store.update_job("job-a", status="failed")
    second = store.claim_sync_item(
        "run", "600000.SH", job_id="job-b", job_claim_token="claim-b", attempt_token="attempt-b",
    )
    assert second["attempt_token"] == "attempt-b"
    assert not store.finish_sync_item(
        "run", "600000.SH", job_id="job-a", job_claim_token="claim-a", attempt_token="attempt-a",
        status="completed", daily_through="2026-01-01",
    )
    assert store.finish_sync_item(
        "run", "600000.SH", job_id="job-b", job_claim_token="claim-b", attempt_token="attempt-b",
        status="completed", daily_through="2026-01-02",
    )
    assert store.sync_item("run", "600000.SH")["daily_through"] == "2026-01-02"


def test_cancelled_job_cannot_claim_or_publish_sync_checkpoint(tmp_path):
    store = make_store(tmp_path / "catalog")
    running_job(store, "job-cancel", "claim-cancel")
    assert store.update_job("job-cancel", cancel_requested=True)
    assert store.claim_sync_item(
        "run", "000001.SZ", job_id="job-cancel", job_claim_token="claim-cancel",
    ) is None
    assert not store.sync_attempt_current(
        "run", "000001.SZ", job_id="job-cancel", job_claim_token="claim-cancel", attempt_token="missing",
    )


def test_sync_item_patch_clears_old_error_when_running_or_completed(tmp_path):
    store = make_store(tmp_path / "catalog")
    store.sync_item("run", "000001.SZ", status="failed", error="old", error_category="network")
    running = store.sync_item("run", "000001.SZ", status="running", details={"phase": "fetch"})
    assert running["error"] is None
    assert running["error_category"] is None
    assert running["details"] == {"phase": "fetch"}
    store.sync_item("run", "000001.SZ", status="failed", error="again", error_category="write")
    completed = store.sync_item("run", "000001.SZ", status="completed")
    assert completed["error"] is None
    assert completed["error_category"] is None
    assert completed["details"] == {"phase": "fetch"}


def test_legacy_jobs_migrate_and_decode_progress_detail_everywhere(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    connection = sqlite3.connect(root / "catalog.sqlite")
    connection.executescript(
        """
        CREATE TABLE jobs(
            id TEXT PRIMARY KEY,kind TEXT,status TEXT,progress REAL,message TEXT,error TEXT,
            result_id TEXT,payload TEXT,created_at TEXT,updated_at TEXT,cancel_requested INTEGER DEFAULT 0,
            pid INTEGER,claim_token TEXT,claimed_at TEXT,dedupe_key TEXT
        );
        INSERT INTO jobs(id,kind,status,progress,message,payload,created_at,updated_at)
        VALUES('legacy-job','market_sync','queued',0,'queued','{}','2026-01-01','2026-01-01');
        CREATE TABLE sync_items(
            run_key TEXT,symbol TEXT,status TEXT,mode TEXT,daily_through TEXT,minute_through TEXT,
            error TEXT,updated_at TEXT,PRIMARY KEY(run_key,symbol)
        );
        """
    )
    connection.commit()
    connection.close()

    store = make_store(root)
    assert store.update_job("legacy-job", progress_detail={"stage": "queued", "fraction": 0.2})
    expected = {"stage": "queued", "fraction": 0.2}
    assert store.job("legacy-job")["progress_detail"] == expected
    assert store.jobs()[0]["progress_detail"] == expected
    assert store.jobs_by_status("queued")[0]["progress_detail"] == expected
    assert store.job_history()["items"][0]["progress_detail"] == expected
    assert store.active_jobs()[0]["progress_detail"] == expected

    with store.db() as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(sync_items)")}
    assert {
        "last_attempt_job_id", "job_claim_token", "attempt_token", "error_category",
        "target_start", "target_end", "data_version", "quality_revision", "quality_status", "details",
    } <= columns


def test_save_result_with_same_id_replaces_file_and_catalog_row_idempotently(tmp_path):
    store = make_store(tmp_path / "catalog")
    first = store.save_result("sync_cancel", {"message": "first"}, "600000.SH", "Sync", result_id="stable-id")
    second = store.save_result("sync_cancel", {"message": "second"}, "600000.SH", "Sync", result_id="stable-id")

    assert first == second == "stable-id"
    assert store.result("stable-id") == {"message": "second", "id": "stable-id"}
    assert len(store.results(kind="sync_cancel")) == 1
    files = list((store.root / "results").glob("*.json"))
    assert [path.name for path in files] == ["stable-id.json"]
