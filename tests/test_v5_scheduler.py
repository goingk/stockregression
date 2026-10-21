"""S5 scheduler contract for the v5 research protocol.

The v5 scheduler is a claim-based queue: a worker takes a queued job with an
atomic ``claim_jobs`` call, owns it through a claim token, and must release
the token when the job reaches a terminal state.  These tests pin the storage
level guarantees the dispatcher in ``jobs.py`` relies on:

* queued jobs are claimed oldest-first and never double-claimed;
* a queued job whose cancellation was requested is cancelled, not claimed;
* a write guarded by ``require_claim`` only succeeds for the live token;
* releasing the token (terminal write) permanently blocks later guarded writes;
* ``dedupe_key`` is unique among queued/running jobs only.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from stockregression.jobs import (
    JobManager,
    _elapsed_seconds,
    _request_stop,
    _runtime_limit_seconds,
    _with_completion_revision,
    execute_job,
    timeout_stage,
)
from stockregression.resources import ResourceLease
from stockregression.settings import PerformanceSettings
from stockregression.storage import Store, now_iso

TOKEN_A = "token-a"
TOKEN_B = "token-b"


def _store(root: Path) -> Store:
    return Store(str(root))


def _fifo_order(store: Store, job_ids: list[str]) -> list[str]:
    """Sort job ids the way the scheduler must hand them out."""
    stamps = {job_id: store.job(job_id)["created_at"] for job_id in job_ids}
    return sorted(job_ids, key=lambda job_id: (stamps[job_id], job_id))


def test_claim_jobs_claims_oldest_queued_job_before_newer_batch(tmp_path):
    """The oldest queued job is claimed first, even behind 1200 newer rows."""
    store = _store(tmp_path)

    oldest = store.new_job("backtest", {"ordinal": 0})
    oldest_row = store.job(oldest)
    assert oldest_row["status"] == "queued"
    assert oldest_row["claim_token"] is None
    assert oldest_row["created_at"] <= now_iso()

    # Keep the oldest job strictly older than every later insert so the
    # (created_at, id) ordering cannot be broken by a same-microsecond tie.
    time.sleep(0.01)
    newer = [store.new_job("backtest", {"ordinal": index}) for index in range(1, 1201)]
    assert len(newer) == 1200
    assert store.job(newer[-1])["created_at"] > oldest_row["created_at"]

    claimed = store.claim_jobs(TOKEN_A, 1)

    assert claimed == [oldest]
    row = store.job(oldest)
    assert row["status"] == "running"
    assert row["claim_token"] == TOKEN_A
    assert row["claimed_at"] is not None
    assert row["message"] == "正在启动计算进程"
    assert row["error"] is None

    # The newest rows stay queued: the scheduler must not skip ahead.
    assert store.job(newer[-1])["status"] == "queued"
    assert store.job(newer[-1])["claim_token"] is None
    assert store.job(newer[-1])["message"] == "等待执行"


def test_claim_jobs_cancels_queued_cancellation_before_claiming(tmp_path):
    """A queued job flagged for cancellation is cancelled, never claimed."""
    store = _store(tmp_path)

    cancelled = store.new_job("backtest", {"tag": "cancel"})
    kept = store.new_job("backtest", {"tag": "keep"})
    assert store.update_job(cancelled, cancel_requested=True) is True

    claimed = store.claim_jobs(TOKEN_A, 5)

    assert cancelled not in claimed
    assert kept in claimed
    row = store.job(cancelled)
    assert row["status"] == "cancelled"
    assert row["message"] == "排队期间已取消"
    assert row["claim_token"] is None
    assert row["claimed_at"] is None
    assert row["cancel_requested"] is True
    assert store.job(kept)["status"] == "running"


def test_claim_jobs_respects_global_worker_capacity_and_limit(tmp_path):
    """Running jobs consume the shared worker budget; limit caps one call."""
    store = _store(tmp_path)
    queued = [store.new_job("backtest", {"i": index}) for index in range(10)]
    expected = _fifo_order(store, queued)

    first = store.claim_jobs(TOKEN_A, 3)
    assert first == expected[:3]

    # Three rows are running, so a 3-slot budget has no capacity left.
    assert store.claim_jobs(TOKEN_B, 3) == []
    # Raising the budget to 4 frees exactly one slot.
    assert store.claim_jobs(TOKEN_B, 4) == expected[3:4]
    assert store.claim_jobs(TOKEN_B, 5) == expected[4:5]
    assert store.claim_jobs(TOKEN_B, 100) == expected[5:]

    for job_id in queued:
        row = store.job(job_id)
        assert row["status"] == "running"
        assert row["claim_token"] in {TOKEN_A, TOKEN_B}


def test_claim_jobs_limit_argument_caps_claim_count(tmp_path):
    """``limit`` narrows a claim batch below the worker budget."""
    store = _store(tmp_path)
    queued = [store.new_job("backtest", {"i": index}) for index in range(5)]
    expected = _fifo_order(store, queued)

    assert store.claim_jobs(TOKEN_A, 10, limit=2) == expected[:2]
    assert store.claim_jobs(TOKEN_A, 10, limit=0) == []


def test_claim_job_claims_only_a_queued_job(tmp_path):
    """Single-job claiming rejects already-running and cancel-flagged jobs."""
    store = _store(tmp_path)
    first = store.new_job("backtest", {"i": 1})
    second = store.new_job("backtest", {"i": 2})

    assert store.claim_job(first, TOKEN_A) is True
    assert store.claim_job(first, TOKEN_B) is False
    assert store.job(first)["claim_token"] == TOKEN_A
    assert store.job(first)["status"] == "running"

    assert store.claim_job(second, TOKEN_A) is True
    assert store.job(second)["claim_token"] == TOKEN_A

    third = store.new_job("backtest", {"i": 3})
    assert store.update_job(third, cancel_requested=True) is True
    assert store.claim_job(third, TOKEN_A) is False
    assert store.job(third)["status"] == "queued"
    assert store.job(third)["claim_token"] is None


def test_update_job_rejects_stale_claim_token_without_touching_row(tmp_path):
    """A guarded write with a foreign token fails and changes nothing."""
    store = _store(tmp_path)
    job_id = store.new_job("backtest", {"tag": "token"})
    assert store.claim_jobs(TOKEN_A, 1) == [job_id]
    before = store.job(job_id)
    assert before["status"] == "running"
    assert before["claim_token"] == TOKEN_A

    assert store.update_job(
        job_id, require_claim=True, expected_claim_token="stale-token",
        status="failed", message="被过期令牌覆盖",
    ) is False

    after = store.job(job_id)
    assert after["status"] == "running"
    assert after["claim_token"] == TOKEN_A
    assert after["claimed_at"] == before["claimed_at"]
    assert after["message"] == before["message"]
    assert after["updated_at"] == before["updated_at"]
    assert after["error"] == before["error"]

    # The live token still writes through.
    assert store.update_job(
        job_id, require_claim=True, expected_claim_token=TOKEN_A,
        status="failed", message="真正的失败", error="失败原因",
    ) is True
    updated = store.job(job_id)
    assert updated["status"] == "failed"
    assert updated["message"] == "真正的失败"
    assert updated["error"] == "失败原因"
    assert updated["claim_token"] == TOKEN_A


def test_guarded_write_requires_a_running_job_and_a_live_token(tmp_path):
    """Guarded writes never touch queued, unclaimed, or token-less jobs."""
    store = _store(tmp_path)
    queued = store.new_job("backtest", {"tag": "queued"})

    # Queued jobs are not running, so the guarded predicate cannot match.
    assert store.update_job(
        queued, require_claim=True, expected_claim_token=TOKEN_A, status="running",
    ) is False
    assert store.job(queued)["status"] == "queued"

    assert store.claim_jobs(TOKEN_A, 1) == [queued]
    # ``claim_token=NULL`` never compares equal, even for a running job.
    assert store.update_job(
        queued, require_claim=True, expected_claim_token=None, status="failed",
    ) is False
    assert store.job(queued)["status"] == "running"

    # Unguarded writes stay available (the API uses them to request cancel).
    assert store.update_job(queued, cancel_requested=True) is True
    assert store.job(queued)["cancel_requested"] is True


def test_terminal_write_releases_claim_and_blocks_later_claim_writes(tmp_path):
    """Releasing the token on a terminal write locks the job permanently."""
    store = _store(tmp_path)
    job_id = store.new_job("backtest", {"tag": "terminal"})
    assert store.claim_jobs(TOKEN_A, 1) == [job_id]

    # This mirrors jobs.py: the terminal write still validates the token and
    # then releases ownership by clearing claim_token / claimed_at.
    assert store.update_job(
        job_id, require_claim=True, expected_claim_token=TOKEN_A,
        status="completed", message="已完成",
        claim_token=None, claimed_at=None,
    ) is True

    row = store.job(job_id)
    assert row["status"] == "completed"
    assert row["message"] == "已完成"
    assert row["claim_token"] is None
    assert row["claimed_at"] is None

    assert store.update_job(
        job_id, require_claim=True, expected_claim_token=TOKEN_A, status="failed",
    ) is False
    assert store.update_job(
        job_id, require_claim=True, expected_claim_token=None, status="failed",
    ) is False
    final = store.job(job_id)
    assert final["status"] == "completed"
    assert final["message"] == "已完成"
    assert final["claim_token"] is None


def test_new_job_dedupe_key_reuses_active_job(tmp_path):
    """``dedupe_key`` is reused while a job is queued or running."""
    store = _store(tmp_path)

    first = store.new_job("backtest", {"tag": "a"}, dedupe_key="v5:research:abc")
    second = store.new_job("backtest", {"tag": "b"}, dedupe_key="v5:research:abc")
    assert first == second
    assert store.job(first)["status"] == "queued"
    assert store.job(first)["dedupe_key"] == "v5:research:abc"

    assert store.claim_jobs(TOKEN_A, 1) == [first]
    # Still active while running: the same slot is reused.
    third = store.new_job("backtest", {"tag": "c"}, dedupe_key="v5:research:abc")
    assert third == first

    store.update_job(
        first, require_claim=True, expected_claim_token=TOKEN_A,
        status="completed", claim_token=None, claimed_at=None,
    )
    # The dedupe slot is released once the job leaves the active statuses.
    fourth = store.new_job("backtest", {"tag": "d"}, dedupe_key="v5:research:abc")
    assert fourth != first
    assert store.job(fourth)["status"] == "queued"


def test_active_dedupe_unique_index_rejects_duplicate_active_rows(tmp_path):
    """The database enforces dedupe uniqueness for active jobs."""
    store = _store(tmp_path)
    job_id = store.new_job("backtest", {"tag": "a"}, dedupe_key="v5:research:dup")

    with store.db() as db:
        indexes = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )}
    assert "jobs_active_dedupe" in indexes

    with pytest.raises(sqlite3.IntegrityError):
        with store.db() as db:
            db.execute(
                "INSERT INTO jobs(id,kind,status,progress,message,payload,created_at,updated_at,dedupe_key) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    "duplicate-row", "backtest", "queued", 0, "等待执行", "{}",
                    now_iso(), now_iso(), "v5:research:dup",
                ),
            )
    assert store.job("duplicate-row") is None
    assert store.job(job_id)["status"] == "queued"


def test_concurrent_claim_jobs_partition_batch_without_duplicate_claims(tmp_path):
    """Two concurrent claimers never hand out the same job twice."""
    store = _store(tmp_path)
    queued = [store.new_job("backtest", {"i": index}) for index in range(200)]
    expected = _fifo_order(store, queued)

    store_a = _store(tmp_path)
    store_b = _store(tmp_path)
    barrier = threading.Barrier(2)
    results: dict[str, list[str]] = {}
    failures: list[BaseException] = []

    def claim(worker: Store, token: str, key: str) -> None:
        try:
            barrier.wait(timeout=30)
            results[key] = worker.claim_jobs(token, 50)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            failures.append(exc)

    threads = [
        threading.Thread(target=claim, args=(store_a, TOKEN_A, "a")),
        threading.Thread(target=claim, args=(store_b, TOKEN_B, "b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for thread in threads:
        assert not thread.is_alive()
    assert not failures, failures

    claimed_a = results["a"]
    claimed_b = results["b"]
    assert set(claimed_a).isdisjoint(claimed_b)

    union = set(claimed_a) | set(claimed_b)
    # Running rows occupy the shared worker budget, so at most 50 of the 200
    # queued jobs may be in flight at any time.
    assert len(union) == 50
    assert union == set(expected[:50])

    for job_id in union:
        row = store.job(job_id)
        assert row["status"] == "running"
        assert row["claim_token"] in {TOKEN_A, TOKEN_B}
        assert row["claimed_at"] is not None
    # One claim transaction wins the race, so every claimed row carries the
    # same owner token and no job was claimed twice.
    assert len({store.job(job_id)["claim_token"] for job_id in union}) == 1
    assert sum(1 for job_id in queued if store.job(job_id)["status"] == "running") == len(union)


def test_concurrent_workers_drain_queue_exactly_once(tmp_path):
    """Concurrent workers cover the whole queue, claiming every job once."""
    store = _store(tmp_path)
    queued = [store.new_job("backtest", {"i": index}) for index in range(200)]
    queued_set = set(queued)
    created_at = {job_id: store.job(job_id)["created_at"] for job_id in queued}

    store_a = _store(tmp_path)
    store_b = _store(tmp_path)
    barrier = threading.Barrier(2)
    claims: dict[str, list[tuple[str, str]]] = {}
    failures: list[BaseException] = []

    def worker(worker_store: Store, token: str, key: str) -> None:
        events: list[tuple[str, str]] = []
        try:
            barrier.wait(timeout=30)
            while True:
                claimed = worker_store.claim_jobs(token, 50)
                if not claimed:
                    break
                for job_id in claimed:
                    row = worker_store.job(job_id)
                    assert row["status"] == "running", row
                    assert row["claim_token"] == token, row
                    events.append((job_id, token))
                    assert worker_store.update_job(
                        job_id, require_claim=True, expected_claim_token=token,
                        status="completed", message="完成",
                        claim_token=None, claimed_at=None,
                    ) is True
            claims[key] = events
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(store_a, TOKEN_A, "a")),
        threading.Thread(target=worker, args=(store_b, TOKEN_B, "b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    for thread in threads:
        assert not thread.is_alive()
    assert not failures, failures

    ids_a = [job_id for job_id, _ in claims["a"]]
    ids_b = [job_id for job_id, _ in claims["b"]]
    assert set(ids_a).isdisjoint(ids_b)

    claimed_ids = ids_a + ids_b
    assert len(claimed_ids) == len(set(claimed_ids)) == 200
    assert set(claimed_ids) == queued_set

    # Each worker takes jobs in queue order.
    for ids in (ids_a, ids_b):
        assert ids == sorted(ids, key=lambda job_id: (created_at[job_id], job_id))

    # Every job was claimed by exactly one owner and released on completion.
    owners = {job_id: token for job_id, token in claims["a"] + claims["b"]}
    assert set(owners.values()) <= {TOKEN_A, TOKEN_B}
    for job_id in queued_set:
        row = store.job(job_id)
        assert row["status"] == "completed"
        assert row["claim_token"] is None
        assert row["claimed_at"] is None


def test_jobs_by_status_cursor_paginates_without_gaps(tmp_path):
    """The recovery cursor walks a status page by page without repeats."""
    store = _store(tmp_path)
    queued = [store.new_job("backtest", {"i": index}) for index in range(5)]
    expected = _fifo_order(store, queued)

    first = store.jobs_by_status("queued", limit=2)
    assert [row["id"] for row in first] == expected[:2]
    assert first[0]["payload"] == {"i": 0}

    second = store.jobs_by_status(
        "queued", after=(first[-1]["created_at"], first[-1]["id"]), limit=2,
    )
    assert [row["id"] for row in second] == expected[2:4]

    third = store.jobs_by_status(
        "queued", after=(second[-1]["created_at"], second[-1]["id"]), limit=2,
    )
    assert [row["id"] for row in third] == expected[4:]

    tail = store.jobs_by_status(
        "queued", after=(third[-1]["created_at"], third[-1]["id"]), limit=2,
    )
    assert tail == []


def test_jobs_listing_filters_by_status_and_pages_by_offset(tmp_path):
    """``jobs`` filters by status and pages newest-first by offset."""
    store = _store(tmp_path)
    first = store.new_job("backtest", {"i": 1})
    second = store.new_job("backtest", {"i": 2})
    assert store.claim_jobs(TOKEN_A, 1) == [first]

    running = store.jobs(status="running")
    queued = store.jobs(status="queued")
    assert [row["id"] for row in running] == [first]
    assert [row["id"] for row in queued] == [second]
    assert queued[0]["payload"] == {"i": 2}

    newest_first = store.jobs(limit=2)
    assert [row["id"] for row in newest_first] == [second, first]
    assert [row["id"] for row in store.jobs(limit=1, offset=1)] == [first]


def test_execute_job_leaves_an_unclaimed_queued_job_untouched(tmp_path):
    """A worker may not start a job it has not claimed.

    An unguarded write would move the row to ``running`` outside the queue's
    ordering, cancellation and capacity rules, so the worker has to leave it
    exactly as it found it.
    """
    store = _store(tmp_path)
    job_id = store.new_job("backtest", {"config": {}})
    before = store.job(job_id)
    assert before["status"] == "queued"
    assert before["claim_token"] is None

    execute_job(str(tmp_path), job_id)

    after = store.job(job_id)
    assert after["status"] == "queued"
    assert after["claim_token"] is None
    assert after["pid"] is None
    assert after["message"] == before["message"]
    assert after["updated_at"] == before["updated_at"]


def test_execute_job_leaves_a_running_job_without_a_claim_token_untouched(tmp_path):
    """A running row that lost its token must not be worked on."""
    store = _store(tmp_path)
    job_id = store.new_job("backtest", {"config": {}})
    assert store.claim_jobs(TOKEN_A, 1) == [job_id]
    with store.db() as db:
        db.execute("UPDATE jobs SET claim_token=NULL WHERE id=?", (job_id,))
    before = store.job(job_id)
    assert before["status"] == "running"

    execute_job(str(tmp_path), job_id)

    after = store.job(job_id)
    assert after["status"] == "running"
    assert after["pid"] is None
    assert after["updated_at"] == before["updated_at"]


def test_recover_orphans_releases_the_resource_lease(tmp_path):
    """A dead job that is failed by the recovery must not keep its lease."""
    store = _store(tmp_path)
    job_id = store.new_job("optimization", {"config": {}})
    assert store.claim_jobs(TOKEN_A, 1) == [job_id]
    with store.db() as db:
        db.execute(
            "INSERT INTO resource_leases(job_id,pid,kind,units,updated_at) VALUES(?,?,?,?,?)",
            (job_id, 999999, "cpu", 2, now_iso()),
        )

    manager = JobManager(str(tmp_path), workers=1, poll_seconds=1.0)
    manager._recover_orphans(store, startup_grace=0.0)

    row = store.job(job_id)
    assert row["status"] == "failed"
    assert row["claim_token"] is None
    assert row["claimed_at"] is None
    with store.db() as db:
        held = db.execute(
            "SELECT COUNT(*) FROM resource_leases WHERE job_id=?", (job_id,)
        ).fetchone()[0]
    assert held == 0


def test_cpu_leases_never_exceed_the_configured_budget(tmp_path):
    """The granted units never sum above the configured CPU budget."""
    store = _store(tmp_path)
    settings = PerformanceSettings(cpu_budget_mode="custom", cpu_budget=4)
    running = [store.new_job("optimization", {"i": index}) for index in range(4)]
    assert set(store.claim_jobs(TOKEN_A, 4)) == set(running)

    granted = 0
    for job_id in running:
        lease = ResourceLease(store, job_id, settings, budget=4)
        units = lease.acquire()
        assert units >= 0
        granted += units
        assert granted <= 4
    assert granted <= 4

    with store.db() as db:
        total = db.execute(
            "SELECT COALESCE(SUM(units), 0) FROM resource_leases WHERE kind='cpu'"
        ).fetchone()[0]
    assert total <= 4


def test_cpu_leases_refuse_surplus_jobs_instead_of_flooring_to_one_unit(tmp_path):
    """More jobs than the budget can serve are refused, not floored to one."""
    store = _store(tmp_path)
    settings = PerformanceSettings(cpu_budget_mode="custom", cpu_budget=2)
    running = [store.new_job("optimization", {"i": index}) for index in range(4)]
    assert set(store.claim_jobs(TOKEN_A, 4)) == set(running)

    granted = 0
    units: list[int] = []
    for job_id in running:
        lease = ResourceLease(store, job_id, settings, budget=2)
        value = lease.acquire()
        units.append(value)
        granted += value
        assert granted <= 2
    assert granted <= 2
    # The budget is exhausted, so the surplus jobs get nothing at all.
    assert units[-1] == 0
    assert units[:2] == [1, 1]


def test_concurrent_leases_never_exceed_the_configured_budget(tmp_path):
    """Jobs that acquire at the same time still share one budget."""
    store = _store(tmp_path)
    settings = PerformanceSettings(cpu_budget_mode="custom", cpu_budget=4)
    running = [store.new_job("optimization", {"i": index}) for index in range(8)]
    assert set(store.claim_jobs(TOKEN_A, 8)) == set(running)

    barrier = threading.Barrier(len(running))
    failures: list[BaseException] = []

    def acquire(job_id: str) -> None:
        try:
            barrier.wait(timeout=30)
            ResourceLease(store, job_id, settings, budget=4).acquire()
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            failures.append(exc)

    threads = [threading.Thread(target=acquire, args=(job_id,)) for job_id in running]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for thread in threads:
        assert not thread.is_alive()
    assert not failures, failures

    with store.db() as db:
        total = db.execute(
            "SELECT COALESCE(SUM(units), 0) FROM resource_leases WHERE kind='cpu'"
        ).fetchone()[0]
    assert total <= 4


def test_timeout_stage_asks_for_a_stop_before_it_terminates():
    """The worker is asked to stop first and only killed after the grace."""
    assert timeout_stage(10, 0) == "none"
    assert timeout_stage(60, 60) == "none"
    assert timeout_stage(61, 60) == "stop"
    assert timeout_stage(90, 60) == "stop"
    assert timeout_stage(91, 60) == "terminate"


def test_runtime_limit_is_read_from_the_frozen_payload(tmp_path):
    """The runtime limit comes from the payload the job was submitted with."""
    store = _store(tmp_path)
    research = store.new_job("optimization", {"config": {"strategy": {"max_runtime_seconds": 120}}})
    completion = store.new_job("research_completion", {"max_runtime_seconds": 90})
    other = store.new_job("backtest", {"config": {}})

    assert _runtime_limit_seconds(store.job(research)) == 120
    assert _runtime_limit_seconds(store.job(completion)) == 90
    assert _runtime_limit_seconds(store.job(other)) == 0


def test_elapsed_seconds_is_measured_from_the_claim_time(tmp_path):
    """The elapsed time counts from the moment the job was claimed."""
    store = _store(tmp_path)
    job_id = store.new_job("optimization", {"config": {}})
    assert store.claim_jobs(TOKEN_A, 1) == [job_id]

    assert _elapsed_seconds(store.job(job_id)) >= 0.0
    assert _elapsed_seconds({"claimed_at": None}) == 0.0
    assert _elapsed_seconds({"claimed_at": "not-a-date"}) == 0.0


def test_request_stop_asks_the_worker_to_publish_its_evidence(tmp_path):
    """The timeout asks the worker to stop; it stays running until it does."""
    store = _store(tmp_path)
    job_id = store.new_job("optimization", {"config": {"strategy": {"max_runtime_seconds": 60}}})
    assert store.claim_jobs(TOKEN_A, 1) == [job_id]
    job = store.job(job_id)

    assert _request_stop(store, job_id, job, 60) is True
    row = store.job(job_id)
    assert row["cancel_requested"] is True
    assert "时间上限" in row["message"]
    # The job keeps running and keeps its claim so the worker can save the
    # candidates it already evaluated.
    assert row["status"] == "running"
    assert row["claim_token"] == TOKEN_A

    # A repeated request does not rewrite the row.
    assert _request_stop(store, job_id, row, 60) is False


def test_completion_revision_is_attached_to_the_job_payload(tmp_path):
    """The produced revision is recorded without dropping the other fields."""
    payload = {"result_id": "abc", "decision": "current", "reporting_budget": 1000}

    updated = _with_completion_revision(payload, {"completion_revision": 3})

    assert updated["completion_revision"] == 3
    assert updated["result_id"] == "abc"
    assert updated["decision"] == "current"
    assert updated["reporting_budget"] == 1000
    # The source payload is left alone.
    assert "completion_revision" not in payload
    # A result without a revision leaves the payload untouched.
    assert _with_completion_revision(payload, {}) is payload


def test_completion_revision_payload_is_persisted_for_the_job(tmp_path):
    """The revision survives in the job row so the endpoint can resolve it."""
    store = _store(tmp_path)
    job_id = store.new_job("research_completion", {"result_id": "abc", "decision": "current"})
    assert store.claim_jobs(TOKEN_A, 1) == [job_id]

    payload = _with_completion_revision(store.job(job_id)["payload"], {"completion_revision": 7})
    assert store.update_job(
        job_id, require_claim=True, expected_claim_token=TOKEN_A, payload=payload,
    ) is True

    row = store.job(job_id)
    assert row["payload"]["completion_revision"] == 7
    assert row["payload"]["result_id"] == "abc"
    assert row["payload"]["decision"] == "current"


def test_execute_job_refuses_to_run_without_a_cpu_share(tmp_path, monkeypatch):
    """A job that cannot get a CPU share is failed, never floored to one unit.

    The performance layer floors a CPU budget at one unit, so a job that
    received no share must never reach ``activate_performance``: it would
    silently occupy a unit the budget does not have.  The job is failed
    explicitly and both its claim and its lease are released.
    """
    store = _store(tmp_path)
    # The configured budget is four units and one running job holds all of
    # them, so the second job cannot be served.
    monkeypatch.setattr(
        "stockregression.resources.effective_cpu_budget", lambda performance: 4,
    )

    holder = store.new_job("optimization", {"config": {}})
    tested = store.new_job(
        "optimization", {"config": {}, "configuration_snapshot": {"settings": {}}},
    )
    assert set(store.claim_jobs(TOKEN_A, 2)) == {holder, tested}
    with store.db() as db:
        db.execute(
            "INSERT INTO resource_leases(job_id,pid,kind,units,updated_at) VALUES(?,?,?,?,?)",
            (holder, 4242, "cpu", 4, now_iso()),
        )

    activated: list[int] = []

    def _activate(**kwargs):
        activated.append(int(kwargs.get("cpu_budget", 0)))
        return ()

    monkeypatch.setattr("stockregression.jobs.activate_performance", _activate)

    execute_job(str(tmp_path), tested)

    # The performance context was never activated, so the job cannot occupy
    # the unit the performance layer would have floored it to.
    assert activated == []
    row = store.job(tested)
    assert row["status"] == "failed"
    assert row["claim_token"] is None
    assert row["claimed_at"] is None
    assert "配额" in row["error"]
    # The worker body never ran, so no result was produced.
    assert row["result_id"] is None
    assert row["pid"] is None

    with store.db() as db:
        rows = db.execute(
            "SELECT job_id, units FROM resource_leases WHERE kind='cpu'"
        ).fetchall()
    leases = {job_id: int(units) for job_id, units in rows}
    # The refused job released its lease; the other job keeps its share.
    assert leases == {holder: 4}
    assert sum(max(1, units) for units in leases.values()) <= 4


def test_effective_units_of_running_compute_jobs_stay_within_the_budget(tmp_path):
    """The units of the compute jobs that really run stay inside the budget."""
    store = _store(tmp_path)
    settings = PerformanceSettings(cpu_budget_mode="custom", cpu_budget=2)
    running = [store.new_job("optimization", {"i": index}) for index in range(3)]
    assert set(store.claim_jobs(TOKEN_A, 3)) == set(running)

    for job_id in running:
        ResourceLease(store, job_id, settings, budget=2).acquire()

    with store.db() as db:
        rows = db.execute(
            "SELECT l.job_id, l.units FROM resource_leases l "
            "JOIN jobs j ON j.id=l.job_id WHERE l.kind='cpu' AND j.status='running'"
        ).fetchall()
    leases = {job_id: int(units) for job_id, units in rows}
    # The job that got no share does not occupy a unit: it is refused by
    # ``execute_job`` before the performance context is activated.
    assert sum(max(1, units) for units in leases.values() if units > 0) <= 2
    assert sum(1 for units in leases.values() if units > 0) <= 2
