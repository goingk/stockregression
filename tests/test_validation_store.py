from __future__ import annotations

import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import stockregression.validation_store as validation_store_module
from stockregression.storage import Store
from stockregression.validation_store import (
    ValidationCheckpointIntegrityError,
    ValidationLedgerIntegrityError,
    ValidationRegistrationConflict,
    ValidationRevisionConflict,
    ValidationStore,
)


SYMBOL = "600000.SH"


def _request(*, family_id: str | None = "family-a") -> dict:
    return {
        "suite_version": "research_validation_v1",
        "family_id": family_id,
        "parent_result_id": None,
        "symbols": [SYMBOL],
        "research": {
            "research_context": {
                "symbol": SYMBOL,
                "start_date": "2024-01-02",
                "end_date": "2024-03-29",
            },
        },
        "total_evaluation_budget": 50,
        "max_runtime_seconds": 3600,
        "human_untouched_verified": True,
    }


def _plan(*, plan_hash: str = "declared-plan-hash") -> dict:
    # This mirrors the interval shape emitted by validation_planner.py.
    return {
        "suite_version": "research_validation_v1",
        "status": "ready",
        "symbols": [{
            "symbol": SYMBOL,
            "status": "ready",
            "evaluation_start": "2024-01-02",
            "evaluation_end": "2024-03-29",
            "blocks": [{
                "index": 0,
                "start": "2024-01-02",
                "end": "2024-03-29",
            }],
        }],
        "estimated_work": {
            "total_evaluation_budget": 50,
            "max_runtime_seconds": 3600,
        },
        "plan_hash": plan_hash,
    }


def _provenance() -> dict:
    return {
        "code_version": "test-v1",
        "data_versions": {SYMBOL: "snapshot-a"},
        "reproducible": True,
    }


def _new_store(tmp_path) -> tuple[Store, ValidationStore]:
    store = Store(tmp_path / "data")
    return store, ValidationStore(store)


def test_register_plan_is_idempotent_and_rejects_contract_changes(tmp_path):
    store, validations = _new_store(tmp_path)

    first = validations.register_plan(
        "run-idempotent", _request(), _plan(), _provenance(),
    )

    assert first["idempotent_replay"] is False
    assert first["family_id"] == "family-a"
    assert first["plan_hash"] == "declared-plan-hash"
    assert first["human_untouched_verified"] is False
    assert first["contaminated"] is None
    assert first["contamination_assessed"] is False
    assert first["recorded_overlap_status"] == "no_recorded_overlap"
    assert first["symbol_intervals"][SYMBOL][0]["start"] == "2024-01-02"
    assert first["symbol_intervals"][SYMBOL][0]["end"] == "2024-03-29"
    assert first["budget"]["current_total_evaluation_budget"] == 50

    replay = validations.register_plan(
        "run-idempotent", _request(), _plan(), _provenance(),
    )
    assert replay["idempotent_replay"] is True
    assert replay["created_at"] == first["created_at"]

    with pytest.raises(ValidationRegistrationConflict):
        validations.register_plan(
            "run-idempotent", {**_request(), "total_evaluation_budget": 51},
            _plan(), _provenance(),
        )
    with pytest.raises(ValidationRegistrationConflict):
        validations.register_plan(
            "run-idempotent", _request(), _plan(plan_hash="different-plan"),
            _provenance(),
        )
    with pytest.raises(ValidationRegistrationConflict):
        validations.register_plan(
            "run-idempotent", _request(), _plan(),
            {**_provenance(), "code_version": "test-v2"},
        )

    with store.db() as db:
        rows = db.execute("SELECT COUNT(*) FROM validation_runs").fetchone()[0]
        intervals = db.execute(
            "SELECT COUNT(*) FROM validation_symbol_intervals"
        ).fetchone()[0]
    assert rows == 1
    assert intervals == 1


def test_schema_initialization_is_reused_for_a_catalog(tmp_path, monkeypatch):
    first_store, _ = _new_store(tmp_path)
    second_store = Store(tmp_path / "data")

    def unexpected_database_open():
        raise AssertionError("schema was initialized more than once")

    monkeypatch.setattr(second_store, "db", unexpected_database_open)
    ValidationStore(second_store)

    with first_store.db() as db:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'validation_%'"
            )
        }
    assert {
        "validation_runs", "validation_family_parents",
        "validation_symbol_intervals", "validation_events", "validation_views",
        "validation_revisions", "validation_checkpoints", "validation_store_state",
    } <= tables


def test_concurrent_registration_and_event_append_are_serialized(tmp_path):
    _, validations = _new_store(tmp_path)

    def register(_):
        return validations.register_plan(
            "run-concurrent", _request(), _plan(), _provenance(),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        registrations = list(pool.map(register, range(8)))

    assert sum(not item["idempotent_replay"] for item in registrations) == 1
    assert all(item["run_id"] == "run-concurrent" for item in registrations)

    def append(index):
        return validations.append_event(
            "run-concurrent", f"unit-{index}", "completed",
            {"candidate_evaluations": 2, "payload": index},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        events = list(pool.map(append, range(24)))

    assert len({item["id"] for item in events}) == 24
    summary = validations.audit_summary("run-concurrent")
    assert summary["events"]["total"] == 24
    assert summary["events"]["actual_evaluations"] == 48
    assert summary["actual_trial_evaluations"] == 48


def test_explicit_views_preserve_the_first_view_and_every_repeat(tmp_path):
    store, validations = _new_store(tmp_path)
    validations.register_plan(
        "run-views", _request(), _plan(), _provenance(),
    )
    store.save_result(
        "research_validation",
        {"run_id": "run-views", "status": "completed"},
        SYMBOL,
        "完整研究验证",
        result_id="result-views",
    )

    first = validations.record_view("result-views")
    second = validations.record_view("result-views", section="overview")
    third = validations.record_view("result-views", section="overview")
    other_section = validations.record_view("result-views", section="evidence")
    store.save_result(
        "research_validation",
        {"status": "completed"},
        SYMBOL,
        "无 run_id 的旧结果",
        result_id="result-without-run",
    )
    result_id_fallback = validations.record_view("result-without-run")

    assert first["run_id"] == "run-views"
    assert first["first_view"] is True
    assert second["first_view"] is False
    assert third["first_view"] is False
    assert third["view_count"] == 3
    assert first["first_viewed_at"] == third["first_viewed_at"]
    assert other_section["first_view"] is True
    assert result_id_fallback["run_id"] == "result-without-run"

    summary = validations.audit_summary("run-views")
    assert summary["views"]["total"] == 4
    assert summary["views"]["sections"]["overview"]["count"] == 3
    assert summary["views"]["sections"]["overview"]["first_viewed_at"] == first["viewed_at"]
    assert summary["views"]["sections"]["evidence"]["count"] == 1


def test_overlap_is_recorded_across_families_without_contamination_claim(tmp_path):
    store, validations = _new_store(tmp_path)
    intervals = [{"start": "2024-01-01", "end": "2024-04-30", "kind": "holdout"}]
    store.register_research(
        "legacy-study", "different-method", SYMBOL,
        "2023-01-01", "2024-04-30", "snapshot-old", intervals,
    )

    with store.db() as db:
        before = [dict(row) for row in db.execute(
            "SELECT * FROM research_registry ORDER BY registry_id"
        )]

    validations.register_plan(
        "run-family-a", _request(family_id="family-a"), _plan(), _provenance(),
    )
    validations.register_plan(
        "run-family-b", _request(family_id="family-b"), _plan(),
        {**_provenance(), "family_id": "family-b"},
    )

    summary = validations.audit_summary("run-family-a")
    assert summary["recorded_overlap"]["detected"] is True
    assert summary["recorded_overlap"]["recorded_registry_studies"] == 1
    assert summary["recorded_overlap"]["validation_runs"] == 1
    assert summary["recorded_overlap"]["different_family_runs"] == 1
    assert summary["recorded_overlap"]["families"] == ["family-b"]
    assert summary["family"]["family_id"] == "family-a"
    assert {item["run_id"] for item in summary["family"]["members"]} == {
        "run-family-a",
    }
    assert summary["recorded_overlap_status"] == "overlap_detected"
    assert summary["overlapping_prior_studies"] == 1
    assert summary["overlapping_validation_runs"] == 1
    assert summary["exact_repeat_prior_studies"] == 0
    assert summary["exact_repeat_validation_runs"] == 1
    assert summary["contamination_assessed"] is False
    assert summary["contamination_status"] == "not_assessed"
    assert summary["contaminated"] is None
    assert summary["overlap_implies_contamination"] is False
    assert summary["human_untouched_verified"] is False

    with store.db() as db:
        after = [dict(row) for row in db.execute(
            "SELECT * FROM research_registry ORDER BY registry_id"
        )]
    assert after == before


def test_failed_and_cached_events_remain_visible_without_double_charging(tmp_path):
    _, validations = _new_store(tmp_path)
    validations.register_plan("run-events", _request(), _plan(), _provenance())

    validations.append_event("run-events", "unit-a", "started")
    failed = validations.append_event(
        "run-events", "unit-a", "failed",
        {"error": "missing minute", "failure_category": "quality",
         "candidate_evaluations": 999},
    )
    cached = validations.append_event(
        "run-events", "unit-a", "completed",
        {"candidate_evaluations": 100, "cache_hit": True},
    )
    reused = validations.append_event(
        "run-events", "unit-a", "cache_reused",
        {"source": "qualification_base", "candidate_evaluations": 50},
    )
    completed = validations.append_event(
        "run-events", "unit-a", "completed",
        {"candidate_evaluations": 7},
    )

    assert failed["failed"] is True
    assert failed["failure_category"] == "quality"
    assert failed["actual_evaluations"] == 0
    assert cached["cache_hit"] is True
    assert cached["actual_evaluations"] == 0
    assert reused["cache_hit"] is True
    assert reused["actual_evaluations"] == 0
    assert completed["actual_evaluations"] == 7

    summary = validations.audit_summary("run-events")
    assert summary["events"]["failed"] == 1
    assert summary["events"]["cache_hits"] == 2
    assert summary["events"]["failure_categories"] == {"quality": 1}
    assert summary["actual_trial_evaluations"] == 7
    failed_row = next(item for item in summary["events"]["log"] if item["failed"])
    assert failed_row["unit_id"] == "unit-a"
    assert failed_row["payload"]["error"] == "missing minute"
    assert failed_row["created_at"]


def test_checkpoint_replace_is_atomic_and_budget_revisions_are_monotonic(
    tmp_path, monkeypatch,
):
    _, validations = _new_store(tmp_path)
    validations.register_plan("run-checkpoint", _request(), _plan(), _provenance())

    folder = tmp_path / "data" / "validation-checkpoints"
    assert validations.load_checkpoint("run-checkpoint") is None
    assert not folder.exists()

    validations.save_checkpoint(
        "run-checkpoint", {"stage": "training", "consumed": 10},
    )
    assert validations.load_checkpoint("run-checkpoint") == {
        "consumed": 10, "stage": "training",
    }

    original_replace = validation_store_module.os.replace
    replace_calls = []

    def transient_sharing_error(source, destination):
        replace_calls.append(source)
        if len(replace_calls) < 3:
            raise PermissionError("temporary sharing violation")
        original_replace(source, destination)

    monkeypatch.setattr(
        validation_store_module.os, "replace", transient_sharing_error,
    )
    monkeypatch.setattr(validation_store_module.time, "sleep", lambda _: None)
    validations.save_checkpoint(
        "run-checkpoint", {"stage": "training", "consumed": 20},
    )
    assert len(replace_calls) == 3
    assert validations.load_checkpoint("run-checkpoint")["consumed"] == 20

    def failed_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(validation_store_module.os, "replace", failed_replace)
    with pytest.raises(OSError):
        validations.save_checkpoint(
            "run-checkpoint", {"stage": "training", "consumed": 99},
        )
    assert validations.load_checkpoint("run-checkpoint")["consumed"] == 20
    assert not list(folder.glob("*.writing"))

    for bad_run_id in ("../escape", "nested/escape", "..", "CON"):
        with pytest.raises(ValueError):
            validations.save_checkpoint(bad_run_id, {"stage": "bad"})
        with pytest.raises(ValueError):
            validations.load_checkpoint(bad_run_id)

    first = validations.register_revision(
        "run-checkpoint", {"updates": {"total_evaluation_budget": 60}},
    )
    second = validations.register_revision(
        "run-checkpoint", {
            "max_runtime_seconds": 7200, "plan_hash": "revised-plan-hash",
        },
    )
    assert first["revision"] == 1
    assert second["revision"] == 2

    summary = validations.audit_summary("run-checkpoint")
    assert [item["revision"] for item in summary["revisions"]] == [1, 2]
    assert summary["budget"]["revision_count"] == 2
    assert summary["budget"]["current_total_evaluation_budget"] == 60
    assert summary["budget"]["current_max_runtime_seconds"] == 7200
    assert summary["current_plan_hash"] == "revised-plan-hash"
    assert Path(summary["checkpoint"]["path"]) == (
        folder / "run-checkpoint.json"
    )


def test_frozen_checkpoint_uses_saved_payload_and_survives_latest_updates(tmp_path):
    store, validations = _new_store(tmp_path)
    validations.register_plan(
        "run-frozen", _request(), _plan(), _provenance(),
    )
    assert validations.freeze_checkpoint("run-frozen") is None

    old_payload = {
        "schema_version": 1,
        "revision": 1,
        "ledgers": {"unit-old": {"cash": [1, 2, 3]}},
    }
    saved = validations.save_checkpoint("run-frozen", old_payload)

    # Simulate a crash/race after the mutable pointer was written. Freezing
    # must use the atomically saved SQLite payload, not this damaged file.
    (store.root / saved["relative_path"]).write_text(
        '{"ledgers":{"wrong":true}}', encoding="utf-8",
    )
    old_ref = validations.freeze_checkpoint("run-frozen")
    assert old_ref == (
        f"validation-checkpoints/immutable/{saved['sha256']}.json"
    )
    assert validations.load_frozen_checkpoint(old_ref) == old_payload
    frozen_path = store.root / old_ref
    old_bytes = frozen_path.read_bytes()

    new_payload = {
        "schema_version": 1,
        "revision": 2,
        "ledgers": {"unit-new": {"cash": [4, 5, 6]}},
    }
    validations.save_checkpoint("run-frozen", new_payload)
    validations.register_revision(
        "run-frozen", {
            "plan_hash": "new-plan-hash",
            "updates": {"total_evaluation_budget": 75},
        },
    )

    assert validations.load_checkpoint("run-frozen") == new_payload
    assert validations.load_frozen_checkpoint(old_ref) == old_payload
    assert frozen_path.read_bytes() == old_bytes

    with ThreadPoolExecutor(max_workers=8) as pool:
        new_refs = list(pool.map(
            lambda _: validations.freeze_checkpoint("run-frozen"), range(8),
        ))
    assert len(set(new_refs)) == 1
    assert new_refs[0] != old_ref
    assert validations.load_frozen_checkpoint(new_refs[0]) == new_payload
    assert validations.load_frozen_checkpoint(old_ref) == old_payload


def test_frozen_checkpoint_reads_are_read_only_and_integrity_checked(
    tmp_path, monkeypatch,
):
    store, validations = _new_store(tmp_path)
    validations.register_plan(
        "run-read-only", _request(), _plan(), _provenance(),
    )
    validations.save_checkpoint("run-read-only", {"ledgers": {"u": {"x": 1}}})
    ref = validations.freeze_checkpoint("run-read-only")
    assert ref is not None

    for bad_ref in (
        "../validation-checkpoints/immutable/" + "0" * 64 + ".json",
        "/validation-checkpoints/immutable/" + "0" * 64 + ".json",
        "validation-checkpoints/immutable/" + "A" * 64 + ".json",
        "validation-checkpoints/other/" + "0" * 64 + ".json",
    ):
        with pytest.raises(ValueError):
            validations.load_frozen_checkpoint(bad_ref)

    missing_ref = "validation-checkpoints/immutable/" + "0" * 64 + ".json"

    def unexpected_database_access():
        raise AssertionError("frozen checkpoint reads must not use SQLite")

    monkeypatch.setattr(store, "db", unexpected_database_access)
    assert validations.load_frozen_checkpoint(None) is None
    assert validations.load_frozen_checkpoint(missing_ref) is None
    assert validations.load_frozen_checkpoint(ref) == {
        "ledgers": {"u": {"x": 1}},
    }

    # A read still verifies the content-address hash instead of trusting the
    # filename or materializing corrupted ledgers.
    bad_digest = "1" * 64
    bad_ref = f"validation-checkpoints/immutable/{bad_digest}.json"
    bad_path = store.root / bad_ref
    bad_path.write_text('{"ledgers":{"tampered":true}}', encoding="utf-8")
    with pytest.raises(ValidationCheckpointIntegrityError):
        validations.load_frozen_checkpoint(bad_ref)


def test_ledger_is_content_addressed_compressible_and_streamable(tmp_path, monkeypatch):
    store, validations = _new_store(tmp_path)
    validations.register_plan(
        "run-ledger", _request(), _plan(), _provenance(),
    )
    ledger = {
        "grid": {
            "trades": [
                {"timestamp": "2024-01-02T09:31:00", "quantity": index}
                for index in range(100)
            ],
        },
        "large_text": "x" * (2 * 64 * 1024),
    }

    ref = validations.save_ledger("run-ledger", "qualification:unit", ledger)
    assert set(ref) == {"__validation_ledger__", "sha256"}
    relative = ref["__validation_ledger__"]
    assert relative.startswith("validation-ledgers/immutable/")
    assert relative.endswith(".json.gz")
    assert ref["sha256"] in relative

    target = store.root / relative
    compressed = target.read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == ref["sha256"]
    decoded = gzip.decompress(compressed).decode("utf-8")
    assert decoded == json.dumps(
        ledger, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    assert validations.load_ledger(ref) == ledger

    chunks = list(validations.iter_ledger_json(ref))
    assert len(chunks) > 1
    assert "".join(chunks) == decoded

    with ThreadPoolExecutor(max_workers=8) as pool:
        duplicates = list(pool.map(
            lambda index: validations.save_ledger(
                "run-ledger", f"duplicate-{index}", ledger,
            ),
            range(8),
        ))
    assert all(item == ref for item in duplicates)

    # Ledger reads are independent of the mutable SQLite catalog.
    def unexpected_database_access():
        raise AssertionError("ledger reads must not use SQLite")

    monkeypatch.setattr(store, "db", unexpected_database_access)
    assert validations.load_ledger(ref) == ledger
    assert "".join(validations.iter_ledger_json(ref)) == decoded


def test_checkpoint_keeps_ledger_references_and_frozen_copy_is_stable(tmp_path):
    store, validations = _new_store(tmp_path)
    validations.register_plan(
        "run-ledger-checkpoint", _request(), _plan(), _provenance(),
    )
    ledger = {"grid": {"equity": [1, 2, 3], "large_text": "x" * 200_000}}
    ref = validations.save_ledger(
        "run-ledger-checkpoint", "qualification:unit", ledger,
    )
    checkpoint = {
        "schema_version": 1,
        "revision": 1,
        "cells": [{"unit_id": "qualification:unit", "status": "completed"}],
        "ledgers": {"qualification:unit": ref},
    }
    validations.save_checkpoint("run-ledger-checkpoint", checkpoint)
    with store.db() as db:
        stored = db.execute(
            "SELECT payload_json FROM validation_checkpoints WHERE run_id=?",
            ("run-ledger-checkpoint",),
        ).fetchone()["payload_json"]
    assert "large_text" not in stored
    assert ref["__validation_ledger__"] in stored
    assert len(stored) < 2048
    frozen_ref = validations.freeze_checkpoint("run-ledger-checkpoint")
    assert frozen_ref is not None

    frozen = validations.load_frozen_checkpoint(frozen_ref)
    assert frozen is not None
    assert frozen["ledgers"]["qualification:unit"] == ref
    assert "large_text" not in json.dumps(frozen)
    assert validations.load_ledger(frozen["ledgers"]["qualification:unit"]) == ledger

    second = {
        **checkpoint,
        "revision": 2,
        "cells": [{"unit_id": "qualification:unit", "status": "completed"}],
    }
    validations.save_checkpoint("run-ledger-checkpoint", second)
    assert validations.load_checkpoint("run-ledger-checkpoint")["revision"] == 2
    assert validations.load_frozen_checkpoint(frozen_ref)["revision"] == 1
    assert validations.load_ledger(ref) == ledger


def test_ledger_references_reject_unsafe_paths_and_tampering(tmp_path):
    store, validations = _new_store(tmp_path)
    validations.register_plan(
        "run-ledger-safety", _request(), _plan(), _provenance(),
    )
    ref = validations.save_ledger(
        "run-ledger-safety", "unit", {"ledger": [1, 2, 3]},
    )
    digest = ref["sha256"]
    bad_refs = [
        {},
        *[
            {"__validation_ledger__": relative, "sha256": digest}
            for relative in (
                f"../validation-ledgers/immutable/{digest}.json.gz",
                f"/validation-ledgers/immutable/{digest}.json.gz",
                f"validation-ledgers/immutable/{digest.upper()}.json.gz",
                f"validation-ledgers/other/{digest}.json.gz",
            )
        ],
    ]
    bad_refs.extend([
        {
            "__validation_ledger__": f"validation-ledgers/immutable/{digest}.json.gz",
            "sha256": "0" * 64,
        },
        {"__validation_ledger__": "C:\\ledger.json.gz", "sha256": digest},
    ])
    for bad in bad_refs:
        with pytest.raises((TypeError, ValidationLedgerIntegrityError)):
            validations.load_ledger(bad)

    missing = {
        "__validation_ledger__": (
            "validation-ledgers/immutable/" + "0" * 64 + ".json.gz"
        ),
        "sha256": "0" * 64,
    }
    with pytest.raises(ValidationLedgerIntegrityError):
        validations.load_ledger(missing)

    target = store.root / ref["__validation_ledger__"]
    target.write_bytes(b"tampered gzip")
    with pytest.raises(ValidationLedgerIntegrityError):
        validations.load_ledger(ref)
    with pytest.raises(ValidationLedgerIntegrityError):
        validations.iter_ledger_json(ref)


def test_revision_limits_cannot_decrease_and_failures_do_not_append(tmp_path):
    _, validations = _new_store(tmp_path)
    validations.register_plan(
        "run-revisions", _request(), _plan(), _provenance(),
    )

    with pytest.raises(ValidationRevisionConflict):
        validations.register_revision(
            "run-revisions", {"updates": {"total_evaluation_budget": 49}},
        )
    with pytest.raises(ValidationRevisionConflict):
        validations.register_revision(
            "run-revisions", {"max_runtime_seconds": 3599},
        )
    with pytest.raises(ValidationRevisionConflict):
        validations.register_revision(
            "run-revisions", {"total_evaluation_budget": 0},
        )
    assert validations.audit_summary("run-revisions")["revisions"] == []

    # Equal values remain valid because a resume may revise the plan without
    # consuming additional budget; every later limit remains non-decreasing.
    first = validations.register_revision(
        "run-revisions", {"total_evaluation_budget": 50},
    )
    second = validations.register_revision(
        "run-revisions", {
            "total_evaluation_budget": 80,
            "max_runtime_seconds": 7200,
            "plan_hash": "revision-plan-2",
        },
    )
    assert (first["revision"], second["revision"]) == (1, 2)

    with pytest.raises(ValidationRevisionConflict):
        validations.register_revision(
            "run-revisions", {"updates": {"total_evaluation_budget": 79}},
        )
    summary = validations.audit_summary("run-revisions")
    assert [item["revision"] for item in summary["revisions"]] == [1, 2]
    assert summary["budget"]["current_total_evaluation_budget"] == 80
    assert summary["budget"]["current_max_runtime_seconds"] == 7200


def test_audit_and_checkpoint_reads_do_not_write(tmp_path, monkeypatch):
    from contextlib import contextmanager

    store, validations = _new_store(tmp_path)
    validations.register_plan(
        "run-get", _request(), _plan(), _provenance(),
    )
    validations.save_checkpoint("run-get", {"ledgers": {"unit": {"x": 1}}})
    ref = validations.freeze_checkpoint("run-get")
    assert ref is not None
    real_db = store.db

    @contextmanager
    def query_only_db():
        with real_db() as db:
            db.execute("PRAGMA query_only=ON")
            yield db

    with monkeypatch.context() as patch:
        patch.setattr(store, "db", query_only_db)
        summary = validations.audit_summary("run-get")
        assert summary["human_untouched_verified"] is False
        assert validations.load_checkpoint("run-get") == {
            "ledgers": {"unit": {"x": 1}},
        }
        assert validations.load_frozen_checkpoint(ref) == {
            "ledgers": {"unit": {"x": 1}},
        }
