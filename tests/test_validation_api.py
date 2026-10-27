from copy import deepcopy
import json

import pytest
from fastapi.testclient import TestClient

from stockregression import settings as settings_module
from stockregression.api import create_app
from stockregression.validation_models import ResearchValidationRequest
from stockregression.validation_planner import plan_hash, stable_hash
from stockregression.validation_store import ValidationStore


def payload():
    return {"research": {"research_context": {"symbol": "159611.SZ", "start_date": "2024-01-01"},
            "ranges": {"buy_step": {"low": .01, "high": .03, "step": .01},
                       "sell_step": {"low": .01, "high": .03, "step": .01},
                       "buy_shares": {"low": 100, "high": 300, "step": 100},
                       "sell_shares": {"low": 100, "high": 300, "step": 100}}},
            "accounts": {"159611.SZ": {"total_budget": 100000, "max_shares": 2000}}}


@pytest.fixture
def app(tmp_path, monkeypatch):
    from stockregression import validation_planner, reproducibility
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", tmp_path / "application.json")
    def provenance(root, *, configuration=None):
        return {"engine_version": "test", "source_fingerprint": "fixed-source",
                "uncommitted_diff_fingerprint": "fixed", "dependency_lock_hash": "fixed",
                "configuration_fingerprint": stable_hash(configuration), "runtime": {"packages": {}, "python": "3.12"}}
    monkeypatch.setattr(reproducibility, "computation_provenance", provenance)
    def preview(store, request, **kwargs):
        result = {"status": "ready", "suite_version": request.suite_version,
            "primary_arm": "same_price:off:adaptive", "data_versions": {"159611.SZ": "frozen"},
            "symbols": [{"symbol": "159611.SZ", "status": "ready", "data_version": "frozen",
                "evaluation_start": "2024-01-02", "evaluation_end": "2024-12-31", "blocks": []}],
            "estimated_work": {"total_evaluation_budget": request.total_evaluation_budget or 500,
                               "max_runtime_seconds": request.max_runtime_seconds}}
        result["plan_hash"] = plan_hash(request, result)
        return result
    monkeypatch.setattr(validation_planner, "plan_validation", preview)
    return create_app(str(tmp_path / "data"), start_jobs=False)


def submit(client):
    value = payload()
    plan = client.post("/api/v1/research-validations/preview", json=value)
    assert plan.status_code == 200, plan.text
    value["plan_hash"] = plan.json()["plan_hash"]
    value["requested_data_versions"] = plan.json()["data_versions"]
    response = client.post("/api/v1/research-validations", json=value)
    assert response.status_code == 202, response.text
    return response.json(), value


def test_submission_pins_plan_registers_before_results_and_deduplicates(app):
    with TestClient(app) as client:
        first, value = submit(client)
        assert first["audit"]["human_untouched_verified"] is False
        second = client.post("/api/v1/research-validations", json=value).json()
        assert first["job_id"] == second["job_id"]
        assert first["run_id"] == second["run_id"]
        value["accounts"]["159611.SZ"]["total_budget"] += 100
        conflict = client.post("/api/v1/research-validations", json=value)
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "VALIDATION_PLAN_CHANGED"
        del value["plan_hash"]
        assert client.post("/api/v1/research-validations", json=value).status_code == 409


def test_reads_do_not_record_views_and_full_export_contains_saved_ledgers(app):
    with TestClient(app) as client:
        submitted, _ = submit(client)
        store = app.state.store
        audit = ValidationStore(store)
        ledger = {"grid": {"equity": [{"timestamp": "2024-01-02T15:00:00", "equity": 100000}],
                           "trades": [{"timestamp": "2024-01-02T14:31:00", "quantity": 100}]}}
        audit.save_checkpoint(submitted["run_id"], {"ledgers": {"q:a": ledger}})
        result = {"kind": "research_validation", "run_id": submitted["run_id"],
                  "checkpoint_snapshot": audit.freeze_checkpoint(submitted["run_id"]),
                  "result_schema_version": 1, "cells": [{"unit_id": "q:a", "checkpoint_ref": "q:a", "symbol": "159611.SZ"}]}
        result_id = store.save_result("research_validation", result, "159611.SZ", "fixture")
        changed = deepcopy(ledger)
        changed["grid"]["trades"][0]["quantity"] = 999
        audit.save_checkpoint(submitted["run_id"], {"ledgers": {"q:a": changed}})
        for _ in range(2):
            response = client.get(f"/api/v1/results/{result_id}")
            assert response.status_code == 200, response.text
        assert audit.audit_summary(submitted["run_id"])["views"]["total"] == 0
        viewed = client.post(f"/api/v1/results/{result_id}/evidence-views", json={"section": "overview"})
        assert viewed.status_code == 200, viewed.text
        assert audit.audit_summary(submitted["run_id"])["views"]["total"] == 1
        full = client.get(f"/api/v1/results/{result_id}?full=true").json()
        assert full["cells"][0]["ledger"]["grid"]["trades"][0]["quantity"] == 100
        exported = client.get(f"/api/v1/results/{result_id}/export?kind=trades")
        assert exported.status_code == 200
        assert "quantity" in exported.text and "100" in exported.text


def test_claimed_job_connects_runner_storage_and_immutable_streaming_exports(app, monkeypatch):
    from stockregression import validation_planner, validation_runner
    from stockregression.jobs import execute_job
    with TestClient(app) as client:
        submitted, _ = submit(client)
        store = app.state.store
        calls = []
        monkeypatch.setattr(validation_planner, "load_validation_bundles", lambda *a, **kw: {})
        def runner(request, plan, bundles, **kwargs):
            calls.append(request.primary_branch)
            ref = kwargs["save_unit_ledger"]("q:one", {"grid": {"trades": [{"quantity": 321}]}})
            kwargs["save_checkpoint"]({"suite_version": request.suite_version, "plan_hash": plan["plan_hash"],
                                      "cells": [], "ledgers": {"q:one": ref}})
            kwargs["audit_event"]("q:one", "completed", {"ledger_evaluations": 1})
            return {"status": "partial", "completion_status": "incomplete", "candidate_status": "no_candidate",
                    "cells": [{"unit_id": "q:one", "checkpoint_ref": "q:one"}]}
        monkeypatch.setattr(validation_runner, "run_validation", runner)
        assert store.claim_job(submitted["job_id"], "validation-test-worker")
        execute_job(str(store.root), submitted["job_id"])
        job = store.job(submitted["job_id"])
        assert job["status"] == "partial", job.get("error")
        assert calls == ["adaptive"]
        result = store.result(job["result_id"])
        assert result["checkpoint_snapshot"]
        assert "__validation_ledger__" in ValidationStore(store).load_checkpoint(submitted["run_id"])["ledgers"]["q:one"]
        summary = client.get(f"/api/v1/jobs/{job['id']}/result")
        assert summary.status_code == 200, summary.text
        full = client.get(f"/api/v1/results/{job['result_id']}?full=true")
        assert full.status_code == 200, full.text
        assert full.json()["cells"][0]["ledger"]["grid"]["trades"][0]["quantity"] == 321
        exported = client.get(f"/api/v1/results/{job['result_id']}/export")
        assert exported.status_code == 200
        assert exported.json()["cells"][0]["ledger"]["grid"]["trades"][0]["quantity"] == 321
        assert exported.json()["ledger_refs"]["q:one"]["storage"] == "immutable_ledger"
        assert "grid" not in exported.json()["ledger_refs"]["q:one"]
        ref = ValidationStore(store).load_checkpoint(submitted["run_id"])["ledgers"]["q:one"]
        (store.root / ref["__validation_ledger__"]).unlink()
        unavailable = client.get(f"/api/v1/results/{job['result_id']}?full=true")
        assert unavailable.status_code == 409
        assert unavailable.json()["detail"]["code"] == "VALIDATION_LEDGER_UNAVAILABLE"
        assert client.get(f"/api/v1/results/{job['result_id']}").status_code == 200


def test_resume_preserves_contract_and_budget_extension_is_explicit(app, monkeypatch):
    with TestClient(app) as client:
        submitted, _ = submit(client)
        store = app.state.store
        store.update_job(submitted["job_id"], status="partial")
        monkeypatch.setattr(store, "snapshot", lambda *a, **kw: {"version": "frozen"})
        response = client.post(f"/api/v1/jobs/{submitted['job_id']}/resume", json={"max_runtime_seconds": 100000})
        assert response.status_code == 202, response.text
        resumed = store.job(response.json()["job_id"])
        assert resumed["payload"]["run_id"] == submitted["run_id"]
        assert resumed["payload"]["config"]["max_runtime_seconds"] == 100000
        assert resumed["payload"]["plan"]["symbols"] == store.job(submitted["job_id"])["payload"]["plan"]["symbols"]
        replay = client.post(f"/api/v1/jobs/{submitted['job_id']}/resume", json={"max_runtime_seconds": 100000})
        assert replay.status_code == 202
        assert replay.json()["job_id"] == resumed["id"]
        changed_while_active = client.post(f"/api/v1/jobs/{submitted['job_id']}/resume", json={"max_runtime_seconds": 110000})
        assert changed_while_active.status_code == 409
        assert len(ValidationStore(store).audit_summary(submitted["run_id"])["revisions"]) == 1
        smaller = client.post(f"/api/v1/jobs/{submitted['job_id']}/resume", json={"max_runtime_seconds": 1})
        assert smaller.status_code == 422


def test_resume_refuses_changed_computation_source(app, monkeypatch):
    from stockregression import reproducibility
    with TestClient(app) as client:
        submitted, _ = submit(client)
        app.state.store.update_job(submitted["job_id"], status="partial")
        monkeypatch.setattr(reproducibility, "computation_provenance", lambda *a, **kw: {"source_fingerprint": "changed"})
        response = client.post(f"/api/v1/jobs/{submitted['job_id']}/resume")
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "VALIDATION_PROVENANCE_CHANGED"


def test_validation_shares_cpu_budget_with_other_research(app):
    from stockregression.resources import ResourceLease
    from stockregression.jobs import _runtime_limit_seconds
    store = app.state.store
    first = store.new_job("research_validation", {"config": {"max_runtime_seconds": 86400}})
    second = store.new_job("optimization", {})
    assert store.claim_job(first, "worker-a")
    assert store.claim_job(second, "worker-b")
    one, two = ResourceLease(store, first, budget=4), ResourceLease(store, second, budget=4)
    try:
        assert one.acquire() == 2
        assert two.acquire() == 2
        assert one.units + two.units <= 4
        assert _runtime_limit_seconds(store.job(first)) == 86400
    finally:
        one.release()
        two.release()
