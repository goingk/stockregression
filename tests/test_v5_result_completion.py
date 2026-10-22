"""S3/S6 回归测试：结果读取的惰性、补算去重与计算来源门禁。

契约要点：

* GET /api/v1/results/{id} 只读归档，不得调用评价器或加载行情；
* 同一结果的并发补算必须去重成同一个后台任务；
* 来源指纹、依赖锁与配置指纹不一致时必须拒绝补算；
* 运行期源码变化必须把结果标记为不可复现，并据此拒绝补算。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.reproducibility import computation_provenance, reproducibility_mismatches
from stockregression.research_completion import complete_research_result
from stockregression.settings import PROJECT_ROOT
from stockregression.storage import CompletionRevisionError, Store, _write_candidate_archive


def provenance(*, reproducible=True, completion_allowed=True, source_fingerprint="src-1",
               dependency_lock_hash="lock-1", configuration_fingerprint="config-1") -> dict:
    return {
        "reproducible": reproducible,
        "completion_allowed": completion_allowed,
        "mismatches": [],
        "finished": {
            "engine_version": "0.1.0", "git_commit": "abc", "source_fingerprint": source_fingerprint,
            "dependency_lock_hash": dependency_lock_hash, "configuration_fingerprint": configuration_fingerprint,
            "runtime": {"packages": {"numpy": "2.1.0"}},
        },
    }


def v5_result(**overrides) -> dict:
    result = {
        "protocol_version": 5, "result_schema_version": 2,
        "symbol": "600000.SH", "data_version": "v-1",
        "config": {"protocol_version": 5, "base": {"symbol": "600000.SH"}},
        "computation_provenance": provenance(),
        "evidence": {"candidate_archives": {"current": []}},
        "fixed": {"status": "completed", "test_result": None},
        "adaptive": {"status": "not_requested", "schedule": [], "test_result": None},
    }
    result.update(overrides)
    return result


def landscape_row(index: int, excess: float | None = None) -> dict:
    """构造一条训练候选行。"""
    return {
        "candidate_id": index,
        "params": {
            "buy_step": .001 * index, "sell_step": .02,
            "buy_shares": 100, "sell_shares": 200,
        },
        "evaluation_status": "completed",
        "training_excess_return": index / 100 if excess is None else excess,
        "max_principal_drawdown": .05, "trade_count": 20,
        "buy_count": 5, "sell_count": 5,
        "actual_min_pos_ratio": 0.0, "actual_max_pos_ratio": 1.0,
        "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
        "j_score": .01 * index, "neighborhood_mean": .01,
        "downside_risk_tau": 0.0, "pos_penalty": 0.0,
        "score_complete": True, "score_missing_count": 0,
    }


def landscape_result(rows) -> dict:
    """构造一个可以读取参数分布的 v5 结果。"""
    return {
        "protocol_version": 5, "result_schema_version": 2,
        "symbol": "600000.SH", "data_version": "v-1",
        "config": {
            "protocol_version": 5,
            "strategy": {"strategy_version": 5, "recommendation_count": 3},
        },
        "evidence": {
            "search_candidate_archives": {"current": list(rows), "fixed": list(rows), "adaptive": []},
        },
        "fixed": {"current": {"as_of": "2026-09-30", "recommendation_regions": [], "diagnostics": {}}},
    }


def completion_revision(store, result_id, *, decision="current", decision_index=0, overlay_rows=()):
    """写一条补算修订：合并档案覆盖某个决策分支。"""
    stored = store.result(result_id)
    base = stored["evidence"]["search_candidate_archives"][decision]
    overlay = _write_candidate_archive(store.root, list(overlay_rows))
    merged = store.merge_candidate_archives(base, overlay)
    return store.save_completion_revision(result_id, {
        "decision": decision, "decision_index": decision_index, "status": "completed",
        "candidate_archive": overlay, "merged_candidate_archive": merged,
    })


def test_result_get_never_evaluates_or_loads_market_data(tmp_path, monkeypatch):
    app = create_app(str(tmp_path), start_jobs=False)
    result_id = app.state.store.save_result("optimization", v5_result(), "600000.SH", "v5 结果")

    def forbidden(*args, **kwargs):
        raise AssertionError("结果读取不得调用评价器或加载行情")

    monkeypatch.setattr("stockregression.data.DataService.load_bundle", forbidden)
    monkeypatch.setattr("stockregression.market_data.service.MarketDataService.load_backtest_bundle", forbidden)
    monkeypatch.setattr("stockregression.research_v4.evaluator.BatchEvaluator.__init__", forbidden)
    monkeypatch.setattr("stockregression.research_completion.complete_research_result", forbidden)

    with TestClient(app) as client:
        response = client.get(f"/api/v1/results/{result_id}")
        assert response.status_code == 200
        full = client.get(f"/api/v1/results/{result_id}", params={"full": "true"})
        assert full.status_code == 200


def test_result_get_reports_the_completion_state_without_running_it(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    result_id = app.state.store.save_result("optimization", v5_result(), "600000.SH", "v5 结果")

    with TestClient(app) as client:
        response = client.get(f"/api/v1/results/{result_id}")

    body = response.json()
    assert body["completion"]["status"] == "available"
    assert body["completion"]["latest_revision"] is None
    assert body["completion"]["endpoint"].endswith("/completion")


def test_concurrent_completion_submissions_are_deduplicated(tmp_path, monkeypatch):
    app = create_app(str(tmp_path), start_jobs=False)
    recorded = provenance()
    result_id = app.state.store.save_result(
        "optimization", v5_result(computation_provenance=recorded), "600000.SH", "v5 结果",
    )
    monkeypatch.setattr(
        "stockregression.reproducibility.computation_provenance",
        lambda *args, **kwargs: recorded["finished"],
    )
    monkeypatch.setattr(app.state.store, "snapshot", lambda *args, **kwargs: {"version": "v-1"})
    payload = {"decision": "current", "decision_index": 0, "reporting_budget": 50, "target": "return_leader"}

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(
                lambda _: client.post(f"/api/v1/results/{result_id}/completion", json=payload), range(4)
            ))

    assert all(response.status_code == 202 for response in responses)
    job_ids = {response.json()["job_id"] for response in responses}
    assert len(job_ids) == 1, "并发补算必须去重成同一个任务"

    store = Store(str(tmp_path))
    jobs = store.jobs(status="queued")
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "research_completion"


def test_completion_is_refused_when_the_result_is_not_reproducible(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    result = v5_result(computation_provenance=provenance(reproducible=False, completion_allowed=False))
    result_id = app.state.store.save_result("optimization", result, "600000.SH", "不可复现结果")

    with TestClient(app) as client:
        response = client.post(f"/api/v1/results/{result_id}/completion", json={"decision": "current"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "COMPLETION_PROVENANCE_MISMATCH"


def test_completion_is_refused_when_the_provenance_is_missing(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    result = v5_result()
    result.pop("computation_provenance")
    result_id = app.state.store.save_result("optimization", result, "600000.SH", "缺少来源的结果")

    with TestClient(app) as client:
        response = client.post(f"/api/v1/results/{result_id}/completion", json={"decision": "current"})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "COMPLETION_PROVENANCE_MISMATCH"


def test_completion_worker_refuses_to_mix_fingerprints(tmp_path):
    store = Store(str(tmp_path))
    result = v5_result(computation_provenance=provenance(source_fingerprint="stale-source"))
    result_id = store.save_result("optimization", result, "600000.SH", "来源不一致")

    with pytest.raises(ValueError, match="计算来源或依赖版本不一致"):
        complete_research_result(
            store, None, result_id, decision="current", decision_index=0, reporting_budget=10,
        )


def test_completion_worker_passes_the_provenance_gate_for_a_matching_result(tmp_path):
    store = Store(str(tmp_path))
    current = computation_provenance(
        PROJECT_ROOT, configuration=v5_result()["config"]
    )
    result = v5_result(computation_provenance=provenance(
        source_fingerprint=current["source_fingerprint"],
        dependency_lock_hash=current["dependency_lock_hash"],
        configuration_fingerprint=current["configuration_fingerprint"],
    ))
    result["computation_provenance"]["finished"] = current
    result_id = store.save_result("optimization", result, "600000.SH", "来源一致")

    # The provenance gate passes, so the run stops at the next gate instead.
    with pytest.raises(ValueError, match="行情数据版本不可用"):
        complete_research_result(
            store, None, result_id, decision="current", decision_index=0, reporting_budget=10,
        )


@pytest.mark.parametrize("field", [
    "source_fingerprint", "dependency_lock_hash", "configuration_fingerprint",
])
def test_reproducibility_mismatches_name_the_inconsistent_fingerprint(field):
    original = provenance()["finished"]
    current = {**original, field: "changed"}

    mismatches = reproducibility_mismatches(original, current)

    assert mismatches, "指纹不一致必须被拒绝"
    assert any(field in reason for reason in mismatches)


def test_unrelated_commit_metadata_does_not_invalidate_identical_computation_content():
    original = provenance()["finished"]
    current = {
        **original,
        "git_commit": "different-commit",
        "git_dirty": not bool(original.get("git_dirty")),
        "git_status_fingerprint": "different-status",
    }

    assert reproducibility_mismatches(original, current) == []


def test_configuration_fingerprint_changes_with_the_configuration():
    root = PROJECT_ROOT
    first = computation_provenance(root, configuration={"symbol": "600000.SH"})
    second = computation_provenance(root, configuration={"symbol": "600519.SH"})

    assert first["configuration_fingerprint"] != second["configuration_fingerprint"]
    assert reproducibility_mismatches(first, second)


def test_source_change_during_the_run_marks_the_result_not_reproducible(tmp_path):
    started = provenance()["finished"]
    finished = {**started, "source_fingerprint": "source-after-edit"}

    mismatches = reproducibility_mismatches(started, finished)

    assert any("source_fingerprint" in reason for reason in mismatches)
    store = Store(str(tmp_path))
    result = v5_result(computation_provenance={
        "reproducible": not mismatches, "completion_allowed": not mismatches,
        "mismatches": mismatches, "finished": finished,
    })
    result_id = store.save_result("optimization", result, "600000.SH", "运行期改码")

    listed = [row for row in store.results() if row["id"] == result_id]
    assert listed and listed[0]["reproducible"] is False

    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        response = client.post(f"/api/v1/results/{result_id}/completion", json={"decision": "current"})
    assert response.status_code == 409


def test_result_reports_the_scope_of_the_selected_revision(tmp_path):
    """补算状态必须公开修订号覆盖的决策分支。"""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    rows = [landscape_row(index) for index in range(1, 4)]
    result_id = store.save_result(
        "optimization", landscape_result(rows), "600000.SH", "修订作用域",
    )
    completion_revision(store, result_id, decision="current", overlay_rows=[rows[0]])

    with TestClient(app) as client:
        body = client.get(f"/api/v1/results/{result_id}").json()

    assert body["completion"]["selected_revision"] == 1
    assert body["completion"]["revision_scope"] == {
        "revision": 1, "decision": "current", "decision_index": 0,
    }


def test_completion_job_result_keeps_the_revision_it_produced(tmp_path):
    """A completion task must not drift to a newer revision on another task."""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    rows = [landscape_row(index) for index in range(1, 4)]
    result_id = store.save_result(
        "optimization", landscape_result(rows), "600000.SH", "v5 result",
    )
    first = completion_revision(store, result_id, overlay_rows=[rows[0]])
    second = completion_revision(store, result_id, overlay_rows=[rows[1]])
    job_id = store.new_job("research_completion", {
        "result_id": result_id, "completion_revision": first["revision"],
    })
    store.update_job(job_id, result_id=result_id)

    with TestClient(app) as client:
        response = client.get(f"/api/v1/jobs/{job_id}/result")

    assert response.status_code == 200
    completion = response.json()["completion"]
    assert completion["selected_revision"] == first["revision"]
    assert completion["latest_revision"]["revision"] == second["revision"]
    assert completion["revision_scope"] == {
        "revision": first["revision"], "decision": "current", "decision_index": 0,
    }


def test_landscape_with_an_out_of_scope_revision_is_refused(tmp_path):
    """修订只覆盖自己的决策分支，别的分支必须明确失败而不是显示补算前的数据。"""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    rows = [landscape_row(index) for index in range(1, 4)]
    result_id = store.save_result(
        "optimization", landscape_result(rows), "600000.SH", "修订作用域",
    )
    completion_revision(store, result_id, decision="current", overlay_rows=[rows[0]])

    with TestClient(app) as client:
        out_of_scope = client.get(
            f"/api/v1/results/{result_id}/landscape?decision=fixed&revision=1"
        )
        in_scope = client.get(
            f"/api/v1/results/{result_id}/landscape?decision=current&revision=1"
        )

    assert out_of_scope.status_code == 409
    detail = out_of_scope.json()["detail"]
    assert detail["code"] == "COMPLETION_REVISION_OUT_OF_SCOPE"
    assert detail["scope"] == {"decision": "current", "decision_index": 0}

    assert in_scope.status_code == 200
    payload = in_scope.json()
    assert payload["revision"] == 1
    assert payload["revision_scope"] == {
        "revision": 1, "decision": "current", "decision_index": 0,
    }


def test_landscape_reads_the_merged_archive_of_the_revised_branch(tmp_path):
    """带修订的参数分布必须读合并档案，而不是补算前的档案。"""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    rows = [landscape_row(index) for index in range(1, 4)]
    result_id = store.save_result(
        "optimization", landscape_result(rows), "600000.SH", "修订参数分布",
    )
    overlay = [{
        key: value for key, value in {**landscape_row(2, excess=.9), "j_score": .5}.items()
        if key != "candidate_id"
    }]
    completion_revision(store, result_id, decision="current", overlay_rows=overlay)

    with TestClient(app) as client:
        base = client.get(f"/api/v1/results/{result_id}/landscape?decision=current").json()
        revised = client.get(
            f"/api/v1/results/{result_id}/landscape?decision=current&revision=1"
        ).json()

    assert base["focus"]["candidate_id"] == 3
    assert revised["focus"]["candidate_id"] == 2, "修订后的焦点必须来自合并档案"


def test_completion_requests_with_different_budgets_are_not_deduplicated(tmp_path, monkeypatch):
    """不同的展示预算不是同一个补算任务，不能被去重成同一个后台任务。"""
    app = create_app(str(tmp_path), start_jobs=False)
    recorded = provenance()
    result_id = app.state.store.save_result(
        "optimization", v5_result(computation_provenance=recorded), "600000.SH", "v5 结果",
    )
    monkeypatch.setattr(
        "stockregression.reproducibility.computation_provenance",
        lambda *args, **kwargs: recorded["finished"],
    )
    monkeypatch.setattr(app.state.store, "snapshot", lambda *args, **kwargs: {"version": "v-1"})

    with TestClient(app) as client:
        first = client.post(
            f"/api/v1/results/{result_id}/completion",
            json={"decision": "current", "decision_index": 0, "reporting_budget": 50},
        )
        second = client.post(
            f"/api/v1/results/{result_id}/completion",
            json={"decision": "current", "decision_index": 0, "reporting_budget": 80},
        )

    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["job_id"] != second.json()["job_id"], "预算不同必须是不同任务"


def test_fresh_completion_starts_a_new_checkpoint_while_resume_reuses_the_original(tmp_path, monkeypatch):
    """终态之后的新提交必须用新的检查点，只有显式恢复才复用原检查点。"""
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    recorded = provenance()
    result_id = store.save_result(
        "optimization", v5_result(computation_provenance=recorded), "600000.SH", "v5 结果",
    )
    monkeypatch.setattr(
        "stockregression.reproducibility.computation_provenance",
        lambda *args, **kwargs: recorded["finished"],
    )
    monkeypatch.setattr(store, "snapshot", lambda *args, **kwargs: {"version": "v-1"})
    payload = {"decision": "current", "decision_index": 0, "reporting_budget": 50}

    with TestClient(app) as client:
        first = client.post(f"/api/v1/results/{result_id}/completion", json=payload).json()
        first_job = first["job_id"]
        store.update_job(first_job, status="failed")
        second = client.post(f"/api/v1/results/{result_id}/completion", json=payload).json()
        second_job = second["job_id"]
        resumed = client.post(f"/api/v1/jobs/{first_job}/resume").json()
        resumed_job = resumed["job_id"]

    assert first_job != second_job
    assert store.job(first_job)["payload"]["checkpoint_id"] == first_job
    assert store.job(second_job)["payload"]["checkpoint_id"] == second_job
    assert store.job(resumed_job)["payload"]["checkpoint_id"] == first_job
    assert store.job(resumed_job)["payload"]["resume_from_job"] == first_job


def test_resume_is_not_blocked_by_a_stale_configuration_revision(tmp_path):
    """恢复任务重放冻结快照，不能被当前配置版本漂移挡住。"""
    store = Store(str(tmp_path))
    job_id = store.new_job("research_completion", {
        "result_id": "missing", "decision": "current", "decision_index": 0,
        "reporting_budget": 10, "target": "return_leader",
        "configuration_revision": "stale-revision",
    })
    store.update_job(job_id, status="failed")

    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        stale = client.post("/api/v1/data/sync", json={
            "scope": "watchlist", "mode": "incremental",
            "configuration_revision": "stale-revision",
        })
        resumed = client.post(f"/api/v1/jobs/{job_id}/resume")

    assert stale.status_code == 409, "新的提交仍然必须检查配置版本"
    assert stale.json()["detail"]["code"] == "CONFIG_REVISION_CONFLICT"
    assert resumed.status_code == 202, resumed.text


def test_completion_checkpoint_merges_concurrent_writes(tmp_path):
    """并发写检查点时，已消耗预算与已处理点不得被后来的写入抹掉。"""
    store = Store(str(tmp_path))
    store.save_completion_checkpoint("checkpoint", {
        "consumed": 5, "processed": [[1, 2]], "in_flight": [[3, 4]],
        "status": "running", "new_rows": [{"params": {"buy_step": .01}}],
    })
    store.save_completion_checkpoint("checkpoint", {
        "consumed": 2, "processed": [[5, 6]], "in_flight": [],
        "status": "running", "new_rows": [{"params": {"buy_step": .02}}],
    })

    checkpoint = store.completion_checkpoint("checkpoint")
    assert checkpoint["consumed"] == 5, "已消耗预算不得被较小的写入覆盖"
    assert {tuple(point) for point in checkpoint["processed"]} == {(1, 2), (5, 6)}
    assert {tuple(point) for point in checkpoint["in_flight"]} == {(3, 4)}
    assert len(checkpoint["new_rows"]) == 2


def test_concurrent_checkpoint_writers_keep_the_highest_consumption(tmp_path):
    """多个线程同时写同一个检查点时，消耗与已处理点必须合并而不是互相覆盖。"""
    store = Store(str(tmp_path))

    def write(index: int) -> None:
        store.save_completion_checkpoint("checkpoint", {
            "consumed": index, "processed": [[index]], "in_flight": [],
            "status": "running", "new_rows": [{"params": {"index": index}}],
        })

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(1, 9)))

    checkpoint = store.completion_checkpoint("checkpoint")
    assert checkpoint["consumed"] == 8
    assert {tuple(point) for point in checkpoint["processed"]} == {(index,) for index in range(1, 9)}
    assert len(checkpoint["new_rows"]) == 8


def test_checkpoint_identity_separates_fresh_runs_from_resumes(tmp_path):
    """检查点键必须区分不同的补算运行，恢复才复用原检查点。"""
    from stockregression import research_completion

    current = computation_provenance(PROJECT_ROOT, configuration={"symbol": "600000.SH"})
    first = research_completion._checkpoint_key(
        result_id="r-1", data_version="v-1", decision="current", decision_index=0,
        provenance=current, checkpoint_id="job-1",
    )
    second = research_completion._checkpoint_key(
        result_id="r-1", data_version="v-1", decision="current", decision_index=0,
        provenance=current, checkpoint_id="job-2",
    )
    resumed = research_completion._checkpoint_key(
        result_id="r-1", data_version="v-1", decision="current", decision_index=0,
        provenance=current, checkpoint_id="job-1",
    )
    assert first != second, "新的补算运行必须使用新的检查点"
    assert first == resumed, "恢复必须复用原检查点"


def test_candidate_reference_raises_for_a_missing_revision(tmp_path):
    """引用不存在的修订必须明确失败，不能退回基础档案。"""
    store = Store(str(tmp_path))
    result_id = store.save_result(
        "optimization", landscape_result([landscape_row(1)]), "600000.SH", "缺失修订",
    )
    stored = store.result(result_id)
    with pytest.raises(CompletionRevisionError) as failure:
        store.candidate_reference(stored, "search_candidate_archives", "current", 0, revision=3)
    assert failure.value.code == "COMPLETION_REVISION_NOT_FOUND"
