"""v5 研究预算回归测试：阶段预算、缓存计费与总预算上限。

契约要点：

* ``training`` / ``validation`` / ``reporting`` 三个阶段各有独立上限；
* 命中缓存的点不得重复计费，也不得因此把任务标记为预算耗尽；
* ``reporting_budget`` 只限制展示补算，必须通过 ``reporting_stopped`` 单独
  汇报，不得冒充任务总预算耗尽；
* 任务总预算在任何阶段都不得被突破；
* 费用压力评价与回放评价同样计费。
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockregression.models import (
    BacktestConfig, DataBundle, OptimizationConfig, ResearchStrategySpec,
)
from stockregression.research import robust as robust_research
from stockregression import research_v5
from stockregression import research_completion
from stockregression.research.robust import EvaluationBudget, EvaluationSession, _complete_reporting_scores
from stockregression.research.space import ROBUST_FIELDS, ParameterSpace


def make_config(**strategy_overrides) -> OptimizationConfig:
    base = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date="2023-01-02", frequency="1d",
        initial_shares=1000, min_shares=0, max_shares=None, buy_shares=100, sell_shares=100,
        grid_mode="geometric", buy_step=0.01, sell_step=0.01,
    )
    strategy = {
        "strategy_version": 5, "search_method": "hierarchical", "ranking_mode": "robust_j",
        "granularity": {"buy_step": 0.005, "sell_step": 0.005, "buy_shares": 100, "sell_shares": 100},
        "hypercube_edge": 3, "training_budget": 1, "validation_budget": 3, "reporting_budget": 2,
        "recommendation_count": 1, "stress_region_count": 1, "stress_cost_multiplier": 2.0,
        "starts_per_candidate": 1, "start_min": 1, "start_max": 1,
        "training_min_days": 250, "training_max_days": 500,
        **strategy_overrides,
    }
    return OptimizationConfig(
        base=base, protocol_version=5,
        strategy=ResearchStrategySpec.model_validate(strategy),
        ranges={
            "buy_step": {"low": 0.005, "high": 0.015, "step": 0.005},
            "sell_step": {"low": 0.005, "high": 0.005, "step": 0.005},
            "buy_shares": {"low": 100, "high": 100, "step": 100},
            "sell_shares": {"low": 100, "high": 100, "step": 100},
        },
    )


class StubEvaluator:
    """Deterministic evaluator with a strictly monotonic return ladder."""

    chunk_size = 4

    def __init__(self, space: ParameterSpace):
        self.space = space
        self.calls: list[list[tuple[int, ...]]] = []
        self.stress_calls: list[tuple[list[tuple[int, ...]], float]] = []

    def _value(self, coordinate: tuple[int, ...]) -> float:
        return 0.001 * sum(int(item) for item in coordinate)

    def _row(self, coordinate: tuple[int, ...]) -> dict:
        return {
            "key": coordinate, "coordinate": list(coordinate),
            "params": self.space.params(coordinate),
            "evaluation_status": "completed",
            "training_excess_return": self._value(coordinate),
            "max_principal_drawdown": 0.05, "trade_count": 20,
            "buy_count": 10, "sell_count": 10,
            "actual_max_pos_ratio": 1.0, "actual_min_pos_ratio": 0.5,
            "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
        }

    def evaluate_training(self, chunk):
        self.calls.append([tuple(item) for item in chunk])
        return [self._row(tuple(item)) for item in chunk]

    def evaluate_training_stress(self, points, cost_multiplier=1.0):
        listed = [tuple(item) for item in points]
        self.stress_calls.append((listed, float(cost_multiplier)))
        return {
            point: {"training_excess_stress": self._value(point) - 0.001 * float(cost_multiplier)}
            for point in listed
        }


def make_session(**strategy_overrides):
    total = strategy_overrides.pop("total", None)
    config = make_config(**strategy_overrides)
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    evaluator = StubEvaluator(space)
    budget = EvaluationBudget(total=total, max_runtime_seconds=600) if total is not None else None
    session = EvaluationSession(evaluator, space, config, None, budget=budget)
    return session, evaluator, space


def test_training_phase_never_spends_more_than_its_own_budget():
    session, evaluator, space = make_session()
    coordinates = list(space.iterate())
    assert len(coordinates) == 3

    evaluated = session.ensure(coordinates, "training", 1)

    assert evaluated == 1
    assert session.spent["training"] == 1
    assert sum(len(call) for call in evaluator.calls) == 1
    assert session.budget_stopped is True
    assert "training" in (session.budget_reason or "")


def test_cached_coordinates_are_not_charged_twice_or_reported_as_budget_exhaustion():
    session, evaluator, space = make_session()
    coordinates = list(space.iterate())

    session.ensure(coordinates, "training", 5)
    assert session.spent["training"] == 3
    calls = len(evaluator.calls)

    # The very same coordinates are requested again with a fresh phase cap.
    evaluated = session.ensure(coordinates, "training", 10)

    assert evaluated == 0
    assert session.spent["training"] == 3, "缓存命中不得重复计费"
    assert len(evaluator.calls) == calls, "缓存命中不得重复评价"
    assert session.budget_stopped is False, "缓存命中不是预算耗尽"
    assert session.budget_reason is None


def test_reporting_budget_caps_the_reporting_phase_without_exceeding_the_total_budget():
    session, evaluator, space = make_session(reporting_budget=1)
    budget = EvaluationBudget(total=10, max_runtime_seconds=600)
    session.task_budget = budget
    centre = space.full_coordinate(tuple(space.dimensions[name].low for name in space.active))

    session.ensure([centre], "training", 5)
    spent_before = sum(session.spent.values())

    _complete_reporting_scores(session)

    assert session.spent["reporting"] <= 1, "展示补算不得越过 reporting_budget"
    assert sum(session.spent.values()) - spent_before <= 1
    assert budget.consumed <= budget.total, "任务总预算不得被突破"
    assert session.spent["validation"] == 0, "展示补算不得借用邻域预算"


def test_reporting_limit_is_reported_separately_from_the_task_budget():
    session, evaluator, space = make_session(reporting_budget=1)
    session.task_budget = EvaluationBudget(total=10, max_runtime_seconds=600)
    centre = space.full_coordinate(tuple(space.dimensions[name].low for name in space.active))
    session.ensure([centre], "training", 5)

    _complete_reporting_scores(session)

    assert session.spent["reporting"] == 1
    assert session.reporting_stopped is True, "展示预算用尽必须用 reporting_stopped 汇报"
    assert "reporting_budget" in (session.reporting_reason or "")
    assert session.budget_stopped is False, "展示预算独立限制，不得冒充任务总预算耗尽"
    assert session.budget_reason is None
    assert session.task_budget.stopped is False, "展示预算上限不得停止任务总预算"


def test_cache_hits_do_not_count_points_truncated_by_a_phase_cap():
    """被阶段上限截断的点既不是缓存命中，也不是已评价。"""
    session, evaluator, space = make_session()
    coordinates = list(space.iterate())

    session.ensure(coordinates, "training", 1)

    assert session.spent["training"] == 1
    assert session.attempts == 1
    assert session.requested == 3
    assert session.skipped == 2
    assert session.cache_hits == 0, "截断不得被报告成缓存命中"

    # 只有已评价的点再次被请求才是缓存命中。
    session.ensure(coordinates[:1], "training", 5)

    assert session.requested == 4
    assert session.attempts == 1
    assert session.skipped == 2
    assert session.cache_hits == 1


def test_memory_budget_stops_reservations_when_rss_exceeds_the_workspace(monkeypatch):
    """进程 RSS 超过工作区上限时必须停止任务预算，且不再评价任何点。"""
    session, evaluator, space = make_session()
    budget = EvaluationBudget(total=10, max_runtime_seconds=600, workspace_bytes=1024)
    monkeypatch.setattr(robust_research, "current_process_rss_bytes", lambda: 4096)
    session.task_budget = budget

    assert budget.reserve(1, "训练") is False
    assert budget.stopped is True
    assert budget.consumed == 0
    assert "内存" in (budget.reason or ""), "内存超限必须有独立的停止原因"

    session.ensure(list(space.iterate()), "training", 5)
    assert session.spent["training"] == 0
    assert evaluator.calls == [], "预算停止后不得继续评价"


def test_stress_budget_overflow_stops_later_phases(monkeypatch):
    """费用压力批次放不下时必须停止任务预算，后续 current/OOS 阶段不能再花钱。"""
    config = make_config(total_evaluation_budget=3)
    base = config.base
    dates = pd.bdate_range("2020-01-02", periods=600)
    close = [10.0] * len(dates)
    daily = pd.DataFrame({
        "date": dates, "open": close, "high": [value + 0.5 for value in close],
        "low": [value - 0.5 for value in close], "close": close,
        "preclose": [close[0]] + close[:-1], "volume": 100000,
        "turnover": [value * 100000 for value in close], "tradestatus": 1, "is_st": False,
    })
    data = DataBundle(symbol="600000.SH", bars=pd.DataFrame(), daily=daily, calendar=pd.DatetimeIndex(dates))
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    evaluator = StubEvaluator(space)
    monkeypatch.setattr(robust_research, "BatchEvaluator", lambda *args, **kwargs: evaluator)
    budget = EvaluationBudget(total=3, max_runtime_seconds=600)

    result = robust_research.select_robust_at(config, base, data, dates[-1], budget=budget)

    assert evaluator.stress_calls == [], "放不下的费用压力批次不得执行"
    assert budget.stopped is True, "费用压力放不下必须停止任务总预算"
    assert "费用压力" in (budget.reason or "")
    assert result["diagnostics"]["budget_stopped"] is True
    assert result["recommendation_regions"][0]["stress"]["status"] == "budget_exhausted"
    assert budget.reserve(1, "当前参数") is False, "后续阶段不得继续消耗预算"
    assert budget.consumed == 3, "放不下的批次不得预先消耗预算"
    assert budget.consumed <= budget.total


def test_task_budget_stops_the_run_and_reports_the_total_limit():
    session, evaluator, space = make_session()
    budget = EvaluationBudget(total=2, max_runtime_seconds=600)
    session.task_budget = budget
    coordinates = list(space.iterate())

    session.ensure(coordinates, "training", 5)

    assert budget.consumed == 2
    assert budget.remaining == 0
    assert budget.stopped is True, "任务总预算用尽必须记录停止状态"
    assert "任务总" in (budget.reason or "")
    assert session.budget_stopped is True
    assert "任务总" in (session.budget_reason or ""), (
        "截断阶段的是任务总预算时，停止原因必须写明总预算，而不是阶段上限"
    )
    assert session.spent["training"] <= 2
    assert session.ledger_evaluations == 2 * budget.consumed


def test_stress_evaluations_are_charged_against_the_remaining_budget():
    session, evaluator, space = make_session()
    remaining = session._budget_remaining()

    session.stress_spent += 4

    assert session._budget_remaining() == remaining - 4
    assert session.ledger_evaluations == 2 * (sum(session.spent.values()) + session.stress_spent)


def test_end_to_end_selection_respects_every_budget_and_charges_the_stress_run(monkeypatch):
    config = make_config(total_evaluation_budget=20)
    base = config.base
    dates = pd.bdate_range("2020-01-02", periods=600)
    close = [10.0] * len(dates)
    daily = pd.DataFrame({
        "date": dates, "open": close, "high": [value + 0.5 for value in close],
        "low": [value - 0.5 for value in close], "close": close,
        "preclose": [close[0]] + close[:-1], "volume": 100000, "turnover": [value * 100000 for value in close],
        "tradestatus": 1, "is_st": False,
    })
    data = DataBundle(symbol="600000.SH", bars=pd.DataFrame(), daily=daily, calendar=pd.DatetimeIndex(dates))

    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    evaluator = StubEvaluator(space)
    monkeypatch.setattr(robust_research, "BatchEvaluator", lambda *args, **kwargs: evaluator)

    budget = EvaluationBudget(total=20, max_runtime_seconds=600)
    result = robust_research.select_robust_at(
        config, base, data, dates[-1], budget=budget,
    )
    diagnostics = result["diagnostics"]

    assert diagnostics["training_unique_count"] <= config.strategy.training_budget
    assert diagnostics["neighborhood_unique_count"] <= config.strategy.validation_budget
    assert diagnostics["reporting_unique_count"] <= config.strategy.reporting_budget
    assert diagnostics["stress_unique_count"] == len(evaluator.stress_calls[0][0]) if evaluator.stress_calls else True
    assert evaluator.stress_calls, "费用压力评价必须真正执行"
    assert evaluator.stress_calls[0][1] == pytest.approx(config.strategy.stress_cost_multiplier)

    total = (
        diagnostics["training_unique_count"] + diagnostics["neighborhood_unique_count"]
        + diagnostics["reporting_unique_count"] + diagnostics["stress_unique_count"]
    )
    assert total <= budget.total, "任务总预算不得被突破"
    assert diagnostics["ledger_evaluation_count"] == 2 * total


FIXED_PROVENANCE = {
    "schema_version": 1, "engine_version": "0.1.0", "git_commit": "abc", "git_dirty": False,
    "git_status_fingerprint": "status", "uncommitted_diff_fingerprint": "diff",
    "source_fingerprint": "src-fixed", "source_file_count": 1,
    "dependency_lock_file": "uv.lock", "dependency_lock_hash": "lock-fixed",
    "configuration_fingerprint": "cfg-fixed",
    "runtime": {"python": "3.12", "implementation": "CPython", "platform": "win32", "packages": {}},
}


def completion_rows(space: ParameterSpace, coordinates) -> list[dict]:
    rows = []
    for index, coordinate in enumerate(coordinates):
        rows.append({
            "candidate_id": index + 1, "key": list(coordinate), "coordinate": list(coordinate),
            "params": space.params(coordinate), "evaluation_status": "completed",
            "training_excess_return": 0.01 * (index + 1),
            "max_principal_drawdown": 0.05, "trade_count": 20, "buy_count": 10, "sell_count": 10,
            "actual_max_pos_ratio": 1.0, "actual_min_pos_ratio": 0.5,
            "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
        })
    return rows


class CompletionEvaluator:
    """Stand-in for the batch evaluator used by the completion worker."""

    chunk_size = 4

    def __init__(self, space: ParameterSpace):
        self.space = space
        self.chunks: list[list[tuple[int, ...]]] = []

    def evaluate_training(self, chunk):
        self.chunks.append([tuple(item) for item in chunk])
        return completion_rows(self.space, [tuple(item) for item in chunk])


class InterruptingCompletionEvaluator(CompletionEvaluator):
    def evaluate_training(self, chunk):
        self.chunks.append([tuple(item) for item in chunk])
        raise InterruptedError("completion cancelled")


class FailingCompletionEvaluator(CompletionEvaluator):
    def evaluate_training(self, chunk):
        self.chunks.append([tuple(item) for item in chunk])
        raise ValueError("synthetic evaluation failure")


class IncompleteCompletionEvaluator(CompletionEvaluator):
    def evaluate_training(self, chunk):
        self.chunks.append([tuple(item) for item in chunk])
        return []


def make_completion_result(config, space: ParameterSpace, coordinates) -> dict:
    payload = {
        "config": config.model_dump(mode="json"),
        "protocol_version": 5, "result_schema_version": 2,
        "symbol": "600000.SH", "data_version": "v-1",
        "computation_provenance": {
            "reproducible": True, "completion_allowed": True, "mismatches": [],
            "finished": FIXED_PROVENANCE, "started": FIXED_PROVENANCE,
        },
        "evidence": {"search_candidate_archives": {"current": completion_rows(space, coordinates)}},
        "fixed": {"current": {"diagnostics": {"training_start": "2023-01-02", "training_end": "2023-06-30"}}},
    }
    return payload


class CompletionService:
    """Stub data service returning a bundle that reproduces the frozen window."""

    def load_bundle(self, config, version=None):
        dates = pd.bdate_range("2023-01-02", "2023-06-30")
        daily = pd.DataFrame({
            "date": dates, "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
            "preclose": 10.0, "volume": 100000, "tradestatus": 1, "is_st": False,
        })
        return DataBundle(
            symbol="600000.SH", bars=pd.DataFrame(), daily=daily,
            calendar=pd.DatetimeIndex(dates), version="v-1",
        )


def prepare_completion(tmp_path, monkeypatch, *, consumed: int, processed: list[tuple[int, ...]],
                        archived: list[tuple[int, ...]] | None = None):
    """Store a result plus a resumable completion checkpoint."""
    from stockregression.storage import Store

    config = make_config()
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    coordinates = list(space.iterate())
    store = Store(str(tmp_path))
    store.put_market_snapshot({"version": "v-1", "files": {}}, "600000.SH")
    payload = make_completion_result(config, space, archived if archived is not None else coordinates)
    result_id = store.save_result("optimization", payload, "600000.SH", "补算结果")

    monkeypatch.setattr(research_completion, "computation_provenance", lambda *args, **kwargs: FIXED_PROVENANCE)
    key = research_completion._checkpoint_key(
        result_id=result_id, data_version="v-1", decision="current",
        decision_index=0, provenance=FIXED_PROVENANCE,
    )
    store.save_completion_checkpoint(key, {
        "result_id": result_id, "decision": "current", "decision_index": 0, "target": "return_leader",
        "data_version": "v-1", "targets": [list(coordinates[1])],
        "consumed": consumed, "processed": [list(point) for point in processed],
        "new_rows": completion_rows(space, processed), "status": "running", "stop_reason": None,
        "original_archive_count": len(coordinates), "required_count": len(coordinates),
    })
    return store, result_id, key, space, coordinates


def test_resumed_completion_keeps_the_restored_budget_and_skips_finished_points(tmp_path, monkeypatch):
    config = make_config()
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    coordinates = list(space.iterate())
    processed = coordinates[:2]
    store, result_id, key, space, coordinates = prepare_completion(
        tmp_path, monkeypatch, consumed=2, processed=processed, archived=[coordinates[1]],
    )
    evaluator = CompletionEvaluator(space)
    monkeypatch.setattr(research_completion, "BatchEvaluator", lambda *args, **kwargs: evaluator)

    outcome = research_completion.complete_research_result(
        store, CompletionService(), result_id, decision="current", decision_index=0, reporting_budget=10,
    )

    assert outcome["status"] in {"completed", "partial"}
    asked = [point for chunk in evaluator.chunks for point in chunk]
    assert asked, "未完成的候选必须继续评价"
    assert not set(asked) & set(processed), "已完成的点不得重复评价"
    checkpoint = store.completion_checkpoint(key)
    assert checkpoint["consumed"] == 2 + len(asked), "恢复后只对新增评价计费"


def test_resumed_completion_refuses_a_budget_below_the_already_consumed_count(tmp_path, monkeypatch):
    config = make_config()
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    coordinates = list(space.iterate())
    store, result_id, key, space, coordinates = prepare_completion(
        tmp_path, monkeypatch, consumed=5, processed=coordinates[:2], archived=[coordinates[1]],
    )

    with pytest.raises(ValueError, match="恢复后的补算预算低于已消耗预算"):
        research_completion.complete_research_result(
            store, CompletionService(), result_id, decision="current", decision_index=0, reporting_budget=3,
        )


def test_cancelled_completion_persists_the_reserved_batch_and_does_not_recharge_it(
    tmp_path, monkeypatch,
):
    config = make_config()
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    coordinates = list(space.iterate())
    store, result_id, key, space, coordinates = prepare_completion(
        tmp_path, monkeypatch, consumed=2, processed=coordinates[:2],
        archived=[coordinates[1]],
    )
    failing = InterruptingCompletionEvaluator(space)
    monkeypatch.setattr(research_completion, "BatchEvaluator", lambda *args, **kwargs: failing)

    with pytest.raises(InterruptedError, match="completion cancelled"):
        research_completion.complete_research_result(
            store, CompletionService(), result_id, decision="current", decision_index=0,
            reporting_budget=10,
        )

    first_batch = tuple(failing.chunks[0])
    checkpoint = store.completion_checkpoint(key)
    assert checkpoint["consumed"] == 2 + len(first_batch)
    assert set(first_batch).issubset({tuple(point) for point in checkpoint["processed"]})
    assert checkpoint["in_flight"] == []

    resumed = CompletionEvaluator(space)
    monkeypatch.setattr(research_completion, "BatchEvaluator", lambda *args, **kwargs: resumed)
    research_completion.complete_research_result(
        store, CompletionService(), result_id, decision="current", decision_index=0,
        reporting_budget=10,
    )
    asked = {point for chunk in resumed.chunks for point in chunk}
    assert not asked.intersection(first_batch)
    assert store.completion_checkpoint(key)["consumed"] == checkpoint["consumed"] + len(asked)


def test_cancellation_carries_the_published_revision_for_the_job(tmp_path, monkeypatch):
    """A cancelled completion job still points at the revision it published."""
    config = make_config()
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    coordinates = list(space.iterate())
    store, result_id, key, space, coordinates = prepare_completion(
        tmp_path, monkeypatch, consumed=2, processed=coordinates[:2],
        archived=[coordinates[1]],
    )
    monkeypatch.setattr(
        research_completion, "BatchEvaluator", lambda *args, **kwargs: CompletionEvaluator(space),
    )

    with pytest.raises(InterruptedError) as caught:
        research_completion.complete_research_result(
            store, CompletionService(), result_id, decision="current", decision_index=0,
            reporting_budget=10, cancelled=lambda: True,
        )

    revisions = store.completion_revisions(result_id)
    assert revisions
    assert caught.value.completion_revision == revisions[-1]["revision"]
    assert revisions[-1]["status"] == "cancelled"


def test_partial_completion_revision_records_the_final_partial_status(tmp_path, monkeypatch):
    config = make_config()
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    coordinates = list(space.iterate())
    store, result_id, _key, space, coordinates = prepare_completion(
        tmp_path, monkeypatch, consumed=0, processed=[],
        archived=[coordinates[1]],
    )
    failing = FailingCompletionEvaluator(space)
    monkeypatch.setattr(research_completion, "BatchEvaluator", lambda *args, **kwargs: failing)

    outcome = research_completion.complete_research_result(
        store, CompletionService(), result_id, decision="current", decision_index=0,
        reporting_budget=10,
    )

    assert outcome["status"] == "partial"
    revision = store.completion_revision(result_id, outcome["completion_revision"])
    assert revision["status"] == "failed"
    assert revision["remaining_count"] > 0
    assert "synthetic evaluation failure" in (revision["stop_reason"] or "")


def test_revision_is_downgraded_before_an_incomplete_result_is_persisted(tmp_path, monkeypatch):
    config = make_config()
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    coordinates = list(space.iterate())
    store, result_id, _key, space, coordinates = prepare_completion(
        tmp_path, monkeypatch, consumed=0, processed=[],
        archived=[coordinates[1]],
    )
    incomplete = IncompleteCompletionEvaluator(space)
    monkeypatch.setattr(research_completion, "BatchEvaluator", lambda *args, **kwargs: incomplete)

    outcome = research_completion.complete_research_result(
        store, CompletionService(), result_id, decision="current", decision_index=0,
        reporting_budget=10,
    )

    assert outcome["status"] == "partial"
    revision = store.completion_revision(result_id, outcome["completion_revision"])
    assert revision["status"] == "partial"
    assert revision["remaining_count"] > 0


def test_task_budget_reserve_accounts_for_the_restored_consumed_count():
    budget = EvaluationBudget(total=10, max_runtime_seconds=600)
    budget.consumed = 6

    assert budget.remaining == 4
    assert budget.reserve(4, "展示补算") is True
    assert budget.consumed == 10
    assert budget.reserve(1, "展示补算") is False
    assert budget.stopped is True
    assert "任务总" in (budget.reason or "")


def test_submission_budget_contract_rejects_a_different_frozen_limit(monkeypatch):
    config = make_config(total_evaluation_budget=100)
    dates = pd.DatetimeIndex([pd.Timestamp("2023-01-02")])
    daily = pd.DataFrame({
        "date": dates, "open": [10.0], "high": [10.5], "low": [9.5],
        "close": [10.0], "preclose": [10.0], "volume": [100000],
    })
    data = DataBundle(
        symbol=config.base.symbol, bars=pd.DataFrame(), daily=daily,
        calendar=dates, version="v-1",
    )
    monkeypatch.setattr(
        research_v5, "estimate_robust_search",
        lambda *args, **kwargs: {
            "effective_total_evaluation_budget": 100,
            "task_evaluation_upper_bound": 100,
            "estimated_task_memory_mb": 8.0,
            "workspace_budget_mb": 1024.0,
        },
    )

    with pytest.raises(ValueError, match="冻结的任务评价上限"):
        research_v5.optimize_v5(
            config, data, budget_contract={
                "data_version": "v-1", "effective_total_evaluation_budget": 99,
                "max_runtime_seconds": config.strategy.max_runtime_seconds,
                "workspace_budget_mb": 1024.0,
            },
        )
