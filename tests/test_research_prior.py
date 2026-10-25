"""Protocol-v5 research-prior provenance and injection contracts."""
from __future__ import annotations

from copy import deepcopy
import math

import pandas as pd
import pytest

from stockregression.models import (
    BacktestConfig, DataBundle, OptimizationConfig, ResearchRequestV5, ResearchStrategySpec,
)
from stockregression.research import robust as robust_research
from stockregression.research_origin import prior_contract_hash
from stockregression.research.robust import (
    _attach_prior_sources, _build_research_prior, _initial_coordinates, _public_row,
    estimate_robust_search, select_robust_at,
)
from stockregression.research.search import coarse_coordinates
from stockregression.research.space import ParameterSpace
from stockregression.research.statistics import market_statistical_profile


def _config(
    *, prior_policy: str = "off", source_as_of: str | None = None,
    seeds: list[dict] | None = None, training_budget: int = 40,
    search_method: str = "hierarchical", hypercube_edge: int = 3,
    ranges: dict | None = None,
) -> OptimizationConfig:
    base = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date="2024-01-02", frequency="1d",
        initial_shares=1000, min_shares=0, max_shares=None,
        buy_shares=100, sell_shares=100, grid_mode="geometric",
        buy_step=.03, sell_step=.03,
    )
    strategy = ResearchStrategySpec(
        strategy_version=5, search_method=search_method,
        granularity={"buy_step": .005, "sell_step": .005, "buy_shares": 100, "sell_shares": 100},
        hypercube_edge=hypercube_edge, training_budget=training_budget,
        validation_budget=400, reporting_budget=20, recommendation_count=1,
        stress_region_count=0, training_min_days=20, training_max_days=30,
    )
    origin = {
        "etf_result_id": "etf-result-1", "symbol": "510300.SH", "method_version": "daily_state_v1",
        "source_as_of": source_as_of, "source_data_version": "etf-data-v1",
        "source_hash": "source-hash", "seeds": deepcopy(seeds or []),
    }
    return OptimizationConfig(
        base=base, protocol_version=5, strategy=strategy,
        ranges=ranges or {
            "buy_step": {"low": .005, "high": .05},
            "sell_step": {"low": .005, "high": .05},
            "buy_shares": {"low": 100, "high": 500},
            "sell_shares": {"low": 100, "high": 300},
        },
        prior_policy=prior_policy, source_etf={"result_id": "etf-result-1", "candidate_ids": [1, 2]},
        requested_data_version="research-data-v1", research_origin=origin,
    )


def _seed(candidate_id: int, *, buy_step: float, sell_step: float = .02,
          buy_shares: int = 200, sell_shares: int = 200) -> dict:
    return {
        "candidate_id": candidate_id,
        "params": {
            "buy_step": buy_step, "sell_step": sell_step,
            "buy_shares": buy_shares, "sell_shares": sell_shares,
        },
    }


def _patch_market_state(monkeypatch, candidates: list[dict], observed: list | None = None):
    def state(data, as_of):
        if observed is not None:
            observed.append((data, pd.Timestamp(as_of)))
        return {
            "status": "completed", "as_of": pd.Timestamp(as_of).date().isoformat(),
            "metrics": {"natr14": .02, "hv_ratio": 1.1},
        }

    monkeypatch.setattr(robust_research, "_market_state_for_prior", state)
    monkeypatch.setattr(robust_research, "_state_candidates_for_prior", lambda value: deepcopy(candidates))


def _history_data(days: int = 600) -> DataBundle:
    dates = pd.bdate_range("2020-01-02", periods=days)
    daily = pd.DataFrame({
        "date": dates, "open": 20.0, "high": 20.5, "low": 19.5,
        "close": 20.0, "preclose": 20.0, "volume": 100000,
        "turnover": 2_000_000.0, "tradestatus": 1, "is_st": False,
    })
    return DataBundle(
        symbol="600000.SH", bars=pd.DataFrame(), daily=daily, calendar=pd.DatetimeIndex(dates),
        quality={"minute_ratio": 1.0}, version="prior-test-data",
    )


def _row(space: ParameterSpace, coordinate: tuple[int, ...], value: float) -> dict:
    return {
        "key": coordinate, "coordinate": list(coordinate), "params": space.params(coordinate),
        "evaluation_status": "completed", "training_excess_return": float(value),
        "training_return": float(value), "training_baseline_return": 0.0,
        "training_baseline_kind": "matched_funding_hold",
        "max_principal_drawdown": .05, "trade_count": 20,
        "buy_count": 10, "sell_count": 10,
        "actual_max_pos_ratio": 1.0, "actual_min_pos_ratio": .5,
        "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
    }


class _FakeEvaluator:
    chunk_size = 1000

    def __init__(self, space: ParameterSpace):
        self.space = space
        self.calls: list[list[tuple[int, ...]]] = []

    def evaluate_training(self, coordinates):
        points = [tuple(item) for item in coordinates]
        self.calls.append(points)
        return [_row(self.space, point, .02 + .002 * point[0]) for point in points]

    def evaluate_training_stress(self, coordinates, cost_multiplier=2.0):
        return {
            tuple(point): {"training_excess_stress": .01, "max_principal_drawdown": .05,
                           "trade_count": 20, "actual_min_pos_ratio": .5,
                           "actual_max_pos_ratio": 1.0, "floor_episode_count": 0,
                           "floor_episode_grids": [], "floor_penalty": 0.0}
            for point in coordinates
        }


def test_request_fields_pass_through_to_the_frozen_optimization_config():
    request = ResearchRequestV5(
        research_context={
            "symbol": "600000.SH", "start_date": "2024-01-02", "end_date": "2024-12-31",
            "frequency": "1d", "initial_shares": 1000,
        },
        ranges={
            "buy_step": {"low": .005, "high": .05},
            "sell_step": {"low": .005, "high": .05},
            "buy_shares": {"low": 100, "high": 500},
            "sell_shares": {"low": 100, "high": 300},
        },
        prior_policy="daily_state_v1",
        source_etf={"result_id": "etf-result-1", "candidate_ids": [3, 7]},
        requested_data_version="research-data-v1",
        research_origin={"etf_result_id": "etf-result-1", "source_as_of": "2024-01-01", "seeds": []},
    )

    config = request.to_optimization()

    assert config.prior_policy == "daily_state_v1"
    assert config.source_etf.result_id == "etf-result-1"
    assert config.source_etf.candidate_ids == [3, 7]
    assert config.requested_data_version == "research-data-v1"
    assert config.research_origin == request.research_origin


def test_off_policy_preserves_the_original_sparse_coordinate_sequence():
    config = _config()
    space = ParameterSpace(config)
    profile = market_statistical_profile(_history_data(), pd.bdate_range("2020-01-02", periods=30))

    actual = _initial_coordinates(space, config, "2024-02-29", profile)

    budget = int(config.strategy.training_budget)
    seed = robust_research._stable_seed(config, "2024-02-29", "robust_sample")
    low = [space.dimensions[name].low for name in space.active]
    high = [space.dimensions[name].high for name in space.active]
    anchors = [
        coordinate for values in (low, high, [(left + right) // 2 for left, right in zip(low, high)])
        if space.legal(values) and (coordinate := space.full_coordinate(values))
    ]
    expected = robust_research._dedupe([
        *anchors,
        *robust_research._statistical_coordinates(space, profile),
        *coarse_coordinates(space, max(1, budget // 4)),
        *robust_research._sobol_coordinates(space, budget, seed),
    ], budget)

    assert actual == expected
    assert len(actual) == budget


def test_prior_quota_strict_source_validation_and_stable_dedupe(monkeypatch):
    seeds = [
        _seed(1, buy_step=.005),
        _seed(2, buy_step=.005),
        _seed(3, buy_step=.2),
        _seed(4, buy_step=.007),
        *[_seed(10 + index, buy_step=.005 * index) for index in range(2, 9)],
    ]
    config = _config(prior_policy="daily_state_v1", source_as_of="2024-02-20", seeds=seeds)
    space = ParameterSpace(config)
    state_coordinate = space.coordinate({
        "buy_step": .03, "sell_step": .03, "buy_shares": 200, "sell_shares": 200,
    })
    _patch_market_state(monkeypatch, [
        {"buy_step": .03, "sell_step": .03, "source": "natr14"},
    ])

    plan = _build_research_prior(config, space, _history_data(), pd.Timestamp("2024-03-01"))
    checks = plan.diagnostics["candidate_checks"]

    assert plan.diagnostics["quota"] == 8
    assert plan.diagnostics["planned_count"] == 8
    assert plan.diagnostics["quota_used"] == 0
    assert checks[0]["status"] == "planned"
    assert len(plan.coordinates) == 8
    assert plan.diagnostics["source_check"]["status"] == "eligible"
    assert checks[1]["reason"] == "duplicate_coordinate"
    assert checks[1]["deduped_into"] == 1
    assert checks[2]["reason"] == "out_of_range:buy_step"
    assert checks[3]["reason"] == "invalid_source_granularity:buy_step"
    assert checks[-1]["reason"] == "quota_exhausted"
    assert state_coordinate not in plan.coordinates
    first = plan.coordinates[0]
    assert len(plan.sources[first]) == 2

    rows = {first: {"key": first, "params": space.params(first)}}
    _attach_prior_sources(rows, plan)
    public = _public_row(rows[first])
    assert len(public["prior_sources"]) == 2
    assert set(public["prior_sources"]) == {
        "etf:etf-result-1:1", "etf:etf-result-1:2",
    }


def test_state_steps_use_nearest_lower_tie_and_share_axis_midpoints(monkeypatch):
    config = _config(prior_policy="daily_state_v1", source_as_of=None, seeds=[])
    space = ParameterSpace(config)
    _patch_market_state(monkeypatch, [
        {"buy_step": .0125, "sell_step": .0175, "source": "natr14"},
    ])

    plan = _build_research_prior(config, space, _history_data(), pd.Timestamp("2024-03-01"))
    first = plan.diagnostics["candidate_checks"][0]

    assert first["original"] == {"buy_step": .0125, "sell_step": .0175}
    assert first["converted"]["buy_step"] == pytest.approx(.01)
    assert first["converted"]["sell_step"] == pytest.approx(.015)
    assert first["converted"]["buy_shares"] == 300
    assert first["converted"]["sell_shares"] == 200


def test_out_of_range_generated_steps_are_skipped_without_clipping(monkeypatch):
    config = _config(prior_policy="daily_state_v1", source_as_of=None, seeds=[])
    space = ParameterSpace(config)
    _patch_market_state(monkeypatch, [
        {"buy_step": .004, "sell_step": .02, "source": "below_min"},
        {"buy_step": .052, "sell_step": .02, "source": "above_max"},
        {"buy_step": .0075, "sell_step": .0225, "source": "inside_tie"},
        {"buy_step": .05, "sell_step": .045, "source": "inside_upper"},
    ])

    plan = _build_research_prior(config, space, _history_data(), pd.Timestamp("2024-03-01"))
    checks = plan.diagnostics["candidate_checks"]

    assert checks[0]["reason"] == "out_of_range:buy_step"
    assert checks[1]["reason"] == "out_of_range:buy_step"
    assert checks[0].get("converted") is None
    assert checks[1].get("converted") is None
    assert checks[2]["converted"]["buy_step"] == pytest.approx(.005)
    assert checks[2]["converted"]["sell_step"] == pytest.approx(.02)
    assert checks[3]["converted"]["buy_step"] == pytest.approx(.05)
    assert len(plan.coordinates) == 2


def test_real_daily_state_module_supplies_the_prior_without_a_source_etf():
    config = _config(prior_policy="daily_state_v1", source_as_of=None, seeds=[])
    config = config.model_copy(update={"source_etf": None, "research_origin": None})
    space = ParameterSpace(config)
    data = _history_data(days=80)

    plan = _build_research_prior(config, space, data, pd.Timestamp(data.calendar[60]))

    state = plan.diagnostics["market_state"]
    assert state["method_version"] == "daily_state_v1"
    assert plan.diagnostics["market_state_error"] is None
    assert plan.diagnostics["state_candidate_count"] >= 1
    assert plan.diagnostics["planned_count"] == len(plan.coordinates)
    assert plan.diagnostics["quota_used"] == 0


def test_future_source_skips_static_seeds_but_keeps_state_prior(monkeypatch):
    source = _seed(99, buy_step=.015)
    future = _config(
        prior_policy="daily_state_v1", source_as_of="2024-03-02", seeds=[source],
    )
    space = ParameterSpace(future)
    state_candidate = {"buy_step": .025, "sell_step": .02, "source": "adx14"}
    _patch_market_state(monkeypatch, [state_candidate])
    decision = pd.Timestamp("2024-03-01")

    future_plan = _build_research_prior(future, space, _history_data(), decision)

    source_coordinate = space.coordinate(source["params"])
    state_coordinate = space.coordinate({
        "buy_step": .025, "sell_step": .02, "buy_shares": 300, "sell_shares": 200,
    })
    assert future_plan.diagnostics["source_check"]["status"] == "future_source"
    assert future_plan.diagnostics["candidate_checks"][0]["reason"] == "future_source"
    assert source_coordinate not in future_plan.coordinates
    assert future_plan.coordinates == [state_coordinate]
    assert future_plan.diagnostics["source_check"]["requested_data_version"] == "research-data-v1"
    assert future_plan.diagnostics["source_check"]["source_data_version"] == "etf-data-v1"

    past = future.model_copy(update={
        "research_origin": {**future.research_origin, "source_as_of": "2024-03-01"},
    })
    past_plan = _build_research_prior(past, space, _history_data(), decision)
    assert past_plan.diagnostics["source_check"]["status"] == "eligible"
    assert past_plan.coordinates == [source_coordinate, state_coordinate]


def test_select_calls_market_state_at_window_end_and_injects_between_coarse_and_sobol(monkeypatch):
    data = _history_data()
    source = _seed(99, buy_step=.015)
    config = _config(
        prior_policy="daily_state_v1", source_as_of="2024-03-02", seeds=[source],
    )
    asof = pd.Timestamp(data.calendar[400])
    base = config.base.model_copy(update={
        "start_date": data.calendar[401].date(), "end_date": data.calendar[-1].date(),
    })
    config = config.model_copy(update={"base": base, "selection_target": "fixed"})
    observed = []
    _patch_market_state(
        monkeypatch,
        [{"buy_step": .025, "sell_step": .02, "source": "adx14"}],
        observed,
    )
    space = ParameterSpace(config)
    evaluator = _FakeEvaluator(space)
    monkeypatch.setattr(robust_research, "BatchEvaluator", lambda *args, **kwargs: evaluator)

    result = select_robust_at(config, base, data, asof)

    assert observed and observed[0][1] == asof.normalize()
    state_coordinate = space.coordinate({
        "buy_step": .025, "sell_step": .02, "buy_shares": 300, "sell_shares": 200,
    })
    source_coordinate = space.coordinate(source["params"])
    first_batch = evaluator.calls[0]
    assert state_coordinate in first_batch
    assert source_coordinate not in first_batch
    profile = market_statistical_profile(data, pd.bdate_range(asof, periods=1))
    plan = _build_research_prior(config, space, data, asof)
    expected = _initial_coordinates(space, config, asof.date().isoformat(), profile, plan.coordinates)
    assert first_batch == expected
    assert result["diagnostics"]["research_prior"]["source_check"]["status"] == "future_source"


def test_full_exhaustive_only_labels_prior_rows_and_preserves_results(monkeypatch):
    data = _history_data()
    ranges = {
        "buy_step": {"low": .005, "high": .01},
        "sell_step": {"low": .01, "high": .01},
        "buy_shares": {"low": 100, "high": 100},
        "sell_shares": {"low": 100, "high": 100},
    }
    common = _config(
        training_budget=40, search_method="exhaustive", hypercube_edge=1, ranges=ranges,
    )
    asof = pd.Timestamp(data.calendar[400])
    base = common.base.model_copy(update={
        "start_date": data.calendar[401].date(), "end_date": data.calendar[-1].date(),
    })
    off = common.model_copy(update={"base": base})
    on = common.model_copy(update={
        "base": base, "prior_policy": "daily_state_v1",
        "research_origin": {
            **common.research_origin, "source_as_of": "2020-02-20",
            "seeds": [_seed(7, buy_step=.005, sell_step=.01, buy_shares=100, sell_shares=100)],
        },
    })
    _patch_market_state(monkeypatch, [])
    space = ParameterSpace(off)
    evaluator = _FakeEvaluator(space)
    monkeypatch.setattr(robust_research, "BatchEvaluator", lambda *args, **kwargs: evaluator)

    without = select_robust_at(off, base, data, asof)
    with_prior = select_robust_at(on, base, data, asof)

    assert without["diagnostics"]["search_method"] == "exhaustive"
    assert without["diagnostics"]["global_exhaustive"] is True
    assert "research_prior" not in without["diagnostics"]
    assert {row["key"] for row in without["candidate_archive"]} == {
        row["key"] for row in with_prior["candidate_archive"]
    }
    assert without["recommended_params"] == with_prior["recommended_params"]
    prior_coordinate = space.coordinate({
        "buy_step": .005, "sell_step": .01, "buy_shares": 100, "sell_shares": 100,
    })
    labelled = next(
        row for row in with_prior["candidate_archive"] if tuple(row["key"]) == prior_coordinate
    )
    assert labelled["prior_sources"] == ["etf:etf-result-1:7"]
    assert with_prior["diagnostics"]["research_prior"]["coordinate_values"]
    prior = with_prior["diagnostics"]["research_prior"]
    assert prior["phase"] == "final"
    assert prior["planned_count"] == 1
    assert prior["quota_used"] == 0
    assert prior["actually_evaluated_count"] == 1
    assert prior["annotation_only_count"] == 1
    assert prior["candidate_checks"][0]["status"] == "annotation_only"
    assert prior["candidate_checks"][0]["reason"] == "exhaustive"
    assert prior["candidate_checks"][0]["actually_evaluated"] is True


@pytest.mark.parametrize(
    "scenario,status,reason,used,evaluated,annotated",
    [
        ("crowded", "skipped", "initial_budget_exhausted", 0, 0, 0),
        ("duplicate", "annotation_only", "duplicate_initial_coordinate", 0, 1, 1),
        ("admitted", "actually_evaluated", None, 1, 1, 0),
        ("unreturned", "skipped", "not_evaluated", 0, 0, 0),
    ],
)
def test_prior_usage_is_reconciled_with_initial_and_actual_rows(
    monkeypatch, scenario, status, reason, used, evaluated, annotated,
):
    data = _history_data(days=80)
    seed = (
        _seed(1, buy_step=.005, sell_step=.005, buy_shares=100, sell_shares=100)
        if scenario == "duplicate" else _seed(1, buy_step=.015)
    )
    config = _config(
        prior_policy="daily_state_v1", source_as_of="2020-01-02", seeds=[seed],
        training_budget=5, hypercube_edge=1,
    )
    space = ParameterSpace(config)
    coordinate = space.coordinate(seed["params"])
    _patch_market_state(monkeypatch, [])
    # Five existing distinct points fill the budget before priors are considered.
    statistical = list(space.iterate(5)) if scenario == "crowded" else []
    monkeypatch.setattr(robust_research, "_statistical_coordinates", lambda *args: statistical)
    monkeypatch.setattr(robust_research, "_local_search", lambda *args: [])
    evaluator = _FakeEvaluator(space)
    if scenario == "unreturned":
        evaluate = evaluator.evaluate_training
        monkeypatch.setattr(
            evaluator, "evaluate_training",
            lambda points: [row for row in evaluate(points) if tuple(row["key"]) != coordinate],
        )
    monkeypatch.setattr(robust_research, "BatchEvaluator", lambda *args, **kwargs: evaluator)

    result = select_robust_at(config, config.base, data, data.calendar[60])

    prior = result["diagnostics"]["research_prior"]
    check = prior["candidate_checks"][0]
    assert prior["phase"] == "final"
    assert prior["planned_count"] == 1
    assert prior["quota_used"] == used
    assert prior["actually_evaluated_count"] == evaluated
    assert prior["annotation_only_count"] == annotated
    assert check["status"] == status
    assert check["reason"] == reason
    assert check["actually_evaluated"] is bool(evaluated)
    assert (coordinate in evaluator.calls[0]) is (scenario != "crowded")
    assert len(evaluator.calls[0]) == 5


def test_fixed_current_and_adaptive_results_expose_prior_diagnostics(monkeypatch):
    from stockregression import share_engine
    from stockregression import research_v5

    data = _history_data(days=90)
    dates = pd.DatetimeIndex(data.calendar)
    config = _config(prior_policy="daily_state_v1", source_as_of="2024-02-20")
    base = config.base.model_copy(update={
        "start_date": dates[30].date(), "end_date": dates[-1].date(),
    })
    config = config.model_copy(update={
        "base": base, "selection_target": "both", "rebalance_days": 100,
        "requested_data_version": data.version,
    })
    _patch_market_state(monkeypatch, [])
    prior = {"policy": "daily_state_v1", "decision_date": dates[29].date().isoformat(), "quota": 8}
    synthetic = {
        "status": "completed", "recommended_params": None,
        "params": {"buy_step": .01, "sell_step": .01, "buy_shares": 100, "sell_shares": 100},
        "as_of": dates[29].date().isoformat(), "score": None, "eligible": False,
        "reasons": [], "candidate_archive": [], "search_candidate_archive": [],
        "trial_archive": [], "recommendation_regions": [],
        "diagnostics": {
            "score_complete": True, "candidate_count": 0, "eligible_center_count": 0,
            "research_prior": prior,
        },
    }
    monkeypatch.setattr(research_v5, "select_robust_at", lambda *args, **kwargs: deepcopy(synthetic))
    monkeypatch.setattr(
        share_engine, "run_share_backtest",
        lambda *args, **kwargs: {"strategies": {"grid": {"metrics": {}}}},
    )
    preview = estimate_robust_search(config, data, bars_memory_bytes=1_000_000)
    contract = {
        "data_version": data.version,
        "prior_policy_hash": prior_contract_hash(config),
        "effective_total_evaluation_budget": preview["effective_total_evaluation_budget"],
        "max_runtime_seconds": config.strategy.max_runtime_seconds,
        "estimated_task_memory_mb": preview["estimated_task_memory_mb"],
        "workspace_budget_mb": preview["workspace_budget_mb"],
    }

    result = research_v5.optimize_v5(config, data, budget_contract=contract)

    assert result["research_origin"] == config.research_origin
    assert result["fixed"]["diagnostics"]["research_prior"] == prior
    assert result["fixed"]["current"]["diagnostics"]["research_prior"] == prior
    assert result["evidence"]["current_decision"]["diagnostics"]["research_prior"] == prior
    assert result["evidence"]["decisions"][0]["diagnostics"]["research_prior"] == prior
    assert result["adaptive"]["diagnostics"]["research_prior"][0]["policy"] == "daily_state_v1"


def test_optimize_v5_rejects_data_version_and_prior_contract_mismatches():
    from stockregression import research_v5

    data = _history_data(days=90)
    dates = pd.DatetimeIndex(data.calendar)
    config = _config(prior_policy="daily_state_v1", source_as_of="2024-02-20").model_copy(update={
        "base": _config().base.model_copy(update={
            "start_date": dates[30].date(), "end_date": dates[-1].date(),
        }),
        "requested_data_version": data.version,
    })

    with pytest.raises(ValueError, match="先验来源合同"):
        research_v5.optimize_v5(
            config, data, budget_contract={"prior_policy_hash": "stale-prior-hash"},
        )
    with pytest.raises(ValueError, match="行情数据版本"):
        research_v5.optimize_v5(
            config.model_copy(update={"requested_data_version": "stale-data-version"}), data,
        )
    empty = research_v5.optimize_v5(
        config.model_copy(update={
            "base": config.base.model_copy(update={
                "start_date": pd.Timestamp("2030-01-02").date(), "end_date": None,
            }),
        }),
        data,
    )
    assert empty["research_origin"] == config.research_origin


def test_eval_end_source_is_future_for_fixed_current_and_every_adaptive_decision(monkeypatch):
    from stockregression import research_v5, share_engine

    data = _history_data(days=600)
    dates = pd.DatetimeIndex(data.calendar)
    eval_start, eval_end = dates[400], dates[430]
    config = _config(
        prior_policy="daily_state_v1", source_as_of=eval_end.date().isoformat(),
        seeds=[_seed(88, buy_step=.01)],
    ).model_copy(update={
        "base": _config().base.model_copy(update={
            "start_date": eval_start.date(), "end_date": eval_end.date(),
        }),
        "selection_target": "both", "rebalance_days": 10,
        "requested_data_version": data.version,
    })
    _patch_market_state(monkeypatch, [])
    space = ParameterSpace(config)
    monkeypatch.setattr(
        robust_research, "BatchEvaluator",
        lambda *args, **kwargs: _FakeEvaluator(space),
    )
    monkeypatch.setattr(
        share_engine, "run_share_backtest",
        lambda *args, **kwargs: {"strategies": {"grid": {"metrics": {}}}},
    )

    result = research_v5.optimize_v5(config, data)

    assert result["status"] in {"completed", "partial"}
    fixed_prior = result["fixed"]["diagnostics"]["research_prior"]
    current_prior = result["fixed"]["current"]["diagnostics"]["research_prior"]
    assert fixed_prior["decision_date"] == dates[399].date().isoformat()
    assert current_prior["decision_date"] == dates[399].date().isoformat()
    assert fixed_prior["source_check"]["status"] == "future_source"
    assert current_prior["source_check"]["status"] == "future_source"

    adaptive = result["adaptive"]["diagnostics"]["research_prior"]
    assert [item["decision_date"] for item in adaptive] == [
        dates[index].date().isoformat() for index in (399, 409, 419, 429)
    ]
    for item in adaptive:
        assert item["source_check"]["decision_date"] == item["decision_date"]
        assert item["source_check"]["status"] == "future_source"
        assert item["coordinate_values"] == []

    for archive in result["evidence"]["candidate_archives"].values():
        assert not any(row.get("prior_sources") for row in archive)


def test_preflight_reports_prior_without_evaluating_candidates(monkeypatch):
    config = _config(prior_policy="daily_state_v1", source_as_of="2020-01-02", seeds=[_seed(1, buy_step=.01)])
    _patch_market_state(monkeypatch, [])
    monkeypatch.setattr(
        robust_research, "BatchEvaluator",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preflight evaluated candidates")),
    )
    data = _history_data(days=80)
    config = config.model_copy(update={
        "base": config.base.model_copy(update={
            "start_date": data.calendar[31].date(), "end_date": data.calendar[-1].date(),
        }),
    })

    preview = estimate_robust_search(config, data, bars_memory_bytes=1_000_000)

    assert preview["research_prior"]["policy"] == "daily_state_v1"
    assert preview["research_prior"]["quota"] == 8
    assert preview["research_prior"]["source_check"]["eligible"] in (True, False)
    assert preview["research_prior"]["phase"] == "planned"
    assert preview["research_prior"]["planned_count"] == 1
    assert preview["research_prior"]["quota_used"] == 0
    assert preview["research_prior"]["actually_evaluated_count"] == 0
    assert preview["research_prior"]["candidate_checks"][0]["status"] == "planned"
