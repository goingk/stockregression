"""稳健 J 数值与搜索契约回归测试（protocol_version=5 / strategy_version=5）。

契约要点：

* 四个搜索轴固定为 buy_step、sell_step、buy_shares、sell_shares，
  公开坐标顺序就是这四轴；
* J 分数建立在峰值本金超额收益上，配对基准是 matched_funding_hold，
  风险口径是 ``max_principal_drawdown``；
* 邻域缺失或评价失败永远不构成完整样本；
* 训练、邻域与展示三个阶段分别计费，展示阶段只补算冻结的目标集合，
  不会因为新 leader 出现而扩大目标；
* 枚举小空间会产生唯一平台峰值，评价失败时保留失败证据。
"""
from __future__ import annotations

from itertools import product
import math

import pandas as pd
import pytest

from stockregression.models import BacktestConfig, DataBundle, OptimizationConfig, ResearchStrategySpec
from stockregression.numerics.robust import (
    NeighborhoodOperator,
    PeakAnalyzer,
    eligible,
    position_penalty,
    robust_score,
    tensor_robust_scores,
)
from stockregression.research import robust as robust_research
from stockregression.research.robust import (
    EvaluationSession,
    _complete_reporting_scores,
    _strides,
    estimate_robust_search,
    select_robust_at,
)
from stockregression.research.space import ParameterSpace


def make_config(*, buy_high: float = .005, share_high: int = 500, strategy: dict | None = None) -> OptimizationConfig:
    base = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date="2024-01-02", frequency="1d",
        initial_shares=1000, min_shares=0, max_shares=None, buy_shares=100, sell_shares=100,
        grid_mode="geometric", buy_step=.03, sell_step=.03,
    )
    parameters = {
        "strategy_version": 5, "search_method": "exhaustive", "ranking_mode": "robust_j",
        "granularity": {"buy_step": .001, "sell_step": .001, "buy_shares": 100, "sell_shares": 100},
        "hypercube_edge": 3, "sigma": 1.0, "alpha": .3, "beta": 1.2, "tau": .01,
        "max_shares_mode": "unlimited", **dict(strategy or {}),
    }
    return OptimizationConfig(
        base=base, protocol_version=5, strategy=ResearchStrategySpec.model_validate(parameters),
        ranges={
            "buy_step": {"low": .001, "high": buy_high},
            "sell_step": {"low": .001, "high": .001},
            "buy_shares": {"low": 100, "high": share_high},
            "sell_shares": {"low": 100, "high": 100},
        },
    )


def field(space: ParameterSpace):
    return lambda name, coordinate: dict(zip(space.fields, coordinate))[name]


def row(space: ParameterSpace, coordinate, value: float, *, drawdown=.05, trades=20, max_pos=1.0, min_pos=0.0):
    """一条 v5 训练候选行。

    ``training_excess_return`` 是相对 matched_funding_hold 配对基准的
    峰值本金超额收益，``max_principal_drawdown`` 是本金利润回撤。
    """
    return {
        "key": coordinate, "coordinate": list(coordinate), "params": space.params(coordinate),
        "evaluation_status": "completed", "training_excess_return": float(value),
        "training_return": float(value), "training_baseline_return": 0.0,
        "training_baseline_kind": "matched_funding_hold",
        "max_principal_drawdown": float(drawdown), "trade_count": int(trades),
        "buy_count": max(1, int(trades) // 2), "sell_count": max(1, int(trades) // 2),
        "actual_max_pos_ratio": float(max_pos), "actual_min_pos_ratio": float(min_pos),
        "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
    }


def complete_field(space: ParameterSpace, value):
    rows = {}
    for coordinate in space.iterate():
        rows[coordinate] = row(space, coordinate, value(coordinate))
    return rows


def test_public_robust_coordinate_order_is_the_four_axis_contract():
    config = make_config(buy_high=.003, share_high=300)
    space = ParameterSpace(config)
    assert space.fields == ("buy_step", "sell_step", "buy_shares", "sell_shares")
    assert space.active == ("buy_step", "buy_shares")
    coordinate = space.full_coordinate((2, 2))
    assert coordinate == (2, 1, 2, 1)
    assert space.params(coordinate) == {
        "buy_step": .002, "sell_step": .001, "buy_shares": 200,
        "sell_shares": 100, "initial_shares": 1000, "min_shares": 0, "max_shares": None,
    }


def test_direct_and_tensor_j_agree_and_boundary_weights_are_renormalised():
    config = make_config()
    space = ParameterSpace(config)
    values = field(space)
    rows = complete_field(
        space,
        lambda coordinate: .04 * (values("buy_step", coordinate) - 3) + .015 * (values("buy_shares", coordinate) / 100),
    )
    direct = {coordinate: robust_score(coordinate, rows, space, config.strategy) for coordinate in space.iterate()}
    tensor = tensor_robust_scores(space, rows, config.strategy)
    for coordinate, expected in direct.items():
        actual = tensor[coordinate]
        assert actual.complete and expected.complete
        assert actual.j_score == pytest.approx(expected.j_score, abs=1e-12)
        assert actual.neighborhood_mean == pytest.approx(expected.neighborhood_mean, abs=1e-12)
        assert actual.downside_risk == pytest.approx(expected.downside_risk, abs=1e-12)

    boundary = space.full_coordinate((1, 1))
    operator = NeighborhoodOperator(space, 1)
    points = operator.points(boundary)
    weights = operator.weights(boundary, 1.0, points)
    assert len(points) == 4
    assert math.fsum(weights) == pytest.approx(1.0, abs=1e-15)
    score = direct[boundary]
    manual_mean = math.fsum(
        weight * rows[point]["training_excess_return"] for weight, point in zip(weights, points)
    )
    assert score.neighborhood_mean == pytest.approx(manual_mean, abs=1e-12)
    assert score.neighborhood_size == 4


def test_large_local_strides_are_limited_per_axis():
    config = make_config(buy_high=.005, share_high=4100)
    space = ParameterSpace(config)
    assert _strides(space, config.strategy, "buy_step") == [1, 2]
    assert _strides(space, config.strategy, "buy_shares") == [1, 2, 4, 8, 10]


def test_j_peak_is_not_the_raw_return_peak_and_ineligible_high_return_is_rejected():
    config = make_config(buy_high=.005, share_high=500)
    space = ParameterSpace(config)
    values = field(space)
    raw_peak = space.full_coordinate((3, 1, 5, 1))
    j_peak = space.full_coordinate((5, 1, 5, 1))
    rows = complete_field(space, lambda coordinate: .08)
    # The raw-return centre is surrounded by deliberately weak, over-positioned
    # and under-traded neighbours.  They still contribute their real returns.
    for coordinate in space.iterate():
        by_name = dict(zip(space.active, space.active_coords(coordinate)))
        if max(abs(by_name["buy_step"] - 3), abs(by_name["buy_shares"] - 5)) <= 1:
            rows[coordinate] = row(space, coordinate, -.05, drawdown=.9, trades=0, max_pos=3.6)
    rows[raw_peak] = row(space, raw_peak, .20, drawdown=.5, trades=20, max_pos=4.0)
    rows[j_peak] = row(space, j_peak, .08, drawdown=.0, trades=20, max_pos=1.0)

    raw_score = robust_score(raw_peak, rows, space, config.strategy)
    steady_score = robust_score(j_peak, rows, space, config.strategy)
    assert rows[raw_peak]["training_excess_return"] > rows[j_peak]["training_excess_return"]
    assert raw_score.neighborhood_mean < raw_score.center_return
    assert steady_score.j_score > raw_score.j_score
    assert not eligible(rows[raw_peak], raw_score, max_principal_drawdown=.2, min_trades=10, strategy=config.strategy)
    assert eligible(rows[j_peak], steady_score, max_principal_drawdown=.2, min_trades=10, strategy=config.strategy)


def test_missing_or_failed_neighbour_never_becomes_a_complete_zero_sample():
    config = make_config()
    space = ParameterSpace(config)
    rows = complete_field(space, lambda coordinate: .05)
    center = space.full_coordinate((2, 1))
    neighbour = NeighborhoodOperator(space, 1).points(center)[1]
    removed = rows.pop(neighbour)
    direct = robust_score(center, rows, space, config.strategy)
    tensor = tensor_robust_scores(space, rows, config.strategy)[center]
    assert not direct.complete and direct.missing_count == 1
    assert not tensor.complete and tensor.missing_count == 1
    assert not math.isfinite(direct.j_score) and not math.isfinite(tensor.j_score)
    removed["evaluation_status"] = "failed"
    rows[neighbour] = removed
    assert not robust_score(center, rows, space, config.strategy).complete


def test_floor_penalty_accumulates_episode_grid_squares():
    config = make_config(strategy={"gamma_min": .4, "eta_floor": .5, "floor_k_ref": 10.0})
    assert config.base.floor_penalty_config is not None
    assert config.base.floor_penalty_config.eta_floor == .5
    assert config.base.floor_penalty_config.gamma_min == .4
    assert config.base.floor_penalty_config.floor_k_ref == 10.0
    assert config.base.model_dump(mode="json")["floor_penalty_config"]["floor_k_ref"] == 10.0
    item = row(
        ParameterSpace(make_config()), (1, 1, 1, 1), .05,
        min_pos=.1, max_pos=1.0,
    )
    item["floor_episode_count"] = 2
    item["floor_episode_grids"] = [3, 4]
    total, maximum, minimum = position_penalty(item, config.strategy)
    assert maximum == pytest.approx(0.0)
    assert minimum == pytest.approx(.4 * (.3 ** 2 + .4 ** 2))
    assert total == pytest.approx(.1)

    item["floor_episode_grids"] = [5]
    assert position_penalty(item, config.strategy)[2] == pytest.approx(.1)
    item["floor_episode_grids"] = [0, 0, 0, 0, 0]
    assert position_penalty(item, config.strategy)[2] == 0.0


def test_missing_floor_episode_evidence_makes_the_score_incomplete():
    config = make_config(strategy={"gamma_min": .4, "eta_floor": .5})
    space = ParameterSpace(config)
    rows = complete_field(space, lambda coordinate: .05)
    center = space.full_coordinate((2, 1))
    rows[center].pop("floor_episode_grids")
    direct = robust_score(center, rows, space, config.strategy)
    tensor = tensor_robust_scores(space, rows, config.strategy)[center]
    assert not direct.complete and not tensor.complete
    assert not math.isfinite(direct.position_penalty)


def test_peak_analyzer_closes_a_flat_plateau_and_detects_a_higher_exit():
    config = make_config()
    space = ParameterSpace(config)
    rows = complete_field(space, lambda coordinate: .05)

    def score_of(coordinate):
        return robust_score(coordinate, rows, space, config.strategy)

    def eligible_of(coordinate):
        return eligible(
            rows.get(coordinate), score_of(coordinate), max_principal_drawdown=.2,
            min_trades=10, strategy=config.strategy,
        )

    analyzer = PeakAnalyzer(space, 1e-12)
    start = space.full_coordinate((1, 1))
    analysis = analyzer.inspect(start, score_of, eligible_of)
    assert analysis.state == "peak" and analysis.complete and analysis.local_peak
    assert analysis.flat_region and len(analysis.plateau) == len(list(space.iterate()))
    assert analysis.representative == min(analysis.plateau)

    exit_point = space.full_coordinate((5, 5))
    rows[exit_point] = row(space, exit_point, .09)
    higher = analyzer.inspect(start, score_of, eligible_of)
    assert higher.state == "slope"
    assert higher.higher is not None
    assert not higher.local_peak


def test_boundary_complete_peak_does_not_mark_outside_lattice_as_missing():
    config = make_config()
    space = ParameterSpace(config)
    rows = complete_field(space, lambda coordinate: .05)

    def score_of(coordinate):
        return robust_score(coordinate, rows, space, config.strategy)

    corner = space.full_coordinate((1, 1))
    analysis = PeakAnalyzer(space, 1e-12).inspect(
        corner, score_of,
        lambda coordinate: eligible(rows.get(coordinate), score_of(coordinate), max_principal_drawdown=.2, min_trades=10, strategy=config.strategy),
    )
    assert analysis.state == "peak" and analysis.complete
    assert not analysis.missing


class FakeEvaluator:
    chunk_size = 2

    def __init__(self, space: ParameterSpace, *, fail=False):
        self.space = space
        self.fail = fail
        self.calls = []

    def evaluate_training(self, coordinates):
        self.calls.append(list(coordinates))
        if self.fail:
            raise RuntimeError("simulated evaluation failure")
        return [row(self.space, coordinate, .05) for coordinate in coordinates]


def test_evaluation_session_charges_unique_points_and_keeps_failure_evidence():
    config = make_config()
    space = ParameterSpace(config)
    points = list(space.iterate())
    evaluator = FakeEvaluator(space)
    session = EvaluationSession(evaluator, space, config, None)
    assert session.ensure(points[:2], "training", 2) == 2
    assert session.ensure(points[:2], "training", 2) == 0
    assert session.spent["training"] == 2
    assert session.requested - session.attempts == 2

    failed = EvaluationSession(FakeEvaluator(space, fail=True), space, config, None)
    assert failed.ensure(points[:1], "training", 1) == 1
    assert failed.spent["training"] == 1 and failed.failures == 1
    assert failed.rows[points[0]]["evaluation_status"] == "failed"
    assert not robust_score(points[0], failed.rows, space, config.strategy).complete

    cancelled = EvaluationSession(FakeEvaluator(space), space, config, lambda: True)
    with pytest.raises(InterruptedError):
        cancelled.ensure(points[:1], "training", 1)


def test_neighborhood_budget_exhaustion_returns_incomplete_score():
    config = make_config()
    space = ParameterSpace(config)
    evaluator = FakeEvaluator(space)
    session = EvaluationSession(evaluator, space, config, None)
    center = space.full_coordinate((3, 3))
    score = session.score(center, cap=2)
    assert not score.complete
    assert score.missing_count > 0


def test_reporting_completes_the_frozen_return_leader_without_charging_the_search_budgets():
    """展示补算只补齐冻结目标，不追逐补算中出现的新 leader。

    v5 的展示阶段有自己的预算，并且目标集合在阶段开始时冻结：
    补算邻域时出现收益更高的点，也不会成为新的展示目标，
    更不会因此动用训练或邻域阶段的预算。
    """
    config = make_config(
        buy_high=.011, share_high=100,
        strategy={"recommendation_count": 4, "reporting_budget": 4},
    )
    space = ParameterSpace(config)

    class RisingEvaluator(FakeEvaluator):
        def evaluate_training(self, coordinates):
            return [row(space, coordinate, coordinate[0] / 100, trades=0, max_pos=4.0) for coordinate in coordinates]

    session = EvaluationSession(RisingEvaluator(space), space, config, None)
    start = space.full_coordinate((3,))
    session.ensure([start], "training", 1)
    spent_before = dict(session.spent)
    _complete_reporting_scores(session)

    # 阶段开始时只有一个已完成候选，它就是被冻结的展示目标。
    leader = start
    score = session.scores[leader]
    assert score.complete and score.missing_count == 0
    assert score.neighborhood_mean == pytest.approx(robust_score(leader, session.rows, space, config.strategy).neighborhood_mean)

    # 补算邻域时出现的更高收益点不会成为新的展示目标，
    # 因此它自己的邻域不会被补算，分数保持不完整。
    new_leader = max(session.rows, key=lambda coordinate: session.rows[coordinate]["training_excess_return"])
    assert new_leader != leader
    assert session.scores[new_leader].complete is False
    assert space.full_coordinate((space.dimensions["buy_step"].high,)) not in session.rows

    # 展示阶段的花费独立记账，不得借用训练或邻域阶段的预算。
    assert session.spent["training"] == spent_before["training"]
    assert session.spent["validation"] == spent_before["validation"]
    assert session.spent["reporting"] > 0
    assert session.spent["reporting"] <= config.strategy.reporting_budget
    assert session.budget_stopped is False
    assert session.reporting_stopped is False
    assert not eligible(session.rows[leader], score, max_principal_drawdown=config.max_principal_drawdown, min_trades=config.min_trades, strategy=config.strategy)


def test_reporting_preserves_failed_neighbours_and_does_not_retry_them_forever():
    config = make_config(buy_high=.003, share_high=100)
    space = ParameterSpace(config)
    session = EvaluationSession(FakeEvaluator(space, fail=True), space, config, None)
    center = space.full_coordinate((2,))
    session.rows[center] = row(space, center, .1)
    _complete_reporting_scores(session)
    assert not session.scores[center].complete
    assert session.scores[center].missing_count == 2
    assert session.spent["reporting"] == 2


def _long_history_data() -> tuple[DataBundle, pd.Timestamp]:
    dates = pd.bdate_range("2023-01-02", periods=340)
    daily = pd.DataFrame({
        "date": dates, "open": 10.0, "high": 11.0, "low": 9.0,
        "close": 10.0, "preclose": 10.0, "volume": 1_000_000.0,
    })
    data = DataBundle(
        symbol="600000.SH", bars=pd.DataFrame(), daily=daily,
        calendar=dates, quality={"minute_ratio": 1.0},
    )
    return data, dates[270]


class SearchFakeEvaluator:
    chunk_size = 4

    def __init__(self, config, base, data, space, train_dates, *args, **kwargs):
        self.space = space

    def evaluate_training(self, coordinates):
        return [row(self.space, coordinate, .02) for coordinate in coordinates]

    def evaluate_training_stress(self, coordinates, cost_multiplier=2.0):
        return {
            coordinate: {
                "training_excess_stress": .01, "max_principal_drawdown": .05,
                "trade_count": 20, "actual_min_pos_ratio": 0.0,
                "actual_max_pos_ratio": 1.0,
                "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
            }
            for coordinate in coordinates
        }


class IneligibleSearchFakeEvaluator(SearchFakeEvaluator):
    def evaluate_training(self, coordinates):
        rows = []
        for coordinate in coordinates:
            item = row(self.space, coordinate, .02, trades=0, max_pos=4.0)
            rows.append(item)
        return rows


def test_robust_selector_exhausts_small_space_and_emits_one_flat_peak(monkeypatch):
    data, evaluation_start = _long_history_data()
    config = make_config(
        buy_high=.003, share_high=300,
        strategy={"training_budget": 100, "validation_budget": 1000, "recommendation_count": 1, "stress_region_count": 0},
    )
    base = config.base.model_copy(update={"start_date": evaluation_start.date(), "end_date": data.calendar[-1].date()})
    config = config.model_copy(update={"base": base, "selection_target": "fixed"})
    monkeypatch.setattr(robust_research, "BatchEvaluator", SearchFakeEvaluator)

    result = select_robust_at(config, base, data, data.calendar[269])
    diagnostics = result["diagnostics"]
    region = result["recommendation_regions"][0]
    assert diagnostics["protocol_version"] == 5
    assert diagnostics["strategy_version"] == 5
    assert diagnostics["ranking_mode"] == "robust_j"
    assert diagnostics["return_basis"] == "peak_principal"
    assert diagnostics["ranking_baseline"] == "matched_funding_hold"
    assert diagnostics["risk_basis"] == "principal_profit_drawdown"
    assert diagnostics["training_only"] is True
    assert diagnostics["internal_validation_folds"] == 0
    assert diagnostics["statistics_applicability"]["status"] == "not_applicable"
    assert diagnostics["search_method"] == "exhaustive"
    assert diagnostics["global_exhaustive"] is True
    assert diagnostics["score_complete"] is True
    assert diagnostics["local_peak_verified"] is True
    assert diagnostics["coverage_status"] == "global_exhaustive"
    assert diagnostics["fields"] == ["buy_step", "sell_step", "buy_shares", "sell_shares"]
    assert region["geometry"] == "robust_j"
    assert region["scoring_basis"] == "robust_j_neighborhood"
    assert region["platform_type"] == "flat"
    assert region["platform_size"] == diagnostics["legal_combination_count"]
    assert result["recommended_params"] == region["center_params"]
    assert result["recommended_params"]["min_shares"] == 0
    assert result["recommended_params"]["max_shares"] is None
    center = next(
        item for item in result["candidate_archive"]
        if tuple(item["key"]) == tuple(region["center_coordinate"])
    )
    assert region["region_id"] in center["region_ids"]
    assert center["is_stable_point"] is True


def test_explicit_exhaustive_over_budget_fails_in_preflight(monkeypatch):
    data, evaluation_start = _long_history_data()
    config = make_config(
        buy_high=.003, share_high=300,
        strategy={"training_budget": 3, "validation_budget": 1000, "stress_region_count": 0},
    )
    base = config.base.model_copy(update={"start_date": evaluation_start.date(), "end_date": data.calendar[-1].date()})
    config = config.model_copy(update={"base": base})
    monkeypatch.setattr(robust_research, "BatchEvaluator", SearchFakeEvaluator)
    with pytest.raises(ValueError, match="完整枚举.*训练预算"):
        select_robust_at(config, base, data, data.calendar[269])


def test_exhaustive_peak_verification_reuses_training_cache(monkeypatch):
    data, evaluation_start = _long_history_data()
    config = make_config(
        buy_high=.003, share_high=300,
        strategy={
            "training_budget": 100, "validation_budget": 1,
            "recommendation_count": 1, "stress_region_count": 0,
        },
    )
    base = config.base.model_copy(update={
        "start_date": evaluation_start.date(), "end_date": data.calendar[-1].date(),
    })
    config = config.model_copy(update={"base": base, "selection_target": "fixed"})
    monkeypatch.setattr(robust_research, "BatchEvaluator", SearchFakeEvaluator)

    result = select_robust_at(config, base, data, data.calendar[269])
    assert result["diagnostics"]["search_method"] == "exhaustive"
    assert result["diagnostics"]["neighborhood_unique_count"] == 0
    assert result["recommendation_regions"]


def test_exhaustive_preflight_does_not_charge_cached_points_to_validation_budget():
    data, evaluation_start = _long_history_data()
    config = make_config(
        buy_high=.003, share_high=300,
        strategy={
            "training_budget": 100, "validation_budget": 1,
            "recommendation_count": 1, "stress_region_count": 0,
        },
    )
    base = config.base.model_copy(update={
        "start_date": evaluation_start.date(), "end_date": data.calendar[-1].date(),
    })
    preview = estimate_robust_search(config.model_copy(update={"base": base}), data)
    assert preview["status"] == "ready"
    assert preview["search_method"] == "exhaustive"


def test_exhaustive_scoring_can_be_complete_without_a_qualified_peak(monkeypatch):
    data, evaluation_start = _long_history_data()
    config = make_config(
        buy_high=.003, share_high=300,
        strategy={"training_budget": 100, "validation_budget": 1000, "recommendation_count": 1, "stress_region_count": 0},
    )
    base = config.base.model_copy(update={"start_date": evaluation_start.date(), "end_date": data.calendar[-1].date()})
    config = config.model_copy(update={"base": base, "selection_target": "fixed"})
    monkeypatch.setattr(robust_research, "BatchEvaluator", IneligibleSearchFakeEvaluator)
    result = select_robust_at(config, base, data, data.calendar[269])
    diagnostics = result["diagnostics"]
    assert result["recommended_params"] is None
    assert diagnostics["score_complete"] is True
    assert diagnostics["local_peak_verified"] is False
    assert diagnostics["global_exhaustive"] is True
    assert diagnostics["eligible_center_count"] == 0
