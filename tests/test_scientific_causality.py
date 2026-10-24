"""v5 研究路径的因果回归测试。

契约要点：

* 决策只使用决策日之前的信息：未来行情、截断快照或更长的历史都不得
  改变已经做出的决策；
* 每个决策的训练时段必须结束于决策日当天；
* 首轮自适应沿用样本外开始前最后一个收盘的既有估值，决策日严格早于
  生效日，账户本金按这个估值起始；
* 账户没有样本外之前的收盘时，本金估值回退到显式的既有估值锚点。

行情是确定性合成夹具，声明为测试生成数据，不代表任何真实标的或行情。
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from stockregression.models import (
    BacktestConfig, DataBundle, OptimizationConfig, ResearchStrategySpec,
)
from stockregression.research import robust as robust_research
from stockregression.research_v5 import optimize_v5
from stockregression.share_engine import funding


def _close_path(indices: np.ndarray) -> np.ndarray:
    return 10.0 + 0.65 * np.sin(indices * 0.17) + 0.002 * indices


def _daily_bundle(days: int, *, version: str = "synthetic-test-v1", index_offset: int = 0) -> DataBundle:
    calendar = pd.bdate_range("2022-01-03", periods=days)
    indices = np.arange(index_offset, index_offset + days, dtype=np.float64)
    close = _close_path(indices)
    previous = _close_path(indices - 1.0)
    open_price = previous * (1.0 + 0.0015 * np.sin(indices * 0.37))
    high = np.maximum(open_price, close) * 1.018
    low = np.minimum(open_price, close) * 0.982
    volume = (100_000 + (indices.astype(np.int64) % 17) * 1_000).astype(np.int64)
    turnover = close * volume

    daily = pd.DataFrame({
        "date": calendar,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "preclose": previous,
        "volume": volume,
        "turnover": turnover,
        "tradestatus": 1,
        "is_st": False,
    })
    bars = pd.DataFrame({
        "timestamp": calendar + pd.Timedelta(hours=15),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "preclose": previous,
        "volume": volume,
        "turnover": turnover,
        "frequency": "1d",
        "tradable": True,
        "is_synthetic": False,
        "is_st": False,
    })
    reports = [{"date": day.date().isoformat(), "mode": "daily"} for day in calendar]
    prior_date = (calendar[0] - pd.offsets.BDay(1)).date().isoformat()
    return DataBundle(
        symbol="159611.SZ",
        bars=bars,
        daily=daily,
        calendar=calendar,
        quality={
            "minute_ratio": 0.0,
            "minute_days": 0,
            "daily_days": days,
            "suspended_days": 0,
            "total_days": days,
            "days": reports,
        },
        version=version,
        metadata={
            "is_demo": False,
            "fixture_source": "deterministic synthetic regression test; not market data",
            "reference_date": prior_date,
            "valuation_price": float(previous[0]),
        },
    )


def _config(data: DataBundle, *, end_index: int | None = None, start_index: int = 504) -> OptimizationConfig:
    """v5 研究配置：四个搜索轴、完整枚举、单一训练时段。"""
    end_date = data.calendar[end_index].date() if end_index is not None else None
    base = BacktestConfig(
        schema_version=3,
        symbol=data.symbol,
        start_date=data.calendar[start_index].date(),
        end_date=end_date,
        frequency="1d",
        initial_shares=1_000,
        min_shares=0,
        max_shares=None,
        buy_shares=100,
        sell_shares=100,
        grid_mode="geometric",
        buy_step=0.02,
        sell_step=0.02,
    )
    strategy = ResearchStrategySpec.model_validate({
        "strategy_version": 5, "search_method": "exhaustive", "ranking_mode": "robust_j",
        "granularity": {"buy_step": 0.001, "sell_step": 0.001, "buy_shares": 100, "sell_shares": 100},
        "hypercube_edge": 3, "training_budget": 100, "validation_budget": 1_000,
        "reporting_budget": 100, "recommendation_count": 1, "stress_region_count": 0,
        "training_min_days": 250, "training_max_days": 500,
    })
    return OptimizationConfig(
        base=base,
        protocol_version=5,
        strategy=strategy,
        selection_target="both",
        rebalance_days=63,
        seed=91,
        min_trades=10,
        max_principal_drawdown=0.2,
        min_minute_ratio=0.0,
        ranges={
            # Keep the actual kernel work small while exercising candidate
            # generation, peak verification and the full rolling schedule.
            "buy_step": {"low": 0.010, "high": 0.012},
            "sell_step": {"low": 0.010, "high": 0.010},
            "buy_shares": {"low": 100, "high": 100},
            "sell_shares": {"low": 100, "high": 100},
        },
    )


def _mutate_after(data: DataBundle, cutoff: pd.Timestamp) -> DataBundle:
    changed = deepcopy(data)
    daily = changed.daily.copy(deep=True)
    bars = changed.bars.copy(deep=True)
    daily_dates = pd.to_datetime(daily["date"]).dt.normalize()
    future = daily_dates > pd.Timestamp(cutoff).normalize()
    previous_close = float(daily.loc[~future, "close"].iloc[-1])
    for row_index in daily.index[future]:
        day_number = int(row_index)
        old = daily.loc[row_index]
        scale = 1.20 + 0.0002 * day_number
        new_open = float(old["open"]) * scale
        new_close = float(old["close"]) * scale
        new_high = max(float(old["high"]) * scale, new_open, new_close)
        new_low = min(float(old["low"]) * scale, new_open, new_close)
        daily.loc[row_index, ["open", "high", "low", "close", "preclose"]] = [
            new_open, new_high, new_low, new_close, previous_close,
        ]
        previous_close = new_close

    bar_dates = pd.to_datetime(bars["timestamp"]).dt.normalize()
    bar_future = bar_dates > pd.Timestamp(cutoff).normalize()
    daily_by_day = daily.assign(_day=pd.to_datetime(daily["date"]).dt.normalize()).set_index("_day")
    for row_index in bars.index[bar_future]:
        day = bar_dates.loc[row_index]
        row = daily_by_day.loc[day]
        bars.loc[row_index, ["open", "high", "low", "close", "preclose", "volume", "turnover"]] = [
            row["open"], row["high"], row["low"], row["close"], row["preclose"], row["volume"], row["turnover"],
        ]
    changed.daily = daily
    changed.bars = bars
    changed.version = "synthetic-test-v2-mutated-after-asof"
    return changed


class WindowSensitiveEvaluator:
    """评价结果依赖训练时段末日。

    窗口一旦越过决策日，选出的参数就可能改变，因此未来数据泄漏会让
    下面「既有决策不变」的断言立刻失败。
    """

    chunk_size = 8
    retain_daily_returns = False

    def __init__(self, config, base, data, space, train_dates, *args, **kwargs):
        self.space = space
        self.train_dates = list(train_dates)

    def _factor(self) -> float:
        if not self.train_dates:
            return 0.0
        days = int(pd.Timestamp(self.train_dates[-1]).value // 86_400_000_000_000)
        return (days % 1000) / 1000.0

    def _excess(self, coordinate) -> float:
        factor = self._factor()
        # 窗口因子对坐标是非单调的：窗口变化可能改变名次。
        return 0.012 + 0.0004 * int(coordinate[0]) + 0.0012 * factor * (2 - int(coordinate[0]))

    def _row(self, coordinate) -> dict:
        return {
            "key": tuple(coordinate), "coordinate": list(coordinate),
            "params": self.space.params(coordinate), "evaluation_status": "completed",
            "training_excess_return": self._excess(coordinate),
            "training_return": self._excess(coordinate), "training_baseline_return": 0.0,
            "training_baseline_kind": "matched_funding_hold",
            "max_principal_drawdown": 0.05, "trade_count": 20,
            "buy_count": 10, "sell_count": 10,
            "actual_max_pos_ratio": 1.0, "actual_min_pos_ratio": 0.5,
            "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
        }

    def evaluate_training(self, coordinates):
        return [self._row(coordinate) for coordinate in coordinates]

    def evaluate_training_stress(self, coordinates, cost_multiplier=2.0):
        return {
            tuple(coordinate): {
                "training_excess_stress": self._excess(coordinate) - 0.001 * float(cost_multiplier),
                "max_principal_drawdown": 0.05, "trade_count": 20,
                "actual_min_pos_ratio": 0.5, "actual_max_pos_ratio": 1.0,
                "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
            }
            for coordinate in coordinates
        }


def _past_decisions(result: dict, cutoff: pd.Timestamp) -> list[dict]:
    limit = pd.Timestamp(cutoff).normalize()
    return [
        row for row in result["evidence"]["decisions"]
        if row.get("decision_date") and pd.Timestamp(row["decision_date"]).normalize() <= limit
    ]


def _assert_same_decisions(left: dict, right: dict, cutoff: pd.Timestamp) -> None:
    left_rows = _past_decisions(left, cutoff)
    right_rows = _past_decisions(right, cutoff)
    assert len(left_rows) == len(right_rows) >= 3
    for first, second in zip(left_rows, right_rows, strict=True):
        assert first["effective_date"] == second["effective_date"]
        assert first["decision_date"] == second["decision_date"]
        assert first["selected_params"] == second["selected_params"]
        assert first["reused_from_fixed"] == second["reused_from_fixed"]
        if first["score"] is None or second["score"] is None:
            assert first["score"] is second["score"] is None
        else:
            assert first["score"] == pytest.approx(second["score"], abs=1e-12)
        # 决策所用的训练时段必须逐字相同，并且结束于决策日当天。
        first_window = first["diagnostics"]["training_window"]
        second_window = second["diagnostics"]["training_window"]
        assert first_window["start"] == second_window["start"]
        assert first_window["end"] == second_window["end"]
        assert first_window["days"] == second_window["days"]
        assert pd.Timestamp(first_window["end"]) <= pd.Timestamp(first["decision_date"])


def _prefix(test_result: dict, cutoff: pd.Timestamp) -> dict:
    limit = pd.Timestamp(cutoff).normalize()
    grid = test_result["strategies"]["grid"]
    return {
        key: [
            deepcopy(row) for row in grid[key]
            if pd.Timestamp(row["timestamp"]).normalize() <= limit
        ]
        for key in ("trades", "equity")
    }


def _assert_same_past_path(left: dict, right: dict, cutoff: pd.Timestamp) -> None:
    for procedure in ("fixed", "adaptive"):
        assert _prefix(left[procedure]["test_result"], cutoff) == _prefix(right[procedure]["test_result"], cutoff)


def test_v5_decisions_are_invariant_to_future_data_and_snapshot_boundaries(monkeypatch):
    monkeypatch.setattr(robust_research, "BatchEvaluator", WindowSensitiveEvaluator)
    data = _daily_bundle(720)
    # The fourth 63-session decision is made at this close and becomes active
    # on the next session. It is an as-of cutoff for every past-path check.
    eval_start = 504
    cutoff_index = eval_start + 189 - 1
    cutoff = pd.Timestamp(data.calendar[cutoff_index])
    next_effective_index = cutoff_index + 1

    full = optimize_v5(_config(data), data)
    shortened = optimize_v5(_config(data, end_index=next_effective_index), data)
    mutated_data = _mutate_after(data, cutoff)
    mutated = optimize_v5(_config(mutated_data), mutated_data)
    extended_data = _daily_bundle(740)
    extended = optimize_v5(_config(extended_data), extended_data)

    for variant in (shortened, mutated, extended):
        _assert_same_decisions(full, variant, cutoff)
        _assert_same_past_path(full, variant, cutoff)

    fixed = full["fixed"]
    adaptive = full["adaptive"]
    current = fixed["current"]
    assert current["as_of"] == fixed["as_of"]
    assert current["diagnostics"]["training_window"] == fixed["diagnostics"]["training_window"]
    assert pd.Timestamp(current["diagnostics"]["training_window"]["end"]) < pd.Timestamp(
        full["config"]["base"]["start_date"]
    )
    # 各个决策的训练时段与分数必须互不相同，否则上面的不变性断言会退化为
    # 空断言：真正泄漏未来数据时，决策会随窗口一起改变。
    windows = [row["diagnostics"]["training_window"]["end"] for row in full["evidence"]["decisions"]]
    scores = [row["score"] for row in full["evidence"]["decisions"]]
    assert len(set(windows)) == len(windows)
    assert len(set(scores)) == len(scores)
    assert fixed["test_result"] is not None
    assert adaptive["test_result"] is not None
    fixed_config = fixed["test_result"]["config"]
    assert fixed_config["initial_shares"] == 1_000
    for name, value in fixed["params"].items():
        assert fixed_config[name] == value
    assert all(row["params"]["initial_shares"] == 1_000 for row in adaptive["schedule"])
    assert adaptive["test_result"]["parameter_schedule"] == adaptive["schedule"]

    # v5 的样本外账本以峰值本金口径给出，并与配对的资金持有基准比较。
    metrics = adaptive["test_result"]["strategies"]["grid"]["metrics"]
    assert metrics["return_basis"] == "peak_principal"
    assert metrics["risk_basis"] == "principal_profit_drawdown"
    assert metrics["matched_funding_hold_return"] is not None
    assert full["outer_validation"]["status"] == "not_applicable"


def test_v5_current_recommendation_uses_evaluation_start_without_a_fixed_branch(monkeypatch):
    monkeypatch.setattr(robust_research, "BatchEvaluator", WindowSensitiveEvaluator)
    data = _daily_bundle(720)
    eval_start = 504
    config = _config(data).model_copy(update={
        "selection_target": "adaptive", "rebalance_days": 10_000,
    })

    result = optimize_v5(config, data)

    current = result["fixed"]["current"]
    expected_asof = data.calendar[eval_start - 1].date().isoformat()
    assert current["as_of"] == expected_asof
    assert current["diagnostics"]["training_window"]["end"] == expected_asof
    assert pd.Timestamp(expected_asof) < pd.Timestamp(result["config"]["base"]["start_date"])


def test_v5_first_adaptive_decision_uses_the_existing_valuation_anchor(monkeypatch):
    monkeypatch.setattr(robust_research, "BatchEvaluator", WindowSensitiveEvaluator)
    data = _daily_bundle(720)
    eval_start = 504

    result = optimize_v5(_config(data), data)

    first = result["adaptive"]["schedule"][0]
    decision = result["evidence"]["decisions"][0]
    assert first["effective_date"] == data.calendar[eval_start].date().isoformat()
    # The first adaptive round decides at the last close before the
    # out-of-sample window starts, and reuses that selection as is.
    assert first["decision_date"] == data.calendar[eval_start - 1].date().isoformat()
    assert pd.Timestamp(first["decision_date"]) < pd.Timestamp(first["effective_date"])
    assert decision["reused_from_fixed"] is True
    assert decision["selected_params"] == result["fixed"]["params"]
    assert first["params"] == decision["selected_params"]

    # The account is funded with that existing valuation, not with a close
    # from inside the out-of-sample window.
    money = result["adaptive"]["test_result"]["funding"]
    valuation_close = float(data.daily.loc[data.daily["date"] == data.calendar[eval_start - 1], "close"].iloc[0])
    first_oos_close = float(data.daily.loc[data.daily["date"] == data.calendar[eval_start], "close"].iloc[0])
    assert money["reference_date"] == first["decision_date"]
    assert money["valuation_price"] == pytest.approx(valuation_close)
    assert money["valuation_price"] != pytest.approx(first_oos_close)
    assert money["initial_principal"] == pytest.approx(1_000 * valuation_close)


def test_the_causality_fixture_reacts_to_the_training_window():
    """夹具本身必须随训练时段变化，否则因果断言没有鉴别力。"""
    class _Space:
        def params(self, coordinate):
            return {}

    def leader_for(day: str) -> int:
        evaluator = WindowSensitiveEvaluator(None, None, None, _Space(), [pd.Timestamp(day)], [])
        candidates = list(range(1, 4))
        return max(candidates, key=lambda coordinate: (evaluator._excess((coordinate,)), -coordinate))

    assert leader_for("2022-01-08") != leader_for("2022-01-03")


def test_v5_funding_falls_back_to_the_explicit_valuation_anchor_without_bundle_history():
    """账户没有样本外之前的收盘时，本金估值回退到显式的既有估值锚点。"""
    data = _daily_bundle(30)
    start = pd.Timestamp(data.calendar[0])
    reference_date = (start - pd.offsets.BDay(1)).date().isoformat()
    valuation_price = float(data.daily.iloc[0]["preclose"])
    data.metadata["reference_date"] = reference_date
    data.metadata["valuation_price"] = valuation_price
    assert not (pd.DatetimeIndex(data.calendar) < start).any()

    base = BacktestConfig(
        schema_version=3, symbol=data.symbol, start_date=start.date(), frequency="1d",
        initial_shares=1_000, min_shares=0, max_shares=None, buy_shares=100, sell_shares=100,
        grid_mode="geometric", buy_step=0.02, sell_step=0.02,
    )
    money = funding(base, data)

    assert money["reference_date"] == reference_date
    assert money["valuation_price"] == pytest.approx(valuation_price)
    assert money["initial_principal"] == pytest.approx(1_000 * valuation_price)
    assert money["initial_equity"] == pytest.approx(1_000 * valuation_price)
