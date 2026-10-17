"""Causal regression checks for the real scientific and share-kernel path.

The bars below are a deterministic synthetic fixture with populated market
fields. They are deliberately identified as test-generated data and make no
claim to represent an actual security or market history.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockregression.models import BacktestConfig, DataBundle, OptimizationConfig, SearchRange
from stockregression.scientific import optimize_scientific


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
    end_date = data.calendar[end_index].date() if end_index is not None else None
    base = BacktestConfig(
        schema_version=2,
        symbol=data.symbol,
        start_date=data.calendar[start_index].date(),
        end_date=end_date,
        frequency="1d",
        initial_shares=1_000,
        min_shares=0,
        max_shares=2_000,
        buy_shares=100,
        sell_shares=100,
        reserve_cash=10_000,
    )
    return OptimizationConfig(
        base=base,
        protocol_version=2,
        selection_target="both",
        n_trials=4,
        seed=91,
        rebalance_days=63,
        bootstrap_reps=100,
        min_trades=0,
        max_drawdown=0.95,
        min_minute_ratio=0.0,
        ranges={
            # Keep the actual kernel work small while exercising candidate
            # generation, frozen validation and the full rolling schedule.
            "buy_step": SearchRange(low=0.02, high=0.04, step=0.01),
            "min_shares": SearchRange(low=0, high=0),
            "max_shares": SearchRange(low=2_000, high=2_000),
            "buy_shares": SearchRange(low=100, high=100),
            "sell_shares": SearchRange(low=100, high=100),
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


def _past_decisions(result: dict, cutoff: pd.Timestamp) -> list[dict]:
    limit = pd.Timestamp(cutoff).normalize()
    return [
        row for row in result["evidence"]["decisions"]
        if row.get("decision_date") and pd.Timestamp(row["decision_date"]).normalize() <= limit
    ]


def _assert_same_decisions(left: dict, right: dict, cutoff: pd.Timestamp) -> None:
    left_fixed = left["evidence"]["fixed_decision"]
    right_fixed = right["evidence"]["fixed_decision"]
    assert left_fixed["decision_date"] == right_fixed["decision_date"]
    assert left_fixed["selected_params"] == right_fixed["selected_params"]
    if left_fixed.get("score") is None or right_fixed.get("score") is None:
        assert left_fixed.get("score") is right_fixed.get("score") is None
    else:
        assert left_fixed["score"] == pytest.approx(right_fixed["score"], abs=1e-12)

    left_rows = _past_decisions(left, cutoff)
    right_rows = _past_decisions(right, cutoff)
    assert len(left_rows) == len(right_rows) >= 3
    for first, second in zip(left_rows, right_rows, strict=True):
        assert first["effective_date"] == second["effective_date"]
        assert first["decision_date"] == second["decision_date"]
        assert first["selected_params"] == second["selected_params"]
        if first["score"] is None or second["score"] is None:
            assert first["score"] is second["score"] is None
        else:
            assert first["score"] == pytest.approx(second["score"], abs=1e-12)


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


def test_scientific_history_is_invariant_to_future_data_and_snapshot_boundaries():
    data = _daily_bundle(720)
    # The third 63-session decision is made at this close and becomes active
    # on the next session. It is an as-of cutoff for every past-path check.
    eval_start = 504
    cutoff_index = eval_start + 126 - 1
    cutoff = pd.Timestamp(data.calendar[cutoff_index])
    next_effective_index = cutoff_index + 1

    full = optimize_scientific(_config(data), data)
    shortened = optimize_scientific(_config(data, end_index=next_effective_index), data)
    mutated_data = _mutate_after(data, cutoff)
    mutated = optimize_scientific(_config(mutated_data), mutated_data)
    extended_data = _daily_bundle(740)
    extended = optimize_scientific(_config(extended_data), extended_data)

    for variant in (shortened, mutated, extended):
        _assert_same_decisions(full, variant, cutoff)
        _assert_same_past_path(full, variant, cutoff)

    fixed = full["fixed"]
    adaptive = full["adaptive"]
    assert fixed["test_result"] is not None
    assert adaptive["test_result"] is not None
    fixed_config = fixed["test_result"]["config"]
    assert fixed_config["initial_shares"] == 1_000
    for name, value in fixed["params"].items():
        assert fixed_config[name] == value
    assert all(row["params"]["initial_shares"] == 1_000 for row in adaptive["schedule"])
    assert adaptive["test_result"]["parameter_schedule"] == adaptive["schedule"]

    locked = full["locked_funding"]
    assert locked["reserve_cash"] == fixed["test_result"]["funding"]["reserve_cash"]
    assert locked["reserve_cash"] == adaptive["test_result"]["funding"]["reserve_cash"]
    assert locked["initial_equity"] == fixed["test_result"]["funding"]["initial_equity"]
    assert locked["initial_equity"] == adaptive["test_result"]["funding"]["initial_equity"]
    assert fixed["test_result"]["funding"]["initial_shares"] == 1_000
    assert adaptive["test_result"]["funding"]["initial_shares"] == 1_000
    assert fixed_config["initial_shares"] == 1_000


def test_first_adaptive_decision_uses_valid_prior_valuation_close_without_bundle_history():
    data = _daily_bundle(70, index_offset=504)
    start = pd.Timestamp(data.calendar[0])
    reference_date = (start - pd.offsets.BDay(1)).date().isoformat()
    # The DataBundle begins on the OOS start date. The only prior close is
    # the explicit, valid funding valuation anchor in metadata.
    data.metadata["reference_date"] = reference_date
    data.metadata["valuation_price"] = float(data.daily.iloc[0]["preclose"])
    assert not (pd.DatetimeIndex(data.calendar) < start).any()

    result = optimize_scientific(_config(data, start_index=0, end_index=10), data)

    first = result["adaptive"]["schedule"][0]
    assert first["effective_date"] == start.date().isoformat()
    assert first["decision_date"] == reference_date
    assert pd.Timestamp(first["decision_date"]) < pd.Timestamp(first["effective_date"])
    assert first["params"]["initial_shares"] == 1_000
    assert result["adaptive"]["test_result"] is not None
