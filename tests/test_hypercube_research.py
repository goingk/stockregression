from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockregression.models import BacktestConfig, DataBundle, OptimizationConfig
from stockregression.research_v4.hypercube_selector import select_hypercube_at
from stockregression.research_v4.selector import estimate_search
from stockregression.research_v4.space import ParameterSpace
from stockregression.research_v4.window import training_window


def _bundle(days=560, prices=None, band=0.05):
    dates = pd.bdate_range("2024-01-01", periods=days)
    values = np.full(days, 11.0) if prices is None else np.asarray(prices, dtype=float)
    daily = pd.DataFrame({
        "date": dates, "open": values, "close": values,
        "high": values + band, "low": values - band,
        "volume": 100000, "preclose": np.r_[values[0], values[:-1]], "tradestatus": 1,
    })
    bars = daily.iloc[1:].copy()
    bars["timestamp"] = bars.date + pd.Timedelta(hours=15)
    bars["frequency"] = "1d"
    bars["tradable"] = True
    return DataBundle(
        symbol="600000.SH", daily=daily, bars=bars, calendar=dates[1:],
        metadata={"prior_closes": {str(day.date()): float(price) for day, price in zip(dates, values)}},
        quality={"minute_ratio": 1.0},
    )


def _config(data, *, strategy=None, start_index=260):
    base = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date=data.calendar[start_index].date(),
        end_date=data.calendar[-1].date(), frequency="1d", initial_shares=1000,
        min_shares=0, max_shares=2000, buy_shares=100, sell_shares=100,
        grid_mode="geometric", buy_step=.0003, sell_step=.0003,
    )
    ranges = {
        "buy_step": {"low": .0001, "high": .0005},
        "sell_step": {"low": .0003, "high": .0003},
        "sell_shares": {"low": 100, "high": 500},
        "buy_shares": {"low": 100, "high": 100},
        "min_shares": {"low": 0, "high": 0},
        "max_shares": {"low": 2000, "high": 2000},
    }
    return OptimizationConfig(
        base=base, protocol_version=4, selection_target="fixed", ranges=ranges,
        strategy={
            "strategy_version": 3, "search_method": "exhaustive",
            "training_budget": 1000, "validation_budget": 1000,
            "hypercube_edge": 3, **(strategy or {}),
        },
    )


def test_training_window_uses_farthest_same_price_day():
    prices = np.full(600, 11.0)
    anchor = 550
    prices[[51, 101, 301, anchor]] = 10.0
    data = _bundle(days=600, prices=prices)
    window = training_window(data, data.calendar[anchor - 1])
    assert window.matched is True
    assert window.fallback is False
    assert window.start == pd.Timestamp(data.daily.iloc[51].date)
    assert window.end == pd.Timestamp(data.daily.iloc[anchor].date)
    assert window.days == 500
    assert window.diagnostic()["match_rule"] == "daily_low_lt_anchor_close_lt_daily_high"
    assert window.diagnostic()["start_reason"] == "same_price_farthest"


def test_training_window_falls_back_to_500_days_without_same_price():
    prices = np.full(600, 11.0)
    anchor = 550
    prices[anchor] = 10.0
    data = _bundle(days=600, prices=prices)
    window = training_window(data, data.calendar[anchor - 1])
    assert window.matched is False
    assert window.fallback is True
    assert window.days == 500
    assert window.start == pd.Timestamp(data.daily.iloc[51].date)


def test_training_window_requires_strict_low_and_high_bracketing():
    prices = np.full(600, 11.0)
    anchor = 550
    prices[anchor] = 10.0
    data = _bundle(days=600, prices=prices)
    data.daily.loc[301, "high"] = 10.0
    data.daily.loc[301, "low"] = 9.95
    data.daily.loc[101, "low"] = 10.0
    data.daily.loc[101, "high"] = 10.05
    window = training_window(data, data.calendar[anchor - 1])
    assert window.matched is False
    assert window.fallback is True


def test_research_granularity_is_independent_of_minimum_parameter_unit():
    data = _bundle(days=320)
    value = _config(data)
    value = value.model_copy(update={"ranges": {
        **value.ranges,
        "buy_step": {"low": .003, "high": .005},
        "sell_step": {"low": .003, "high": .003},
        "min_shares": {"low": 0, "high": 1000},
        "max_shares": {"low": 1000, "high": 1000},
    }})
    fine = value.strategy.model_copy(update={"granularity": {
        **value.strategy.granularity, "buy_step": .0005, "min_shares": 100,
    }})
    fine = value.model_copy(update={"strategy": fine})
    coarse = value.model_copy(update={"strategy": value.strategy.model_copy(update={"granularity": {
        **value.strategy.granularity, "buy_step": .001, "min_shares": 500,
    }})})
    fine_space, coarse_space = ParameterSpace(fine), ParameterSpace(coarse)
    assert fine_space.value("buy_step", 6) == .003
    assert fine_space.dimensions["buy_step"].low == 6
    assert fine_space.dimensions["buy_step"].count > coarse_space.dimensions["buy_step"].count
    assert fine_space.dimensions["min_shares"].count > coarse_space.dimensions["min_shares"].count
    # The minimum unit remains 0.0001 and 100 shares; only the sampling stride changed.
    assert fine_space.value("buy_step", 1) == .0005
    assert fine_space.value("min_shares", 1) == 100


def test_hypercube_selection_uses_one_training_block_and_mean_excess():
    data = _bundle(days=320)
    value = _config(data)
    selected = select_hypercube_at(value, value.base, data, data.calendar[259])
    diagnostics = selected["diagnostics"]
    assert selected["status"] == "completed"
    assert diagnostics["strategy_version"] == 3
    assert diagnostics["geometry"] == "hypercube_mean"
    assert diagnostics["training_only"] is True
    assert diagnostics["validation_folds"] == []
    assert diagnostics["selection_history_days"] >= 250
    assert diagnostics["box_count"] > 0
    region = selected["recommendation_regions"][0]
    assert region["edge_length"] == 3
    assert region["combination_count"] == 9
    members = [
        row for row in selected["candidate_archive"]
        if region["region_id"] in (row.get("region_ids") or [])
    ]
    assert len(members) == 9
    assert region["mean_excess_return"] == pytest.approx(
        float(np.mean([row["training_excess_return"] for row in members]))
    )


def test_hypercube_preflight_reports_single_training_window():
    data = _bundle(days=400)
    value = _config(data)
    preview = estimate_search(value, data)
    assert preview["status"] == "ready"
    assert preview["strategy_version"] == 3
    assert preview["hypercube_volume"] == 9
    assert preview["validation_days"] == 0
    assert preview["training_window"]["days"] >= 250
    assert preview["training_window"]["max_offset_trading_days"] == 500


def test_hypercube_preflight_rejects_edge_larger_than_a_search_range():
    data = _bundle(days=400)
    value = _config(data)
    value = value.model_copy(update={
        "strategy": value.strategy.model_copy(update={"hypercube_edge": 7})
    })
    preview = estimate_search(value, data)
    assert preview["status"] == "insufficient_range"
    assert preview["minimum_neighborhood_possible"] is False
