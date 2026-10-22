"""Protocol facts must have one implementation shared by every public surface."""

from __future__ import annotations

import pandas as pd

from stockregression.models import (
    BacktestConfig, DataBundle, OptimizationConfig, research_protocol_description,
)
from stockregression.research.robust import estimate_robust_search, select_robust_at
from stockregression.research import strategy_descriptions
from stockregression.research_v5 import optimize_v5


def _config(**strategy_updates) -> OptimizationConfig:
    config = OptimizationConfig(
        base={"schema_version": 3, "symbol": "600000.SH", "start_date": "2024-01-02"},
    )
    if strategy_updates:
        strategy = config.strategy.model_copy(update=strategy_updates)
        config = config.model_copy(update={"strategy": strategy})
    return config


def test_strategy_catalog_and_result_protocol_share_the_same_fact_source():
    config = _config()
    catalog = strategy_descriptions()["strategies"][0]
    protocol = config.protocol_description()

    assert protocol == research_protocol_description(
        config.strategy, config.selection_target,
        max_principal_drawdown=config.max_principal_drawdown,
        min_trades=config.min_trades,
    )
    for name in (
        "protocol_version", "result_schema_version", "strategy_id", "strategy_version",
        "ranking_mode", "training", "out_of_sample", "return_definition",
        "risk_definition", "ranking_baseline", "ranking_basis", "j_formula",
        "j_inputs", "eligibility", "statistics", "budgets",
    ):
        assert catalog[name] == protocol[name]


def test_protocol_declares_one_training_block_and_no_internal_significance_claim():
    protocol = _config(training_window_mode="same_price").protocol_description()

    assert protocol["training"]["single_block"] is True
    assert protocol["training"]["internal_validation_folds"] == 0
    assert protocol["training"]["same_price_rule"] == "daily_low_lt_anchor_close_lt_daily_high"
    assert protocol["ranking_baseline"] == "matched_funding_hold"
    assert protocol["ranking_basis"] == "peak_principal_return_difference"
    assert protocol["statistics"]["status"] == "not_applicable"
    assert "显著性" in protocol["statistics"]["reason"]
    assert "matched_funding_hold" in protocol["j_inputs"]["center_excess"]


def test_fixed_window_descriptor_matches_the_fixed_implementation():
    config = _config(training_window_mode="fixed", training_min_days=17, training_max_days=42)
    training = config.protocol_description()["training"]

    assert training["mode"] == "fixed"
    assert training["match_rule"] == "fixed_training_max_days"
    assert training["max_days"] == 42
    assert training["fallback"] is None
    assert training["insufficient_history"] == "data_issue"
    assert "same_price_rule" not in training
    assert "min_days" not in training


def test_protocol_derives_eligibility_constraints_from_the_configuration():
    config = _config(eta_hard_cap=1.75)
    config = config.model_copy(update={"max_principal_drawdown": .17, "min_trades": 23})
    eligibility = config.protocol_description()["eligibility"]

    assert eligibility == {
        "min_j_score": 0.0,
        "j_score_is_exclusive": True,
        "max_principal_drawdown": .17,
        "min_trades": 23,
        "require_two_way_trades": True,
        "eta_hard_cap": 1.75,
    }


def _research_bundle(*, implemented: bool = False, days: int = 60) -> DataBundle:
    """Daily fixture whose only corporate action is switched on and off."""
    dates = pd.bdate_range("2024-01-02", periods=days)
    close = [20.0] * days
    daily = pd.DataFrame({
        "date": dates, "open": close, "high": [21.0] * days, "low": [19.0] * days,
        "close": close, "preclose": close, "volume": 100000, "turnover": [2000000.0] * days,
        "tradestatus": 1, "is_st": False,
    })
    actions = pd.DataFrame([{
        "record_date": dates[5], "ex_date": dates[10], "pay_date": dates[10],
        "stock_date": dates[10], "cash_per_share": 1.0, "bonus_ratio": 0.0,
        "implemented": implemented, "unsupported_reason": "",
    }])
    return DataBundle(
        symbol="600000.SH", bars=pd.DataFrame(), daily=daily, actions=actions,
        calendar=pd.DatetimeIndex(dates), quality={}, version="v5-protocol-test",
        metadata={}, warnings=[],
    )


def _research_config(dates: pd.DatetimeIndex) -> OptimizationConfig:
    base = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date=dates[20].date(),
        end_date=dates[-1].date(), frequency="1d", initial_shares=1000, min_shares=0,
        max_shares=None, buy_shares=100, sell_shares=100, grid_mode="geometric",
        buy_step=0.01, sell_step=0.01,
    )
    return OptimizationConfig(base=base, protocol_version=5, selection_target="fixed")


def test_window_data_issue_is_reported_as_its_own_status():
    """训练窗口的数据问题必须以独立状态上报，不得混同为历史不足。"""
    data = _research_bundle(implemented=False)
    dates = pd.DatetimeIndex(data.calendar)
    config = _research_config(dates)

    selection = select_robust_at(config, config.base, data, dates[-1])

    assert selection["status"] == "data_issue"
    assert selection["recommended_params"] is None
    diagnostics = selection["diagnostics"]
    assert diagnostics["status"] == "data_issue"
    assert diagnostics["data_issue"]
    window = diagnostics["training_window"]
    assert window["mode"] == "same_price"
    assert window["data_issue"]
    assert window["days"] == 0
    assert window["action_factor_fingerprint"] is None

    estimate = estimate_robust_search(config, data)
    assert estimate["status"] == "data_issue"
    assert estimate["training_window"]["data_issue"]

    result = optimize_v5(config, data)
    assert result["status"] == "data_issue"
    assert result["data_issue"]
    assert result["fixed"]["status"] == "data_issue"
    assert result["fixed"]["diagnostics"]["training_window"]["data_issue"]


def test_genuinely_short_history_keeps_its_own_status():
    """历史确实不足时仍然报告 insufficient_history，不得被数据问题状态吞掉。"""
    data = _research_bundle(implemented=True)
    dates = pd.DatetimeIndex(data.calendar)
    config = _research_config(dates)

    selection = select_robust_at(config, config.base, data, dates[-1])

    assert selection["status"] == "insufficient_history"
    assert selection["diagnostics"]["status"] == "insufficient_history"
    assert selection["diagnostics"]["data_issue"] is None
    assert selection["diagnostics"]["training_window"]["data_issue"] is None

    estimate = estimate_robust_search(config, data)
    assert estimate["status"] == "insufficient_history"

    result = optimize_v5(config, data)
    assert result["status"] == "partial"
    assert result["data_issue"] is None
