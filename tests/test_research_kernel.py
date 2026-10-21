"""Scientific evaluation must share the production ledger, including boundaries.

The v5 protocol scores every candidate on one peak-principal ledger:
``peak_principal_return``, ``max_principal_drawdown`` and the
``matched_funding_hold`` baseline.  The checks below keep the shared-kernel
guarantees that survive that contract change: research and production run on
the same ledger, corporate actions land on that ledger, the account stays
continuous across parameter changes, the interpreted kernel matches the
compiled one, and risk is measured on every bar instead of the session close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockregression import engine
from stockregression.models import BacktestConfig, DataBundle, FloorPenaltyConfig
from stockregression.providers import normalize_actions
from stockregression.rules import instrument_terms
from stockregression.share_engine import (
    research_batch, run_passive_backtest, run_share_backtest,
)

# Research and production must agree on these ledger fields exactly.  The
# production backtest also reports ``peak_invested``; the batch path keeps the
# same number in its funding statistics instead of in the metric row.
LEDGER_FIELDS = (
    "peak_principal_return", "max_principal_drawdown",
    "matched_funding_hold_return", "matched_funding_hold_drawdown",
    "initial_equity", "total_invested", "external_topups", "total_profit",
    "matched_funding_hold_invested", "trade_count", "buy_count", "sell_count",
    "total_fees", "turnover", "return_basis", "risk_basis",
)


def market(intraday=False):
    """A v5 share account: schema_version=3, geometric grid, no cash budget."""
    dates = pd.bdate_range("2024-01-02", periods=9)
    prices = np.array([10., 10., 9., 8.5, 10.5, 9.2, 11., 9.5, 10.])
    if intraday:
        prices[:] = 10.
    daily = pd.DataFrame({"date": dates, "open": prices, "close": prices,
        "high": prices + .01, "low": prices - .01, "preclose": np.r_[10., prices[:-1]],
        "volume": 100000, "turnover": prices * 100000, "tradestatus": 1,
        "is_st": False, "limit_up": 100., "limit_down": .01})
    bars = daily.iloc[1:].copy()
    bars["timestamp"] = bars.date + pd.Timedelta(hours=15)
    bars["frequency"], bars["tradable"], bars["is_synthetic"] = "1d", True, False
    if intraday:
        records = []
        for day in dates[1:]:
            for minute, price in enumerate([10., 12., 8., 10.]):
                records.append({"timestamp": day + pd.Timedelta(hours=9, minutes=31 + minute),
                    "open": price, "high": price, "low": price, "close": price,
                    "volume": 10000, "frequency": "1m", "tradable": True, "is_synthetic": False})
        bars = pd.DataFrame(records)
    config = BacktestConfig(schema_version=3, symbol="159611.SZ", start_date=dates[1].date(),
        end_date=dates[-1].date(), frequency="1m" if intraday else "1d", initial_shares=1000,
        min_shares=0, max_shares=None, grid_mode="geometric",
        fees={"commission_rate": 0, "min_commission": 0, "slippage_bps": 0})
    bundle = DataBundle(symbol=config.symbol, bars=bars, daily=daily, calendar=dates[1:],
        actions=normalize_actions(pd.DataFrame()), version="research-fixture",
        metadata=instrument_terms(config.symbol, {}), quality={"minute_ratio": 1.0})
    return config, bundle


def folds(bundle):
    return [(str(bundle.calendar[0].date()), str(bundle.calendar[3].date())),
            (str(bundle.calendar[4].date()), str(bundle.calendar[-1].date()))]


def session_close_indices(bundle):
    """Bar index of the last bar of every session, in session order."""
    stamps = pd.DatetimeIndex(bundle.bars["timestamp"])
    closes: dict[pd.Timestamp, int] = {}
    for index, stamp in enumerate(stamps):
        closes[stamp.normalize()] = index
    return list(closes.values())


def legacy_config(config):
    """The v2 contract kept a fixed cash budget; v3 funds buys on demand."""
    values = config.model_dump(mode="python")
    values.update(schema_version=2, reserve_cash=10000)
    return BacktestConfig.model_validate(values)


def test_research_batch_matches_full_ledger_and_peak_principal():
    config, bundle = market()
    params = [{"buy_step": .03}, {"buy_step": .06}]
    batched = research_batch(config, bundle, params, folds(bundle))
    serial = research_batch(config, bundle, params, folds(bundle), parallel=False)
    assert "daily_returns" not in batched
    assert batched["metrics"] == serial["metrics"]
    assert batched["fold_metrics"] == serial["fold_metrics"]
    for index, patch in enumerate(params):
        run = run_share_backtest(config.model_copy(update=patch), bundle)
        funding = run["funding"]
        grid = run["strategies"]["grid"]
        # v3 owns no cash budget: the account is funded from successful buys.
        assert funding["funding_mode"] == "top_up_on_fill"
        assert funding["reserve_cash"] == 0
        assert funding["initial_equity"] == funding["initial_principal"]
        assert grid["metrics"]["total_invested"] == (
            funding["initial_principal"] + funding["topups"])
        # Research and production must report one and the same ledger.
        for field in LEDGER_FIELDS:
            assert batched["metrics"][index][field] == grid["metrics"][field]
        # ... including the peak-principal curve and the per-bar risk.
        assert batched["metrics"][index]["peak_principal_return"] == pytest.approx(
            grid["equity"][-1]["principal_curve"] - 1)
        assert batched["metrics"][index]["max_principal_drawdown"] == pytest.approx(
            max(point["principal_drawdown"] for point in grid["equity"]))


def test_research_fold_drawdown_includes_intraday_path():
    config, bundle = market(intraday=True)
    config = config.model_copy(update={"min_shares": 1000, "max_shares": 1000})
    result = research_batch(config, bundle, [{}], folds(bundle))
    # The intraday path runs 10 -> 12 -> 8 -> 10 on a 1000-share account, so
    # the profit peaks at +2000 and troughs at -2000 while the peak principal
    # stays at the 10000 seed: a 4000 profit drawdown on 10000 of principal.
    np.testing.assert_allclose(result["fold_drawdowns"], .4)
    assert result["fold_metrics"][0][0]["trade_count"] == 0


def test_single_fold_measures_the_same_ledger_as_the_account():
    """A fold is a window on one continuous account, never a second account."""
    config, bundle = market()
    window = [(str(bundle.calendar[0].date()), str(bundle.calendar[-1].date()))]
    result = research_batch(config, bundle, [{}], window)
    fold = result["fold_metrics"][0][0]
    metrics = result["metrics"][0]
    run = run_share_backtest(config, bundle)
    assert fold["profit"] == pytest.approx(metrics["total_profit"], abs=1e-9)
    assert fold["principal_drawdown"] == pytest.approx(metrics["max_principal_drawdown"], abs=1e-12)
    assert fold["principal"] == pytest.approx(run["funding"]["peak_invested"])
    assert fold["end_assets"] == pytest.approx(
        run["strategies"]["grid"]["equity"][-1]["equity"], abs=1e-9)
    assert fold["trade_count"] == metrics["trade_count"]


@pytest.mark.parametrize("grid_mode", ["trailing", "arithmetic", "geometric"])
def test_identical_schedule_is_exactly_the_same_continuous_account(grid_mode):
    config, bundle = market()
    config = config.model_copy(update={"grid_mode": grid_mode})
    params = {"buy_step": .03, "sell_step": .03}
    schedule = [{"effective_date": str(bundle.calendar[index].date()),
                 "decision_date": str(bundle.calendar[index - 1].date()), "params": params}
                for index in (2, 4, 6)]
    plain = run_share_backtest(config, bundle)
    scheduled = run_share_backtest(config, bundle, parameter_schedule=schedule)
    assert plain["funding"] == scheduled["funding"]
    assert plain["strategies"] == scheduled["strategies"]


def test_parameter_change_replaces_pending_size_using_prior_close():
    config, bundle = market()
    schedule = [{"effective_date": str(bundle.calendar[2].date()),
                 "decision_date": str(bundle.calendar[1].date()), "params": {"buy_shares": 300}}]
    result = run_share_backtest(config, bundle, parameter_schedule=schedule)
    grid = result["strategies"]["grid"]
    first = grid["trades"][0]
    assert first["timestamp"][:10] == schedule[0]["effective_date"]
    assert first["quantity"] == 300
    # v3 owns no cash budget: the fill is funded by an external contribution
    # of exactly the traded value, and the account keeps no residual cash.
    assert first["external_topup"] == pytest.approx(300 * 8.5)
    assert first["cash"] == 0
    assert any(row["reason"] == "parameter_update" for row in grid["rejections"])
    assert not any(row["reason"] == "initial_position" for row in grid["trades"])


def test_daily_details_preserve_fills_and_exact_risk():
    config, bundle = market(intraday=True)
    full = run_share_backtest(config, bundle)
    compact = run_share_backtest(config, bundle, daily_details=True)
    closes = session_close_indices(bundle)
    for name, strategy in full["strategies"].items():
        daily = compact["strategies"][name]
        assert daily["metrics"] == strategy["metrics"]
        assert daily["trades"] == strategy["trades"]
        assert daily["rejections"] == strategy["rejections"]
        assert len(daily["equity"]) == len(bundle.calendar)
        assert len(strategy["equity"]) == len(bundle.bars)
        # The daily view is the same ledger, sampled once per session close.
        for point, index in zip(daily["equity"], closes):
            full_point = strategy["equity"][index]
            assert point["equity"] == full_point["equity"]
            assert point["cash"] == full_point["cash"]
            assert point["position"] == full_point["position"]
            assert point["total_invested"] == full_point["total_invested"]
            assert point["cumulative_profit"] == full_point["cumulative_profit"]
            assert point["principal_curve"] == full_point["principal_curve"]
            assert point["principal_drawdown"] == full_point["principal_drawdown"]
        # Risk is measured on every bar, not only at the session close.
        assert daily["metrics"]["max_principal_drawdown"] == max(
            point["principal_drawdown"] for point in strategy["equity"])
        assert daily["metrics"]["max_principal_drawdown"] > max(
            point["principal_drawdown"] for point in daily["equity"])


def test_passive_market_reference_carries_dividend_and_bonus_claims():
    config, bundle = market()
    bundle.actions = normalize_actions(pd.DataFrame([{
        "record_date": str(bundle.calendar[1].date()), "ex_date": str(bundle.calendar[2].date()),
        "pay_date": str(bundle.calendar[5].date()), "stock_date": str(bundle.calendar[5].date()),
        "cash_per_share": .6, "bonus_ratio": .1, "implemented": True,
    }]))
    passive = run_passive_backtest(config, bundle, daily_details=True)
    grid = passive["strategies"]["grid"]
    metrics = grid["metrics"]
    assert grid["trades"] == []
    # 1000 shares become 1100 after the 10% bonus and pay 600 in cash.
    assert grid["equity"][-1]["cash"] == 600
    assert grid["equity"][-1]["position"] == 1100
    assert metrics["peak_principal_return"] == pytest.approx(
        (grid["equity"][-1]["equity"] - metrics["total_invested"]) / metrics["peak_invested"])
    # Changing the rule while dividend/bonus rights are outstanding must retain
    # those exact claims rather than initialize a fresh account.
    frozen = config.model_copy(update={"min_shares": 1000, "max_shares": 1000})
    schedule = [{"effective_date": str(bundle.calendar[3].date()),
                 "decision_date": str(bundle.calendar[2].date()), "params": {"buy_shares": 200}}]
    before = run_share_backtest(frozen, bundle)
    changed = run_share_backtest(frozen, bundle, parameter_schedule=schedule)
    assert before["strategies"]["grid"] == changed["strategies"]["grid"]


def test_batch_refuses_changed_initial_account_and_cancel():
    config, bundle = market()
    with pytest.raises(ValueError, match="初始持股"):
        research_batch(config, bundle, [{"initial_shares": 2000}], [])
    with pytest.raises(InterruptedError):
        research_batch(config, bundle, [{}], [], cancelled=lambda: True)


def test_legacy_fixed_cash_budget_cannot_change_inside_a_batch():
    """Only the retired v2 contract owns a cash budget, and it stays fixed."""
    config, bundle = market()
    legacy = legacy_config(config)
    assert legacy.schema_version == 2
    with pytest.raises(ValueError, match="schema_version=3"):
        research_batch(legacy, bundle, [{"reserve_cash": 20000}], [])


def test_latest_parameters_replay_is_explicitly_retrospective():
    config, bundle = market()
    config = config.model_copy(update={"selection_asof": config.end_date})
    result = run_share_backtest(config, bundle)
    assert result["parameter_selection"]["retrospective"] is True
    assert any("样本内回看" in warning for warning in result["warnings"])


def test_batch_python_reference_matches_compiled(monkeypatch):
    config, bundle = market()
    config = config.model_copy(update={
        "floor_penalty_config": FloorPenaltyConfig(eta_floor=.5, gamma_min=.4, floor_k_ref=10),
    })
    fast = research_batch(config, bundle, [{}, {"sell_shares": 200}], folds(bundle))
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    slow = research_batch(config, bundle, [{}, {"sell_shares": 200}], folds(bundle))
    assert fast["metrics"] == slow["metrics"]
    assert fast["fold_metrics"] == slow["fold_metrics"]
    assert fast["position_stats"] == slow["position_stats"]
    assert all("floor_episode_grids" in row for row in fast["position_stats"])
