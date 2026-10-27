import numpy as np
import pandas as pd
import pytest

from stockregression import engine
from stockregression.models import BacktestConfig, DataBundle, FeeConfig
from stockregression.providers import normalize_actions
from stockregression.rules import instrument_terms
from stockregression.share_engine import run_qualification
from stockregression.validation_models import QualificationAccount


def _bundle(
    prices,
    *, symbol="600000.SH", frequency="1d", volumes=None,
    bars=None, actions=None, precloses=None,
):
    prices = np.asarray(prices, dtype=float)
    dates = pd.bdate_range("2024-01-02", periods=len(prices))
    volumes = np.full(len(prices), 100_000.0) if volumes is None else np.asarray(volumes, dtype=float)
    precloses = (
        np.r_[prices[0], prices[:-1]] if precloses is None
        else np.asarray(precloses, dtype=float)
    )
    daily = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": prices + .01,
        "low": prices - .01,
        "close": prices,
        "preclose": precloses,
        "volume": volumes,
        "turnover": prices * volumes,
        "tradestatus": 1,
        "is_st": False,
        "limit_up": prices * 1.30,
        "limit_down": prices * .70,
    })
    if bars is None:
        bars = daily.iloc[1:].copy()
        bars["timestamp"] = bars.date + pd.Timedelta(hours=15)
        bars["frequency"] = frequency
        bars["tradable"] = True
        bars["is_synthetic"] = False
    bundle = DataBundle(
        symbol=symbol,
        bars=pd.DataFrame(bars).reset_index(drop=True),
        daily=daily,
        actions=normalize_actions(pd.DataFrame() if actions is None else pd.DataFrame(actions)),
        calendar=pd.DatetimeIndex(dates[1:]),
        metadata=instrument_terms(symbol, {
            "reference_date": str(dates[0].date()),
            "valuation_price": float(prices[0]),
        }),
        version="qualification-fixture",
    )
    return bundle


def _config(bundle, *, initial_shares=0, max_shares=2000, buy_shares=100, sell_shares=100,
            buy_step=.01, sell_step=.01, frequency="1d", fees=None):
    return BacktestConfig(
        schema_version=3,
        symbol=bundle.symbol,
        start_date=bundle.calendar[0].date(),
        end_date=bundle.calendar[-1].date(),
        frequency=frequency,
        initial_shares=initial_shares,
        min_shares=0,
        max_shares=max_shares,
        buy_shares=buy_shares,
        sell_shares=sell_shares,
        grid_mode="trailing",
        buy_step=buy_step,
        sell_step=sell_step,
        reference_price_mode="manual",
        reference_price=10.0,
        fees=fees or FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
    )


def _intraday_bars(date, closes, volumes):
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    timestamps = [date + pd.Timedelta(hours=hour) for hour in (10, 11, 14, 15)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes + .01,
        "low": closes - .01,
        "close": closes,
        "preclose": np.r_[10.0, closes[:-1]],
        "volume": volumes,
        "turnover": closes * volumes,
        "frequency": "60m",
        "tradable": True,
        "is_synthetic": False,
        "limit_up": closes * 1.30,
        "limit_down": closes * .70,
    })


def test_initial_cash_and_contributions_are_identical_but_deployed_independently():
    prices = [10, 10, 10, 10, 10]
    bundle = _bundle(prices)
    config = _config(bundle, initial_shares=1000, max_shares=3000, buy_shares=100)
    account = QualificationAccount(
        total_budget=20_000,
        max_shares=3000,
        contributions=[{"available_at": "2024-01-04T00:00:00", "amount": 1000}],
    )

    result = run_qualification(config, bundle, account)

    assert result["status"] == "completed"
    assert result["qualification_status"] == "qualified"
    assert result["initial_cash"] == pytest.approx(9000)
    assert result["external_topups"] == pytest.approx(1000)
    grid = result["strategies"]["grid"]
    baseline = result["strategies"]["scheduled_funding_hold"]
    assert grid["metrics"]["initial_cash"] == baseline["metrics"]["initial_cash"] == pytest.approx(9000)
    assert grid["metrics"]["external_topups"] == baseline["metrics"]["external_topups"] == pytest.approx(1000)
    assert [row["amount"] for row in grid["contributions"]] == [1000]
    assert [row["amount"] for row in baseline["contributions"]] == [1000]
    assert baseline["trades"][0]["timestamp"] == bundle.bars.timestamp.iloc[0].isoformat()
    assert baseline["trades"][0]["reason"] == "qualification_initial_cash"
    funded = [trade for trade in baseline["trades"] if trade["reason"] == "qualification_scheduled_funding"]
    assert funded and funded[0]["timestamp"][:10] == "2024-01-05"
    assert all(trade["execution"] == "next_bar_end" for strategy in result["strategies"].values() for trade in strategy["trades"])
    assert all(trade["timestamp"][-8:] == "15:00:00" for strategy in result["strategies"].values() for trade in strategy["trades"])


def test_participation_capacity_is_independent_per_account_and_remainder_expires():
    day = pd.Timestamp("2024-01-03")
    closes = [9.8, 9.8, 9.8, 9.8]
    volumes = [10_000.0] * 4
    bundle = _bundle(
        [10, 9.8], frequency="60m",
        bars=_intraday_bars(day, closes, volumes),
    )
    config = _config(
        bundle, buy_shares=1000, frequency="60m",
        fees=FeeConfig(commission_rate=.001, min_commission=1, slippage_bps=0),
    )
    account = QualificationAccount(total_budget=10_000, max_shares=2000, ttl_bars=2)

    result = run_qualification(config, bundle, account)
    grid = result["strategies"]["grid"]
    baseline = result["strategies"]["scheduled_funding_hold"]

    assert all(trade["quantity"] == 100 for trade in grid["trades"][:2])
    assert all(trade["fees"] == pytest.approx(1.01) for trade in grid["trades"][:2])
    assert any(row["reason"] == "ttl_expired" and row["requested_quantity"] == 800
               for row in grid["rejections"])
    grid_at_eleven = [trade for trade in grid["trades"] if trade["timestamp"].endswith("11:00:00")]
    baseline_at_eleven = [trade for trade in baseline["trades"] if trade["timestamp"].endswith("11:00:00")]
    assert grid_at_eleven and baseline_at_eleven
    assert grid_at_eleven[0]["quantity"] == baseline_at_eleven[0]["quantity"] == 100


def test_insufficient_cash_cancels_the_remainder_without_resizing_it():
    bundle = _bundle([10, 9.5, 9.5])
    config = _config(bundle, buy_shares=1000)
    account = QualificationAccount(total_budget=500, max_shares=2000)

    result = run_qualification(config, bundle, account)
    grid = result["strategies"]["grid"]

    assert grid["trades"] == []
    assert any(row["reason"] == "insufficient_cash" and row["requested_quantity"] == 1000
               for row in grid["rejections"])
    assert grid["metrics"]["final_cash"] == pytest.approx(500)
    assert grid["metrics"]["funding_diagnostics"]["cash_shortfall_action"] == "cancel_remainder"
    assert result["qualification_status"] == "execution_limited"
    assert result["qualification_basis"]["grid_insufficient_cash_rejections"] > 0


def test_baseline_funding_rejection_does_not_make_the_grid_unqualified():
    bundle = _bundle([10, 10, 10])
    config = _config(bundle, initial_shares=1000, max_shares=2000)
    account = QualificationAccount(total_budget=10_500, max_shares=2000)

    result = run_qualification(config, bundle, account, passive_grid=True)
    grid = result["strategies"]["grid"]
    baseline = result["strategies"]["scheduled_funding_hold"]

    assert not any(row["reason"] == "insufficient_cash" for row in grid["rejections"])
    assert any(row["reason"] == "insufficient_cash" for row in baseline["rejections"])
    assert result["qualification_status"] == "qualified"
    assert result["qualification_basis"]["funding_rejections_by_strategy"]["grid"] == 0
    assert result["qualification_basis"]["funding_rejections_by_strategy"]["scheduled_funding_hold"] > 0


def test_reverse_signal_and_parameter_pause_cancel_pending_remainder():
    day = pd.Timestamp("2024-01-03")
    bars = _intraday_bars(day, [9.8, 9.8, 10.3, 10.3], [10_000.0] * 4)
    bars.loc[2, "limit_up"] = 10.3
    bundle = _bundle(
        [10, 9.8], frequency="60m",
        bars=bars,
    )
    config = _config(bundle, buy_shares=1000, sell_shares=1000, frequency="60m")
    account = QualificationAccount(total_budget=10_000, max_shares=2000, ttl_bars=10)
    grid = run_qualification(config, bundle, account)["strategies"]["grid"]
    assert any(row["reason"] == "reverse_signal" for row in grid["rejections"])

    daily = _bundle([10, 9.8, 9.8, 9.8])
    paused = _config(daily, buy_shares=1000)
    schedule = [{
        "effective_date": str(daily.calendar[1].date()),
        "decision_date": str(daily.calendar[0].date()),
        "paused": True,
        "params": {"buy_step": .01, "sell_step": .01, "buy_shares": 1000, "sell_shares": 1000},
    }]
    paused_result = run_qualification(
        paused, daily, QualificationAccount(total_budget=10_000, max_shares=2000),
        parameter_schedule=schedule,
    )
    paused_grid = paused_result["strategies"]["grid"]
    assert paused_grid["trades"] == []
    assert any(row["reason"] == "parameter_update" for row in paused_grid["rejections"])
    assert paused_result["strategies"]["scheduled_funding_hold"]["trades"]


def test_passive_grid_has_no_orders_but_still_has_a_valued_ledger():
    bundle = _bundle([10, 9.8, 10.2])
    config = _config(bundle, initial_shares=100, buy_shares=100, max_shares=2000)
    result = run_qualification(
        config, bundle, QualificationAccount(total_budget=5000, max_shares=2000),
        passive_grid=True,
    )
    grid = result["strategies"]["grid"]
    assert grid["trades"] == []
    assert len(grid["equity"]) == 2
    assert grid["equity"][-1]["equity"] > 0
    assert result["execution_profile"]["passive_grid"] is True


def test_nan_volume_keeps_an_unevaluable_mask_after_zero_fill():
    bundle = _bundle([10, 10, 10, 10])
    bundle.bars.loc[1, "volume"] = np.nan
    config = _config(bundle, initial_shares=100)
    result = run_qualification(
        config, bundle, QualificationAccount(total_budget=5000, max_shares=2000),
    )
    assert result["qualification_status"] == "unevaluable"
    assert "NaN" in result["unevaluable_reason"]
    assert result["volume_unavailable_bars"] == [bundle.bars.timestamp.iloc[1].isoformat()]
    assert not any(
        trade["timestamp"] == bundle.bars.timestamp.iloc[1].isoformat()
        and trade["quantity"] > 100_000 * result["execution_profile"]["participation_rate"]
        for strategy in result["strategies"].values() for trade in strategy["trades"]
    )


def test_nan_volume_takes_priority_over_a_risk_breach():
    bundle = _bundle([10, 10, 8, 8])
    bundle.bars.loc[1, "volume"] = np.nan
    config = _config(bundle, initial_shares=1000, max_shares=1000).model_copy(
        update={"max_principal_drawdown": .1},
    )
    result = run_qualification(
        config, bundle, QualificationAccount(total_budget=10_000, max_shares=1000),
    )

    assert result["qualification_basis"]["risk_breached"] is True
    assert result["qualification_status"] == "unevaluable"
    assert "NaN" in result["unevaluable_reason"]


def test_omitted_contribution_changes_initial_cash_and_keeps_the_removed_receipt_out():
    bundle = _bundle([10, 10, 10, 10])
    config = _config(bundle, initial_shares=1000, max_shares=3000)
    account = QualificationAccount(
        total_budget=20_000, max_shares=3000,
        contributions=[{"available_at": "2024-01-04T00:00:00", "amount": 1000}],
    )
    result = run_qualification(
        config, bundle, account, omitted_contribution_index=0,
    )
    assert result["initial_cash"] == pytest.approx(9_000)
    assert result["funding"]["total_budget"] == pytest.approx(20_000)
    assert result["funding"]["scheduled_contributions"] == pytest.approx(1_000)
    assert result["funding"]["eligible_scheduled_contributions"] == 0
    assert result["funding"]["omitted_contribution_amount"] == pytest.approx(1_000)
    assert result["external_topups"] == 0
    for strategy in result["strategies"].values():
        assert strategy["contributions"] == []
        assert strategy["metrics"]["initial_cash"] == pytest.approx(9_000)
        assert strategy["metrics"]["total_budget"] == pytest.approx(20_000)
        assert strategy["metrics"]["external_topups"] == 0
        assert strategy["metrics"]["funding_diagnostics"]["omitted_contribution_index"] == 0


def test_omitting_one_receipt_keeps_every_planned_receipt_out_of_initial_cash():
    bundle = _bundle([10, 10, 10, 10])
    config = _config(bundle, initial_shares=1000, max_shares=3000)
    account = QualificationAccount(
        total_budget=22_000, max_shares=3000,
        contributions=[
            {"available_at": "2024-01-04T00:00:00", "amount": 1000},
            {"available_at": "2024-01-05T00:00:00", "amount": 2000},
        ],
    )

    result = run_qualification(
        config, bundle, account, omitted_contribution_index=0,
    )

    assert result["initial_cash"] == pytest.approx(9_000)
    assert result["funding"]["scheduled_contributions"] == pytest.approx(3_000)
    assert result["funding"]["eligible_scheduled_contributions"] == pytest.approx(2_000)
    assert result["external_topups"] == pytest.approx(2_000)
    assert [row["index"] for row in result["contributions"]] == [1]
    assert [row["amount"] for row in result["contributions"]] == [2_000]


def test_intraday_contribution_is_credited_at_its_matching_bar():
    day = pd.Timestamp("2024-01-03")
    bars = _intraday_bars(day, [10.0] * 4, [100_000.0] * 4)
    bundle = _bundle([10, 10], frequency="60m", bars=bars)
    config = _config(
        bundle, initial_shares=1000, max_shares=1000, buy_shares=100,
        frequency="60m",
    )
    account = QualificationAccount(
        total_budget=11_000, max_shares=1000,
        contributions=[{"available_at": day + pd.Timedelta(hours=11), "amount": 1000}],
    )

    result = run_qualification(config, bundle, account)
    credited_at = bars.loc[1, "timestamp"].isoformat()
    assert result["initial_cash"] == 0
    assert result["external_topups"] == pytest.approx(1000)
    assert result["contributions"][0]["credited_at"] == credited_at
    assert result["contributions"][0]["bar"] == 1
    for strategy in result["strategies"].values():
        equity = {row["timestamp"]: row for row in strategy["equity"]}
        assert equity[bars.loc[0, "timestamp"].isoformat()]["cash"] == pytest.approx(0)
        assert equity[credited_at]["cash"] == pytest.approx(1000)


def test_risk_breach_takes_priority_over_a_grid_execution_limit():
    bundle = _bundle([10, 8, 8])
    config = _config(
        bundle, initial_shares=1000, max_shares=2000, buy_shares=1000,
    ).model_copy(update={"max_principal_drawdown": .1})
    account = QualificationAccount(total_budget=10_000, max_shares=2000)

    result = run_qualification(config, bundle, account)

    assert any(row["reason"] == "insufficient_cash"
               for row in result["strategies"]["grid"]["rejections"])
    assert result["qualification_basis"]["risk_breached"] is True
    assert result["qualification_status"] == "risk_breached"


def test_partial_fill_remainder_is_canceled_at_the_last_bar_of_the_day():
    day = pd.Timestamp("2024-01-03")
    bundle = _bundle(
        [10, 9.8], frequency="60m",
        bars=_intraday_bars(day, [9.8] * 4, [10_000.0] * 4),
    )
    config = _config(bundle, buy_shares=1000, frequency="60m")
    account = QualificationAccount(total_budget=10_000, max_shares=2000, ttl_bars=10)
    grid = run_qualification(config, bundle, account)["strategies"]["grid"]
    assert [trade["quantity"] for trade in grid["trades"]] == [100, 100, 100]
    terminal = [row for row in grid["rejections"] if row["reason"] == "day_end_cancel"]
    assert len(terminal) == 1
    assert terminal[0]["requested_quantity"] == 700
    assert terminal[0]["timestamp"].endswith("15:00:00")


def test_close_execution_uses_adverse_slippage_and_tick_rounding():
    bundle = _bundle([10, 9.8, 9.8])
    config = _config(
        bundle,
        fees=FeeConfig(commission_rate=0, min_commission=0, slippage_bps=100),
    )
    result = run_qualification(
        config, bundle, QualificationAccount(total_budget=10_000, max_shares=2000),
    )
    grid_trade = result["strategies"]["grid"]["trades"][0]
    assert grid_trade["side"] == "buy"
    assert grid_trade["price"] == pytest.approx(9.9)
    assert grid_trade["timestamp"] == bundle.bars.timestamp.iloc[1].isoformat()


def test_extra_delay_holds_a_full_order_until_its_later_bar_end():
    bundle = _bundle([10, 9.8, 9.8, 9.8, 9.8])
    config = _config(bundle)
    result = run_qualification(
        config, bundle,
        QualificationAccount(total_budget=10_000, max_shares=2000),
        extra_delay_bars=1,
    )
    grid = result["strategies"]["grid"]
    assert grid["trades"]
    assert grid["trades"][0]["timestamp"] == bundle.bars.timestamp.iloc[2].isoformat()
    assert result["execution_profile"]["extra_delay_bars"] == 1


def test_t_plus_one_and_limit_locked_orders_retry_within_their_ttl():
    day = pd.Timestamp("2024-01-03")
    bars = _intraday_bars(day, [9.8, 9.8, 9.8, 10.3], [10_000.0] * 4)
    bars.loc[1, "limit_up"] = 9.8
    bars.loc[2, "limit_up"] = 9.8
    bundle = _bundle([10, 9.8], frequency="60m", bars=bars)
    config = _config(bundle, buy_shares=100, sell_shares=100, frequency="60m")
    grid = run_qualification(
        config, bundle,
        QualificationAccount(total_budget=10_000, max_shares=2000, ttl_bars=5),
    )["strategies"]["grid"]
    assert any(row["reason"] == "limit_up_locked" for row in grid["rejections"])
    assert grid["trades"] and grid["trades"][0]["timestamp"].endswith("15:00:00")

    t1_bars = _intraday_bars(day, [9.8, 9.8, 10.3, 10.3], [100_000.0] * 4)
    t1_bundle = _bundle([10, 9.8], frequency="60m", bars=t1_bars)
    t1_config = _config(t1_bundle, buy_shares=100, sell_shares=100, frequency="60m")
    t1_grid = run_qualification(
        t1_config, t1_bundle,
        QualificationAccount(total_budget=10_000, max_shares=2000, ttl_bars=5),
    )["strategies"]["grid"]
    assert any(row["reason"] == "t_plus_1" for row in t1_grid["rejections"])


def test_company_action_stock_and_cash_claim_apply_to_both_qualification_ledgers():
    bundle = _bundle(
        [10, 10, 5, 5],
        precloses=[10, 10, 5, 5],
    )
    bundle.actions = normalize_actions(pd.DataFrame([{
        "record_date": bundle.calendar[0], "ex_date": bundle.calendar[1],
        "stock_date": bundle.calendar[1], "pay_date": bundle.calendar[1],
        "cash_per_share": .1, "bonus_ratio": 1.0,
    }]))
    config = _config(bundle, initial_shares=100, max_shares=2000)
    result = run_qualification(
        config, bundle,
        QualificationAccount(total_budget=1000, max_shares=2000),
        passive_grid=True,
    )
    for strategy in result["strategies"].values():
        assert strategy["trades"] == []
        assert strategy["equity"][-1]["position"] == 200
        assert strategy["equity"][-1]["cash"] == pytest.approx(10)
        assert strategy["equity"][-1]["equity"] == pytest.approx(1010)


def test_legacy_share_backtest_matches_compiled_and_interpreted_paths(monkeypatch):
    bundle = _bundle([10, 9.4, 10.4, 9.2, 10.5])
    config = _config(bundle, initial_shares=1000, max_shares=2000, buy_shares=100, sell_shares=100)
    compiled = engine.run_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    interpreted = engine.run_backtest(config, bundle)
    for name in ("grid", "dca", "buy_hold", "matched_funding_hold"):
        assert compiled["strategies"][name]["trades"] == interpreted["strategies"][name]["trades"]
        assert compiled["strategies"][name]["equity"] == interpreted["strategies"][name]["equity"]


def test_qualification_compiled_and_interpreted_kernel_match(monkeypatch):
    bundle = _bundle([10, 9.8, 10.2, 9.9, 10.4])
    config = _config(bundle, initial_shares=1000, max_shares=2000, buy_shares=100, sell_shares=100)
    account = QualificationAccount(
        total_budget=12_000, max_shares=2000,
        contributions=[{"available_at": "2024-01-04T00:00:00", "amount": 500}],
    )
    compiled = run_qualification(config, bundle, account)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    interpreted = run_qualification(config, bundle, account)
    for name in ("grid", "scheduled_funding_hold"):
        assert compiled["strategies"][name]["trades"] == interpreted["strategies"][name]["trades"]
        assert compiled["strategies"][name]["equity"] == interpreted["strategies"][name]["equity"]
        assert compiled["strategies"][name]["rejections"] == interpreted["strategies"][name]["rejections"]


def test_qualification_honors_an_existing_cancellation_request():
    bundle = _bundle([10, 10, 10])
    config = _config(bundle, initial_shares=100)
    with pytest.raises(InterruptedError):
        run_qualification(
            config, bundle,
            QualificationAccount(total_budget=5000, max_shares=2000),
            cancelled=lambda: True,
        )

