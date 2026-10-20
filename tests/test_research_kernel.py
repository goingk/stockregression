"""Scientific evaluation must share the production ledger, including boundaries."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockregression import engine
from stockregression.models import BacktestConfig, DataBundle, OptimizationConfig
from stockregression.research_v4.evaluator import BatchEvaluator
from stockregression.research_v4.space import ParameterSpace
from stockregression.providers import normalize_actions
from stockregression.rules import instrument_terms
from stockregression.share_engine import (
    funding, research_batch, research_passive, run_passive_backtest, run_share_backtest,
)


def market(intraday=False):
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
    config = BacktestConfig(schema_version=2, symbol="159611.SZ", start_date=dates[1].date(),
        end_date=dates[-1].date(), frequency="1m" if intraday else "1d", initial_shares=1000,
        max_shares=2000, reserve_cash=10000,
        fees={"commission_rate": 0, "min_commission": 0, "slippage_bps": 0})
    bundle = DataBundle(symbol=config.symbol, bars=bars, daily=daily, calendar=dates[1:],
        actions=normalize_actions(pd.DataFrame()), version="research-fixture",
        metadata=instrument_terms(config.symbol, {}), quality={"minute_ratio": 1.0})
    return config, bundle


def folds(bundle):
    return [(str(bundle.calendar[0].date()), str(bundle.calendar[3].date())),
            (str(bundle.calendar[4].date()), str(bundle.calendar[-1].date()))]


def test_research_batch_matches_full_ledger_and_fixed_cash():
    config, bundle = market()
    params = [{"buy_step": .03, "max_shares": 2000}, {"buy_step": .06, "max_shares": 4000}]
    batched = research_batch(config, bundle, params, folds(bundle))
    serial = research_batch(config, bundle, params, folds(bundle), parallel=False)
    np.testing.assert_array_equal(batched["daily_returns"], serial["daily_returns"])
    assert batched["metrics"] == serial["metrics"]
    assert batched["fold_metrics"] == serial["fold_metrics"]
    for index, patch in enumerate(params):
        run = run_share_backtest(config.model_copy(update=patch), bundle)
        assert run["funding"]["reserve_cash"] == 10000
        assert run["funding"]["initial_equity"] == 20000
        assert run["strategies"]["grid"]["metrics"] == batched["metrics"][index]
        np.testing.assert_array_equal(batched["daily_returns"][:, index],
            [p["return"] for p in run["strategies"]["grid"]["period_returns"]])


def test_research_fold_drawdown_includes_intraday_path():
    config, bundle = market(intraday=True)
    config = config.model_copy(update={"min_shares": 1000, "max_shares": 1000, "reserve_cash": 0})
    result = research_batch(config, bundle, [{}], folds(bundle))
    np.testing.assert_array_equal(result["daily_returns"], 0.)
    np.testing.assert_allclose(result["fold_drawdowns"], 1 / 3)
    assert result["fold_metrics"][0][0]["trade_count"] == 0


@pytest.mark.parametrize("grid_mode", ["trailing", "arithmetic", "geometric"])
def test_identical_schedule_is_exactly_the_same_continuous_account(grid_mode):
    config, bundle = market()
    config = config.model_copy(update={"grid_mode": grid_mode})
    params = {"buy_step": .03, "sell_step": .03} if grid_mode == "trailing" else {"grid_step": .03}
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
    assert first["cash"] == pytest.approx(10000 - 300 * 8.5)
    assert any(row["reason"] == "parameter_update" for row in grid["rejections"])
    assert not any(row["reason"] == "initial_position" for row in grid["trades"])


def test_daily_details_preserve_fills_and_exact_risk():
    config, bundle = market(intraday=True)
    full = run_share_backtest(config, bundle)
    compact = run_share_backtest(config, bundle, daily_details=True)
    for name, strategy in full["strategies"].items():
        daily = compact["strategies"][name]
        assert daily["metrics"] == strategy["metrics"]
        assert daily["trades"] == strategy["trades"]
        assert daily["rejections"] == strategy["rejections"]
        assert daily["period_returns"] == strategy["period_returns"]
        assert len(daily["equity"]) == len(bundle.calendar)
        assert len(strategy["equity"]) == len(bundle.bars)


def test_passive_uses_same_cash_and_shares_and_carries_dividend_across_change():
    config, bundle = market()
    bundle.actions = normalize_actions(pd.DataFrame([{
        "record_date": str(bundle.calendar[1].date()), "ex_date": str(bundle.calendar[2].date()),
        "pay_date": str(bundle.calendar[5].date()), "stock_date": str(bundle.calendar[5].date()),
        "cash_per_share": .6, "bonus_ratio": .1, "implemented": True,
    }]))
    passive = run_passive_backtest(config, bundle, daily_details=True)
    compact = research_passive(config, bundle)
    grid = passive["strategies"]["grid"]
    assert grid["trades"] == []
    assert grid["equity"][-1]["cash"] == 10600
    assert grid["equity"][-1]["position"] == 1100
    np.testing.assert_array_equal(compact["daily_returns"], [p["return"] for p in grid["period_returns"]])
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
    with pytest.raises(ValueError, match="现金预算"):
        research_batch(config, bundle, [{"reserve_cash": 20000}], [])
    with pytest.raises(InterruptedError):
        research_batch(config, bundle, [{}], [], cancelled=lambda: True)


def test_latest_parameters_replay_is_explicitly_retrospective():
    config, bundle = market()
    config = config.model_copy(update={"selection_asof": config.end_date})
    result = run_share_backtest(config, bundle)
    assert result["parameter_selection"]["retrospective"] is True
    assert any("样本内回看" in warning for warning in result["warnings"])


def test_batch_python_reference_matches_compiled(monkeypatch):
    config, bundle = market()
    fast = research_batch(config, bundle, [{}, {"sell_shares": 200}], folds(bundle))
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    slow = research_batch(config, bundle, [{}, {"sell_shares": 200}], folds(bundle))
    assert fast["metrics"] == slow["metrics"]
    assert fast["fold_metrics"] == slow["fold_metrics"]
    np.testing.assert_array_equal(fast["daily_returns"], slow["daily_returns"])


@pytest.mark.parametrize("intraday", [False, True])
@pytest.mark.parametrize("corporate_actions", [False, True])
def test_v4_fresh_validation_matches_standalone_backtest(intraday, corporate_actions):
    config, bundle = market(intraday=intraday)
    config = config.model_copy(update={
        "schema_version": 3, "grid_mode": "geometric", "reserve_cash": None,
        "fees": config.fees.model_copy(update={
            "commission_rate": .0003, "min_commission": 5., "slippage_bps": 5.,
        }),
    })
    if corporate_actions:
        bundle.actions = normalize_actions(pd.DataFrame([
            {"record_date": str(bundle.calendar[2].date()),
             "ex_date": str(bundle.calendar[4].date()),
             "pay_date": str(bundle.calendar[5].date()),
             "stock_date": str(bundle.calendar[5].date()),
             "cash_per_share": .2, "bonus_ratio": .1, "implemented": True},
            {"record_date": str(bundle.calendar[5].date()),
             "ex_date": str(bundle.calendar[6].date()),
             "pay_date": str(bundle.calendar[7].date()),
             "stock_date": str(bundle.calendar[7].date()),
             "cash_per_share": .1, "bonus_ratio": .1, "implemented": True},
        ]))
    request = OptimizationConfig(
        base=config, protocol_version=4,
        ranges={
            "grid_step": {"low": .03, "high": .031},
            "sell_shares": {"low": 100, "high": 200},
            "buy_shares": {"low": 200, "high": 200},
            "min_shares": {"low": 0, "high": 0},
            "max_shares": {"low": 2000, "high": 2000},
        },
    )
    space = ParameterSpace(request)
    train, validation = bundle.calendar[:4], bundle.calendar[4:]
    evaluator = BatchEvaluator(request, request.base, bundle, space, train, validation)
    coordinates = list(space.iterate())
    rows = evaluator.evaluate_validation(coordinates)
    stressed = evaluator.evaluate_stress(coordinates)
    trained = evaluator.evaluate_training(coordinates)
    assert evaluator.validation_base.start_date == validation[0].date()
    assert len(evaluator.validation_folds) == 4
    for index, (coordinate, row) in enumerate(zip(coordinates, rows)):
        candidate = evaluator.validation_base.model_copy(update=space.params(coordinate))
        backtest = run_share_backtest(candidate, bundle)
        grid = backtest["strategies"]["grid"]
        baseline = backtest["strategies"]["buy_hold"]["metrics"]["total_return"]
        assert row["grid_return"] == grid["metrics"]["total_return"]
        assert row["baseline_return"] == pytest.approx(baseline, abs=1e-14)
        assert row["validation_excess_original"] == pytest.approx(
            grid["metrics"]["total_return"] - baseline, abs=1e-14,
        )
        assert row["trade_count"] == grid["metrics"]["trade_count"]
        np.testing.assert_array_equal(row["_current_validation_returns"],
            [item["return"] for item in grid["period_returns"]])
        assert list(row["_current_validation_dates"]) == list(validation)
        # Every diagnostic fold measures this one ledger; no fold reopens it.
        assert sum(fold["profit"] for fold in row["folds"]) == pytest.approx(
            grid["metrics"]["total_profit"], abs=1e-9,
        )
        stress_fees = candidate.fees.model_copy(update={
            name: getattr(candidate.fees, name) * 2
            for name in ("commission_rate", "min_commission", "slippage_bps")
        })
        stress_run = run_share_backtest(candidate.model_copy(update={"fees": stress_fees}), bundle)
        assert stressed[coordinate]["validation_return_stress"] == stress_run["strategies"]["grid"]["metrics"]["total_return"]
        training_run = run_share_backtest(evaluator.train_base.model_copy(update=space.params(coordinate)), bundle)
        assert trained[index]["training_return"] == training_run["strategies"]["grid"]["metrics"]["total_return"]
    if not intraday and not corporate_actions:
        # A replay starting in training must not silently become validation's
        # starting account again. This fixture deliberately changes holdings.
        old_run = research_batch(request.base, bundle, [space.params(coordinates[0])],
            [(str(validation[0].date()), str(validation[-1].date()))], parallel=False)
        fold = old_run["fold_metrics"][0][0]
        assert fold["profit"] / fold["principal"] != pytest.approx(rows[0]["grid_return"])
