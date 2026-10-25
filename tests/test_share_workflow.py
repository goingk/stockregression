import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression import engine
from stockregression.api import create_app
from stockregression.data import DataService
from stockregression.models import BacktestConfig, DataBundle, DcaConfig, FeeConfig, FloorPenaltyConfig, OptimizationConfig
from stockregression.providers import normalize_actions
from stockregression.rules import instrument_terms
from stockregression.storage import Store


def fixture(symbol="600000.SH", days=8, **metadata):
    dates = pd.bdate_range("2024-01-02", periods=days + 1)
    prices = np.resize([10., 9.4, 10.4, 9.2, 10.5, 9.1, 10.7, 9.0, 10.6], len(dates))
    daily = pd.DataFrame({"date": dates, "open": prices, "high": prices + .01, "low": prices - .01, "close": prices, "preclose": np.r_[prices[0], prices[:-1]], "volume": 100000, "turnover": prices * 100000, "tradestatus": 1, "is_st": False, "limit_up": 100., "limit_down": .01})
    bars = daily.iloc[1:].copy()
    bars["timestamp"] = bars.date + pd.Timedelta(hours=15)
    bars["frequency"], bars["tradable"], bars["is_synthetic"] = "1d", True, False
    config = BacktestConfig(schema_version=2, symbol=symbol, start_date=dates[1].date(), end_date=dates[-1].date(), frequency="1d", fees={"commission_rate": 0, "min_commission": 0, "slippage_bps": 0}, dca={"frequency": "daily", "shares": 100})
    bundle = DataBundle(symbol=symbol, bars=bars, daily=daily, calendar=dates[1:], actions=normalize_actions(pd.DataFrame()), metadata=instrument_terms(symbol, metadata), version="fixture-v2")
    return config, bundle


def floor_grid_fixture(opens, closes):
    config, bundle = fixture(days=len(opens) - 1)
    opens, closes = np.asarray(opens, dtype=float), np.asarray(closes, dtype=float)
    for name, values in (
        ("open", opens), ("close", closes),
        ("high", np.maximum(opens, closes) + .01), ("low", np.minimum(opens, closes) - .01),
        ("preclose", np.r_[closes[0], closes[:-1]]), ("turnover", closes * 100000),
    ):
        bundle.daily[name] = values
    bundle.bars = bundle.daily.iloc[1:].copy()
    bundle.bars["timestamp"] = bundle.bars.date + pd.Timedelta(hours=15)
    bundle.bars["frequency"], bundle.bars["tradable"], bundle.bars["is_synthetic"] = "1d", True, False
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 10000, "min_shares": 0, "max_shares": None,
        "buy_shares": 1000, "sell_shares": 5500, "grid_mode": "geometric",
        "buy_step": .02, "sell_step": .02, "reference_price_mode": "manual", "reference_price": 10,
        "fees": FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
        "floor_penalty_config": FloorPenaltyConfig(eta_floor=.5, gamma_min=.4, floor_k_ref=10),
    })
    return config, bundle


def test_grid_position_audit_records_stockout_once_for_a_repeated_sell_signal():
    config, bundle = fixture()
    prices = np.array([8.0, 9.5, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9])
    for name, values in (
        ("open", prices), ("close", prices), ("high", prices + .01),
        ("low", prices - .01), ("preclose", np.r_[prices[0], prices[:-1]]),
        ("turnover", prices * 100000),
    ):
        bundle.daily[name] = values
    bundle.bars = bundle.daily.iloc[1:].copy()
    bundle.bars["timestamp"] = bundle.bars.date + pd.Timedelta(hours=15)
    bundle.bars["frequency"], bundle.bars["tradable"], bundle.bars["is_synthetic"] = "1d", True, False
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 200, "min_shares": 100,
        "max_shares": None, "buy_shares": 100, "sell_shares": 300,
        "grid_mode": "geometric", "buy_step": .01, "sell_step": .01,
        "reference_price_mode": "manual", "reference_price": 8.5,
        "fees": FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
    })
    grid = engine.run_backtest(config, bundle)["strategies"]["grid"]
    assert grid["metrics"]["stockout_count"] == 1
    assert grid["metrics"]["tplus1_block_count"] == 0
    assert grid["metrics"]["actual_min_pos_ratio"] == pytest.approx(.5)
    assert grid["metrics"]["actual_max_pos_ratio"] == pytest.approx(1.0)


def test_floor_episode_counts_five_grids_even_after_the_position_reaches_zero():
    config, bundle = floor_grid_fixture(
        [10, 10, 10.2, 10.2, 11.2700, 11.2700],
        [10, 10, 10.2, 10.4, 11.2700, 11.2700],
    )
    result = engine.run_backtest(config, bundle)
    grid = result["strategies"]["grid"]
    assert result["floor_episode_grids"] == [5]
    assert result["floor_penalty"] == pytest.approx(.1)
    assert grid["floor_episode_count"] == 1
    assert grid["floor_episode_grids"] == [5]
    assert grid["floor_penalty"] == pytest.approx(.1)
    assert grid["trades"][-1]["shares"] == 0


def test_floor_episode_without_a_new_sell_grid_has_zero_penalty():
    config, bundle = floor_grid_fixture(
        [10, 10, 10.2, 9.9, 9.8, 9.8],
        [10, 10, 10.2, 9.9, 9.8, 9.8],
    )
    grid = engine.run_backtest(config, bundle)["strategies"]["grid"]
    assert grid["floor_episode_count"] == 1
    assert grid["floor_episode_grids"] == [0]
    assert grid["floor_penalty"] == 0.0


def test_initial_position_at_the_floor_starts_an_episode_and_buyback_closes_it():
    config, bundle = floor_grid_fixture(
        [10, 10, 10.2, 10.2, 9.9, 9.8],
        [10, 10, 10.2, 9.9, 9.8, 9.8],
    )
    config = config.model_copy(update={
        "floor_penalty_config": FloorPenaltyConfig(eta_floor=1.0, gamma_min=.4, floor_k_ref=10),
    })
    grid = engine.run_backtest(config, bundle)["strategies"]["grid"]
    assert grid["floor_episode_count"] == 1
    assert len(grid["floor_episode_grids"]) == 1
    assert grid["trades"][-1]["side"] == "buy"
    assert grid["trades"][-1]["shares"] > config.initial_shares * .5


def test_floor_penalty_does_not_change_the_trading_path():
    opens = [10, 10, 10.2, 10.2, 11.2700, 11.2700]
    closes = [10, 10, 10.2, 10.4, 11.2700, 11.2700]
    penalized, bundle = floor_grid_fixture(opens, closes)
    plain = penalized.model_copy(update={"floor_penalty_config": None})
    left = engine.run_backtest(penalized, bundle)["strategies"]["grid"]
    right = engine.run_backtest(plain, bundle)["strategies"]["grid"]
    assert left["trades"] == right["trades"]
    assert left["equity"] == right["equity"]
    for key in ("peak_principal_return", "max_principal_drawdown", "trade_count"):
        assert left["metrics"][key] == right["metrics"][key]


def test_existing_shares_sell_first_day_and_orders_use_hundreds(monkeypatch):
    config, bundle = fixture()
    config = config.model_copy(update={"reference_price_mode": "manual", "reference_price": 8.5})
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    result = engine.run_backtest(config, bundle)
    grid = result["strategies"]["grid"]
    assert not any(trade["reason"] == "initial_position" for trade in grid["trades"])
    assert grid["trades"][0]["side"] == "sell"
    assert all(trade["quantity"] >= 100 and trade["quantity"] % 100 == 0 for strategy in result["strategies"].values() for trade in strategy["trades"])
    assert result["funding"]["valuation_price"] == 10
    assert result["funding"]["initial_equity"] == 18500.1  # Ten buy orders reserve exchange transfer fees.


def test_dca_initial_holdings_and_external_contributions_share_one_return_basis():
    config, bundle = fixture("510300.SH")
    config = config.model_copy(update={"dca": DcaConfig(frequency="daily", shares=100, initial_shares=200), "fees": FeeConfig()})
    full = engine.run_backtest(config, bundle)
    preview = engine.run_backtest(config, bundle, preview=True)
    trades = full["strategies"]["dca"]["trades"]
    total = sum(trade["quantity"] * trade["price"] + trade["fees"] for trade in trades)
    dca = full["strategies"]["dca"]
    assert len(trades) == 8
    assert dca["metrics"]["initial_equity"] == 2000
    assert dca["metrics"]["total_invested"] == pytest.approx(2000 + total)
    assert dca["metrics"]["peak_principal_return"] == pytest.approx((1000 * bundle.daily.close.iloc[-1]) / (2000 + total) - 1)
    assert "cagr" not in dca["metrics"] and "period_returns" not in dca
    for name in full["strategies"]:
        assert full["strategies"][name]["equity"] == preview["strategies"][name]["equity"]
    assert full["accounting_basis"] == "independent_contributions_v1"
    running_dca_principal = 0.0
    for point in dca["equity"]:
        running_dca_principal = max(running_dca_principal, point["total_invested"])
        assert point["principal_curve"] == pytest.approx(
            1 + (point["equity"] - point["total_invested"]) / running_dca_principal
        )
    assert running_dca_principal == pytest.approx(dca["metrics"]["total_invested"])
    for name in ("grid", "buy_hold"):
        for point in full["strategies"][name]["equity"]:
            assert point["principal_curve"] == pytest.approx(
                1 + (point["equity"] - point["total_invested"]) / point["peak_principal"]
            )


def test_initial_shares_defaults_and_fee_defaults():
    assert FeeConfig().commission_rate == .0005
    assert FeeConfig().min_commission == .1
    assert DcaConfig().initial_shares == 0
    for value in (1, 99, 150, 100.5, -100):
        with pytest.raises(ValueError):
            DcaConfig(initial_shares=value)
    assert DcaConfig(initial_shares=0).initial_shares == 0
    assert DcaConfig(initial_shares=100).initial_shares == 100
    config, _ = fixture("510300.SH")
    older_v2_payload = config.model_dump(mode="json")
    older_v2_payload["dca"].pop("initial_shares")
    loaded = BacktestConfig.model_validate(older_v2_payload)
    assert loaded.schema_version == 2 and loaded.dca.initial_shares == 0


def test_buy_hold_and_dca_are_independent_of_grid_funding_fields():
    config, bundle = fixture("510300.SH")
    config = config.model_copy(update={"dca": DcaConfig(frequency="daily", shares=100, initial_shares=200)})
    baseline = engine.run_backtest(config, bundle)
    dca_variant_raw = config.model_dump(mode="json")
    dca_variant_raw["dca"].update(frequency="weekly", shares=300, initial_shares=400)
    dca_variant = engine.run_backtest(BacktestConfig.model_validate(dca_variant_raw), bundle)
    for name in ("grid", "buy_hold"):
        assert dca_variant["strategies"][name] == baseline["strategies"][name]
    assert dca_variant["strategies"]["dca"] != baseline["strategies"]["dca"]
    raw = config.model_dump(mode="json")
    raw.update(initial_shares=0, min_shares=0, max_shares=0, grid_mode="arithmetic", buy_step=.15, sell_step=.15, reference_price_mode="manual", reference_price=20)
    gridless = engine.run_backtest(BacktestConfig.model_validate(raw), bundle)
    assert gridless["funding"]["initial_equity"] == 0
    assert gridless["strategies"]["grid"]["metrics"]["peak_principal_return"] is None
    for name in ("dca", "buy_hold"):
        assert gridless["strategies"][name] == baseline["strategies"][name]
    hold = gridless["strategies"]["buy_hold"]
    assert hold["trades"] == []
    assert all(point["position"] == 1000 and point["cash"] == 0 for point in hold["equity"])
    assert hold["metrics"]["initial_equity"] == 10000
    assert all(point["principal_curve"] == pytest.approx(point["equity"] / 10000) for point in hold["equity"])


def test_compiled_and_interpreted_share_accounting_match(monkeypatch):
    config, bundle = fixture("518880.SH")
    config = config.model_copy(update={"dca": DcaConfig(frequency="daily", shares=100, initial_shares=200)})
    fast = engine.run_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    slow = engine.run_backtest(config, bundle)
    assert fast == slow
    summary = engine.run_backtest(config, bundle, summary_only=True)
    assert summary["accounting_basis"] == fast["accounting_basis"] == "independent_contributions_v1"
    for name in fast["strategies"]:
        assert summary["strategies"][name]["metrics"] == fast["strategies"][name]["metrics"]


def test_fee_decimals_quantities_and_inactive_fields():
    config, _ = fixture()
    raw = config.model_dump(mode="json")
    raw.update(grid_mode="geometric", buy_step=.02, sell_step=.04, fees={"commission_rate": .00012345})
    parsed = BacktestConfig.model_validate(raw)
    assert parsed.fees.commission_rate == .00012345
    assert parsed.buy_step == .02 and parsed.sell_step == .04
    assert "grid_step" not in parsed.model_dump()
    with pytest.raises(ValueError):
        BacktestConfig.model_validate({**raw, "buy_step": 0})
    with pytest.raises(ValueError):
        OptimizationConfig.model_validate({"base": raw, "ranges": {"buy_step": {"low": -1, "high": -2, "step": .01}, "buy_shares": {"low": 100, "high": 200, "step": 100}}})
    for value in (1, 99, 150, 100.5):
        with pytest.raises(ValueError):
            BacktestConfig.model_validate({**raw, "buy_shares": value})




def test_retired_optimization_modes_and_trial_controls_are_rejected():
    config, bundle = fixture("510300.SH")
    unversioned = config.model_dump(mode="json")
    unversioned.pop("schema_version")
    defaults = OptimizationConfig(base=unversioned)
    assert "buy_shares" in defaults.ranges and "buy_ratio" not in defaults.ranges and "grid_step" not in defaults.ranges
    with pytest.raises(ValueError):
        OptimizationConfig(base=config, mode="historical", n_trials=2)


def test_v2_api_preview_does_not_persist_jobs_or_results(tmp_path, monkeypatch):
    config, bundle = fixture("510300.SH")
    store = Store(tmp_path)
    store.publish(config.symbol, bundle.bars, bundle.daily, bundle.actions, pd.DataFrame({"date": bundle.daily.date}), {**bundle.metadata, "actions_verified": True}, {})
    sync_calls = []
    monkeypatch.setattr(DataService, "sync", lambda self, symbols, mode, *args, **kwargs: sync_calls.append((symbols, mode)) or {"status": "failed", "symbols": [], "failures": [{"symbol": config.symbol, "reason": "mock source unavailable"}]})
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        raw = config.model_dump(mode="json")
        raw["end_date"] = "2024-01-31"
        response = client.post("/api/v1/backtests/preview", json={"session_id": "test", "revision": 1, "config": raw})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "DATA_NOT_READY"
        assert sync_calls == []
        assert client.get("/api/v1/jobs").json() == []
        assert client.get("/api/v1/results").json() == []


@pytest.mark.parametrize("symbol,can_sell", [("600000.SH", False), ("518880.SH", True)])
def test_new_purchases_respect_product_t0_t1(symbol, can_sell):
    config, bundle = fixture(symbol, days=1)
    config = config.model_copy(update={"frequency": "1m", "initial_shares": 0})
    day = bundle.calendar[0]
    bundle.bars = pd.DataFrame([{"timestamp": day + pd.Timedelta(hours=9, minutes=31 + index), "open": opening, "close": close, "high": max(opening, close), "low": min(opening, close), "volume": 100000, "frequency": "1m", "tradable": True, "is_synthetic": False} for index, (opening, close) in enumerate([(10, 9), (9, 11), (11, 11)])])
    result = engine.run_backtest(config, bundle)
    grid = result["strategies"]["grid"]
    assert [trade["side"] for trade in grid["trades"]] == (["buy", "sell"] if can_sell else ["buy"])
    if not can_sell:
        assert any(row["reason"] == "t_plus_1" for row in grid["rejections"])
    assert grid["trades"][0]["fees"] == (0 if can_sell else .01)


def test_initial_holding_receives_first_day_dividend_without_nav_jump():
    config, bundle = fixture()
    config = config.model_copy(update={"max_shares": 1000, "buy_step": 1., "sell_step": 1., "dca": DcaConfig(frequency="daily", shares=100, initial_shares=200)})
    bundle.daily.loc[bundle.daily.date == bundle.calendar[0], "preclose"] = 9.4
    bundle.actions = normalize_actions(pd.DataFrame([{"record_date": "2024-01-02", "ex_date": "2024-01-03", "pay_date": "2024-01-04", "cash_per_share": .6, "bonus_ratio": 0}]))
    result = engine.run_backtest(config, bundle)
    assert result["strategies"]["grid"]["equity"][0]["equity"] == pytest.approx(10000)
    assert result["strategies"]["grid"]["equity"][1]["cash"] == 600
    assert result["strategies"]["dca"]["metrics"]["initial_equity"] == 2000
    assert result["strategies"]["dca"]["equity"][0]["position"] == 300
    assert result["strategies"]["dca"]["equity"][0]["equity"] == pytest.approx(2940)
    first_dca_trade = result["strategies"]["dca"]["trades"][0]
    first_invested = 2000 + first_dca_trade["quantity"] * first_dca_trade["price"] + first_dca_trade["fees"]
    assert result["strategies"]["dca"]["equity"][0]["total_invested"] == pytest.approx(first_invested)
    assert result["strategies"]["dca"]["equity"][0]["principal_curve"] == pytest.approx(
        1 + (2940 - first_invested) / first_invested
    )
    assert result["strategies"]["dca"]["equity"][0]["cash"] == 0
    assert result["strategies"]["dca"]["equity"][1]["cash"] == 120
    hold = result["strategies"]["buy_hold"]
    assert hold["trades"] == []
    assert hold["metrics"]["initial_equity"] == 10000
    assert hold["equity"][0]["equity"] == pytest.approx(10000)
    assert hold["equity"][1]["cash"] == 600


def test_fixed_grid_does_not_whipsaw_around_the_same_grid_line(monkeypatch):
    config, bundle = fixture("512880.SH", days=6)
    opens = np.array([1.031, 1.031, 1.081, 1.066, 1.068, 1.043, 1.084])
    closes = np.array([1.031, 1.081, 1.066, 1.068, 1.043, 1.084, 1.084])
    for name, values in (
        ("open", opens), ("close", closes),
        ("high", np.maximum(opens, closes) + .001),
        ("low", np.minimum(opens, closes) - .001),
        ("preclose", np.r_[closes[0], closes[:-1]]),
        ("turnover", closes * 100000),
    ):
        bundle.daily[name] = values
    bundle.bars = bundle.daily.iloc[1:].copy()
    bundle.bars["timestamp"] = bundle.bars.date + pd.Timedelta(hours=15)
    bundle.bars["frequency"], bundle.bars["tradable"], bundle.bars["is_synthetic"] = "1d", True, False
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 20_000, "min_shares": 0,
        "max_shares": None, "buy_shares": 2_500, "sell_shares": 2_500,
        "grid_mode": "geometric", "buy_step": .035, "sell_step": .035,
        "reference_price_mode": "manual", "reference_price": 1.031,
    })

    trades = engine.run_backtest(config, bundle)["strategies"]["grid"]["trades"]
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    assert engine.run_backtest(config, bundle)["strategies"]["grid"]["trades"] == trades
    assert [(trade["side"], trade["price"]) for trade in trades] == [
        ("sell", 1.081), ("buy", 1.043), ("sell", 1.084),
    ]
    assert trades[1]["price"] <= trades[0]["price"] * (1 - .035)
    assert trades[2]["price"] >= trades[1]["price"] * (1 + .035)


@pytest.mark.parametrize("bonus_ratio,expected_position,price_factor", [(1.0, 200, .5), (-.5, 50, 2.0)])
def test_share_conversion_updates_position_and_preserves_nav(bonus_ratio, expected_position, price_factor, monkeypatch):
    config, bundle = fixture(days=3)
    dates = bundle.calendar
    prices = np.array([10.0, 10.0, 10.0 * price_factor, 10.0 * price_factor])
    for name, values in (("open", prices), ("close", prices), ("high", prices + .01),
                         ("low", prices - .01), ("preclose", np.r_[10.0, prices[:-1]]),
                         ("turnover", prices * 100000)):
        bundle.daily[name] = values
    bundle.bars = bundle.daily.iloc[1:].copy()
    bundle.bars["timestamp"] = bundle.bars.date + pd.Timedelta(hours=15)
    bundle.bars["frequency"], bundle.bars["tradable"], bundle.bars["is_synthetic"] = "1d", True, False
    bundle.actions = normalize_actions(pd.DataFrame([{
        "record_date": dates[0], "ex_date": dates[1], "stock_date": dates[1],
        "cash_per_share": 0, "bonus_ratio": bonus_ratio,
    }]))
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 100, "min_shares": 0,
        "max_shares": None, "buy_shares": 0, "sell_shares": 0,
        "buy_step": 1., "sell_step": 1., "reference_price_mode": "manual",
        "reference_price": 10., "fees": FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
    })
    result = engine.run_backtest(config, bundle)["strategies"]["grid"]
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    interpreted = engine.run_backtest(config, bundle)["strategies"]["grid"]
    converted = next(row for row in result["equity"] if row["timestamp"][:10] == str(dates[1].date()))
    assert converted["position"] == expected_position
    assert converted["equity"] == pytest.approx(1000, abs=.02)
    assert [row["position"] for row in interpreted["equity"]] == [row["position"] for row in result["equity"]]
    assert [row["equity"] for row in interpreted["equity"]] == pytest.approx([row["equity"] for row in result["equity"]])


def _share_action_market(bundle, actions, prices):
    dates = bundle.calendar
    prices = np.asarray(prices, dtype=float)
    for name, values in (("open", prices), ("close", prices), ("high", prices + .001),
                         ("low", prices - .001), ("preclose", np.r_[prices[0], prices[:-1]]),
                         ("turnover", prices * 100000)):
        bundle.daily[name] = values
    bundle.bars = bundle.daily.iloc[1:].copy()
    bundle.bars["timestamp"] = bundle.bars.date + pd.Timedelta(hours=15)
    bundle.bars["frequency"], bundle.bars["tradable"], bundle.bars["is_synthetic"] = "1d", True, False
    bundle.actions = normalize_actions(pd.DataFrame([
        {"record_date": dates[record], "ex_date": dates[ex], "stock_date": dates[stock],
         "pay_date": dates[ex] if cash else None,
         "cash_per_share": cash, "bonus_ratio": ratio}
        for item in actions
        for record, ex, stock, ratio, cash in [(*item, 0) if len(item) == 4 else item]
    ]))


def test_pending_order_quantity_and_execution_price_follow_split_once(monkeypatch):
    config, bundle = fixture(days=3)
    # Prior close signals a 100 share buy; the split is effective before the
    # next open, so its pending amount becomes 200 and fills at half price.
    _share_action_market(bundle, [(0, 1, 1, 1.0)], [10, 9, 5, 5])
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 100, "buy_shares": 100,
        "sell_shares": 100, "min_shares": 0, "max_shares": None,
        "buy_step": .05, "sell_step": 1., "reference_price_mode": "manual",
        "reference_price": 10., "fees": FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
    })
    grid = engine.run_backtest(config, bundle)["strategies"]["grid"]
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    interpreted = engine.run_backtest(config, bundle)["strategies"]["grid"]
    fill = next(trade for trade in grid["trades"] if trade["timestamp"][:10] == str(bundle.calendar[1].date()))
    assert fill["quantity"] == 200
    assert fill["price"] == pytest.approx(5)
    py_fill = next(trade for trade in interpreted["trades"] if trade["timestamp"][:10] == str(bundle.calendar[1].date()))
    assert (py_fill["quantity"], py_fill["price"]) == pytest.approx((fill["quantity"], fill["price"]))


def test_dca_share_amount_is_converted_before_ex_date_purchase():
    config, bundle = fixture(days=3)
    _share_action_market(bundle, [(1, 2, 2, 1.0)], [10, 10, 5, 5])
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 100, "buy_shares": 100,
        "sell_shares": 100, "min_shares": 0, "max_shares": None,
        "buy_step": 1., "sell_step": 1., "reference_price_mode": "manual",
        "reference_price": 10., "dca": DcaConfig(frequency="daily", shares=100),
        "fees": FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
    })
    dca = engine.run_backtest(config, bundle)["strategies"]["dca"]
    ex_fill = next(trade for trade in dca["trades"] if trade["timestamp"][:10] == str(bundle.calendar[2].date()))
    assert ex_fill["quantity"] == 200
    assert ex_fill["price"] == pytest.approx(5)


def test_continuous_splits_compound_and_execute_on_latest_share_basis():
    config, bundle = fixture(days=4)
    # 200 * 1.0025 * 2.495 = 500.2475, so one final floor produces 500.
    # Flooring after the first action would instead produce a 400 share lot.
    _share_action_market(bundle, [(0, 1, 1, .0025), (1, 2, 2, 1.495)], [10, 10, 9.975, 3.5, 3.5])
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 100, "buy_shares": 200,
        "sell_shares": 200, "min_shares": 0, "max_shares": None,
        "buy_step": .05, "sell_step": 1., "reference_price_mode": "manual",
        "reference_price": 10., "fees": FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
    })
    grid = engine.run_backtest(config, bundle)["strategies"]["grid"]
    fill = next(trade for trade in grid["trades"] if trade["timestamp"][:10] == str(bundle.calendar[3].date()))
    assert fill["quantity"] == 500
    assert fill["price"] == pytest.approx(3.5)


def test_parameter_schedule_uses_decision_basis_and_does_not_double_scale_ex_date():
    from stockregression.share_engine import _inputs, _parameter_arrays

    config, bundle = fixture(days=4)
    _share_action_market(bundle, [(1, 2, 2, 1.0)], [10, 10, 5, 5, 5])
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 100, "buy_shares": 100,
        "sell_shares": 100, "min_shares": 0, "max_shares": None,
        "reference_price_mode": "manual", "reference_price": 10.,
    })
    prepared = _inputs(config, bundle)
    same_day = [{"decision_date": str(bundle.calendar[1].date()),
                 "effective_date": str(bundle.calendar[2].date()),
                 "params": {"buy_shares": 200, "sell_shares": 200}}]
    next_day = [{"decision_date": str(bundle.calendar[2].date()),
                 "effective_date": str(bundle.calendar[3].date()),
                 "params": {"buy_shares": 200, "sell_shares": 200}}]
    decided_before_split = [{"decision_date": str(bundle.calendar[1].date()),
                             "effective_date": str(bundle.calendar[3].date()),
                             "params": {"buy_shares": 200, "sell_shares": 200}}]
    same_values = _parameter_arrays(config, prepared, same_day)[1]
    next_values = _parameter_arrays(config, prepared, next_day)[1]
    prior_basis_values = _parameter_arrays(config, prepared, decided_before_split)[1]
    # On the ex-date, the kernel converts these values once after applying the
    # schedule. On the following day the decision already uses post-split units.
    assert same_values[0, 5:7].tolist() == [200, 200]
    assert next_values[0, 5:7].tolist() == [200, 200]
    assert prior_basis_values[0, 5:7].tolist() == [400, 400]


def test_parameter_arrays_keep_fractional_share_basis_for_a_later_split():
    from stockregression.share_engine import _inputs, _parameter_arrays

    config, bundle = fixture(days=5)
    _share_action_market(
        bundle, [(0, 1, 1, -2 / 3), (2, 3, 3, 2.0)],
        [10, 10, 30, 30, 10, 10],
    )
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 100, "min_shares": 0,
        "max_shares": None, "buy_shares": 100, "sell_shares": 100,
        "reference_price_mode": "manual", "reference_price": 10.,
    })
    prepared = _inputs(config, bundle)
    schedule = [{
        "decision_date": str(bundle.calendar[0].date()),
        "effective_date": str(bundle.calendar[2].date()),
        "params": {
            "min_shares": 100, "max_shares": 400,
            "buy_shares": 100, "sell_shares": 100,
        },
    }]
    _, values, _ = _parameter_arrays(config, prepared, schedule)
    # The decision basis passes through a 1-for-3 reverse split. Preserve the
    # fractional schedule value so a later 3-for-1 action restores it exactly.
    assert values[0, 0] == pytest.approx(100 / 3)
    assert values[0, 1] == pytest.approx(400 / 3)
    assert values[0, 5] == pytest.approx(100 / 3)
    assert values[0, 6] == pytest.approx(100 / 3)
    assert values[0, 5] * 3 == pytest.approx(100)
    assert values[0, 6] * 3 == pytest.approx(100)


def test_delayed_reverse_split_settles_once_and_never_oversells():
    config, bundle = fixture(days=6)
    _share_action_market(bundle, [(1, 2, 4, -.5)], [10, 10, 10, 22, 24, 26, 26])
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 200, "buy_shares": 100,
        "sell_shares": 200, "min_shares": 0, "max_shares": None,
        "buy_step": 1., "sell_step": .05, "reference_price_mode": "manual",
        "reference_price": 10., "fees": FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
    })
    grid = engine.run_backtest(config, bundle)["strategies"]["grid"]
    ex_point = next(row for row in grid["equity"] if row["timestamp"][:10] == str(bundle.calendar[2].date()))
    pre_stock_point = next(row for row in grid["equity"] if row["timestamp"][:10] == str(bundle.calendar[3].date()))
    assert ex_point["position"] == 100
    assert pre_stock_point["position"] == 100
    sells = [trade for trade in grid["trades"] if trade["side"] == "sell"]
    assert [trade["quantity"] for trade in sells] == [100]
    assert sells[0]["timestamp"][:10] == str(bundle.calendar[4].date())
    ex_date = pd.Timestamp(bundle.actions.iloc[0]["ex_date"]).date()
    stock_date = pd.Timestamp(bundle.actions.iloc[0]["stock_date"]).date()
    pre_stock_rejections = [
        rejection for rejection in grid["rejections"]
        if rejection["reason"] == "t_plus_1"
        and ex_date < pd.Timestamp(rejection["timestamp"]).date() < stock_date
    ]
    assert pre_stock_rejections, grid["rejections"]
    assert all(row["position"] >= 0 for row in grid["equity"])


def test_same_ex_date_positive_and_negative_claims_with_different_unlock_dates_are_rejected():
    config, bundle = fixture(days=4)
    ex = bundle.calendar[1]
    bundle.actions = normalize_actions(pd.DataFrame([
        {"record_date": bundle.calendar[0], "ex_date": ex,
         "stock_date": bundle.calendar[2], "cash_per_share": 0, "bonus_ratio": .5},
        {"record_date": bundle.calendar[0], "ex_date": ex,
         "stock_date": bundle.calendar[3], "cash_per_share": 0, "bonus_ratio": -.25},
    ]))
    with pytest.raises(ValueError, match="可交易日不一致"):
        engine._prepare_actions(bundle, config.start_date, config.end_date)


def test_same_day_actions_sum_bonus_and_cash_before_calculating_one_price_factor():
    config, bundle = fixture(days=3)
    # Two simultaneous 1-for-2 entitlements combine to r=2 (not 2.25),
    # while distinct cash distributions add per pre-action share.
    _share_action_market(bundle, [(0, 1, 1, .5, .2), (0, 1, 1, .5, .3)], [10, 10, 4.75, 4.75])
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 100, "buy_shares": 100,
        "sell_shares": 100, "min_shares": 0, "max_shares": None,
        "buy_step": 1., "sell_step": .05, "reference_price_mode": "manual",
        "reference_price": 10., "fees": FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
    })
    grid = engine.run_backtest(config, bundle)["strategies"]["grid"]
    ex_point = next(row for row in grid["equity"] if row["timestamp"][:10] == str(bundle.calendar[1].date()))
    assert ex_point["position"] == 200
    assert ex_point["cash"] == pytest.approx(50)
    assert ex_point["equity"] == pytest.approx(1000)
    assert grid["trades"] == []

    from stockregression.share_engine import _inputs, _parameter_arrays
    prepared = _inputs(config, bundle)
    schedule = [{"decision_date": str(bundle.calendar[0].date()),
                 "effective_date": str(bundle.calendar[2].date()),
                 "params": {"buy_shares": 100, "sell_shares": 100}}]
    _, values, _ = _parameter_arrays(config, prepared, schedule)
    # Schedule scaling uses the same per-ex-date sum as price_basis; cash is
    # aggregated once and the price spacing follows the resulting 0.475 factor.
    assert values[0, 5:7].tolist() == [200, 200]
    assert values[0, 4] == pytest.approx(config.grid_step * .475)


def test_fractional_entitlements_are_floored_after_same_day_bonus_aggregation(monkeypatch):
    config, bundle = fixture(days=3)
    # The two same-day claims are 0.4 and 0.6 shares. They jointly issue one
    # share; flooring each claim independently would lose both fractions.
    _share_action_market(bundle, [(0, 1, 1, .004), (0, 1, 1, .006)], [100, 100, 99.01, 99.01])
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 100, "buy_shares": 100,
        "sell_shares": 100, "min_shares": 0, "max_shares": None,
        "buy_step": 1., "sell_step": 1., "reference_price_mode": "manual",
        "reference_price": 100., "fees": FeeConfig(commission_rate=0, min_commission=0, slippage_bps=0),
    })
    compiled = engine.run_backtest(config, bundle)["strategies"]["grid"]
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    interpreted = engine.run_backtest(config, bundle)["strategies"]["grid"]
    compiled_ex = next(row for row in compiled["equity"] if row["timestamp"][:10] == str(bundle.calendar[1].date()))
    interpreted_ex = next(row for row in interpreted["equity"] if row["timestamp"][:10] == str(bundle.calendar[1].date()))
    assert compiled_ex["position"] == 101
    assert compiled_ex["equity"] == pytest.approx(10000, abs=.02)
    assert interpreted_ex["position"] == compiled_ex["position"]
    assert interpreted_ex["equity"] == pytest.approx(compiled_ex["equity"])




@pytest.mark.parametrize("frequency", ["weekly", "monthly"])
def test_dca_first_available_day_once_per_period(frequency):
    from stockregression.models import DcaConfig
    from stockregression.share_engine import period_key
    config, bundle = fixture("510300.SH", days=40)
    config = config.model_copy(update={"dca": DcaConfig(frequency=frequency, shares=300)})
    # The first three exchange days cannot execute, so the first contribution
    # moves within its period without accumulating missed purchases.
    blocked = set(bundle.calendar[:3])
    bundle.daily.loc[bundle.daily.date.isin(blocked), ["tradestatus", "volume"]] = 0
    bundle.bars.loc[bundle.bars.date.isin(blocked), "volume"] = 0
    # The next observed session's preclose refers to the last traded close,
    # not the final synthetic quote from the suspended fixture dates.
    resume = bundle.daily.date == bundle.calendar[3]
    bundle.daily.loc[resume, "preclose"] = bundle.daily.loc[bundle.daily.date == bundle.calendar[0] - pd.Timedelta(days=1), "close"].iloc[0]
    result = engine.run_backtest(config, bundle)["strategies"]["dca"]
    expected = {}
    for day in bundle.calendar:
        if day not in blocked:
            expected.setdefault(period_key(day, frequency), str(day.date()))
    assert [trade["timestamp"][:10] for trade in result["trades"]] == list(expected.values())
    assert all(trade["quantity"] == 300 for trade in result["trades"])


def test_dca_wholly_unavailable_period_has_no_investment_or_return():
    config, bundle = fixture("510300.SH")
    bundle.daily["tradestatus"], bundle.daily["volume"] = 0, 0
    bundle.daily.loc[0, ["tradestatus", "volume"]] = [1, 100000]
    bundle.bars["volume"] = 0
    result = engine.run_backtest(config, bundle)["strategies"]["dca"]
    assert result["metrics"]["total_invested"] == 0
    assert result["metrics"]["peak_principal_return"] is None
    assert all(row["principal_curve"] is None for row in result["equity"])
    assert sum(row["reason"] == "dca_period_skipped" for row in result["rejections"]) == 8


def test_zero_position_bounds_do_not_create_assets_or_fixed_capital_returns():
    config, bundle = fixture("510300.SH")
    raw = config.model_dump(mode="json")
    config = BacktestConfig.model_validate({**raw, "initial_shares": 0, "min_shares": 0, "max_shares": 0})
    result = engine.run_backtest(config, bundle)
    assert result["funding"]["initial_equity"] == 0
    assert result["strategies"]["grid"]["trades"] == []
    assert result["strategies"]["grid"]["metrics"]["peak_principal_return"] is None
    assert result["strategies"]["dca"]["metrics"]["total_invested"] > 0
    assert result["strategies"]["buy_hold"]["metrics"]["initial_equity"] == 10000
    assert result["strategies"]["buy_hold"]["trades"] == []
    assert all(point["position"] == 1000 for point in result["strategies"]["buy_hold"]["equity"])






def test_latest_preview_edit_reuses_inflight_same_range_preparation(tmp_path, monkeypatch):
    import threading
    import stockregression.preview as module

    config, bundle = fixture()
    store = Store(tmp_path)
    manifest = store.publish(config.symbol, bundle.bars, bundle.daily, bundle.actions, pd.DataFrame({"date": bundle.daily.date}), bundle.metadata, {})
    bundle.version = manifest["version"]
    started, release = threading.Event(), threading.Event()
    preparation_checks, calculated = [], []

    def ensure(self, config, version=None, progress=None, cancelled=None, **kwargs):
        preparation_checks.append(bool(cancelled()))
        if len(preparation_checks) == 1:
            started.set()
            assert release.wait(5)
            preparation_checks[-1] = bool(cancelled())
        return kwargs.get("cached_bundle") or bundle

    def run(config, data, progress=None, cancelled=None, preview=False, cancel_flag=None):
        calculated.append(config.buy_shares)
        return {"shares": config.buy_shares}

    monkeypatch.setattr(module.DataService, "ensure_bundle", ensure)
    monkeypatch.setattr(module, "run_backtest", run)
    manager = module.PreviewManager(store)
    try:
        manager.submit("page", 1, config)
        assert started.wait(5)
        manager.submit("page", 2, config.model_copy(update={"buy_shares": 200}))
        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and manager.snapshot("page")["status"] != "completed":
            time.sleep(.01)
        state = manager.snapshot("page")
        assert state["revision"] == 2 and state["result"] == {"shares": 200}
        assert preparation_checks == [False, False]
        assert calculated == [200]
    finally:
        release.set()
        manager.stop()


def test_share_kernel_preview_cancel_flag_interrupts_at_first_bar():
    import numpy as np

    config, bundle = fixture()
    with pytest.raises(InterruptedError, match="回测已取消"):
        engine.run_backtest(config, bundle, preview=True, cancel_flag=np.ones(1, dtype=np.int32))


def test_share_kernel_running_preview_cancels_from_another_thread(monkeypatch):
    import threading

    import stockregression.share_kernel as share_kernel
    from stockregression.engine import _NUMBA_FASTPATH_ENABLED

    assert _NUMBA_FASTPATH_ENABLED, "该回归测试需要 Numba fast path"
    config, bundle = fixture()
    flag = np.zeros(1, dtype=np.int32)
    expected = engine.run_backtest(config, bundle, summary_only=True, preview=True)
    warm = engine.run_backtest(config, bundle, summary_only=True, preview=True,
                               cancel_flag=flag)
    assert warm == expected

    first_bar = bundle.bars.iloc[[0]]
    long_bars = first_bar.loc[first_bar.index.repeat(250_000)].reset_index(drop=True)
    long_bundle = DataBundle(
        symbol=bundle.symbol, bars=long_bars, daily=bundle.daily.copy(),
        actions=bundle.actions.copy(), calendar=bundle.calendar.copy(),
        quality=bundle.quality.copy(), version="long-preview-cancel",
        metadata=bundle.metadata.copy(), warnings=list(bundle.warnings),
    )
    entered_kernel = threading.Event()
    release_kernel = threading.Event()
    original_simulate = share_kernel.simulate

    def signal_then_simulate(*args, **kwargs):
        entered_kernel.set()
        assert release_kernel.wait(10)
        return original_simulate(*args, **kwargs)

    monkeypatch.setattr(share_kernel, "simulate", signal_then_simulate)
    long_flag = np.zeros(1, dtype=np.int32)
    outcome = []

    def run():
        try:
            engine.run_backtest(config, long_bundle, summary_only=True,
                                preview=True, cancel_flag=long_flag)
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert entered_kernel.wait(10), "long preview never entered the compiled kernel"
    canceller = threading.Thread(target=lambda: (time.sleep(.005), long_flag.__setitem__(0, 1)))
    canceller.start()
    release_kernel.set()
    worker.join(10)
    canceller.join(2)
    assert not worker.is_alive(), "cancelled kernel worker did not exit"
    assert len(outcome) == 1 and isinstance(outcome[0], InterruptedError)




def test_etf_dated_limits_and_fees_do_not_use_stock_defaults():
    from stockregression.rules import etf_price_limit_rate, commissions_and_taxes
    terms = instrument_terms("159915.SZ")
    assert etf_price_limit_rate(terms, "2020-08-21") == .1
    assert etf_price_limit_rate(terms, "2020-08-24") == .2
    assert instrument_terms("159985.SZ")["settlement"] == "T0"
    assert instrument_terms("159954.SZ", {"name": "H股ETF"})["settlement"] == "T0"
    assert commissions_and_taxes(10000, "sell", "2014-01-01", 0, 0, symbol="510300.SH")["total"] == 0




def test_market_dry_run_reports_universe_without_creating_data_job(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from stockregression import cli
    from stockregression.settings import Settings
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(data_dir=str(tmp_path)))
    monkeypatch.setattr("stockregression.providers.fetch_instruments", lambda *args, **kwargs: [{"symbol": "600000.SH"}, {"symbol": "510300.SH"}])
    result = CliRunner().invoke(cli.app, ["sync", "--scope", "market", "--mode", "history", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "ETF 1" in result.output
    assert not (tmp_path / "catalog.sqlite").exists()


def test_incremental_publish_reuses_other_year_and_keeps_old_snapshot(tmp_path):
    config, bundle = fixture("510300.SH")
    store = Store(tmp_path)
    before = bundle.daily.iloc[:1].copy()
    before["date"] = pd.Timestamp("2023-12-29")
    daily = pd.concat([before, bundle.daily], ignore_index=True)
    original = store.publish(config.symbol, bundle.bars, daily, bundle.actions, pd.DataFrame({"date": daily.date}), bundle.metadata, {})
    correction = bundle.daily.tail(1).copy()
    correction["volume"] += 100
    updated = store.publish(config.symbol, pd.DataFrame(), correction, pd.DataFrame(), pd.DataFrame(), bundle.metadata, {}, incremental=True)
    old_year = [path for path in original["files"]["daily"] if "/2023/" in path]
    assert old_year and all(path in updated["files"]["daily"] for path in old_year)
    assert updated["version"] != original["version"]
    historical = store.snapshot(config.symbol, original["version"])
    assert store.read_files(historical["files"]["daily"]).volume.sum() == daily.volume.sum()




