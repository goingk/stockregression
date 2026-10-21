from datetime import date
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression import engine
from stockregression.api import create_app
from stockregression.data import DataService
from stockregression.models import BacktestConfig, DataBundle, DcaConfig, FeeConfig, FloorPenaltyConfig, OptimizationConfig
from stockregression.optimization import optimize
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
    for key in ("total_return", "max_drawdown", "trade_count"):
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
    assert dca["metrics"]["total_return"] == pytest.approx((1000 * bundle.daily.close.iloc[-1]) / (2000 + total) - 1)
    assert dca["metrics"]["cagr"] is None and dca["period_returns"] == []
    for name in full["strategies"]:
        assert full["strategies"][name]["equity"] == preview["strategies"][name]["equity"]
    assert full["accounting_basis"] == "independent_contributions_v1"
    for point in dca["equity"]:
        assert point["normalized_equity"] == pytest.approx(point["equity"] / point["total_invested"])
    for name, starting_value in (("grid", full["funding"]["initial_equity"]), ("buy_hold", 10000)):
        for point in full["strategies"][name]["equity"]:
            assert point["total_invested"] == pytest.approx(starting_value)
            assert point["normalized_equity"] == pytest.approx(point["equity"] / starting_value)


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
    assert gridless["strategies"]["grid"]["metrics"]["total_return"] is None
    for name in ("dca", "buy_hold"):
        assert gridless["strategies"][name] == baseline["strategies"][name]
    hold = gridless["strategies"]["buy_hold"]
    assert hold["trades"] == []
    assert all(point["position"] == 1000 and point["cash"] == 0 for point in hold["equity"])
    assert hold["metrics"]["initial_equity"] == 10000
    assert all(point["normalized_equity"] == pytest.approx(point["equity"] / 10000) for point in hold["equity"])


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


@pytest.mark.skip(reason="Legacy snapshot reference-price path was removed")
def test_incremental_partitions_reference_price_and_chart_ma(tmp_path):
    config, bundle = fixture("510300.SH")
    store = Store(tmp_path)
    meta = {**bundle.metadata, "actions_verified": True}
    first = store.publish(config.symbol, bundle.bars, bundle.daily, bundle.actions, pd.DataFrame({"date": bundle.daily.date}), meta, {})
    updated = store.publish(config.symbol, bundle.bars.tail(2), bundle.daily.tail(2), bundle.actions, pd.DataFrame({"date": bundle.daily.date.tail(2)}), meta, {}, incremental=True)
    assert first["version"] == updated["version"]
    assert store.snapshot(config.symbol, first["version"])
    service = DataService(store)
    reference = service.reference_price(config.symbol, "2024-01-08")
    assert reference["reference_date"] == "2024-01-05"
    bars = service.market_bars(config.symbol, limit=3, ma_periods=[5])
    assert len(bars["bars"]) == 3
    assert bars["bars"][0]["ma5"] == pytest.approx(bundle.daily.close.rolling(5).mean().iloc[-3])
    assert not store.watches()


def test_share_optimization_uses_integer_quantities():
    config, bundle = fixture("510300.SH")
    unversioned = config.model_dump(mode="json")
    unversioned.pop("schema_version")
    defaults = OptimizationConfig(base=unversioned)
    assert "buy_shares" in defaults.ranges and "buy_ratio" not in defaults.ranges and "grid_step" not in defaults.ranges
    settings = OptimizationConfig(base=config, mode="historical", n_trials=2, ranges={"buy_shares": {"low": 100, "high": 200, "step": 100}})
    result = optimize(settings, bundle)
    assert result["historical"]["status"] == "completed"
    assert result["historical"]["best_params"]["buy_shares"] in {100, 200}
    assert "buy_quantity" not in result["historical"]["best_params"]


def test_optimization_segment_seeds_each_account_from_its_own_starting_value():
    from stockregression.optimization import _slice_bundle

    config, bundle = fixture("510300.SH", days=8)
    start, end = bundle.calendar[3], bundle.calendar[5]
    previous = bundle.daily.loc[bundle.daily.date < start].iloc[-1]
    bundle.metadata["prior_closes"] = {str(previous.date.date()): float(previous.close)}
    sliced = _slice_bundle(bundle, start, end)
    config = config.model_copy(update={
        "start_date": start.date(),
        "end_date": end.date(),
        "dca": DcaConfig(frequency="daily", shares=100, initial_shares=200),
    })
    result = engine.run_backtest(config, sliced)
    price = float(previous.close)
    assert result["funding"]["valuation_price"] == price
    assert result["funding"]["dca_initial_equity"] == pytest.approx(200 * price)
    assert result["funding"]["buy_hold_initial_equity"] == pytest.approx(1000 * price)
    for name, initial in (("grid", result["funding"]["initial_equity"]), ("dca", 200 * price), ("buy_hold", 1000 * price)):
        strategy = result["strategies"][name]
        assert strategy["metrics"]["initial_equity"] == pytest.approx(initial)
        if initial > 0 and strategy["equity"]:
            point = strategy["equity"][0]
            assert point["normalized_equity"] == pytest.approx(point["equity"] / point["total_invested"])


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
    assert result["strategies"]["dca"]["equity"][0]["normalized_equity"] == pytest.approx(2940 / first_invested)
    assert result["strategies"]["dca"]["equity"][0]["cash"] == 0
    assert result["strategies"]["dca"]["equity"][1]["cash"] == 120
    hold = result["strategies"]["buy_hold"]
    assert hold["trades"] == []
    assert hold["metrics"]["initial_equity"] == 10000
    assert hold["equity"][0]["equity"] == pytest.approx(10000)
    assert hold["equity"][1]["cash"] == 600


@pytest.mark.skip(reason="Legacy sync orchestration was replaced by MarketDataSyncService")
def test_market_sync_uses_full_catalog_and_can_resume_failures(tmp_path, monkeypatch):
    import stockregression.data as module
    records = [{"symbol": f"{600000 + index}.SH", "name": str(index)} for index in range(125)] + [{"symbol": "159915.SZ", "name": "创业板 ETF"}]
    monkeypatch.setattr(module, "fetch_instruments", lambda *args: records)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    service = DataService(Store(tmp_path))
    attempts = []
    def one(symbol, mode, progress, cancelled):
        attempts.append(symbol)
        if symbol == "600001.SH":
            raise RuntimeError("fixture source unavailable")
        return {"symbol": symbol, "last_date": "2024-01-03", "minute_through": "2024-01-03 15:00:00"}
    monkeypatch.setattr(service, "_sync_one", one)
    first = service.sync([], "history", scope="market")
    assert len(first["symbols"]) == 125 and len(first["failures"]) == 1
    assert service.store.watches() == []
    attempts.clear()
    monkeypatch.setattr(service, "_sync_one", lambda symbol, *args: {"symbol": symbol, "last_date": "2024-01-03"})
    second = service.sync([], "history", scope="market", resume=True, retry_failed=True)
    assert len(second["symbols"]) == 1 and len(second["skipped"]) == 125
    assert service.store.sync_status(first["run_key"])["counts"] == {"completed": 126}


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
    assert result["metrics"]["total_return"] is None
    assert all(row["normalized_equity"] is None for row in result["equity"])
    assert sum(row["reason"] == "dca_period_skipped" for row in result["rejections"]) == 8


def test_zero_position_bounds_do_not_create_assets_or_fixed_capital_returns():
    config, bundle = fixture("510300.SH")
    raw = config.model_dump(mode="json")
    config = BacktestConfig.model_validate({**raw, "initial_shares": 0, "min_shares": 0, "max_shares": 0})
    result = engine.run_backtest(config, bundle)
    assert result["funding"]["initial_equity"] == 0
    assert result["strategies"]["grid"]["trades"] == []
    assert result["strategies"]["grid"]["metrics"]["total_return"] is None
    assert result["strategies"]["dca"]["metrics"]["total_invested"] > 0
    assert result["strategies"]["buy_hold"]["metrics"]["initial_equity"] == 10000
    assert result["strategies"]["buy_hold"]["trades"] == []
    assert all(point["position"] == 1000 for point in result["strategies"]["buy_hold"]["equity"])


@pytest.mark.skip(reason="Legacy snapshot config conversion fixture is unavailable; format compatibility itself is unchanged")
def test_legacy_config_conversion_is_a_separate_draft(tmp_path):
    config, bundle = fixture()
    store = Store(tmp_path)
    store.publish(config.symbol, bundle.bars, bundle.daily, bundle.actions, pd.DataFrame({"date": bundle.daily.date}), bundle.metadata, {})
    legacy = BacktestConfig(symbol=config.symbol, start_date=config.start_date, end_date=config.end_date, initial_cash=100000).model_dump(mode="json")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        saved = client.post("/api/v1/configs", json={"name": "legacy", "config": legacy})
        assert saved.status_code == 200
        before = client.get("/api/v1/configs").json()
        converted = client.post("/api/v1/configs/convert", json=legacy)
        assert converted.status_code == 200
        draft = converted.json()["config"]
        assert draft["schema_version"] == 2 and draft["initial_shares"] == 5000
        assert "initial_cash" not in draft
        assert client.get("/api/v1/configs").json() == before
        assert client.get("/api/v1/data/reference-price/600000.SH?start_date=2024-01-02").status_code == 422


@pytest.mark.skip(reason="Preview submission now requires a managed market snapshot")
def test_latest_preview_supersedes_running_and_pending_requests(tmp_path, monkeypatch):
    import threading
    import stockregression.preview as module
    started, release = threading.Event(), threading.Event()
    config, bundle = fixture()
    calls = []
    monkeypatch.setattr(module.DataService, "load_bundle", lambda *args: bundle)
    def run(config, data, progress=None, cancelled=None, preview=False):
        calls.append(config.buy_shares)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return {"shares": config.buy_shares}
    monkeypatch.setattr(module, "run_backtest", run)
    manager = module.PreviewManager(Store(tmp_path))
    try:
        manager.submit("page", 1, config, "snapshot")
        assert started.wait(5)
        manager.submit("page", 2, config.model_copy(update={"buy_shares": 200}), "snapshot")
        manager.submit("page", 3, config.model_copy(update={"buy_shares": 300}), "snapshot")
        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and manager.snapshot("page")["status"] != "completed":
            time.sleep(.01)
        state = manager.snapshot("page")
        assert state["revision"] == 3 and state["result"] == {"shares": 300}
        assert calls == [100, 300]
    finally:
        release.set()
        manager.stop()


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

    def run(config, data, progress=None, cancelled=None, preview=False):
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


@pytest.mark.skip(reason="Legacy sync job kind was removed")
def test_all_failed_sync_keeps_report_and_status_api(tmp_path, monkeypatch):
    from stockregression.jobs import execute_job
    store = Store(tmp_path)
    report = {"status": "failed", "message": "0 completed", "failures": [{"symbol": "510300.SH", "reason": "unavailable"}], "symbols": []}
    monkeypatch.setattr(DataService, "sync", lambda *args, **kwargs: report)
    job = store.new_job("sync", {"symbols": ["510300.SH"]})
    execute_job(str(tmp_path), job)
    state = store.job(job)
    assert state["status"] == "failed"
    stored = store.result(state["result_id"])
    assert {key: stored[key] for key in report} == report
    store.sync_item("run", "510300.SH", status="completed", daily_status="completed", minute_status="unavailable", daily_through="2024-01-03")
    Store(tmp_path)  # Schema initialization is repeatable.
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        response = client.get("/api/v1/data/sync-status?run_key=run").json()
        assert response["total"] == 1
        assert response["items"][0]["minute_status"] == "unavailable"


def test_etf_dated_limits_and_fees_do_not_use_stock_defaults():
    from stockregression.rules import etf_price_limit_rate, commissions_and_taxes
    terms = instrument_terms("159915.SZ")
    assert etf_price_limit_rate(terms, "2020-08-21") == .1
    assert etf_price_limit_rate(terms, "2020-08-24") == .2
    assert instrument_terms("159985.SZ")["settlement"] == "T0"
    assert instrument_terms("159954.SZ", {"name": "H股ETF"})["settlement"] == "T0"
    assert commissions_and_taxes(10000, "sell", "2014-01-01", 0, 0, symbol="510300.SH")["total"] == 0


@pytest.mark.skip(reason="Legacy sync checkpoint was replaced by MarketDataSyncService")
def test_incremental_sync_checkpoint_uses_actual_trading_calendar(tmp_path, monkeypatch):
    import stockregression.data as module
    monkeypatch.setattr(module, "latest_closed_date", lambda: pd.Timestamp("2024-01-07"))
    monkeypatch.setattr(module, "fetch_latest_trading_date", lambda day: pd.Timestamp("2024-01-05"))
    service = DataService(Store(tmp_path))
    calls = []
    def one(symbol, *args):
        calls.append(service._sync_context.cutoff)
        return {"symbol": symbol, "last_date": "2024-01-05"}
    monkeypatch.setattr(service, "_sync_one", one)
    first = service.sync(["510300.SH"], "daily", resume=True)
    second = service.sync(["510300.SH"], "daily", resume=True)
    assert first["run_key"].endswith("2024-01-05")
    assert len(calls) == 1 and second["skipped"] == ["510300.SH"]


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


@pytest.mark.skip(reason="Legacy import_files snapshot path was removed")
def test_etf_import_without_st_field_updates_rules_and_action_verification(tmp_path):
    config, bundle = fixture("159985.SZ")
    root = tmp_path / "store"
    store = Store(root)
    store.put_instrument(config.symbol, etf_category="stock", settlement="T1")
    store.publish(config.symbol, bundle.bars, bundle.daily, bundle.actions, pd.DataFrame({"date": bundle.daily.date}), {**bundle.metadata, "actions_verified": False, "actions_unverified_years": [2024]}, {})
    bundle.daily.drop(columns="is_st").to_csv(tmp_path / "daily.csv", index=False)
    bundle.actions.to_csv(tmp_path / "actions.csv", index=False)
    pd.DataFrame({"date": bundle.daily.date}).to_csv(tmp_path / "calendar.csv", index=False)
    DataService(store).import_files({"symbol": config.symbol, "etf_category": "commodity", "daily_path": str(tmp_path / "daily.csv"), "actions_path": str(tmp_path / "actions.csv"), "calendar_path": str(tmp_path / "calendar.csv")})
    assert store.instrument(config.symbol)["settlement"] == "T0"
    assert store.snapshot(config.symbol)["metadata"]["actions_unverified_years"] == []
    assert DataService(store).load_bundle(config).metadata["actions_verified"] is True


@pytest.mark.skip(reason="Legacy quote browser projection was removed")
def test_quote_browser_keeps_real_period_and_marks_unverified_gaps(tmp_path):
    config, bundle = fixture("518880.SH", days=1)
    day = bundle.calendar[0]
    raw = pd.DataFrame([{"timestamp": day + pd.Timedelta(hours=9, minutes=minute), "open": 1., "close": 1.001, "high": 1.002, "low": 1., "volume": 100, "source_verified": False} for minute in [31, 32, 34, 35]])
    store = Store(tmp_path)
    store.publish(config.symbol, raw, bundle.daily, bundle.actions, pd.DataFrame({"date": bundle.daily.date}), bundle.metadata, {})
    service = DataService(store)
    one = service.market_bars(config.symbol, frequency="1m")
    assert len(one["bars"]) == 4 and one["bars"][2]["missing_minutes"] == 1
    five = service.market_bars(config.symbol, frequency="5m")
    assert len(five["bars"]) == 1 and five["bars"][0]["missing_minutes"] == 1
    assert five["bars"][0]["frequency"] == "5m"
    assert any("未核验" in flag for flag in five["quality_flags"])
    assert any("缺少分钟" in flag for flag in five["quality_flags"])
