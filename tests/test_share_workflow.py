from datetime import date
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression import engine
from stockregression.api import create_app
from stockregression.data import DataService
from stockregression.models import BacktestConfig, DataBundle, OptimizationConfig
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


def test_dca_external_contributions_are_not_profit_and_preview_matches_full():
    config, bundle = fixture("510300.SH")
    full = engine.run_backtest(config, bundle)
    preview = engine.run_backtest(config, bundle, preview=True)
    trades = full["strategies"]["dca"]["trades"]
    total = sum(trade["quantity"] * trade["price"] + trade["fees"] for trade in trades)
    dca = full["strategies"]["dca"]
    assert len(trades) == 8
    assert dca["metrics"]["total_invested"] == pytest.approx(total)
    assert dca["metrics"]["total_return"] == pytest.approx((8 * 100 * bundle.daily.close.iloc[-1]) / total - 1)
    assert dca["metrics"]["cagr"] is None and dca["period_returns"] == []
    for name in full["strategies"]:
        assert full["strategies"][name]["equity"] == preview["strategies"][name]["equity"]


def test_compiled_and_interpreted_share_accounting_match(monkeypatch):
    config, bundle = fixture("518880.SH")
    fast = engine.run_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    slow = engine.run_backtest(config, bundle)
    assert fast == slow
    summary = engine.run_backtest(config, bundle, summary_only=True)
    assert summary["strategies"]["grid"]["metrics"] == fast["strategies"]["grid"]["metrics"]


def test_fee_decimals_quantities_and_inactive_fields():
    config, _ = fixture()
    raw = config.model_dump(mode="json")
    raw.update(grid_mode="geometric", buy_step=0, sell_step="", fees={"commission_rate": .00012345})
    parsed = BacktestConfig.model_validate(raw)
    assert parsed.fees.commission_rate == .00012345
    assert "buy_step" not in parsed.model_dump()
    search = OptimizationConfig.model_validate({"base": raw, "ranges": {"buy_step": {"low": -1, "high": -2, "step": 0}, "buy_shares": {"low": 100, "high": 200, "step": 100}}})
    assert set(search.ranges) == {"buy_shares"}
    for value in (1, 99, 150, 100.5):
        with pytest.raises(ValueError):
            BacktestConfig.model_validate({**raw, "buy_shares": value})


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


def test_v2_api_preview_does_not_persist_jobs_or_results(tmp_path):
    config, bundle = fixture("510300.SH")
    store = Store(tmp_path)
    store.publish(config.symbol, bundle.bars, bundle.daily, bundle.actions, pd.DataFrame({"date": bundle.daily.date}), {**bundle.metadata, "actions_verified": True}, {})
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        raw = config.model_dump(mode="json")
        raw["end_date"] = "2024-01-31"
        response = client.post("/api/v1/backtests/preview", json={"session_id": "test", "revision": 1, "config": raw})
        assert response.status_code == 200
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            state = client.get("/api/v1/backtests/preview/test").json()
            if state["status"] in {"completed", "failed"}:
                break
            time.sleep(.05)
        assert state["status"] == "completed", state
        assert state["result"]["effective_end_date"] == str(bundle.calendar[-1].date())
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
    config = config.model_copy(update={"max_shares": 1000, "buy_step": 1., "sell_step": 1.})
    bundle.daily.loc[bundle.daily.date == bundle.calendar[0], "preclose"] = 9.4
    bundle.actions = normalize_actions(pd.DataFrame([{"record_date": "2024-01-02", "ex_date": "2024-01-03", "pay_date": "2024-01-04", "cash_per_share": .6, "bonus_ratio": 0}]))
    result = engine.run_backtest(config, bundle)
    assert result["strategies"]["grid"]["equity"][0]["equity"] == pytest.approx(10000)
    assert result["strategies"]["grid"]["equity"][1]["cash"] == 600
    assert result["strategies"]["dca"]["equity"][0]["cash"] == 0


def test_market_sync_uses_full_catalog_and_can_resume_failures(tmp_path, monkeypatch):
    import stockregression.data as module
    records = [{"symbol": f"{600000 + index}.SH", "name": str(index)} for index in range(125)] + [{"symbol": "159915.SZ", "name": "创业板 ETF"}]
    monkeypatch.setattr(module, "fetch_instruments", lambda revision: records)
    monkeypatch.setattr("stockregression.providers.estimate_history_bytes", lambda symbols, settings: 1000)
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


def test_latest_preview_supersedes_running_and_pending_requests(tmp_path, monkeypatch):
    import threading
    import stockregression.preview as module
    started, release = threading.Event(), threading.Event()
    config, bundle = fixture()
    calls = []
    monkeypatch.setattr(module.DataService, "load_bundle", lambda *args: bundle)
    def run(config, data, cancelled, preview):
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
    monkeypatch.setattr("stockregression.providers.fetch_instruments", lambda revision: [{"symbol": "600000.SH"}, {"symbol": "510300.SH"}])
    monkeypatch.setattr("stockregression.providers.estimate_history_bytes", lambda symbols, settings: 2 ** 30)
    result = CliRunner().invoke(cli.app, ["sync", "--scope", "market", "--mode", "history", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "ETF 1" in result.output and "1.00 GiB" in result.output
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
