import pandas as pd
import pytest

from stockregression import engine
from stockregression.data_resolution import _digest
from stockregression.models import BacktestConfig
from stockregression.providers import normalize_actions
from stockregression.quality import prepare_bundle
from stockregression.share_engine import _inputs
from test_share_workflow import fixture


POLICY = "minute_verified_daily_then_skip_v1"


def _skip_fixture():
    config, bundle = fixture(symbol="159611.SZ", days=5)
    config = config.model_copy(update={
        "schema_version": 3, "initial_shares": 1000, "min_shares": 0,
        "max_shares": 3000, "buy_shares": 100, "sell_shares": 100,
        "reference_price_mode": "manual", "reference_price": 10.0,
        "grid_mode": "geometric", "buy_step": .02, "sell_step": .02,
        "data_policy": POLICY,
    })
    day = bundle.calendar[1]
    resolution = {
        "policy": POLICY, "plan_hash": "frozen-test-plan",
        "symbol": bundle.symbol,
        "days": [{"date": str(d.date()), "mode": "skip" if d == day else "daily",
                  "reason_code": "daily_unavailable" if d == day else "minute_unavailable",
                  "reason": "测试跳过" if d == day else "测试日线降级"} for d in bundle.calendar],
        "counts": {"skip": 1, "daily": 4},
    }
    resolution["plan_hash"] = _digest({key: value for key, value in resolution.items() if key != "plan_hash"})
    bundle.metadata["data_resolution"] = resolution
    bundle.bars = bundle.bars[bundle.bars.timestamp.dt.normalize() != day].copy()
    return config, bundle, day


def test_backtest_config_omits_default_data_policy_from_serialized_fingerprint():
    config, _ = fixture()
    assert "data_policy" not in config.model_dump(mode="json")
    with pytest.raises(ValueError, match="仅适用于 ETF"):
        BacktestConfig.model_validate({**config.model_dump(mode="python"), "data_policy": POLICY})


def test_skip_event_expires_pending_and_is_absent_from_equity_series():
    config, bundle, skip_day = _skip_fixture()
    result = engine.run_backtest(config, bundle)
    grid = result["strategies"]["grid"]
    assert all(point["timestamp"][:10] != str(skip_day.date()) for point in grid["equity"])
    # A pending intent presented at a skip event is explicitly reported as unavailable_day.
    assert any(rejection.get("reason") == "unavailable_day" for rejection in grid["rejections"])


def test_skip_event_compiled_and_python_kernel_outputs_match():
    config, bundle, _ = _skip_fixture()
    compiled_enabled = engine._NUMBA_FASTPATH_ENABLED
    try:
        engine._NUMBA_FASTPATH_ENABLED = True
        compiled = engine.run_backtest(config, bundle)
        engine._NUMBA_FASTPATH_ENABLED = False
        interpreted = engine.run_backtest(config, bundle)
    finally:
        engine._NUMBA_FASTPATH_ENABLED = compiled_enabled
    assert compiled["strategies"]["grid"]["metrics"] == interpreted["strategies"]["grid"]["metrics"]
    assert compiled["strategies"]["grid"]["trades"] == interpreted["strategies"]["grid"]["trades"]


def test_cached_inputs_do_not_bypass_resolution_integrity_check():
    config, bundle, _ = _skip_fixture()
    _inputs(config, bundle)
    bundle.metadata["data_resolution"]["days"][0]["mode"] = "skip"
    with pytest.raises(ValueError, match="RESOLUTION_PLAN_MISMATCH"):
        _inputs(config, bundle)


def test_prepare_bundle_uses_frozen_daily_and_skip_modes_without_forward_filling():
    from test_quality_rules import daily_fixture

    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    plan = {
        "policy": POLICY, "symbol": "159611.SZ",
        "days": [
            {"date": str(days[0].date()), "mode": "daily", "reason_code": "minute_unavailable", "reason": "minute missing"},
            {"date": str(days[1].date()), "mode": "skip", "reason_code": "daily_unavailable", "reason": "daily missing"},
        ],
    }
    plan["plan_hash"] = _digest(plan)
    daily = daily_fixture(days[:1])
    bundle = prepare_bundle("159611.SZ", pd.DataFrame(), daily, normalize_actions(pd.DataFrame()),
        pd.DataFrame({"date": days}), days[0], days[-1], "1m", "frozen-v1",
        {"asset_type": "etf", "price_tick": .001, "data_resolution": plan}, {})
    assert bundle.quality["days"][0]["mode"] == "daily"
    assert bundle.quality["days"][1]["mode"] == "skip"
    assert len(bundle.bars.loc[bundle.bars.frequency == "1d"]) == 1
    assert not (bundle.bars.timestamp.dt.normalize() == days[1]).any()
    config = BacktestConfig(schema_version=3, symbol="159611.SZ", start_date=days[0].date(),
        end_date=days[1].date(), frequency="1m", initial_shares=1000, max_shares=3000,
        buy_shares=100, sell_shares=100, data_policy=POLICY)
    prepared = _inputs(config, bundle)
    assert list(prepared["days"]) == list(days)
    assert prepared["skip"].tolist() == [False, True]


def test_dividend_recorded_before_skip_is_booked_on_skip_ex_date():
    from stockregression.providers import normalize_actions

    config, bundle, skip_day = _skip_fixture()
    record_day = bundle.calendar[0]
    pay_day = bundle.calendar[2]
    bundle.actions = normalize_actions(pd.DataFrame([{
        "record_date": record_day, "ex_date": skip_day, "pay_date": pay_day,
        "cash_per_share": .1, "bonus_ratio": 0,
    }]))
    result = engine.run_backtest(config, bundle)
    grid = result["strategies"]["grid"]
    after_pay = next(point for point in grid["equity"] if point["timestamp"][:10] == str(pay_day.date()))
    assert after_pay["cash"] >= .1 * config.initial_shares


def test_dividend_record_date_on_skip_preserves_entitlement():
    config, bundle, skip_day = _skip_fixture()
    pay_day = bundle.calendar[3]
    bundle.actions = normalize_actions(pd.DataFrame([{
        "record_date": skip_day, "ex_date": bundle.calendar[2], "pay_date": pay_day,
        "cash_per_share": .1, "bonus_ratio": .1, "stock_date": pay_day,
    }]))
    result = engine.run_backtest(config, bundle)
    grid = result["strategies"]["grid"]
    assert all(point["timestamp"][:10] != str(skip_day.date()) for point in grid["equity"])
    after_pay = next(point for point in grid["equity"] if point["timestamp"][:10] == str(pay_day.date()))
    assert after_pay["cash"] >= .1 * config.initial_shares
    assert after_pay["position"] >= config.initial_shares


def test_t_plus_one_advances_across_skipped_session_without_cross_gap_fill():
    config, bundle, skip_day = _skip_fixture()
    config = config.model_copy(update={"initial_shares": 0})
    plan = bundle.metadata["data_resolution"]
    for day in plan["days"]:
        if day["mode"] != "skip":
            day["mode"] = "minute"
    plan["plan_hash"] = _digest({key: value for key, value in plan.items() if key != "plan_hash"})
    bars = []
    for index, day in enumerate(bundle.calendar):
        if day == skip_day:
            continue
        for minute, (opening, close) in enumerate([(10., 9.7), (9.7, 10.4), (10.4, 10.4)] if index == 0
                                                  else [(10.4, 10.4), (10.4, 10.4)]):
            bars.append({"timestamp": day + pd.Timedelta(hours=9, minutes=31 + minute),
                "open": opening, "high": max(opening, close), "low": min(opening, close),
                "close": close, "volume": 10000., "frequency": "1m", "tradable": True,
                "is_synthetic": False})
    bundle.bars = pd.DataFrame(bars)
    result = engine.run_backtest(config, bundle)
    grid = result["strategies"]["grid"]
    buys = [trade for trade in grid["trades"] if trade["side"] == "buy"]
    sells = [trade for trade in grid["trades"] if trade["side"] == "sell"]
    assert buys and sells
    assert buys[0]["timestamp"][:10] == str(bundle.calendar[0].date())
    assert all(trade["timestamp"][:10] != str(skip_day.date()) for trade in grid["trades"])
    assert any(trade["timestamp"][:10] == str(bundle.calendar[2].date()) for trade in sells)
    assert any(rejection["reason"] == "unavailable_day" for rejection in grid["rejections"])
