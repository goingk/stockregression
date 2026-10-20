from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

import stockregression.engine as engine
from stockregression.engine import open_timestamp, run_backtest
from stockregression.models import BacktestConfig, DataBundle


SYMBOL = "600000.SH"


def _config(start: str, end: str | None = None, **values) -> BacktestConfig:
    return BacktestConfig(
        symbol=SYMBOL,
        start_date=date.fromisoformat(start),
        end_date=date.fromisoformat(end) if end else None,
        initial_cash=values.pop("initial_cash", 100_000),
        fees=values.pop("fees", {"commission_rate": 0, "min_commission": 0, "slippage_bps": 0}),
        **values,
    )


def _bar(ts: str, op: float, close: float, *, frequency="1m", **values):
    return {
        "timestamp": pd.Timestamp(ts), "open": op, "high": max(op, close),
        "low": min(op, close), "close": close, "volume": 100_000,
        "turnover": 0, "frequency": frequency, "tradable": True,
        "is_synthetic": False, "preclose": None, "is_st": False,
        "limit_up": 100.0, "limit_down": 0.01, **values,
    }


def _daily(day: str, op: float, close: float, *, preclose=None, tradestatus=1, **values):
    return {
        "date": pd.Timestamp(day), "open": op, "high": max(op, close),
        "low": min(op, close), "close": close, "volume": 100_000,
        "turnover": 0, "preclose": preclose, "tradestatus": tradestatus,
        "is_st": False, **values,
    }


def _bundle(bars, days, *, daily=None, actions=None, quality=None):
    return DataBundle(
        symbol=SYMBOL,
        bars=pd.DataFrame(bars),
        daily=pd.DataFrame(daily or []),
        actions=pd.DataFrame(actions or []),
        calendar=pd.DatetimeIndex(days),
        quality=quality or {},
        version="test-v1",
        metadata={"board": "main"},
    )


def test_real_close_signal_fills_at_next_real_open_and_skips_synthetic_bar():
    bars = [
        _bar("2024-01-02 09:31", 10.0, 9.0),  # close crosses the trailing buy band
        _bar("2024-01-02 09:32", 9.0, 9.0, is_synthetic=True),
        _bar("2024-01-02 09:33", 8.5, 8.6),
    ]
    result = run_backtest(_config("2024-01-02"), _bundle(bars, ["2024-01-02"]))
    trades = result["strategies"]["grid"]["trades"]
    assert [t["reason"] for t in trades] == ["initial_position", "grid_buy"]
    assert trades[0]["timestamp"] == "2024-01-02T09:30:00"
    assert trades[1]["timestamp"] == "2024-01-02T09:32:00"
    # The order generated at the first bar's close did not fill at the synthetic 09:31.
    assert trades[1]["price"] == 8.5


def test_new_shares_obey_t_plus_one():
    bars = [
        _bar("2024-01-02 09:31", 10.0, 11.0),
        _bar("2024-01-02 09:32", 11.0, 11.0),
    ]
    result = run_backtest(_config("2024-01-02"), _bundle(bars, ["2024-01-02"]))
    grid = result["strategies"]["grid"]
    assert [trade["side"] for trade in grid["trades"]] == ["buy"]
    assert any(row["reason"] == "t_plus_1" for row in grid["rejections"])


def test_fixed_grid_uses_independent_buy_and_sell_spacing():
    config = _config(
        "2024-01-02", schema_version=3, grid_mode="geometric",
        buy_step=.10, sell_step=.20,
    )
    assert engine._cell(10.0, 10.0, config) == 0
    assert engine._cell(9.95, 10.0, config) == 0
    assert engine._cell(9.0, 10.0, config) == -1
    assert engine._cell(8.19, 10.0, config) == -1
    assert engine._cell(8.1, 10.0, config) == -2
    assert engine._cell(10.05, 10.0, config) == 0
    assert engine._cell(12.0, 10.0, config) == 1
    narrow_sell = config.model_copy(update={"sell_step": .02})
    assert engine._cell(10.25, 10.0, narrow_sell) > 0
    for price in (8.1, 9.0, 9.95, 10.0, 10.25, 12.0):
        assert engine._nb_cell(price, 10.0, 2, config.buy_step, config.sell_step) == engine._cell(price, 10.0, config)


def test_pending_order_is_cancelled_on_real_suspension_and_daily_fallback_is_one_bar():
    bars = [
        _bar("2024-01-02 09:31", 10.0, 9.0),
        _bar("2024-01-04 15:00", 8.8, 8.9, frequency="1d"),
    ]
    daily = [
        _daily("2024-01-02", 10, 9, preclose=10),
        _daily("2024-01-03", 9, 9, preclose=9, tradestatus=0),
        _daily("2024-01-04", 8.8, 8.9, preclose=9),
    ]
    quality = {"days": [
        {"date": "2024-01-02", "mode": "minute"},
        {"date": "2024-01-03", "mode": "suspended"},
        {"date": "2024-01-04", "mode": "daily"},
    ]}
    result = run_backtest(
        _config("2024-01-02", "2024-01-04"),
        _bundle(bars, ["2024-01-02", "2024-01-03", "2024-01-04"], daily=daily, quality=quality),
    )
    grid = result["strategies"]["grid"]
    assert [trade["reason"] for trade in grid["trades"]] == ["initial_position"]
    assert any(row["reason"] == "suspended_day" for row in grid["rejections"])
    assert grid["equity"][-1]["timestamp"] == "2024-01-04T15:00:00"


def test_dca_and_buy_hold_start_with_their_own_cash_and_fees_are_accounted():
    bars = [_bar("2024-01-02 09:31", 10, 10)]
    result = run_backtest(
        _config("2024-01-02", initial_cash=10_000),
        _bundle(bars, ["2024-01-02"]),
    )
    strategies = result["strategies"]
    dca_buy = strategies["dca"]["trades"][0]
    hold_buy = strategies["buy_hold"]["trades"][0]
    assert dca_buy["reason"] == "dca"
    assert hold_buy["reason"] == "buy_hold"
    assert hold_buy["quantity"] >= dca_buy["quantity"]
    assert all(item["cash"] >= 0 for strategy in strategies.values() for item in strategy["trades"])
    assert strategies["buy_hold"]["metrics"]["total_fees"] > 0
    assert strategies["buy_hold"]["metrics"]["total_return"] < 0


def test_company_action_receivable_preserves_value_and_fixed_grid_adjusts_reference():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    bars = [
        _bar("2024-01-02 15:00", 10, 10, frequency="1d", preclose=10),
        _bar("2024-01-03 15:00", 9, 9, frequency="1d", preclose=9),
        _bar("2024-01-04 15:00", 9, 9, frequency="1d", preclose=9),
    ]
    daily = [
        _daily(dates[0], 10, 10, preclose=10),
        _daily(dates[1], 9, 9, preclose=9),
        _daily(dates[2], 9, 9, preclose=9),
    ]
    actions = [{
        "record_date": pd.Timestamp(dates[0]), "ex_date": pd.Timestamp(dates[1]),
        "pay_date": pd.Timestamp(dates[2]), "stock_date": pd.NaT,
        "cash_per_share": 1.0, "bonus_ratio": 0.0,
        "implemented": True, "unsupported_reason": "",
    }]
    config = _config(
        dates[0], dates[-1], frequency="1d", grid_mode="arithmetic",
        reference_price=10, grid_step=0.1, buy_step=0.05, sell_step=0.5,
    )
    result = run_backtest(config, _bundle(bars, dates, daily=daily, actions=actions))
    grid = result["strategies"]["grid"]
    # The ex-date 10 -> 9 adjustment was synchronized with the fixed grid, so it
    # did not fabricate a buy signal.  The cash dividend remained in equity before pay.
    assert [trade["reason"] for trade in grid["trades"]] == ["initial_position"]
    before_ex = grid["equity"][0]["equity"]
    on_ex = grid["equity"][1]["equity"]
    assert on_ex == pytest.approx(before_ex, rel=0.005)
    assert grid["equity"][2]["equity"] == pytest.approx(on_ex, rel=1e-6)


def test_summary_only_keeps_metrics_and_callbacks_can_cancel():
    days = ["2024-01-02", "2024-01-03"]
    bars = [
        _bar("2024-01-02 09:31", 10, 10),
        _bar("2024-01-03 09:31", 10, 10),
    ]
    bundle = _bundle(bars, days)
    summary = run_backtest(_config(days[0], days[-1]), bundle, summary_only=True)
    assert summary["strategies"]["buy_hold"]["metrics"]["trade_count"] == 1
    assert summary["strategies"]["buy_hold"]["equity"] == []
    calls = iter([False, True])
    with pytest.raises(InterruptedError):
        run_backtest(_config(days[0], days[-1]), bundle, cancelled=lambda: next(calls))


def test_open_timestamp_maps_daily_and_intraday_session_starts():
    assert open_timestamp(pd.Timestamp("2024-01-02 15:00"), "1d") == datetime(2024, 1, 2, 9, 30)
    assert open_timestamp(pd.Timestamp("2024-01-02 13:05"), "5m") == datetime(2024, 1, 2, 13, 0)
    assert open_timestamp(pd.Timestamp("2024-01-02 09:31"), "1m") == datetime(2024, 1, 2, 9, 30)


def test_numba_fastpath_matches_reference_ledger(monkeypatch):
    days = ["2024-01-02", "2024-01-03", "2024-02-01", "2024-03-01"]
    bars = [
        _bar("2024-01-02 09:31", 10.0, 10.0),
        _bar("2024-01-02 09:32", 10.0, 9.0),
        _bar("2024-01-02 09:33", 8.8, 9.0),
        _bar("2024-01-03 09:31", 9.2, 11.0),
        _bar("2024-01-03 09:32", 11.0, 11.0),
        _bar("2024-02-01 09:31", 10.5, 10.2),
        _bar("2024-02-01 09:32", 10.2, 10.2),
        _bar("2024-03-01 09:31", 10.2, 9.8),
        _bar("2024-03-01 09:32", 9.8, 9.8),
    ]
    bundle = _bundle(bars, days)
    config = _config(
        days[0], days[-1], initial_cash=20_000, initial_position=0.5,
        buy_step=0.05, sell_step=0.05,
        buy_quantity={"mode": "shares", "value": 200},
        sell_quantity={"mode": "shares", "value": 100},
    )
    assert engine._fast_market_inputs(config, bundle) is not None
    fast = run_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    reference = run_backtest(config, bundle)

    for strategy in ("grid", "dca", "buy_hold"):
        fast_result = fast["strategies"][strategy]
        reference_result = reference["strategies"][strategy]
        assert fast_result["trades"] == reference_result["trades"]
        assert fast_result["rejections"] == reference_result["rejections"]
        assert len(fast_result["period_returns"]) == len(reference_result["period_returns"])
        for actual, expected in zip(fast_result["period_returns"], reference_result["period_returns"]):
            assert actual["timestamp"] == expected["timestamp"]
            assert actual["return"] == pytest.approx(expected["return"], abs=1e-9)
        assert len(fast_result["equity"]) == len(reference_result["equity"])
        for actual, expected in zip(fast_result["equity"], reference_result["equity"]):
            assert actual["timestamp"] == expected["timestamp"]
            for field in ("equity", "cash", "position", "position_ratio", "drawdown"):
                assert actual[field] == pytest.approx(expected[field], abs=1e-8)
        for key, expected in reference_result["metrics"].items():
            if key in {"trade_count", "buy_count", "sell_count"}:
                assert fast_result["metrics"][key] == expected
            else:
                assert fast_result["metrics"][key] == pytest.approx(expected, abs=1e-9)


def test_fastpath_preserves_synthetic_skip_and_daily_fallback_fill(monkeypatch):
    days = ["2024-01-02", "2024-01-03"]
    bars = [
        _bar("2024-01-02 09:31", 10.0, 9.0),
        _bar("2024-01-02 09:32", 9.0, 8.5, is_synthetic=True, tradable=False, volume=0),
        _bar("2024-01-03 15:00", 8.8, 9.0, frequency="1d", preclose=9.0),
    ]
    daily = [
        _daily(days[0], 10.0, 9.0, preclose=10.0),
        _daily(days[1], 8.8, 9.0, preclose=9.0),
    ]
    bundle = _bundle(
        bars, days, daily=daily,
        quality={"days": [
            {"date": days[0], "mode": "minute", "missing_minutes": 1},
            {"date": days[1], "mode": "daily", "missing_minutes": 240},
        ]},
    )
    config = _config(days[0], days[-1])
    assert engine._fast_market_inputs(config, bundle) is not None
    fast = run_backtest(config, bundle)
    fast_summary = run_backtest(config, bundle, summary_only=True)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    reference = run_backtest(config, bundle)
    reference_summary = run_backtest(config, bundle, summary_only=True)
    for strategy in ("grid", "dca", "buy_hold"):
        actual = fast["strategies"][strategy]
        expected = reference["strategies"][strategy]
        assert actual["trades"] == expected["trades"]
        assert actual["rejections"] == expected["rejections"]
        assert len(actual["equity"]) == len(expected["equity"])
        for a, e in zip(actual["equity"], expected["equity"]):
            assert a["timestamp"] == e["timestamp"]
            for field in ("equity", "cash", "position", "position_ratio", "drawdown"):
                assert a[field] == pytest.approx(e[field], abs=1e-8)
        actual_summary = fast_summary["strategies"][strategy]
        expected_summary = reference_summary["strategies"][strategy]
        assert not actual_summary["equity"] and not actual_summary["trades"]
        for key, value in expected_summary["metrics"].items():
            if key in {"trade_count", "buy_count", "sell_count"}:
                assert actual_summary["metrics"][key] == value
            else:
                assert actual_summary["metrics"][key] == pytest.approx(value, abs=1e-9)


def test_fastpath_nontradable_real_bar_cancels_pending_without_signal_or_mark(monkeypatch):
    days = ["2024-01-02", "2024-01-03"]
    bars = [
        _bar("2024-01-02 09:31", 10.0, 10.0),
        _bar("2024-01-02 09:32", 10.0, 9.4),  # queues one next-open grid buy
        _bar(
            "2024-01-02 09:33", 9.3, 7.0, volume=0,
            tradable=False, is_synthetic=False,
        ),  # cancels that pending order; its close must not create another
        _bar("2024-01-02 09:34", 9.3, 10.0),
        _bar("2024-01-03 09:31", 10.0, 10.0),
    ]
    bundle = _bundle(bars, days)
    config = _config(
        days[0], days[-1], initial_cash=100_000, initial_position=0.5,
        buy_step=0.05,
        buy_quantity={"mode": "shares", "value": 100},
    )
    fast_inputs = engine._fast_market_inputs(config, bundle)
    assert fast_inputs is not None
    assert fast_inputs["tradable_mask"].tolist() == [True, True, False, True, True]
    fast = run_backtest(config, bundle)
    fast_summary = run_backtest(config, bundle, summary_only=True)

    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    reference = run_backtest(config, bundle)
    reference_summary = run_backtest(config, bundle, summary_only=True)
    for strategy in ("grid", "dca", "buy_hold"):
        actual = fast["strategies"][strategy]
        expected = reference["strategies"][strategy]
        assert actual["trades"] == expected["trades"]
        assert actual["rejections"] == expected["rejections"]
        assert [row["timestamp"] for row in actual["equity"]] == [
            row["timestamp"] for row in expected["equity"]
        ]
        assert "2024-01-02T09:33:00" not in {
            row["timestamp"] for row in actual["equity"]
        }
        for key, value in expected["metrics"].items():
            if key in {"trade_count", "buy_count", "sell_count"}:
                assert actual["metrics"][key] == value
            else:
                assert actual["metrics"][key] == pytest.approx(value, abs=1e-9)
        assert fast_summary["strategies"][strategy]["metrics"] == pytest.approx(
            reference_summary["strategies"][strategy]["metrics"], abs=1e-9,
        )
        assert fast_summary["strategies"][strategy]["equity"] == []
    grid = fast["strategies"]["grid"]
    assert [trade["reason"] for trade in grid["trades"]] == ["initial_position"]
    assert any(
        row["reason"] == "non_tradable_bar"
        and row["timestamp"] == "2024-01-02T09:32:00"
        for row in grid["rejections"]
    )


def test_no_price_limit_flag_overrides_supplied_and_calculated_limits(monkeypatch):
    day = "2024-01-02"
    bars = [_bar(
        f"{day} 09:31", 12.0, 12.0, preclose=None,
        limit_up=None, limit_down=None, no_price_limit=True,
    )]
    daily = [_daily(
        day, 12.0, 12.0, preclose=None,
        limit_up=10.0, limit_down=10.0, no_price_limit=True,
    )]
    bundle = _bundle(
        bars, [day], daily=daily,
        quality={"days": [{"date": day, "mode": "minute"}]},
    )
    config = _config(day)
    assert engine._fast_market_inputs(config, bundle) is not None
    fast = run_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    reference = run_backtest(config, bundle)
    for strategy in ("grid", "dca", "buy_hold"):
        assert fast["strategies"][strategy]["trades"] == reference["strategies"][strategy]["trades"]
    assert fast["strategies"]["grid"]["trades"][0]["price"] == 12.0
    assert not any("涨跌停" in warning for warning in reference["warnings"])
    assert engine._nb_sell_lot(100, 100, 2) == 100


def test_numba_fastpath_matches_cash_dividend_receivable_and_settlement(monkeypatch):
    days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    prices = [10.0, 9.5, 9.5, 9.5]
    precloses = [10.0, 9.5, 9.5, 9.5]
    bars = [
        _bar(f"{day} 09:31", price, price, preclose=preclose)
        for day, price, preclose in zip(days, prices, precloses)
    ]
    daily = [
        _daily(day, price, price, preclose=preclose)
        for day, price, preclose in zip(days, prices, precloses)
    ]
    actions = [{
        "record_date": pd.Timestamp(days[0]), "ex_date": pd.Timestamp(days[1]),
        "pay_date": pd.Timestamp(days[3]), "stock_date": pd.NaT,
        "cash_per_share": 0.5, "bonus_ratio": 0.0,
        "implemented": True, "unsupported_reason": "",
    }]
    bundle = _bundle(
        bars, days, daily=daily, actions=actions,
        quality={"days": [{"date": day, "mode": "minute"} for day in days]},
    )
    config = _config(
        days[0], days[-1], frequency="1m", initial_cash=20_000,
        grid_mode="arithmetic", reference_price=10.0,
        grid_step=0.1, buy_step=0.05, sell_step=0.5,
    )
    assert engine._fast_market_inputs(config, bundle) is not None
    fast = run_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    reference = run_backtest(config, bundle)
    for strategy in ("grid", "dca", "buy_hold"):
        actual = fast["strategies"][strategy]
        expected = reference["strategies"][strategy]
        assert actual["trades"] == expected["trades"]
        assert actual["rejections"] == expected["rejections"]
        assert len(actual["equity"]) == len(expected["equity"])
        for a, e in zip(actual["equity"], expected["equity"]):
            assert a["timestamp"] == e["timestamp"]
            for field in ("equity", "cash", "position", "position_ratio", "drawdown"):
                assert a[field] == pytest.approx(e[field], abs=1e-8)
        for key, expected_value in expected["metrics"].items():
            if key in {"trade_count", "buy_count", "sell_count"}:
                assert actual["metrics"][key] == expected_value
            else:
                assert actual["metrics"][key] == pytest.approx(expected_value, abs=1e-8)
    grid = fast["strategies"]["grid"]
    assert grid["equity"][1]["cash"] == grid["equity"][0]["cash"]
    assert grid["equity"][1]["equity"] == pytest.approx(grid["equity"][0]["equity"], abs=0.01)
    assert grid["equity"][2]["cash"] == grid["equity"][1]["cash"]
    assert grid["equity"][3]["cash"] > grid["equity"][2]["cash"]


def test_numba_historical_sh_face_value_transfer_fee_matches_reference(monkeypatch):
    days = ["2014-06-02", "2014-06-03"]
    bundle = _bundle(
        [_bar(f"{day} 09:31", 10.0, 10.0) for day in days], days,
    )
    config = _config(
        days[0], days[-1], initial_cash=100_000, initial_position=0.5,
        fees={"commission_rate": 0, "min_commission": 0, "slippage_bps": 0},
    )
    assert engine._fast_market_inputs(config, bundle) is not None
    fast = run_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    reference = run_backtest(config, bundle)

    for strategy in ("grid", "dca", "buy_hold"):
        actual = fast["strategies"][strategy]
        expected = reference["strategies"][strategy]
        assert actual["trades"] == expected["trades"]
        assert actual["metrics"] == pytest.approx(expected["metrics"])
    grid_initial = fast["strategies"]["grid"]["trades"][0]
    assert grid_initial["quantity"] == 5_000
    assert grid_initial["fees"] == 1.50
    assert not any("统一按 0.02‰" in warning for warning in fast["warnings"])


def test_position_ratio_counts_bonus_receivable_but_excludes_cash_receivable(monkeypatch):
    days = ["2024-01-02", "2024-01-03"]
    prices = [10.0, 6.0]
    bars = [
        _bar(f"{day} 09:31", price, price)
        for day, price in zip(days, prices)
    ]
    actions = [{
        "record_date": pd.Timestamp(days[0]), "ex_date": pd.Timestamp(days[1]),
        "pay_date": pd.Timestamp("2024-01-04"), "stock_date": pd.Timestamp("2024-01-05"),
        "cash_per_share": 1.0, "bonus_ratio": 0.5,
        "implemented": True, "unsupported_reason": "",
    }]
    bundle = _bundle(bars, days, actions=actions)
    config = _config(
        days[0], days[-1], initial_cash=100_000, initial_position=0.5,
    )
    assert engine._fast_market_inputs(config, bundle) is not None
    fast = run_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    reference = run_backtest(config, bundle)

    for strategy in ("grid", "dca", "buy_hold"):
        fast_equity = fast["strategies"][strategy]["equity"]
        reference_equity = reference["strategies"][strategy]["equity"]
        assert len(fast_equity) == len(reference_equity)
        assert [row["timestamp"] for row in fast_equity] == [
            row["timestamp"] for row in reference_equity
        ]
        for actual, expected in zip(fast_equity, reference_equity):
            assert actual["position"] == pytest.approx(expected["position"])
            assert actual["position_ratio"] == pytest.approx(
                expected["position_ratio"], abs=1e-12,
            )
            if actual["timestamp"] == "2024-01-03T09:31:00":
                assert actual["position_ratio"] == pytest.approx(
                    actual["position"] * prices[1] / actual["equity"], abs=1e-12,
                )

    ex_date_grid = next(
        row for row in fast["strategies"]["grid"]["equity"]
        if row["timestamp"] == "2024-01-03T09:31:00"
    )
    record_date_grid = fast["strategies"]["grid"]["equity"][0]
    assert ex_date_grid["position"] > record_date_grid["position"]
    cash_receivable = (
        ex_date_grid["equity"]
        - ex_date_grid["cash"]
        - ex_date_grid["position"] * prices[1]
    )
    assert cash_receivable > 0
    assert ex_date_grid["position_ratio"] < (
        ex_date_grid["position"] * prices[1] + cash_receivable
    ) / ex_date_grid["equity"]


def test_future_ohlc_perturbation_cannot_change_pre_break_trades_in_either_engine(monkeypatch):
    days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    base_bars = [
        _bar("2024-01-02 09:31", 10.0, 10.0),
        _bar("2024-01-02 09:32", 10.0, 9.0),
        _bar("2024-01-02 09:33", 8.8, 8.8),
        _bar("2024-01-03 09:31", 8.8, 10.0),
        _bar("2024-01-03 09:32", 10.0, 10.0),
        _bar("2024-01-04 09:31", 10.0, 9.0),
        _bar("2024-01-04 09:32", 9.0, 9.0),
        _bar("2024-01-05 09:31", 9.0, 11.0),
        _bar("2024-01-05 09:32", 11.0, 11.0),
    ]
    perturbed_bars = [dict(row) for row in base_bars]
    # Leave timestamps and every non-OHLC field intact; change only bars after
    # the first two sessions, with valid high/low envelopes for both paths.
    for index, (opening, close) in zip(range(5, len(perturbed_bars)), [
        (15.0, 16.0), (16.0, 16.0), (16.0, 5.0), (5.0, 5.0),
    ]):
        perturbed_bars[index].update(
            open=opening, high=max(opening, close) + 0.2,
            low=min(opening, close) - 0.2, close=close,
        )

    original = _bundle(base_bars, days)
    changed_future = _bundle(perturbed_bars, days)
    assert original.calendar.equals(changed_future.calendar)
    pd.testing.assert_frame_equal(
        original.bars.iloc[:5].reset_index(drop=True),
        changed_future.bars.iloc[:5].reset_index(drop=True),
    )
    for index in range(5, len(base_bars)):
        assert all(
            original.bars.iloc[index][column] == changed_future.bars.iloc[index][column]
            for column in original.bars.columns
            if column not in {"open", "high", "low", "close"}
        )

    config = _config(
        days[0], days[-1], initial_cash=100_000, initial_position=0.5,
        buy_step=0.05, sell_step=0.05,
        buy_quantity={"mode": "shares", "value": 200},
        sell_quantity={"mode": "shares", "value": 100},
    )
    assert engine._fast_market_inputs(config, original) is not None
    assert engine._fast_market_inputs(config, changed_future) is not None
    original_fast = run_backtest(config, original)
    changed_fast = run_backtest(config, changed_future)

    # The changed path begins at the 09:30 opening event for Jan 4.  All fills
    # executed before that event must be identical in the two histories.
    break_open = open_timestamp(pd.Timestamp("2024-01-04 09:31"), "1m").isoformat()
    for strategy in ("grid", "dca", "buy_hold"):
        original_before = [
            trade for trade in original_fast["strategies"][strategy]["trades"]
            if trade["timestamp"] < break_open
        ]
        changed_before = [
            trade for trade in changed_fast["strategies"][strategy]["trades"]
            if trade["timestamp"] < break_open
        ]
        assert original_before == changed_before

    # The future mutation is material, so the test also proves that the runs
    # did not accidentally compare the same market path.
    assert (
        original_fast["strategies"]["grid"]["trades"]
        != changed_fast["strategies"]["grid"]["trades"]
    )

    for fast_result in (original_fast, changed_fast):
        monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
        reference = run_backtest(config, original if fast_result is original_fast else changed_future)
        monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", True)
        for strategy in ("grid", "dca", "buy_hold"):
            fast_ledger = fast_result["strategies"][strategy]
            reference_ledger = reference["strategies"][strategy]
            assert fast_ledger["trades"] == reference_ledger["trades"]
            assert fast_ledger["rejections"] == reference_ledger["rejections"]
            for key, expected in reference_ledger["metrics"].items():
                if key in {"trade_count", "buy_count", "sell_count"}:
                    assert fast_ledger["metrics"][key] == expected
                else:
                    assert fast_ledger["metrics"][key] == pytest.approx(expected, abs=1e-9)

    # The reference engine must obey the same no-look-ahead property.
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    original_reference = run_backtest(config, original)
    changed_reference = run_backtest(config, changed_future)
    for strategy in ("grid", "dca", "buy_hold"):
        original_before = [
            trade for trade in original_reference["strategies"][strategy]["trades"]
            if trade["timestamp"] < break_open
        ]
        changed_before = [
            trade for trade in changed_reference["strategies"][strategy]["trades"]
            if trade["timestamp"] < break_open
        ]
        assert original_before == changed_before

