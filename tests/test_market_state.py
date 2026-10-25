from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pandas as pd
import pytest

from stockregression.market_state import (
    MAX_TRADING_DAYS,
    daily_market_state,
    state_step_candidates,
)
from stockregression.models import DataBundle


def _bundle(
    days: pd.DatetimeIndex,
    closes: np.ndarray,
    *,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    prior_close: float | None = None,
    actions: pd.DataFrame | None = None,
) -> DataBundle:
    closes = np.asarray(closes, dtype=float)
    high = closes * 1.01 if highs is None else np.asarray(highs, dtype=float)
    low = closes * .99 if lows is None else np.asarray(lows, dtype=float)
    preclose = np.r_[prior_close if prior_close is not None else closes[0], closes[:-1]]
    daily = pd.DataFrame({
        "date": days,
        "open": closes,
        "high": high,
        "low": low,
        "close": closes,
        "preclose": preclose,
        "volume": 1_000_000.0,
    })
    return DataBundle(
        symbol="510050.SH", bars=pd.DataFrame(), daily=daily,
        actions=actions if actions is not None else pd.DataFrame(),
        calendar=days, version="market-state-fixture",
    )


def _trend_bundle(periods: int = 260) -> DataBundle:
    days = pd.bdate_range("2024-01-02", periods=periods)
    closes = np.arange(periods, dtype=float) + 10.0
    highs = closes + 1.0
    lows = closes - 1.0
    return _bundle(days, closes, highs=highs, lows=lows, prior_close=9.0)


def _expected_natr(highs, lows, closes, prior_close, period=14):
    ranges = []
    previous = prior_close
    for high, low, close in zip(highs, lows, closes):
        ranges.append(max(high - low, abs(high - previous), abs(low - previous)))
        previous = close
    atr = sum(ranges[:period]) / period
    for value in ranges[period:]:
        atr = (atr * (period - 1) + value) / period
    return atr / closes[-1]


def test_wilder_adx_and_natr_match_fixed_talib_064_golden():
    # Independent golden generated with Python ta-lib 0.6.8 / C TA-Lib 0.6.4.
    t = np.arange(80, dtype=float)
    close = 100.0 + .07 * t + 3.0 * np.sin(.37 * t) + .8 * np.cos(.11 * t)
    high = close + .8 + .4 * np.sin(.23 * t) ** 2
    low = close - .7 - .3 * np.cos(.19 * t) ** 2
    days = pd.bdate_range("2025-01-02", periods=len(t))
    data = _bundle(days, close, highs=high, lows=low, prior_close=None)
    data.daily.loc[0, "open"] = close[0]
    data.daily.loc[0, "preclose"] = np.nan

    golden = {
        27: (.018706892703918225, 19.1736882367426, 17.13092095195736, 23.486491023206497),
        40: (.01815601896479866, 23.825251249165778, 23.862591806926858, 11.846137264731786),
        79: (.01826857530406928, 19.80058685898613, 15.097080220312723, 25.439392670232664),
    }
    for index, expected in golden.items():
        state = daily_market_state(data, days[index])
        assert state["samples"]["daily_rows"] == index
        assert state["samples"]["selected_start"] == days[1].date().isoformat()
        assert state["samples"]["prior_close_date"] == days[0].date().isoformat()
        assert state["samples"]["prior_close_source"] == "first_selected_bar"
        actual = (
            state["metrics"]["natr14"], state["metrics"]["adx14"],
            state["metrics"]["plus_di14"], state["metrics"]["minus_di14"],
        )
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_daily_state_numeric_contract_uses_250_sessions_and_decimal_natr():
    data = _trend_bundle()
    as_of = data.calendar[-1]
    state = daily_market_state(data, as_of)

    assert state["method_version"] == "daily_state_v1"
    assert state["as_of"] == as_of.date().isoformat()
    assert state["status"] == "completed", state["metric_details"]
    assert set(state["metrics"]) == {
        "natr14", "hv_ratio", "adx14", "plus_di14", "minus_di14", "amplitude_q75",
    }
    assert state["samples"]["daily_rows"] == MAX_TRADING_DAYS
    assert state["samples"]["truncated_at_limit"] is True
    assert state["samples"]["prior_close_date"] == data.calendar[-251].date().isoformat()
    assert state["samples"]["close_samples_hv"] == MAX_TRADING_DAYS + 1

    selected = data.daily.iloc[-MAX_TRADING_DAYS:]
    prior = float(data.daily.iloc[-251].close)
    expected_natr = _expected_natr(
        selected.high.to_numpy(), selected.low.to_numpy(),
        selected.close.to_numpy(), prior,
    )
    close_series = np.r_[prior, selected.close.to_numpy()]
    returns = np.diff(np.log(close_series))
    expected_hv = np.std(returns[-10:], ddof=1) / np.std(returns[-60:], ddof=1)
    amplitudes = (
        (selected.high.to_numpy()[-20:] - selected.low.to_numpy()[-20:])
        / close_series[-21:-1]
    )
    assert state["metrics"]["natr14"] == pytest.approx(expected_natr, rel=1e-12)
    assert state["metrics"]["natr14"] < .1
    assert state["metrics"]["hv_ratio"] == pytest.approx(expected_hv, rel=1e-12)
    assert state["metrics"]["adx14"] == pytest.approx(100.0)
    assert state["metrics"]["plus_di14"] > state["metrics"]["minus_di14"]
    assert state["metrics"]["amplitude_q75"] == pytest.approx(
        np.quantile(amplitudes, .75, method="linear"), rel=1e-12,
    )
    assert all(
        detail["status"] == "available"
        for detail in state["metric_details"].values()
    )
    assert state["input_fingerprint"]


def test_daily_state_ignores_future_daily_rows_and_actions():
    data = _trend_bundle()
    as_of = data.calendar[-15]
    baseline = daily_market_state(data, as_of)

    future = deepcopy(data)
    extra = future.daily.iloc[-1:].copy()
    extra["date"] = pd.Timestamp(as_of) + pd.offsets.BDay(1)
    extra[["open", "high", "low", "close"]] *= 1000.0
    future.daily = pd.concat([future.daily, extra], ignore_index=True)
    future.actions = pd.DataFrame([{
        "ex_date": as_of + pd.offsets.BDay(1),
        "cash_per_share": 0.0, "bonus_ratio": 0.0,
        "implemented": True, "unsupported_reason": "future unsupported action",
    }])

    repeated = daily_market_state(future, as_of)
    assert repeated["metrics"] == baseline["metrics"]
    assert repeated["samples"] == baseline["samples"]
    assert repeated["input_fingerprint"] == baseline["input_fingerprint"]
    assert repeated["samples"]["selected_end"] == as_of.date().isoformat()


@pytest.mark.parametrize("kind", ["cash", "split"])
def test_daily_state_adjusts_dividend_and_split_without_double_adjusting(kind):
    days = pd.bdate_range("2024-01-02", periods=260)
    latent_close = 10.0 + np.arange(len(days)) * .01
    latent_high = latent_close * 1.008
    latent_low = latent_close * .992
    ex_index = 240
    factor = .95 if kind == "cash" else .5
    raw_close = latent_close.copy()
    raw_high = latent_high.copy()
    raw_low = latent_low.copy()
    raw_close[:ex_index] /= factor
    raw_high[:ex_index] /= factor
    raw_low[:ex_index] /= factor
    prior_raw = raw_close[ex_index - 1]
    prior_latent = latent_close[ex_index - 1]
    raw_preclose = raw_close.copy()
    raw_preclose[ex_index] = prior_latent
    if kind == "cash":
        action_values = {"cash_per_share": prior_raw - prior_latent, "bonus_ratio": 0.0}
    else:
        action_values = {"cash_per_share": 0.0, "bonus_ratio": 1.0}
    actions = pd.DataFrame([{
        "ex_date": days[ex_index], **action_values,
        "implemented": True, "unsupported_reason": "",
    }])
    adjusted_bundle = _bundle(
        days, raw_close, highs=raw_high, lows=raw_low,
        prior_close=raw_close[0], actions=actions,
    )
    adjusted_bundle.daily["preclose"] = raw_preclose
    clean = _bundle(
        days, latent_close, highs=latent_high, lows=latent_low,
        prior_close=raw_preclose[0],
    )

    adjusted = daily_market_state(adjusted_bundle, days[-1])
    reference = daily_market_state(clean, days[-1])
    assert adjusted["price_basis"]["action_count"] == 1
    assert adjusted["price_basis"]["actions_applied"][0]["factor"] == pytest.approx(factor)
    for key in ("natr14", "hv_ratio", "adx14", "plus_di14", "minus_di14", "amplitude_q75"):
        assert adjusted["metrics"][key] == pytest.approx(reference["metrics"][key], rel=1e-10, abs=1e-12)


def test_missing_session_breaks_history_and_is_never_bridged():
    data = _trend_bundle()
    missing = data.calendar[100]
    data.daily = data.daily[data.daily.date.ne(missing)].reset_index(drop=True)
    baseline = daily_market_state(data, data.calendar[-1])
    assert baseline["samples"]["selected_start"] == data.calendar[102].date().isoformat()
    assert baseline["samples"]["gap_before_start"] == missing.date().isoformat()
    assert baseline["samples"]["prior_close_date"] == data.calendar[101].date().isoformat()
    assert baseline["samples"]["prior_close_source"] == "first_selected_bar"
    assert baseline["status"] == "completed"
    assert baseline["metrics"]["natr14"] is not None
    assert baseline["metrics"]["hv_ratio"] is not None
    assert baseline["metrics"]["amplitude_q75"] is not None

    changed = deepcopy(data)
    changed.daily.loc[changed.daily.date.lt(missing), ["high", "low", "close"]] *= 50.0
    repeated = daily_market_state(changed, changed.calendar[-1])
    assert repeated["metrics"] == baseline["metrics"]
    assert repeated["input_fingerprint"] == baseline["input_fingerprint"]


def test_missing_latest_trading_session_returns_unavailable_without_fallback():
    data = _trend_bundle()
    latest = data.calendar[-1]
    data.daily = data.daily[data.daily.date.lt(latest)].reset_index(drop=True)
    state = daily_market_state(data, latest)
    assert state["status"] == "unavailable"
    assert all(value is None for value in state["metrics"].values())
    assert state["samples"]["selected_end"] is None
    assert state["samples"]["missing_latest_date"] == latest.date().isoformat()
    assert latest.date().isoformat() in state["message"]
    assert all(
        detail["reason_code"] == "missing_latest_daily"
        for detail in state["metric_details"].values()
    )

    data.daily = data.daily.iloc[0:0].copy()
    empty = daily_market_state(data, latest)
    assert empty["status"] == "unavailable"
    assert empty["samples"]["missing_latest_date"] == latest.date().isoformat()


def test_weekend_as_of_uses_last_calendar_trading_session():
    data = _trend_bundle()
    latest = data.calendar[-1]
    weekend = latest + pd.offsets.Week(weekday=5)
    assert weekend.weekday() == 5
    state = daily_market_state(data, weekend)
    assert state["status"] == "completed"
    assert state["samples"]["selected_end"] == latest.date().isoformat()
    assert state["samples"]["missing_latest_date"] is None


@pytest.mark.parametrize("actions", [
    pd.DataFrame([{"cash_per_share": 0.1, "bonus_ratio": 0.0}]),
    pd.DataFrame([{
        "ex_date": "not-a-date", "cash_per_share": 0.1,
        "bonus_ratio": 0.0,
    }]),
])
def test_nonempty_actions_without_valid_ex_date_are_not_ignored(actions):
    data = _trend_bundle()
    data.actions = actions
    state = daily_market_state(data, data.calendar[-1])
    assert state["status"] == "data_not_ready"
    assert all(value is None for value in state["metrics"].values())
    assert "ex_date" in state["price_basis"]["error"]
    assert all(
        detail["reason_code"] == "invalid_actions"
        for detail in state["metric_details"].values()
    )


def test_invalid_latest_open_is_not_accepted_but_absent_open_is_allowed():
    invalid = _trend_bundle()
    invalid.daily.loc[invalid.daily.index[-1], "open"] = np.nan
    rejected = daily_market_state(invalid, invalid.calendar[-1])
    assert rejected["status"] == "unavailable"
    assert rejected["samples"]["missing_latest_date"] == invalid.calendar[-1].date().isoformat()
    assert all(value is None for value in rejected["metrics"].values())

    without_open = _trend_bundle()
    without_open.daily = without_open.daily.drop(columns="open")
    accepted = daily_market_state(without_open, without_open.calendar[-1])
    assert accepted["status"] == "completed"
    assert accepted["metrics"]["natr14"] is not None


def test_invalid_daily_history_nulls_every_metric_with_reasons():
    data = _trend_bundle()
    data.daily["high"] = np.nan
    state = daily_market_state(data, data.calendar[-1])
    assert state["status"] == "unavailable"
    assert all(value is None for value in state["metrics"].values())
    assert all(
        detail["reason_code"] == "missing_latest_daily"
        for detail in state["metric_details"].values()
    )


def test_flat_prices_keep_zero_natr_and_amplitude_but_adx_has_no_direction():
    days = pd.bdate_range("2025-01-02", periods=100)
    close = np.full(len(days), 10.0)
    data = _bundle(
        days, close, highs=close.copy(), lows=close.copy(), prior_close=10.0,
    )
    state = daily_market_state(data, days[-1])
    assert state["status"] == "completed"
    assert state["metrics"]["natr14"] == 0.0
    assert state["metrics"]["amplitude_q75"] == 0.0
    assert state["metrics"]["hv_ratio"] is None
    assert state["metric_details"]["hv_ratio"]["reason_code"] == "zero_denominator"
    assert state["metrics"]["adx14"] is None
    assert state["metrics"]["plus_di14"] is None
    assert state["metrics"]["minus_di14"] is None


def test_short_history_reports_metric_specific_insufficient_samples():
    days = pd.bdate_range("2025-01-02", periods=5)
    data = _bundle(days, np.linspace(10.0, 11.0, len(days)), prior_close=9.9)
    state = daily_market_state(data, days[-1])
    assert state["status"] == "insufficient_history"
    assert all(value is None for value in state["metrics"].values())
    assert all(
        detail["reason_code"] == "insufficient_samples"
        for detail in state["metric_details"].values()
    )


def test_state_step_candidates_build_two_bases_and_adx_asymmetry():
    state = {
        "metrics": {
            "natr14": .01, "hv_ratio": 1.5, "adx14": 30.0,
            "plus_di14": 28.0, "minus_di14": 9.0, "amplitude_q75": .02,
        },
        "metric_details": {"adx14": {"direction": "up"}},
    }
    candidates = state_step_candidates(state)
    base = .01 * (.65 + .5 * (1.5 - 1.0))
    alpha = 1.0 + min(1.0, (30.0 - 22.0) / 28.0 * .85)
    assert [(item["buy_step"], item["sell_step"]) for item in candidates] == pytest.approx([
        (base, base), (.85 * base, alpha * base),
        (.02, .02), (.85 * .02, alpha * .02),
    ])
    assert [item["source"] for item in candidates] == [
        "natr_hv_symmetric", "natr_hv_up_asymmetric",
        "amplitude_q75_symmetric", "amplitude_q75_up_asymmetric",
    ]

    down = deepcopy(state)
    down["metrics"].update({"plus_di14": 5.0, "minus_di14": 30.0})
    down["metric_details"]["adx14"]["direction"] = "down"
    asymmetric = [item for item in state_step_candidates(down) if "asymmetric" in item["source"]]
    assert [item["buy_step"] for item in asymmetric] == pytest.approx([alpha * base, alpha * .02])
    assert [item["sell_step"] for item in asymmetric] == pytest.approx([.85 * base, .85 * .02])

    weak = deepcopy(down)
    weak["metrics"]["adx14"] = 25.0
    assert len(state_step_candidates(weak)) == 2
    assert all(item["buy_step"] == item["sell_step"] for item in state_step_candidates(weak))
    assert not any(math.isclose(item["buy_step"], 0.0) for item in candidates)
