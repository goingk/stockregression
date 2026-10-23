from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pandas as pd
import pytest

import stockregression.etf_statistics as statistics
from stockregression.etf_statistics import (
    compute_etf_statistics,
    generate_candidates,
    score_etfs,
)
from stockregression.models import DataBundle
from stockregression.rules import round_to_tick


def _synthetic_bundle(
    *, window_days: int = 120, mddd_verified: bool = True,
) -> DataBundle:
    """Deterministic minute fixture with an independent 250-day daily tail."""
    rng = np.random.default_rng(20261023)
    end = pd.Timestamp("2026-06-30")
    calendar = pd.bdate_range(end=end, periods=window_days)
    all_daily_days = pd.bdate_range(end=end, periods=250)
    rows: list[pd.DataFrame] = []
    daily_closes: dict[pd.Timestamp, float] = {}
    mu = math.log(1.1)
    phi = 0.72
    state = mu

    for day in calendar:
        morning = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min",
        )
        afternoon = pd.date_range(
            day + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min",
        )
        timestamps = morning.append(afternoon)
        prices: list[float] = []
        for _session in range(2):
            for _bucket in range(4):
                start = state
                target = mu + phi * (state - mu) + rng.normal(0.0, 0.0028)
                for offset in range(30):
                    fraction = (offset + 1) / 30.0
                    shape = 0.00022 * math.sin(math.pi * fraction * 2.0)
                    price = math.exp(
                        start * (1.0 - fraction) + target * fraction + shape
                    )
                    prices.append(price)
                # Preserve exact bucket endpoints while retaining minute-scale
                # movement inside the bucket.
                prices[-1] = math.exp(target)
                state = target
        close = np.asarray(prices, dtype=float)
        open_price = np.r_[close[0], close[:-1]]
        volume = 100_000.0 + np.arange(240) % 37 * 1_000.0
        bars = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.00008,
            "low": np.minimum(open_price, close) * 0.99992,
            "close": close,
            "volume": volume,
            "turnover": volume * close,
            "frequency": "1m",
            "source": "fixture",
            "source_verified": True,
            "tradable": True,
            "is_synthetic": False,
            "is_auction_placeholder": False,
        })
        rows.append(bars)
        daily_closes[day] = float(close[-1])

    minute = pd.concat(rows, ignore_index=True)
    prior_days = all_daily_days[~all_daily_days.isin(calendar)]
    prior_close = np.linspace(1.02, 1.1, len(prior_days))
    prior_close += np.sin(np.arange(len(prior_days)) * 0.17) * 0.008
    daily_close = np.asarray(
        list(prior_close) + [daily_closes[day] for day in calendar], dtype=float,
    )
    preclose = np.r_[daily_close[0], daily_close[:-1]]
    open_daily = daily_close * (1.0 + 0.00035 * np.sin(np.arange(250) * 0.31))
    high = np.maximum(open_daily, daily_close) * 1.002
    low = np.minimum(open_daily, daily_close) * 0.998
    daily = pd.DataFrame({
        "date": all_daily_days,
        "open": open_daily,
        "high": high,
        "low": low,
        "close": daily_close,
        "preclose": preclose,
        "volume": 24_000_000.0,
        "turnover": 24_000_000.0 * daily_close,
        "tradestatus": 1,
        "is_st": False,
    })
    actions = pd.DataFrame([
        {
            "ex_date": all_daily_days[0] - pd.Timedelta(days=10),
            "cash_per_share": 0.0,
            "bonus_ratio": 0.0,
            "implemented": True,
            "unsupported_reason": "old action outside both statistical windows",
        },
        {
            "ex_date": end + pd.Timedelta(days=10),
            "cash_per_share": 0.0,
            "bonus_ratio": 0.0,
            "implemented": True,
            "unsupported_reason": "future action must be ignored",
        },
    ])
    metadata = {
        "price_tick": 0.001,
        "mddd_history_verified": mddd_verified,
        "prior_closes": {
            str(row.date.date()): float(row.preclose) for row in daily.itertuples()
        },
    }
    return DataBundle(
        symbol="159611.SZ",
        bars=minute,
        daily=daily,
        actions=actions,
        calendar=calendar,
        quality={"is_demo": False, "days": []},
        metadata=metadata,
        version="synthetic-etf-statistics-v1",
    )


def _assert_metric_contract(result: dict) -> None:
    assert {
        "rv_annualized", "path_redundancy", "iovr", "variance_ratio_60m",
        "variance_ratio_240m", "variance_ratio_1200m", "ou_half_life_days",
        "zero_crossing_rate", "jump_ratio", "max_drawdown_duration_days",
        "tick_to_price_ratio", "rv_5m_annualized", "mean_raw_price",
    }.issubset(result["metrics"])
    for key, value in result["metrics"].items():
        assert value is None or math.isfinite(value)
        detail = result["metric_details"][key]
        assert {"status", "reason", "sample_count"}.issubset(detail)
        assert isinstance(detail["sample_count"], int) and detail["sample_count"] >= 0
    assert {
        "reference_price", "lower_price", "upper_price", "current_raw_price",
        "atr30_pct", "svr", "realized_skewness", "realized_kurtosis", "price_tick",
    } == set(result["grid_basis"])


def test_compute_contract_uses_two_scale_minute_statistics_without_lookahead():
    data = _synthetic_bundle()
    as_of = data.calendar[-1]
    result = compute_etf_statistics(data, as_of, 120)

    assert result["status"] == "completed"
    _assert_metric_contract(result)
    assert result["diagnostics"]["window_start"] == str(data.calendar[0].date())
    assert result["diagnostics"]["window_end"] == str(as_of.date())
    assert result["diagnostics"]["sampling_frequencies"]["ou"] == "30m"
    assert result["diagnostics"]["sampling_frequencies"]["variance_ratio"] == "5m"
    assert result["diagnostics"]["source_quality"]["volume_coverage"] == pytest.approx(1.0)
    assert result["diagnostics"]["source_quality"]["turnover_source"] == "unknown"
    assert result["diagnostics"]["source_quality"]["vwap_quality"] == "unknown"
    assert result["diagnostics"]["profile"]["price_quality"] == "ohlc4_approximation"
    assert result["metrics"]["rv_annualized"] > 0
    assert result["metrics"]["rv_5m_annualized"] > 0
    assert result["metrics"]["variance_ratio_60m"] is not None
    assert result["metrics"]["variance_ratio_1200m"] is not None
    assert 0.0 <= result["metrics"]["zero_crossing_rate"] <= 1.0
    assert result["metrics"]["path_redundancy"] == pytest.approx(
        1.0 - abs(
            data.bars.close.iloc[-1] - data.bars.close.iloc[0]
        ) / data.bars.close.diff().abs().sum(),
        rel=1e-9,
    )
    basis = result["grid_basis"]
    assert basis["lower_price"] < basis["reference_price"] < basis["upper_price"]
    assert result["actions_applied"] == []
    assert all(reason != "old action" for reason in result["warnings"])
    assert result["eligible"] == (not result["filter_reasons"])

    future = deepcopy(data)
    next_day = as_of + pd.offsets.BDay(1)
    extra = data.bars.iloc[-240:].copy()
    extra["timestamp"] = extra.timestamp + (next_day - as_of)
    extra[["open", "high", "low", "close"]] *= 8.0
    future.bars = pd.concat([future.bars, extra], ignore_index=True)
    future.daily = pd.concat([
        future.daily,
        pd.DataFrame([{
            **future.daily.iloc[-1].to_dict(),
            "date": next_day,
            "open": 8.0,
            "high": 9.0,
            "low": 7.0,
            "close": 8.0,
            "preclose": 8.0,
        }]),
    ], ignore_index=True)
    future.actions = pd.concat([
        future.actions,
        pd.DataFrame([{
            "ex_date": next_day,
            "cash_per_share": 0.0,
            "bonus_ratio": 0.0,
            "implemented": True,
            "unsupported_reason": "future unsupported action",
        }]),
    ], ignore_index=True)
    repeated = compute_etf_statistics(future, as_of, 120)
    assert repeated["metrics"] == result["metrics"]
    assert repeated["grid_basis"] == result["grid_basis"]
    assert repeated["diagnostics"]["window_end"] == str(as_of.date())


def test_real_fill_is_excluded_while_auction_and_zero_volume_points_remain():
    clean = _synthetic_bundle()
    replacement_time = clean.calendar[-1] + pd.Timedelta(hours=9, minutes=45)
    replacement_index = clean.bars.index[clean.bars.timestamp.eq(replacement_time)][0]
    zero_time = clean.calendar[-1] + pd.Timedelta(hours=10, minutes=22)
    zero_index = clean.bars.index[clean.bars.timestamp.eq(zero_time)][0]
    auction_time = clean.calendar[-1] + pd.Timedelta(hours=15)
    auction_index = clean.bars.index[clean.bars.timestamp.eq(auction_time)][0]

    dirty = deepcopy(clean)
    dirty.bars.loc[replacement_index, ["open", "high", "low", "close"]] *= 50.0
    dirty.bars.loc[replacement_index, "is_synthetic"] = True
    dirty.bars.loc[zero_index, "volume"] = 0.0
    dirty.bars.loc[zero_index, "is_synthetic"] = False
    dirty.bars.loc[zero_index, "is_auction_placeholder"] = False
    dirty.bars.loc[auction_index, "volume"] = 0.0
    dirty.bars.loc[auction_index, "is_synthetic"] = True
    dirty.bars.loc[auction_index, "is_auction_placeholder"] = True

    removed = deepcopy(dirty)
    removed.bars = removed.bars.drop(index=replacement_index).reset_index(drop=True)
    dirty_result = compute_etf_statistics(dirty, dirty.calendar[-1], 120)
    removed_result = compute_etf_statistics(removed, removed.calendar[-1], 120)

    assert dirty_result["status"] == "data_not_ready"
    assert dirty_result["eligible"] is False
    assert removed_result["status"] == "data_not_ready"
    assert str(dirty.calendar[-1].date()) in dirty_result["diagnostics"]["minute_missing_days"]
    for key, value in dirty_result["metrics"].items():
        other = removed_result["metrics"][key]
        if value is None:
            assert other is None
        else:
            assert other == pytest.approx(value, rel=1e-12, abs=1e-15)
    quality = dirty_result["diagnostics"]["source_quality"]
    assert quality["synthetic_missing_minutes"] == 1
    assert quality["auction_placeholder_minutes"] == 1
    assert quality["zero_volume_retained_minutes"] == 1


def test_explicitly_unverified_minute_blocks_completeness_and_eligibility():
    data = _synthetic_bundle()
    target = data.calendar[-2] + pd.Timedelta(hours=10, minutes=13)
    index = data.bars.index[data.bars.timestamp.eq(target)][0]
    data.bars.loc[index, "source_verified"] = False

    result = compute_etf_statistics(data, data.calendar[-1], 120)
    assert result["status"] == "data_not_ready"
    assert result["eligible"] is False
    integrity = result["diagnostics"]["minute_integrity"]
    assert integrity["unverified_slots"] == 1
    assert str(data.calendar[-2].date()) in result["diagnostics"]["minute_missing_days"]
    rows = [result, {**_score_row("CONTROL"), "symbol": "CONTROL"}]
    score_etfs(rows)
    assert result["gss_score"] is None
    assert result["score_cohort_size"] == 1


def test_legal_auction_placeholder_can_fill_an_uncalibrated_slot_without_failing_day():
    data = _synthetic_bundle()
    auction = data.calendar[-1] + pd.Timedelta(hours=15)
    index = data.bars.index[data.bars.timestamp.eq(auction)][0]
    data.bars.loc[index, ["open", "high", "low"]] = np.nan
    data.bars.loc[index, "is_synthetic"] = True
    data.bars.loc[index, "is_auction_placeholder"] = True
    data.bars.loc[index, "source_verified"] = False
    data.bars.loc[index, "volume"] = 0.0

    result = compute_etf_statistics(data, data.calendar[-1], 120)
    assert result["status"] == "completed"
    integrity = result["diagnostics"]["minute_integrity"]
    assert integrity["expected_slots"] == integrity["accepted_slots"] == 120 * 240
    assert integrity["placeholder_slots"] == 1


def test_missing_calibration_column_and_nonzero_volume_placeholder_are_not_accepted():
    data = _synthetic_bundle()
    context = statistics._prepare(data, data.calendar[-1], 120)
    missing_calibration = context.points_1m.drop(columns=["source_verified"])
    integrity = statistics._minute_integrity(missing_calibration, context.days, set())
    assert not integrity["complete_days"]
    assert integrity["unverified_slots"] == len(context.days) * 240

    suspended_day = context.days[-3]
    without_suspended = context.points_1m[
        context.points_1m.day.ne(suspended_day)
    ].copy()
    ignored = statistics._minute_integrity(
        without_suspended, context.days, {suspended_day},
    )
    assert not ignored["incomplete_days"]
    assert len(ignored["complete_days"]) == len(context.days) - 1

    fake_placeholder = context.points_1m.copy()
    fake_placeholder.loc[100, "is_auction_placeholder"] = True
    fake_placeholder.loc[100, "volume"] = 1.0
    fake_placeholder.loc[100, "source_verified"] = False
    rejected = statistics._minute_integrity(fake_placeholder, context.days, set())
    assert rejected["placeholder_slots"] == 0
    assert rejected["unverified_slots"] == 1


def test_mddd_requires_verification_is_censored_and_does_not_null_other_metrics():
    verified = _synthetic_bundle(mddd_verified=True)
    daily = verified.daily.copy()
    close = np.r_[np.linspace(1.0, 2.0, 150), np.linspace(1.99, 0.8, 100)]
    daily["close"] = close
    daily["open"] = close
    daily["high"] = close * 1.002
    daily["low"] = close * 0.998
    daily["preclose"] = np.r_[close[0], close[:-1]]
    verified.daily = daily
    unverified = deepcopy(verified)
    unverified.metadata["mddd_history_verified"] = False

    checked = compute_etf_statistics(verified, verified.calendar[-1], 120)
    blocked = compute_etf_statistics(unverified, unverified.calendar[-1], 120)
    assert checked["metrics"]["max_drawdown_duration_days"] >= 100
    assert checked["diagnostics"]["mddd_censored"] is True
    assert checked["diagnostics"]["mddd_recovery_date"] is None
    assert blocked["status"] == "completed"
    assert blocked["metrics"]["max_drawdown_duration_days"] is None
    assert "校验账本" in blocked["metric_details"]["max_drawdown_duration_days"]["reason"]
    for key in (
        "rv_annualized", "path_redundancy", "iovr", "variance_ratio_240m",
        "zero_crossing_rate", "jump_ratio",
    ):
        assert blocked["metrics"][key] == checked["metrics"][key]


def test_unverified_action_inside_statistical_window_blocks_adjusted_statistics():
    data = _synthetic_bundle()
    action_day = data.calendar[60]
    extra = pd.DataFrame([{
        "ex_date": action_day,
        "cash_per_share": 0.0,
        "bonus_ratio": 0.0,
        "implemented": True,
        "unsupported_reason": "unsupported action inside window",
    }])
    data.actions = pd.concat([data.actions, extra], ignore_index=True)
    result = compute_etf_statistics(data, data.calendar[-1], 120)
    assert result["status"] == "data_not_ready"
    assert any("unsupported action inside window" in warning for warning in result["warnings"])


@pytest.mark.parametrize("cash,bonus", [(0.10, 0.0), (0.0, 0.10)])
def test_action_on_first_statistical_day_uses_retained_prior_close(cash, bonus):
    data = _synthetic_bundle(mddd_verified=False)
    ex_date = data.calendar[0]
    daily_index = data.daily.index[data.daily.date.eq(ex_date)][0]
    prior_index = daily_index - 1
    prior_close = float(data.daily.loc[prior_index, "close"])
    implied_reference = (prior_close - cash) / (1.0 + bonus)
    data.daily.loc[daily_index, "preclose"] = implied_reference
    data.actions = pd.concat([
        data.actions,
        pd.DataFrame([{
            "ex_date": ex_date,
            "cash_per_share": cash,
            "bonus_ratio": bonus,
            "implemented": True,
            "unsupported_reason": "",
        }]),
    ], ignore_index=True)

    result = compute_etf_statistics(data, data.calendar[-1], 120)
    assert result["status"] == "completed"
    assert len(result["actions_applied"]) == 1
    action = result["actions_applied"][0]
    assert action["ex_date"] == ex_date.date().isoformat()
    assert action["bonus_ratio"] == pytest.approx(bonus)
    assert action["factor"] == pytest.approx(implied_reference / prior_close)


@pytest.mark.parametrize("cash,bonus", [(0.10, 0.0), (0.0, 1.0)])
def test_iovr_uses_action_reference_when_exday_preclose_is_stale(cash, bonus):
    data = _synthetic_bundle(mddd_verified=False)
    as_of = data.calendar[-1]
    baseline = compute_etf_statistics(data, as_of, 120)

    ex_date = data.calendar[-10]
    daily_index = data.daily.index[data.daily.date.eq(ex_date)][0]
    prior_close = float(data.daily.loc[daily_index - 1, "close"])
    original_ratio = (
        float(data.daily.loc[daily_index, "open"])
        / float(data.daily.loc[daily_index, "preclose"])
    )
    implied_reference = (prior_close - cash) / (1.0 + bonus)
    data.daily.loc[daily_index, "preclose"] = prior_close
    data.daily.loc[daily_index, "open"] = implied_reference * original_ratio
    data.actions = pd.concat([
        data.actions,
        pd.DataFrame([{
            "ex_date": ex_date,
            "cash_per_share": cash,
            "bonus_ratio": bonus,
            "implemented": True,
            "unsupported_reason": "",
        }]),
    ], ignore_index=True)

    repaired = compute_etf_statistics(data, as_of, 120)
    assert repaired["status"] == "completed"
    assert len(repaired["actions_applied"]) == 1
    action = repaired["actions_applied"][0]
    assert action["source"] == "cash_and_bonus"
    assert action["preclose_mismatch"] is not None
    assert action["factor"] == pytest.approx(implied_reference / prior_close)
    assert repaired["metrics"]["iovr"] == pytest.approx(
        baseline["metrics"]["iovr"], rel=1e-10, abs=1e-12,
    )


def _five_minute_clock(returns: np.ndarray) -> pd.DataFrame:
    days = pd.bdate_range("2025-01-02", periods=math.ceil((len(returns) + 1) / 48))
    clock = statistics._slot_table(days)
    clock = clock[clock.slot.mod(5).eq(4)].reset_index(drop=True)
    clock = clock.iloc[: len(returns) + 1].copy()
    clock["valid"] = True
    clock["close"] = np.exp(np.r_[math.log(1.1), math.log(1.1) + np.cumsum(returns)])
    return clock


def test_variance_ratio_distinguishes_random_walk_trend_and_mean_reversion():
    rng = np.random.default_rng(81)
    innovations = rng.normal(0.0, 0.0007, size=4_000)
    trend = np.empty_like(innovations)
    reverting = np.empty_like(innovations)
    trend[0] = innovations[0]
    reverting[0] = innovations[0]
    for index in range(1, len(innovations)):
        trend[index] = 0.28 * trend[index - 1] + innovations[index]
        reverting[index] = -0.28 * reverting[index - 1] + innovations[index]

    random_ratio = statistics._variance_ratio(_five_minute_clock(innovations), 48)[0]
    trend_ratio = statistics._variance_ratio(_five_minute_clock(trend), 48)[0]
    reverting_ratio = statistics._variance_ratio(_five_minute_clock(reverting), 48)[0]
    assert random_ratio == pytest.approx(1.0, abs=0.20)
    assert trend_ratio > random_ratio * 1.10
    assert reverting_ratio < random_ratio * 0.90


def test_zero_crossing_uses_arithmetic_price_ma_not_geometric_log_ma():
    days = pd.bdate_range("2025-01-02", periods=20)
    clock = statistics._slot_table(days)
    clock = clock[clock.slot.mod(5).eq(4)].reset_index(drop=True)
    rng = np.random.default_rng(0)
    price = pd.Series(np.exp(rng.normal(0.0, 0.6, len(clock))) * 5.0)
    clock["close"] = price
    clock["valid"] = True

    arithmetic_centered = price - price.rolling(240, min_periods=240).mean()
    arithmetic_pairs = arithmetic_centered.notna() & arithmetic_centered.shift().notna()
    arithmetic_expected = (
        (arithmetic_centered * arithmetic_centered.shift() < 0) & arithmetic_pairs
    ).sum() / arithmetic_pairs.sum()
    log_centered = np.log(price) - np.log(price).rolling(240, min_periods=240).mean()
    log_pairs = log_centered.notna() & log_centered.shift().notna()
    log_expected = (
        (log_centered * log_centered.shift() < 0) & log_pairs
    ).sum() / log_pairs.sum()

    actual = statistics._zero_crossing(clock)[0]
    assert actual == pytest.approx(arithmetic_expected)
    assert arithmetic_expected != pytest.approx(log_expected)


def test_thirty_minute_ou_fit_matches_stationary_ar1_and_rejects_invalid_phi():
    rng = np.random.default_rng(83)
    days = pd.bdate_range("2025-01-02", periods=90)
    clock = statistics._slot_table(days)
    clock = clock[clock.slot.mod(30).eq(29)].reset_index(drop=True)
    assert len(clock) == 90 * 8
    clock["bucket"] = clock.slot // 30
    mu = math.log(1.1)
    phi = 0.84
    values: list[float] = []
    for _ in range(90):
        for _session in range(2):
            state = mu + rng.normal(0.0, 0.004)
            for _bucket in range(4):
                state = mu + phi * (state - mu) + rng.normal(0.0, 0.0012)
                values.append(state)
    clock["close"] = np.exp(values)
    clock["valid"] = True
    details, error = statistics._ou_fit(clock)
    assert error is None
    fitted = details["phi"]
    expected_half_life = math.log(2.0) / (8.0 * -math.log(fitted))
    assert fitted == pytest.approx(phi, abs=0.06)
    assert details["half_life_days"] == pytest.approx(expected_half_life)
    assert details["lower_price"] < details["center_price"] < details["upper_price"]

    invalid = clock.copy()
    invalid["close"] = 1.1
    invalid_details, invalid_error = statistics._ou_fit(invalid)
    assert "half_life_days" not in invalid_details or invalid_details["half_life_days"] is None
    assert "不满足0<phi<1" in invalid_error or "不可拟合" in invalid_error


def test_jump_ratio_rises_with_a_jump_but_continuous_path_stays_low():
    base = 0.00025 * np.sin(np.arange(3_000) * 0.31)
    days = pd.bdate_range("2025-01-02", periods=math.ceil(len(base) / 46))
    labels = []
    sessions = []
    for index in range(len(base)):
        labels.append(days[index // 46])
        sessions.append((index % 46) // 23)
    smooth = pd.DataFrame({"day": labels, "session": sessions, "r": base})
    jumpy = smooth.copy()
    jumpy.loc[1_500, "r"] = 0.03

    smooth_result = statistics._jump_and_semivariance(smooth)
    jump_result = statistics._jump_and_semivariance(jumpy)
    assert smooth_result[3] is not None and smooth_result[3] < 0.03
    assert jump_result[3] > smooth_result[3] + 0.005


def test_jump_bv_does_not_pair_returns_across_a_missing_five_minute_interval():
    day = pd.Timestamp("2026-06-02")
    clock = statistics._slot_table(pd.DatetimeIndex([day]))
    clock = clock[clock.slot.mod(5).eq(4)].head(6).reset_index(drop=True)
    clock["valid"] = True
    clock["open"] = 1.0
    clock["close"] = [1.1, np.nan, 1.1, 1.10011, 1.09989, 1.09967]
    clock.loc[1, "valid"] = False

    returns = statistics._within_session_returns(clock)
    assert list(returns.end_slot) == [4, 19, 24, 29]
    assert list(returns.start_slot) == [4, 14, 19, 24]
    assert returns.end_slot.iloc[1] != returns.end_slot.iloc[0]

    values = returns.r.to_numpy(dtype=float)
    rv = float(values @ values)
    adjacent_after_gap = np.abs(values[2:]) * np.abs(values[1:-1])
    pair_count = int(adjacent_after_gap.size)
    correction = (pair_count + 1.0) / pair_count
    bv = 0.5 * math.pi * correction * float(adjacent_after_gap.sum())
    expected_jump = max(rv - bv, 0.0) / rv
    assert statistics._jump_and_semivariance(returns)[3] == pytest.approx(
        expected_jump, rel=1e-12, abs=1e-15,
    )

    data = _synthetic_bundle()
    missing_time = data.calendar[-2] + pd.Timedelta(hours=10, minutes=13)
    data.bars = data.bars[data.bars.timestamp.ne(missing_time)].reset_index(drop=True)
    incomplete = compute_etf_statistics(data, data.calendar[-1], 120)
    assert incomplete["status"] == "data_not_ready"
    assert incomplete["diagnostics"]["minute_integrity"]["unaccepted_slots"] >= 1
    assert incomplete["metrics"]["jump_ratio"] is not None
    assert incomplete["metric_details"]["jump_ratio"]["status"] == "available"


def test_atr30_uses_real_lunch_overnight_and_stops_at_an_unbridged_gap():
    days = pd.bdate_range("2025-01-02", periods=7)
    clock = statistics._slot_table(days)
    clock = clock[clock.slot.mod(30).eq(29)].reset_index(drop=True)
    clock["bucket"] = clock.slot // 30
    clock["open"] = 10.0
    clock["high"] = 10.5
    clock["low"] = 9.5
    clock["close"] = 10.0
    clock["valid"] = True
    clock["valid_ohlc"] = True

    daily_days = pd.bdate_range(end=days[-1], periods=8)
    daily = pd.DataFrame({"date": daily_days, "close": 10.0})
    daily.loc[daily.date.eq(days[5]), "close"] = 8.0

    lunch = clock.index[
        clock.day.eq(days[5]) & clock.session.eq(1) & clock.bucket.eq(0)
    ][0]
    clock.loc[lunch, ["high", "low"]] = [11.2, 9.5]
    morning_last = clock.index[
        clock.day.eq(days[5]) & clock.session.eq(0) & clock.bucket.eq(3)
    ][0]
    clock.loc[morning_last, "close"] = 8.0

    overnight = clock.index[
        clock.day.eq(days[6]) & clock.session.eq(0) & clock.bucket.eq(0)
    ][0]
    clock.loc[overnight, ["high", "low"]] = [11.2, 9.5]
    missing_bucket = clock.index[
        clock.day.eq(days[6]) & clock.session.eq(0) & clock.bucket.eq(1)
    ][0]
    clock.loc[missing_bucket, "valid_ohlc"] = False
    clock.loc[missing_bucket, "valid"] = False
    after_gap = clock.index[
        clock.day.eq(days[6]) & clock.session.eq(0) & clock.bucket.eq(2)
    ][0]
    clock.loc[after_gap, ["high", "low"]] = [11.2, 9.5]

    value, sample_count, day_count, error = statistics._atr30(clock, days, daily)
    expected_count = 7 * 8 - 2
    expected_mean = (52 * 0.10 + 0.40 + 0.40) / expected_count
    assert error is None
    assert sample_count == expected_count
    assert day_count == 7
    assert value == pytest.approx(expected_mean)


def test_source_quality_marks_ths_amount_estimates_and_never_infers_amount_from_verified_flag():
    day = pd.Timestamp("2026-06-01")
    days = pd.DatetimeIndex([day])
    clock = statistics._slot_table(days)
    base = clock.copy()
    base["open"] = 1.0
    base["high"] = 1.0
    base["low"] = 1.0
    base["close"] = 1.0
    base["volume"] = 10.0
    base["turnover"] = 10.0
    base["frequency"] = "1m"
    base["source"] = "ths"
    base["source_verified"] = True
    base["is_synthetic"] = False
    base["is_auction_placeholder"] = False
    daily = pd.DataFrame({"date": [day], "turnover": [2_400.0]})

    scaled = statistics._source_quality(base, days, daily)
    assert scaled["volume_coverage"] == pytest.approx(1.0)
    assert scaled["turnover_source"] == "estimated_ohlc4_daily_scaled"
    assert scaled["vwap_quality"] == "estimated_ohlc4_daily_scaled"

    daily["turnover"] = 4_800.0
    estimated = statistics._source_quality(base, days, daily)
    assert estimated["turnover_source"] == estimated["vwap_quality"] == "estimated"

    other = base.copy()
    other["source"] = "sina"
    assert statistics._source_quality(other, days, daily)["turnover_source"] == "unknown"
    assert statistics._source_quality(other, days, daily)["vwap_quality"] == "unknown"

    missing = base.copy()
    missing.loc[0, "is_synthetic"] = True
    coverage = statistics._source_quality(missing, days, daily)
    assert coverage["volume_coverage"] == pytest.approx(239 / 240)


def _score_row(symbol: str, *, eligible: bool = True, complete: bool = True) -> dict:
    metrics = {
        "rv_annualized": 0.20,
        "path_redundancy": 0.90,
        "variance_ratio_240m": 1.0,
        "iovr": 2.0,
        "zero_crossing_rate": 0.10,
        "jump_ratio": 0.10,
    }
    if not complete:
        metrics.pop("jump_ratio")
    return {
        "symbol": symbol, "status": "completed", "eligible": eligible,
        "metrics": metrics, "gss_score": 999, "rank": 999,
    }


def test_score_etfs_uses_average_tie_ranks_and_keeps_eligibility_separate():
    rows = [
        _score_row("B", eligible=False),
        _score_row("A", eligible=False),
        {
            **_score_row("C"),
            "metrics": {
                "rv_annualized": 0.30,
                "path_redundancy": 0.95,
                "variance_ratio_240m": 0.80,
                "iovr": 4.0,
                "zero_crossing_rate": 0.20,
                "jump_ratio": 0.05,
            },
        },
        _score_row("D", complete=False),
    ]
    returned = score_etfs(rows)
    assert returned is rows
    assert rows[3]["score_cohort_size"] == 3
    assert rows[3]["gss_score"] is None and rows[3]["rank"] is None
    # A and B have average ranks 1.5 in every component, hence percentile .25.
    assert rows[0]["gss_score"] == rows[1]["gss_score"] == pytest.approx(25.0)
    assert rows[2]["gss_score"] == pytest.approx(100.0)
    assert rows[2]["rank"] == 1
    assert rows[1]["rank"] == 2  # stable symbol breaks the exact GSS tie
    assert rows[0]["rank"] == 3
    assert sum(rows[0]["score_contributions"].values()) == pytest.approx(25.0, abs=1e-6)
    assert rows[0]["eligible"] is False and rows[0]["gss_score"] is not None


def test_score_etfs_has_no_percentile_for_a_single_row_cohort():
    rows = [_score_row("ONLY")]
    score_etfs(rows)
    assert rows[0]["score_cohort_size"] == 1
    assert rows[0]["gss_score"] is None
    assert rows[0]["rank"] is None
    assert rows[0]["score_contributions"] == {}


def test_score_etfs_orients_only_variance_ratio_and_jump_ratio_downward():
    lower_risk = _score_row("LOW")
    lower_risk["metrics"].update({"variance_ratio_240m": 0.80, "jump_ratio": 0.05})
    higher_risk = _score_row("HIGH")
    higher_risk["metrics"].update({"variance_ratio_240m": 1.05, "jump_ratio": 0.20})
    rows = [higher_risk, lower_risk]
    score_etfs(rows)
    # The four other components tie at percentile .5 and contribute 35 points.
    assert lower_risk["score_contributions"]["variance_ratio_240m"] == pytest.approx(20.0)
    assert higher_risk["score_contributions"]["variance_ratio_240m"] == pytest.approx(0.0)
    assert lower_risk["score_contributions"]["jump_ratio"] == pytest.approx(10.0)
    assert higher_risk["score_contributions"]["jump_ratio"] == pytest.approx(0.0)
    assert lower_risk["gss_score"] == pytest.approx(65.0)
    assert higher_risk["gss_score"] == pytest.approx(35.0)
    assert lower_risk["rank"] == 1 and higher_risk["rank"] == 2


def _candidate_basis(**overrides) -> dict:
    basis = {
        "reference_price": 1.0,
        "lower_price": 0.72,
        "upper_price": 1.38,
        "current_raw_price": 1.01,
        "atr30_pct": 0.01,
        "svr": 1.60,
        "realized_skewness": 0.45,
        "realized_kurtosis": 1.2,
        "price_tick": 0.001,
        "mean_raw_price": 1.0,
    }
    basis.update(overrides)
    return basis


def _actual_lines(candidate: dict, tick: float) -> tuple[list[float], list[float]]:
    mid = round_to_tick(candidate["reference_price"], tick)
    down = [
        round_to_tick(mid * (1.0 - candidate["buy_step"]) ** index, tick)
        for index in range(1, candidate["grids_down_count"] + 1)
    ]
    up = [
        round_to_tick(mid * (1.0 + candidate["sell_step"]) ** index, tick)
        for index in range(1, candidate["grids_up_count"] + 1)
    ]
    return down, up


def test_candidates_keep_symmetric_family_and_tick_safe_asymmetric_mirror():
    basis = _candidate_basis()
    candidates = generate_candidates(basis)
    assert candidates
    assert len(candidates) <= 100
    minimum = max(1.2 * basis["atr30_pct"], 5 * basis["price_tick"] / basis["mean_raw_price"])
    assert min(candidate["base_step"] for candidate in candidates) >= minimum
    assert all(
        minimum - 1e-12 <= step <= 0.05 + 1e-12
        for candidate in candidates
        for step in (candidate["buy_step"], candidate["sell_step"])
    )
    bases = {candidate["base_step"] for candidate in candidates}
    symmetric = {
        candidate["base_step"]: candidate for candidate in candidates
        if not candidate["asymmetric"]
    }
    assert bases == set(symmetric)
    assert all(
        symmetric[base]["buy_step"] == symmetric[base]["sell_step"] == base
        for base in bases
    )

    asymmetric = [candidate for candidate in candidates if candidate["asymmetric"]]
    assert asymmetric
    assert all(candidate["sell_step"] > candidate["buy_step"] for candidate in asymmetric)
    assert all(
        candidate["sell_step"] / candidate["buy_step"] <= 1.60 + 1e-12
        for candidate in asymmetric
    )
    for candidate in candidates:
        assert candidate["lower_price"] == round_to_tick(candidate["lower_price"], basis["price_tick"])
        assert candidate["upper_price"] == round_to_tick(candidate["upper_price"], basis["price_tick"])
        assert candidate["lower_price"] < candidate["reference_price"] < candidate["upper_price"]
        down, up = _actual_lines(candidate, basis["price_tick"])
        assert len(down) == len(set(down)) == candidate["grids_down_count"]
        assert len(up) == len(set(up)) == candidate["grids_up_count"]
        assert down and up
        assert min(down) >= candidate["lower_price"]
        assert max(up) <= candidate["upper_price"]
        assert all(line < candidate["reference_price"] for line in down)
        assert all(line > candidate["reference_price"] for line in up)


def test_candidate_with_any_rounded_duplicate_line_is_rejected_as_a_whole():
    assert statistics._candidate_for_steps(
        base_step=0.00001, buy_step=0.00001, sell_step=0.01,
        reference=10.0, lower=9.0, upper=11.0,
        current=10.0, tick=0.001,
    ) is None


def test_negative_skew_mirrors_asymmetry_and_invalid_ranges_produce_no_candidate():
    negative = generate_candidates(_candidate_basis(svr=0.80, realized_skewness=-0.45))
    asymmetric = [candidate for candidate in negative if candidate["asymmetric"]]
    assert asymmetric
    assert all(candidate["buy_step"] > candidate["sell_step"] for candidate in asymmetric)
    assert all(
        candidate["buy_step"] / candidate["sell_step"] <= 1.60 + 1e-12
        for candidate in asymmetric
    )
    assert generate_candidates(_candidate_basis(current_raw_price=1.50)) == []
    assert generate_candidates(_candidate_basis(lower_price=1.10, upper_price=1.20)) == []
    assert generate_candidates(_candidate_basis(atr30_pct=0.06)) == []
