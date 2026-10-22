"""Regressions for minute gap filling and the coverage metrics around it.

The gaps that matter are the ones a reader would otherwise mistake for a
filled bar: a run that is too long to rebuild, a slot blocked by an invalid
price and the session boundaries that cannot be carried forward.  Every one
of them has to stay missing and has to be counted exactly once.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockregression.market_data.quality import _fill_minutes, validate_and_fill
from stockregression.quality import minute_slots


DAY = pd.Timestamp("2026-10-15")
MORNING = 120  # 11:30, the last bar of the morning session
AFTERNOON = 120  # 13:01, the first bar of the afternoon session


def minute_frame(day: pd.Timestamp = DAY, *, drop=(), source: str = "sina") -> pd.DataFrame:
    slots = minute_slots(day)
    frame = pd.DataFrame({
        "timestamp": slots,
        "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0,
        "volume": 100.0, "turnover": 1000.0,
        "source": source, "source_verified": True,
        "is_synthetic": False, "is_auction_placeholder": False,
    })
    if drop:
        frame = frame.drop(index=list(drop)).reset_index(drop=True)
    return frame


def validate(frame: pd.DataFrame, *, symbol: str = "510300.SH", day: pd.Timestamp = DAY):
    return validate_and_fill(
        frame, "1min", calendar=pd.DataFrame({"date": [day]}), fill=True,
        coverage_start=day, coverage_end=day + pd.Timedelta(days=1), symbol=symbol,
    )


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6])
def test_consecutive_gap_is_filled_only_up_to_the_run_limit(count):
    dropped = list(range(50, 50 + count))
    result = validate(minute_frame(drop=dropped))
    report = result.report
    expected_synthetic = count if count <= 3 else 0

    assert report["synthetic_rows"] == expected_synthetic
    assert len(result.frame) == 240 - count + expected_synthetic
    assert report["missing_rows"] == count - expected_synthetic
    assert report["unfillable"] == count - expected_synthetic
    assert report["gap_count"] == (0 if count <= 3 else 1)
    missing_stamps = minute_slots(DAY)[dropped]
    present = result.frame.timestamp.isin(missing_stamps).sum()
    assert present == expected_synthetic
    assert result.frame.is_synthetic.sum() == expected_synthetic
    assert result.frame.source_verified.all()


def test_four_consecutive_missing_bars_stay_a_real_gap():
    dropped = list(range(50, 54))
    result = validate(minute_frame(drop=dropped))
    report = result.report

    assert len(result.frame) == 236
    assert report["synthetic_rows"] == 0
    assert report["missing_rows"] == 4
    assert report["unfillable"] == 4
    assert not result.frame.is_synthetic.any()
    assert not result.frame.timestamp.isin(minute_slots(DAY)[dropped]).any()
    assert report["status"] == "partial"
    assert any("未补齐" in item for item in result.pending)


def test_two_consecutive_gaps_are_each_measured_once():
    result = validate(minute_frame(drop=[50, 51, 52, 53, 60]))
    report = result.report

    assert report["synthetic_rows"] == 1
    assert report["missing_rows"] == 4
    assert report["unfillable"] == 4
    assert report["gap_count"] == 1
    assert len(result.frame) == 236
    assert not result.frame.timestamp.isin(minute_slots(DAY)[[50, 51, 52, 53]]).any()


def test_lunch_break_boundary_slots_are_never_carried_forward():
    result = validate(minute_frame(drop=[MORNING - 1, MORNING]))
    report = result.report

    assert report["synthetic_rows"] == 0
    assert report["missing_rows"] == 2
    assert report["unfillable"] == 2
    assert len(result.frame) == 238
    boundary = minute_slots(DAY)[[MORNING - 1, MORNING]]
    assert not result.frame.timestamp.isin(boundary).any()


def test_session_open_and_close_slots_stay_missing():
    result = validate(minute_frame(drop=[0]))
    assert result.report["synthetic_rows"] == 0
    assert result.report["missing_rows"] == 1
    assert result.report["unfillable"] == 1

    result = validate(minute_frame(drop=[239]))
    assert result.report["synthetic_rows"] == 0
    assert result.report["missing_rows"] == 1
    assert result.report["unfillable"] == 1


def test_afternoon_gap_away_from_the_boundary_is_filled():
    dropped = [MORNING + 30]
    result = validate(minute_frame(drop=dropped))

    assert result.report["synthetic_rows"] == 1
    assert result.report["missing_rows"] == 0
    assert result.report["unfillable"] == 0
    assert len(result.frame) == 240
    assert result.frame.timestamp.isin(minute_slots(DAY)[dropped]).any()


def test_blocked_bar_is_not_filled_and_is_counted_once():
    blocked = minute_slots(DAY)[50]
    frame = minute_frame(drop=[50])
    frame.attrs["blocked_timestamps"] = [blocked.isoformat()]

    result = validate(frame)

    assert result.report["blocked_price_rows"] == 1
    assert result.report["synthetic_rows"] == 0
    assert result.report["missing_rows"] == 1
    assert result.report["unfillable"] == 1
    assert not result.frame.timestamp.eq(blocked).any()


def test_blocked_gap_is_not_counted_twice_inside_the_fill():
    blocked = minute_slots(DAY)[50]
    frame = minute_frame(drop=[50])

    filled, count, unfillable = _fill_minutes(
        frame, pd.DatetimeIndex([DAY]), blocked=pd.DatetimeIndex([blocked]),
    )

    assert count == 0
    assert len(unfillable) == 1
    assert unfillable == [blocked]
    assert not filled.timestamp.eq(blocked).any()
    assert len(filled) == 239


def test_blocked_bar_next_to_a_short_gap_keeps_both_missing():
    blocked = minute_slots(DAY)[50]
    frame = minute_frame(drop=[50, 51])

    filled, count, unfillable = _fill_minutes(
        frame, pd.DatetimeIndex([DAY]), blocked=pd.DatetimeIndex([blocked]),
    )

    assert count == 0
    assert len(unfillable) == 2
    assert not filled.timestamp.isin(minute_slots(DAY)[[50, 51]]).any()


def test_filled_bars_are_labelled_and_real_bars_keep_their_source():
    result = validate(minute_frame(drop=[50], source="tdx"))
    filled = result.frame[result.frame.is_synthetic]
    real = result.frame[~result.frame.is_synthetic]

    assert len(filled) == 1
    assert filled.source.eq("fill").all()
    assert not filled.tradable.any()
    assert real.source.eq("tdx").all()
    assert real.source_verified.all()
    assert not real.is_synthetic.any()


def test_is_synthetic_is_boolean_after_filling():
    result = validate(minute_frame(drop=[50, 51, 52, 53, 54, 55]))

    assert result.frame.is_synthetic.dtype == bool
    assert result.frame.source_verified.dtype == bool
    assert result.frame.is_auction_placeholder.dtype == bool
    assert not result.frame.is_synthetic.any()


def test_a_day_without_any_record_stays_a_real_gap():
    other = DAY + pd.Timedelta(days=1)
    frame = minute_frame(DAY)

    result = validate_and_fill(
        frame, "1min", calendar=pd.DataFrame({"date": [DAY, other]}), fill=True,
        coverage_start=DAY, coverage_end=other + pd.Timedelta(days=1), symbol="510300.SH",
    )

    assert len(result.frame) == 240
    assert result.report["synthetic_rows"] == 0
    assert result.report["missing_rows"] == 240
    assert result.report["unfillable"] == 240
    assert result.report["status"] == "partial"


def test_closing_auction_placeholders_are_kept_and_never_filled():
    dropped = [237, 238]
    result = validate(minute_frame(drop=dropped), symbol="159611.SZ")
    report = result.report

    assert report["auction_placeholders"] == 2
    assert report["synthetic_rows"] == 0
    assert report["missing_rows"] == 0
    placeholders = result.frame[result.frame.is_auction_placeholder]
    assert len(placeholders) == 2
    assert not placeholders.is_synthetic.any()
    assert (placeholders.volume == 0).all()
    assert placeholders.source.eq("auction-placeholder").all()
    assert len(result.frame) == 240


def test_auction_placeholder_next_to_a_real_gap_keeps_both_facts():
    dropped = [80, 237, 238]
    result = validate(minute_frame(drop=dropped), symbol="159611.SZ")
    report = result.report

    assert report["auction_placeholders"] == 2
    assert report["synthetic_rows"] == 1
    assert report["missing_rows"] == 0
    assert len(result.frame) == 240
    assert result.frame.source_verified.all()


def test_report_counts_agree_with_the_returned_frame():
    result = validate(minute_frame(drop=[50, 51, 52, 53, 200, 201]))
    report = result.report

    assert report["expected_rows"] == 240
    assert report["observed_rows"] == 234
    assert report["synthetic_rows"] == 0
    assert report["missing_rows"] == 6
    assert report["unfillable"] == 6
    assert report["gap_count"] == 2
    assert len(result.frame) == 234
    assert result.frame.timestamp.is_unique
    assert result.frame.timestamp.is_monotonic_increasing
