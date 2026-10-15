import pandas as pd
import pytest

from stockregression.data import DataService
from stockregression.quality import apply_price_limits, minute_slots
from stockregression.storage import Store


def daily_fixture(days):
    return pd.DataFrame({"date": days, "open": 10.0, "high": 10.1, "low": 9.9,
                         "close": 10.0, "preclose": 10.0, "volume": 24000,
                         "turnover": 240000, "tradestatus": 1, "is_st": False})


@pytest.mark.parametrize("board,listing", [("main", "2024-01-02"), ("chinext", "2021-01-04"), ("star", "2019-07-22")])
def test_modern_ipo_first_five_actual_trading_days_have_no_limit(board, listing):
    days = pd.bdate_range(listing, periods=7)
    result = apply_price_limits(daily_fixture(days), pd.DataFrame({"date": days}), {"board": board, "listing_date": listing})
    assert result.no_price_limit.tolist() == [True] * 5 + [False] * 2
    assert result.iloc[:5].limit_up.isna().all()
    expected = 11 if board == "main" else 12
    assert result.iloc[5].limit_up == expected


def test_nuclear_ipo_first_day_and_old_listing_outside_calendar():
    days = pd.bdate_range("2016-01-04", periods=3)
    daily = daily_fixture(days)
    calendar = pd.DataFrame({"date": days})
    result = apply_price_limits(daily, calendar, {"board": "main", "listing_date": "2016-01-04"})
    assert result.iloc[0].limit_up == 14.4 and result.iloc[0].limit_down == 6.4
    assert result.iloc[1].limit_up == 11
    old = apply_price_limits(daily, calendar, {"board": "main", "listing_date": "1999-11-10"})
    assert not old.no_price_limit.any() and (old.limit_up == 11).all()


def test_import_without_explicit_label_is_not_silently_verified(tmp_path):
    days = pd.bdate_range("2026-01-05", periods=3)
    daily = daily_fixture(days)
    timestamps = minute_slots(days[0]).append(minute_slots(days[1])).append(minute_slots(days[2]))
    minute = pd.DataFrame({"timestamp": timestamps, "open": 10, "high": 10.1, "low": 9.9, "close": 10, "volume": 100, "turnover": 1000})
    daily_path, minute_path, calendar_path = [tmp_path / name for name in ("daily.csv", "minute.csv", "calendar.csv")]
    daily.to_csv(daily_path, index=False)
    minute.to_csv(minute_path, index=False)
    pd.DataFrame({"date": days}).to_csv(calendar_path, index=False)
    service = DataService(Store(tmp_path / "data"))
    result = service.import_files({"symbol": "600000.SH", "daily_path": str(daily_path), "minute_path": str(minute_path), "calendar_path": str(calendar_path)})
    assert not result["profiles"]["import"]["verified"]
    explicit = service.import_files({"symbol": "600000.SH", "daily_path": str(daily_path), "minute_path": str(minute_path), "calendar_path": str(calendar_path), "label_convention": "end"})
    assert explicit["profiles"]["import"]["verified"]
