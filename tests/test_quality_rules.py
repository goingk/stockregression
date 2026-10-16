import pandas as pd
import pytest

from stockregression.data import DataService
from stockregression.quality import DataQualityError, apply_price_limits, canonical_frame, minute_slots
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


def index_clock_minute(days, volume=100.0):
    """Tushare-style export: the bar clock lives in a trade_date/trade_time index."""
    stamps = pd.DatetimeIndex([stamp for day in days for stamp in minute_slots(day)])
    return pd.DataFrame({"ts_code": "159611.SZ", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0,
                         "vol": volume, "amount": 1000.0},
                        index=pd.MultiIndex.from_arrays([stamps.normalize(), stamps], names=["trade_date", "trade_time"]))


def test_index_clock_export_is_promoted_to_timestamp():
    raw = index_clock_minute(pd.bdate_range("2026-01-05", periods=2))

    frame = canonical_frame(raw, source="import")

    assert frame.timestamp.tolist() == list(pd.DatetimeIndex([stamp for day in raw.index.levels[0].unique() for stamp in minute_slots(day)]))
    assert frame.volume.tolist() == [100.0] * 480
    assert frame.turnover.tolist() == [1000.0] * 480


def test_minute_frame_with_non_positive_price_is_rejected():
    stamps = minute_slots(pd.Timestamp("2026-01-05"))
    raw = pd.DataFrame({"timestamp": stamps, "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 100.0})
    raw.loc[5, "open"] = 0.0

    with pytest.raises(DataQualityError, match="非正价格"):
        canonical_frame(raw, source="sina")


def test_unnamed_datetime_index_is_promoted_to_timestamp():
    stamps = minute_slots(pd.Timestamp("2026-01-05"))
    raw = pd.DataFrame({"open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "vol": 100.0}, index=stamps)

    frame = canonical_frame(raw, source="import")

    assert frame.timestamp.tolist() == list(stamps)


def test_import_accepts_index_clock_minute_file(tmp_path):
    days = pd.bdate_range("2026-01-05", periods=3)
    daily = daily_fixture(days)
    minute = index_clock_minute(days)
    daily_path, minute_path, calendar_path = [tmp_path / name for name in ("daily.csv", "159611.SZ.parquet", "calendar.csv")]
    daily.to_csv(daily_path, index=False)
    minute.to_parquet(minute_path)
    pd.DataFrame({"date": days}).to_csv(calendar_path, index=False)
    store = Store(tmp_path / "data")
    service = DataService(store)

    result = service.import_files({"symbol": "159611.SZ", "etf_category": "stock", "daily_path": str(daily_path),
                                   "minute_path": str(minute_path), "calendar_path": str(calendar_path), "label_convention": "end"})

    assert result["profiles"]["import"]["verified"] is True
    assert result["profiles"]["import"]["convention"] == "end"
    published = store.read_files(store.snapshot("159611.SZ", result["version"])["files"]["minute"])
    assert len(published) == len(days) * 240


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
