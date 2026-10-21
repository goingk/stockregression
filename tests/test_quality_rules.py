import pandas as pd
import pytest

from stockregression.data import DataService
from stockregression.market_data.models import MarketDataError
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










@pytest.mark.parametrize("kind", ["minute", "daily"])
def test_import_rejects_mismatched_file_symbol_before_network(tmp_path, monkeypatch, kind):
    days = pd.bdate_range("2026-01-05", periods=3)
    frame = index_clock_minute(days) if kind == "minute" else daily_fixture(days).assign(ts_code="159611.SZ")
    path = tmp_path / "input.parquet"
    frame.to_parquet(path)
    store = Store(tmp_path / "data")
    monkeypatch.setattr("stockregression.data.fetch_auxiliary_routed", lambda *args: pytest.fail("代码不一致时不得联网"))
    with pytest.raises((DataQualityError, MarketDataError), match="195611.SZ.*159611.SZ.*不一致"):
        DataService(store).import_files({"symbol": "195611.SZ", kind + "_path": str(path)})
    assert store.snapshot("195611.SZ") is None


def test_opening_auction_is_allocated_once_to_first_end_labelled_bar():
    from stockregression.quality import align_minutes
    day = pd.Timestamp("2026-01-05")
    raw = canonical_frame(index_clock_minute(pd.DatetimeIndex([day])))
    auction = raw.iloc[[0]].copy()
    auction["timestamp"] = day + pd.Timedelta(hours=9, minutes=30)
    auction["open"], auction["high"], auction["low"] = 9.95, 10.2, 9.8
    auction["volume"], auction["turnover"] = 500.0, 5000.0
    aligned = align_minutes(pd.concat([auction, auction, raw], ignore_index=True), "end")
    assert len(aligned) == 240
    assert aligned.volume.sum() == 24500
    first = aligned.iloc[0]
    assert first.open == 9.95 and first.high == 10.2 and first.low == 9.8
    assert first.close == 10.0 and first.volume == 600
    assert first.raw_volume == 600 and first.opening_auction_volume == 500


@pytest.mark.parametrize("error,expected", [(1e-14, "minute"), (0.01, "daily")])
def test_minute_high_low_check_tolerates_only_float_noise(error, expected):
    from stockregression.quality import prepare_bundle
    from stockregression.providers import normalize_actions
    day = pd.Timestamp("2026-01-05")
    minute = canonical_frame(index_clock_minute(pd.DatetimeIndex([day])))
    minute["source_verified"] = True
    minute["high"] = 10.0 - error
    minute["low"] = 10.0 + error
    daily = daily_fixture(pd.DatetimeIndex([day])).assign(high=10.0, low=10.0)
    bundle = prepare_bundle("159611.SZ", minute, daily, normalize_actions(pd.DataFrame()),
                            pd.DataFrame({"date": [day]}), day, day, "1m", "test",
                            {"asset_type": "etf", "price_tick": 0.001}, {})
    assert bundle.quality["days"][0]["mode"] == expected
