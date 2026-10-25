import numpy as np
import pandas as pd

from stockregression.providers import normalize_actions
from stockregression.quality import minute_slots, prepare_bundle
from stockregression.rules import instrument_terms


def fixture(missing_auction=True, extra_missing=(), unverified=False):
    day = pd.Timestamp("2026-10-08")
    stamps = minute_slots(day)
    minute = pd.DataFrame({"timestamp": stamps, "open": 10., "high": 10.1,
        "low": 9.9, "close": 10., "volume": 100., "turnover": 1000.,
        "source": "sina", "source_verified": not unverified})
    # A file source may retain zero-volume clock slots during the auction.
    minute.loc[237:238, ["open", "high", "low", "close"]] = 10.
    minute.loc[237:238, ["volume", "turnover"]] = 0.
    minute.loc[239, "close"] = 10.02
    daily = pd.DataFrame({"date": [day], "open": [10.], "high": [10.1], "low": [9.9],
        "close": [10.02], "preclose": [10.], "volume": [minute.volume.sum()],
        "turnover": [minute.turnover.sum()], "tradestatus": [1], "is_st": [False]})
    removed = ([237, 238] if missing_auction else []) + list(extra_missing)
    minute = minute.drop(index=removed).reset_index(drop=True)
    return minute, daily, pd.DataFrame({"date": [day]})


def prepare(minute, daily, calendar, *, symbol="159611.SZ", frequency="1m"):
    return prepare_bundle(symbol, minute, daily, normalize_actions(pd.DataFrame()), calendar,
        daily.date.min(), daily.date.max(), frequency, "auction-test",
        instrument_terms(symbol, {"actions_verified": True}), {"sina": {"verified": True}})


def test_trade_only_closing_auction_slots_are_not_missing_market_data():
    minute, daily, calendar = fixture()
    result = prepare(minute, daily, calendar)
    quality, report = result.quality, result.quality["days"][0]
    assert len(minute) == 238
    assert quality["minute_ratio"] == 1.
    assert report["missing_minutes"] == 2
    assert report["missing_data_minutes"] == 0
    assert report["auction_placeholders"] == quality["auction_placeholders"] == 2
    assert not report["download_needed"]
    bars = result.bars.set_index("timestamp")
    placeholders = bars[bars.is_auction_placeholder.eq(True)]
    assert len(placeholders) == 2
    assert placeholders.is_synthetic.all() and not placeholders.tradable.any()
    assert (placeholders.volume == 0).all()
    assert np.array_equal(placeholders.close.to_numpy(), [10., 10.])
    assert bars.loc[pd.Timestamp("2026-10-08 15:00"), "close"] == 10.02


def test_real_gap_and_unverified_sources_remain_visible():
    minute, daily, calendar = fixture(extra_missing=(80,))
    result = prepare(minute, daily, calendar)
    assert result.quality["days"][0]["missing_data_minutes"] == 1
    assert result.quality["days"][0]["auction_placeholders"] == 2
    assert result.quality["synthetic_bars"] == 3
    minute, daily, calendar = fixture(unverified=True)
    result = prepare(minute, daily, calendar)
    assert result.quality["pending_calibration_days"] == 1
    assert result.quality["days"][0]["mode"] == "daily"
    assert result.quality["days"][0]["auction_placeholders"] == 0


def test_missing_closing_anchor_is_not_reclassified_and_sh_closing_auction_is():
    minute, daily, calendar = fixture(extra_missing=(239,))
    result = prepare(minute, daily, calendar)
    assert result.quality["days"][0]["auction_placeholders"] == 0
    assert result.quality["days"][0]["missing_data_minutes"] == 3
    minute, daily, calendar = fixture()
    result = prepare(minute, daily, calendar, symbol="510300.SH")
    assert result.quality["days"][0]["auction_placeholders"] == 2
    assert result.quality["days"][0]["missing_data_minutes"] == 0


def test_auction_omissions_do_not_disable_observed_five_minute_closing_bar():
    sparse = prepare(*fixture(), frequency="5m")
    padded = prepare(*fixture(missing_auction=False), frequency="5m")
    columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover", "tradable", "is_synthetic"]
    pd.testing.assert_frame_equal(sparse.bars[columns], padded.bars[columns], check_dtype=False)
    closing = sparse.bars.iloc[-1]
    assert closing.timestamp == pd.Timestamp("2026-10-08 15:00")
    assert closing.tradable and not closing.is_synthetic
    assert sparse.quality["synthetic_bars"] == sparse.quality["auction_placeholders"] == 2
