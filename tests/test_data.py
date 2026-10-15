from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockregression.data import DataService, merge_frames
from stockregression.models import BacktestConfig
from stockregression.quality import (
    DataQualityError,
    align_minutes,
    infer_profile,
    minute_slots,
    prepare_bundle,
)
from stockregression.settings import load_settings, save_settings
from stockregression.storage import Store


def _market(days: int = 3, convention: str = "end"):
    calendar = pd.bdate_range("2024-01-02", periods=days)
    raw_rows = []
    reference_rows = []
    daily_rows = []
    for day_index, day in enumerate(calendar):
        slots = minute_slots(day)
        offset = day_index * 0.1
        close = 10 + offset + np.arange(240) * 0.0001
        opening = np.r_[close[0], close[:-1]]
        high = np.maximum(opening, close) + 0.001
        low = np.minimum(opening, close) - 0.001
        raw_volume = np.arange(1, 241, dtype="int64") + day_index
        raw_timestamps = slots - pd.Timedelta(minutes=1) if convention == "start" else slots
        for index in range(240):
            raw_rows.append({
                "timestamp": raw_timestamps[index],
                "open": opening[index],
                "high": high[index],
                "low": low[index],
                "close": close[index],
                "volume": int(raw_volume[index]),
                "turnover": float(raw_volume[index] * close[index]),
                "source": "hf",
            })
            reference_rows.append({
                "timestamp": slots[index],
                "open": opening[index],
                "high": high[index],
                "low": low[index],
                "close": close[index],
                "volume": int(raw_volume[index] * 100),
                "turnover": float(raw_volume[index] * close[index] * 100),
                "source": "reference",
            })
        daily_rows.append({
            "date": day,
            "open": float(opening[0]),
            "high": float(high.max()),
            "low": float(low.min()),
            "close": float(close[-1]),
            "volume": int(raw_volume.sum() * 100),
            "turnover": float((raw_volume * close).sum() * 100),
            "preclose": float(close[-1] - 0.1) if day_index else float(opening[0]),
            "tradestatus": 1,
            "is_st": False,
        })
    return (
        pd.DataFrame(raw_rows),
        pd.DataFrame(daily_rows),
        pd.DataFrame(reference_rows),
        calendar,
    )


def _profiles_for_import():
    return {"fixture": {"verified": True, "convention": "end", "volume_multiplier": 1}}


def _aggregate_five_minute_reference(reference: pd.DataFrame, days: pd.DatetimeIndex):
    rows = []
    for day in days:
        slots = minute_slots(day)
        part = reference[reference.timestamp.dt.normalize() == day].set_index("timestamp")
        observed = part.reindex(slots)
        for offset in (0, 120):
            session = slots[offset:offset + 120]
            for index in range(0, 120, 5):
                bucket = observed.iloc[offset + index:offset + index + 5]
                assert len(bucket) == 5
                rows.append({
                    "timestamp": session[index + 4],
                    "open": bucket.iloc[0].open,
                    "high": bucket.high.max(),
                    "low": bucket.low.min(),
                    "close": bucket.iloc[-1].close,
                    "volume": bucket.volume.sum(),
                })
    return pd.DataFrame(rows)


def _volatile_market():
    raw, daily, reference, calendar = _market()
    pattern = np.array([0.0, 0.04, 0.10, 0.02, -0.06, 0.08, -0.10, 0.03, 0.07, -0.03])
    for day_index, day in enumerate(calendar):
        mask = raw.timestamp.dt.normalize() == day
        indexes = raw.index[mask]
        close = 10 + day_index * 0.2 + pattern[np.arange(240) % len(pattern)]
        opening = np.r_[close[0], close[:-1]]
        high = np.maximum(opening, close) + 0.002
        low = np.minimum(opening, close) - 0.002
        raw.loc[indexes, ["open", "high", "low", "close"]] = np.column_stack([opening, high, low, close])
        raw.loc[indexes, "turnover"] = raw.loc[indexes, "volume"].to_numpy() * close
        ref_indexes = reference.index[reference.timestamp.dt.normalize() == day]
        reference.loc[ref_indexes, ["open", "high", "low", "close"]] = np.column_stack([opening, high, low, close])
        reference.loc[ref_indexes, "turnover"] = reference.loc[ref_indexes, "volume"].to_numpy() * close
        daily_index = daily.index[daily.date == day][0]
        daily.loc[daily_index, ["open", "high", "low", "close"]] = [opening[0], high.max(), low.min(), close[-1]]
    return raw, daily, reference, calendar


def test_demo_load_bundle_reports_causal_fill_and_exclusive_daily_fallback(tmp_path):
    service = DataService(Store(tmp_path / "demo-store"))
    demo = service.demo(symbol="000001.SZ", days=100)
    manifest = service.store.snapshot(demo["symbol"], demo["version"])
    daily = service.store.read_files(manifest["files"]["daily"])
    config = BacktestConfig(
        symbol=demo["symbol"],
        start_date=daily.date.min().date(),
        end_date=daily.date.max().date(),
        frequency="1m",
    )

    bundle = service.load_bundle(config, version=demo["version"])

    assert demo["is_demo"] is True
    assert len(bundle.daily) == 100
    assert bundle.quality["minute_days"] == 99
    assert bundle.quality["daily_days"] == 1
    assert bundle.quality["minute_ratio"] == pytest.approx(0.99)
    assert any("演示数据" in warning for warning in bundle.warnings)

    gap_day = pd.Timestamp(bundle.quality["days"][50]["date"])
    gap_slots = minute_slots(gap_day)[90:92]
    day_bars = bundle.bars.set_index("timestamp")
    gap = day_bars.loc[gap_slots]
    preceding_close = day_bars.loc[minute_slots(gap_day)[89], "close"]
    assert bundle.quality["days"][50]["mode"] == "minute"
    assert bundle.quality["days"][50]["missing_minutes"] == 2
    assert gap["is_synthetic"].all()
    assert (gap["volume"] == 0).all()
    assert (gap["close"] == preceding_close).all()
    assert not gap["tradable"].any()

    fallback_day = pd.Timestamp(bundle.quality["days"][90]["date"])
    fallback = bundle.bars[bundle.bars.timestamp.dt.normalize() == fallback_day]
    assert bundle.quality["days"][90]["mode"] == "daily"
    assert bundle.quality["days"][90]["missing_minutes"] == 60
    assert len(fallback) == 1
    assert fallback.iloc[0]["frequency"] == "1d"
    assert not fallback.iloc[0]["is_synthetic"]


def test_instrument_board_metadata_agrees_with_trading_rules(tmp_path):
    from stockregression.rules import infer_board

    store = Store(tmp_path / "instruments")
    for symbol in ["600000.SH", "000001.SZ", "300001.SZ", "688001.SH", "689009.SH"]:
        store.put_instrument(symbol)
        record = store.instruments(symbol)[0]
        assert record["board"] == infer_board(symbol)


def test_parquet_duckdb_deduplication_and_republishing_keep_version(tmp_path):
    store = Store(tmp_path / "store")
    old = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-12-29 15:00", "2024-01-02 15:00"]),
        "close": [10.0, 11.0],
    })
    new = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02 15:00", "2024-01-03 15:00"]),
        "close": [12.0, 13.0],
    })
    minute = merge_frames(old, new, "timestamp")
    daily_old = pd.DataFrame({"date": pd.to_datetime(["2023-12-29", "2024-01-02"]), "close": [10.0, 11.0]})
    daily_new = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "close": [12.0, 13.0]})
    daily = merge_frames(daily_old, daily_new, "date")
    actions = pd.DataFrame(columns=["ex_date"])
    calendar = pd.DataFrame({"date": pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"])})
    metadata = {"name": "Fixture", "actions_verified": True}
    profiles = _profiles_for_import()

    first = store.publish("600000.SH", minute, daily, actions, calendar, metadata, profiles)
    second = store.publish("600000.SH", minute, daily, actions, calendar, metadata, profiles)
    read_minute = store.read_files(first["files"]["minute"])
    read_daily = store.read_files(first["files"]["daily"])
    filtered = store.read_files(
        first["files"]["minute"], "timestamp", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")
    )

    assert first["version"] == second["version"]
    assert len(read_minute) == 3
    assert read_minute.loc[read_minute.timestamp == pd.Timestamp("2024-01-02 15:00"), "close"].item() == 12.0
    assert read_daily.date.nunique() == len(read_daily) == 3
    assert filtered.timestamp.tolist() == [pd.Timestamp("2024-01-02 15:00")]


def test_load_bundle_rejects_calendar_trading_day_without_daily(tmp_path):
    store = Store(tmp_path / "strict-calendar")
    calendar_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    daily = pd.DataFrame({
        "date": [calendar_dates[0], calendar_dates[2]],
        "open": [10.0, 12.0],
        "high": [10.5, 12.5],
        "low": [9.5, 11.5],
        "close": [10.0, 12.0],
        "volume": [1000, 1000],
        "turnover": [10000, 12000],
        "preclose": [10.0, 10.0],
        "tradestatus": [1, 1],
        "is_st": [False, False],
    })
    empty = pd.DataFrame()
    manifest = store.publish(
        "600000.SH",
        empty,
        daily,
        pd.DataFrame(columns=["ex_date"]),
        pd.DataFrame({"date": calendar_dates}),
        {"actions_verified": True},
        _profiles_for_import(),
    )
    config = BacktestConfig(
        symbol="600000.SH",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 4),
        frequency="1m",
    )

    with pytest.raises(DataQualityError, match="交易日没有日线"):
        DataService(store).load_bundle(config, version=manifest["version"])


@pytest.mark.parametrize("convention", ["start", "end"])
def test_independent_minute_reference_calibrates_labels_and_volume_100(convention):
    raw, daily, reference, _ = _market(convention=convention)

    profile = infer_profile(raw, daily, "hf", reference=reference)

    assert profile["verified"] is True
    assert profile["convention"] == convention
    assert profile["volume_multiplier"] == 100
    assert profile["overlap_bars"] == 3 * 240
    assert profile["match_score"] == pytest.approx(1.0)


def test_independent_five_minute_reference_calibrates_volume_100():
    raw, daily, reference, calendar = _volatile_market()
    reference_5m = _aggregate_five_minute_reference(reference, calendar)

    profile = infer_profile(
        raw,
        daily,
        "hf",
        reference=reference_5m,
        reference_frequency="5m",
    )

    assert profile["verified"] is True
    assert profile["convention"] == "end"
    assert profile["volume_multiplier"] == 100
    assert profile["overlap_bars"] == 3 * 48
    assert profile["match_score"] == pytest.approx(1.0)
    assert profile["reference_volume_score"] == pytest.approx(1.0)


def test_incomplete_five_minute_groups_are_excluded_from_reference_score():
    raw, daily, reference, calendar = _volatile_market()
    reference_5m = _aggregate_five_minute_reference(reference, calendar)
    # Remove the first end-labelled minute of each day. Each 09:31–09:35
    # reference bucket is now partial and must not count as a matched 5m bar.
    missing = pd.DatetimeIndex([minute_slots(day)[0] for day in calendar])
    partial = raw[~raw.timestamp.isin(missing)].copy()

    profile = infer_profile(
        partial,
        daily,
        "hf",
        reference=reference_5m,
        reference_frequency="5m",
    )

    assert profile["verified"] is True
    assert profile["convention"] == "end"
    assert profile["volume_multiplier"] == 100
    assert profile["overlap_bars"] == len(reference_5m) - len(calendar)
    assert profile["overlap_bars"] == 3 * 47


def test_uncalibrated_provider_is_downgraded_to_daily():
    raw, daily, _, calendar = _market()
    profile = infer_profile(raw, daily, "hf")
    aligned = align_minutes(raw, profile["convention"], profile["volume_multiplier"])
    aligned["source_verified"] = profile["verified"]

    bundle = prepare_bundle(
        "600000.SH",
        aligned,
        daily,
        pd.DataFrame(),
        pd.DataFrame({"date": calendar}),
        calendar[0],
        calendar[-1],
        "1m",
        "fixture",
        {},
        {"hf": profile},
    )

    assert profile["verified"] is False
    assert all(report["mode"] == "daily" for report in bundle.quality["days"])
    assert all("未校准" in report["reason"] for report in bundle.quality["days"])
    assert all(bundle.bars.groupby(bundle.bars.timestamp.dt.normalize()).size() == 1)


def test_settings_environment_override_and_nested_schedule_save(tmp_path, monkeypatch):
    import stockregression.settings as settings_module

    settings_path = tmp_path / ".local" / "settings.json"
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", settings_path)
    monkeypatch.setenv("STOCKREGRESSION_DATA_DIR", str(tmp_path / "configured-data"))

    loaded = load_settings()
    saved = save_settings({"schedule": {"enabled": True}})
    reloaded = load_settings()

    assert loaded.data_dir == str((tmp_path / "configured-data").resolve())
    assert saved.schedule.enabled is True
    assert saved.schedule.daily_time == "18:30"
    assert reloaded.schedule.enabled is True
    assert settings_path.is_file()
