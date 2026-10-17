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
from stockregression.providers import normalize_actions
from stockregression.settings import load_settings, save_settings
from stockregression.storage import Store
from stockregression.rules import instrument_terms


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


def _daily_only_snapshot(store: Store, symbol="600000.SH"):
    _, daily, _, calendar = _market(4)
    daily["preclose"] = daily.close.shift(1).fillna(daily.close.iloc[0])
    metadata = instrument_terms(symbol, {"actions_verified": True, "hf_revision": load_settings().hf_revision})
    manifest = store.publish(
        symbol,
        pd.DataFrame(),
        daily,
        normalize_actions(pd.DataFrame()),
        pd.DataFrame({"date": calendar}),
        metadata,
        {},
    )
    config = BacktestConfig(symbol=symbol, start_date=calendar[0].date(), end_date=calendar[-1].date(), frequency="1m")
    return config, manifest


def test_ensure_bundle_retries_degraded_snapshot_after_source_failure_backoff(tmp_path, monkeypatch):
    import stockregression.data as data_module

    store = Store(tmp_path)
    config, manifest = _daily_only_snapshot(store)
    calls = []

    def fail_sync(self, symbols, mode, *args, **kwargs):
        calls.append(mode)
        return {
            "status": "completed",
            "symbols": [{"symbol": config.symbol, "version": manifest["version"], "source_errors": ["近期分钟源不可用：mock network timeout"]}],
            "failures": [],
        }

    monkeypatch.setattr(DataService, "sync", fail_sync)
    service = DataService(store)
    first = service.ensure_bundle(config)
    assert first.version == manifest["version"]
    assert first.quality["daily_days"] == len(first.quality["days"])
    assert not any("mock network timeout" in warning for warning in first.warnings)
    assert any("mock network timeout" in event["message"] for event in first.metadata["source_events"])
    cached = DataService(store).ensure_bundle(config)
    assert calls == ["repair"]
    assert any("mock network timeout" in event["message"] for event in cached.metadata["source_events"])

    original_time = data_module.time.time
    monkeypatch.setattr(data_module.time, "time", lambda: original_time() + 61)
    DataService(store).ensure_bundle(config)
    assert calls == ["repair", "repair"]


def test_ensure_bundle_retries_missing_snapshot_after_source_failure_backoff(tmp_path, monkeypatch):
    import stockregression.data as data_module

    store = Store(tmp_path)
    config = BacktestConfig(symbol="600000.SH", start_date=date(2024, 1, 2), end_date=date(2024, 1, 5), frequency="1m")
    calls = []

    def fail_sync(self, symbols, mode, *args, **kwargs):
        calls.append(mode)
        return {"status": "failed", "symbols": [], "failures": [{"symbol": config.symbol, "reason": "mock history endpoint unavailable"}]}

    monkeypatch.setattr(DataService, "sync", fail_sync)
    with pytest.raises(DataQualityError, match="mock history endpoint unavailable"):
        DataService(store).ensure_bundle(config)
    with pytest.raises(DataQualityError, match="mock history endpoint unavailable"):
        DataService(store).ensure_bundle(config)
    assert calls == ["history"]

    original_time = data_module.time.time
    monkeypatch.setattr(data_module.time, "time", lambda: original_time() + 61)
    with pytest.raises(DataQualityError, match="mock history endpoint unavailable"):
        DataService(store).ensure_bundle(config)
    assert calls == ["history", "history"]


def test_submission_readiness_probe_is_cached_for_same_immutable_request(tmp_path, monkeypatch):
    store = Store(tmp_path)
    config, manifest = _daily_only_snapshot(store)
    # Use a one-day configuration with a daily-only frequency so no minute
    # download is required and the local snapshot is immediately ready.
    config = config.model_copy(update={"frequency": "1d"})
    service = DataService(store)
    calls = 0
    original = service._probe_bundle

    def count_probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_probe_bundle", count_probe)
    assert service.current_version_if_ready(config) == manifest["version"]
    assert service.current_version_if_ready(config) == manifest["version"]
    assert calls == 1


def test_preparation_lock_wait_checks_cancellation_in_short_intervals(tmp_path):
    import threading
    import time
    from filelock import FileLock

    path = str(tmp_path / "prepare-600000.SH.lock")
    acquired, release = threading.Event(), threading.Event()

    def hold_lock():
        with FileLock(path):
            acquired.set()
            release.wait(5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert acquired.wait(2)
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3

    started = time.monotonic()
    try:
        with pytest.raises(InterruptedError, match="任务已取消"):
            DataService._acquire_prepare_lock(FileLock(path), cancelled, timeout=20)
        assert time.monotonic() - started < 1
    finally:
        release.set()
        holder.join(2)


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


def test_load_bundle_does_not_warn_for_weekend_after_latest_closed_session(tmp_path, monkeypatch):
    store = Store(tmp_path / "weekend-request")
    config, manifest = _daily_only_snapshot(store)
    old = store.snapshot(config.symbol, manifest["version"])
    manifest = store.publish(
        config.symbol,
        store.read_files(old["files"]["minute"]),
        store.read_files(old["files"]["daily"]),
        store.read_files(old["files"]["actions"]),
        store.read_files(old["files"]["calendar"]),
        {
            **old["metadata"],
            "requested_end_date": "2024-01-06",
            "warnings": ["请求截至 2024-01-06，实际本地已收盘数据截至 2024-01-05"],
        },
        old["profiles"],
    )
    config = config.model_copy(update={
        "schema_version": 2,
        "frequency": "1d",
        "start_date": date(2024, 1, 3),
        "end_date": date(2024, 1, 6),  # Saturday; the fixture ends Friday.
    })
    monkeypatch.setattr("stockregression.data.latest_closed_date", lambda: pd.Timestamp("2024-01-05"))

    bundle = DataService(store).load_bundle(config, version=manifest["version"])

    assert bundle.bars.timestamp.dt.normalize().max() == pd.Timestamp("2024-01-05")
    assert not any("请求截至 2024-01-06" in warning for warning in bundle.warnings)


@pytest.mark.parametrize("frequency", ["1m", "5m"])
@pytest.mark.parametrize("verified", [True, False])
def test_market_bars_excludes_only_verified_sz_closing_auction_slots(tmp_path, frequency, verified):
    minute, daily, _, calendar = _market(days=1)
    minute["source_verified"] = verified
    closing_slots = minute_slots(calendar[0])[[237, 238]]
    minute = minute[~minute.timestamp.isin(closing_slots)].copy()
    symbol = "000001.SZ"
    store = Store(tmp_path / frequency)
    manifest = store.publish(
        symbol,
        minute,
        daily,
        normalize_actions(pd.DataFrame()),
        pd.DataFrame({"date": calendar}),
        instrument_terms(symbol, {"actions_verified": True}),
        {"hf": {"verified": verified, "source_kind": "provider"}},
    )

    result = DataService(store).market_bars(symbol, frequency=frequency, limit=300)

    assert result["version"] == manifest["version"]
    incomplete = any("K 线内缺少分钟" in warning or "此前分钟有缺口" in warning for warning in result["warnings"])
    assert incomplete is not verified
    if verified:
        assert all(bar["missing_minutes"] == 0 for bar in result["bars"])


def test_market_bars_keeps_other_verified_minute_gaps_visible(tmp_path):
    minute, daily, _, calendar = _market(days=1)
    minute["source_verified"] = True
    # 14:56 is an ordinary continuous-session bar, not an auction placeholder.
    minute = minute[~minute.timestamp.isin([minute_slots(calendar[0])[235], minute_slots(calendar[0])[237], minute_slots(calendar[0])[238]])].copy()
    symbol = "000001.SZ"
    store = Store(tmp_path / "real-gap")
    store.publish(
        symbol, minute, daily, normalize_actions(pd.DataFrame()), pd.DataFrame({"date": calendar}),
        instrument_terms(symbol, {"actions_verified": True}),
        {"hf": {"verified": True, "source_kind": "provider"}},
    )

    result = DataService(store).market_bars(symbol, frequency="5m", limit=300)

    assert any("K 线内缺少分钟" in warning for warning in result["warnings"])
    assert max(bar["missing_minutes"] for bar in result["bars"]) == 1


def test_market_bars_rebuilds_legacy_status_without_coverage_page(tmp_path):
    from stockregression.quality import QUALITY_RULES_VERSION

    minute, daily, _, calendar = _market(days=1)
    minute["volume"] *= 100
    minute["turnover"] *= 100
    minute["source_verified"] = True
    symbol = "600000.SH"
    store = Store(tmp_path / "first-chart")
    manifest = store.publish(
        symbol,
        minute,
        daily,
        normalize_actions(pd.DataFrame()),
        pd.DataFrame({"date": calendar}),
        instrument_terms(symbol, {"actions_verified": True, "quality_summary": {"minute_days": 0, "daily_days": 1}}),
        {"hf": {"verified": True, "source_kind": "provider"}},
    )
    # A legacy cache with a valid-looking diagnostics shape must not be reused.
    store.state("coverage:" + manifest["version"], {
        "diagnostics": {"schema_version": 1, "status": "unavailable"},
        "source_events": [],
        "quality_rules_version": 1,
        "sentinel": "stale",
    })

    result = DataService(store).market_bars(symbol, frequency="1m", limit=300)
    projection = store.state(f"coverage:{manifest['version']}:q{QUALITY_RULES_VERSION}")

    assert result["diagnostics"]["status"] == "complete"
    assert result["diagnostics"]["minute_days"] == 1
    assert result["diagnostics"]["quality_rules_version"] == QUALITY_RULES_VERSION
    assert projection["diagnostics"]["status"] == "complete"
    assert projection["diagnostics"]["minute_through"].startswith(str(calendar[-1].date()))
    assert projection["diagnostics"]["daily_through"] == str(calendar[-1].date())
    assert projection.get("sentinel") is None


def test_load_bundle_projects_fresh_watermarks_before_diagnostics(tmp_path, monkeypatch):
    import stockregression.data as module

    raw, daily, _, calendar = _market(days=4)
    raw["volume"] *= 100
    raw["turnover"] *= 100
    raw["source_verified"] = True
    symbol = "600000.SH"
    metadata = instrument_terms(symbol, {
        "actions_verified": True,
        "daily_through": "2000-01-01",
        "minute_through": "2000-01-01",
        "diagnostics": {"quality_rules_version": 1, "status": "unavailable"},
    })
    store = Store(tmp_path / "watermarks")
    manifest = store.publish(
        symbol, raw, daily, normalize_actions(pd.DataFrame()), pd.DataFrame({"date": calendar}), metadata,
        {"hf": {"verified": True, "source_kind": "provider"}},
    )
    config = BacktestConfig(
        schema_version=2,
        symbol=symbol,
        start_date=calendar[1].date(),
        end_date=calendar[-1].date(),
        frequency="1m",
    )
    monkeypatch.setattr(module, "latest_closed_date", lambda: calendar[-1])
    observed = {}
    original = module.project_data_diagnostics

    def inspect(metadata, *args, **kwargs):
        observed["daily_through"] = metadata.get("daily_through")
        observed["minute_through"] = metadata.get("minute_through")
        return original(metadata, *args, **kwargs)

    monkeypatch.setattr(module, "project_data_diagnostics", inspect)
    DataService(store).load_bundle(config, version=manifest["version"])

    assert observed["daily_through"] == str(daily.date.max().date())
    assert observed["minute_through"].startswith(str(calendar[-1].date()))


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


def test_auto_import_reuses_cached_independent_reference_without_sync(tmp_path, monkeypatch):
    raw, daily, reference, calendar = _volatile_market()
    store = Store(tmp_path / "store")
    store.publish("600000.SH", pd.DataFrame(), daily, normalize_actions(pd.DataFrame()),
                  pd.DataFrame({"date": calendar}), instrument_terms("600000.SH", {"actions_verified": True}), {})
    cached = _aggregate_five_minute_reference(reference, calendar)
    reference_path = store.root / "reference.parquet"
    cached.to_parquet(reference_path)
    store.state("import-reference:600000.SH", {"path": "reference.parquet", "frequency": "5m", "source": "independent fixture"})
    minute_path = tmp_path / "minute.parquet"
    raw.to_parquet(minute_path)
    monkeypatch.setattr("stockregression.data.fetch_auxiliary", lambda *args: pytest.fail("complete local data must not sync"))
    monkeypatch.setattr("baostock.login", lambda: pytest.fail("cached calendar must not sync"))
    result = DataService(store).import_files({"symbol": "600000.SH", "minute_path": str(minute_path)})
    profile = result["profiles"]["import"]
    assert profile["verified"] and profile["convention"] == "end"
    assert profile["reference_frequency"] == "5m"
    assert profile["reference_source"] == "independent fixture"
    active = store.snapshot("600000.SH")
    assert result["diagnostics"]["status"] == "complete"
    assert active["metadata"]["minute_status"] == "available"
    assert active["metadata"]["minute_through"].startswith(str(calendar[-1].date()))
    assert active["metadata"]["quality_summary"]["minute_days"] == len(calendar)
    assert active["metadata"]["quality_summary"]["daily_days"] == 0
    assert len(active["metadata"]["diagnostics"]["source_spans"]) == 1
    assert active["metadata"]["diagnostics"]["source_spans"][0]["source"] == "hf"
    assert active["metadata"]["diagnostics"]["source_spans"][0]["verified"] is True
