from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stockregression.market_data.consumer import load_native_bundle
from stockregression.models import BacktestConfig
from stockregression.market_data.models import DataNotReady
from stockregression.quality import minute_slots


DAY = pd.Timestamp("2024-03-18")


class _Store:
    def __init__(self, manifests):
        self.manifests = manifests
        self.active = next(reversed(manifests.values()))

    def snapshot(self, symbol, version=None):
        if version is None:
            return self.active
        return next((item for item in self.manifests.values() if item["version"] == version), None)


class _Repository:
    def __init__(self, minute_by_version, daily_by_version):
        self.minute_by_version = minute_by_version
        self.daily_by_version = daily_by_version
        self.bar_versions = []

    def version_info(self, symbol, timeframe, version):
        return None

    def bars(self, symbol, timeframe, *, version, start=None, end=None):
        self.bar_versions.append((timeframe, version))
        source = self.daily_by_version if timeframe == "1day" else self.minute_by_version
        frame = source[version].copy()
        if start is not None:
            frame = frame[frame.timestamp >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.timestamp < pd.Timestamp(end)]
        return frame.reset_index(drop=True)

    def auxiliary(self, symbol, name, *, version=None):
        if name == "calendar":
            return {"payload": [{"date": str(DAY.date())}]}
        if name == "actions":
            return {"payload": [], "verified": True}
        return None


def _frames(symbol="159611.SZ", *, missing=(), minute_volume=100.0, snapshot="snapshot-current", market_version=8):
    slots = minute_slots(DAY)
    close = 1.2 + np.arange(240) * 0.0001
    opening = close - 0.0001
    minute = pd.DataFrame({
        "timestamp": slots, "open": opening, "high": close + 0.0001,
        "low": opening - 0.0001, "close": close,
        "volume": np.full(240, minute_volume),
        "turnover": np.full(240, minute_volume * 1.2),
        "source": "import", "source_verified": True,
        "is_synthetic": False, "is_auction_placeholder": False,
        "frequency": "1min",
    }).drop(index=list(missing)).reset_index(drop=True)
    daily = pd.DataFrame([{
        "timestamp": DAY, "open": opening[0], "high": close.max() + 0.0001,
        "low": opening.min() - 0.0001, "close": close[-1],
        "preclose": 1.2, "volume": 240.0, "turnover": 288.0,
        "tradestatus": 1, "is_st": False, "source_verified": True,
        "is_synthetic": False,
    }])
    manifest = {
        "version": snapshot, "symbol": symbol, "market_data": True,
        "metadata": {"asset_type": "etf", "price_tick": 0.001},
        "market_versions": {
            "1day": {"version": 4}, "1min": {"version": market_version},
        },
        "auxiliary_versions": {"calendar": "calendar-1", "actions": "actions-1"},
        "profiles": {},
    }
    return manifest, minute, daily


def _service(manifests_and_frames):
    manifests = {}
    minute_by_version = {}
    daily_by_version = {}
    for manifest, minute, daily in manifests_and_frames:
        manifests[manifest["version"]] = manifest
        minute_by_version[manifest["market_versions"]["1min"]["version"]] = minute
        daily_by_version[manifest["market_versions"]["1day"]["version"]] = daily
    return SimpleNamespace(
        store=_Store(manifests),
        repository=_Repository(minute_by_version, daily_by_version),
    )


def _config(symbol="159611.SZ"):
    return BacktestConfig(
        schema_version=3, symbol=symbol, start_date=DAY.date(),
        end_date=DAY.date(), frequency="1m",
    )


def test_default_bundle_keeps_minutes_for_volume_mismatch():
    manifest, minute, daily = _frames(minute_volume=100.0)
    service = _service([(manifest, minute, daily)])

    bundle = load_native_bundle(service, _config())

    assert bundle.quality["minute_days"] == 1
    assert bundle.quality["daily_days"] == 0
    assert len(bundle.bars) == 240
    assert set(bundle.bars.frequency) == {"1m"}


def test_preflight_summary_reads_daily_but_not_minute_bodies():
    manifest, minute, daily = _frames()
    service = _service([(manifest, minute, daily)])

    bundle = load_native_bundle(service, _config(), summary_only=True)

    assert bundle.bars.empty
    assert len(bundle.daily) == 1
    assert list(bundle.calendar) == [DAY]
    assert service.repository.bar_versions == [("1day", 4)]
    assert bundle.metadata["preflight_summary"] is True


def test_statistics_bundle_preserves_raw_prices_and_reports_volume_mismatch():
    manifest, minute, daily = _frames(minute_volume=100.0)
    manifest["metadata"]["etf_replay_blocked_reason"] = "旧版本因成交量差异阻断回放"
    service = _service([(manifest, minute, daily)])

    bundle = load_native_bundle(service, _config(), statistics_only=True)

    assert len(bundle.bars) == 240
    assert set(bundle.bars.frequency) == {"1m"}
    assert bundle.bars.source_verified.all()
    assert not bundle.bars.is_synthetic.any()
    assert bundle.metadata["etf_volume_verified"] is False
    assert bundle.metadata["volume_unverified_days"] == [str(DAY.date())]
    assert "etf_replay_blocked_reason" not in bundle.metadata
    assert any("按分钟成交量继续" in warning for warning in bundle.warnings)
    assert bundle.bars.volume.sum() == pytest.approx(minute.volume.sum())


def test_statistics_bundle_leaves_ordinary_small_gap_missing():
    manifest, minute, daily = _frames(missing=(20,), minute_volume=1.0)
    service = _service([(manifest, minute, daily)])

    bundle = load_native_bundle(service, _config(), statistics_only=True)

    assert len(bundle.bars) == 239
    assert minute_slots(DAY)[20] not in set(bundle.bars.timestamp)
    assert not bundle.bars.is_synthetic.any()


def test_statistics_bundle_only_fills_rule_validated_sz_closing_auction_slots():
    sz_manifest, sz_minute, sz_daily = _frames("159611.SZ", missing=(237, 238), minute_volume=1.0)
    sz = load_native_bundle(_service([(sz_manifest, sz_minute, sz_daily)]), _config(), statistics_only=True)
    filled = sz.bars[sz.bars.is_auction_placeholder]
    assert len(sz.bars) == 240
    assert set(filled.timestamp) == {minute_slots(DAY)[237], minute_slots(DAY)[238]}
    assert filled.volume.eq(0).all() and filled.turnover.eq(0).all()
    assert filled.source_verified.all() and not filled.is_synthetic.any()

    sh_manifest, sh_minute, sh_daily = _frames("510210.SH", missing=(237, 238), minute_volume=1.0)
    sh = load_native_bundle(_service([(sh_manifest, sh_minute, sh_daily)]), _config("510210.SH"), statistics_only=True)
    assert len(sh.bars) == 238
    assert not sh.bars.is_auction_placeholder.any()


def test_statistics_bundle_reads_the_requested_frozen_market_version():
    old_manifest, old_minute, old_daily = _frames(snapshot="snapshot-old", market_version=7, minute_volume=100.0)
    current_manifest, current_minute, current_daily = _frames(snapshot="snapshot-current", market_version=8, minute_volume=2.0)
    service = _service([
        (old_manifest, old_minute, old_daily),
        (current_manifest, current_minute, current_daily),
    ])

    bundle = load_native_bundle(service, _config(), version="snapshot-old", statistics_only=True)

    assert bundle.version == "snapshot-old"
    assert ("1min", 7) in service.repository.bar_versions
    assert ("1min", 8) not in service.repository.bar_versions
    assert bundle.bars.volume.sum() == 240 * 100


def test_statistics_bundle_keeps_a_whole_missing_session_on_the_calendar():
    manifest, minute, daily = _frames(minute_volume=1.)
    second_day = DAY + pd.offsets.BDay(1)
    extra = daily.copy()
    extra["timestamp"] = second_day
    extra["preclose"] = daily.iloc[-1].close
    service = _service([(manifest, minute, pd.concat([daily, extra], ignore_index=True))])
    config = _config().model_copy(update={"end_date": second_day.date()})
    bundle = load_native_bundle(service, config, statistics_only=True)
    assert len(bundle.calendar) == 2 and len(bundle.bars) == 240
    assert not bundle.bars.timestamp.dt.normalize().eq(second_day).any()
    assert bundle.metadata["etf_volume_verified"] is True


def test_statistics_bundle_still_rejects_invalid_observed_prices():
    manifest, minute, daily = _frames(minute_volume=1.)
    minute.loc[10, "close"] = 0
    with pytest.raises(DataNotReady, match="无效价格"):
        load_native_bundle(_service([(manifest, minute, daily)]), _config(), statistics_only=True)


def test_statistics_bundle_auction_rows_do_not_hide_the_observed_minute_frequency():
    from stockregression.etf_statistics import compute_etf_statistics

    manifest, minute, daily = _frames(missing=(237, 238), minute_volume=1.)
    minute = minute.drop(columns="frequency")
    bundle = load_native_bundle(_service([(manifest, minute, daily)]), _config(), statistics_only=True)
    result = compute_etf_statistics(bundle, DAY, 1)
    assert set(bundle.bars.frequency) == {"1m"}
    assert result["status"] == "completed", result["warnings"]
    assert result["diagnostics"]["minute_integrity"]["accepted_slots"] == 240
