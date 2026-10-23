from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.consumer import load_native_bundle
from stockregression.market_data.repository import VALIDATION_RULES_VERSION
from stockregression.models import BacktestConfig
from stockregression.storage import Store


SYMBOL = "510050.SH"


def _daily_frame(day, close):
    return pd.DataFrame([{
        "timestamp": pd.Timestamp(day) + pd.Timedelta(hours=15),
        "open": close, "high": close, "low": close, "close": close,
        "preclose": close, "volume": 245.0, "turnover": 245.0,
        "tradestatus": 1, "is_st": False, "source": "fixture",
        "source_verified": True, "is_synthetic": False,
    }])


def _minute_frame(day, close):
    slots = pd.date_range(pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min")
    slots = slots.append(pd.date_range(pd.Timestamp(day) + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min"))
    return pd.DataFrame({
        "timestamp": slots, "open": close, "high": close, "low": close,
        "close": close, "preclose": close, "volume": 1.0, "turnover": close,
        "tradestatus": 1, "is_st": False, "source": "fixture",
        "source_verified": True, "is_synthetic": False,
    })


def _dataset(service, day, close, *, mode="merge", report=None):
    service.repository.commit_bars(
        SYMBOL, "1day", _daily_frame(day, close), source="fixture",
        report=report or {}, import_mode=mode,
    )
    service.repository.commit_bars(
        SYMBOL, "1min", _minute_frame(day, close), source="fixture",
        report=report or {}, import_mode=mode,
    )


def _config(day):
    return BacktestConfig(
        schema_version=3, symbol=SYMBOL, start_date=pd.Timestamp(day).date(),
        end_date=pd.Timestamp(day).date(), frequency="1m",
    )


def _with_quality_evidence(manifest, start, end):
    frozen = copy.deepcopy(manifest)
    versions = {
        timeframe: int(value["version"])
        for timeframe, value in frozen["market_versions"].items()
    }
    intervals = {
        timeframe: [{
            "start": pd.Timestamp(start).date().isoformat(),
            "end": pd.Timestamp(end).date().isoformat(),
            "status": "complete",
        }]
        for timeframe in versions
    }
    frozen["quality_evidence"] = {
        "schema_version": 1,
        "rules_version": VALIDATION_RULES_VERSION,
        "market_versions": versions,
        "auxiliary_versions": copy.deepcopy(frozen.get("auxiliary_versions", {})),
        "intervals": intervals,
    }
    frozen["version"] = "mkt-frozen-" + hashlib.sha256(
        json.dumps(frozen, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    return frozen


def test_validation_state_does_not_apply_current_ledger_to_old_version(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    old_day = pd.Timestamp("2026-01-05")
    _dataset(service, old_day, 1.0, report={"status": "pending_validation"})
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": old_day}], version="calendar-old", verified=True,
    )
    service.repository.save_auxiliary(
        SYMBOL, "actions", [], version="actions-old", verified=True,
    )
    old = service.store.snapshot(SYMBOL)
    old_version = int(old["market_versions"]["1min"]["version"])

    new_day = pd.Timestamp("2026-01-06")
    _dataset(service, old_day, 1.0)
    _dataset(service, new_day, 2.0)
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": old_day}, {"date": new_day}],
        version="calendar-new", verified=True,
    )
    service.repository.record_day_validations([
        {
            "symbol": SYMBOL, "timeframe": "1min", "adjustment": "none",
            "trade_date": "2026-01-05", "status": "passed", "row_count": 240,
            "rules_version": VALIDATION_RULES_VERSION, "report": {"status": "complete"},
        },
        {
            "symbol": SYMBOL, "timeframe": "1min", "adjustment": "none",
            "trade_date": "2026-01-06", "status": "passed", "row_count": 240,
            "rules_version": VALIDATION_RULES_VERSION, "report": {"status": "complete"},
        },
    ])
    current_version = int(service.repository.dataset(SYMBOL, "1min")["current_version"])
    current_daily_version = int(service.repository.dataset(SYMBOL, "1day")["current_version"])
    historical_daily_version = int(old["market_versions"]["1day"]["version"])

    original_read = service.repository._read_version
    reads = []

    def timestamp_only(*args, **kwargs):
        reads.append((args[1], kwargs.get("fields")))
        assert args[1] == "1day"
        assert kwargs.get("fields") == ("timestamp",)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(service.repository, "_read_version", timestamp_only)
    current = service.repository.validation_state(
        SYMBOL, "1min", start=old_day, end=new_day,
        version=current_version,
        calendar_version="calendar-new",
        daily_version=current_daily_version,
    )
    historical = service.repository.validation_state(
        SYMBOL, "1min", start=old_day, end=old_day,
        version=old_version, calendar_version="calendar-old",
        daily_version=historical_daily_version,
    )

    assert current["available"] is True
    assert current["status"] == "complete"
    assert current["quality_source"] == "ledger"
    assert historical["available"] is False
    assert historical["status"] == "unavailable"
    assert historical["verified_from"] is None
    assert historical["verified_through"] is None
    assert reads and all(fields == ("timestamp",) for _, fields in reads)


def test_short_locked_calendar_is_completed_from_the_versioned_daily_timestamps(
    tmp_path, monkeypatch,
):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.bdate_range("2026-05-01", periods=10)
    for day in days:
        _dataset(service, day, 1.0)
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": day} for day in days[-5:]],
        version="calendar-short", verified=True,
    )
    snapshot = service.store.snapshot(SYMBOL)
    minute_version = int(snapshot["market_versions"]["1min"]["version"])
    daily_version = int(snapshot["market_versions"]["1day"]["version"])

    original_read = service.repository._read_version
    reads = []

    def timestamp_only(*args, **kwargs):
        reads.append((args[1], kwargs.get("fields")))
        assert args[1] == "1day"
        assert kwargs.get("fields") == ("timestamp",)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(service.repository, "_read_version", timestamp_only)
    combined = service.repository.trading_days(
        SYMBOL, days[0], days[-1], calendar_version="calendar-short",
        daily_version=daily_version,
    )
    assert pd.DatetimeIndex(combined).equals(days)

    service.repository.record_day_validations([
        {
            "symbol": SYMBOL, "timeframe": "1min", "adjustment": "none",
            "trade_date": day.date().isoformat(), "status": "passed",
            "row_count": 240, "rules_version": VALIDATION_RULES_VERSION,
            "report": {"status": "complete"},
        }
        for day in days
    ])
    state = service.repository.validation_state(
        SYMBOL, "1min", start=days[0], end=days[-1],
        version=minute_version, calendar_version="calendar-short",
        daily_version=daily_version,
    )

    assert state["available"] is True
    assert state["status"] == "complete"
    assert state["daily_version"] == daily_version
    assert state["expected_start"] == days[0].date().isoformat()
    assert state["expected_end"] == days[-1].date().isoformat()
    assert state["validated_days"] == len(days)
    assert state["pending"] == []
    assert reads and all(fields == ("timestamp",) for _, fields in reads)

    without_daily = service.repository.validation_state(
        SYMBOL, "1min", start=days[0], end=days[-1],
        version=minute_version, calendar_version="calendar-short",
    )
    assert without_daily["available"] is False
    assert "没有锁定日线版本" in without_daily["reason"]

    # A locked calendar that is absent remains a hard failure; the daily
    # timestamps must never act as a substitute for the calendar contract.
    assert not len(service.repository.trading_days(
        SYMBOL, days[0], days[-1], calendar_version="calendar-missing",
        daily_version=daily_version,
    ))
    missing = service.repository.validation_state(
        SYMBOL, "1min", start=days[0], end=days[-1], version=minute_version,
        calendar_version="calendar-missing", daily_version=daily_version,
    )
    assert missing["available"] is False


def test_readiness_and_consumer_use_the_snapshot_calendar_and_versions(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    old_day = pd.Timestamp("2026-02-02")
    new_day = pd.Timestamp("2026-02-03")
    _dataset(service, old_day, 1.0, report={"status": "pending_validation"})
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": old_day}], version="calendar-old", verified=True,
    )
    service.repository.save_auxiliary(
        SYMBOL, "actions", [], version="actions-old", verified=True,
    )
    old = service.store.snapshot(SYMBOL)

    # Replace the current view with a different session so using even one live
    # range or auxiliary field would produce an observable mismatch.
    _dataset(service, new_day, 2.0, mode="snapshot")
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": new_day}], version="calendar-new", verified=True,
    )
    service.repository.save_auxiliary(
        SYMBOL, "actions", [{"ex_date": new_day, "cash_per_share": 0.5}],
        version="actions-new", verified=True,
    )

    readiness = service.readiness(_config(old_day), old["version"])
    bundle = load_native_bundle(service, _config(old_day), old["version"])

    assert readiness["snapshot_version"] == old["version"]
    assert readiness["calendar_version"] == "calendar-old"
    assert not any(
        gap.get("kind") == "range" and gap.get("timeframe") in {"1day", "1min"}
        for gap in readiness["gaps"]
    )
    assert readiness["ready"] is False
    assert any(gap.get("kind") == "quality" for gap in readiness["gaps"])
    assert pd.DatetimeIndex(bundle.calendar).equals(pd.DatetimeIndex([old_day]))
    assert bundle.actions.empty
    assert float(bundle.daily.close.iloc[-1]) == pytest.approx(1.0)


def test_versioned_validation_rejects_a_different_locked_calendar(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    day = pd.Timestamp("2026-03-02")
    _dataset(service, day, 1.0)
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": day}], version="calendar-a", verified=True,
    )
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": day}], version="calendar-b", verified=True,
    )
    service.repository.record_day_validations([{
        "symbol": SYMBOL, "timeframe": "1min", "adjustment": "none",
        "trade_date": "2026-03-02", "status": "passed", "row_count": 240,
        "rules_version": VALIDATION_RULES_VERSION, "report": {"status": "complete"},
    }])
    version = int(service.repository.dataset(SYMBOL, "1min")["current_version"])
    daily_version = int(service.repository.dataset(SYMBOL, "1day")["current_version"])

    state = service.repository.validation_state(
        SYMBOL, "1min", start=day, end=day, version=version,
        calendar_version="calendar-a", daily_version=daily_version,
    )

    assert state["available"] is False
    assert state["status"] == "unavailable"
    assert state["verified_through"] is None


def test_complete_historical_report_is_the_only_fallback_for_an_old_version(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    old_day = pd.Timestamp("2026-04-01")
    new_day = pd.Timestamp("2026-04-02")
    _dataset(service, old_day, 1.0, report={"status": "complete", "coverage_status": "complete"})
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": old_day}], version="calendar-old", verified=True,
    )
    old = service.store.snapshot(SYMBOL)
    old_version = int(old["market_versions"]["1min"]["version"])
    old_daily_version = int(old["market_versions"]["1day"]["version"])

    _dataset(service, new_day, 2.0, mode="snapshot")
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": new_day}], version="calendar-new", verified=True,
    )

    without_daily = service.repository.validation_state(
        SYMBOL, "1min", start=old_day, end=old_day,
        version=old_version, calendar_version="calendar-old",
    )
    without_calendar = service.repository.validation_state(
        SYMBOL, "1min", start=old_day, end=old_day,
        version=old_version, daily_version=old_daily_version,
    )
    state = service.repository.validation_state(
        SYMBOL, "1min", start=old_day, end=old_day,
        version=old_version, calendar_version="calendar-old",
        daily_version=old_daily_version,
    )

    assert without_daily["available"] is True
    assert without_daily["status"] == "complete"
    assert without_daily["quality_source"] == "version_report"
    assert without_daily["daily_version"] is None
    assert without_calendar["available"] is False
    assert without_calendar["status"] == "unavailable"
    assert state["available"] is True
    assert state["status"] == "complete"
    assert state["quality_source"] == "version_report"
    assert state["verified_from"] == old_day.date().isoformat()
    assert state["verified_through"] == old_day.date().isoformat()
    assert state["daily_version"] == old_daily_version


def test_frozen_quality_evidence_survives_later_data_and_failed_live_ledger(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    prior = pd.Timestamp("2026-06-01")
    start = pd.Timestamp("2026-06-02")
    _dataset(service, prior, 1.0, report={"status": "pending_validation"})
    _dataset(service, start, 1.0, report={"status": "pending_validation"})
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": prior}, {"date": start}],
        version="calendar-old", verified=True,
    )
    service.repository.save_auxiliary(
        SYMBOL, "actions", [], version="actions-old", verified=True,
    )
    source = service.store.snapshot(SYMBOL)
    frozen = _with_quality_evidence(source, start, start)
    service.store.remember_market_snapshots([frozen])

    # Later synchronization changes every live dependency and deliberately
    # leaves failed day checks in the current ledger.
    later = pd.Timestamp("2026-06-03")
    _dataset(service, later, 1.2)
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": prior}, {"date": start}, {"date": later}],
        version="calendar-new", verified=True,
    )
    service.repository.save_auxiliary(
        SYMBOL, "actions", [{"ex_date": later, "cash_per_share": 0.2}],
        version="actions-new", verified=True,
    )
    service.repository.record_day_validations([
        {
            "symbol": SYMBOL, "timeframe": timeframe, "adjustment": "none",
            "trade_date": start.date().isoformat(), "status": "failed",
            "row_count": 240 if timeframe == "1min" else 1,
            "rules_version": VALIDATION_RULES_VERSION,
            "report": {"status": "failed", "reason": "current sync changed"},
        }
        for timeframe in ("1day", "1min")
    ])

    readiness = service.readiness(_config(start), frozen["version"])
    bundle = load_native_bundle(service, _config(start), frozen["version"])

    assert readiness["ready"] is True, readiness["gaps"]
    assert readiness["calendar_version"] == "calendar-old"
    assert bundle.actions.empty
    assert pd.Timestamp(bundle.daily.date.max()).normalize() == start
    assert float(bundle.daily.loc[bundle.daily.date == start, "close"].iloc[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("tamper", ["version", "auxiliary", "rules", "interval"])
def test_frozen_quality_evidence_identity_tampering_is_rejected(tmp_path, tamper):
    service = MarketDataService(Store(tmp_path / "data"))
    day = pd.Timestamp("2026-07-01")
    _dataset(service, day, 1.0, report={"status": "pending_validation"})
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": day}], version="calendar-old", verified=True,
    )
    service.repository.save_auxiliary(
        SYMBOL, "actions", [], version="actions-old", verified=True,
    )
    source = service.store.snapshot(SYMBOL)
    frozen = _with_quality_evidence(source, day, day)
    version = int(frozen["market_versions"]["1min"]["version"])
    daily_version = int(frozen["market_versions"]["1day"]["version"])
    dataset = service.repository.version_info(SYMBOL, "1min", version)

    assert service._validation_covers(
        dataset, day, day, version=version,
        calendar_version="calendar-old", daily_version=daily_version,
        manifest=frozen,
    )
    assert service._degradable_minute_days(
        dataset, day, day, version=version,
        calendar_version="calendar-old", manifest=frozen,
    ) == []

    tampered = copy.deepcopy(frozen)
    evidence = tampered["quality_evidence"]
    if tamper == "version":
        tampered["market_versions"]["1min"]["version"] = version + 1
    elif tamper == "auxiliary":
        evidence["auxiliary_versions"]["actions"] = "actions-other"
    elif tamper == "rules":
        evidence["rules_version"] = VALIDATION_RULES_VERSION - 1
    else:
        evidence["intervals"]["1min"][0]["start"] = (
            day + pd.Timedelta(days=1)
        ).date().isoformat()

    assert not service._validation_covers(
        dataset, day, day, version=version,
        calendar_version="calendar-old", daily_version=daily_version,
        manifest=tampered,
    )
    assert service._frozen_quality_state(
        tampered, dataset, day, day, version,
    ) is None
