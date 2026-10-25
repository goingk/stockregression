from __future__ import annotations

import json
import shutil
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.jobs import execute_job
from stockregression.market_data import MarketDataService
from stockregression.market_data.sync import MarketDataSyncService, _recent_minute_incomplete
from stockregression.market_data.aggregation import AggregationService
from stockregression.market_data.models import MarketDataError
from stockregression.market_data.quality import validate_and_fill
from stockregression.models import BacktestConfig
from stockregression.quality import minute_slots
from stockregression.storage import Store


def _days(count=3):
    return pd.bdate_range("2026-01-05", periods=count)


def _verified_minute_without_closing_placeholder_ticks(days):
    rows = []
    for day in days:
        slots = minute_slots(day)
        keep = [slot for index, slot in enumerate(slots) if index not in {237, 238}]
        for timestamp in keep:
            rows.append({
                "timestamp": timestamp, "open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0,
                "volume": 100, "turnover": 100.0, "source": "sina", "source_verified": True,
                "is_synthetic": False, "is_auction_placeholder": False,
            })
    return pd.DataFrame(rows)


def _minute_rows(day, slots):
    return pd.DataFrame({
        "timestamp": list(slots), "open": 1.0, "high": 1.01, "low": 0.99,
        "close": 1.0, "volume": 100, "turnover": 100.0,
    })


def test_recent_minute_incomplete_detects_a_late_started_session():
    day = pd.Timestamp("2026-10-08")
    slots = minute_slots(day)[10:]
    frame = _minute_rows(day, slots)

    reason = _recent_minute_incomplete(frame, "517520.SH", day, day + pd.Timedelta(days=1))

    assert reason is not None and "缺" in reason


def test_recent_minute_incomplete_accepts_a_complete_session():
    day = pd.Timestamp("2026-10-08")
    frame = _minute_rows(day, minute_slots(day))

    assert _recent_minute_incomplete(frame, "517520.SH", day, day + pd.Timedelta(days=1)) is None


def test_recent_minute_incomplete_accepts_sz_closing_auction_placeholders():
    day = pd.Timestamp("2026-10-08")
    slots = [slot for index, slot in enumerate(minute_slots(day)) if index not in {237, 238}]
    frame = _minute_rows(day, slots)

    assert _recent_minute_incomplete(frame, "159611.SZ", day, day + pd.Timedelta(days=1)) is None


def test_recent_minute_incomplete_detects_a_partial_incremental_tail():
    day = pd.Timestamp("2026-10-08")
    slots = [day + pd.Timedelta(hours=15)]
    frame = _minute_rows(day, slots)

    reason = _recent_minute_incomplete(
        frame, "517520.SH", day + pd.Timedelta(hours=14, minutes=58), day + pd.Timedelta(hours=15),
    )

    assert reason is not None and "缺 2 根" in reason


@pytest.mark.parametrize("symbol,expected_status,expected_missing", [
    ("159611.SZ", "complete", 0),
    ("510300.SH", "complete", 0),
])
def test_read_only_revalidation_keeps_verified_closing_auction_slots(symbol, expected_status, expected_missing):
    days = _days(2)
    frame = _verified_minute_without_closing_placeholder_ticks(days)
    validation = validate_and_fill(
        frame, "1min", calendar=pd.DataFrame({"date": days}), fill=False, symbol=symbol,
    )
    assert validation.report["missing_rows"] == expected_missing
    assert validation.report["status"] == expected_status
    assert validation.report["auction_placeholders"] == 4


def _daily(days):
    return pd.DataFrame({
        "date": days, "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.2,
        "volume": 24_000.0, "turnover": 240_000.0, "preclose": 10.0,
        "tradestatus": 1, "is_st": 0,
    })


def _minute(days, timeframe=1):
    size = max(1, 240 // (240 // timeframe))
    stamps = []
    for day in days:
        slots = minute_slots(day)
        stamps.extend(slots[(timeframe - 1)::timeframe] if timeframe > 1 else slots)
    bars = 240 // timeframe
    return pd.DataFrame({
        "timestamp": stamps, "open": 10.0, "high": 10.5, "low": 9.5,
        "close": 10.2, "volume": 100.0 * (size if timeframe > 1 else 1),
        "turnover": 1_000.0 * (timeframe if timeframe > 1 else 1),
    })


def _calendar(tmp_path, days):
    path = tmp_path / "calendar.csv"
    pd.DataFrame({"date": days}).to_csv(path, index=False)
    return path


def test_import_persists_in_sqlite_and_filename_duplicate_is_skipped(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days()
    calendar = _calendar(tmp_path, days)
    path = tmp_path / "600000_daily.csv"
    _daily(days).to_csv(path, index=False)

    first = service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    renamed = tmp_path / "SH600000_export.csv"
    shutil.copy2(path, renamed)
    second = service.import_file("600000.SH", "1day", renamed, calendar_path=calendar)
    path.unlink()
    renamed.unlink()

    assert first["inserted"] == 3 and first["version"] == 1
    assert second["status"] == "skipped"
    assert service.bars("600000.SH", "1day")["count"] == 3
    assert service.repository.dataset("600000.SH", "1day")["row_count"] == 3
    assert service.repository.dataset("600000.SH", "1min") is None


def test_coarse_minute_files_are_rejected_without_one_minute_base(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days()
    calendar = _calendar(tmp_path, days)
    daily_path, minute_path = tmp_path / "600000_daily.csv", tmp_path / "600000_15m.csv"
    _daily(days).to_csv(daily_path, index=False)
    _minute(days, 15).to_csv(minute_path, index=False)
    service.import_file("600000.SH", "1day", daily_path, calendar_path=calendar)
    with pytest.raises(MarketDataError, match="派生周期"):
        service.import_file("600000.SH", "15min", minute_path, calendar_path=calendar, label_convention="end")

    assert service.repository.dataset("600000.SH", "15min") is None
    assert service.repository.dataset("600000.SH", "1min") is None
    config = BacktestConfig(symbol="600000.SH", start_date=date(2026, 1, 5), end_date=date(2026, 1, 7), frequency="15m", schema_version=2)
    with pytest.raises(MarketDataError, match="1min"):
        service.load_backtest_bundle(config)


def test_conservative_minute_fill_and_real_gap_threshold(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex([pd.Timestamp("2026-01-05")])
    calendar = _calendar(tmp_path, days)
    daily_path, minute_path = tmp_path / "600000_daily.csv", tmp_path / "600000_1min.csv"
    _daily(days).to_csv(daily_path, index=False)
    frame = _minute(days)
    gaps = frame[frame.timestamp.dt.strftime("%H:%M").isin({"10:01", "10:02"})].index
    frame.drop(gaps, inplace=True)
    frame.to_csv(minute_path, index=False)
    service.import_file("600000.SH", "1day", daily_path, calendar_path=calendar)
    result = service.import_file("600000.SH", "1min", minute_path, calendar_path=calendar, label_convention="end")
    bars = service.repository.active_rows("600000.SH", "1min")

    assert result["synthetic"] == 2
    assert len(bars) == 240
    assert bars.is_synthetic.sum() == 2
    assert bars[bars.is_synthetic].volume.eq(0).all()


def test_end_labelled_opening_auction_is_folded_into_first_minute(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    day = pd.Timestamp("2026-01-05")
    calendar = _calendar(tmp_path, pd.DatetimeIndex([day]))
    daily_path, minute_path = tmp_path / "600000_daily.csv", tmp_path / "600000_1min.csv"
    _daily(pd.DatetimeIndex([day])).to_csv(daily_path, index=False)
    frame = _minute(pd.DatetimeIndex([day]))
    auction = {
        "timestamp": day + pd.Timedelta(hours=9, minutes=30), "open": 9.9, "high": 10.3,
        "low": 9.8, "close": 10.0, "volume": 500.0, "turnover": 5_000.0,
    }
    frame = pd.concat([pd.DataFrame([auction]), frame], ignore_index=True)
    frame.to_csv(minute_path, index=False)
    service.import_file("600000.SH", "1day", daily_path, calendar_path=calendar)
    service.import_file("600000.SH", "1min", minute_path, calendar_path=calendar, label_convention="end")
    bars = service.repository.active_rows("600000.SH", "1min")
    first = bars.iloc[0]

    assert len(bars) == 240
    assert first.timestamp == day + pd.Timedelta(hours=9, minutes=31)
    assert first.open == pytest.approx(9.9)
    assert first.high == pytest.approx(10.5)
    assert first.volume == pytest.approx(600)


def test_version_history_is_queryable_after_correction(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days(2)
    calendar = _calendar(tmp_path, days)
    path = tmp_path / "600000_daily.csv"
    _daily(days).to_csv(path, index=False)
    first = service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    revised = _daily(days)
    revised.loc[1, "close"] = 11.0
    revised.loc[1, "high"] = 11.5
    revised.to_csv(path, index=False)
    second = service.import_file("600000.SH", "1day", path, calendar_path=calendar)

    old = service.repository.bars("600000.SH", "1day", version=first["version"])
    current = service.repository.bars("600000.SH", "1day", version=second["version"])
    assert second["version"] == first["version"] + 1
    assert old.close.tolist() == [10.2, 10.2]
    assert current.close.tolist() == [10.2, 11.0]


def test_batch_folder_matches_errors_is_partial_and_resumable(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days()
    folder = tmp_path / "imports" / "1day"
    folder.mkdir(parents=True)
    _daily(days).to_csv(folder / "600000_daily.csv", index=False)
    _daily(days).to_csv(folder / "000001_daily.csv", index=False)
    _daily(days).to_csv(folder / "no-code.csv", index=False)
    preview = service.preview_folder(folder, "1day")
    assert preview["target_count"] == 2 and preview["invalid_count"] == 1

    first = service.import_folder(folder, "1day")
    second = service.import_folder(folder, "1day")
    assert first["status"] == "partial" and first["completed"] == 2 and first["failed"] == 1
    assert second["completed"] == 0 and second["skipped"] == 2


def test_explicit_daily_to_week_aggregation(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days(5)
    calendar = _calendar(tmp_path, days)
    path = tmp_path / "600000_daily.csv"
    _daily(days).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    result = AggregationService(service).aggregate("600000.SH", "1day", "1week")
    week = service.repository.active_rows("600000.SH", "1week")
    assert result["target_timeframe"] == "1week"
    assert len(week) == 1
    assert week.iloc[0].volume == 120_000
    assert service.repository.dataset("600000.SH", "30min") is None


@pytest.mark.parametrize("timeframe", ["1week", "1month"])
def test_period_files_must_be_generated_from_daily_base(tmp_path, timeframe):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days(5)
    calendar = _calendar(tmp_path, days)
    frame = pd.DataFrame({
        "date": [days[-1]], "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.2,
        "volume": 120_000.0, "turnover": 1_200_000.0,
    })
    path = tmp_path / "600000_period.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(MarketDataError, match="派生周期"):
        service.import_file("600000.SH", timeframe, path, calendar_path=calendar)
    assert service.repository.dataset("600000.SH", timeframe) is None
    assert service.repository.dataset("600000.SH", "1day") is None


def test_local_label_revalidation_does_not_require_source_file(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex([pd.Timestamp("2026-01-05")])
    calendar = _calendar(tmp_path, days)
    daily_path, minute_path = tmp_path / "600000_daily.csv", tmp_path / "600000_1min.csv"
    _daily(days).to_csv(daily_path, index=False)
    # Start-labelled input, deliberately imported with auto semantics.
    frame = _minute(days)
    frame["timestamp"] = frame.timestamp - pd.Timedelta(minutes=1)
    frame.to_csv(minute_path, index=False)
    service.import_file("600000.SH", "1day", daily_path, calendar_path=calendar)
    service.import_file("600000.SH", "1min", minute_path, calendar_path=calendar)
    minute_path.unlink()
    result = service.revalidate("600000.SH", "1min", label_convention="start")
    bars = service.repository.active_rows("600000.SH", "1min")
    assert result["report"]["status"] == "complete"
    assert bars.timestamp.min() == pd.Timestamp("2026-01-05 09:31")
    assert bars.source_verified.all()


def test_consistent_lot_volume_is_corrected_only_with_daily_reference(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days(3)
    calendar = _calendar(tmp_path, days)
    daily_path, minute_path = tmp_path / "600000_daily.csv", tmp_path / "600000_1min.csv"
    _daily(days).to_csv(daily_path, index=False)
    minute = _minute(days)
    minute["volume"] = 1.0
    minute.to_csv(minute_path, index=False)
    service.import_file("600000.SH", "1day", daily_path, calendar_path=calendar)
    result = service.import_file("600000.SH", "1min", minute_path, calendar_path=calendar, label_convention="end")
    bars = service.repository.active_rows("600000.SH", "1min")
    assert result["status"] == "ready"
    assert bars.volume.sum() == pytest.approx(72_000)


def test_minute_price_aggregation_must_match_daily_ohlc(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days(3)
    calendar = _calendar(tmp_path, days)
    daily_path, minute_path = tmp_path / "600000_daily.csv", tmp_path / "600000_1min.csv"
    _daily(days).to_csv(daily_path, index=False)
    minute = _minute(days)
    first_day = minute.timestamp.dt.normalize() == days[0]
    minute.loc[first_day, "high"] = 11.0
    minute.to_csv(minute_path, index=False)
    service.import_file("600000.SH", "1day", daily_path, calendar_path=calendar)

    result = service.import_file(
        "600000.SH", "1min", minute_path,
        calendar_path=calendar, label_convention="end",
    )

    assert result["status"] != "ready"
    assert any("分钟价格聚合与日线不一致" in item for item in result["pending_validation"])


def test_skip_invalid_rows_removes_zero_price_source_placeholders(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex([pd.Timestamp("2026-01-05")])
    calendar = _calendar(tmp_path, days)
    path = tmp_path / "600000_1min.csv"
    frame = _minute(days)
    frame.loc[10, ["open", "high", "low", "close"]] = 0
    frame.to_csv(path, index=False)
    with pytest.raises(MarketDataError, match="非正"):
        service.import_file("600000.SH", "1min", path, calendar_path=calendar, label_convention="end")
    result = service.import_file(
        "600000.SH", "1min", path, calendar_path=calendar, label_convention="end",
        skip_invalid_rows=True,
    )
    assert result["rows_read"] == 239
    assert any("已跳过 1 条" in item for item in result["pending_validation"])


def test_minutes_without_daily_volume_evidence_are_usable_and_revalidate(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days(3)
    calendar = _calendar(tmp_path, days)
    daily_path, minute_path = tmp_path / "600000_daily.csv", tmp_path / "600000_1min.csv"
    _daily(days).to_csv(daily_path, index=False)
    _minute(days).to_csv(minute_path, index=False)
    first = service.import_file("600000.SH", "1min", minute_path, calendar_path=calendar, label_convention="end")
    assert first["status"] == "ready"
    assert not first["pending_validation"]
    service.import_file("600000.SH", "1day", daily_path, calendar_path=calendar)
    result = service.revalidate("600000.SH", "1min")
    assert result["report"]["status"] == "complete"
    assert service.repository.dataset("600000.SH", "1min")["status"] == "ready"


def test_volume_unit_calibration_corrects_clear_100x_ratio_without_vetoing_other_ratios(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    day = pd.Timestamp("2026-01-05")
    frame = _minute_rows(day, minute_slots(day))
    frame["volume"] = 1.0
    daily = pd.DataFrame({
        "timestamp": [day], "tradestatus": [1], "volume": [24000.0],
        "open": [1.0], "high": [1.01], "low": [0.99], "close": [1.0],
    })
    monkeypatch.setattr(service.repository, "active_rows", lambda *_args, **_kwargs: daily)
    corrected, pending = service._verify_volume_units("600000.SH", "1min", frame)
    assert corrected.volume.sum() == pytest.approx(24000)
    assert corrected.attrs["unit_verification"]["volume_factor"] == 100
    assert pending is None

    daily.loc[0, "volume"] = 50000
    corrected, pending = service._verify_volume_units("600000.SH", "1min", frame)
    assert corrected.volume.sum() == pytest.approx(240)
    assert corrected.attrs["unit_verification"]["volume_diagnostic"]
    assert pending is None


def test_source_revision_marks_derived_dataset_stale(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days(5)
    calendar = _calendar(tmp_path, days)
    path = tmp_path / "600000_daily.csv"
    _daily(days).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    AggregationService(service).aggregate("600000.SH", "1day", "1week")
    assert service.repository.dataset("600000.SH", "1week")["derived_stale"] is False
    revised = _daily(days)
    revised.loc[4, "close"] = 10.4
    revised.loc[4, "high"] = 10.8
    revised.to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    assert service.repository.dataset("600000.SH", "1week")["derived_stale"] is True


def test_incomplete_month_is_provisional_and_replaced_by_complete_period(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    first_days = _days(5)
    first_calendar = _calendar(tmp_path, first_days)
    path = tmp_path / "600000_daily.csv"
    _daily(first_days).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=first_calendar)
    provisional = service.bars("600000.SH", "1month")
    dataset = service.repository.dataset("600000.SH", "1month")
    assert provisional["count"] == 1
    assert dataset["derived"]["source_timeframe"] == "1day"
    complete_days = pd.bdate_range("2026-01-05", "2026-01-30")
    complete_calendar = _calendar(tmp_path, complete_days)
    _daily(complete_days).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=complete_calendar)
    monthly_bars = service.bars("600000.SH", "1month")
    rows = service.repository.active_rows("600000.SH", "1month")
    assert len(rows) == 1 and rows.iloc[0].timestamp == pd.Timestamp(complete_days[-1])
    assert monthly_bars["count"] == 1


def test_incremental_requires_initial_history_and_rechecks_closed_watermark(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    sync = MarketDataSyncService(service)
    missing = sync.sync(["600000.SH"], "1day", "incremental")
    assert missing["status"] == "failed" and "初始化" in missing["failures"][0]["reason"]
    days = _days()
    calendar = _calendar(tmp_path, days)
    path = tmp_path / "600000_daily.csv"
    _daily(days).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    monkeypatch.setattr("stockregression.market_data.sync.latest_closed_date", lambda: days[-1])
    calls = []

    def fetch(self, symbol, timeframe, mode, start, end, cancelled):
        calls.append((start, end))
        return _daily(days).copy(), "fixture", {"label_convention": "end"}

    monkeypatch.setattr(MarketDataSyncService, "_fetch", fetch)
    result = sync.sync(["600000.SH"], "1day", "incremental")
    assert calls
    assert len(result["symbols"]) == 1
    assert result["skipped"] == []


@pytest.mark.parametrize("timeframe,fields", [
    ("1week", "date,open,high,low,close,volume,amount,pctChg"),
    ("1month", "date,open,high,low,close,volume,amount,pctChg"),
])
def test_period_history_uses_baostock_period_fields(tmp_path, monkeypatch, timeframe, fields):
    import sys
    import types

    calls = []
    field_list = fields.split(",")

    class Result:
        fields = ()
        error_code = "0"
        error_msg = "success"

        def __init__(self):
            self.rows = [["2026-10-16", "1.0", "1.1", "0.9", "1.05", "120000", "126000", "1.2"]]

        def next(self):
            if not self.rows:
                return False
            self.rows.pop(0)
            return True

        def get_row_data(self):
            return ["2026-10-16", "1.0", "1.1", "0.9", "1.05", "120000", "126000", "1.2"]

    Result.fields = field_list

    def query(_code, requested_fields, **_kwargs):
        calls.append(requested_fields)
        return Result()

    fake = types.SimpleNamespace(
        login=lambda: types.SimpleNamespace(error_code="0", error_msg="success"),
        logout=lambda: None,
        query_history_k_data_plus=query,
    )
    monkeypatch.setitem(sys.modules, "baostock", fake)
    frame = MarketDataSyncService._baostock_history("600000.SH", timeframe, "2026-10-01", "2026-10-16")
    assert calls == [fields]
    assert "preclose" not in calls[0] and "isST" not in calls[0]
    assert float(frame.iloc[0].close) == pytest.approx(1.05)


def test_one_minute_history_never_calls_online_source(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    monkeypatch.setattr(MarketDataSyncService, "_fetch", lambda *args: pytest.fail("1min history must not call network"))
    with pytest.raises(Exception, match="本地数据包"):
        MarketDataSyncService(service)._one(
            "600000.SH", "1min", "history", "2026-01-01", "2026-10-16", lambda: False
        )


def test_one_minute_repair_fetches_one_tail_window(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    sync = MarketDataSyncService(service)
    windows = [
        (pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-02")),
        (pd.Timestamp("2026-10-12"), pd.Timestamp("2026-10-16")),
    ]
    monkeypatch.setattr(sync, "_repair_windows", lambda *args: windows)
    monkeypatch.setattr("stockregression.market_data.sync.latest_closed_date", lambda: pd.Timestamp("2026-10-16"))
    calls = []
    frame = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-10-16 14:59"), pd.Timestamp("2026-10-16 15:00")],
        "open": [10.0, 10.1], "high": [10.1, 10.2], "low": [9.9, 10.0],
        "close": [10.1, 10.2], "volume": [100, 120], "turnover": [1010.0, 1224.0],
        "source": ["sina", "sina"],
    })

    def fetch(symbol, timeframe, mode, start, end, cancelled):
        calls.append((start, end))
        return frame.copy(), "sina", {"label_convention": "end"}

    monkeypatch.setattr(sync, "_fetch", fetch)
    committed = []
    monkeypatch.setattr(service, "_commit_frame", lambda *args, **kwargs: committed.append(kwargs) or {
        "status": "ready", "version": 1
    })
    result = sync._one(
        "600000.SH", "1min", "repair", "2026-09-30", "2026-10-16", lambda: False
    )
    assert len(calls) == 1
    assert calls[0] == (windows[0][0], windows[-1][1])
    assert committed[0]["coverage_end"] == pd.Timestamp("2026-10-16 15:00")
    assert result["source"] == "sina"


def test_daily_sync_commits_many_symbols_as_independent_year_shards(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    first_day = pd.DatetimeIndex(["2026-01-05"])
    two_days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": two_days}).to_csv(calendar, index=False)
    path = tmp_path / "daily.csv"
    _daily(first_day).to_csv(path, index=False)
    for symbol in ("600000.SH", "000001.SZ"):
        service.import_file(symbol, "1day", path, calendar_path=calendar)
    before = {
        symbol: service.repository.dataset(symbol, "1day")["current_version"]
        for symbol in ("600000.SH", "000001.SZ")
    }
    monkeypatch.setattr("stockregression.market_data.sync.latest_closed_date", lambda: two_days[-1])

    def fetch(self, symbol, timeframe, mode, start, end, cancelled):
        return _daily(two_days[-1:]).copy(), "fixture", {"label_convention": "end"}

    monkeypatch.setattr(MarketDataSyncService, "_fetch", fetch)
    result = MarketDataSyncService(service).sync(
        ["600000.SH", "000001.SZ"], "1day", "incremental", resume=False, workers=2,
    )

    assert result["status"] in {"completed", "partial"}
    assert result["batches"] == 1 and result["batch_size"] == 128
    assert {item["symbol"] for item in result["symbols"]} == {"600000.SH", "000001.SZ"}
    assert len(service.repository.active_rows("600000.SH", "1day")) == 2
    assert len(service.repository.active_rows("600000.SH", "1day")) == 2
    assert len(service.repository.active_rows("000001.SZ", "1day")) == 2
    for symbol, version in before.items():
        assert service.repository.dataset(symbol, "1day")["current_version"] > version
    published = json.loads(service.repository.package.manifest_path.read_text(encoding="utf-8"))
    assert published["shard_count"] == 2
    assert len(list(service.repository.package.partitions_dir.rglob("*.parquet"))) == 4


def test_purged_delisted_instrument_cannot_reenter_market_sync_scope(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = _days()
    calendar = _calendar(tmp_path, days)
    path = tmp_path / "159001_daily.csv"
    _daily(days).to_csv(path, index=False)
    symbol = "159001.SZ"
    service.import_file(symbol, "1day", path, calendar_path=calendar)
    service.store.put_instrument(
        symbol, "历史基金", asset_type="etf",
        delisting_date=str(days[-1].date()),
        permanent_exclusion="test-cleanup",
    )

    assert service.store.delisted_symbols() == [symbol]
    assert service.store.instruments() == []
    assert service.repository.symbols_for_timeframe("1day") == []
    service.repository.rebuild_index(
        symbols=[symbol], exclude_symbols=[symbol],
    )
    assert service.repository.dataset(symbol, "1day") is None

    result = service.repository.purge_instruments([symbol])

    assert result["symbols"] == 1
    assert result["datasets"] >= 1
    assert service.repository.dataset(symbol, "1day") is None
    assert not (
        service.repository.package.partitions_dir / "1day" / f"symbol={symbol}"
    ).exists()
    assert service.store.instrument(symbol)["permanent_exclusion"] == "test-cleanup"


def test_market_data_api_imports_and_defaults_sync_to_base_timeframes(tmp_path):
    root = tmp_path / "data"
    days = _days(2)
    calendar = _calendar(tmp_path, days)
    path = tmp_path / "600000_daily.csv"
    _daily(days).to_csv(path, index=False)
    app = create_app(str(root), start_jobs=False)
    with TestClient(app) as client:
        submitted = client.post("/api/v1/data/sync", json={"symbols": ["600000.SH"], "mode": "incremental"})
        assert submitted.status_code == 200
        submitted = client.post("/api/v1/data/import", json={
            "symbol": "600000.SH", "timeframe": "1day", "path": str(path),
            "calendar_path": str(calendar), "label_convention": "end",
        })
        assert submitted.status_code == 200
        job_id = submitted.json()["job_id"]
        assert Store(root).claim_job(job_id, "test-worker")
        execute_job(str(root), job_id)
        datasets = client.get("/api/v1/data/datasets?timeframe=1day").json()["items"]
        bars = client.get("/api/v1/data/bars/600000.SH?frequency=1day").json()
        assert len(datasets) == 1 and datasets[0]["storage"] == "parquet"
        assert len(bars) == 2
        assert client.get("/api/v1/data/coverage/600000.SH?timeframe=1min").status_code == 400


@pytest.mark.parametrize("outcome", ["completed", "partial", "failed"])
def test_sync_job_summary_counts_and_failure_periods(tmp_path, monkeypatch, outcome):
    def sync(self, symbols, timeframe, mode, **kwargs):
        failed = outcome == "failed" or (outcome == "partial" and timeframe == "1min")
        return {
            "status": "failed" if failed else "completed", "timeframe": timeframe,
            "symbols": [] if failed else [{"symbol": symbols[0]}],
            "skipped": [],
            "failures": [{"symbol": symbols[0], "reason": "calendar write denied"}] if failed else [],
        }
    monkeypatch.setattr(MarketDataSyncService, "sync", sync)
    root = tmp_path / "data"
    with TestClient(create_app(str(root), start_jobs=False)) as client:
        job_id = client.post("/api/v1/data/sync", json={
            "symbols": ["159516.SZ"], "timeframes": ["1day", "1min"],
        }).json()["job_id"]
        assert Store(root).claim_job(job_id, "test-worker")
        execute_job(str(root), job_id)
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"] == outcome, job
        report = client.get(f"/api/v1/results/{job['result_id']}").json()
        assert len(report["runs"]) == 2
        assert all(item["timeframe"] in {"1day", "1min"} for item in report["failures"])
        if outcome == "partial":
            assert "成功 1 项，跳过 0 项，失败 1 项" in job["message"]
            assert report["failures"][0]["timeframe"] == "1min"
        if outcome == "failed":
            assert "calendar write denied" in job["error"]


def test_single_upload_picker_path_saves_job_and_removes_staging_files(tmp_path):
    root = tmp_path / "data"
    days = _days(2)
    daily_csv = _daily(days).to_csv(index=False)
    calendar_csv = pd.DataFrame({"date": days}).to_csv(index=False)
    app = create_app(str(root), start_jobs=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/data/upload",
            data={"symbol": "600000.SH", "timeframe": "1day", "label_convention": "end"},
            files={
                "market_file": ("600000_daily.csv", daily_csv, "text/csv"),
                "calendar_file": ("calendar.csv", calendar_csv, "text/csv"),
            },
        )
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        payload = app.state.store.job(job_id)["payload"]
        assert all((root / relative).exists() for relative in payload["upload_dirs"])
        assert app.state.store.claim_job(job_id, "test-worker")
        execute_job(str(root), job_id)
        assert client.get("/api/v1/jobs/" + job_id).json()["status"] == "completed"
        assert app.state.store.snapshot("600000.SH")["market_data"] is True
        assert not any((root / relative).exists() for relative in payload["upload_dirs"])


def test_folder_upload_accepts_relative_paths_and_keeps_calendar_outside_batch_scan(tmp_path):
    root = tmp_path / "data"
    days = _days(2)
    daily_csv = _daily(days).to_csv(index=False)
    calendar_csv = pd.DataFrame({"date": days}).to_csv(index=False)
    app = create_app(str(root), start_jobs=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/data/upload-batch",
            data={"timeframe": "1day", "label_convention": "end"},
            files=[
                ("files", ("selected/600000_daily.csv", daily_csv.encode(), "text/csv")),
                ("files", ("selected/000001_daily.csv", daily_csv.encode(), "text/csv")),
                ("calendar_file", ("calendar.csv", calendar_csv.encode(), "text/csv")),
            ],
        )
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        assert app.state.store.claim_job(job_id, "test-worker")
        execute_job(str(root), job_id)
        job = client.get("/api/v1/jobs/" + job_id).json()
        assert job["status"] == "completed", job.get("error")
        assert {row["symbol"] for row in client.get("/api/v1/data/datasets?timeframe=1day").json()["items"]} == {"600000.SH", "000001.SZ"}
