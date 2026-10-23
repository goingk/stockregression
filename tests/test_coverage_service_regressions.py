from __future__ import annotations

from concurrent.futures import Future

import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.coverage_index import CoverageArchive, coverage_frame_hash
from stockregression.quality import minute_slots
from stockregression.storage import Store


def test_import_without_saved_calendar_does_not_crash(tmp_path):
    path = tmp_path / "daily.csv"
    pd.DataFrame({
        "date": ["2026-01-05"], "open": [10.0], "high": [10.5],
        "low": [9.5], "close": [10.2], "volume": [1000],
        "preclose": [10.0], "tradestatus": [1], "is_st": [0],
    }).to_csv(path, index=False)

    service = MarketDataService(Store(tmp_path / "data"))
    result = service.import_file(
        "600000.SH", "1day", path,
    )

    assert result["symbol"] == "600000.SH"
    assert result["status"] in {"ready", "partial", "pending_validation"}
    rows = CoverageArchive(service.repository).rows("600000.SH", "1day")
    row = next(item for item in rows if item["trade_date"] == "2026-01-05")
    assert row["scanned"] is True
    assert row["is_trading_day"] is True
    assert not (row.get("report") or {}).get("non_trading")


def test_compatibility_coverage_does_not_build_missing_archive(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    path = tmp_path / "daily.csv"
    pd.DataFrame({
        "date": ["2026-01-05"], "open": [10.0], "high": [10.5],
        "low": [9.5], "close": [10.2], "volume": [1000],
        "preclose": [10.0], "tradestatus": [1], "is_st": [0],
    }).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path)
    monkeypatch.setattr(CoverageArchive, "rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(CoverageArchive, "trusted_head", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "stockregression.market_data.service.validate_incremental",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must stay read-only")),
    )

    result = service.compatibility_coverage("600000.SH")

    assert result["diagnostics"]["price_quality"]["daily"]["status"] == "pending"
    assert result["diagnostics"]["status"] == "partial"


def test_dirty_archive_rows_are_projected_as_pending(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    path = tmp_path / "daily.csv"
    pd.DataFrame({
        "date": ["2026-01-05"], "open": [10.0], "high": [10.5],
        "low": [9.5], "close": [10.2], "volume": [1000],
        "preclose": [10.0], "tradestatus": [1], "is_st": [0],
    }).to_csv(path, index=False)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": ["2026-01-05"]}).to_csv(calendar, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    cached = service.compatibility_coverage("600000.SH")
    assert cached["diagnostics"]["price_quality"]["daily"]["status"] == "passed"
    archive = CoverageArchive(service.repository)
    archive.mark_dirty("600000.SH", "1day", ["2026-01-05"], reason="rules changed")

    result = service.compatibility_coverage("600000.SH")

    assert result["days"][0]["status"] == "pending"
    assert result["diagnostics"]["price_quality"]["daily"]["status"] == "pending"
    assert result["diagnostics"]["status"] == "partial"


def test_cached_coverage_marks_only_corrupt_archive_month_pending(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    path = tmp_path / "daily.csv"
    dates = ["2026-01-05", "2026-02-02"]
    pd.DataFrame({
        "date": dates, "open": [10.0, 10.0], "high": [10.5, 10.5],
        "low": [9.5, 9.5], "close": [10.2, 10.2], "volume": [1000, 1000],
        "preclose": [10.0, 10.0], "tradestatus": [1, 1], "is_st": [0, 0],
    }).to_csv(path, index=False)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": dates}).to_csv(calendar, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    assert service.compatibility_coverage("600000.SH")["diagnostics"]["price_quality"]["daily"]["status"] == "passed"

    archive = CoverageArchive(service.repository)
    head = archive.load_head("600000.SH", "1day")
    february = next(item for item in head["months"] if item["month"] == "2026-02")
    (service.repository.package_dir / february["path"]).write_bytes(b"corrupt month")

    result = service.compatibility_coverage("600000.SH")

    states = {row["date"]: row["status"] for row in result["days"]}
    assert states["2026-01-05"] == "passed"
    assert states["2026-02-02"] == "pending"
    assert result["diagnostics"]["price_quality"]["daily"]["status"] == "pending"
    assert result["diagnostics"]["status"] == "partial"


def test_full_daily_validation_reuses_the_loaded_frame(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    path = tmp_path / "daily.csv"
    pd.DataFrame({
        "date": ["2025-12-31", "2026-01-02"], "open": [10.0, 10.0],
        "high": [10.5, 10.5], "low": [9.5, 9.5], "close": [10.2, 10.2],
        "volume": [1000, 1000], "preclose": [10.0, 10.0],
        "tradestatus": [1, 1], "is_st": [0, 0],
    }).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path)
    calls = []
    path_reads = []
    original = service.repository.active_rows
    original_read_paths = service.repository._read_paths

    def counted(*args, **kwargs):
        calls.append((kwargs.get("start"), kwargs.get("end")))
        return original(*args, **kwargs)

    def counted_read_paths(*args, **kwargs):
        path_reads.append(tuple(args[0]))
        return original_read_paths(*args, **kwargs)

    monkeypatch.setattr(service.repository, "active_rows", counted)
    monkeypatch.setattr(service.repository, "_read_paths", counted_read_paths)
    result = service.revalidate("600000.SH", "1day", mode="full")

    assert len(calls) == 1
    assert len(path_reads) == 1
    assert result["report"]["market_data_read_calls"] == 1
    assert result["report"]["market_shard_reads"] == 2

    calls.clear()
    path_reads.clear()
    second_result = service.revalidate("600000.SH", "1day", mode="full")

    assert len(calls) == 1
    assert len(path_reads) == 1
    assert second_result["report"]["market_data_read_calls"] == 1
    assert second_result["report"]["market_shard_reads"] == 2


def test_parallel_validation_retries_only_stale_dates_and_counts_once(tmp_path, monkeypatch):
    import stockregression.market_data.service as service_module

    service = MarketDataService(Store(tmp_path / "data"))
    path = tmp_path / "daily.csv"
    calendar = tmp_path / "calendar.csv"
    days = ["2026-01-05", "2026-02-02"]
    pd.DataFrame({
        "date": days, "open": [10.0, 10.0], "high": [10.5, 10.5],
        "low": [9.5, 9.5], "close": [10.2, 10.2], "volume": [1000, 1000],
        "preclose": [10.0, 10.0], "tradestatus": [1, 1], "is_st": [0, 0],
    }).to_csv(path, index=False)
    pd.DataFrame({"date": days}).to_csv(calendar, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)

    def make_rows(day, status="passed"):
        base = {
            "symbol": "600000.SH", "timeframe": "1day", "adjustment": "none",
            "trade_date": day, "status": status, "row_count": 1,
            "source_hash": "same", "dependency_hash": "same", "rules_version": 1,
            "report": {}, "checked_at": "now",
        }
        coverage = {
            **base, "is_trading_day": True, "scanned": True,
            "gaps": [], "pending": [], "dirty_reason": None,
        }
        return coverage, base

    worker_daily_payloads = []

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, _fn, *args):
            future = Future()
            date = args[8][0]
            preloaded_daily = args[10]
            assert preloaded_daily is not None
            assert set(pd.to_datetime(preloaded_daily.timestamp).dt.strftime("%Y-%m-%d")) == {date}
            worker_daily_payloads.append(
                (date, preloaded_daily.attrs.get("_coverage_data_version"))
            )
            # The second worker fails, but its result is discarded as stale;
            # the retry succeeds and must determine the final failed count.
            coverage, validation = make_rows(
                date, status="failed" if date == days[1] else "passed"
            )
            future.set_result({
                "checked_days": 1, "failed_days": 0,
                "_coverage_rows": [coverage], "_validation_rows": [validation],
                "market_data_read_calls": 0, "market_shard_reads": 0,
            })
            return future

        def shutdown(self, **kwargs):
            pass

    commit_inputs = []

    def commit_rows(_archive, rows, *, validation_rows=(), **kwargs):
        trade_dates = [row["trade_date"] for row in rows]
        commit_inputs.append((trade_dates, [row["trade_date"] for row in validation_rows]))
        if len(commit_inputs) == 1:
            return {
                "revision": 1, "_accepted_dates": [days[0]],
                "_stale_dates": [days[1]], "_evidence_read_calls": 1,
            }
        return {
            "revision": 2, "_accepted_dates": [days[1]],
            "_stale_dates": [], "_evidence_read_calls": 0,
        }

    def retry(_service, _symbol, _timeframe, *, dates, **kwargs):
        assert dates == [days[1]]
        coverage, validation = make_rows(days[1])
        return {
            "_coverage_rows": [coverage], "_validation_rows": [validation],
            "failed_days": 0, "market_data_read_calls": 0,
            "market_shard_reads": 0,
        }

    monkeypatch.setattr(service_module, "ProcessPoolExecutor", FakeExecutor)
    # Route this daily fixture through the multi-period scheduler branch while
    # keeping the worker and publication protocol itself under test.
    monkeypatch.setattr(service_module, "MINUTE_TIMEFRAMES", ("1day",))
    monkeypatch.setattr(service_module, "effective_setting", lambda *_args: 2)
    monkeypatch.setattr(service_module, "effective_cpu_budget", lambda *_args: 2)
    monkeypatch.setattr(CoverageArchive, "commit_rows", commit_rows)
    monkeypatch.setattr(service_module, "validate_incremental", retry)
    daily_prefetch_calls = []
    real_daily_prefetch = service_module._read_daily_dependencies

    def counted_daily_prefetch(*args, **kwargs):
        daily_prefetch_calls.append(args[2])
        return real_daily_prefetch(*args, **kwargs)

    monkeypatch.setattr(service_module, "_read_daily_dependencies", counted_daily_prefetch)

    result = service._validate_ledger(
        "600000.SH", "1day", mode="full", adjustment="none",
        start=days[0], end=days[-1],
    )

    assert sorted(commit_inputs[0][0]) == days
    assert sorted(commit_inputs[0][1]) == days
    assert commit_inputs[1] == ([days[1]], [days[1]])
    assert len(daily_prefetch_calls) == 1
    assert sorted(date for date, _version in worker_daily_payloads) == days
    assert all(version == service.repository.dataset("600000.SH", "1day")["current_version"]
               for _date, version in worker_daily_payloads)
    assert result["checked_days"] == 2
    assert result["market_data_read_calls"] == 2
    assert result["market_shard_reads"] == 2


def test_legacy_single_day_metadata_counts_days_but_keeps_quality_pending(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    dataset = {
        "symbol": "510050.SH", "timeframe": "1min", "adjustment": "none",
        "current_version": 7, "status": "ready", "first_time": "2026-01-05 09:31:00",
        "last_time": "2026-01-05 15:00:00", "row_count": 240,
        "current_bytes": 1024, "storage": "parquet", "synthetic_rows": 0,
        "versions": [],
    }
    monkeypatch.setattr(
        service.repository, "coverage",
        lambda _symbol, timeframe: None if timeframe == "1day" else dict(dataset),
    )
    monkeypatch.setattr(
        service.repository, "dataset",
        lambda _symbol, _timeframe: dict(dataset),
    )
    monkeypatch.setattr(CoverageArchive, "trusted_head", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(CoverageArchive, "rows", lambda *_args, **_kwargs: [])

    result = service.compatibility_coverage("510050.SH")

    assert result["diagnostics"]["minute_days"] == 1
    assert result["diagnostics"]["minute_days_known"] is True
    assert result["diagnostics"]["price_quality"]["minute"]["status"] == "pending"
    assert result["diagnostics"]["status"] == "partial"


def test_legacy_multiday_metadata_without_report_stays_unknown(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    dataset = {
        "symbol": "510050.SH", "timeframe": "1min", "adjustment": "none",
        "current_version": 7, "status": "ready", "first_time": "2026-01-05 09:31:00",
        "last_time": "2026-01-06 15:00:00", "row_count": 480,
        "current_bytes": 2048, "storage": "parquet", "synthetic_rows": 0,
        "versions": [],
    }
    monkeypatch.setattr(
        service.repository, "coverage",
        lambda _symbol, timeframe: None if timeframe == "1day" else dict(dataset),
    )
    monkeypatch.setattr(
        service.repository, "dataset",
        lambda _symbol, _timeframe: dict(dataset),
    )
    monkeypatch.setattr(CoverageArchive, "trusted_head", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(CoverageArchive, "rows", lambda *_args, **_kwargs: [])

    result = service.compatibility_coverage("510050.SH")

    assert result["diagnostics"]["minute_days"] == 0
    assert result["diagnostics"]["minute_days_known"] is False
    assert result["diagnostics"]["price_quality"]["minute"]["status"] == "pending"


def test_full_minute_volume_correction_preloads_the_committed_version(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    day = pd.Timestamp("2026-01-05")
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": [day]}).to_csv(calendar, index=False)
    daily = tmp_path / "daily.csv"
    pd.DataFrame({
        "date": [day], "open": [10.0], "high": [10.5], "low": [9.5],
        "close": [10.2], "volume": [2400.0], "preclose": [10.0],
        "tradestatus": [1], "is_st": [0],
    }).to_csv(daily, index=False)
    service.import_file("600000.SH", "1day", daily, calendar_path=calendar)
    minute = tmp_path / "minute.csv"
    pd.DataFrame({
        "timestamp": minute_slots(day), "open": 10.0, "high": 10.5,
        "low": 9.5, "close": 10.2, "volume": 0.1,
    }).to_csv(minute, index=False)
    service.import_file("600000.SH", "1min", minute, calendar_path=calendar)
    old_version = service.repository.dataset("600000.SH", "1min")["current_version"]
    raw_sum = service.repository.active_rows(
        "600000.SH", "1min", start=day, end=day + pd.Timedelta(days=1),
        use_cache=False,
    ).volume.sum()

    def correct_units(_symbol, _timeframe, frame):
        verified = frame.copy()
        verified["volume"] = verified.volume * 100
        return verified, None

    monkeypatch.setattr(service, "_verify_volume_units", correct_units)

    result = service.revalidate("600000.SH", "1min", mode="full")

    current = service.repository.dataset("600000.SH", "1min")
    assert current["current_version"] > old_version
    rows = service.repository.active_rows(
        "600000.SH", "1min", start=day, end=day + pd.Timedelta(days=1),
        use_cache=False,
    )
    archive_row = next(
        row for row in CoverageArchive(service.repository).rows("600000.SH", "1min")
        if row["trade_date"] == str(day.date())
    )

    assert rows.volume.sum() == pytest.approx(raw_sum * 100)
    assert archive_row["source_hash"] == coverage_frame_hash(rows)
    assert result["report"]["checked_days"] == 1
