from concurrent.futures import Future

import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.coverage_index import CoverageArchive
from stockregression.market_data.models import MarketSyncCancelled
from stockregression.market_data.sync import MarketDataSyncService, SOURCE_VERSION
from stockregression.settings import Settings
from stockregression.storage import Store

SYMBOL = "600000.SH"
ETF_SYMBOL = "159001.SZ"


def seed(tmp_path, count=10, start="2026-01-05", symbol=SYMBOL):
    settings = Settings(data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"),
                        performance={"cpu_budget_mode": "custom", "cpu_budget": 2})
    service = MarketDataService(Store(settings.data_dir, settings=settings),
                                package_dir=settings.package_dir, settings=settings)
    asset_type = "etf" if symbol.startswith(("1", "5")) else "stock"
    days = pd.bdate_range(start, periods=count)
    frame = pd.DataFrame({"timestamp": days, "open": 10., "high": 11., "low": 9.,
                          "close": 10., "preclose": 10., "tradestatus": 1, "is_st": False,
                          "volume": 1000., "turnover": 10000., "source": "fixture",
                          "source_verified": True, "is_synthetic": False,
                          "is_auction_placeholder": False})
    service.repository.commit_bars(symbol, "1day", frame, source="fixture", metadata={"asset_type": asset_type})
    service.repository.save_auxiliary(symbol, "calendar", pd.DataFrame({"date": days}).to_dict("records"), verified=True)
    service.store.put_instrument(symbol, name="fixture", asset_type=asset_type)
    return service, days, frame


def run_key(service, cutoff):
    return f"market-sync:{service.repository.package_id}:1day:incremental:{cutoff}:{SOURCE_VERSION}"


def coverage_reports(service, symbol):
    with service.repository._connect() as db:
        return [
            row[0] for row in db.execute(
                "SELECT report FROM parquet_coverage "
                "WHERE symbol=? AND timeframe='1day' AND adjustment='none' "
                "ORDER BY end_time",
                (symbol,),
            ).fetchall()
        ]


def test_legacy_completed_without_ledger_bootstraps_locally(tmp_path, monkeypatch):
    service, days, frame = seed(tmp_path)
    cutoff = str(days[-1].date())
    key = run_key(service, cutoff)
    service.store.sync_item(key, SYMBOL, status="completed", error="stale failure")
    sync = MarketDataSyncService(service, cutoff_date=cutoff)
    monkeypatch.setattr(sync, "_fetch", lambda *_: pytest.fail("ledger lag cannot trigger network history"))
    result = sync.sync([SYMBOL], "1day", "incremental", resume=True)
    assert result["status"] == "completed", result
    assert result["symbols"][0]["local_validation"]
    assert len(CoverageArchive(service.repository).rows(SYMBOL, "1day")) == len(days)
    assert service.store.sync_item(key, SYMBOL)["error"] is None
    assert service.repository.dataset(SYMBOL, "1day")["current_version"] == 1


def test_market_download_tail_is_independent_of_missing_ledger(tmp_path):
    service, days, frame = seed(tmp_path)
    sync = MarketDataSyncService(service, cutoff_date="2026-01-19")
    lower, upper = sync._increment_range(service.repository.dataset(SYMBOL, "1day"), "1day")
    assert lower == days[-1] - pd.Timedelta(days=3)
    assert upper == pd.Timestamp("2026-01-19")
    assert CoverageArchive(service.repository).load_head(SYMBOL, "1day") is None


def test_incremental_skips_stale_local_history_without_network(tmp_path, monkeypatch):
    service, days, frame = seed(tmp_path, start="2025-01-06", symbol=ETF_SYMBOL)
    cutoff = "2026-10-23"
    sync = MarketDataSyncService(service, cutoff_date=cutoff)
    monkeypatch.setattr(sync, "_fetch", lambda *_: pytest.fail("stale incremental history must not call network"))
    key = run_key(service, cutoff)
    service.store.sync_item(key, ETF_SYMBOL, status="failed", error="previous empty response")

    result = sync.sync([ETF_SYMBOL], "1day", "incremental", resume=True)

    assert result["status"] == "completed", result
    assert result["failures"] == []
    assert len(result["skipped"]) == 1
    skipped = result["skipped"][0]
    assert skipped["incremental_stale"] is True
    assert "缺口修补" in skipped["message"]
    assert service.repository.dataset(ETF_SYMBOL, "1day")["last_time"] == str(days[-1])
    state = service.store.sync_item(key, ETF_SYMBOL)
    assert state["status"] == "completed"
    assert state["error"] is None


def test_preparation_returns_auxiliary_without_writing_package(tmp_path, monkeypatch):
    service, days, frame = seed(tmp_path)
    sync = MarketDataSyncService(service, cutoff_date=days[-1])
    manifest = service.repository.package.manifest_path.read_bytes()
    calendar = pd.DataFrame({"date": days})
    monkeypatch.setattr(sync, "_fetch", lambda *_: (frame, "fixture", {"calendar": calendar,
                         "actions": pd.DataFrame(), "name": "new name", "label_convention": "end"}))
    monkeypatch.setattr(service.repository, "save_auxiliary", lambda *_args, **_kwargs: pytest.fail("download cannot persist"))
    monkeypatch.setattr(service.store, "put_instrument", lambda *_args, **_kwargs: pytest.fail("download cannot change catalog metadata"))
    result = sync._prepare_daily(SYMBOL, "incremental", None, None, lambda: False)
    assert "calendar" in result["auxiliary"]
    assert service.repository.package.manifest_path.read_bytes() == manifest


def test_empty_does_not_complete_or_advance_watermark(tmp_path, monkeypatch):
    service, days, frame = seed(tmp_path)
    sync = MarketDataSyncService(service, cutoff_date="2026-01-19")
    monkeypatch.setattr(sync, "_fetch", lambda *_: (pd.DataFrame(), "fixture", {}))
    result = sync.sync([SYMBOL], "1day", "incremental")
    assert result["status"] == "failed"
    assert result["failures"][0]["category"] == "empty_unresolved"
    state = service.store.sync_item(run_key(service, "2026-01-19"), SYMBOL)
    assert state["status"] == "failed" and not state.get("daily_through")
    assert service.repository.dataset(SYMBOL, "1day")["last_time"] == str(days[-1])


def test_cancel_local_validation_preserves_published_batch_and_resumes(tmp_path, monkeypatch):
    from stockregression.market_data import validation
    service, days, frame = seed(tmp_path, count=30)
    cutoff = str(days[-1].date())
    key = run_key(service, cutoff)
    service.store.sync_item(key, SYMBOL, status="completed")
    monkeypatch.setattr(validation, "_DAILY_COMMIT_BATCH_SIZE", 5)
    stopped = False
    details = []
    original_commit = CoverageArchive.commit_rows_many
    def stop_after_first_publication(archive, chunks):
        nonlocal stopped
        result = original_commit(archive, chunks)
        stopped = True
        return result
    monkeypatch.setattr(CoverageArchive, "commit_rows_many", stop_after_first_publication)
    sync = MarketDataSyncService(service, cutoff_date=cutoff, detail=details.append)
    monkeypatch.setattr(sync, "_fetch", lambda *_: pytest.fail("must validate local history"))
    with pytest.raises(MarketSyncCancelled) as captured:
        sync.sync([SYMBOL], "1day", "incremental", resume=True,
                  cancelled=lambda: stopped)
    partial = captured.value.report
    assert partial["status"] == "cancelled"
    assert partial["symbols"] == []
    assert [item["symbol"] for item in partial["published_quality_pending"]] == [SYMBOL]
    assert len(CoverageArchive(service.repository).rows(SYMBOL, "1day")) == 30
    assert any(item.get("total_days") == 30 for item in details)
    monkeypatch.setattr(CoverageArchive, "commit_rows_many", original_commit)
    reports_before_resume = coverage_reports(service, SYMBOL)
    resumed = MarketDataSyncService(service, cutoff_date=cutoff)
    monkeypatch.setattr(resumed, "_fetch", lambda *_: pytest.fail("resume must read local history"))
    result = resumed.sync([SYMBOL], "1day", "incremental", resume=True)
    assert result["status"] == "completed"
    assert len(CoverageArchive(service.repository).rows(SYMBOL, "1day")) == 30
    assert result["skipped"][0]["metrics"]["validated_days"] == 0
    assert coverage_reports(service, SYMBOL) == reports_before_resume


def test_checked_but_invalid_history_remains_partial(tmp_path, monkeypatch):
    service, days, frame = seed(tmp_path)
    cutoff = str(days[-1].date())
    # An explicit calendar day with no bar is a real quality gap.
    missing_day = days[-1] + pd.offsets.BDay()
    service.repository.save_auxiliary(SYMBOL, "calendar", [{"date": missing_day}], verified=True)
    sync = MarketDataSyncService(service, cutoff_date=missing_day)
    monkeypatch.setattr(sync, "_prepare_daily", lambda *_: {"symbol": SYMBOL, "skipped": True,
                                                          "message": "provider range skipped"})
    result = sync.sync([SYMBOL], "1day", "incremental")
    assert result["status"] == "partial"
    assert result["skipped"][0]["quality_status"] == "partial"
    state = service.store.sync_item(run_key(service, str(missing_day.date())), SYMBOL)
    assert state["status"] == "completed" and state["quality_status"] == "partial"
    gap = CoverageArchive(service.repository).coverage_gaps(SYMBOL, "1day", days[0], missing_day)
    assert gap["gaps"], "strict coverage must retain the missing target date"


def test_incremental_commit_and_snapshot_do_not_read_unrelated_years(tmp_path, monkeypatch):
    from stockregression.market_data.repository import MarketDataRepository
    from stockregression.market_data.validation import validate_incremental

    service, days, frame = seed(tmp_path, count=540, start="2024-01-05")
    assert validate_incremental(service, SYMBOL, "1day")["status"] == "complete"
    next_day = days[-1] + pd.offsets.BDay()
    extra = frame.iloc[-1:].copy()
    extra["timestamp"] = next_day
    calendar = pd.DataFrame({"date": days.append(pd.DatetimeIndex([next_day]))})
    paths_read = []
    original = MarketDataRepository._read_paths
    def tracked(repository, paths, **kwargs):
        paths_read.extend(str(path) for path in paths)
        return original(repository, paths, **kwargs)
    monkeypatch.setattr(MarketDataRepository, "_read_paths", tracked)
    sync = MarketDataSyncService(service, cutoff_date=next_day)
    monkeypatch.setattr(sync, "_fetch", lambda *_: (extra, "fixture", {
        "calendar": calendar, "label_convention": "end",
    }))
    result = sync.sync([SYMBOL], "1day", "incremental")
    assert result["status"] == "completed", result
    assert result["symbols"][0]["metrics"]["validated_days"] == 1
    assert paths_read
    assert all(f"year={next_day.year}" in path.replace("\\", "/") for path in paths_read), paths_read
