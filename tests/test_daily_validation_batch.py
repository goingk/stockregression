from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from stockregression.market_data import validation
from stockregression.market_data.repository import MarketDataRepository
from stockregression.data import DataService


def _daily_frame(days: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame({
        "timestamp": days,
        "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1,
        "volume": 1000.0, "turnover": 10100.0, "preclose": 10.0,
        "tradestatus": 1, "is_st": False, "source_verified": True,
        "is_synthetic": False, "is_auction_placeholder": False,
    })
    return frame


def _harness(frame, days, *, trading_days=None):
    class Repository:
        def __init__(self):
            self.reads = []
            self.saved = []

        def _storage_period(self, timeframe, day):
            year = pd.Timestamp(day).year
            return f"year={year}", f"{year}-01-01", f"{year + 1}-01-01"

        def dataset(self, *args):
            return {"current_version": 1, "adjustment": "none"}

        def active_rows(self, symbol, timeframe, *, adjustment, start, end, **kwargs):
            self.reads.append((start, end))
            return frame.copy(deep=True)

        def auxiliary(self, symbol, kind):
            return None

        def validation_state(self, *args, **kwargs):
            return {"verified_from": str(days[0].date()), "verified_through": str(days[-1].date()),
                    "failed": [], "pending": []}

        def save_coverage(self, *args):
            self.saved.append(("coverage", args))

        def update_dataset_status(self, *args):
            self.saved.append(("status", args))

    class Archive:
        def __init__(self, repository):
            pass

        def commit_rows(self, rows, **kwargs):
            commits.append((len(rows), {row["trade_date"][:7] for row in rows}))
            return {"revision": len(commits), "_stale_dates": [], "_evidence_read_calls": 0}

    commits = []
    repository = Repository()
    store = SimpleNamespace(instrument=lambda symbol: {"asset_type": "stock"})
    service = SimpleNamespace(repository=repository, store=store)
    targets = list(pd.DatetimeIndex(days).normalize())
    selection = {
        "dataset": {"current_version": 1, "adjustment": "none"},
        "lower": targets[0], "upper": targets[-1], "calendar_days": pd.DatetimeIndex(targets),
        "trading_days": pd.DatetimeIndex(trading_days if trading_days is not None else targets),
        "targets": targets, "calendar_available": True, "reused_days": 0,
        "archive": {"revision": 1, "data_version": 1}, "file_state": {"status": "readable"},
        "has_pending_scope": True,
    }
    return service, selection, Archive, commits


def _semantic_rows(result):
    coverage = [{key: value for key, value in row.items() if key != "checked_at"}
                for row in result["_coverage_rows"]]
    validations = [{key: value for key, value in row.items() if key != "checked_at"}
                   for row in result["_validation_rows"]]
    return coverage, validations


def test_daily_fast_path_matches_legacy_reports_and_ledger_rows(monkeypatch):
    days = pd.DatetimeIndex([
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
    ])
    trading = days.delete(2)  # Jan 4 remains in the target calendar but is non-trading.
    frame = _daily_frame(days.delete(2))
    # Missing strict field, blocked timestamp, anomalous price, and a duplicate
    # timestamp all exercise the existing failure and fallback semantics.
    frame["is_st"] = frame.is_st.astype("boolean")
    frame.loc[frame.timestamp == days[1], "is_st"] = pd.NA
    frame.loc[frame.timestamp == days[3], "preclose"] = pd.NA
    frame = pd.concat([frame, frame.loc[frame.timestamp == days[4]]], ignore_index=True)
    frame.attrs["blocked_timestamps"] = [pd.Timestamp("2024-01-05 09:31")]
    frame.attrs["price_anomaly_rows"] = [{
        "timestamp": pd.Timestamp("2024-01-08"), "source": "fixture",
        "quality_fields": "close", "quality_reason": "close invalid",
    }]
    service, selection, archive, _ = _harness(frame, days, trading_days=trading)
    monkeypatch.setattr(validation, "CoverageArchive", archive)

    fast = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection,
        persist_results=False, _daily_fast_path=True,
    )
    legacy = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection,
        persist_results=False, _daily_fast_path=False,
    )

    assert _semantic_rows(fast) == _semantic_rows(legacy)
    assert fast["failed_days"] == legacy["failed_days"]
    assert fast["passed_days"] == legacy["passed_days"]
    assert fast["checked_days"] == len(days)


@pytest.mark.parametrize("case", [
    "normal", "missing_day", "suspended_status", "zero_volume",
    "missing_preclose", "missing_tradestatus", "missing_is_st",
    "source_unverified",
])
def test_daily_scalar_semantic_matrix_matches_legacy(monkeypatch, case):
    day = pd.Timestamp("2024-06-03")
    frame = _daily_frame(pd.DatetimeIndex([day]))
    if case == "missing_day":
        frame = frame.iloc[0:0].copy()
    elif case == "suspended_status":
        frame.loc[0, "tradestatus"] = 0
    elif case == "zero_volume":
        frame.loc[0, ["volume", "turnover"]] = 0
    elif case == "missing_preclose":
        frame.loc[0, "preclose"] = pd.NA
    elif case == "missing_tradestatus":
        frame.loc[0, "tradestatus"] = pd.NA
    elif case == "missing_is_st":
        frame["is_st"] = frame.is_st.astype("boolean")
        frame.loc[0, "is_st"] = pd.NA
    elif case == "source_unverified":
        frame.loc[0, "source_verified"] = False
    service, selection, archive, _ = _harness(frame, pd.DatetimeIndex([day]))
    monkeypatch.setattr(validation, "CoverageArchive", archive)

    fast = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection, persist_results=False,
    )
    legacy = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection,
        persist_results=False, _daily_fast_path=False,
    )

    assert _semantic_rows(fast) == _semantic_rows(legacy), case
    assert fast["passed_days"] == legacy["passed_days"], case
    assert fast["failed_days"] == legacy["failed_days"], case
    if case == "source_unverified":
        # Daily ledger policy is intentionally unchanged; downstream strict
        # backtest readiness remains responsible for verified-source gating.
        assert fast["_coverage_rows"][0]["status"] == legacy["_coverage_rows"][0]["status"]
        assert frame.source_verified.eq(False).all()
        assert DataService._minute_status(frame[["timestamp", "source_verified"]]) == "unverified"


def test_daily_preloaded_duplicate_day_across_month_year_views_matches_legacy(monkeypatch):
    december = pd.DatetimeIndex(["2025-12-31", "2025-12-31 15:00"])
    january = pd.DatetimeIndex(["2025-12-31 15:00", "2026-01-02"])
    days = pd.DatetimeIndex(["2025-12-31", "2026-01-02"])
    dec_frame, jan_frame = _daily_frame(december), _daily_frame(january)
    service, selection, archive, _ = _harness(pd.concat([dec_frame, jan_frame.iloc[1:]], ignore_index=True), days)
    monkeypatch.setattr(validation, "CoverageArchive", archive)
    preloaded = {"2025-12": dec_frame, "2026-01": jan_frame}

    fast = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection, persist_results=False,
        preloaded_months=preloaded,
    )
    legacy = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection, persist_results=False,
        preloaded_months=preloaded, _daily_fast_path=False,
    )

    assert _semantic_rows(fast) == _semantic_rows(legacy)
    assert [row["row_count"] for row in fast["_coverage_rows"]] == [2, 1]
    # The two year partitions came from the caller's preloaded monthly views;
    # grouping/deduplicating them must not count as two additional market reads.
    assert fast["market_data_read_calls"] == 0
    assert fast["market_data_shard_reads"] == 0


def test_quarantined_invalid_ohlc_matches_legacy_daily_ledger(monkeypatch):
    day = pd.Timestamp("2024-07-01")
    raw = _daily_frame(pd.DatetimeIndex([day]))
    raw.loc[0, "high"] = 8.0  # Inverted OHLC: repository consumer quarantine removes it.
    quarantined = MarketDataRepository._frame_for_consumer(raw)
    assert quarantined.empty
    assert quarantined.attrs.get("price_anomaly_rows")
    service, selection, archive, _ = _harness(quarantined, pd.DatetimeIndex([day]))
    monkeypatch.setattr(validation, "CoverageArchive", archive)

    fast = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection, persist_results=False,
    )
    legacy = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection, persist_results=False,
        _daily_fast_path=False,
    )

    assert _semantic_rows(fast) == _semantic_rows(legacy)
    assert fast["_coverage_rows"][0]["status"] == "failed"
    assert fast["_coverage_rows"][0]["blocked_rows"] == 1


def test_daily_hash_batch_and_instrument_snapshot_are_once_per_run(monkeypatch):
    days = pd.date_range("2024-02-01", periods=5, freq="D")
    frame = _daily_frame(days)
    service, selection, archive, _ = _harness(frame, days)
    instrument_calls = []
    service.store.instrument = lambda symbol: instrument_calls.append(symbol) or {"asset_type": "stock"}
    hashes = []
    original_hash = validation._day_hash
    monkeypatch.setattr(validation, "_day_hash", lambda value: hashes.append(len(value)) or original_hash(value))
    monkeypatch.setattr(validation, "CoverageArchive", archive)

    result = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection, persist_results=False,
    )

    assert len(instrument_calls) == 1
    # A single yearly JSON pass makes the per-day digest without invoking the
    # generic DataFrame hash serializer once per target day.
    assert hashes == []
    assert result["checked_days"] == len(days)


def test_daily_preloaded_months_merge_year_partition_and_deduplicate_overlap(monkeypatch):
    days = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-02-02"])
    frame = _daily_frame(days)
    service, selection, archive, _ = _harness(frame, days)
    monkeypatch.setattr(validation, "CoverageArchive", archive)
    january = frame.iloc[:2].copy()
    february = frame.iloc[1:].copy()
    # The February view overlaps Jan's last row; the merged year partition must
    # choose one copy, while preserving duplicate rows native to either view.
    result = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection, persist_results=False,
        preloaded_months={"2024-01": january, "2024-02": february},
    )
    assert result["checked_days"] == 3
    assert service.repository.reads == []
    assert [row["row_count"] for row in result["_coverage_rows"]] == [1, 1, 1]


def test_daily_archive_publication_flushes_at_512_target_days(monkeypatch):
    days = pd.date_range("2024-01-01", periods=513, freq="D")
    frame = _daily_frame(days)
    service, selection, archive, commits = _harness(frame, days)
    monkeypatch.setattr(validation, "CoverageArchive", archive)

    result = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection,
        persist_results=True, persist_archive=True,
    )

    assert [count for count, _ in commits] == [512, 1]
    assert result["checked_days"] == 513
    assert service.repository.saved


def test_daily_cancel_after_512_days_keeps_completed_chunk_published(monkeypatch):
    days = pd.date_range("2024-01-01", periods=513, freq="D")
    frame = _daily_frame(days)
    service, selection, archive, commits = _harness(frame, days)
    monkeypatch.setattr(validation, "CoverageArchive", archive)
    calls = 0

    def cancel_before_next_day():
        nonlocal calls
        calls += 1
        return calls > 512

    try:
        validation.validate_incremental(
            service, "600000.SH", "1day", selection=selection,
            persist_results=True, persist_archive=True, cancelled=cancel_before_next_day,
        )
    except InterruptedError as exc:
        assert str(exc) == "任务已取消"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("validation should stop before target day 513")

    assert [count for count, _ in commits] == [512]


def test_incremental_target_subset_never_validates_unaffected_history(monkeypatch):
    all_days = pd.date_range("2020-01-01", periods=900, freq="D")
    targets = all_days[-4:]
    frame = _daily_frame(all_days)
    service, selection, archive, _ = _harness(frame, targets)
    selection["calendar_days"] = all_days
    selection["trading_days"] = all_days
    selection["lower"] = all_days[0]
    selection["upper"] = all_days[-1]
    selection["targets"] = list(targets)
    selection["reused_days"] = len(all_days) - len(targets)
    monkeypatch.setattr(validation, "CoverageArchive", archive)

    result = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection, persist_results=False,
    )

    assert result["checked_days"] == 4
    assert result["reused_days"] == 896
    assert [row["trade_date"] for row in result["_coverage_rows"]] == [
        str(day.date()) for day in targets
    ]


def test_no_target_reuse_returns_checkpoint_archive_revision(monkeypatch):
    days = pd.date_range("2026-01-05", periods=2, freq="D")
    frame = _daily_frame(days)
    service, selection, archive, _ = _harness(frame, days)
    selection["targets"] = []
    selection["reused_days"] = len(days)
    selection["archive"] = {"revision": 17, "data_version": 1, "quality_version": 17}
    selection["archive_hit"] = True
    monkeypatch.setattr(validation, "CoverageArchive", archive)

    result = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection, persist_results=False,
    )

    assert result["checked_days"] == 0
    assert result["archive_revision"] == 17
    assert result["_archive_head"] is selection["archive"]
