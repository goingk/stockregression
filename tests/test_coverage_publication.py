from __future__ import annotations

import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.coverage_index import CoverageArchive
from stockregression.market_data.validation import validate_incremental
from stockregression.storage import Store


def _import(service, root, name, dates, close=10.2):
    path = root / name
    pd.DataFrame({
        "date": dates, "open": 10.0, "high": 11.5, "low": 9.5,
        "close": close, "volume": 24_000.0, "turnover": 240_000.0,
        "preclose": 10.0, "tradestatus": 1, "is_st": 0,
    }).to_csv(path, index=False)
    calendar = root / "calendar.csv"
    existing = service.repository.auxiliary("600000.SH", "calendar")
    prior = [row["date"] for row in (existing or {}).get("payload") or []]
    pd.DataFrame({"date": sorted(set(str(value)[:10] for value in [*prior, *dates]))}).to_csv(
        calendar, index=False,
    )
    return service.import_file("600000.SH", "1day", path, calendar_path=calendar)


def _results(service, **kwargs):
    return validate_incremental(service, "600000.SH", "1day", mode="full",
                                persist_results=False, persist_archive=False, **kwargs)


def test_failure_before_publication_cannot_change_either_sqlite_ledger(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    _import(service, tmp_path, "first.csv", ["2026-01-05"])
    archive = CoverageArchive(service.repository)
    before = archive.rows("600000.SH", "1day")
    ledger = service.repository.day_validations("600000.SH", "1day")
    reference = archive.head_reference("600000.SH", "1day")
    result = _results(service)

    def interrupted(*args, **kwargs):
        raise RuntimeError("before publication")

    monkeypatch.setattr(service.repository.package, "publish_quality_archive", interrupted)
    with pytest.raises(RuntimeError, match="before publication"):
        archive.commit_rows(result["_coverage_rows"], validation_rows=result["_validation_rows"],
                            expected_data_version=1)
    assert archive.head_reference("600000.SH", "1day") == reference
    assert archive.rows("600000.SH", "1day") == before
    assert service.repository.day_validations("600000.SH", "1day") == ledger
    recovered = MarketDataService(Store(tmp_path / "data"))
    assert CoverageArchive(recovered.repository).rows("600000.SH", "1day") == before


def test_failure_after_publication_restores_both_ledgers(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    _import(service, tmp_path, "first.csv", ["2026-01-05"])
    archive = CoverageArchive(service.repository)
    before = archive.head_reference("600000.SH", "1day")
    result = _results(service)
    checked_at = result["_coverage_rows"][0]["checked_at"]

    with monkeypatch.context() as context:
        def interrupted(*args, **kwargs):
            raise RuntimeError("mirror interrupted")
        context.setattr(archive, "record_archive", interrupted)
        with pytest.raises(RuntimeError, match="mirror interrupted"):
            archive.commit_rows(result["_coverage_rows"], validation_rows=result["_validation_rows"],
                                expected_data_version=1)
    assert archive.head_reference("600000.SH", "1day")["revision"] > before["revision"]
    recovered = MarketDataService(Store(tmp_path / "data"))
    assert CoverageArchive(recovered.repository).rows("600000.SH", "1day")[0]["checked_at"] == checked_at
    assert recovered.repository.day_validations("600000.SH", "1day")[0]["checked_at"] == checked_at


def test_workers_from_same_base_merge_without_losing_other_month(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    _import(service, tmp_path, "first.csv", ["2026-01-05", "2026-02-02"])
    january = _results(service, start="2026-01-05", end="2026-01-05")
    february = _results(service, start="2026-02-02", end="2026-02-02")
    archive = CoverageArchive(service.repository)
    for result in (january, february):
        head = archive.commit_rows(result["_coverage_rows"], validation_rows=result["_validation_rows"],
                                   expected_data_version=1)
        assert not head["_stale_dates"]
    rows = {row["trade_date"]: row for row in archive.rows("600000.SH", "1day")}
    assert rows["2026-01-05"]["checked_at"] == january["_coverage_rows"][0]["checked_at"]
    assert rows["2026-02-02"]["checked_at"] == february["_coverage_rows"][0]["checked_at"]


def test_concurrent_revision_discards_only_the_changed_day(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = ["2026-01-05", "2026-01-06"]
    _import(service, tmp_path, "first.csv", days)
    result = _results(service)
    _import(service, tmp_path, "revision.csv", [days[1]], close=10.3)
    archive = CoverageArchive(service.repository)
    new_day = archive.rows("600000.SH", "1day", start=days[1], end=days[1])[0]
    head = archive.commit_rows(result["_coverage_rows"], validation_rows=result["_validation_rows"],
                               expected_data_version=1)
    assert head["_accepted_dates"] == [days[0]]
    assert head["_stale_dates"] == [days[1]]
    assert archive.rows("600000.SH", "1day", start=days[1], end=days[1])[0] == new_day


def test_unrelated_new_year_does_not_discard_or_reread_historical_results(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    _import(service, tmp_path, "first.csv", ["2025-01-06"])
    result = _results(service)
    _import(service, tmp_path, "append.csv", ["2026-01-05"])

    def no_read(*args, **kwargs):
        raise AssertionError("unchanged historical shard must not be reread")

    monkeypatch.setattr(service.repository, "active_rows", no_read)
    archive = CoverageArchive(service.repository)
    head = archive.commit_rows(result["_coverage_rows"], validation_rows=result["_validation_rows"],
                               expected_data_version=1)
    assert head["_accepted_dates"] == ["2025-01-06"]
    assert not head["_stale_dates"]


def test_damaged_month_rechecks_only_that_month_and_preserves_other_files(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = ["2026-01-05", "2026-02-02", "2026-03-02"]
    _import(service, tmp_path, "first.csv", days)
    archive = CoverageArchive(service.repository)
    head = archive.load_head("600000.SH", "1day")
    references = {item["month"]: item["path"] for item in head["months"]}
    checked = {row["trade_date"]: row["checked_at"] for row in archive.rows("600000.SH", "1day")}
    service.repository.package.absolute(references["2026-02"]).write_text("{broken", encoding="utf-8")

    assert service.repository.validation_state("600000.SH", "1day")["pending"] == [days[1]]
    result = validate_incremental(service, "600000.SH", "1day")

    assert result["checked_days"] == 1
    rows = {row["trade_date"]: row for row in archive.rows("600000.SH", "1day")}
    assert rows[days[0]]["checked_at"] == checked[days[0]]
    assert rows[days[2]]["checked_at"] == checked[days[2]]
    restored = archive.trusted_head("600000.SH", "1day")
    assert restored is not None
    assert {item["month"]: item["path"] for item in restored["months"] if item["month"] != "2026-02"} == {
        month: path for month, path in references.items() if month != "2026-02"
    }


def test_unchanged_validation_reads_each_archive_file_at_most_once(tmp_path, monkeypatch):
    from collections import Counter
    from pathlib import Path

    first = MarketDataService(Store(tmp_path / "data"))
    _import(first, tmp_path, "first.csv", ["2026-01-05", "2026-02-02", "2026-03-02"])
    service = MarketDataService(Store(tmp_path / "data"))
    reads = Counter()
    original = Path.read_bytes

    def tracked(path):
        if "/quality/" in path.as_posix():
            reads[str(path)] += 1
        return original(path)

    def no_market(*args, **kwargs):
        raise AssertionError("unchanged validation must not read market bodies")

    monkeypatch.setattr(Path, "read_bytes", tracked)
    monkeypatch.setattr(service.repository, "active_rows", no_market)
    result = service.revalidate("600000.SH", "1day")
    assert result["report"]["checked_days"] == 0
    assert all(count == 1 for count in reads.values())
    reads.clear()
    assert service.revalidate("600000.SH", "1day")["report"]["checked_days"] == 0
    assert not reads


def test_month_publication_does_not_query_other_month_day_rows(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    _import(service, tmp_path, "first.csv", ["2026-01-05", "2026-02-02", "2026-03-02"])
    result = _results(service, start="2026-02-02", end="2026-02-02")
    original = CoverageArchive.rows

    def limited(archive, symbol, timeframe, adjustment="none", *, start=None, end=None):
        assert start is not None and str(pd.Timestamp(start).date())[:7] == "2026-02"
        assert end is not None and str(pd.Timestamp(end).date())[:7] == "2026-02"
        return original(archive, symbol, timeframe, adjustment, start=start, end=end)

    monkeypatch.setattr(CoverageArchive, "rows", limited)
    archive = CoverageArchive(service.repository)
    head = archive.commit_rows(result["_coverage_rows"], validation_rows=result["_validation_rows"],
                               expected_data_version=1)
    assert head["_accepted_dates"] == ["2026-02-02"]


def test_upgrade_discards_unpublished_rows_left_by_previous_writer(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    _import(service, tmp_path, "first.csv", ["2026-01-05"])
    with service.repository._connect() as db:
        db.execute("UPDATE parquet_coverage_days SET status='failed',gap_count=99")
        db.execute("UPDATE parquet_day_validation SET status='failed'")
        db.execute("PRAGMA user_version=3")
    recovered = MarketDataService(Store(tmp_path / "data"))
    row = CoverageArchive(recovered.repository).rows("600000.SH", "1day")[0]
    assert row["status"] == "passed" and row["gap_count"] == 0
    assert recovered.repository.day_validations("600000.SH", "1day")[0]["status"] == "passed"


def test_preloaded_frame_keeps_its_version_when_selection_observes_a_newer_one(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    _import(service, tmp_path, "first.csv", ["2026-01-05"])
    frame = service.repository.active_rows("600000.SH", "1day")
    frame.attrs["_coverage_data_version"] = 1
    _import(service, tmp_path, "changed.csv", ["2026-01-05"], close=10.3)

    result = _results(service, preloaded_months={"2026-01": frame})
    assert result["_coverage_rows"][0]["_data_version"] == 1
    head = CoverageArchive(service.repository).commit_rows(
        result["_coverage_rows"], expected_data_version=2,
    )
    assert head["_stale_dates"] == ["2026-01-05"]
    assert not head["_accepted_dates"]


def test_recent_calendar_does_not_turn_older_observed_sessions_into_nontrading_days(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    path = tmp_path / "history.csv"
    days = ["2025-01-06", "2026-01-05"]
    pd.DataFrame({
        "date": days, "open": 10.0, "high": 10.5, "low": 9.5,
        "close": 10.2, "volume": 24_000.0, "turnover": 240_000.0,
        "preclose": 10.0, "tradestatus": 1, "is_st": 0,
    }).to_csv(path, index=False)
    calendar = tmp_path / "recent-calendar.csv"
    pd.DataFrame({"date": [days[1]]}).to_csv(calendar, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    row = CoverageArchive(service.repository).rows(
        "600000.SH", "1day", start=days[0], end=days[0],
    )[0]
    assert row["is_trading_day"] is True
    assert row["row_count"] == 1
    assert not row["report"].get("non_trading")

    def no_market(*args, **kwargs):
        raise AssertionError("an unchanged older failure or pass must remain reusable")

    service.repository.active_rows = no_market
    assert service.revalidate("600000.SH", "1day")["report"]["checked_days"] == 0
