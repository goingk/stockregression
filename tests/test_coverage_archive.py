from __future__ import annotations

import json

import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.coverage_index import CoverageArchive
from stockregression.market_data.maintenance import reset_market_storage
from stockregression.market_data.package_layout import auxiliary_relative_path
from stockregression.market_data.repository import MarketDataRepository
from stockregression.market_data.sync import MarketDataSyncService
from stockregression.quality import minute_slots
from stockregression.storage import Store


def _daily(days, close=10.2):
    return pd.DataFrame({
        "date": list(days), "open": 10.0, "high": 10.5, "low": 9.5,
        "close": close, "volume": 24_000.0, "turnover": 240_000.0,
        "preclose": 10.0, "tradestatus": 1, "is_st": 0,
    })


def _write_daily(root, name, days, close=10.2):
    path = root / name
    _daily(days, close).to_csv(path, index=False)
    return path


def _calendar(root, days):
    path = root / "calendar.csv"
    pd.DataFrame({"date": list(days)}).to_csv(path, index=False)
    return path


def _minute(days):
    rows = []
    for day in days:
        for timestamp in minute_slots(day):
            rows.append({
                "timestamp": timestamp, "open": 10.0, "high": 10.5,
                "low": 9.5, "close": 10.2, "volume": 100,
                "turnover": 1_020.0, "source": "fixture",
                "source_verified": True, "is_synthetic": False,
                "is_auction_placeholder": False,
            })
    return pd.DataFrame(rows)


def _service_with_gap(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    calendar_days = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07"])
    source_days = calendar_days.delete(1)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", source_days),
        calendar_path=_calendar(tmp_path, calendar_days),
    )
    return service


def test_restart_reuses_stored_gap_without_reading_market_bodies(tmp_path, monkeypatch):
    first = _service_with_gap(tmp_path)
    gap = next(
        row for row in CoverageArchive(first.repository).rows("600000.SH", "1day")
        if row["trade_date"] == "2026-01-06"
    )
    assert gap["status"] == "failed" and gap["gap_count"] == 1
    package = first.repository.package_dir

    restarted = MarketDataService(Store(tmp_path / "data"), package_dir=package)

    def fail_read(*_args, **_kwargs):
        raise AssertionError("unchanged failures and gaps must not rescan market rows")

    monkeypatch.setattr(restarted.repository, "active_rows", fail_read)
    monkeypatch.setattr(restarted.repository, "read_frame", fail_read)
    monkeypatch.setattr(restarted.repository, "bars", fail_read)
    result = restarted.revalidate("600000.SH", "1day")

    assert result["report"]["checked_days"] == 0
    assert result["report"]["archive_hit"] is True
    assert result["status"] == "partial"


def test_new_import_builds_archive_from_the_committed_frame(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07"])

    def fail_read(*_args, **_kwargs):
        raise AssertionError("newly committed rows must not be read back for validation")

    monkeypatch.setattr(service.repository, "active_rows", fail_read)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    head = CoverageArchive(service.repository).trusted_head("600000.SH", "1day")

    assert head["gap_count"] == 0
    assert len(CoverageArchive(service.repository).rows("600000.SH", "1day")) == 3


def test_sparse_revision_rechecks_only_named_days(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.bdate_range("2026-01-05", periods=5)
    calendar = _calendar(tmp_path, days)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "first.csv", days),
        calendar_path=calendar,
    )
    before = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }
    revised = _daily(days)
    revised.loc[1, "close"] = 11.1
    revised.loc[1, "high"] = 11.5
    revised.loc[3, "close"] = 9.9
    revised.loc[3, "low"] = 9.4
    path = tmp_path / "revised.csv"
    revised.to_csv(path, index=False)
    calls = []
    original = service.repository.active_rows

    def bounded_read(*args, **kwargs):
        calls.append((kwargs.get("start"), kwargs.get("end")))
        return original(*args, **kwargs)

    monkeypatch.setattr(service.repository, "active_rows", bounded_read)
    result = service.import_file(
        "600000.SH", "1day", path, calendar_path=calendar,
    )
    after = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }

    assert result["changed_dates"] == ["2026-01-06", "2026-01-08"]
    assert result["metrics"]["validated_days"] == 2
    assert after["2026-01-05"] == before["2026-01-05"]
    assert after["2026-01-07"] == before["2026-01-07"]
    assert after["2026-01-09"] == before["2026-01-09"]
    assert after["2026-01-06"] != before["2026-01-06"]
    assert after["2026-01-08"] != before["2026-01-08"]
    assert calls == []


def test_pending_revision_outside_request_does_not_reopen_other_months(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-02-02"])
    calendar = _calendar(tmp_path, days)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "first.csv", days),
        calendar_path=calendar,
    )
    revised = _daily(days)
    revised.loc[2, "close"] = 11.1
    revised.loc[2, "high"] = 11.5
    path = tmp_path / "revised.csv"
    revised.to_csv(path, index=False)

    def interrupt(*_args, **_kwargs):
        raise RuntimeError("interrupted after market manifest")

    monkeypatch.setattr(service.repository, "_copy_version_shards", interrupt)
    with pytest.raises(RuntimeError, match="after market manifest"):
        service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    monkeypatch.undo()
    service = MarketDataService(Store(tmp_path / "data"))

    def fail_read(*_args, **_kwargs):
        raise AssertionError("an unrelated pending month must not reopen January")

    monkeypatch.setattr(service.repository, "active_rows", fail_read)
    result = service.revalidate(
        "600000.SH", "1day", start="2026-01-05", end="2026-01-06",
    )

    assert result["report"]["checked_days"] == 0
    assert result["report"]["reused_days"] == 2


def test_append_reuses_unchanged_month_archive_file(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-02-02"])
    calendar = _calendar(tmp_path, days)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "first.csv", days),
        calendar_path=calendar,
    )
    first = CoverageArchive(service.repository).trusted_head("600000.SH", "1day")
    first_months = {row["month"]: row for row in first["months"]}
    newer = days.append(pd.DatetimeIndex(["2026-01-06"]))
    _calendar(tmp_path, newer)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "second.csv", newer),
        calendar_path=tmp_path / "calendar.csv",
    )
    second = CoverageArchive(service.repository).trusted_head("600000.SH", "1day")
    second_months = {row["month"]: row for row in second["months"]}

    assert second_months["2026-02"]["path"] == first_months["2026-02"]["path"]
    assert second_months["2026-01"]["path"] != first_months["2026-01"]["path"]
    assert int(second["revision"]) > int(first["revision"])


def test_damaged_archive_head_rebuilds_all_indexed_months(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-02-02"])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    archive = CoverageArchive(service.repository)
    reference = archive.head_reference("600000.SH", "1day")
    service.repository.package.absolute(reference["path"]).write_text(
        "{damaged", encoding="utf-8",
    )

    rebuilt = archive.persist(
        "600000.SH", "1day", changed_months={"2026-02"},
    )
    trusted = archive.trusted_head("600000.SH", "1day")

    assert {item["month"] for item in rebuilt["months"]} == {
        "2026-01", "2026-02"
    }
    assert trusted is not None and trusted["revision"] == rebuilt["revision"]


def test_repair_of_complete_archive_never_reads_or_downloads(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    sync = MarketDataSyncService(service)

    def fail(*_args, **_kwargs):
        raise AssertionError("a complete archive must not read or download market data")

    monkeypatch.setattr(service.repository, "active_rows", fail)
    monkeypatch.setattr(sync, "_fetch", fail)
    result = sync.sync(
        ["600000.SH"], "1day", "repair",
        start="2026-01-05", end="2026-01-06",
    )

    assert result["status"] == "completed"
    assert result["skipped"]
    assert not result["symbols"]


def test_repair_downloads_only_the_stored_gap_and_shrinks_it(tmp_path, monkeypatch):
    service = _service_with_gap(tmp_path)
    sync = MarketDataSyncService(service)
    calendar_days = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07"])

    def fetched(_symbol, timeframe, _mode, _start, _end, _cancelled):
        return (
            _daily([pd.Timestamp("2026-01-06")]), "fixture",
            {
                "label_convention": "end", "daily_source": "fixture",
                "calendar": pd.DataFrame({"date": calendar_days}),
                "actions": pd.DataFrame(), "actions_verified": True,
            },
        )

    monkeypatch.setattr(sync, "_fetch", fetched)
    result = sync.sync(
        ["600000.SH"], "1day", "repair",
        start="2026-01-05", end="2026-01-07",
    )
    repair = next(row for row in CoverageArchive(service.repository).rows("600000.SH", "1day") if row["trade_date"] == "2026-01-06")

    assert result["status"] == "completed"
    assert repair["status"] == "passed"
    assert repair["gap_count"] == 0 and repair["gaps"] == []


def test_copying_package_restores_gap_records_without_market_scan(tmp_path, monkeypatch):
    first = _service_with_gap(tmp_path)
    package = first.repository.package_dir
    copied = MarketDataService(Store(tmp_path / "new-catalog"), package_dir=package)

    def fail_read(*_args, **_kwargs):
        raise AssertionError("copying a package must restore its coverage archive")

    monkeypatch.setattr(copied.repository, "active_rows", fail_read)
    rows = CoverageArchive(copied.repository).rows("600000.SH", "1day")
    gap = next(row for row in rows if row["trade_date"] == "2026-01-06")
    head = CoverageArchive(copied.repository).trusted_head("600000.SH", "1day")

    assert gap["status"] == "failed" and gap["gap_count"] == 1
    assert head and len(head["months"]) == 1
    manifest = json.loads((package / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    reference = manifest["datasets"]["600000.SH|1day|none"]["quality_archive"]
    assert reference["data_version"] == head["data_version"]


def test_explicit_full_mode_rechecks_the_requested_range(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.bdate_range("2026-01-05", periods=3)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    before = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }
    result = service.revalidate(
        "600000.SH", "1day", mode="full",
        start=days[0], end=days[-1],
    )
    after = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }

    assert result["report"]["checked_days"] == len(days)
    assert all(after[str(day.date())] != before[str(day.date())] for day in days)


def test_calendar_tail_extension_checks_only_the_new_day(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05"])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    before = service.repository.day_validations("600000.SH", "1day")[0]["checked_at"]
    service.repository.save_auxiliary(
        "600000.SH", "calendar", [{"date": "2026-01-06"}], verified=True,
    )

    result = service.revalidate("600000.SH", "1day")
    rows = {row["trade_date"]: row for row in CoverageArchive(service.repository).rows("600000.SH", "1day")}
    after = next(row["checked_at"] for row in service.repository.day_validations("600000.SH", "1day") if row["trade_date"] == "2026-01-05")

    assert result["report"]["checked_days"] == 1
    assert rows["2026-01-06"]["status"] == "failed"
    assert rows["2026-01-06"]["gap_count"] == 1
    assert after == before


def test_archive_keeps_more_than_500_complete_gaps(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.bdate_range("2024-01-02", periods=600)
    source_days = pd.DatetimeIndex([days[0], days[-1]])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "sparse.csv", source_days),
        calendar_path=_calendar(tmp_path, days),
    )
    archive = CoverageArchive(service.repository)
    rows = archive.rows("600000.SH", "1day")
    head = archive.trusted_head("600000.SH", "1day")
    gaps = [gap for row in rows for gap in row.get("gaps") or []]

    assert len(gaps) == 598
    assert head["gap_count"] == 598
    assert sum(row["gap_count"] for row in rows) == 598


def test_quality_only_manifest_commit_is_recovered_by_a_new_catalog(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05"])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    archive = CoverageArchive(service.repository)
    before = archive.head_reference("600000.SH", "1day")
    monkeypatch.setattr(
        CoverageArchive, "record_archive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("database interrupted")),
    )
    with pytest.raises(RuntimeError, match="database interrupted"):
        archive.persist("600000.SH", "1day")
    published = archive.head_reference("600000.SH", "1day")
    assert int(published["revision"]) > int(before["revision"])

    monkeypatch.undo()
    recovered = MarketDataService(Store(tmp_path / "data"))
    with recovered.repository._connect() as db:
        indexed = db.execute(
            "SELECT revision FROM parquet_coverage_archives "
            "WHERE symbol='600000.SH' AND timeframe='1day' AND adjustment='none'"
        ).fetchone()

    assert int(indexed["revision"]) == int(published["revision"])


def test_market_publication_keeps_old_archive_but_marks_only_changed_day(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    calendar = _calendar(tmp_path, days)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "first.csv", days),
        calendar_path=calendar,
    )

    def interrupt(*_args, **_kwargs):
        raise RuntimeError("interrupted after manifest publication")

    revised = _daily(days)
    revised.loc[1, "close"] = 11.0
    revised.loc[1, "high"] = 11.5
    revised_path = tmp_path / "revised.csv"
    revised.to_csv(revised_path, index=False)
    monkeypatch.setattr(service.repository, "_copy_version_shards", interrupt)
    with pytest.raises(RuntimeError, match="after manifest"):
        service.import_file("600000.SH", "1day", revised_path, calendar_path=calendar)

    recovered = MarketDataService(Store(tmp_path / "data"))
    head = CoverageArchive(recovered.repository).trusted_head("600000.SH", "1day")
    pending = CoverageArchive(recovered.repository).pending_dates(head)
    assert pending == {pd.Timestamp("2026-01-06")}
    result = recovered.revalidate("600000.SH", "1day")
    assert result["report"]["checked_days"] == 1


def test_month_workers_return_rows_to_one_parent_publication(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03"])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    result = service.revalidate(
        "600000.SH", "1day", mode="full", start=days[0], end=days[-1],
    )

    assert result["report"]["checked_days"] == len(days)
    assert result["report"]["scanned_months"] == 2
    assert result["report"]["archive_revision"]
    assert len(CoverageArchive(service.repository).rows("600000.SH", "1day")) == len(days)


def test_company_action_invalidates_only_its_effective_window(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-12", "2026-01-20"])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    before = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }
    service.repository.save_auxiliary("600000.SH", "actions", [{
        "ex_date": "2026-01-12", "cash_per_share": 1.0,
    }], version="same-label", verified=True)

    result = service.revalidate("600000.SH", "1day")
    after = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }

    assert result["report"]["checked_days"] == 2
    assert after["2026-01-05"] == before["2026-01-05"]
    assert after["2026-01-12"] != before["2026-01-12"]
    assert after["2026-01-20"] != before["2026-01-20"]


def test_auxiliary_dependency_content_change_is_detected_at_same_version(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-12", "2026-01-20"])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    actions = [{"ex_date": "2026-01-12", "cash_per_share": 1.0}]
    service.repository.save_auxiliary(
        "600000.SH", "actions", actions, version="same-label", verified=True,
    )
    service.revalidate("600000.SH", "1day")
    before = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }
    path = service.repository.package.absolute(
        auxiliary_relative_path("600000.SH", "actions")
    )
    pd.DataFrame([{"ex_date": "2026-01-12", "cash_per_share": 2.0}]).to_parquet(
        path, index=False,
    )

    result = service.revalidate("600000.SH", "1day")
    after = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }

    assert result["report"]["checked_days"] == 2
    assert all(after[day] != before[day] for day in before)


def test_action_invalidation_survives_package_copy_without_market_reads(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-12", "2026-01-20"])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    service.repository.save_auxiliary("600000.SH", "actions", [{
        "ex_date": "2026-01-12", "cash_per_share": 1.0,
    }], version="same-label", verified=True)
    package = service.repository.package_dir

    copied = MarketDataService(Store(tmp_path / "copied-catalog"), package_dir=package)
    head = CoverageArchive(copied.repository).trusted_head("600000.SH", "1day")
    pending = CoverageArchive(copied.repository).pending_dates(head)

    assert pending == {
        pd.Timestamp("2026-01-12"), pd.Timestamp("2026-01-20")
    }


def test_daily_revision_pending_is_published_to_minute_archive(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    calendar = _calendar(tmp_path, days)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=calendar,
    )
    minute_path = tmp_path / "minute.csv"
    _minute(days).to_csv(minute_path, index=False)
    service.import_file(
        "600000.SH", "1min", minute_path, calendar_path=calendar,
    )
    revised = _daily(days)
    revised.loc[1, "close"] = 11.1
    revised.loc[1, "high"] = 11.5
    revised_path = tmp_path / "revised.csv"
    revised.to_csv(revised_path, index=False)
    service.import_file("600000.SH", "1day", revised_path, calendar_path=calendar)

    copied = MarketDataService(
        Store(tmp_path / "copied-catalog"), package_dir=service.repository.package_dir,
    )
    head = CoverageArchive(copied.repository).trusted_head("600000.SH", "1min")

    assert CoverageArchive(copied.repository).pending_dates(head) == {
        pd.Timestamp("2026-01-06")
    }


def test_old_catalog_layout_gets_coverage_tables_on_open(tmp_path):
    root = tmp_path / "data"
    repository = MarketDataRepository(root)
    with repository._connect() as db:
        db.execute("DROP TABLE parquet_coverage_days")
        db.execute("DROP TABLE parquet_coverage_archives")
        db.execute("PRAGMA user_version=2")

    reopened = MarketDataRepository(root)
    with reopened._connect() as db:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert {"parquet_coverage_days", "parquet_coverage_archives"} <= tables


def test_reset_detaches_package_instead_of_reindexing_old_market(tmp_path):
    root = tmp_path / "data"
    service = MarketDataService(Store(root))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=_calendar(tmp_path, days),
    )
    source = tmp_path / "empty-source"
    source.mkdir()

    reset_market_storage(root, source_folder=source)
    reopened = MarketDataService(Store(root))

    assert reopened.repository.dataset("600000.SH", "1day") is None
