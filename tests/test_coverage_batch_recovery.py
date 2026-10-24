from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stockregression import performance
from stockregression.market_data import MarketDataService
from stockregression.market_data.coverage_index import CoverageArchive
from stockregression.market_data.repository import MarketDataRepository
from stockregression.market_data.validation import validate_incremental
from stockregression.storage import Store


SYMBOL = "600000.SH"
ALL_DAYS = ["2026-01-05", "2026-02-02", "2026-03-02", "2026-04-01"]
TOUCH_DAYS = ALL_DAYS[:3]


def _service(tmp_path: Path) -> MarketDataService:
    service = MarketDataService(Store(tmp_path / "catalog"))
    bars = tmp_path / "bars.csv"
    pd.DataFrame({
        "date": ALL_DAYS,
        "open": [10.0] * len(ALL_DAYS),
        "high": [11.0] * len(ALL_DAYS),
        "low": [9.0] * len(ALL_DAYS),
        "close": [10.2] * len(ALL_DAYS),
        "volume": [24_000.0] * len(ALL_DAYS),
        "turnover": [240_000.0] * len(ALL_DAYS),
        "preclose": [10.0] * len(ALL_DAYS),
        "tradestatus": [1] * len(ALL_DAYS),
        "is_st": [0] * len(ALL_DAYS),
    }).to_csv(bars, index=False)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": ALL_DAYS}).to_csv(calendar, index=False)
    service.import_file(SYMBOL, "1day", bars, calendar_path=calendar)
    return service


def _evidence(service: MarketDataService, days: list[str]) -> dict:
    return validate_incremental(
        service, SYMBOL, "1day", mode="full", start=days[0], end=days[-1],
        dates=days, force_dates=True, persist_results=False, persist_archive=False,
    )


def _seed(service: MarketDataService) -> tuple[CoverageArchive, dict, dict[str, str]]:
    archive = CoverageArchive(service.repository)
    evidence = _evidence(service, ALL_DAYS)
    archive.commit_rows(
        evidence["_coverage_rows"],
        validation_rows=evidence["_validation_rows"],
        expected_data_version=1,
    )
    head = archive.load_head(SYMBOL, "1day")
    assert head is not None
    month_paths = {str(item["month"]): str(item["path"]) for item in head["months"]}
    return archive, evidence, month_paths


def _touch(archive: CoverageArchive, service: MarketDataService, *, days=TOUCH_DAYS):
    evidence = _evidence(service, list(days))
    result = archive.commit_rows(
        evidence["_coverage_rows"],
        validation_rows=evidence["_validation_rows"],
        expected_data_version=1,
    )
    return result, evidence


def _assert_all_dates_preserved(archive: CoverageArchive) -> None:
    rows = archive.rows(SYMBOL, "1day")
    assert {str(row["trade_date"])[:10] for row in rows} == set(ALL_DAYS)
    assert all(row["scanned"] for row in rows)


def test_multimonth_chunk_publishes_one_head_and_one_sqlite_transaction(tmp_path, monkeypatch):
    service = _service(tmp_path)
    archive, _, original_paths = _seed(service)
    evidence = _evidence(service, TOUCH_DAYS)
    publishes = 0
    begins = 0
    recorded_seconds = []
    original_publish = service.repository.package.publish_quality_archive
    original_connect = service.repository._connect
    monkeypatch.setattr(
        performance, "record_seconds",
        lambda stage, seconds: recorded_seconds.append((stage, seconds)),
    )

    def count_publish(*args, **kwargs):
        nonlocal publishes
        publishes += 1
        return original_publish(*args, **kwargs)

    class ConnectionProxy:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, *args, **kwargs):
            nonlocal begins
            if str(sql).strip().upper() == "BEGIN IMMEDIATE":
                begins += 1
            return self.connection.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    class ConnectionContext:
        def __enter__(self):
            self.context = original_connect()
            return ConnectionProxy(self.context.__enter__())

        def __exit__(self, exc_type, exc, traceback):
            return self.context.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(service.repository.package, "publish_quality_archive", count_publish)
    monkeypatch.setattr(service.repository, "_connect", ConnectionContext)
    result = archive.commit_rows(
        evidence["_coverage_rows"], validation_rows=evidence["_validation_rows"],
        expected_data_version=1,
    )

    assert result["_accepted_dates"] == TOUCH_DAYS
    assert publishes == 1
    assert begins == 1
    assert len(recorded_seconds) == 1
    assert recorded_seconds[0][0] == "ledger_publication"
    assert recorded_seconds[0][1] >= 0
    _assert_all_dates_preserved(archive)
    new_paths = {
        str(item["month"]): str(item["path"])
        for item in archive.load_head(SYMBOL, "1day")["months"]
    }
    assert new_paths["2026-04"] == original_paths["2026-04"]


@pytest.mark.parametrize("failure_point", ["month", "head", "mirror"])
def test_publish_fault_then_same_process_retry_keeps_all_published_dates(
    tmp_path, monkeypatch, failure_point,
):
    service = _service(tmp_path)
    archive, _, original_paths = _seed(service)
    evidence = _evidence(service, TOUCH_DAYS)
    original_reference = archive.head_reference(SYMBOL, "1day")

    if failure_point == "month":
        original_replace = Path.replace
        injected = False

        def fail_month_replace(path, target):
            nonlocal injected
            if not injected and "/quality/archives/" in target.as_posix() and "/months/" in target.as_posix():
                injected = True
                raise OSError("injected month payload failure")
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", fail_month_replace)
        with pytest.raises(OSError, match="month payload"):
            archive.commit_rows(
                evidence["_coverage_rows"], validation_rows=evidence["_validation_rows"],
                expected_data_version=1,
            )
        monkeypatch.setattr(Path, "replace", original_replace)
        assert injected
        assert archive.head_reference(SYMBOL, "1day") == original_reference

    elif failure_point == "head":
        original_publish = service.repository.package.publish_quality_archive
        injected = False

        def publish_then_fail(*args, **kwargs):
            nonlocal injected
            value = original_publish(*args, **kwargs)
            if not injected:
                injected = True
                raise RuntimeError("injected after head publication")
            return value

        monkeypatch.setattr(service.repository.package, "publish_quality_archive", publish_then_fail)
        with pytest.raises(RuntimeError, match="after head"):
            archive.commit_rows(
                evidence["_coverage_rows"], validation_rows=evidence["_validation_rows"],
                expected_data_version=1,
            )
        monkeypatch.setattr(service.repository.package, "publish_quality_archive", original_publish)
        assert injected
        assert archive.head_reference(SYMBOL, "1day")["revision"] > original_reference["revision"]

    else:
        original_record = archive.record_archive
        injected = False

        def fail_mirror(*args, **kwargs):
            nonlocal injected
            if not injected:
                injected = True
                raise RuntimeError("injected SQLite mirror failure")
            return original_record(*args, **kwargs)

        monkeypatch.setattr(archive, "record_archive", fail_mirror)
        with pytest.raises(RuntimeError, match="mirror failure"):
            archive.commit_rows(
                evidence["_coverage_rows"], validation_rows=evidence["_validation_rows"],
                expected_data_version=1,
            )
        assert injected
        assert archive.head_reference(SYMBOL, "1day")["revision"] > original_reference["revision"]

    # Retry on the same repository instance. The package head, not a stale
    # SQLite month, must carry forward the date rows from the first attempt.
    retried = archive.commit_rows(
        evidence["_coverage_rows"], validation_rows=evidence["_validation_rows"],
        expected_data_version=1,
    )
    assert retried["_accepted_dates"] == TOUCH_DAYS
    _assert_all_dates_preserved(archive)
    retry_revision = retried["revision"]

    # Publish a second time on the same process after recovery. Its merge base
    # must be the just-published package head, never the older SQLite mirror.
    published_again = archive.commit_rows(
        evidence["_coverage_rows"], validation_rows=evidence["_validation_rows"],
        expected_data_version=1,
    )
    assert published_again["revision"] > retry_revision
    _assert_all_dates_preserved(archive)
    paths = {
        str(item["month"]): str(item["path"])
        for item in archive.load_head(SYMBOL, "1day")["months"]
    }
    assert paths["2026-04"] == original_paths["2026-04"]


def test_reopen_after_mirror_failure_restores_head_before_retry(tmp_path, monkeypatch):
    service = _service(tmp_path)
    archive, _, original_paths = _seed(service)
    evidence = _evidence(service, TOUCH_DAYS)
    original_record = archive.record_archive
    injected = False

    def fail_mirror(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            raise RuntimeError("injected mirror failure before SQLite commit")
        return original_record(*args, **kwargs)

    monkeypatch.setattr(archive, "record_archive", fail_mirror)
    with pytest.raises(RuntimeError, match="mirror failure"):
        archive.commit_rows(
            evidence["_coverage_rows"], validation_rows=evidence["_validation_rows"],
            expected_data_version=1,
        )
    assert injected

    reopened = MarketDataService(Store(tmp_path / "catalog"))
    recovered = CoverageArchive(reopened.repository)
    retried, _ = _touch(recovered, reopened)
    assert retried["_accepted_dates"] == TOUCH_DAYS
    _assert_all_dates_preserved(recovered)
    paths = {
        str(item["month"]): str(item["path"])
        for item in recovered.load_head(SYMBOL, "1day")["months"]
    }
    assert paths["2026-04"] == original_paths["2026-04"]


def test_readonly_rows_reads_published_head_without_repair_or_file_changes(
    tmp_path, monkeypatch,
):
    service = _service(tmp_path)
    archive, _, _ = _seed(service)
    head = archive.load_head(SYMBOL, "1day")
    updated = next(
        row for row in archive._published_month_rows(head, {"2026-03"})["2026-03"]
        if str(row["trade_date"])[:10] == "2026-03-02"
    )
    updated["report"] = {**(updated.get("report") or {}), "published_only_marker": "head"}

    # Simulate a process that atomically published the new package head and
    # stopped before its SQLite mirror transaction.
    archive.persist(
        SYMBOL, "1day", expected_data_version=1,
        changed_months={"2026-03"}, _updates=[updated], _record_mirror=False,
    )
    published_head = archive.load_head(SYMBOL, "1day")
    assert published_head["revision"] > head["revision"]

    readonly = MarketDataRepository(
        service.store.root, package_dir=service.repository.package.root,
        market_settings=service.repository.market_settings, initialize=False,
    )
    readonly_archive = CoverageArchive(readonly)
    mirror_path = readonly.db_path
    with readonly._connect() as db:
        mirror_revision = db.execute(
            "SELECT revision FROM parquet_coverage_archives "
            "WHERE symbol=? AND timeframe=? AND adjustment=?",
            (SYMBOL, "1day", "none"),
        ).fetchone()["revision"]
    assert mirror_revision < published_head["revision"]

    def fail_if_locked():
        pytest.fail("read-only row lookup attempted to acquire package transaction lock")

    monkeypatch.setattr(readonly.package, "transaction_guard", fail_if_locked)
    package_before = {
        str(path.relative_to(readonly.package.root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in readonly.package.root.rglob("*") if path.is_file()
    }
    catalog_before = (mirror_path.read_bytes(), mirror_path.stat().st_mtime_ns)

    readonly_archive._ensure_mirror(SYMBOL, "1day", "none")
    rows = readonly_archive.rows(
        SYMBOL, "1day", start="2026-03-01", end="2026-03-31",
    )
    assert [str(row["trade_date"])[:10] for row in rows] == ["2026-03-02"]
    assert rows[0]["report"]["published_only_marker"] == "head"

    package_after = {
        str(path.relative_to(readonly.package.root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in readonly.package.root.rglob("*") if path.is_file()
    }
    catalog_after = (mirror_path.read_bytes(), mirror_path.stat().st_mtime_ns)
    assert package_after == package_before
    assert catalog_after == catalog_before

    # No hidden mirror repair took place: the stale reference remains stale.
    with readonly._connect() as db:
        after_revision = db.execute(
            "SELECT revision FROM parquet_coverage_archives "
            "WHERE symbol=? AND timeframe=? AND adjustment=?",
            (SYMBOL, "1day", "none"),
        ).fetchone()["revision"]
    assert after_revision == mirror_revision
