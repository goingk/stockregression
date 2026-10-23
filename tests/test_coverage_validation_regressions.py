import pandas as pd
from types import SimpleNamespace

from stockregression.market_data import validation


class _CalendarRepository:
    def auxiliary(self, symbol, kind):
        return None

    def _calendar_dates(self, records):
        return pd.DatetimeIndex([])


class _CalendarService:
    repository = _CalendarRepository()


def test_prefetched_dates_remain_targets_without_becoming_calendar_facts(monkeypatch):
    class Archive:
        def __init__(self, repository):
            pass

        def rows(self, *args, **kwargs):
            return [{"trade_date": "2024-01-02"}]

        def load_head(self, *args):
            return {"revision": 1}

    monkeypatch.setattr(validation, "CoverageArchive", Archive)
    dates, available, calendar = validation._expected_days(
        _CalendarService(), "600000.SH", "1day", "2024-01-01", "2024-01-05", "none",
        prefetched_dates=pd.DatetimeIndex(["2024-01-03", "2024-01-05"]),
    )

    assert available is False
    assert calendar.empty
    assert dates.equals(pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-05"]))
    assert calendar.empty


def test_daily_dependencies_read_each_year_shard_once_for_sparse_targets():
    class Repository:
        def __init__(self):
            self.reads = []

        def _storage_period(self, timeframe, day):
            year = pd.Timestamp(day).year
            return (
                f"year={year}", pd.Timestamp(f"{year}-01-01"),
                pd.Timestamp(f"{year + 1}-01-01"),
            )

        def active_rows(self, symbol, timeframe, *, adjustment, start, end):
            self.reads.append((start, end))
            return pd.DataFrame({"timestamp": [start, end]})

    repository = Repository()
    targets = [pd.Timestamp("2023-01-04"), pd.Timestamp("2023-09-12"), pd.Timestamp("2025-03-03")]

    result = validation._read_daily_dependencies(repository, "600000.SH", targets, "none")

    assert repository.reads == [
        (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
    ]
    assert len(result) == 4


def test_corrupt_archive_month_only_targets_dates_in_that_month(monkeypatch):
    days = pd.DatetimeIndex(["2024-01-02", "2024-02-01"])

    class Repository:
        def dataset(self, *args):
            return {
                "first_time": days[0], "last_time": days[-1],
                "current_version": 7, "adjustment": "none",
            }

        def auxiliary(self, symbol, kind):
            return {"payload": [day.date().isoformat() for day in days]} if kind == "calendar" else None

        def _calendar_dates(self, records):
            return pd.DatetimeIndex(records)

        def market_file_state(self, *args):
            return {"status": "readable"}

    class Archive:
        def __init__(self, repository):
            pass

        def rows(self, *args, **kwargs):
            return [
                {"trade_date": str(day.date()), "scanned": True,
                 "rules_version": validation.VALIDATION_RULES_VERSION,
                 "dependency_hash": None}
                for day in days
            ]

        def load_head(self, *args):
            return {"revision": 2}

        def validation_head(self, *args, **kwargs):
            return {"revision": 2, "data_version": 7, "_invalid_months": ["2024-01"]}

        def load_for_validation(self, *args, **kwargs):
            return self.rows()

        @staticmethod
        def pending_dates(*args):
            return set()

    service = SimpleNamespace(repository=Repository())
    monkeypatch.setattr(validation, "CoverageArchive", Archive)

    selection = validation.validation_targets(
        service, "600000.SH", "1day", start=days[0], end=days[-1],
    )

    assert selection["targets"] == [days[0]]
    assert selection["reused_days"] == 1


def test_corrupt_month_without_calendar_rebuilds_only_its_manifest_date_range(monkeypatch):
    class Repository:
        def dataset(self, *args):
            return {
                "first_time": "2024-01-01", "last_time": "2024-02-02",
                "current_version": 7, "adjustment": "none",
            }

        def auxiliary(self, symbol, kind):
            return None

        def _calendar_dates(self, records):
            return pd.DatetimeIndex([])

        def market_file_state(self, *args):
            return {"status": "readable"}

    good_day = "2024-02-01"
    head = {
        "revision": 3, "data_version": 7, "_invalid_months": ["2024-01"],
        "months": [
            {"month": "2024-01", "first_day": "2024-01-02", "last_day": "2024-01-04"},
            {"month": "2024-02", "first_day": good_day, "last_day": good_day},
        ],
    }

    class Archive:
        def __init__(self, repository):
            pass

        def rows(self, *args, **kwargs):
            return [{"trade_date": good_day}]

        def load_head(self, *args):
            return head

        def validation_head(self, *args, **kwargs):
            return head

        def load_for_validation(self, *args, **kwargs):
            return [{"trade_date": good_day, "scanned": True, "rules_version":
                     validation.VALIDATION_RULES_VERSION, "dependency_hash": None}]

        @staticmethod
        def pending_dates(*args):
            return set()

    monkeypatch.setattr(validation, "CoverageArchive", Archive)
    selection = validation.validation_targets(
        SimpleNamespace(repository=Repository()), "600000.SH", "1day",
        start="2024-01-01", end="2024-02-02",
    )

    assert selection["calendar_available"] is False
    assert selection["trading_days"].empty
    assert selection["targets"] == list(pd.date_range("2024-01-02", "2024-01-04"))
    assert selection["reused_days"] == 1


def test_minute_validation_reads_each_month_shard_once(monkeypatch):
    days = pd.DatetimeIndex(["2024-01-31", "2024-02-01"])
    reads = []
    events = []

    class Repository:
        def dataset(self, symbol, timeframe, adjustment="none"):
            events.append(("dataset", timeframe))
            return {"current_version": 9} if timeframe == "1min" else {"current_version": 4}

        def active_rows(self, symbol, timeframe, *, adjustment, start, end):
            if timeframe == "1day":
                events.append(("read", "1day"))
                return pd.DataFrame({
                    "timestamp": days,
                    "tradestatus": [1, 1],
                    "volume": [100, 100],
                    "close": [10.0, 10.0],
                })
            reads.append((start, end))
            month = pd.Timestamp(start)
            stamp = pd.Timestamp("2024-01-31 09:31" if month.month == 1 else "2024-02-01 09:31")
            return pd.DataFrame({
                "timestamp": [stamp], "open": [10.0], "high": [10.0],
                "low": [10.0], "close": [10.0], "volume": [1],
                "turnover": [10.0], "source_verified": [True],
                "is_synthetic": [False],
            })

        def _storage_period(self, timeframe, day):
            value = pd.Timestamp(day)
            if timeframe == "1day":
                return (f"year={value.year}", f"{value.year}-01-01", f"{value.year + 1}-01-01")
            start = value.normalize().replace(day=1)
            end = start + pd.offsets.MonthBegin(1)
            return (f"month={start:%Y-%m}", start.isoformat(), end.isoformat())

        def auxiliary(self, symbol, kind):
            return {"payload": [day.date().isoformat() for day in days]} if kind == "calendar" else None

    service = SimpleNamespace(repository=Repository())
    selection = {
        "dataset": {"current_version": 9, "adjustment": "none"},
        "lower": days[0], "upper": days[-1], "calendar_days": days,
        "trading_days": days, "targets": list(days), "calendar_available": True,
        "reused_days": 0, "archive": {"revision": 1, "data_version": 9},
        "file_state": {"status": "readable"}, "has_pending_scope": True,
    }

    class Archive:
        def __init__(self, repository):
            pass

    class ValidationResult:
        report = {"status": "complete", "expected_rows": 1, "observed_rows": 1}
        pending = []
        details = {}

    monkeypatch.setattr(validation, "CoverageArchive", Archive)
    monkeypatch.setattr(validation, "validate_and_fill", lambda *args, **kwargs: ValidationResult())
    monkeypatch.setattr(validation, "anomalies_in_range", lambda *args, **kwargs: pd.DataFrame())

    result = validation.validate_incremental(
        service, "600000.SH", "1min", selection=selection, persist_results=False,
    )

    assert result["checked_days"] == 2
    assert len(reads) == 2
    assert [pd.Timestamp(start).month for start, _ in reads] == [1, 2]
    assert result["market_data_read_calls"] == 2
    assert result["market_data_shard_reads"] == 2
    assert result["daily_dependency_read_calls"] == 1
    assert result["daily_dependency_shard_reads"] == 1
    assert events.index(("dataset", "1day")) < events.index(("read", "1day"))

    reads.clear()
    january_preloaded = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-31 09:31")],
        "open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0],
        "volume": [1], "turnover": [10.0], "source_verified": [True],
        "is_synthetic": [False],
    })
    preloaded_result = validation.validate_incremental(
        service, "600000.SH", "1min", selection=selection, persist_results=False,
        preloaded_months={"2024-01": january_preloaded},
    )
    assert preloaded_result["market_data_read_calls"] == 1
    assert preloaded_result["market_data_shard_reads"] == 1
    assert [pd.Timestamp(start).month for start, _ in reads] == [2]
    assert preloaded_result["daily_dependency_read_calls"] == 1

    reads.clear()
    february_preloaded = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-02-01 09:31")],
        "open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0],
        "volume": [1], "turnover": [10.0], "source_verified": [True],
        "is_synthetic": [False],
    })
    storage_preloaded_result = validation.validate_incremental(
        service, "600000.SH", "1min", selection=selection, persist_results=False,
        preloaded_months={
            "month=2024-01": january_preloaded,
            "month=2024-02": february_preloaded,
        },
    )
    assert storage_preloaded_result["market_data_read_calls"] == 0
    assert storage_preloaded_result["market_data_shard_reads"] == 0
    assert reads == []


def test_serial_commit_and_stale_retry_reads_are_included_in_metrics(monkeypatch):
    day = pd.Timestamp("2024-03-04")
    read_periods = []
    commit_count = 0

    class Repository:
        def active_rows(self, symbol, timeframe, *, adjustment, start, end):
            read_periods.append((start, end))
            return pd.DataFrame({
                "timestamp": [pd.Timestamp("2024-03-04 00:00")],
                "open": [10.0], "high": [10.0], "low": [10.0],
                "close": [10.0], "volume": [1], "turnover": [10.0],
                "preclose": [10.0], "tradestatus": [1],
                "source_verified": [True], "is_synthetic": [False],
            })

        def _storage_period(self, timeframe, stamp):
            return ("year=2024", "2024-01-01", "2025-01-01")

        def auxiliary(self, symbol, kind):
            return None

        def validation_state(self, *args, **kwargs):
            return {"verified_from": str(day.date()), "verified_through": str(day.date()),
                    "failed": [], "pending": []}

        def save_coverage(self, *args, **kwargs):
            pass

        def update_dataset_status(self, *args, **kwargs):
            pass

    class Archive:
        def __init__(self, repository):
            pass

        def commit_rows(self, *args, **kwargs):
            nonlocal commit_count
            commit_count += 1
            return {
                "revision": commit_count,
                "_stale_dates": [str(day.date())] if commit_count == 1 else [],
                "_evidence_read_calls": 2 if commit_count == 1 else 1,
            }

    selection = {
        "dataset": {"current_version": 1, "adjustment": "none"},
        "lower": day, "upper": day, "calendar_days": pd.DatetimeIndex([day]),
        "trading_days": pd.DatetimeIndex([day]), "targets": [day],
        "calendar_available": True, "reused_days": 0,
        "archive": {"revision": 1, "data_version": 1},
        "file_state": {"status": "readable"}, "has_pending_scope": True,
    }
    service = SimpleNamespace(
        repository=Repository(),
        store=SimpleNamespace(instrument=lambda symbol: {"asset_type": "stock"}),
    )

    class ValidationResult:
        report = {"status": "complete", "expected_rows": 1, "observed_rows": 1}
        pending = []
        details = {}

    monkeypatch.setattr(validation, "CoverageArchive", Archive)
    monkeypatch.setattr(validation, "validation_targets", lambda *args, **kwargs: selection)
    monkeypatch.setattr(validation, "validate_and_fill", lambda *args, **kwargs: ValidationResult())
    monkeypatch.setattr(validation, "anomalies_in_range", lambda *args, **kwargs: pd.DataFrame())

    result = validation.validate_incremental(
        service, "600000.SH", "1day", selection=selection,
    )

    assert commit_count == 2
    assert len(read_periods) == 2
    assert result["market_data_read_calls"] == 5
    assert result["market_data_shard_reads"] == 5
    assert result["evidence_read_calls"] == 3
