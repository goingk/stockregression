"""Offline benchmark for persisted daily validation and batched ledger commits.

Run with ``.venv/Scripts/python.exe scripts/benchmark_daily_validation.py``.
All catalogs, market files, and calendars are created under a temporary directory.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

import pandas as pd

from stockregression.market_data import MarketDataService
from stockregression.market_data.validation import validate_incremental
from stockregression.storage import Store


SYMBOL = "600000.SH"
TRADING_DAYS = 1_250


def _bars(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": dates,
        "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1,
        "volume": 100_000.0, "turnover": 1_010_000.0,
        "preclose": 10.0, "tradestatus": 1, "is_st": 0,
        "source": "offline-benchmark", "source_verified": True,
        "is_synthetic": False,
    })


def _seed_local_repository(service: MarketDataService, dates: pd.DatetimeIndex) -> None:
    service.store.put_instrument(SYMBOL, "Offline benchmark", asset_type="stock")
    service.repository.save_auxiliary(
        SYMBOL, "calendar", [{"date": day} for day in dates],
        version="offline-benchmark", verified=True,
    )
    service.repository.commit_bars(
        SYMBOL, "1day", _bars(dates), source="offline-benchmark",
        file_name="synthetic-fixed-daily.csv",
    )


def _count_publications(service: MarketDataService):
    package = service.repository.package
    original = package.publish_quality_archive
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    package.publish_quality_archive = counted

    def restore():
        package.publish_quality_archive = original

    return calls, restore


def _run_bootstrap(root: Path, *, fast_path: bool) -> dict:
    dates = pd.bdate_range("2019-01-02", periods=TRADING_DAYS)
    service = MarketDataService(Store(root / "catalog"))
    _seed_local_repository(service, dates)
    publications, restore = _count_publications(service)
    started = time.perf_counter()
    try:
        report = validate_incremental(
            service, SYMBOL, "1day", mode="full",
            start=dates[0], end=dates[-1], dates=dates, force_dates=True,
            persist_results=True, persist_archive=True,
            _daily_fast_path=fast_path,
        )
    finally:
        elapsed = time.perf_counter() - started
        restore()
    return {
        "seconds": elapsed,
        "publications": len(publications),
        "checked_days": report.get("checked_days"),
        "status": report.get("status"),
        "failed_days": report.get("failed_days"),
        "service": service,
        "dates": dates,
    }


def _run_incremental_check(bootstrap: dict, root: Path) -> dict:
    service = bootstrap["service"]
    next_day = pd.bdate_range(bootstrap["dates"][-1] + pd.Timedelta(days=1), periods=1)[0]
    repository = service.repository
    original_active_rows = repository.active_rows
    original_read_paths = repository._read_paths
    reads = []
    phase = "incremental_commit"

    def tracked_active_rows(symbol, timeframe, *, start=None, end=None, **kwargs):
        frame = original_active_rows(
            symbol, timeframe, start=start, end=end, **kwargs,
        )
        actual = pd.to_datetime(frame.get("timestamp"), errors="coerce") if not frame.empty else pd.DatetimeIndex([])
        reads.append({
            "reader": "active_rows", "phase": phase,
            "requested_start": str(pd.Timestamp(start).date()) if start is not None else None,
            "requested_end": str(pd.Timestamp(end).date()) if end is not None else None,
            "returned_min": str(actual.min().date()) if len(actual) else None,
            "returned_max": str(actual.max().date()) if len(actual) else None,
            "rows": len(frame),
        })
        return frame

    def tracked_read_paths(paths, **kwargs):
        result = original_read_paths(paths, **kwargs)
        years = sorted({
            int(match.group(1)) for path in paths
            if (match := re.search(r"year=(\d{4})", str(path).replace("\\", "/")))
        })
        reads.append({
            "reader": "_read_paths", "phase": phase,
            "requested_start": str(pd.Timestamp(kwargs["start"]).date()) if kwargs.get("start") is not None else None,
            "requested_end": str(pd.Timestamp(kwargs["end"]).date()) if kwargs.get("end") is not None else None,
            "shard_years": years,
            "rows": len(result),
        })
        return result

    repository.active_rows = tracked_active_rows
    repository._read_paths = tracked_read_paths
    publications, restore = _count_publications(service)
    started = time.perf_counter()
    try:
        service.repository.save_auxiliary(
            SYMBOL, "calendar", [{"date": next_day}],
            version="offline-benchmark-incremental", verified=True,
        )
        service.repository.commit_bars(
            SYMBOL, "1day", _bars(pd.DatetimeIndex([next_day])),
            source="offline-benchmark-incremental",
            file_name="synthetic-one-day-append.csv",
        )
        phase = "incremental_validation"
        report = validate_incremental(
            service, SYMBOL, "1day", mode="incremental",
            start=next_day, end=next_day, dates=[next_day],
            force_dates=True,
            persist_results=True, persist_archive=True,
        )
    finally:
        elapsed = time.perf_counter() - started
        restore()
        repository.active_rows = original_active_rows
        repository._read_paths = original_read_paths

    unrelated = [
        item for item in reads
        if (
            any(year != next_day.year for year in item.get("shard_years", []))
            if item.get("shard_years") else any(
                value is not None and pd.Timestamp(value).year != next_day.year
                for value in (item.get("returned_min"), item.get("returned_max"))
            )
        )
    ]
    return {
        "seconds": elapsed,
        "publications": len(publications),
        "checked_days": report.get("checked_days"),
        "status": report.get("status"),
        "target_date": str(next_day.date()),
        "reads": reads,
        "unrelated_history_reads": unrelated,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stockregression-daily-bench-") as temporary:
        root = Path(temporary)
        legacy = _run_bootstrap(root / "legacy", fast_path=False)
        fast = _run_bootstrap(root / "fast", fast_path=True)
        incremental = _run_incremental_check(fast, root / "fast")

        speedup = legacy["seconds"] / fast["seconds"] if fast["seconds"] else float("inf")
        result = {
            "trading_days": TRADING_DAYS,
            "legacy_monthly_bootstrap": {
                "seconds": round(legacy["seconds"], 3),
                "publications": legacy["publications"],
                "checked_days": legacy["checked_days"],
                "status": legacy["status"],
            },
            "fast_batched_bootstrap": {
                "seconds": round(fast["seconds"], 3),
                "under_5_seconds": fast["seconds"] < 5,
                "publications": fast["publications"],
                "expected_publications": 3,
                "checked_days": fast["checked_days"],
                "status": fast["status"],
            },
            "speedup_multiple": round(speedup, 2),
            "incremental_one_day": {
                **incremental,
                "seconds": round(incremental["seconds"], 3),
                "reads": incremental["reads"],
                "unrelated_history_reads": incremental["unrelated_history_reads"],
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if fast["publications"] != 3:
            raise SystemExit("batched bootstrap should publish exactly 3 heads for 1250 days")
        if fast["seconds"] >= 5:
            raise SystemExit("batched bootstrap exceeded the 5-second target")
        if speedup < 5:
            raise SystemExit("batched bootstrap speedup was below 5x")
        if incremental["checked_days"] != 1 or incremental["publications"] != 1:
            raise SystemExit("single-day incremental validation should check and publish one day")
        if incremental["unrelated_history_reads"]:
            raise SystemExit("incremental validation read unrelated historical dates")


if __name__ == "__main__":
    main()
