from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2

import pandas as pd
import pytest
import numpy as np
from fastapi.testclient import TestClient

from stockregression.market_data import MarketDataService
from stockregression.market_data.cache import ByteLRU, configure_cache
from stockregression.market_data.maintenance import reset_market_storage
from stockregression.market_data.package import import_etf_package, index_etf_package_direct
from stockregression.market_data.package import scan_etf_package
from stockregression.market_data.package_layout import resolve_package_root
from stockregression.market_data.models import DataVersionNotFound
from stockregression.models import BacktestConfig
from stockregression.share_engine import funding
from stockregression.storage import Store
from stockregression.api import create_app
from stockregression.jobs import execute_job
from stockregression.settings import save_settings


def _daily(days):
    return pd.DataFrame({
        "date": days, "open": 10.0, "high": 10.5, "low": 9.5,
        "close": 10.2, "volume": 24_000.0, "turnover": 240_000.0,
        "preclose": 10.0, "tradestatus": 1, "is_st": 0,
    })


def test_exchange_calendar_parallel_writes_preserve_all_dates(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    service = MarketDataService(Store(tmp_path / "data"))
    repository = service.repository
    days = pd.bdate_range("2026-01-05", periods=8)
    ready = Barrier(len(days))

    def write(index):
        symbol = f"{159500 + index}.SZ"
        ready.wait(timeout=15)
        repository.save_auxiliary(
            symbol, "calendar", [{"date": days[index]}],
            version=f"calendar-{index}", verified=True,
        )
        # Read alongside the remaining writers, including on Windows where
        # an open reader can otherwise prevent os.replace from succeeding.
        assert repository.auxiliary(symbol, "calendar")["payload"]

    with ThreadPoolExecutor(max_workers=len(days)) as pool:
        list(pool.map(write, range(len(days))))
    saved = repository.auxiliary("159516.SZ", "calendar")
    assert pd.DatetimeIndex([item["date"] for item in saved["payload"]]).equals(days)
    assert not list(repository.package_dir.rglob("*.writing"))


def test_repository_writes_parquet_and_cached_results_are_isolated(tmp_path):
    configure_cache(1)
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.bdate_range("2026-01-05", periods=3)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": days}).to_csv(calendar, index=False)
    path = tmp_path / "600000_daily.csv"
    _daily(days).to_csv(path, index=False)

    imported = service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    dataset = service.repository.dataset("600000.SH", "1day")
    first = service.repository.active_rows("600000.SH", "1day")
    first.loc[0, "close"] = -1
    second = service.repository.active_rows("600000.SH", "1day")

    assert imported["version"] == 1
    assert dataset["storage"] == "parquet-source"
    assert dataset["file_count"] == 1
    assert not any((tmp_path / "data" / "market").rglob("bars.sqlite"))
    assert len(list(service.repository.package_dir.rglob("*.parquet"))) >= 1
    assert second.iloc[0].close == pytest.approx(10.2)
    with pytest.raises(DataVersionNotFound, match="DATA_VERSION_NOT_FOUND"):
        service.repository.bars("600000.SH", "1day", version=99)


def test_cache_uses_byte_budget_and_bypasses_oversized_objects():
    cache = ByteLRU(1)
    large = pd.DataFrame({"value": np.arange(200_000, dtype="float64")})
    small = pd.DataFrame({"value": [1.0, 2.0]})
    loaded = cache.get_or_load("large", lambda: large)
    assert len(loaded) == len(large)
    assert cache.used_bytes == 0
    first = cache.get_or_load("small", lambda: small)
    first.loc[0, "value"] = 99
    second = cache.get_or_load("small", lambda: small)
    assert second.iloc[0].value == 1.0
    assert 0 < cache.used_bytes <= cache.budget_bytes


def test_incremental_writes_overlay_without_rewriting_base_or_losing_history(tmp_path):
    configure_cache(0)
    service = MarketDataService(Store(tmp_path / "data"))
    calendar = tmp_path / "calendar.csv"
    first_days = pd.DatetimeIndex(["2025-12-30", "2026-01-05"])
    pd.DataFrame({"date": first_days}).to_csv(calendar, index=False)
    path = tmp_path / "600000_daily.csv"
    _daily(first_days).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    package = service.repository.package_dir
    before = sorted(path.relative_to(package).as_posix() for path in package.rglob("*.parquet"))
    base_path = package / "stock_daily.parquet"
    base_stat = (base_path.stat().st_size, base_path.stat().st_mtime_ns)

    next_days = pd.DatetimeIndex(["2025-12-30", "2026-01-05", "2026-01-06"])
    pd.DataFrame({"date": next_days}).to_csv(calendar, index=False)
    _daily(next_days).to_csv(path, index=False)
    second = service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    after = sorted(path.relative_to(package).as_posix() for path in package.rglob("*.parquet"))

    assert second["version"] == 2
    assert "stock_daily.parquet" in before
    assert "stock_daily.parquet" in after
    assert (base_path.stat().st_size, base_path.stat().st_mtime_ns) == base_stat
    assert second["increment_count"] == 1
    assert list((package / ".stockregression" / "increments").rglob("*.parquet"))
    assert not list((package / ".stockregression" / "transactions").glob("*.json"))
    assert service.repository.bars("600000.SH", "1day", version=1).shape[0] == 2


def test_native_bundle_keeps_prior_close_for_share_accounting(tmp_path):
    configure_cache(0)
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.bdate_range("2026-01-05", periods=2)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": days}).to_csv(calendar, index=False)
    path = tmp_path / "600000_daily.csv"
    _daily(days).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)

    manifest = service.store.snapshot("600000.SH")
    config = BacktestConfig(
        symbol="600000.SH", start_date=days[1].date(), frequency="1d", schema_version=2,
    )
    bundle = service.load_backtest_bundle(config, manifest["version"])
    assert bundle.metadata["prior_closes"][str(days[0].date())] == pytest.approx(10.2)
    funding(config, bundle)


def _etf_package(root: Path) -> None:
    one_minute = root / "etf_1min"
    one_minute.mkdir(parents=True)
    day = pd.Timestamp("2026-01-05")
    pd.DataFrame([{
        "ts_code": "510050.SH", "cname": "Fixture ETF", "exchange": "SH",
        "list_date": "2009-01-05", "mgt_fee": 0.5, "etf_type": "stock",
    }]).to_parquet(root / "etf_basic_data.parquet", index=False)
    # Source daily units are lots and thousand yuan.
    pd.DataFrame([{
        "ts_code": "510050.SH", "trade_date": day, "open": 1.0, "high": 1.0,
        "low": 1.0, "close": 1.0, "pre_close": 1.0, "vol": 2.45,
        "amount": 0.245, "adj_factor": 1.0,
    }]).to_parquet(root / "etf_daily.parquet", index=False)
    stamps = [
        day + pd.Timedelta(hours=9, minutes=30),
        *pd.date_range(day + pd.Timedelta(hours=9, minutes=31), day + pd.Timedelta(hours=11, minutes=30), freq="min"),
        *pd.date_range(day + pd.Timedelta(hours=13, minutes=1), day + pd.Timedelta(hours=15), freq="min"),
    ]
    volume = [5.0] + [1.0] * (len(stamps) - 1)
    pd.DataFrame({
        "ts_code": ["510050.SH"] * len(stamps), "trade_time": stamps,
        "trade_date": [day] * len(stamps), "open": [1.0] * len(stamps),
        "high": [1.0] * len(stamps), "low": [1.0] * len(stamps),
        "close": [1.0] * len(stamps), "vol": volume, "amount": volume,
        "adj_factor": [1.0] * len(stamps),
    }).set_index(["trade_date", "trade_time"]).to_parquet(
        one_minute / "510050.SH.parquet"
    )


def test_dataset_index_resolves_asset_subpackage(tmp_path):
    dataset = tmp_path / "data" / "dataset"
    etf = dataset / "etf"
    etf.mkdir(parents=True)
    (dataset / "stock").mkdir()
    _etf_package(etf)

    assert resolve_package_root(dataset) == etf.resolve()
    assert scan_etf_package(dataset)["folder"] == str(etf.resolve())

    service = MarketDataService(Store(tmp_path / "app"), package_dir=dataset)
    assert service.repository.package_dir == etf.resolve()
    assert service.repository.package_info()["path"] == str(etf.resolve())

    (dataset / "stock" / "stock_daily.parquet").touch()
    with pytest.raises(ValueError, match="多个数据包"):
        resolve_package_root(dataset)


def test_settings_store_project_paths_relative(tmp_path, monkeypatch):
    import stockregression.settings as settings_module

    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", tmp_path / ".local" / "settings.json")
    monkeypatch.delenv("STOCKREGRESSION_DATA_DIR", raising=False)
    monkeypatch.delenv("STOCKREGRESSION_PACKAGE_DIR", raising=False)

    saved = save_settings({"data_dir": "data", "package_dir": "data/dataset"})

    assert saved.data_dir == str((tmp_path / "data").resolve())
    assert saved.package_dir == str((tmp_path / "data" / "dataset").resolve())
    stored = json.loads((tmp_path / ".local" / "settings.json").read_text(encoding="utf-8"))
    assert stored["paths"]["data_dir"] == "data"
    assert stored["paths"]["package_dir"] == "data/dataset"


def test_etf_package_normalizes_units_and_opening_auction(tmp_path):
    configure_cache(0)
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    service = MarketDataService(Store(tmp_path / "data"))
    result = import_etf_package(
        service, package, timeframes=["1day", "1min"],
        symbols=["510050.SH"], label_convention="auto",
    )

    daily = service.repository.active_rows("510050.SH", "1day")
    minute = service.repository.active_rows("510050.SH", "1min")
    assert result["status"] == "completed"
    assert daily.iloc[0].volume == pytest.approx(245)
    assert daily.iloc[0].turnover == pytest.approx(245)
    assert len(minute) == 240
    assert minute.iloc[0].timestamp == pd.Timestamp("2026-01-05 09:31")
    assert minute.volume.sum() == pytest.approx(245)
    assert minute.turnover.sum() == pytest.approx(245)
    assert service.repository.dataset("510050.SH", "1min")["status"] == "ready"


def test_reset_removes_market_versions_and_preserves_results(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.bdate_range("2026-01-05", periods=2)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": days}).to_csv(calendar, index=False)
    path = tmp_path / "600000_daily.csv"
    _daily(days).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    store = Store(tmp_path / "data")
    store.add_watch("600000.SH", "Fixture")
    result_id = store.save_result("backtest", {"status": "completed"}, "600000.SH", "Fixture result")
    package = tmp_path / "source"
    package.mkdir()
    _etf_package(package)

    report = reset_market_storage(store.root, source_folder=package)

    assert report["status"] == "completed"
    assert service.repository.dataset("600000.SH", "1day") is None
    assert not any((tmp_path / "data" / "market").rglob("*.parquet"))
    assert store.result(result_id)["status"] == "completed"
    assert [row["symbol"] for row in store.watches()] == ["600000.SH"]


def test_etf_package_api_scans_and_submits_parquet_import(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    root = tmp_path / "data"
    app = create_app(str(root), start_jobs=False)
    with TestClient(app) as client:
        scanned = client.post(
            "/api/v1/data/import-package/scan",
            json={"folder": str(package), "format": "etf"},
        )
        assert scanned.status_code == 200, scanned.text
        assert scanned.json()["timeframes"]["1min"]["symbols"] == 1
        submitted = client.post(
            "/api/v1/data/import-package",
            json={
                "folder": str(package), "format": "etf",
                "timeframes": ["1day", "1min"], "symbols": ["510050.SH"],
            },
        )
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job_id"]
        execute_job(str(root), job_id)
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
        assert client.get("/api/v1/data/coverage/510050.SH?timeframe=1min").json()["storage"] == "parquet-source"


def test_direct_package_index_reads_original_files_without_copying(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    root = tmp_path / "data"
    service = MarketDataService(Store(root))
    source_file = package / "etf_1min" / "510050.SH.parquet"
    before = source_file.stat().st_mtime_ns
    result = index_etf_package_direct(
        service, package, timeframes=["1day", "1min"], symbols=["510050.SH"],
    )
    minute = service.repository.active_rows("510050.SH", "1min")

    assert result["mode"] == "direct_parquet"
    assert service.repository.dataset("510050.SH", "1min")["storage"] == "parquet-source"
    assert len(minute) == 240
    assert minute.volume.sum() == pytest.approx(245)
    assert not any((root / "market").rglob("*.parquet"))
    assert source_file.stat().st_mtime_ns == before
    active = service.store.snapshot("510050.SH")
    assert active["market_data"] is True
    assert service.store.snapshot("510050.SH", active["version"]) is not None


def test_direct_source_commit_preserves_base_and_writes_overlay(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    root = tmp_path / "data"
    service = MarketDataService(Store(root))
    index_etf_package_direct(
        service, package, timeframes=["1day", "1min"], symbols=["510050.SH"],
    )
    source_daily = package / "etf_daily.parquet"
    before = source_daily.stat().st_mtime_ns
    days = pd.DatetimeIndex(["2026-01-06"])
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": days}).to_csv(calendar, index=False)
    path = tmp_path / "new_daily.csv"
    _daily(days).to_csv(path, index=False)

    service.import_file("510050.SH", "1day", path, calendar_path=calendar)

    assert len(service.repository.active_rows("510050.SH", "1day")) == 2
    assert source_daily.stat().st_mtime_ns == before
    overlays = list((package / ".stockregression" / "increments").rglob("*.parquet"))
    assert len(overlays) == 1
    assert not any((root / "market").rglob("*.parquet"))


def test_partial_registration_is_idempotent_and_preserves_other_periods(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    service = MarketDataService(Store(tmp_path / "data"))

    first = index_etf_package_direct(
        service, package, timeframes=["1day", "1min"], symbols=["510050.SH"],
    )
    assert first["changed"] == 2
    day_before = service.repository.dataset("510050.SH", "1day")

    second = index_etf_package_direct(
        service, package, timeframes=["1min"], symbols=["510050.SH"],
    )
    assert second["unchanged"] == 1 and second["changed"] == 0
    day_after = service.repository.dataset("510050.SH", "1day")
    minute = service.repository.dataset("510050.SH", "1min")
    assert day_after["current_version"] == day_before["current_version"]
    assert day_after["row_count"] == day_before["row_count"]
    assert minute is not None


def test_shared_daily_file_freezes_other_symbol_snapshot(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    first_day = pd.DatetimeIndex(["2026-01-05"])
    two_days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": two_days}).to_csv(calendar, index=False)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _daily(first_day).to_csv(first, index=False)
    _daily(two_days).to_csv(second, index=False)
    service.import_file("600000.SH", "1day", first, calendar_path=calendar)
    service.import_file("000001.SZ", "1day", first, calendar_path=calendar)
    old = service.store.snapshot("000001.SZ")

    service.import_file("600000.SH", "1day", second, calendar_path=calendar)

    old_version = int(old["market_versions"]["1day"]["version"])
    assert len(service.repository.bars("000001.SZ", "1day", version=old_version)) == 1
    assert len(service.repository.bars("000001.SZ", "1day")) == 1
    current = service.repository.dataset("000001.SZ", "1day")
    assert current["current_version"] == old_version
    assert len(service.repository.bars("600000.SH", "1day")) == 2


def test_inventory_and_minute_coverage_use_dates_not_rows(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    service = MarketDataService(Store(tmp_path / "data"))
    index_etf_package_direct(service, package, timeframes=["1day", "1min"], symbols=["510050.SH"])

    coverage = service.compatibility_coverage("510050.SH")
    inventory = service.inventory(limit=50, offset=0)

    assert coverage["diagnostics"]["minute_days"] == 1
    assert inventory["count"] == 1 and inventory["dataset_count"] == 2
    assert set(inventory["items"][0]["periods"]) == {"1day", "1min"}
    assert inventory["stats"]["bytes"] > 0


def test_package_preparation_validates_history_with_only_a_recent_calendar(tmp_path, monkeypatch):
    from stockregression.market_data.preparation import MarketPreparationService
    from stockregression.market_data.sync import MarketDataSyncService

    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    daily_path = package / "etf_daily.parquet"
    minute_path = package / "etf_1min" / "510050.SH.parquet"
    daily = pd.read_parquet(daily_path)
    minute = pd.read_parquet(minute_path).reset_index()
    offsets = [0, 1, 2, 3, 4, 7]  # Includes the Friday before a weekend start.
    daily_parts, minute_parts = [], []
    for offset in offsets:
        part = daily.copy()
        part["trade_date"] += pd.Timedelta(days=offset)
        daily_parts.append(part)
        part = minute.copy()
        part["trade_date"] += pd.Timedelta(days=offset)
        part["trade_time"] += pd.Timedelta(days=offset)
        minute_parts.append(part)
    pd.concat(daily_parts).to_parquet(daily_path, index=False)
    pd.concat(minute_parts).set_index(["trade_date", "trade_time"]).to_parquet(minute_path)
    original_bytes = minute_path.read_bytes()
    service = MarketDataService(Store(tmp_path / "data"))
    index_etf_package_direct(service, package, timeframes=["1day", "1min"], symbols=["510050.SH"])
    # Simulate the imported package carrying only its latest auxiliary increment.
    service.repository.save_auxiliary("510050.SH", "calendar", [{"date": "2026-01-12"}], verified=True)
    service.repository.save_auxiliary("510050.SH", "actions", [], verified=True)
    monkeypatch.setattr(MarketDataSyncService, "sync", lambda *args, **kwargs: pytest.fail("quality-only preparation must stay local"))
    config = BacktestConfig(
        schema_version=3, symbol="510050.SH", start_date=pd.Timestamp("2026-01-06").date(),
        end_date=pd.Timestamp("2026-01-12").date(), frequency="1m",
    )
    preparation = MarketPreparationService(service)
    plan = preparation.plan(config)
    assert [(step["kind"], step.get("timeframe")) for step in plan["steps"]] == [
        ("validate", "1day"), ("validate", "1min"),
    ]
    progress = []
    result = preparation.execute(config, progress=lambda fraction, message: progress.append((fraction, message)))
    assert result["status"] == "completed", result
    assert result["remaining_gaps"] == []
    assert service.readiness(config)["ready"]
    assert any("/" in message and "校验" in message for _, message in progress)
    assert [value for value, _ in progress] == sorted(value for value, _ in progress)
    assert minute_path.read_bytes() == original_bytes
    assert preparation.execute(config)["mode"] == "already_ready"
    weekend_config = config.model_copy(update={"start_date": pd.Timestamp("2026-01-10").date()})
    assert service.readiness(weekend_config)["ready"]
    daily_dataset = service.repository.dataset("510050.SH", "1day")
    assert not service._validation_covers(daily_dataset, "2026-01-06", "2026-01-13")
    # A date present in imported daily history must remain pending if its minute
    # validation is absent, even though it predates the auxiliary calendar.
    service.repository.invalidate_day_validations("510050.SH", "1min", ["2026-01-07"])
    state = service.repository.validation_state("510050.SH", "1min", start="2026-01-06", end="2026-01-12")
    assert "2026-01-07" in state["pending"]
    inventory = service.inventory(limit=50, offset=0)
    minute_inventory = inventory["items"][0]["periods"]["1min"]
    assert "2026-01-07" in minute_inventory["validation"]["pending"]
    assert not service.readiness(config)["ready"]


def test_revalidation_is_bounded_and_reuses_month_reads(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    service = MarketDataService(Store(tmp_path / "data"))
    index_etf_package_direct(service, package, timeframes=["1day", "1min"], symbols=["510050.SH"])
    reads = []
    original = service.repository.active_rows

    def read(symbol, timeframe, *args, **kwargs):
        if timeframe == "1min":
            reads.append((kwargs.get("start"), kwargs.get("end")))
        return original(symbol, timeframe, *args, **kwargs)

    monkeypatch.setattr(service.repository, "active_rows", read)
    cancelled = lambda: True
    with pytest.raises(InterruptedError):
        service.revalidate("510050.SH", "1min", start="2026-01-05", end="2026-01-05", cancelled=cancelled)
    assert not reads
    service.revalidate("510050.SH", "1min", start="2026-01-05", end="2026-01-05")
    assert reads and all(start is not None and end is not None for start, end in reads)
    reads.clear()
    monkeypatch.setattr(
        service.repository, "bars",
        lambda *args, **kwargs: pytest.fail("clean validation must not rescan bars"),
    )
    monkeypatch.setattr(
        service.repository, "active_rows",
        lambda *args, **kwargs: pytest.fail("clean validation must not rescan source rows"),
    )
    cached = service.revalidate("510050.SH", "1min", start="2026-01-05", end="2026-01-05")
    assert cached["report"]["cached"] is True
    assert reads == []


def test_failed_increment_publication_keeps_previous_version(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    first_day = pd.DatetimeIndex(["2026-01-05"])
    two_days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": two_days}).to_csv(calendar, index=False)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _daily(first_day).to_csv(first, index=False)
    _daily(two_days).to_csv(second, index=False)
    service.import_file("600000.SH", "1day", first, calendar_path=calendar)
    package_file = service.repository.package_file_path("600000.SH", "1day")
    original = package_file.read_bytes()

    def fail_publish(*_args, **_kwargs):
        raise OSError("simulated increment publication failure")

    monkeypatch.setattr(service.repository, "_write_increment_file", fail_publish)
    with pytest.raises(OSError, match="simulated"):
        service.import_file("600000.SH", "1day", second, calendar_path=calendar)

    recovered = MarketDataService(Store(tmp_path / "data"))
    assert recovered.repository.package_file_path("600000.SH", "1day").read_bytes() == original
    assert recovered.repository.dataset("600000.SH", "1day")["current_version"] == 1


def test_source_reregistration_reuses_unchanged_day_validation(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    service = MarketDataService(Store(tmp_path / "data"))
    index_etf_package_direct(
        service, package, timeframes=["1day"], symbols=["510050.SH"],
    )
    service.revalidate("510050.SH", "1day")
    before = service.repository.day_validations("510050.SH", "1day")
    assert before and before[0]["status"] == "passed"
    checked_at = before[0]["checked_at"]
    service.repository.archive_dataset("510050.SH", "1day")

    daily_path = package / "etf_daily.parquet"
    daily = pd.read_parquet(daily_path)
    extra = daily.iloc[[0]].copy()
    extra["trade_date"] = pd.Timestamp("2026-01-06")
    pd.concat([daily, extra], ignore_index=True).to_parquet(daily_path, index=False)
    index_etf_package_direct(
        service, package, timeframes=["1day"], symbols=["510050.SH"],
    )

    after = service.repository.day_validations("510050.SH", "1day")
    assert len(after) == 1
    assert after[0]["trade_date"] == "2026-01-05"
    assert after[0]["checked_at"] == checked_at
    assert service.repository.dataset("510050.SH", "1day")["quality_status"] == "partial"


def test_default_package_initialization_indexes_only_base_timeframes(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    (package / "etf_5min").mkdir()
    copy2(package / "etf_1min" / "510050.SH.parquet", package / "etf_5min" / "510050.SH.parquet")
    service = MarketDataService(Store(tmp_path / "data"))

    result = index_etf_package_direct(service, package, symbols=["510050.SH"])

    assert result["changed"] == 2
    assert service.repository.dataset("510050.SH", "1day") is not None
    assert service.repository.dataset("510050.SH", "1min") is not None
    assert service.repository.dataset("510050.SH", "5min") is None
    assert not list((package / ".stockregression" / "bases").rglob("*.parquet"))
    assert not list((package / ".stockregression" / "increments").rglob("*.parquet"))


def test_package_registration_indexes_stock_layout(tmp_path):
    package = tmp_path / "package"
    (package / "stock_1min").mkdir(parents=True)
    day = pd.Timestamp("2026-01-05")
    pd.DataFrame([{
        "ts_code": "600000.SH", "cname": "Fixture Stock", "exchange": "SH",
        "list_date": "1999-11-10",
    }]).to_parquet(package / "stock_basic_data.parquet", index=False)
    pd.DataFrame([{
        "ts_code": "600000.SH", "trade_date": day, "open": 10.0, "high": 10.2,
        "low": 9.9, "close": 10.1, "pre_close": 10.0, "vol": 100.0,
        "amount": 101.0, "adj_factor": 1.0,
    }]).to_parquet(package / "stock_daily.parquet", index=False)
    stamps = [
        *pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min"),
        *pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min"),
    ]
    pd.DataFrame({
        "ts_code": "600000.SH", "trade_time": stamps,
        "trade_date": [day] * len(stamps), "open": 10.1, "high": 10.1,
        "low": 10.1, "close": 10.1, "vol": 1.0, "amount": 10.1,
        "adj_factor": 1.0,
    }).to_parquet(package / "stock_1min" / "600000.SH.parquet", index=False)
    service = MarketDataService(Store(tmp_path / "data"))

    result = index_etf_package_direct(
        service, package, timeframes=["1day", "1min"], symbols=["600000.SH"],
    )

    assert result["changed"] == 2 and result["errors"] == []
    assert service.store.instrument("600000.SH")["asset_type"] == "stock"
    assert service.repository.dataset("600000.SH", "1day")["storage"] == "parquet-source"
    assert len(service.repository.bars("600000.SH", "1min")) == 240


def test_external_package_file_change_invalidates_current_quality(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    day = pd.DatetimeIndex(["2026-01-05"])
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": day}).to_csv(calendar, index=False)
    source = tmp_path / "source.csv"
    _daily(day).to_csv(source, index=False)
    service.import_file("600000.SH", "1day", source, calendar_path=calendar)
    assert service.repository.dataset("600000.SH", "1day")["file_status"] == "readable"

    package_file = service.repository.package_file_path("600000.SH", "1day")
    externally_changed = pd.read_parquet(package_file)
    externally_changed["close"] = externally_changed["close"] + 1
    externally_changed.to_parquet(package_file, index=False)

    dataset = service.repository.dataset("600000.SH", "1day")
    assert dataset["file_status"] == "changed"
    assert dataset["status"] == "partial"
