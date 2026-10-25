from __future__ import annotations

import json
from pathlib import Path

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
from stockregression.configuration import current_configuration, patch_configuration
from stockregression.settings import load_settings


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


@pytest.mark.parametrize("existing_file", [False, True])
def test_basic_info_parallel_writes_preserve_all_symbols(tmp_path, existing_file):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    repository = MarketDataService(Store(tmp_path / "data")).repository
    symbols = [f"{159500 + index}.SZ" for index in range(8)]
    expected = {symbol: f"ETF {index}" for index, symbol in enumerate(symbols)}
    if existing_file:
        # Preserve an unrelated row while updating one existing symbol and
        # inserting the remaining symbols into the shared asset file.
        repository.save_basic_info("159999.SZ", {"cname": "Existing ETF"})
        repository.save_basic_info(symbols[0], {"cname": "Old name"})
        expected["159999.SZ"] = "Existing ETF"
    ready = Barrier(len(symbols))

    def write(symbol):
        ready.wait(timeout=15)
        return repository.save_basic_info(symbol, {"cname": expected[symbol]})

    with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
        paths = list(pool.map(write, symbols))

    assert len(set(paths)) == 1
    saved = pd.read_parquet(paths[0])
    assert len(saved) == len(expected)
    assert saved.ts_code.is_unique
    assert saved.set_index("ts_code").cname.to_dict() == expected
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
    assert dataset["storage"] == "parquet"
    assert dataset["file_count"] == 1
    assert not any((tmp_path / "data" / "market").rglob("bars.sqlite"))
    assert len(list((service.repository.package_dir / "partitions").rglob("*.parquet"))) == 1
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


def test_incremental_rewrites_only_the_affected_year_shard_and_keeps_history(tmp_path):
    configure_cache(0)
    service = MarketDataService(Store(tmp_path / "data"))
    calendar = tmp_path / "calendar.csv"
    first_days = pd.DatetimeIndex(["2025-12-30", "2026-01-05"])
    pd.DataFrame({"date": first_days}).to_csv(calendar, index=False)
    path = tmp_path / "600000_daily.csv"
    _daily(first_days).to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    package = service.repository.package_dir
    before = json.loads((package / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    first_head = before["datasets"]["600000.SH|1day|none"]
    first_dataset = json.loads((package / first_head["manifest"]).read_text(encoding="utf-8"))
    old_shards = {
        item["path"]: (package / item["path"]).stat().st_mtime_ns
        for item in first_dataset["shards"]
    }

    next_days = pd.DatetimeIndex(["2025-12-30", "2026-01-05", "2026-01-06"])
    pd.DataFrame({"date": next_days}).to_csv(calendar, index=False)
    _daily(next_days).to_csv(path, index=False)
    second = service.import_file("600000.SH", "1day", path, calendar_path=calendar)
    after = json.loads((package / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    second_dataset = json.loads((package / after["datasets"]["600000.SH|1day|none"]["manifest"]).read_text(encoding="utf-8"))

    assert second["version"] == 2
    current_paths = {item["path"] for item in second_dataset["shards"]}
    assert any("year=2025" in path for path in current_paths)
    assert any("year=2026" in path for path in current_paths)
    assert set(old_shards).issubset({path.relative_to(package).as_posix() for path in package.rglob("*.parquet")})
    assert all((package / path).stat().st_mtime_ns == mtime for path, mtime in old_shards.items())
    assert second["shard_count"] == 1
    assert list((package / "partitions").rglob("*.parquet"))
    assert not (package / ".stockregression" / "increments").exists()
    assert not list((package / ".stockregression" / "transactions").glob("*.json"))
    assert service.repository.bars("600000.SH", "1day", version=1).shape[0] == 2


def test_native_bundle_keeps_prior_close_for_share_accounting(tmp_path):
    configure_cache(0)
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.bdate_range("2026-01-05", periods=2)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": days}).to_csv(calendar, index=False)
    path = tmp_path / "600000_daily.csv"
    daily = _daily(days)
    # This fixture has no corporate actions: each subsequent preclose must
    # equal the preceding close, otherwise settlement correctly rejects it.
    daily["preclose"] = daily["close"].shift().fillna(daily["preclose"])
    daily.to_csv(path, index=False)
    service.import_file("600000.SH", "1day", path, calendar_path=calendar)

    manifest = service.store.snapshot("600000.SH")
    config = BacktestConfig(
        symbol="600000.SH", start_date=days[1].date(), frequency="1d", schema_version=2,
    )
    bundle = service.load_backtest_bundle(config, manifest["version"])
    assert bundle.metadata["prior_closes"][str(days[0].date())] == pytest.approx(10.2)
    funding(config, bundle)


def _etf_package(root: Path, days=None) -> None:
    """Build a schema-2 package with only manifest-published base shards."""
    root.mkdir(parents=True, exist_ok=True)
    days = pd.DatetimeIndex(days if days is not None else ["2026-01-05"])
    builder = MarketDataService(
        Store(root.parent / f"{root.name}-builder-data"), package_dir=root,
    )
    builder.repository.save_basic_info("510050.SH", {
        "cname": "Fixture ETF", "exchange": "SH", "list_date": "2009-01-05",
        "mgt_fee": 0.5, "etf_type": "stock",
    })
    daily = pd.DataFrame({
        "timestamp": days, "open": 1.0, "high": 1.0, "low": 1.0,
        "close": 1.0, "preclose": 1.0, "volume": 245.0, "turnover": 245.0,
        "tradestatus": 1, "is_st": False, "source": "fixture",
        "source_verified": True, "is_synthetic": False,
    })
    builder.repository.commit_bars("510050.SH", "1day", daily, source="fixture")
    stamps = [
        stamp + pd.Timedelta(hours=9, minutes=31)
        for day in days for stamp in pd.date_range(day, periods=120, freq="min")
    ]
    stamps += [
        stamp + pd.Timedelta(hours=13, minutes=1)
        for day in days for stamp in pd.date_range(day, periods=120, freq="min")
    ]
    stamps = sorted(stamps)
    volume = [6.0 if stamp.hour == 9 and stamp.minute == 31 else 1.0 for stamp in stamps]
    minute = pd.DataFrame({
        "timestamp": stamps, "open": 1.0, "high": 1.0, "low": 1.0,
        "close": 1.0, "volume": volume, "turnover": volume, "preclose": 1.0,
        "tradestatus": 1, "is_st": False, "source": "fixture",
        "source_verified": True, "is_synthetic": False,
    }).sort_values("timestamp").reset_index(drop=True)
    builder.repository.commit_bars("510050.SH", "1min", minute, source="fixture")


def _stock_package(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    builder = MarketDataService(
        Store(root.parent / f"{root.name}-builder-data"), package_dir=root,
    )
    builder.repository.save_basic_info("600000.SH", {
        "cname": "Fixture Stock", "exchange": "SH", "list_date": "1999-11-10",
    })
    day = pd.Timestamp("2026-01-05")
    daily = pd.DataFrame([{
        "timestamp": day, "open": 10.0, "high": 10.2, "low": 9.9,
        "close": 10.1, "preclose": 10.0, "volume": 240.0,
        "turnover": 2424.0, "tradestatus": 1, "is_st": False,
        "source": "fixture", "source_verified": True,
    }])
    builder.repository.commit_bars("600000.SH", "1day", daily, source="fixture")
    stamps = [
        *pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min"),
        *pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min"),
    ]
    minute = pd.DataFrame({
        "timestamp": stamps, "open": 10.1, "high": 10.1, "low": 10.1,
        "close": 10.1, "volume": 1.0, "turnover": 10.1, "preclose": 10.0,
        "tradestatus": 1, "is_st": False, "source": "fixture",
        "source_verified": True,
    })
    builder.repository.commit_bars("600000.SH", "1min", minute, source="fixture")


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

    stock_package = dataset / "stock"
    (stock_package / ".stockregression").mkdir(parents=True)
    (stock_package / ".stockregression" / "manifest.json").write_text(
        json.dumps({
            "schema_version": 2, "package_id": "stock-package",
            "generation": "stock-generation", "datasets": {},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="多个数据包"):
        resolve_package_root(dataset)


def test_settings_store_project_paths_relative(tmp_path, monkeypatch):
    import stockregression.settings as settings_module

    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", tmp_path / "config" / "application.json")
    monkeypatch.delenv("STOCKREGRESSION_DATA_DIR", raising=False)
    monkeypatch.delenv("STOCKREGRESSION_PACKAGE_DIR", raising=False)

    document = current_configuration()
    patch_configuration(
        {"paths": {"data_dir": "data", "package_dir": "data/dataset"}},
        base_revision=document["config_revision"],
    )
    saved = load_settings()

    assert saved.data_dir == str((tmp_path / "data").resolve())
    assert saved.package_dir == str((tmp_path / "data" / "dataset").resolve())
    stored = json.loads((tmp_path / "config" / "application.json").read_text(encoding="utf-8"))
    assert stored["paths"]["data_dir"] == "data"
    assert stored["paths"]["package_dir"] == "data/dataset"


def test_schema_two_package_keeps_canonical_volume_and_turnover_units(tmp_path):
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
    assert service.repository.dataset("510050.SH", "1min")["status"] == "pending_validation"


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
        assert Store(root).claim_job(job_id, "test-worker")
        execute_job(str(root), job_id)
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
        assert client.get("/api/v1/data/coverage/510050.SH?timeframe=1min").json()["storage"] == "parquet"


def test_package_index_rebuilds_from_manifest_without_rewriting_shards(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    root = tmp_path / "data"
    service = MarketDataService(Store(root))
    manifest = json.loads((package / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    head = manifest["datasets"]["510050.SH|1min|none"]
    dataset_manifest = json.loads((package / head["manifest"]).read_text(encoding="utf-8"))
    shard_stats = {
        item["path"]: (package / item["path"]).stat().st_mtime_ns
        for item in dataset_manifest["shards"]
    }
    result = index_etf_package_direct(
        service, package, timeframes=["1day", "1min"], symbols=["510050.SH"],
    )
    minute = service.repository.active_rows("510050.SH", "1min")

    assert result["datasets"] == 2
    assert service.repository.dataset("510050.SH", "1min")["storage"] == "parquet"
    assert len(minute) == 240
    assert minute.volume.sum() == pytest.approx(245)
    assert not any((root / "market").rglob("*.parquet"))
    assert all((package / path).stat().st_mtime_ns == mtime for path, mtime in shard_stats.items())
    active = service.store.snapshot("510050.SH")
    assert active["market_data"] is True
    assert service.store.snapshot("510050.SH", active["version"]) is not None


def test_import_replaces_only_the_current_year_shard_and_keeps_old_version(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    root = tmp_path / "data"
    service = MarketDataService(Store(root))
    index_etf_package_direct(
        service, package, timeframes=["1day", "1min"], symbols=["510050.SH"],
    )
    head = json.loads((package / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))["datasets"]["510050.SH|1day|none"]
    first_dataset = json.loads((package / head["manifest"]).read_text(encoding="utf-8"))
    old_shard = package / first_dataset["shards"][0]["path"]
    original = old_shard.read_bytes()
    days = pd.DatetimeIndex(["2026-01-06"])
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": days}).to_csv(calendar, index=False)
    path = tmp_path / "new_daily.csv"
    _daily(days).to_csv(path, index=False)

    service.import_file("510050.SH", "1day", path, calendar_path=calendar)

    assert len(service.repository.active_rows("510050.SH", "1day")) == 2
    assert old_shard.read_bytes() == original
    assert len(service.repository.bars("510050.SH", "1day", version=1)) == 1
    assert len(list((package / "partitions" / "1day").rglob("*.parquet"))) == 2
    assert not (package / ".stockregression" / "increments").exists()
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
    days = pd.DatetimeIndex(pd.Timestamp("2026-01-05") + pd.to_timedelta([0, 1, 2, 3, 4, 7], unit="D"))
    _etf_package(package, days=days)
    manifest = json.loads((package / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    minute_dataset = json.loads((package / manifest["datasets"]["510050.SH|1min|none"]["manifest"]).read_text(encoding="utf-8"))
    minute_shard = package / minute_dataset["shards"][0]["path"]
    original_bytes = minute_shard.read_bytes()
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
    assert minute_shard.read_bytes() == original_bytes
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
    manifest = json.loads((service.repository.package_dir / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    head = manifest["datasets"]["600000.SH|1day|none"]
    version_manifest = json.loads((service.repository.package_dir / head["manifest"]).read_text(encoding="utf-8"))
    original_paths = [service.repository.package_dir / item["path"] for item in version_manifest["shards"]]
    original = [path.read_bytes() for path in original_paths]

    def fail_publish(*_args, **_kwargs):
        raise OSError("simulated increment publication failure")

    monkeypatch.setattr(service.repository, "_write_period_shard", fail_publish)
    with pytest.raises(OSError, match="simulated"):
        service.import_file("600000.SH", "1day", second, calendar_path=calendar)

    recovered = MarketDataService(Store(tmp_path / "data"))
    assert [path.read_bytes() for path in original_paths] == original
    assert recovered.repository.dataset("600000.SH", "1day")["current_version"] == 1


def test_manifest_index_rebuild_reuses_unchanged_day_validation(tmp_path):
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
    rebuilt = index_etf_package_direct(
        service, package, timeframes=["1day"], symbols=["510050.SH"],
    )

    after = service.repository.day_validations("510050.SH", "1day")
    assert rebuilt["changed"] == 0 and rebuilt["unchanged"] == 1
    assert len(after) == len(before)
    assert after[0]["checked_at"] == checked_at
    assert service.repository.dataset("510050.SH", "1day")["quality_status"] == "passed"


def test_default_package_initialization_indexes_only_base_timeframes(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _etf_package(package)
    (package / "etf_5min").mkdir()
    (package / "etf_5min" / "510050.SH.parquet").touch()
    service = MarketDataService(Store(tmp_path / "data"))

    result = index_etf_package_direct(service, package, symbols=["510050.SH"])

    assert result["changed"] == 2
    assert service.repository.dataset("510050.SH", "1day") is not None
    assert service.repository.dataset("510050.SH", "1min") is not None
    assert service.repository.dataset("510050.SH", "5min") is None
    assert not (package / ".stockregression" / "bases").exists()
    assert not (package / ".stockregression" / "increments").exists()


def test_package_registration_indexes_stock_layout(tmp_path):
    package = tmp_path / "package"
    _stock_package(package)
    service = MarketDataService(Store(tmp_path / "data"))

    result = index_etf_package_direct(
        service, package, timeframes=["1day", "1min"], symbols=["600000.SH"],
    )

    assert result["changed"] == 2 and result["errors"] == []
    assert service.store.instrument("600000.SH")["asset_type"] == "stock"
    assert service.repository.dataset("600000.SH", "1day")["storage"] == "parquet"
    assert len(service.repository.bars("600000.SH", "1min")) == 240


def test_full_package_verification_detects_modified_shard_bytes(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    day = pd.DatetimeIndex(["2026-01-05"])
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": day}).to_csv(calendar, index=False)
    source = tmp_path / "source.csv"
    _daily(day).to_csv(source, index=False)
    service.import_file("600000.SH", "1day", source, calendar_path=calendar)
    assert service.repository.dataset("600000.SH", "1day")["file_status"] == "readable"

    head = json.loads((service.repository.package_dir / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))["datasets"]["600000.SH|1day|none"]
    version_manifest = json.loads((service.repository.package_dir / head["manifest"]).read_text(encoding="utf-8"))
    package_file = service.repository.package_dir / version_manifest["shards"][0]["path"]
    externally_changed = pd.read_parquet(package_file)
    externally_changed["close"] = externally_changed["close"] + 1
    externally_changed.to_parquet(package_file, index=False)

    from stockregression.market_data.migration import verify_package
    with pytest.raises(ValueError, match="SHA-256"):
        verify_package(tmp_path / "data", service.repository.package_dir, full=True)
