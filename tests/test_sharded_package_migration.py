from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.migration import (
    migrate_package, preflight, verify_package, verify_source_independent,
)
from stockregression.market_data.repository import MarketDataRepository
from stockregression.storage import Store


def _frozen_schema_one_fixture(tmp_path):
    data_root = tmp_path / "data"
    source = tmp_path / "ETF-source"
    target = data_root / "dataset" / "etf"
    target.mkdir(parents=True)
    source.mkdir()
    service = MarketDataService(Store(data_root), package_dir=source)
    service.repository.save_basic_info("510050.SH", {"cname": "ETF fixture", "exchange": "SH"})
    bars = pd.DataFrame([{
        "timestamp": pd.Timestamp("2026-01-05"), "open": 1.0, "high": 1.0,
        "low": 1.0, "close": 1.0, "volume": 100.0, "turnover": 100.0,
        "preclose": 1.0, "tradestatus": 1, "is_st": False,
        "source": "fixture", "source_verified": True,
    }])
    service.repository.commit_bars("510050.SH", "1day", bars, source="fixture")
    service.repository.save_auxiliary(
        "510050.SH", "calendar", [{"date": "2026-01-05"}],
        version="calendar-v1", verified=True,
    )
    service.store.add_watch("510050.SH", "ETF fixture")
    service.refresh_snapshot("510050.SH")
    snapshot = service.store.snapshot("510050.SH")
    result_id = service.store.save_result(
        "backtest", {"status": "completed", "data_version": snapshot["version"]},
        "510050.SH", "Historical result",
    )

    orphan = source / "partitions" / "1day" / "symbol=000001.BJ" / "adjustment=none" / "year=2026" / "orphan.parquet"
    orphan.parent.mkdir(parents=True)
    pd.DataFrame({"value": [1]}).to_parquet(orphan, index=False)

    old_target_manifest = {
        "schema_version": 1, "package_id": "old-target-package",
        "files": {}, "auxiliary": {},
    }
    (target / ".stockregression").mkdir(parents=True)
    (target / ".stockregression" / "manifest.json").write_text(
        json.dumps(old_target_manifest), encoding="utf-8",
    )
    manifest_path = source / ".stockregression" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with service.repository.connection() as db:
        db.execute("""CREATE TABLE parquet_files AS
            SELECT symbol,timeframe,adjustment,version,period_key partition_key,
                   path,sha256,row_count,first_time,last_time,size,schema_version
            FROM parquet_shards WHERE 0""")
        db.execute("ALTER TABLE parquet_shards RENAME TO parquet_increments")
    return data_root, source, target, service, snapshot["version"], result_id


def test_migration_freezes_active_refs_quarantines_orphans_and_rebuilds_index(tmp_path, monkeypatch):
    data_root, source, target, service, old_version, result_id = _frozen_schema_one_fixture(tmp_path)
    baseline = preflight(data_root, source, target)
    assert baseline["dataset_count"] == 1
    assert baseline["shard_count"] == 1
    assert baseline["quarantine_count"] == 1
    assert baseline["by_timeframe"]["1day"]["rows"] == 1

    result = migrate_package(data_root, source, target, apply=True)
    assert result["status"] == "applied"
    assert result["dataset_count"] == 1 and result["shard_count"] == 1
    assert result["quarantine_count"] == 1
    verified = verify_package(data_root, target, full=True)
    assert verified["status"] == "passed" and verified["full"]
    assert verified["sample_queries"][0]["symbol"] == "510050.SH"

    new_manifest = json.loads((target / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    assert new_manifest["schema_version"] == 2
    assert new_manifest["package_id"] == baseline["package_id"]
    assert new_manifest["generation"] != baseline["old_generation"]
    quarantined = new_manifest["quarantine"][0]
    assert (target / quarantined["path"]).is_file()

    store = Store(data_root)
    assert store.result(result_id)["status"] == "completed"
    assert [row["symbol"] for row in store.watches()] == ["510050.SH"]
    assert store.snapshot("510050.SH", old_version) is None

    repository = MarketDataRepository(data_root)
    before = store.result(result_id)
    rebuilt = repository.rebuild_index()
    assert rebuilt["datasets"] == 1
    assert store.result(result_id) == before
    assert [row["symbol"] for row in store.watches()] == ["510050.SH"]

    monkeypatch.setattr(
        "stockregression.market_data.migration._service_smoke",
        lambda *_args, **_kwargs: {"package": "ready", "backtest_readiness": True},
    )
    offline = verify_source_independent(data_root, target, full=True)
    assert offline["source_unavailable"] is True
    assert not source.exists()
    assert (data_root / "migrations" / result["migration_id"] / "source-package-hold").is_dir()


def test_failed_cutover_restores_previous_package_and_catalog(tmp_path, monkeypatch):
    data_root, source, target, _service, _old_version, _result_id = _frozen_schema_one_fixture(tmp_path)
    original_target = json.loads((target / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))

    def fail_rebuild(*_args, **_kwargs):
        raise RuntimeError("injected index rebuild failure")

    monkeypatch.setattr(MarketDataRepository, "rebuild_index", fail_rebuild)
    with pytest.raises(RuntimeError, match="injected index rebuild failure"):
        migrate_package(data_root, source, target, apply=True)

    restored = json.loads((target / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    assert restored == original_target
    with Store(data_root).db() as db:
        package_path = db.execute(
            "SELECT value FROM parquet_meta WHERE key='package_dir'"
        ).fetchone()[0]
    assert package_path == str(source.resolve())
    failed = list((data_root / "migrations").glob("*/migration.json"))[-1]
    report = json.loads(failed.read_text(encoding="utf-8"))
    assert report["rolled_back"] is True
    assert report["failed_package"]
    assert source.is_dir()


def test_resume_reuses_completed_stage_after_cutover_rename_failure(tmp_path, monkeypatch):
    data_root, source, target, _service, _old_version, _result_id = _frozen_schema_one_fixture(tmp_path)
    real_replace = os.replace

    def fail_first_target_move(src, dst):
        if Path(src).resolve() == target.resolve() and Path(dst).name == "replaced-package":
            raise PermissionError("simulated external directory handle")
        return real_replace(src, dst)

    monkeypatch.setattr("stockregression.market_data.migration.os.replace", fail_first_target_move)
    with pytest.raises(PermissionError, match="simulated external directory handle"):
        migrate_package(data_root, source, target, apply=True)

    report_path = next((data_root / "migrations").glob("*/migration.json"))
    failed = json.loads(report_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and failed["rolled_back"]
    stage = Path(failed["stage_path"])
    assert (stage / ".stockregression" / "manifest.json").is_file()
    assert target.is_dir() and source.is_dir()

    monkeypatch.setattr("stockregression.market_data.migration.os.replace", real_replace)
    resumed = migrate_package(
        data_root, source, target, apply=True,
        resume_migration_id=failed["migration_id"],
    )
    assert resumed["status"] == "applied"
    assert resumed["resumed_from_failed_attempt"] is True
    assert resumed["migration_id"] == failed["migration_id"]
    assert not stage.exists()
    assert (data_root / "migrations" / failed["migration_id"] / "attempts" / "failed-before-resume.json").is_file()
    assert verify_package(data_root, target, full=True)["status"] == "passed"
