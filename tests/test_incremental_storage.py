from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.jobs import execute_job
from stockregression.market_data import MarketDataService
from stockregression.market_data.models import MarketDataError
from stockregression.market_data.repository import MarketDataRepository
from stockregression.storage import Store


def _daily(days, close=10.2):
    return pd.DataFrame({
        "date": list(days),
        "open": 10.0,
        "high": 10.5,
        "low": 9.5,
        "close": close,
        "volume": 24_000.0,
        "turnover": 240_000.0,
        "preclose": 10.0,
        "tradestatus": 1,
        "is_st": 0,
    })


def _write_daily(root, name, days, close=10.2):
    path = root / name
    _daily(days, close=close).to_csv(path, index=False)
    return path


def _calendar(root, days):
    path = root / "calendar.csv"
    pd.DataFrame({"date": list(days)}).to_csv(path, index=False)
    return path


def test_append_keeps_immutable_prior_shards_and_reuses_validation(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    calendar = _calendar(tmp_path, days.insert(2, pd.Timestamp("2026-01-07")))
    first = service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "first.csv", days),
        calendar_path=calendar,
    )
    package = service.repository.package_dir
    before_manifest = json.loads((package / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    before_dataset = json.loads((package / before_manifest["datasets"]["600000.SH|1day|none"]["manifest"]).read_text(encoding="utf-8"))
    before_shards = {row["path"]: (package / row["path"]).read_bytes() for row in before_dataset["shards"]}
    original_checked = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }

    newer = days.append(pd.DatetimeIndex(["2026-01-07"]))
    second = service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "second.csv", newer),
        calendar_path=calendar,
    )

    assert second["version"] == first["version"] + 1
    assert second["shard_count"] == 1
    assert all((package / path).read_bytes() == data for path, data in before_shards.items())
    after_manifest = json.loads((package / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    after_dataset = json.loads((package / after_manifest["datasets"]["600000.SH|1day|none"]["manifest"]).read_text(encoding="utf-8"))
    assert after_dataset["shards"]
    assert len(service.repository.bars("600000.SH", "1day", version=1)) == 2
    assert len(service.repository.active_rows("600000.SH", "1day")) == 3
    current_checked = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }
    assert current_checked["2026-01-05"] == original_checked["2026-01-05"]
    assert service.repository.validation_state(
        "600000.SH", "1day"
    )["verified_through"] == "2026-01-07"

    revised = _daily(newer)
    revised.loc[1, "close"] = 11.1
    revised.loc[1, "high"] = 11.5
    revised_path = tmp_path / "revised.csv"
    revised.to_csv(revised_path, index=False)
    before_revision = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }
    third = service.import_file(
        "600000.SH", "1day", revised_path, calendar_path=calendar,
    )
    after_revision = {
        row["trade_date"]: row["checked_at"]
        for row in service.repository.day_validations("600000.SH", "1day")
    }
    assert third["changed_dates"] == ["2026-01-06"]
    assert after_revision["2026-01-05"] == before_revision["2026-01-05"]
    assert after_revision["2026-01-07"] == before_revision["2026-01-07"]
    assert after_revision["2026-01-06"] != before_revision["2026-01-06"]


def test_snapshot_replaces_view_but_preserves_previous_version(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    calendar = _calendar(tmp_path, days)
    first = service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "first.csv", days),
        calendar_path=calendar,
    )
    second = service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "snapshot.csv", days[:1]),
        calendar_path=calendar, import_mode="snapshot",
    )

    assert second["version"] > first["version"]
    assert service.repository.active_rows("600000.SH", "1day").timestamp.tolist() == [
        pd.Timestamp("2026-01-05")
    ]
    assert len(service.repository.bars("600000.SH", "1day", version=first["version"])) == 2
    assert service.repository.day_validations("600000.SH", "1day")[0]["trade_date"] == "2026-01-05"


def test_published_manifest_recovers_when_sqlite_commit_is_interrupted(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    service = MarketDataService(Store(data_root))
    days = pd.DatetimeIndex(["2026-01-05"])
    calendar = _calendar(tmp_path, days)
    source = _write_daily(tmp_path, "daily.csv", days)

    def fail_after_manifest(*_args, **_kwargs):
        raise RuntimeError("injected interruption after package manifest publish")

    monkeypatch.setattr(service.repository, "_copy_version_shards", fail_after_manifest)
    with pytest.raises(RuntimeError, match="after package manifest publish"):
        service.import_file("600000.SH", "1day", source, calendar_path=calendar)

    assert service.repository.package.pending_transactions()
    recovered = MarketDataRepository(data_root)
    assert recovered.active_rows("600000.SH", "1day").shape[0] == 1
    assert recovered.package.pending_transactions() == []
    assert recovered.package_info()["index_status"] == "ready"


def test_validation_skips_kline_read_when_version_is_already_compliant(tmp_path, monkeypatch):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    calendar = _calendar(tmp_path, days)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=calendar,
    )

    def fail_read(*_args, **_kwargs):
        raise AssertionError("已合规的数据不应重新读取 K 线")

    monkeypatch.setattr(service.repository, "active_rows", fail_read)
    result = service.revalidate("600000.SH", "1day")
    assert result["report"]["checked_days"] == 0
    assert result["verified_through"] == "2026-01-06"


def test_immutable_shards_do_not_need_duplicate_archive_copies(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.DatetimeIndex(["2026-01-05"])
    calendar = _calendar(tmp_path, days)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=calendar,
    )
    first = service.repository.archive_dataset("600000.SH", "1day")
    second = service.repository.archive_dataset("600000.SH", "1day")
    archives = list((service.repository.package_dir / ".stockregression" / "archives").rglob("*.parquet"))

    assert first["copied"] == 0 and second["reused"] == 0
    assert archives == []
    assert first["archived"] is True
    assert service.repository.archive_status("600000.SH", "1day")["protected"] is True


def test_year_shards_remain_separate_without_cross_year_compaction(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    first_days = pd.DatetimeIndex(["2025-12-30"])
    days = pd.DatetimeIndex(["2025-12-30", "2026-12-30", "2027-12-30"])
    calendar = _calendar(tmp_path, days)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "first.csv", first_days),
        calendar_path=calendar,
    )
    for index, day in enumerate(days[1:], start=2):
        service.import_file(
            "600000.SH", "1day", _write_daily(tmp_path, f"{index}.csv", days[:index]),
            calendar_path=calendar,
        )

    dataset = service.repository.dataset("600000.SH", "1day")
    assert dataset["shard_count"] == 3
    assert dataset["row_count"] == 3
    assert service.repository.active_rows("600000.SH", "1day").timestamp.dt.date.nunique() == 3


def test_derived_timeframe_is_generated_lazily_and_native_import_is_rejected(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))
    days = pd.bdate_range("2026-01-05", periods=5)
    calendar = _calendar(tmp_path, days)
    service.import_file(
        "600000.SH", "1day", _write_daily(tmp_path, "daily.csv", days),
        calendar_path=calendar,
    )

    weekly = service.bars("600000.SH", "1week")
    assert weekly["count"] == 1
    derived = service.repository.dataset("600000.SH", "1week")
    assert derived["derived"]["source_timeframe"] == "1day"

    with pytest.raises(MarketDataError, match="派生周期"):
        service.import_file(
            "600000.SH", "1week", _write_daily(tmp_path, "week.csv", days[-1:]),
            calendar_path=calendar,
        )


def test_api_snapshot_and_compliance_are_exposed_as_task_results(tmp_path):
    root = tmp_path / "data"
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    calendar = _calendar(tmp_path, days)
    first = _write_daily(tmp_path, "first.csv", days)
    snapshot = _write_daily(tmp_path, "snapshot.csv", days[:1])
    client = TestClient(create_app(str(root), start_jobs=False))

    for path, mode in ((first, "merge"), (snapshot, "snapshot")):
        response = client.post("/api/v1/data/import", json={
            "symbol": "600000.SH", "timeframe": "1day", "path": str(path),
            "calendar_path": str(calendar), "label_convention": "end",
            "import_mode": mode,
        })
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        assert Store(root).claim_job(job_id, "test-worker")
        execute_job(str(root), job_id)

    bars = client.get("/api/v1/data/bars/600000.SH?frequency=1day").json()
    compliance = client.get("/api/v1/data/compliance/600000.SH").json()
    assert len(bars) == 1
    assert compliance["1day"]["verified_through"] == "2026-01-05"

    invalid = client.post("/api/v1/data/validate", json={
        "symbol": "600000.SH", "timeframe": "1day", "mode": "unknown",
    })
    assert invalid.status_code == 422
