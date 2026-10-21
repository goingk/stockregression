from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression import data as data_module
from stockregression.api import create_app, public_result
from stockregression.data import DataService
from stockregression.storage import Store, dumps


def _seed_snapshot(store: Store) -> str:
    days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    minute_parts = []
    for day, source, count in ((days[0], "import", 240), (days[1], "import", 240), (days[2], "sina", 238)):
        timestamps = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=count, freq="min")
        minute_parts.append(pd.DataFrame({
            "timestamp": timestamps,
            "open": 1.0,
            "high": 1.01,
            "low": 0.99,
            "close": 1.0,
            "volume": 1000 if source == "import" else 900,
            "turnover": 1000.0,
            "source": source,
            "source_verified": True,
            "is_synthetic": False,
        }))
    daily = pd.DataFrame({
        "date": days,
        "open": 1.0,
        "high": 1.02,
        "low": 0.98,
        "close": 1.0,
        "preclose": 1.0,
        "volume": 100_000,
        "turnover": 100_000.0,
        "tradestatus": 1,
        "is_st": False,
    })
    manifest = store.publish(
        "159611.SZ",
        pd.concat(minute_parts, ignore_index=True),
        daily,
        pd.DataFrame(),
        pd.DataFrame({"date": days}),
        {
            "name": "diagnostics test fixture",
            "is_demo": False,
            "warnings": ["历史分钟下载失败：Sina endpoint timeout"],
            "source_errors": ["近期分钟源不可用：cached provider failure"],
            "actions_verified": False,
            "actions_unverified_years": [2024],
        },
        {
            "import": {"verified": True, "source_kind": "import"},
            "sina": {"verified": True, "source_kind": "provider"},
        },
    )
    return manifest["version"]


def _quality() -> dict:
    return {
        "minute_days": 1,
        "daily_days": 1,
        "suspended_days": 0,
        "total_days": 2,
        "pending_calibration_days": 1,
        "minute_ratio": 0.5,
        "synthetic_bars": 2,
        "auction_placeholders": 1,
        "days": [
            {
                "date": "2024-01-03",
                "mode": "minute",
                "missing_minutes": 2,
                "missing_data_minutes": 1,
                "auction_placeholders": 1,
                "download_needed": False,
            },
            {"date": "2024-01-04", "mode": "daily", "download_needed": True},
        ],
        "warnings": ["1 个交易日的本地分钟数据尚未校准，已按日线处理"],
    }


def _saved_result(version: str) -> dict:
    stale = "历史分钟下载失败：Sina endpoint timeout"
    top_quality = _quality()
    top_quality["warnings"].append("历史分钟独立校验失败：quality-only reference mismatch")
    fixed_quality = _quality()
    fixed_quality["warnings"].append("新浪分钟备源不可用：nested quality-only provider failure")
    return {
        "protocol_version": 5,
        "result_schema_version": 2,
        "data_version": version,
        "symbol": "159611.SZ",
        "config": {
            "protocol_version": 5,
            "base": {
                "schema_version": 3,
                "symbol": "159611.SZ",
                "start_date": "2024-01-03",
                "end_date": "2024-01-04",
                "frequency": "1m",
            },
        },
        "warnings": [stale],
        "source_events": [],
        "quality": top_quality,
        "evidence": {
            "trial_archive": [{"trial": 17, "params": {"grid_step": 0.02}, "payload": [1, 2, 3]}],
            "search_candidate_archive": [{"candidate_id": 9, "params": {"buy_step": .01}}],
        },
        "fixed": {
            "candidate_archive": [{"candidate_id": 9, "costs": [[0.1, 0.2]]}],
            "test_result": {
                "warnings": ["近期分钟源不可用：cached provider failure"],
                "quality": fixed_quality,
                "strategies": {"grid": {"equity": []}},
            },
        },
        "adaptive": {
            "test_result": {
                "warnings": [stale],
                "quality": _quality(),
                "strategies": {"grid": {"equity": []}},
            },
        },
    }


def _warnings(result: dict):
    values = []
    if isinstance(result, dict):
        values.extend(result.get("warnings", []))
        for key in ("fixed", "adaptive", "historical", "robust"):
            branch = result.get(key)
            if isinstance(branch, dict):
                nested = branch.get("test_result") or branch.get("best_result")
                if isinstance(nested, dict):
                    values.extend(_warnings(nested))
    return values


def test_public_v5_result_moves_old_provider_warnings_but_keeps_current_risks_and_import_span():
    result = _saved_result("snapshot-v1")
    metadata = {
        "warnings": ["历史分钟下载失败：Sina endpoint timeout"],
        "source_errors": ["近期分钟源不可用：cached provider failure"],
        "actions_verified": False,
        "actions_unverified_years": [2024],
        "diagnostics": {
            "source_spans": [
                {"source": "import", "source_kind": "import", "start": "2024-01-02", "end": "2024-01-03", "days": 2, "verified": True, "rows": 480},
                {"source": "sina", "source_kind": "provider", "start": "2024-01-04", "end": "2024-01-04", "days": 1, "verified": True, "rows": 238},
            ],
        },
    }

    public = public_result(result, data_metadata=metadata)

    assert "trial_archive" not in public["evidence"]
    assert "search_candidate_archive" not in public["evidence"]
    assert "candidate_archive" not in public["fixed"]
    assert not any("endpoint timeout" in warning or "provider failure" in warning for warning in _warnings(public))
    assert {event["message"] for event in public["source_events"]} >= {
        "历史分钟下载失败：Sina endpoint timeout",
        "历史分钟独立校验失败：quality-only reference mismatch",
        "近期分钟源不可用：cached provider failure",
    }
    assert "新浪分钟备源不可用：nested quality-only provider failure" in {
        event["message"] for event in public["fixed"]["test_result"]["source_events"]
    }
    diagnostics = public["diagnostics"]
    assert diagnostics["checked_start"] == "2024-01-03"
    assert diagnostics["checked_end"] == "2024-01-04"
    assert diagnostics["source_spans"] == [
        {"source": "import", "source_kind": "import", "start": "2024-01-03", "end": "2024-01-03", "days": None, "verified": True, "rows": None},
        {"source": "sina", "source_kind": "provider", "start": "2024-01-04", "end": "2024-01-04", "days": 1, "verified": True, "rows": 238},
    ]
    assert diagnostics["auction_placeholders"] == 1
    assert diagnostics["missing_raw_minutes"] == 1
    assert any("公司行动" in warning for warning in public["warnings"])
    assert any("未校准" in warning for warning in public["warnings"])
    assert any("分钟数据当前缺口" in warning for warning in public["warnings"])
    assert any("因果前值补齐" in warning for warning in public["warnings"])
    assert public["diagnostics"]["status"] == public["fixed"]["test_result"]["diagnostics"]["status"]
    assert public["diagnostics"]["status"] == public["adaptive"]["test_result"]["diagnostics"]["status"]


def test_public_daily_backtest_does_not_claim_minute_quality_downgrade():
    result = {
        "config": {
            "symbol": "159611.SZ",
            "start_date": "2024-01-02",
            "end_date": "2024-01-04",
            "frequency": "1d",
        },
        "quality": {
            "minute_days": 0,
            "daily_days": 3,
            "suspended_days": 0,
            "total_days": 3,
            "pending_calibration_days": 0,
            # Old summaries sometimes carried a minute ratio even though the
            # saved run explicitly requested daily bars.
            "minute_ratio": 0.0,
            "days": [
                {"date": day, "mode": "daily", "download_needed": False}
                for day in ("2024-01-02", "2024-01-03", "2024-01-04")
            ],
            "warnings": [],
        },
        "warnings": [],
    }

    public = public_result(result)

    diagnostics = public["diagnostics"]
    assert diagnostics["requested_frequency"] == "1d"
    assert diagnostics["minute_quality_applicable"] is False
    assert diagnostics["status"] == "complete"
    assert diagnostics["minute_ratio"] is None
    assert diagnostics["raw_complete"] is None
    assert not any("整日降级" in warning or "分钟数据当前缺口" in warning for warning in public["warnings"])
