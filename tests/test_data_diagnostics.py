from __future__ import annotations

import pandas as pd

from stockregression.data_diagnostics import derive_source_spans, project_data_diagnostics


def _minutes(day, source, count, verified=True):
    stamps = pd.date_range(pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=31), periods=count, freq="min")
    return pd.DataFrame({
        "timestamp": stamps,
        "source": source,
        "source_verified": verified,
    })


def test_source_spans_use_merged_rows_for_mixed_import_and_sync_sources():
    minute = pd.concat([
        _minutes("2024-01-02", "import", 240),
        _minutes("2024-01-03", "import", 240),
        _minutes("2024-01-04", "sina", 238),
        _minutes("2024-01-05", "sina", 240, verified=False),
    ], ignore_index=True)

    spans = derive_source_spans(
        minute,
        {"import": {"verified": True}, "sina": {"verified": True}},
        "2024-01-03",
        "2024-01-04",
    )

    assert spans == [
        {
            "source": "import", "source_kind": "import", "start": "2024-01-03",
            "end": "2024-01-03", "days": 1, "verified": True, "rows": 240,
        },
        {
            "source": "sina", "source_kind": "provider", "start": "2024-01-04",
            "end": "2024-01-04", "days": 1, "verified": True, "rows": 238,
        },
    ]


def test_projection_separates_historical_source_failures_from_current_risks():
    metadata = {
        "warnings": [
            "历史分钟下载失败：endpoint timeout",
            "ETF 历史分钟库仅覆盖部分标的；其他缺口需导入补齐",
            "公司行动资料尚未完成核验",
        ],
        "source_errors": ["近期分钟源不可用：mock timeout"],
        "actions_verified": False,
        "actions_unverified_years": [2024],
        "minute_status": "unavailable",  # deliberately stale legacy state
        "quality_summary": {"minute_days": 0, "daily_days": 99},  # deliberately stale
        "diagnostics": {"source_spans": derive_source_spans(_minutes("2024-01-02", "import", 240))},
    }
    quality = {
        "minute_days": 2,
        "daily_days": 1,
        "suspended_days": 1,
        "total_days": 4,
        "pending_calibration_days": 1,
        "minute_ratio": 2 / 3,
        "synthetic_bars": 2,
        "days": [
            {"date": "2024-01-02", "mode": "minute", "missing_minutes": 2},
            {"date": "2024-01-03", "mode": "minute", "missing_minutes": 0},
            {"date": "2024-01-04", "mode": "daily", "download_needed": False, "reason": "来源分钟标签/单位未校准"},
            {"date": "2024-01-05", "mode": "suspended", "missing_minutes": 0},
        ],
        "warnings": ["1 个交易日已有本地分钟数据，但时间标签/单位尚未校准；按日线处理"],
    }

    projected = project_data_diagnostics(metadata, quality, "2024-01-02", "2024-01-05")

    assert projected["diagnostics"]["status"] == "partial"
    assert projected["diagnostics"]["minute_days"] == 2
    assert projected["diagnostics"]["daily_days"] == 1
    assert projected["diagnostics"]["pending_calibration_days"] == 1
    assert projected["diagnostics"]["missing_raw_minutes"] == 2
    assert projected["diagnostics"]["synthetic_bars"] == 2
    assert any("公司行动资料" in warning for warning in projected["current_warnings"])
    assert any("未校准" in warning for warning in projected["current_warnings"])
    assert any("因果前值补齐" in warning for warning in projected["current_warnings"])
    assert not any("endpoint timeout" in warning or "mock timeout" in warning for warning in projected["current_warnings"])
    assert {event["message"] for event in projected["source_events"]} >= {
        "历史分钟下载失败：endpoint timeout",
        "ETF 历史分钟库仅覆盖部分标的；其他缺口需导入补齐",
        "近期分钟源不可用：mock timeout",
    }


def test_complete_current_quality_overrides_old_failure_and_quality_summary():
    metadata = {
        "warnings": ["近期分钟源不可用：old provider timeout"],
        "source_errors": ["历史分钟下载失败：old endpoint error"],
        "quality_summary": {"minute_days": 0, "daily_days": 100},
        "minute_status": "unavailable",
        "diagnostics": {"source_spans": derive_source_spans(_minutes("2024-01-02", "import", 240))},
    }
    quality = {
        "minute_days": 1,
        "daily_days": 0,
        "suspended_days": 0,
        "total_days": 1,
        "pending_calibration_days": 0,
        "minute_ratio": 1.0,
        "synthetic_bars": 0,
        "days": [{"date": "2024-01-02", "mode": "minute", "missing_minutes": 0}],
        "warnings": [],
    }

    projected = project_data_diagnostics(metadata, quality, "2024-01-02", "2024-01-02")

    assert projected["diagnostics"]["status"] == "complete"
    assert projected["diagnostics"]["daily_days"] == 0
    assert projected["current_warnings"] == []
    assert len(projected["source_events"]) == 2


def test_exchange_auction_placeholders_are_not_reported_as_missing_data():
    projected = project_data_diagnostics(
        {},
        {
            "minute_days": 1,
            "daily_days": 0,
            "suspended_days": 0,
            "total_days": 1,
            "pending_calibration_days": 0,
            "minute_ratio": 1.0,
            "synthetic_bars": 14,
            "auction_placeholders": 14,
            "days": [{
                "date": "2026-09-30", "mode": "minute", "missing_minutes": 14,
                "missing_data_minutes": 0, "auction_placeholders": 14,
                "download_needed": False,
            }],
            "warnings": [],
        },
        "2026-09-30",
        "2026-09-30",
    )

    diagnostics = projected["diagnostics"]
    assert diagnostics["status"] == "complete"
    assert diagnostics["auction_placeholders"] == 14
    assert diagnostics["synthetic_bars"] == 14
    assert diagnostics["repaired_bars"] == 0
    assert diagnostics["missing_raw_minutes"] == 0
    assert diagnostics["grid_missing_minutes"] == 14
    assert diagnostics["raw_complete"] is True
    assert not any("因果前值补齐" in warning for warning in projected["current_warnings"])


def test_raw_complete_requires_complete_calibrated_minute_coverage():
    projected = project_data_diagnostics(
        {},
        {
            "minute_days": 0,
            "daily_days": 2,
            "suspended_days": 0,
            "total_days": 2,
            "pending_calibration_days": 0,
            "days": [
                {"date": "2024-01-02", "mode": "daily", "missing_data_minutes": 0},
                {"date": "2024-01-03", "mode": "daily", "missing_data_minutes": 0},
            ],
            "warnings": ["近期分钟源不可用：stale source"],
        },
        "2024-01-02",
        "2024-01-03",
    )

    assert projected["diagnostics"]["status"] == "partial"
    assert projected["diagnostics"]["missing_raw_minutes"] == 0
    assert projected["diagnostics"]["raw_complete"] is False
    assert not any("stale source" in warning for warning in projected["current_warnings"])
    assert any("stale source" in event["message"] for event in projected["source_events"])


def test_unverified_action_risk_survives_empty_years_and_missing_metadata():
    warning = "公司行动资料尚未完成核验"
    for metadata in (
        {"actions_verified": False, "actions_unverified_years": [], "warnings": [warning]},
        {"warnings": [warning]},
    ):
        projected = project_data_diagnostics(metadata, {}, "2024-01-02", "2024-12-31")
        assert warning in projected["current_warnings"]
        assert not any(event["status"] == "resolved" and event["message"] == warning for event in projected["source_events"])


def test_clipped_cached_source_spans_clear_full_span_row_counts():
    projected = project_data_diagnostics(
        {"diagnostics": {"source_spans": [{
            "source": "hf", "source_kind": "provider", "start": "2024-01-02",
            "end": "2024-01-10", "days": 7, "verified": True, "rows": 1680,
        }]}},
        {},
        "2024-01-03",
        "2024-01-05",
    )

    span, = projected["diagnostics"]["source_spans"]
    assert span["start"] == "2024-01-03"
    assert span["end"] == "2024-01-05"
    assert span["days"] is None
    assert span["rows"] is None


def test_unmapped_source_is_not_mislabeled_as_provider():
    spans = derive_source_spans(_minutes("2024-01-02", "mystery", 240))

    assert spans[0]["source"] == "mystery"
    assert spans[0]["source_kind"] == "unknown"


def test_intentionally_daily_request_is_not_reported_as_minute_downgrade():
    projected = project_data_diagnostics(
        {},
        {
            "requested_frequency": "1d",
            "minute_days": 0,
            "daily_days": 2,
            "suspended_days": 0,
            "total_days": 2,
            "pending_calibration_days": 0,
            "minute_ratio": 0,
            "days": [
                {"date": "2024-01-02", "mode": "daily", "download_needed": False},
                {"date": "2024-01-03", "mode": "daily", "download_needed": False},
            ],
            "warnings": [],
        },
        "2024-01-02",
        "2024-01-03",
    )

    diagnostics = projected["diagnostics"]
    assert diagnostics["status"] == "complete"
    assert diagnostics["requested_frequency"] == "1d"
    assert diagnostics["minute_quality_applicable"] is False
    assert diagnostics["minute_ratio"] is None
    assert diagnostics["raw_complete"] is None
    assert not any("降级" in warning or "分钟数据当前缺口" in warning for warning in projected["current_warnings"])

