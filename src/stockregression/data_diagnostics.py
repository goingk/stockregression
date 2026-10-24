"""Pure projections for current local market-data quality and source history.

The snapshot's ``diagnostics`` describe the data that was actually merged and
validated when that snapshot was published.  Request-specific projections use
the quality report produced by ``prepare_bundle``; old provider failures and
stale quality summaries are never treated as current data gaps.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .quality import QUALITY_RULES_VERSION


DIAGNOSTICS_SCHEMA_VERSION = 1

_ACTION_WORDS = ("公司行动", "分红", "拆分", "除权", "除息", "actions")
_HISTORICAL_SOURCE_PREFIXES = (
    "历史分钟下载失败",
    "近期分钟源不可用",
    "新浪分钟备源不可用",
    "历史分钟独立校验失败",
    "历史分钟源版本已变化",
    "ETF 历史分钟库仅覆盖",
    "最新交易日预检暂不可用",
    "无法取得精确下载量估算",
    "东财 ETF 日线不可用",
    "ETF 日线来源均不可用",
)
_SOURCE_FAILURE_WORDS = (
    "下载失败", "源不可用", "接口失败", "请求失败", "连接失败", "超时",
    "endpoint unavailable", "network timeout", "request failed", "connection error",
    "源版本已变化", "备源不可用", "仅覆盖部分标的",
)


def _day(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    if result.tz is not None:
        result = result.tz_convert("Asia/Shanghai").tz_localize(None)
    return result.normalize()


def _iso_day(value: Any) -> str | None:
    parsed = _day(value)
    return parsed.date().isoformat() if parsed is not None else None


def _is_import_source(source: str) -> bool:
    normalized = source.strip().lower().replace("-", "_")
    return normalized in {"import", "manual", "user_import", "local_import"} or normalized.startswith("import_")


def derive_source_spans(
    minute: pd.DataFrame | None,
    profiles: dict | None = None,
    start=None,
    end=None,
) -> list[dict[str, Any]]:
    """Summarize actual minute-row provenance, retaining mixed import/sync data.

    A profile describes how a source was calibrated; it does not establish which
    dates that source supplied.  Date spans therefore come only from the merged
    minute rows themselves.
    """
    if minute is None or minute.empty or "timestamp" not in minute:
        return []
    frame = minute.copy()
    frame["_day"] = pd.to_datetime(frame.timestamp, errors="coerce")
    frame = frame[frame._day.notna()].copy()
    if frame.empty:
        return []
    frame["_day"] = frame._day.dt.normalize()
    lower, upper = _day(start), _day(end)
    if lower is not None:
        frame = frame[frame._day >= lower]
    if upper is not None:
        frame = frame[frame._day <= upper]
    if frame.empty:
        return []
    if "source" not in frame:
        frame["source"] = "unknown"
    frame["source"] = frame.source.fillna("unknown").astype(str).str.strip().replace("", "unknown")
    profile_map = profiles or {}
    if "source_verified" in frame:
        frame["_verified"] = frame.source_verified.fillna(False).map(_as_bool)
    else:
        frame["_verified"] = frame.source.map(lambda source: bool(profile_map.get(source, {}).get("verified", False)))
    rows = []
    for (source, verified), group in frame.groupby(["source", "_verified"], sort=True, dropna=False):
        dates = pd.DatetimeIndex(group._day.unique()).sort_values()
        if not len(dates):
            continue
        if _is_import_source(str(source)):
            source_kind = "import"
        elif str(source) in profile_map:
            source_kind = str(profile_map[str(source)].get("source_kind") or "provider")
        elif str(source).lower() in {"tdx", "sina", "hf", "baostock", "eastmoney"}:
            source_kind = "provider"
        else:
            source_kind = "unknown"
        rows.append({
            "source": str(source),
            "source_kind": source_kind,
            "start": dates[0].date().isoformat(),
            "end": dates[-1].date().isoformat(),
            "days": int(len(dates)),
            "verified": bool(verified),
            "rows": int(len(group)),
        })
    return rows


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "verified"}
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return False


def _event_key(event: dict[str, Any]) -> tuple[str, str, str]:
    return (str(event.get("kind", "source")), str(event.get("source", "")), str(event.get("message", "")))


def _source_from_message(message: str) -> str | None:
    lowered = message.lower()
    for source in ("tdx", "sina", "hf", "baostock", "eastmoney", "import"):
        if source in lowered:
            return source
    if "历史分钟" in message:
        return "hf"
    if "近期分钟" in message:
        return "tdx"
    return None


def _is_action_warning(message: str) -> bool:
    lowered = message.lower()
    return any(word.lower() in lowered for word in _ACTION_WORDS)


def _is_source_history(message: str) -> bool:
    if _is_action_warning(message):
        return False
    lowered = message.lower()
    return message.startswith(_HISTORICAL_SOURCE_PREFIXES) or any(word.lower() in lowered for word in _SOURCE_FAILURE_WORDS)


def _event(message: str, *, kind: str = "source_attempt", source: str | None = None, status: str = "historical", at=None) -> dict[str, Any]:
    return {
        "kind": kind,
        "source": source or _source_from_message(message),
        "message": str(message),
        "status": status,
        "at": at,
    }


def _dedupe_events(events) -> list[dict[str, Any]]:
    output, seen = [], set()
    for value in events:
        if isinstance(value, str):
            event = _event(value)
        elif isinstance(value, dict):
            message = str(value.get("message") or value.get("error") or "").strip()
            if not message:
                continue
            event = {
                "kind": str(value.get("kind") or "source_attempt"),
                "source": value.get("source") or _source_from_message(message),
                "message": message,
                "status": str(value.get("status") or "historical"),
                "at": value.get("at"),
            }
        else:
            continue
        key = _event_key(event)
        if key in seen:
            continue
        seen.add(key)
        output.append(event)
    return output


def _range_source_spans(metadata: dict, start, end) -> list[dict[str, Any]]:
    source = ((metadata or {}).get("diagnostics") or {}).get("source_spans", [])
    lower, upper = _day(start), _day(end)
    result = []
    for span in source if isinstance(source, list) else []:
        if not isinstance(span, dict):
            continue
        span_start, span_end = _day(span.get("start")), _day(span.get("end"))
        if span_start is None or span_end is None:
            continue
        clipped_start = max(span_start, lower) if lower is not None else span_start
        clipped_end = min(span_end, upper) if upper is not None else span_end
        if clipped_start > clipped_end:
            continue
        item = dict(span)
        item["start"] = clipped_start.date().isoformat()
        item["end"] = clipped_end.date().isoformat()
        # Spans are observed-row summaries.  Do not invent an exact count for a
        # clipped historical span when per-day observations were not persisted.
        if clipped_start != span_start or clipped_end != span_end:
            item["days"] = None
            item["rows"] = None
        result.append(item)
    return result


def project_data_diagnostics(
    metadata: dict | None,
    quality: dict | None,
    start,
    end,
    profiles: dict | None = None,
) -> dict[str, Any]:
    """Project current quality and historical source events without stale fields.

    ``quality`` must be the report calculated for the current request or merged
    snapshot.  ``metadata.quality_summary`` and legacy ``minute_status`` are
    intentionally ignored.
    """
    metadata = dict(metadata or {})
    quality = dict(quality or {})
    lower, upper = _day(start), _day(end)
    days = quality.get("days") if isinstance(quality.get("days"), list) else []
    minute_days = max(0, int(quality.get("minute_days", 0) or 0))
    daily_days = max(0, int(quality.get("daily_days", 0) or 0))
    total_days = max(0, int(quality.get("total_days", len(days)) or 0))
    pending_calibration_days = max(0, int(quality.get("pending_calibration_days", 0) or 0))
    suspended_days = max(0, int(quality.get("suspended_days", 0) or 0))
    requested_frequency = quality.get("requested_frequency")
    minute_quality_applicable = requested_frequency != "1d"
    minute_ratio = quality.get("minute_ratio")
    if minute_ratio is None:
        minute_ratio = minute_days / max(1, minute_days + daily_days)
    try:
        minute_ratio = min(1.0, max(0.0, float(minute_ratio)))
    except (TypeError, ValueError):
        minute_ratio = 0.0
    if not minute_quality_applicable:
        minute_ratio = None

    missing_minute_dates = [
        str(row.get("date")) for row in days
        if row.get("mode") == "daily" and row.get("download_needed", False) and row.get("date")
    ]
    if not days:
        missing_minute_dates = list(quality.get("missing_minute_dates") or [])
    missing_raw_minutes = sum(
        max(0, int(row.get("missing_data_minutes", row.get("missing_minutes", 0)) or 0))
        for row in days
        if row.get("mode") == "minute"
    ) if days else max(0, int(quality.get("missing_raw_minutes", 0) or 0))
    grid_missing_minutes = sum(
        max(0, int(row.get("missing_minutes", 0) or 0))
        for row in days
        if row.get("mode") == "minute"
    ) if days else max(0, int(quality.get("grid_missing_minutes", 0) or 0))
    auction_placeholders = max(
        0,
        int(quality.get("auction_placeholders", 0) or 0),
        sum(max(0, int(row.get("auction_placeholders", 0) or 0)) for row in days),
    )
    repaired_days = sum(
        1 for row in days
        if row.get("mode") == "minute" and int(row.get("missing_data_minutes", row.get("missing_minutes", 0)) or 0) > 0
    ) if days else max(0, int(quality.get("repaired_days", 0) or 0))
    synthetic_bars = max(0, int(quality.get("synthetic_bars", 0) or 0))
    repaired_bars = max(0, synthetic_bars - auction_placeholders)
    if total_days == 0:
        status = "unavailable"
    elif pending_calibration_days or (minute_quality_applicable and (daily_days or missing_minute_dates)):
        status = "partial"
    elif (minute_days if minute_quality_applicable else daily_days) + suspended_days >= total_days:
        status = "complete"
    else:
        status = "partial"

    spans = _range_source_spans(metadata, lower, upper)
    diagnostics = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "quality_rules_version": QUALITY_RULES_VERSION,
        "requested_frequency": requested_frequency,
        "minute_quality_applicable": minute_quality_applicable,
        "status": status,
        "checked_start": lower.date().isoformat() if lower is not None else None,
        "checked_end": upper.date().isoformat() if upper is not None else None,
        "minute_days": minute_days,
        "daily_days": daily_days,
        "total_days": total_days,
        "suspended_days": suspended_days,
        "pending_calibration_days": pending_calibration_days,
        "minute_ratio": minute_ratio,
        "synthetic_bars": synthetic_bars,
        "auction_placeholders": auction_placeholders,
        "repaired_bars": repaired_bars,
        "repaired_days": repaired_days,
        "missing_raw_minutes": missing_raw_minutes,
        "grid_missing_minutes": grid_missing_minutes,
        "raw_complete": None if not minute_quality_applicable else status == "complete" and pending_calibration_days == 0 and missing_raw_minutes == 0,
        "minute_through": metadata.get("minute_through"),
        "daily_through": metadata.get("daily_through"),
        "missing_minute_dates": missing_minute_dates,
        "source_spans": spans,
    }

    events = list(metadata.get("source_events") or [])
    for error in metadata.get("source_errors") or []:
        events.append(_event(str(error), kind="source_error"))

    current_warnings = []
    # Global/yearly verification flags are not request-scoped evidence. Action
    # warnings are emitted only by the window's concrete event validation.
    action_risk = False
    verification_state = None
    action_years = None

    for warning in metadata.get("warnings") or []:
        message = str(warning).strip()
        if not message:
            continue
        if _is_action_warning(message):
            if action_risk or (verification_state is None and action_years is None):
                current_warnings.append(message)
            else:
                events.append(_event(message, kind="resolved_risk", status="resolved"))
        elif _is_source_history(message):
            events.append(_event(message))
        elif message.startswith("请求截至 "):
            # A clipped-end message only applies to the exact request it names.
            requested = metadata.get("requested_end_date")
            if requested and str(requested) in message:
                current_warnings.append(message)
        else:
            current_warnings.append(message)


    for warning in quality.get("warnings") or []:
        message = str(warning).strip()
        if not message:
            continue
        if _is_source_history(message):
            events.append(_event(message, kind="quality_source_warning"))
        elif _is_action_warning(message):
            if action_risk or (verification_state is None and action_years is None):
                current_warnings.append(message)
            else:
                events.append(_event(message, kind="resolved_risk", status="resolved"))
        else:
            current_warnings.append(message)
    if minute_quality_applicable and diagnostics["daily_days"] and not any("整日降级" in warning for warning in current_warnings):
        current_warnings.append(f"{diagnostics['daily_days']} 个交易日已整日降级为日线；不还原盘中路径")
    if diagnostics["pending_calibration_days"] and not any("未校准" in warning for warning in current_warnings):
        current_warnings.append(f"{diagnostics['pending_calibration_days']} 个交易日的本地分钟数据尚未校准，已按日线处理")
    if minute_quality_applicable and missing_minute_dates:
        current_warnings.append("分钟数据当前缺口：" + ", ".join(missing_minute_dates[:12]) + ("…" if len(missing_minute_dates) > 12 else ""))
    if missing_raw_minutes:
        current_warnings.append(f"{missing_raw_minutes} 根有效交易分钟缺失已通过因果前值补齐；这些价格不属于原始分钟记录")

    events = _dedupe_events(events)
    current_warnings = list(dict.fromkeys(current_warnings))
    return {"diagnostics": diagnostics, "source_events": events, "current_warnings": current_warnings}


def add_source_events(metadata: dict | None, events) -> dict:
    """Return metadata with append-only, normalized source attempt events."""
    result = dict(metadata or {})
    existing = list(result.get("source_events") or [])
    for value in events or []:
        if isinstance(value, dict):
            existing.append(value)
        elif value:
            existing.append(_event(str(value), kind="source_error"))
    result["source_events"] = _dedupe_events(existing)
    return result

