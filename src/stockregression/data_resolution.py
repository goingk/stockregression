"""Deterministic, snapshot-bound choices of trustworthy data for each session.

Planning reads the quality catalogue, not price arrays. Consumers still verify
the selected immutable prices; a bad minute day is never certified as complete.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

import pandas as pd


POLICY = "minute_verified_daily_then_skip_v1"
MODES = ("minute", "daily", "skip", "suspended", "no_trades")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _plain(value.item())
    return str(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _date(value) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("DATA_RESOLUTION_MISMATCH: 日期无效")
    return stamp.date().isoformat()


def _versions(manifest: dict) -> dict:
    return {
        "market_versions": {key: int(value["version"])
                            for key, value in manifest.get("market_versions", {}).items()},
        "auxiliary_versions": dict(manifest.get("auxiliary_versions", {})),
    }


def _records(repository, symbol, timeframe, start, end) -> list[dict]:
    # The published coverage archive supersedes the legacy per-day ledger.
    if hasattr(repository, "_connect"):
        from .market_data.coverage_index import CoverageArchive
        archive = CoverageArchive(repository)
        rows = archive.rows(symbol, timeframe, start=start, end=end)
        if rows or archive.head_reference(symbol, timeframe):
            return rows
    return repository.day_validations(symbol, timeframe, start=start, end=end)


def _state(repository, symbol, timeframe, manifest, start, end) -> dict:
    reference = manifest.get("market_versions", {}).get(timeframe)
    daily = manifest.get("market_versions", {}).get("1day")
    if reference is None or daily is None:
        return {"available": False, "failed": [], "pending": []}
    return repository.validation_state(symbol, timeframe, start=start, end=end,
        version=int(reference["version"]),
        calendar_version=manifest.get("auxiliary_versions", {}).get("calendar"),
        daily_version=int(daily["version"]))


def build_data_resolution(market, symbol: str, manifest: dict, days) -> dict:
    """Choose daily fallbacks/skip independently, with no cohort ratio gate."""
    from .market_data.repository import (
        VALIDATION_RULES_VERSION, minute_grid_report_stale, validation_rules_version,
    )

    dates = sorted({_date(day) for day in days})
    if not dates:
        raise ValueError("DATA_RESOLUTION_MISSING: 请求窗口没有可信交易日历")
    repository = market.repository
    states, records = {}, {}
    for timeframe in ("1day", "1min"):
        states[timeframe] = _state(repository, symbol, timeframe, manifest, dates[0], dates[-1])
        record_start = min(pd.Timestamp(dates[0]) - pd.Timedelta(days=35),
                           pd.Timestamp(dates[-1]) - pd.Timedelta(days=550)) if timeframe == "1day" else dates[0]
        records[timeframe] = {str(row["trade_date"])[:10]: row for row in
            _records(repository, symbol, timeframe, record_start, dates[-1])}

    def verdict(timeframe, day):
        state = states[timeframe]
        row = records[timeframe].get(day)
        pending = set(state.get("pending") or [])
        failed = set(state.get("failed") or [])
        if state.get("available") is False or day in pending:
            return None
        reference = manifest.get("market_versions", {}).get(timeframe, {})
        if ((reference.get("first") and day < _date(reference["first"]))
                or (reference.get("last") and day > _date(reference["last"]))):
            return None
        if row:
            rules_version = validation_rules_version(timeframe)
            if (row.get("dirty_reason") or not row.get("scanned", True)
                    or int(row.get("rules_version", rules_version)) < rules_version
                    or row.get("pending") or day in failed
                    or row.get("status") not in {"passed", "suspended"}):
                return None
            return row
        # Complete immutable version reports (and small catalogue adapters)
        # can certify a range without exposing individual ledger rows.
        if (not failed and not pending and not state.get("pending_count")
                and state.get("verified_from") and state.get("verified_through")
                and _date(state["verified_from"]) <= day <= _date(state["verified_through"])):
            return {"status": "passed", "row_count": 240 if timeframe == "1min" else 1,
                    "rules_version": validation_rules_version(timeframe),
                    "source_hash": _digest({"state": state, "day": day}),
                    "report": {}, "quality_source": state.get("quality_source", "range_certificate")}
        return None

    def evidence(row):
        if row is None:
            return None
        return _plain({key: row.get(key) for key in (
            "status", "row_count", "source_hash", "dependency_hash", "rules_version", "report",
        )})

    def needs_validation(timeframe, day):
        reference = manifest.get("market_versions", {}).get(timeframe)
        if not reference:
            return False
        if ((reference.get("first") and day < _date(reference["first"]))
                or (reference.get("last") and day > _date(reference["last"]))):
            return False
        if timeframe == "1min":
            daily = verdict("1day", day)
            report = (daily or {}).get("report") or {}
            if daily and (daily.get("status") == "suspended" or report.get("non_trading")
                          or report.get("suspended") or report.get("no_trades")):
                return False
        state = states[timeframe]
        if state.get("available") is False or day in set(state.get("pending") or []):
            return True
        row = records[timeframe].get(day)
        if row:
            return bool(
                row.get("dirty_reason") or not row.get("scanned", True)
                or int(row.get("rules_version", validation_rules_version(timeframe)))
                < validation_rules_version(timeframe)
                or minute_grid_report_stale(timeframe, row)
                or row.get("status") not in {"passed", "failed", "suspended"}
            )
        if day in set(state.get("failed") or []):
            return False
        return verdict(timeframe, day) is None

    output = []
    for day in dates:
        daily, minute = verdict("1day", day), verdict("1min", day)
        report = (daily or {}).get("report") or {}
        minute_report = (minute or {}).get("report") or {}
        observed = int((minute or {}).get("row_count") or minute_report.get("observed_rows") or 0)
        auction = int(minute_report.get("auction_placeholders") or 0)
        complete = bool(minute and observed + auction >= 240
                        and not minute_report.get("synthetic_rows")
                        and not minute_report.get("missing_data_minutes")
                        and not minute_report.get("blocked_price_rows")
                        and not minute_report.get("price_anomalies"))
        if not daily:
            mode, reason_code, reason = "skip", "daily_unavailable", "当天没有可核实的日线，跳过当天交易事件"
        elif daily.get("status") == "suspended" or report.get("non_trading") or report.get("suspended"):
            mode, reason_code, reason = "suspended", "suspended", "已确认停牌"
        elif report.get("no_trades"):
            mode, reason_code, reason = "no_trades", "no_trades", "已确认全天无成交"
        elif complete:
            mode, reason_code, reason = "minute", "minute_complete", "真实完整分钟线"
        else:
            mode, reason_code, reason = "daily", "minute_unavailable", "分钟线不完整或不可核实，整日使用已校验日线"
        output.append({"date": day, "mode": mode, "reason_code": reason_code, "reason": reason,
                       "evidence": {"1day": evidence(daily), "1min": evidence(minute) if complete else None}})
    counts = Counter(item["mode"] for item in output)
    evaluated = [item["date"] for item in output if item["mode"] in {"minute", "daily"}]
    validation_rules_versions = {
        timeframe: validation_rules_version(timeframe)
        for timeframe in ("1day", "1min")
    }
    result = {"schema_version": 1, "policy": POLICY, "rules_version": 1,
        "validation_rules_version": VALIDATION_RULES_VERSION,
        "validation_rules_versions": validation_rules_versions, "symbol": symbol,
        "start": dates[0], "end": dates[-1], "versions": _versions(manifest), "days": output,
        "counts": {mode: counts[mode] for mode in MODES},
        "prior_daily_days": [day for day in sorted(records["1day"])
                             if _date(pd.Timestamp(dates[0]) - pd.Timedelta(days=35)) <= day < dates[0]
                             and verdict("1day", day) is not None],
        "history_daily_days": [day for day in sorted(records["1day"])
                               if verdict("1day", day) is not None],
        "history_evidence_hash": _digest({day: evidence(verdict("1day", day))
            for day in sorted(records["1day"]) if verdict("1day", day) is not None}),
        "first_evaluable_date": evaluated[0] if evaluated else None,
        "last_evaluable_date": evaluated[-1] if evaluated else None}
    pending_validation = {
        timeframe: [day for day in dates if needs_validation(timeframe, day)]
        for timeframe in ("1day", "1min")
    }
    if any(pending_validation.values()):
        result["pending_validation"] = pending_validation
    result["plan_hash"] = _digest(result)
    return result


def validate_data_resolution(manifest: dict, start=None, end=None) -> dict:
    """Never rebuild a frozen plan from the mutable live quality ledger."""
    from .market_data.repository import VALIDATION_RULES_VERSION, validation_rules_version

    plan = manifest.get("data_resolution")
    if not isinstance(plan, dict):
        raise ValueError("DATA_RESOLUTION_MISSING: 冻结版本没有逐日数据策略")
    if (plan.get("schema_version") != 1 or plan.get("policy") != POLICY
            or plan.get("rules_version") != 1
            or plan.get("symbol") != manifest.get("symbol")
            or plan.get("versions") != _versions(manifest)
            or plan.get("plan_hash") != _digest({key: value for key, value in plan.items() if key != "plan_hash"})):
        raise ValueError("DATA_RESOLUTION_MISMATCH: 冻结的逐日策略或依赖版本不匹配")
    if any((plan.get("pending_validation") or {}).values()):
        raise ValueError("DATA_VALIDATION_PENDING: 请求区间存在待校验数据，请先运行本地校验")
    days = plan.get("days")
    if (not isinstance(days, list) or not days
            or any(not isinstance(day, dict) or day.get("mode") not in MODES for day in days)):
        raise ValueError("DATA_RESOLUTION_MISMATCH: 逐日数据模式无效")
    try:
        global_rules_version = int(plan.get("validation_rules_version", 0))
        timeframe_versions = plan.get("validation_rules_versions") or {}
        if not isinstance(timeframe_versions, dict):
            raise ValueError
        used_timeframes = set()
        for day in days:
            if day["mode"] == "minute":
                used_timeframes.update(("1day", "1min"))
            elif day["mode"] in {"daily", "suspended", "no_trades"}:
                used_timeframes.add("1day")
        if (global_rules_version < 1 or global_rules_version > VALIDATION_RULES_VERSION
                or any(
                    int(timeframe_versions.get(timeframe, global_rules_version))
                    < validation_rules_version(timeframe)
                    or int(timeframe_versions.get(timeframe, global_rules_version))
                    > VALIDATION_RULES_VERSION
                    for timeframe in used_timeframes
                )):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("DATA_RESOLUTION_MISMATCH: 冻结方案使用了过期或未知的验证口径") from None
    dates = [_date(day.get("date")) for day in days]
    counts = Counter(day["mode"] for day in days)
    if (dates != sorted(set(dates)) or dates[0] != plan.get("start") or dates[-1] != plan.get("end")
            or plan.get("counts") != {mode: counts[mode] for mode in MODES}
            or (start is not None and _date(start) != plan.get("requested_start", plan["start"]))
            or (end is not None and _date(end) != plan.get("requested_end", plan["end"]))):
        raise ValueError("DATA_RESOLUTION_MISMATCH: 逐日策略与请求窗口不一致")
    return deepcopy(plan)


def resolution_intervals(plan: dict, timeframe: str) -> list[dict]:
    accepted = {"minute"} if timeframe == "1min" else {"minute", "daily", "suspended", "no_trades"}
    result = []
    continuous = False
    for day in plan["days"]:
        if day["mode"] not in accepted:
            continuous = False
        elif continuous:
            result[-1]["end"] = day["date"]
        else:
            result.append({"start": day["date"], "end": day["date"], "status": "complete"})
            continuous = True
    return result


def resolution_sampling(plan: dict) -> dict:
    excluded = [day["date"] for day in plan["days"] if day["mode"] != "minute"]
    expected = len(plan["days"])
    valid = expected - len(excluded)
    return {"expected_days": expected, "valid_days": valid, "coverage_ratio": valid / max(1, expected),
        "excluded_days": excluded, "statistics_excluded_days": excluded.copy(),
        "statistics_days": valid, "comparable": True, "data_policy": POLICY}


def resolution_snapshot_version(manifest: dict) -> str:
    content = {key: value for key, value in manifest.items()
               if key not in {"version", "created_at", "files", "profiles"}}
    return "mkt-" + _digest(content)[:24]
