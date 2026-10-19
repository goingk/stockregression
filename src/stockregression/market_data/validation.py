from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from ..price_validation import anomalies_from_attrs
from .models import MarketDataError
from .quality import validate_and_fill
from .repository import VALIDATION_RULES_VERSION, _json, _time_text, utc_now
from .timeframes import MINUTE_TIMEFRAMES, normalize_timeframe


def _day_hash(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty:
        return None
    fields = [
        field for field in (
            "timestamp", "open", "high", "low", "close", "volume", "turnover",
            "preclose", "tradestatus", "is_st", "source", "source_verified",
            "is_synthetic", "is_auction_placeholder",
        ) if field in frame
    ]
    canonical = frame[fields].sort_values("timestamp").reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(canonical, index=False).values.tobytes()
    ).hexdigest()


def _dependency_hash(repository, symbol: str, timeframe: str) -> str:
    calendar = repository.auxiliary(symbol, "calendar")
    actions = repository.auxiliary(symbol, "actions")
    daily = repository.dataset(symbol, "1day")
    payload = {
        "rules": VALIDATION_RULES_VERSION,
        "calendar": (calendar or {}).get("version"),
        "actions": (actions or {}).get("version"),
        "daily": int(daily["current_version"]) if daily else None,
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def _calendar_days(service, symbol: str, start, end) -> pd.DatetimeIndex:
    calendar = service._calendar(symbol)
    values = pd.DatetimeIndex([])
    if calendar is not None and not calendar.empty and "date" in calendar:
        values = pd.DatetimeIndex(pd.to_datetime(calendar.date, errors="coerce").dropna())
    daily = service.repository.active_rows(
        symbol, "1day", start=pd.Timestamp(start).normalize(), end=pd.Timestamp(end).normalize()
    )
    if not daily.empty and "timestamp" in daily:
        values = values.append(pd.DatetimeIndex(pd.to_datetime(daily.timestamp, errors="coerce").dropna()))
    return values.normalize().unique().sort_values().intersection(
        pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    )


def validate_incremental(
    service,
    symbol: str,
    timeframe: str,
    *,
    mode: str = "incremental",
    start=None,
    end=None,
    progress=lambda *_: None,
    cancelled=lambda: False,
) -> dict:
    """Validate only absent, failed or explicitly re-opened trading days."""
    repository = service.repository
    symbol, timeframe = symbol, normalize_timeframe(timeframe)
    dataset = repository.dataset(symbol, timeframe)
    if not dataset:
        raise MarketDataError("没有可校验的本地行情")
    if mode not in {"incremental", "full"}:
        raise MarketDataError("校验模式应为 incremental/full")
    lower = pd.Timestamp(start or dataset["first_time"]).normalize()
    upper = pd.Timestamp(end or dataset["last_time"]).normalize()
    if upper < lower:
        raise MarketDataError("校验开始日期不能晚于结束日期")
    period_mode = timeframe in {"1week", "1month"}
    if period_mode:
        period_frame = repository.active_rows(symbol, timeframe, start=lower, end=upper)
        calendar_days = (
            pd.DatetimeIndex(period_frame.timestamp.dt.normalize().unique()).sort_values()
            if not period_frame.empty else pd.DatetimeIndex([])
        )
        calendar_available = bool(len(calendar_days))
    else:
        calendar_days = _calendar_days(service, symbol, lower, upper)
        calendar_available = bool(len(calendar_days))
    if not calendar_available:
        sample = repository.active_rows(symbol, timeframe, start=lower, end=upper)
        calendar_days = pd.DatetimeIndex(
            sample.timestamp.dt.normalize().unique()
        ).sort_values() if not sample.empty else pd.DatetimeIndex([])
    if mode == "full":
        repository.invalidate_day_validations(
            symbol, timeframe, [str(day.date()) for day in calendar_days]
        )
    existing = {
        pd.Timestamp(row["trade_date"]).normalize(): row
        for row in repository.day_validations(symbol, timeframe, start=lower, end=upper)
    }
    targets = [
        day for day in calendar_days
        if mode == "full"
        or pd.Timestamp(day).normalize() not in existing
        or existing[pd.Timestamp(day).normalize()].get("status") not in {"passed", "suspended"}
        or int(existing[pd.Timestamp(day).normalize()].get("rules_version") or 0)
        < VALIDATION_RULES_VERSION
    ]
    if not targets:
        state = repository.validation_state(symbol, timeframe, start=lower, end=upper)
        return {
            "symbol": symbol, "timeframe": timeframe, "status": state.get("verified_through") and "complete" or "partial",
            "mode": mode, "checked_days": 0, "skipped_days": len(calendar_days),
            "verified_from": state.get("verified_from"),
            "verified_through": state.get("verified_through"),
            "pending_validation": state.get("pending", []),
            "calendar_available": state.get("calendar_available"),
        }

    daily_by_day: dict[pd.Timestamp, pd.Series] = {}
    if timeframe in MINUTE_TIMEFRAMES:
        daily = repository.active_rows(symbol, "1day", start=lower, end=upper)
        if not daily.empty:
            daily["day"] = pd.to_datetime(daily.timestamp).dt.normalize()
            daily_by_day = {
                pd.Timestamp(day).normalize(): part.iloc[-1]
                for day, part in daily.groupby("day", sort=True)
            }
    dependency = _dependency_hash(repository, symbol, timeframe)
    checked: list[dict] = []
    checked_days = 0
    for index, day in enumerate(targets, start=1):
        if cancelled():
            raise InterruptedError("任务已取消")
        day = pd.Timestamp(day).normalize()
        progress(index / max(1, len(targets)), f"校验 {day.date()} {timeframe}")
        report: dict[str, Any] = {}
        status = "failed"
        reason = None
        skip_minute = False
        if timeframe in MINUTE_TIMEFRAMES:
            daily_row = daily_by_day.get(day)
            if daily_row is not None:
                daily_status = pd.to_numeric(
                    pd.Series([daily_row.get("tradestatus")]), errors="coerce"
                ).iloc[0]
                daily_volume = pd.to_numeric(
                    pd.Series([daily_row.get("volume")]), errors="coerce"
                ).iloc[0]
                if pd.notna(daily_status) and int(daily_status) == 0:
                    status = "suspended"
                    reason = "确认停牌"
                    report = {"status": "complete", "expected_rows": 0, "observed_rows": 0,
                              "reason": reason}
                    skip_minute = True
                elif pd.notna(daily_volume) and float(daily_volume) == 0:
                    status = "passed"
                    reason = "交易状态正常但全天无成交量"
                    report = {"status": "complete", "expected_rows": 0, "observed_rows": 0,
                              "no_trades": True, "reason": reason}
                    skip_minute = True
            if status != "suspended" and daily_row is None:
                reason = "缺少对应日线，无法确认交易状态与分钟依赖"
        part = pd.DataFrame()
        if not skip_minute and status != "suspended":
            part = repository.active_rows(symbol, timeframe, start=day, end=day)
            price_anomalies = anomalies_from_attrs(part)
            if not price_anomalies.empty:
                status = "failed"
                reason = f"存在 {len(price_anomalies)} 条无效价格，已隔离且禁止补造"
                report = {
                    "status": "failed", "expected_rows": 1,
                    "observed_rows": 0, "gap_count": len(price_anomalies),
                    "blocked_price_rows": len(price_anomalies),
                    "price_anomalies": price_anomalies.head(50).astype(str).to_dict("records"),
                    "reason": reason,
                }
            elif part.empty:
                reason = reason or "交易日没有行情记录"
            elif period_mode:
                if part.empty:
                    reason = "周期没有行情记录"
                elif part.source_verified.all():
                    status = "passed"
                    report = {
                        "status": "complete", "expected_rows": 1,
                        "observed_rows": 1, "gap_count": 0,
                    }
                else:
                    reason = "周期来源尚未核验"
            else:
                calendar_part = pd.DataFrame({"date": [day]})
                pending: list[str] = []
                if timeframe == "1day":
                    required = ["open", "high", "low", "close", "volume", "preclose", "tradestatus"]
                    missing = [field for field in required if field not in part or part[field].isna().all()]
                    asset_type = str((service.store.instrument(symbol) or {}).get("asset_type") or "")
                    if asset_type == "stock" and ("is_st" not in part or part.is_st.isna().any()):
                        missing.append("is_st")
                    if missing:
                        pending.append("缺少严格回测字段：" + ", ".join(missing))
                validation = validate_and_fill(
                    part, timeframe, calendar=calendar_part, fill=True,
                    pending=pending, coverage_start=day, coverage_end=day, symbol=symbol,
                )
                report = validation.report
                if (
                    validation.report.get("status") == "complete"
                    and not validation.pending
                ):
                    status = "passed"
                else:
                    reason = "; ".join(validation.pending) or "所选区间存在覆盖缺口"
        checked.append({
            "symbol": symbol, "timeframe": timeframe,
            "adjustment": dataset.get("adjustment", "none"),
            "trade_date": str(day.date()), "status": status,
            "row_count": len(part), "source_hash": _day_hash(part),
            "dependency_hash": dependency, "rules_version": VALIDATION_RULES_VERSION,
            "report": {**report, "reason": reason},
            "checked_at": utc_now(),
        })
        checked_days += 1
    repository.record_day_validations(checked)
    state = repository.validation_state(symbol, timeframe, start=lower, end=upper)
    passed = sum(row["status"] == "passed" for row in checked)
    suspended = sum(row["status"] == "suspended" for row in checked)
    failed = sum(row["status"] == "failed" for row in checked)
    repository.save_coverage(
        symbol, timeframe, dataset.get("adjustment", "none"), lower, upper,
        {
            "status": "complete" if not failed else "partial",
            "expected_rows": int(sum(int(row.get("report", {}).get("expected_rows", 0) or 0) for row in checked)),
            "observed_rows": int(sum(int(row.get("report", {}).get("observed_rows", 0) or 0) for row in checked)),
            "synthetic_rows": int(sum(int(row.get("report", {}).get("synthetic_rows", 0) or 0) for row in checked)),
            "gap_count": int(sum(int(row.get("report", {}).get("gap_count", 0) or 0) for row in checked)),
            "checked_days": checked_days, "failed_days": failed,
            "calendar_available": calendar_available,
            "coverage_status": "complete" if not failed and state.get("verified_through") else "partial",
            "reason": None if not failed else f"{failed} 个交易日未通过校验",
        },
    )
    repository.update_dataset_status(
        symbol, timeframe, dataset.get("adjustment", "none"),
        "ready" if not failed and state.get("verified_through") else "partial",
        None if not failed else f"{failed} 个交易日未通过校验",
    )
    return {
        "symbol": symbol, "timeframe": timeframe,
        "status": "complete" if not failed else "partial", "mode": mode,
        "checked_days": checked_days, "passed_days": passed,
        "suspended_days": suspended, "failed_days": failed,
        "skipped_days": max(0, len(calendar_days) - checked_days),
        "verified_from": state.get("verified_from"),
        "verified_through": state.get("verified_through"),
        "pending_validation": state.get("pending", []),
        "calendar_available": calendar_available,
    }
