from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..quality import _closing_auction_placeholders, minute_slots
from ..price_validation import anomalies_from_attrs
from .models import MarketDataError
from .timeframes import normalize_timeframe, timeframe_spec


MAX_DAILY_FILL = 5
MAX_RUN_FILL = 3


@dataclass
class ValidationResult:
    frame: pd.DataFrame
    report: dict
    pending: list[str]


def _calendar_dates(calendar) -> pd.DatetimeIndex:
    if calendar is None:
        return pd.DatetimeIndex([])
    if isinstance(calendar, pd.DataFrame):
        if calendar.empty or "date" not in calendar:
            return pd.DatetimeIndex([])
        values = calendar.date
    else:
        values = calendar
    return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce").dropna()).normalize().unique().sort_values()


def _period_labels(dates: pd.DatetimeIndex, timeframe: str) -> pd.DatetimeIndex:
    if not len(dates):
        return pd.DatetimeIndex([])
    if timeframe == "1week":
        groups = pd.Series(dates, index=dates).groupby(dates.to_period("W-FRI"))
    elif timeframe == "1month":
        groups = pd.Series(dates, index=dates).groupby(dates.to_period("M"))
    else:
        return dates
    return pd.DatetimeIndex([group.max() for _, group in groups])


def validate_and_fill(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    calendar=None,
    fill: bool = True,
    pending: Iterable[str] = (),
    coverage_start=None,
    coverage_end=None,
    symbol: str = "",
) -> ValidationResult:
    timeframe = normalize_timeframe(timeframe)
    spec = timeframe_spec(timeframe)
    if frame.empty:
        raise MarketDataError("没有可校验的行情记录")
    pending = list(dict.fromkeys(pending))
    blocked = pd.DatetimeIndex(
        pd.to_datetime(frame.attrs.get("blocked_timestamps", []), errors="coerce").dropna()
    )
    price_anomalies = anomalies_from_attrs(frame)
    if blocked.size:
        pending.append(f"存在 {blocked.size} 条无效价格记录，相关时间点保持缺口且不自动填充")
    result = frame.copy().sort_values("timestamp", kind="stable").reset_index(drop=True)
    dates = _calendar_dates(calendar)
    observed_days = pd.DatetimeIndex(result.timestamp.dt.normalize().unique()).sort_values()
    calendar_available = bool(len(dates))
    if not len(dates):
        dates = observed_days
        pending.append("未提供交易日历；覆盖检查仅限已有日期")
    lower = pd.Timestamp(coverage_start) if coverage_start is not None else result.timestamp.min()
    upper = pd.Timestamp(coverage_end) if coverage_end is not None else result.timestamp.max()
    dates = dates[(dates >= lower.normalize()) & (dates <= upper.normalize())]
    if spec.kind == "minute":
        bar_size = spec.minutes or 1
        if pending or not fill:
            return _minute_count_validation(
                result, spec, timeframe, symbol, dates, lower, upper, pending, blocked
            )
        slots = []
        for day in dates:
            session_slots = minute_slots(day)
            if bar_size == 1:
                slots.extend(session_slots)
            else:
                slots.extend(session_slots[(bar_size - 1)::bar_size])
        expected = pd.DatetimeIndex(sorted(set(slots)))
        expected = expected[(expected >= lower) & (expected <= upper)]
        auction_rows = []
        if timeframe == "1min" and symbol.endswith(".SZ"):
            for day in dates:
                observed = result[result.timestamp.dt.normalize() == day].set_index("timestamp").reindex(minute_slots(day))
                mask = observed.close.isna().to_numpy()
                if not mask.any():
                    continue
                placeholders = _closing_auction_placeholders(symbol, day, observed, mask)
                for timestamp in observed.index[placeholders]:
                    if timestamp in blocked:
                        continue
                    previous = observed.loc[observed.index < timestamp, "close"].dropna()
                    if previous.empty:
                        continue
                    price = float(previous.iloc[-1])
                    auction_rows.append({
                        "timestamp": timestamp, "open": price, "high": price, "low": price, "close": price,
                        "volume": 0, "turnover": 0, "source": "auction-placeholder",
                        "source_verified": True, "is_synthetic": False, "is_auction_placeholder": True,
                    })
        if auction_rows:
            result = pd.concat([result, pd.DataFrame(auction_rows)], ignore_index=True).sort_values("timestamp")
        observed = pd.DatetimeIndex(result.timestamp)
        missing = expected.difference(observed)
        filled = 0
        if fill and timeframe == "1min" and bool(result.source_verified.all()) and not pending:
            result, filled, unfillable = _fill_minutes(result, dates, blocked=blocked)
            missing = missing.difference(pd.DatetimeIndex(result.timestamp))
        else:
            unfillable = list(missing)
        gaps = _group_gaps(missing, bar_size)
        incomplete = len(unfillable) > 0
        report = {
            "expected_rows": len(expected), "observed_rows": len(observed),
            "missing_rows": len(missing), "synthetic_rows": filled,
            "gap_count": len(gaps), "gaps": gaps[:500], "unfillable": len(unfillable),
            "auction_placeholders": len(auction_rows),
            "blocked_price_rows": int(len(blocked)),
            "status": "partial" if incomplete else "complete",
        }
        _attach_price_anomalies(report, price_anomalies)
        if incomplete:
            pending.append(f"存在 {len(unfillable)} 个未补齐的分钟缺口")
    else:
        all_labels = _period_labels(dates, timeframe)
        observed_last = result.timestamp.max()
        if spec.kind in {"week", "month"}:
            observed_day = pd.Timestamp(observed_last).normalize()
            if spec.kind == "week":
                boundary = observed_day.to_period("W-FRI").end_time.normalize()
                incomplete_period = observed_day < boundary - pd.Timedelta(days=3)
            else:
                boundary = observed_day.to_period("M").end_time.normalize()
                today = pd.Timestamp.today().normalize()
                current_period = boundary.to_period("M") == today.to_period("M")
                incomplete_period = observed_day < boundary - pd.Timedelta(days=10) or (current_period and today < boundary)
            if incomplete_period:
                pending.append("最新记录属于尚未结束的周或月周期，完成水位暂不推进")
        expected_labels = all_labels
        expected_labels = expected_labels[(expected_labels >= lower.normalize()) & (expected_labels <= upper.normalize())]
        observed_labels = pd.DatetimeIndex(result.timestamp.dt.normalize())
        missing_labels = expected_labels.difference(observed_labels)
        gaps = [{"start": str(value.date()), "end": str(value.date()), "missing": 1} for value in missing_labels]
        report = {
            "expected_rows": len(expected_labels), "observed_rows": len(observed_labels),
            "expected_days": len(expected_labels), "observed_days": int(observed_labels.nunique()),
            "calendar_available": calendar_available,
            "missing_rows": len(missing_labels), "synthetic_rows": 0,
            "gap_count": len(gaps), "gaps": gaps[:500], "unfillable": len(missing_labels),
            "status": "complete" if not len(missing_labels) and len(pending) == 0 else "partial" if len(missing_labels) else "pending_validation",
            "blocked_price_rows": int(len(blocked)),
        }
        _attach_price_anomalies(report, price_anomalies)
        if len(missing_labels):
            pending.append(f"所选区间缺少 {len(missing_labels)} 个{spec.label}记录")
    result["source_verified"] = result.source_verified.map(bool)
    result["is_synthetic"] = result.is_synthetic.map(bool)
    result["is_auction_placeholder"] = result.is_auction_placeholder.map(bool)
    report["pending_validation"] = list(dict.fromkeys(pending))
    report["calendar_available"] = calendar_available
    report["coverage_status"] = "unknown" if not calendar_available else (
        "complete" if report.get("status") == "complete" else "partial"
    )
    report["first_time"] = result.timestamp.min().isoformat() if len(result) else None
    report["last_time"] = result.timestamp.max().isoformat() if len(result) else None
    return ValidationResult(result, report, report["pending_validation"])


def _minute_count_validation(
    result: pd.DataFrame, spec, timeframe: str, symbol: str,
    dates: pd.DatetimeIndex, lower, upper, pending: list[str],
    blocked: pd.DatetimeIndex,
) -> ValidationResult:
    """Fast coverage validation when interpolation is disabled or unavailable."""
    per_day = 240 // (spec.minutes or 1)
    expected_days = dates[(dates >= lower.normalize()) & (dates <= upper.normalize())]
    observed_by_day = {
        pd.Timestamp(day).normalize(): pd.DatetimeIndex(part.timestamp)
        for day, part in result.groupby(result.timestamp.dt.normalize())
    }
    missing_slots = []
    blocked_slots = []
    auction_placeholders = 0
    expected_count = 0
    for day in expected_days:
        session_slots = minute_slots(day)
        expected = session_slots if (spec.minutes or 1) == 1 else session_slots[(spec.minutes - 1)::spec.minutes]
        expected = pd.DatetimeIndex(expected)
        expected = expected[(expected >= lower) & (expected <= upper)]
        expected_count += len(expected)
        observed = observed_by_day.get(pd.Timestamp(day).normalize(), pd.DatetimeIndex([]))
        missing = expected.difference(observed)
        day_blocked = missing.intersection(blocked)
        blocked_slots.extend(day_blocked.tolist())
        missing = missing.difference(blocked)
        if timeframe == "1min" and len(expected):
            part = pd.DataFrame(index=session_slots)
            part["close"] = np.where(part.index.isin(observed), 1.0, np.nan)
            part["source_verified"] = True
            mask = part.close.isna().to_numpy()
            placeholders = _closing_auction_placeholders(symbol, day, part, mask)
            placeholder_times = pd.DatetimeIndex(part.index[placeholders])
            auction_placeholders += int(len(placeholder_times.intersection(missing)))
            missing = missing.difference(placeholder_times)
        missing_slots.extend(missing.tolist())
    missing_index = pd.DatetimeIndex(sorted(set(missing_slots)))
    deficit = len(missing_index)
    blocked_count = len(set(blocked_slots))
    gaps = _group_gaps(missing_index, spec.minutes or 1)
    if deficit:
        pending.append(f"存在 {deficit} 个未补齐的分钟缺口")
    if blocked.size:
        pending.append(f"存在 {blocked.size} 条无效价格记录，相关时间点保持缺口")
    report = {
        "expected_rows": expected_count, "observed_rows": len(result),
        "expected_days": len(expected_days),
        "observed_days": int(pd.DatetimeIndex(result.timestamp.dt.normalize().unique()).size),
        "missing_rows": deficit + blocked_count, "synthetic_rows": 0,
        "gap_count": len(gaps), "gaps": gaps[:500], "unfillable": deficit + blocked_count,
        "auction_placeholders": auction_placeholders,
        "blocked_price_rows": int(blocked_count),
        "status": "partial" if deficit or blocked_count else "pending_validation" if pending else "complete",
    }
    _attach_price_anomalies(report, anomalies_from_attrs(result))
    output = result.copy()
    output["source_verified"] = output.source_verified.map(bool)
    output["is_synthetic"] = output.is_synthetic.map(bool)
    output["is_auction_placeholder"] = output.is_auction_placeholder.map(bool)
    report["pending_validation"] = list(dict.fromkeys(pending))
    report["first_time"] = output.timestamp.min().isoformat() if len(output) else None
    report["last_time"] = output.timestamp.max().isoformat() if len(output) else None
    return ValidationResult(output, report, report["pending_validation"])


def _attach_price_anomalies(report: dict, price_anomalies) -> None:
    if price_anomalies is None or price_anomalies.empty:
        return
    columns = [
        column for column in ("timestamp", "date", "source", "quality_reason")
        if column in price_anomalies
    ]
    report["price_anomalies"] = price_anomalies.head(200)[columns].astype(str).to_dict("records")


def _fill_minutes(frame: pd.DataFrame, days: pd.DatetimeIndex, *, blocked: pd.DatetimeIndex | None = None):
    pieces = []
    filled = 0
    unfillable: list[pd.Timestamp] = []
    grouped = {day.normalize(): part.set_index("timestamp").sort_index() for day, part in frame.groupby(frame.timestamp.dt.normalize())}
    blocked = pd.DatetimeIndex([] if blocked is None else blocked)
    for day in days:
        observed = grouped.get(day.normalize(), pd.DataFrame())
        aligned = observed.reindex(minute_slots(day))
        missing = aligned.close.isna()
        blocked_mask = aligned.index.isin(blocked)
        if blocked_mask.any():
            unfillable.extend(aligned.index[blocked_mask & missing])
        if int(missing.sum()) > MAX_DAILY_FILL:
            unfillable.extend(aligned.index[missing])
            pieces.append(aligned[aligned.close.notna()].reset_index().rename(columns={"index": "timestamp"}))
            continue
        sessions = [(0, 120), (120, 240)]
        fillable = pd.Series(False, index=aligned.index)
        for start, end in sessions:
            segment_missing = missing.iloc[start:end].to_numpy()
            if missing.iloc[start] or missing.iloc[end - 1]:
                unfillable.extend(aligned.index[start:end][segment_missing])
                continue
            for pos, flag in enumerate(segment_missing):
                if not flag:
                    continue
                run_start = pos
                while pos + 1 < len(segment_missing) and segment_missing[pos + 1]:
                    pos += 1
                run = pos - run_start + 1
                if run <= MAX_RUN_FILL and segment_missing[:run_start].sum() + run <= MAX_DAILY_FILL:
                    stretch = aligned.index[start + run_start:start + pos + 1]
                    if not pd.DatetimeIndex(stretch).isin(blocked).any():
                        fillable.iloc[start + run_start:start + pos + 1] = True
                    else:
                        unfillable.extend(stretch[segment_missing[run_start:pos + 1]])
                else:
                    unfillable.extend(aligned.index[start + run_start:start + pos + 1])
        if fillable.any():
            for start, end in sessions:
                session = aligned.iloc[start:end].copy()
                session["close"] = session.close.ffill()
                session["open"] = session.open.fillna(session.close)
                session["high"] = session.high.fillna(session.close)
                session["low"] = session.low.fillna(session.close)
                session["volume"] = session.volume.fillna(0)
                session["turnover"] = session.turnover.fillna(0)
                session.loc[fillable.iloc[start:end], "is_synthetic"] = True
                session.loc[fillable.iloc[start:end], "cutfill_reason"] = "少量内部缺口以前值补齐"
                aligned.iloc[start:end] = session
            filled += int(fillable.sum())
        kept = aligned[aligned.close.notna()].reset_index().rename(columns={"index": "timestamp"})
        pieces.append(kept)
    result = pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()
    result["source"] = result.source.fillna("fill")
    result.loc[result.is_synthetic, "source"] = "fill"
    result["tradable"] = ~result.is_synthetic.astype(bool)
    return result.sort_values("timestamp").reset_index(drop=True), filled, unfillable


def _group_gaps(missing: pd.DatetimeIndex, interval: int):
    if not len(missing):
        return []
    groups = []
    start = previous = missing[0]
    count = 1
    for value in missing[1:]:
        if (value - previous) == pd.Timedelta(minutes=interval):
            previous = value
            count += 1
            continue
        groups.append({"start": start.isoformat(), "end": previous.isoformat(), "missing": count})
        start = previous = value
        count = 1
    groups.append({"start": start.isoformat(), "end": previous.isoformat(), "missing": count})
    return groups
