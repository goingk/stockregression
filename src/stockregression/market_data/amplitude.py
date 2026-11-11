"""Pure daily amplitude calculations for the market analysis API."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_price(value: Any) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number > 0 else None


def _date_text(value: Any) -> str | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.normalize().date().isoformat()


def summarize_amplitude(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return arithmetic mean and median for the shared valid sample set."""
    result: dict[str, dict[str, float]] = {}
    for column in ("amplitude", "upside", "downside"):
        values = sorted(float(sample[column]) for sample in samples)
        count = len(values)
        if not count:
            result[column] = {"mean": math.nan, "median": math.nan}
            continue
        mean = sum(values) / count
        if count % 2:
            median = values[count // 2]
        else:
            median = (values[count // 2 - 1] + values[count // 2]) / 2.0
        result[column] = {"mean": float(mean), "median": float(median)}
    return result


def calculate_daily_amplitude(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate one amplitude sample per usable daily bar.

    ``records`` preserves every input bar in date order so callers can apply
    their own lookback/date range without losing the reason a row was skipped.
    Only ``samples`` participates in the three requested metrics.
    """
    columns = ["timestamp", "high", "low", "preclose", "tradestatus"]
    if frame is None or frame.empty:
        return {
            "samples": [], "metrics": summarize_amplitude([]),
            "excluded_count": 0, "excluded_records": [], "records": [],
        }

    working = frame.copy()
    if "timestamp" not in working and "date" in working:
        working["timestamp"] = working["date"]
    for column in columns:
        if column not in working:
            working[column] = pd.NA
    working["_timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.sort_values("_timestamp", kind="stable", na_position="last")

    samples: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, str]] = []
    previous_valid_close: float | None = None
    has_open = "open" in working.columns
    has_close = "close" in working.columns
    price_tolerance = 1e-9

    for _, row in working.iterrows():
        day = _date_text(row.get("timestamp"))
        reason: str | None = None
        sample: dict[str, Any] | None = None

        if day is None:
            reason = "行情日期无效"

        status = _finite_number(row.get("tradestatus"))
        if reason is None and status is not None and status == 0:
            reason = "停牌（tradestatus=0）"

        open_price = _valid_price(row.get("open")) if has_open else None
        high = _valid_price(row.get("high"))
        low = _valid_price(row.get("low"))
        close = _valid_price(row.get("close")) if has_close else None
        prices_present = (
            high is not None and low is not None
            and (not has_open or open_price is not None)
            and (not has_close or close is not None)
        )
        bounds_valid = prices_present and high >= low
        range_valid = bounds_valid and (
            high >= max(open_price or low, close or low) - price_tolerance
            and low <= min(open_price or high, close or high) + price_tolerance
        )
        ohlc_valid = prices_present and range_valid
        if reason is None and not prices_present:
            reason = "OHLC 价格无效"
        if reason is None and not bounds_valid:
            reason = "最高价低于最低价"
        if reason is None and not range_valid:
            reason = "开盘价或收盘价超出最高价与最低价范围"

        if reason is None:
            reference = _valid_price(row.get("preclose"))
            if reference is None:
                reference = previous_valid_close
            if reference is None:
                reason = "缺少可用的前收盘价"
            else:
                amplitude = (high - low) / reference
                upside = max(0.0, high / reference - 1.0)
                downside = max(0.0, 1.0 - low / reference)
                if not all(math.isfinite(value) for value in (amplitude, upside, downside)):
                    reason = "振幅计算结果无效"
                else:
                    sample = {
                        "date": day,
                        "previous_close": float(reference),
                        "high": float(high),
                        "low": float(low),
                        "amplitude": float(amplitude),
                        "upside": float(upside),
                        "downside": float(downside),
                    }

        # A valid traded close remains the fallback reference even when this
        # row cannot form a sample because it has no previous close yet.
        if status != 0 and ohlc_valid and close is not None:
            previous_valid_close = close

        if sample is not None:
            samples.append(sample)
        else:
            excluded_records.append({"date": day or "", "reason": reason or "无效记录"})
        records.append({"date": day, "sample": sample, "reason": reason})

    return {
        "samples": samples,
        "metrics": summarize_amplitude(samples),
        "excluded_count": len(excluded_records),
        "excluded_records": excluded_records,
        "records": records,
    }
