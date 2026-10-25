"""Causal daily market-state diagnostics shared by research workflows.

The state is intentionally independent from the minute-level ETF statistics.
It reads at most 250 consecutive daily sessions plus one preceding close,
expresses prices in the decision-day corporate-action basis, and reports every
unavailable metric with a reason. Missing exchange sessions break the usable
history; they are never stitched across or filled.
"""
from __future__ import annotations

from copy import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import price_basis
from .models import DataBundle


METHOD_VERSION = "daily_state_v1"
MAX_TRADING_DAYS = 250
ATR_PERIOD = 14
ADX_PERIOD = 14
HV_SHORT_PERIOD = 10
HV_LONG_PERIOD = 60
AMPLITUDE_PERIOD = 20

METRIC_KEYS = (
    "natr14",
    "hv_ratio",
    "adx14",
    "plus_di14",
    "minus_di14",
    "amplitude_q75",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _day(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if stamp is pd.NaT:
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return stamp.normalize()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        number = _finite(value)
        return number if number is not None else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _detail(
    *, status: str, reason: str, sample_count: int = 0,
    required_samples: int = 0, reason_code: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "reason_code": reason_code,
        "sample_count": int(max(0, sample_count)),
        "required_samples": int(max(0, required_samples)),
        **_json_value(extra),
    }


def _warmup() -> dict[str, Any]:
    return {
        "maximum_trading_days": MAX_TRADING_DAYS,
        "preceding_close_required": True,
        "natr14_daily_periods": ATR_PERIOD,
        "hv10_log_returns": HV_SHORT_PERIOD,
        "hv60_log_returns": HV_LONG_PERIOD,
        "hv60_daily_days_with_preclose": HV_LONG_PERIOD,
        "adx14_daily_periods": 2 * ADX_PERIOD - 1,
        "amplitude20_daily_days_with_prior_close": AMPLITUDE_PERIOD,
    }


def _empty_state(as_of: Any, reason: str, *, reason_code: str = "unavailable") -> dict[str, Any]:
    metric_details = {
        key: _detail(
            status="unavailable", reason=reason, reason_code=reason_code,
            required_samples={
                "natr14": ATR_PERIOD, "hv_ratio": HV_LONG_PERIOD,
                "adx14": 2 * ADX_PERIOD - 1, "plus_di14": ATR_PERIOD,
                "minus_di14": ATR_PERIOD, "amplitude_q75": AMPLITUDE_PERIOD,
            }[key],
        )
        for key in METRIC_KEYS
    }
    stamp = _day(as_of)
    fingerprint_payload = {
        "method_version": METHOD_VERSION,
        "as_of": stamp.date().isoformat() if stamp is not None else str(as_of),
        "daily": [], "actions": [],
    }
    return {
        "method_version": METHOD_VERSION,
        "as_of": stamp.date().isoformat() if stamp is not None else None,
        "status": "data_not_ready" if reason_code != "insufficient_samples" else "insufficient_history",
        "metrics": {key: None for key in METRIC_KEYS},
        "metric_details": metric_details,
        "samples": {
            "daily_rows": 0, "selected_start": None, "selected_end": None,
            "prior_close_date": None, "prior_close": None,
            "truncated_at_limit": False, "gap_before_start": None,
        },
        "warmup": _warmup(),
        "price_basis": {
            "method": "raw_to_as_of_corporate_action_factor",
            "as_of": stamp.date().isoformat() if stamp is not None else None,
            "actions_applied": [], "action_count": 0, "error": reason,
        },
        "input_fingerprint": hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "message": reason,
    }


def _daily_rows(data: DataBundle, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = data.daily if isinstance(data.daily, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=["date", "high", "low", "close", "preclose"])
    column = "date" if "date" in frame else "timestamp" if "timestamp" in frame else None
    if column is None:
        return pd.DataFrame(columns=["date", "high", "low", "close", "preclose"])
    rows = frame.copy()
    dates = pd.to_datetime(rows[column], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    rows["date"] = dates.dt.normalize()
    rows = rows[rows.date.notna() & rows.date.le(cutoff)].copy()
    rows = rows.sort_values("date", kind="stable").drop_duplicates("date", keep="last")
    open_provided = "open" in rows.columns
    for key in ("high", "low", "close", "preclose", "open"):
        rows[key] = (
            pd.to_numeric(rows[key], errors="coerce")
            if key in rows else np.nan
        )
    rows["_open_provided"] = open_provided
    return rows.reset_index(drop=True)


def _calendar(data: DataBundle, cutoff: pd.Timestamp) -> pd.DatetimeIndex:
    source = data.calendar
    if isinstance(source, pd.DataFrame):
        values = source["date"] if "date" in source else source.index
    else:
        values = source
    try:
        days = pd.to_datetime(list(values), errors="coerce")
    except TypeError:
        days = pd.DatetimeIndex([])
    stamps = pd.DatetimeIndex(days)
    if stamps.tz is not None:
        stamps = stamps.tz_convert("Asia/Shanghai").tz_localize(None)
    stamps = stamps.dropna().normalize().unique().sort_values()
    return pd.DatetimeIndex(stamps[stamps <= cutoff], name="date")


def _actions_before(
    data: DataBundle, cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, str | None]:
    actions = data.actions if isinstance(data.actions, pd.DataFrame) else pd.DataFrame()
    if actions.empty:
        return pd.DataFrame(columns=["ex_date"]), None
    if "ex_date" not in actions:
        return actions.copy(), "非空公司行动缺少 ex_date 字段，无法判断行动是否已发生"
    rows = actions.copy()
    dates = pd.to_datetime(rows.ex_date, errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    rows["ex_date"] = dates.dt.normalize()
    if rows.ex_date.isna().any():
        return rows, "非空公司行动含无效或缺失的 ex_date，不能忽略未知行动"
    return rows[rows.ex_date.le(cutoff)].reset_index(drop=True), None


def _valid_daily(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(False, index=rows.index, dtype=bool)
    high = pd.to_numeric(rows.high, errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(rows.low, errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(rows.close, errors="coerce").to_numpy(dtype=float)
    open_values = pd.to_numeric(rows.open, errors="coerce").to_numpy(dtype=float)
    open_provided = (
        rows["_open_provided"].fillna(False).astype(bool).to_numpy()
        if "_open_provided" in rows else np.zeros(len(rows), dtype=bool)
    )
    with np.errstate(invalid="ignore"):
        valid_hlc = (
            np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
            & (high > 0) & (low > 0) & (close > 0)
            & (high >= low) & (close >= low) & (close <= high)
        )
        valid_open = (
            np.isfinite(open_values) & (open_values > 0)
            & (open_values >= low) & (open_values <= high)
        )
    return pd.Series(valid_hlc & (~open_provided | valid_open), index=rows.index, dtype=bool)


def _select_history(
    rows: pd.DataFrame, expected: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Timestamp | None, dict[str, Any]]:
    if rows.empty or not len(expected):
        return pd.DataFrame(), None, {
            "truncated_at_limit": False, "gap_before_start": None,
            "invalid_daily_dates": [],
        }
    valid = _valid_daily(rows)
    valid_by_date = {
        pd.Timestamp(day): bool(flag)
        for day, flag in zip(rows.date, valid)
    }
    earliest = pd.Timestamp(expected[0])
    latest = pd.Timestamp(expected[-1])
    by_date = rows.set_index("date", drop=False).sort_index()
    end = len(expected) - 1
    latest_expected = pd.Timestamp(expected[end])
    if latest_expected not in by_date.index:
        return pd.DataFrame(), None, {
            "truncated_at_limit": False, "gap_before_start": None,
            "invalid_daily_dates": [],
            "latest_expected_date": latest_expected.date().isoformat(),
            "missing_latest_date": latest_expected.date().isoformat(),
        }
    if not bool(valid_by_date.get(latest_expected, False)):
        return pd.DataFrame(), None, {
            "truncated_at_limit": False, "gap_before_start": None,
            "invalid_daily_dates": [latest_expected.date().isoformat()],
            "latest_expected_date": latest_expected.date().isoformat(),
            "missing_latest_date": latest_expected.date().isoformat(),
        }

    start = end
    count = 0
    while start >= 0 and count < MAX_TRADING_DAYS:
        day = pd.Timestamp(expected[start])
        if not valid_by_date.get(day, False):
            break
        start -= 1
        count += 1
    selected_start = start + 1
    truncated = count == MAX_TRADING_DAYS and selected_start > 0
    if truncated:
        # If the stop was the 250-day limit, do not call a valid predecessor
        # a gap merely because it was not selected.
        gap = None
    else:
        gap = pd.Timestamp(expected[selected_start - 1]) if selected_start > 0 else None
    prior = pd.Timestamp(expected[selected_start - 1]) if selected_start > 0 else None
    selected_days = pd.DatetimeIndex(expected[selected_start:end + 1])
    selected = by_date.loc[selected_days].copy().reset_index(drop=True)
    invalid_dates = [
        pd.Timestamp(day).date().isoformat()
        for day in expected
        if pd.Timestamp(day) in by_date.index
        and not bool(valid_by_date.get(pd.Timestamp(day), False))
    ]
    return selected, prior, {
        "truncated_at_limit": bool(truncated),
        "gap_before_start": gap.date().isoformat() if gap is not None else None,
        "invalid_daily_dates": invalid_dates,
        "calendar_earliest": earliest.date().isoformat(),
        "calendar_latest": latest.date().isoformat(),
        "latest_expected_date": latest.date().isoformat(),
        "missing_latest_date": None,
    }


def _fingerprint(
    symbol: str, as_of: pd.Timestamp, history: pd.DataFrame, actions: pd.DataFrame,
) -> str:
    daily_rows = []
    for row in history.itertuples(index=False):
        daily_rows.append({
            "date": pd.Timestamp(row.date).date().isoformat(),
            "high": _finite(row.high), "low": _finite(row.low),
            "close": _finite(row.close), "preclose": _finite(row.preclose),
        })
    action_rows = []
    for row in actions.to_dict("records"):
        action_rows.append({
            "ex_date": _day(row.get("ex_date")).date().isoformat() if _day(row.get("ex_date")) is not None else None,
            "cash_per_share": _finite(row.get("cash_per_share")),
            "bonus_ratio": _finite(row.get("bonus_ratio")),
            "implemented": row.get("implemented"),
            "unsupported_reason": row.get("unsupported_reason"),
        })
    payload = {
        "method_version": METHOD_VERSION, "symbol": str(symbol or ""),
        "as_of": as_of.date().isoformat(), "daily": daily_rows,
        "actions": action_rows,
    }
    return hashlib.sha256(
        json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _true_ranges(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
    prior_close: float | None,
) -> list[float] | None:
    if prior_close is None or not highs:
        return None
    values: list[float] = []
    previous = prior_close
    for high, low, close in zip(highs, lows, closes):
        candidates = (float(high) - float(low), abs(float(high) - previous), abs(float(low) - previous))
        value = max(candidates)
        if not math.isfinite(value) or value < 0:
            return None
        values.append(value)
        previous = float(close)
    return values


def _natr14(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
    prior_close: float | None,
) -> tuple[float | None, dict[str, Any]]:
    required = ATR_PERIOD
    ranges = _true_ranges(highs, lows, closes, prior_close)
    if prior_close is None:
        return None, _detail(
            status="unavailable", reason="缺少连续历史的前置收盘价，无法计算真实波幅",
            reason_code="missing_preclose", required_samples=required,
        )
    if ranges is None or len(ranges) < required:
        return None, _detail(
            status="insufficient_samples", reason=f"Wilder ATR14 需要至少 {required} 个连续日线波幅",
            reason_code="insufficient_samples", sample_count=0 if ranges is None else len(ranges),
            required_samples=required,
        )
    atr = math.fsum(ranges[:required]) / required
    for value in ranges[required:]:
        atr = (atr * (required - 1) + value) / required
    close = _positive(closes[-1])
    if close is None:
        return None, _detail(
            status="unavailable", reason="最新有效收盘价为零或无效，NATR14 分母不可用",
            reason_code="zero_denominator", sample_count=len(ranges), required_samples=required,
        )
    value = float(atr / close)
    if not math.isfinite(value):
        return None, _detail(
            status="unavailable", reason="NATR14 计算结果非有限值", reason_code="invalid_value",
            sample_count=len(ranges), required_samples=required,
        )
    return value, _detail(
        status="available", reason="Wilder ATR14 除以最新有效收盘价；内部使用小数比例",
        sample_count=len(ranges), required_samples=required,
    )


def _hv_ratio(closes: Sequence[float]) -> tuple[float | None, dict[str, Any]]:
    values = np.asarray(closes, dtype=float)
    required_closes = HV_LONG_PERIOD + 1
    if len(values) < required_closes or np.any(~np.isfinite(values)) or np.any(values <= 0):
        return None, _detail(
            status="insufficient_samples",
            reason=f"HV 比值需要 {HV_LONG_PERIOD + 1} 个连续有效收盘价（60 个对数收益）",
            reason_code="insufficient_samples", sample_count=max(0, len(values) - 1),
            required_samples=HV_LONG_PERIOD,
        )
    returns = np.diff(np.log(values))
    short = returns[-HV_SHORT_PERIOD:]
    long = returns[-HV_LONG_PERIOD:]
    sigma10 = float(np.std(short, ddof=1))
    sigma60 = float(np.std(long, ddof=1))
    extra = {"hv10": sigma10 if math.isfinite(sigma10) else None,
             "hv60": sigma60 if math.isfinite(sigma60) else None}
    if not math.isfinite(sigma10) or not math.isfinite(sigma60):
        return None, _detail(
            status="unavailable", reason="HV 对数收益标准差不可计算",
            reason_code="invalid_value", sample_count=len(long),
            required_samples=HV_LONG_PERIOD, **extra,
        )
    if sigma60 <= 0:
        return None, _detail(
            status="unavailable", reason="60 日对数收益标准差为零，HV 比值分母不可用",
            reason_code="zero_denominator", sample_count=len(long),
            required_samples=HV_LONG_PERIOD, **extra,
        )
    value = float(sigma10 / sigma60)
    return value, _detail(
        status="available", reason="最近10日与60日对数收益标准差之比，std(ddof=1)",
        sample_count=len(long), required_samples=HV_LONG_PERIOD, **extra,
    )


def _adx14(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
    prior_high: float | None, prior_low: float | None, prior_close: float | None,
) -> tuple[float | None, float | None, float | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
    required_di = ATR_PERIOD
    required_adx = 2 * ADX_PERIOD - 1
    ranges = _true_ranges(highs, lows, closes, prior_close)
    if prior_high is None or prior_low is None or prior_close is None:
        base_detail = _detail(
            status="unavailable", reason="缺少连续历史的前置最高/最低/收盘价，无法计算 Wilder DM、TR 与 DI",
            reason_code="missing_preclose", required_samples=required_di,
        )
        adx_detail = {**base_detail, "required_samples": required_adx}
        return None, None, None, adx_detail, base_detail, {**base_detail}
    if ranges is None or len(ranges) < required_di:
        count = 0 if ranges is None else len(ranges)
        di_detail = _detail(
            status="insufficient_samples", reason=f"Wilder DI14 需要至少 {required_di} 个连续波幅",
            reason_code="insufficient_samples", sample_count=count, required_samples=required_di,
        )
        adx_detail = {**di_detail, "required_samples": required_adx, "reason": f"Wilder ADX14 需要至少 {required_adx} 个连续日线波幅"}
        return None, None, None, adx_detail, di_detail, {**di_detail}

    plus_dm: list[float] = []
    minus_dm: list[float] = []
    previous_high = float(prior_high)
    previous_low = float(prior_low)
    previous_close = float(prior_close)
    for high, low, close in zip(highs, lows, closes):
        up = float(high) - previous_high
        down = previous_low - float(low)
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        previous_high = float(high)
        previous_low = float(low)
        previous_close = float(close)

    # TA-Lib 0.6.4 seeds the Wilder sums with the first period-1 bars, then
    # performs one smoothing update before each of the first period DX values.
    # This yields the first ADX output at 2*period-1 completed bars.
    seed_count = required_di - 1
    smooth_tr = math.fsum(ranges[:seed_count])
    smooth_plus = math.fsum(plus_dm[:seed_count])
    smooth_minus = math.fsum(minus_dm[:seed_count])
    dx_values: list[float | None] = []
    plus_value: float | None = None
    minus_value: float | None = None

    def current_di(tr_value: float, positive: float, negative: float) -> tuple[float | None, float | None, float | None]:
        if not math.isfinite(tr_value) or tr_value <= 0:
            return None, None, None
        plus = float(100.0 * positive / tr_value)
        minus = float(100.0 * negative / tr_value)
        total = plus + minus
        dx = float(100.0 * abs(plus - minus) / total) if total > 0 else None
        return plus, minus, dx

    plus_value = None
    minus_value = None
    for index in range(seed_count, len(ranges)):
        smooth_tr = smooth_tr - smooth_tr / required_di + ranges[index]
        smooth_plus = smooth_plus - smooth_plus / required_di + plus_dm[index]
        smooth_minus = smooth_minus - smooth_minus / required_di + minus_dm[index]
        plus_value, minus_value, dx = current_di(smooth_tr, smooth_plus, smooth_minus)
        dx_values.append(dx)

    direction = None
    if plus_value is not None and minus_value is not None:
        if plus_value > minus_value + 1e-12:
            direction = "up"
        elif minus_value > plus_value + 1e-12:
            direction = "down"
    di_counts = len(ranges)
    di_detail = _detail(
        status="available" if plus_value is not None and minus_value is not None else "unavailable",
        reason="Wilder +DI14；以 14 周期平滑 TR 归一" if plus_value is not None else "平滑 TR 为零，DI14 分母不可用",
        reason_code=None if plus_value is not None else "zero_denominator",
        sample_count=di_counts, required_samples=required_di, direction=direction,
    )
    minus_detail = _detail(
        status="available" if minus_value is not None else "unavailable",
        reason="Wilder -DI14；以 14 周期平滑 TR 归一" if minus_value is not None else "平滑 TR 为零，DI14 分母不可用",
        reason_code=None if minus_value is not None else "zero_denominator",
        sample_count=di_counts, required_samples=required_di, direction=direction,
    )
    if len(dx_values) < ADX_PERIOD:
        adx_detail = _detail(
            status="insufficient_samples", reason=f"Wilder ADX14 需要至少 {required_adx} 个连续日线波幅",
            reason_code="insufficient_samples", sample_count=len(dx_values),
            required_samples=required_adx, direction=direction,
        )
        return None, plus_value, minus_value, adx_detail, di_detail, minus_detail
    if any(value is None for value in dx_values[:ADX_PERIOD]):
        adx_detail = _detail(
            status="unavailable", reason="初始 DX 期间没有方向或平滑 TR 为零，ADX14 不生成",
            reason_code="no_direction", sample_count=len(dx_values),
            required_samples=required_adx, direction=direction,
        )
        return None, plus_value, minus_value, adx_detail, di_detail, minus_detail

    adx = math.fsum(float(value) for value in dx_values[:ADX_PERIOD]) / ADX_PERIOD
    for dx in dx_values[ADX_PERIOD:]:
        if dx is None:
            adx_detail = _detail(
                status="unavailable", reason="ADX 平滑期间出现零方向分母",
                reason_code="zero_denominator", sample_count=len(dx_values),
                required_samples=required_adx, direction=direction,
            )
            return None, plus_value, minus_value, adx_detail, di_detail, minus_detail
        adx = (adx * (ADX_PERIOD - 1) + float(dx)) / ADX_PERIOD
    if direction is None:
        adx_detail = _detail(
            status="unavailable", reason="+DI14 与 -DI14 没有明确方向，ADX14 不生成",
            reason_code="no_direction", sample_count=len(dx_values),
            required_samples=required_adx, direction=None,
        )
        return None, plus_value, minus_value, adx_detail, di_detail, minus_detail
    return float(adx), plus_value, minus_value, _detail(
        status="available", reason="Wilder ADX14；14 个初始 DX 均值后递推平滑",
        sample_count=len(dx_values), required_samples=required_adx, direction=direction,
    ), di_detail, minus_detail


def _amplitude_q75(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
    prior_close: float | None,
) -> tuple[float | None, dict[str, Any]]:
    if len(highs) < AMPLITUDE_PERIOD:
        return None, _detail(
            status="insufficient_samples", reason=f"振幅Q75需要最近 {AMPLITUDE_PERIOD} 个连续交易日",
            reason_code="insufficient_samples", sample_count=len(highs),
            required_samples=AMPLITUDE_PERIOD,
        )
    start = len(highs) - AMPLITUDE_PERIOD
    amplitudes: list[float] = []
    for index in range(start, len(highs)):
        previous = prior_close if index == 0 else closes[index - 1]
        denominator = _positive(previous)
        if denominator is None:
            return None, _detail(
                status="unavailable", reason="振幅计算缺少最近20日的前置收盘价",
                reason_code="missing_preclose", sample_count=len(amplitudes),
                required_samples=AMPLITUDE_PERIOD,
            )
        value = float((float(highs[index]) - float(lows[index])) / denominator)
        if not math.isfinite(value):
            return None, _detail(
                status="unavailable", reason="振幅计算结果非有限值",
                reason_code="invalid_value", sample_count=len(amplitudes),
                required_samples=AMPLITUDE_PERIOD,
            )
        amplitudes.append(value)
    value = float(np.quantile(np.asarray(amplitudes), 0.75, method="linear"))
    return value, _detail(
        status="available", reason="最近20日 (high-low)/前收 的线性插值75分位；零振幅保留",
        sample_count=len(amplitudes), required_samples=AMPLITUDE_PERIOD,
    )


def daily_market_state(data: DataBundle, as_of: Any) -> dict[str, Any]:
    """Return a causal, corporate-action-aware daily state at ``as_of``."""
    cutoff = _day(as_of)
    if cutoff is None:
        return _empty_state(as_of, "as_of 不是有效日期", reason_code="invalid_as_of")
    rows = _daily_rows(data, cutoff)
    if rows.empty:
        calendar = _calendar(data, cutoff)
        if len(calendar):
            missing = calendar[-1]
            result = _empty_state(
                cutoff,
                f"截止交易日 {missing.date().isoformat()} 没有有效的原始日线，状态不可用",
                reason_code="missing_latest_daily",
            )
            result["status"] = "unavailable"
            result["samples"]["latest_expected_date"] = missing.date().isoformat()
            result["samples"]["missing_latest_date"] = missing.date().isoformat()
            return result
        return _empty_state(cutoff, "截止日之前没有可用日线", reason_code="insufficient_samples")

    expected = _calendar(data, cutoff)
    if not len(expected):
        expected = pd.DatetimeIndex(rows.date.drop_duplicates().sort_values(), name="date")
    else:
        expected = expected.union(pd.DatetimeIndex(rows.date, name="date")).sort_values()
    selected, prior_date, selection = _select_history(rows, expected)
    if selected.empty:
        missing_latest = selection.get("missing_latest_date")
        if missing_latest:
            reason = f"截止交易日 {missing_latest} 没有有效的原始日线，状态不可用"
            reason_code = "missing_latest_daily"
        else:
            reason = "截止日附近没有有效日线 OHLC"
            reason_code = "invalid_value"
        result = _empty_state(cutoff, reason, reason_code=reason_code)
        result["status"] = "unavailable" if missing_latest else "data_not_ready"
        result["samples"].update(selection)
        return result

    by_date = rows.set_index("date", drop=False).sort_index()
    prior_row = pd.DataFrame()
    borrowed_prior = False
    if prior_date is not None and prior_date in by_date.index:
        candidate = by_date.loc[[prior_date]].copy()
        if bool(_valid_daily(candidate).iloc[0]):
            prior_row = candidate
    if prior_row.empty and len(selected) > 1:
        # With no earlier calendar row, the first usable bar supplies the
        # previous high/low/close for Wilder DM/TR and is not a metric sample.
        prior_row = selected.iloc[[0]].copy()
        prior_date = pd.Timestamp(prior_row.date.iloc[0])
        selected = selected.iloc[1:].reset_index(drop=True)
        borrowed_prior = True
    raw_history = pd.concat([prior_row, selected], ignore_index=True).sort_values("date")
    raw_history["preclose"] = pd.to_numeric(raw_history.preclose, errors="coerce")

    relevant_actions, action_date_error = _actions_before(data, cutoff)
    if action_date_error:
        result = _empty_state(cutoff, action_date_error, reason_code="invalid_actions")
        result["status"] = "data_not_ready"
        result["samples"].update(selection)
        result["samples"].update({
            "daily_rows": int(len(selected)),
            "selected_start": selected.date.iloc[0].date().isoformat(),
            "selected_end": selected.date.iloc[-1].date().isoformat(),
            "prior_close_date": prior_date.date().isoformat() if prior_date is not None else None,
            "prior_close": None,
        })
        result["price_basis"]["error"] = action_date_error
        result["input_fingerprint"] = _fingerprint(
            str(data.symbol or ""), cutoff, raw_history, relevant_actions,
        )
        return result
    history_start = prior_date if prior_date is not None else pd.Timestamp(selected.date.iloc[0])
    if not relevant_actions.empty:
        relevant_actions = relevant_actions[relevant_actions.ex_date.gt(history_start)].copy()
    action_data = copy(data)
    action_data.daily = raw_history.copy()
    action_data.actions = relevant_actions.copy()
    factors, action_error = price_basis.action_factors(
        action_data, cutoff,
        prices=raw_history.set_index("date")[["close", "preclose"]].copy(),
    )
    fingerprint = _fingerprint(str(data.symbol or ""), cutoff, raw_history, relevant_actions)
    price_basis_result = {
        "method": "raw_to_as_of_corporate_action_factor",
        "as_of": cutoff.date().isoformat(),
        "actions_applied": [_json_value(item) for item in (factors or [])],
        "action_count": len(factors or []),
        "error": action_error,
        "source_prices": "DataBundle.daily raw (unadjusted)",
        "adjusted_columns": ["high", "low", "close", "preclose"],
    }
    if action_error:
        result = _empty_state(cutoff, f"公司行动价格调整不可用：{action_error}", reason_code="action_error")
        result["price_basis"] = price_basis_result
        result["input_fingerprint"] = fingerprint
        result["samples"].update(selection)
        result["samples"].update({
            "daily_rows": int(len(selected)),
            "selected_start": selected.date.iloc[0].date().isoformat(),
            "selected_end": selected.date.iloc[-1].date().isoformat(),
            "prior_close_date": prior_date.date().isoformat() if prior_date is not None else None,
            "prior_close": None,
        })
        return result

    try:
        adjusted_history = price_basis.adjust_frame(
            raw_history, factors or [], timestamp_column="date", adjust_volume=False,
        ).sort_values("date").reset_index(drop=True)
    except Exception as exc:
        result = _empty_state(cutoff, f"公司行动价格调整失败：{exc}", reason_code="action_error")
        result["price_basis"] = price_basis_result
        result["input_fingerprint"] = fingerprint
        return result

    adjusted_selected = adjusted_history[
        adjusted_history.date.isin(selected.date)
    ].sort_values("date").reset_index(drop=True)
    prior_close = None
    prior_high = None
    prior_low = None
    prior_close_source = None
    if prior_date is not None and not prior_row.empty:
        prior_candidates = adjusted_history[adjusted_history.date.eq(prior_date)]
        if not prior_candidates.empty:
            prior_high = _positive(prior_candidates.high.iloc[-1])
            prior_low = _positive(prior_candidates.low.iloc[-1])
            prior_close = _positive(prior_candidates.close.iloc[-1])
            if prior_close is not None:
                prior_close_source = "first_selected_bar" if borrowed_prior else "preceding_daily_row"
    highs = [float(value) for value in adjusted_selected.high]
    lows = [float(value) for value in adjusted_selected.low]
    closes = [float(value) for value in adjusted_selected.close]
    close_series = ([prior_close] if prior_close is not None else []) + closes

    natr14, natr_detail = _natr14(highs, lows, closes, prior_close)
    hv_ratio, hv_detail = _hv_ratio(close_series)
    adx14, plus_di14, minus_di14, adx_detail, plus_detail, minus_detail = _adx14(
        highs, lows, closes, prior_high, prior_low, prior_close,
    )
    amplitude_q75, amplitude_detail = _amplitude_q75(highs, lows, closes, prior_close)
    metrics = {
        "natr14": natr14, "hv_ratio": hv_ratio, "adx14": adx14,
        "plus_di14": plus_di14, "minus_di14": minus_di14,
        "amplitude_q75": amplitude_q75,
    }
    details = {
        "natr14": natr_detail, "hv_ratio": hv_detail, "adx14": adx_detail,
        "plus_di14": plus_detail, "minus_di14": minus_detail,
        "amplitude_q75": amplitude_detail,
    }
    shortage = any(
        item.get("reason_code") in {"insufficient_samples", "missing_preclose"}
        for item in details.values()
    )
    status = "insufficient_history" if shortage else "completed"
    samples = {
        **selection,
        "daily_rows": int(len(selected)),
        "selected_start": selected.date.iloc[0].date().isoformat(),
        "selected_end": selected.date.iloc[-1].date().isoformat(),
        "prior_close_date": prior_date.date().isoformat() if prior_date is not None else None,
        "prior_close": prior_close,
        "prior_high": prior_high,
        "prior_low": prior_low,
        "prior_close_source": prior_close_source,
        "close_samples_hv": int(len(close_series)),
        "return_10_samples": int(min(HV_SHORT_PERIOD, max(0, len(close_series) - 1))),
        "return_60_samples": int(min(HV_LONG_PERIOD, max(0, len(close_series) - 1))),
        "atr_samples": int(min(ATR_PERIOD, len(selected)) if prior_close is not None else 0),
        "adx_samples": int(min(2 * ADX_PERIOD - 1, len(selected)) if prior_close is not None else 0),
        "amplitude_samples": int(min(AMPLITUDE_PERIOD, len(selected))),
        "enabled_calendar_days": int(len(expected)),
    }
    result = {
        "method_version": METHOD_VERSION,
        "as_of": cutoff.date().isoformat(),
        "status": status,
        "metrics": metrics,
        "metric_details": details,
        "samples": samples,
        "warmup": _warmup(),
        "price_basis": price_basis_result,
        "input_fingerprint": fingerprint,
    }
    if shortage:
        result["message"] = "部分指标因连续历史不足不可用；有效指标仍保留"
    return result


def state_step_candidates(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return NATR/HV and amplitude-Q75 step families from a daily state."""
    if not isinstance(state, Mapping):
        return []
    metrics = state.get("metrics") if isinstance(state.get("metrics"), Mapping) else state
    details = state.get("metric_details") if isinstance(state.get("metric_details"), Mapping) else {}
    natr = _finite(metrics.get("natr14"))
    hv = _finite(metrics.get("hv_ratio"))
    adx = _finite(metrics.get("adx14"))
    plus_di = _finite(metrics.get("plus_di14"))
    minus_di = _finite(metrics.get("minus_di14"))
    amplitude = _finite(metrics.get("amplitude_q75"))
    bases: list[tuple[float, str]] = []
    if natr is not None and natr > 0:
        hv_multiplier = .65 + .5 * max(0.0, hv - 1.0) if hv is not None else .65
        base = float(natr * hv_multiplier)
        if math.isfinite(base) and base > 0:
            bases.append((base, "natr_hv"))
    if amplitude is not None and amplitude > 0:
        bases.append((float(amplitude), "amplitude_q75"))

    direction = None
    adx_info = details.get("adx14") if isinstance(details.get("adx14"), Mapping) else {}
    detail_direction = adx_info.get("direction")
    if detail_direction in {"up", "down"}:
        direction = detail_direction
    elif plus_di is not None and minus_di is not None:
        if plus_di > minus_di + 1e-12:
            direction = "up"
        elif minus_di > plus_di + 1e-12:
            direction = "down"
    asymmetric = adx is not None and adx > 25.0 and direction in {"up", "down"}
    alpha = 1.0 + min(1.0, (float(adx) - 22.0) / 28.0 * .85) if asymmetric else 1.0

    result: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()

    def append(buy_step: float, sell_step: float, source: str) -> None:
        if not math.isfinite(buy_step) or not math.isfinite(sell_step):
            return
        if buy_step <= 0 or sell_step <= 0:
            return
        key = (round(buy_step, 12), round(sell_step, 12))
        if key in seen:
            return
        seen.add(key)
        result.append({"buy_step": buy_step, "sell_step": sell_step, "source": source})

    for base, source in bases:
        append(base, base, f"{source}_symmetric")
        if asymmetric:
            if direction == "up":
                append(.85 * base, alpha * base, f"{source}_up_asymmetric")
            else:
                append(alpha * base, .85 * base, f"{source}_down_asymmetric")
    return result


__all__ = ["METHOD_VERSION", "daily_market_state", "state_step_candidates"]
