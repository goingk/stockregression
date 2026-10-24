"""Shared corporate-action price factors and frame adjustments.

The market ledger stores prices and volumes in their original share basis.  A
consumer that compares dates across an ex-date must express the earlier values
in the later basis.  This module centralizes that conversion so training-window
selection, statistics and diagnostics cannot silently disagree.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .models import DataBundle
from .corporate_actions import resolve_actions


_PRICE_COLUMNS = ("open", "high", "low", "close", "preclose")


def _normalise_day(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if stamp is pd.NaT:
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _daily_prices(data: DataBundle) -> pd.DataFrame:
    """Return raw daily close/preclose/high/low keyed by session date."""
    frame = data.daily if isinstance(data.daily, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=["close", "high", "low", "preclose"])
    values = frame.copy()
    if "date" in values.columns or "timestamp" in values.columns:
        index = pd.to_datetime(values.pop("date" if "date" in values else "timestamp"), errors="coerce")
    else:
        index = pd.to_datetime(values.index, errors="coerce")
    values.index = index
    values = values[values.index.notna()].copy()
    values.index = values.index.normalize()
    values = values.sort_index()
    values = values[~values.index.duplicated(keep="last")]
    for column in ("close", "high", "low", "preclose"):
        if column not in values.columns:
            values[column] = math.nan
        else:
            values[column] = pd.to_numeric(values[column], errors="coerce")
    return values[["close", "high", "low", "preclose"]]


def _normalise_prices(prices: pd.DataFrame | None, data: DataBundle) -> pd.DataFrame:
    values = _daily_prices(data) if prices is None else prices.copy()
    if values.empty:
        return values
    index = pd.to_datetime(values.index, errors="coerce")
    values.index = index
    values = values[values.index.notna()].copy()
    values.index = values.index.normalize()
    values = values.sort_index()
    values = values[~values.index.duplicated(keep="last")]
    for column in ("close", "preclose"):
        if column not in values.columns:
            values[column] = math.nan
        else:
            values[column] = pd.to_numeric(values[column], errors="coerce")
    return values


def _price_tick(data: DataBundle) -> float:
    """Exchange tick used to verify an ex-rights reference price."""
    try:
        from .rules import instrument_terms

        metadata = data.metadata if isinstance(data.metadata, dict) else {}
        tick = float(instrument_terms(data.symbol, metadata).get("price_tick") or 0.0)
    except Exception:
        tick = 0.0
    return tick if tick > 0 else 0.01


def _explicitly_not_implemented(value: Any) -> bool:
    """Only an explicit false value marks a ledger action as unimplemented."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "n", "否"}
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        return False
    return not bool(value)


def action_factors(
    data: DataBundle,
    anchor: pd.Timestamp,
    prices: pd.DataFrame | None = None,
    *, start=None, allow_reference_only: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return verified ex-date factors effective through ``anchor``.

    Rows on the same ex-date are merged.  The exchange preclose is trusted only
    when it agrees with the cash-and-bonus implied reference; otherwise the
    latter is used and the mismatch remains in the diagnostic.  An explicitly
    unimplemented action or an unsupported action is a hard data issue.

    ``prices`` must have a date index and contain raw ``close`` and
    ``preclose`` columns.  When omitted it is built from ``data.daily``.
    """
    anchor_day = _normalise_day(anchor)
    if anchor_day is None:
        return [], "除权锚点日期无效，无法计算公司行动价格因子"
    raw = _normalise_prices(prices, data)
    actions = resolve_actions(data.actions, raw, _price_tick(data))
    lower = _normalise_day(start) if start is not None else None

    rows = actions.copy()
    for column, default in (
        ("cash_per_share", 0.0),
        ("bonus_ratio", 0.0),
        ("implemented", True),
        ("unsupported_reason", ""),
    ):
        if column not in rows:
            rows[column] = default
    rows["ex_date"] = pd.to_datetime(rows["ex_date"], errors="coerce").dt.normalize()
    invalid_dates = rows["ex_date"].isna()
    if invalid_dates.any() and not allow_reference_only:
        return [], "公司行动缺少有效除权日"
    rows = rows[~invalid_dates & (rows["ex_date"] <= anchor_day)].copy()
    if lower is not None:
        rows = rows[rows.ex_date >= lower]

    applied: list[dict[str, Any]] = []
    tick = _price_tick(data)
    tolerance = tick * 1.1
    for ex_date, same_day in rows.groupby("ex_date", sort=True):
        action_rows = list(same_day.itertuples(index=False))
        if any(
            _explicitly_not_implemented(getattr(row, "implemented", True))
            for row in action_rows
        ):
            if allow_reference_only:
                continue
            return applied, (
                f"选窗遇到未确认已实施的公司行动（{ex_date:%Y-%m-%d}）："
                "除权事实无法核实，训练窗口不可用"
            )
        reasons = [
            str(getattr(row, "unsupported_reason", "") or "").strip()
            for row in action_rows
            if str(getattr(row, "unsupported_reason", "") or "").strip()
        ]
        if reasons and not allow_reference_only:
            return applied, f"选窗遇到未支持的公司行动（{ex_date:%Y-%m-%d}）：{reasons[0]}"

        ex_day = pd.Timestamp(ex_date).normalize()
        if allow_reference_only and any(bool(getattr(row, "date_unresolved", False)) for row in action_rows):
            # The fold date is not an ex-date. The reference-only pass below
            # finds the real jump without consuming this placeholder date.
            continue
        before = raw.index[raw.index < ex_day]
        if not len(before):
            continue
        try:
            prior_close = float(raw.loc[before[-1], "close"])
            reference = float(raw.loc[ex_day, "preclose"]) if ex_day in raw.index else math.nan
            cash = math.fsum(float(getattr(row, "cash_per_share", 0.0) or 0.0) for row in action_rows)
            bonus = math.fsum(float(getattr(row, "bonus_ratio", 0.0) or 0.0) for row in action_rows)
        except (TypeError, ValueError):
            return applied, f"选窗公司行动 {ex_day.date()} 缺少有效分红或送转数据"
        if not math.isfinite(prior_close) or prior_close <= 0:
            return applied, (
                f"选窗公司行动 {ex_day.date()} 的除权因子无效："
                "除权日前一交易日缺少有效收盘价"
            )
        if not math.isfinite(bonus) or not math.isfinite(cash) or cash < 0 or 1.0 + bonus <= 0:
            if allow_reference_only and math.isfinite(reference) and reference > 0:
                applied.append({"ex_date": ex_day, "factor": reference / prior_close,
                                "source": "preclose", "bonus_ratio": 0.0, "action_count": len(action_rows)})
                continue
            return applied, f"选窗公司行动 {ex_day.date()} 的除权因子无效：送转比例不可用"
        implied_reference = (prior_close - cash) / (1.0 + bonus)
        if allow_reference_only and reasons:
            if not math.isfinite(reference) or reference <= 0:
                return applied, f"{ex_day.date()} 缺少有效除权参考价"
            implied_reference = reference
        if not math.isfinite(implied_reference) or implied_reference <= 0:
            return applied, f"选窗公司行动 {ex_day.date()} 的除权因子无效：除权参考价不可用"

        mismatch: dict[str, Any] | None = None
        if math.isfinite(reference) and reference > 0:
            difference = abs(reference - implied_reference)
            if difference <= tolerance:
                factor = reference / prior_close
                source = "verified_preclose"
            else:
                factor = implied_reference / prior_close
                source = "cash_and_bonus"
                mismatch = {
                    "observed_reference": reference,
                    "implied_reference": float(implied_reference),
                    "difference": float(difference),
                    "tolerance": float(tolerance),
                }
        else:
            factor = implied_reference / prior_close
            source = "cash_and_bonus"
        if not math.isfinite(factor) or factor <= 0:
            return applied, f"选窗公司行动 {ex_day.date()} 的除权因子无效"
        applied.append({
            "ex_date": ex_day,
            "factor": float(factor),
            "source": source,
            "cash_per_share": cash,
            "bonus_ratio": bonus,
            "action_count": len(action_rows),
            "preclose": reference if math.isfinite(reference) else None,
            "implied_reference": float(implied_reference),
            "preclose_mismatch": mismatch,
        })
    if allow_reference_only and not raw.empty:
        prior = pd.to_numeric(raw.close, errors="coerce").shift(1)
        known = {item["ex_date"] for item in applied}
        for stamp in raw.index:
            if stamp > anchor_day or (lower is not None and stamp < lower) or stamp in known:
                continue
            previous, reference = prior.loc[stamp], raw.loc[stamp, "preclose"]
            if not math.isfinite(previous) or previous <= 0:
                continue
            if not math.isfinite(reference) or reference <= 0:
                return applied, f"{stamp.date()} 缺少有效前收盘价，无法计算前复权"
            if abs(reference - previous) > tolerance:
                applied.append({"ex_date": stamp, "factor": reference / previous,
                                "source": "preclose", "bonus_ratio": 0.0, "action_count": 0})
        applied.sort(key=lambda item: item["ex_date"])
    return applied, None


def _factor_rows(
    factors: list[dict[str, Any]], days: pd.Series, *, volume: bool = False,
) -> pd.Series:
    multiplier = pd.Series(1.0, index=days.index, dtype="float64")
    for item in factors:
        ex_date = _normalise_day(item.get("ex_date"))
        if ex_date is None:
            continue
        if volume:
            value = 1.0 + float(item.get("bonus_ratio", 0.0) or 0.0)
        else:
            value = float(item.get("factor", 1.0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("公司行动价格因子无效")
        multiplier.loc[days < ex_date] *= value
    return multiplier


def adjust_frame(
    frame: pd.DataFrame,
    factors: list[dict[str, Any]],
    timestamp_column: str = "timestamp",
    adjust_volume: bool = False,
) -> pd.DataFrame:
    """Express rows before every ex-date in the current decision-day basis.

    Prices other than traded notional fields are multiplied by the cumulative
    ex-rights factor.  Adjusted volume uses only ``1 + bonus_ratio``; cash
    dividends do not alter shares.  ``turnover``/``amount`` are deliberately
    left untouched and must not be divided by a cash factor.
    """
    if not factors or frame.empty or timestamp_column not in frame.columns:
        return frame.copy()
    adjusted = frame.copy()
    days = pd.to_datetime(adjusted[timestamp_column], errors="coerce")
    if days.isna().any():
        raise ValueError("行情时间列包含无法解析的日期")
    if getattr(days.dt, "tz", None) is not None:
        days = days.dt.tz_localize(None)
    days = days.dt.normalize()
    price_multiplier = _factor_rows(factors, days)
    for column in _PRICE_COLUMNS:
        if column in adjusted.columns:
            values = pd.to_numeric(adjusted[column], errors="coerce")
            adjusted[column] = values * price_multiplier
    if adjust_volume and "volume" in adjusted.columns:
        volume_multiplier = _factor_rows(factors, days, volume=True)
        adjusted["volume"] = pd.to_numeric(adjusted["volume"], errors="coerce") * volume_multiplier
    return adjusted


def price_factor_at(factors: list[dict[str, Any]], day: pd.Timestamp) -> float:
    """Return the cumulative price factor for all ex-dates after ``day``."""
    stamp = _normalise_day(day)
    if stamp is None:
        raise ValueError("价格基准日期无效")
    factor = 1.0
    for item in factors:
        ex_date = _normalise_day(item.get("ex_date"))
        if ex_date is None:
            continue
        value = float(item.get("factor", 1.0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("公司行动价格因子无效")
        if stamp < ex_date:
            factor *= value
    return float(factor)
