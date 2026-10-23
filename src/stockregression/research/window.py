"""Decision-date training-window selection shared by v5 research and completion.

The training sample is one contiguous block.  Its start is the farthest
decision-date-adjusted day in the configured lookback whose daily low is below,
and whose daily high is above, the close immediately before the evaluation
date.  When no such day exists, the configured maximum lookback is used.  Fixed
mode requires the complete maximum lookback and reports insufficient history
instead of shortening it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import pandas as pd

from ..models import DataBundle
from ..price_basis import action_factors


MIN_TRAINING_OFFSET = 250
MAX_TRAINING_OFFSET = 500


@dataclass(frozen=True)
class TrainingWindow:
    dates: pd.DatetimeIndex
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    anchor_date: pd.Timestamp | None
    anchor_price: float | None
    matched_date: pd.Timestamp | None
    matched: bool
    fallback: bool
    search_start: pd.Timestamp | None
    search_end: pd.Timestamp | None
    match_rule: str
    reason: str
    mode: str = "same_price"
    price_basis: str = "decision_date_adjusted"
    actions_applied: list[dict[str, Any]] | None = None
    data_issue: str | None = None
    min_offset: int = MIN_TRAINING_OFFSET
    max_offset: int = MAX_TRAINING_OFFSET

    @property
    def days(self) -> int:
        return int(len(self.dates))

    @property
    def start_reason(self) -> str:
        if self.mode == "fixed":
            return "fixed_training_max_days"
        return "same_price_farthest" if self.matched else "five_hundred_day_fallback"

    def diagnostic(self) -> dict[str, Any]:
        return {
            "start": self.start.date().isoformat() if self.start is not None else None,
            "end": self.end.date().isoformat() if self.end is not None else None,
            "days": self.days,
            "anchor_date": self.anchor_date.date().isoformat() if self.anchor_date is not None else None,
            "anchor_price": self.anchor_price,
            "matched_date": self.matched_date.date().isoformat() if self.matched_date is not None else None,
            "matched": bool(self.matched),
            "fallback": bool(self.fallback),
            "start_reason": self.start_reason,
            "search_start": self.search_start.date().isoformat() if self.search_start is not None else None,
            "search_end": self.search_end.date().isoformat() if self.search_end is not None else None,
            "min_offset_trading_days": self.min_offset,
            "max_offset_trading_days": self.max_offset,
            "match_rule": self.match_rule,
            "reason": self.reason,
            "mode": self.mode,
            "price_basis": self.price_basis,
            "actions_applied": list(self.actions_applied or []),
            "action_factor_fingerprint": (
                hashlib.sha256(
                    repr([
                        (item.get("ex_date"), item.get("factor"), item.get("source"))
                        for item in self.actions_applied or []
                    ]).encode("utf-8")
                ).hexdigest()
                if self.actions_applied else None
            ),
            "data_issue": self.data_issue,
        }


def _normalise(value: Any) -> pd.Timestamp | None:
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
    """Daily high/low/close keyed by session date, including metadata history."""
    records: dict[pd.Timestamp, dict[str, float]] = {}

    def record(day: pd.Timestamp, field: str, value: Any, *, overwrite: bool = False) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(number) or number <= 0:
            return
        item = records.setdefault(day, {})
        if overwrite or field not in item:
            item[field] = number

    if isinstance(data.daily, pd.DataFrame) and not data.daily.empty:
        frame = data.daily
        if "date" in frame.columns:
            for position, raw_date in enumerate(frame["date"]):
                day = _normalise(raw_date)
                if day is None:
                    continue
                for field in ("close", "high", "low", "preclose"):
                    if field in frame.columns:
                        record(day, field, frame[field].iloc[position])

    # Intraday bars are a fallback for dates whose daily high/low is absent.
    if isinstance(data.bars, pd.DataFrame) and not data.bars.empty:
        frame = data.bars
        if {"timestamp", "close"}.issubset(frame.columns):
            usable = frame.copy()
            usable["date"] = pd.to_datetime(usable["timestamp"], errors="coerce").dt.normalize()
            usable = usable[usable["date"].notna()].sort_values("timestamp")
            for day, group in usable.groupby("date", sort=True):
                stamp = pd.Timestamp(day).normalize()
                record(stamp, "close", group["close"].iloc[-1])
                if "high" in group.columns:
                    high = pd.to_numeric(group["high"], errors="coerce").max()
                    record(stamp, "high", high)
                if "low" in group.columns:
                    low = pd.to_numeric(group["low"], errors="coerce").min()
                    record(stamp, "low", low)

    metadata = data.metadata if isinstance(data.metadata, dict) else {}
    prior = metadata.get("prior_closes")
    if isinstance(prior, dict):
        for raw_date, raw_price in prior.items():
            day = _normalise(raw_date)
            if day is not None:
                record(day, "close", raw_price)
    if not records:
        return pd.DataFrame(columns=["close", "high", "low"])
    frame = pd.DataFrame.from_dict(records, orient="index")
    for column in ("close", "high", "low", "preclose"):
        if column not in frame:
            frame[column] = math.nan
    return frame[["close", "high", "low", "preclose"]].sort_index()


def _adjusted_daily_prices(
    data: DataBundle, anchor: pd.Timestamp, prices: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str | None]:
    """Compatibility wrapper that adjusts only the window's high/low values."""
    factors, data_issue = action_factors(data, anchor, prices)
    adjusted = prices.copy()
    applied: list[dict[str, Any]] = []
    for item in factors:
        ex_date = pd.Timestamp(item["ex_date"]).normalize()
        before = adjusted.index < ex_date
        adjusted.loc[before, ["high", "low"]] = (
            adjusted.loc[before, ["high", "low"]] * float(item["factor"])
        )
        applied.append({**item, "ex_date": ex_date.date().isoformat()})
    return adjusted, applied, data_issue


def training_window(
    data: DataBundle,
    asof: pd.Timestamp | None,
    *,
    min_offset: int = MIN_TRAINING_OFFSET,
    max_offset: int = MAX_TRAINING_OFFSET,
    mode: str = "same_price",
) -> TrainingWindow:
    """Return the new single-block training window ending at ``asof``."""
    min_offset = max(1, int(min_offset))
    max_offset = max(min_offset, int(max_offset))
    empty = TrainingWindow(
        pd.DatetimeIndex([]), None, None, None, None, None, False, True,
        None, None, "daily_low_lt_anchor_close_lt_daily_high", "没有可用的决策日收盘价",
        mode=mode,
    )
    anchor_limit = _normalise(asof)
    if anchor_limit is None:
        return empty
    calendar = pd.DatetimeIndex(pd.to_datetime(data.calendar, errors="coerce")).dropna().normalize().sort_values().unique()
    calendar = pd.DatetimeIndex(calendar[calendar <= anchor_limit])
    if not len(calendar):
        return empty
    prices = _daily_prices(data)
    if prices.empty or prices["close"].dropna().empty:
        return empty

    # Use the last session with a known daily close as the decision anchor.
    available = prices[(prices.index <= anchor_limit) & prices["close"].notna()]
    if available.empty:
        return empty
    anchor = pd.Timestamp(available.index[-1]).normalize()
    anchor_price = float(available.iloc[-1]["close"])
    if mode == "fixed":
        evaluation_index = int(calendar.get_indexer([anchor])[0]) + 1
        start_index = max(0, evaluation_index - max_offset)
        dates = pd.DatetimeIndex(calendar[start_index:evaluation_index])
        full = len(dates) >= max_offset
        issue = None if full else f"固定训练窗口需要 {max_offset} 个交易日，仅有 {len(dates)} 个交易日"
        if not full:
            dates = pd.DatetimeIndex([])
        return TrainingWindow(
            dates=dates, start=pd.Timestamp(dates[0]).normalize() if len(dates) else None,
            end=pd.Timestamp(dates[-1]).normalize() if len(dates) else None,
            anchor_date=anchor, anchor_price=anchor_price, matched_date=None, matched=False,
            fallback=not full, search_start=dates[0] if len(dates) else None,
            search_end=anchor, match_rule="fixed_training_max_days",
            reason=(f"固定训练窗口取评估日前 {max_offset} 个交易日" if full else issue or "固定训练窗口不完整"),
            min_offset=min_offset, max_offset=max_offset, mode="fixed",
            data_issue=issue,
        )
    adjusted, actions_applied, data_issue = _adjusted_daily_prices(data, anchor, prices)
    if data_issue:
        return TrainingWindow(
            pd.DatetimeIndex([]), None, None, anchor, anchor_price, None, False, False,
            None, anchor, "decision_date_adjusted_same_price", data_issue,
            min_offset=min_offset, max_offset=max_offset, mode="same_price",
            actions_applied=actions_applied, data_issue=data_issue,
        )
    positions = pd.DatetimeIndex(calendar).get_indexer([anchor])
    if len(positions) == 0 or positions[0] < 0:
        return empty
    anchor_index = int(positions[0])
    evaluation_index = anchor_index + 1
    earliest_index = max(0, evaluation_index - max_offset)
    latest_index = max(earliest_index, evaluation_index - min_offset)
    latest_index = min(latest_index, anchor_index)
    search_start = pd.Timestamp(calendar[earliest_index]).normalize()
    search_end = pd.Timestamp(calendar[latest_index]).normalize()

    def same_price(day: pd.Timestamp) -> bool:
        if day not in adjusted.index:
            return False
        row = adjusted.loc[day]
        try:
            low, high = float(row["low"]), float(row["high"])
        except (TypeError, ValueError):
            return False
        return math.isfinite(low) and math.isfinite(high) and low < anchor_price < high

    matched_index: int | None = None
    # Ascending order means the farthest matching day is considered first.
    for index in range(earliest_index, latest_index + 1):
        if same_price(pd.Timestamp(calendar[index]).normalize()):
            matched_index = index
            break

    if matched_index is not None:
        start_index = matched_index
        matched = True
        fallback = False
        matched_date = pd.Timestamp(calendar[start_index]).normalize()
        reason = f"找到同价日（当日最低价 < 评估前最后收盘价 < 当日最高价），取 {min_offset} 至 {max_offset} 个交易日内距离评估日最远的一天作为训练时段起点"
    else:
        start_index = max(0, evaluation_index - max_offset)
        matched = False
        fallback = True
        matched_date = None
        reason = f"{min_offset} 至 {max_offset} 个交易日内没有同价日，按评估日前 {max_offset} 个交易日取训练时段"

    dates = pd.DatetimeIndex(calendar[start_index:anchor_index + 1])
    return TrainingWindow(
        dates=dates,
        start=pd.Timestamp(dates[0]).normalize() if len(dates) else None,
        end=pd.Timestamp(dates[-1]).normalize() if len(dates) else None,
        anchor_date=anchor,
        anchor_price=anchor_price,
        matched_date=matched_date,
        matched=matched,
        fallback=fallback,
        search_start=search_start,
        search_end=search_end,
        match_rule="daily_low_lt_anchor_close_lt_daily_high",
        reason=reason,
        min_offset=min_offset,
        max_offset=max_offset,
        mode="same_price",
        actions_applied=actions_applied,
    )


@dataclass(frozen=True)
class TrainingWindowPolicy:
    """Validated lookback policy shared by preflight and production selection."""

    min_offset: int = MIN_TRAINING_OFFSET
    max_offset: int = MAX_TRAINING_OFFSET
    mode: str = "same_price"

    def __post_init__(self) -> None:
        if int(self.min_offset) < 1:
            raise ValueError("训练窗口最短回看日数必须为正整数")
        if int(self.max_offset) < int(self.min_offset):
            raise ValueError("训练窗口最长回看日数不能小于最短回看日数")
        if self.mode not in {"same_price", "fixed"}:
            raise ValueError("训练窗口模式必须为 same_price 或 fixed")

    def window(self, data: DataBundle, asof: pd.Timestamp | None) -> TrainingWindow:
        return training_window(
            data, asof, min_offset=int(self.min_offset), max_offset=int(self.max_offset), mode=self.mode,
        )

    def diagnostic(self) -> dict[str, int]:
        return {"min_offset_trading_days": int(self.min_offset), "max_offset_trading_days": int(self.max_offset), "mode": self.mode}
