"""Deterministic stress and synthetic scenarios for research validation.

The functions in this module only construct in-memory specifications and
``DataBundle`` values.  They never publish market data and never inspect a
future market schedule.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .models import DataBundle
from .rules import (
    etf_price_limit_rate,
    instrument_terms,
    price_limit_rate,
    round_limit_price,
    round_to_tick,
)


STRESS_CASES = (
    "base",
    "cost2",
    "volume_half",
    "delay1",
    "max_contribution_omit",
    "combined",
)

SYNTHETIC_CASES = ("down30", "up30", "low_volatility", "gapdown")
SYNTHETIC_DAYS = 63
STRESS_SCENARIOS = STRESS_CASES
SYNTHETIC_SCENARIOS = SYNTHETIC_CASES


def _contributions(account: Any) -> list[Any]:
    if isinstance(account, Mapping):
        values = list(account.get("contributions") or [])
    else:
        values = list(getattr(account, "contributions", []) or [])
    return sorted(values, key=lambda item: (
        item.get("available_at") if isinstance(item, Mapping) else getattr(item, "available_at")
    ))


def largest_contribution_index(account: Any) -> int | None:
    """Return the index of the largest scheduled top-up, if one exists.

    ``QualificationAccount`` sorts contributions by availability date.  Using
    the sorted list makes the index stable across a model round trip and makes
    it agree with the ``omitted_contribution_index`` engine argument.
    """
    rows = _contributions(account)
    if not rows:
        return None
    return max(
        range(len(rows)),
        key=lambda index: (
            float(rows[index].get("amount", 0) if isinstance(rows[index], Mapping)
                  else getattr(rows[index], "amount", 0)),
            -index,
        ),
    )


def stress_cases(account: Any) -> list[dict[str, Any]]:
    """Build the six diagnostic executions for one independent account.

    The maximum top-up omission is meaningful only when the account has at
    least one scheduled contribution.  ``combined`` contains that omission,
    so both cases are marked not applicable for an unfunded account instead
    of silently running a different scenario.
    """
    omitted = largest_contribution_index(account)
    omission_kwargs = (
        {"omitted_contribution_index": omitted}
        if omitted is not None else {}
    )
    cases = [
        {"case": "base", "applicable": True, "kwargs": {}, "reason": None},
        {"case": "cost2", "applicable": True,
         "kwargs": {"cost_multiplier": 2.0}, "reason": None},
        {"case": "volume_half", "applicable": True,
         "kwargs": {"volume_multiplier": 0.5}, "reason": None},
        {"case": "delay1", "applicable": True,
         "kwargs": {"extra_delay_bars": 1}, "reason": None},
        {
            "case": "max_contribution_omit", "applicable": omitted is not None,
            "kwargs": omission_kwargs,
            "reason": None if omitted is not None else "no_scheduled_contributions",
        },
        {
            "case": "combined", "applicable": omitted is not None,
            "kwargs": {
                "cost_multiplier": 2.0, "volume_multiplier": 0.5,
                "extra_delay_bars": 1, **omission_kwargs,
            },
            "reason": None if omitted is not None else "no_scheduled_contributions",
        },
    ]
    return [{**item, "omitted_contribution_index": omitted} for item in cases]


def synthetic_closes(
    case: str, *, days: int = SYNTHETIC_DAYS, start_price: float = 10.0,
    gap_limit_rate: float | None = None,
) -> np.ndarray:
    """Return a deterministic synthetic close path with no market lineage."""
    if case not in SYNTHETIC_CASES:
        raise ValueError(f"未知人造情景：{case}")
    if int(days) != days or days < 1:
        raise ValueError("人造情景日数必须为正整数")
    if not math.isfinite(float(start_price)) or start_price <= 0:
        raise ValueError("人造情景起始价格必须为正数")
    days = int(days)
    start = float(start_price)
    if days == 1:
        endpoint = {"down30": .70, "up30": 1.30}.get(case, 1.0)
        return np.asarray([start * endpoint], dtype=np.float64)
    positions = np.arange(days, dtype=np.float64)
    if case == "down30":
        return start * (1.0 - .30 * positions / (days - 1))
    if case == "up30":
        return start * (1.0 + .30 * positions / (days - 1))
    if case == "low_volatility":
        # Stay inside a fixed +/-1% band around the anchor.  Compounding
        # alternating returns would drift persistently below the lower edge.
        offsets = np.where(np.arange(days) % 2 == 0, .01, -.01)
        offsets[0] = 0.0
        return start * (1.0 + offsets)
    if gap_limit_rate is None:
        raise ValueError("gapdown 必须提供锚日适用的跌停幅度")
    gap_rate = float(gap_limit_rate)
    if not math.isfinite(gap_rate) or gap_rate <= 0 or gap_rate > 1:
        raise ValueError("锚日跌停幅度必须位于 (0, 1]")
    # Use the tighter of the required 10% stress and the anchor day's frozen
    # exchange limit.  A 5%-limit ST stock therefore gaps down at 5%.
    gap_multiplier = 1.0 - min(.10, gap_rate)
    prices = np.full(days, start, dtype=np.float64)
    gap_index = max(1, days // 2)
    prices[gap_index:] = start * gap_multiplier
    return prices


def _scenario_dates(start: date | datetime | pd.Timestamp | None, days: int) -> pd.DatetimeIndex:
    first = pd.Timestamp(start).normalize() if start is not None else pd.Timestamp("2024-01-02")
    while first.weekday() >= 5:
        first += pd.Timedelta(days=1)
    values = pd.bdate_range(first, periods=int(days))
    return pd.DatetimeIndex(values)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _daily_dates(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "date" not in frame:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(frame.date, errors="coerce").dt.normalize()


def _anchor_market_state(
    template: DataBundle, decision_date: Any,
) -> dict[str, Any]:
    """Resolve the causal anchor price and frozen exchange rules."""
    cutoff = pd.Timestamp(decision_date).normalize()
    metadata = template.metadata or {}
    for key, future_invalid in (("listing_date", lambda value: value > cutoff),
                                ("delisting_date", lambda value: value <= cutoff)):
        raw = metadata.get(key)
        if raw in (None, ""):
            continue
        try:
            value = pd.Timestamp(raw).normalize()
        except (TypeError, ValueError):
            continue
        if pd.notna(value) and future_invalid(value):
            raise ValueError(f"锚日 {cutoff.date()} 不在证券 {template.symbol} 的有效上市区间")
    daily = template.daily.copy()
    dates = _daily_dates(daily)
    valid = dates.le(cutoff) & pd.to_numeric(daily.get("close"), errors="coerce").gt(0)
    if "tradestatus" in daily:
        valid &= pd.to_numeric(daily.tradestatus, errors="coerce").fillna(1).ne(0)
    prefix = daily.loc[valid].copy()
    if prefix.empty:
        raise ValueError(f"锚日 {cutoff.date()} 及以前没有 {template.symbol} 的可验证收盘")
    positions = prefix.index
    anchor_index = positions[-1]
    anchor_row = daily.loc[anchor_index]
    price = float(anchor_row.close)
    price_date = pd.Timestamp(anchor_row.date).normalize()
    terms = instrument_terms(template.symbol, metadata)
    tick = float(terms.get("price_tick") or 0)
    if not math.isfinite(tick) or tick <= 0:
        raise ValueError("锚日 instrument terms 缺少有效价格 tick")
    if bool(anchor_row.get("no_price_limit", False)):
        rate = None
    elif terms.get("asset_type") == "etf":
        rate = etf_price_limit_rate(terms, cutoff)
    else:
        rate = price_limit_rate(
            str(terms.get("board") or "main"), cutoff,
            bool(anchor_row.get("is_st", False)),
        )
    return {
        "decision_date": cutoff,
        "price_date": price_date,
        "price": price,
        "tick": tick,
        "limit_rate": None if rate is None else float(rate),
        "is_st": bool(anchor_row.get("is_st", False)),
        "no_price_limit": rate is None,
        "terms": terms,
    }


def synthetic_bundle(
    symbol: str, case: str, *, template: DataBundle | None = None,
    start: date | datetime | pd.Timestamp | str | None = None,
    days: int = SYNTHETIC_DAYS, start_price: float | None = None,
    decision_date: date | datetime | pd.Timestamp | str | None = None,
) -> DataBundle:
    """Create an in-memory 63-day daily ``DataBundle`` for stress replay.

    No bars, actions, or versions from the market store are mutated.  The
    template contributes instrument metadata only; all OHLC values and the
    calendar are synthetic.  A ten-percent move is represented by explicit
    limit fields so the replay cannot accidentally exceed legal bounds.
    """
    if case not in SYNTHETIC_CASES:
        raise ValueError(f"未知人造情景：{case}")
    if decision_date is None:
        decision_date = (
            pd.Timestamp(start).normalize() - pd.Timedelta(days=1)
            if start is not None else pd.Timestamp("2024-01-01")
        )
    cutoff = pd.Timestamp(decision_date).normalize()
    anchor = _anchor_market_state(template, cutoff) if template is not None else None
    if anchor is not None:
        price = float(anchor["price"])
        if start_price is not None and not math.isclose(
            float(start_price), price, rel_tol=1e-9, abs_tol=float(anchor["tick"]) / 2,
        ):
            raise ValueError("人造场景起始价与 decision_date 及以前的可验证收盘不一致")
    else:
        price = float(start_price if start_price is not None else 10.0)
    tick = float(anchor["tick"] if anchor is not None else .01)
    limit_rate = anchor["limit_rate"] if anchor is not None else .10
    price = round_to_tick(price, tick)
    if start is None:
        start = cutoff + pd.Timedelta(days=1)
    start_stamp = pd.Timestamp(start).normalize()
    if start_stamp <= cutoff:
        start_stamp = cutoff + pd.Timedelta(days=1)
    dates = _scenario_dates(start_stamp, days)
    closes = synthetic_closes(
        case, days=days, start_price=price, gap_limit_rate=limit_rate,
    )
    if case == "gapdown":
        gap_index = max(1, days // 2)
        gap_rate = min(.10, float(limit_rate))
        closes[gap_index:] = closes[gap_index - 1] * (1.0 - gap_rate)
    precloses = np.r_[closes[0], closes[:-1]]
    precloses = np.asarray([round_to_tick(float(value), tick) for value in precloses])
    opens = precloses.copy()
    if case == "gapdown":
        gap_index = max(1, days // 2)
        opens[gap_index] = closes[gap_index]
    closes = np.asarray([round_to_tick(float(value), tick) for value in closes])
    highs = np.maximum(opens, closes)
    lows = np.minimum(opens, closes)
    normal = case != "gapdown"
    if normal:
        highs = highs * 1.002
        lows = lows * .998
    else:
        gap_index = max(1, days // 2)
        highs[gap_index:] = opens[gap_index:]
        lows[gap_index:] = opens[gap_index:]
    volume = np.full(days, 100_000.0, dtype=np.float64)
    if limit_rate is None:
        limit_up = np.full(days, np.nan)
        limit_down = np.full(days, np.nan)
        no_price_limit = np.ones(days, dtype=bool)
    else:
        limit_up = np.asarray([
            round_limit_price(float(value) * (1.0 + float(limit_rate)), tick)
            for value in precloses
        ])
        limit_down = np.asarray([
            round_limit_price(float(value) * (1.0 - float(limit_rate)), tick)
            for value in precloses
        ])
        no_price_limit = np.zeros(days, dtype=bool)
    opens = np.asarray([round_to_tick(float(value), tick) for value in opens])
    highs = np.asarray([
        round_to_tick(float(value), tick, side="buy") for value in highs
    ])
    lows = np.asarray([
        round_to_tick(float(value), tick, side="sell") for value in lows
    ])
    for index in range(days):
        if not no_price_limit[index]:
            opens[index] = min(max(opens[index], limit_down[index]), limit_up[index])
            closes[index] = min(max(closes[index], limit_down[index]), limit_up[index])
            highs[index] = min(highs[index], limit_up[index])
            lows[index] = max(lows[index], limit_down[index])
        highs[index] = max(highs[index], opens[index], closes[index])
        lows[index] = min(lows[index], opens[index], closes[index])
    daily = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
        "turnover": closes * volume,
        "preclose": precloses,
        "tradestatus": 1,
        "is_st": bool(anchor["is_st"]) if anchor is not None else False,
        "limit_up": limit_up,
        "limit_down": limit_down,
        "no_price_limit": no_price_limit,
    })
    bars = daily.copy()
    bars["timestamp"] = bars.date + pd.Timedelta(hours=15)
    bars["frequency"] = "1d"
    bars["tradable"] = True
    # The shared engine treats ``is_synthetic`` rows as missing data and drops
    # them.  These are intentional validation inputs, so their provenance is
    # carried by the dedicated column and metadata instead.
    bars["is_synthetic"] = False
    bars["validation_synthetic"] = True
    bars["is_skip"] = False
    bars["template_symbol"] = symbol
    template_metadata = dict(template.metadata) if template is not None else {}
    digest_payload = {
        "symbol": symbol, "case": case, "days": days,
        "start": dates[0].date().isoformat(), "start_price": price,
        "decision_date": cutoff.date().isoformat(), "tick": tick,
        "limit_rate": limit_rate,
    }
    version = "synthetic-validation-v1-" + _fingerprint(digest_payload)[:20]
    metadata = {
        **template_metadata,
        "reference_date": str((dates[0] - pd.Timedelta(days=1)).date()),
        "valuation_price": price,
        "anchor_decision_date": cutoff.date().isoformat(),
        "anchor_price_date": (
            anchor["price_date"].date().isoformat() if anchor is not None
            else None
        ),
        "anchor_price_source": (
            "template_daily_at_or_before_decision_date"
            if anchor is not None else "explicit_synthetic_origin"
        ),
        "price_tick": tick,
        "price_limit_rate": limit_rate,
        "gapdown_limit_rate": (
            min(.10, float(limit_rate))
            if case == "gapdown" and limit_rate is not None else None
        ),
        "template_future_ignored": template is not None,
        "is_synthetic": True,
        "is_demo": True,
        "synthetic_scenario": case,
        "synthetic_validation": True,
        "market_store_written": False,
        "no_real_future_schedule": True,
        "warnings": ["人造压力数据，不进入市场库，也不代表真实未来行情"],
    }
    # Frozen per-day resolution policies must never label synthetic data as a
    # verified market snapshot.
    metadata.pop("data_resolution", None)
    quality = {
        "is_demo": True, "synthetic": True, "minute_ratio": 0.0,
        "days": [{"date": day.date().isoformat(), "mode": "daily"} for day in dates],
    }
    return DataBundle(
        symbol=symbol, bars=bars, daily=daily, actions=pd.DataFrame(),
        calendar=pd.DatetimeIndex(dates), quality=quality, version=version,
        metadata=metadata,
        warnings=["人造压力数据，不进入市场库，也不代表真实未来行情"],
    )


def synthetic_cases(
    symbol: str, *, template: DataBundle | None = None,
    start: date | datetime | pd.Timestamp | str | None = None,
    days: int = SYNTHETIC_DAYS, start_price: float | None = None,
    decision_date: date | datetime | pd.Timestamp | str | None = None,
) -> list[tuple[str, DataBundle]]:
    """Return one bundle for each of the four required synthetic paths."""
    return [
        (case, synthetic_bundle(
            symbol, case, template=template, start=start, days=days,
            start_price=start_price, decision_date=decision_date,
        ))
        for case in SYNTHETIC_CASES
    ]


__all__ = [
    "STRESS_CASES", "SYNTHETIC_CASES", "STRESS_SCENARIOS", "SYNTHETIC_SCENARIOS",
    "SYNTHETIC_DAYS",
    "largest_contribution_index", "stress_cases", "synthetic_closes",
    "synthetic_bundle", "synthetic_cases",
]
