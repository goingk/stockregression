"""Read-only price adjustment helpers for chart bars."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from ..models import DataBundle
from ..price_basis import action_factors, adjust_frame


def apply_chart_adjustment(service, symbol, timeframe, frame, *, anchor, start=None):
    """Adjust chart prices against a fixed closed-session anchor.

    Weekly/monthly bars are rebuilt from adjusted daily rows so an event inside
    a bucket cannot apply one factor to the entire already-aggregated candle.
    """
    if frame.empty:
        return frame.copy(), None, None
    daily_dataset = service.repository.dataset(symbol, "1day")
    if not daily_dataset:
        return frame.copy(), "复权失败：缺少基础日线数据", None
    # Factor identity is based on the complete local daily history, independent
    # of the visible page/window. The anchor still bounds which events apply.
    daily = service.repository.active_rows(symbol, "1day")
    if daily.empty:
        return frame.copy(), "复权失败：锚点前没有日线价格", None
    prices = daily.copy()
    prices.index = pd.to_datetime(prices.pop("timestamp"), errors="coerce").dt.normalize()
    prices = prices[~prices.index.duplicated(keep="last")]
    action_aux = service.repository.auxiliary(symbol, "actions") or {}
    actions = pd.DataFrame(action_aux.get("payload") or [])
    from ..rules import instrument_terms
    store = getattr(service, "store", None)
    instrument = store.instrument(symbol) if store is not None else None
    instrument = instrument or {}
    bundle = DataBundle(
        symbol=symbol, bars=pd.DataFrame(), daily=daily, actions=actions,
        metadata=instrument_terms(symbol, instrument),
    )
    factors, reason = action_factors(
        bundle, pd.Timestamp(anchor), prices=prices,
        allow_reference_only=True,
    )
    if reason:
        return frame.copy(), f"复权失败：{reason}", None
    version_data = {
        "anchor": pd.Timestamp(anchor).normalize().date().isoformat(),
        "daily_version": daily_dataset.get("current_version"),
        "actions_version": action_aux.get("version"),
        "factors": [
            (str(pd.Timestamp(item["ex_date"]).date()), float(item["factor"]))
            for item in factors
        ],
    }
    version = hashlib.sha256(json.dumps(version_data, sort_keys=True).encode()).hexdigest()[:16]
    if timeframe in {"1week", "1month"}:
        adjusted_daily = adjust_frame(daily, factors)
        adjusted_daily["timestamp"] = pd.to_datetime(adjusted_daily["timestamp"])
        adjusted_daily = adjusted_daily[
            adjusted_daily.timestamp < pd.Timestamp(anchor).normalize() + pd.Timedelta(days=1)
        ].copy()
        defaults = {
            "source_verified": True, "is_synthetic": False,
            "is_auction_placeholder": False, "opening_auction_volume": 0.0,
            "tradestatus": 1, "is_st": False, "adj_factor": 1.0,
        }
        for column, value in defaults.items():
            if column not in adjusted_daily:
                adjusted_daily[column] = value
        from .aggregation import AggregationService
        target = AggregationService._aggregate_frame(
            adjusted_daily, "1day", timeframe,
        )
        result = target
    else:
        result = adjust_frame(frame, factors)
    return result, None, version


def aggregate_daily_chart(service, symbol, timeframe, *, end=None):
    """Build an unadjusted coarse chart directly from the local daily base."""
    daily = service.repository.active_rows(symbol, "1day", end=end)
    if daily.empty:
        return daily
    defaults = {
        "source_verified": True, "is_synthetic": False,
        "is_auction_placeholder": False, "opening_auction_volume": 0.0,
        "tradestatus": 1, "is_st": False, "adj_factor": 1.0,
    }
    for column, value in defaults.items():
        if column not in daily:
            daily[column] = value
    from .aggregation import AggregationService
    return AggregationService._aggregate_frame(daily, "1day", timeframe)
