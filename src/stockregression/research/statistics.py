"""Deterministic market-shape diagnostics used to seed research searches."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..models import DataBundle


def market_statistical_profile(data: DataBundle, dates: pd.DatetimeIndex) -> dict[str, Any]:
    frame = data.daily.copy()
    if frame.empty or not len(dates):
        return {"status": "unavailable", "reason": "候选生成段没有日线行情", "quantiles": {}}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[frame["date"].between(dates[0], dates[-1])].sort_values("date")
    for name in ("open", "high", "low", "close", "preclose"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    required = {"open", "high", "low", "close"}
    if frame.empty or not required <= set(frame):
        return {"status": "unavailable", "reason": "候选生成段没有有效价格", "quantiles": {}}
    frame = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)].copy()
    if frame.empty:
        return {"status": "unavailable", "reason": "候选生成段没有有效价格", "quantiles": {}}
    if "preclose" not in frame or frame["preclose"].isna().any():
        frame["preclose"] = frame["preclose"].fillna(frame["close"].shift(1)).fillna(frame["open"])
    frame = frame[frame["preclose"] > 0]
    if frame.empty:
        return {"status": "unavailable", "reason": "候选生成段没有有效前收盘", "quantiles": {}}
    open_values = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    preclose = frame["preclose"].to_numpy(dtype=np.float64)
    up = np.maximum(0.0, high - open_values) / open_values
    down = np.maximum(0.0, open_values - low) / open_values
    true_range = (np.maximum(high, preclose) - np.minimum(low, preclose)) / preclose
    x = np.arange(len(close), dtype=np.float64)
    trend = float(np.polyfit(x, np.log(close), 1)[0] * 252.0) if len(close) > 2 else 0.0
    quantiles = {
        "up": [float(np.quantile(up, q)) for q in (.25, .5, .75)],
        "down": [float(np.quantile(down, q)) for q in (.25, .5, .75)],
        "range": [float(np.quantile(true_range, q)) for q in (.25, .5, .75)],
    }
    return {
        "status": "completed", "days": int(len(frame)),
        "start": frame["date"].iloc[0].date().isoformat(),
        "end": frame["date"].iloc[-1].date().isoformat(),
        "trend_strength": trend, "quantile_labels": ["p25", "p50", "p75"],
        "quantiles": quantiles,
    }
