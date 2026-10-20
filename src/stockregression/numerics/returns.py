"""Pure daily-return conversions shared by research and backtesting adapters."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..models import BacktestConfig, DataBundle


def compound_daily(values: np.ndarray) -> float:
    """Compound simple daily returns without hiding invalid observations."""
    values = np.asarray(values, dtype=np.float64)
    growth = 1.0 + values
    if not len(values) or np.any(growth <= 0) or not np.isfinite(growth).all():
        return math.nan
    return float(np.prod(growth) - 1.0)


def buy_hold_returns(base: "BacktestConfig", data: "DataBundle") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Evaluate the independent buy-and-hold account through the share engine."""
    from ..share_engine import research_buy_hold

    result = research_buy_hold(base, data)
    dates = pd.DatetimeIndex(pd.to_datetime(result["dates"])).normalize()
    values = np.asarray(result["daily_returns"], dtype=np.float64)
    if len(dates) != len(values):
        raise ValueError("buy_hold dates do not align with daily_returns")
    return dates, values
