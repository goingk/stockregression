"""Protocol-v5 optimization entry point."""
from __future__ import annotations

import math
from typing import Any, Callable

from .models import DataBundle, OptimizationConfig


Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

_RESEARCH_RANGES = {"buy_step", "sell_step", "buy_shares", "sell_shares"}


def _check_ranges(config: OptimizationConfig) -> None:
    if set(config.ranges) != _RESEARCH_RANGES:
        raise ValueError("稳健 J 研究范围必须且只能包含四个搜索轴")
    for name, search_range in config.ranges.items():
        low, high = float(search_range.low), float(search_range.high)
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise ValueError(f"搜索区间 {name} 必须是有限数值且上限不小于下限")
        if search_range.step is not None:
            raise ValueError("稳健 J 研究的步长只能配置在 strategy.granularity 中")
        if name in {"buy_step", "sell_step"} and (low <= 0 or high > 1):
            raise ValueError(f"搜索区间 {name} 必须位于 (0, 1]")
        if name in {"buy_shares", "sell_shares"}:
            if low < 100 or high < 100:
                raise ValueError(f"股数搜索区间 {name} 不能低于 100")
            for value in (low, high):
                if abs(value / 100 - round(value / 100)) > 1e-9:
                    raise ValueError(f"股数搜索区间 {name} 必须以 100 为单位")


def optimize(
    config: OptimizationConfig,
    data: DataBundle,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
    runtime=None,
    budget_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the current v5 robust-J research protocol."""
    _check_ranges(config)
    if config.protocol_version != 5 or config.strategy.strategy_version != 5:
        raise ValueError("旧研究协议已移除；请使用 protocol_version=5 和 strategy_version=5")
    from .research_v5 import optimize_v5

    return optimize_v5(config, data, progress, cancelled, runtime, budget_contract)
