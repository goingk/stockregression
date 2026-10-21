"""Public v5 entry points for research selection and the strategy catalogue.

The v1-v4 protocols, their connected-lattice and hypercube geometries and the
bootstrap statistics they relied on have been removed.  What remains here is a
thin facade over :mod:`stockregression.research.robust`, which owns the only
supported selection implementation.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from ..models import (
    GRANULARITY_DEFAULTS, BacktestConfig, DataBundle, OptimizationConfig,
    ResearchStrategySpec, research_protocol_description,
)
from .space import ROBUST_FIELDS


Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


def select_at(
    config: OptimizationConfig,
    base: BacktestConfig,
    data: DataBundle,
    asof: pd.Timestamp | None,
    *,
    seed_offset: int = 0,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
    progress_start: float = 0.0,
    progress_width: float = 1.0,
    runtime=None,
) -> dict[str, Any]:
    """Run the current four-axis v5 robust-J selection."""
    if config.protocol_version != 5 or config.strategy.strategy_version != 5:
        raise ValueError("旧研究协议已移除；请使用 protocol_version=5 和 strategy_version=5")
    from ..research.robust import select_robust_at

    return select_robust_at(
        config, base, data, asof, seed_offset=seed_offset, progress=progress,
        cancelled=cancelled, progress_start=progress_start,
        progress_width=progress_width, runtime=runtime,
    )


def estimate_search(config: OptimizationConfig, data: DataBundle) -> dict[str, Any]:
    """Return the current v5 preflight estimate."""
    if config.protocol_version != 5 or config.strategy.strategy_version != 5:
        raise ValueError("旧研究协议已移除；请使用 protocol_version=5 和 strategy_version=5")
    from ..research.robust import estimate_robust_search

    return estimate_robust_search(config, data)


def strategy_descriptions() -> dict[str, Any]:
    strategy = ResearchStrategySpec()
    protocol = research_protocol_description(
        strategy, "both", max_principal_drawdown=.2, min_trades=10,
    )
    return {
        "strategies": [{
            "label": "等比网格四维稳健 J 峰值研究",
            "grid_mode": "geometric",
            "fields": list(ROBUST_FIELDS),
            "granularity": dict(GRANULARITY_DEFAULTS),
            "max_shares_mode": "unlimited",
            "neighborhood_fields": list(strategy.neighborhood_fields),
            "search_methods": ["auto", "hierarchical", "exhaustive"],
            "ranking_modes": ["robust_j"],
            "default_ranking_mode": "robust_j",
            **protocol,
            "defaults": {
                "max_principal_drawdown": .2,
                **strategy.model_dump(mode="json"),
            },
        }]
    }
