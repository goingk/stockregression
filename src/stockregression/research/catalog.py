"""Public strategy catalogue for the current research protocol."""
from __future__ import annotations

from typing import Any

from ..models import (
    GRANULARITY_DEFAULTS,
    ResearchStrategySpec, research_protocol_description,
)
from .space import ROBUST_FIELDS


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
