"""Research orchestration and evidence contracts."""

from .robust import EvaluationSession, estimate_robust_search, select_robust_at
from .window import TrainingWindowPolicy
from .catalog import strategy_descriptions

__all__ = [
    "EvaluationSession", "TrainingWindowPolicy",
    "estimate_robust_search", "select_robust_at", "strategy_descriptions",
]
