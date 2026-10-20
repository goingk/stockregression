"""Research orchestration and evidence contracts."""

from .robust import EvaluationSession, estimate_robust_search, select_robust_at
from .statistics import StatisticalEvaluator
from .window import TrainingWindowPolicy

__all__ = [
    "EvaluationSession", "StatisticalEvaluator", "TrainingWindowPolicy",
    "estimate_robust_search", "select_robust_at",
]
