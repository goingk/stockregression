"""Batch evaluation adapters for the current v5 training ledger."""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import pandas as pd

from ..models import BacktestConfig, DataBundle, OptimizationConfig
from ..share_engine import research_batch
from .space import ParameterSpace


Cancelled = Callable[[], bool]
Progress = Callable[[float, str], None]


def _chunks(values: list[tuple[int, ...]], size: int) -> Iterable[list[tuple[int, ...]]]:
    for begin in range(0, len(values), size):
        yield values[begin:begin + size]


def _finite(value: Any) -> float:
    number = float(value)
    return number if math.isfinite(number) else math.nan


class BatchEvaluator:
    def __init__(
        self,
        config: OptimizationConfig,
        base: BacktestConfig,
        data: DataBundle,
        space: ParameterSpace,
        train_dates: pd.DatetimeIndex,
        *,
        cancelled: Cancelled | None = None,
        progress: Progress | None = None,
        allow_odd_lot_exit: bool = False,
        chunk_size: int | None = None,
    ):
        self.config = config
        self.base = base.model_copy(update={"grid_mode": "geometric"})
        self.data = data
        self.space = space
        self.train_dates = pd.DatetimeIndex(train_dates)
        self.cancelled = cancelled
        self.progress = progress
        self.allow_odd_lot_exit = bool(allow_odd_lot_exit)
        self.chunk_size = int(chunk_size or 512)
        self.train_base = self.base.model_copy(update={
            "start_date": self.train_dates[0].date(),
            "end_date": self.train_dates[-1].date(),
            "grid_mode": "geometric",
        })
        # The production engine bounds trading by config dates. Keep prior closes
        # and corporate actions so initialization matches a standalone backtest.
        self.train_data = data

    def evaluate_training(self, coordinates: list[tuple[int, ...]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        fold = [(self.train_dates[0].date().isoformat(), self.train_dates[-1].date().isoformat())]
        for chunk_index, chunk in enumerate(_chunks(coordinates, self.chunk_size)):
            if self.cancelled and self.cancelled():
                raise InterruptedError("研究已取消")
            params = [self.space.params(item) for item in chunk]
            result = research_batch(
                self.train_base, self.train_data, params, fold, cancelled=self.cancelled,
                allow_odd_lot_exit=self.allow_odd_lot_exit,
            )
            for index, coordinate in enumerate(chunk):
                evidence = result["fold_metrics"][index][0]
                positions = result.get("position_stats", [{}] * len(chunk))[index]
                metrics = result["metrics"][index]
                grid_return = float(metrics.get("peak_principal_return", math.nan))
                baseline_return = float(metrics.get("matched_funding_hold_return", math.nan))
                excess = grid_return - baseline_return if math.isfinite(grid_return) and math.isfinite(baseline_return) else math.nan
                drawdown = float(metrics.get("max_principal_drawdown", math.inf))
                trade_count = int(metrics.get("trade_count", 0) or 0)
                buy_count = int(metrics.get("buy_count", 0) or 0)
                sell_count = int(metrics.get("sell_count", 0) or 0)
                output.append({
                    "params": self.space.params(coordinate),
                    "key": coordinate,
                    "coordinate": list(coordinate),
                    "evaluation_status": "completed" if math.isfinite(excess) else "failed",
                    "status": "completed" if math.isfinite(excess) else "failed",
                    "training_return": grid_return,
                    "training_baseline_return": baseline_return,
                    "training_baseline_kind": "matched_funding_hold",
                    "training_excess_return": excess,
                    "training_score": excess,
                    "training_principal_drawdown": drawdown,
                    "training_trade_count": trade_count,
                    "training_buy_count": buy_count,
                    "training_sell_count": sell_count,
                    "max_principal_drawdown": drawdown,
                    "trade_count": trade_count,
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    **positions,
                    "training_folds": [evidence],
                })
            if self.progress:
                self.progress((chunk_index + 1) * self.chunk_size / max(1, len(coordinates)), "评估训练参数")
        return output
    def evaluate_training_stress(self, coordinates: list[tuple[int, ...]], cost_multiplier: float = 2.0) -> dict[tuple[int, ...], dict[str, Any]]:
        """Double-cost diagnostic over the single training block."""
        output: dict[tuple[int, ...], dict[str, Any]] = {}
        if not len(self.train_dates):
            return output
        fold = [(self.train_dates[0].date().isoformat(), self.train_dates[-1].date().isoformat())]
        for chunk in _chunks(coordinates, self.chunk_size):
            if self.cancelled and self.cancelled():
                raise InterruptedError("研究已取消")
            params = [self.space.params(item) for item in chunk]
            result = research_batch(
                self.train_base, self.train_data, params, fold,
                cost_multiplier=float(cost_multiplier), cancelled=self.cancelled,
                allow_odd_lot_exit=self.allow_odd_lot_exit,
            )
            for index, coordinate in enumerate(chunk):
                evidence = result["fold_metrics"][index][0]
                positions = result.get("position_stats", [{}] * len(chunk))[index]
                metrics = result["metrics"][index]
                grid_return = float(metrics.get("peak_principal_return", math.nan))
                baseline_return = float(metrics.get("matched_funding_hold_return", math.nan))
                excess = grid_return - baseline_return if math.isfinite(grid_return) and math.isfinite(baseline_return) else math.nan
                drawdown = float(metrics.get("max_principal_drawdown", math.inf))
                output[coordinate] = {
                    "training_return_stress": grid_return,
                    "training_excess_stress": excess,
                    "max_principal_drawdown": drawdown,
                    "trade_count": int(metrics.get("trade_count", 0) or 0),
                    **positions,
                }
        return output
