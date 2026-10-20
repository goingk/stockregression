"""Batch evaluation adapters for protocol v4."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..models import BacktestConfig, DataBundle, OptimizationConfig
from ..share_engine import research_batch
from ..scientific import _buy_hold_returns, _compound_daily
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
        validation_dates: pd.DatetimeIndex,
        *,
        validation_folds: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
        cancelled: Cancelled | None = None,
        progress: Progress | None = None,
    ):
        self.config = config
        self.base = base.model_copy(update={"grid_mode": "geometric"})
        self.data = data
        self.space = space
        self.train_dates = pd.DatetimeIndex(train_dates)
        self.validation_dates = pd.DatetimeIndex(validation_dates)
        self.cancelled = cancelled
        self.progress = progress
        self.chunk_size = 512
        self.retain_daily_returns = not (
            config.strategy.strategy_version == 2 and config.strategy.search_method == "hierarchical"
        )
        self.train_base = self.base.model_copy(update={
            "start_date": self.train_dates[0].date(),
            "end_date": self.train_dates[-1].date(),
            "grid_mode": "geometric",
        })
        # The production engine bounds trading by config dates. Keep prior closes
        # and corporate actions so initialization matches a standalone backtest.
        self.train_data = data
        self._train_baseline = self._baseline(self.train_base, self.train_dates)
        self.validation_base = self.base.model_copy(update={
            "start_date": self.validation_dates[0].date(),
            "end_date": self.validation_dates[-1].date(),
            "grid_mode": "geometric",
        })
        self.validation_data = data
        self._validation_baseline = self._baseline(self.validation_base, self.validation_dates)
        self._validation_folds = list(validation_folds) if validation_folds else self._fold_ranges()

    def _baseline(self, base: BacktestConfig, dates: pd.DatetimeIndex) -> float:
        baseline_dates, values = _buy_hold_returns(base, self.data)
        mask = (baseline_dates >= dates[0]) & (baseline_dates <= dates[-1])
        return _compound_daily(values[mask])

    def _fold_ranges(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        dates = self.validation_dates
        count = min(4, len(dates))
        if count == 0:
            return []
        boundaries = np.linspace(0, len(dates), count + 1, dtype=int)
        return [
            (pd.Timestamp(dates[boundaries[index]]), pd.Timestamp(dates[boundaries[index + 1] - 1]))
            for index in range(count)
            if boundaries[index + 1] > boundaries[index]
        ]

    @property
    def validation_folds(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        return list(self._validation_folds)

    def evaluate_training(self, coordinates: list[tuple[int, ...]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        fold = [(self.train_dates[0].date().isoformat(), self.train_dates[-1].date().isoformat())]
        for chunk_index, chunk in enumerate(_chunks(coordinates, self.chunk_size)):
            if self.cancelled and self.cancelled():
                raise InterruptedError("研究已取消")
            params = [self.space.params(item) for item in chunk]
            result = research_batch(self.train_base, self.train_data, params, fold, cancelled=self.cancelled)
            for index, coordinate in enumerate(chunk):
                evidence = result["fold_metrics"][index][0]
                grid_return = float(result["metrics"][index]["total_return"])
                excess = grid_return - self._train_baseline if math.isfinite(grid_return) else math.nan
                output.append({
                    "params": self.space.params(coordinate),
                    "key": coordinate,
                    "coordinate": list(coordinate),
                    "evaluation_status": "completed" if math.isfinite(excess) else "failed",
                    "status": "completed" if math.isfinite(excess) else "failed",
                    "training_return": grid_return,
                    "training_baseline_return": self._train_baseline,
                    "training_excess_return": excess,
                    "training_score": excess,
                    "training_drawdown": float(evidence.get("max_drawdown", math.inf) or math.inf),
                    "training_trade_count": int(evidence.get("trade_count", 0) or 0),
                    "training_folds": [evidence],
                })
            if self.progress:
                self.progress((chunk_index + 1) * self.chunk_size / max(1, len(coordinates)), "评估训练参数")
        return output

    def _aggregate_validation(
        self,
        coordinate: tuple[int, ...],
        folds: list[dict[str, Any]],
        cost_key: str,
        metrics: dict[str, Any],
        daily_dates: pd.DatetimeIndex | None = None,
        daily_values: np.ndarray | None = None,
    ) -> dict[str, Any]:
        # Folds only diagnose risk; the whole validation interval is one fresh
        # account, using the same peak-invested ROI as the production backtest.
        grid_return = float(metrics["total_return"])
        excess = grid_return - self._validation_baseline if math.isfinite(grid_return) else math.nan
        drawdown = max((float(row.get("max_drawdown", 0.0) or 0.0) for row in folds), default=math.inf)
        trade_count = sum(int(row.get("trade_count", 0) or 0) for row in folds)
        buy_count = sum(int(row.get("buy_count", 0) or 0) for row in folds)
        sell_count = sum(int(row.get("sell_count", 0) or 0) for row in folds)
        row = {
            "params": self.space.params(coordinate),
            "key": coordinate,
            "coordinate": list(coordinate),
            "evaluation_status": "completed" if math.isfinite(excess) else "failed",
            "folds": deepcopy(folds),
            "grid_return": grid_return,
            "baseline_return": self._validation_baseline,
            "validation_return_original": grid_return,
            "validation_excess_original": excess,
            "raw_score": excess,
            "max_drawdown": drawdown,
            "trade_count": trade_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "constraint_reasons": [],
            "complete": True,
        }
        if cost_key == "stress":
            row["validation_return_stress"] = grid_return
            row["validation_excess_stress"] = excess
        if daily_dates is not None and daily_values is not None:
            row["_current_validation_dates"] = daily_dates
            row["_current_validation_returns"] = daily_values
        return row

    def evaluate_validation(self, coordinates: list[tuple[int, ...]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        fold_ranges = [(start.date().isoformat(), end.date().isoformat()) for start, end in self._validation_folds]
        if not fold_ranges:
            return [{
                "params": self.space.params(item), "key": item, "coordinate": list(item),
                "evaluation_status": "insufficient_history", "complete": False,
                "validation_excess_original": math.nan, "raw_score": math.nan,
                "constraint_reasons": ["没有分离的验证窗口"],
            } for item in coordinates]
        for chunk_index, chunk in enumerate(_chunks(coordinates, self.chunk_size)):
            if self.cancelled and self.cancelled():
                raise InterruptedError("研究已取消")
            params = [self.space.params(item) for item in chunk]
            result = research_batch(self.validation_base, self.validation_data, params, fold_ranges, cancelled=self.cancelled)
            dates = pd.DatetimeIndex(result["dates"])
            mask = (dates >= self.validation_dates[0]) & (dates <= self.validation_dates[-1])
            for index, coordinate in enumerate(chunk):
                row = self._aggregate_validation(
                    coordinate,
                    result["fold_metrics"][index],
                    "original",
                    result["metrics"][index],
                    dates[mask] if self.retain_daily_returns else None,
                    np.asarray(result["daily_returns"][mask, index], dtype=np.float64) if self.retain_daily_returns else None,
                )
                output.append(row)
            if self.progress:
                self.progress((chunk_index + 1) * self.chunk_size / max(1, len(coordinates)), "验证冻结参数域")
        return output

    def evaluate_stress(self, coordinates: list[tuple[int, ...]]) -> dict[tuple[int, ...], dict[str, Any]]:
        output: dict[tuple[int, ...], dict[str, Any]] = {}
        fold_ranges = [(start.date().isoformat(), end.date().isoformat()) for start, end in self._validation_folds]
        if not fold_ranges:
            return output
        for chunk_index, chunk in enumerate(_chunks(coordinates, self.chunk_size)):
            if self.cancelled and self.cancelled():
                raise InterruptedError("研究已取消")
            params = [self.space.params(item) for item in chunk]
            result = research_batch(
                self.validation_base,
                self.validation_data,
                params,
                fold_ranges,
                cost_multiplier=2.0,
                cancelled=self.cancelled,
            )
            for index, coordinate in enumerate(chunk):
                output[coordinate] = self._aggregate_validation(
                    coordinate, result["fold_metrics"][index], "stress", result["metrics"][index],
                )
            if self.progress:
                self.progress((chunk_index + 1) * self.chunk_size / max(1, len(coordinates)), "评估双倍成本压力")
        return output


def center_constraints(row: dict[str, Any], max_drawdown: float, min_trades: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(row.get("max_drawdown", math.inf)) > float(max_drawdown):
        reasons.append("验证折最大回撤超过上限")
    if int(row.get("trade_count", 0) or 0) < int(min_trades):
        reasons.append("验证折成交次数不足")
    if int(row.get("buy_count", 0) or 0) <= 0 or int(row.get("sell_count", 0) or 0) <= 0:
        reasons.append("验证折未同时出现买入和卖出")
    return not reasons, reasons


def positive(row: dict[str, Any] | None, tolerance: float = 1e-12) -> bool:
    if not row or row.get("evaluation_status") != "completed":
        return False
    value = row.get("validation_excess_original")
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > tolerance
