"""Four-axis robust-J research: objective, budgeted search and peak evidence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..models import BacktestConfig, DataBundle, OptimizationConfig, SearchRange
from ..numerics.robust import (
    NeighborhoodOperator,
    PeakAnalysis,
    PeakAnalyzer,
    RobustScore,
    completed,
    eligible,
    position_penalty,
    robust_score,
    tensor_robust_scores,
)
from ..performance import (
    current_process_rss_bytes, current_workspace_bytes, record_count, record_seconds,
)
from .evaluator import BatchEvaluator
from .search import coarse_coordinates
from .space import ROBUST_FIELDS, ParameterSpace
from .window import TrainingWindow, TrainingWindowPolicy
from .statistics import market_statistical_profile

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


@dataclass
class EvaluationBudget:
    """Task-scoped numeric budget shared by selection and replay phases."""

    total: int
    max_runtime_seconds: int
    started: float = field(default_factory=time.monotonic)
    consumed: int = 0
    stopped: bool = False
    reason: str | None = None
    workspace_bytes: int = field(default_factory=current_workspace_bytes)

    @property
    def remaining(self) -> int:
        return max(0, int(self.total) - int(self.consumed))

    @property
    def timed_out(self) -> bool:
        return time.monotonic() - self.started >= max(1, int(self.max_runtime_seconds))

    def stop(self, reason: str) -> None:
        self.stopped = True
        self.reason = reason

    def reserve(self, count: int, purpose: str) -> bool:
        count = max(0, int(count))
        if self.stopped:
            return False
        if self.timed_out:
            self.stop("任务达到时间上限")
            return False
        if count > self.remaining:
            self.stop(f"任务总评价上限不足以启动{purpose}批次")
            return False
        resident = current_process_rss_bytes()
        if resident and self.workspace_bytes and resident > int(self.workspace_bytes):
            self.stop(
                f"任务内存占用 {resident / 1048576:.0f} MB 超过工作区上限 "
                f"{int(self.workspace_bytes) / 1048576:.0f} MB"
            )
            return False
        self.consumed += count
        return True


def _emit(progress: Progress | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, float(fraction))), message)


def _stable_seed(config: OptimizationConfig, asof: str, purpose: str) -> int:
    strategy = config.strategy.model_dump(mode="json")
    # Sampling must remain stable when the chosen window or an operational
    # budget changes.  Those settings alter how much of the same deterministic
    # sequence is consumed, rather than which candidate sequence is drawn.
    for name in (
        "training_window_mode", "training_budget", "validation_budget",
        "reporting_budget", "total_evaluation_budget", "max_runtime_seconds",
        "stress_cost_multiplier", "stress_region_count",
    ):
        strategy.pop(name, None)
    payload = {
        "seed": int(config.seed), "asof": asof, "purpose": purpose,
        "strategy": strategy,
        "ranges": {name: value.model_dump(mode="json") for name, value in config.ranges.items() if name in ROBUST_FIELDS},
        "initial_shares": int(config.base.initial_shares),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _dedupe(coordinates: Iterable[tuple[int, ...]], limit: int) -> list[tuple[int, ...]]:
    output: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for coordinate in coordinates:
        if coordinate in seen:
            continue
        seen.add(coordinate)
        output.append(coordinate)
        if len(output) >= limit:
            break
    return output


def _sobol_coordinates(space: ParameterSpace, limit: int, seed: int) -> list[tuple[int, ...]]:
    if limit <= 0 or not space.active:
        return [space.full_coordinate(())] if limit > 0 else []
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=len(space.active), scramble=True, seed=int(seed))
        batches = max(1, int(math.ceil(math.log2(max(2, limit * 3)))))
        samples = sampler.random_base2(m=batches)
    except Exception:
        samples = np.random.default_rng(seed).random((limit * 3, len(space.active)))
    output: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw in samples:
        values = []
        for value, name in zip(raw, space.active):
            dim = space.dimensions[name]
            tick = dim.low + min(dim.width, int(float(value) * max(1, dim.count)))
            values.append(tick)
        if not space.legal(values):
            continue
        coordinate = space.full_coordinate(values)
        if coordinate in seen:
            continue
        seen.add(coordinate)
        output.append(coordinate)
        if len(output) >= limit:
            break
    return output


def _snap_value(space: ParameterSpace, name: str, target: float) -> int:
    dim = space.dimensions[name]
    unit = float(dim.unit)
    tick = int(round(float(target) / unit))
    return min(dim.high, max(dim.low, tick))


def _statistical_coordinates(space: ParameterSpace, profile: dict[str, Any]) -> list[tuple[int, ...]]:
    if profile.get("status") != "completed":
        return []
    quantiles = profile.get("quantiles") or {}
    output: list[tuple[int, ...]] = []
    for index in range(3):
        values: list[int] = []
        for name in space.active:
            dim = space.dimensions[name]
            if name == "buy_step":
                target = float((quantiles.get("down") or [0])[index])
                tick = _snap_value(space, name, target)
            elif name == "sell_step":
                target = float((quantiles.get("up") or [0])[index])
                tick = _snap_value(space, name, target)
            elif name in {"buy_shares", "sell_shares"}:
                tick = (dim.low + dim.high) // 2
            else:
                tick = (dim.low + dim.high) // 2
            values.append(tick)
        if space.legal(values):
            coordinate = space.full_coordinate(values)
            if coordinate not in output:
                output.append(coordinate)
    return output


def _initial_coordinates(space: ParameterSpace, config: OptimizationConfig, asof: str, profile: dict[str, Any]) -> list[tuple[int, ...]]:
    budget = int(config.strategy.training_budget)
    total = space.size()
    if total <= budget:
        return list(space.iterate(budget))
    seeds = [_stable_seed(config, asof, "robust_sample")]
    anchors: list[tuple[int, ...]] = []
    if space.active:
        low = [space.dimensions[name].low for name in space.active]
        high = [space.dimensions[name].high for name in space.active]
        for values in (low, high):
            if space.legal(values):
                anchors.append(space.full_coordinate(values))
        middle = [(space.dimensions[name].low + space.dimensions[name].high) // 2 for name in space.active]
        if space.legal(middle):
            anchors.append(space.full_coordinate(middle))
    coarse_limit = max(1, budget // 4)
    statistical = _statistical_coordinates(space, profile)
    sobol = _sobol_coordinates(space, budget, seeds[0])
    return _dedupe([*anchors, *statistical, *coarse_coordinates(space, coarse_limit), *sobol], budget)


class EvaluationSession:
    def __init__(self, evaluator: BatchEvaluator, space: ParameterSpace, config: OptimizationConfig,
                 cancelled: Cancelled | None, budget: EvaluationBudget | None = None):
        self.evaluator = evaluator
        self.space = space
        self.config = config
        self.cancelled = cancelled
        self.task_budget = budget
        self.rows: dict[tuple[int, ...], dict[str, Any]] = {}
        self.attempts = 0
        self.requested = 0
        self.failures = 0
        # Points that were requested but never admitted: truncated batches,
        # exhausted budgets and timeouts.  They are neither cache hits nor
        # charged evaluations.
        self.skipped = 0
        self.spent = {"training": 0, "validation": 0, "reporting": 0}
        self.stress_spent = 0
        self.budget_stopped = False
        self.budget_reason: str | None = None
        self.reporting_stopped = False
        self.reporting_reason: str | None = None
        self.started = time.monotonic()
        self.scores: dict[tuple[int, ...], RobustScore | None] = {}

    @property
    def total_cap(self) -> int:
        if self.task_budget is not None:
            return int(self.task_budget.total)
        configured = self.config.strategy.total_evaluation_budget
        if configured is not None:
            return int(configured)
        strategy = self.config.strategy
        return int(
            strategy.training_budget + strategy.validation_budget + strategy.reporting_budget
            + strategy.stress_region_count * max(1, strategy.hypercube_edge ** 4)
        )

    def _budget_remaining(self) -> int:
        if self.task_budget is not None:
            return self.task_budget.remaining
        return max(0, self.total_cap - sum(self.spent.values()) - self.stress_spent)

    @property
    def ledger_evaluations(self) -> int:
        # Every numeric candidate evaluates the strategy and its matched-funding
        # hold account as one paired ledger evaluation.  The count is this
        # session's own work: one task budget is shared by the fixed, current
        # and adaptive decisions, so the task-wide total must never be reported
        # as a single decision's ledger count.
        return 2 * (sum(self.spent.values()) + self.stress_spent)

    @property
    def cache_hits(self) -> int:
        """Requested points that were already evaluated and cost nothing."""
        return max(0, int(self.requested) - int(self.attempts) - int(self.skipped))

    def _stop_for_budget(self, reason: str) -> None:
        """Stop the session and the task budget together.

        The task budget must never keep spending once the session stopped, so
        the reason text no longer decides whether the task budget is stopped.
        """
        self.budget_stopped = True
        self.budget_reason = reason
        if self.task_budget is not None:
            self.task_budget.stop(reason)

    def _stop_for_phase_cap(self, phase: str) -> None:
        reason = f"{phase} 阶段达到评价预算上限"
        if phase == "reporting":
            self.reporting_stopped = True
            self.reporting_reason = "展示补算达到 reporting_budget"
        else:
            # A phase cap truncates only this phase.  The task budget keeps the
            # remaining phases alive, so it must not be stopped here.
            self.budget_stopped = True
            self.budget_reason = reason

    def ensure(self, coordinates: Iterable[tuple[int, ...]], phase: str, cap: int) -> int:
        requested = list(dict.fromkeys(coordinates))
        self.requested += len(requested)
        missing = [coordinate for coordinate in requested if coordinate not in self.rows]
        original_missing = len(missing)
        timed_out = (
            self.task_budget.timed_out if self.task_budget is not None
            else time.monotonic() - self.started >= int(self.config.strategy.max_runtime_seconds)
        )
        if timed_out:
            self.skipped += original_missing
            self._stop_for_budget("任务达到时间上限")
            return 0
        phase_available = max(0, int(cap) - int(self.spent[phase]))
        task_available = self._budget_remaining()
        allowed = min(phase_available, task_available)
        task_limited = task_available <= phase_available and original_missing > task_available
        missing = missing[:allowed]
        truncated = original_missing - len(missing)
        if truncated > 0:
            # Requested but not admitted: skipped, never a cache hit.
            self.skipped += truncated
        if not missing:
            # Cache hits are free and must not be reported as an exhausted
            # phase budget.  Only an actually requested, uncached point that
            # cannot be admitted stops the phase.
            if original_missing > 0 and allowed <= 0:
                if task_available <= phase_available:
                    reason = "任务总评价上限已用尽"
                    if self.task_budget is not None:
                        self.task_budget.stop(reason)
                    self._stop_for_budget(reason)
                else:
                    self._stop_for_phase_cap(phase)
            return 0
        if len(missing) < original_missing:
            if not task_limited:
                self._stop_for_phase_cap(phase)
        if self.cancelled and self.cancelled():
            raise InterruptedError("研究已取消")
        attempted = 0
        for begin in range(0, len(missing), self.evaluator.chunk_size):
            timed_out = (
                self.task_budget.timed_out if self.task_budget is not None
                else time.monotonic() - self.started >= int(self.config.strategy.max_runtime_seconds)
            )
            if timed_out:
                self._stop_for_budget("任务达到时间上限")
                break
            chunk = missing[begin:begin + self.evaluator.chunk_size]
            if self.task_budget is not None and not self.task_budget.reserve(len(chunk), phase):
                self._stop_for_budget(self.task_budget.reason or f"{phase} 阶段达到任务总评价上限")
                break
            # Numerical work is reserved before the native batch starts.  A
            # failed or cancelled batch stays charged, while work that never
            # started remains available for a checkpointed retry.
            self.spent[phase] += len(chunk)
            self.attempts += len(chunk)
            attempted += len(chunk)
            try:
                evaluated = self.evaluator.evaluate_training(chunk)
                self.rows.update({tuple(row["key"]): row for row in evaluated})
            except InterruptedError:
                raise
            except Exception as exc:
                # Preserve failure evidence and continue with the remaining
                # budget rather than silently dropping legal lattice points.
                self.failures += len(chunk)
                for coordinate in chunk:
                    self.rows[coordinate] = {
                        "params": self.space.params(coordinate), "key": coordinate,
                        "coordinate": list(coordinate), "evaluation_status": "failed",
                        "training_excess_return": math.nan, "max_principal_drawdown": math.inf,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
        admitted_but_not_run = len(missing) - attempted
        if admitted_but_not_run > 0:
            self.skipped += admitted_but_not_run
        if task_limited:
            reason = f"任务总评价上限不足以启动{phase}阶段完整批次"
            if self.task_budget is not None:
                self.task_budget.stop(reason)
            self._stop_for_budget(reason)
        if phase == "reporting" and self.spent["reporting"] >= int(cap):
            self.reporting_stopped = True
            self.reporting_reason = "展示补算达到 reporting_budget"
        return len(missing)

    def score(self, center: tuple[int, ...], phase: str = "validation", cap: int | None = None) -> RobustScore:
        operator = NeighborhoodOperator(self.space, int(self.config.strategy.hypercube_edge) // 2)
        points = operator.points(center)
        missing = [point for point in points if point not in self.rows]
        if missing:
            remaining = int(self.config.strategy.validation_budget) if cap is None else int(cap)
            self.ensure(missing, phase, remaining)
        score = robust_score(center, self.rows, self.space, self.config.strategy)
        self.scores[center] = score
        return score


def _score_sort(score: RobustScore, coordinate: tuple[int, ...]) -> tuple:
    return (
        -float(score.j_score), -float(score.neighborhood_mean),
        float(score.downside_risk), float(score.position_penalty),
        -float(score.center_return), coordinate,
    )


def _complete_reporting_scores(session: EvaluationSession) -> None:
    """Complete a reporting target set frozen before reporting starts.

    Newly evaluated neighbours may change the ranking, but they never expand
    the target list or receive an automatic follow-up neighbourhood.  Failed
    points remain missing and are reported as incomplete.
    """
    target = int(session.config.strategy.recommendation_count)
    frozen = sorted(
        (coordinate for coordinate, row in session.rows.items() if completed(row)),
        key=lambda coordinate: (-float(session.rows[coordinate]["training_excess_return"]), coordinate),
    )
    if not frozen:
        return
    # The return leader is always the first frozen target; later targets are
    # only the candidates visible at the start of this phase.
    reporting_cap = int(session.config.strategy.reporting_budget)
    for coordinate in frozen[: max(1, target)]:
        if session.budget_stopped or session.reporting_stopped or session.spent["reporting"] >= reporting_cap:
            if session.spent["reporting"] >= reporting_cap:
                session.reporting_stopped = True
                session.reporting_reason = "展示补算达到 reporting_budget"
            break
        session.score(coordinate, phase="reporting", cap=reporting_cap)
        for point in NeighborhoodOperator(session.space, int(session.config.strategy.hypercube_edge) // 2).points(coordinate):
            if session.budget_stopped or session.reporting_stopped or session.spent["reporting"] >= reporting_cap:
                if session.spent["reporting"] >= reporting_cap:
                    session.reporting_stopped = True
                    session.reporting_reason = "展示补算达到 reporting_budget"
                break
            score = session.scores.get(point)
            if score is None or not score.complete:
                session.scores[point] = robust_score(point, session.rows, session.space, session.config.strategy)
def _normalized_distance(space: ParameterSpace, left: tuple[int, ...], right: tuple[int, ...]) -> float:
    total = 0.0
    for name in space.active:
        index = space._field_index[name]
        width = max(1, space.dimensions[name].width)
        total += ((left[index] - right[index]) / width) ** 2
    return math.sqrt(total)


def _start_coordinates(session: EvaluationSession, space: ParameterSpace, target: int) -> list[tuple[int, ...]]:
    rows = [row for coordinate, row in session.rows.items() if completed(row)]
    if not rows:
        return []
    raw = sorted(rows, key=lambda row: (-float(row["training_excess_return"]), tuple(row["key"])))
    adjusted = []
    for row in rows:
        penalty = position_penalty(row, session.config.strategy)[0]
        adjusted.append((float(row["training_excess_return"]) - penalty, row))
    adjusted.sort(key=lambda item: (-item[0], tuple(item[1]["key"])))
    coverage_rows = rows
    selected: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for bucket in (raw, [item[1] for item in adjusted]):
        for row in bucket:
            coordinate = tuple(row["key"])
            if coordinate not in seen:
                selected.append(coordinate)
                seen.add(coordinate)
            if len(selected) >= max(1, target // 3):
                break
    while len(selected) < target and coverage_rows:
        remaining = [row for row in coverage_rows if tuple(row["key"]) not in seen]
        if not remaining:
            break
        if not selected:
            row = raw[0]
        else:
            row = max(
                remaining,
                key=lambda item: (_normalized_distance(space, tuple(item["key"]), selected[-1]), -float(item["training_excess_return"])),
            )
        coordinate = tuple(row["key"])
        selected.append(coordinate)
        seen.add(coordinate)
    return selected[:target]


def _strides(space: ParameterSpace, strategy, axis: str | None = None) -> list[int]:
    maximum = (
        space.dimensions[axis].width if axis is not None
        else max((space.dimensions[name].width for name in space.active), default=0)
    )
    cap = max(1, int(math.floor(max(0, maximum) * float(strategy.max_stride_fraction))))
    values = set(int(value) for value in strategy.initial_strides)
    current = max(values) if values else 1
    while current < cap:
        current *= int(strategy.stride_multiplier)
        values.add(min(cap, current))
    return sorted(value for value in values if value >= 1)


def _shift(space: ParameterSpace, coordinate: tuple[int, ...], axis: str, delta: int) -> tuple[int, ...] | None:
    if axis not in space.active:
        return None
    index = space._field_index[axis]
    dim = space.dimensions[axis]
    value = coordinate[index] + int(delta)
    if value < dim.low or value > dim.high:
        return None
    values = dict(zip(space.active, space.active_coords(coordinate)))
    values[axis] = value
    active_values = [values[name] for name in space.active]
    return coordinate if not space.legal(active_values) else space.full_coordinate(active_values)


def _local_search(session: EvaluationSession, space: ParameterSpace, starts: list[tuple[int, ...]], search_cap: int) -> list[tuple[int, ...]]:
    if not starts or search_cap <= 0:
        return []
    per_start = max(1, search_cap // max(1, len(starts)))
    starts_before = session.spent["validation"]
    visited: set[tuple[int, ...]] = set()
    verified: list[tuple[int, ...]] = []
    fine_strides = sorted({int(value) for value in session.config.strategy.initial_strides if int(value) >= 1})
    strides_by_axis = {
        axis: _strides(space, session.config.strategy, axis) for axis in space.active
    }
    for start in starts:
        if session.cancelled and session.cancelled():
            raise InterruptedError("研究已取消")
        if session.spent["validation"] >= search_cap:
            break
        local_cap = min(search_cap, session.spent["validation"] + per_start)
        current = start
        current_score = session.score(current, cap=local_cap)
        visited.add(current)
        for _ in range(24):
            if not current_score.complete:
                break
            best_center = current
            best_score = current_score

            def probe(stride_values: list[int], *, coarse: bool = False) -> tuple[tuple[int, ...], RobustScore]:
                nonlocal best_center, best_score
                for axis in space.active:
                    axis_strides = (
                        [value for value in strides_by_axis[axis] if value > max(fine_strides, default=0)]
                        if coarse else stride_values
                    )
                    for stride in axis_strides:
                        for delta in (-stride, stride):
                            if session.spent["validation"] >= local_cap:
                                return best_center, best_score
                            shifted = _shift(space, current, axis, delta)
                            if shifted is None or shifted == current or shifted in visited:
                                continue
                            visited.add(shifted)
                            score = session.score(shifted, cap=local_cap)
                            if not score.complete or not math.isfinite(score.j_score):
                                continue
                            if score.j_score > best_score.j_score + float(session.config.strategy.peak_tolerance):
                                best_center, best_score = shifted, score
                return best_center, best_score

            probe(fine_strides or [1])
            if best_center == current and any(
                any(value > max(fine_strides, default=0) for value in strides_by_axis[axis])
                for axis in space.active
            ):
                # Only spend validation evidence on larger jumps after the
                # local one- and two-tick refinement has stalled.
                probe([], coarse=True)
            verified.append(current)
            if best_center == current:
                break
            current, current_score = best_center, best_score
        if current_score.complete:
            verified.append(current)
    return _dedupe(verified, len(verified))


def _score_of(session: EvaluationSession, center: tuple[int, ...]):
    return session.scores.get(center)


def _eligible_of(session: EvaluationSession, center: tuple[int, ...]) -> bool:
    score = session.scores.get(center)
    return eligible(
        session.rows.get(center), score,
        max_principal_drawdown=session.config.max_principal_drawdown,
        min_trades=session.config.min_trades, strategy=session.config.strategy,
    )


def _verify_peak(
    session: EvaluationSession,
    analyzer: PeakAnalyzer,
    center: tuple[int, ...],
    verify_cap: int,
) -> PeakAnalysis:
    current = center
    visited: set[tuple[int, ...]] = set()
    analysis = PeakAnalysis("incomplete", center, missing=(center,))
    for _ in range(64):
        if current in visited:
            return PeakAnalysis("slope", center, higher=current)
        visited.add(current)
        if current not in session.scores or session.scores[current] is None or not session.scores[current].complete:
            session.score(current, cap=verify_cap)
        analysis = analyzer.inspect(current, lambda point: _score_of(session, point), lambda point: _eligible_of(session, point))
        if analysis.state == "peak":
            return analysis
        if analysis.state == "slope":
            higher = analysis.higher
            if higher is None or not _eligible_of(session, higher):
                return analysis
            if session.spent["validation"] >= verify_cap:
                return PeakAnalysis("incomplete", center, higher=higher, complete=False)
            current = higher
            continue
        missing = [point for point in analysis.missing if point not in session.scores or session.scores.get(point) is None or not session.scores[point].complete]
        if not missing or session.spent["validation"] >= verify_cap:
            return analysis
        for point in missing:
            if session.spent["validation"] >= verify_cap:
                break
            session.score(point, cap=verify_cap)
    return PeakAnalysis("incomplete", center, missing=tuple(sorted(analysis.missing)))


def _public_row(row: dict[str, Any], score: RobustScore | None = None) -> dict[str, Any]:
    payload = deepcopy({key: value for key, value in row.items() if not key.startswith("_")})
    if score is not None:
        payload.update(score.as_dict())
    payload["j_score"] = float(score.j_score) if score is not None and math.isfinite(score.j_score) else None
    payload["center_excess_return"] = float(score.center_return) if score is not None and math.isfinite(score.center_return) else payload.get("training_excess_return")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    initial = float(params.get("initial_shares") or 0)
    if initial > 0:
        payload["q_buy"] = float(params.get("buy_shares", 0.0)) / initial
        payload["q_sell"] = float(params.get("sell_shares", 0.0)) / initial
    return payload


def _region(space: ParameterSpace, center: tuple[int, ...], row: dict[str, Any], score: RobustScore, analysis: PeakAnalysis, session: EvaluationSession, region_id: int) -> dict[str, Any]:
    operator = NeighborhoodOperator(space, int(session.config.strategy.hypercube_edge) // 2)
    points = operator.points(center)
    bounds = {}
    for name in space.active:
        index = space._field_index[name]
        axis = [point[index] for point in points]
        bounds[name] = {
            "low": space.value(name, min(axis)), "high": space.value(name, max(axis)),
            "low_ticks": min(axis), "high_ticks": max(axis),
        }
    values = [float(session.rows[point]["training_excess_return"]) for point in points if completed(session.rows.get(point))]
    return {
        "geometry": "robust_j", "connectivity": "fixed_edge_cube",
        "scoring_basis": "robust_j_neighborhood", "edge_length": int(session.config.strategy.hypercube_edge),
        "radius_ticks": int(session.config.strategy.hypercube_edge) // 2,
        "region_id": region_id, "candidate_id": row.get("candidate_id"),
        "center_params": dict(row["params"]), "center_coordinate": list(center),
        "q_buy": float(row["params"].get("buy_shares", 0.0)) / max(1, int(row["params"].get("initial_shares", 0) or 0)),
        "q_sell": float(row["params"].get("sell_shares", 0.0)) / max(1, int(row["params"].get("initial_shares", 0) or 0)),
        "radii": {name: int(session.config.strategy.hypercube_edge) // 2 for name in space.active},
        "stable_point": _public_row(row, score), "peak_point": _public_row(row, score),
        "combination_count": len(points), "completed_count": sum(completed(session.rows.get(point)) for point in points),
        "raw_volume": len(points), "normalized_volume": len(points) / max(1, space.size()),
        "mean_excess_return": score.neighborhood_mean, "min_excess_return": min(values) if values else None,
        "max_excess_return": max(values) if values else None,
        "center_excess_return": score.center_return, "j_score": score.j_score,
        "downside_risk_tau": score.downside_risk, "pos_penalty": score.position_penalty,
        "actual_max_pos_ratio": row.get("actual_max_pos_ratio"),
        "actual_min_pos_ratio": row.get("actual_min_pos_ratio"),
        "floor_episode_count": score.floor_episode_count,
        "floor_episode_grids": list(score.floor_episode_grids),
        "floor_penalty": score.min_position_penalty,
        "floor_penalty_basis": "episode_grids_v1",
        "recommended_p_max": row.get("actual_max_pos_ratio"),
        "recommended_p_min": row.get("actual_min_pos_ratio"),
        "neighborhood_max_eta": score.neighborhood_max_eta,
        "trade_count": row.get("trade_count"), "stockout_count": row.get("stockout_count"),
        "tplus1_block_count": row.get("tplus1_block_count"),
        "pos_win_rate": score.positive_ratio, "safe_win_rate": score.safe_ratio,
        "score_complete": score.complete, "local_peak_verified": analysis.complete,
        "platform_type": "flat" if analysis.flat_region else "single",
        "platform_size": len(analysis.plateau), "global_exhaustive": False,
        "bounds": bounds, "bounds_are_envelope": False,
        "constraint_eligible": True, "constraint_reasons": [],
        "region_points": [list(point) for point in points],
        "stress": {},
    }


def _method(config: OptimizationConfig, total: int, space: ParameterSpace) -> str:
    if config.strategy.search_method != "auto":
        return config.strategy.search_method
    estimate_mb = total * int(config.strategy.estimated_bytes_per_point) / (1024 * 1024)
    if total <= int(config.strategy.training_budget) and estimate_mb <= float(config.strategy.exhaustive_memory_limit_mb):
        return "exhaustive"
    return "hierarchical"


def _base_natural(config: OptimizationConfig, base: BacktestConfig) -> BacktestConfig:
    return base.model_copy(update={"min_shares": 0, "max_shares": None, "grid_mode": "geometric", "schema_version": 3})


def select_robust_at(
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
    budget: EvaluationBudget | None = None,
) -> dict[str, Any]:
    del seed_offset
    if config.strategy.strategy_version != 5:
        raise ValueError("稳健 J 选择器需要 strategy_version=5")
    if config.base.initial_shares <= 0:
        raise ValueError("稳健 J 研究要求初始持仓大于零")
    natural = _base_natural(config, base)
    config = config.model_copy(update={"base": natural})
    strategy = config.strategy
    policy = TrainingWindowPolicy(
        int(strategy.training_min_days), int(strategy.training_max_days),
        mode=str(strategy.training_window_mode),
    )
    window = policy.window(data, asof)
    as_of = window.end
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    active = tuple(name for name in ROBUST_FIELDS if name in space.active)
    if window.data_issue:
        # A corporate-action or price problem is not a short history and must
        # never be reported as one.
        return _empty_result(
            config, space, as_of, [window.data_issue], window.dates,
            status="data_issue", window=window,
        )
    if not active:
            return _empty_result(
                config, space, as_of, ["四个搜索轴全部固定，无法形成稳健 J 峰值"],
                window.dates, status="insufficient_range", window=window,
            )
    if window.days < int(strategy.training_min_days):
        return _empty_result(
            config, space, as_of,
            [window.reason, f"训练时段只有 {window.days} 个交易日，少于最低 {strategy.training_min_days} 日"],
            window.dates, window=window,
        )
    edge = int(strategy.hypercube_edge)
    impossible = [name for name in space.active if space.dimensions[name].count < edge]
    if impossible:
        return _empty_result(
            config, space, as_of,
            ["超正方体边长超出这些参数的有效范围：" + "、".join(impossible)],
            window.dates, status="insufficient_range", window=window,
        )
    total = space.size()
    method = _method(config, total, space)
    operator = NeighborhoodOperator(space, edge // 2)
    test_center = space.full_coordinate([(dim.low + dim.high) // 2 for dim in (space.dimensions[name] for name in space.active)])
    minimum_box = len(operator.points(test_center))
    if method != "exhaustive" and minimum_box > int(strategy.validation_budget):
        raise ValueError(f"一个稳健 J 邻域需要 {minimum_box} 个评价点，超过邻域评价预算 {strategy.validation_budget}")
    if method == "exhaustive" and total > int(strategy.training_budget):
        raise ValueError(f"完整枚举需要 {total} 个训练点，超过训练预算 {strategy.training_budget}")
    exhaustive_mb = total * int(strategy.estimated_bytes_per_point) / (1024 * 1024)
    if method == "exhaustive" and exhaustive_mb > float(strategy.exhaustive_memory_limit_mb):
        raise ValueError(
            f"完整枚举预计需要 {exhaustive_mb:.1f} MB，超过内存上限 "
            f"{strategy.exhaustive_memory_limit_mb} MB"
        )
    if method == "hierarchical" and int(strategy.validation_budget) < minimum_box:
        raise ValueError("邻域评价预算不足以计算一个完整稳健 J 分数")
    _emit(progress, progress_start + .02 * progress_width, f"训练时段 {window.start} 至 {window.end}（{window.days} 日）")
    evaluator = BatchEvaluator(
        config, natural, data, space, window.dates,
        cancelled=cancelled, allow_odd_lot_exit=True,
        chunk_size=getattr(runtime, "evaluation_chunk_size", None),
    )
    session = EvaluationSession(evaluator, space, config, cancelled, budget=budget)
    started_seconds = time.perf_counter()
    try:
        if method == "exhaustive":
            coordinates = list(space.iterate(int(strategy.training_budget) + 1))
            if len(coordinates) > int(strategy.training_budget):
                raise ValueError(f"完整枚举需要 {total} 个训练点，超过训练预算 {strategy.training_budget}")
            session.ensure(coordinates, "training", len(coordinates))
            session.scores.update(tensor_robust_scores(space, session.rows, strategy))
            _emit(progress, progress_start + .45 * progress_width, f"完整枚举完成：{len(session.rows):,} 点")
            scored_centers = list(coordinates)
        else:
            profile = market_statistical_profile(data, window.dates)
            initial = _initial_coordinates(space, config, as_of.date().isoformat() if as_of is not None else "", profile)
            session.ensure(initial, "training", int(strategy.training_budget))
            _emit(progress, progress_start + .32 * progress_width, f"覆盖采样完成：{len(session.rows):,} 点")
            target_starts = min(
                int(strategy.start_max),
                max(int(strategy.start_min), int(strategy.starts_per_candidate) * int(strategy.recommendation_count)),
                len(session.rows),
            )
            starts = _start_coordinates(session, space, target_starts)
            reserve = float(strategy.validation_reserve_fraction)
            search_cap = max(0, int(math.floor(int(strategy.validation_budget) * (1.0 - reserve))))
            scored_centers = _local_search(session, space, starts, search_cap)
            _emit(progress, progress_start + .68 * progress_width, f"多起点局部搜索完成：{len(scored_centers):,} 个中心")
        # Exhaustive scores every center; sparse scores only centres that have a
        # complete support set or were reached by the local search.
        for coordinate in session.rows:
            if coordinate not in session.scores:
                session.scores[coordinate] = robust_score(coordinate, session.rows, space, strategy)
        analyzer = PeakAnalyzer(space, float(strategy.peak_tolerance))
        candidates = [
            coordinate for coordinate, score in session.scores.items()
            if score is not None and eligible(
                session.rows.get(coordinate), score,
                max_principal_drawdown=config.max_principal_drawdown,
                min_trades=config.min_trades, strategy=strategy,
            )
        ]
        candidates.sort(key=lambda coordinate: _score_sort(session.scores[coordinate], coordinate))
        verify_started = session.spent["validation"]
        verify_cap = int(strategy.validation_budget)
        peaks: dict[tuple[int, ...], tuple[tuple[int, ...], PeakAnalysis]] = {}
        analyses: list[PeakAnalysis] = []
        seen_representatives: set[tuple[int, ...]] = set()
        for coordinate in candidates:
            if len(peaks) >= int(strategy.recommendation_count):
                break
            if method != "exhaustive" and session.spent["validation"] >= verify_cap:
                break
            analysis = _verify_peak(session, analyzer, coordinate, verify_cap)
            analyses.append(analysis)
            if not analysis.complete or analysis.representative is None:
                continue
            representative = analysis.representative
            if representative in seen_representatives:
                continue
            seen_representatives.add(representative)
            peaks[representative] = (coordinate, analysis)
        ordered = sorted(peaks.items(), key=lambda item: _score_sort(session.scores[item[0]], item[0]))
        regions = []
        for region_id, (representative, (coordinate, analysis)) in enumerate(ordered, 1):
            score = session.scores[representative]
            row = session.rows[representative]
            region = _region(space, representative, row, score, analysis, session, region_id)
            region["global_exhaustive"] = method == "exhaustive" and len(session.rows) == total
            for point in region["region_points"]:
                candidate = session.rows[tuple(point)]
                candidate["region_ids"] = list(dict.fromkeys([
                    *(candidate.get("region_ids") or []), int(region_id),
                ]))
            row["is_stable_point"] = True
            row["is_peak_point"] = True
            regions.append(region)
        _emit(progress, progress_start + .78 * progress_width, "补齐最高超额收益点邻域与 J 排名")
        _complete_reporting_scores(session)
        if int(strategy.stress_region_count) > 0 and regions and not (cancelled and cancelled()):
            stress_regions = regions[:int(strategy.stress_region_count)]
            stress_points = sorted({tuple(point) for region in stress_regions for point in region["region_points"]})
            stress_blocked = False
            if len(stress_points) > session._budget_remaining():
                # The stress batch does not fit in the remaining task budget.
                # The task budget is stopped explicitly here, so the later
                # current-decision and out-of-sample phases cannot keep
                # spending after the stress phase was refused.
                if session.task_budget is not None:
                    session.task_budget.stop("费用压力评价超出总预算")
                session._stop_for_budget("费用压力评价超出总预算")
                stress_blocked = True
            elif session.task_budget is not None and not session.task_budget.reserve(len(stress_points), "费用压力"):
                session._stop_for_budget(session.task_budget.reason or "费用压力评价超出总预算")
                stress_blocked = True
            if stress_blocked:
                if session.task_budget is not None and session.task_budget.reason:
                    session.budget_reason = session.task_budget.reason
                for region in stress_regions:
                    region["stress"] = {"status": "budget_exhausted", "cost_multiplier": float(strategy.stress_cost_multiplier)}
            else:
                try:
                    session.stress_spent += len(stress_points)
                    evidence = evaluator.evaluate_training_stress(stress_points, cost_multiplier=float(strategy.stress_cost_multiplier))
                    for region in stress_regions:
                        values = [float(evidence[tuple(point)]["training_excess_stress"]) for point in region["region_points"] if tuple(point) in evidence and math.isfinite(float(evidence[tuple(point)].get("training_excess_stress", math.nan)))]
                        region["stress"] = {
                            "status": "completed" if len(values) == region["combination_count"] else "partial",
                            "cost_multiplier": float(strategy.stress_cost_multiplier),
                            "mean_excess_return": math.fsum(values) / len(values) if values else None,
                            "min_excess_return": min(values) if values else None,
                        }
                except InterruptedError:
                    raise
                except Exception as exc:
                    for region in stress_regions:
                        region["stress"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        fallback_coordinate = space.full_coordinate(tuple(space.dimensions[name].low for name in space.active))
        diagnostic = candidates[0] if candidates else next(iter(session.rows), fallback_coordinate)
        top = regions[0] if regions else None
        selected = deepcopy(top["center_params"]) if top else deepcopy(session.rows.get(diagnostic, {}).get("params", space.params(fallback_coordinate)))
        public_rows = [_public_row(row, session.scores.get(tuple(row["key"]))) for row in session.rows.values()]
        public_rows.sort(key=lambda row: (-(float(row.get("j_score")) if row.get("j_score") is not None else -math.inf), tuple(row.get("key", ()))))
        for index, row in enumerate(public_rows, 1):
            row["candidate_id"] = index
        public_ranked = []
        for region in regions:
            representative = tuple(region["center_coordinate"])
            public_ranked.append({
                **_public_row(session.rows[representative], session.scores[representative]),
                "region": deepcopy({key: value for key, value in region.items() if key != "region_points"}),
                "selection_score": region["j_score"],
            })
        reasons = [window.reason]
        if method == "hierarchical":
            reasons.append("大范围使用覆盖采样与多起点局部搜索；输出仅包含已完成一格邻域核验的局部峰值")
        if not regions:
            reasons.append("没有发现满足交易、风险、持仓约束且 J>0 的已完成局部峰值")
        diagnostics = {
            "status": "completed", "protocol_version": 5, "strategy_id": "geometric_robust_j",
            "strategy_version": 5, "geometry": "robust_j", "connectivity": "fixed_edge_cube",
            "search_method": method, "ranking_mode": "robust_j",
            "fields": list(ROBUST_FIELDS), "active_fields": list(active),
            "legal_combination_count": int(total), "neighborhood_edge": edge,
            "neighborhood_volume": minimum_box, "peak_support_volume": min(total, int(math.prod(min(space.dimensions[name].count, edge + 2) for name in space.active)) if space.active else 1),
            "training_budget": int(strategy.training_budget), "neighborhood_budget": int(strategy.validation_budget),
            "training_unique_count": session.spent["training"], "neighborhood_unique_count": session.spent["validation"],
            "reporting_unique_count": session.spent["reporting"],
            "stress_unique_count": session.stress_spent,
            "ledger_evaluation_count": session.ledger_evaluations,
            "total_evaluation_cap": session.total_cap,
            "budget_stopped": session.budget_stopped,
            "budget_reason": session.budget_reason,
            "reporting_stopped": session.reporting_stopped,
            "reporting_reason": session.reporting_reason,
            "training_attempt_count": session.attempts, "failed_count": session.failures,
            "candidate_count": len(session.rows),
            "generation_evaluation_count": session.spent["training"],
            "validation_evaluation_count": session.spent["validation"] + session.spent["reporting"],
            "scored_center_count": sum(1 for score in session.scores.values() if score is not None and score.complete),
            "eligible_center_count": len(candidates), "recommendation_count": len(regions),
            "recommendation_status": "qualified" if regions else "no_qualified",
            "score_complete": (
                all(score.complete for score in session.scores.values() if score is not None)
                if method == "exhaustive"
                else bool(regions and all(region["score_complete"] for region in regions))
            ),
            "local_peak_verified": bool(regions and all(region["local_peak_verified"] for region in regions)),
            "global_exhaustive": bool(method == "exhaustive" and len(session.rows) == total),
            "coverage_status": "global_exhaustive" if method == "exhaustive" else "budgeted_local_peaks",
            "training_window": window.diagnostic(), "training_start": window.start.date().isoformat() if window.start is not None else None,
            "training_end": window.end.date().isoformat() if window.end is not None else None,
            "selection_history_days": window.days,
            "selection_history_start": window.start.date().isoformat() if window.start is not None else None,
            "selection_history_end": window.end.date().isoformat() if window.end is not None else None,
            "training_only": True, "internal_validation_folds": 0,
            "return_basis": "peak_principal", "ranking_baseline": "matched_funding_hold",
            "risk_basis": "principal_profit_drawdown",
            "statistics_applicability": {
                "status": "not_applicable",
                "reason": "本版本未提供峰值本金超额收益的显著性检验",
            },
            "platforms": [{
                "platform_id": index, "center": list(analysis.center), "representative": list(analysis.representative or ()),
                "state": analysis.state, "complete": analysis.complete, "flat_region": analysis.flat_region,
                "plateau_size": len(analysis.plateau), "missing_count": len(analysis.missing),
            } for index, analysis in enumerate(analyses, 1)],
            "computation": {
                "training_evaluations": session.spent["training"], "neighborhood_evaluations": session.spent["validation"],
                "reporting_evaluations": session.spent["reporting"],
                "stress_evaluations": sum(int(region.get("combination_count", 0)) for region in regions[:int(strategy.stress_region_count)]),
                "stress_evaluations_charged": session.stress_spent,
                "ledger_evaluations": session.ledger_evaluations,
                "total_evaluation_cap": session.total_cap,
                "budget_stopped": session.budget_stopped,
                "budget_reason": session.budget_reason,
                "reporting_stopped": session.reporting_stopped,
                "reporting_reason": session.reporting_reason,
                "cache_hits": session.cache_hits,
                "skipped_count": int(session.skipped),
            },
            "selected_constraint_reasons": [],
            "method_parameters": strategy.model_dump(mode="json"),
        }
        record_seconds("robust_j_selection", time.perf_counter() - started_seconds)
        record_count("robust_j_training_points", session.spent["training"])
        record_count("robust_j_neighborhood_points", session.spent["validation"])
        record_count("robust_j_reporting_points", session.spent["reporting"])
        return {
            "status": "partial" if session.budget_stopped or session.reporting_stopped else "completed",
            "recommendation_status": "qualified" if regions else "no_qualified",
            "params": selected,
            "recommended_params": deepcopy(regions[0]["center_params"]) if regions else None,
            "as_of": as_of.date().isoformat() if as_of is not None else None,
            "score": float(regions[0]["j_score"]) if regions else None,
            "raw_score": float(regions[0]["center_excess_return"]) if regions else None,
            "eligible": bool(regions), "reasons": list(dict.fromkeys(reasons)),
            "candidates": public_ranked,
            "candidate_archive": public_rows,
            "search_candidate_archive": public_rows,
            "trial_archive": public_rows,
            "recommendation_regions": regions,
            "diagnostics": diagnostics,
        }
    except InterruptedError:
        _emit(progress, .95, "稳健 J 研究已取消")
        return _cancelled_selection(config, space, session, window, as_of)


def _empty_result(
    config: OptimizationConfig,
    space: ParameterSpace,
    asof: pd.Timestamp | None,
    reasons: list[str],
    history: pd.DatetimeIndex,
    *,
    status: str = "insufficient_history",
    window: TrainingWindow | None = None,
) -> dict[str, Any]:
    fallback = space.params(space.full_coordinate(tuple(space.dimensions[name].low for name in space.active)))
    diagnostic = window.diagnostic() if window is not None else {
        "start": history[0].date().isoformat() if len(history) else None,
        "end": history[-1].date().isoformat() if len(history) else None,
        "days": len(history),
    }
    return {
        "status": status, "recommendation_status": "no_qualified",
        "params": fallback, "recommended_params": None,
        "as_of": asof.date().isoformat() if asof is not None else None,
        "score": None, "raw_score": None, "eligible": False,
        "reasons": list(dict.fromkeys(reasons)), "candidates": [], "candidate_archive": [],
        "search_candidate_archive": [], "trial_archive": [], "recommendation_regions": [],
        "diagnostics": {
            "status": status, "protocol_version": 5, "strategy_id": "geometric_robust_j",
            "strategy_version": 5, "geometry": "robust_j",
            "selection_history_days": len(history), "candidate_count": 0, "eligible_count": 0,
            "training_only": True, "internal_validation_folds": 0,
            "search_method": config.strategy.search_method,
            "ranking_mode": "robust_j", "training_window": diagnostic,
            "data_issue": diagnostic.get("data_issue"),
        },
    }


def _cancelled_selection(
    config: OptimizationConfig, space: ParameterSpace, session: EvaluationSession,
    window: TrainingWindow, as_of: pd.Timestamp | None,
) -> dict[str, Any]:
    public_rows = [_public_row(row, session.scores.get(tuple(row["key"]))) for row in session.rows.values()]
    public_rows.sort(
        key=lambda row: (
            -(float(row.get("j_score")) if row.get("j_score") is not None else -math.inf),
            tuple(row.get("key", ())),
        )
    )
    for index, row in enumerate(public_rows, 1):
        row["candidate_id"] = index
    fallback = space.params(space.full_coordinate(tuple(space.dimensions[name].low for name in space.active)))
    diagnostic = next(iter(session.rows.values()), {})
    params = deepcopy(diagnostic.get("params", fallback))
    return {
        "status": "cancelled", "recommendation_status": "no_qualified",
        "params": params, "recommended_params": None,
        "as_of": as_of.date().isoformat() if as_of is not None else None,
        "score": None, "raw_score": None, "eligible": False,
        "reasons": ["用户取消，已保留完成的训练候选"],
        "candidates": [], "candidate_archive": public_rows,
        "search_candidate_archive": public_rows, "trial_archive": public_rows,
        "recommendation_regions": [],
        "diagnostics": {
            "status": "cancelled", "protocol_version": 5,
            "strategy_id": "geometric_robust_j", "strategy_version": 5,
            "training_window": window.diagnostic(),
            "training_start": window.start.date().isoformat() if window.start is not None else None,
            "training_end": window.end.date().isoformat() if window.end is not None else None,
            "selection_history_days": window.days,
            "training_only": True, "internal_validation_folds": 0,
            "candidate_count": len(session.rows),
            "training_unique_count": session.spent["training"],
            "neighborhood_unique_count": session.spent["validation"],
            "reporting_unique_count": session.spent["reporting"],
            "ledger_evaluation_count": session.ledger_evaluations,
            "budget_stopped": session.budget_stopped,
            "budget_reason": session.budget_reason,
            "reporting_stopped": session.reporting_stopped,
            "reporting_reason": session.reporting_reason,
            "score_complete": False, "local_peak_verified": False,
            "global_exhaustive": False, "coverage_status": "cancelled",
            "return_basis": "peak_principal", "ranking_baseline": "matched_funding_hold",
            "risk_basis": "principal_profit_drawdown",
            "statistics_applicability": {
                "status": "not_applicable",
                "reason": "本版本未提供峰值本金超额收益的显著性检验",
            },
        },
    }


def estimate_robust_search(config: OptimizationConfig, data: DataBundle) -> dict[str, Any]:
    if config.protocol_version != 5 or config.strategy.strategy_version != 5:
        raise ValueError("稳健 J 预检需要 protocol_version=5 和 strategy_version=5")
    natural = _base_natural(config, config.base)
    normalized = config.model_copy(update={"base": natural})
    space = ParameterSpace(normalized, field_order=ROBUST_FIELDS)
    dates = pd.DatetimeIndex(data.calendar)
    eval_start = pd.Timestamp(config.base.start_date).normalize()
    asof = pd.Timestamp(eval_start) - pd.Timedelta(days=1)
    closer = dates[dates < eval_start]
    if len(closer):
        asof = closer[-1]
    strategy = config.strategy
    policy = TrainingWindowPolicy(
        int(strategy.training_min_days), int(strategy.training_max_days),
        mode=str(strategy.training_window_mode),
    )
    window = policy.window(data, asof)
    total = space.size()
    method = _method(normalized, total, space)
    edge = int(strategy.hypercube_edge)
    active = tuple(name for name in space.active)
    volume = edge ** max(1, len(active)) if active else 1
    peak_support = int(math.prod(min(space.dimensions[name].count, edge + 2) for name in space.active)) if space.active else 1
    reasons: list[str] = []
    impossible = [name for name in active if space.dimensions[name].count < edge]
    bytes_per_point = int(strategy.estimated_bytes_per_point)
    stress_points = int(strategy.stress_region_count) * peak_support
    candidate_upper = min(
        total,
        int(strategy.training_budget) + int(strategy.validation_budget)
        + int(strategy.reporting_budget) + stress_points,
    )
    memory_mb = total * bytes_per_point / (1024 * 1024)
    sampled_memory_mb = candidate_upper * bytes_per_point / (1024 * 1024)
    frame_bytes = 0
    for frame in (data.bars, data.daily, data.actions, pd.DataFrame({"date": data.calendar})):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame_bytes += int(frame.memory_usage(index=True, deep=True).sum())
    # One native batch holds its daily x candidate result plus the paired
    # account state.  Candidate cache and JSON archives are retained alongside.
    evaluation_days_estimate = max(1, int(len(dates[dates <= asof])))
    batch_candidates = min(512, candidate_upper)
    ledger_buffer_mb = batch_candidates * evaluation_days_estimate * 8 * 4 / (1024 * 1024)
    tensor_mb = min(total, int(strategy.training_budget)) * 6 * 8 / (1024 * 1024) if active else 0.0
    memory_estimate_mb = (
        frame_bytes / (1024 * 1024) + sampled_memory_mb
        + ledger_buffer_mb + tensor_mb
    )
    workspace_mb = current_workspace_bytes() / (1024 * 1024)
    per_decision_upper = (
        int(strategy.training_budget) + int(strategy.validation_budget)
        + int(strategy.reporting_budget) + stress_points
    )
    evaluation_days = int(len(dates[dates >= eval_start]))
    fixed_count = 1 if config.selection_target in ("fixed", "both") else 0
    adaptive_count = math.ceil(evaluation_days / max(1, int(config.rebalance_days))) if config.selection_target in ("adaptive", "both") else 0
    replay_runs = fixed_count + (1 if adaptive_count else 0)
    task_upper = per_decision_upper * (fixed_count + adaptive_count + 1) + replay_runs
    effective_total = task_upper if strategy.total_evaluation_budget is None else min(task_upper, int(strategy.total_evaluation_budget))
    if window.data_issue:
        status = "data_issue"
        reasons.append(window.data_issue)
    elif window.days < int(strategy.training_min_days):
        status = "insufficient_history"
        reasons.append(f"训练时段只有 {window.days} 个交易日，少于最低 {strategy.training_min_days} 日")
    elif impossible:
        status = "insufficient_range"
        reasons.append("超正方体边长超出这些参数的有效范围：" + "、".join(impossible))
    elif method != "exhaustive" and volume > int(strategy.validation_budget):
        status = "budget_exceeded"
        reasons.append(f"一个稳健 J 邻域需要 {volume} 个评价点，超过邻域评价预算 {strategy.validation_budget}")
    elif method == "exhaustive" and total > int(strategy.training_budget):
        status = "budget_exceeded"
        reasons.append(f"完整枚举需要 {total} 个训练点，超过训练预算 {strategy.training_budget}")
    elif method == "exhaustive" and memory_mb > float(strategy.exhaustive_memory_limit_mb):
        status = "budget_exceeded"
        reasons.append(
            f"完整枚举预计需要 {memory_mb:.1f} MB，超过内存上限 "
            f"{strategy.exhaustive_memory_limit_mb} MB"
        )
    elif memory_estimate_mb > workspace_mb:
        status = "budget_exceeded"
        reasons.append(
            f"任务预计占用 {memory_estimate_mb:.1f} MB，超过工作区上限 {workspace_mb:.1f} MB"
        )
    else:
        status = "ready"
    return {
        "status": status, "protocol_version": 5, "strategy_id": "geometric_robust_j",
        "strategy_version": 5, "geometry": "robust_j", "grid_mode": "geometric",
        "protocol": config.protocol_description(),
        "fields": list(ROBUST_FIELDS), "active_fields": list(active),
        "granularity": {name: space.granularity[name] for name in ROBUST_FIELDS},
        "legal_combination_count": int(total), "search_method": method,
        "hypercube_edge": edge, "hypercube_volume": volume, "peak_support_volume": peak_support,
        "training_budget": int(strategy.training_budget), "neighborhood_budget": int(strategy.validation_budget),
        "reporting_budget": int(strategy.reporting_budget),
        "estimated_memory_mb": memory_mb, "exhaustive_memory_limit_mb": int(strategy.exhaustive_memory_limit_mb),
        "estimated_bytes_per_point": bytes_per_point, "estimated_sampled_memory_mb": sampled_memory_mb,
        "estimated_task_memory_mb": memory_estimate_mb,
        "workspace_budget_mb": workspace_mb,
        "estimated_peak_support_points": peak_support,
        "per_decision_evaluation_upper_bound": per_decision_upper,
        "replay_evaluation_upper_bound": replay_runs,
        "task_evaluation_upper_bound": task_upper,
        "effective_total_evaluation_budget": effective_total,
        "max_runtime_seconds": int(strategy.max_runtime_seconds),
        "decision_count_estimate": int(fixed_count + adaptive_count + 1),
        "history_days": window.days, "training_days": window.days, "validation_days": 0,
        "full_training_window": window.days >= int(strategy.training_min_days),
        "training_window": window.diagnostic(), "minimum_neighborhood_possible": not impossible,
        "minimum_neighborhood_size": volume,
        "effective_bounds": {name: {"low": space.value(name, dim.low), "high": space.value(name, dim.high), "low_ticks": dim.low, "high_ticks": dim.high, "fixed": dim.fixed} for name, dim in space.dimensions.items() if name in ROBUST_FIELDS},
        "slippage_mode": strategy.slippage_mode, "reasons": list(dict.fromkeys(reasons)),
    }
