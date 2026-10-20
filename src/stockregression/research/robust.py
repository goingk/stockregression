"""Four-axis robust-J research: objective, budgeted search and peak evidence."""
from __future__ import annotations

from copy import deepcopy
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
from ..performance import record_count, record_seconds
from ..research_v4.evaluator import BatchEvaluator
from ..research_v4.search import coarse_coordinates
from ..research_v4.space import ROBUST_FIELDS, ParameterSpace
from .window import TrainingWindowPolicy
from .statistics import market_statistical_profile

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


def _emit(progress: Progress | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, float(fraction))), message)


def _stable_seed(config: OptimizationConfig, asof: str, purpose: str) -> int:
    payload = {
        "seed": int(config.seed), "asof": asof, "purpose": purpose,
        "strategy": config.strategy.model_dump(mode="json"),
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
    def __init__(self, evaluator: BatchEvaluator, space: ParameterSpace, config: OptimizationConfig, cancelled: Cancelled | None):
        self.evaluator = evaluator
        self.space = space
        self.config = config
        self.cancelled = cancelled
        self.rows: dict[tuple[int, ...], dict[str, Any]] = {}
        self.attempts = 0
        self.requested = 0
        self.failures = 0
        self.spent = {"training": 0, "validation": 0}
        self.scores: dict[tuple[int, ...], RobustScore | None] = {}

    def ensure(self, coordinates: Iterable[tuple[int, ...]], phase: str, cap: int) -> int:
        requested = list(dict.fromkeys(coordinates))
        self.requested += len(requested)
        missing = [coordinate for coordinate in requested if coordinate not in self.rows]
        allowed = max(0, int(cap) - int(self.spent[phase]))
        missing = missing[:allowed]
        if not missing:
            return 0
        if self.cancelled and self.cancelled():
            raise InterruptedError("研究已取消")
        self.spent[phase] += len(missing)
        self.attempts += len(missing)
        for begin in range(0, len(missing), self.evaluator.chunk_size):
            chunk = missing[begin:begin + self.evaluator.chunk_size]
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
                        "training_excess_return": math.nan, "training_drawdown": math.inf,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
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
    return eligible(session.rows.get(center), score, max_drawdown=session.config.max_drawdown, min_trades=session.config.min_trades, strategy=session.config.strategy)


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
) -> dict[str, Any]:
    del seed_offset
    if config.strategy.strategy_version != 4:
        raise ValueError("稳健 J 选择器需要 strategy_version=4")
    if config.base.initial_shares <= 0:
        raise ValueError("稳健 J 研究要求初始持仓大于零")
    natural = _base_natural(config, base)
    config = config.model_copy(update={"base": natural})
    strategy = config.strategy
    policy = TrainingWindowPolicy(int(strategy.training_min_days), int(strategy.training_max_days))
    window = policy.window(data, asof)
    as_of = window.end
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    active = tuple(name for name in ROBUST_FIELDS if name in space.active)
    if not active:
            return _empty_result(
                config, space, as_of, ["四个搜索轴全部固定，无法形成稳健 J 峰值"],
                window.dates, status="insufficient_range",
            )
    if window.days < int(strategy.training_min_days):
        return _empty_result(config, space, as_of, [window.reason, f"训练时段只有 {window.days} 个交易日，少于最低 {strategy.training_min_days} 日"], window.dates)
    edge = int(strategy.hypercube_edge)
    impossible = [name for name in space.active if space.dimensions[name].count < edge]
    if impossible:
        return _empty_result(
            config, space, as_of,
            ["超正方体边长超出这些参数的有效范围：" + "、".join(impossible)],
            window.dates, status="insufficient_range",
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
        config, natural, data, space, window.dates, window.dates,
        validation_folds=[], cancelled=cancelled, allow_odd_lot_exit=True,
        chunk_size=getattr(runtime, "evaluation_chunk_size", None),
    )
    evaluator.retain_daily_returns = False
    session = EvaluationSession(evaluator, space, config, cancelled)
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
            if score is not None and eligible(session.rows.get(coordinate), score, max_drawdown=config.max_drawdown, min_trades=config.min_trades, strategy=strategy)
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
            regions.append(region)
        if int(strategy.stress_region_count) > 0 and regions and not (cancelled and cancelled()):
            stress_regions = regions[:int(strategy.stress_region_count)]
            stress_points = sorted({tuple(point) for region in stress_regions for point in region["region_points"]})
            try:
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
            "status": "completed", "protocol_version": 4, "strategy_id": "geometric_robust_j",
            "strategy_version": 4, "geometry": "robust_j", "connectivity": "fixed_edge_cube",
            "search_method": method, "ranking_mode": "robust_j",
            "fields": list(ROBUST_FIELDS), "active_fields": list(active),
            "legal_combination_count": int(total), "neighborhood_edge": edge,
            "neighborhood_volume": minimum_box, "peak_support_volume": min(total, int(math.prod(min(space.dimensions[name].count, edge + 2) for name in space.active)) if space.active else 1),
            "training_budget": int(strategy.training_budget), "neighborhood_budget": int(strategy.validation_budget),
            "training_unique_count": session.spent["training"], "neighborhood_unique_count": session.spent["validation"],
            "training_attempt_count": session.attempts, "failed_count": session.failures,
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
            "selection_history_days": window.days, "validation_folds": [],
            "platforms": [{
                "platform_id": index, "center": list(analysis.center), "representative": list(analysis.representative or ()),
                "state": analysis.state, "complete": analysis.complete, "flat_region": analysis.flat_region,
                "plateau_size": len(analysis.plateau), "missing_count": len(analysis.missing),
            } for index, analysis in enumerate(analyses, 1)],
            "computation": {
                "training_evaluations": session.spent["training"], "neighborhood_evaluations": session.spent["validation"],
                "stress_evaluations": sum(int(region.get("combination_count", 0)) for region in regions[:int(strategy.stress_region_count)]),
                "cache_hits": max(0, session.requested - session.attempts),
            },
            "selected_constraint_reasons": [],
            "method_parameters": strategy.model_dump(mode="json"),
        }
        neighborhoods = {tuple(region["center_coordinate"]): [tuple(point) for point in region["region_points"]] for region in regions}
        record_seconds("robust_j_selection", time.perf_counter() - started_seconds)
        record_count("robust_j_training_points", session.spent["training"])
        record_count("robust_j_neighborhood_points", session.spent["validation"])
        return {
            "status": "completed", "recommendation_status": "qualified" if regions else "no_qualified",
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
            "_ranked_rows": public_ranked, "_family_rows": public_rows,
            "_neighborhoods": neighborhoods, "_search_rows": public_rows,
            "_fold_ranges": [], "_selection_data": data, "_selection_dates": window.dates,
        }
    except InterruptedError:
        _emit(progress, .95, "稳健 J 研究已取消")
        raise


def _empty_result(
    config: OptimizationConfig,
    space: ParameterSpace,
    asof: pd.Timestamp | None,
    reasons: list[str],
    history: pd.DatetimeIndex,
    *,
    status: str = "insufficient_history",
) -> dict[str, Any]:
    fallback = space.params(space.full_coordinate(tuple(space.dimensions[name].low for name in space.active)))
    return {
        "status": status, "recommendation_status": "no_qualified",
        "params": fallback, "recommended_params": None,
        "as_of": asof.date().isoformat() if asof is not None else None,
        "score": None, "raw_score": None, "eligible": False,
        "reasons": list(dict.fromkeys(reasons)), "candidates": [], "candidate_archive": [],
        "search_candidate_archive": [], "trial_archive": [], "recommendation_regions": [],
        "diagnostics": {
            "status": status, "protocol_version": 4, "strategy_id": "geometric_robust_j",
            "strategy_version": 4, "geometry": "robust_j", "training_only": True,
            "selection_history_days": len(history), "candidate_count": 0, "eligible_count": 0,
            "validation_folds": [], "search_method": config.strategy.search_method,
            "ranking_mode": "robust_j", "training_window": {
                "start": history[0].date().isoformat() if len(history) else None,
                "end": history[-1].date().isoformat() if len(history) else None,
                "days": len(history),
            },
        },
        "_ranked_rows": [], "_family_rows": [], "_neighborhoods": {}, "_search_rows": [],
        "_fold_ranges": [], "_selection_data": None, "_selection_dates": history,
    }


def estimate_robust_search(config: OptimizationConfig, data: DataBundle) -> dict[str, Any]:
    if config.protocol_version != 4 or config.strategy.strategy_version != 4:
        raise ValueError("稳健 J 预检需要 protocol_version=4 和 strategy_version=4")
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
    policy = TrainingWindowPolicy(int(strategy.training_min_days), int(strategy.training_max_days))
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
    memory_mb = total * bytes_per_point / (1024 * 1024)
    sampled_memory_mb = min(total, int(strategy.training_budget) + int(strategy.validation_budget)) * bytes_per_point / (1024 * 1024)
    if window.days < int(strategy.training_min_days):
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
    else:
        status = "ready"
    evaluation_days = int(len(dates[dates >= eval_start]))
    fixed_count = 1 if config.selection_target in ("fixed", "both") else 0
    adaptive_count = math.ceil(evaluation_days / max(1, int(config.rebalance_days))) if config.selection_target in ("adaptive", "both") else 0
    return {
        "status": status, "protocol_version": 4, "strategy_id": "geometric_robust_j",
        "strategy_version": 4, "geometry": "robust_j", "grid_mode": "geometric",
        "fields": list(ROBUST_FIELDS), "active_fields": list(active),
        "granularity": {name: space.granularity[name] for name in ROBUST_FIELDS},
        "legal_combination_count": int(total), "search_method": method,
        "hypercube_edge": edge, "hypercube_volume": volume, "peak_support_volume": peak_support,
        "training_budget": int(strategy.training_budget), "neighborhood_budget": int(strategy.validation_budget),
        "estimated_memory_mb": memory_mb, "exhaustive_memory_limit_mb": int(strategy.exhaustive_memory_limit_mb),
        "estimated_bytes_per_point": bytes_per_point, "estimated_sampled_memory_mb": sampled_memory_mb,
        "estimated_peak_support_points": peak_support,
        "decision_count_estimate": int(fixed_count + adaptive_count + 1),
        "history_days": window.days, "training_days": window.days, "validation_days": 0,
        "full_training_window": window.days >= int(strategy.training_min_days),
        "training_window": window.diagnostic(), "minimum_neighborhood_possible": not impossible,
        "minimum_neighborhood_size": volume,
        "effective_bounds": {name: {"low": space.value(name, dim.low), "high": space.value(name, dim.high), "low_ticks": dim.low, "high_ticks": dim.high, "fixed": dim.fixed} for name, dim in space.dimensions.items() if name in ROBUST_FIELDS},
        "slippage_mode": strategy.slippage_mode, "reasons": list(dict.fromkeys(reasons)),
    }
