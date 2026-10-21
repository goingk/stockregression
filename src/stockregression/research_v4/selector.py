"""Protocol v4 selection orchestration."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable

import pandas as pd

from ..models import GRANULARITY_DEFAULTS, BacktestConfig, DataBundle, OptimizationConfig
from .. import scientific
from .evaluator import BatchEvaluator, center_constraints
from .neighborhoods import find_regions, region_sort_key
from .search import (
    ValidationPlan,
    build_validation_plan,
    initial_training_coordinates,
    refine_coordinates,
    statistical_coordinates,
)
from .space import FIELDS, UNITS, ParameterSpace, value_to_ticks


Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


def _history(data: DataBundle, asof: pd.Timestamp | None) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(data.calendar)
    if asof is None:
        return pd.DatetimeIndex([])
    return dates[dates <= pd.Timestamp(asof).normalize()][-504:]


def _method(config: OptimizationConfig, total: int) -> str:
    if config.strategy.search_method != "auto":
        return config.strategy.search_method
    if total <= int(config.strategy.training_budget) and total <= int(config.strategy.validation_budget):
        return "exhaustive"
    return "hierarchical"


def _movable_fields(space: ParameterSpace, config: OptimizationConfig) -> tuple[str, ...]:
    selected = set(config.strategy.neighborhood_fields)
    movable = tuple(name for name in space.active if name in selected)
    if space.active and not movable:
        raise ValueError("细搜索邻域至少需要一个落在有效搜索范围内的可移动参数")
    return movable


def _empty_result(
    config: OptimizationConfig,
    space: ParameterSpace,
    asof: pd.Timestamp | None,
    reasons: list[str],
    history_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    movable = tuple(name for name in space.active if name in set(config.strategy.neighborhood_fields))
    fallback = space.params(space.full_coordinate(tuple(space.dimensions[name].low for name in space.active)))
    return {
        "status": "insufficient_history",
        "recommendation_status": "no_qualified",
        "params": fallback,
        "recommended_params": None,
        "as_of": asof.date().isoformat() if asof is not None else None,
        "score": None,
        "raw_score": None,
        "eligible": False,
        "reasons": list(dict.fromkeys(reasons)),
        "candidates": [],
        "candidate_archive": [],
        "search_candidate_archive": [],
        "trial_archive": [],
        "recommendation_regions": [],
        "diagnostics": {
            "status": "insufficient_history",
            "protocol_version": 4,
            "strategy_version": config.strategy.strategy_version,
            "geometry": ("connected_lattice" if config.strategy.strategy_version == 2
                         else "hypercube_mean" if config.strategy.strategy_version == 3 else "symmetric_box"),
            "budget_role": ("seed_discovery" if config.strategy.strategy_version == 2
                            else "training_sample_and_box_cap" if config.strategy.strategy_version == 3 else "evaluation_cap"),
            "expand_until_closed": config.strategy.strategy_version == 2,
            "training_only": config.strategy.strategy_version == 3,
            "selection_history_days": int(len(history_dates)),
            "candidate_count": 0,
            "eligible_count": 0,
            "validation_folds": [],
            "search_method": config.strategy.search_method,
            "ranking_mode": config.strategy.ranking_mode,
            "neighborhood_fields": list(movable),
        },
        "_ranked_rows": [],
        "_family_rows": [],
        "_neighborhoods": {},
        "_search_rows": [],
        "_fold_ranges": [],
        "_selection_data": None,
        "_selection_dates": history_dates,
    }


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in row.items()
        if not key.startswith("_")
    }


def _public_region(region: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in region.items() if key != "region_points"}


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
    if config.strategy.strategy_version == 4:
        from ..research.robust import select_robust_at
        return select_robust_at(
            config, base, data, asof, seed_offset=seed_offset, progress=progress,
            cancelled=cancelled, progress_start=progress_start, progress_width=progress_width,
            runtime=runtime,
        )
    if config.strategy.strategy_version == 3:
        from .hypercube_selector import select_hypercube_at
        return select_hypercube_at(
            config, base, data, asof, seed_offset=seed_offset, progress=progress,
            cancelled=cancelled, progress_start=progress_start, progress_width=progress_width,
        )
    if config.strategy.strategy_version == 2:
        from .connected_selector import select_connected_at
        return select_connected_at(
            config, base, data, asof, seed_offset=seed_offset, progress=progress,
            cancelled=cancelled, progress_start=progress_start, progress_width=progress_width,
        )
    if int(config.protocol_version) != 4:
        raise ValueError("v4 selector requires protocol_version=4")
    if base.schema_version != 3:
        raise ValueError("protocol_version=4 requires schema_version=3")
    if base.grid_mode != "geometric":
        raise ValueError("v4 参数研究只支持等比网格")
    history = _history(data, asof)
    as_of = history[-1] if len(history) else (pd.Timestamp(asof).normalize() if asof is not None else None)
    space = ParameterSpace(config)
    movable = _movable_fields(space, config)
    reasons: list[str] = []
    generation, folds, split, window_reasons = scientific._folds_for_history(history)
    reasons.extend(window_reasons)
    if len(history) < 168 or not folds:
        return _empty_result(config, space, as_of, [*reasons, "没有分离的后252日验证窗口，不授予正式推荐"], history)
    validation_dates = pd.DatetimeIndex([])
    for start, end in folds:
        validation_dates = validation_dates.append(history[(history >= start) & (history <= end)])
    validation_dates = validation_dates.unique().sort_values()
    if len(generation) == 0 or len(validation_dates) == 0:
        return _empty_result(config, space, as_of, [*reasons, "训练或验证区间为空"], history)
    total = space.size()
    method = _method(config, total)
    if method == "exhaustive" and (
        total > int(config.strategy.training_budget) or total > int(config.strategy.validation_budget)
    ):
        raise ValueError(
            f"完整枚举需要 {total} 个参数点，超过训练预算 {config.strategy.training_budget} 或验证预算 {config.strategy.validation_budget}"
        )
    evaluator = BatchEvaluator(
        config, base, data, space, generation, validation_dates,
        validation_folds=list(folds),
        cancelled=cancelled, progress=progress,
    )
    if progress:
        progress(progress_start + 0.02 * progress_width, "构建等比网格参数空间")

    if method == "exhaustive":
        training_coordinates = list(space.iterate(int(config.strategy.training_budget) + 1))
        if len(training_coordinates) > int(config.strategy.training_budget):
            raise ValueError(
                f"完整枚举需要 {total} 个训练点，超过训练预算 {config.strategy.training_budget}"
            )
    else:
        training_coordinates = initial_training_coordinates(space, config, as_of.date().isoformat() if as_of else "")
        try:
            profile = scientific._statistical_profile(base, data, generation)
            seeds = statistical_coordinates(space, profile)
        except Exception:
            seeds = []
        if seeds:
            existing = set(training_coordinates)
            room = max(0, int(config.strategy.training_budget) - len(training_coordinates))
            additions = [item for item in seeds if item not in existing][:room]
            if not additions and len(training_coordinates) >= 3:
                training_coordinates = training_coordinates[:-len(seeds)] + seeds
            else:
                training_coordinates.extend(additions)
    training_rows = evaluator.evaluate_training(training_coordinates)
    if method == "hierarchical":
        remaining = max(0, int(config.strategy.training_budget) - len(training_coordinates))
        extra = refine_coordinates(
            space,
            training_rows,
            existing=set(training_coordinates),
            limit=remaining,
            recommendation_count=int(config.strategy.recommendation_count),
            movable_fields=movable,
        )
        if extra:
            training_rows.extend(evaluator.evaluate_training(extra))
    training_rows = sorted({tuple(row["key"]): row for row in training_rows}.values(), key=lambda row: tuple(row["key"]))
    for index, row in enumerate(training_rows, start=1):
        row["search_id"] = index
    if cancelled and cancelled():
        raise InterruptedError("研究已取消")
    if progress:
        progress(progress_start + 0.48 * progress_width, "训练搜索完成，冻结验证参数域")

    exhaustive_plan = method == "exhaustive"
    plan: ValidationPlan = build_validation_plan(
        space, config, training_rows, asof=as_of.date().isoformat() if as_of else "", exhaustive=exhaustive_plan,
    )
    # No validation score is read before this point.
    if progress:
        progress(progress_start + 0.52 * progress_width, f"验证域已冻结：{len(plan.coordinates)} 个参数点")
    validation_rows = evaluator.evaluate_validation(plan.coordinates)
    validation_rows.sort(key=lambda row: tuple(row["key"]))
    for index, row in enumerate(validation_rows, start=1):
        row["candidate_id"] = index
        legal, constraint_reasons = center_constraints(row, config.max_drawdown, config.min_trades)
        row["constraint_eligible"] = legal
        row["constraint_reasons"] = constraint_reasons
        row["eligible"] = bool(legal and row.get("evaluation_status") == "completed")
        row["is_center"] = False
        row["is_baseline"] = False
    rows = {tuple(row["key"]): row for row in validation_rows}
    global_coverage = bool(
        len(plan.coordinates) == total
        and all(row.get("evaluation_status") == "completed" for row in validation_rows)
    )
    regions, platforms = find_regions(
        space,
        rows,
        ranking_mode=config.strategy.ranking_mode,
        fixed_half_widths=config.strategy.fixed_half_widths,
        neighborhood_fields=movable,
        max_drawdown=float(config.max_drawdown),
        min_trades=int(config.min_trades),
        global_exhaustive=global_coverage,
    )
    for region in regions:
        rows[tuple(region["center_coordinate"])]["is_center"] = True
        for point in region["region_points"]:
            row = rows.get(tuple(point))
            if row is not None:
                row.setdefault("region_ids", []).append(int(region.get("region_id", 0)))
    recommendations = regions[: int(config.strategy.recommendation_count)]

    # Stress is a diagnostic over the union of recommended regions only.
    stress_coords = sorted({tuple(point) for region in recommendations for point in region["region_points"]})
    if not recommendations:
        completed = [row for row in validation_rows if row.get("evaluation_status") == "completed"]
        if completed:
            diagnostic = max(completed, key=lambda row: (float(row.get("validation_excess_original", -math.inf)), -int(row["candidate_id"])))
            stress_coords = [tuple(diagnostic["key"])]
    if stress_coords and not (cancelled and cancelled()):
        try:
            stress = evaluator.evaluate_stress(stress_coords)
            for coordinate, evidence in stress.items():
                if coordinate in rows:
                    rows[coordinate]["validation_excess_stress"] = evidence.get("validation_excess_stress")
                    rows[coordinate]["stress_folds"] = evidence.get("folds", [])
            for region in regions:
                evidence = [stress.get(tuple(point)) for point in region["region_points"]]
                values = [
                    float(item["validation_excess_stress"])
                    for item in evidence
                    if item and isinstance(item.get("validation_excess_stress"), (int, float))
                    and math.isfinite(float(item["validation_excess_stress"]))
                ]
                center_stress = stress.get(tuple(region["center_coordinate"]))
                region["stress"] = {
                    "status": "completed" if len(values) == len(region["region_points"]) else "partial",
                    "center_excess_return": float(center_stress["validation_excess_stress"])
                    if center_stress and isinstance(center_stress.get("validation_excess_stress"), (int, float)) else None,
                    "mean_excess_return": sum(values) / len(values) if values else None,
                    "min_excess_return": min(values) if values else None,
                }
        except InterruptedError:
            raise
        except Exception as exc:  # diagnostics must not revoke original-cost qualification
            for region in regions:
                region["stress"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    diagnostic = max(
        (row for row in validation_rows if row.get("evaluation_status") == "completed"),
        key=lambda row: (float(row.get("validation_excess_original", -math.inf)), -int(row["candidate_id"])),
        default=None,
    )
    top = recommendations[0] if recommendations else None
    selected = deepcopy(top["center_params"]) if top else deepcopy(diagnostic["params"]) if diagnostic else space.params(space.full_coordinate(()))
    if not recommendations:
        reasons.append("没有发现完整且全部为正的局部邻域")
    if method == "hierarchical":
        reasons.append("大范围使用分层搜索；结果只代表冻结预算内发现的候选")
    if progress:
        progress(progress_start + progress_width, "峰与邻域分析完成")
    public_regions = [_public_region(region) for region in recommendations]
    public_candidates = [_public_row(row) for row in validation_rows]
    diagnostics = {
        "status": "completed",
        "protocol_version": 4,
        "strategy_id": config.strategy.strategy_id,
        "strategy_version": config.strategy.strategy_version,
        "grid_mode": "geometric",
        "search_method": method,
        "ranking_mode": config.strategy.ranking_mode,
        "neighborhood_fields": list(movable),
        "units": {**UNITS, "shares": 100},
        "training_budget": int(config.strategy.training_budget),
        "validation_budget": int(config.strategy.validation_budget),
        "legal_combination_count": int(total),
        "training_unique_count": len(training_rows),
        "validation_unique_count": len(validation_rows),
        "validation_planned_count": len(plan.coordinates),
        "unique_generated_count": len(training_rows),
        "generation_evaluation_count": len(training_rows),
        "validation_evaluation_count": len(validation_rows),
        "local_search_unique": len(validation_rows),
        "neighbor_count": sum(int(region["combination_count"]) for region in regions),
        "candidate_count": len(validation_rows),
        "eligible_count": len(regions),
        "peak_platform_count": sum(1 for item in platforms if item["local_peak"] and item["complete"]),
        "recommendation_count": len(recommendations),
        "recommendation_status": "qualified" if recommendations else "no_qualified",
        "global_exhaustive": global_coverage,
        "coverage_status": "global_exhaustive" if global_coverage else "budgeted_freeze",
        "validation_plan_hash": plan.plan_hash,
        "selection_history_days": int(len(history)),
        "selection_history_start": history[0].date().isoformat() if len(history) else None,
        "selection_history_end": history[-1].date().isoformat() if len(history) else None,
        "training_start": generation[0].date().isoformat(),
        "training_end": generation[-1].date().isoformat(),
        "validation_start": validation_dates[0].date().isoformat(),
        "validation_end": validation_dates[-1].date().isoformat(),
        "validation_account_mode": "fresh_initial_config",
        "validation_return_basis": "peak_invested_roi",
        "validation_folds": [
            {"start": start.date().isoformat(), "end": end.date().isoformat(), "days": int(((validation_dates >= start) & (validation_dates <= end)).sum())}
            for start, end in folds
        ],
        "selected_constraint_reasons": list(top.get("constraint_reasons", [])) if top else ["没有完整正超额邻域"],
        "platforms": platforms,
        "computation": {
            "training_evaluations": len(training_rows),
            "validation_evaluations": len(validation_rows),
            "stress_evaluations": len(stress_coords),
            "cost_evaluations": len(training_rows) + len(validation_rows) + len(stress_coords),
        },
    }
    neighborhoods = {
        tuple(region["center_coordinate"]): [tuple(point) for point in region["region_points"]]
        for region in regions
    }
    public_ranked = []
    for region in recommendations:
        row = rows[tuple(region["center_coordinate"])]
        public_ranked.append(_public_row({**row, "region": _public_region(region), "selection_score": region["mean_excess_return"]}))
    return {
        "status": "completed",
        "recommendation_status": "qualified" if recommendations else "no_qualified",
        "params": selected,
        "recommended_params": deepcopy(top["center_params"]) if top else None,
        "as_of": as_of.date().isoformat() if as_of is not None else None,
        "score": float(top["mean_excess_return"]) if top else (float(diagnostic["validation_excess_original"]) if diagnostic else None),
        "raw_score": float(top["center_excess_return"]) if top else (float(diagnostic["validation_excess_original"]) if diagnostic else None),
        "eligible": bool(recommendations),
        "reasons": list(dict.fromkeys(reasons)),
        "candidates": public_ranked,
        "candidate_archive": public_candidates,
        "search_candidate_archive": [_public_row(row) for row in training_rows],
        "trial_archive": [_public_row(row) for row in training_rows],
        "recommendation_regions": public_regions,
        "diagnostics": diagnostics,
        "_ranked_rows": public_ranked,
        "_family_rows": validation_rows,
        "_neighborhoods": neighborhoods,
        "_search_rows": training_rows,
        "_fold_ranges": list(folds),
        "_selection_data": data,
        "_selection_dates": history,
    }


def estimate_search(config: OptimizationConfig, data: DataBundle) -> dict[str, Any]:
    if int(config.protocol_version) != 4:
        raise ValueError("v4 preflight requires protocol_version=4")
    if config.base.grid_mode != "geometric":
        raise ValueError("v4 参数研究只支持等比网格")
    if config.strategy.strategy_version == 4:
        from ..research.robust import estimate_robust_search
        return estimate_robust_search(config, data)
    space = ParameterSpace(config)
    movable = _movable_fields(space, config)
    dates = pd.DatetimeIndex(data.calendar)
    eval_start = pd.Timestamp(config.base.start_date).normalize()
    asof = scientific._known_preceding_close(data, eval_start)
    reasons: list[str] = []
    if config.strategy.strategy_version == 3:
        from .window import MIN_TRAINING_OFFSET, training_window
        window = training_window(data, asof)
        total = space.size()
        method = _method(config, total)
        edge = int(config.strategy.hypercube_edge)
        box_volume = edge ** len(space.active) if space.active else 1
        impossible = [name for name in space.active if space.dimensions[name].count < edge]
        if len(window.dates) < MIN_TRAINING_OFFSET:
            status = "insufficient_history"
            reasons = [*reasons, f"训练时段只有 {len(window.dates)} 个交易日，少于最低 {MIN_TRAINING_OFFSET} 日"]
        elif impossible:
            status = "insufficient_range"
            reasons = [*reasons, "超正方体边长超出这些参数的有效范围：" + "、".join(impossible)]
        elif box_volume > int(config.strategy.validation_budget):
            status = "budget_exceeded"
            reasons = [*reasons, f"一个超正方体需要 {box_volume} 个评价点，超过验证预算 {config.strategy.validation_budget}"]
        elif method == "exhaustive" and total > int(config.strategy.training_budget):
            status = "budget_exceeded"
            reasons = [*reasons, f"完整枚举需要 {total} 个训练点，超过训练预算 {config.strategy.training_budget}"]
        else:
            status = "ready"
        evaluation_days = int(len(dates[dates >= eval_start]))
        fixed_count = 1 if config.selection_target in ("fixed", "both") else 0
        adaptive_count = math.ceil(evaluation_days / max(1, int(config.rebalance_days))) if config.selection_target in ("adaptive", "both") else 0
        return {
            "status": status, "protocol_version": 4, "strategy_id": config.strategy.strategy_id,
            "strategy_version": 3, "geometry": "hypercube_mean", "connectivity": "fixed_edge_cube",
            "budget_role": "training_sample_and_box_cap", "expand_until_closed": False,
            "grid_mode": "geometric", "ranking_mode": "fixed_mean", "hypercube_edge": edge,
            "hypercube_volume": int(box_volume), "active_dimensions": list(space.active),
            "granularity": dict(space.granularity), "neighborhood_fields": list(space.active),
            "fields": list(space.fields), "active_fields": list(space.active),
            "units": {**space.granularity},
            "effective_bounds": {
                name: {
                    "low_ticks": dim.low, "high_ticks": dim.high,
                    "low": space.value(name, dim.low), "high": space.value(name, dim.high),
                    "fixed": dim.fixed,
                }
                for name, dim in space.dimensions.items()
            },
            "legal_combination_count": int(total), "search_method": method,
            "training_budget": int(config.strategy.training_budget),
            "validation_budget": int(config.strategy.validation_budget),
            "minimum_neighborhood_size": int(box_volume),
            "minimum_neighborhood_possible": not impossible,
            "decision_count_estimate": int(fixed_count + adaptive_count + 1),
            "history_days": int(window.days), "training_days": int(window.days),
            "validation_days": 0, "full_504_day_history": bool(window.days >= MIN_TRAINING_OFFSET),
            "full_training_window": bool(window.days >= MIN_TRAINING_OFFSET),
            "training_window": window.diagnostic(),
            "slippage_mode": config.strategy.slippage_mode,
            "reasons": list(dict.fromkeys(reasons)),
        }
    history = _history(data, asof)
    generation, folds, split, reasons = scientific._folds_for_history(history)
    total = space.size()
    method = _method(config, total)
    if config.strategy.strategy_version == 2:
        minimum_neighborhood = 1
        neighborhood_possible = total > 0
    elif config.strategy.ranking_mode == "fixed_mean" and movable:
        radii = {
            name: value_to_ticks(name, config.strategy.fixed_half_widths.get(name, UNITS[name]))
            for name in movable
        }
        minimum_neighborhood = math.prod(2 * radii[name] + 1 for name in movable)
        neighborhood_possible = all(2 * radii[name] + 1 <= space.dimensions[name].count for name in movable)
    else:
        minimum_neighborhood = space.min_box_size(movable)
        neighborhood_possible = all(space.dimensions[name].count >= 3 for name in movable) if movable else False
    if not neighborhood_possible:
        reasons = [*reasons, "有效搜索范围无法容纳完整对称最小邻域；只能产生单点诊断"]
    if method == "exhaustive" and (
        total > int(config.strategy.training_budget) or total > int(config.strategy.validation_budget)
    ):
        status = "budget_exceeded"
        reasons = [*reasons, f"完整枚举需要 {total} 个参数点，超过训练或验证预算"]
    elif len(history) < 504 or not folds:
        status = "insufficient_history"
        reasons = [*reasons, "正式推荐需要完整 504 日历史"]
    else:
        status = "ready"
    evaluation_days = int(len(dates[dates >= eval_start]))
    fixed_count = 1 if config.selection_target in ("fixed", "both") else 0
    adaptive_count = math.ceil(evaluation_days / max(1, int(config.rebalance_days))) if config.selection_target in ("adaptive", "both") else 0
    decision_count = fixed_count + adaptive_count + 1
    return {
        "status": status,
        "protocol_version": 4,
        "strategy_id": config.strategy.strategy_id,
        "strategy_version": config.strategy.strategy_version,
        "geometry": "connected_lattice" if config.strategy.strategy_version == 2 else "symmetric_box",
        "budget_role": "seed_discovery" if config.strategy.strategy_version == 2 else "evaluation_cap",
        "expand_until_closed": config.strategy.strategy_version == 2,
        "grid_mode": "geometric",
        "ranking_mode": config.strategy.ranking_mode,
        "neighborhood_fields": list(movable),
        "fields": list(space.fields),
        "active_fields": list(space.active),
        "units": {**UNITS, "shares": 100},
        "effective_bounds": {
            name: {
                "low_ticks": dim.low,
                "high_ticks": dim.high,
                "low": dim.low * dim.unit if name in {"buy_step", "sell_step"} else dim.low * 100,
                "high": dim.high * dim.unit if name in {"buy_step", "sell_step"} else dim.high * 100,
                "fixed": dim.fixed,
            }
            for name, dim in space.dimensions.items()
        },
        "legal_combination_count": int(total),
        "search_method": method,
        "training_budget": int(config.strategy.training_budget),
        "validation_budget": int(config.strategy.validation_budget),
        "minimum_neighborhood_size": int(minimum_neighborhood),
        "minimum_neighborhood_possible": bool(neighborhood_possible),
        "decision_count_estimate": int(decision_count),
        "history_days": int(len(history)),
        "training_days": int(len(generation)),
        "validation_days": int(sum(len(history[(history >= start) & (history <= end)]) for start, end in folds)),
        "full_504_day_history": bool(len(history) >= 504 and split),
        "slippage_mode": config.strategy.slippage_mode,
        "reasons": list(dict.fromkeys(reasons)),
    }


def strategy_descriptions() -> dict[str, Any]:
    return {
        "strategies": [{
            "strategy_id": "geometric_robust_j",
            "strategy_version": 4,
            "label": "等比网格四维稳健 J 峰值研究",
            "grid_mode": "geometric",
            "fields": ["buy_step", "sell_step", "buy_shares", "sell_shares"],
            "granularity": {**GRANULARITY_DEFAULTS},
            "max_shares_mode": "unlimited",
            "neighborhood_fields": ["buy_step", "sell_step", "buy_shares", "sell_shares"],
            "search_methods": ["auto", "hierarchical", "exhaustive"],
            "ranking_modes": ["robust_j"],
            "default_ranking_mode": "robust_j",
            "defaults": {
                "training_budget": 20_000, "validation_budget": 100_000,
                "recommendation_count": 10, "hypercube_edge": 3,
                "sigma": 1.0, "alpha": .30, "beta": 1.20, "tau": .01,
                "eta_limit": 2.0, "eta_hard_cap": 3.5, "gamma_max": .40,
                "eta_floor": 0.0, "gamma_min": 0.0, "floor_k_ref": 10.0,
            },
        }, {
            "strategy_id": "geometric_excess_neighborhood",
            "strategy_version": 3,
            "label": "等比网格超正方体平均超额收益",
            "grid_mode": "geometric",
            "fields": list(FIELDS),
            "granularity": {**UNITS},
            "search_methods": ["auto", "hierarchical", "exhaustive"],
            "ranking_modes": ["fixed_mean"],
            "default_ranking_mode": "fixed_mean",
            "defaults": {
                "training_budget": 20_000,
                "validation_budget": 100_000,
                "recommendation_count": 10,
                "hypercube_edge": 3,
                "max_shares_mode": "search",
                "slippage_mode": "configured",
                "neighborhood_fields": ["buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares"],
                "granularity": {**UNITS},
                "fixed_half_widths": {
                    "buy_step": 0.001, "sell_step": 0.001, "sell_shares": 100, "buy_shares": 100,
                    "min_shares": 500, "max_shares": 500,
                },
            },
        }]
    }
