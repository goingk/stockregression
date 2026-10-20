"""Strategy v3: rank fixed-size hypercubes by mean training excess return.

The historical sample is one training block.  It is never split into a
training and an internal-validation period.  Every formal score is the mean
grid-minus-buy-and-hold excess over all legal points in a hypercube of the
configured edge length.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import time

from .. import scientific
from ..performance import record_count, record_seconds
from .evaluator import BatchEvaluator, center_constraints
from .search import box_coordinates, initial_training_coordinates, refine_coordinates, statistical_coordinates
from .space import FIELDS, ParameterSpace
from .window import MIN_TRAINING_OFFSET, training_window


def _fit_center(space: ParameterSpace, coordinate: tuple[int, ...], edge: int) -> tuple[int, ...] | None:
    radius = edge // 2
    values = list(coordinate)
    for name in space.active:
        index = FIELDS.index(name)
        dim = space.dimensions[name]
        low, high = dim.low + radius, dim.high - radius
        if low > high:
            return None
        values[index] = min(high, max(low, values[index]))
    if "min_shares" in space.dimensions and "max_shares" in space.dimensions:
        min_index, max_index = FIELDS.index("min_shares"), FIELDS.index("max_shares")
        if values[min_index] > values[max_index]:
            return None
    return tuple(values)


def _complete(row: dict | None) -> bool:
    if not row or row.get("evaluation_status") != "completed":
        return False
    value = row.get("training_excess_return")
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _box_key(points: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(points))


def _region_key(region: dict) -> tuple:
    return (
        -float(region.get("mean_excess_return", -math.inf)),
        -float(region.get("min_excess_return", -math.inf)),
        -float(region.get("center_excess_return", -math.inf)),
        tuple(region.get("center_coordinate", ())),
    )


def _representative(row: dict) -> dict:
    return {
        "candidate_id": row.get("candidate_id"),
        "params": dict(row.get("params", {})),
        "coordinate": list(row.get("key", ())),
        "excess_return": row.get("training_excess_return"),
        "stability_margin": None,
        "max_drawdown": row.get("training_drawdown"),
        "trade_count": row.get("training_trade_count"),
        "constraint_eligible": bool(row.get("constraint_eligible")),
        "constraint_reasons": list(row.get("constraint_reasons", [])),
    }


def select_hypercube_at(config, base, data, asof, *, seed_offset=0, progress=None, cancelled=None,
                        progress_start=0.0, progress_width=1.0):
    from .selector import _empty_result, _method, _public_region, _public_row

    if config.protocol_version != 4 or base.schema_version != 3 or base.grid_mode != "geometric":
        raise ValueError("超正方体平均超额收益策略需要 protocol_version=4、schema_version=3 和等比网格")
    window = training_window(data, asof)
    as_of = window.end
    space = ParameterSpace(config)
    active = space.active
    edge = int(config.strategy.hypercube_edge)
    radius = edge // 2
    if window.days < MIN_TRAINING_OFFSET:
        return _empty_result(
            config, space, as_of,
            [window.reason, f"训练时段只有 {window.days} 个交易日，少于最低 {MIN_TRAINING_OFFSET} 日"],
            window.dates,
        )
    impossible = [name for name in active if space.dimensions[name].count < edge]
    if impossible:
        labels = "、".join(name for name in impossible)
        return _empty_result(
            config, space, as_of,
            [f"超正方体边长 {edge} 超出有效研究范围：{labels}"],
            window.dates,
        )
    box_volume = edge ** len(active) if active else 1
    if box_volume > int(config.strategy.validation_budget):
        raise ValueError(
            f"一个超正方体需要 {box_volume} 个评价点，超过验证预算 "
            f"{config.strategy.validation_budget}；请减小边长、增大细粒度或提高预算"
        )
    total = space.size()
    method = _method(config, total)
    if method == "exhaustive" and total > int(config.strategy.training_budget):
        raise ValueError(
            f"完整枚举需要 {total} 个训练点，超过训练预算 {config.strategy.training_budget}"
        )

    def emit(fraction, message):
        if progress:
            progress(progress_start + progress_width * max(0.0, min(1.0, fraction)), message)

    emit(0.02, f"训练时段 {window.start.date() if window.start is not None else '—'} 至 "
              f"{window.end.date() if window.end is not None else '—'}（{window.days} 日，"
              f"{'同价格日' if window.matched else '500 日回退'}）")
    evaluator = BatchEvaluator(
        config, base, data, space, window.dates, window.dates,
        validation_folds=[], cancelled=cancelled,
    )
    evaluator.retain_daily_returns = False
    date_key = as_of.date().isoformat() if as_of is not None else ""

    started = time.perf_counter()
    if method == "exhaustive":
        center_coordinates = list(space.iterate(int(config.strategy.training_budget) + 1))
    else:
        center_coordinates = initial_training_coordinates(space, config, date_key)
        try:
            statistical = statistical_coordinates(space, scientific._statistical_profile(base, data, window.dates))
        except Exception:
            statistical = []
        if statistical:
            existing = set(center_coordinates)
            room = max(0, int(config.strategy.training_budget) - len(center_coordinates))
            additions = [item for item in statistical if item not in existing][:room]
            if not additions and len(center_coordinates) >= len(statistical):
                center_coordinates = center_coordinates[:-len(statistical)] + statistical
            else:
                center_coordinates.extend(additions)
    center_coordinates = list(dict.fromkeys(center_coordinates))[: int(config.strategy.training_budget)]
    center_rows: list[dict] = []
    try:
        for begin in range(0, len(center_coordinates), evaluator.chunk_size):
            chunk = center_coordinates[begin:begin + evaluator.chunk_size]
            center_rows.extend(evaluator.evaluate_training(chunk))
            emit(0.05 + 0.28 * min(1.0, (begin + len(chunk)) / max(1, len(center_coordinates))),
                 f"训练采样：{len(center_rows):,} 点")
        if method != "exhaustive":
            extra = refine_coordinates(
                space, center_rows, existing=set(center_coordinates),
                limit=max(0, int(config.strategy.training_budget) - len(center_rows)),
                recommendation_count=int(config.strategy.recommendation_count),
                movable_fields=active,
            )
            for begin in range(0, len(extra), evaluator.chunk_size):
                chunk = extra[begin:begin + evaluator.chunk_size]
                center_rows.extend(evaluator.evaluate_training(chunk))
                emit(0.33, f"训练细化：{len(center_rows):,} 点")
    except InterruptedError:
        emit(0.95, "训练采样已取消")
    record_seconds("hypercube_training", time.perf_counter() - started)
    record_count("hypercube_training_points", len(center_rows))
    center_rows = sorted({tuple(row["key"]): row for row in center_rows}.values(), key=lambda row: tuple(row["key"]))
    for index, row in enumerate(center_rows, 1):
        row["search_id"] = index
    completed_centers = sorted(
        (row for row in center_rows if _complete(row)),
        key=lambda row: (-float(row["training_excess_return"]), tuple(row["key"])),
    )
    if not completed_centers:
        return _empty_result(
            config, space, as_of,
            [*([window.reason] if window.reason else []), "训练时段没有完成的参数评价"],
            window.dates,
        )

    cache: dict[tuple[int, ...], dict] = {tuple(row["key"]): row for row in center_rows}
    box_limits = max(1, int(config.strategy.validation_budget) // max(1, box_volume))
    target_boxes = min(
        len(completed_centers),
        max(20, 4 * int(config.strategy.recommendation_count)),
        box_limits,
    )
    chosen: list[tuple[tuple[int, ...], list[tuple[int, ...]]]] = []
    seen_boxes: set[tuple[tuple[int, ...], ...]] = set()
    box_evaluated = 0
    box_started = time.perf_counter()
    emit(0.36, f"超正方体评价：边长 {edge}，每个最多 {box_volume:,} 点")
    try:
        for row in completed_centers:
            if len(chosen) >= target_boxes:
                break
            center = _fit_center(space, tuple(row["key"]), edge)
            if center is None:
                continue
            # Keep the candidate boxes distinguishable; overlapping boxes are
            # allowed only when they are not the same lattice cube.
            if any(
                all(abs(center[FIELDS.index(name)] - other[FIELDS.index(name)]) <= radius for name in active)
                for other, _ in chosen
            ):
                continue
            points = box_coordinates(
                space, center, {name: radius for name in active}, movable_fields=active,
            )
            if not points:
                continue
            key = _box_key(points)
            if key in seen_boxes:
                continue
            missing = [point for point in points if point not in cache]
            if box_evaluated + len(missing) > int(config.strategy.validation_budget):
                break
            if missing:
                for begin in range(0, len(missing), evaluator.chunk_size):
                    chunk = missing[begin:begin + evaluator.chunk_size]
                    for evaluated in evaluator.evaluate_training(chunk):
                        cache[tuple(evaluated["key"])] = evaluated
                box_evaluated += len(missing)
            if not all(_complete(cache.get(point)) for point in points):
                continue
            seen_boxes.add(key)
            chosen.append((center, points))
            emit(0.38 + 0.38 * min(1.0, len(chosen) / max(1, target_boxes)),
                 f"超正方体：{len(chosen):,} / {target_boxes:,} 个")
    except InterruptedError:
        emit(0.95, "超正方体评价已取消")
    record_seconds("hypercube_boxes", time.perf_counter() - box_started)
    record_count("hypercube_boxes", len(chosen))
    record_count("hypercube_points", len(cache))

    all_rows = sorted(cache.values(), key=lambda row: tuple(row["key"]))
    for index, row in enumerate(all_rows, 1):
        row["candidate_id"] = index
        row["coordinate"] = list(row["key"])
        row["validation_return_original"] = row.get("training_return")
        row["validation_excess_original"] = row.get("training_excess_return")
        row["grid_return"] = row.get("training_return")
        row["baseline_return"] = row.get("training_baseline_return")
        row["validation_return_stress"] = row.get("training_return_stress")
        row["validation_excess_stress"] = row.get("training_excess_stress")
        row["raw_score"] = row.get("training_excess_return")
        row["complete"] = _complete(row)
        row["is_center"] = False
        row["is_baseline"] = False
        eligible, reasons = center_constraints(row, config.max_drawdown, config.min_trades)
        row["constraint_eligible"] = eligible
        row["constraint_reasons"] = reasons
        row["eligible"] = bool(eligible and row["complete"])

    regions: list[dict] = []
    search_volume = math.prod(space.dimensions[name].count for name in active) if active else 1
    for center, points in chosen:
        rows = [cache[point] for point in points]
        values = [float(row["training_excess_return"]) for row in rows]
        if not values:
            continue
        center_row = cache[center]
        peak_row = max(rows, key=lambda row: (float(row["training_excess_return"]), tuple(-value for value in row["key"])))
        bounds = {}
        for name in space.dimensions:
            index = FIELDS.index(name)
            axis = [point[index] for point in points]
            bounds[name] = {
                "low": space.value(name, min(axis)),
                "high": space.value(name, max(axis)),
                "low_ticks": min(axis), "high_ticks": max(axis),
            }
        region = {
            "geometry": "hypercube_mean", "connectivity": "fixed_edge_cube",
            "scoring_basis": "training_mean_excess_return",
            "edge_length": edge, "radius_ticks": radius,
            "center_params": dict(center_row["params"]),
            "center_coordinate": list(center), "candidate_id": center_row["candidate_id"],
            "radii": {name: radius for name in active},
            "stable_point": _representative(center_row), "peak_point": _representative(peak_row),
            "combination_count": len(points), "completed_count": len(points),
            "raw_volume": len(points), "normalized_volume": len(points) / max(1, search_volume),
            "mean_excess_return": math.fsum(values) / len(values),
            "min_excess_return": min(values), "max_excess_return": max(values),
            "center_excess_return": float(center_row["training_excess_return"]),
            "complete": True, "boundary_closed": True, "maximum_search_complete": True,
            "global_exhaustive": bool(method == "exhaustive" and edge == 1 and len(cache) == total),
            "bounds": bounds, "bounds_are_envelope": False,
            "constraint_eligible": bool(center_row.get("constraint_eligible")),
            "constraint_reasons": list(center_row.get("constraint_reasons", [])),
            "region_points": [list(point) for point in points],
            "member_hash": hashlib.sha256(json.dumps([list(p) for p in sorted(points)]).encode()).hexdigest(),
            "stress": {},
        }
        regions.append(region)
    regions.sort(key=_region_key)
    for index, region in enumerate(regions, 1):
        region["region_id"] = index
        for point in region["region_points"]:
            row = cache.get(tuple(point))
            if row is not None:
                row.setdefault("region_ids", []).append(index)
        center_row = cache[tuple(region["center_coordinate"])]
        center_row["is_center"] = True

    # Double-cost stress is a diagnostic over the best few boxes only.
    stress_regions = regions[:min(3, len(regions))]
    stress_coordinates = sorted({tuple(point) for region in stress_regions for point in region["region_points"]})
    if stress_coordinates and not (cancelled and cancelled()):
        try:
            stress = evaluator.evaluate_training_stress(stress_coordinates)
            for point, evidence in stress.items():
                row = cache.get(point)
                if row is None:
                    continue
                row["training_excess_stress"] = evidence.get("training_excess_stress")
                row["training_return_stress"] = evidence.get("training_return_stress")
                row["validation_excess_stress"] = evidence.get("training_excess_stress")
                row["validation_return_stress"] = evidence.get("training_return_stress")
            for region in stress_regions:
                values = [
                    float(stress[tuple(point)]["training_excess_stress"])
                    for point in region["region_points"]
                    if tuple(point) in stress and math.isfinite(float(stress[tuple(point)].get("training_excess_stress", math.nan)))
                ]
                region["stress"] = {
                    "status": "completed" if len(values) == region["combination_count"] else "partial",
                    "mean_excess_return": math.fsum(values) / len(values) if values else None,
                    "min_excess_return": min(values) if values else None,
                }
        except InterruptedError:
            emit(0.95, "压力诊断已取消")
        except Exception as exc:
            for region in stress_regions:
                region["stress"] = {"status": "failed", "reason": str(exc)}

    top = regions[0] if regions else None
    displayed_regions = regions[:int(config.strategy.recommendation_count)]
    diagnostic = completed_centers[0]
    selected = deepcopy(top["center_params"]) if top else deepcopy(diagnostic["params"])
    top_center = cache.get(tuple(top["center_coordinate"])) if top else None
    qualified = bool(
        top is not None and float(top["mean_excess_return"]) > 0
        and top.get("constraint_eligible", False)
    )
    reasons = [window.reason]
    if not window.matched:
        reasons.append("未找到同价格日，已按 500 日回退")
    if method != "exhaustive":
        reasons.append("训练采样使用预算内分层搜索；超正方体评分覆盖所选中心的完整合法邻域")
    if top is None:
        reasons.append("没有形成可评价的完整超正方体")
    elif not qualified:
        reasons.append("最佳超正方体平均超额收益未为正或中心未满足约束，未授予推荐")
    public_regions = [_public_region(region) for region in displayed_regions]
    public_ranked = [
        _public_row({
            **cache[tuple(region["center_coordinate"])],
            "region": _public_region(region),
            "selection_score": region["mean_excess_return"],
        })
        for region in displayed_regions
    ]
    diagnostics = {
        "status": "cancelled" if (cancelled and cancelled()) else "completed",
        "protocol_version": 4, "strategy_id": config.strategy.strategy_id,
        "strategy_version": 3, "geometry": "hypercube_mean", "connectivity": "fixed_edge_cube",
        "scoring_basis": "training_only_mean_excess_return",
        "training_only": True, "constraints_role": "diagnostic_only",
        "grid_mode": "geometric", "search_method": method,
        "ranking_mode": "fixed_mean", "hypercube_edge": edge, "radius_ticks": radius,
        "hypercube_volume": box_volume, "active_dimensions": list(active),
        "granularity": dict(space.granularity),
        "units": {name: float(space.granularity[name]) for name in FIELDS},
        "training_budget": int(config.strategy.training_budget),
        "validation_budget": int(config.strategy.validation_budget),
        "legal_combination_count": total, "training_unique_count": len(center_rows),
        "validation_unique_count": len(all_rows), "candidate_count": len(all_rows),
        "box_count": len(displayed_regions), "evaluated_box_count": len(regions), "eligible_count": sum(
            1 for region in displayed_regions if region["mean_excess_return"] > 0 and region.get("constraint_eligible")
        ),
        "recommendation_count": len(displayed_regions),
        "recommendation_status": "qualified" if qualified else "no_qualified",
        "global_exhaustive": bool(method == "exhaustive" and edge == 1 and len(cache) == total),
        "coverage_status": "global_exhaustive" if (method == "exhaustive" and edge == 1 and len(cache) == total) else "training_sample_with_complete_boxes",
        "generation_evaluation_count": len(center_rows),
        "validation_evaluation_count": len(all_rows),
        "box_point_count": int(box_evaluated),
        "neighbor_count": sum(region["combination_count"] for region in regions),
        "selection_history_days": window.days,
        "selection_history_start": window.start.date().isoformat() if window.start is not None else None,
        "selection_history_end": window.end.date().isoformat() if window.end is not None else None,
        "training_start": window.start.date().isoformat() if window.start is not None else None,
        "training_end": window.end.date().isoformat() if window.end is not None else None,
        "training_window": window.diagnostic(),
        "validation_start": base.start_date.isoformat(),
        "validation_end": (base.end_date or base.start_date).isoformat(),
        "validation_account_mode": "fresh_initial_config",
        "validation_return_basis": "peak_invested_roi",
        "validation_folds": [],
        "selected_constraint_reasons": list(top.get("constraint_reasons", [])) if top else ["没有完整超正方体"],
        "computation": {
            "training_evaluations": len(center_rows), "box_evaluations": len(all_rows),
            "box_count": len(regions), "stress_evaluations": len(stress_coordinates),
            "cost_evaluations": len(center_rows) + len(all_rows) + len(stress_coordinates),
        },
    }
    emit(1.0, f"超正方体评分完成：{len(regions):,} 个候选")
    return {
        "status": diagnostics["status"],
        "recommendation_status": "qualified" if qualified else "no_qualified",
        "params": selected,
        "recommended_params": deepcopy(top["center_params"]) if qualified and top else None,
        "as_of": date_key, "score": top["mean_excess_return"] if top else diagnostic["training_excess_return"],
        "raw_score": top["center_excess_return"] if top else diagnostic["training_excess_return"],
        "eligible": qualified, "reasons": list(dict.fromkeys(reasons)),
        "candidates": public_ranked,
        "candidate_archive": [_public_row(row) for row in all_rows],
        "search_candidate_archive": [_public_row(row) for row in center_rows],
        "trial_archive": [_public_row(row) for row in center_rows],
        "recommendation_regions": public_regions,
        "diagnostics": diagnostics, "_ranked_rows": public_ranked,
        "_family_rows": all_rows,
        "_neighborhoods": {tuple(region["center_coordinate"]): region["region_points"] for region in regions},
        "_search_rows": center_rows, "_fold_ranges": [],
        "_selection_data": data, "_selection_dates": window.dates,
    }
