"""Strategy v2 orchestration; v1 remains available for historical box studies."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import time

import pandas as pd

from .. import scientific
from ..performance import record_count, record_seconds
from .connected import analyze_regions, completed, coordinate_hash, explore, positive, refine_training, seed_coordinates
from .evaluator import BatchEvaluator, center_constraints
from .search import initial_training_coordinates, statistical_coordinates
from .space import UNITS, ParameterSpace


def select_connected_at(config, base, data, asof, *, seed_offset=0, progress=None, cancelled=None,
                        progress_start=0.0, progress_width=1.0):
    from .selector import _empty_result, _history, _method, _movable_fields, _public_region, _public_row

    if config.protocol_version != 4 or base.schema_version != 3 or base.grid_mode != "geometric":
        raise ValueError("连通域策略需要 protocol_version=4、schema_version=3 和等比网格")
    history = _history(data, asof)
    as_of = history[-1] if len(history) else (pd.Timestamp(asof).normalize() if asof is not None else None)
    space = ParameterSpace(config)
    movable = _movable_fields(space, config)
    generation, folds, split, reasons = scientific._folds_for_history(history)
    # Strategy v2 formal recommendations always require the full 252+252 window.
    if len(history) < 504 or not folds or not len(generation):
        return _empty_result(config, space, as_of, [*reasons, "正式连通域推荐需要完整 504 日历史"], history)
    validation_dates = pd.DatetimeIndex([])
    for start, end in folds:
        validation_dates = validation_dates.append(history[(history >= start) & (history <= end)])
    validation_dates = validation_dates.unique().sort_values()
    total = space.size()
    method = _method(config, total)
    if method == "exhaustive" and (total > config.strategy.training_budget or total > config.strategy.validation_budget):
        raise ValueError(f"完整枚举需要 {total} 个参数点，超过训练或验证预算")

    def emit(fraction, message):
        if progress:
            progress(progress_start + progress_width * fraction, message)

    evaluator = BatchEvaluator(config, base, data, space, generation, validation_dates,
                               validation_folds=list(folds), cancelled=cancelled)
    evaluator.retain_daily_returns = method == "exhaustive"
    date_key = as_of.date().isoformat() if as_of is not None else ""
    started = time.perf_counter()
    training_coordinates = (list(space.iterate()) if method == "exhaustive"
                            else initial_training_coordinates(space, config, date_key))
    if method != "exhaustive":
        try:
            statistical = statistical_coordinates(space, scientific._statistical_profile(base, data, generation))
        except Exception:
            statistical = []
        training_coordinates = list(dict.fromkeys([*statistical, *training_coordinates]))[:config.strategy.training_budget]
    training_rows = []
    training_cancelled = False
    try:
        for begin in range(0, len(training_coordinates), evaluator.chunk_size):
            training_rows.extend(evaluator.evaluate_training(training_coordinates[begin:begin+evaluator.chunk_size]))
            emit(0.04 + 0.30 * min(1.0, (begin+evaluator.chunk_size)/max(1, len(training_coordinates))),
                 f"训练搜索：{len(training_rows):,} 点")
        if method != "exhaustive":
            extra = refine_training(space, training_rows, existing=set(training_coordinates),
                                    limit=max(0, config.strategy.training_budget-len(training_rows)),
                                    recommendation_count=config.strategy.recommendation_count, movable_fields=movable)
            for begin in range(0, len(extra), evaluator.chunk_size):
                training_rows.extend(evaluator.evaluate_training(extra[begin:begin+evaluator.chunk_size]))
                emit(0.35 + 0.10 * min(1.0, (begin+evaluator.chunk_size)/max(1, len(extra))),
                     f"多区域细化：共 {len(training_rows):,} 个训练点")
    except InterruptedError:
        training_cancelled = True
    record_seconds("connected_training", time.perf_counter()-started)
    record_count("connected_training_points", len(training_rows))
    training_rows.sort(key=lambda row: tuple(row["key"]))
    for index, row in enumerate(training_rows, 1):
        row["search_id"] = index
    seeds = (list(space.iterate()) if method == "exhaustive" else seed_coordinates(space, config, training_rows, date_key))
    if training_cancelled:
        from .connected import Exploration
        discovery = Exploration({}, "cancelled", len(seeds))
    else:
        discovery = explore(space, seeds, movable, evaluator.evaluate_validation,
                            chunk_size=evaluator.chunk_size, cancelled=cancelled,
                            progress=lambda message: emit(0.50, message))
    validation_rows = sorted(discovery.rows.values(), key=lambda row: tuple(row["key"]))
    for index, row in enumerate(validation_rows, 1):
        point = tuple(row["key"])
        row.update(candidate_id=index, coordinate=list(point), params=space.params(point),
                   is_center=False, is_baseline=False, is_stable_point=False, is_peak_point=False)
        eligible, constraint_reasons = center_constraints(row, config.max_drawdown, config.min_trades)
        row.update(constraint_eligible=eligible, constraint_reasons=constraint_reasons, eligible=positive(row))
    rows = {tuple(row["key"]): row for row in validation_rows}
    global_coverage = len(rows) == total and all(completed(row) for row in validation_rows)
    regions, incomplete = [], []
    emit(0.70, f"入口与边界回放结束：{len(rows):,} 点，开始连通分组与稳定距离计算")
    try:
        regions, incomplete = analyze_regions(space, rows, movable, global_exhaustive=global_coverage,
                                              cancelled=cancelled)
    except InterruptedError:
        discovery.status = "cancelled"
    if incomplete and discovery.status == "completed":
        discovery.status = "partial"
    # Abnormal termination never promotes the incomplete discovery to formal recommendations.
    recommendations = (regions[:config.strategy.recommendation_count]
                       if discovery.status in {"completed", "partial"} else [])
    stress_coordinates = sorted({p for region in recommendations for p in region["region_points"]})
    if stress_coordinates:
        evaluator.progress = lambda fraction, message: emit(0.85 + 0.12 * min(1.0, fraction), message)
        emit(0.85, f"压力诊断：{len(stress_coordinates):,} 个唯一域成员")
        started = time.perf_counter()
        try:
            stress = evaluator.evaluate_stress(stress_coordinates)
            for point, evidence in stress.items():
                rows[point]["validation_excess_stress"] = evidence.get("validation_excess_stress")
                rows[point]["validation_return_stress"] = evidence.get("validation_return_stress")
                rows[point]["stress_folds"] = evidence.get("folds", [])
            for region in recommendations:
                values = [stress[p]["validation_excess_stress"] for p in region["region_points"]
                          if p in stress and math.isfinite(float(stress[p].get("validation_excess_stress", math.nan)))]
                region["stress"] = {"status": "completed" if len(values) == region["combination_count"] else "partial",
                                    "mean_excess_return": math.fsum(values)/len(values) if values else None,
                                    "min_excess_return": min(values) if values else None}
        except InterruptedError:
            discovery.status = "cancelled"
            recommendations = []
        except Exception as exc:
            for region in recommendations:
                region["stress"] = {"status": "failed", "reason": str(exc)}
        record_seconds("connected_stress", time.perf_counter()-started)
        record_count("connected_stress_points", len(stress_coordinates))
    top = recommendations[0] if recommendations else None
    diagnostic = max((row for row in validation_rows if completed(row)),
                     key=lambda row: (row["validation_excess_original"], tuple(-v for v in row["key"])), default=None)
    fallback = space.params(space.full_coordinate(tuple(space.dimensions[name].low for name in space.active)))
    selected = deepcopy(top["stable_point"]["params"] if top else diagnostic["params"] if diagnostic else fallback)
    if method != "exhaustive":
        reasons = [*reasons, "入口预算限制发现范围；已发现正收益域完整扩张，不保证找到所有孤立区域"]
    if discovery.status != "completed":
        reasons = [*reasons, "探查被取消或存在失败点；只展示已绑定边界的完整参数域"]
    elif not top:
        reasons = [*reasons, "没有发现正超额收益连通域"]
    public_regions = [_public_region(region) for region in recommendations]
    public_ranked = [_public_row({**rows[tuple(region["center_coordinate"])], "region": _public_region(region),
                                  "selection_score": region["mean_excess_return"]}) for region in recommendations]
    outcome_hash = hashlib.sha256(json.dumps([
        [list(p), rows[p].get("evaluation_status"), rows[p].get("validation_excess_original") if completed(rows[p]) else None]
        for p in sorted(rows)], separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    diagnostics = {
        "status": discovery.status, "protocol_version": 4, "strategy_id": config.strategy.strategy_id,
        "strategy_version": 2, "geometry": "connected_lattice", "connectivity": "axis_one_tick",
        "stability_metric": "normalized_euclidean_to_inner_boundary", "constraints_role": "diagnostic_only",
        "grid_mode": "geometric", "search_method": method, "ranking_mode": "max_volume",
        "neighborhood_fields": list(movable), "units": dict(UNITS),
        "training_budget": config.strategy.training_budget, "validation_budget": config.strategy.validation_budget,
        "budget_role": "seed_discovery", "expand_until_closed": True,
        "legal_combination_count": total, "training_unique_count": len(training_rows),
        "validation_unique_count": len(rows), "validation_planned_count": len(seeds),
        "seed_count": len(seeds), "budget_overrun_count": max(0, len(rows)-len(seeds)),
        "retry_count": discovery.retry_count, "generation_evaluation_count": len(training_rows),
        "validation_evaluation_count": len(rows), "local_search_unique": len(rows),
        "candidate_count": len(rows), "eligible_count": len(recommendations),
        "region_count": len(regions), "closed_region_count": len(regions), "incomplete_regions": incomplete,
        "neighbor_count": sum(r["combination_count"] for r in regions),
        "recommendation_count": len(recommendations), "recommendation_status": "qualified" if top else "no_qualified",
        "global_exhaustive": global_coverage, "discovery_complete": discovery.status == "completed" and not incomplete,
        "coverage_status": "global_exhaustive" if global_coverage else "discovered_components_closed" if discovery.status == "completed" else "incomplete",
        "adaptive_validation_search": method != "exhaustive", "seed_plan_hash": coordinate_hash(seeds),
        "evaluation_set_hash": coordinate_hash(rows), "evaluation_trace_hash": outcome_hash,
        "selection_history_days": len(history), "selection_history_start": history[0].date().isoformat(),
        "selection_history_end": history[-1].date().isoformat(),
        "training_start": generation[0].date().isoformat(), "training_end": generation[-1].date().isoformat(),
        "validation_start": validation_dates[0].date().isoformat(), "validation_end": validation_dates[-1].date().isoformat(),
        "validation_account_mode": "fresh_initial_config", "validation_return_basis": "peak_invested_roi",
        "validation_folds": [{"start": start.date().isoformat(), "end": end.date().isoformat()} for start, end in folds],
        "selected_constraint_reasons": list(top["stable_point"]["constraint_reasons"]) if top else [],
        "computation": {"training_evaluations": len(training_rows), "validation_evaluations": len(rows),
                        "stress_evaluations": len(stress_coordinates), "retry_evaluations": discovery.retry_count,
                        "cost_evaluations": len(training_rows)+len(rows)+len(stress_coordinates)+discovery.retry_count},
    }
    emit(1.0 if discovery.status == "completed" else 0.95,
         f"正收益连通域探查{ '完成' if discovery.status == 'completed' else '未完成'}：{len(regions):,} 片闭合域")
    return {
        "status": discovery.status, "recommendation_status": "qualified" if top else "no_qualified",
        "params": selected, "recommended_params": deepcopy(top["stable_point"]["params"]) if top else None,
        "as_of": date_key, "score": top["mean_excess_return"] if top else None,
        "raw_score": top["center_excess_return"] if top else None, "eligible": bool(top),
        "reasons": list(dict.fromkeys(reasons)), "candidates": public_ranked,
        "candidate_archive": [_public_row(row) for row in validation_rows],
        "search_candidate_archive": [_public_row(row) for row in training_rows],
        "trial_archive": [_public_row(row) for row in training_rows], "recommendation_regions": public_regions,
        "diagnostics": diagnostics, "_ranked_rows": public_ranked, "_family_rows": validation_rows,
        "_neighborhoods": {tuple(r["center_coordinate"]): r["region_points"] for r in regions},
        "_search_rows": training_rows, "_fold_ranges": list(folds), "_selection_data": data, "_selection_dates": history,
    }
