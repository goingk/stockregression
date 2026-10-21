"""Explicit, resumable completion of archived research neighbourhoods."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

import pandas as pd

from .models import BacktestConfig, OptimizationConfig
from .numerics.robust import NeighborhoodOperator, robust_score
from .reproducibility import computation_provenance, reproducibility_mismatches
from .research.robust import EvaluationBudget, _public_row
from .research_preparation import preparation_config
from .research_v4.evaluator import BatchEvaluator
from .research_v4.space import ROBUST_FIELDS, ParameterSpace
from .research_v4.window import TrainingWindowPolicy
from .share_engine import run_share_backtest
from .settings import PROJECT_ROOT
from .storage import Store, _write_candidate_archive, dumps


Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


def _key(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _checkpoint_key(*, result_id: str, data_version: str | None, decision: str,
                    decision_index: int, provenance: dict[str, Any],
                    checkpoint_id: str | None = None) -> str:
    """Build the checkpoint identity of one completion run.

    The checkpoint id separates the runs of the same decision: a fresh
    submission starts a new checkpoint, while a resume reuses the checkpoint
    of the original job.
    """
    payload = {
        "result_id": result_id, "data_version": data_version,
        "decision": decision, "decision_index": int(decision_index),
        "checkpoint_id": str(checkpoint_id or ""),
        "source_fingerprint": provenance.get("source_fingerprint"),
        "dependency_lock_hash": provenance.get("dependency_lock_hash"),
        "configuration_fingerprint": provenance.get("configuration_fingerprint"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def _emit(progress: Progress | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, fraction)), message)


def _failed_row(space: ParameterSpace, coordinate: tuple[int, ...], reason: str) -> dict[str, Any]:
    return {
        "params": space.params(coordinate), "key": coordinate, "coordinate": list(coordinate),
        "evaluation_status": "failed", "training_excess_return": math.nan,
        "max_principal_drawdown": math.inf, "failure_reason": reason,
    }


def complete_research_result(
    store: Store,
    service,
    result_id: str,
    *,
    decision: str,
    decision_index: int,
    reporting_budget: int,
    target: str = "return_leader",
    checkpoint_id: str | None = None,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> dict[str, Any]:
    result = store.result(result_id)
    if result is None:
        raise LookupError("结果不存在")
    if result.get("protocol_version") != 5 or result.get("result_schema_version") != 2:
        raise ValueError("该历史版本不支持补算")
    stored_provenance = result.get("computation_provenance") or {}
    current = computation_provenance(PROJECT_ROOT, configuration=result.get("config"))
    mismatches = reproducibility_mismatches(stored_provenance.get("finished"), current)
    if stored_provenance.get("reproducible") is False or mismatches:
        raise ValueError("计算来源或依赖版本不一致，拒绝混合旧候选与当前引擎：" + "；".join(mismatches))
    if not stored_provenance.get("completion_allowed"):
        raise ValueError("原始结果缺少可复现的计算来源指纹，无法补算")
    version = result.get("data_version")
    if not version or store.snapshot(result.get("symbol") or result.get("config", {}).get("base", {}).get("symbol"), version) is None:
        raise ValueError("原始行情数据版本不可用，无法补算")
    config = OptimizationConfig.model_validate(result.get("config"))
    group = "search_candidate_archives"
    base_archive_ref = store.candidate_reference(result, group, decision, decision_index)
    archive_stats = store.candidate_stats(result, group, decision, decision_index)
    if not archive_stats.get("total"):
        raise LookupError("研究决策不存在或候选档案为空")
    base_values = config.base.model_dump(mode="python")
    base_values.update({"schema_version": 3, "grid_mode": "geometric", "min_shares": 0, "max_shares": None})
    base = BacktestConfig.model_validate(base_values)
    normalized = config.model_copy(update={"base": base})
    space = ParameterSpace(normalized, field_order=ROBUST_FIELDS)
    checkpoint_key = _checkpoint_key(
        result_id=result_id, data_version=version, decision=decision,
        decision_index=decision_index, provenance=current,
        checkpoint_id=checkpoint_id,
    )
    checkpoint = store.completion_checkpoint(checkpoint_key) or {}
    consumed = int(checkpoint.get("consumed", 0) or 0)
    # The rows produced by the previous run are restored before the reserved
    # points are turned into failures, so the recovery keeps its evidence.
    new_rows = [row for row in checkpoint.get("new_rows") or [] if isinstance(row, dict)]
    processed = {_key(value) for value in checkpoint.get("processed", []) if isinstance(value, (list, tuple))}
    in_flight = {
        _key(value) for value in checkpoint.get("in_flight", [])
        if isinstance(value, (list, tuple))
    }
    frozen_targets = checkpoint.get("targets")
    rows: dict[tuple[int, ...], dict[str, Any]] = {}
    if not frozen_targets:
        target_rows = store.candidate_archive(
            result, group, decision, decision_index,
            sort_by="training_excess_return", descending=True,
            limit=max(1, int(config.strategy.recommendation_count)),
        )["rows"]
        frozen_targets = [list(space.coordinate(row["params"])) for row in target_rows]
    else:
        target_rows = []
    targets = [_key(value) for value in frozen_targets]
    operator = NeighborhoodOperator(space, int(config.strategy.hypercube_edge) // 2)
    # Warm exactly the cached neighborhoods around the frozen targets.  A
    # complete cached neighbourhood is a free lookup and must not be charged as
    # a new evaluation merely because the main result stores it in SQLite.
    for target in targets:
        center_params = space.params(target)
        radius = int(config.strategy.hypercube_edge) // 2
        choices = {}
        for name in ROBUST_FIELDS:
            value = center_params.get(name)
            if not isinstance(value, (int, float)):
                continue
            unit = float(space.dimensions[name].unit)
            tick = int(round(float(value) / unit)) if unit else 0
            choices[name] = [
                space.value(name, tick + offset)
                for offset in range(-radius, radius + 1)
                if space.dimensions[name].low <= tick + offset <= space.dimensions[name].high
            ]
        for row in store.candidate_archive(
            result, group, decision, decision_index,
            in_filters=choices, sort_by="ordinal", limit=max(1, int(config.strategy.hypercube_edge) ** 4 + 8),
        )["rows"]:
            if isinstance(row.get("params"), dict):
                rows[space.coordinate(row["params"])] = row
    for row in [*target_rows, *(checkpoint.get("new_rows") or [])]:
        if isinstance(row, dict) and isinstance(row.get("params"), dict):
            rows[space.coordinate(row["params"])] = row
    # A previous process may have reserved a batch and exited before producing
    # results.  The reservation remains charged, and the started points become
    # explicit failures so a retry cannot spend the same budget again.
    for point in sorted(in_flight - processed):
        rows[point] = _failed_row(space, point, "补算进程在评价完成前中断")
        new_rows.append(rows[point])
        processed.add(point)
    all_required: list[tuple[int, ...]] = []
    for point in targets:
        for neighbor in operator.points(point):
            if neighbor not in all_required:
                all_required.append(neighbor)
    required = [point for point in all_required if point not in rows and point not in processed]
    target_rows = [rows[point] for point in targets if point in rows]
    if not target_rows:
        raise LookupError("补算目标在当前候选档案中不存在")
    if decision == "fixed":
        diagnostics = (result.get("fixed") or {}).get("diagnostics") or {}
    elif decision == "current":
        diagnostics = ((result.get("fixed") or {}).get("current") or {}).get("diagnostics") or {}
    else:
        decisions = ((result.get("evidence") or {}).get("decisions") or [])
        diagnostics = (decisions[decision_index].get("diagnostics") or {}) if decision_index < len(decisions) else {}
    start = diagnostics.get("training_start")
    end = diagnostics.get("training_end")
    if not start or not end:
        raise ValueError("原始结果缺少训练时段范围")
    prepared = preparation_config(store, base, version, int(config.strategy.training_max_days))
    bundle = service.load_bundle(prepared, version)
    policy = TrainingWindowPolicy(
        int(config.strategy.training_min_days), int(config.strategy.training_max_days),
        mode=str(config.strategy.training_window_mode),
    )
    recomputed_window = policy.window(bundle, pd.Timestamp(end))
    stored_window = diagnostics.get("training_window") or {}
    if recomputed_window.data_issue:
        raise ValueError("原结果的训练窗口无法按当前数据版本复现：" + recomputed_window.data_issue)
    if (
        str(recomputed_window.start.date()) if recomputed_window.start is not None else None
    ) != str(start) or (
        str(recomputed_window.end.date()) if recomputed_window.end is not None else None
    ) != str(end):
        raise ValueError("原结果训练窗口与冻结的窗口策略不一致，拒绝补算")
    if stored_window and stored_window.get("action_factor_fingerprint") not in (
        None, recomputed_window.diagnostic().get("action_factor_fingerprint")
    ):
        raise ValueError("原结果公司行动转换指纹不一致，拒绝补算")
    dates = pd.DatetimeIndex(bundle.calendar)
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if not len(dates):
        raise ValueError("原始数据版本缺少训练时段行情")
    budget = EvaluationBudget(max(0, int(reporting_budget)), max(1, int(config.strategy.max_runtime_seconds)))
    if budget.total < consumed:
        raise ValueError("恢复后的补算预算低于已消耗预算")
    budget.consumed = consumed
    evaluator = BatchEvaluator(
        normalized, base, bundle, space, dates,
        cancelled=cancelled, allow_odd_lot_exit=True,
    )
    chunks = max(1, int(evaluator.chunk_size))
    total_required = len(all_required)
    evaluated = 0
    status = "completed"
    stop_reason = None
    for begin in range(0, len(required), chunks):
        if cancelled and cancelled():
            status, stop_reason = "cancelled", "用户取消，已保存补算检查点"
            break
        chunk = required[begin:min(len(required), begin + chunks)]
        remaining = budget.remaining
        if remaining <= 0:
            status, stop_reason = "budget_exhausted", budget.reason or "展示补算预算已用尽"
            break
        chunk = chunk[:remaining]
        if not chunk:
            break
        reserved = budget.reserve(len(chunk), "展示补算")
        if not reserved:
            status, stop_reason = "budget_exhausted", budget.reason or "展示补算预算已用尽"
            break
        store.save_completion_checkpoint(checkpoint_key, {
            "result_id": result_id, "decision": decision, "decision_index": decision_index,
            "checkpoint_id": str(checkpoint_id or ""),
            "target": target, "data_version": version, "targets": [list(point) for point in targets],
            "consumed": budget.consumed, "processed": [list(point) for point in sorted(processed)],
            "in_flight": [list(point) for point in chunk], "new_rows": new_rows,
            "status": "running", "stop_reason": None,
            "original_archive_count": int(archive_stats.get("total", 0)), "required_count": total_required,
            "remaining_count": max(0, total_required - sum(point in rows for point in all_required)),
        })
        try:
            evaluated_rows = evaluator.evaluate_training(chunk)
        except InterruptedError:
            for point in chunk:
                rows[point] = _failed_row(space, point, "补算在评价完成前取消")
                new_rows.append(rows[point])
                processed.add(point)
            store.save_completion_checkpoint(checkpoint_key, {
                "result_id": result_id, "decision": decision, "decision_index": decision_index,
                "checkpoint_id": str(checkpoint_id or ""),
                "target": target, "data_version": version, "targets": [list(point) for point in targets],
                "consumed": budget.consumed, "processed": [list(point) for point in sorted(processed)],
                "in_flight": [], "new_rows": new_rows, "status": "cancelled",
                "stop_reason": "用户取消，已保存补算检查点",
                "original_archive_count": int(archive_stats.get("total", 0)),
                "required_count": total_required,
                "remaining_count": max(0, total_required - sum(point in rows for point in all_required)),
            })
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for point in chunk:
                rows[point] = _failed_row(space, point, reason)
                new_rows.append(rows[point])
                processed.add(point)
            status, stop_reason = "failed", reason
            store.save_completion_checkpoint(checkpoint_key, {
                "result_id": result_id, "decision": decision, "decision_index": decision_index,
                "checkpoint_id": str(checkpoint_id or ""),
                "target": target, "data_version": version, "targets": [list(point) for point in targets],
                "consumed": budget.consumed, "processed": [list(point) for point in sorted(processed)],
                "in_flight": [], "new_rows": new_rows, "status": "failed", "stop_reason": reason,
                "original_archive_count": int(archive_stats.get("total", 0)),
                "required_count": total_required,
                "remaining_count": max(0, total_required - sum(point in rows for point in all_required)),
            })
            break
        for row in evaluated_rows:
            if isinstance(row.get("params"), dict):
                coordinate = space.coordinate(row["params"])
                rows[coordinate] = row
                new_rows.append(row)
        processed.update(chunk)
        evaluated += len(chunk)
        completed_now = sum(
            point in rows and rows[point].get("evaluation_status") == "completed"
            for point in all_required
        )
        store.save_completion_checkpoint(checkpoint_key, {
            "result_id": result_id, "decision": decision, "decision_index": decision_index,
            "checkpoint_id": str(checkpoint_id or ""),
            "target": target, "data_version": version, "targets": [list(point) for point in targets],
            "consumed": budget.consumed, "processed": [list(point) for point in sorted(processed)],
            "in_flight": [], "new_rows": new_rows, "status": "running", "stop_reason": None,
            "original_archive_count": int(archive_stats.get("total", 0)), "required_count": total_required,
            "remaining_count": max(0, total_required - completed_now),
        })
        _emit(progress, .1 + .8 * (begin + len(chunk)) / max(1, len(required)), "保存补算候选批次")
    overlay_coordinates = sorted(set(processed) | set(targets))
    overlay = [_public_row(rows[point], robust_score(point, rows, space, config.strategy))
               for point in overlay_coordinates if point in rows]
    archive_ref = _write_candidate_archive(store.root, overlay) if overlay else None
    merged_ref = None
    if base_archive_ref is not None and archive_ref is not None:
        merged_ref = store.merge_candidate_archives(base_archive_ref, archive_ref)
    completed_required = sum(
        point in rows and rows[point].get("evaluation_status") == "completed"
        for point in all_required
    )
    if status == "completed" and completed_required < total_required:
        status = "partial"
        stop_reason = stop_reason or "部分候选评价失败，已保留失败证据"
    checkpoint_payload = {
        "result_id": result_id, "decision": decision, "decision_index": decision_index,
        "checkpoint_id": str(checkpoint_id or ""),
        "target": target, "data_version": version, "targets": [list(point) for point in targets],
        "consumed": budget.consumed, "processed": [list(point) for point in sorted(processed)],
        "in_flight": [], "new_rows": new_rows, "status": status, "stop_reason": stop_reason,
        "original_archive_count": int(archive_stats.get("total", 0)), "required_count": total_required,
        "remaining_count": max(0, total_required - completed_required),
    }
    store.save_completion_checkpoint(checkpoint_key, checkpoint_payload)
    revision = store.save_completion_revision(result_id, {
        **{key: value for key, value in checkpoint_payload.items() if key not in {"new_rows", "processed"}},
        "candidate_archive": archive_ref, "merged_candidate_archive": merged_ref,
        "completed_count": completed_required,
        "remaining_count": max(0, total_required - completed_required),
        "provenance": current,
    })
    if status == "cancelled":
        cancelled_error = InterruptedError(stop_reason or "补算已取消")
        # A cancellation between batches still publishes the evidence already
        # produced.  Carry the revision through the exception so the job can
        # point at that exact revision instead of whatever is newest later.
        cancelled_error.completion_revision = revision["revision"]
        raise cancelled_error
    _emit(progress, 1.0, "补算修订已保存")
    return {
        "status": "completed" if status == "completed" else "partial",
        "result_id": result_id, "completion_revision": revision["revision"],
        "decision": decision, "decision_index": decision_index,
        "evaluated_count": evaluated, "consumed": budget.consumed,
        "remaining_count": max(0, total_required - completed_required),
        "stop_reason": stop_reason, "candidate_archive": archive_ref,
        "merged_candidate_archive": merged_ref,
        "protocol_version": 5, "result_schema_version": 2,
    }
