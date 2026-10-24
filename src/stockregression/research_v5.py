"""Protocol-v5 research orchestration on peak-principal returns.

The search is a four-axis robust-J landscape.  Internal bootstrap selection is
deliberately absent: the reported evidence is the actual continuous out-of-
sample ledger and its matched-funding hold comparison.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable

import pandas as pd

from .models import BacktestConfig, DataBundle, OptimizationConfig
from .research.robust import EvaluationBudget, estimate_robust_search, select_robust_at

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


def _emit(progress: Progress | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, float(fraction))), message)


def _cancelled(callback: Cancelled | None) -> bool:
    return bool(callback and callback())


def _date_key(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _calendar(data: DataBundle) -> pd.DatetimeIndex:
    if len(data.calendar):
        values = pd.to_datetime(data.calendar, errors="coerce").dropna()
    else:
        values = pd.to_datetime(data.daily.get("date", []), errors="coerce").dropna()
    normalized = []
    for value in values:
        stamp = pd.Timestamp(value)
        normalized.append(stamp.tz_localize(None).normalize() if stamp.tzinfo is not None else stamp.normalize())
    return pd.DatetimeIndex(sorted(set(normalized)))


def _base_with(
    base: BacktestConfig,
    params: dict[str, Any] | None = None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> BacktestConfig:
    values = base.model_dump(mode="python")
    if params:
        values.update(deepcopy(params))
    if start is not None:
        values["start_date"] = start.date()
    if end is not None:
        values["end_date"] = end.date()
    values["schema_version"] = 3
    values["grid_mode"] = "geometric"
    values["min_shares"] = 0
    values["max_shares"] = None
    return BacktestConfig.model_validate(values)


def _metrics_pair(result: dict[str, Any] | None) -> dict[str, Any]:
    section = (((result or {}).get("strategies") or {}).get("grid") or {})
    metrics = section.get("metrics") if isinstance(section, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    grid = metrics.get("peak_principal_return")
    baseline = metrics.get("matched_funding_hold_return")
    excess = float(grid) - float(baseline) if grid is not None and baseline is not None else None
    return {
        "peak_principal_return": grid,
        "principal_drawdown": metrics.get("max_principal_drawdown"),
        "matched_funding_hold_return": baseline,
        "matched_funding_hold_drawdown": metrics.get("matched_funding_hold_drawdown"),
        "excess_return": excess,
        "trade_count": metrics.get("trade_count"),
    }


def _protocol(config: OptimizationConfig) -> dict[str, Any]:
    return config.protocol_description()


def _empty(config: OptimizationConfig, eval_start: pd.Timestamp, reason: str) -> dict[str, Any]:
    return {
        "status": "insufficient_history",
        "protocol_version": 5,
        "result_schema_version": 2,
        "config": config.model_dump(mode="json"),
        "protocol": _protocol(config),
        "fixed": {"status": "insufficient_history", "test_result": None},
        "adaptive": {"status": "not_requested", "schedule": [], "test_result": None},
        "evidence": {
            "candidate_archives": {"fixed": [], "current": [], "adaptive": []},
            "trial_archives": {"fixed": [], "current": [], "adaptive": []},
            "search_candidate_archives": {"fixed": [], "current": [], "adaptive": []},
            "decisions": [], "reasons": [reason],
        },
        "computation": {"decision_count": 0, "candidate_evaluations": 0, "ledger_evaluations": 0},
        "warnings": [reason],
    }


def optimize_v5(
    config: OptimizationConfig,
    data: DataBundle,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
    runtime=None,
    budget_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config.protocol_version != 5 or config.strategy.strategy_version != 5:
        raise ValueError("新版研究需要 protocol_version=5 和 strategy_version=5")
    if config.base.schema_version != 3:
        raise ValueError("新版研究需要 schema_version=3 的股数配置")
    dates = _calendar(data)
    start = _date_key(config.base.start_date)
    end = _date_key(config.base.end_date) if config.base.end_date else (dates[-1] if len(dates) else start)
    eval_dates = dates[(dates >= start) & (dates <= end)]
    if not len(eval_dates):
        result = _empty(config, start, "评估区间没有可用交易日")
        result.update(symbol=config.base.symbol, data_version=data.version)
        return result
    base = _base_with(config.base, start=eval_dates[0], end=eval_dates[-1])
    estimate = estimate_robust_search(config, data)
    contract = budget_contract if isinstance(budget_contract, dict) else None
    task_limit = int(estimate.get("effective_total_evaluation_budget") or 1)
    task_runtime = int(config.strategy.max_runtime_seconds)
    if contract is not None:
        if contract.get("data_version") not in (None, data.version):
            raise ValueError("提交时的行情数据版本与执行版本不一致，请重新提交研究")
        frozen_limit = int(contract.get("effective_total_evaluation_budget") or 0)
        if frozen_limit != task_limit:
            raise ValueError("提交时冻结的任务评价上限与执行预检不一致，请重新提交研究")
        frozen_runtime = int(contract.get("max_runtime_seconds") or task_runtime)
        if frozen_runtime != task_runtime:
            raise ValueError("提交时冻结的任务时限与执行配置不一致，请重新提交研究")
        frozen_workspace = float(contract.get("workspace_budget_mb") or 0)
        estimated_memory = float(estimate.get("estimated_task_memory_mb") or 0)
        if frozen_workspace and estimated_memory > frozen_workspace:
            raise ValueError("任务预计内存超过提交时冻结的工作区上限，请重新提交研究")
    task_budget = EvaluationBudget(
        total=max(1, task_limit),
        max_runtime_seconds=task_runtime,
    )
    target = config.selection_target
    decisions: list[dict[str, Any]] = []
    archives: dict[str, Any] = {"fixed": [], "current": [], "adaptive": []}
    trials: dict[str, Any] = {"fixed": [], "current": [], "adaptive": []}
    search_archives: dict[str, Any] = {"fixed": [], "current": [], "adaptive": []}
    fixed: dict[str, Any] = {"status": "not_requested", "test_result": None}
    fixed_selection: dict[str, Any] | None = None
    fixed_result = None
    adaptive_result = None
    schedule: list[dict[str, Any]] = []
    selection_count = 0
    # Every research decision is anchored to the evaluation start.  The last
    # close before that date is the price anchor; using the evaluation end
    # here would select parameters with hindsight and would let the current
    # recommendation's training window overlap the out-of-sample period.
    prior = dates[dates < eval_dates[0]]
    selection_asof = prior[-1] if len(prior) else eval_dates[0] - pd.Timedelta(days=1)

    if target in ("fixed", "both"):
        asof = selection_asof
        _emit(progress, .02, "选择固定样本外参数")
        fixed_selection = select_robust_at(
            config, base, data, asof, progress=progress, cancelled=cancelled,
            progress_start=.02, progress_width=.23, runtime=runtime, budget=task_budget,
        )
        selection_count += 1
        archives["fixed"] = fixed_selection.get("candidate_archive", [])
        trials["fixed"] = fixed_selection.get("trial_archive", [])
        search_archives["fixed"] = fixed_selection.get("search_candidate_archive", [])
        params = deepcopy(fixed_selection.get("recommended_params"))
        if params is None:
            selection_status = str(fixed_selection.get("status") or "no_qualified")
            if selection_status == "completed":
                selection_status = "no_qualified"
            fixed = {
                "status": selection_status, "params": None, "as_of": fixed_selection.get("as_of"),
                "test_result": None, "recommendation_status": fixed_selection.get("recommendation_status", "no_qualified"),
                "reasons": fixed_selection.get("reasons", []), "diagnostics": fixed_selection.get("diagnostics", {}),
                "recommendation_regions": fixed_selection.get("recommendation_regions", []),
            }
        elif not _cancelled(cancelled) and task_budget.reserve(1, "固定样本外回放"):
            fixed_base = _base_with(base, params, start=eval_dates[0], end=eval_dates[-1])
            from .share_engine import run_share_backtest
            try:
                fixed_result = run_share_backtest(fixed_base, data, progress=_mapped(progress, .26, .32), cancelled=cancelled)
                fixed_result["protocol"] = _protocol(config)
            except InterruptedError:
                fixed_result = None
            if fixed_result is not None:
                fixed = {
                    "status": "completed", "params": params, "recommended_params": params,
                    "diagnostic_params": fixed_selection.get("params"), "as_of": fixed_selection.get("as_of"),
                    "score": fixed_selection.get("score"), "eligible": fixed_selection.get("eligible"),
                    "test_result": fixed_result, "diagnostics": fixed_selection.get("diagnostics", {}),
                    "recommendation_status": "qualified", "reasons": fixed_selection.get("reasons", []),
                    "recommendation_regions": fixed_selection.get("recommendation_regions", []),
                    "oos_metrics": _metrics_pair(fixed_result),
                }
            else:
                fixed = {"status": "cancelled", "test_result": None, "reasons": ["用户取消固定样本外回放"]}
        elif _cancelled(cancelled):
            fixed = {"status": "cancelled", "test_result": None}
        else:
            fixed = {
                "status": "budget_exhausted", "test_result": None,
                "reasons": [task_budget.reason or "任务总预算已用尽"],
            }

    if target in ("adaptive", "both"):
        step_indices = list(range(0, len(eval_dates), max(1, int(config.rebalance_days))))
        for step, index in enumerate(step_indices):
            if _cancelled(cancelled) or task_budget.stopped:
                break
            effective = eval_dates[index]
            asof = eval_dates[index - 1] if index else (dates[dates < effective][-1] if len(dates[dates < effective]) else None)
            reused = bool(step == 0 and fixed_selection is not None and fixed_selection.get("as_of") == (asof.date().isoformat() if asof is not None else None))
            selected = deepcopy(fixed_selection) if reused else select_robust_at(
                config, base, data, asof, progress=progress, cancelled=cancelled,
                progress_start=.30 + .45 * step / max(1, len(step_indices)),
                progress_width=.45 / max(1, len(step_indices)), runtime=runtime, budget=task_budget,
            )
            if not reused:
                selection_count += 1
            archives["adaptive"].append({
                "effective_date": effective.date().isoformat(), "decision_date": selected.get("as_of"),
                "candidate_archive": selected.get("candidate_archive", []),
                "search_candidate_archive": selected.get("search_candidate_archive", []),
            })
            trials["adaptive"].append({"effective_date": effective.date().isoformat(), "trial_archive": selected.get("trial_archive", [])})
            params = deepcopy(selected.get("recommended_params"))
            paused = params is None
            schedule.append({
                "effective_date": effective.date().isoformat(), "decision_date": selected.get("as_of"),
                "params": deepcopy(selected["params"] if paused else params), "paused": paused,
                "recommendation_status": "no_qualified" if paused else "qualified",
            })
            decisions.append({
                "effective_date": effective.date().isoformat(), "decision_date": selected.get("as_of"),
                "status": selected.get("status"), "reused_from_fixed": reused,
                "selected_params": params, "diagnostic_params": deepcopy(selected.get("params")),
                "paused": paused, "score": selected.get("score"), "reasons": selected.get("reasons", []),
                "diagnostics": selected.get("diagnostics", {}), "recommendation_regions": selected.get("recommendation_regions", []),
            })
        if schedule and not _cancelled(cancelled) and task_budget.reserve(1, "自适应样本外回放"):
            from .share_engine import run_share_backtest
            try:
                adaptive_result = run_share_backtest(base, data, progress=_mapped(progress, .76, .18), cancelled=cancelled, parameter_schedule=schedule)
                adaptive_result["protocol"] = _protocol(config)
            except InterruptedError:
                adaptive_result = None
        adaptive = {
            "status": "completed" if adaptive_result is not None else ("cancelled" if _cancelled(cancelled) else "budget_exhausted"),
            "schedule": schedule, "test_result": adaptive_result,
            "latest_params": deepcopy(schedule[-1]["params"]) if schedule else None,
            "oos_metrics": _metrics_pair(adaptive_result),
            "reasons": [] if adaptive_result is not None else [task_budget.reason or "任务总预算已用尽"],
        }
    else:
        adaptive = {"status": "not_requested", "schedule": [], "test_result": None}

    _emit(progress, .95, "按评估开始日前收盘生成参数建议")
    if fixed_selection is not None:
        # The fixed and current branches deliberately share the same causal
        # anchor, so the second search would reproduce the first result.
        current = deepcopy(fixed_selection)
    else:
        current = select_robust_at(
            config, base, data, selection_asof, progress=progress, cancelled=cancelled,
            progress_start=.95, progress_width=.04, runtime=runtime, budget=task_budget,
        )
    selection_count += 1
    archives["current"] = current.get("candidate_archive", [])
    trials["current"] = current.get("trial_archive", [])
    search_archives["current"] = current.get("search_candidate_archive", [])
    current_recommended = deepcopy(current.get("recommended_params"))
    fixed["current"] = {
        "params": current_recommended, "recommended_params": current_recommended,
        "diagnostic_params": deepcopy(current.get("params")), "as_of": current.get("as_of"),
        "score": current.get("score"), "status": current.get("status") if current_recommended is not None else "no_qualified",
        "eligible": current.get("eligible"), "reasons": current.get("reasons", []),
        "diagnostics": current.get("diagnostics", {}),
        "recommendation_regions": current.get("recommendation_regions", []),
        "candidates": current.get("candidates", [])[: int(config.top_candidates)],
    }

    selection_diagnostics_for_status = [
        fixed.get("diagnostics") or {}, current.get("diagnostics") or {},
        *[(decision.get("diagnostics") or {}) for decision in decisions],
    ]
    data_issue_reasons = [
        str(item.get("data_issue"))
        for item in selection_diagnostics_for_status
        if isinstance(item, dict) and item.get("data_issue")
    ]
    data_issue_statuses = [
        fixed.get("status"), current.get("status"),
        *[decision.get("status") for decision in decisions],
    ]
    has_data_issue = bool(data_issue_reasons) or any(
        value == "data_issue" for value in data_issue_statuses
    )
    status = "cancelled" if _cancelled(cancelled) else ("data_issue" if has_data_issue else "completed")
    if not has_data_issue and (
        fixed.get("status") in {"partial", "insufficient_history", "insufficient_range", "cancelled", "budget_exhausted"}
        or adaptive.get("status") in {"partial", "cancelled", "budget_exhausted"}
        or current.get("status") in {"partial", "insufficient_history", "insufficient_range", "budget_exhausted"}
    ):
        status = "partial" if status != "cancelled" else status
    selection_diagnostics = [
        selection.get("diagnostics") or {}
        for selection in (fixed_selection, current)
        if isinstance(selection, dict)
    ]
    selection_diagnostics.extend(
        decision.get("diagnostics") or {}
        for decision in decisions
        if isinstance(decision.get("diagnostics"), dict)
    )
    score_complete = bool(selection_diagnostics) and all(
        item.get("score_complete") is True for item in selection_diagnostics
    )
    result = {
        "status": status, "protocol_version": 5, "result_schema_version": 2,
        "symbol": config.base.symbol, "data_version": data.version,
        "config": config.model_dump(mode="json"), "protocol": _protocol(config),
        "data_issue": data_issue_reasons[0] if data_issue_reasons else None,
        "fixed": fixed, "adaptive": adaptive,
        "evidence": {
            "candidate_archives": archives, "trial_archives": trials,
            "search_candidate_archives": search_archives, "decisions": decisions,
            "current_decision": {
                "decision_date": current.get("as_of"), "status": current.get("status"),
                "selected_params": current_recommended, "diagnostic_params": current.get("params"),
                "score": current.get("score"), "candidate_count": current.get("diagnostics", {}).get("candidate_count", 0),
                "eligible_count": current.get("diagnostics", {}).get("eligible_center_count", 0),
                "selected_eligible": current.get("eligible", False), "reasons": current.get("reasons", []),
                "recommendation_regions": current.get("recommendation_regions", []),
            },
        },
        "outer_validation": {
            "status": "not_applicable", "reason": "本版本未提供显著性检验；以下为真实连续样本外账本",
            "fixed": _metrics_pair(fixed_result), "adaptive": _metrics_pair(adaptive_result),
        },
        "computation": {
            "decision_count": selection_count,
            "candidate_evaluations": int(task_budget.consumed),
            "ledger_evaluations": 2 * int(task_budget.consumed),
            "training_budget": int(config.strategy.training_budget),
            "neighborhood_budget": int(config.strategy.validation_budget),
            "reporting_budget": int(config.strategy.reporting_budget),
            "total_evaluation_budget": config.strategy.total_evaluation_budget,
            "max_runtime_seconds": int(config.strategy.max_runtime_seconds),
            "evaluation_upper_bound": int(estimate["task_evaluation_upper_bound"]),
            "budget_contract_frozen": contract is not None,
            "frozen_evaluation_budget": task_limit,
            "frozen_data_version": (contract or {}).get("data_version"),
            "evaluation_budget_used": int(task_budget.consumed),
            "evaluation_budget_remaining": int(task_budget.remaining),
            "budget_stopped": bool(task_budget.stopped),
            "budget_reason": task_budget.reason,
            "estimated_task_memory_mb": float(estimate["estimated_task_memory_mb"]),
            "workspace_budget_mb": float(estimate["workspace_budget_mb"]),
        },
        "completion_status": "complete" if score_complete else "available",
        "warnings": list(dict.fromkeys([*data.warnings, *(fixed.get("reasons") or []), *(current.get("reasons") or [])])),
    }
    _emit(progress, 1.0, "新版稳健 J 研究完成")
    return result


def _mapped(progress: Progress | None, start: float, width: float) -> Progress | None:
    if progress is None:
        return None

    def callback(fraction: float, message: str) -> None:
        progress(start + width * max(0.0, min(1.0, float(fraction))), message)

    return callback
