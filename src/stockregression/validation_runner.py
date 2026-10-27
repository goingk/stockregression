"""Orchestration for complete research validation.

The runner keeps the selected v5 result as the primary decision source.  It
retests those decisions through the independent qualification kernel, then
records execution pressure and synthetic paths as descriptive evidence.  It
never replaces the primary arm after seeing a stress result.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
from pydantic import BaseModel

from .models import BacktestConfig, DataBundle
from .research_v5 import optimize_v5
from .validation_models import ResearchValidationRequest
from .validation_scenarios import SYNTHETIC_CASES, stress_cases, synthetic_bundle
from .validation_scoring import (
    common_decision_anchors,
    descriptive_summary,
    gss_rows_for_date,
    gss_screens,
    independent_account_summary,
    paired_forward_outcome,
    segment_principal_returns,
)


RESULT_SCHEMA_VERSION = 1
VALIDATION_PROTOCOL_VERSION = 1
_LEDGER_REFERENCE = "__validation_ledger__"

_BUSINESS_SELECTION_STATUSES = {
    "completed", "no_qualified", "not_requested",
}
_BUSINESS_SOURCE_STATUSES = {
    "completed", "partial", "no_qualified",
}
_INCOMPLETE_SELECTION_STATUSES = {
    "failed", "cancelled", "data_issue", "budget_exhausted",
}
_NOT_APPLICABLE_SELECTION_STATUSES = {
    "not_applicable", "not_requested", "insufficient_history",
    "insufficient_range",
}
_NEW_RESEARCH_SOURCE_REASONS = {
    "source_budget_incomplete", "source_branch_incomplete",
    "source_status_data_issue", "source_status_insufficient_history",
    "source_status_insufficient_range",
}


def _json_value(value: Any) -> Any:
    """Convert checkpoint input to strict JSON values without data frames."""
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _day(value: Any) -> date:
    return pd.Timestamp(value).date()


def _progress(callback, fraction: float, message: str) -> None:
    if callback:
        callback(max(0.0, min(1.0, float(fraction))), message)


def _runtime_seconds(runtime) -> float | None:
    if runtime is None:
        return None
    try:
        value = runtime() if callable(runtime) else runtime
    except TypeError:
        value = runtime
    if isinstance(value, Mapping):
        value = value.get("elapsed_seconds", value.get("elapsed"))
    elif hasattr(value, "elapsed_seconds"):
        value = getattr(value, "elapsed_seconds")
        value = value() if callable(value) else value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _source_business_complete(raw: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Separate a legitimate no-candidate outcome from a stopped search."""
    status = str(raw.get("status") or "")
    computation = raw.get("computation") if isinstance(raw.get("computation"), Mapping) else {}
    if not status:
        return False, "source_status_missing"
    if status not in _BUSINESS_SOURCE_STATUSES:
        return False, f"source_status_{status}"
    if bool(computation.get("budget_stopped")):
        return False, "source_budget_incomplete"
    time_stop = (
        bool(computation.get("runtime_stopped") or computation.get("timed_out"))
        or computation.get("stop_reason") in {"runtime_limit", "timeout"}
        or raw.get("stop_reason") in {"runtime_limit", "timeout"}
    )
    if time_stop:
        return False, "source_runtime_stopped"
    reason = computation.get("budget_reason")
    if reason and status == "budget_exhausted":
        return False, "source_budget_exhausted"
    fixed = raw.get("fixed") if isinstance(raw.get("fixed"), Mapping) else {}
    adaptive = raw.get("adaptive") if isinstance(raw.get("adaptive"), Mapping) else {}
    current = fixed.get("current") if isinstance(fixed.get("current"), Mapping) else {}
    branches = [str(item.get("status") or "") for item in (fixed, adaptive, current)]
    if any(item not in _BUSINESS_SELECTION_STATUSES for item in branches):
        return False, "source_branch_incomplete"
    if status in {"completed", "partial"}:
        # A finite search that normally reaches ``no_qualified`` is complete;
        # it does not need to prove global exhaustion.  A selected result with
        # incomplete scoring is still interrupted work.
        if any(item in _INCOMPLETE_SELECTION_STATUSES for item in branches):
            return False, "source_branch_incomplete"
        if bool(computation.get("budget_stopped")):
            return False, "source_budget_stopped"
        schedule = adaptive.get("schedule") if isinstance(adaptive.get("schedule"), list) else []
        selected = (
            fixed.get("params", fixed.get("recommended_params")) is not None
            or current.get("params", current.get("recommended_params")) is not None
            or any(not row.get("paused") for row in schedule if isinstance(row, Mapping))
        )
        if selected and raw.get("completion_status") not in {None, "complete"}:
            return False, "source_scoring_incomplete"
        return True, None
    if status == "no_qualified":
        return True, None
    return False, f"source_status_{status}"


def _adaptive_candidate_state(summary: Mapping[str, Any]) -> str:
    """Classify only the primary adaptive schedule, never fixed-selected."""
    adaptive = summary.get("adaptive")
    adaptive = adaptive if isinstance(adaptive, Mapping) else {}
    schedule = [
        row for row in adaptive.get("schedule") or []
        if isinstance(row, Mapping)
    ]
    has_effective_row = any(
        not bool(row.get("paused"))
        and isinstance(row.get("params"), Mapping)
        and bool(row.get("params"))
        for row in schedule
    )
    if has_effective_row:
        return "qualified"
    if schedule:
        return "no_qualified"
    statuses = {
        str(summary.get("source_status") or ""),
        str(adaptive.get("status") or ""),
    }
    if statuses & {"no_qualified", "no_candidate", "not_qualified"}:
        return "no_qualified"
    if statuses & _NOT_APPLICABLE_SELECTION_STATUSES:
        return "not_applicable"
    return "not_applicable"


def _qualification_state(
    candidate_state: str, qualification: Mapping[str, Any] | None,
) -> str:
    """Keep execution qualification independent from candidate existence."""
    if candidate_state in {"incomplete", "no_qualified", "not_applicable"}:
        return candidate_state
    cell = qualification if isinstance(qualification, Mapping) else {}
    result = cell.get("result")
    result = result if isinstance(result, Mapping) else {}
    if cell.get("status") != "completed":
        return "incomplete"
    status = str(result.get("qualification_status") or "")
    if status in {
        "unevaluable", "risk_breached", "execution_limited", "qualified",
        "not_applicable",
    }:
        return status
    return "incomplete"


def _aggregate_candidate_states(states: list[str]) -> str:
    if not states or "incomplete" in states:
        return "incomplete"
    if all(state == "qualified" for state in states):
        return "qualified"
    if all(state == "no_qualified" for state in states):
        return "no_qualified"
    if all(state == "not_applicable" for state in states):
        return "not_applicable"
    return "mixed"


def _aggregate_qualification_states(states: list[str]) -> str:
    """Apply the frozen business-evidence priority at the root."""
    if not states or "incomplete" in states:
        return "incomplete"
    for status in (
        "no_qualified", "not_applicable", "unevaluable",
        "risk_breached", "execution_limited", "qualified",
    ):
        if status in states:
            return status
    return "incomplete"


def _select_schedule(adaptive: Mapping[str, Any]) -> list[dict[str, Any]]:
    schedule = adaptive.get("schedule") if isinstance(adaptive.get("schedule"), list) else []
    normalized = []
    for row in schedule:
        if not isinstance(row, Mapping):
            continue
        normalized.append({
            "effective_date": row.get("effective_date"),
            "decision_date": row.get("decision_date"),
            "params": _json_value(row.get("params") or {}),
            "paused": bool(row.get("paused")),
            "recommendation_status": row.get("recommendation_status",
                                               "no_qualified" if row.get("paused") else "qualified"),
        })
    return normalized


def _compact_diagnostics(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    fields = (
        "candidate_count", "eligible_center_count", "selected_center_count",
        "scored_center_count", "score_missing_count", "score_complete",
        "local_peak_verified", "global_exhaustive", "data_issue",
        "training_window", "research_prior", "budget_reason", "status",
    )
    return {key: _json_value(value[key]) for key in fields if key in value}


def _optimization_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    fixed = raw.get("fixed") if isinstance(raw.get("fixed"), Mapping) else {}
    current = fixed.get("current") if isinstance(fixed.get("current"), Mapping) else {}
    adaptive = raw.get("adaptive") if isinstance(raw.get("adaptive"), Mapping) else {}
    evidence = raw.get("evidence") if isinstance(raw.get("evidence"), Mapping) else {}
    decisions = []
    for row in evidence.get("decisions") or []:
        if not isinstance(row, Mapping):
            continue
        decisions.append({
            "effective_date": row.get("effective_date"),
            "decision_date": row.get("decision_date"),
            "status": row.get("status"),
            "selected_params": _json_value(row.get("selected_params")),
            "paused": bool(row.get("paused")),
            "score": _json_value(row.get("score")),
            "reasons": _json_value(row.get("reasons") or []),
            "diagnostics": _compact_diagnostics(row.get("diagnostics")),
        })
    fixed_params = fixed.get("params", fixed.get("recommended_params"))
    current_params = current.get("params", current.get("recommended_params"))
    complete, incomplete_reason = _source_business_complete(raw)
    computation = raw.get("computation") if isinstance(raw.get("computation"), Mapping) else {}
    return {
        "source_status": raw.get("status"),
        "source_completion_status": raw.get("completion_status"),
        "business_complete": complete,
        "incomplete_reason": incomplete_reason,
        "data_issue": raw.get("data_issue"),
        "fixed": {
            "status": fixed.get("status"),
            "selected": _json_value(fixed_params),
            "diagnostics": _compact_diagnostics(fixed.get("diagnostics")),
            "as_of": fixed.get("as_of"),
        },
        "current": {
            "status": current.get("status"),
            "selected": _json_value(current_params),
            "diagnostics": _compact_diagnostics(current.get("diagnostics")),
            "decision_date": current.get("as_of"),
        },
        "adaptive": {
            "status": adaptive.get("status"),
            "schedule": _select_schedule(adaptive),
            "latest_params": _json_value(adaptive.get("latest_params")),
        },
        "decisions": decisions,
        "computation": {
            key: _json_value(computation.get(key))
            for key in (
                "candidate_evaluations", "evaluation_budget_used",
                "evaluation_budget_remaining", "budget_stopped", "budget_reason",
                "decision_count", "ledger_evaluations", "max_runtime_seconds",
            ) if key in computation
        },
        "warnings": _json_value(raw.get("warnings") or []),
    }


def _qualification_ledgers(raw: Mapping[str, Any]) -> dict[str, Any]:
    strategies = raw.get("strategies") if isinstance(raw.get("strategies"), Mapping) else {}
    grid = strategies.get("grid") if isinstance(strategies.get("grid"), Mapping) else {}
    hold = raw.get("scheduled_funding_hold")
    if not isinstance(hold, Mapping):
        hold = strategies.get("scheduled_funding_hold")
    if not isinstance(hold, Mapping):
        hold = strategies.get("matched_funding_hold")
    return {
        "grid": _json_value(grid),
        "scheduled_funding_hold": _json_value(hold or {}),
        "funding": _json_value(raw.get("funding") or {}),
        "parameter_schedule": _json_value(raw.get("parameter_schedule") or []),
    }


def _qualification_summary(
    raw: Mapping[str, Any], blocks: Iterable[Mapping[str, Any]] | None,
    *, checkpoint_ref: str,
) -> dict[str, Any]:
    strategies = raw.get("strategies") if isinstance(raw.get("strategies"), Mapping) else {}
    grid = strategies.get("grid") if isinstance(strategies.get("grid"), Mapping) else {}
    hold = raw.get("scheduled_funding_hold")
    if not isinstance(hold, Mapping):
        hold = strategies.get("scheduled_funding_hold")
    if not isinstance(hold, Mapping):
        hold = strategies.get("matched_funding_hold")
    hold = hold if isinstance(hold, Mapping) else {}
    grid_metrics = grid.get("metrics") or {}
    returns = segment_principal_returns(
        grid.get("equity") or [], blocks,
        initial_equity=(
            grid_metrics.get("initial_equity")
            if isinstance(grid_metrics, Mapping) else None
        ),
        initial_principal=(
            grid_metrics.get("initial_principal")
            if isinstance(grid_metrics, Mapping) else None
        ),
    )
    equity = grid.get("equity") or []
    trades = grid.get("trades") or []
    rejections = grid.get("rejections") or []
    contributions = hold.get("contributions") or raw.get("contributions") or []
    last = equity[-1] if equity and isinstance(equity[-1], Mapping) else {}
    return {
        "status": raw.get("status", "completed"),
        "qualification_status": raw.get("qualification_status", "qualified"),
        "unevaluable_reason": raw.get("unevaluable_reason"),
        "qualification_basis": _json_value(raw.get("qualification_basis") or {}),
        "strategy_metrics": _json_value(grid.get("metrics") or {}),
        "scheduled_funding_hold_metrics": _json_value(hold.get("metrics") or {}),
        "parameter_schedule": _json_value(raw.get("parameter_schedule") or []),
        "segment_returns": returns,
        "bounded_ledger": {
            "equity_points": len(equity), "trade_count": len(trades),
            "rejection_count": len(rejections),
            "contribution_count": len(contributions),
            "ending_equity": _json_value(last.get("equity")),
            "ending_cash": _json_value(last.get("cash")),
            "ending_position": _json_value(last.get("position")),
            "ending_principal": _json_value(last.get("total_invested")),
        },
        "checkpoint_ref": checkpoint_ref,
    }


class _CellState:
    def __init__(self, checkpoint: Mapping[str, Any] | None):
        value = checkpoint if isinstance(checkpoint, Mapping) else {}
        cells = value.get("cells") if isinstance(value.get("cells"), list) else []
        self.cells: dict[str, dict[str, Any]] = {
            str(cell["unit_id"]): _json_value(cell)
            for cell in cells if isinstance(cell, Mapping) and cell.get("unit_id")
        }
        self.ledgers: dict[str, Any] = _json_value(
            value.get("ledgers") if isinstance(value.get("ledgers"), Mapping) else {}
        )
        self.candidate_evaluations = int(value.get("candidate_evaluations") or 0)
        self.ledger_evaluations = int(value.get("ledger_evaluations") or 0)
        self.cache_hits = int(value.get("cache_hits") or 0)
        self.statistics_evaluations = int(value.get("statistics_evaluations") or 0)
        self.elapsed_seconds = float(value.get("elapsed_seconds") or 0.0)
        self.checkpoint_container: dict[str, Any] | None = None
        self.revision = int(value.get("revision") or 0)
        self.suite_version = value.get("suite_version")
        self.plan_hash = value.get("plan_hash")

    def completed(self, unit_id: str) -> dict[str, Any] | None:
        cell = self.cells.get(unit_id)
        if not cell or cell.get("status") != "completed":
            return None
        if (
            cell.get("kind") == "optimization"
            or unit_id.startswith("optimization:")
        ):
            result = cell.get("result")
            if not isinstance(result, Mapping) or result.get("business_complete") is not True:
                return None
        return cell

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "revision": self.revision,
            "suite_version": self.suite_version,
            "plan_hash": self.plan_hash,
            "cells": sorted(self.cells.values(), key=lambda item: str(item.get("unit_id"))),
            "ledgers": self.ledgers,
            "candidate_evaluations": self.candidate_evaluations,
            "actual_candidate_evaluations": self.candidate_evaluations,
            "ledger_evaluations": self.ledger_evaluations,
            "actual_ledger_evaluations": self.ledger_evaluations,
            "cache_hits": self.cache_hits,
            "total_evaluations": self.candidate_evaluations + self.ledger_evaluations,
            "statistics_evaluations": self.statistics_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _schedule_base_params(config: BacktestConfig) -> dict[str, Any]:
    fields = ("buy_step", "sell_step", "grid_step", "buy_shares", "sell_shares",
              "min_shares", "max_shares")
    values = config.model_dump(mode="python")
    return {key: values[key] for key in fields if key in values}


def _paused_schedule(
    config: BacktestConfig, bundle: DataBundle, starts: Iterable[Any],
) -> list[dict[str, Any]]:
    calendar = pd.DatetimeIndex(pd.to_datetime(bundle.calendar)).normalize().sort_values()
    rows = []
    for value in starts:
        try:
            effective = pd.Timestamp(value).normalize()
        except (TypeError, ValueError):
            continue
        previous = calendar[calendar < effective]
        decision = previous[-1] if len(previous) else effective - pd.Timedelta(days=1)
        rows.append({
            "effective_date": effective.date().isoformat(),
            "decision_date": decision.date().isoformat(),
            "params": _schedule_base_params(config),
            "paused": True,
            "recommendation_status": "no_qualified",
        })
    return rows


def _config_with_params(config: BacktestConfig, params: Mapping[str, Any] | None) -> BacktestConfig:
    if not params:
        return config
    return BacktestConfig.model_validate({**config.model_dump(mode="python"), **params})


def _run_qualification_kernel(
    config: BacktestConfig, bundle: DataBundle, account: Any,
    schedule: list[dict[str, Any]] | None, cancelled=None, **kwargs,
) -> dict[str, Any]:
    from . import share_engine
    worker = getattr(share_engine, "run_qualification", None)
    if not callable(worker):
        raise RuntimeError("share_engine.run_qualification 尚未实现")
    return worker(
        config, bundle, account, parameter_schedule=schedule or None,
        passive_grid=False, cancelled=cancelled, **kwargs,
    )


def _normalise_schedule(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("effective_date"):
            continue
        result.append({
            "effective_date": str(row.get("effective_date")),
            "decision_date": str(row.get("decision_date") or
                                 (pd.Timestamp(row["effective_date"]) - pd.Timedelta(days=1)).date()),
            "params": _json_value(row.get("params") or {}),
            "paused": bool(row.get("paused")),
            "recommendation_status": row.get("recommendation_status"),
        })
    return result


def _stamp(value: Any) -> pd.Timestamp | None:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(result):
        return None
    return result.normalize()


def _visible_etf_prefix(bundle: DataBundle, stamp: pd.Timestamp) -> bool:
    """Return availability from data strictly visible at ``stamp``.

    The full bundle's terminal date is deliberately irrelevant: a future row
    cannot make the symbol available for an earlier cross-section.
    """
    metadata = bundle.metadata or {}
    for key, invalid in (
        ("listing_date", lambda value: value > stamp),
        ("delisting_date", lambda value: value <= stamp),
    ):
        raw = metadata.get(key)
        if raw in (None, ""):
            continue
        value = _stamp(raw)
        if value is not None and invalid(value):
            return False
    if bundle.daily.empty or "date" not in bundle.daily:
        return False
    dates = pd.to_datetime(bundle.daily.date, errors="coerce").dt.normalize()
    closes = pd.to_numeric(bundle.daily.get("close"), errors="coerce")
    visible = dates.le(stamp) & closes.gt(0)
    if "tradestatus" in bundle.daily:
        visible &= pd.to_numeric(
            bundle.daily.tradestatus, errors="coerce",
        ).fillna(1).ne(0)
    return bool(visible.any())


def _is_etf_bundle(bundle: DataBundle) -> bool:
    """Resolve ETF scope from frozen metadata before using symbol fallback."""
    metadata = bundle.metadata if isinstance(bundle.metadata, Mapping) else {}
    asset_type = metadata.get("asset_type")
    if asset_type is not None:
        return str(asset_type).strip().lower() == "etf"
    board = metadata.get("board")
    if board is not None:
        return str(board).strip().lower() == "etf"
    from .rules import instrument_terms

    return instrument_terms(bundle.symbol, dict(metadata))["asset_type"] == "etf"


def _evaluation_dates(bundle: DataBundle, item: Mapping[str, Any]) -> set[pd.Timestamp]:
    if bundle.daily.empty or "date" not in bundle.daily:
        return set()
    start, end = _stamp(item.get("evaluation_start")), _stamp(item.get("evaluation_end"))
    if start is None or end is None:
        return set()
    dates = pd.to_datetime(bundle.daily.date, errors="coerce").dt.normalize()
    closes = pd.to_numeric(bundle.daily.get("close"), errors="coerce")
    valid = dates.between(start, end, inclusive="both") & closes.gt(0)
    if "tradestatus" in bundle.daily:
        valid &= pd.to_numeric(
            bundle.daily.tradestatus, errors="coerce",
        ).fillna(1).ne(0)
    blocks = [block for block in item.get("blocks") or [] if isinstance(block, Mapping)]
    if blocks:
        in_block = pd.Series(False, index=bundle.daily.index)
        for block in blocks:
            block_start, block_end = _stamp(block.get("start")), _stamp(block.get("end"))
            if block_start is not None and block_end is not None:
                in_block |= dates.between(block_start, block_end, inclusive="both")
        valid &= in_block
    return set(dates.loc[valid].tolist())


def _active_schedule_row(
    schedule: Iterable[Mapping[str, Any]], decision_date: pd.Timestamp,
) -> Mapping[str, Any] | None:
    active = []
    for row in schedule:
        if not isinstance(row, Mapping):
            continue
        stamp = _stamp(row.get("effective_date"))
        if stamp is not None and stamp <= decision_date:
            active.append((stamp, row))
    return max(active, key=lambda item: item[0])[1] if active else None


def _screen_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [row.get("paired_excess") for row in rows]
    drawdowns = [row.get("max_principal_drawdown") for row in rows]
    no_candidate = sum(1 for row in rows if row.get("no_candidate") is True)
    missing = sum(1 for row in rows if row.get("status") == "missing_outcome")
    selected = len(rows)
    return {
        "selected_count": selected,
        "outcome_count": sum(
            1 for row in rows if row.get("paired_excess") is not None
        ),
        "missing_outcome_count": missing,
        "no_candidate_count": no_candidate,
        "no_candidate_ratio": no_candidate / selected if selected else None,
        "paired_excess": descriptive_summary(paired),
        "max_principal_drawdown": descriptive_summary(drawdowns),
        "missing_policy": "retain_selected_symbol_no_replacement",
    }


def _aggregate_screen_outcomes(windows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = {name: [] for name in (
        "random_all", "score_only", "filter_only", "score_filter",
    )}
    for window in windows:
        screens = window.get("screens") if isinstance(window, Mapping) else None
        if not isinstance(screens, Mapping):
            continue
        for name in aggregate:
            screen = screens.get(name)
            if not isinstance(screen, Mapping):
                continue
            rows = screen.get("selected_outcomes")
            if isinstance(rows, list):
                aggregate[name].extend(
                    dict(row) for row in rows if isinstance(row, Mapping)
                )
    return {name: _screen_distribution(rows) for name, rows in aggregate.items()}


def _primary_axes(request: ResearchValidationRequest, plan: Mapping[str, Any]) -> tuple[str, str]:
    primary = plan.get("primary_arm")
    if isinstance(primary, Mapping):
        window = primary.get("training_window_mode", primary.get("window_mode", "same_price"))
        prior = primary.get("prior_policy", "daily_state_v1")
    elif isinstance(primary, str):
        parts = primary.split(":")
        window = parts[0] if parts and parts[0] in {"same_price", "fixed"} else "same_price"
        prior = parts[1] if len(parts) > 1 and parts[1] in {"off", "daily_state_v1"} else "daily_state_v1"
    else:
        window = request.research.strategy.training_window_mode
        prior = request.research.prior_policy
    return str(window), str(prior)


def _arms(request: ResearchValidationRequest, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary_window, primary_prior = _primary_axes(request, plan)
    standard = [
        (primary_window, primary_prior),
        ("same_price", "off"), ("same_price", "daily_state_v1"),
        ("fixed", "off"), ("fixed", "daily_state_v1"),
    ]
    seen, arms = set(), []
    for window, prior in standard:
        if window not in {"same_price", "fixed"} or prior not in {"off", "daily_state_v1"}:
            continue
        group = f"{window}:{prior}"
        if group in seen:
            continue
        seen.add(group)
        arms.append({
            "arm_group": group, "window_mode": window, "prior_policy": prior,
            "primary": group == f"{primary_window}:{primary_prior}",
        })
    if len(arms) != 4:
        raise ValueError("完整验证必须包含四个冻结的窗口/先验组合")
    return arms


def _cell_public(cell: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(cell)
    # Ledger arrays only live under checkpoint["ledgers"] and never in root.
    value.pop("ledger", None)
    value.pop("source_result", None)
    return value


def materialize_full_validation_result(
    result: Mapping[str, Any], checkpoint: Mapping[str, Any] | None,
    *, resolve: bool = False, load_unit_ledger=None,
) -> dict[str, Any]:
    """Attach ledger values or references to a full validation export.

    ``resolve=False`` is the API-safe mode: persisted ledgers remain compact
    ``__validation_ledger__`` references.  Tests or a single detail endpoint may
    opt into materialising them with ``resolve=True`` and a loader.
    """
    value = deepcopy(dict(result))
    ledgers = checkpoint.get("ledgers") if isinstance(checkpoint, Mapping) else {}
    ledgers = ledgers if isinstance(ledgers, Mapping) else {}

    def indexed_reference(unit_id: str, ledger_value: Any) -> dict[str, Any]:
        if isinstance(ledger_value, Mapping) and _LEDGER_REFERENCE in ledger_value:
            return {
                "checkpoint_ref": unit_id,
                "storage": "immutable_ledger",
                "relative_path": ledger_value.get(_LEDGER_REFERENCE),
                "sha256": ledger_value.get("sha256"),
            }
        # Legacy checkpoints may still contain an inline ledger.  The export
        # index remains a reference map even for those test/compatibility
        # inputs, so raw minute rows only appear at ``cells[].ledger``.
        return {
            "checkpoint_ref": unit_id,
            "storage": "checkpoint_inline",
            "materialized_location": "cells[].ledger",
        }

    references = {
        str(unit_id): indexed_reference(str(unit_id), ledger_value)
        for unit_id, ledger_value in ledgers.items()
    }

    def resolved(ref: Any) -> Any:
        if not isinstance(ref, Mapping) or _LEDGER_REFERENCE not in ref:
            return deepcopy(ref)
        if not resolve:
            return deepcopy(ref)
        if not load_unit_ledger:
            raise ValueError("解析持久化验证账本需要 load_unit_ledger 回调")
        return deepcopy(load_unit_ledger(deepcopy(ref)))

    cells = []
    for cell in value.get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        item = dict(cell)
        ref = item.get("checkpoint_ref")
        if ref in ledgers:
            item["ledger"] = resolved(ledgers[ref])
        cells.append(item)
    value["cells"] = cells
    for key in ("stress", "synthetic"):
        if isinstance(value.get(key), list):
            summaries = []
            for raw in value[key]:
                if not isinstance(raw, Mapping):
                    continue
                item = deepcopy(dict(raw))
                item.pop("ledger", None)
                item.pop("ledger_ref", None)
                summaries.append(item)
            value[key] = summaries
    for symbol in value.get("symbols") or []:
        if not isinstance(symbol, dict):
            continue
        for arm in symbol.get("arms") or []:
            if not isinstance(arm, dict):
                continue
            for scheme in (arm.get("schemes") or {}).values():
                if not isinstance(scheme, dict):
                    continue
                qualification = scheme.get("qualification")
                if isinstance(qualification, dict):
                    qualification.pop("ledger", None)
                    qualification.pop("ledger_ref", None)
    value["ledger_refs"] = references
    value.pop("ledgers", None)
    value["full_export"] = True
    value["ledgers_resolved"] = bool(resolve)
    value["ledger_export"] = {
        "materialized_location": "cells[].ledger",
        "references_location": "ledger_refs",
        "duplicated_raw_ledgers": False,
    }
    return value


def materialize_stored_validation_result(
    store: Any, result: Mapping[str, Any], *, resolve: bool = False,
    load_unit_ledger=None,
) -> dict[str, Any]:
    """Hydrate a stored validation root from its durable numbered checkpoint."""
    from .validation_store import ValidationStore

    run_id = result.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("验证结果缺少 run_id，不能读取完整账本")
    validations = ValidationStore(store)
    snapshot = result.get("checkpoint_snapshot")
    if isinstance(snapshot, str) and snapshot:
        checkpoint = validations.load_frozen_checkpoint(snapshot)
        if checkpoint is None:
            raise ValueError("验证结果的冻结检查点不存在或不可读取")
    else:
        checkpoint = validations.load_checkpoint(run_id)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("验证结果没有可读取的检查点")
    if resolve and load_unit_ledger is None:
        load_unit_ledger = validations.load_ledger
    return materialize_full_validation_result(
        result, checkpoint, resolve=resolve, load_unit_ledger=load_unit_ledger,
    )


def run_validation(
    request: ResearchValidationRequest, plan: dict,
    bundles: dict[str, DataBundle], progress=None, cancelled=None,
    runtime=None, checkpoint=None, save_checkpoint=None, audit_event=None,
    save_inner_result: Callable[[dict, str, str], str] | None = None,
    save_unit_ledger=None, load_unit_ledger=None,
) -> dict:
    """Run the complete validation suite and return its bounded public root."""
    started = time.monotonic()
    cancel = cancelled or (lambda: False)
    if checkpoint is not None and not isinstance(checkpoint, Mapping):
        raise ValueError("完整验证检查点必须是映射对象")
    if checkpoint:
        checkpoint_plan = checkpoint.get("plan_hash")
        checkpoint_suite = checkpoint.get("suite_version")
        checkpoint_schema = checkpoint.get("schema_version", 1)
        if (
            checkpoint_suite != request.suite_version
            or checkpoint_plan != plan.get("plan_hash")
            or int(checkpoint_schema or 0) != 1
        ):
            raise ValueError(
                "检查点合同与冻结验证方案不一致，拒绝复用或覆盖旧证据"
            )
    state = _CellState(checkpoint)
    prior_elapsed = state.elapsed_seconds
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    state.suite_version = request.suite_version
    state.plan_hash = plan.get("plan_hash")
    stop_reason: str | None = None
    budget = (
        (plan.get("estimated_work") or {}).get("total_evaluation_budget")
        if isinstance(plan.get("estimated_work"), Mapping) else None
    )
    budget = int(budget if budget is not None else request.total_evaluation_budget or 0) or None

    def elapsed_now() -> float:
        elapsed = prior_elapsed + max(0.0, time.monotonic() - started)
        provided = _runtime_seconds(runtime)
        return max(elapsed, provided or 0.0)

    def audit(unit_id: str, event: str, payload: Mapping[str, Any] | None = None) -> None:
        nonlocal stop_reason
        value = _json_value(payload or {})
        events.append({"unit_id": unit_id, "event": event, "payload": value})
        if audit_event:
            try:
                audit_event(unit_id, event, value)
            except Exception as exc:
                message = f"审计写入失败（{unit_id}/{event}）：{type(exc).__name__}: {exc}"
                warnings.append(message)
                stop_reason = stop_reason or "audit_write_failed"

    def save() -> None:
        nonlocal stop_reason
        state.revision += 1
        state.elapsed_seconds = elapsed_now()
        if save_checkpoint:
            try:
                value = save_checkpoint(state.payload())
                if isinstance(value, Mapping):
                    state.checkpoint_container = _json_value(value)
            except Exception as exc:
                message = f"检查点写入失败：{type(exc).__name__}: {exc}"
                warnings.append(message)
                stop_reason = stop_reason or "checkpoint_write_failed"

    def stop_reason_now() -> str | None:
        if cancel():
            return "cancelled"
        elapsed = elapsed_now()
        if elapsed >= request.max_runtime_seconds:
            return "runtime_limit"
        consumed = state.candidate_evaluations + state.ledger_evaluations
        if budget is not None and consumed >= budget:
            return "evaluation_budget"
        return None

    def load_ledger(unit_id: str) -> Any:
        value = state.ledgers.get(unit_id)
        if not isinstance(value, Mapping) or _LEDGER_REFERENCE not in value:
            return value
        if not load_unit_ledger:
            raise RuntimeError("持久化验证账本缺少 load_unit_ledger 回调")
        return load_unit_ledger(_json_value(value))

    def execute(
        unit_id: str, kind: str, symbol: str, arm_group: str | None,
        scheme: str | None, function: Callable[[], tuple[dict[str, Any], Mapping[str, Any] | None]],
        *, evaluations: int = 0, reservation: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        nonlocal stop_reason
        existing = state.cells.get(unit_id)
        cached = state.completed(unit_id)
        if cached is not None:
            state.cache_hits += 1
            cached = {**cached, "cache_reused": True}
            state.cells[unit_id] = cached
            audit(unit_id, "cache_reused", {
                "source": "cache", "kind": kind, "symbol": symbol,
                "arm_group": arm_group, "scheme": scheme,
            })
            save()
            return cached
        retry_history = []
        if isinstance(existing, Mapping):
            previous = existing.get("retry_history")
            if isinstance(previous, list):
                retry_history.extend(_json_value(item) for item in previous)
            previous_result = existing.get("result")
            previous_result = (
                previous_result if isinstance(previous_result, Mapping) else {}
            )
            previous_computation = (
                previous_result.get("computation")
                if isinstance(previous_result.get("computation"), Mapping) else {}
            )
            retry_history.append({
                "status": existing.get("status"),
                "source_status": previous_result.get("source_status"),
                "business_complete": previous_result.get("business_complete"),
                "incomplete_reason": previous_result.get("incomplete_reason"),
                "candidate_evaluations": previous_computation.get(
                    "candidate_evaluations", 0
                ),
                "selection_result_id": previous_result.get("selection_result_id"),
            })
        stop_reason = stop_reason or stop_reason_now()
        if stop_reason:
            return None
        reserve = max(0, int(evaluations if reservation is None else reservation))
        consumed = state.candidate_evaluations + state.ledger_evaluations
        if budget is not None and consumed + reserve > budget:
            stop_reason = "evaluation_budget"
            return None
        audit(unit_id, "started", {
            "kind": kind, "symbol": symbol, "arm_group": arm_group,
            "scheme": scheme,
            "attempt": len(retry_history) + 1,
            **({"retry_history": retry_history} if retry_history else {}),
            **(metadata or {}),
        })
        if stop_reason:
            return None
        try:
            result, ledger = function()
            if kind == "optimization" and not evaluations:
                evaluations = int((result.get("computation") or {}).get(
                    "candidate_evaluations") or 0
                )
            cell = {
                "unit_id": unit_id, "kind": kind, "symbol": symbol,
                "arm_group": arm_group, "scheme": scheme,
                "status": "completed", "checkpoint_ref": unit_id,
                "result": _json_value(result),
                **({"metadata": _json_value(metadata)} if metadata else {}),
            }
            if retry_history:
                cell["attempt"] = len(retry_history) + 1
                cell["retry_history"] = retry_history
            if ledger is not None:
                if save_unit_ledger is not None:
                    reference = save_unit_ledger(unit_id, ledger)
                    reference = _json_value(reference)
                    if (
                        not isinstance(reference, Mapping)
                        or _LEDGER_REFERENCE not in reference
                    ):
                        raise RuntimeError(
                            "save_unit_ledger 必须返回含 __validation_ledger__ 的引用"
                        )
                    state.ledgers[unit_id] = reference
                else:
                    state.ledgers[unit_id] = _json_value(ledger)
            state.cells[unit_id] = cell
            if kind == "optimization":
                state.candidate_evaluations += max(0, int(evaluations))
                if reservation is not None and int(evaluations) > int(reservation):
                    warnings.append(
                        f"{unit_id} 实际候选评价 {int(evaluations)} 超过冻结上界 {int(reservation)}"
                    )
                    stop_reason = stop_reason or "evaluation_budget_contract_exceeded"
            else:
                ledger_increment = max(1, int(evaluations or 1)) if ledger is not None else 0
                state.ledger_evaluations += ledger_increment
            if kind == "gss":
                state.statistics_evaluations += int(
                    result.get("statistics_evaluations") or 0
                )
            audit(unit_id, "completed", {
                "status": "completed", "kind": kind,
                "candidate_evaluations": evaluations if kind == "optimization" else 0,
                "ledger_evaluations": (
                    0 if kind == "optimization" else
                    max(1, int(evaluations or 1)) if ledger is not None else 0
                ),
                "actual_evaluations": max(0, int(evaluations)),
            })
            save()
            return cell
        except InterruptedError as exc:
            stop_reason = "cancelled"
            cell = {
                "unit_id": unit_id, "kind": kind, "symbol": symbol,
                "arm_group": arm_group, "scheme": scheme,
                "status": "cancelled", "error": str(exc),
                "checkpoint_ref": unit_id,
            }
            if retry_history:
                cell["attempt"] = len(retry_history) + 1
                cell["retry_history"] = retry_history
            state.cells[unit_id] = cell
            audit(unit_id, "failed", {"status": "cancelled", "error": str(exc),
                                       "failure_category": "cancelled"})
            save()
            return cell
        except Exception as exc:
            cell = {
                "unit_id": unit_id, "kind": kind, "symbol": symbol,
                "arm_group": arm_group, "scheme": scheme,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "checkpoint_ref": unit_id,
            }
            if retry_history:
                cell["attempt"] = len(retry_history) + 1
                cell["retry_history"] = retry_history
            state.cells[unit_id] = cell
            audit(unit_id, "failed", {
                "status": "failed", "error": cell["error"],
                "failure_category": type(exc).__name__,
            })
            save()
            return cell

    try:
        arms = _arms(request, plan)
    except Exception as exc:
        return {
            "kind": "research_validation", "status": "failed",
            "completion_status": "incomplete", "suite_version": request.suite_version,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "plan_hash": plan.get("plan_hash"), "primary_arm": plan.get("primary_arm"),
            "stop_reason": "arm_configuration",
            "selection_result_id": None, "selection_result_ids": [],
            "symbols": [], "cells": [], "stress": [], "synthetic": [],
            "gss": {"status": "incomplete", "decision_dates": [], "windows": [],
                    "dates": [], "historical_universe_verified": False,
                    "portfolio_nav_constructed": False},
            "summary": {"evidence_kind": "descriptive", "significance_tests_used": False},
            "computation": {"candidate_evaluations": 0, "actual_candidate_evaluations": 0,
                            "ledger_evaluations": 0, "total_evaluations": 0,
                            "cache_hits": 0, "missing_units": []},
            "checkpoint": None, "audit": events, "descriptive_only": True,
            "warnings": [f"验证方案无法展开：{type(exc).__name__}: {exc}"],
        }
    if request.research.source_etf is not None and len(request.symbols) != 1:
        raise ValueError("source_etf 只能绑定单一标的")

    items = [item for item in plan.get("symbols") or [] if isinstance(item, Mapping)]
    optimization_cells: dict[tuple[str, str], dict[str, Any]] = {}
    qualification_cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    required_units: list[str] = []
    skipped_symbols: list[str] = []

    # Freeze the complete known unit manifest before any branch can stop.  The
    # synthetic discovery marker is replaced only after an optimization result
    # reveals its selected parameter sets.
    for item in items:
        symbol = str(item.get("symbol") or "")
        bundle = bundles.get(symbol)
        if (
            item.get("status") != "ready" or bundle is None
            or (item.get("data_version") and bundle.version != item.get("data_version"))
        ):
            continue
        for arm in arms:
            group = arm["arm_group"]
            required_units.append(f"optimization:{symbol}:{group}")
            for scheme in ("fixed", "adaptive"):
                qualify_unit = f"qualification:{symbol}:{group}:{scheme}"
                required_units.append(qualify_unit)
                required_units.extend(
                    f"stress:{symbol}:{group}:{scheme}:{case['case']}"
                    for case in stress_cases(request.accounts[symbol])
                )
            required_units.append(f"synthetic-selection:{symbol}:{group}")

    total_symbols = max(1, len(items))
    for symbol_index, item in enumerate(items):
        symbol = str(item.get("symbol") or "")
        if item.get("status") != "ready" or symbol not in bundles:
            warnings.append(f"{symbol or '未知标的'} 未进入执行：冻结方案状态或数据不可用")
            skipped_symbols.append(symbol)
            continue
        bundle = bundles[symbol]
        if item.get("data_version") and bundle.version != item.get("data_version"):
            warnings.append(f"{symbol} 数据版本与冻结方案不一致")
            skipped_symbols.append(symbol)
            continue
        for arm_index, arm in enumerate(arms):
            group = arm["arm_group"]
            arm_fraction = (symbol_index + (arm_index + 1) / len(arms)) / total_symbols
            opt_unit = f"optimization:{symbol}:{group}"

            # The callback needs the full inner result; the summary is expanded
            # here without retaining candidate archives in the public result.
            def optimize_function(item=item, bundle=bundle, arm=arm, opt_unit=opt_unit):
                _progress(progress, .02 + .36 * arm_fraction,
                          f"{bundle.symbol} · {arm['arm_group']} · 冻结参数选择")
                training_days = int(
                    item.get("training_days") or plan.get("training_days")
                    or request.research.strategy.training_max_days
                )
                minimum_days = min(int(
                    item.get("training_min_days") or plan.get("training_min_days")
                    or request.research.strategy.training_min_days
                ), training_days)
                min_history = min(int(request.research.min_history_days), training_days)
                optimization = request.research.to_optimization()
                context = optimization.research_context.model_copy(update={
                    "symbol": bundle.symbol, "start_date": _day(item["evaluation_start"]),
                    "end_date": _day(item["evaluation_end"]),
                })
                strategy = optimization.strategy.model_copy(update={
                    "training_window_mode": arm["window_mode"],
                    "training_max_days": training_days, "training_min_days": minimum_days,
                })
                base = optimization.base.model_copy(update={
                    "symbol": bundle.symbol, "start_date": _day(item["evaluation_start"]),
                    "end_date": _day(item["evaluation_end"]), "data_policy": None,
                })
                config = optimization.model_copy(update={
                    "base": base, "research_context": context, "strategy": strategy,
                    "selection_target": "both", "prior_policy": arm["prior_policy"],
                    "requested_data_version": item.get("data_version") or bundle.version,
                    "min_history_days": max(20, min_history),
                })
                callback = (lambda fraction, message:
                            _progress(progress, .02 + .36 * arm_fraction + .08 * float(fraction), message))
                raw = optimize_v5(config, bundle, progress=callback,
                                  cancelled=cancel, runtime=runtime)
                inner_id = None
                if save_inner_result is not None:
                    inner_id = save_inner_result(raw, bundle.symbol, arm["arm_group"])
                summary = _optimization_summary(raw)
                if inner_id is not None:
                    summary["selection_result_id"] = inner_id
                evaluations = int((raw.get("computation") or {}).get("candidate_evaluations") or 0)
                return summary, None

            estimate = item.get("estimated_work") if isinstance(
                item.get("estimated_work"), Mapping
            ) else {}
            reservation = estimate.get("selection_evaluations_per_run_upper_bound")
            if reservation is None:
                plan_estimate = plan.get("estimated_work") if isinstance(
                    plan.get("estimated_work"), Mapping
                ) else {}
                aggregate_bound = plan_estimate.get(
                    "candidate_evaluations_upper_bound"
                )
                try:
                    aggregate_bound = int(aggregate_bound or 0)
                except (TypeError, ValueError):
                    aggregate_bound = 0
                selection_runs = int(estimate.get("selection_runs") or 4)
                if aggregate_bound > 0:
                    reservation = math.ceil(
                        aggregate_bound / max(1, selection_runs * total_symbols)
                    )
                else:
                    reservation = budget or 1
                if budget is not None:
                    reservation = min(int(reservation), int(budget))
            reservation = max(1, int(reservation))

            cell = execute(
                opt_unit, "optimization", symbol, group, None,
                optimize_function,
                reservation=reservation,
                metadata={"training_days": item.get("training_days")},
            )
            if cell and cell.get("status") == "completed":
                optimization_cells[(symbol, group)] = cell

            summary = cell.get("result", {}) if cell else {}
            fixed = summary.get("fixed", {}) if isinstance(summary, Mapping) else {}
            adaptive = summary.get("adaptive", {}) if isinstance(summary, Mapping) else {}
            scheme_order = ("adaptive", "fixed") if arm["primary"] else ("fixed", "adaptive")
            for scheme in scheme_order:
                qualify_unit = f"qualification:{symbol}:{group}:{scheme}"
                scheme_data = fixed if scheme == "fixed" else adaptive
                selected = scheme_data.get("selected") if isinstance(scheme_data, Mapping) else None
                schedule = _normalise_schedule(scheme_data.get("schedule", [])) if scheme == "adaptive" else []
                if scheme == "fixed":
                    paused = selected is None or not isinstance(selected, Mapping)
                else:
                    paused = not any(not row.get("paused") for row in schedule)
                base_config = request.research.to_optimization().base.model_copy(update={
                    "symbol": symbol, "start_date": _day(item["evaluation_start"]),
                    "end_date": _day(item["evaluation_end"]), "data_policy": None,
                })
                if paused:
                    starts = ([row["effective_date"] for row in schedule] or
                              [str(item["evaluation_start"])])
                    schedule = schedule or _paused_schedule(base_config, bundle, starts)
                    for row in schedule:
                        row["paused"] = True
                    kernel_config = base_config
                else:
                    kernel_config = base_config if scheme == "adaptive" else _config_with_params(base_config, selected)

                def qualification_function(scheme=scheme, kernel_config=kernel_config,
                                            schedule=schedule, qualify_unit=qualify_unit):
                    _progress(progress, .40 + .30 * arm_fraction / 2,
                              f"{symbol} · {group} · {scheme} · 现金与成交复核")
                    raw = _run_qualification_kernel(
                        kernel_config, bundle, request.accounts[symbol],
                        schedule, cancelled=cancel,
                        max_principal_drawdown=request.research.max_principal_drawdown,
                    )
                    return _qualification_summary(raw, item.get("blocks"), checkpoint_ref=qualify_unit), _qualification_ledgers(raw), 1

                qcell = execute(
                    qualify_unit, "qualification", symbol, group, scheme,
                    lambda: (lambda value: (value[0], value[1]))(qualification_function()),
                    evaluations=1,
                )
                if qcell and qcell.get("status") == "completed":
                    qualification_cells[(symbol, group, scheme)] = qcell

                # Six execution pressure scenarios are cells even when the
                # base cell is a reference to the just-completed qualification.
                cases = stress_cases(request.accounts[symbol])
                for case in cases:
                    stress_unit = f"stress:{symbol}:{group}:{scheme}:{case['case']}"
                    stopped = stop_reason_now()
                    if stopped:
                        stop_reason = stop_reason or stopped
                        break
                    if stop_reason:
                        break
                    if not case["applicable"]:
                        state.cells[stress_unit] = {
                            "unit_id": stress_unit, "kind": "stress", "symbol": symbol,
                            "arm_group": group, "scheme": scheme, "status": "not_applicable",
                            "reason": case["reason"], "checkpoint_ref": qualify_unit,
                        }
                        audit(stress_unit, "completed", {
                            "status": "not_applicable", "reason": case["reason"],
                        })
                        save()
                        continue
                    if case["case"] == "base":
                        state.cells[stress_unit] = {
                            "unit_id": stress_unit, "kind": "stress", "symbol": symbol,
                            "arm_group": group, "scheme": scheme, "status": "completed",
                            "checkpoint_ref": qualify_unit,
                            "result": deepcopy(qcell.get("result", {})) if qcell else {},
                            "source": "qualification_base",
                        }
                        audit(stress_unit, "cache_reused", {
                            "source": "qualification_base", "kind": "stress",
                        })
                        save()
                        continue

                    def stress_function(case=case, kernel_config=kernel_config,
                                        schedule=schedule, stress_unit=stress_unit):
                        _progress(progress, .70 + .10 * arm_fraction,
                                  f"{symbol} · {group} · {scheme} · 压力 {case['case']}")
                        raw = _run_qualification_kernel(
                            kernel_config, bundle, request.accounts[symbol], schedule,
                            cancelled=cancel,
                            max_principal_drawdown=request.research.max_principal_drawdown,
                            **dict(case["kwargs"]),
                        )
                        return _qualification_summary(
                            raw, item.get("blocks"), checkpoint_ref=stress_unit,
                        ), _qualification_ledgers(raw), 1

                    scell = execute(
                        stress_unit, "stress", symbol, group, scheme,
                        lambda: (lambda value: (value[0], value[1]))(stress_function()),
                        evaluations=1, metadata={
                            "stress_case": case["case"],
                            "stress_kwargs": _json_value(case["kwargs"]),
                            "omitted_contribution_index": case["omitted_contribution_index"],
                        },
                    )

        # Synthetic paths are anchored at the earliest decision that selected
        # each parameter set.  They never read a later market path.
        selected_parameters: dict[str, dict[str, Any]] = {}

        def record_selection(
            params: Mapping[str, Any], source: str, decision: Any,
        ) -> None:
            key = _digest(params)
            value = selected_parameters.setdefault(key, {
                "params": _json_value(params), "sources": [], "decision_date": None,
            })
            value["sources"].append(source)
            stamp = _stamp(decision)
            existing = _stamp(value["decision_date"])
            if stamp is not None and (existing is None or stamp < existing):
                value["decision_date"] = stamp.date().isoformat()

        for arm in arms:
            selection_marker = f"synthetic-selection:{symbol}:{arm['arm_group']}"
            cell = optimization_cells.get((symbol, arm["arm_group"]))
            if not cell:
                continue
            if selection_marker in required_units:
                required_units.remove(selection_marker)
            summary = cell.get("result") or {}
            fixed = summary.get("fixed") or {}
            if isinstance(fixed.get("selected"), Mapping):
                record_selection(
                    fixed["selected"], f"{arm['arm_group']}:fixed",
                    fixed.get("as_of") or
                    (pd.Timestamp(item["evaluation_start"]) - pd.Timedelta(days=1)),
                )
            adaptive = summary.get("adaptive") or {}
            for row in adaptive.get("schedule") or []:
                if row.get("paused") or not isinstance(row.get("params"), Mapping):
                    continue
                record_selection(
                    row["params"],
                    f"{arm['arm_group']}:adaptive:{row.get('effective_date')}",
                    row.get("decision_date") or
                    (pd.Timestamp(row.get("effective_date")) - pd.Timedelta(days=1)),
                )
        template = bundles[symbol]
        for param_hash, selection in selected_parameters.items():
            params = selection["params"]
            decision_date = selection.get("decision_date")
            if decision_date is None:
                warnings.append(f"{symbol} 的已选参数缺少因果决策日，未生成人造路径")
                continue
            for case in SYNTHETIC_CASES:
                synth_unit = f"synthetic:{symbol}:{param_hash}:{case}"
                required_units.append(synth_unit)

                def synthetic_function(
                    case=case, params=params, decision_date=decision_date,
                    synth_unit=synth_unit,
                ):
                    synthetic = synthetic_bundle(
                        symbol, case, template=template,
                        decision_date=decision_date,
                    )
                    synthetic_config = request.research.to_optimization().base.model_copy(update={
                        "symbol": symbol, "start_date": synthetic.calendar[0].date(),
                        "end_date": synthetic.calendar[-1].date(), "frequency": "1d",
                        "data_policy": None,
                    })
                    synthetic_config = _config_with_params(synthetic_config, params)
                    _progress(progress, .82 + .12 * (symbol_index + 1) / total_symbols,
                              f"{symbol} · 人造路径 · {synthetic.metadata.get('synthetic_scenario')}")
                    raw = _run_qualification_kernel(
                        synthetic_config, synthetic, request.accounts[symbol], None,
                        cancelled=cancel,
                        max_principal_drawdown=request.research.max_principal_drawdown,
                    )
                    synthetic_blocks = [{
                        "start": synthetic.calendar[0].date().isoformat(),
                        "end": synthetic.calendar[-1].date().isoformat(),
                        "days": len(synthetic.calendar),
                    }]
                    summary = _qualification_summary(
                        raw, synthetic_blocks, checkpoint_ref=synth_unit,
                    )
                    summary.update({
                        "synthetic": True, "scenario": synthetic.metadata.get("synthetic_scenario"),
                        "real_future_schedule_used": False,
                        "market_store_written": False,
                        "decision_date": decision_date,
                        "anchor_price": synthetic.metadata.get("valuation_price"),
                        "anchor_price_date": synthetic.metadata.get("anchor_price_date"),
                        "price_limit_rate": synthetic.metadata.get("price_limit_rate"),
                    })
                    return summary, _qualification_ledgers(raw), 1

                execute(
                    synth_unit, "synthetic", symbol, None, None,
                    lambda: (lambda value: (value[0], value[1]))(synthetic_function()),
                    evaluations=1, metadata={
                        "scenario": case, "parameter_hash": param_hash,
                        "selected_params": _json_value(params),
                        "selection_sources": _json_value(selection["sources"]),
                        "decision_date": decision_date,
                    },
                )

    # GSS is a cross-sectional ETF screen.  Account-level qualification and
    # stress evidence for stocks remains available, but stocks never enter the
    # ETF statistic call or its cross-section.
    primary_group = next((arm["arm_group"] for arm in arms if arm["primary"]), None)
    dates_by_symbol = {}
    etf_symbols = []
    non_etf_symbols = []
    for item in items:
        symbol = str(item.get("symbol") or "")
        bundle = bundles.get(symbol)
        if item.get("status") == "ready" and bundle is not None:
            if _is_etf_bundle(bundle):
                etf_symbols.append(symbol)
                dates_by_symbol[symbol] = _evaluation_dates(bundle, item)
            else:
                non_etf_symbols.append(symbol)
    gss_applicable = bool(etf_symbols)
    gss_horizon = int(
        plan.get("gss_horizon_days") or request.research.rebalance_days
    )
    gss_calendar = (
        common_decision_anchors(dates_by_symbol, horizon_days=gss_horizon)
        if gss_applicable else {
            "common_dates": [], "anchors": [], "status": "not_applicable",
        }
    )
    common_dates = gss_calendar["common_dates"]
    anchors = gss_calendar["anchors"]
    gss_cells = []
    required_units.extend(f"gss:{anchor['decision_date']}" for anchor in anchors)
    gss_window = int(plan.get("gss_window_days") or min(250, request.research.strategy.training_max_days))
    from . import etf_statistics
    for number, anchor in enumerate(anchors):
        stopped = stop_reason_now()
        if stop_reason or stopped:
            stop_reason = stop_reason or stopped
            break
        stamp = _stamp(anchor["decision_date"])
        gss_unit = f"gss:{anchor['decision_date']}"

        def gss_function(stamp=stamp, anchor=anchor, gss_unit=gss_unit):
            rows, available = {}, {}
            for symbol in etf_symbols:
                bundle = bundles.get(symbol)
                if bundle is None:
                    continue
                available[symbol] = _visible_etf_prefix(bundle, stamp)
                if not available[symbol]:
                    continue
                rows[symbol] = etf_statistics.compute_etf_statistics(
                    bundle, stamp, gss_window,
                )
            prepared = gss_rows_for_date(rows, available_by_symbol=available)
            screens = gss_screens(
                prepared, top_n=3, seed=request.research.seed,
                decision_date=stamp.date().isoformat(),
            )

            outcome_cache: dict[str, dict[str, Any]] = {}

            def selected_outcome(symbol: str) -> dict[str, Any]:
                if symbol in outcome_cache:
                    return deepcopy(outcome_cache[symbol])
                qualification = qualification_cells.get(
                    (symbol, primary_group, "adaptive")
                )
                qualification_result = (qualification or {}).get("result") or {}
                optimization = optimization_cells.get((symbol, primary_group))
                adaptive = ((optimization or {}).get("result") or {}).get("adaptive") or {}
                active = _active_schedule_row(
                    adaptive.get("schedule") or [], stamp,
                )
                no_candidate = (
                    bool(active.get("paused")) if isinstance(active, Mapping) else None
                )
                candidate_state = (
                    "no_qualified" if no_candidate is True else
                    "qualified" if no_candidate is False else "unknown"
                )
                ledger = load_ledger(
                    f"qualification:{symbol}:{primary_group}:adaptive"
                )
                base = {
                    "symbol": symbol, "decision_date": anchor["decision_date"],
                    "outcome_start": anchor["outcome_start"],
                    "outcome_end": anchor["outcome_end"],
                    "no_candidate": no_candidate,
                    "candidate_state": candidate_state,
                    "qualification_status": qualification_result.get("qualification_status"),
                }
                if not isinstance(ledger, Mapping):
                    result = {
                        **base, "status": "missing_outcome",
                        "reason": "required_primary_qualification_ledger_missing",
                        "grid_principal_return": None,
                        "matched_funding_return": None,
                        "paired_excess": None,
                        "max_principal_drawdown": None,
                    }
                    outcome_cache[symbol] = result
                    return deepcopy(result)
                grid = ledger.get("grid") if isinstance(ledger.get("grid"), Mapping) else {}
                hold = (
                    ledger.get("scheduled_funding_hold")
                    if isinstance(ledger.get("scheduled_funding_hold"), Mapping) else {}
                )
                grid_metrics = grid.get("metrics") if isinstance(
                    grid.get("metrics"), Mapping
                ) else {}
                hold_metrics = hold.get("metrics") if isinstance(
                    hold.get("metrics"), Mapping
                ) else {}
                outcome = paired_forward_outcome(
                    grid.get("equity") or [], hold.get("equity") or [],
                    start=anchor["outcome_start"], end=anchor["outcome_end"],
                    days=anchor["outcome_observations"],
                    grid_initial_equity=grid_metrics.get("initial_equity"),
                    grid_initial_principal=grid_metrics.get("initial_principal"),
                    matched_funding_initial_equity=hold_metrics.get("initial_equity"),
                    matched_funding_initial_principal=hold_metrics.get("initial_principal"),
                )
                result = {**base, **outcome}
                outcome_cache[symbol] = result
                return deepcopy(result)

            screen_results = {}
            for name, screen in screens["screens"].items():
                outcomes = [selected_outcome(member["symbol"]) for member in screen["members"]]
                screen_results[name] = {
                    **screen, "selected_outcomes": outcomes,
                    **{
                        key: value for key, value in _screen_distribution(outcomes).items()
                        if key not in {"selected_count", "outcome_count",
                                       "missing_outcome_count", "no_candidate_count",
                                       "no_candidate_ratio", "missing_policy"}
                    },
                    "distribution": _screen_distribution(outcomes),
                    "forward_start": anchor["outcome_start"],
                    "forward_end": anchor["outcome_end"],
                    "horizon_days": anchor["outcome_observations"],
                }
            return {
                **screens, "screens": screen_results,
                "status": "completed",
                "decision_date": anchor["decision_date"],
                "outcome_start": anchor["outcome_start"],
                "outcome_end": anchor["outcome_end"],
                "horizon_days": anchor["outcome_observations"],
                "cohort_sufficient": int(screens["quality_cohort_size"]) >= 2,
                "score_cohort_sufficient": int(screens["score_cohort_size"]) >= 2,
                "required_common_symbols": 2,
                "statistics_evaluations": len(rows),
                "future_missing_policy": "retain_selected_symbol_no_replacement",
            }, None, 0

        _progress(progress, .94 + .05 * (number + 1) / max(1, len(anchors)),
                  f"GSS 共同决策日 {anchor['decision_date']}")
        gcell = execute(
            gss_unit, "gss", "", primary_group, None,
            lambda: (lambda value: (value[0], None))(gss_function()),
        )
        if gcell and gcell.get("status") == "completed":
            gss_cells.append(gcell)

    gss_results = [
        cell.get("result") for cell in gss_cells
        if isinstance(cell.get("result"), Mapping)
    ]
    gss_reasons = []
    if not gss_applicable:
        gss_reasons.append("no_etf_members_in_frozen_request")
        gss_status = "not_applicable"
    else:
        if not anchors:
            gss_reasons.append("insufficient_common_forward_horizon")
        if len(gss_results) != len(anchors):
            gss_reasons.append("incomplete_gss_windows")
        if any(not bool(result.get("cohort_sufficient")) for result in gss_results):
            gss_reasons.append("fewer_than_two_visible_common_symbols")
        if any(not bool(result.get("score_cohort_sufficient")) for result in gss_results):
            gss_reasons.append("fewer_than_two_common_scored_symbols")
        if any(
            any(
                outcome.get("status") != "completed"
                for screen in (result.get("screens") or {}).values()
                if isinstance(screen, Mapping)
                for outcome in screen.get("selected_outcomes") or []
                if isinstance(outcome, Mapping)
            )
            for result in gss_results
        ):
            gss_reasons.append("selected_symbol_forward_outcome_incomplete")
        gss_status = "completed" if anchors and not gss_reasons else "incomplete"
    gss_outcome_distribution = _aggregate_screen_outcomes(gss_results)

    candidate_state_rows = []
    qualification_state_rows = []
    for item in items:
        symbol = str(item.get("symbol") or "")
        if item.get("status") != "ready" or symbol not in bundles:
            continue
        optimization = optimization_cells.get((symbol, primary_group))
        optimization_result = (optimization or {}).get("result") or {}
        if not optimization or not optimization_result.get("business_complete", False):
            candidate_state = "incomplete"
        else:
            candidate_state = _adaptive_candidate_state(optimization_result)
        qualification = qualification_cells.get((symbol, primary_group, "adaptive"))
        qualification_state = _qualification_state(candidate_state, qualification)
        candidate_state_rows.append({"symbol": symbol, "state": candidate_state})
        qualification_state_rows.append({"symbol": symbol, "state": qualification_state})
    candidate_states = [row["state"] for row in candidate_state_rows]
    qualification_states = [row["state"] for row in qualification_state_rows]
    candidate_status = _aggregate_candidate_states(candidate_states)
    qualification_status = _aggregate_qualification_states(qualification_states)
    qualification_incomplete = "incomplete" in qualification_states

    def pending_units() -> list[str]:
        return [unit for unit in required_units
                if (state.cells.get(unit) or {}).get("status") != "completed"
                and (state.cells.get(unit) or {}).get("status") != "not_applicable"]

    if stop_reason is None and pending_units():
        stop_reason = stop_reason_now()
    missing_units = pending_units()
    business_incomplete = []
    for key, cell in optimization_cells.items():
        if not (cell.get("result") or {}).get("business_complete", False):
            reason = (cell.get("result") or {}).get("incomplete_reason")
            entry = {
                "symbol": key[0], "arm_group": key[1],
                "reason": reason,
                "retryable_same_frozen_settings": reason not in _NEW_RESEARCH_SOURCE_REASONS,
                "requires_new_research": reason in _NEW_RESEARCH_SOURCE_REASONS,
            }
            business_incomplete.append(entry)
            if reason == "source_budget_incomplete":
                warnings.append(
                    f"{key[0]} / {key[1]} 的内层固定评价预算不足；提高父任务总预算不能改变该内层上限，请新建研究并调整内层预算设置"
                )
    qualification_problems = []
    for key, cell in qualification_cells.items():
        result = cell.get("result") if isinstance(cell.get("result"), Mapping) else {}
        if result.get("qualification_status") == "unevaluable":
            qualification_problems.append({
                "symbol": key[0], "arm_group": key[1], "scheme": key[2],
                "reason": result.get("unevaluable_reason") or "qualification_unevaluable",
            })
    gss_incomplete = gss_status == "incomplete"
    incomplete = bool(
        stop_reason or missing_units or business_incomplete
        or qualification_incomplete or qualification_problems or gss_incomplete
    )
    source_budget_incomplete = any(
        item.get("reason") == "source_budget_incomplete"
        for item in business_incomplete
    )
    parent_budget_exhausted = stop_reason == "evaluation_budget"
    requires_new_research = any(
        bool(item.get("requires_new_research")) for item in business_incomplete
    )
    if parent_budget_exhausted and source_budget_incomplete:
        budget_incomplete_scope = "parent_and_source"
    elif parent_budget_exhausted:
        budget_incomplete_scope = "parent"
    elif source_budget_incomplete:
        budget_incomplete_scope = "source"
    else:
        budget_incomplete_scope = None
    if parent_budget_exhausted and source_budget_incomplete:
        completion_reason = "parent_and_source_evaluation_budget"
    elif parent_budget_exhausted:
        completion_reason = "parent_evaluation_budget"
    elif source_budget_incomplete:
        completion_reason = "source_budget_incomplete"
    elif stop_reason:
        completion_reason = stop_reason
    elif missing_units:
        completion_reason = "missing_required_units"
    elif business_incomplete:
        completion_reason = "source_incomplete"
    elif qualification_incomplete or qualification_problems:
        completion_reason = "qualification_incomplete"
    elif gss_incomplete:
        completion_reason = "gss_incomplete"
    elif incomplete:
        completion_reason = "incomplete"
    else:
        completion_reason = None
    if stop_reason == "cancelled":
        status = "cancelled"
    elif (stop_reason or missing_units or business_incomplete
          or qualification_incomplete or qualification_problems or gss_incomplete):
        # The job dispatcher exposes resumable work as ``partial``; the public
        # completion contract remains explicit and separate.
        status = "partial"
    elif not items or skipped_symbols or any(item.get("status") == "data_issue" for item in items):
        status = "data_issue"
        incomplete = True
    else:
        status = "completed"

    primary_opt = next((optimization_cells.get((str(item.get("symbol") or ""), primary_group))
                        for item in items
                        if optimization_cells.get((str(item.get("symbol") or ""), primary_group))), None)
    selection_result_id = ((primary_opt or {}).get("result") or {}).get("selection_result_id")
    all_selection_ids = []
    for cell in optimization_cells.values():
        result = cell.get("result") if isinstance(cell.get("result"), Mapping) else {}
        current_id = result.get("selection_result_id")
        if current_id:
            all_selection_ids.append(current_id)
        for attempt in cell.get("retry_history") or []:
            attempt_id = (
                attempt.get("selection_result_id")
                if isinstance(attempt, Mapping) else None
            )
            if attempt_id and attempt_id not in all_selection_ids:
                all_selection_ids.append(attempt_id)
    public_cells = [_cell_public(cell) for cell in
                    sorted(state.cells.values(), key=lambda item: str(item.get("unit_id")))]
    qualification_returns = [
        ((cell.get("result") or {}).get("strategy_metrics") or {}).get("peak_principal_return")
        for (symbol, group, scheme), cell in qualification_cells.items()
        if group == primary_group and scheme == "adaptive"
    ]

    evidence_status = "incomplete" if gss_incomplete else "recorded"
    symbol_rows = []
    for item in items:
        symbol = str(item.get("symbol") or "")
        arm_rows = []
        for arm in arms:
            group = arm["arm_group"]
            opt = (optimization_cells.get((symbol, group)) or {}).get("result") or {}
            schemes = {}
            for scheme in ("fixed", "adaptive"):
                selected = (opt.get("fixed") if scheme == "fixed" else opt.get("adaptive")) or {}
                qualification = ((qualification_cells.get((symbol, group, scheme)) or {})
                                 .get("result") or {})
                schedule = selected.get("schedule", [])
                if not schedule and qualification.get("parameter_schedule"):
                    schedule = qualification["parameter_schedule"]
                schemes[scheme] = {
                    "status": selected.get("status"),
                    "selected": selected.get("selected"),
                    "latest_selected": selected.get("latest_params", selected.get("selected")),
                    "schedule": schedule,
                    "diagnostics": selected.get("diagnostics", {}),
                    "selection_result_id": opt.get("selection_result_id"),
                    "qualification": qualification,
                }
            arm_rows.append({**arm, "schemes": schemes})
        symbol_rows.append({
            "symbol": symbol, "status": item.get("status"),
            "data_version": item.get("data_version"),
            "training_days": item.get("training_days"),
            "evaluation_start": item.get("evaluation_start"),
            "evaluation_end": item.get("evaluation_end"),
            "blocks": _json_value(item.get("blocks") or []),
            "initial_cash": item.get("initial_cash"),
            "candidate_status": next((
                row["state"] for row in candidate_state_rows
                if row["symbol"] == symbol
            ), "incomplete"),
            "qualification_status": next((
                row["state"] for row in qualification_state_rows
                if row["symbol"] == symbol
            ), "incomplete"),
            "arms": arm_rows,
        })

    primary_account_metrics = {}
    if symbol_rows:
        primary_account_metrics = (
            ((qualification_cells.get((symbol_rows[0]["symbol"], primary_group, "adaptive"))
              or {}).get("result") or {}).get("strategy_metrics") or {}
        )
    message = {
        "completed": "完整研究验证完成",
        "partial": "完整研究验证未完成；已完成的单元与账本已保留",
        "cancelled": "完整研究验证已取消；已完成的单元与账本已保留",
        "data_issue": "完整研究验证存在冻结数据问题",
    }.get(status, "完整研究验证结束")

    state.elapsed_seconds = elapsed_now()

    result = {
        "kind": "research_validation",
        "suite_version": request.suite_version,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "status": status,
        "message": message,
        "completion_status": "complete" if not incomplete else "incomplete",
        "completion_reason": completion_reason,
        "stop_reason": stop_reason,
        "plan_hash": plan.get("plan_hash"), "primary_arm": plan.get("primary_arm"),
        "candidate_status": candidate_status,
        "qualification_status": qualification_status,
        "selection_result_id": selection_result_id,
        "selection_result_ids": all_selection_ids,
        "symbols": symbol_rows,
        "cells": public_cells,
        "stress": [cell for cell in public_cells if cell.get("kind") == "stress"],
        "synthetic": [cell for cell in public_cells if cell.get("kind") == "synthetic"],
        "gss": {
            "status": gss_status,
            "scope": "etf_only",
            "applicable": gss_applicable,
            "etf_members": etf_symbols,
            "excluded_symbols": non_etf_symbols,
            "reasons": list(dict.fromkeys(gss_reasons)),
            "decision_dates": [anchor["decision_date"] for anchor in anchors],
            "common_dates": common_dates,
            "common_date_count": len(common_dates),
            "anchors": _json_value(anchors),
            "horizon_days": gss_horizon,
            "required_common_symbols": 2,
            "windows": [_cell_public(cell) for cell in gss_cells],
            "dates": [
                {"unit_id": cell.get("unit_id"), **(cell.get("result") or {})}
                for cell in gss_cells
            ],
            "outcome_distribution": gss_outcome_distribution,
            "historical_universe_verified": False,
            "portfolio_nav_constructed": False,
            "future_missing_policy": "retain_selected_symbol_no_replacement",
        },
        "evidence": {
            "status": evidence_status,
            "gss_status": gss_status,
            "gss_applicable": gss_applicable,
            "gss_etf_members": etf_symbols,
            "gss_excluded_symbols": non_etf_symbols,
            "gss_reasons": list(dict.fromkeys(gss_reasons)),
            "qualification_problems": qualification_problems,
            "candidate_states": candidate_state_rows,
            "qualification_states": qualification_state_rows,
            "qualification_priority": [
                "no_qualified", "not_applicable", "unevaluable",
                "risk_breached", "execution_limited", "qualified",
            ],
            "descriptive_only": True,
            "significance_tests_used": False,
            "historical_universe_verified": False,
            "human_untouched_verified": False,
        },
        "summary": {
            "primary_arm": primary_group,
            "primary_peak_principal_returns": descriptive_summary(qualification_returns),
            "primary_peak_principal_return": primary_account_metrics.get("peak_principal_return"),
            "independent_accounts": independent_account_summary({
                row["symbol"]: ((qualification_cells.get((row["symbol"], primary_group, "adaptive")) or {})
                                .get("result", {}) or {}).get("strategy_metrics", {})
                for row in symbol_rows
            }),
            "evidence_kind": "descriptive",
            "significance_tests_used": False,
            "primary_arm_changed": False,
        },
        "computation": {
            "candidate_evaluations": state.candidate_evaluations,
            "actual_candidate_evaluations": state.candidate_evaluations,
            "ledger_evaluations": state.ledger_evaluations,
            "actual_ledger_evaluations": state.ledger_evaluations,
            "total_evaluations": state.candidate_evaluations + state.ledger_evaluations,
            "statistics_evaluations": state.statistics_evaluations,
            "total_evaluation_budget": budget,
            "planned_total_evaluation_budget": budget,
            "total_evaluation_remaining": (
                max(0, budget - state.candidate_evaluations - state.ledger_evaluations)
                if budget is not None else None
            ),
            "cache_hits": state.cache_hits,
            "missing_units": missing_units,
            "business_incomplete_sources": business_incomplete,
            "budget_incomplete_scope": budget_incomplete_scope,
            "source_budget_incomplete": source_budget_incomplete,
            "parent_budget_exhausted": parent_budget_exhausted,
            "requires_new_research": requires_new_research,
            "qualification_problems": qualification_problems,
            "gss_incomplete": gss_incomplete,
            "elapsed_seconds": state.elapsed_seconds,
        },
        "checkpoint": state.checkpoint_container,
        "audit": events,
        "warnings": list(dict.fromkeys(warnings)),
        "descriptive_only": True,
    }
    _progress(progress, 1.0, "完整研究验证结束")
    return result


__all__ = [
    "run_validation", "materialize_full_validation_result",
    "materialize_stored_validation_result",
]
