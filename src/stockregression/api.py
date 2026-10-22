from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import math
import shutil
import uuid
from contextlib import asynccontextmanager
from itertools import combinations, product
from pathlib import Path, PurePosixPath

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal
from .preview import PreviewManager
from .configuration import (
    ConfigurationConflict,
    configuration_schema as public_configuration_schema,
    current_configuration,
    patch_configuration,
    pending_restart_fields,
    task_configuration_snapshot,
    validate_configuration_patch,
)

from .data import DataService
from .data_diagnostics import derive_source_spans, project_data_diagnostics
from .quality import QUALITY_RULES_VERSION, _closing_auction_placeholders, minute_slots
from .jobs import JobManager
from .models import BacktestConfig, OptimizationConfig, ResearchRequestV5, normalize_symbol
from .market_data import MarketDataService
from .market_data.aggregation import AggregationService
from .market_data.models import DataVersionNotFound, MarketDataError
from .market_data.cache import configure_cache
from .market_data.package import scan_etf_package
from .market_data.preparation import MarketPreparationService
from .market_data.timeframes import normalize_timeframe, public_timeframe
from .performance import effective_cpu_budget, logical_cpu_count
from .rules import instrument_terms
from .research_preparation import preparation_config
from .schedule import configure_schedule
from .settings import ConfigurationDocument, PROJECT_ROOT, Settings, load_settings, resolve_project_path
from .source_health import configured_sources, validate_source_settings
from .storage import CompletionRevisionError, Store, dumps, json_safe, now_iso
from .watchlist_sync import import_watchlist_snapshot


class SyncRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    mode: str = "history"
    scope: Literal["market", "watchlist", "selected"] | None = None
    resume: bool = False
    retry_failed: bool = False
    workers: int = Field(default=4, ge=1, le=4)
    configuration_revision: str | None = None


class MarketSyncRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    frequency: str | None = None
    timeframes: list[str] = Field(default_factory=list)
    mode: Literal["history", "incremental", "daily", "repair"] = "incremental"
    scope: Literal["market", "watchlist", "selected"] | None = None
    start_date: str | None = None
    end_date: str | None = None
    resume: bool = False
    retry_failed: bool = False
    initialize: bool = False
    workers: int = Field(default=4, ge=1, le=4)
    configuration_revision: str | None = None


class ConfigurationPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_revision: str
    patch: dict[str, Any]


class ConfigurationValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patch: dict[str, Any]


class ConfigurationEnvelope(BaseModel):
    config: ConfigurationDocument
    revision: str
    pending_restart: list[str] = Field(default_factory=list)
    active_data_dir: str | None = None
    active_package_dir: str | None = None


class ConfigurationValidationEnvelope(BaseModel):
    valid: Literal[True] = True
    config: ConfigurationDocument
    revision: str


class MarketImportRequest(BaseModel):
    symbol: str
    timeframe: str | None = None
    frequency: str | None = None
    path: str | None = None
    label_convention: Literal["auto", "start", "end"] = "auto"
    adjustment: str = "none"
    etf_category: str | None = None
    calendar_path: str | None = None
    actions_path: str | None = None
    skip_invalid_rows: bool = False
    import_mode: Literal["merge", "snapshot"] = "merge"


class MarketBatchRequest(BaseModel):
    folder: str | None = None
    timeframe: str | None = None
    frequency: str | None = None
    recursive: bool = False
    label_convention: Literal["auto", "start", "end"] = "auto"
    adjustment: str = "none"
    etf_category: str | None = None
    resume: bool = True
    calendar_path: str | None = None
    workers: int = Field(default=4, ge=1, le=4)
    skip_invalid_rows: bool = False
    import_mode: Literal["merge", "snapshot"] = "merge"


class MarketPackageRequest(BaseModel):
    folder: str
    format: Literal["etf", "stock", "package"] = "package"
    timeframes: list[str] | None = None
    symbols: list[str] | None = None
    label_convention: Literal["auto", "start", "end"] = "auto"
    resume: bool = True
    workers: int = Field(default=4, ge=1, le=4)
    skip_invalid_rows: bool = False


class MarketValidateRequest(BaseModel):
    symbol: str
    timeframe: str | None = None
    frequency: str | None = None
    label_convention: Literal["auto", "start", "end"] | None = None
    mode: Literal["incremental", "full"] = "incremental"
    start_date: str | None = None
    end_date: str | None = None


class MarketArchiveRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    frequency: str | None = None


class MarketAggregateRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    source_timeframe: str
    target_timeframe: str
    start_date: str | None = None
    end_date: str | None = None


class MarketPreparationRequest(BaseModel):
    config: BacktestConfig
    version: str | None = None


class MarketSourceProbeRequest(BaseModel):
    source: Literal["sina", "tdx", "ths", "baostock", "eastmoney"]
    capability: Literal["recent_minute", "etf_daily"]
    market: Literal["all", "stock", "etf"] = "all"


class RestartRequest(BaseModel):
    force: bool = False


def _safe_upload_path(name: str) -> Path:
    raw = str(name or "upload.csv").replace("\\", "/")
    pure = PurePosixPath(raw)
    parts = [part for part in pure.parts if part not in {"", "."}]
    if pure.is_absolute() or not parts or any(part == ".." or ":" in part for part in parts):
        raise ValueError("上传文件名包含不安全路径")
    if parts[-1].lower().rsplit(".", 1)[-1] not in {"csv", "parquet"}:
        raise ValueError("仅支持上传 CSV 和 Parquet 文件")
    return Path(*parts)


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                output.write(chunk)
                size += len(chunk)
    finally:
        await upload.close()
    if size == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("上传文件为空")
    return size


class PreviewRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    revision: int = Field(ge=1)
    config: BacktestConfig


class BatchRequest(BaseModel):
    configs: list[BacktestConfig] = Field(min_length=1, max_length=100)


class ResearchCompletionRequest(BaseModel):
    decision: Literal["current", "fixed", "adaptive"] = "current"
    decision_index: int = Field(default=0, ge=0)
    reporting_budget: int = Field(default=1000, ge=0, le=100_000)
    target: Literal["return_leader"] = "return_leader"


_RESULT_ARCHIVE_FIELDS = {
    "candidate_archive", "trial_archive", "candidate_archives", "trial_archives",
    "search_candidate_archive", "search_candidate_archives",
}


def _compact_research_result(value, top_candidates: int, parent_key: str | None = None):
    """Remove research-only matrices/archives while retaining UI summaries."""
    if isinstance(value, dict):
        compact = {}
        for key, item in value.items():
            if key in _RESULT_ARCHIVE_FIELDS:
                continue
            if key == "candidate_ids":
                continue
            if parent_key == "diagnostics" and key == "costs":
                # This is the full [candidate][fold] matrix. Top candidates
                # keep their compact cost_scores, and diagnostics retain the
                # candidate/evaluation counts needed to interpret the result.
                continue
            if key == "candidates" and isinstance(item, list):
                compact[key] = [_compact_research_result(row, top_candidates) for row in item[:top_candidates]]
            else:
                compact[key] = _compact_research_result(item, top_candidates, key)
        return compact
    if isinstance(value, list):
        return [_compact_research_result(item, top_candidates, parent_key) for item in value]
    if isinstance(value, tuple):
        return [_compact_research_result(item, top_candidates, parent_key) for item in value]
    return value


def _landscape_metric(row: dict, sample: str, cost: str, fold: int | None) -> float | None:
    if sample == "training":
        value = row.get("training_excess_return")
    elif fold is not None:
        field = "stress_folds" if cost == "stress" else "folds"
        folds = row.get(field) if isinstance(row.get(field), list) else []
        value = folds[fold].get("excess_return") if fold < len(folds) and isinstance(folds[fold], dict) else None
    else:
        value = row.get("validation_excess_stress" if cost == "stress" else "validation_excess_original")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)):
        return float(value)
    return None


def _landscape_point(row: dict, sample: str, cost: str, fold: int | None, fallback_id: int | None = None) -> dict:
    folds = row.get("stress_folds" if cost == "stress" else "folds", []) if sample == "validation" else []
    folds = folds if isinstance(folds, list) else []
    status = row.get("evaluation_status") if sample == "validation" else row.get("status", row.get("evaluation_status"))
    if sample == "training":
        grid_value, baseline_value = row.get("training_return"), row.get("training_baseline_return")
    elif fold is not None and fold < len(folds) and isinstance(folds[fold], dict):
        evidence_row = folds[fold]
        grid_value, baseline_value = evidence_row.get("grid_return"), evidence_row.get("baseline_return")
    else:
        grid_value = row.get("validation_return_stress" if cost == "stress" else "validation_return_original")
        baseline_value = None
    return {
        "candidate_id": row.get("candidate_id", row.get("search_id", fallback_id)),
        "params": row.get("params", {}), "is_center": bool(row.get("is_center")),
        "is_baseline": bool(row.get("is_baseline")), "is_focus": False,
        "status": status or "unknown",
        "excess_return": _landscape_metric(row, sample, cost, fold),
        "grid_return": grid_value, "baseline_return": baseline_value,
        "drawdown": row.get("max_principal_drawdown") if sample == "validation" else row.get("max_principal_drawdown"),
        "trade_count": row.get("trade_count") if sample == "validation" else row.get("training_trade_count", row.get("trade_count")),
        "peak_invested": row.get("peak_invested"),
        "neighbor_candidate_ids": row.get("neighbor_candidate_ids", []),
        "neighbor_of_candidate_ids": row.get("neighbor_of_candidate_ids", []),
        "robustness_status": row.get("robustness_status"),
        "neighbor_positive_ratio": row.get("neighbor_positive_ratio", row.get("pos_win_rate")),
        "neighbor_q25_excess": row.get("neighbor_q25_excess"),
        "j_score": row.get("j_score"),
        "neighborhood_mean": row.get("neighborhood_mean"),
        "downside_risk_tau": row.get("downside_risk_tau"),
        "pos_penalty": row.get("pos_penalty"),
        "neighborhood_size": row.get("neighborhood_size"),
        "neighborhood_support_size": row.get("neighborhood_support_size"),
        "pos_win_rate": row.get("pos_win_rate"),
        "safe_win_rate": row.get("safe_win_rate"),
        "score_complete": row.get("score_complete"),
        "score_missing_count": row.get("score_missing_count"),
        "island_suspect": row.get("island_suspect", False),
        "neighbor_boundary_clipped_count": row.get("neighbor_boundary_clipped_count", 0),
        "region_ids": row.get("region_ids", []),
        "is_stable_point": bool(row.get("is_stable_point")),
        "is_peak_point": bool(row.get("is_peak_point")),
        "stability_margin": row.get("stability_margin"),
        "constraint_eligible": row.get("constraint_eligible"),
        "constraint_reasons": row.get("constraint_reasons", []),
        **{name: row.get(name) for name in (
            "actual_min_pos_ratio", "actual_max_pos_ratio", "neighborhood_max_eta",
            "stockout_count", "tplus1_block_count", "buy_count", "sell_count",
            "floor_episode_count", "floor_episode_grids", "floor_penalty", "floor_penalty_basis",
        )},
    }


def _robust_return_key(row: dict) -> tuple:
    value = _landscape_metric(row, "training", "original", None)
    return (-(value if value is not None else -math.inf),
            _landscape_param_key(row["params"], ["buy_step", "sell_step", "buy_shares", "sell_shares"]))


def _robust_landscape_ranking(archive: list[dict], config: dict, selection: dict) -> dict:
    """Rank complete J scores independently of execution/peak eligibility."""
    strategy = config.get("strategy", {})
    count = max(1, min(100, int(strategy.get("recommendation_count", 10))))
    fields = ["buy_step", "sell_step", "buy_shares", "sell_shares"]
    regions = {
        _landscape_param_key(region.get("center_params", {}), fields): region
        for region in selection.get("recommendation_regions", [])
    }

    def finite(row, name, fallback=0.0):
        value = row.get(name)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else fallback

    ranked = sorted(
        (row for row in archive if row.get("score_complete") is not False
         and finite(row, "j_score", None) is not None),
        key=lambda row: (
            -finite(row, "j_score"), -finite(row, "neighborhood_mean"),
            finite(row, "downside_risk_tau"), finite(row, "pos_penalty"),
            -finite(row, "training_excess_return"),
            _landscape_param_key(row["params"], fields),
        ),
    )
    points = []
    for rank, row in enumerate(ranked[:count], 1):
        point = _landscape_point(row, "training", "original", None)
        reasons = []
        if finite(row, "j_score") <= 0:
            reasons.append("J 未大于 0")
        if finite(row, "max_principal_drawdown", math.inf) > config.get("max_principal_drawdown", .3):
            reasons.append("回撤超过约束")
        if finite(row, "trade_count") < config.get("min_trades", 10):
            reasons.append("成交笔数不足")
        if finite(row, "buy_count") <= 0 or finite(row, "sell_count") <= 0:
            reasons.append("未同时发生买入和卖出")
        hard_cap = strategy.get("eta_hard_cap", 3.5)
        if hard_cap is not None and finite(row, "actual_max_pos_ratio", math.inf) > hard_cap:
            reasons.append("实际持仓超过约束")
        region = regions.get(_landscape_param_key(row["params"], fields), {})
        point.update({
            "rank": rank, "constraint_eligible": not reasons, "constraint_reasons": reasons,
            "local_peak_verified": region.get("local_peak_verified", False),
            "platform_size": region.get("platform_size"), "stress": region.get("stress", {}),
        })
        points.append(point)
    diagnostics = selection.get("diagnostics", {})
    return {
        "j_focus": points[0] if points else None, "j_ranking": points,
        "j_candidate_count": len(ranked), "recommendation_count": count,
        "decision_as_of": selection.get("as_of", selection.get("decision_date")),
        "training_start": diagnostics.get("training_start"),
        "training_end": diagnostics.get("training_end"),
        "coverage_status": diagnostics.get("coverage_status"),
    }


def _landscape_rank_key(row: dict, sample: str, cost: str, fold: int | None, index: int) -> tuple[float, int]:
    value = _landscape_metric(row, sample, cost, fold)
    candidate_id = row.get("candidate_id", row.get("search_id", index + 1))
    try:
        identity = int(candidate_id)
    except (TypeError, ValueError):
        identity = index + 1
    return (value if value is not None else -math.inf, -identity)


def _landscape_sample_indices(total: int, limit: int, focus: int | None) -> list[int]:
    if total <= 0:
        return []
    if limit >= total:
        return list(range(total))
    if limit == 1:
        return [focus if focus is not None else 0]
    indices = {round(index * (total - 1) / (limit - 1)) for index in range(limit)}
    if focus is not None and focus not in indices:
        indices.add(focus)
        if len(indices) > limit:
            removable = sorted(value for value in indices if value not in (0, total - 1, focus))
            if removable:
                indices.remove(removable[len(removable) // 2])
            else:
                indices.remove(total - 1 if total - 1 != focus else 0)
    return sorted(indices)[:limit]


def _landscape_same(left, right) -> bool:
    if left is None and right is None:
        return True
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return left == right


def _landscape_slices(
    archive: list[dict], focus_params: dict, names: list[str],
    sample: str, cost: str, fold: int | None, limit: int = 2500,
    totals: dict[str, int] | None = None,
) -> dict[str, dict]:
    """Return complete two-dimensional planes through the relative-best point.

    A neighborhood slice is complete: every legal candidate in that plane is
    available in the result archive, so blank heatmap cells should only mean an
    unevaluated or invalid combination.  The plane size comes from the archive
    query's SQL count, not from the number of rows that were materialised for
    the response, and a truncated plane is reported as sampled.
    """
    output = {}
    for x_name, y_name in combinations(names, 2):
        fixed = [name for name in names if name not in (x_name, y_name)]
        rows = [
            row for row in archive
            if all(_landscape_same(row.get("params", {}).get(name), focus_params.get(name)) for name in fixed)
        ]
        if not rows:
            continue
        focus_index = next((
            index for index, row in enumerate(rows)
            if all(_landscape_same(row.get("params", {}).get(name), focus_params.get(name)) for name in names)
        ), None)
        if len(rows) > limit:
            indices = _landscape_sample_indices(len(rows), limit, focus_index)
            sampled = True
        else:
            indices = list(range(len(rows)))
            sampled = False
        points = []
        for index in indices:
            point = _landscape_point(rows[index], sample, cost, fold, index + 1)
            point["is_focus"] = index == focus_index
            points.append(point)
        key = "|".join(sorted((x_name, y_name)))
        total = int((totals or {}).get(key, len(rows)))
        output[key] = {
            "x": x_name, "y": y_name, "points": points,
            "total": total, "sampled": bool(sampled or total > len(points)),
        }
    return output


def _landscape_param_key(params: dict, names: list[str]) -> tuple:
    values = []
    for name in names:
        value = params.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(round(float(value), 9))
        else:
            values.append(value)
    return tuple(values)


def _research_decision_context(result: dict, decision: str, index: int = 0) -> dict:
    """Resolve one v5 selection without consulting or recomputing market data."""
    if decision == "current":
        return ((result.get("fixed") or {}).get("current") or {})
    if decision == "fixed":
        return result.get("fixed") or {}
    decisions = ((result.get("evidence") or {}).get("decisions") or [])
    if index >= len(decisions):
        raise LookupError("研究决策不存在")
    return decisions[index]


def _landscape_neighborhood(
    archive: list[dict], focus_params: dict, names: list[str],
    sample: str, cost: str, fold: int | None, fields: list[str] | None = None,
) -> dict:
    """Summarize the immediate full-dimensional neighbor box for the focus."""
    axes = []
    selected = set(names if fields is None else fields)
    for name in names:
        if name not in selected:
            continue
        values = sorted({
            float(row["params"][name])
            for row in archive
            if isinstance(row.get("params", {}).get(name), (int, float))
        })
        if len(values) < 2 or not isinstance(focus_params.get(name), (int, float)):
            continue
        center = float(focus_params[name])
        index = min(range(len(values)), key=lambda item: abs(values[item] - center))
        choices = values[max(0, index - 1):index + 2]
        if center not in choices:
            choices = sorted({*choices, center})
        axes.append((name, choices[:3]))
    if not axes:
        return {"available": False, "expected_count": 0, "evaluated_count": 0}
    lookup = {_landscape_param_key(row["params"], names): row for row in archive}
    expected = math.prod(len(choices) for _, choices in axes) - 1
    values = []
    completed = 0
    missing = 0
    metric_missing = 0
    for combination in product(*(choices for _, choices in axes)):
        if all(
            abs(combination[index] - float(focus_params.get(name, combination[index]))) < 1e-9
            for index, (name, _) in enumerate(axes)
        ):
            continue
        params = {**focus_params, **{
            name: combination[index] for index, (name, _) in enumerate(axes)
        }}
        row = lookup.get(_landscape_param_key(params, names))
        if row is None:
            missing += 1
            continue
        completed += 1
        value = _landscape_metric(row, sample, cost, fold)
        if value is not None:
            values.append(value)
        else:
            metric_missing += 1
    return {
        "available": True,
        "dimensions": [name for name, _ in axes],
        "expected_count": expected,
        "evaluated_count": completed,
        "missing_count": missing,
        "metric_count": len(values),
        "metric_missing_count": metric_missing,
        "metric_available": bool(values),
        "positive_count": sum(value > 1e-12 for value in values),
        "positive_ratio": sum(value > 1e-12 for value in values) / len(values) if values else None,
        "q25_excess": float(np.percentile(values, 25)) if values else None,
        "min_excess": min(values) if values else None,
        "max_excess": max(values) if values else None,
        "all_positive": bool(values) and len(values) == expected and all(value > 1e-12 for value in values),
    }


def _robust_training_landscape(
    store: Store, result: dict, result_id: str, *, decision: str,
    decision_index: int, region_id: int | None, offset: int, limit: int,
    revision: int | None,
) -> dict[str, Any]:
    """Read a training landscape through storage-layer queries only."""
    group = "search_candidate_archives"
    stats = store.candidate_stats(
        result, group, decision, decision_index, region_id=region_id, revision=revision,
    )
    total = int(stats.get("total", 0))
    if not total:
        return {
            "result_id": result_id, "decision": decision, "decision_index": decision_index,
            "revision": revision, "sample": "training", "cost": "original", "fold": None,
            "region_id": region_id, "offset": offset, "limit": limit, "total": 0,
            "points": [], "bounds": None, "focus": None, "focus_neighborhood": None,
            "top_candidates": [], "slices": {}, "sampled": False, "center_count": 0,
        }

    def query(**kwargs):
        return store.candidate_archive(
            result, group, decision, decision_index, revision=revision, **kwargs,
        )["rows"]

    focus_rows = query(
        region_id=region_id, sort_by="training_excess_return", descending=True, limit=1,
    )
    if not focus_rows:
        raise LookupError("候选档案为空")
    focus_row = focus_rows[0]
    ordinals = store.candidate_ordinals(
        result, group, decision, decision_index, region_id=region_id, revision=revision,
    )
    focus_ordinal = int(focus_row.get("_archive_ordinal", ordinals[0]))
    focus_position = ordinals.index(focus_ordinal) if focus_ordinal in ordinals else 0
    if offset > 0:
        selected_ordinals = ordinals[offset:min(total, offset + limit)]
        sampled = False
    else:
        positions = _landscape_sample_indices(total, limit, focus_position)
        selected_ordinals = [ordinals[position] for position in positions]
        sampled = len(positions) < total

    subset: dict[int, dict] = {}

    def remember(rows: list[dict]) -> None:
        for row in rows:
            subset[int(row.get("_archive_ordinal", -1))] = row

    selected_rows = query(
        region_id=region_id, ordinals=selected_ordinals,
        sort_by="ordinal", limit=max(1, len(selected_ordinals)),
    )
    remember(selected_rows)
    remember([focus_row])
    names = ["buy_step", "sell_step", "buy_shares", "sell_shares"]
    focus_params = focus_row.get("params", {})

    # Fetch only each two-dimensional plane through the focus.  The storage
    # query performs the fixed-axis filter; rows outside these planes are never
    # decoded for this response.
    plane_totals: dict[str, int] = {}
    for x_name, y_name in combinations(names, 2):
        fixed = {
            name: focus_params[name]
            for name in names if name not in (x_name, y_name)
            and isinstance(focus_params.get(name), (int, float))
        }
        remember(query(
            region_id=region_id, filters=fixed, sort_by="ordinal", limit=2501,
        ))
        # The plane size is counted by SQLite; the materialised rows above are
        # truncated and must not be reported as the complete plane.
        plane_totals["|".join(sorted((x_name, y_name)))] = store.candidate_total(
            result, group, decision, decision_index, region_id=region_id,
            filters=fixed, revision=revision,
        )

    axis_values = store.candidate_axis_values(
        result, group, decision, decision_index, names=names, revision=revision,
    )
    neighborhood_choices = {}
    for name in names:
        values = axis_values.get(name, [])
        center = focus_params.get(name)
        if len(values) < 2 or not isinstance(center, (int, float)):
            continue
        index = min(range(len(values)), key=lambda item: abs(values[item] - float(center)))
        neighborhood_choices[name] = values[max(0, index - 1):index + 2]
    if neighborhood_choices:
        remember(query(
            region_id=region_id, in_filters=neighborhood_choices,
            sort_by="ordinal", limit=256,
        ))

    top_candidates = []
    for rank, row in enumerate(query(
        region_id=region_id, sort_by="training_excess_return", descending=True, limit=10,
    ), 1):
        point = _landscape_point(row, "training", "original", None)
        point["is_focus"] = int(row.get("_archive_ordinal", -1)) == focus_ordinal
        point["rank"] = rank
        top_candidates.append(point)
        remember([row])

    config = result.get("config", {})
    strategy = config.get("strategy") if isinstance(config, dict) else {}
    recommendation_count = max(1, min(100, int((strategy or {}).get("recommendation_count", 10))))
    ranking_rows = query(
        region_id=region_id, filters={"score_complete": 1},
        sort_by="j_score", descending=True,
        limit=min(1000, max(32, recommendation_count * 8)),
    )
    remember(ranking_rows)
    selection = _research_decision_context(result, decision, decision_index)
    ranking = _robust_landscape_ranking(ranking_rows, config, selection)
    ranking["j_candidate_count"] = int(stats.get("j_complete", 0))

    rows = list(subset.values())
    points = []
    for row in selected_rows:
        point = _landscape_point(row, "training", "original", None)
        point["is_focus"] = int(row.get("_archive_ordinal", -1)) == focus_ordinal
        points.append(point)
    focus = next((point for point in points if point["is_focus"]), None)
    if focus is None:
        focus = _landscape_point(focus_row, "training", "original", None)
        focus["is_focus"] = True
        points.append(focus)
    slices = _landscape_slices(
        rows, focus_params, names, "training", "original", None, totals=plane_totals,
    )
    neighborhood_fields = [
        str(name) for name in ((strategy or {}).get("neighborhood_fields") or names)
        if name in names
    ] or None
    focus_neighborhood = _landscape_neighborhood(
        rows, focus_params, names, "training", "original", None, fields=neighborhood_fields,
    )
    finite = [
        float(point["excess_return"]) for point in points
        if isinstance(point.get("excess_return"), (int, float))
        and np.isfinite(float(point["excess_return"]))
    ]
    return {
        "result_id": result_id, "decision": decision, "decision_index": decision_index,
        "revision": revision, "sample": "training", "cost": "original", "fold": None,
        "region_id": region_id, "offset": offset, "limit": limit, "total": total,
        "parameters": names, "points": points,
        "bounds": {"min": min(finite), "max": max(finite)} if finite else None,
        "center_count": sum(bool(point["is_center"]) for point in points),
        "focus": focus, "focus_neighborhood": focus_neighborhood,
        "top_candidates": top_candidates, "slices": slices,
        "sampled": sampled, "sample_strategy": "even" if sampled else "page",
        **ranking,
        "completion": {
            "status": "available" if int(stats.get("j_complete", 0)) < total else "complete",
            "endpoint": f"/api/v1/results/{result_id}/completion",
        },
    }


def _quality_dates(result: dict) -> tuple[str | None, str | None]:
    quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
    days = [str(row["date"])[:10] for row in quality.get("days", [])
            if isinstance(row, dict) and row.get("date")]
    if days:
        return min(days), max(days)
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    base = config.get("base") if isinstance(config.get("base"), dict) else config
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    return (diagnostics.get("checked_start") or base.get("start_date"),
            diagnostics.get("checked_end") or result.get("effective_end_date") or base.get("end_date"))


def _result_data_metadata(store: Store, result: dict) -> dict | None:
    """Read provenance for the result's locked snapshot, without market fetches.

    Only a small projection of immutable Parquet columns is read once per
    version/date range. This does not re-run minute validation or replace the
    original result's quality report with today's active snapshot.
    """
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    base = config.get("base") if isinstance(config.get("base"), dict) else config
    symbol, version = result.get("symbol") or base.get("symbol"), result.get("data_version")
    if not symbol or not version:
        return None
    manifest = store.snapshot(symbol, version)
    if not manifest:
        return None
    start, end = _quality_dates(result)
    key = f"data-source-spans:v2:q{QUALITY_RULES_VERSION}:{version}:{start or ''}:{end or ''}"
    cached = store.state(key)
    if not isinstance(cached, dict) or not isinstance(cached.get("spans"), list):
        frames = []
        if manifest.get("market_data"):
            market_versions = manifest.get("market_versions", {})
            minute_name = next(
                (name for name in ("1min", "5min", "15min", "30min", "60min")
                 if name in market_versions), None,
            )
            if minute_name:
                frames.append(MarketDataService(store).repository.read_frame(
                    symbol, minute_name,
                    version=int(market_versions[minute_name]["version"]),
                    start=start, end=end,
                    fields=("timestamp", "source", "source_verified", "is_synthetic", "close"),
                ))
        else:
            import pyarrow.parquet as pq
            for relative in manifest.get("files", {}).get("minute", []):
                path = store.root / relative
                names = set(pq.read_schema(path).names)
                columns = [
                    name for name in ("timestamp", "source", "source_verified", "is_synthetic", "close")
                    if name in names
                ]
                if "timestamp" in columns:
                    frames.append(pd.read_parquet(path, columns=columns))
        minute = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        spans = derive_source_spans(minute, manifest.get("profiles"), start, end)
        auction_by_day = {}
        if not minute.empty and "close" in minute:
            stamps = pd.to_datetime(minute.timestamp)
            if start:
                minute = minute[stamps >= pd.Timestamp(start)]
            if end:
                minute = minute[pd.to_datetime(minute.timestamp) < pd.Timestamp(end) + pd.Timedelta(days=1)]
            if "source_verified" not in minute and "source" in minute:
                minute = minute.assign(source_verified=minute.source.map(
                    lambda source: bool(manifest.get("profiles", {}).get(source, {}).get("verified", False))))
            for day, group in minute.groupby(pd.to_datetime(minute.timestamp).dt.normalize()):
                if len(group) >= 240:
                    continue
                indexed = group.drop_duplicates("timestamp", keep="last").set_index("timestamp").reindex(minute_slots(day))
                count = int(_closing_auction_placeholders(manifest["symbol"], day, indexed, indexed.close.isna().to_numpy()).sum())
                if count:
                    auction_by_day[str(day.date())] = count
        cached = {"spans": spans, "auction_placeholders_by_day": auction_by_day}
    metadata = dict(manifest.get("metadata") or {})
    metadata["diagnostics"] = {**metadata.get("diagnostics", {}), "source_spans": cached["spans"]}
    metadata["_snapshot_profiles"] = manifest.get("profiles", {})
    metadata["_auction_placeholders_by_day"] = cached.get("auction_placeholders_by_day", {})
    return metadata


def _project_result_quality(result: dict, data_metadata: dict | None) -> dict:
    quality = result.get("quality")
    if not isinstance(quality, dict):
        return result
    start, end = _quality_dates(result)
    metadata = dict(data_metadata or {})
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    base = config.get("base") if isinstance(config.get("base"), dict) else config
    if "requested_frequency" not in quality and base.get("frequency"):
        quality = {**quality, "requested_frequency": base["frequency"]}
    # Old results remain immutable. Their display projection can distinguish
    # known auction grid slots using the exact locked raw snapshot, without
    # recomputing any historical fills, strategy metrics or parameter choices.
    auction_by_day = metadata.get("_auction_placeholders_by_day", {})
    if auction_by_day and isinstance(quality.get("days"), list):
        projected_days = []
        for row in quality["days"]:
            row = dict(row)
            count = int(auction_by_day.get(str(row.get("date"))[:10], 0))
            if count and row.get("mode") == "minute" and "missing_data_minutes" not in row:
                row["auction_placeholders"] = min(count, int(row.get("missing_minutes", 0)))
                row["missing_data_minutes"] = max(0, int(row.get("missing_minutes", 0)) - row["auction_placeholders"])
                if not row["missing_data_minutes"] and str(row.get("reason", "")).startswith("少量缺口"):
                    row["reason"] = "收盘集合竞价无逐分钟成交，占位保留"
            projected_days.append(row)
        quality = {**quality, "days": projected_days,
                   "auction_placeholders": sum(int(row.get("auction_placeholders", 0)) for row in projected_days if row.get("mode") == "minute")}
    own_diagnostics = result.get("diagnostics")
    if isinstance(own_diagnostics, dict) and own_diagnostics.get("schema_version") == 1:
        metadata["diagnostics"] = {**metadata.get("diagnostics", {}), **own_diagnostics}
    metadata["warnings"] = list(dict.fromkeys([*metadata.get("warnings", []), *result.get("warnings", [])]))
    metadata["source_events"] = [*metadata.get("source_events", []), *result.get("source_events", [])]
    metadata["requested_end_date"] = result.get("requested_end_date") or base.get("end_date")
    # Clean the quality warning list separately, so execution/parameter notices
    # do not become quality failures while archived provider errors disappear.
    quality_projection = project_data_diagnostics({}, quality, start, end)
    metadata["source_events"].extend(quality_projection["source_events"])
    quality = {**quality, "warnings": quality_projection["current_warnings"]}
    projection = project_data_diagnostics(metadata, quality, start, end, metadata.get("_snapshot_profiles"))
    return {**result, "quality": quality, "warnings": projection["current_warnings"],
            "diagnostics": projection["diagnostics"], "source_events": projection["source_events"]}


def public_result(result: dict, full: bool = False, data_metadata: dict | None = None) -> dict:
    """Send UI-sized results by default, preserving raw data for full requests."""
    if full:
        return result
    result = dict(result)
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    is_v5_research = (
        result.get("protocol_version") == 5
        and result.get("result_schema_version") == 2
        and config.get("protocol_version") == 5
    )
    if is_v5_research:
        try:
            top_candidates = max(1, min(100, int(config.get("top_candidates", 20))))
        except (TypeError, ValueError):
            top_candidates = 20
        result = _compact_research_result(result, top_candidates)
    result = _project_result_quality(result, data_metadata)
    if "strategies" in result:
        result["equity_frequency"] = "1d"
        strategies = {}
        for name, raw_strategy in result["strategies"].items():
            strategy = dict(raw_strategy)
            daily = {}
            for row in strategy.get("equity", []):
                daily[str(row.get("timestamp", ""))[:10]] = row
            strategy["equity"] = list(daily.values())
            strategy["equity_frequency"] = "1d"
            strategies[name] = strategy
        result["strategies"] = strategies
    for mode in ("fixed", "adaptive"):
        if result.get(mode) and result[mode].get("test_result"):
            result[mode] = {**result[mode], "test_result": public_result(result[mode]["test_result"], data_metadata=data_metadata)}
    return result


def _stream_json_value(value: Any, store: Store):
    """Serialize a result while streaming candidate archives from SQLite."""
    if isinstance(value, dict) and isinstance(value.get("__candidate_archive__"), str):
        yield "["
        first = True
        for row in store.iter_candidate_reference(value):
            if not first:
                yield ","
            yield dumps(row)
            first = False
        yield "]"
        return
    if isinstance(value, dict):
        yield "{"
        first = True
        for key, item in value.items():
            if not first:
                yield ","
            yield json.dumps(str(key), ensure_ascii=False)
            yield ":"
            yield from _stream_json_value(item, store)
            first = False
        yield "}"
        return
    if isinstance(value, (list, tuple)):
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from _stream_json_value(item, store)
        yield "]"
        return
    yield dumps(value)


def _with_data_availability(store: Store, result: dict) -> dict:
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    base = config.get("base") if isinstance(config.get("base"), dict) else config
    symbol = result.get("symbol") or base.get("symbol")
    version = result.get("data_version")
    if not symbol or not version:
        return result
    available = store.snapshot(symbol, version, persist=False) is not None
    return {
        **result, "data_version_available": available,
        "data_version_warning": None if available else "原数据版本已清理，可查看历史结果，但不能按原版本重算",
    }


def _require_supported_result(result: dict) -> None:
    protocol = result.get("protocol_version")
    if protocol is None:
        return
    if protocol == 5 and result.get("result_schema_version") == 2:
        return
    raise HTTPException(410, {
        "code": "UNSUPPORTED_RESULT_VERSION",
        "message": "该历史研究结果产生于已移除的协议版本，不支持查看或导出",
    })


def _revision_scope(revision: dict | None) -> dict | None:
    """Describe which decision branch a completion revision covers."""
    if not isinstance(revision, dict):
        return None
    return {
        "revision": int(revision.get("revision", 0) or 0),
        "decision": revision.get("decision"),
        "decision_index": int(revision.get("decision_index", 0) or 0),
    }


def _completion_state(result_id: str, result: dict, revisions: list[dict],
                      selected_revision: dict | None) -> dict:
    """Describe the revision selected for one result read."""
    if selected_revision is not None:
        revision_status = selected_revision.get("status")
        completion_status = (
            "complete" if revision_status == "completed"
            and int(selected_revision.get("remaining_count", 0) or 0) == 0
            else "available" if revision_status in {"partial", "budget_exhausted", "cancelled"}
            else str(revision_status or "available")
        )
    elif result.get("protocol_version") == 5:
        completion_status = str(result.get("completion_status") or "available")
    else:
        completion_status = "not_applicable"
    return {
        "latest_revision": revisions[-1] if revisions else None,
        "revision": selected_revision,
        "selected_revision": selected_revision.get("revision") if selected_revision else None,
        "revision_scope": _revision_scope(selected_revision),
        "status": completion_status,
        "endpoint": f"/api/v1/results/{result_id}/completion",
    }


def _csv_value(value: Any) -> Any:
    """Render one ledger value for CSV output."""
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _revision_conflict(exc: CompletionRevisionError) -> HTTPException:
    """Turn an unusable completion revision into a clear client error."""
    return HTTPException(409, {
        "code": exc.code,
        "message": str(exc),
        "revision": exc.revision,
        "scope": exc.scope,
    })


def _revision_substituted_result(result: dict, revision: dict | None) -> dict:
    """Resolve the branch covered by a completion revision to its merged archive.

    The three archive groups of a decision hold the same candidate rows, so a
    revision is applied to every one of them for the covered decision only.
    The other branches keep their original archive and the export states which
    branch was revised.
    """
    scope = _revision_scope(revision)
    merged = revision.get("merged_candidate_archive") if isinstance(revision, dict) else None
    evidence = result.get("evidence")
    if scope is None:
        return result
    if not isinstance(merged, dict) or not isinstance(evidence, dict):
        raise CompletionRevisionError(
            f"补算修订 {scope['revision']} 没有可用的候选档案",
            code="COMPLETION_REVISION_ARCHIVE_MISSING", revision=scope["revision"], scope=scope,
        )
    updated = {**result, "evidence": {**evidence}}
    applied = False
    for group_name, field in (
        ("candidate_archives", "candidate_archive"),
        ("trial_archives", "trial_archive"),
        ("search_candidate_archives", "search_candidate_archive"),
    ):
        group = evidence.get(group_name)
        if not isinstance(group, dict):
            continue
        new_group = dict(group)
        if scope["decision"] == "adaptive":
            entries = group.get("adaptive")
            if isinstance(entries, list) and scope["decision_index"] < len(entries):
                entry = entries[scope["decision_index"]]
                if isinstance(entry, dict):
                    new_entry = dict(entry)
                    if isinstance(new_entry.get(field), dict):
                        new_entry[field] = merged
                        applied = True
                    new_group["adaptive"] = [
                        *entries[:scope["decision_index"]], new_entry, *entries[scope["decision_index"] + 1:],
                    ]
        else:
            branch = group.get(scope["decision"])
            if isinstance(branch, dict) and "__candidate_archive__" in branch:
                new_group[scope["decision"]] = merged
                applied = True
        updated["evidence"][group_name] = new_group
    if not applied:
        raise CompletionRevisionError(
            f"补算修订 {scope['revision']} 覆盖的决策 {scope['decision']} "
            f"#{scope['decision_index']} 没有候选档案",
            code="COMPLETION_REVISION_OUT_OF_SCOPE", revision=scope["revision"], scope=scope,
        )
    updated["export_revision"] = {
        **scope,
        "archive": "merged_candidate_archive",
        "applied_branches": [f"{scope['decision']}#{scope['decision_index']}"],
    }
    return updated


def create_app(data_dir: str | None = None, start_jobs: bool = True, *, openapi_only: bool = False):
    configured = load_settings() if not openapi_only else Settings()
    root = str(Path(data_dir or configured.data_dir).resolve())
    configure_cache(configured.market_cache_mb)
    store = None if openapi_only else Store(root)
    watchlist_snapshot = PROJECT_ROOT / "config" / "watchlist.json" if data_dir is None else None

    def refresh_watchlist_snapshot():
        if watchlist_snapshot is not None:
            try:
                import_watchlist_snapshot(store, watchlist_snapshot)
            except (OSError, ValueError) as exc:
                logging.getLogger(__name__).warning("自选快照导入失败，保留本地列表：%s", exc)

    if store is not None:
        refresh_watchlist_snapshot()
    initial_package = (
        str(resolve_project_path(configured.package_dir))
        if data_dir is None and configured.package_dir else None
    )
    if openapi_only:
        package_dir = str(resolve_project_path(configured.package_dir)) if configured.package_dir else root
    else:
        initial_market = MarketDataService(store, package_dir=initial_package)
        package_dir = str(initial_market.repository.package_dir)
    active_max_workers = int(configured.max_workers)
    active_scheduler_poll_seconds = float(configured.market_data.scheduler_poll_seconds)
    manager = JobManager(root, active_max_workers, active_scheduler_poll_seconds)
    previews = PreviewManager(store)
    instance_id = uuid.uuid4().hex

    @asynccontextmanager
    async def lifespan(app):
        if start_jobs:
            manager.start()
        yield
        previews.stop()
        if start_jobs:
            manager.stop()

    app = FastAPI(title="A 股及 ETF 行情与网格研究台", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.package_dir = package_dir
    app.state.previews = previews
    app.state.instance_id = instance_id
    app.state.restart_callback = None
    app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

    def submit(kind, payload, dedupe_key=None, *, include_created=False):
        frozen = dict(payload)
        snapshot = task_configuration_snapshot(
            active_data_dir=root,
            active_package_dir=package_dir,
            active_host=configured.host,
            active_port=configured.port,
        )
        requested_revision = str(frozen.get("configuration_revision") or "")
        # A resumed task replays the frozen configuration snapshot of the
        # original submission, so a later configuration save must not block it.
        if (
            requested_revision
            and not frozen.get("resume_from_job")
            and requested_revision != snapshot.get("revision")
        ):
            raise HTTPException(409, detail={
                "code": "CONFIG_REVISION_CONFLICT",
                "message": "研究提交所使用的配置版本已过期，请等待配置保存后重试",
            })
        frozen.setdefault("configuration_revision", snapshot.get("revision"))
        frozen.setdefault("configuration_snapshot", snapshot)
        if kind == "optimization" and "submission_provenance" not in frozen:
            from .reproducibility import computation_provenance
            from .settings import PROJECT_ROOT
            frozen["submission_provenance"] = computation_provenance(
                PROJECT_ROOT, configuration=frozen.get("config"),
            )
        job_id, created = store.new_job_info(kind, frozen, dedupe_key=dedupe_key)
        return {
            "job_id": job_id,
            **({"created": created} if include_created else {}),
        }

    def require_ready(config):
        manifest = store.snapshot(config.symbol)
        if manifest and manifest.get("market_data"):
            readiness = MarketDataService(store).readiness(config)
            if not readiness["ready"]:
                raise HTTPException(409, detail={
                    "code": "DATA_NOT_READY", **readiness,
                    "config": config.model_dump(mode="json"),
                })
            return manifest["version"]
        raise HTTPException(409, detail={
            "code": "DATA_NOT_READY", "ready": False, "symbol": config.symbol,
            "frequency": config.frequency,
            "config": config.model_dump(mode="json"),
            "gaps": [{"kind": "local", "message": "当前有效数据包中没有本次请求的行情；请先登记或同步基础行情"}],
            "preparation": {"symbol": config.symbol, "frequency": config.frequency,
                              "start": str(config.start_date), "end": str(config.end_date)},
        })

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "ok", "data_dir": root, "package_dir": package_dir,
            "version": "0.1.0", "instance_id": instance_id,
            "restart_supported": callable(getattr(app.state, "restart_callback", None)),
        }

    @app.post("/api/v1/system/restart", status_code=202)
    def restart_project(payload: RestartRequest = RestartRequest()):
        callback = getattr(app.state, "restart_callback", None)
        if not callable(callback):
            raise HTTPException(409, detail={
                "code": "RESTART_UNAVAILABLE",
                "message": "当前进程不是由项目启动脚本托管，无法自动重启。请关闭后重新运行 start.cmd。",
            })
        active_jobs = store.active_job_count()
        if active_jobs and not payload.force:
            raise HTTPException(409, detail={
                "code": "RESTART_JOBS_ACTIVE",
                "message": f"当前有 {active_jobs} 个排队或运行中的任务；确认重启后运行任务会被中断。",
                "active_jobs": active_jobs,
            })
        try:
            callback(active_jobs)
        except Exception as exc:
            raise HTTPException(503, detail={
                "code": "RESTART_FAILED", "message": "无法启动重启助手：" + str(exc),
            }) from exc
        return {
            "status": "restarting", "active_jobs": active_jobs,
            "previous_instance_id": instance_id,
            "message": "服务将重启，运行中的任务会中断",
        }

    @app.get("/api/v1/instruments")
    def instruments(q: str = "", asset_type: Literal["stock", "etf"] | None = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
        return store.instruments(q, limit, asset_type, offset)

    @app.get("/api/v1/workbench/recent-symbols")
    def recent_workbench_symbols():
        items = []
        for submitted in store.recent_submitted_symbols(10):
            symbol = submitted["symbol"]
            stored = store.instrument(symbol)
            if stored is None:
                terms = instrument_terms(symbol, {"name": symbol})
                item = {
                    "symbol": symbol,
                    "name": symbol,
                    "exchange": symbol.rsplit(".", 1)[-1],
                    "board": terms["board"],
                    **terms,
                }
            else:
                item = stored
            items.append({**item, "last_submitted_at": submitted["last_submitted_at"]})
        return items

    @app.post("/api/v1/instruments/refresh")
    def refresh_instruments():
        return submit("instruments", {})

    @app.get("/api/v1/watchlist")
    def watches():
        refresh_watchlist_snapshot()
        return store.watches()

    @app.post("/api/v1/watchlist")
    def add_watch(payload: dict):
        try:
            symbol = normalize_symbol(payload["symbol"])
            store.add_watch(symbol, payload.get("name", ""))
            return {"symbol": symbol}
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/v1/watchlist/{symbol}")
    def delete_watch(symbol: str):
        with store.db() as db:
            db.execute("DELETE FROM watchlist WHERE symbol=?", (normalize_symbol(symbol),))
        return {"deleted": True}

    @app.get("/api/v1/data/frequencies")
    def data_frequencies():
        return {"timeframes": MarketDataService(store).frequency_catalog()}

    @app.get("/api/v1/data/package")
    def data_package():
        return json_safe(MarketDataService(store).package_info())

    @app.get("/api/v1/data/inventory")
    def data_inventory(
        q: str = "", asset_type: Literal["stock", "etf"] | None = None,
        timeframe: str | None = None, status: str | None = None,
        limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
    ):
        try:
            return json_safe(MarketDataService(store).inventory(
                query=q, asset_type=asset_type, timeframe=timeframe,
                status=status, limit=limit, offset=offset,
            ))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/import-package/scan")
    def import_package_scan(payload: MarketPackageRequest):
        try:
            return json_safe(scan_etf_package(payload.folder))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/import-package")
    def import_package(payload: MarketPackageRequest):
        nonlocal package_dir
        try:
            target = Path(payload.folder).expanduser().resolve()
            if data_dir is None and initial_package and target != Path(initial_package).resolve():
                raise ValueError(
                    "项目已固定使用当前行情数据包；如需切换基础数据包，请修改本机设置中的 package_dir 后重启"
                )
            values = payload.model_dump()
            values["folder"] = str(target)
            values["timeframes"] = [
                normalize_timeframe(item) for item in (payload.timeframes or [])
            ] or None
            values["symbols"] = [
                normalize_symbol(item) for item in (payload.symbols or [])
            ] or None
            scan_etf_package(target)
            market = MarketDataService(store)
            market.repository.attach_package(target)
            package_dir = str(market.repository.package_dir)
            return submit("market_index_package", values)
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/datasets")
    def data_datasets(timeframe: str | None = None, q: str = "", status: str | None = None, limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0)):
        try:
            return json_safe(MarketDataService(store).datasets(timeframe, q, status, limit, offset))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/coverage/{symbol}")
    def coverage(symbol: str, timeframe: str | None = None, frequency: str | None = None):
        try:
            requested = timeframe or frequency
            if requested:
                return json_safe(MarketDataService(store).coverage(normalize_symbol(symbol), normalize_timeframe(requested)))
            return json_safe(DataService(store).coverage(symbol))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/v1/data/sync")
    def sync_data(payload: MarketSyncRequest):
        try:
            requested = payload.timeframes or [payload.timeframe or payload.frequency]
            requested = [normalize_timeframe(value) for value in requested if value]
            if not requested:
                requested = ["1day", "1min"]
            invalid = [value for value in requested if value not in {"1day", "1min"}]
            if invalid:
                raise ValueError("线上更新只支持 1day 和 1min；粗周期由基础行情按需聚合")
            timeframes = sorted(set(requested), key=lambda value: 0 if value == "1day" else 1)
            values = payload.model_dump()
            values["timeframe"] = timeframes[0]
            values["timeframes"] = timeframes
            values["symbols"] = [normalize_symbol(symbol) for symbol in payload.symbols]
            if "1min" in timeframes and payload.mode == "history":
                raise ValueError("1 分钟历史行情只从本地数据包读取；请先登记数据包或导入文件，近期缺口使用缺口修补。")
            if "1min" in timeframes and payload.scope == "market" and payload.initialize:
                raise ValueError("1 分钟行情不支持全市场在线初始化；请登记本地数据包或导入文件。")
            if payload.scope == "market" and payload.symbols:
                raise ValueError("全市场范围不能同时指定代码")
            if payload.scope in ("market", "watchlist"):
                values["symbols"] = []
            return submit("market_sync", values)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/sources")
    def data_sources():
        return configured_sources(store)

    @app.post("/api/v1/data/sources/probe")
    def probe_data_source(payload: MarketSourceProbeRequest):
        try:
            values = payload.model_dump()
            if payload.capability == "recent_minute" and payload.source not in {"sina", "tdx", "ths"}:
                raise ValueError("近期 1 分钟探测来源应为 sina、tdx 或 ths")
            if payload.capability == "etf_daily" and payload.source not in {"baostock", "eastmoney"}:
                raise ValueError("ETF 日线探测来源应为 baostock 或 eastmoney")
            if payload.capability == "etf_daily" and payload.market == "stock":
                raise ValueError("ETF 日线探测仅支持 etf 或 all")
            return submit("market_source_probe", values)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/sync-status")
    def sync_status(run_key: str | None = None, status: str | None = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
        return store.sync_status(run_key, status, limit, offset)

    @app.post("/api/v1/data/import")
    def import_data(payload: MarketImportRequest):
        try:
            values = payload.model_dump()
            values["symbol"] = normalize_symbol(payload.symbol)
            values["timeframe"] = normalize_timeframe(payload.timeframe or payload.frequency)
            if not payload.path:
                raise ValueError("必须提供本地 CSV 或 Parquet 文件路径")
            return submit("market_import", values)
        except (ValueError, KeyError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/upload")
    async def upload_import_data(
        market_file: UploadFile = File(...),
        symbol: str = Form(...),
        timeframe: str = Form(...),
        label_convention: str = Form("auto"),
        etf_category: str | None = Form(None),
        calendar_file: UploadFile | None = File(None),
        actions_file: UploadFile | None = File(None),
        skip_invalid_rows: bool = Form(False),
        import_mode: str = Form("merge"),
    ):
        try:
            timeframe = normalize_timeframe(timeframe)
            symbol = normalize_symbol(symbol)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if label_convention not in {"auto", "start", "end"}:
            raise HTTPException(422, "时间标签应为 auto/start/end")
        if import_mode not in {"merge", "snapshot"}:
            raise HTTPException(422, "导入模式应为 merge/snapshot")
        market = MarketDataService(store)
        stage = market.repository.package.staging_dir / "uploads" / timeframe / uuid.uuid4().hex
        metadata_stage = stage.parent / (stage.name + "-metadata")
        try:
            market_path = stage / _safe_upload_path(market_file.filename or "market.csv")
            size = await _save_upload(market_file, market_path)
            calendar_path = None
            if calendar_file is not None and calendar_file.filename:
                calendar_path = metadata_stage / _safe_upload_path(calendar_file.filename)
                await _save_upload(calendar_file, calendar_path)
            actions_path = None
            if actions_file is not None and actions_file.filename:
                actions_path = stage / "actions" / _safe_upload_path(actions_file.filename)
                await _save_upload(actions_file, actions_path)
            payload = {
                "symbol": symbol, "timeframe": timeframe, "path": str(market_path),
                "label_convention": label_convention, "adjustment": "none",
                "etf_category": etf_category or None,
                "calendar_path": str(calendar_path) if calendar_path else None,
                "actions_path": str(actions_path) if actions_path else None,
                "skip_invalid_rows": skip_invalid_rows,
                "import_mode": import_mode,
                "upload_dirs": [
                    str(stage), str(metadata_stage),
                ],
            }
            submitted = submit("market_import", payload)
            return {**submitted, "file_name": market_file.filename, "bytes": size, "timeframe": timeframe}
        except (ValueError, MarketDataError) as exc:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(metadata_stage, ignore_errors=True)
            raise HTTPException(422, str(exc)) from exc
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(metadata_stage, ignore_errors=True)
            raise

    @app.post("/api/v1/data/upload-batch")
    async def upload_import_batch(
        files: list[UploadFile] = File(...),
        timeframe: str = Form(...),
        label_convention: str = Form("auto"),
        etf_category: str | None = Form(None),
        calendar_file: UploadFile | None = File(None),
        skip_invalid_rows: bool = Form(False),
        import_mode: str = Form("merge"),
    ):
        try:
            timeframe = normalize_timeframe(timeframe)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if label_convention not in {"auto", "start", "end"}:
            raise HTTPException(422, "时间标签应为 auto/start/end")
        if import_mode not in {"merge", "snapshot"}:
            raise HTTPException(422, "导入模式应为 merge/snapshot")
        if not files:
            raise HTTPException(422, "请选择至少一个 CSV 或 Parquet 文件")
        market = MarketDataService(store)
        stage = market.repository.package.staging_dir / "uploads" / timeframe / uuid.uuid4().hex
        metadata_stage = stage.parent / (stage.name + "-metadata")
        count = 0
        total_bytes = 0
        try:
            for upload in files:
                relative = _safe_upload_path(upload.filename or "upload.csv")
                destination = stage / relative
                if destination.exists():
                    raise ValueError("上传文件存在重复相对路径：" + relative.as_posix())
                total_bytes += await _save_upload(upload, destination)
                count += 1
            calendar_path = None
            if calendar_file is not None and calendar_file.filename:
                calendar_path = metadata_stage / _safe_upload_path(calendar_file.filename)
                await _save_upload(calendar_file, calendar_path)
            payload = {
                "folder": str(stage), "timeframe": timeframe, "recursive": True,
                "label_convention": label_convention, "adjustment": "none",
                "etf_category": etf_category or None, "resume": True,
                "calendar_path": str(calendar_path) if calendar_path else None,
                "skip_invalid_rows": skip_invalid_rows,
                "import_mode": import_mode,
                "upload_dirs": [
                    str(stage), str(metadata_stage),
                ],
            }
            submitted = submit("market_import_batch", payload)
            return {**submitted, "files": count, "bytes": total_bytes, "timeframe": timeframe}
        except (ValueError, MarketDataError) as exc:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(metadata_stage, ignore_errors=True)
            raise HTTPException(422, str(exc)) from exc
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(metadata_stage, ignore_errors=True)
            raise

    @app.post("/api/v1/data/import-batch/scan")
    def import_batch_scan(payload: MarketBatchRequest):
        try:
            timeframe = normalize_timeframe(payload.timeframe or payload.frequency)
            folder = payload.folder or str(Path(root) / "imports" / timeframe)
            return json_safe(MarketDataService(store).preview_folder(folder, timeframe, recursive=payload.recursive))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/import-batch")
    def import_batch(payload: MarketBatchRequest):
        try:
            timeframe = normalize_timeframe(payload.timeframe or payload.frequency)
            values = payload.model_dump()
            values["timeframe"] = timeframe
            values["folder"] = payload.folder or str(Path(root) / "imports" / timeframe)
            return submit("market_import_batch", values)
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/validate")
    def validate_data(payload: MarketValidateRequest):
        try:
            timeframe = normalize_timeframe(payload.timeframe or payload.frequency)
            return submit("market_validate", {"symbol": normalize_symbol(payload.symbol), "timeframe": timeframe,
                                             "label_convention": payload.label_convention,
                                             "mode": payload.mode,
                                             "start_date": payload.start_date,
                                             "end_date": payload.end_date})
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/archive")
    def archive_data(payload: MarketArchiveRequest):
        try:
            timeframe = normalize_timeframe(payload.timeframe or payload.frequency)
            return submit("market_archive", {
                "symbols": [normalize_symbol(value) for value in payload.symbols],
                "timeframe": timeframe,
            })
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/compliance/{symbol}")
    def data_compliance(symbol: str):
        try:
            market = MarketDataService(store)
            normalized = normalize_symbol(symbol)
            return json_safe({
                timeframe: market.repository.validation_state(normalized, timeframe)
                for timeframe in ("1day", "1min")
                if market.repository.dataset(normalized, timeframe)
            })
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/aggregate")
    def aggregate_data(payload: MarketAggregateRequest):
        try:
            values = payload.model_dump()
            values["symbols"] = [normalize_symbol(symbol) for symbol in payload.symbols]
            values["source_timeframe"] = normalize_timeframe(payload.source_timeframe)
            values["target_timeframe"] = normalize_timeframe(payload.target_timeframe)
            return submit("market_aggregate", values)
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/readiness")
    def data_readiness(config: BacktestConfig, version: str | None = None):
        try:
            return json_safe(MarketDataService(store).readiness(config, version))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/preparation/plan")
    def preparation_plan(payload: MarketPreparationRequest):
        try:
            return json_safe(MarketPreparationService(MarketDataService(store)).plan(payload.config))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/preparation")
    def preparation_submit(payload: MarketPreparationRequest):
        try:
            return submit("market_prepare", {
                "config": payload.config.model_dump(mode="json"),
                "version": payload.version,
            })
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/bars/{symbol}")
    def bars(
        symbol: str, start_date: str | None = None, end_date: str | None = None,
        frequency: str = "1m", limit: int = Query(1500, ge=1, le=10000),
        version: str | None = None, before: str | None = None, latest: bool = False,
    ):
        normalized = normalize_timeframe(frequency)
        market = MarketDataService(store)
        manifest = store.snapshot(symbol, version, persist=False)
        if version and manifest is None:
            raise HTTPException(404, {
                "code": "DATA_VERSION_NOT_FOUND",
                "message": f"原数据版本 {version} 已清理，不能按该版本读取",
            })
        if manifest and not manifest.get("market_data"):
            raise HTTPException(404, {
                "code": "DATA_VERSION_NOT_FOUND",
                "message": f"数据版本 {version or manifest.get('version')} 属于已删除的旧行情格式",
            })
        if market.repository.dataset(normalize_symbol(symbol), normalized) or manifest:
            try:
                market_version = None
                if manifest and manifest.get("market_data"):
                    reference = (manifest.get("market_versions", {}).get(normalized) or {}).get("version")
                    if version and reference is None:
                        raise HTTPException(404, {
                            "code": "DATA_VERSION_NOT_FOUND",
                            "message": f"数据版本 {version} 没有 {normalized} 行情",
                        })
                    market_version = int(reference) if reference is not None else None
                payload = market.bars(
                    symbol, normalized, start=pd.Timestamp(start_date) if start_date else None,
                    end=pd.Timestamp(end_date) if end_date else None,
                    version=market_version,
                    limit=limit,
                    before=pd.Timestamp(before) if before else None,
                    latest=latest,
                    materialize=False,
                )
                return json_safe(payload["rows"])
            except DataVersionNotFound as exc:
                raise HTTPException(404, {"code": exc.code, "message": str(exc)}) from exc
            except MarketDataError:
                return []
        return []

    @app.post("/api/v1/backtests")
    def backtest(config: BacktestConfig):
        version = require_ready(config)
        if config.schema_version != 3:
            raise HTTPException(422, "只接受 schema_version=3 的当前股数配置")
        return submit("backtest", {"config": config.model_dump(mode="json"), "version": version})

    @app.get("/api/v1/data/reference-price/{symbol}")
    def reference_price(symbol: str, start_date: str):
        try:
            return DataService(store).reference_price(normalize_symbol(symbol), pd.Timestamp(start_date).date())
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/market/bars/{symbol}")
    def market_bars(symbol: str, frequency: str = "1d", start_date: str | None = None, end_date: str | None = None, before: str | None = None, limit: int = Query(600, ge=1, le=5000), ma: str = "5,10,20,60"):
        try:
            timeframe = normalize_timeframe(frequency)
            market = MarketDataService(store)
            return json_safe(market.market_bars(
                symbol, timeframe, start=start_date, end=end_date, before=before,
                limit=limit, ma_periods=[int(value) for value in ma.split(",") if value.strip()],
                materialize=False,
            ))
        except MarketDataError as exc:
            return {"symbol": normalize_symbol(symbol), "frequency": normalize_timeframe(frequency),
                    "bars": [], "warnings": [str(exc)], "quality_flags": [str(exc)], "diagnostics": {}, "source_events": []}
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/backtests/preview")
    def preview(payload: PreviewRequest):
        version = require_ready(payload.config)
        if payload.config.schema_version != 3:
            raise HTTPException(409, "预览只接受 schema_version=3 的当前股数配置")
        # Preview is deliberately submitted without a synchronous coverage scan;
        # the worker reports preparation progress and pins the result version.
        return json_safe(previews.submit(payload.session_id, payload.revision, payload.config, version))

    @app.get("/api/v1/backtests/preview/{session_id}")
    def preview_status(session_id: str):
        status = previews.snapshot(session_id)
        if status is None:
            raise HTTPException(404, "预览已过期")
        return json_safe(status)

    @app.delete("/api/v1/backtests/preview/{session_id}")
    def cancel_preview(session_id: str):
        previews.cancel(session_id)
        return {"cancelled": True}

    @app.post("/api/v1/backtests/batch")
    def batch(payload: BatchRequest):
        versions = [require_ready(config) for config in payload.configs]
        if any(config.schema_version != 3 for config in payload.configs):
            raise HTTPException(422, "批量回测只接受 schema_version=3 的当前股数配置")
        return {"job_ids": [submit("backtest", {"config": config.model_dump(mode="json"), "version": version})["job_id"] for config, version in zip(payload.configs, versions)]}

    @app.post("/api/v1/optimizations")
    def optimization(config: OptimizationConfig | ResearchRequestV5):
        if isinstance(config, ResearchRequestV5):
            config = config.to_optimization()
        prepared = preparation_config(store, config.base, lookback_days=int(config.strategy.training_max_days))
        version = require_ready(prepared)
        bundle = DataService(store).load_bundle(prepared, version)
        if config.protocol_version == 5:
            from .research.robust import estimate_robust_search
            try:
                preview = estimate_robust_search(config, bundle)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        else:
            raise HTTPException(422, "旧研究协议已移除；请使用 protocol_version=5")
        if preview.get("status") == "budget_exceeded":
            reason = next((value for value in preview.get("reasons", []) if "超正方体" in str(value) or "预算" in str(value)), None)
            raise HTTPException(422, {
                "code": "SEARCH_BUDGET_EXCEEDED",
                "message": reason or f"完整枚举需要 {preview.get('legal_combination_count')} 个验证点，超过预算 {preview.get('validation_budget')}",
                **preview,
            })
        submitted = submit("optimization", {
            "config": config.model_dump(mode="json"), "version": version,
            "configuration_revision": config.configuration_revision,
            "budget_contract": {
                "data_version": version,
                "effective_total_evaluation_budget": preview["effective_total_evaluation_budget"],
                "task_evaluation_upper_bound": preview["task_evaluation_upper_bound"],
                "max_runtime_seconds": preview["max_runtime_seconds"],
                "estimated_task_memory_mb": preview["estimated_task_memory_mb"],
                "workspace_budget_mb": preview["workspace_budget_mb"],
            },
        })
        return {**submitted, "search_preview": preview}

    @app.post("/api/v1/optimizations/preview")
    def optimization_preview(config: OptimizationConfig | ResearchRequestV5):
        if isinstance(config, ResearchRequestV5):
            config = config.to_optimization()
        prepared = preparation_config(store, config.base, lookback_days=int(config.strategy.training_max_days))
        version = require_ready(prepared)
        bundle = DataService(store).load_bundle(prepared, version)
        if config.protocol_version == 5:
            from .research.robust import estimate_robust_search
            try:
                return json_safe(estimate_robust_search(config, bundle))
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        raise HTTPException(422, "旧研究协议已移除；请使用 protocol_version=5")

    @app.get("/api/v1/research/strategies")
    def research_strategies():
        from .research import strategy_descriptions
        return strategy_descriptions()

    @app.get("/api/v1/jobs")
    def jobs(
        status: str | None = Query(None),
        kind: str | None = Query(None, pattern=r"^[a-z_]{1,64}$"),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ):
        return store.jobs(limit=limit, status=status, kind=kind, offset=offset)

    # Keep the bare-list /jobs contract for existing polling clients.  History
    # and filtering use this dedicated paged resource so a long-lived install
    # never truncates the task center at the newest hundred rows.
    @app.get("/api/v1/jobs/history")
    def job_history(
        status: str | None = Query(None, pattern=r"^[a-z_]{1,32}$"),
        kind: str | None = Query(None, pattern=r"^[a-z_]{1,64}$"),
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
    ):
        return store.job_history(limit=limit, offset=offset, status=status, kind=kind)

    @app.get("/api/v1/jobs/active")
    def active_jobs():
        return store.active_jobs()

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str):
        job = store.job(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return job

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = get_job(job_id)
        if job["status"] == "queued":
            if store.cancel_queued_job(job_id):
                from .jobs import _cleanup_uploaded_input
                _cleanup_uploaded_input(store, job_id)
            else:
                # The dispatcher claimed the row between the read and the
                # atomic queue cancellation.  The worker now owns cleanup.
                store.update_job(job_id, cancel_requested=1)
        elif job["status"] == "running":
            store.update_job(job_id, cancel_requested=1)
        return store.job(job_id)

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        get_job(job_id)
        async def events():
            previous = None
            while True:
                job = store.job(job_id)
                serialized = dumps(job)
                if serialized != previous:
                    yield f"data: {serialized}\n\n"
                    previous = serialized
                if not job or job["status"] in ("completed", "partial", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.5)
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/v1/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = get_job(job_id)
        if not job.get("result_id"):
            raise HTTPException(409, "任务尚无结果")
        result_id = str(job["result_id"])
        result = store.result(result_id)
        if result is None:
            raise HTTPException(404, "结果不存在")
        _require_supported_result(result)
        revisions = store.completion_revisions(result_id)
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        requested_revision = payload.get("completion_revision")
        if requested_revision is not None:
            try:
                revision_number = int(requested_revision)
            except (TypeError, ValueError) as exc:
                raise HTTPException(409, {
                    "code": "COMPLETION_REVISION_INVALID",
                    "message": "任务保存的补算修订号无效",
                }) from exc
            selected_revision = store.completion_revision(result_id, revision_number)
            if selected_revision is None:
                raise HTTPException(409, {
                    "code": "COMPLETION_REVISION_NOT_FOUND",
                    "message": f"任务对应的补算修订 {revision_number} 不存在",
                    "revision": revision_number,
                })
        else:
            selected_revision = revisions[-1] if revisions else None
        public = public_result(result, data_metadata=None)
        public["completion"] = _completion_state(
            result_id, result, revisions, selected_revision,
        )
        return _with_data_availability(store, public)

    @app.post("/api/v1/jobs/{job_id}/resume", status_code=202)
    def resume_job(job_id: str):
        job = get_job(job_id)
        if job.get("kind") != "research_completion":
            raise HTTPException(422, "只有可恢复的研究补算任务支持继续")
        if job.get("status") not in {"failed", "cancelled", "partial"}:
            raise HTTPException(409, "任务当前不需要恢复")
        payload = dict(job.get("payload") or {})
        payload.pop("submission_provenance", None)
        payload["resume_from_job"] = job_id
        # Only a resume continues the original checkpoint; a fresh submission
        # always starts a new one.
        payload["checkpoint_id"] = payload.get("checkpoint_id") or job_id
        return {**submit("research_completion", payload), "resumed_from": job_id}

    @app.get("/api/v1/jobs/{job_id}/items")
    def job_items(job_id: str, section: str = "all", offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        job = get_job(job_id)
        if not job.get("result_id"):
            raise HTTPException(409, "任务尚无结果")
        result = store.result(job["result_id"])
        content = result.get("result", result)
        groups: list[dict] = []
        for name in ("results", "failures", "skipped_items", "symbols", "skipped"):
            if section not in ("all", name):
                continue
            values = content.get(name, []) if isinstance(content, dict) else []
            for item in values:
                groups.append({"section": name, **item} if isinstance(item, dict) else {"section": name, "value": item})
        if not groups and isinstance(content, dict):
            groups.append({"section": "result", **content})
        total = len(groups)
        return {"items": groups[offset:offset + limit], "total": total, "offset": offset, "limit": limit}

    @app.get("/api/v1/results")
    def results(
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
        kind: str | None = Query(None, pattern=r"^[a-z_]{1,64}$"),
    ):
        return store.results(limit=limit, offset=offset, kind=kind)

    @app.get("/api/v1/results/{result_id}/summary")
    def result_summary(result_id: str, revision: int | None = Query(None, ge=1)):
        return get_result(result_id, full=False, revision=revision)

    @app.get("/api/v1/results/{result_id}")
    def get_result(result_id: str, full: bool = False, revision: int | None = Query(None, ge=1)):
        result = store.result(result_id)
        if result is None:
            raise HTTPException(404, "结果不存在")
        _require_supported_result(result)
        public = public_result(
            result, full=full, data_metadata=None,
        )
        revisions = store.completion_revisions(result_id)
        selected_revision = revisions[-1] if revisions else None
        if revision is not None:
            selected_revision = store.completion_revision(result_id, revision)
            if selected_revision is None:
                raise HTTPException(404, "补算修订不存在")
        public["completion"] = _completion_state(
            result_id, result, revisions, selected_revision,
        )
        return public if full else _with_data_availability(store, public)

    @app.get("/api/v1/results/{result_id}/landscape")
    def result_landscape(
        result_id: str,
        decision: str = Query("current", pattern="^(current|fixed|adaptive)$"),
        decision_index: int = Query(0, ge=0),
        sample: str = Query("training", pattern="^(training|validation)$"),
        cost: str = Query("original", pattern="^(original|stress)$"),
        fold: int | None = Query(None, ge=0),
        region_id: int | None = Query(None, ge=1),
        offset: int = Query(0, ge=0),
        limit: int = Query(1000, ge=1, le=10000),
        view: str = Query("sample", pattern="^(sample|members|slice|projection)$"),
        x: str | None = None,
        y: str | None = None,
        anchor: str = Query("stable", pattern="^(stable|peak)$"),
        revision: int | None = Query(None, ge=1),
    ):
        result = store.result(result_id)
        if result is None:
            raise HTTPException(404, "结果不存在")
        _require_supported_result(result)
        config = result.get("config", {})
        strategy_version = (config.get("strategy") or {}).get("strategy_version")
        if strategy_version != 5:
            raise HTTPException(410, {
                "code": "UNSUPPORTED_RESULT_VERSION",
                "message": "该研究结果缺少 v5 协议事实，不支持参数分布读取",
            })
        if sample != "training":
            raise HTTPException(422, {
                "code": "NO_INTERNAL_VALIDATION_LANDSCAPE",
                "message": "v5 只有单段训练和真实连续样本外账本，没有内部验证候选分布",
            })
        try:
            payload = _robust_training_landscape(
                store, result, result_id, decision=decision,
                decision_index=decision_index, region_id=region_id,
                offset=offset, limit=limit, revision=revision,
            )
            if revision is not None:
                payload["revision_scope"] = _revision_scope(
                    store.completion_revision(result_id, revision)
                )
            return payload
        except CompletionRevisionError as exc:
            # A revision only covers one decision branch.  Showing the
            # pre-completion archive as if it had been revised is worse than
            # refusing the request.
            raise _revision_conflict(exc) from exc
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/v1/results/{result_id}/completion", status_code=202)
    def submit_result_completion(result_id: str, request: ResearchCompletionRequest):
        result = store.result(result_id)
        if result is None:
            raise HTTPException(404, "结果不存在")
        _require_supported_result(result)
        if result.get("protocol_version") != 5 or result.get("result_schema_version") != 2:
            raise HTTPException(410, {
                "code": "UNSUPPORTED_RESULT_VERSION",
                "message": "该历史版本不支持补算",
            })
        provenance = result.get("computation_provenance") or {}
        if provenance.get("reproducible") is False or not provenance.get("completion_allowed"):
            raise HTTPException(409, {
                "code": "COMPLETION_PROVENANCE_MISMATCH",
                "message": "原结果缺少完整计算来源，或运行期间代码与依赖发生变化，不能补算",
                "mismatches": provenance.get("mismatches", []),
            })
        finished = provenance.get("finished") or {}
        from .reproducibility import computation_provenance, reproducibility_mismatches
        from .settings import PROJECT_ROOT
        current = computation_provenance(PROJECT_ROOT, configuration=result.get("config"))
        mismatches = reproducibility_mismatches(finished, current)
        if mismatches:
            raise HTTPException(409, {
                "code": "COMPLETION_PROVENANCE_MISMATCH",
                "message": "当前代码、配置或依赖环境与原结果不一致，拒绝补算",
                "mismatches": mismatches,
            })
        symbol = result.get("symbol") or (result.get("config") or {}).get("base", {}).get("symbol")
        data_version = result.get("data_version")
        if not symbol or not data_version or store.snapshot(symbol, data_version, persist=False) is None:
            raise HTTPException(409, {
                "code": "COMPLETION_DATA_UNAVAILABLE",
                "message": "原行情数据版本不可用，不能补算",
            })
        payload = {
            "result_id": result_id, **request.model_dump(mode="json"),
            "data_version": data_version,
            "max_runtime_seconds": int(
                (result.get("config") or {}).get("strategy", {}).get("max_runtime_seconds", 3600)
            ),
        }
        identity = {
            "result_id": result_id, "decision": request.decision,
            "decision_index": request.decision_index, "target": request.target,
            "reporting_budget": int(request.reporting_budget),
            "data_version": data_version,
            "source_fingerprint": current.get("source_fingerprint"),
            "dependency_lock_hash": current.get("dependency_lock_hash"),
            "configuration_fingerprint": current.get("configuration_fingerprint"),
        }
        dedupe_key = "completion:" + hashlib.sha256(
            dumps(identity).encode("utf-8")
        ).hexdigest()
        submitted = submit(
            "research_completion", payload, dedupe_key=dedupe_key,
            include_created=True,
        )
        created = bool(submitted.pop("created", True))
        job = store.job(submitted["job_id"]) or {}
        return {
            **submitted, "status": job.get("status"),
            "deduplicated": not created and job.get("status") in {"queued", "running"},
        }

    @app.get("/api/v1/results/{result_id}/regions/{region_id}/export")
    def export_region(result_id: str, region_id: int,
                      decision: str = Query("current", pattern="^(current|fixed|adaptive)$"),
                      decision_index: int = Query(0, ge=0),
                      format: str = Query("csv", pattern="^(csv|json)$"),
                      revision: int | None = Query(None, ge=1)):
        result = store.result(result_id)
        if result is None:
            raise HTTPException(404, "结果不存在")
        _require_supported_result(result)
        try:
            context = _research_decision_context(result, decision, decision_index)
            region = next((r for r in context.get("recommendation_regions", []) if r.get("region_id") == region_id), None)
            if not region or region.get("geometry") != "robust_j":
                raise LookupError("推荐参数域不存在")
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        if revision is not None:
            try:
                # The revision scope is checked before the response starts:
                # a revision that covers another branch must not silently
                # export the pre-completion rows.
                store.candidate_reference(
                    result, "candidate_archives", decision, decision_index, revision=revision,
                )
            except CompletionRevisionError as exc:
                raise _revision_conflict(exc) from exc
            except LookupError as exc:
                raise HTTPException(404, str(exc)) from exc
        filename = f"{result_id}-{decision}-{decision_index}-region-{region_id}.{format}"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        if format == "json":
            def json_rows():
                yield '{"region":' + dumps(region) + ',"members":['
                first = True
                for row in store.iter_candidate_archive(
                    result, "candidate_archives", decision, decision_index,
                    region_id=region_id, sort_by="candidate_id", revision=revision,
                ):
                    if not first:
                        yield ","
                    yield dumps(_landscape_point(row, "training", "original", None))
                    first = False
                yield "]}"
            return StreamingResponse(json_rows(), media_type="application/json", headers=headers)

        def csv_rows():
            from .research.space import ROBUST_FIELDS
            output = io.StringIO(newline="")
            writer = csv.writer(output)
            writer.writerow(["candidate_id", *ROBUST_FIELDS, "excess_return", "max_principal_drawdown", "score_complete", "constraint_eligible", "constraint_reasons"])
            yield "\ufeff" + output.getvalue()
            for row in store.iter_candidate_archive(
                result, "candidate_archives", decision, decision_index,
                region_id=region_id, sort_by="candidate_id", revision=revision,
            ):
                output.seek(0)
                output.truncate(0)
                writer.writerow([row["candidate_id"], *(row["params"].get(name) for name in ROBUST_FIELDS),
                                 row.get("training_excess_return"), row.get("max_principal_drawdown"),
                                 bool(row.get("score_complete")),
                                 row.get("constraint_eligible"), "；".join(row.get("constraint_reasons", []))])
                yield output.getvalue()
        return StreamingResponse(csv_rows(), media_type="text/csv; charset=utf-8", headers=headers)

    @app.get("/api/v1/results/{result_id}/export")
    def export(result_id: str, kind: str = "json", revision: int | None = Query(None, ge=1)):
        result = get_result(result_id, full=True, revision=revision)
        if kind == "json":
            revision_record = store.completion_revision(result_id, revision) if revision is not None else None
            try:
                payload = _revision_substituted_result(result, revision_record)
            except CompletionRevisionError as exc:
                raise _revision_conflict(exc) from exc
            return StreamingResponse(
                _stream_json_value(payload, store), media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{result_id}-{kind}.json"'},
            )
        elif kind in ("equity", "trades"):
            def ledger_records():
                """Yield the equity or trade ledger rows one at a time."""
                if result.get("protocol_version") == 5:
                    runs = []
                    for name in ("fixed", "adaptive"):
                        branch = result.get(name)
                        if isinstance(branch, dict):
                            runs.append((name, branch.get("test_result") or {}))
                else:
                    runs = [(None, result if "strategies" in result else {})]
                for selection, run in runs:
                    for strategy, values in (run.get("strategies") or {}).items():
                        for row in values.get(kind, []) or []:
                            record = {"strategy": strategy, **row}
                            if selection is not None:
                                record = {"selection": selection, **record}
                            yield record

            def ledger_rows():
                """Stream the equity or trade ledger without building a frame in memory."""
                columns: list[str] = []
                seen: set[str] = set()
                for record in ledger_records():
                    for key in record:
                        if key not in seen:
                            seen.add(key)
                            columns.append(key)
                if not columns:
                    yield "\ufeff\n"
                    return
                output = io.StringIO(newline="")
                writer = csv.writer(output, lineterminator="\n")
                writer.writerow(columns)
                yield "\ufeff" + output.getvalue()
                for record in ledger_records():
                    output.seek(0)
                    output.truncate(0)
                    writer.writerow([_csv_value(record.get(name)) for name in columns])
                    yield output.getvalue()
            return StreamingResponse(
                ledger_rows(), media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{result_id}-{kind}.csv"'},
            )
        elif kind == "candidates":
            revision_record = store.completion_revision(result_id, revision) if revision is not None else None
            scope = _revision_scope(revision_record)

            def applies(selection: str, index: int) -> bool:
                return bool(
                    scope is not None
                    and scope["decision"] == selection
                    and scope["decision_index"] == int(index)
                )

            def csv_rows():
                output = io.StringIO(newline="")
                writer = csv.writer(output)
                writer.writerow([
                    "selection", "decision_index", "candidate_id", "buy_step", "sell_step",
                    "buy_shares", "sell_shares", "j_score", "training_excess_return",
                    "max_principal_drawdown", "score_complete", "revision", "archive_source",
                ])
                yield "\ufeff" + output.getvalue()
                evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
                selections = [("fixed", 0), ("current", 0)]
                adaptive = (evidence.get("candidate_archives") or {}).get("adaptive") or []
                selections.extend(("adaptive", index) for index in range(len(adaptive)))
                for selection, index in selections:
                    revised = applies(selection, index)
                    for row in store.iter_candidate_archive(
                        result, "candidate_archives", selection, index,
                        revision=revision if revised else None,
                    ):
                        params = row.get("params") if isinstance(row.get("params"), dict) else {}
                        output.seek(0)
                        output.truncate(0)
                        writer.writerow([
                            selection, index, row.get("candidate_id"), params.get("buy_step"),
                            params.get("sell_step"), params.get("buy_shares"), params.get("sell_shares"),
                            row.get("j_score"), row.get("training_excess_return"),
                            row.get("max_principal_drawdown"), row.get("score_complete"),
                            revision if revised else "", "revised" if revised else "base",
                        ])
                        yield output.getvalue()
            return StreamingResponse(
                csv_rows(), media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{result_id}-{kind}.csv"'},
            )
        else:
            raise HTTPException(422, "导出类型应为 json/equity/trades/candidates")

    @app.get("/api/v1/configs")
    def configs():
        with store.db() as db:
            rows = db.execute("SELECT * FROM configs ORDER BY created_at DESC").fetchall()
        return [{**dict(row), "config": json.loads(row["config"])} for row in rows]

    @app.post("/api/v1/configs")
    def save_config(payload: dict):
        try:
            config = BacktestConfig.model_validate(payload["config"])
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if config.schema_version != 3:
            raise HTTPException(422, "只接受 schema_version=3 的当前股数配置")
        config_id = uuid.uuid4().hex
        with store.db() as db:
            db.execute("INSERT INTO configs VALUES(?,?,?,?)", (config_id, payload.get("name", "网格配置"), config.model_dump_json(), now_iso()))
        return {"id": config_id}

    @app.delete("/api/v1/configs/{config_id}")
    def delete_config(config_id: str):
        with store.db() as db:
            db.execute("DELETE FROM configs WHERE id=?", (config_id,))
        return {"deleted": True}

    @app.get("/api/v1/configuration", response_model=ConfigurationEnvelope)
    def get_configuration():
        document = current_configuration()
        return {
            "config": document,
            "revision": document.get("config_revision"),
            "pending_restart": pending_restart_fields(
                document, active_data_dir=root, active_package_dir=package_dir,
                active_host=configured.host, active_port=configured.port,
                active_max_workers=active_max_workers,
                active_scheduler_poll_seconds=active_scheduler_poll_seconds,
            ),
            "active_data_dir": root,
            "active_package_dir": package_dir,
        }

    @app.get("/api/v1/configuration/schema")
    def get_configuration_schema():
        return public_configuration_schema()

    @app.post("/api/v1/configuration/validate", response_model=ConfigurationValidationEnvelope)
    def validate_configuration(payload: ConfigurationValidationRequest):
        try:
            document = validate_configuration_patch(payload.patch)
            return {"valid": True, "config": document, "revision": document.get("config_revision")}
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.patch("/api/v1/configuration", response_model=ConfigurationEnvelope)
    def update_configuration(payload: ConfigurationPatchRequest):
        base_revision = payload.base_revision
        patch = payload.patch
        try:
            candidate = validate_configuration_patch(patch, base_revision=base_revision)
            validate_source_settings(store, candidate["market_data"])
            document = patch_configuration(patch, base_revision=base_revision)
            configure_cache(int(document["market_data"]["cache_mb"]))
            return {
                "config": document,
                "revision": document.get("config_revision"),
                "pending_restart": pending_restart_fields(
                    document, active_data_dir=root, active_package_dir=package_dir,
                    active_host=configured.host, active_port=configured.port,
                    active_max_workers=active_max_workers,
                    active_scheduler_poll_seconds=active_scheduler_poll_seconds,
                ),
            }
        except ConfigurationConflict as exc:
            raise HTTPException(409, {"code": "CONFIG_REVISION_CONFLICT", "message": str(exc)}) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/performance")
    def performance_status():
        current = load_settings()
        with store.db() as db:
            leases = [dict(row) for row in db.execute(
                "SELECT job_id,pid,kind,units,updated_at FROM resource_leases ORDER BY updated_at DESC"
            ).fetchall()]
        return {
            "logical_cpus": logical_cpu_count(),
            "effective_cpu_budget": effective_cpu_budget(current.performance),
            "settings": current.performance.model_dump(mode="json"),
            "active_leases": leases,
            "reserved_cpu_units": sum(int(row.get("units") or 0) for row in leases),
        }

    @app.post("/api/v1/schedule/install")
    def install_schedule():
        try:
            return configure_schedule(True)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/v1/schedule/remove")
    def remove_schedule():
        try:
            return configure_schedule(False)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    frontend = PROJECT_ROOT / "frontend" / "dist"
    if (frontend / "assets").exists():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def index(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "接口不存在")
        if (frontend / "index.html").exists():
            return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})
        return {"message": "后端已启动。请在 frontend 目录运行 npm ci 和 npm run build，然后刷新", "docs": "/docs"}

    return app
