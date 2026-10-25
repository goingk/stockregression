"""Offline comparison of research priors on three local, ready ETFs.

The experiment is deliberately separate from application research jobs.  It
reads the active local snapshots, holds the snapshot, evaluation dates, seed,
parameter ranges and budgets fixed, and runs ``off`` immediately before
``daily_state_v1`` for each selected symbol.  It never downloads data and does
not save a research result through the application store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stockregression.data import DataService
from stockregression.etf_models import EtfResearchRequest
from stockregression.etf_research import preflight
from stockregression.market_data import MarketDataService
from stockregression.models import BacktestConfig, OptimizationConfig, ResearchStrategySpec
from stockregression.providers import latest_closed_date
from stockregression.reproducibility import computation_provenance
from stockregression.research.robust import estimate_robust_search
from stockregression.research_preparation import prepare_research_bundle, preparation_config
from stockregression.research_v5 import optimize_v5
from stockregression.settings import PROJECT_ROOT, load_settings, resolve_project_path
from stockregression.storage import Store


POLICIES = ("off", "daily_state_v1")
EVALUATION_TRADING_DAYS = 250
CALENDAR_LOOKBACK_DAYS = 550
SYMBOL_LIMIT = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research-priors-comparison.json"),
        help="experiment report path (the application result store is not used)",
    )
    parser.add_argument("--training-budget", type=int, default=100)
    parser.add_argument("--validation-budget", type=int, default=200)
    parser.add_argument("--reporting-budget", type=int, default=20)
    parser.add_argument("--total-evaluation-budget", type=int, default=1_000)
    parser.add_argument("--max-runtime-seconds", type=int, default=180, help="per policy run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-only", action="store_true",
        help="resolve symbols, common dates and prior capability without running research",
    )
    return parser


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float,)):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base(
    settings, symbol: str, start: date, end: date,
) -> BacktestConfig:
    defaults = settings.workbench_defaults
    grid, position = defaults.grid, defaults.position
    return BacktestConfig(
        schema_version=3,
        symbol=symbol,
        start_date=start,
        end_date=end,
        frequency="1m",
        initial_shares=position.initial_shares,
        min_shares=0,
        max_shares=None,
        buy_shares=position.buy_shares,
        sell_shares=position.sell_shares,
        grid_mode="geometric",
        buy_step=grid.buy_step,
        sell_step=grid.sell_step,
        # ETF studies use relative grid steps.  Absolute price bounds are not
        # portable to another fund and are deliberately cleared.
        lower_price=None,
        upper_price=None,
        fees=settings.fee_defaults.model_dump(mode="python"),
        dca=defaults.dca.model_dump(mode="python"),
    )


def _common_config(settings, base: BacktestConfig, arguments) -> OptimizationConfig:
    research = settings.workbench_defaults.research.model_dump(mode="python")
    research.update({
        "training_budget": arguments.training_budget,
        "validation_budget": arguments.validation_budget,
        "reporting_budget": arguments.reporting_budget,
        "total_evaluation_budget": arguments.total_evaluation_budget,
        "max_runtime_seconds": arguments.max_runtime_seconds,
        "seed": arguments.seed,
        "training_min_days": 250,
        "training_max_days": 500,
    })
    strategy_fields = set(ResearchStrategySpec.model_fields)
    strategy_values = {
        name: value for name, value in research.items() if name in strategy_fields
    }
    return OptimizationConfig(
        base=base,
        selection_target="fixed",
        ranges={
            name: value.model_dump(mode="python")
            for name, value in settings.workbench_defaults.research.ranges.items()
        },
        strategy=ResearchStrategySpec.model_validate(strategy_values),
        seed=int(research["seed"]),
        top_candidates=settings.workbench_defaults.research.top_candidates,
        max_principal_drawdown=settings.workbench_defaults.research.max_principal_drawdown,
        min_trades=settings.workbench_defaults.research.min_trades,
        min_history_days=settings.workbench_defaults.research.min_history_days,
        min_minute_ratio=settings.workbench_defaults.research.min_minute_ratio,
        prior_policy="off",
        source_etf=None,
        requested_data_version=None,
        research_origin=None,
    )


def _policy_config(
    common: OptimizationConfig, policy: str, data_version: str,
) -> OptimizationConfig:
    if policy not in POLICIES:
        raise ValueError(f"未知先验策略：{policy}")
    if "prior_policy" not in OptimizationConfig.model_fields:
        raise RuntimeError(
            "当前代码尚未提供 OptimizationConfig.prior_policy；"
            "等待来源集成完成后再运行 benchmark"
        )
    payload = common.model_dump(mode="python")
    payload.update({
        "prior_policy": policy,
        # This experiment isolates the intrinsic daily-state prior.  A source
        # ETF reference is intentionally absent for both arms.
        "source_etf": None,
        "requested_data_version": data_version,
        "research_origin": None,
    })
    config = OptimizationConfig.model_validate(payload)
    if config.prior_policy != policy:
        raise RuntimeError(f"prior_policy={policy} 未被当前模型接受")
    return config


def _day(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if stamp is pd.NaT else stamp.normalize()


def _candidate_cutoffs(
    inventory: list[dict[str, Any]], latest: pd.Timestamp,
) -> list[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for item in inventory:
        for key in ("last_time", "verified_through"):
            stamp = _day(item.get(key))
            if stamp is not None and stamp <= latest:
                dates.add(stamp)
    return sorted(dates, reverse=True)


def _cutoff_candidates(
    inventory: list[dict[str, Any]], cutoff: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows = []
    for item in inventory:
        if not item.get("symbol") or str(item.get("file_status") or "readable") != "readable":
            continue
        last = _day(item.get("last_time"))
        verified_from = _day(item.get("verified_from"))
        verified_through = _day(item.get("verified_through"))
        if (
            last is None or last < cutoff
            or verified_from is None or verified_from > cutoff
            or verified_through is None or verified_through < cutoff
            or int(item.get("row_count") or 0) <= 0
        ):
            continue
        rows.append(item)
    rows.sort(key=lambda item: str(item["symbol"]))
    return rows


def _select_cohort(
    store: Store, market: MarketDataService, inventory: list[dict[str, Any]],
    settings, arguments,
) -> dict[str, Any]:
    """Find the latest cutoff with at least three runnable research-ready ETFs."""
    latest = min(_day(latest_closed_date()), max(
        (_day(item.get("last_time")) for item in inventory if _day(item.get("last_time")) is not None),
        default=_day(latest_closed_date()),
    ))
    attempts: list[dict[str, Any]] = []
    best_partial: dict[str, Any] | None = None
    for cutoff in _candidate_cutoffs(inventory, latest):
        candidates = _cutoff_candidates(inventory, cutoff)
        if len(candidates) < SYMBOL_LIMIT:
            continue
        symbols = [str(item["symbol"]) for item in candidates]
        plan = preflight(
            store,
            EtfResearchRequest(
                scope="selected", symbols=symbols, as_of=cutoff.date(), window_days=250,
                validation_top_k=0, max_runtime_seconds=arguments.max_runtime_seconds,
            ),
            market=market, persist=False,
        )
        ready_by_symbol = {
            row["symbol"]: row for row in plan.get("rows", [])
            if row.get("status") == "ready"
        }
        exact: list[tuple[str, str, dict[str, Any]]] = []
        exact_failures: list[dict[str, str]] = []
        for symbol in sorted(ready_by_symbol):
            manifest = store.snapshot(symbol, persist=False, repository=market.repository)
            version = str((manifest or {}).get("version") or "")
            runnable = bool(
                manifest and version
                and store.snapshot(symbol, version, persist=False, repository=market.repository)
            )
            if runnable:
                exact.append((symbol, version, manifest))
            else:
                exact_failures.append({"symbol": symbol, "reason": "活动快照不能按版本重新加载"})
        all_calendars = {
            symbol: _calendar_for(market, symbol, manifest, cutoff)
            for symbol, _version, manifest in exact
        }
        all_common = None
        for days in all_calendars.values():
            all_common = days if all_common is None else all_common.intersection(days)
        all_common = all_common.sort_values() if all_common is not None else pd.DatetimeIndex([])
        training_ready: list[tuple[str, str, dict[str, Any]]] = []
        training_failures: list[dict[str, Any]] = []
        if len(all_common) >= EVALUATION_TRADING_DAYS:
            all_start = all_common[-EVALUATION_TRADING_DAYS].date()
            for symbol, version, manifest in exact:
                base = _base(settings, symbol, all_start, cutoff.date())
                prepared = preparation_config(store, base, version, lookback_days=500)
                readiness = market.readiness(prepared, version)
                if readiness.get("ready"):
                    training_ready.append((symbol, version, manifest))
                else:
                    training_failures.append({
                        "symbol": symbol, "stage": "full_training_readiness",
                        "gaps": readiness.get("gaps") or [],
                    })
        chosen = training_ready[:SYMBOL_LIMIT]
        calendars = {
            symbol: all_calendars[symbol]
            for symbol, _version, _manifest in chosen
        }
        common = None
        for days in calendars.values():
            common = days if common is None else common.intersection(days)
        common = common.sort_values() if common is not None else pd.DatetimeIndex([])
        attempt = {
            "cutoff": cutoff.date().isoformat(),
            "candidate_count": len(candidates),
            "preflight_counts": plan.get("counts"),
            "formal_ready": sorted(ready_by_symbol),
            "runnable_ready": [symbol for symbol, _version, _manifest in exact],
            "exact_snapshot_failures": exact_failures,
            "training_ready": [symbol for symbol, _version, _manifest in training_ready],
            "training_readiness_failures": training_failures,
            "selected": [symbol for symbol, _version, _manifest in chosen],
            "lookback_common_trading_days": int(len(common)),
            "evaluation_trading_days": int(min(EVALUATION_TRADING_DAYS, len(common))),
        }
        attempts.append(attempt)
        selection = {
            "cutoff": cutoff,
            "symbols": [symbol for symbol, _version, _manifest in chosen],
            "ready_by_symbol": ready_by_symbol,
            "source_versions": {symbol: version for symbol, version, _manifest in chosen},
            "common": common,
            "plan": plan,
        }
        if len(chosen) >= SYMBOL_LIMIT and len(common) >= EVALUATION_TRADING_DAYS:
            return {**selection, "attempts": attempts, "sufficient": True}
        if len(chosen) and len(common) >= EVALUATION_TRADING_DAYS:
            if best_partial is None or len(chosen) > len(best_partial["symbols"]):
                best_partial = selection
    if best_partial is not None:
        return {**best_partial, "attempts": attempts, "sufficient": False}
    return {
        "cutoff": latest, "symbols": [], "ready_by_symbol": {},
        "source_versions": {}, "common": pd.DatetimeIndex([]), "plan": None,
        "attempts": attempts, "sufficient": False,
    }


def _calendar_for(
    service: MarketDataService, symbol: str, manifest: dict[str, Any], cutoff: pd.Timestamp,
) -> pd.DatetimeIndex:
    daily = (manifest.get("market_versions") or {}).get("1day") or {}
    auxiliary = (manifest.get("auxiliary_versions") or {}).get("calendar")
    if not daily or daily.get("version") is None:
        return pd.DatetimeIndex([])
    return service.repository.trading_days(
        symbol,
        cutoff - pd.Timedelta(days=CALENDAR_LOOKBACK_DAYS),
        cutoff,
        calendar_version=auxiliary,
        daily_version=int(daily["version"]),
    )


def _normalised_day_audit(values: Any, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    """Normalize source timestamps in memory without changing the source rows."""
    try:
        stamps = pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce"))
    except (TypeError, ValueError):
        stamps = pd.DatetimeIndex([])
    if getattr(stamps, "tz", None) is not None:
        stamps = stamps.tz_convert("Asia/Shanghai").tz_localize(None)
    valid = ~stamps.isna()
    timestamps = stamps[valid]
    intraday = timestamps != timestamps.normalize()
    dates = timestamps.normalize()
    in_range = dates[(dates >= start) & (dates <= end)]
    weekend = in_range.weekday >= 5
    unique = pd.DatetimeIndex(in_range.unique()).sort_values()
    trading = unique[unique.weekday < 5]
    return trading, {
        "source_timestamps": int(len(stamps)),
        "invalid_timestamps": int((~valid).sum()),
        "intraday_timestamps": int(intraday.sum()),
        "weekend_timestamps": int(weekend.sum()),
        "unique_dates_all_after_normalization": int(len(dates.unique())),
        "outside_range_timestamps": int(len(dates) - len(in_range)),
        "duplicate_dates_after_normalization": int(len(in_range) - len(unique)),
        "unique_dates_after_normalization": int(len(unique)),
        "trading_days_after_weekday_filter": int(len(trading)),
    }


def _calendar_audit(
    service: MarketDataService, symbol: str, manifest: dict[str, Any],
    symbol_days: pd.DatetimeIndex, common_days: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    start = cutoff - pd.Timedelta(days=CALENDAR_LOOKBACK_DAYS)
    auxiliary = (manifest.get("auxiliary_versions") or {}).get("calendar")
    payload = service.repository.auxiliary(symbol, "calendar", version=auxiliary) or {}
    raw_values = [row.get("date") for row in (payload.get("payload") or []) if isinstance(row, dict)]
    _raw_days, raw_audit = _normalised_day_audit(raw_values, start, cutoff)
    _symbol_days, symbol_audit = _normalised_day_audit(symbol_days, start, cutoff)
    _common_days, common_audit = _normalised_day_audit(common_days, start, cutoff)
    return {
        "lookback_start": start.date().isoformat(),
        "lookback_end": cutoff.date().isoformat(),
        "auxiliary_calendar_version": auxiliary,
        "auxiliary_payload": raw_audit,
        "symbol_trading_calendar": symbol_audit,
        "common_trading_calendar": common_audit,
        "normalization": (
            "source timestamps converted in memory to Asia/Shanghai, timezone removed, "
            "normalized to dates, deduplicated, range-filtered and weekend-filtered; "
            "source rows are unchanged"
        ),
    }


def _prior_summary(result: dict[str, Any]) -> dict[str, Any] | None:
    fixed = result.get("fixed") if isinstance(result.get("fixed"), dict) else {}
    diagnostics = fixed.get("diagnostics") if isinstance(fixed.get("diagnostics"), dict) else {}
    prior = diagnostics.get("research_prior")
    if not isinstance(prior, dict):
        return None
    checks = prior.get("candidate_checks") if isinstance(prior.get("candidate_checks"), list) else []
    status_counts = Counter(
        str(row.get("status")) for row in checks
        if isinstance(row, dict) and row.get("status") is not None
    )
    return {
        "policy": prior.get("policy"),
        "decision_date": prior.get("decision_date"),
        "phase": prior.get("phase"),
        "execution_data_version": prior.get("execution_data_version"),
        "quota": prior.get("quota"),
        "planned_count": prior.get("planned_count"),
        "quota_used": prior.get("quota_used"),
        "actually_evaluated_count": prior.get("actually_evaluated_count"),
        "annotation_only_count": prior.get("annotation_only_count"),
        "source_seed_count": prior.get("source_seed_count"),
        "state_candidate_count": prior.get("state_candidate_count"),
        "market_state": prior.get("market_state"),
        "market_state_error": prior.get("market_state_error"),
        "source_check": prior.get("source_check"),
        "candidate_check_status_counts": dict(sorted(status_counts.items())),
        "candidate_checks": checks[:64],
        "candidate_checks_truncated": len(checks) > 64,
        "coordinate_values": prior.get("coordinate_values"),
    }


def _candidate_failures(result: dict[str, Any]) -> dict[str, int]:
    fixed = result.get("fixed") if isinstance(result.get("fixed"), dict) else {}
    diagnostics = fixed.get("diagnostics") if isinstance(fixed.get("diagnostics"), dict) else {}
    reasons: list[str] = []
    for value in (
        *(fixed.get("reasons") or []),
        *(diagnostics.get("selected_constraint_reasons") or []),
    ):
        if value not in (None, ""):
            reasons.append(str(value))
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    archives = evidence.get("candidate_archives") if isinstance(evidence.get("candidate_archives"), dict) else {}
    for archive in archives.values():
        if not isinstance(archive, list):
            continue
        for row in archive:
            if not isinstance(row, dict):
                continue
            for value in (*(row.get("reasons") or []), *(row.get("constraint_reasons") or [])):
                if value not in (None, ""):
                    reasons.append(str(value))
    return dict(Counter(reasons).most_common(24))


def _run_summary(
    result: dict[str, Any], policy: str, elapsed: float, warning: str | None = None,
) -> dict[str, Any]:
    fixed = result.get("fixed") if isinstance(result.get("fixed"), dict) else {}
    diagnostics = fixed.get("diagnostics") if isinstance(fixed.get("diagnostics"), dict) else {}
    current = fixed.get("current") if isinstance(fixed.get("current"), dict) else {}
    current_diagnostics = (
        current.get("diagnostics") if isinstance(current.get("diagnostics"), dict) else {}
    )
    computation = result.get("computation") if isinstance(result.get("computation"), dict) else {}
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    archives = evidence.get("candidate_archives") if isinstance(evidence.get("candidate_archives"), dict) else {}
    labelled_prior_rows = {
        tuple(row.get("key") or ())
        for archive in archives.values() if isinstance(archive, list)
        for row in archive if isinstance(row, dict) and row.get("prior_sources")
    }
    metrics = fixed.get("oos_metrics") if isinstance(fixed.get("oos_metrics"), dict) else None
    status = result.get("status")
    summary = {
        "status": status,
        "partial": status == "partial",
        "policy": policy,
        "elapsed_seconds": round(float(elapsed), 6),
        "runtime_limit_seconds": computation.get("max_runtime_seconds"),
        "runtime_limit_hit": bool(
            computation.get("budget_stopped")
            and "时间" in str(computation.get("budget_reason") or "")
        ),
        "data_version": result.get("data_version"),
        "fixed_status": fixed.get("status"),
        "j": {
            "score": fixed.get("score"),
            "selected_eligible": fixed.get("eligible"),
            "qualified_count": diagnostics.get("eligible_center_count"),
            "recommendation_count": diagnostics.get("recommendation_count"),
            "candidate_count": diagnostics.get("candidate_count"),
            "score_complete": diagnostics.get("score_complete"),
            "coverage_status": diagnostics.get("coverage_status"),
            "training_unique_count": diagnostics.get("training_unique_count"),
            "neighborhood_unique_count": diagnostics.get("neighborhood_unique_count"),
            "reporting_unique_count": diagnostics.get("reporting_unique_count"),
        },
        "candidate_evaluations": computation.get("candidate_evaluations"),
        "ledger_evaluations": computation.get("ledger_evaluations"),
        "budget": {
            "training": computation.get("training_budget"),
            "validation": computation.get("neighborhood_budget"),
            "reporting": computation.get("reporting_budget"),
            "total": computation.get("total_evaluation_budget"),
            "effective": computation.get("frozen_evaluation_budget"),
            "used": computation.get("evaluation_budget_used"),
            "remaining": computation.get("evaluation_budget_remaining"),
            "stopped": computation.get("budget_stopped"),
            "reason": computation.get("budget_reason"),
        },
        "training_window": diagnostics.get("training_window"),
        "selected_params": fixed.get("recommended_params"),
        "prior_source_labelled_candidate_count": len(labelled_prior_rows),
        "prior": _prior_summary(result),
        "candidate_failure_reasons": _candidate_failures(result),
        "failure_reasons": list(dict.fromkeys([
            *(fixed.get("reasons") or []), *(result.get("warnings") or []),
        ])),
        "oos": {
            "excess_return": (metrics or {}).get("excess_return"),
            "grid_peak_principal_return": (metrics or {}).get("peak_principal_return"),
            "matched_funding_hold_return": (metrics or {}).get("matched_funding_hold_return"),
            "grid_principal_drawdown": (metrics or {}).get("principal_drawdown"),
            "matched_funding_hold_drawdown": (metrics or {}).get("matched_funding_hold_drawdown"),
            "trade_count": (metrics or {}).get("trade_count"),
        },
        "current": {
            "semantics": "same evaluation-start anchor as fixed; independent display only",
            "same_anchor_as_fixed": current.get("as_of") == fixed.get("as_of"),
            "status": current.get("status"),
            "decision_date": current.get("as_of"),
            "selected_params": current.get("recommended_params"),
            "score": current.get("score"),
            "eligible": current.get("eligible"),
            "candidate_count": current_diagnostics.get("candidate_count"),
            "eligible_count": current_diagnostics.get("eligible_center_count"),
            "reasons": current.get("reasons") or [],
        },
        "config_fingerprint": _digest(result.get("config")),
    }
    if warning:
        summary["execution_error"] = warning
    return summary


def _run_one(
    config: OptimizationConfig, bundle, *, warning: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    last = {"stage": None}

    def progress(fraction: float, message: str) -> None:
        last["stage"] = {"fraction": round(float(fraction), 6), "message": str(message)}

    try:
        result = optimize_v5(config, bundle, progress=progress, cancelled=lambda: False)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "status": "failed", "partial": True, "policy": config.prior_policy,
            "elapsed_seconds": round(elapsed, 6),
            "runtime_limit_seconds": config.strategy.max_runtime_seconds,
            "execution_error": f"{type(exc).__name__}: {exc}",
            "last_progress": last["stage"],
            "config_fingerprint": _digest(config.model_dump(mode="json")),
        }
    elapsed = time.perf_counter() - started
    summary = _run_summary(result, config.prior_policy, elapsed, warning)
    summary["last_progress"] = last["stage"]
    return summary


def _preview_summary(
    preview: dict[str, Any], stress_region_count: int,
) -> dict[str, Any]:
    prior = preview.get("research_prior") if isinstance(preview.get("research_prior"), dict) else None
    stress = None
    if "peak_support_volume" in preview:
        stress = int(stress_region_count) * int(preview["peak_support_volume"])
    return {
        "status": preview.get("status"),
        "reasons": preview.get("reasons") or [],
        "training_window": preview.get("training_window"),
        "minimum_neighborhood_size": preview.get("minimum_neighborhood_size"),
        "validation_budget": preview.get("neighborhood_budget"),
        "stress_points_at_three_regions": stress,
        "task_evaluation_upper_bound": preview.get("task_evaluation_upper_bound"),
        "effective_total_evaluation_budget": preview.get("effective_total_evaluation_budget"),
        "decision_count_estimate": preview.get("decision_count_estimate"),
        "estimated_task_memory_mb": preview.get("estimated_task_memory_mb"),
        "workspace_budget_mb": preview.get("workspace_budget_mb"),
        "prior_quota": (prior or {}).get("quota"),
        "prior_quota_used": (prior or {}).get("quota_used"),
    }


def main() -> int:
    arguments = _parser().parse_args()
    settings = load_settings()
    store = Store(str(resolve_project_path(settings.data_dir)), settings=settings)
    market = MarketDataService(store, settings=settings)
    service = DataService(store, settings=settings)
    inventory = market.datasets(timeframe="1min", limit=10_000).get("items", [])
    cohort = _select_cohort(store, market, inventory, settings, arguments)
    symbols = list(cohort["symbols"])
    cutoff = cohort["cutoff"]
    common = cohort["common"]
    source_versions = cohort["source_versions"]
    ready_by_symbol = cohort["ready_by_symbol"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "research_priors_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "benchmark_run": False,
        "diagnostic_code": {
            "market_state_sha256": hashlib.sha256(
                (PROJECT_ROOT / "src/stockregression/market_state.py").read_bytes()
            ).hexdigest(),
            "market_state_test_sha256": hashlib.sha256(
                (PROJECT_ROOT / "tests/test_market_state.py").read_bytes()
            ).hexdigest(),
            "adx_golden": "test_wilder_adx_and_natr_match_fixed_talib_064_golden",
            "adx_golden_reference": "C TA-Lib 0.6.4",
        },
        "experiment": {
            "hypothesis": "在同一评估区间比较 off 与 daily_state_v1；不预设正收益或优越性",
            "selection": "从最近共同截止日回溯，取正式研究 ready 且快照可重载的 ETF，按代码取前三只",
            "frequency": "1m",
            "evaluation_trading_days": EVALUATION_TRADING_DAYS,
            "training_days": {"minimum": 250, "maximum": 500, "window_mode": "settings"},
            "selection_target": "fixed",
            "current_semantics": (
                "current reuses the same evaluation-start fixed_selection; "
                "it is not a latest-market recommendation"
            ),
            "seed": arguments.seed,
            "source_etf": None,
            "prior_policy_order": list(POLICIES),
            "persisted_application_result": False,
        },
        "budgets": {
            "training": arguments.training_budget,
            "validation": arguments.validation_budget,
            "reporting": arguments.reporting_budget,
            "total": arguments.total_evaluation_budget,
            "max_runtime_seconds_per_run": arguments.max_runtime_seconds,
        },
        "catalog": {
            "inventory_1min_rows": len(inventory),
            "latest_catalog_ready_count": sum(item.get("ready") is True for item in inventory),
            "ready_count": len(symbols),
            "requested_symbol_count": SYMBOL_LIMIT,
            "selected_symbols": symbols,
            "unfilled_symbol_slots": max(0, SYMBOL_LIMIT - len(symbols)),
        },
        "symbols": {},
        "failures": [],
        "warnings": [],
    }
    if len(symbols) < SYMBOL_LIMIT:
        report["warnings"].append(
            f"最新可运行共同区间仅找到 {len(symbols)} 只正式研究 ready 且快照可重载的 ETF，少于目标 3 只；按实际可用数量记录 partial"
        )
    if not symbols:
        report["status"] = "blocked"
        report["failures"].append({"stage": "selection", "reason": "没有本地研究就绪 ETF"})
    if "prior_policy" not in OptimizationConfig.model_fields:
        report["status"] = "blocked"
        report["failures"].append({
            "stage": "capability",
            "reason": "OptimizationConfig.prior_policy 尚未集成；未运行 benchmark",
        })

    if symbols and report["status"] != "blocked":
        lookback_start = cutoff - pd.Timedelta(days=CALENDAR_LOOKBACK_DAYS)
        evaluation_days = common[-EVALUATION_TRADING_DAYS:] if len(common) >= EVALUATION_TRADING_DAYS else pd.DatetimeIndex([])
        _common_trading, common_audit = _normalised_day_audit(common, lookback_start, cutoff)
        calendar_audits = {}
        for symbol in symbols:
            manifest = store.snapshot(symbol, persist=False, repository=market.repository)
            if manifest:
                symbol_days = _calendar_for(market, symbol, manifest, cutoff)
                calendar_audits[symbol] = _calendar_audit(
                    market, symbol, manifest, symbol_days, common, cutoff,
                )
        report["selection"] = {
            "method": "latest_common_formal_ready_backscan",
            "cutoff": cutoff.date().isoformat(),
            "lookback_start": lookback_start.date().isoformat(),
            "lookback_end": cutoff.date().isoformat(),
            "lookback_common_trading_days": int(len(common)),
            "lookback_common_calendar_audit": common_audit,
            "evaluation_trading_days_required": EVALUATION_TRADING_DAYS,
            "evaluation_trading_days": int(len(evaluation_days)),
            "evaluation_start": evaluation_days[0].date().isoformat() if len(evaluation_days) else None,
            "evaluation_end": evaluation_days[-1].date().isoformat() if len(evaluation_days) else None,
            "evaluation_weekend_days": int(sum(day.weekday() >= 5 for day in evaluation_days)),
            "evaluation_duplicate_dates": int(len(evaluation_days) - len(evaluation_days.unique())),
            "evaluation_intraday_timestamps": int(sum(day != day.normalize() for day in evaluation_days)),
            "calendar_audit_by_symbol": calendar_audits,
            "shortened": False,
            "sufficient_symbols": len(symbols) >= SYMBOL_LIMIT,
            "attempts": cohort["attempts"],
        }
        report["catalog"]["source_versions"] = source_versions
        if len(common) < EVALUATION_TRADING_DAYS:
            report["status"] = "blocked"
            report["failures"].append({
                "stage": "common_calendar",
                "reason": f"共同交易日仅 {len(common)} 日，少于要求的 {EVALUATION_TRADING_DAYS} 日；未改用日线或缩短评估期",
            })
        else:
            end = common[-1].date()
            start = common[-EVALUATION_TRADING_DAYS].date()
            plan = cohort["plan"] or {}
            report["selection"]["preflight"] = {
                "as_of": plan.get("as_of"),
                "counts": plan.get("counts"),
                "rows": [
                    {
                        "symbol": row.get("symbol"), "status": row.get("status"),
                        "version": row.get("version"),
                        "snapshot_version": source_versions.get(row.get("symbol")),
                        "window_start": row.get("window_start"),
                        "window_end": row.get("window_end"),
                        "gaps": row.get("gaps") or [], "warnings": row.get("warnings") or [],
                    }
                    for row in plan.get("rows", []) if row.get("symbol") in symbols
                ],
            }
            if not arguments.selection_only:
                for symbol in symbols:
                    row = ready_by_symbol[symbol]
                    base = _base(settings, symbol, start, end)
                    # preflight augments the in-memory manifest version with its
                    # quality-evidence hash when persist=False.  Execution must
                    # lock the active snapshot that was resolved before that
                    # temporary document transformation.
                    version = source_versions.get(symbol) or ""
                    prepared = preparation_config(store, base, version, lookback_days=500)
                    readiness = market.readiness(prepared, version)
                    symbol_report: dict[str, Any] = {
                        "symbol": symbol,
                        "source_version": version,
                        "target_version": version,
                        "source_etf": None,
                        "evaluation": {
                            "start": start.isoformat(), "end": end.isoformat(),
                            "frequency": "1m", "trading_days": EVALUATION_TRADING_DAYS,
                        },
                        "readiness": readiness,
                        "runs": {},
                    }
                    if not readiness.get("ready"):
                        symbol_report["status"] = "data_not_ready"
                        report["failures"].append({
                            "stage": "readiness", "symbol": symbol,
                            "reason": "训练前置与评估区间未通过本地 1 分钟数据就绪检查",
                            "gaps": readiness.get("gaps") or [],
                        })
                        report["symbols"][symbol] = symbol_report
                        continue
                    try:
                        bundle = prepare_research_bundle(
                            service, base, version,
                            lookback_days=500,
                        )
                    except Exception as exc:
                        symbol_report["status"] = "failed"
                        symbol_report["error"] = f"{type(exc).__name__}: {exc}"
                        report["failures"].append({
                            "stage": "prepare", "symbol": symbol,
                            "reason": symbol_report["error"],
                        })
                        report["symbols"][symbol] = symbol_report
                        continue
                    prior_days = int(
                        (pd.DatetimeIndex(pd.to_datetime(bundle.calendar)).normalize() < pd.Timestamp(start)).sum()
                    ) if len(bundle.calendar) else 0
                    symbol_report["preparation"] = {
                        "requested_start": prepared.start_date.isoformat(),
                        "bundle_version": bundle.version,
                        "prior_trading_days": prior_days,
                        "warning_count": len(bundle.warnings or []),
                        "warnings": (bundle.warnings or [])[:20],
                    }
                    if bundle.version != version or prior_days < 250:
                        symbol_report["status"] = "insufficient_history"
                        report["failures"].append({
                            "stage": "training_window", "symbol": symbol,
                            "reason": (
                                "数据版本在运行前发生变化" if bundle.version != version
                                else f"评估前仅有 {prior_days} 个交易日，不足现有模型最低 250 日；未缩短评估期"
                            ),
                        })
                        report["symbols"][symbol] = symbol_report
                        continue
                    common_config = _common_config(settings, base, arguments)
                    configs = {
                        policy: _policy_config(common_config, policy, version)
                        for policy in POLICIES
                    }
                    symbol_report["configs"] = {
                        policy: config.model_dump(mode="json") for policy, config in configs.items()
                    }
                    symbol_report["config_fingerprints"] = {
                        policy: _digest(config.model_dump(mode="json"))
                        for policy, config in configs.items()
                    }
                    previews = {
                        policy: estimate_robust_search(config, bundle)
                        for policy, config in configs.items()
                    }
                    symbol_report["previews"] = {
                        policy: _preview_summary(
                            preview,
                            int(configs[policy].strategy.stress_region_count),
                        )
                        for policy, preview in previews.items()
                    }
                    blocked = [
                        policy for policy, preview in previews.items()
                        if preview.get("status") not in {"ready", "data_issue", "insufficient_history", "insufficient_range"}
                        or preview.get("status") != "ready"
                    ]
                    if blocked:
                        symbol_report["status"] = "budget_or_data_blocked"
                        report["failures"].append({
                            "stage": "preview", "symbol": symbol,
                            "reason": "先验预算或研究预检未通过",
                            "policies": blocked,
                        })
                        report["symbols"][symbol] = symbol_report
                        continue
                    prior_preview = previews["daily_state_v1"].get("research_prior") or {}
                    quota = int(prior_preview.get("quota") or 0)
                    stress = (
                        int(configs["daily_state_v1"].strategy.stress_region_count)
                        * int(previews["daily_state_v1"].get("peak_support_volume") or 0)
                    )
                    if quota > 8 or quota > int(arguments.training_budget) * 0.20:
                        symbol_report["status"] = "budget_exceeded"
                        report["failures"].append({
                            "stage": "prior_budget", "symbol": symbol,
                            "reason": f"先验配额 {quota} 超过 20%/8 上限",
                        })
                        report["symbols"][symbol] = symbol_report
                        continue
                    if stress != 81:
                        report["warnings"].append(
                            f"{symbol} 的三区域压力评价上界为 {stress}，不是约定的 81；请检查 3x3x3 邻域与压力区域数"
                        )
                    for policy in POLICIES:
                        symbol_report["runs"][policy] = _run_one(configs[policy], bundle)
                    off, on = symbol_report["runs"]["off"], symbol_report["runs"]["daily_state_v1"]
                    off_excess = ((off.get("oos") or {}).get("excess_return"))
                    on_excess = ((on.get("oos") or {}).get("excess_return"))
                    off_dd = ((off.get("oos") or {}).get("grid_principal_drawdown"))
                    on_dd = ((on.get("oos") or {}).get("grid_principal_drawdown"))
                    off_score, on_score = (off.get("j") or {}).get("score"), (on.get("j") or {}).get("score")
                    symbol_report["comparison"] = {
                        "daily_state_minus_off": {
                            "oos_excess_return": (
                                float(on_excess) - float(off_excess)
                                if on_excess is not None and off_excess is not None else None
                            ),
                            "oos_principal_drawdown": (
                                float(on_dd) - float(off_dd)
                                if on_dd is not None and off_dd is not None else None
                            ),
                            "j_score": (
                                float(on_score) - float(off_score)
                                if on_score is not None and off_score is not None else None
                            ),
                        },
                        "outcome_claim": "仅记录差异；单次小预算对照不能证明先验带来正收益",
                    }
                    statuses = [symbol_report["runs"][policy].get("status") for policy in POLICIES]
                    symbol_report["status"] = (
                        "completed" if statuses == ["completed", "completed"] else "partial"
                    )
                    report["symbols"][symbol] = symbol_report

    if arguments.selection_only and report["status"] != "blocked":
        report["status"] = "selection_only"
    if not arguments.selection_only:
        statuses = [item.get("status") for item in report["symbols"].values()]
        if report["status"] != "blocked":
            if not statuses or any(status != "completed" for status in statuses):
                report["status"] = "partial"
            elif len(symbols) < SYMBOL_LIMIT or report["warnings"]:
                report["status"] = "partial"
            else:
                report["status"] = "completed"
        report["partial"] = report["status"] == "partial"
        report["benchmark_run"] = any(
            bool(item.get("runs"))
            for item in report["symbols"].values() if isinstance(item, dict)
        )
        report["provenance"] = computation_provenance(
            PROJECT_ROOT,
            configuration={
                "budgets": report["budgets"], "selection": report.get("selection"),
                "symbols": {
                    symbol: item.get("configs")
                    for symbol, item in report["symbols"].items()
                    if isinstance(item, dict)
                },
            },
        )
        report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = _json_safe(report)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 2 if report.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
