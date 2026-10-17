"""Reproducible real-data acceptance; defaults to the user's 159611 snapshot.

Run from the repository: .venv/Scripts/python.exe scripts/validate_scientific.py
Only existing real snapshots are used. A completed research job is saved for
inspection in the application; the primary market data is never rewritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from stockregression.api import public_result
from stockregression.data import DataService
from stockregression.jobs import execute_job
from stockregression.models import BacktestConfig, OptimizationConfig
from stockregression.optimization import _slice_bundle
from stockregression.research_preparation import prepare_research_bundle
from stockregression.research_statistics import stationary_bootstrap_indices
from stockregression.scientific import _select_at, _seed_offset_for_date
from stockregression.settings import load_settings
from stockregression.share_engine import research_batch, run_share_backtest
from stockregression.storage import Store, dumps, now_iso


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="159611.SZ")
    parser.add_argument("--start", default="2025-10-16")
    parser.add_argument("--end", default="2026-10-16")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--result-id", help="Reuse an already completed result for additional checks")
    parser.add_argument("--output", default=str(ROOT / "docs" / "159611-scientific-acceptance.json"))
    args = parser.parse_args()
    settings = load_settings()
    store = Store(settings.data_dir)
    manifest = store.snapshot(args.symbol)
    if not manifest or manifest.get("metadata", {}).get("is_demo"):
        raise SystemExit("Acceptance requires an existing real-data snapshot")
    preferences = settings.workbench_defaults
    base = BacktestConfig.model_validate({
        **preferences.grid.model_dump(), **preferences.position.model_dump(),
        "schema_version": 2, "symbol": args.symbol, "start_date": args.start,
        "end_date": args.end, "frequency": "1m", "dca": preferences.dca.model_dump(),
        "fees": settings.fee_defaults.model_dump(),
    })
    research = preferences.research.model_dump()
    research["ranges"].pop("initial_shares", None)
    config = OptimizationConfig.model_validate({**research, "base": base.model_dump(mode="json"),
        "protocol_version": 2, "selection_target": "both", "rebalance_days": 63,
        "n_trials": args.trials, "bootstrap_reps": args.bootstrap_reps})
    started = time.perf_counter()
    last_print = [0.0]
    observations = []

    def progress(fraction, message):
        observations.append((float(fraction), str(message)))
        tick = time.perf_counter()
        if tick - last_print[0] >= 5 or fraction >= 1:
            print(f"{fraction:.1%} {message}", flush=True)
            last_print[0] = tick

    job_id = None
    research_seconds_source = "elapsed_this_invocation"
    if args.result_id:
        result_id = args.result_id
        result = store.result(result_id)
        if not result or result.get("protocol_version") != 2:
            raise SystemExit("Result is not a scientific v2 study")
        config = OptimizationConfig.model_validate(result["config"])
        base = config.base
        manifest = store.snapshot(base.symbol, result["data_version"])
        existing_job = next((job for job in store.jobs(1000) if job.get("result_id") == result_id), None)
        if existing_job:
            job_id = existing_job["id"]
    else:
        # Start in running state atomically so a separately running application
        # worker cannot also claim this acceptance job from the queued list.
        job_id = uuid.uuid4().hex
        with store.db() as db:
            db.execute("INSERT INTO jobs(id,kind,status,progress,message,payload,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (job_id, "optimization", "running", 0, "159611 科学研究验收",
                 dumps({"config": config.model_dump(mode="json"), "version": manifest["version"]}), now_iso(), now_iso()))
        print(f"job_id={job_id} snapshot={manifest['version']}", flush=True)
        execute_job(str(store.root), job_id, observer=progress)
        job = store.job(job_id)
        if job["status"] != "completed" or not job.get("result_id"):
            raise RuntimeError(dumps(job))
        result_id = job["result_id"]
        result = store.result(result_id)
    research_seconds = time.perf_counter() - started
    if args.result_id and existing_job:
        research_seconds = (pd.Timestamp(existing_job["updated_at"]) - pd.Timestamp(existing_job["created_at"])).total_seconds()
        research_seconds_source = "persisted_job_wall_time"
    bundle = prepare_research_bundle(DataService(store), base, manifest["version"])
    if bundle.metadata.get("is_demo"):
        raise AssertionError("Real-data acceptance cannot use demo data")
    fixed = result["fixed"]["test_result"]
    adaptive = result["adaptive"]["test_result"]
    schedule = result["adaptive"]["schedule"]
    current = result["fixed"]["current"]
    reserve = fixed["funding"]["reserve_cash"]
    checks = {
        "completed_real_minute_research": result["status"] == "completed" and not result.get("is_demo"),
        "minute_coverage_above_default_gate": bundle.quality["minute_ratio"] >= .95,
        "same_initial_funding": fixed["funding"] == adaptive["funding"],
        "fixed_selection_precedes_evaluation": pd.Timestamp(result["fixed"]["as_of"]) < pd.Timestamp(base.start_date),
        "all_schedule_decisions_precede_execution": all(pd.Timestamp(x["decision_date"]) < pd.Timestamp(x["effective_date"]) for x in schedule),
        "current_asof_is_final_evaluation_close": current["as_of"] == adaptive["effective_end_date"],
    }
    # A counterfactual future must not change the selected historical params,
    # even if the snapshot identifier also changes.
    decision = result["evidence"]["decisions"][min(1, len(result["evidence"]["decisions"]) - 1)]
    cutoff = pd.Timestamp(decision["decision_date"])
    changed = replace(bundle, bars=bundle.bars.copy(), daily=bundle.daily.copy(),
                      metadata=deepcopy(bundle.metadata), version=bundle.version + "-counterfactual-future")
    for frame, column in [(changed.bars, "timestamp"), (changed.daily, "date")]:
        future = pd.to_datetime(frame[column]).dt.normalize() > cutoff
        for price in ("open", "high", "low", "close", "preclose"):
            if price in frame:
                frame.loc[future, price] *= 1.25
    locked = base.model_copy(update={"reserve_cash": reserve})
    print(f"Checking future independence at {cutoff.date()}", flush=True)
    selected = _select_at(config, locked, changed, cutoff, reserve_cash=reserve,
        seed_offset=_seed_offset_for_date(cutoff), progress=progress, cancelled=None)
    checks["future_change_keeps_historical_parameters"] = selected["params"] == decision["selected_params"]
    checks["future_change_keeps_historical_score"] = bool(np.isclose(selected["score"], decision["score"], rtol=0, atol=1e-12))
    counterfactual_run = run_share_backtest(locked, changed, parameter_schedule=schedule, daily_details=True)

    def prefix(rows):
        return [row for row in rows if pd.Timestamp(row["timestamp"]).normalize() <= cutoff]

    checks["future_change_keeps_past_trades"] = prefix(counterfactual_run["strategies"]["grid"]["trades"]) == prefix(adaptive["strategies"]["grid"]["trades"])
    checks["future_change_keeps_past_equity"] = prefix(counterfactual_run["strategies"]["grid"]["equity"]) == prefix(adaptive["strategies"]["grid"]["equity"])
    # Verify the accelerated kernel against scalar execution on the same real
    # minute path and fixed account, including exact intraday fold drawdown.
    days = bundle.calendar[-126:]
    small = _slice_bundle(bundle, days[0], days[-1])
    small_base = locked.model_copy(update={"start_date": days[0].date(), "end_date": days[-1].date()})
    current_archive = result["evidence"]["candidate_archives"]["current"]
    parameters = [row["params"] for row in current_archive
                  if row.get("evaluation_status") == "completed"][:128]
    if len(parameters) < 2:
        parameters = [decision["selected_params"], current["params"]]
    windows = [(str(days[0].date()), str(days[62].date())), (str(days[63].date()), str(days[-1].date()))]
    # Warm both kernels and shared market preparation before timing. Report
    # medians over identical inputs, with no minimum speedup acceptance gate.
    research_batch(small_base, small, parameters, windows)
    research_batch(small_base, small, parameters, windows, parallel=False)
    timings = {"batch": [], "scalar": []}
    for _ in range(3):
        before = time.perf_counter()
        fast = research_batch(small_base, small, parameters, windows)
        timings["batch"].append(time.perf_counter() - before)
        before = time.perf_counter()
        scalar = research_batch(small_base, small, parameters, windows, parallel=False)
        timings["scalar"].append(time.perf_counter() - before)
    fast_seconds = float(np.median(timings["batch"]))
    scalar_seconds = float(np.median(timings["scalar"]))
    maximum_difference = float(np.max(np.abs(fast["daily_returns"] - scalar["daily_returns"])))
    checks["batch_matches_scalar_returns"] = maximum_difference <= 1e-12
    checks["batch_matches_scalar_intraday_risk_and_fees"] = fast["fold_metrics"] == scalar["fold_metrics"] and fast["metrics"] == scalar["metrics"]
    # Compare only the resampling reduction operator: the same shared stationary
    # indices drive a reference loop and the batched count-matrix product.
    centered = np.log1p(fast["daily_returns"])
    centered -= centered.mean(axis=0)
    rng = np.random.default_rng(config.seed)
    indices = np.asarray([stationary_bootstrap_indices(len(days), 20, rng) for _ in range(256)])
    weights = np.asarray([np.bincount(row, minlength=len(days)) for row in indices], dtype=np.float64)
    before = time.perf_counter()
    reference_maxima = np.asarray([centered[row].mean(axis=0).max() for row in indices])
    reduction_scalar_seconds = time.perf_counter() - before
    before = time.perf_counter()
    matrix_maxima = (weights @ centered / len(days)).max(axis=1)
    reduction_matrix_seconds = time.perf_counter() - before
    reduction_difference = float(np.max(np.abs(reference_maxima - matrix_maxima)))
    checks["bootstrap_matrix_matches_shared_index_reference"] = reduction_difference <= 1e-12
    preview = run_share_backtest(locked.model_copy(update={**current["params"], "selection_asof": date.fromisoformat(current["as_of"])}), bundle, preview=True)
    checks["latest_parameter_replay_is_marked_in_sample"] = preview["parameter_selection"]["retrospective"]
    public = public_result(result)
    checks["api_keeps_both_scientific_curves"] = all(public[key]["test_result"]["strategies"]["grid"]["equity_frequency"] == "1d" for key in ("fixed", "adaptive"))
    checks["api_keeps_full_archives_off_default_response"] = "candidate_archives" not in public["evidence"] and "trial_archives" not in public["evidence"]
    check = result["evidence"]["final_126_day_check"]
    checks["final_check_uses_126_days"] = check["days"] == 126
    checks["bootstrap_completed_requested_repetitions"] = check["bootstrap"]["primary"]["completed_reps"] == config.bootstrap_reps
    family = current.get("diagnostics", {}).get("candidate_family_bootstrap", {})
    checks["current_family_tests_all_frozen_candidates"] = family.get("family_count", 0) == len(current_archive) and len(current_archive) > 2
    checks["current_family_is_labeled_internal_not_independent_oos"] = family.get("sample_role") == "internal_validation_used_for_selection" and family.get("independent_oos") is False
    checks["current_family_bootstrap_completed_requested_repetitions"] = family.get("bootstrap", {}).get("primary", {}).get("completed_reps") == config.bootstrap_reps
    archives = result["evidence"]["candidate_archives"]
    unique_archives = [archives["fixed"], archives["current"], *[
        archive["candidate_archive"]
        for decision_row, archive in zip(result["evidence"]["decisions"], archives["adaptive"])
        if not decision_row.get("reused_from_fixed")
    ]]
    checks["computation_counts_reused_selection_only_once"] = (
        result["computation"]["unique_selection_calls"] == len(unique_archives)
        and result["computation"]["candidate_count"] == sum(len(archive) for archive in unique_archives)
        and result["computation"]["neighborhood_count"] == sum(not row.get("is_center") for archive in unique_archives for row in archive)
    )
    if not args.result_id:
        checks["progress_completes_only_after_scientific_selection"] = all(
            "科学参数选择完成" in message for fraction, message in observations if fraction >= 1
        )
    compact_current = deepcopy({key: value for key, value in current.items() if key != "candidates"})
    compact_current["diagnostics"].pop("costs", None)
    compact_current["diagnostics"].get("candidate_family_bootstrap", {}).pop("candidate_ids", None)
    report = {
        "symbol": base.symbol, "snapshot": bundle.version, "is_demo": False,
        "created_at": now_iso(), "job_id": job_id, "result_id": result_id,
        "evaluation_start": str(base.start_date), "evaluation_end": str(base.end_date),
        "config": config.model_dump(mode="json"), "locked_funding": fixed["funding"],
        "minute_ratio": bundle.quality["minute_ratio"], "prepared_trading_days": len(bundle.calendar),
        "evaluation_trading_days": result["evidence"]["evaluation_days"],
        "checks": checks, "all_checks_passed": all(checks.values()),
        "fixed_metrics": fixed["strategies"]["grid"]["metrics"],
        "adaptive_metrics": adaptive["strategies"]["grid"]["metrics"],
        "same_account_passive_metrics": fixed["strategies"]["same_account_passive"]["metrics"],
        "current": compact_current,
        "schedule": schedule, "final_126_day_check": check,
        "local_audit": result["evidence"].get("local_audit"), "computation": result["computation"],
        "timing": {"research_job_seconds": research_seconds, "research_seconds_source": research_seconds_source,
                   "total_acceptance_seconds": time.perf_counter() - started,
                   "warm_benchmark_candidates": len(parameters), "warm_benchmark_days": len(days),
                   "warm_benchmark_repetitions": 3,
                   "batch_median_seconds": fast_seconds, "scalar_median_seconds": scalar_seconds,
                   "scalar_over_batch_speedup": scalar_seconds / fast_seconds,
                   "maximum_return_difference": maximum_difference,
                   "bootstrap_reduction_reps": len(indices),
                   "bootstrap_reduction_scalar_seconds": reduction_scalar_seconds,
                   "bootstrap_reduction_matrix_seconds": reduction_matrix_seconds,
                   "bootstrap_reduction_maximum_difference": reduction_difference},
        "artifacts": {"full_result": str((store.root / "results" / f"{result_id}.json").resolve()),
                      "full_result_bytes": (store.root / "results" / f"{result_id}.json").stat().st_size,
                      "default_api_bytes": len(dumps(public).encode("utf-8"))},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    temporary_output.write_text(dumps(report), encoding="utf-8")
    temporary_output.replace(output)
    print(json.dumps({"report": str(output.resolve()), "result_id": result_id, "checks": checks,
                      "all_checks_passed": report["all_checks_passed"], "timing": report["timing"]}, ensure_ascii=False), flush=True)
    if not all(checks.values()):
        raise SystemExit("Acceptance failed: " + ", ".join(key for key, passed in checks.items() if not passed))


if __name__ == "__main__":
    main()
