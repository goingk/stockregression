"""Reproducible protocol-v4 acceptance against a locked local snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockregression.data import DataService
from stockregression.models import BacktestConfig, OptimizationConfig
from stockregression.research_preparation import prepare_research_bundle
from stockregression.research_v4 import estimate_search
from stockregression.scientific import optimize_scientific
from stockregression.settings import load_settings
from stockregression.storage import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="159611.SZ")
    parser.add_argument("--start", default="2026-04-17")
    parser.add_argument("--end", default="2026-10-16")
    parser.add_argument("--frequency", choices=("1d", "1m"), default="1d")
    parser.add_argument("--method", choices=("auto", "hierarchical", "exhaustive"), default="exhaustive")
    parser.add_argument("--ranking", choices=("max_volume", "fixed_mean"), default="max_volume")
    parser.add_argument("--training-budget", type=int, default=100)
    parser.add_argument("--validation-budget", type=int, default=100)
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--output", default=str(ROOT / "docs" / "v4-acceptance.json"))
    args = parser.parse_args()
    settings = load_settings()
    store = Store(settings.data_dir)
    service = DataService(store)
    manifest = store.snapshot(args.symbol)
    if not manifest or manifest.get("metadata", {}).get("is_demo"):
        raise SystemExit("Acceptance requires an existing real-data snapshot")
    base = BacktestConfig(
        schema_version=3, symbol=args.symbol, start_date=args.start, end_date=args.end,
        frequency=args.frequency, initial_shares=1000, min_shares=0, max_shares=2000,
        buy_shares=100, sell_shares=100, grid_mode="geometric", grid_step=.015,
        fees=settings.fee_defaults.model_dump(),
    )
    ranges = {
        "grid_step": {"low": .01, "high": .02},
        "sell_shares": {"low": 100, "high": 100},
        "buy_shares": {"low": 100, "high": 100},
        "min_shares": {"low": 0, "high": 0},
        "max_shares": {"low": 2000, "high": 2000},
    }
    config = OptimizationConfig.model_validate({
        "base": base.model_dump(mode="python"), "protocol_version": 4,
        "selection_target": "fixed", "ranges": ranges, "bootstrap_reps": args.bootstrap_reps,
        "min_trades": 0, "max_drawdown": .95, "min_minute_ratio": 0,
        "strategy": {
            "search_method": args.method, "ranking_mode": args.ranking,
            "training_budget": args.training_budget, "validation_budget": args.validation_budget,
        },
    })
    started = time.perf_counter()
    bundle = prepare_research_bundle(service, base, manifest["version"])
    prepared_seconds = time.perf_counter() - started
    preview = estimate_search(config, bundle)
    research_started = time.perf_counter()
    result = optimize_scientific(config, bundle)
    research_seconds = time.perf_counter() - research_started
    fixed = result.get("fixed", {})
    diagnostics = fixed.get("diagnostics", {})
    payload = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": result.get("protocol_version"),
        "symbol": args.symbol,
        "frequency": args.frequency,
        "data_version": bundle.version,
        "evaluation": {"start": args.start, "end": args.end},
        "preview": preview,
        "result": {
            "status": result.get("status"),
            "fixed_status": fixed.get("status"),
            "recommendation_status": fixed.get("recommendation_status"),
            "recommendation_count": len(fixed.get("recommendation_regions", [])),
            "training_unique_count": diagnostics.get("training_unique_count"),
            "validation_unique_count": diagnostics.get("validation_unique_count"),
            "peak_platform_count": diagnostics.get("peak_platform_count"),
            "global_exhaustive": diagnostics.get("global_exhaustive"),
            "coverage_status": diagnostics.get("coverage_status"),
            "validation_plan_hash": diagnostics.get("validation_plan_hash"),
            "regions": [
                {
                    "region_id": item.get("region_id"), "radii": item.get("radii"),
                    "volume": item.get("normalized_volume"), "points": item.get("combination_count"),
                    "mean_excess_return": item.get("mean_excess_return"),
                    "min_excess_return": item.get("min_excess_return"),
                }
                for item in fixed.get("recommendation_regions", [])
            ],
        },
        "timing_seconds": {"data_preparation": prepared_seconds, "research": research_seconds},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
