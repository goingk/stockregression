"""Offline comparison of same-price and fixed training windows.

The two studies share the data snapshot, evaluation dates, seed, parameter
space and every evaluation budget.  They are intentionally not appended to an
ordinary research job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stockregression.data import DataService
from stockregression.models import BacktestConfig, OptimizationConfig, ResearchStrategySpec
from stockregression.research_preparation import prepare_research_bundle
from stockregression.research_v5 import optimize_v5
from stockregression.settings import load_settings, resolve_project_path
from stockregression.storage import Store


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="evaluation start date")
    parser.add_argument("--end", required=True, help="evaluation end date")
    parser.add_argument("--version", help="locked market data version; default active")
    parser.add_argument("--initial-shares", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-budget", type=int)
    parser.add_argument("--validation-budget", type=int)
    parser.add_argument("--reporting-budget", type=int)
    parser.add_argument("--total-evaluation-budget", type=int)
    parser.add_argument("--max-runtime-seconds", type=int, default=3_600)
    parser.add_argument("--seed", type=int)
    return parser


def _summary(result: dict) -> dict:
    return {
        "status": result.get("status"),
        "fixed_status": (result.get("fixed") or {}).get("status"),
        "fixed": (result.get("fixed") or {}).get("oos_metrics"),
        "current": ((result.get("evidence") or {}).get("current_decision") or {}),
        "training_window": (
            ((result.get("fixed") or {}).get("diagnostics") or {}).get("training_window")
        ),
        "protocol": result.get("protocol"),
        "computation": result.get("computation"),
    }


def main() -> int:
    arguments = _parser().parse_args()
    settings = load_settings()
    defaults = settings.workbench_defaults
    store = Store(str(resolve_project_path(settings.data_dir)), settings=settings)
    grid = defaults.grid
    position = defaults.position
    base = BacktestConfig(
        schema_version=3,
        symbol=arguments.symbol,
        start_date=arguments.start,
        end_date=arguments.end,
        frequency="1d",
        initial_shares=arguments.initial_shares,
        min_shares=0,
        max_shares=None,
        buy_shares=position.buy_shares,
        sell_shares=position.sell_shares,
        grid_mode="geometric",
        buy_step=grid.buy_step,
        sell_step=grid.sell_step,
        lower_price=grid.lower_price,
        upper_price=grid.upper_price,
        fees=settings.fee_defaults.model_dump(mode="python"),
        dca=defaults.dca.model_dump(mode="python"),
    )
    research = defaults.research.model_dump(mode="python")
    for name in ("training_budget", "validation_budget", "reporting_budget", "total_evaluation_budget", "max_runtime_seconds", "seed"):
        value = getattr(arguments, name)
        if value is not None:
            research[name] = value
    strategy_fields = set(ResearchStrategySpec.model_fields)
    strategy_values = {
        name: value for name, value in research.items() if name in strategy_fields
    }
    common = OptimizationConfig(
        base=base,
        selection_target="fixed",
        ranges={
            name: value.model_dump(mode="python")
            for name, value in defaults.research.ranges.items()
        },
        strategy=ResearchStrategySpec.model_validate(strategy_values),
        seed=research["seed"],
        top_candidates=defaults.research.top_candidates,
        max_principal_drawdown=defaults.research.max_principal_drawdown,
        min_trades=defaults.research.min_trades,
        min_history_days=defaults.research.min_history_days,
        min_minute_ratio=defaults.research.min_minute_ratio,
    )
    service = DataService(store, settings=settings)
    bundle = prepare_research_bundle(
        service, base, arguments.version,
        lookback_days=int(research["training_max_days"]),
    )
    version = bundle.version
    report = {
        "symbol": base.symbol,
        "data_version": version,
        "evaluation_start": arguments.start,
        "evaluation_end": arguments.end,
        "seed": int(research["seed"]),
        "budgets": {
            "training": int(research["training_budget"]),
            "neighborhood": int(research["validation_budget"]),
            "reporting": int(research["reporting_budget"]),
            "total": research.get("total_evaluation_budget"),
        },
        "windows": {},
    }
    for mode in ("same_price", "fixed"):
        config = common.model_copy(update={
            "strategy": common.strategy.model_copy(update={"training_window_mode": mode})
        })
        report["windows"][mode] = _summary(optimize_v5(config, bundle))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
