"""Frozen local-data acceptance for the complete validation feature.

The explicit account amounts below are simulation inputs, never trade orders.
No network download or market-data modification is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from stockregression.reproducibility import computation_provenance
from stockregression.settings import PROJECT_ROOT, load_settings
from stockregression.storage import Store
from stockregression.validation_models import ResearchValidationRequest
from stockregression.validation_planner import plan_validation, load_validation_bundles
from stockregression.validation_runner import run_validation
from stockregression.validation_store import ValidationStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["159611.SZ", "159995.SZ", "512880.SH"])
    parser.add_argument("--budget", type=float, required=True, help="每个独立模拟账户的总预算")
    parser.add_argument("--max-shares", type=int, required=True)
    parser.add_argument("--run", action="store_true", help="运行全部冻结单元；默认只预检")
    parser.add_argument("--output", default=".local/research-validation-acceptance.json")
    args = parser.parse_args()
    settings = load_settings()
    store = Store(settings.data_dir, settings=settings)
    request = ResearchValidationRequest.model_validate({
        "research": {"research_context": {"symbol": args.symbols[0], "start_date": "2015-01-01", "frequency": "1m"},
            "ranges": {"buy_step": {"low": .02, "high": .06, "step": .02},
                       "sell_step": {"low": .02, "high": .06, "step": .02},
                       "buy_shares": {"low": 100, "high": 300, "step": 100},
                       "sell_shares": {"low": 100, "high": 300, "step": 100}},
            "strategy": {"search_method": "exhaustive", "training_budget": 81,
                         "granularity": {"buy_step": .02, "sell_step": .02, "buy_shares": 100, "sell_shares": 100},
                         "validation_budget": 81, "reporting_budget": 20}},
        "symbols": args.symbols,
        "accounts": {symbol: {"total_budget": args.budget, "max_shares": args.max_shares} for symbol in args.symbols},
        "family_id": "complete-validation-acceptance-v1",
    })
    plan = plan_validation(store, request, settings=settings)
    output = {"acceptance_version": 1, "source": "existing local frozen market data",
              "synthetic_market_replacement": False, "request": request.model_dump(mode="json"), "plan": plan}
    if args.run and plan["status"] == "ready":
        request = request.model_copy(update={"plan_hash": plan["plan_hash"], "requested_data_versions": plan["data_versions"]})
        run_id = "acceptance-" + uuid.uuid4().hex
        audit = ValidationStore(store)
        audit.register_plan(run_id, request.model_dump(mode="json"), plan,
                            computation_provenance(PROJECT_ROOT, configuration=request.model_dump(mode="json")))
        bundles = load_validation_bundles(store, request, plan, settings=settings)
        def progress(fraction, message):
            print(json.dumps({"progress": round(fraction, 3), "message": message}, ensure_ascii=False), flush=True)
        result = run_validation(request, plan, bundles, progress=progress, runtime=settings.performance,
            save_checkpoint=lambda value: audit.save_checkpoint(run_id, value),
            audit_event=lambda unit, event, value=None: audit.append_event(run_id, unit, event, value),
            save_unit_ledger=lambda unit, value: audit.save_ledger(run_id, unit, value),
            load_unit_ledger=audit.load_ledger,
            save_inner_result=lambda inner, symbol, group: store.save_result("optimization", inner, symbol, "完整验证验收内部研究 · " + group))
        result.update(run_id=run_id, audit=audit.audit_summary(run_id), plan=plan)
        if hasattr(audit, "freeze_checkpoint"):
            result["checkpoint_snapshot"] = audit.freeze_checkpoint(run_id)
        result_id = store.save_result("research_validation", result, ",".join(args.symbols), "本地真实数据完整验证验收")
        output.update(run_id=run_id, result_id=result_id, status=result["status"],
                      completion_status=result["completion_status"], computation=result["computation"])
    elif args.run:
        output.update(status="data_issue", completion_status="incomplete",
                      reason="真实数据或预算未满足冻结预检；未替换合成行情、未缩减必需项目")
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"report": str(target), "plan_status": plan["status"],
                      "status": output.get("status", "preflight_only"),
                      "symbols": [{"symbol": row["symbol"], "status": row["status"],
                                   "gaps": row.get("gaps", [])[:3]} for row in plan["symbols"]]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
