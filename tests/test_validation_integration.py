"""Exercise actual v5 search and the shared qualification ledger end to end.

All prices in this module are declared synthetic test fixtures, not market
evidence.  No optimizer or execution function is replaced by a test double.
"""
from pathlib import Path

from stockregression.models import ResearchRequestV5
from stockregression.research.robust import estimate_robust_search
from stockregression.storage import Store
from stockregression.validation_models import ResearchValidationRequest
from stockregression.validation_planner import history_plan, plan_hash
from stockregression.validation_runner import run_validation, materialize_stored_validation_result
from stockregression.validation_scenarios import synthetic_bundle
from stockregression.validation_store import ValidationStore


def test_real_search_and_execution_complete_frozen_stock_suite(tmp_path):
    bundle = synthetic_bundle("600000.SH", "low_volatility", start="2024-01-02", days=104)
    bundle.metadata.update(asset_type="stock", board="main", settlement="T1", price_tick=.01)
    research = ResearchRequestV5(research_context={"symbol": bundle.symbol,
        "start_date": bundle.calendar[41].date(), "end_date": bundle.calendar[-1].date(), "frequency": "1d"},
        ranges={"buy_step": {"low": .005, "high": .015, "step": .005},
                "sell_step": {"low": .005, "high": .015, "step": .005},
                "buy_shares": {"low": 100, "high": 300, "step": 100},
                "sell_shares": {"low": 100, "high": 300, "step": 100}},
        strategy={"training_min_days": 20, "training_max_days": 40,
                  "granularity": {"buy_step": .005, "sell_step": .005, "buy_shares": 100, "sell_shares": 100},
                  "search_method": "exhaustive", "training_budget": 81,
                  "validation_budget": 81, "reporting_budget": 10, "max_runtime_seconds": 180},
        min_history_days=20, rebalance_days=21)
    request = ResearchValidationRequest(research=research, accounts={bundle.symbol: {
        "total_budget": 50000, "max_shares": 3500}}, total_evaluation_budget=20000, max_runtime_seconds=180)
    item = history_plan(bundle.calendar, bundle.calendar, request, symbol=bundle.symbol)
    assert item["status"] == "ready"
    item["data_version"] = bundle.version
    item["initial_cash"] = 40000
    base = research.to_optimization()
    estimate = estimate_robust_search(base.model_copy(update={"selection_target": "both"}), bundle)
    item["estimated_work"] = {"selection_evaluations_per_run_upper_bound": estimate["task_evaluation_upper_bound"]}
    plan = {"status": "ready", "suite_version": request.suite_version,
            "primary_arm": "same_price:off:adaptive", "symbols": [item],
            "data_versions": {bundle.symbol: bundle.version},
            "universe": {"members": [{"symbol": bundle.symbol, "asset_type": "stock"}],
                         "historical_universe_verified": False},
            "estimated_work": {"total_evaluation_budget": 20000}}
    plan["plan_hash"] = plan_hash(request, plan)
    store = Store(tmp_path / "data")
    audit = ValidationStore(store)
    audit.register_plan("numeric-integration", request.model_dump(mode="json"), plan, {"fixture": True})
    result = run_validation(request, plan, {bundle.symbol: bundle},
        save_checkpoint=lambda value: audit.save_checkpoint("numeric-integration", value),
        audit_event=lambda unit, event, value=None: audit.append_event("numeric-integration", unit, event, value),
        save_unit_ledger=lambda unit, value: audit.save_ledger("numeric-integration", unit, value),
        load_unit_ledger=audit.load_ledger,
        save_inner_result=lambda value, symbol, arm: store.save_result("optimization", value, symbol, arm))
    assert result["completion_status"] == "complete", {"computation": result.get("computation"),
        "warnings": result.get("warnings"), "failed": [(c["unit_id"], c.get("error")) for c in result["cells"] if c["status"] == "failed"]}
    assert result["gss"]["status"] == "not_applicable"
    assert len(result["selection_result_ids"]) == 4
    assert sum(len(arm["schemes"]) for arm in result["symbols"][0]["arms"]) == 8
    assert result["computation"]["candidate_evaluations"] > 0
    assert result["summary"]["primary_arm_changed"] is False
    result.update(run_id="numeric-integration", checkpoint_snapshot=audit.freeze_checkpoint("numeric-integration"))
    materialized = materialize_stored_validation_result(store, result, resolve=True)
    qualified = [cell for cell in materialized["cells"] if cell["kind"] == "qualification"]
    assert len(qualified) == 8
    assert all("grid" in cell["ledger"] and "scheduled_funding_hold" in cell["ledger"] for cell in qualified)
    assert all("__validation_ledger__" in ref for ref in audit.load_checkpoint("numeric-integration")["ledgers"].values())
