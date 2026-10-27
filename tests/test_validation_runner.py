from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import pytest

from stockregression.models import BacktestConfig, ResearchRequestV5
from stockregression.validation_models import QualificationAccount, ResearchValidationRequest
from stockregression.validation_runner import (
    _run_qualification_kernel,
    _qualification_summary,
    _source_business_complete,
    _visible_etf_prefix,
    materialize_full_validation_result,
    materialize_stored_validation_result,
    run_validation,
)
from stockregression.validation_scenarios import synthetic_bundle


def _metrics(value: float = .31) -> dict[str, float]:
    return {
        "rv_annualized": value,
        "path_redundancy_daily_mean": value,
        "path_redundancy_5d_mean": value,
        "variance_ratio_240m": value,
        "iovr": value,
        "zero_crossing_rate": value,
        "jump_ratio": value,
    }


def _request(bundle, *, contributions=True, extra_bundles=()) -> ResearchValidationRequest:
    bundles = [bundle, *extra_bundles]
    research = ResearchRequestV5(
        research_context={
            "symbol": bundle.symbol,
            "start_date": bundle.calendar[0].date(),
            "end_date": bundle.calendar[-1].date(),
            "frequency": "1d", "initial_shares": 1000,
            "fees": {"commission_rate": 0, "min_commission": 0, "slippage_bps": 0},
        },
        ranges={
            "buy_step": {"low": .005, "high": .05},
            "sell_step": {"low": .005, "high": .05},
            "buy_shares": {"low": 100, "high": 500},
            "sell_shares": {"low": 100, "high": 500},
        },
        strategy={
            "training_min_days": 300, "training_max_days": 500,
            "training_budget": 20, "validation_budget": 20,
            "reporting_budget": 2, "max_runtime_seconds": 3600,
        },
        selection_target="both", prior_policy="daily_state_v1",
        min_history_days=504, rebalance_days=30,
    )
    accounts = {
        item.symbol: {
            "total_budget": 30000, "max_shares": 2000,
            "contributions": ([{
                "available_at": item.calendar[5].strftime("%Y-%m-%dT11:00:00"),
                "amount": 1000,
            }] if contributions else []),
        }
        for item in bundles
    }
    return ResearchValidationRequest(
        research=research, symbols=[item.symbol for item in bundles], accounts=accounts,
        requested_data_versions={item.symbol: item.version for item in bundles},
        max_runtime_seconds=3600,
    )


def _plan(bundle, *, extra_bundles=()) -> dict:
    bundles = [bundle, *extra_bundles]
    start = bundle.calendar[0]
    end = min(item.calendar[-1] for item in bundles)
    return {
        "suite_version": "research_validation_v1", "plan_hash": "fixture-plan",
        "primary_arm": "same_price:daily_state_v1:adaptive", "status": "ready",
        "symbols": [{
            "symbol": item.symbol, "data_version": item.version, "status": "ready",
            "training_days": 250, "training_min_days": 300,
            "evaluation_start": item.calendar[0].date().isoformat(),
            "evaluation_end": end.date().isoformat(),
            "blocks": [
                {"start": item.calendar[0].date().isoformat(),
                 "end": item.calendar[min(30, len(item.calendar) - 1)].date().isoformat(),
                 "days": min(31, len(item.calendar))},
                {"start": item.calendar[min(31, len(item.calendar) - 1)].date().isoformat(),
                 "end": end.date().isoformat(),
                 "days": max(1, len(item.calendar) - 31)},
            ],
            "prepared_start": item.calendar[0].date().isoformat(), "initial_cash": 19000,
            "estimated_work": {
                "selection_runs": 4, "arms": 8,
                "selection_evaluations_per_run_upper_bound": 100,
            },
        } for item in bundles],
        "data_versions": {item.symbol: item.version for item in bundles},
        "universe": {"members": [{"symbol": item.symbol} for item in bundles],
                     "historical_universe_verified": False},
        "estimated_work": {"candidate_evaluations_upper_bound": 400, "total_evaluation_budget": 400},
    }


def _selection(bundle) -> dict:
    return {"buy_step": .03, "sell_step": .04, "buy_shares": 100, "sell_shares": 200}


def _raw_optimization(bundle, *, selected=True, partial_no_candidates=False):
    params = _selection(bundle) if selected else None
    schedule = []
    if selected:
        schedule = [
            {
                "effective_date": bundle.calendar[0].date().isoformat(),
                "decision_date": bundle.calendar[0].date().isoformat(),
                "params": params, "paused": False, "recommendation_status": "qualified",
            },
            {
                "effective_date": bundle.calendar[30].date().isoformat(),
                "decision_date": bundle.calendar[29].date().isoformat(),
                "params": params, "paused": False, "recommendation_status": "qualified",
            },
        ]
    else:
        schedule = [{
            "effective_date": bundle.calendar[0].date().isoformat(),
            "decision_date": (bundle.calendar[0] - pd.Timedelta(days=1)).date().isoformat(),
            "params": {"buy_step": .03, "sell_step": .03, "buy_shares": 100, "sell_shares": 100},
            "paused": True, "recommendation_status": "no_qualified",
        }]
    return {
        "status": "partial" if partial_no_candidates else "completed",
        "completion_status": "available" if partial_no_candidates else "complete",
        "data_version": bundle.version, "data_issue": None,
        "fixed": {
            "status": "no_qualified" if not selected else "completed",
            "params": params, "diagnostics": {"candidate_count": 4 if selected else 0,
                                               "score_complete": selected},
            "as_of": bundle.calendar[0].date().isoformat(),
            "current": {
                "status": "no_qualified" if not selected else "completed",
                "params": params, "diagnostics": {"score_complete": selected},
            },
        },
        "adaptive": {
            "status": "completed", "schedule": schedule,
            "latest_params": params,
        },
        "evidence": {
            "candidate_archives": {"fixed": [{"hidden": "inner-only-archive"}],
                                   "adaptive": [], "current": []},
            "trial_archives": {"fixed": [{"hidden": "trial"}], "adaptive": [], "current": []},
            "decisions": [{
                "effective_date": row["effective_date"], "decision_date": row["decision_date"],
                "status": "completed" if selected else "no_qualified",
                "selected_params": row["params"], "paused": row["paused"],
                "score": .2 if selected else None, "reasons": [], "diagnostics": {"score_complete": selected},
            } for row in schedule],
        },
        "computation": {
            "candidate_evaluations": 7, "budget_stopped": False,
            "budget_reason": None, "decision_count": len(schedule),
        },
        "warnings": [],
    }


def _fake_qualification(config, bundle, schedule, account=None, **kwargs):
    curve = []
    for index, stamp in enumerate(bundle.calendar):
        invested = 1000.0
        profit = .01 * index
        curve.append({
            "timestamp": (stamp + pd.Timedelta(hours=15)).isoformat(),
            "equity": invested + profit, "cash": 0.0, "position": 100.0,
            "total_invested": invested, "peak_principal": invested,
            "cumulative_profit": profit, "principal_curve": 1 + profit / invested,
            "principal_drawdown": 0.0,
        })
    metric = {"peak_principal_return": .01, "max_principal_drawdown": .02,
              "total_profit": 1.0, "peak_invested": 1000.0}
    return {
        "status": "completed",
        "strategies": {
            "grid": {"metrics": dict(metric), "equity": curve,
                     "trades": [], "rejections": [], "contributions": []},
            "scheduled_funding_hold": {"metrics": dict(metric), "equity": curve,
                                       "trades": [], "rejections": [], "contributions": []},
        },
        "funding": {"initial_cash": 1000.0},
        "parameter_schedule": schedule or [],
        "stress_kwargs": {key: value for key, value in kwargs.items() if value is not None},
    }


def test_runner_uses_inner_result_ids_and_keeps_archives_out_of_root(monkeypatch):
    import stockregression.validation_runner as runner
    import stockregression.etf_statistics as statistics

    bundle = synthetic_bundle("159611.SZ", "low_volatility", start="2024-01-02", days=130)
    peer = synthetic_bundle("512880.SH", "low_volatility", start="2024-01-02", days=130)
    request = _request(bundle, extra_bundles=(peer,))
    request = request.model_copy(update={
        "research": request.research.model_copy(update={
            "max_principal_drawdown": .17,
        }),
    })
    plan = _plan(bundle, extra_bundles=(peer,))
    calls, inner, checkpoints, audits, qualification_kwargs = [], [], [], [], []
    configs = []
    ledger_values = {}

    def fake_optimize(config, data, progress=None, cancelled=None, runtime=None):
        configs.append(deepcopy(config))
        calls.append(config.strategy.training_window_mode)
        return _raw_optimization(data)

    def fake_qualification(config, data, account, parameter_schedule=None,
                           passive_grid=False, cancelled=None, **kwargs):
        qualification_kwargs.append(kwargs)
        return _fake_qualification(config, data, parameter_schedule, **kwargs)

    def fake_statistics(data, as_of, window_days):
        return {
            "status": "completed", "metrics": _metrics(),
            "eligible": True, "filter_reasons": [], "sampling": {},
            "diagnostics": {},
        }

    def save_inner(raw, symbol, arm_group):
        inner.append((deepcopy(raw), symbol, arm_group))
        return f"inner-{len(inner)}"

    def save_ledger(unit, value):
        marker = f"test-ledger/{unit}.json.gz"
        ref = {"__validation_ledger__": marker, "sha256": "fixture"}
        ledger_values[marker] = deepcopy(value)
        return ref

    def load_ledger(ref):
        return deepcopy(ledger_values[ref["__validation_ledger__"]])

    monkeypatch.setattr(runner, "optimize_v5", fake_optimize)
    monkeypatch.setattr(runner, "_run_qualification_kernel", fake_qualification)
    monkeypatch.setattr(statistics, "compute_etf_statistics", fake_statistics)
    result = run_validation(
        request, plan, {bundle.symbol: bundle, peer.symbol: peer},
        checkpoint=None,
        save_checkpoint=lambda value: (checkpoints.append(deepcopy(value)) or
                                       {"relative_path": "fixture-checkpoint.json",
                                        "revision": value["revision"]}),
        audit_event=lambda unit, event, payload=None: audits.append((unit, event, payload)),
        save_inner_result=save_inner,
        save_unit_ledger=save_ledger, load_unit_ledger=load_ledger,
    )

    assert len(calls) == 8
    assert len(inner) == 8
    assert result["status"] == "completed", {
        "computation": result["computation"],
        "warnings": result["warnings"],
        "cells": [(cell["unit_id"], cell["status"]) for cell in result["cells"][-12:]],
        "failures": [(cell["unit_id"], cell.get("error")) for cell in result["cells"]
                     if cell["status"] == "failed"][:2],
    }
    assert result["completion_status"] == "complete"
    assert result["kind"] == "research_validation"
    assert result["result_schema_version"] == 1
    assert result["validation_protocol_version"] == 1
    assert "protocol_version" not in result
    assert result["selection_result_id"] == "inner-1"
    assert len(result["selection_result_ids"]) == 8
    assert isinstance(result["cells"], list)
    assert len({cell["unit_id"] for cell in result["cells"]}) == len(result["cells"])
    assert "inner-only-archive" not in json.dumps(result, ensure_ascii=False)
    assert checkpoints and isinstance(checkpoints[-1]["cells"], list)
    assert checkpoints[-1]["ledgers"]
    assert all("__validation_ledger__" in value for value in checkpoints[-1]["ledgers"].values())
    assert "\"equity\"" not in json.dumps(checkpoints[-1]["ledgers"])
    assert any(event == "cache_reused" and unit.endswith(":base")
               for unit, event, _ in audits)
    assert all(config.min_history_days == 250 for config in configs)
    assert all(config.strategy.training_min_days == 250 for config in configs)
    assert all(config.strategy.training_max_days == 250 for config in configs)
    assert {config.prior_policy for config in configs} == {"off", "daily_state_v1"}
    assert {config.strategy.training_window_mode for config in configs} == {"same_price", "fixed"}
    # All six stress cases exist per scheme.  The base case references the
    # qualification ledger instead of rerunning the kernel.
    stress_cells = [cell for cell in result["cells"] if cell["kind"] == "stress"]
    assert len(stress_cells) == 16 * 6
    combined = [cell for cell in stress_cells
                if (cell.get("metadata") or {}).get("stress_case") == "combined"][0]
    assert combined["metadata"]["stress_kwargs"] == {
        "cost_multiplier": 2.0, "volume_multiplier": .5,
        "extra_delay_bars": 1, "omitted_contribution_index": 0,
    }
    synthetic_cells = [cell for cell in result["cells"] if cell["kind"] == "synthetic"]
    assert synthetic_cells and all(
        cell["metadata"]["selected_params"] == _selection(bundle)
        for cell in synthetic_cells
    )
    assert len([cell for cell in result["cells"] if cell["kind"] == "synthetic"]) == 8
    assert result["summary"]["independent_accounts"]["accounts"][bundle.symbol][
        "peak_principal_return"
    ] == .01
    assert any(item.get("cost_multiplier") == 2.0 for item in qualification_kwargs)
    assert any(item.get("volume_multiplier") == .5 for item in qualification_kwargs)
    assert any(item.get("extra_delay_bars") == 1 for item in qualification_kwargs)
    assert any(item.get("omitted_contribution_index") == 0 for item in qualification_kwargs)
    assert qualification_kwargs and all(
        item.get("max_principal_drawdown") == .17
        for item in qualification_kwargs
    )
    assert result["candidate_status"] == "qualified"
    assert result["qualification_status"] == "qualified"
    assert result["evidence"]["status"] == "recorded"
    assert result["gss"]["status"] == "completed"
    assert result["gss"]["horizon_days"] == 30
    assert result["gss"]["dates"]
    first_gss = result["gss"]["dates"][0]
    assert first_gss["outcome_end"] < result["symbols"][0]["evaluation_end"]
    for screen in first_gss["screens"].values():
        assert screen["selected_outcomes"]
        assert all(row["status"] == "completed" for row in screen["selected_outcomes"])
        assert screen["distribution"]["paired_excess"]["count"] == screen["count"]
    assert result["computation"]["statistics_evaluations"] > 0
    assert result["computation"]["total_evaluations"] == (
        result["computation"]["candidate_evaluations"]
        + result["computation"]["ledger_evaluations"]
    )

    hydrated = materialize_full_validation_result(result, checkpoints[-1])
    assert hydrated["full_export"] is True
    assert hydrated["ledger_export"] == {
        "materialized_location": "cells[].ledger",
        "references_location": "ledger_refs",
        "duplicated_raw_ledgers": False,
    }
    assert "ledgers" not in hydrated
    assert set(hydrated["ledger_refs"]) == set(checkpoints[-1]["ledgers"])
    for unit, index in hydrated["ledger_refs"].items():
        assert "__validation_ledger__" not in index
        assert index["checkpoint_ref"] == unit
        assert index["relative_path"] == checkpoints[-1]["ledgers"][unit][
            "__validation_ledger__"
        ]
    exported = [cell for cell in hydrated["cells"] if cell.get("ledger")]
    assert exported and all("__validation_ledger__" in cell["ledger"] for cell in exported)
    for collection in (hydrated["stress"], hydrated["synthetic"]):
        for cell in collection:
            assert "ledger" not in cell
            assert "ledger_ref" not in cell
    for symbol in hydrated["symbols"]:
        for arm in symbol["arms"]:
            for scheme in arm["schemes"].values():
                qualification = scheme["qualification"]
                assert "ledger" not in qualification
                assert "ledger_ref" not in qualification
    resolved = materialize_full_validation_result(
        result, checkpoints[-1], resolve=True, load_unit_ledger=load_ledger,
    )
    assert "ledgers" not in resolved
    assert resolved["ledger_refs"] == hydrated["ledger_refs"]
    assert all(
        "ledger" not in cell
        for cell in [*resolved["stress"], *resolved["synthetic"]]
    )
    assert all(
        "ledger" not in scheme["qualification"]
        for symbol in resolved["symbols"]
        for arm in symbol["arms"]
        for scheme in arm["schemes"].values()
    )
    assert any(
        isinstance(cell.get("ledger"), dict)
        and isinstance(cell["ledger"].get("grid"), dict)
        and cell["ledger"]["grid"].get("equity")
        for cell in resolved["cells"]
    )

    # A checkpoint replay performs no search or qualification work and cannot
    # double-count candidate evaluations.
    first_count = result["computation"]["candidate_evaluations"]
    calls.clear()
    inner.clear()
    qualification_kwargs.clear()
    resumed = run_validation(
        request, plan, {bundle.symbol: bundle, peer.symbol: peer},
        checkpoint=checkpoints[-1],
        save_checkpoint=lambda value: None, audit_event=lambda *args: None,
        save_inner_result=save_inner,
        save_unit_ledger=save_ledger, load_unit_ledger=load_ledger,
    )
    assert not calls
    assert not inner
    assert not qualification_kwargs
    assert resumed["computation"]["candidate_evaluations"] == first_count
    assert resumed["computation"]["cache_hits"] > 0


def test_stored_export_prefers_the_frozen_checkpoint_reference(monkeypatch):
    import stockregression.validation_store as validation_store

    frozen_ref = {"__validation_ledger__": "frozen-ledger", "sha256": "frozen"}
    fallback_ref = {"__validation_ledger__": "fallback-ledger", "sha256": "fallback"}
    frozen_checkpoint = {
        "ledgers": {"qualification:unit": frozen_ref},
        "revision": 7,
    }
    fallback_checkpoint = {
        "ledgers": {"qualification:unit": fallback_ref},
        "revision": 8,
    }
    lookups = []

    class FakeValidationStore:
        def __init__(self, store):
            self.store = store

        def load_frozen_checkpoint(self, ref):
            lookups.append(("frozen", ref))
            return frozen_checkpoint

        def load_checkpoint(self, run_id):
            lookups.append(("latest", run_id))
            return fallback_checkpoint

        def load_ledger(self, ref):
            raise AssertionError("resolve=False must not read a ledger")

    monkeypatch.setattr(validation_store, "ValidationStore", FakeValidationStore)
    result = {
        "run_id": "run-snapshot",
        "checkpoint_snapshot": "validation-checkpoints/immutable/frozen.json",
        "cells": [{
            "unit_id": "qualification:unit", "kind": "qualification",
            "status": "completed", "checkpoint_ref": "qualification:unit",
        }],
    }

    exported = materialize_stored_validation_result(object(), result)

    assert lookups == [("frozen", "validation-checkpoints/immutable/frozen.json")]
    assert exported["cells"][0]["ledger"] == frozen_ref
    assert exported["ledger_refs"] == {
        "qualification:unit": {
            "checkpoint_ref": "qualification:unit",
            "storage": "immutable_ledger",
            "relative_path": "frozen-ledger",
            "sha256": "frozen",
        },
    }
    assert exported["ledgers_resolved"] is False


def test_full_export_never_repeats_a_legacy_inline_ledger_in_root():
    ledger = {"grid": {"equity": [{"timestamp": "2024-01-02T15:00:00", "equity": 10}]}}
    result = {
        "cells": [{
            "unit_id": "qualification:unit", "kind": "qualification",
            "status": "completed", "checkpoint_ref": "qualification:unit",
        }],
        "stress": [], "synthetic": [], "symbols": [],
    }

    exported = materialize_full_validation_result(
        result, {"ledgers": {"qualification:unit": ledger}},
    )

    assert exported["cells"][0]["ledger"] == ledger
    assert ledger not in list(exported["ledger_refs"].values())
    assert exported["ledger_refs"] == {
        "qualification:unit": {
            "checkpoint_ref": "qualification:unit",
            "storage": "checkpoint_inline",
            "materialized_location": "cells[].ledger",
        },
    }


def test_partial_no_qualified_is_business_complete_but_scoring_stop_is_not():
    bundle = synthetic_bundle("159611.SZ", "low_volatility", start="2024-01-02", days=63)
    no_candidate = _raw_optimization(bundle, selected=False, partial_no_candidates=True)
    assert _source_business_complete(no_candidate) == (True, None)
    stopped = deepcopy(no_candidate)
    stopped["computation"]["budget_stopped"] = True
    assert _source_business_complete(stopped) == (
        False, "source_budget_incomplete",
    )
    selected_partial = _raw_optimization(bundle, selected=True)
    selected_partial["status"] = "partial"
    selected_partial["completion_status"] = "available"
    assert _source_business_complete(selected_partial) == (False, "source_scoring_incomplete")
    for unavailable in ("insufficient_history", "insufficient_range", "unrecognised"):
        raw = deepcopy(no_candidate)
        raw["status"] = unavailable
        assert _source_business_complete(raw) == (False, f"source_status_{unavailable}")
    unknown_branch = deepcopy(no_candidate)
    unknown_branch["fixed"]["status"] = "future_status"
    assert _source_business_complete(unknown_branch) == (
        False, "source_branch_incomplete",
    )


def test_no_candidate_schemes_run_paused_ledgers_without_becoming_a_budget_stop(monkeypatch):
    import stockregression.validation_runner as runner
    import stockregression.etf_statistics as statistics

    bundle = synthetic_bundle("159611.SZ", "low_volatility", start="2024-01-02", days=63)
    request, plan = _request(bundle), _plan(bundle)
    schedules = []

    monkeypatch.setattr(
        runner, "optimize_v5",
        lambda *args, **kwargs: _raw_optimization(bundle, selected=False,
                                                  partial_no_candidates=True),
    )

    def fake_qualification(config, data, account, parameter_schedule=None,
                           passive_grid=False, cancelled=None, **kwargs):
        schedules.append(deepcopy(parameter_schedule))
        return _fake_qualification(config, data, parameter_schedule, **kwargs)

    monkeypatch.setattr(runner, "_run_qualification_kernel", fake_qualification)
    monkeypatch.setattr(statistics, "compute_etf_statistics", lambda *args: {
        "status": "completed", "metrics": _metrics(), "eligible": True,
        "filter_reasons": [], "sampling": {}, "diagnostics": {},
    })
    result = run_validation(
        request, plan, {bundle.symbol: bundle},
        save_inner_result=lambda raw, symbol, arm: f"no-candidate-{arm}",
    )
    assert result["status"] == "partial"
    assert result["completion_status"] == "incomplete"
    assert result["stop_reason"] is None
    assert result["computation"]["business_incomplete_sources"] == []
    assert result["candidate_status"] == "no_qualified"
    assert result["qualification_status"] == "no_qualified"
    assert set(result["gss"]["reasons"]) == {
        "fewer_than_two_visible_common_symbols",
        "fewer_than_two_common_scored_symbols",
    }
    assert schedules
    assert all(row.get("paused") is True for schedule in schedules
               for row in (schedule or []))
    assert not [cell for cell in result["cells"] if cell["kind"] == "synthetic"]


def _two_etf_run(monkeypatch, *, fake_optimization, fake_qualification,
                 statuses=None):
    import stockregression.validation_runner as runner
    import stockregression.etf_statistics as statistics

    first = synthetic_bundle("159611.SZ", "low_volatility",
                             start="2024-01-02", days=63)
    second = synthetic_bundle("512880.SH", "low_volatility",
                              start="2024-01-02", days=63)
    request = _request(first, extra_bundles=(second,))
    plan = _plan(first, extra_bundles=(second,))
    monkeypatch.setattr(runner, "optimize_v5", fake_optimization)
    monkeypatch.setattr(runner, "_run_qualification_kernel", fake_qualification)
    monkeypatch.setattr(statistics, "compute_etf_statistics", lambda *args: {
        "status": "completed", "metrics": _metrics(), "eligible": True,
        "filter_reasons": [], "sampling": {}, "diagnostics": {},
    })
    return run_validation(
        request, plan, {first.symbol: first, second.symbol: second},
    )


@pytest.mark.parametrize("engine_status", ["risk_breached", "execution_limited"])
def test_root_qualification_retains_engine_risk_and_execution_failure(monkeypatch, engine_status):
    raw_optimization = _raw_optimization

    def fake_optimization(config, data, **kwargs):
        return raw_optimization(data)

    def fake_qualification(config, data, account, parameter_schedule=None,
                           passive_grid=False, cancelled=None, **kwargs):
        raw = _fake_qualification(config, data, parameter_schedule, **kwargs)
        raw["qualification_status"] = engine_status
        raw["qualification_basis"] = {
            "priority": ["unevaluable", "risk_breached", "execution_limited", "qualified"],
            "risk_limit": .17,
            "risk_breached": engine_status == "risk_breached",
            "grid_insufficient_cash_rejections": int(engine_status == "execution_limited"),
        }
        return raw

    result = _two_etf_run(
        monkeypatch, fake_optimization=fake_optimization,
        fake_qualification=fake_qualification,
    )

    assert result["status"] == "completed"
    assert result["completion_status"] == "complete"
    assert result["candidate_status"] == "qualified"
    assert result["qualification_status"] == engine_status
    assert result["evidence"]["status"] == "recorded"
    assert not result["computation"]["qualification_problems"]
    assert {row["state"] for row in result["evidence"]["qualification_states"]} == {
        engine_status,
    }
    primary = [
        cell["result"] for cell in result["cells"]
        if cell["kind"] == "qualification" and cell["scheme"] == "adaptive"
    ]
    assert primary and all(
        cell["qualification_status"] == engine_status
        and cell["qualification_basis"]["risk_limit"] == .17
        for cell in primary
    )


def test_primary_adaptive_all_paused_overrides_fixed_selected_and_is_business_complete(monkeypatch):
    schedules = []

    def fake_optimization(config, data, **kwargs):
        raw = _raw_optimization(data, selected=False, partial_no_candidates=True)
        raw["status"] = "completed"
        raw["completion_status"] = "complete"
        raw["fixed"].update({
            "status": "completed", "params": _selection(data),
            "diagnostics": {"candidate_count": 4, "score_complete": True},
        })
        raw["fixed"]["current"].update({
            "status": "completed", "params": _selection(data),
        })
        return raw

    def fake_qualification(config, data, account, parameter_schedule=None,
                           passive_grid=False, cancelled=None, **kwargs):
        schedules.append(deepcopy(parameter_schedule))
        return _fake_qualification(config, data, parameter_schedule, **kwargs)

    result = _two_etf_run(
        monkeypatch, fake_optimization=fake_optimization,
        fake_qualification=fake_qualification,
    )

    assert result["status"] == "completed"
    assert result["completion_status"] == "complete"
    assert result["candidate_status"] == "no_qualified"
    assert result["qualification_status"] == "no_qualified"
    assert schedules and all(
        row.get("paused") is True
        for schedule in schedules for row in (schedule or [])
    )
    assert all(fixed is not None for fixed in (
        arm["schemes"]["fixed"]["selected"]
        for symbol in result["symbols"] for arm in symbol["arms"]
    ))


def test_primary_adaptive_not_applicable_is_distinct_from_no_candidate(monkeypatch):
    def fake_optimization(config, data, **kwargs):
        raw = _raw_optimization(data, selected=False, partial_no_candidates=True)
        raw.update({"status": "completed", "completion_status": "complete"})
        raw["adaptive"].update({"status": "not_requested", "schedule": []})
        raw["fixed"].update({"status": "completed", "params": None})
        raw["fixed"]["current"].update({"status": "completed", "params": None})
        return raw

    result = _two_etf_run(
        monkeypatch, fake_optimization=fake_optimization,
        fake_qualification=lambda *args, **kwargs: _fake_qualification(*args, **kwargs),
    )

    assert result["status"] == "completed"
    assert result["completion_status"] == "complete"
    assert result["candidate_status"] == "not_applicable"
    assert result["qualification_status"] == "not_applicable"
    assert not result["computation"]["business_incomplete_sources"]
    assert not [cell for cell in result["cells"] if cell["kind"] == "synthetic"]


def test_gss_is_not_applicable_for_a_stock_only_request(monkeypatch):
    import stockregression.validation_runner as runner
    import stockregression.etf_statistics as statistics

    first = synthetic_bundle("159611.SZ", "low_volatility",
                             start="2024-01-02", days=63)
    first = deepcopy(first)
    first.metadata = {
        **(first.metadata or {}), "asset_type": "stock", "board": "main",
    }
    second = synthetic_bundle("600000.SH", "low_volatility",
                              start="2024-01-02", days=63)
    request = _request(first, extra_bundles=(second,))
    plan = _plan(first, extra_bundles=(second,))
    calls = []

    monkeypatch.setattr(
        runner, "optimize_v5",
        lambda config, data, **kwargs: _raw_optimization(data),
    )
    monkeypatch.setattr(
        runner, "_run_qualification_kernel",
        lambda *args, **kwargs: _fake_qualification(*args, **kwargs),
    )
    monkeypatch.setattr(
        statistics, "compute_etf_statistics",
        lambda *args: calls.append(args) or pytest.fail("股票不得进入 ETF 统计"),
    )

    result = run_validation(
        request, plan, {first.symbol: first, second.symbol: second},
    )

    assert not calls
    assert result["completion_status"] == "complete"
    assert result["gss"]["status"] == "not_applicable"
    assert result["gss"]["applicable"] is False
    assert result["gss"]["reasons"] == ["no_etf_members_in_frozen_request"]
    assert result["gss"]["etf_members"] == []
    assert set(result["gss"]["excluded_symbols"]) == {first.symbol, second.symbol}
    assert result["gss"]["dates"] == []


def test_mixed_request_retests_every_symbol_but_gss_uses_only_the_etf(monkeypatch):
    import stockregression.validation_runner as runner
    import stockregression.etf_statistics as statistics

    etf = synthetic_bundle("159611.SZ", "low_volatility",
                           start="2024-01-02", days=63)
    stock = synthetic_bundle("600000.SH", "low_volatility",
                             start="2024-01-02", days=63)
    request = _request(etf, extra_bundles=(stock,))
    plan = _plan(etf, extra_bundles=(stock,))
    stat_symbols = []

    monkeypatch.setattr(
        runner, "optimize_v5",
        lambda config, data, **kwargs: _raw_optimization(data),
    )
    monkeypatch.setattr(
        runner, "_run_qualification_kernel",
        lambda *args, **kwargs: _fake_qualification(*args, **kwargs),
    )

    def fake_statistics(bundle, *args):
        stat_symbols.append(bundle.symbol)
        return {
            "status": "completed", "metrics": _metrics(), "eligible": True,
            "filter_reasons": [], "sampling": {}, "diagnostics": {},
        }

    monkeypatch.setattr(statistics, "compute_etf_statistics", fake_statistics)
    result = run_validation(
        request, plan, {etf.symbol: etf, stock.symbol: stock},
    )

    qualification_symbols = {
        cell["symbol"] for cell in result["cells"]
        if cell["kind"] == "qualification"
    }
    assert qualification_symbols == {etf.symbol, stock.symbol}
    assert set(stat_symbols) == {etf.symbol}
    assert result["gss"]["etf_members"] == [etf.symbol]
    assert result["gss"]["excluded_symbols"] == [stock.symbol]
    assert result["gss"]["status"] == "incomplete"
    assert "fewer_than_two_visible_common_symbols" in result["gss"]["reasons"]
    assert result["completion_status"] == "incomplete"


def test_candidate_budget_stop_marks_incomplete_without_running_more_search(monkeypatch):
    import stockregression.validation_runner as runner

    bundle = synthetic_bundle("159611.SZ", "low_volatility", start="2024-01-02", days=63)
    request, plan = _request(bundle), _plan(bundle)
    plan["estimated_work"]["total_evaluation_budget"] = 7
    plan["symbols"][0]["estimated_work"][
        "selection_evaluations_per_run_upper_bound"
    ] = 7
    calls = []

    def fake_optimize(*args, **kwargs):
        calls.append(1)
        return _raw_optimization(bundle)

    monkeypatch.setattr(runner, "optimize_v5", fake_optimize)
    monkeypatch.setattr(runner, "_run_qualification_kernel",
                        lambda *args, **kwargs: _fake_qualification(*args, **kwargs))
    result = run_validation(request, plan, {bundle.symbol: bundle})
    assert len(calls) == 1
    assert result["status"] == "partial"
    assert result["completion_status"] == "incomplete"
    assert result["stop_reason"] == "evaluation_budget"
    assert result["computation"]["candidate_evaluations"] == 7
    assert result["computation"]["ledger_evaluations"] == 0
    assert result["computation"]["total_evaluation_budget"] == 7
    assert result["computation"]["total_evaluation_remaining"] == 0
    assert result["completion_reason"] == "parent_evaluation_budget"
    assert result["computation"]["budget_incomplete_scope"] == "parent"
    assert result["computation"]["parent_budget_exhausted"] is True
    assert result["computation"]["source_budget_incomplete"] is False


def test_source_budget_incomplete_is_not_reported_as_parent_budget(monkeypatch):
    import stockregression.validation_runner as runner

    bundle = synthetic_bundle("159611.SZ", "low_volatility",
                              start="2024-01-02", days=63)
    request, plan = _request(bundle), _plan(bundle)
    cancelled = {"value": False}

    def fake_optimize(config, data, **kwargs):
        raw = _raw_optimization(data)
        raw["status"] = "partial"
        raw["completion_status"] = "available"
        raw["computation"]["candidate_evaluations"] = 4
        raw["computation"]["budget_stopped"] = True
        cancelled["value"] = True
        return raw

    monkeypatch.setattr(runner, "optimize_v5", fake_optimize)
    monkeypatch.setattr(
        runner, "_run_qualification_kernel",
        lambda *args, **kwargs: _fake_qualification(*args, **kwargs),
    )
    result = run_validation(
        request, plan, {bundle.symbol: bundle},
        cancelled=lambda: cancelled["value"],
    )

    assert result["stop_reason"] == "cancelled"
    assert result["completion_reason"] == "source_budget_incomplete"
    assert result["computation"]["budget_incomplete_scope"] == "source"
    assert result["computation"]["source_budget_incomplete"] is True
    assert result["computation"]["parent_budget_exhausted"] is False
    assert result["computation"]["requires_new_research"] is True
    source = result["computation"]["business_incomplete_sources"][0]
    assert source["reason"] == "source_budget_incomplete"
    assert source["requires_new_research"] is True
    assert any("提高父任务总预算不能改变该内层上限" in warning
               for warning in result["warnings"])


def test_cancelled_incomplete_optimization_retries_with_cumulative_attempts(monkeypatch):
    import stockregression.validation_runner as runner
    import stockregression.etf_statistics as statistics

    bundle = synthetic_bundle("159611.SZ", "low_volatility",
                              start="2024-01-02", days=63)
    request, plan = _request(bundle), _plan(bundle)
    cancelled = {"value": False}
    calls = []
    checkpoints = []
    audits = []
    inner_ids = []
    first_attempt = {"value": True}

    def fake_optimize(config, data, **kwargs):
        calls.append(config.strategy.training_window_mode)
        raw = _raw_optimization(data)
        if first_attempt["value"]:
            first_attempt["value"] = False
            raw["status"] = "cancelled"
            raw["completion_status"] = "available"
            raw["computation"]["candidate_evaluations"] = 4
            cancelled["value"] = True
        return raw

    def save_inner(raw, symbol, arm_group):
        inner_ids.append(f"attempt-{len(inner_ids) + 1}")
        return inner_ids[-1]

    def save_checkpoint(value):
        checkpoints.append(deepcopy(value))
        return {"revision": value["revision"]}

    monkeypatch.setattr(runner, "optimize_v5", fake_optimize)
    monkeypatch.setattr(
        runner, "_run_qualification_kernel",
        lambda *args, **kwargs: _fake_qualification(*args, **kwargs),
    )
    monkeypatch.setattr(statistics, "compute_etf_statistics", lambda *args: {
        "status": "completed", "metrics": _metrics(), "eligible": True,
        "filter_reasons": [], "sampling": {}, "diagnostics": {},
    })
    first = run_validation(
        request, plan, {bundle.symbol: bundle},
        cancelled=lambda: cancelled["value"],
        save_checkpoint=save_checkpoint,
        audit_event=lambda unit, event, payload=None: audits.append(
            (unit, event, payload)
        ),
        save_inner_result=save_inner,
    )

    assert first["status"] == "cancelled"
    assert first["computation"]["candidate_evaluations"] == 4
    checkpoint = deepcopy(checkpoints[-1])
    cached_unit = "qualification:159611.SZ:fixed:off:fixed"
    checkpoint["cells"].append({
        "unit_id": cached_unit, "kind": "qualification",
        "symbol": bundle.symbol, "arm_group": "fixed:off", "scheme": "fixed",
        "status": "completed", "checkpoint_ref": cached_unit,
        "result": {"qualification_status": "qualified", "strategy_metrics": _metrics()},
    })

    cancelled["value"] = False
    calls.clear()
    resumed = run_validation(
        request, plan, {bundle.symbol: bundle}, checkpoint=checkpoint,
        save_checkpoint=lambda value: {"revision": value["revision"]},
        audit_event=lambda unit, event, payload=None: audits.append(
            (unit, event, payload)
        ),
        save_inner_result=save_inner,
    )

    optimization_unit = "optimization:159611.SZ:same_price:daily_state_v1"
    retried = next(cell for cell in resumed["cells"]
                   if cell["unit_id"] == optimization_unit)
    assert retried["attempt"] == 2
    assert retried["retry_history"][0]["source_status"] == "cancelled"
    assert retried["retry_history"][0]["candidate_evaluations"] == 4
    assert retried["result"]["business_complete"] is True
    assert "attempt-1" in resumed["selection_result_ids"]
    assert "attempt-2" in resumed["selection_result_ids"]
    assert resumed["computation"]["candidate_evaluations"] == 32
    assert resumed["computation"]["cache_hits"] >= 1
    assert sum(unit == optimization_unit and event == "started"
               for unit, event, _ in audits) == 2
    assert any(unit == cached_unit and event == "cache_reused"
               for unit, event, _ in audits)
    assert inner_ids == ["attempt-1", "attempt-2", "attempt-3", "attempt-4", "attempt-5"]


def test_real_qualification_kernel_integration_produces_independent_ledgers():
    bundle = synthetic_bundle("159611.SZ", "low_volatility", start="2024-01-02", days=8)
    config = BacktestConfig(
        schema_version=3, symbol=bundle.symbol,
        start_date=bundle.calendar[0].date(), end_date=bundle.calendar[-1].date(),
        frequency="1d", initial_shares=1000, min_shares=0, max_shares=None,
        grid_mode="geometric", buy_step=.03, sell_step=.03,
        buy_shares=100, sell_shares=100,
        fees={"commission_rate": 0, "min_commission": 0, "slippage_bps": 0},
    )
    account = QualificationAccount(total_budget=20000, max_shares=2000)
    raw = _run_qualification_kernel(config, bundle, account, None)
    assert raw["status"] == "completed"
    assert set(raw["strategies"]) == {"grid", "scheduled_funding_hold"}
    assert len(raw["strategies"]["grid"]["equity"]) == 8
    assert len(raw["strategies"]["scheduled_funding_hold"]["equity"]) == 8
    assert raw["execution_profile"]["independent_strategy_capacity"] is True


def test_qualification_summary_uses_the_engine_initial_equity():
    bundle = synthetic_bundle("159611.SZ", "low_volatility",
                              start="2024-01-02", days=3)
    raw = _fake_qualification(
        None, bundle, [], max_principal_drawdown=.2,
    )
    raw["strategies"]["grid"]["metrics"]["initial_equity"] = 1000
    raw["strategies"]["grid"]["equity"][0]["equity"] = 995

    summary = _qualification_summary(
        raw, [{"start": bundle.calendar[0].date().isoformat(),
               "end": bundle.calendar[0].date().isoformat()}],
        checkpoint_ref="qualification:fixture",
    )

    segment = summary["segment_returns"]["segments"][0]
    assert segment["initial_source"] == "explicit_initial_equity"
    assert segment["starting_equity"] == 1000
    assert segment["principal_return"] == pytest.approx(-.005)
    assert segment["max_principal_drawdown"] == pytest.approx(.005)


def test_cancelled_run_keeps_incomplete_root_and_existing_cells(monkeypatch):
    import stockregression.validation_runner as runner

    bundle = synthetic_bundle("159611.SZ", "low_volatility", start="2024-01-02", days=63)
    request, plan = _request(bundle), _plan(bundle)
    monkeypatch.setattr(runner, "optimize_v5", lambda *args, **kwargs: _raw_optimization(bundle))
    monkeypatch.setattr(runner, "_run_qualification_kernel",
                        lambda *args, **kwargs: _fake_qualification(*args, **kwargs))
    checkpoint = {
        "schema_version": 1, "suite_version": plan["suite_version"],
        "plan_hash": plan["plan_hash"],
        "cells": [{"unit_id": "optimization:159611.SZ:same_price:daily_state_v1",
                   "kind": "optimization", "status": "completed",
                   "result": {"computation": {"candidate_evaluations": 7}}}],
        "candidate_evaluations": 7, "ledger_evaluations": 0,
    }
    result = run_validation(request, plan, {bundle.symbol: bundle}, checkpoint=checkpoint,
                            cancelled=lambda: True)
    assert result["status"] == "cancelled"
    assert result["completion_status"] == "incomplete"
    assert result["stop_reason"] == "cancelled"
    assert result["computation"]["candidate_evaluations"] == 7

