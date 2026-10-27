from copy import deepcopy

import pandas as pd
import pytest
from pydantic import ValidationError

from stockregression.validation_models import ResearchValidationRequest
from stockregression.validation_planner import history_plan, plan_hash, load_validation_bundles


def request(**updates):
    value = {
        "research": {"research_context": {"symbol": "159611.SZ", "start_date": "2024-01-01"},
                     "ranges": {"buy_step": {"low": .01, "high": .03, "step": .01},
                                "sell_step": {"low": .01, "high": .03, "step": .01},
                                "buy_shares": {"low": 100, "high": 300, "step": 100},
                                "sell_shares": {"low": 100, "high": 300, "step": 100}}},
        "accounts": {"159611.SZ": {"total_budget": 100000, "max_shares": 2000}},
    }
    value.update(updates)
    return ResearchValidationRequest.model_validate(value)


@pytest.mark.parametrize("n,t,o,b", [(1005, 500, 504, 4), (601, 300, 300, 4), (401, 250, 150, 2), (314, 250, 63, 1)])
def test_history_uses_available_dates_and_balances_complete_blocks(n, t, o, b):
    dates = pd.bdate_range("2020-01-01", periods=n)
    result = history_plan(dates, dates, request(), symbol="159611.SZ")
    assert result["status"] == "ready"
    assert result["training_days"] == t
    assert result["evaluation_days"] == o
    assert len(result["blocks"]) == b
    assert sum(block["days"] for block in result["blocks"]) == o
    assert result["blocks"][0]["start"] == result["evaluation_start"]
    assert result["blocks"][-1]["end"] == result["evaluation_end"]
    assert max(x["days"] for x in result["blocks"]) - min(x["days"] for x in result["blocks"]) <= 1


def test_missing_day_cannot_be_deleted_and_histories_stitched():
    dates = pd.bdate_range("2020-01-01", periods=700)
    certified = dates.delete(500)
    result = history_plan(dates, certified, request(), symbol="159611.SZ")
    assert result["status"] == "data_issue"
    assert result["available_days"] == 198


def test_auto_tail_lag_is_visible_and_cutoff_is_respected():
    dates = pd.bdate_range("2020-01-01", periods=601)
    result = history_plan(dates, dates[:-3], request(), symbol="159611.SZ")
    assert result["evaluation_end"] == str(dates[-4].date())
    assert len(result["excluded_tail_days"]) == 3
    cut = request()
    cut.research.research_context.end_date = dates[480].date()
    result = history_plan(dates, dates, cut, symbol="159611.SZ")
    assert result["evaluation_end"] == str(dates[480].date())


def test_explicit_dates_not_clipped_or_replaced():
    dates = pd.bdate_range("2020-01-01", periods=700)
    value = request()
    value.history_policy = "explicit"
    value.research.research_context.start_date = dates[501].date()
    value.research.research_context.end_date = dates[-1].date()
    result = history_plan(dates, dates, value, symbol="159611.SZ")
    assert result["status"] == "ready"
    assert result["evaluation_start"] == str(dates[501].date())
    result = history_plan(dates, dates.delete(600), value, symbol="159611.SZ")
    assert result["status"] == "data_issue"
    assert result["gaps"][0]["date"] == str(dates[600].date())


def test_hash_pins_exact_versions_and_all_account_settings():
    value = request()
    plan = {"symbols": [], "data_versions": {"159611.SZ": "frozen-a"}}
    original = plan_hash(value, plan)
    same = value.model_copy(update={"plan_hash": original, "requested_data_versions": {"159611.SZ": "frozen-a"}})
    assert plan_hash(same, {**plan, "plan_hash": original}) == original
    changed = deepcopy(plan)
    changed["data_versions"]["159611.SZ"] = "frozen-b"
    assert plan_hash(value, changed) != original
    value.accounts["159611.SZ"].participation_rate = .02
    assert plan_hash(value, plan) != original


def test_load_rejects_modified_plan_before_reading_any_data():
    value = request()
    with pytest.raises(ValueError, match="哈希"):
        load_validation_bundles(None, value, {"symbols": [], "plan_hash": "tampered"})


def test_all_accounts_are_explicit_finite_and_contributions_convert_timezone():
    with pytest.raises(ValidationError):
        request(symbols=["159611.SZ", "512880.SH"])
    with pytest.raises(ValidationError):
        request(accounts={"159611.SZ": {"total_budget": float("nan"), "max_shares": 2000}})
    value = request(accounts={"159611.SZ": {"total_budget": 100000, "max_shares": 2000,
        "contributions": [{"available_at": "2026-01-01T02:00:00Z", "amount": 1000}]}})
    assert value.accounts["159611.SZ"].contributions[0].available_at.hour == 10


def test_preview_blocks_inner_search_budget_before_job_execution(monkeypatch):
    from types import SimpleNamespace
    from stockregression import validation_planner
    from stockregression.research import robust
    dates = pd.bdate_range("2020-01-01", periods=601)
    daily = pd.DataFrame({"timestamp": dates, "close": 10., "tradestatus": 1, "volume": 100000})
    manifest = {"symbol": "159611.SZ", "version": "locked", "market_data": True,
                "market_versions": {"1day": {"version": 1, "rows": 601}, "1min": {"version": 1, "rows": 144240}},
                "auxiliary_versions": {"calendar": "calendar-v1"}}
    store = SimpleNamespace(snapshot=lambda *a, **kw: manifest, instrument=lambda symbol: {"symbol": symbol, "asset_type": "etf"})
    repository = SimpleNamespace(bars=lambda *a, **kw: daily, trading_days=lambda *a, **kw: dates)
    market = SimpleNamespace(repository=repository, readiness=lambda *a, **kw: {"ready": True},
                             load_backtest_bundle=lambda *a, **kw: object())
    monkeypatch.setattr(validation_planner, "_verified_days", lambda market, manifest, calendar, timeframe: set(calendar))
    captured = []
    def estimate(config, bundle):
        captured.append(config)
        return {"status": "budget_exceeded", "reasons": ["完整枚举训练预算不足"],
                "task_evaluation_upper_bound": 100, "estimated_task_memory_mb": 2}
    monkeypatch.setattr(robust, "estimate_robust_search", estimate)
    value = request()
    value.research.min_history_days = 504
    result = validation_planner.plan_validation(store, value, market=market)
    assert result["status"] == "budget_exceeded"
    assert result["symbols"][0]["gaps"][0]["message"] == "完整枚举训练预算不足"
    assert captured[0].min_history_days == result["symbols"][0]["training_days"]
    assert result["estimated_work"]["total_evaluations_upper_bound"] > result["estimated_work"]["candidate_evaluations_upper_bound"]
    assert result["estimated_work"]["estimated_memory_mb"] > 2
