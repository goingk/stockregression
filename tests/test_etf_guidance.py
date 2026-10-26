from copy import deepcopy
import json

import pytest

from stockregression.etf_guidance import parameter_guidance
from stockregression.etf_models import EtfResearchRequest
from stockregression.models import BacktestConfig, OptimizationConfig, ResearchStrategySpec
from stockregression.research.space import ParameterSpace


@pytest.fixture
def row():
    return {
        "status": "completed", "data_status": "ready", "version": "frozen-1",
        "window_end": "2026-09-30", "metrics": {"mean_raw_price": 1.25},
        "grid_basis": {"atr30_pct": .005, "price_tick": .001, "current_raw_price_unadjusted": 1.3},
        "diagnostics": {"window_end": "2026-09-30"},
        "market_state": {
            "as_of": "2026-09-30", "status": "ready",
            "metrics": {"natr14": .015, "amplitude_q75": .02},
            "metric_details": {"hv_ratio": {"sample_count": 60, "hv60": .01}},
            "samples": {"selected_start": "2025-10-01", "selected_end": "2026-09-30"},
        },
    }


def test_guidance_is_valid_lattice_and_roundtrips_as_json(row):
    request = EtfResearchRequest()
    before = deepcopy(row)
    result = parameter_guidance(row, request)
    assert result["status"] == "ready"
    assert row == before
    json.dumps(result, allow_nan=False)
    lower = result["ranges"]["buy_step"]["low"]
    assert lower >= max(1.2 * .005, 5 * .001 / 1.25, result["evidence"]["round_trip_cost"])
    assert result["tau"]["value"] == .016
    assert result["tau"]["sensitivity"] == [.008, .016, .024]
    config = OptimizationConfig(
        base=BacktestConfig(schema_version=3, symbol="510300.SH", start_date="2026-10-01"),
        strategy=ResearchStrategySpec(granularity=result["granularity"], tau=result["tau"]["value"]),
        ranges={**result["ranges"], "buy_shares": {"low": 100, "high": 1000}, "sell_shares": {"low": 100, "high": 1000}},
    )
    space = ParameterSpace(config)
    assert 3 <= space.dimensions["buy_step"].count <= 26
    assert space.dimensions["buy_step"].count == result["evidence"]["spacing_points"]


def test_costs_and_volatility_change_the_hints(row):
    request = EtfResearchRequest()
    normal = parameter_guidance(row, request)
    row["market_state"]["metric_details"]["hv_ratio"]["hv60"] = .025
    volatile = parameter_guidance(row, request)
    assert volatile["tau"]["value"] > normal["tau"]["value"]
    expensive = EtfResearchRequest(execution={"fees": {"min_commission": 2}})
    costlier = parameter_guidance(row, expensive)
    assert costlier["ranges"]["buy_step"]["low"] > normal["ranges"]["buy_step"]["low"]
    assert costlier["tau"]["value"] >= costlier["evidence"]["round_trip_cost"]


def test_unqualified_replay_never_drives_the_suggestion(row):
    request = EtfResearchRequest()
    expected = parameter_guidance(row, request)
    row["eligible"] = False
    row["validation"] = {"status": "no_qualified", "candidates": [{"metrics": {"excess_return": 999}}]}
    assert parameter_guidance(row, request) == expected


@pytest.mark.parametrize("problem", ["short", "future", "flat", "nan", "no_price", "expensive"])
def test_unusable_data_has_a_reason_not_fabricated_numbers(row, problem):
    request = EtfResearchRequest()
    if problem == "short":
        row["market_state"]["metric_details"]["hv_ratio"]["sample_count"] = 59
    elif problem == "future":
        row["market_state"]["as_of"] = "2026-10-01"
    elif problem == "flat":
        row["market_state"]["metric_details"]["hv_ratio"]["hv60"] = 0
    elif problem == "nan":
        row["market_state"]["metric_details"]["hv_ratio"]["hv60"] = float("nan")
    elif problem == "no_price":
        row["metrics"] = {}
        row["grid_basis"]["current_raw_price_unadjusted"] = None
    elif problem == "expensive":
        request = EtfResearchRequest(execution={"fees": {"min_commission": 50}})
    result = parameter_guidance(row, request)
    assert result["status"] == "unavailable"
    assert result["reason"]
    assert result["ranges"] == {} and result["tau"] is None
    json.dumps(result, allow_nan=False)


def test_daily_fallback_is_explicit(row):
    row["data_status"] = "limited"
    row["grid_basis"]["atr30_pct"] = None
    result = parameter_guidance(row, EtfResearchRequest())
    assert result["status"] == "limited"
    assert result["ranges"]["buy_step"]["low"] >= .5 * .015
    assert "日线波幅" in result["notes"][-1]
