from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.storage import Store


@pytest.mark.parametrize("count", [1, 4, 100])
def test_j_ranking_uses_configured_count_even_when_every_candidate_is_ineligible(tmp_path, count):
    rows = [{
        "candidate_id": index,
        "params": {"buy_step": .001 * index, "sell_step": .02, "buy_shares": 100, "sell_shares": 200, "initial_shares": 1000, "min_shares": 0},
        "evaluation_status": "completed", "training_excess_return": index / 100,
        "training_drawdown": .8, "trade_count": 0, "buy_count": 0, "sell_count": 0,
        "actual_min_pos_ratio": 0.0, "actual_max_pos_ratio": 4.0,
        "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
        "j_score": -index / 100, "neighborhood_mean": -.01,
        "downside_risk_tau": .02, "pos_penalty": .03, "score_complete": True,
        "neighborhood_size": 3, "neighborhood_support_size": 3, "score_missing_count": 0,
    } for index in range(1, 7)]
    # An incomplete higher score must never outrank complete observations.
    rows.append({**deepcopy(rows[0]), "candidate_id": 99, "score_complete": False, "j_score": 99})
    selection = {"as_of": "2026-09-30", "recommendation_regions": [], "diagnostics": {"coverage_status": "budgeted_local_peaks"}}
    result = {
        "protocol_version": 4,
        "config": {"protocol_version": 4, "top_candidates": 1, "max_drawdown": .2, "min_trades": 10,
                   "strategy": {"strategy_version": 4, "recommendation_count": count}},
        "fixed": {"current": selection},
        "evidence": {"search_candidate_archives": {"current": rows}},
    }
    result_id = Store(tmp_path).save_result("optimization", result, "600000.SH", "robust ranking")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        response = client.get(f"/api/v1/results/{result_id}/landscape?sample=training&limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["points"]) == 1
    assert len(payload["j_ranking"]) == min(count, 6)
    assert payload["recommendation_count"] == count
    assert payload["j_candidate_count"] == 6
    assert payload["focus"]["candidate_id"] == 6
    assert payload["focus"]["score_complete"] is True
    assert payload["j_focus"]["candidate_id"] == 1
    assert payload["decision_as_of"] == "2026-09-30"
    assert "min_shares" not in payload["parameters"]
    assert all(not point["constraint_eligible"] for point in payload["j_ranking"])
    assert all(point["trade_count"] == 0 and point["status"] == "completed" for point in payload["j_ranking"])
    assert "J 未大于 0" in payload["j_focus"]["constraint_reasons"]


def test_j_ranking_ties_use_neighborhood_risk_penalty_return_and_coordinate_order():
    from stockregression.api import _robust_landscape_ranking

    def point(index, mean=.1, risk=.01, penalty=.01, excess=.1):
        return {"candidate_id": index, "params": {"buy_step": index / 1000, "sell_step": .02, "buy_shares": 100, "sell_shares": 200},
                "j_score": .1, "score_complete": True, "neighborhood_mean": mean,
                "downside_risk_tau": risk, "pos_penalty": penalty, "training_excess_return": excess}

    rows = [point(6), point(5), point(4, excess=.2), point(3, penalty=0), point(2, risk=0), point(1, mean=.2)]
    payload = _robust_landscape_ranking(rows, {"strategy": {"recommendation_count": 6}}, {})
    assert [point["candidate_id"] for point in payload["j_ranking"]] == [1, 2, 3, 4, 5, 6]


def test_j_ranking_and_load_date_follow_the_selected_research_decision(tmp_path):
    def archive(index):
        return [{"candidate_id": index, "params": {"buy_step": .001 * index, "sell_step": .02, "buy_shares": 100, "sell_shares": 200},
                 "j_score": .1, "score_complete": True, "training_excess_return": .1}]

    result = {"protocol_version": 4, "config": {"strategy": {"strategy_version": 4, "recommendation_count": 5}},
              "fixed": {"as_of": "2026-01-02", "current": {"as_of": "2026-09-30"}},
              "evidence": {"decisions": [{"decision_date": "2026-06-30"}],
                           "search_candidate_archives": {"current": archive(1), "fixed": archive(2),
                                                         "adaptive": [{"search_candidate_archive": archive(3)}]}}}
    result_id = Store(tmp_path).save_result("optimization", result, "600000.SH", "decisions")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        for decision, index, as_of in (("current", 1, "2026-09-30"), ("fixed", 2, "2026-01-02"), ("adaptive", 3, "2026-06-30")):
            payload = client.get(f"/api/v1/results/{result_id}/landscape?sample=training&decision={decision}").json()
            assert payload["j_focus"]["candidate_id"] == index
            assert payload["j_ranking"][0]["candidate_id"] == index
            assert payload["decision_as_of"] == as_of


def test_old_results_complete_focus_from_locked_data_once_without_rewriting_history(tmp_path, monkeypatch):
    import pandas as pd
    from stockregression import api
    from stockregression.models import BacktestConfig, DataBundle, OptimizationConfig, ResearchStrategySpec
    from stockregression.research import robust
    from stockregression.research_v4.space import ParameterSpace

    config = OptimizationConfig(
        base=BacktestConfig(schema_version=3, symbol="600000.SH", start_date="2024-01-02", frequency="1d", initial_shares=1000),
        protocol_version=4,
        strategy=ResearchStrategySpec(strategy_version=4, recommendation_count=2, granularity={"buy_step": .001}),
        ranges={"buy_step": {"low": .001, "high": .005}, "sell_step": {"low": .01, "high": .01},
                "buy_shares": {"low": 100, "high": 100}, "sell_shares": {"low": 100, "high": 100}},
    )
    space = ParameterSpace(config)
    calls = []

    def raw(coordinate):
        return {"params": space.params(coordinate), "key": list(coordinate), "evaluation_status": "completed",
                "training_excess_return": coordinate[0] / 100, "training_drawdown": .01,
                    "trade_count": 20, "buy_count": 10, "sell_count": 10, "actual_max_pos_ratio": 1.0,
                    "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0}

    class Evaluator:
        chunk_size = 100

        def __init__(self, *args, **kwargs):
            pass

        def evaluate_training(self, coordinates):
            calls.extend(coordinates)
            return [raw(coordinate) for coordinate in coordinates]

    archive = [{**raw(space.full_coordinate((3,))), "candidate_id": 1}]
    dates = pd.bdate_range("2024-01-02", periods=3)
    bundle = DataBundle(symbol="600000.SH", bars=pd.DataFrame(), daily=pd.DataFrame(), calendar=dates)
    loaded_versions = []

    def load_bundle(self, base, version):
        loaded_versions.append(version)
        return bundle

    monkeypatch.setattr(robust, "BatchEvaluator", Evaluator)
    monkeypatch.setattr(api, "preparation_config", lambda store, base, version, lookback: base)
    monkeypatch.setattr(api.DataService, "load_bundle", load_bundle)
    selection = {"as_of": "2024-01-04", "diagnostics": {"training_start": "2024-01-02", "training_end": "2024-01-04"}}
    result = {"protocol_version": 4, "data_version": "locked-test-version", "config": config.model_dump(mode="json"),
              "fixed": {"current": selection}, "evidence": {"search_candidate_archives": {"current": archive}}}
    store = Store(tmp_path)
    result_id = store.save_result("optimization", result, "600000.SH", "old result")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        url = f"/api/v1/results/{result_id}/landscape?sample=training"
        first = client.get(url)
        assert first.status_code == 200
        payload = first.json()
        assert payload["focus"]["params"]["buy_step"] == .005
        assert payload["focus"]["score_complete"] is True
        assert payload["focus"]["neighborhood_size"] == 2
        assert payload["focus"]["score_missing_count"] == 0
        assert payload["focus_completion_error"] is None
        calls_after_first = len(calls)
        assert client.get(url).json()["focus"] == payload["focus"]
        assert len(calls) == calls_after_first
    assert loaded_versions == ["locked-test-version"]
    assert store.result(result_id)["evidence"]["search_candidate_archives"]["current"] == archive
