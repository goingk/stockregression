"""稳健 J 训练景观读取契约（protocol_version=5 / result_schema_version=2）。

契约要点：

* 只有 v5 结果可以读取参数分布，旧协议结果一律 410；
* J 排名只按完整分数排序，收益、风险、交易笔数与持仓约束只影响
  ``constraint_eligible``，不影响名次；
* 焦点是收益最高的候选，J 焦点是 J 分数最高的候选，两者互不替代；
* 决策（current / fixed / adaptive）决定读哪一份档案与哪一个决策日；
* 读取路径只读归档，不评价候选、不加载行情，补算必须走显式补算任务。
"""
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
        "max_principal_drawdown": .8, "trade_count": 0, "buy_count": 0, "sell_count": 0,
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
        "protocol_version": 5, "result_schema_version": 2,
        "config": {"protocol_version": 5, "top_candidates": 1, "max_principal_drawdown": .2, "min_trades": 10,
                   "strategy": {"strategy_version": 5, "recommendation_count": count}},
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
    payload = _robust_landscape_ranking(rows, {"strategy": {"strategy_version": 5, "recommendation_count": 6}}, {})
    assert [point["candidate_id"] for point in payload["j_ranking"]] == [1, 2, 3, 4, 5, 6]


def test_j_ranking_and_load_date_follow_the_selected_research_decision(tmp_path):
    def archive(index):
        return [{"candidate_id": index, "params": {"buy_step": .001 * index, "sell_step": .02, "buy_shares": 100, "sell_shares": 200},
                 "j_score": .1, "score_complete": True, "training_excess_return": .1}]

    result = {"protocol_version": 5, "result_schema_version": 2,
              "config": {"protocol_version": 5, "max_principal_drawdown": .2, "min_trades": 10,
                         "strategy": {"strategy_version": 5, "recommendation_count": 5}},
              "fixed": {"as_of": "2026-01-02", "current": {"as_of": "2026-09-30"}},
              "evidence": {"decisions": [{"decision_date": "2026-06-30"}],
                           "search_candidate_archives": {"current": archive(1), "fixed": archive(2),
                                                         "adaptive": [{"search_candidate_archive": archive(3)}]}}}
    result_id = Store(tmp_path).save_result("optimization", result, "600000.SH", "decisions")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        for decision, index, as_of in (("current", 1, "2026-09-30"), ("fixed", 2, "2026-01-02"),
                                       ("adaptive", 3, "2026-06-30")):
            payload = client.get(f"/api/v1/results/{result_id}/landscape?sample=training&decision={decision}").json()
            assert payload["j_focus"]["candidate_id"] == index
            assert payload["j_ranking"][0]["candidate_id"] == index
            assert payload["decision_as_of"] == as_of

        # 自适应分支按决策序号存放自己的档案，绝不能退回其它决策的档案。
        adaptive = client.get(f"/api/v1/results/{result_id}/landscape?sample=training&decision=adaptive")
        assert adaptive.status_code == 200
        adaptive_payload = adaptive.json()
        assert adaptive_payload["j_focus"]["candidate_id"] == 3
        assert adaptive_payload["decision_as_of"] == "2026-06-30"
        # 只读到自适应分支自己的候选，读不到 current / fixed 的档案。
        assert [point["candidate_id"] for point in adaptive_payload["points"]] == [3]
        assert [point["candidate_id"] for point in adaptive_payload["j_ranking"]] == [3]


def test_v5_landscape_read_never_evaluates_candidates_or_loads_market_data(tmp_path, monkeypatch):
    """读取参数分布只读归档；补算必须走显式补算任务。"""
    def forbidden(*args, **kwargs):
        raise AssertionError("参数分布读取不得评价候选或加载行情")

    monkeypatch.setattr("stockregression.data.DataService.load_bundle", forbidden)
    monkeypatch.setattr("stockregression.research_v4.evaluator.BatchEvaluator.__init__", forbidden)
    monkeypatch.setattr("stockregression.research_completion.complete_research_result", forbidden)

    rows = [{
        "candidate_id": index,
        "params": {"buy_step": .001 * index, "sell_step": .02, "buy_shares": 100, "sell_shares": 200},
        "evaluation_status": "completed", "training_excess_return": index / 100,
        "max_principal_drawdown": .05, "trade_count": 20, "buy_count": 10, "sell_count": 10,
        "actual_min_pos_ratio": 0.0, "actual_max_pos_ratio": 1.0,
        "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
        "j_score": .01 * index, "neighborhood_mean": .01, "downside_risk_tau": 0.0,
        "pos_penalty": 0.0, "score_complete": True, "score_missing_count": 0,
    } for index in range(1, 4)]
    result = {
        "protocol_version": 5, "result_schema_version": 2,
        "config": {"protocol_version": 5, "max_principal_drawdown": .2, "min_trades": 10,
                   "strategy": {"strategy_version": 5, "recommendation_count": 3}},
        "fixed": {"current": {"as_of": "2026-09-30", "recommendation_regions": [], "diagnostics": {}}},
        "evidence": {"search_candidate_archives": {"current": rows}},
    }
    result_id = Store(tmp_path).save_result("optimization", result, "600000.SH", "只读读取")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        response = client.get(f"/api/v1/results/{result_id}/landscape?sample=training")
    assert response.status_code == 200
    payload = response.json()
    assert payload["j_candidate_count"] == 3
    assert payload["completion"]["status"] == "complete"


def test_v5_region_export_uses_the_current_robust_j_membership(tmp_path):
    rows = [{
        "candidate_id": 1,
        "params": {"buy_step": .01, "sell_step": .02, "buy_shares": 100, "sell_shares": 200},
        "evaluation_status": "completed", "training_excess_return": .03,
        "max_principal_drawdown": .05, "trade_count": 20,
        "score_complete": True, "region_ids": [1],
    }]
    result = {
        "protocol_version": 5, "result_schema_version": 2,
        "fixed": {"current": {
            "as_of": "2026-09-30", "diagnostics": {},
            "recommendation_regions": [{"region_id": 1, "geometry": "robust_j"}],
        }},
        "evidence": {"candidate_archives": {"current": rows}},
    }
    result_id = Store(tmp_path).save_result("optimization", result, "600000.SH", "推荐域导出")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        response = client.get(
            f"/api/v1/results/{result_id}/regions/1/export?decision=current&format=json"
        )
    assert response.status_code == 200
    assert response.json()["region"]["geometry"] == "robust_j"
    assert [row["candidate_id"] for row in response.json()["members"]] == [1]


def test_landscape_slice_total_comes_from_the_archive_query(tmp_path):
    """二维切片的总数必须来自档案查询，平面被截断时必须标记为抽样。"""
    rows = [{
        "candidate_id": index,
        "params": {
            "buy_step": .0001 * index, "sell_step": .0002 * index,
            "buy_shares": 100, "sell_shares": 200,
        },
        "evaluation_status": "completed", "training_excess_return": index / 100000,
        "max_principal_drawdown": .05, "trade_count": 20,
        "buy_count": 5, "sell_count": 5,
        "actual_min_pos_ratio": 0.0, "actual_max_pos_ratio": 1.0,
        "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
        "j_score": .000001 * index, "neighborhood_mean": .01,
        "downside_risk_tau": 0.0, "pos_penalty": 0.0,
        "score_complete": True, "score_missing_count": 0,
    } for index in range(1, 2601)]
    result = {
        "protocol_version": 5, "result_schema_version": 2,
        "config": {"protocol_version": 5, "max_principal_drawdown": .2, "min_trades": 10,
                   "strategy": {"strategy_version": 5, "recommendation_count": 3}},
        "fixed": {"current": {
            "as_of": "2026-09-30", "recommendation_regions": [], "diagnostics": {},
        }},
        "evidence": {"search_candidate_archives": {"current": rows}},
    }
    result_id = Store(tmp_path).save_result("optimization", result, "600000.SH", "大切片")

    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        response = client.get(f"/api/v1/results/{result_id}/landscape?sample=training&limit=50")

    assert response.status_code == 200
    payload = response.json()
    plane = payload["slices"]["buy_step|sell_step"]
    assert plane["total"] == 2600, "切片总数必须是档案查询的行数，不能是被截断的行数"
    assert plane["sampled"] is True, "被截断的平面必须标记为抽样"
    assert len(plane["points"]) <= 2500
