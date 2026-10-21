"""v5 研究协议下的结果发布、配置迁移与研究登记契约。

旧协议的公开结果压缩测试已经从 v2 形状迁移到 protocol_version=5 /
result_schema_version=2；旧的 v1/v2 模型默认值、旧版研究偏好迁移与 v2
科学研究的 API 兼容用例随协议一并移除。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from stockregression.api import create_app, public_result
from stockregression.jobs import _research_protocol_hash
from stockregression.models import BacktestConfig, OptimizationConfig, ResearchStrategySpec
from stockregression.settings import ResearchDefaults
from stockregression.storage import Store, dumps


def test_public_v5_results_keep_ui_summaries_and_drop_raw_archives_and_cost_matrices(tmp_path):
    candidates = [
        {
            "candidate_id": index,
            "params": {"buy_step": 0.01 * index},
            "score": 0.1 * index,
            "cost_scores": {"original_cost": 0.1 * index},
            "training_excess_return": 0.01 * index,
            "max_principal_drawdown": 0.05,
        }
        for index in range(5)
    ]
    archive = [{"candidate_id": index, "payload": [index] * 10} for index in range(50)]
    trial_archive = [{"trial": index, "status": "completed", "params": {"x": index}} for index in range(40)]
    cost_matrix = [[float(row * 100 + column) for column in range(100)] for row in range(50)]
    curve = [
        {"timestamp": "2024-01-02T10:00:00", "equity": 100.0},
        {"timestamp": "2024-01-02T15:00:00", "equity": 101.0},
    ]
    run = {"strategies": {"grid": {"equity": curve, "trades": [{"timestamp": "2024-01-02T15:00:00", "side": "buy"}]}}}
    result = {
        "protocol_version": 5,
        "result_schema_version": 2,
        "config": {
            "protocol_version": 5,
            "top_candidates": 2,
            "strategy": {"strategy_version": 5, "ranking_mode": "robust_j"},
        },
        "computation": {"candidate_count": 150, "cost_evaluations": 300, "decision_count": 3},
        "fixed": {
            "status": "completed",
            "params": {"buy_step": 0.02},
            "candidates": candidates,
            "candidate_archive": archive,
            "trial_archive": trial_archive,
            "current": {
                "params": {"buy_step": 0.03},
                "candidates": candidates,
                "candidate_archive": archive,
                "trial_archive": trial_archive,
                "diagnostics": {
                    "candidate_count": 50,
                    "validation_evaluation_count": 100,
                    "trial_status_counts": {"completed": 4},
                    "costs": {"original_cost": {"fold_utility": cost_matrix}},
                    "candidate_ids": list(range(5)),
                },
            },
            "test_result": run,
        },
        "adaptive": {
            "status": "completed",
            "schedule": [{"decision_date": "2024-01-02", "effective_date": "2024-01-03", "params": {"buy_step": 0.03}}],
            "latest_params": {"buy_step": 0.03},
            "test_result": run,
        },
        "evidence": {
            "decisions": [{
                "decision_date": "2024-01-02",
                "candidate_count": 50,
                "evaluated_count": 50,
                "candidates": candidates,
                "diagnostics": {"candidate_count": 50, "costs": {"original_cost": {"fold_utility": cost_matrix}}},
            }],
            "candidate_archives": {"fixed": archive, "adaptive": archive},
            "trial_archives": {"fixed": trial_archive, "adaptive": trial_archive},
            "search_candidate_archives": {"current": archive},
            "current_decision": {"decision": "current"},
        },
    }

    compact = public_result(result)
    assert len(compact["fixed"]["candidates"]) == 2
    assert len(compact["fixed"]["current"]["candidates"]) == 2
    assert "candidate_archive" not in compact["fixed"]
    assert "trial_archive" not in compact["fixed"]
    assert "candidate_archive" not in compact["fixed"]["current"]
    assert "trial_archive" not in compact["fixed"]["current"]
    assert "candidate_archives" not in compact["evidence"]
    assert "trial_archives" not in compact["evidence"]
    assert "search_candidate_archives" not in compact["evidence"]
    assert len(compact["evidence"]["decisions"][0]["candidates"]) == 2
    assert compact["evidence"]["decisions"][0]["candidate_count"] == 50
    assert compact["evidence"]["current_decision"] == {"decision": "current"}
    assert compact["fixed"]["current"]["diagnostics"]["candidate_count"] == 50
    assert compact["fixed"]["current"]["diagnostics"]["validation_evaluation_count"] == 100
    assert compact["fixed"]["current"]["diagnostics"]["trial_status_counts"] == {"completed": 4}
    assert "costs" not in compact["fixed"]["current"]["diagnostics"]
    assert "costs" not in compact["evidence"]["decisions"][0]["diagnostics"]
    assert "candidate_ids" not in compact["fixed"]["current"]["diagnostics"]
    assert compact["fixed"]["candidates"][0]["cost_scores"] == candidates[0]["cost_scores"]
    assert compact["adaptive"]["schedule"] == result["adaptive"]["schedule"]
    assert compact["computation"] == result["computation"]
    assert len(compact["fixed"]["test_result"]["strategies"]["grid"]["equity"]) == 1
    assert compact["fixed"]["test_result"]["strategies"]["grid"]["trades"] == run["strategies"]["grid"]["trades"]
    assert len(dumps(compact)) < len(dumps(result)) / 10

    # Full callers retain the raw audit archives, cost matrix and full-resolution curves.
    assert public_result(result, full=True) == result
    assert len(result["fixed"]["candidate_archive"]) == 50
    assert len(result["fixed"]["test_result"]["strategies"]["grid"]["equity"]) == 2

    app = create_app(str(tmp_path), start_jobs=False)
    result_id = app.state.store.save_result("optimization", result, "000001.SZ", "v5 API payload")
    with TestClient(app) as client:
        default_response = client.get(f"/api/v1/results/{result_id}")
        full_response = client.get(f"/api/v1/results/{result_id}?full=true")
    assert default_response.status_code == full_response.status_code == 200
    assert "candidate_archive" not in default_response.json()["fixed"]
    assert "candidate_archive" in full_response.json()["fixed"]
    assert "costs" in full_response.json()["fixed"]["current"]["diagnostics"]
    assert len(full_response.json()["fixed"]["test_result"]["strategies"]["grid"]["equity"]) == 2


def test_v5_models_reject_retired_protocol_fields():
    with pytest.raises(ValidationError):
        BacktestConfig(
            schema_version=3, symbol="600000.SH", start_date=date(2024, 1, 2), reserve_cash=0,
        )
    with pytest.raises(ValidationError):
        OptimizationConfig(
            base={"schema_version": 3, "symbol": "600000.SH", "start_date": "2024-01-02"},
            mode="historical",
        )
    with pytest.raises(ValidationError):
        OptimizationConfig(
            base={"schema_version": 3, "symbol": "600000.SH", "start_date": "2024-01-02"},
            n_trials=8,
        )
    with pytest.raises(ValidationError):
        OptimizationConfig(
            base={"schema_version": 3, "symbol": "600000.SH", "start_date": "2024-01-02"},
            protocol_version=4,
        )
    with pytest.raises(ValidationError):
        ResearchStrategySpec(strategy_version=4)

    serialized = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date=date(2024, 1, 2),
    ).model_dump(mode="json")
    assert "reserve_cash" not in serialized

    config = OptimizationConfig(
        base={"schema_version": 3, "symbol": "600000.SH", "start_date": "2024-01-02"},
    )
    assert config.protocol_version == 5
    assert config.strategy.strategy_version == 5
    assert config.strategy.ranking_mode == "robust_j"
    assert config.strategy.strategy_id == "geometric_robust_j"
    assert set(config.ranges) == {"buy_step", "sell_step", "buy_shares", "sell_shares"}
    assert all(value.step is None for value in config.ranges.values())
    assert config.protocol_description()["result_schema_version"] == 2


def test_legacy_research_preferences_migrate_to_v5_defaults():
    legacy = ResearchDefaults.model_validate({
        "protocol_version": 4,
        "mode": "robust",
        "n_trials": 321,
        "bootstrap_reps": 10_000,
        "max_drawdown": 0.5,
        "ranges": ResearchDefaults().model_dump()["ranges"],
    })

    assert legacy.protocol_version == 5
    assert legacy.strategy_version == 5
    assert legacy.ranking_mode == "robust_j"
    assert legacy.strategy_id == "geometric_robust_j"
    assert legacy.max_shares_mode == "unlimited"
    assert legacy.neighborhood_fields == ["buy_step", "sell_step", "buy_shares", "sell_shares"]
    assert set(legacy.ranges) == {"buy_step", "sell_step", "buy_shares", "sell_shares"}
    assert all(value.step is None for value in legacy.ranges.values())
    assert set(legacy.granularity) == {"buy_step", "sell_step", "buy_shares", "sell_shares"}
    assert set(legacy.fixed_half_widths) == {"buy_step", "sell_step", "buy_shares", "sell_shares"}
    assert legacy.selection_target == "both"
    assert legacy.rebalance_days == 63
    # 只在迁移整份旧对象时丢弃已退役的控制项。
    assert not hasattr(legacy, "mode")
    assert not hasattr(legacy, "n_trials")
    assert not hasattr(legacy, "bootstrap_reps")
    assert not hasattr(legacy, "max_drawdown")

    # 当前协议下提交已退役的控制项必须报错，而不是被静默丢弃。
    with pytest.raises(ValidationError):
        ResearchDefaults.model_validate({"protocol_version": 5, "mode": "robust"})


def test_research_registry_counts_overlap_even_after_seed_or_snapshot_changes(tmp_path):
    store = Store(tmp_path)
    window = [{"start": "2024-01-02", "end": "2024-03-29", "kind": "holdout"}]
    store.register_research("first", "protocol-a", "600000.SH", "2024-01-02", "2024-03-29", "snapshot-a", window)
    second = store.register_research("second", "protocol-b", "600000.SH", "2024-01-02", "2024-03-29", "snapshot-b", window)
    assert second["overlapping_prior_studies"] == 1
    assert second["exact_repeat_prior_studies"] == 1
    replay = store.register_research("second", "protocol-b", "600000.SH", "2024-01-02", "2024-03-29", "snapshot-b", window)
    assert replay["idempotent_replay"] is True
    assert replay["overlapping_prior_studies"] == 1
    assert replay["human_untouched_verified"] is False
    assert replay["recorded_overlap_status"] == "overlap_detected"


def test_research_registry_serializes_concurrent_overlap_checks(tmp_path):
    store = Store(tmp_path)
    window = [{"start": "2024-01-02", "end": "2024-03-29", "kind": "holdout"}]

    def register(identity):
        return store.register_research(identity, "protocol", "600000.SH", "2024-01-02", "2024-03-29", "snapshot", window)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(register, ("parallel-a", "parallel-b")))
    assert sorted(item["overlapping_prior_studies"] for item in outputs) == [0, 1]


def test_protocol_hash_excludes_evaluation_dates_but_tracks_seed():
    first = OptimizationConfig(protocol_version=5, base={
        "schema_version": 3, "symbol": "600000.SH", "start_date": "2024-01-02", "end_date": "2024-06-28",
    })
    shifted = first.model_copy(update={
        "base": first.base.model_copy(update={"start_date": date(2024, 2, 1), "end_date": date(2024, 8, 30)}),
    })
    reseeded = first.model_copy(update={"seed": first.seed + 1})
    assert _research_protocol_hash(first) == _research_protocol_hash(shifted)
    assert _research_protocol_hash(first) != _research_protocol_hash(reseeded)
