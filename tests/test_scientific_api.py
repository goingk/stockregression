from __future__ import annotations

from datetime import date
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from stockregression import optimization, scientific
from stockregression.api import create_app, public_result
from stockregression.jobs import _research_protocol_hash, execute_job
from stockregression.models import BacktestConfig, OptimizationConfig
from stockregression.settings import ResearchDefaults
from stockregression.storage import Store, dumps


def test_public_v2_results_keep_ui_summaries_and_drop_raw_archives_and_cost_matrices(tmp_path):
    candidates = [
        {
            "candidate_id": index,
            "params": {"buy_step": 0.01 * index},
            "score": 0.1 * index,
            "cost_scores": {"original_cost": 0.1 * index},
            "folds": [{"start": "2024-01-02", "end": "2024-01-03", "days": 2}],
        }
        for index in range(5)
    ]
    archive = [{"candidate_id": index, "payload": [index] * 10} for index in range(50)]
    trial_archive = [{"trial": index, "status": "completed", "params": {"x": index}} for index in range(40)]
    cost_matrix = [[float(row * 100 + column) for column in range(100)] for row in range(50)]
    bootstrap = {
        "method": "paired_stationary_bootstrap_white_reality_check",
        "primary": {"p_value": 0.25, "completed_reps": 100, "candidate_count": 5},
        "sensitivity": {"20": {"p_value": 0.25, "completed_reps": 100}},
    }
    curve = [
        {"timestamp": "2024-01-02T10:00:00", "equity": 100.0},
        {"timestamp": "2024-01-02T15:00:00", "equity": 101.0},
    ]
    run = {"strategies": {"grid": {"equity": curve, "trades": [{"timestamp": "2024-01-02T15:00:00", "side": "buy"}]}}}
    result = {
        "protocol_version": 2,
        "config": {"protocol_version": 2, "top_candidates": 2},
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
                    "candidate_family_bootstrap": {"family_count": 5, "candidate_ids": list(range(5)), "bootstrap": bootstrap},
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
            "current_candidate_family_bootstrap": {"family_count": 5, "candidate_ids": list(range(5)), "bootstrap": bootstrap},
            "final_126_day_check": {"status": "completed", "bootstrap": bootstrap},
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
    assert len(compact["evidence"]["decisions"][0]["candidates"]) == 2
    assert compact["evidence"]["decisions"][0]["candidate_count"] == 50
    assert compact["fixed"]["current"]["diagnostics"]["candidate_count"] == 50
    assert compact["fixed"]["current"]["diagnostics"]["validation_evaluation_count"] == 100
    assert compact["fixed"]["current"]["diagnostics"]["trial_status_counts"] == {"completed": 4}
    assert "costs" not in compact["fixed"]["current"]["diagnostics"]
    assert "costs" not in compact["evidence"]["decisions"][0]["diagnostics"]
    assert compact["fixed"]["candidates"][0]["cost_scores"] == candidates[0]["cost_scores"]
    assert compact["evidence"]["final_126_day_check"]["bootstrap"] == bootstrap
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
    result_id = app.state.store.save_result("optimization", result, "000001.SZ", "v2 API payload")
    with TestClient(app) as client:
        default_response = client.get(f"/api/v1/results/{result_id}")
        full_response = client.get(f"/api/v1/results/{result_id}?full=true")
    assert default_response.status_code == full_response.status_code == 200
    assert "candidate_archive" not in default_response.json()["fixed"]
    assert "candidate_archive" in full_response.json()["fixed"]
    assert "costs" in full_response.json()["fixed"]["current"]["diagnostics"]
    assert len(full_response.json()["fixed"]["test_result"]["strategies"]["grid"]["equity"]) == 2


def test_public_result_keeps_legacy_candidate_data_unchanged():
    result = {
        "protocol_version": 1,
        "historical": {
            "candidate_archive": [{"trial": 1}],
            "best_result": {"strategies": {"grid": {"equity": [
                {"timestamp": "2024-01-02T10:00:00", "equity": 100.0},
                {"timestamp": "2024-01-02T15:00:00", "equity": 101.0},
            ]}}},
        },
    }

    compact = public_result(result)

    assert compact["historical"]["candidate_archive"] == [{"trial": 1}]
    assert len(compact["historical"]["best_result"]["strategies"]["grid"]["equity"]) == 1


def _seed_daily_history(store: Store, days: int = 640) -> pd.DatetimeIndex:
    calendar = pd.bdate_range("2020-01-02", periods=days)
    close = pd.Series(10 + pd.RangeIndex(days).to_numpy() * 0.002, dtype=float)
    daily = pd.DataFrame({
        "date": calendar,
        "open": close,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "preclose": close.shift(1).fillna(close),
        "volume": 100_000,
        "turnover": close * 100_000,
        "tradestatus": 1,
        "is_st": False,
    })
    store.publish(
        "000001.SZ",
        pd.DataFrame(),
        daily,
        pd.DataFrame(),
        pd.DataFrame({"date": calendar}),
        {"name": "测试股票", "listing_date": str(calendar[0].date()), "actions_verified": True, "is_demo": False},
        {},
    )
    return calendar


def test_v2_models_keep_legacy_defaults_and_serialize_only_v2_fields():
    legacy = OptimizationConfig(base={"symbol": "600000.SH", "start_date": "2024-01-02"})
    assert legacy.protocol_version == 1
    assert legacy.n_trials == 200
    assert "reserve_cash" not in BacktestConfig(symbol="600000.SH", start_date=date(2024, 1, 2)).model_dump(mode="json")

    with pytest.raises(ValidationError):
        OptimizationConfig(protocol_version=2, base={"symbol": "600000.SH", "start_date": "2024-01-02"})

    scientific = OptimizationConfig(
        protocol_version=2,
        base={"schema_version": 2, "symbol": "600000.SH", "start_date": "2024-01-02", "reserve_cash": 0, "selection_asof": "2023-12-29"},
    )
    assert scientific.n_trials == 1000
    serialized = scientific.base.model_dump(mode="json")
    assert serialized["reserve_cash"] == 0
    assert serialized["selection_asof"] == "2023-12-29"


def test_old_research_preferences_fill_new_defaults_and_keep_custom_ranges():
    old = ResearchDefaults.model_validate({
        "mode": "robust",
        "n_trials": 321,
        "ranges": ResearchDefaults().model_dump()["ranges"],
    })
    assert old.protocol_version == 3
    assert old.n_trials == 321
    assert old.mode == "robust"
    assert len(old.ranges) == 8
    assert old.selection_target == "both"
    assert old.rebalance_days == 63
    assert old.bootstrap_reps == 10_000


@pytest.mark.skip(reason="Legacy snapshot research seed was removed")
def test_scientific_result_daily_clipping_preserves_fields_and_exports_both_selections(tmp_path, monkeypatch):
    app = create_app(str(tmp_path), start_jobs=False)
    dates = _seed_daily_history(app.state.store)
    base = {
        "protocol_version": 2,
        "selection_target": "both",
        "base": {
            "schema_version": 2,
            "symbol": "000001.SZ",
            "start_date": str(dates[550].date()),
            "end_date": str(dates[-1].date()),
            "frequency": "1d",
        },
    }

    def fake_optimize(config, bundle, progress=None, cancelled=None):
        assert config.base.start_date == dates[550].date()
        assert bundle.calendar.min() == dates[45]
        assert bundle.calendar.max() == dates[-1]
        assert bundle.metadata["research_preparation"]["requested_lookback_trading_days"] == 504
        if progress:
            progress(1, "完成")
        equity = [
            {"timestamp": "2022-01-03 10:00:00", "equity": 100.0},
            {"timestamp": "2022-01-03 15:00:00", "equity": 101.0},
        ]
        return {
            "protocol_version": 2,
            "status": "completed",
            "config": config.model_dump(mode="json"),
            "fixed": {"status": "completed", "params": {"buy_shares": 100}, "as_of": "2022-01-03", "test_result": {"strategies": {"grid": {"equity": equity, "trades": [{"date": "2022-01-03"}]}}}},
            "adaptive": {"status": "completed", "schedule": [], "latest_params": {"buy_shares": 200}, "test_result": {"strategies": {"grid": {"equity": equity, "trades": [{"date": "2022-01-03"}]}}}},
            "evidence": {"fixed_decision": {"windows": [{"start": "2022-01-01", "end": "2022-01-03", "days": 3}]}, "notes": ["audit remains visible"]},
            "data_version": bundle.version,
            "quality": {"is_demo": False},
        }

    monkeypatch.setattr(optimization, "optimize", fake_optimize)
    with TestClient(app) as client:
        response = client.post("/api/v1/optimizations", json=base)
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["payload"]["version"] is not None
        assert job["payload"]["config"]["n_trials"] == 1000
        execute_job(str(tmp_path), job_id)

        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"] == "completed", job.get("error")
        result = client.get(f"/api/v1/jobs/{job_id}/result").json()
        assert result["fixed"]["params"]["buy_shares"] == 100
        assert result["adaptive"]["latest_params"]["buy_shares"] == 200
        assert result["evidence"]["notes"] == ["audit remains visible"]
        intervals = result["evidence"]["local_audit"]["validation_intervals"]
        assert {item["kind"] for item in intervals} == {"input_evaluation", "evidence.fixed_decision.windows"}
        # Two intraday rows on the same date are reduced to one daily point.
        assert len(result["fixed"]["test_result"]["strategies"]["grid"]["equity"]) == 1
        assert result["fixed"]["test_result"]["strategies"]["grid"]["equity_frequency"] == "1d"
        exported = client.get(f"/api/v1/results/{job['result_id']}/export?kind=equity")
        assert b"selection,strategy" in exported.content
        assert b"fixed,grid" in exported.content and b"adaptive,grid" in exported.content

        repeated_id = client.post("/api/v1/optimizations", json=base).json()["job_id"]
        execute_job(str(tmp_path), repeated_id)
        repeated_job = client.get(f"/api/v1/jobs/{repeated_id}").json()
        repeated_result = client.get(f"/api/v1/jobs/{repeated_id}/result").json()
        assert repeated_job["status"] == "completed"
        assert repeated_result["evidence"]["local_audit"]["overlapping_prior_studies"] == 1
        assert repeated_result["evidence"]["local_audit"]["exact_repeat_prior_studies"] == 1


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
    first = OptimizationConfig(protocol_version=2, base={
        "schema_version": 2, "symbol": "600000.SH", "start_date": "2024-01-02", "end_date": "2024-06-28",
    })
    shifted = first.model_copy(update={
        "base": first.base.model_copy(update={"start_date": date(2024, 2, 1), "end_date": date(2024, 8, 30)}),
    })
    reseeded = first.model_copy(update={"seed": first.seed + 1})
    assert _research_protocol_hash(first) == _research_protocol_hash(shifted)
    assert _research_protocol_hash(first) != _research_protocol_hash(reseeded)


def test_selection_rng_seed_is_independent_of_asof_date(monkeypatch):
    dates = pd.bdate_range("2022-01-03", periods=220)
    prices = pd.Series(10 + pd.RangeIndex(len(dates)).to_numpy() * 0.001, dtype=float)
    daily = pd.DataFrame({"date": dates, "open": prices, "high": prices, "low": prices, "close": prices, "volume": 1000, "turnover": prices * 1000})
    bars = pd.DataFrame({"timestamp": dates + pd.Timedelta(hours=15), "open": prices, "high": prices, "low": prices, "close": prices, "volume": 1000, "turnover": prices * 1000, "frequency": "1d", "tradable": True, "is_synthetic": False})
    bundle = scientific.DataBundle(symbol="600000.SH", bars=bars, daily=daily, calendar=dates, quality={"minute_ratio": 1.0})
    config = OptimizationConfig(protocol_version=2, n_trials=1, seed=39, base={
        "schema_version": 2,
        "symbol": "600000.SH",
        "start_date": str(dates[-1].date()),
        "frequency": "1d",
    })
    seeds = []

    def trial_params(_config, base, names, **kwargs):
        seeds.append(kwargs["seed"])
        params = {name: getattr(base, name) for name in names}
        params["initial_shares"] = base.initial_shares
        return [params], [{"status": "completed", "params": params}], False

    def candidate_family(sampled, _config, _base, _names):
        row = {
            "key": ("candidate",), "candidate_id": 1, "params": sampled[0], "raw_score": 0.1,
            "eligible": True, "constraint_reasons": [], "max_drawdown": 0.1,
            "trade_count": 20, "buy_count": 10, "sell_count": 10, "turnover": 0.2,
            "folds": [], "stress_folds": [], "cost_scores": {},
        }
        return [row], {row["key"]: [row["key"]]}

    def evaluate_family(family_rows, _config, _base, _data, _dates, _folds, **_kwargs):
        return family_rows, {"status": "completed", "fold_count": 4}, False

    monkeypatch.setattr(scientific, "_trial_params", trial_params)
    monkeypatch.setattr(scientific, "_candidate_family", candidate_family)
    monkeypatch.setattr(scientific, "_evaluate_family", evaluate_family)
    scientific._select_at(config, config.base, bundle, dates[-1], reserve_cash=None, seed_offset=1, progress=None, cancelled=None)
    scientific._select_at(config, config.base, bundle, dates[-2], reserve_cash=None, seed_offset=1, progress=None, cancelled=None)
    assert seeds == [40, 40]
