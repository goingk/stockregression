from __future__ import annotations

import runpy
from datetime import date

import pandas as pd
from fastapi.testclient import TestClient

from stockregression import scientific
from stockregression.api import create_app
from stockregression.models import BacktestConfig, OptimizationConfig
from stockregression.share_engine import run_share_backtest
from stockregression.storage import Store


def market():
    fixture = runpy.run_path("tests/test_research_kernel.py")["market"]
    return fixture()


def v3_config(base: BacktestConfig, **updates) -> BacktestConfig:
    values = base.model_dump(mode="python")
    values.update(schema_version=3, max_shares=None, **updates)
    values.pop("reserve_cash", None)
    return BacktestConfig.model_validate(values)


def params_of(config: BacktestConfig) -> dict:
    names = scientific._v3_names(config)
    return {name: int(getattr(config, name)) if name.endswith("_shares") else float(getattr(config, name)) for name in names} | {
        "initial_shares": int(config.initial_shares),
        "max_shares": config.max_shares,
    }


def test_v3_top_up_accounting_uses_peak_net_contributions():
    config, bundle = market()
    config = v3_config(config)
    result = run_share_backtest(config, bundle)
    funding = result["funding"]
    grid = result["strategies"]["grid"]
    expected_peak = funding["initial_principal"] + funding["topups"]
    assert funding["funding_mode"] == "top_up_on_fill"
    assert funding["topups"] > 0
    assert funding["peak_invested"] == expected_peak
    assert grid["metrics"]["total_return"] == (grid["equity"][-1]["equity"] - expected_peak) / expected_peak
    assert grid["metrics"]["return_basis"] == "peak_invested_roi"
    assert grid["metrics"]["risk_return_basis"] == "cash_flow_adjusted_unit_nav"
    for point in grid["equity"]:
        assert point["normalized_equity"] == (point["equity"] - point["total_invested"]) / expected_peak + 1
    assert all(trade["external_topup"] == 0 for trade in grid["trades"] if trade["side"] == "sell")


def test_v3_paused_schedule_stops_trading_until_resume():
    config, bundle = market()
    config = v3_config(config)
    days = pd.DatetimeIndex(bundle.calendar)
    params = params_of(config)
    schedule = [
        {"effective_date": days[0].date().isoformat(), "decision_date": (days[0] - pd.offsets.BDay(1)).date().isoformat(), "params": params, "paused": True},
        {"effective_date": days[2].date().isoformat(), "decision_date": days[1].date().isoformat(), "params": params, "paused": False},
    ]
    result = run_share_backtest(config, bundle, parameter_schedule=schedule)
    trades = result["strategies"]["grid"]["trades"]
    assert all(pd.Timestamp(trade["timestamp"]).normalize() >= days[2] for trade in trades)
    assert result["parameter_schedule"][0]["paused"] is True


def test_v3_zero_initial_principal_funds_only_the_first_successful_buy():
    config, bundle = market()
    config = v3_config(config, initial_shares=0, min_shares=0, buy_shares=100)
    config = config.model_copy(update={"reference_price_mode": "manual", "reference_price": 20.0})
    result = run_share_backtest(config, bundle)
    funding = result["funding"]
    grid = result["strategies"]["grid"]
    assert funding["initial_principal"] == 0
    assert funding["topups"] > 0
    assert funding["peak_invested"] == funding["topups"]
    assert grid["metrics"]["total_return"] == (grid["equity"][-1]["equity"] - funding["topups"]) / funding["topups"]
    assert grid["equity"][0]["normalized_equity"] is not None


def test_v3_candidate_neighborhood_contains_pairwise_moves():
    config, _ = market()
    config = v3_config(config)
    optimization = OptimizationConfig(
        base=config,
        protocol_version=3,
        ranges={
            "buy_step": {"low": 0.02, "high": 0.06, "step": 0.01},
            "sell_step": {"low": 0.02, "high": 0.06, "step": 0.01},
            "min_shares": {"low": 0, "high": 0, "step": 100},
            "buy_shares": {"low": 100, "high": 100, "step": 100},
            "sell_shares": {"low": 100, "high": 100, "step": 100},
        },
    )
    center = params_of(config) | {"buy_step": 0.03, "sell_step": 0.03}
    rows, links = scientific._candidate_family([center], optimization, config, scientific._v3_names(config), pairwise=True)
    center_key = scientific._params_key(center, scientific._v3_names(config))
    neighbors = [row["params"] for row in rows if row["key"] in links[center_key] and row["key"] != center_key]
    assert any(abs(row["buy_step"] - center["buy_step"]) > 1e-12 and abs(row["sell_step"] - center["sell_step"]) > 1e-12 for row in neighbors)


def test_v3_rank_requires_80_percent_positive_neighbors_and_both_costs():
    center = {
        "key": (0.03,), "candidate_id": 1, "params": {"buy_step": 0.03},
        "is_center": True, "is_baseline": False, "evaluation_status": "completed",
        "eligible": True, "constraint_eligible": True,
        "validation_excess_original": 0.04, "validation_excess_stress": 0.03,
        "raw_score": 0.03, "max_drawdown": 0.1, "peak_invested": 1000,
    }
    values = [(-0.01, 0.01), (0.01, 0.01), (0.02, 0.02), (0.03, 0.03), (0.04, 0.04)]
    rows = [center]
    keys = [center["key"]]
    for index, (original, stress) in enumerate(values, start=2):
        key = (0.02 + index / 100,)
        keys.append(key)
        rows.append({
            "key": key, "candidate_id": index, "params": {"buy_step": key[0]},
            "is_center": False, "is_baseline": False, "evaluation_status": "completed",
            "eligible": True, "validation_excess_original": original,
            "validation_excess_stress": stress, "raw_score": original,
            "max_drawdown": 0.1, "peak_invested": 1000,
        })
    ranked = scientific._rank_family(rows, {center["key"]: keys}, protocol_version=3, full_history=True)
    assert ranked[0]["recommendation_eligible"] is True
    assert ranked[0]["neighbor_positive_ratio"] == 0.8
    for offset in range(4):
        key = (0.90 + offset / 100,)
        rows.append({
            "key": key, "candidate_id": 90 + offset, "params": {"buy_step": key[0]},
            "is_center": False, "is_baseline": False, "evaluation_status": "completed",
            "eligible": True, "validation_excess_original": -0.02,
            "validation_excess_stress": 0.02, "raw_score": -0.02,
            "max_drawdown": 0.1, "peak_invested": 1000,
        })
        keys.append(key)
    ranked = scientific._rank_family(rows, {center["key"]: keys}, protocol_version=3, full_history=True)
    assert ranked[0]["recommendation_eligible"] is False
    assert ranked[0]["island_suspect"] is True


def test_v3_statistical_profile_does_not_read_validation_dates():
    module = runpy.run_path("tests/test_scientific_causality.py")
    data = module["_daily_bundle"](760)
    config, _ = market()
    config = v3_config(config, symbol=data.symbol, start_date=data.calendar[504].date(), end_date=data.calendar[-1].date(), frequency="1d")
    generation = pd.DatetimeIndex(data.calendar[:252])
    profile = scientific._statistical_profile(config, data, generation)
    changed = module["_mutate_after"](data, generation[-1])
    assert scientific._statistical_profile(config, changed, generation) == profile


def test_v3_landscape_endpoint_returns_full_candidates_and_slices(tmp_path):
    store = Store(tmp_path)
    base = {
        "symbol": "159611.SZ", "start_date": "2024-01-02", "frequency": "1d",
        "initial_shares": 1000, "min_shares": 0, "max_shares": None,
        "buy_shares": 100, "sell_shares": 100, "grid_mode": "trailing",
    }
    points = []
    for index, (buy_step, sell_step) in enumerate(((0.02, 0.03), (0.03, 0.03), (0.03, 0.04)), start=1):
        points.append({
            "candidate_id": index, "params": {**base, "schema_version": 3, "buy_step": buy_step, "sell_step": sell_step},
            "is_center": index == 2, "is_baseline": False, "evaluation_status": "completed",
            "validation_excess_original": 0.01 * index, "validation_excess_stress": 0.005 * index,
            "validation_return_original": 0.02 * index, "validation_return_stress": 0.015 * index,
            "max_drawdown": 0.1, "trade_count": 10, "peak_invested": 10000,
            "folds": [{"excess_return": 0.001 * index, "grid_return": 0.002 * index, "baseline_return": 0.001, "max_drawdown": 0.05}],
        })
    result = {
        "protocol_version": 3, "config": {"protocol_version": 3, "base": base, "ranges": {"buy_step": {"low": .01, "high": .05}, "sell_step": {"low": .01, "high": .05}}},
        "evidence": {"candidate_archives": {"current": points}, "search_candidate_archives": {"current": points}},
    }
    result_id = store.save_result("optimization", result, "159611.SZ", "v3 landscape")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        compact = client.get(f"/api/v1/results/{result_id}").json()
        assert "candidate_archives" not in compact["evidence"]
        response = client.get(f"/api/v1/results/{result_id}/landscape")
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["points"]) == 3
        assert "pca" not in payload
        assert payload["slices"]["buy_step|sell_step"]["total"] == 3


def test_landscape_sampling_covers_parameter_space_and_includes_relative_best(tmp_path):
    store = Store(tmp_path)
    base = {
        "symbol": "159611.SZ", "start_date": "2024-01-02", "frequency": "1d",
        "initial_shares": 1000, "min_shares": 0, "max_shares": None,
        "buy_shares": 100, "sell_shares": 100, "grid_mode": "trailing",
    }
    points = []
    for index in range(40):
        points.append({
            "candidate_id": index + 1,
            "params": {
                **base, "schema_version": 3,
                "buy_step": .01 + index * .001, "sell_step": .02 + (index % 7) * .001,
            },
            "is_center": False, "is_baseline": False, "evaluation_status": "completed",
            "validation_excess_original": index / 1000,
            "validation_excess_stress": index / 2000,
            "validation_return_original": .02, "max_drawdown": .1,
            "trade_count": 10, "peak_invested": 10000,
        })
    result = {
        "protocol_version": 3,
        "config": {
            "protocol_version": 3, "base": base,
            "ranges": {"buy_step": {"low": .01, "high": .05}, "sell_step": {"low": .02, "high": .026}},
        },
        "evidence": {"candidate_archives": {"current": points}},
    }
    result_id = store.save_result("optimization", result, "159611.SZ", "landscape sample")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        payload = client.get(f"/api/v1/results/{result_id}/landscape?limit=5").json()

    assert payload["total"] == 40
    assert payload["sampled"] is True
    assert len(payload["points"]) == 5
    assert len({point["params"]["buy_step"] for point in payload["points"]}) > 1
    assert payload["focus"]["candidate_id"] == 40
    assert any(point["is_focus"] for point in payload["points"])
    assert payload["top_candidates"][0]["candidate_id"] == 40
    assert payload["top_candidates"][0]["rank"] == 1
    plane = payload["slices"]["buy_step|sell_step"]
    assert plane["total"] == 40
    assert plane["sampled"] is False
    assert len(plane["points"]) == 40
    assert any(point["is_focus"] for point in plane["points"])
    neighborhood = payload["focus_neighborhood"]
    assert neighborhood["available"] is True
    assert neighborhood["expected_count"] > 0
    assert neighborhood["evaluated_count"] > 0
    assert neighborhood["evaluated_count"] + neighborhood["missing_count"] == neighborhood["expected_count"]


def test_v4_landscape_exposes_independent_buy_and_sell_spacing_slices(tmp_path):
    store = Store(tmp_path)
    base = {
        "symbol": "159611.SZ", "start_date": "2024-01-02", "frequency": "1d",
        "initial_shares": 1000, "min_shares": 0, "max_shares": None,
        "buy_shares": 100, "sell_shares": 100, "grid_mode": "geometric",
    }
    points = []
    for index, (buy_step, sell_step) in enumerate(((.02, .03), (.02, .04), (.03, .03), (.03, .04)), start=1):
        points.append({
            "candidate_id": index,
            "params": {**base, "schema_version": 3, "buy_step": buy_step, "sell_step": sell_step},
            "is_center": index == 3, "is_baseline": False, "evaluation_status": "completed",
            "validation_excess_original": .01 * index, "validation_excess_stress": .005 * index,
            "validation_return_original": .02 * index, "validation_return_stress": .015 * index,
            "max_drawdown": .1, "trade_count": 10, "peak_invested": 10000,
            "folds": [{"excess_return": .001 * index, "grid_return": .002 * index, "baseline_return": .001, "max_drawdown": .05}],
        })
    result = {
        "protocol_version": 4,
        "config": {
            "protocol_version": 4, "base": base,
            "strategy": {"neighborhood_fields": ["buy_step", "sell_step"]},
            "ranges": {"buy_step": {"low": .02, "high": .03}, "sell_step": {"low": .03, "high": .04}},
        },
        "evidence": {"candidate_archives": {"current": points}, "search_candidate_archives": {"current": points}},
    }
    result_id = store.save_result("optimization", result, "159611.SZ", "v4 landscape")
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        payload = client.get(f"/api/v1/results/{result_id}/landscape").json()

    assert payload["parameters"][:2] == ["buy_step", "sell_step"]
    assert "pca" not in payload
    assert payload["slices"]["buy_step|sell_step"]["total"] == 4
    assert payload["focus_neighborhood"]["dimensions"] == ["buy_step", "sell_step"]


def test_v4_strategy_catalog_defaults_to_hypercube_geometric_research(tmp_path):
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        response = client.get("/api/v1/research/strategies")
        assert response.status_code == 200
        strategy = response.json()["strategies"][0]
        assert strategy["grid_mode"] == "geometric"
        assert strategy["default_ranking_mode"] == "fixed_mean"
        assert strategy["strategy_version"] == 3
        assert strategy["granularity"] == {
            "buy_step": .0001, "sell_step": .0001, "sell_shares": 100.0, "buy_shares": 100.0,
            "min_shares": 500.0, "max_shares": 500.0,
        }


def test_v2_config_conversion_removes_cash_budget_and_preserves_null_bound(tmp_path):
    store = Store(tmp_path)
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        converted = client.post("/api/v1/configs/convert", json={
            "schema_version": 2, "symbol": "600000.SH", "start_date": "2024-01-02",
            "initial_shares": 1000, "min_shares": 0, "max_shares": None,
            "buy_shares": 100, "sell_shares": 100, "reserve_cash": 5000,
        })
        assert converted.status_code == 200
        config = converted.json()["config"]
        assert config["schema_version"] == 3
        assert config["max_shares"] is None
        assert "reserve_cash" not in config
