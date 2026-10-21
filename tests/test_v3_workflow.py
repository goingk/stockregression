"""Share-account ledger behaviour under the v5 peak-principal contract.

The v3 research protocol is gone from the product: ``optimize()`` only
accepts ``protocol_version=5`` and the landscape endpoint only serves v5
results.  What survives is the ledger that protocol produced, and that is
what this file pins down: peak net contributions fund the grid on demand,
the matched funding hold is the ranking baseline, and the account stays
continuous across schedule changes.
"""
from __future__ import annotations

import runpy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.models import BacktestConfig
from stockregression.share_engine import run_share_backtest


def market():
    fixture = runpy.run_path("tests/test_research_kernel.py")["market"]
    return fixture()


def v3_config(base: BacktestConfig, **updates) -> BacktestConfig:
    """The v5 share account: schema_version=3, geometric grid, unlimited size."""
    values = base.model_dump(mode="python")
    values.update({
        "schema_version": 3, "grid_mode": "geometric", "min_shares": 0, "max_shares": None,
        **updates,
    })
    values.pop("reserve_cash", None)
    return BacktestConfig.model_validate(values)


def test_top_up_accounting_uses_peak_net_contributions():
    config, bundle = market()
    config = v3_config(config)
    result = run_share_backtest(config, bundle)
    funding = result["funding"]
    grid = result["strategies"]["grid"]
    metrics = grid["metrics"]
    expected_peak = funding["initial_principal"] + funding["topups"]
    assert funding["funding_mode"] == "top_up_on_fill"
    assert funding["topups"] > 0
    assert funding["peak_invested"] == expected_peak
    assert metrics["total_invested"] == expected_peak
    assert metrics["peak_principal_return"] == pytest.approx(
        (grid["equity"][-1]["equity"] - expected_peak) / expected_peak)
    assert metrics["return_basis"] == "peak_principal"
    assert metrics["risk_basis"] == "principal_profit_drawdown"
    # The principal curve divides the running profit by the same peak principal.
    for point in grid["equity"]:
        assert point["principal_curve"] == pytest.approx(
            (point["equity"] - point["total_invested"]) / expected_peak + 1)
    # Only buys fund the account; a sell never asks for outside money.
    assert all(trade["external_topup"] == 0 for trade in grid["trades"] if trade["side"] == "sell")


def test_paused_schedule_stops_trading_until_resume():
    config, bundle = market()
    config = v3_config(config)
    days = pd.DatetimeIndex(bundle.calendar)
    params = {
        "buy_step": config.buy_step, "sell_step": config.sell_step,
        "buy_shares": config.buy_shares, "sell_shares": config.sell_shares,
    }
    schedule = [
        {"effective_date": days[0].date().isoformat(), "decision_date": (days[0] - pd.offsets.BDay(1)).date().isoformat(), "params": params, "paused": True},
        {"effective_date": days[2].date().isoformat(), "decision_date": days[1].date().isoformat(), "params": params, "paused": False},
    ]
    result = run_share_backtest(config, bundle, parameter_schedule=schedule)
    trades = result["strategies"]["grid"]["trades"]
    assert trades
    assert pd.Timestamp(trades[0]["timestamp"]).normalize() == days[2]
    assert all(pd.Timestamp(trade["timestamp"]).normalize() >= days[2] for trade in trades)
    assert result["parameter_schedule"][0]["paused"] is True


def test_zero_initial_principal_is_funded_by_external_contributions():
    config, bundle = market()
    config = v3_config(config, initial_shares=0, min_shares=0, buy_shares=100)
    config = config.model_copy(update={"reference_price_mode": "manual", "reference_price": 20.0})
    result = run_share_backtest(config, bundle)
    funding = result["funding"]
    grid = result["strategies"]["grid"]
    assert funding["initial_principal"] == 0
    assert funding["topups"] > 0
    assert funding["peak_invested"] == funding["topups"]
    assert grid["metrics"]["peak_principal_return"] == pytest.approx(
        (grid["equity"][-1]["equity"] - funding["topups"]) / funding["topups"])
    assert grid["equity"][0]["principal_curve"] is not None
    # With no principal of its own, the account only trades what an external
    # contribution pays for: the first fill is funded to its traded value.
    first = grid["trades"][0]
    assert first["side"] == "buy"
    assert first["external_topup"] == pytest.approx(first["quantity"] * first["price"])


def test_strategy_catalog_defaults_to_robust_j_geometric_research(tmp_path):
    with TestClient(create_app(str(tmp_path), start_jobs=False)) as client:
        response = client.get("/api/v1/research/strategies")
        assert response.status_code == 200
        strategy = response.json()["strategies"][0]
        assert strategy["grid_mode"] == "geometric"
        assert strategy["default_ranking_mode"] == "robust_j"
        assert strategy["protocol_version"] == 5
        assert strategy["strategy_version"] == 5
        assert strategy["fields"] == ["buy_step", "sell_step", "buy_shares", "sell_shares"]
        assert strategy["ranking_baseline"] == "matched_funding_hold"
        assert strategy["ranking_basis"] == "peak_principal_return_difference"
        assert strategy["granularity"] == {
            "buy_step": .0001, "sell_step": .0001, "sell_shares": 100.0, "buy_shares": 100.0,
        }


def test_legacy_config_conversion_removes_cash_budget_and_preserves_null_bound(tmp_path):
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
