"""An actual immutable-package -> ledger -> ETF statistics integration test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockregression import etf_research as research
from stockregression.etf_models import EtfResearchRequest
from stockregression.market_data import MarketDataService
from stockregression.quality import minute_slots
from stockregression.settings import Settings
from stockregression.storage import Store, dumps


def seed_market(tmp_path, *, gap_days=0):
    root = tmp_path / "data"
    settings = Settings(data_dir=str(root), package_dir=str(root / "package"), performance={"validation_workers": 1})
    store = Store(root, settings=settings)
    market = MarketDataService(store, settings=settings)
    symbol = "512880.SH"
    store.put_instrument(symbol, "集成测试 ETF", asset_type="etf", price_tick=.001)
    store.add_watch(symbol, "集成测试 ETF")
    days = pd.bdate_range("2023-01-02", periods=252)
    minute, daily = [], []
    previous = 10.
    rng = np.random.default_rng(518)
    state = 0.
    for index, day in enumerate(days):
        states = []
        for _ in range(240):
            state = .99 * state + rng.normal(0, .02)
            states.append(state)
        close = np.round(10 + np.array(states), 3)
        opening = np.r_[round(previous + (.002 if index % 2 else -.002), 3), close[:-1]]
        high, low = np.maximum(opening, close) + .001, np.minimum(opening, close) - .001
        volume = np.full(240, 1000.)
        turnover = (opening + high + low + close) / 4 * volume
        if index >= len(days) - 120:
            part = pd.DataFrame({
                "timestamp": minute_slots(day), "open": opening, "high": high,
                "low": low, "close": close, "volume": volume, "turnover": turnover,
                "source": "import", "source_verified": True, "is_synthetic": False,
            })
            if len(days) - 1 - gap_days <= index < len(days) - 1:
                part = part.iloc[60:].copy()
            minute.append(part)
        daily.append({
            "timestamp": day, "open": opening[0], "high": high.max(), "low": low.min(),
            "close": close[-1], "preclose": previous, "volume": volume.sum(),
            "turnover": turnover.sum(), "tradestatus": 1, "is_st": False,
            "source": "import", "source_verified": True, "is_synthetic": False,
        })
        previous = close[-1]
    repository = market.repository
    # Real incremental syncs may store only their five-day calendar tail in
    # the versioned auxiliary record. Full dates come from the frozen daily index.
    repository.save_auxiliary(symbol, "calendar", [{"date": str(day.date())} for day in days[-5:]], version="test-calendar", verified=True)
    repository.save_auxiliary(symbol, "actions", [], version="test-actions", verified=True)
    repository.commit_bars(symbol, "1day", pd.DataFrame(daily), source="import")
    repository.commit_bars(symbol, "1min", pd.concat(minute, ignore_index=True), source="import")
    market.refresh_snapshot(symbol)
    return store, market, days


def test_real_package_local_validation_statistics_and_frozen_auxiliary(tmp_path, monkeypatch):
    store, market, days = seed_market(tmp_path)
    monkeypatch.setattr(research, "latest_closed_date", lambda: days[-1])
    request = EtfResearchRequest()
    before = research.preflight(store, request, market=market, persist=True)
    assert before["rows"][0]["status"] == "pending_validation"
    checked = research.run_local_validation(store, request, before, market=market)
    assert checked["status"] == "completed", checked
    frozen = research.preflight(store, request, market=market, persist=True)
    assert frozen["rows"][0]["status"] == "ready", frozen["rows"]
    result = research.run_etf_research(store, request, frozen, market=market)
    assert result["ranking_complete"] and result["status"] == "completed"
    row = result["rows"][0]
    assert row["status"] == "completed", row
    assert row["metrics"]["rv_annualized"] > 0
    assert "path_redundancy" not in row["metrics"]
    for key in ("path_redundancy_daily_mean", "path_redundancy_5d_mean", "path_redundancy_20d_mean",
                "path_redundancy_daily_mean_1m", "path_redundancy_5d_mean_1m", "path_redundancy_20d_mean_1m"):
        assert row["metrics"][key] is None or 0 <= row["metrics"][key] <= 1
    assert row["gss_score"] is None  # one instrument cannot form a cross-section
    assert row["eligible"], row["filter_reasons"]
    assert row["validation"]["status"] in {"completed", "no_qualified"}, row["validation"]
    assert row["validation"]["candidates"]
    from stockregression.engine import run_backtest
    from stockregression.models import BacktestConfig
    replay = row["validation"]["candidates"][0]
    bundle = research._bundle(store, market, request, frozen["rows"][0], frozen["universe"])
    normal = run_backtest(BacktestConfig.model_validate(replay["replay_config"]), bundle)
    assert replay["metrics"]["peak_principal_return"] == normal["strategies"]["grid"]["metrics"]["peak_principal_return"]
    assert "NaN" not in dumps(result) and "Infinity" not in dumps(result)
    original = store.snapshot("512880.SH", frozen["rows"][0]["version"], persist=False)
    assert original["quality_evidence"]["intervals"]["1day"][0]["start"] == frozen["rows"][0]["daily_start"]
    market.repository.save_auxiliary("512880.SH", "actions", [{
        "ex_date": "2030-01-01", "cash_per_share": 99,
        "implemented": False, "unsupported_reason": "未来待核实行动",
    }], version="future-actions", verified=False)
    # A later correction creates new immutable price revisions and invalidates
    # the current quality ledger. The old certified snapshot remains usable.
    for timeframe in ("1day", "1min"):
        changed = market.repository.bars("512880.SH", timeframe, start=days[-1], end=days[-1]).copy()
        for column in ("open", "high", "low", "close", "turnover"):
            changed[column] *= 1.1
        market.repository.commit_bars("512880.SH", timeframe, changed, source="import")
    repeated = research.run_etf_research(store, request, frozen, market=market)
    assert repeated["rows"][0]["status"] == "completed", repeated["rows"][0]
    assert repeated["rows"][0]["metrics"] == row["metrics"]
    assert repeated["rows"][0]["validation"]["candidates"] == row["validation"]["candidates"]


def test_real_partial_package_remains_statistics_only_and_reproducible(tmp_path, monkeypatch):
    from copy import deepcopy
    import pytest
    from stockregression.market_data.models import MarketDataError
    from stockregression.snapshot_quality import frozen_quality_state

    store, market, days = seed_market(tmp_path, gap_days=1)
    monkeypatch.setattr(research, "latest_closed_date", lambda: days[-1])
    request = EtfResearchRequest()
    before = research.preflight(store, request, market=market, persist=True)
    assert research.run_local_validation(store, request, before, market=market)["status"] == "partial"
    plan = research.preflight(store, request, market=market, persist=True)
    assert plan["rows"][0]["status"] == "ready", plan["rows"]
    assert plan["rows"][0]["data_resolution"]["counts"]["daily"] == 1
    assert plan["rows"][0]["sampling"]["statistics_days"] == 119
    result = research.run_etf_research(store, request, plan, market=market)
    row = result["rows"][0]
    assert row["status"] == "completed", row
    assert row["data_status"] == "ready"
    assert row["replay_allowed"] is True
    assert row["validation"]["status"] in {"completed", "no_qualified", "no_candidates"}
    assert row["metric_details"]["rv_annualized"]["sample_count"] == 119
    assert row["data_resolution"]["counts"]["daily"] == 1
    frozen = store.snapshot(row["symbol"], row["version"], persist=False)
    omitted = days[-2]
    assert not frozen_quality_state(frozen, "1min", omitted, omitted)
    assert frozen_quality_state(frozen, "1min", days[-1], days[-1])
    # Reject a plan that tries to include a frozen excluded session.
    tampered = deepcopy(plan["rows"][0])
    tampered["sampling"]["statistics_excluded_days"] = []
    with pytest.raises(MarketDataError, match="统计样本"):
        research._bundle(store, market, request, tampered, plan["universe"])
    auxiliary = market.repository.auxiliary
    with monkeypatch.context() as patch:
        patch.setattr(market.repository, "auxiliary", lambda symbol, kind, **kwargs: None if kind == "actions" else auxiliary(symbol, kind, **kwargs))
        with pytest.raises(ValueError, match="公司行动"):
            research._bundle(store, market, request, plan["rows"][0], plan["universe"], statistics_only=True)
    # New data and ledger revisions cannot change the old complete-day sample.
    changed = market.repository.bars(row["symbol"], "1min", start=days[-1], end=days[-1]).copy()
    for column in ("open", "high", "low", "close", "turnover"):
        changed[column] *= 1.1
    market.repository.commit_bars(row["symbol"], "1min", changed, source="import")
    repeated = research.run_etf_research(store, request, plan, market=market)
    assert repeated["rows"][0]["status"] == "completed", repeated["rows"][0]
    assert repeated["rows"][0]["metrics"] == row["metrics"]


def test_legacy_ready_bundle_uses_strict_config_without_v5_data_policy(tmp_path, monkeypatch):
    from stockregression.models import DataBundle

    store, market, days = seed_market(tmp_path)
    request = EtfResearchRequest()
    manifest = store.snapshot("512880.SH", persist=True)
    assert manifest and "data_resolution" not in manifest
    seen = []
    monkeypatch.setattr(market, "readiness", lambda *args, **kwargs: {"ready": True, "degraded_days": []})
    monkeypatch.setattr(market, "load_backtest_bundle", lambda config, version: seen.append((config, version)) or
                        DataBundle("512880.SH", pd.DataFrame(), pd.DataFrame()))
    universe = {
        "package_id": manifest["package_id"], "generation": manifest["generation"],
        "versions": {"512880.SH": {"version": manifest["version"]}},
    }
    row = {"symbol": "512880.SH", "version": manifest["version"], "window_start": str(days[-120].date()),
           "window_end": str(days[-1].date()), "daily_start": str(days[-121].date()),
           "data_status": "ready", "status": "ready"}
    research._bundle(store, market, request, row, universe)
    assert len(seen) == 1
    assert seen[0][0].data_policy is None


def test_missing_initial_valuation_keeps_statistics_and_disables_replay(tmp_path, monkeypatch):
    from stockregression import etf_statistics
    from stockregression.models import DataBundle

    request = EtfResearchRequest(scope="selected", symbols=["512880.SH"], validation_top_k=1)
    day = pd.Timestamp("2024-01-10")
    candidate = {"buy_step": .01, "sell_step": .01, "base_step": .01, "reference_price": 10.,
                 "lower_price": 9., "upper_price": 11., "grids_down_count": 10,
                 "grids_up_count": 10, "asymmetric": False}
    data = DataBundle("512880.SH", pd.DataFrame(), pd.DataFrame({
        "date": [day], "close": [10.], "volume": [1000.], "tradestatus": [1],
    }))
    stats = {
        "status": "completed", "metrics": {"rv_annualized": .2},
        "metric_details": {}, "grid_basis": {}, "market_state": None,
        "actions_applied": [], "eligible": True, "filter_reasons": [],
        "warnings": [], "diagnostics": {},
    }
    monkeypatch.setattr(etf_statistics, "compute_etf_statistics", lambda *_: stats)
    monkeypatch.setattr(etf_statistics, "generate_candidates", lambda *_: [candidate])
    monkeypatch.setattr(research, "_bundle", lambda *args, **kwargs: data)
    result = research.run_etf_research(Store(tmp_path), request, {
        "method_version": "etf_min_grid_v4", "as_of": day.date().isoformat(),
        "universe": {"symbols": ["512880.SH"]},
        "rows": [{"symbol": "512880.SH", "status": "ready", "window_start": day.date().isoformat(),
                  "window_end": day.date().isoformat(), "replay_allowed": True,
                  "data_status": "ready", "warnings": [], "sampling": {"excluded_days": []}}],
    }, market=object())
    row = result["rows"][0]
    assert row["status"] == "completed"
    assert row["metrics"]["rv_annualized"] == .2
    assert row["validation"]["reason_code"] == "initial_valuation_missing"
    assert row["replay_allowed"] is False
