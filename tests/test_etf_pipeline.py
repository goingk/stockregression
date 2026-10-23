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


def seed_market(tmp_path):
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
            minute.append(pd.DataFrame({
                "timestamp": minute_slots(day), "open": opening, "high": high,
                "low": low, "close": close, "volume": volume, "turnover": turnover,
                "source": "import", "source_verified": True, "is_synthetic": False,
            }))
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
    assert 0 <= row["metrics"]["path_redundancy"] <= 1
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
