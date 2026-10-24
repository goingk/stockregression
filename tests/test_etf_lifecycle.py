from __future__ import annotations

from types import SimpleNamespace

import pytest

from stockregression import etf_research as research
from stockregression import etf_statistics
from stockregression.etf_models import EtfResearchRequest
from stockregression.market_data.models import MarketDataError


SYMBOLS = [f"510{i:03d}.SH" for i in range(1, 13)]
SCORE_METRICS = (
    "rv_annualized",
    "path_redundancy_daily_mean",
    "path_redundancy_5d_mean",
    "variance_ratio_240m",
    "iovr",
    "zero_crossing_rate",
    "jump_ratio",
)


def _request(**values) -> EtfResearchRequest:
    return EtfResearchRequest(scope="selected", symbols=SYMBOLS, **values)


def _plan(request: EtfResearchRequest) -> dict:
    rows = [
        {
            "symbol": symbol,
            "symbol_name": f"测试 ETF {index + 1}",
            "status": "ready",
            "window_start": "2024-01-02",
            "window_end": "2024-06-28",
            "version": f"frozen-{symbol}",
            "gaps": [],
            "warnings": [],
        }
        for index, symbol in enumerate(SYMBOLS)
    ]
    return {
        "as_of": "2024-06-28",
        "requested_as_of": None,
        "window_days": request.window_days,
        "scope": request.scope,
        "symbols": list(SYMBOLS),
        "rows": rows,
        "counts": {
            "total": len(rows), "ready": len(rows), "pending_validation": 0,
            "insufficient_history": 0, "data_not_ready": 0,
        },
        "universe": {
            "scope": request.scope,
            "symbols": list(SYMBOLS),
            "package_id": "package-current",
            "generation": "generation-current",
            "as_of": "2024-06-28",
            "versions": {
                symbol: {"version": f"frozen-{symbol}"}
                for symbol in SYMBOLS
            },
        },
    }


def _completed_statistics(symbol: str) -> dict:
    index = SYMBOLS.index(symbol)
    value = index / 100.0
    metrics = {key: value for key in SCORE_METRICS}
    return {
        "status": "completed",
        "metrics": metrics,
        "metric_details": {key: {"status": "ok", "sample_count": 100} for key in metrics},
        "grid_basis": {
            "reference_price": 1.0,
            "lower_price": 0.8,
            "upper_price": 1.2,
            "current_raw_price": 1.0,
            "atr30_pct": 0.01,
            "svr": 1.0,
            "realized_skewness": 0.0,
            "realized_kurtosis": 0.0,
            "price_tick": 0.001,
        },
        "diagnostics": {"symbol": symbol},
        "eligible": symbol != SYMBOLS[-1],
        "filter_reasons": ["status_unavailable"] if symbol == SYMBOLS[-1] else [],
        "warnings": [],
    }


def _install_statistics(monkeypatch, on_statistics=None):
    def compute(data, _as_of, _window_days):
        if on_statistics:
            on_statistics(data.symbol)
        return _completed_statistics(data.symbol)

    monkeypatch.setattr(etf_statistics, "compute_etf_statistics", compute)


def _run(plan: dict, request: EtfResearchRequest, *, cancelled=None) -> dict:
    return research.run_etf_research(
        SimpleNamespace(), request, plan,
        market=SimpleNamespace(), cancelled=cancelled,
    )


def test_default_auto_validation_replays_only_top_ten_eligible_scores(monkeypatch):
    request = _request()
    plan = _plan(request)
    replayed = []
    monkeypatch.setattr(
        research, "_bundle",
        lambda _store, _market, _request, row, _universe, **_kwargs: SimpleNamespace(symbol=row["symbol"]),
    )
    _install_statistics(monkeypatch)

    def validate(_request, row, _data, **_kwargs):
        replayed.append(row["symbol"])
        return {"status": "no_qualified", "candidates": [], "recommendation": None, "forward_config": None}

    monkeypatch.setattr(research, "validate_candidates", validate)
    result = _run(plan, request)
    rows = {row["symbol"]: row for row in result["rows"]}

    assert request.validation_top_k == 10
    assert result["ranking_complete"] is True
    assert result["counts"]["scored"] == 12
    assert result["counts"]["eligible"] == 11
    assert result["counts"]["validated"] == 10
    assert all(rows[symbol]["gss_score"] is not None for symbol in SYMBOLS)
    assert rows[SYMBOLS[-1]]["rank"] == 1
    assert rows[SYMBOLS[-1]]["eligible"] is False
    assert replayed == list(reversed(SYMBOLS[1:11]))
    assert SYMBOLS[0] not in replayed
    assert SYMBOLS[-1] not in replayed


def test_legacy_volume_diagnostics_do_not_block_replay(monkeypatch):
    request = _request()
    plan = _plan(request)
    blocked = SYMBOLS[-2]
    reason = "分钟与日线成交量单位待核对"
    monkeypatch.setattr(research, "_bundle", lambda _store, _market, _request, row, _universe, **_kwargs: SimpleNamespace(
        symbol=row["symbol"], metadata={"etf_replay_blocked_reason": reason} if row["symbol"] == blocked else {},
    ))
    _install_statistics(monkeypatch)
    replayed = []

    def validate(_request, row, _data, **_kwargs):
        replayed.append(row["symbol"])
        return {"status": "no_qualified", "candidates": [], "recommendation": None, "forward_config": None}

    monkeypatch.setattr(research, "validate_candidates", validate)
    result = _run(plan, request)
    row = next(item for item in result["rows"] if item["symbol"] == blocked)
    assert row["status"] == "completed" and row["data_status"] == "ready"
    assert row["gss_score"] is not None
    assert row["validation"]["status"] == "no_qualified"
    assert blocked in replayed and len(replayed) == 10


def test_statistics_cancellation_keeps_completed_metrics_without_publishing_scores(monkeypatch):
    state = {"cancelled": False}
    request = _request()
    plan = _plan(request)
    monkeypatch.setattr(
        research, "_bundle",
        lambda _store, _market, _request, row, _universe, **_kwargs: SimpleNamespace(symbol=row["symbol"]),
    )

    def statistics(symbol):
        if symbol == SYMBOLS[0]:
            state["cancelled"] = True

    _install_statistics(monkeypatch, statistics)
    monkeypatch.setattr(
        research, "validate_candidates",
        lambda *args, **kwargs: pytest.fail("cancelled statistics must not enter replay"),
    )
    result = _run(plan, request, cancelled=lambda: state["cancelled"])
    completed = next(row for row in result["rows"] if row["symbol"] == SYMBOLS[0])

    assert result["status"] == "cancelled"
    assert result["stop_reason"] == "cancelled"
    assert result["ranking_complete"] is False
    assert completed["status"] == "completed"
    assert completed["metrics"]["rv_annualized"] == 0.0
    assert completed["gss_score"] is None
    assert completed["rank"] is None
    assert all(row["gss_score"] is None and row["rank"] is None for row in result["rows"])
    assert result["counts"]["scored"] == 0


def test_replay_cancellation_keeps_full_ranking_and_completed_candidates(monkeypatch):
    request = _request()
    plan = _plan(request)
    replayed = []
    candidate = {"id": 1, "qualified": True, "reasons": []}
    monkeypatch.setattr(
        research, "_bundle",
        lambda _store, _market, _request, row, _universe, **_kwargs: SimpleNamespace(symbol=row["symbol"]),
    )
    _install_statistics(monkeypatch)

    def validate(_request, row, _data, **_kwargs):
        replayed.append(row["symbol"])
        return {
            "status": "cancelled", "stop_reason": "cancelled",
            "candidates": [candidate], "recommendation": None, "forward_config": None,
        }

    monkeypatch.setattr(research, "validate_candidates", validate)
    result = _run(plan, request)
    stopped_row = next(row for row in result["rows"] if row["symbol"] == replayed[0])

    assert result["status"] == "cancelled"
    assert result["stop_reason"] == "cancelled"
    assert result["ranking_complete"] is True
    assert len(replayed) == 1
    assert all(row["gss_score"] is not None and row["rank"] is not None for row in result["rows"])
    assert stopped_row["validation"]["status"] == "cancelled"
    assert stopped_row["validation"]["candidates"] == [candidate]
    assert stopped_row["validation"]["recommendation"] is None
    assert result["counts"]["validated"] == 0
    assert result["counts"]["recommended"] == 0


def test_generation_change_during_statistics_returns_partial_stop_state(monkeypatch):
    request = _request()
    plan = _plan(request)

    def bundle(*_args, **_kwargs):
        raise MarketDataError("DATA_GENERATION_CHANGED: 数据包代际已改变，请重新预检和提交")

    monkeypatch.setattr(research, "_bundle", bundle)
    _install_statistics(monkeypatch)
    monkeypatch.setattr(
        research, "validate_candidates",
        lambda *args, **kwargs: pytest.fail("generation change must stop before replay"),
    )
    result = _run(plan, request)

    assert result["status"] == "partial"
    assert result["stop_reason"] == "data_generation_changed"
    assert "已停止" in result["message"]
    assert result["ranking_complete"] is False
    assert result["counts"]["scored"] == 0
    assert all(row["gss_score"] is None and row["rank"] is None for row in result["rows"])


def test_cleaned_frozen_version_marks_the_symbol_not_ready_with_a_clear_gap(monkeypatch):
    request = _request()
    plan = _plan(request)
    missing_symbol = SYMBOLS[4]

    def bundle(_store, _market, _request, row, _universe, **_kwargs):
        if row["symbol"] == missing_symbol:
            raise MarketDataError("DATA_VERSION_NOT_FOUND: 冻结的行情版本已不可用")
        return SimpleNamespace(symbol=row["symbol"])

    monkeypatch.setattr(research, "_bundle", bundle)
    _install_statistics(monkeypatch)
    monkeypatch.setattr(
        research, "validate_candidates",
        lambda *_args, **_kwargs: {
            "status": "no_qualified", "candidates": [], "recommendation": None, "forward_config": None,
        },
    )
    result = _run(plan, request)
    missing = next(row for row in result["rows"] if row["symbol"] == missing_symbol)

    assert result["ranking_complete"] is True
    assert result["counts"]["scored"] == 11
    assert result["counts"]["data_not_ready"] == 1
    assert missing["status"] == "data_not_ready"
    assert missing["data_status"] == "data_not_ready"
    assert missing["gss_score"] is None and missing["rank"] is None
    assert any(
        gap["kind"] == "quality" and "DATA_VERSION_NOT_FOUND" in gap["message"]
        for gap in missing["gaps"]
    )


def test_last_statistics_completion_cancel_prevents_score_publication(monkeypatch):
    state = {"cancelled": False}
    request = _request()
    plan = _plan(request)
    monkeypatch.setattr(
        research, "_bundle",
        lambda _store, _market, _request, row, _universe, **_kwargs: SimpleNamespace(symbol=row["symbol"]),
    )

    def compute(data, _as_of, _window_days):
        statistics = _completed_statistics(data.symbol)
        if data.symbol == SYMBOLS[-1]:
            state["cancelled"] = True
        return statistics

    monkeypatch.setattr(etf_statistics, "compute_etf_statistics", compute)
    monkeypatch.setattr(
        research, "validate_candidates",
        lambda *args, **kwargs: pytest.fail("cancelled scoring must not enter replay"),
    )
    result = _run(plan, request, cancelled=lambda: state["cancelled"])

    assert result["status"] == "cancelled"
    assert result["stop_reason"] == "cancelled"
    assert result["ranking_complete"] is False
    assert result["counts"]["scored"] == 0
    assert all(row["status"] == "completed" for row in result["rows"])
    assert all(row["metrics"]["rv_annualized"] is not None for row in result["rows"])
    assert all(row["gss_score"] is None and row["rank"] is None for row in result["rows"])


def test_runtime_limit_after_last_statistics_keeps_metrics_and_returns_partial(monkeypatch):
    request = _request(max_runtime_seconds=1)
    plan = _plan(request)
    statistics_calls = []
    clock = iter([0.0] * 13 + [1.0])
    monkeypatch.setattr(research.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        research, "_bundle",
        lambda _store, _market, _request, row, _universe, **_kwargs: SimpleNamespace(symbol=row["symbol"]),
    )

    def compute(data, _as_of, _window_days):
        statistics_calls.append(data.symbol)
        return _completed_statistics(data.symbol)

    monkeypatch.setattr(etf_statistics, "compute_etf_statistics", compute)
    monkeypatch.setattr(
        research, "validate_candidates",
        lambda *args, **kwargs: pytest.fail("runtime stop must not enter replay"),
    )
    result = _run(plan, request)

    assert statistics_calls == SYMBOLS
    assert result["status"] == "partial"
    assert result["stop_reason"] == "runtime_limit"
    assert result["ranking_complete"] is False
    assert result["counts"]["scored"] == 0
    assert all(row["status"] == "completed" for row in result["rows"])
    assert all(row["metrics"]["rv_annualized"] is not None for row in result["rows"])
    assert all(row["gss_score"] is None and row["rank"] is None for row in result["rows"])
