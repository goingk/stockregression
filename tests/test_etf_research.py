from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from stockregression.api import create_app, public_result
from stockregression.etf_models import EtfResearchRequest
from stockregression import etf_research as research
from stockregression.jobs import _runtime_limit_seconds
from stockregression.market_data.models import MarketDataError
from stockregression.storage import Store


def test_request_fingerprint_changes_with_method_version(monkeypatch):
    request = EtfResearchRequest(scope="selected", symbols=["159611.SZ"])
    plan = {
        "universe": {"package_id": "package-a", "generation": "generation-a", "symbols": ["159611.SZ"]}
    }
    monkeypatch.setattr(research, "METHOD_VERSION", "etf_min_grid_v3")
    previous = research.request_fingerprint(request, plan)
    monkeypatch.setattr(research, "METHOD_VERSION", "etf_min_grid_v4")
    current = research.request_fingerprint(request, plan)
    assert current != previous


def test_current_etf_method_version_is_v5():
    assert research.METHOD_VERSION == "etf_min_grid_v5"


class CatalogOnlyRepository:
    """Fails if preflight tries to read OHLC bodies or prepare/download data."""
    def __init__(self):
        self.identity = {"package_id": "test-package", "generation": "test-generation"}
        self.package = SimpleNamespace(manifest=lambda: {**self.identity, "datasets": {"159611.SZ|1min|none": {}}})
        self.days = pd.bdate_range("2023-01-02", "2024-12-31")
        self.pending = set()
        self.checked_versions = []
        self.checked_date_indexes = []

    def auxiliary(self, symbol, kind, version=None):
        assert version == "frozen-" + kind
        return {"payload": [{"date": day} for day in self.days] if kind == "calendar" else [], "verified": True}

    def active_manifest(self, symbol, metadata=None):
        return self.store.snapshot(symbol, persist=False)

    def trading_days(self, symbol, start, end, *, calendar_version=None, daily_version=None):
        assert calendar_version == "frozen-calendar" and daily_version == 7
        self.checked_date_indexes.append((symbol, daily_version))
        return self.days[(self.days >= start) & (self.days <= end)]

    def validation_state(self, symbol, timeframe, *, start, end, version=None, calendar_version=None, daily_version=None):
        self.checked_versions.append((symbol, timeframe, version, calendar_version))
        assert version == 7
        assert calendar_version == "frozen-calendar"
        assert daily_version == 7
        pending = symbol in self.pending
        return {
            "verified_from": None if pending else pd.Timestamp(start).date().isoformat(),
            "verified_through": None if pending else pd.Timestamp(end).date().isoformat(),
            "failed": [], "pending": [pd.Timestamp(start).date().isoformat()] if pending else [],
        }

    def bars(self, *args, **kwargs):
        raise AssertionError("preflight must not read price bodies")

    read_frame = bars
    active_rows = bars

    def day_validations(self, symbol, timeframe, *, start, end):
        return []  # The fixture's full-range state already certifies every day.


class CatalogStore:
    def __init__(self, repository):
        self.repository = repository
        repository.store = self
        self.values = {}
        self.persisted = []
        for symbol in ("159611.SZ", "512880.SH"):
            self.values[symbol] = {
                **repository.identity, "symbol": symbol, "version": "frozen-" + symbol,
                "market_data": True,
                "market_versions": {timeframe: {"version": 7, "first": "2023-01-02", "last": "2024-12-31", "rows": 100000, "status": "ready"} for timeframe in ("1day", "1min")},
                "auxiliary_versions": {kind: "frozen-" + kind for kind in ("calendar", "actions")},
            }

    def watches(self):
        return [self.instrument(symbol) for symbol in self.values] + [{"symbol": "600000.SH", "asset_type": "stock"}]

    def instrument(self, symbol):
        return {"symbol": symbol, "name": "测试 ETF", "asset_type": "etf"}

    def instruments(self, **kwargs):
        return [self.instrument(symbol) for symbol in self.values]

    def snapshot(self, symbol, version=None, *, persist=True):
        if persist:
            self.persisted.append(symbol)
        value = self.values.get(symbol)
        return deepcopy(value) if value and version in (None, value["version"]) else None

    def remember_market_snapshots(self, manifests):
        for manifest in manifests:
            self.persisted.append(manifest["symbol"])
            self.values[manifest["symbol"]] = deepcopy(manifest)


@pytest.fixture
def catalog(monkeypatch):
    repository = CatalogOnlyRepository()
    store = CatalogStore(repository)
    market = SimpleNamespace(repository=repository)
    monkeypatch.setattr(research, "latest_closed_date", lambda: pd.Timestamp("2024-12-31"))
    return store, market


def test_preflight_is_catalog_only_and_ready_index_does_not_bypass_ledger(catalog):
    store, market = catalog
    market.repository.pending.add("512880.SH")
    plan = research.preflight(store, EtfResearchRequest(), market=market)
    assert plan["as_of"] == "2024-12-31"
    assert plan["counts"] == {"total": 2, "ready": 1, "limited": 0, "pending_validation": 1, "insufficient_history": 0, "data_not_ready": 0}
    assert plan["symbols"] == ["159611.SZ", "512880.SH"]
    assert not store.persisted
    assert {item[1] for item in market.repository.checked_versions} == {"1day", "1min"}
    assert len(market.repository.checked_versions) == 4


def test_explicit_date_never_falls_back_to_an_older_minute_end(catalog):
    store, market = catalog
    store.values["159611.SZ"]["market_versions"]["1min"]["last"] = "2024-12-30"
    plan = research.preflight(store, EtfResearchRequest(as_of="2024-12-31"), market=market, persist=True)
    assert plan["as_of"] == "2024-12-31"
    row = plan["rows"][0]
    assert row["window_end"] == "2024-12-31"
    assert row["status"] == "ready"
    assert row["data_resolution"]["days"][-1]["mode"] == "daily"
    assert len(store.persisted) == 2


def test_short_incremental_calendar_does_not_hide_existing_daily_history(catalog):
    store, market = catalog
    original = market.repository.auxiliary

    def auxiliary(symbol, kind, version=None):
        result = original(symbol, kind, version)
        if kind == "calendar":
            result["payload"] = result["payload"][-5:]
        return result

    market.repository.auxiliary = auxiliary
    plan = research.preflight(store, EtfResearchRequest(), market=market)
    assert plan["counts"]["ready"] == 2
    assert plan["counts"]["insufficient_history"] == 0
    assert all(row["window_start"] == str(market.repository.days[-120].date()) for row in plan["rows"])
    assert market.repository.checked_date_indexes == [("159611.SZ", 7), ("512880.SH", 7)]


def test_default_cutoff_uses_latest_daily_and_falls_back_after_minute_coverage(catalog):
    store, market = catalog
    for snapshot in store.values.values():
        snapshot["market_versions"]["1min"]["last"] = "2024-12-30"
    plan = research.preflight(store, EtfResearchRequest(), market=market)
    assert plan["as_of"] == "2024-12-31"
    assert all(row["window_end"] == "2024-12-31" for row in plan["rows"])
    assert all(row["data_resolution"]["days"][-1]["mode"] == "daily" for row in plan["rows"])


def test_local_pool_contains_only_package_symbols(catalog):
    store, market = catalog
    plan = research.preflight(store, EtfResearchRequest(scope="local"), market=market)
    assert plan["symbols"] == ["159611.SZ"]


def test_generation_change_during_freeze_is_rejected(catalog):
    store, market = catalog
    original = market.repository.auxiliary

    def auxiliary(*args, **kwargs):
        market.repository.identity["generation"] = "changed"
        return original(*args, **kwargs)

    market.repository.auxiliary = auxiliary
    with pytest.raises(MarketDataError, match="DATA_GENERATION_CHANGED"):
        research.preflight(store, EtfResearchRequest(), market=market)


@pytest.mark.parametrize("payload", [
    {"scope": "selected", "symbols": []},
    {"scope": "selected", "symbols": ["600000.SH"]},
    {"symbols": ["159611.SZ"]},
    {"initial_cash": 100000},
    {"execution": {"initial_shares": 150}},
    {"execution": {"initial_shares": 1000, "max_shares": 500}},
    {"window_days": 60},
    {"validation_top_k": 101},
])
def test_domain_contract_rejects_unsupported_or_ambiguous_inputs(payload):
    with pytest.raises(ValidationError):
        EtfResearchRequest.model_validate(payload)


def test_execution_template_keeps_v3_share_funding_and_current_limits():
    request = EtfResearchRequest(execution={"initial_shares": 0, "max_shares": None, "buy_shares": 300, "fees": {"min_commission": 5}})
    config = request.execution.backtest("159611.SZ", "2024-01-02", "2024-02-02")
    payload = config.model_dump(mode="json")
    assert payload["schema_version"] == 3
    assert payload["initial_shares"] == 0 and payload["max_shares"] is None
    assert payload["buy_shares"] == 300 and payload["fees"]["min_commission"] == 5
    assert not {"initial_cash", "initial_position", "reserve_cash"} & payload.keys()
    assert _runtime_limit_seconds({"kind": "etf_research", "payload": {"config": request.model_dump()}}) == 3600


def test_public_result_summary_does_not_embed_etf_candidate_rows():
    raw = {"kind": "etf_research", "result_schema_version": 1, "rows_count": 1, "rows": [{"symbol": "159611.SZ", "validation": {"candidates": list(range(100))}}]}
    assert "rows" not in public_result(raw)
    assert public_result(raw, full=True) == raw


def test_api_result_pagination_and_export_are_read_only(tmp_path, monkeypatch):
    app = create_app(str(tmp_path), start_jobs=False)
    store = Store(tmp_path)
    result = {
        "kind": "etf_research", "method_version": "etf_min_grid_v1", "result_schema_version": 1,
        "status": "completed", "ranking_complete": True, "rows_count": 2,
        "rows": [{"symbol": "159611.SZ", "gss_score": 75}, {"symbol": "512880.SH", "gss_score": 25}],
    }
    result_id = store.save_result("etf_research", result, "", "ETF 测试")
    monkeypatch.setattr(research, "preflight", lambda *args, **kwargs: pytest.fail("GET must not preflight data"))
    with TestClient(app) as client:
        summary = client.get(f"/api/v1/results/{result_id}")
        assert summary.status_code == 200
        assert "rows" not in summary.json()
        page = client.get(f"/api/v1/etf-research/{result_id}/items?offset=1&limit=1").json()
        assert page["total"] == 2 and page["items"][0]["symbol"] == "512880.SH"
        selected = client.get(f"/api/v1/etf-research/{result_id}/items?symbol=159611").json()
        assert selected["total"] == 1
        export = client.get(f"/api/v1/results/{result_id}/export?kind=json")
        assert export.status_code == 200 and len(export.json()["rows"]) == 2
        history = client.get("/api/v1/results?kind=etf_research").json()
        assert history[0]["id"] == result_id and history[0]["supported"]


def test_api_submission_freezes_universe_and_creates_distinct_local_validation_task(tmp_path, catalog, monkeypatch):
    catalog_store, market = catalog
    plan = research.preflight(catalog_store, EtfResearchRequest(), market=market)
    calls = []

    def prepared(*args, **kwargs):
        calls.append(kwargs.get("persist", False))
        return plan

    monkeypatch.setattr(research, "preflight", prepared)
    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        assert client.post("/api/v1/etf-research/preflight", json={}).status_code == 200
        response = client.post("/api/v1/etf-research", json={})
        assert response.status_code == 202
        job = client.get("/api/v1/jobs/" + response.json()["job_id"]).json()
        assert job["kind"] == "etf_research"
        assert job["payload"]["plan"]["universe"]["versions"]["159611.SZ"]["version"] == plan["rows"][0]["version"]
        assert job["payload"]["submission_provenance"]["source_fingerprint"]
        validation = client.post("/api/v1/etf-research/validation", json={})
        assert validation.status_code == 202
        assert client.get("/api/v1/jobs/" + validation.json()["job_id"]).json()["kind"] == "etf_validation"
    assert calls == [False, True, True]


def test_replay_version_is_passed_through_readiness_and_never_replaced(tmp_path, monkeypatch):
    from stockregression.market_data import MarketDataService
    from stockregression.models import BacktestConfig

    app = create_app(str(tmp_path), start_jobs=False)
    version = "locked-etf-snapshot"
    manifest = {"version": version, "market_data": True}
    seen = []
    monkeypatch.setattr(Store, "snapshot", lambda self, symbol, selected=None, **kwargs: manifest if selected == version else None)
    monkeypatch.setattr(MarketDataService, "readiness", lambda self, config, selected=None: seen.append(selected) or {"ready": True})
    config = BacktestConfig(schema_version=3, symbol="159611.SZ", start_date="2024-01-02", end_date="2024-02-02")
    with TestClient(app) as client:
        response = client.post("/api/v1/backtests?version=" + version, json=config.model_dump(mode="json"))
        assert response.status_code == 200
        job = client.get("/api/v1/jobs/" + response.json()["job_id"]).json()
        assert job["payload"]["version"] == version
        assert seen == [version]
        missing = client.post("/api/v1/backtests?version=cleaned-up", json=config.model_dump(mode="json"))
        assert missing.status_code == 409
        assert seen == [version]


def replay_fixture(settlement="T1"):
    from stockregression.models import DataBundle

    day = pd.Timestamp("2024-01-03")
    daily = pd.DataFrame({
        "date": [day - pd.Timedelta(days=1), day], "open": [10., 10.],
        "high": [10., 11.], "low": [10., 9.], "close": [10., 11.],
        "preclose": [10., 10.], "volume": [100000., 300000.],
        "turnover": [1000000., 3000000.], "tradestatus": [1, 1], "is_st": [False, False],
    })
    bars = pd.DataFrame({
        "timestamp": pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=3, freq="min"),
        "open": [10., 9., 11.], "close": [9., 11., 11.],
        "high": [10., 11., 11.], "low": [9., 9., 11.],
        "volume": [100000.] * 3, "frequency": ["1m"] * 3,
        "tradable": [True] * 3, "is_synthetic": [False] * 3,
    })
    data = DataBundle("512880.SH", bars, daily, calendar=pd.DatetimeIndex([day]),
                      metadata={"asset_type": "etf", "price_tick": .001, "settlement": settlement}, version="replay-fixture")
    request = EtfResearchRequest(execution={"initial_shares": 0, "fees": {"commission_rate": 0, "min_commission": 0, "slippage_bps": 0}}, min_trades=1)
    row = {"symbol": data.symbol, "window_start": "2024-01-03", "window_end": "2024-01-03", "grid_basis": {}, "actions_applied": []}
    candidate = {"buy_step": .05, "sell_step": .05, "base_step": .05, "reference_price": 10., "lower_price": 8., "upper_price": 12., "grids_down_count": 4, "grids_up_count": 3, "asymmetric": False}
    return request, row, data, candidate


def test_candidate_replay_obeys_t_plus_one_and_matches_the_normal_engine(monkeypatch):
    from stockregression import etf_statistics
    from stockregression.engine import run_backtest
    from stockregression.models import BacktestConfig

    request, row, data, candidate = replay_fixture("T1")
    monkeypatch.setattr(etf_statistics, "generate_candidates", lambda basis: [candidate])
    t1 = research.validate_candidates(request, row, data, stopped=lambda: None)
    assert t1["candidates"][0]["metrics"]["buy_count"] == 1
    assert t1["candidates"][0]["metrics"]["sell_count"] == 0
    assert not t1["candidates"][0]["qualified"]
    assert "缺少买入或卖出成交" in t1["candidates"][0]["reasons"]
    config = BacktestConfig.model_validate(t1["candidates"][0]["replay_config"])
    normal = run_backtest(config, data)["strategies"]["grid"]["metrics"]
    assert t1["candidates"][0]["metrics"]["peak_principal_return"] == pytest.approx(normal["peak_principal_return"])
    request, row, data, _ = replay_fixture("T0")
    t0 = research.validate_candidates(request, row, data, stopped=lambda: None)
    assert t0["candidates"][0]["metrics"]["trade_count"] == 2
    assert t0["candidates"][0]["metrics"]["sell_count"] == 1


def test_candidate_replay_uses_the_starting_price_basis_and_actual_fees(monkeypatch):
    from stockregression import etf_statistics

    request, row, data, candidate = replay_fixture("T0")
    request.execution.fees.min_commission = 1000
    row["actions_applied"] = [{"ex_date": "2024-01-03", "factor": .9, "bonus_ratio": 0}]
    monkeypatch.setattr(etf_statistics, "generate_candidates", lambda basis: [candidate])
    result = research.validate_candidates(request, row, data, stopped=lambda: None)
    tested = result["candidates"][0]
    assert tested["replay_config"]["reference_price"] == pytest.approx(10 / .9)
    assert tested["replay_config"]["lower_price"] == pytest.approx(8 / .9)
    assert tested["replay_config"]["upper_price"] == pytest.approx(12 / .9)
    assert tested["replay_config"]["selection_asof"] == "2024-01-03"
    assert not tested["qualified"] and result["status"] == "no_qualified"


def test_cancelled_replay_keeps_completed_candidates_without_publishing_a_recommendation(monkeypatch):
    from stockregression import etf_statistics

    request, row, data, candidate = replay_fixture("T0")
    monkeypatch.setattr(etf_statistics, "generate_candidates", lambda basis: [candidate, {**candidate, "buy_step": .06}])
    stop = {"value": False}

    def progress(*_):
        stop["value"] = True

    result = research.validate_candidates(request, row, data, stopped=lambda: "cancelled" if stop["value"] else None, progress=progress)
    assert result["status"] == "cancelled"
    assert len(result["candidates"]) == 1
    assert result["recommendation"] is None


def test_stale_exchange_preclose_does_not_block_verified_action_factor_replay(monkeypatch):
    from stockregression import etf_statistics

    request, row, data, candidate = replay_fixture()
    row["actions_applied"] = [{"ex_date": "2024-01-03", "factor": .9, "preclose_mismatch": {"observed_reference": 9.5, "implied_reference": 9.}}]
    monkeypatch.setattr(etf_statistics, "generate_candidates", lambda basis: [candidate])
    result = research.validate_candidates(request, row, data, stopped=lambda: None)
    assert result["candidates"]
    assert result["candidates"][0]["replay_config"]["reference_price"] == pytest.approx(candidate["reference_price"] / .9)


def test_local_validation_reports_partial_coverage_without_fetching_or_overclaiming(catalog):
    store, market = catalog
    request = EtfResearchRequest(scope="selected", symbols=["159611.SZ"])
    plan = research.preflight(store, request, market=market, persist=True)
    calls = []

    def revalidate(symbol, timeframe, **kwargs):
        calls.append((symbol, timeframe))
        return {"status": "partial" if timeframe == "1min" else "complete", "pending_validation": ["缺口"] if timeframe == "1min" else []}

    market.revalidate = revalidate
    result = research.run_local_validation(store, request, plan, market=market)
    assert result["status"] == "partial"
    assert result["rows"][0]["status"] == "partial"
    assert calls == [("159611.SZ", "1day"), ("159611.SZ", "1min")]


def test_local_validation_does_not_silently_validate_data_changed_while_queued(catalog):
    store, market = catalog
    request = EtfResearchRequest(scope="selected", symbols=["159611.SZ"])
    plan = research.preflight(store, request, market=market, persist=True)
    store.values["159611.SZ"]["market_versions"]["1min"]["version"] = 8
    market.revalidate = lambda *args, **kwargs: pytest.fail("must not validate a different queued input")
    result = research.run_local_validation(store, request, plan, market=market)
    assert result["status"] == "partial"
    assert result["rows"][0]["status"] == "failed"
    assert "DATA_VERSION_CHANGED" in result["rows"][0]["reason"]


@pytest.mark.parametrize("status", ["partial", "cancelled"])
def test_job_terminal_states_keep_etf_result_artifacts(tmp_path, catalog, monkeypatch, status):
    from stockregression import reproducibility
    from stockregression.jobs import execute_job

    catalog_store, market = catalog
    plan = research.preflight(catalog_store, EtfResearchRequest(), market=market)
    monkeypatch.setattr(research, "preflight", lambda *args, **kwargs: plan)
    monkeypatch.setattr(reproducibility, "computation_provenance", lambda *args, **kwargs: {"source_fingerprint": "fixed-test-source"})
    monkeypatch.setattr(research, "run_etf_research", lambda *args, **kwargs: {
        "kind": "etf_research", "method_version": "etf_min_grid_v1", "result_schema_version": 1,
        "status": status, "rows_count": 1, "ranking_complete": False,
        "rows": [{"symbol": "159611.SZ", "status": "completed", "metrics": {"rv_annualized": .2}}],
    })
    app = create_app(str(tmp_path), start_jobs=False)
    with TestClient(app) as client:
        response = client.post("/api/v1/etf-research", json={})
        job_id = response.json()["job_id"]
        assert Store(tmp_path).claim_job(job_id, "etf-test-worker")
        execute_job(str(tmp_path), job_id)
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"] == status, job
        assert job["result_id"]
        rows = client.get(f"/api/v1/etf-research/{job['result_id']}/items").json()
        assert rows["items"][0]["metrics"]["rv_annualized"] == .2


def test_etf_jobs_share_cpu_leases_with_existing_research(tmp_path):
    from stockregression.resources import ResourceLease

    store = Store(tmp_path)
    first = store.new_job("etf_research", {})
    assert store.claim_job(first, "first-worker")
    first_lease = ResourceLease(store, first, budget=4)
    assert first_lease.acquire() == 4
    second = store.new_job("optimization", {})
    assert store.claim_job(second, "second-worker")
    second_lease = ResourceLease(store, second, budget=4)
    try:
        assert first_lease.rebalance() == 2
        assert second_lease.acquire() == 2
        assert first_lease.units + second_lease.units == 4
    finally:
        first_lease.release()
        second_lease.release()


def test_etf_research_stops_at_the_shared_process_working_set_budget(monkeypatch):
    monkeypatch.setattr(research, "current_process_rss_bytes", lambda: 300 * 1024 * 1024)
    monkeypatch.setattr(research, "current_workspace_bytes", lambda: 256 * 1024 * 1024)
    assert research._stop(EtfResearchRequest(), research.time.monotonic(), lambda: False) == "workspace_limit"


def test_individual_validation_uses_original_versions_window_and_execution(tmp_path, catalog, monkeypatch):
    catalog_store, market = catalog
    request = EtfResearchRequest(execution={"buy_shares": 300, "fees": {"min_commission": 3}})
    plan = research.preflight(catalog_store, request, market=market)
    app = create_app(str(tmp_path), start_jobs=False)
    store = Store(tmp_path)
    result_id = store.save_result("etf_research", {
        "kind": "etf_research", "result_schema_version": 1, "as_of": plan["as_of"],
        "request": request.model_dump(mode="json"), "universe": plan["universe"],
        "rows": plan["rows"],
    }, "", "原始 ETF 研究")
    frozen_version = plan["rows"][0]["version"]
    monkeypatch.setattr(Store, "snapshot", lambda self, symbol, version=None, **kwargs: {"version": version} if version == frozen_version else None)
    monkeypatch.setattr(research, "preflight", lambda *args, **kwargs: pytest.fail("individual validation must not switch to current snapshots"))
    with TestClient(app) as client:
        response = client.post(f"/api/v1/etf-research/{result_id}/validate/159611.SZ")
        assert response.status_code == 202
        job = client.get("/api/v1/jobs/" + response.json()["job_id"]).json()
        payload = job["payload"]
        assert payload["source_result_id"] == result_id
        assert payload["config"]["symbols"] == ["159611.SZ"] and payload["config"]["validation_top_k"] == 1
        assert payload["config"]["execution"]["buy_shares"] == 300
        assert payload["config"]["execution"]["fees"]["min_commission"] == 3
        assert payload["plan"]["rows"][0]["version"] == frozen_version
        assert payload["plan"]["rows"][0]["window_start"] == plan["rows"][0]["window_start"]
        missing = client.post(f"/api/v1/etf-research/{result_id}/validate/512880.SH")
        assert missing.status_code == 409 and missing.json()["detail"]["code"] == "DATA_VERSION_NOT_FOUND"


def _install_minute_failures(monkeypatch, market, failures, *, reason="存在 60 个未补齐的分钟缺口", pending=False, invalid=False):
    original = market.repository.validation_state

    def state(symbol, timeframe, **kwargs):
        value = original(symbol, timeframe, **kwargs)
        dates = failures.get(symbol, []) if timeframe == "1min" else []
        if dates:
            window = market.repository.days[(market.repository.days >= kwargs["start"]) & (market.repository.days <= kwargs["end"])]
            value.update(
                failed=dates, failed_count=len(dates), validated_days=len(window) - len(dates),
                pending=[dates[0]] if pending else [], pending_count=int(pending),
            )
        return value

    monkeypatch.setattr(market.repository, "validation_state", state)

    def day_validations(symbol, timeframe, *, start, end):
        sessions = market.repository.days[
            (market.repository.days >= pd.Timestamp(start)) & (market.repository.days <= pd.Timestamp(end))
        ]
        failed = set(failures.get(symbol, [])) if timeframe == "1min" else set()
        result = []
        for session in sessions:
            day = session.date().isoformat()
            broken = timeframe == "1min" and day in failed
            result.append({
                "trade_date": day, "status": "failed" if broken else "passed",
                "row_count": 180 if broken else (240 if timeframe == "1min" else 1),
                "report": ({"reason": reason, "pending_validation": [reason], "unfillable": 60,
                            "blocked_price_rows": int(invalid)} if broken else {}),
            })
        return result

    monkeypatch.setattr(market.repository, "day_validations", day_validations, raising=False)


def test_v5_falls_back_per_day_and_never_certifies_missing_minutes(catalog, monkeypatch):
    from stockregression.snapshot_quality import frozen_quality_state

    store, market = catalog
    failed = "2024-12-05"
    _install_minute_failures(monkeypatch, market, {"159611.SZ": [failed]})
    manifests = []
    monkeypatch.setattr(store, "remember_market_snapshots", lambda values: manifests.extend(deepcopy(values)))
    plan = research.preflight(store, EtfResearchRequest(), market=market, persist=True)
    limited, complete = plan["rows"]
    assert plan["counts"]["ready"] == 2
    assert limited["status"] == "ready" and complete["status"] == "ready"
    assert limited["data_resolution"]["counts"]["daily"] == 1
    assert limited["replay_allowed"] is True
    assert complete["data_resolution"]["counts"]["daily"] == 0
    for row in plan["rows"]:
        assert row["sampling"]["statistics_excluded_days"] == [failed]
        assert row["sampling"]["statistics_days"] == 119
        assert row["sampling"]["comparable"] is True
    frozen = next(value for value in manifests if value["symbol"] == limited["symbol"])
    assert frozen["etf_sampling"] == limited["sampling"]
    assert frozen_quality_state(frozen, "1min", failed, failed) is None
    assert frozen_quality_state(frozen, "1min", "2024-12-06", "2024-12-06")
    assert frozen_quality_state(frozen, "1day", limited["window_start"], limited["window_end"])


@pytest.mark.parametrize("options", [
    {"pending": True}, {"invalid": True},
    {"reason": "分钟来源标签或单位未校准"},
])
def test_pending_invalid_or_uncalibrated_data_is_not_admitted(catalog, monkeypatch, options):
    store, market = catalog
    _install_minute_failures(monkeypatch, market, {"159611.SZ": ["2024-12-05"]}, **options)
    row = research.preflight(store, EtfResearchRequest(), market=market)["rows"][0]
    assert row["status"] == "ready"
    failed_day = next(item for item in row["data_resolution"]["days"] if item["date"] == "2024-12-05")
    assert failed_day["mode"] == "daily"


@pytest.mark.parametrize("failed_days", [12, 13])
def test_old_minute_coverage_thresholds_do_not_reject_daily_fallback(catalog, monkeypatch, failed_days):
    store, market = catalog
    dates = [day.date().isoformat() for day in market.repository.days[-20:-20 + failed_days]]
    _install_minute_failures(monkeypatch, market, {"159611.SZ": dates})
    row = research.preflight(store, EtfResearchRequest(), market=market)["rows"][0]
    assert row["status"] == "ready"
    assert row["data_resolution"]["counts"]["daily"] == failed_days


def test_incompatible_samples_keep_diagnostics_without_mixing_gss(catalog, monkeypatch):
    store, market = catalog
    dates = [day.date().isoformat() for day in market.repository.days[-20:-4]]
    _install_minute_failures(monkeypatch, market, {"159611.SZ": dates[:8], "512880.SH": dates[8:]})
    rows = research.preflight(store, EtfResearchRequest(), market=market)["rows"]
    assert all(row["status"] == "ready" for row in rows)
    assert all(row["sampling"]["comparable"] is False for row in rows)
    assert all(row["sampling"]["statistics_days"] == 112 for row in rows)


def test_individual_replay_cannot_bypass_limited_statistics_policy(tmp_path):
    app = create_app(str(tmp_path), start_jobs=False)
    store = Store(tmp_path)
    result_id = store.save_result("etf_research", {
        "kind": "etf_research", "result_schema_version": 1, "as_of": "2024-12-31",
        "request": EtfResearchRequest().model_dump(mode="json"),
        "rows": [{"symbol": "159611.SZ", "data_status": "limited"}],
    }, "", "有缺口的统计")
    with TestClient(app) as client:
        response = client.post(f"/api/v1/etf-research/{result_id}/validate/159611.SZ")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "DATA_NOT_READY"


def test_short_filled_sessions_are_excluded_even_when_the_ledger_passed(catalog, monkeypatch):
    store, market = catalog
    dates = ["2024-12-05", "2024-12-06"]
    def records(symbol, timeframe, *, start, end):
        sessions = market.repository.days[
            (market.repository.days >= pd.Timestamp(start)) & (market.repository.days <= pd.Timestamp(end))
        ]
        if timeframe == "1day":
            return [{"trade_date": session.date().isoformat(), "status": "passed", "row_count": 1}
                    for session in sessions]
        return [{"trade_date": day, "status": "passed", "row_count": 238,
                 "report": {"synthetic_rows": 2, "auction_placeholders": 0}}
                for day in (dates if symbol == "159611.SZ" and timeframe == "1min" else [])]
    monkeypatch.setattr(market.repository, "day_validations", records)
    rows = research.preflight(store, EtfResearchRequest(), market=market)["rows"]
    assert rows[0]["status"] == "ready"
    assert rows[0]["sampling"]["excluded_days"] == dates
    assert rows[1]["sampling"]["statistics_excluded_days"] == dates


def test_whole_missing_sessions_must_also_pass_the_95_percent_minute_gate(catalog, monkeypatch):
    store, market = catalog
    dates = [day.date().isoformat() for day in market.repository.days[-20:-8]]
    _install_minute_failures(monkeypatch, market, {"159611.SZ": dates})
    original = market.repository.day_validations
    def records(symbol, timeframe, **kwargs):
        rows = original(symbol, timeframe, **kwargs)
        if symbol == "159611.SZ" and timeframe == "1min":
            for row in rows:
                if row["trade_date"] in dates:
                    row.update(row_count=0, report={"reason": "交易日没有行情记录"})
        return rows
    monkeypatch.setattr(market.repository, "day_validations", records)
    assert research.preflight(store, EtfResearchRequest(), market=market)["rows"][0]["status"] == "ready"


def test_sparse_etfs_do_not_shrink_the_healthy_gss_common_sample():
    days = pd.bdate_range("2024-01-01", periods=250)
    rows = []
    windows = {}
    for index in range(8):
        symbol = f"51{index:04d}.SH"
        windows[symbol] = days
        excluded = [day.date().isoformat() for day in days[:8]]
        rows.append({"symbol": symbol, "status": "ready", "warnings": [], "sampling": {
            "expected_days": 250, "valid_days": 242, "excluded_days": excluded,
            "statistics_excluded_days": excluded.copy(), "statistics_days": 242,
            "comparable": True,
        }})
    for symbol, minute_days in (("513310.SH", 219), ("515330.SH", 112), ("530000.SH", 7)):
        windows[symbol] = days
        excluded = [day.date().isoformat() for day in days[minute_days:]]
        rows.append({"symbol": symbol, "status": "ready", "warnings": [], "sampling": {
            "expected_days": 250, "valid_days": minute_days, "excluded_days": excluded,
            "statistics_excluded_days": excluded.copy(), "statistics_days": minute_days,
            "comparable": True,
        }})
    research._align_sampling(rows, windows)
    healthy = rows[:8]
    sparse = rows[8:]
    assert all(row["sampling"]["comparable"] for row in healthy)
    assert all(row["sampling"]["statistics_days"] == 242 for row in healthy)
    assert all(row["sampling"]["comparable"] is False for row in sparse)
    assert [row["sampling"]["statistics_days"] for row in sparse] == [219, 112, 7]
