from __future__ import annotations

import copy
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.data import DataService
from stockregression.market_data import MarketDataService
from stockregression.models import OptimizationConfig, ResearchRequestV5
from stockregression.research_origin import ResearchOriginError, resolve_research_origin, prior_contract_hash
from stockregression.storage import Store, _write_candidate_archive, _candidate_rows


def _source(store):
    candidates = [{"id": i, "qualified": i != 2, "buy_step": i / 100, "sell_step": .02,
                   "replay_config": {"buy_shares": 200, "sell_shares": 300}}
                  for i in range(1, 6)]
    result = {"kind": "etf_research", "method_version": "etf_min_grid_v3", "result_schema_version": 1,
              "as_of": "2025-01-03", "rows": [{"symbol": "510050.SH", "window_end": "2025-01-03",
              "version": "source-retired", "metrics": {}, "validation": {"candidates": candidates}}]}
    return store.save_result("etf_research", result, "", "source")


def _config(result_id=None, **kwargs):
    return OptimizationConfig(base={"symbol": "510050.SH", "start_date": "2024-01-03", "end_date": "2025-01-03"},
                              source_etf={"result_id": result_id} if result_id else None, **kwargs)


def test_resolve_stored_candidates_ignores_forged_origin_and_missing_source_snapshot(tmp_path):
    store = Store(tmp_path)
    source = _source(store)
    config = _config(source, research_origin={"source_as_of": "2000-01-01", "seeds": []})
    origin = resolve_research_origin(store, config)
    assert origin["source_as_of"] == "2025-01-03"
    assert [seed["candidate_id"] for seed in origin["seeds"]] == [1, 3, 4]
    assert origin["seeds"][0]["params"]["buy_shares"] == 200
    assert origin["source_data_version"] == "source-retired"
    assert resolve_research_origin(store, _config(research_origin=origin)) is None
    changed = copy.deepcopy(origin)
    changed["seeds"][0]["params"]["buy_step"] = .07
    assert prior_contract_hash(config.model_copy(update={"research_origin": origin})) != prior_contract_hash(
        config.model_copy(update={"research_origin": changed}))


def test_invalid_reference_and_empty_candidate_list(tmp_path):
    store = Store(tmp_path)
    source = _source(store)
    config = _config(source)
    with pytest.raises(ResearchOriginError):
        resolve_research_origin(store, config.model_copy(update={"base": config.base.model_copy(update={"symbol": "159611.SZ"})}))
    with pytest.raises(ResearchOriginError):
        resolve_research_origin(store, _config("missing"))
    config.source_etf.candidate_ids = [999]
    with pytest.raises(ResearchOriginError):
        resolve_research_origin(store, config)
    config.source_etf.candidate_ids = []
    assert resolve_research_origin(store, config)["seeds"] == []


@pytest.mark.parametrize("standalone", [False, True])
def test_api_freezes_target_version_and_source_and_indexes_submission(tmp_path, monkeypatch, standalone):
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    source = _source(store)
    config = _config(source, prior_policy="daily_state_v1", requested_data_version="target-frozen")
    seen = []
    manifest = {"version": "target-frozen", "market_data": True, "files": {"calendar": []}}
    monkeypatch.setattr(Store, "snapshot", lambda self, symbol, version=None, **kw: manifest if version in (None, "target-frozen") else None)
    monkeypatch.setattr(MarketDataService, "readiness", lambda self, cfg, version=None: seen.append(version) or {"ready": True})
    monkeypatch.setattr("stockregression.api.preparation_config", lambda store, base, **kw: base)
    # No minute-row estimate is needed for a manifest without market_versions.
    bundle = SimpleNamespace(version="target-frozen", bars=pd.DataFrame(), daily=pd.DataFrame())
    monkeypatch.setattr(DataService, "load_bundle", lambda *a, **kw: bundle)
    monkeypatch.setattr("stockregression.research.robust.estimate_robust_search", lambda cfg, data, **kw: {
        "status": "ready", "effective_total_evaluation_budget": 100,
        "task_evaluation_upper_bound": 100, "max_runtime_seconds": 60,
        "estimated_task_memory_mb": 1, "workspace_budget_mb": 10,
    })
    payload = config.model_dump(mode="json")
    if standalone:
        payload = ResearchRequestV5(research_context=config.research_context, ranges=config.ranges,
                                   source_etf=config.source_etf, prior_policy=config.prior_policy,
                                   requested_data_version="target-frozen").model_dump(mode="json")
    payload["research_origin"] = {"source_as_of": "1990-01-01"}
    with TestClient(app) as client:
        preview = client.post("/api/v1/optimizations/preview", json=payload)
        assert preview.status_code == 200, preview.text
        assert preview.json()["data_version"] == "target-frozen"
        response = client.post("/api/v1/optimizations", json=payload)
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        assert client.post("/api/v1/optimizations", json=payload).json()["job_id"] == job_id
        job = store.job(job_id)
        assert job["payload"]["version"] == "target-frozen"
        origin = job["payload"]["config"]["research_origin"]
        assert origin["source_as_of"] == "2025-01-03"
        assert origin["source_data_version"] == "source-retired"
        assert job["payload"]["budget_contract"]["prior_policy_hash"] == preview.json()["prior_policy_hash"]
        store.update_job(job_id, status="cancelled", error="cancelled test")
        links = client.get(f"/api/v1/etf-research/{source}/research-links?symbol=510050").json()
        assert links["total"] == 1 and links["items"][0]["status"] == "cancelled"
        payload["requested_data_version"] = "deleted-target"
        missing = client.post("/api/v1/optimizations", json=payload)
        assert missing.status_code == 409 and missing.json()["detail"]["code"] == "DATA_NOT_READY"
    assert seen and set(seen) == {"target-frozen"}


def test_links_rebuild_read_only_and_completion_preserves_sources(tmp_path, monkeypatch):
    app = create_app(str(tmp_path), start_jobs=False)
    store = app.state.store
    source = _source(store)
    origin = resolve_research_origin(store, _config(source))
    job = store.new_job("optimization", {"config": {"research_origin": origin}})
    rid = store.save_result("optimization", {"fixed": {"oos_metrics": {"excess_return": .1},
        "current": {"status": "no_qualified"}}}, "510050.SH", "study")
    store.update_job(job, status="completed", result_id=rid)
    with store.db() as db:
        db.execute("DELETE FROM research_origins")
    assert store.rebuild_research_origins() == 1
    monkeypatch.setattr(DataService, "load_bundle", lambda *a, **k: pytest.fail("GET loaded market data"))
    with TestClient(app) as client:
        item = client.get(f"/api/v1/etf-research/{source}/research-links?symbol=510050.SH").json()["items"][0]
        assert item["summary"]["fixed"]["oos_metrics"]["excess_return"] == .1
        assert item["summary"]["current"]["status"] == "no_qualified"
    row = {"candidate_id": 1, "params": {"buy_step": .01, "sell_step": .02, "buy_shares": 100, "sell_shares": 200},
           "prior_sources": ["etf:source:1"], "j_score": None}
    base = _write_candidate_archive(store.root, [row])
    overlay = _write_candidate_archive(store.root, [{"candidate_id": 1, "params": row["params"], "j_score": .5}])
    merged = store.merge_candidate_archives(base, overlay)
    assert _candidate_rows(store.root, merged)["rows"][0]["prior_sources"] == row["prior_sources"]
