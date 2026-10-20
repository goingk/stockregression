import copy
import csv
import io

from fastapi.testclient import TestClient
import pytest

from stockregression.api import create_app
from stockregression.models import SearchRange
from stockregression.research_v4.connected import analyze_regions
from stockregression.research_v4.selector import _public_region
from stockregression.research_v4.space import ParameterSpace
from stockregression.storage import Store
from test_connected_regions import make_space, full_rows


@pytest.fixture
def region_result(tmp_path):
    space = make_space(5,5)
    config = space.config.model_copy(update={"ranges": {**space.config.ranges, "buy_shares": SearchRange(low=100,high=300)}})
    space = ParameterSpace(config)
    contexts, archives = [], []
    for reverse in (False,True):
        rows = full_rows(space,lambda p: -.1 if p[:2] == (3,3) else ((20-p[0]-p[1]-.1*p[3]) if reverse else (p[0]+p[1]+.1*p[3]))/100)
        regions,_ = analyze_regions(space,rows,("buy_step","sell_step","buy_shares"),global_exhaustive=True)
        contexts.append({"as_of":"2026-10-19","recommendation_regions":[_public_region(r) for r in regions],
                         "diagnostics":{"strategy_version":2,"geometry":"connected_lattice","status":"completed",
                            "validation_account_mode":"fresh_initial_config","validation_start":"2025-09-26",
                            "validation_end":"2026-10-19","global_exhaustive":True}})
        archive = list(rows.values())
        for r in archive:
            r["validation_return_original"] = r["validation_excess_original"]+.03
            r["validation_excess_stress"] = r["validation_excess_original"]-.01
            r["folds"] = [{"excess_return":r["validation_excess_original"]/2,"grid_return":.02}]
            r["stress_folds"] = [{"excess_return":r["validation_excess_original"]/4,"grid_return":.01}]
        archives.append(archive)
    current,fixed = contexts
    result = {"protocol_version":4,"config":config.model_dump(mode="json"),
              "fixed":{**fixed,"current":current},
              "evidence":{"decisions":[fixed],"candidate_archives":{"current":archives[0],"fixed":archives[1],
                         "adaptive":[{"candidate_archive":archives[1]}]}}}
    store = Store(tmp_path)
    result_id = store.save_result("optimization",result,"600000.SH","connected regions")
    with TestClient(create_app(str(tmp_path),start_jobs=False)) as client:
        yield client,result_id,result,archives


def test_compact_summary_archive_hydration_and_exact_members_paging(region_result):
    client,result_id,result,archives = region_result
    compact = client.get(f"/api/v1/results/{result_id}").json()
    assert "candidate_archives" not in compact["evidence"]
    region = compact["fixed"]["current"]["recommendation_regions"][0]
    assert region["combination_count"] == 72 and "region_points" not in region
    seen = []
    for offset in range(0,72,11):
        response = client.get(f"/api/v1/results/{result_id}/landscape",params={"view":"members","region_id":1,"offset":offset,"limit":11})
        assert response.status_code == 200
        payload = response.json()
        assert not payload["sampled"] and payload["total"] == 72
        seen.extend(p["candidate_id"] for p in payload["points"])
    assert seen == sorted({r["candidate_id"] for r in archives[0] if r.get("region_ids")})


def test_projection_aggregates_every_member_and_slice_preserves_hole(region_result):
    client,result_id,_,_ = region_result
    base = f"/api/v1/results/{result_id}/landscape"
    query = {"region_id":1,"x":"buy_step","y":"sell_step"}
    projection = client.get(base,params={**query,"view":"projection","limit":100}).json()
    assert projection["view_total"] == 24
    assert sum(c["count"] for c in projection["projection"]) == 72
    assert all(c["count"] == 3 for c in projection["projection"])
    assert not any(c["x"] == .0003 and c["y"] == .0003 for c in projection["projection"])
    stable = client.get(base,params={**query,"view":"slice","anchor":"stable"}).json()
    peak = client.get(base,params={**query,"view":"slice","anchor":"peak"}).json()
    assert stable["view_total"] == peak["view_total"] == 24
    assert stable["fixed_params"]["buy_shares"] == 200
    assert peak["fixed_params"]["buy_shares"] == 300
    assert all(p["params"]["buy_shares"] == 200 for p in stable["points"])
    assert stable["stable_point"]["candidate_id"] != stable["peak_point"]["candidate_id"]


def test_representatives_do_not_change_when_cost_fold_or_decision_changes(region_result):
    client,result_id,_,_ = region_result
    base = f"/api/v1/results/{result_id}/landscape"
    original = client.get(base,params={"region_id":1,"view":"members"}).json()
    stress = client.get(base,params={"region_id":1,"view":"members","cost":"stress","fold":0}).json()
    for role in ("stable_point","peak_point"):
        assert original[role]["params"] == stress[role]["params"]
        assert stress[role]["excess_return"] == pytest.approx(original[role]["excess_return"]/4)
    fixed = client.get(base,params={"decision":"fixed","region_id":1}).json()
    adaptive = client.get(base,params={"decision":"adaptive","decision_index":0,"region_id":1}).json()
    assert fixed["peak_point"]["params"] == adaptive["peak_point"]["params"]
    assert fixed["peak_point"]["params"] != original["peak_point"]["params"]


def test_complete_region_exports_and_bad_requests(region_result):
    client,result_id,_,_ = region_result
    base = f"/api/v1/results/{result_id}/regions/1/export"
    exported = client.get(base,params={"format":"json"}).json()
    assert len(exported["members"]) == exported["region"]["combination_count"] == 72
    assert sum(bool(p["is_stable_point"]) for p in exported["members"]) == 1
    assert sum(bool(p["is_peak_point"]) for p in exported["members"]) == 1
    csv_result = client.get(base,params={"format":"csv"})
    records = list(csv.DictReader(io.StringIO(csv_result.text.lstrip('\ufeff'))))
    assert len(records) == 72
    assert all(float(p["excess_return"]) > 0 for p in records)
    assert client.get(f"/api/v1/results/{result_id}/landscape",params={"region_id":999}).status_code == 404
    assert client.get(f"/api/v1/results/{result_id}/landscape",params={"region_id":1,"x":"cash"}).status_code == 422
    assert client.get(base,params={"decision":"adaptive","decision_index":9}).status_code == 404
