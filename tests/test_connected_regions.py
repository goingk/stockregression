from collections import Counter
from datetime import date
import math

import numpy as np
import pytest

from stockregression.models import BacktestConfig, OptimizationConfig, ResearchStrategySpec
from stockregression.research_v4.connected import analyze_regions, explore, refine_training, seed_coordinates
from stockregression.research_v4.space import ParameterSpace


def make_space(width=9, height=1, *, unlimited=False):
    base = BacktestConfig(schema_version=3, symbol="600000.SH", start_date=date(2024, 1, 2),
                          frequency="1d", initial_shares=1000, min_shares=0, max_shares=2000,
                          buy_shares=100, sell_shares=100, grid_mode="geometric", buy_step=.03, sell_step=.03)
    ranges = {"buy_step": {"low": .0001, "high": width*.0001},
              "sell_step": {"low": .0001, "high": height*.0001},
              "sell_shares": {"low": 100, "high": 100}, "buy_shares": {"low": 100, "high": 100},
              "min_shares": {"low": 0, "high": 0}, "max_shares": {"low": 2000, "high": 2000}}
    # Avoid arithmetic-created decimal strings off the exact lattice.
    ranges["buy_step"]["high"] = round(width*.0001, 4)
    ranges["sell_step"]["high"] = round(height*.0001, 4)
    cfg = OptimizationConfig(base=base, protocol_version=4, ranges=ranges,
        strategy={"strategy_version": 2, "search_method": "hierarchical", "training_budget": 10,
                  "validation_budget": 1, "max_shares_mode": "unlimited" if unlimited else "search"})
    return ParameterSpace(cfg)


def row(space, point, value, index=0):
    return {"key": point, "params": space.params(point), "candidate_id": index,
            "evaluation_status": "completed", "validation_excess_original": value,
            "max_drawdown": .99, "trade_count": 0, "buy_count": 0, "sell_count": 0,
            "constraint_eligible": False, "constraint_reasons": ["诊断约束未满足"]}


def full_rows(space, value):
    return {p: row(space, p, value(p), i) for i, p in enumerate(space.iterate(), 1)}


def test_monotone_returns_recommend_entire_domain_with_distinct_representatives():
    space = make_space()
    rows = full_rows(space, lambda p: p[0]/100)
    regions, missing = analyze_regions(space, rows, space.active, global_exhaustive=True)
    assert not missing and len(regions) == 1
    region = regions[0]
    assert region["combination_count"] == 9
    assert region["stable_point"]["coordinate"][0] == 5
    assert region["peak_point"]["coordinate"][0] == 9
    assert region["stable_point"]["stability_margin"] == .5
    assert region["constraint_failure_count"] == 9  # diagnostics do not remove members
    assert region["normalized_volume"] == 1
    assert region["touches_search_boundary"] is True


@pytest.mark.parametrize("shape", ["L", "hole", "channel", "islands", "diagonal"])
def test_arbitrary_shapes_match_reference_axis_components(shape):
    space = make_space(7, 7)
    masks = {
        "L": lambda x,y: x <= 2 or y <= 2,
        "hole": lambda x,y: (x,y) != (4,4),
        "channel": lambda x,y: y == 2 or x == 6,
        "islands": lambda x,y: x <= 2 or x >= 6,
        "diagonal": lambda x,y: (x,y) in {(2,2),(3,3)},
    }
    rows = full_rows(space, lambda p: .01*(p[0]+p[1]) if masks[shape](*p[:2]) else 0)
    regions, _ = analyze_regions(space, rows, space.active)
    positive = {p for p,r in rows.items() if r["validation_excess_original"] > 0}
    reference = []
    while positive:
        queue = [positive.pop()]
        component = set(queue)
        while queue:
            p = queue.pop()
            for axis in (0,1):
                for step in (-1,1):
                    other = list(p); other[axis] += step; other = tuple(other)
                    if other in positive:
                        positive.remove(other); component.add(other); queue.append(other)
        reference.append(component)
    assert {frozenset(r["region_points"]) for r in regions} == {frozenset(c) for c in reference}
    if shape == "diagonal":
        assert len(regions) == 2
    if shape == "hole":
        assert regions[0]["combination_count"] == 48


def test_seed_budget_is_exceeded_until_positive_component_is_closed_and_cached():
    space = make_space()
    counts = Counter()
    messages = []
    def evaluate(points):
        counts.update(points)
        return [row(space,p,p[0]/100) for p in points]
    points = list(space.iterate())
    result = explore(space, [points[0], points[5]], space.active, evaluate, chunk_size=2, progress=messages.append)
    assert result.status == "completed" and len(result.rows) == 9
    assert set(counts.values()) == {1}
    assert "已闭合 1 域" in messages[-1]


def test_undiscovered_island_does_not_become_global_exhaustive():
    space = make_space()
    def evaluate(points):
        return [row(space,p,.1 if p[0] in {1,2,8,9} else -.1) for p in points]
    result = explore(space, [next(space.iterate())], space.active, evaluate)
    regions, _ = analyze_regions(space,result.rows,space.active)
    assert len(regions) == 1 and regions[0]["combination_count"] == 2
    assert len(result.rows) == 3 and not regions[0]["global_exhaustive"]


def test_failed_boundary_retried_once_never_treated_as_negative():
    space = make_space()
    counts = Counter()
    def evaluate(points):
        counts.update(points)
        return [row(space,p,float("nan") if p[0] == 3 else .1) for p in points]
    result = explore(space,[next(space.iterate())],space.active,evaluate)
    regions, incomplete = analyze_regions(space,result.rows,space.active)
    assert result.status == "partial" and counts[next(p for p in counts if p[0] == 3)] == 2
    assert regions == [] and len(incomplete) == 1


def test_partial_exploration_keeps_other_fully_closed_region():
    space = make_space()
    def evaluate(points):
        return [row(space,p,.1 if p[0] in {1,2,8,9} else float('nan') if p[0] == 7 else -.1) for p in points]
    points = list(space.iterate())
    result = explore(space,[points[0],points[-1]],space.active,evaluate)
    regions,incomplete = analyze_regions(space,result.rows,space.active)
    assert result.status == 'partial'
    assert [r['combination_count'] for r in regions] == [2] and regions[0]['region_points'][0][0] == 1
    assert len(incomplete) == 1


def test_cancellation_returns_completed_rows_without_false_closure():
    space = make_space()
    calls = []
    def evaluate(points):
        calls.append(points)
        return [row(space,p,.1) for p in points]
    result = explore(space,[next(space.iterate())],space.active,evaluate,cancelled=lambda:bool(calls))
    assert result.status == "cancelled" and len(result.rows) == 1
    regions, incomplete = analyze_regions(space,result.rows,space.active)
    assert not regions and incomplete


def test_frozen_dimensions_partition_components_and_unlimited_coordinates_work():
    space = make_space(5, 3, unlimited=True)
    rows = full_rows(space,lambda p:.1)
    regions,_ = analyze_regions(space,rows,("buy_step",))
    assert len(regions) == 3 and all(r["combination_count"] == 5 for r in regions)
    assert all(r["normalized_volume"] == 1 for r in regions)
    assert all(r["stable_point"]["params"]["max_shares"] is None for r in regions)


def test_normalized_euclidean_margins_match_brute_force_and_ties_are_deterministic():
    space = make_space(7, 5)
    rows = full_rows(space,lambda p:.1 if p[:2] != (3,3) else -.1)
    regions,_ = analyze_regions(space,rows,space.active)
    points = regions[0]["region_points"]
    boundary = [p for p in points if p[0] in {1,7} or p[1] in {1,5} or abs(p[0]-3)+abs(p[1]-3)==1]
    for p in points:
        expected = min(math.hypot((p[0]-b[0])/6,(p[1]-b[1])/4) for b in boundary)
        assert rows[p]["stability_margin"] == pytest.approx(expected)
    reverse = dict(reversed(list(rows.items())))
    second,_ = analyze_regions(space,reverse,space.active)
    assert second[0]["stable_point"] == regions[0]["stable_point"]
    assert second[0]["peak_point"] == regions[0]["peak_point"]


def test_refinement_allocates_to_multiple_centers_and_starts_locally():
    space = make_space(101, 101)
    centers = [space.full_coordinate((80,80)),space.full_coordinate((20,20))]
    scored = [{"key":p,"training_score":2-i,"evaluation_status":"completed"} for i,p in enumerate(centers)]
    refined = refine_training(space,scored,existing=set(centers),limit=10,recommendation_count=2,movable_fields=space.active)
    assert len(refined) == len(set(refined)) == 10
    assert all(min(sum(abs(a-b) for a,b in zip(p,c)) for c in centers) <= 2 for p in refined)
    assert any(p[0] < 30 for p in refined) and any(p[0] > 70 for p in refined)


def test_strategy_version_defaults_and_legacy_fixed_mode():
    assert ResearchStrategySpec().strategy_version == 4
    assert ResearchStrategySpec().hypercube_edge == 3
    assert ResearchStrategySpec(ranking_mode="fixed_mean").strategy_version == 1
    assert ResearchStrategySpec(ranking_mode="max_volume").strategy_version == 2
    with pytest.raises(ValueError):
        ResearchStrategySpec(strategy_version=2,ranking_mode="fixed_mean")
    with pytest.raises(ValueError):
        ResearchStrategySpec(strategy_version=3, hypercube_edge=2)


def test_protocol_hash_distinguishes_connected_geometry_and_legacy_box():
    from stockregression.jobs import _research_protocol_hash
    connected = make_space().config
    legacy = connected.model_copy(update={"strategy": connected.strategy.model_copy(update={"strategy_version": 1})})
    assert _research_protocol_hash(connected) != _research_protocol_hash(legacy)


def test_seed_plan_is_reproducible_and_all_fixed_domain_is_singleton():
    space = make_space()
    assert seed_coordinates(space,space.config,[],"2026-10-20") == seed_coordinates(space,space.config,[],"2026-10-20")
    fixed = make_space(1,1)
    rows = full_rows(fixed,lambda p:.1)
    regions,_ = analyze_regions(fixed,rows,())
    assert regions[0]["combination_count"] == 1
    assert regions[0]["stable_point"] == regions[0]["peak_point"]


@pytest.mark.parametrize("failure", [False, True])
def test_selector_integrates_complete_expansion_and_skips_adaptive_family_bootstrap(monkeypatch, failure):
    import pandas as pd
    from stockregression import scientific
    from stockregression.models import DataBundle
    from stockregression.research_v4 import connected_selector
    from stockregression.research_v4.selector import select_at

    space = make_space()
    dates = pd.bdate_range('2024-01-01',periods=504)
    bundle = DataBundle(symbol=space.base.symbol,bars=pd.DataFrame(),daily=pd.DataFrame(),calendar=dates)
    evaluations = Counter()
    instances = []
    class FakeEvaluator:
        def __init__(self,*args,**kwargs):
            self.chunk_size = 2
            instances.append(self)
        def evaluate_training(self,points):
            return [{"key":p,"params":space.params(p),"evaluation_status":"completed","training_score":p[0]} for p in points]
        def evaluate_validation(self,points):
            evaluations.update(points)
            return [row(space,p,float('nan') if failure and p[0] == 3 else p[0]/100) for p in points]
        def evaluate_stress(self,points):
            return {p:{"validation_excess_stress":p[0]/100-.01,"validation_return_stress":p[0]/100} for p in points}
    monkeypatch.setattr(connected_selector,'BatchEvaluator',FakeEvaluator)
    monkeypatch.setattr(scientific,'_statistical_profile',lambda *a:{})
    result = select_at(space.config,space.base,bundle,dates[-1])
    assert not instances[0].retain_daily_returns
    assert result['diagnostics']['strategy_version'] == 2
    assert result['diagnostics']['seed_count'] == 1
    assert 'validation_plan_hash' not in result['diagnostics']
    assert result['diagnostics']['evaluation_set_hash']
    bootstrap = scientific.evaluate_current_family(space.config,bundle,result,result['candidate_archive'],None)
    assert bootstrap['status'] == 'not_applicable' and 'bootstrap' not in bootstrap
    if failure:
        assert result['status'] == 'partial' and not result['eligible']
        assert result['recommendation_regions'] == []
        assert max(evaluations.values()) == 2
    else:
        assert result['status'] == 'completed' and result['eligible']
        assert result['diagnostics']['budget_overrun_count'] == 8
        region = result['recommendation_regions'][0]
        assert result['recommended_params'] == region['stable_point']['params']
        assert result['recommended_params']['buy_step'] == .0005
        assert region['peak_point']['params']['buy_step'] == pytest.approx(.0009)
        assert not region['stable_point']['constraint_eligible']
        assert max(evaluations.values()) == 1


def test_minute_selection_and_both_representatives_match_standalone_ledgers():
    import pandas as pd
    from stockregression.models import DataBundle, FeeConfig, SearchRange
    from stockregression.research_v4.selector import select_at
    from stockregression.share_engine import run_share_backtest

    space = make_space(3,3)
    base = space.base.model_copy(update={"frequency":"1m", "fees":FeeConfig(commission_rate=0,min_commission=0,slippage_bps=0)})
    ranges = {**space.config.ranges,"buy_step":SearchRange(low=.01,high=.0102),
              "sell_step":SearchRange(low=.01,high=.0102)}
    config = space.config.model_copy(update={"base":base,"ranges":ranges})
    dates = pd.bdate_range('2024-01-01',periods=505)
    daily = pd.DataFrame({'date':dates,'open':10.,'high':10.3,'low':9.7,'close':10.,
                          'volume':1000000.,'preclose':10.,'tradestatus':1})
    bars = pd.concat([daily.iloc[1:].assign(timestamp=dates[1:]+offset,open=opening,close=price)
                      for offset,opening,price in ((pd.Timedelta(hours=9,minutes=31),10.,9.7),
                                                  (pd.Timedelta(hours=9,minutes=32),9.7,10.3),
                                                  (pd.Timedelta(hours=15),10.3,10.))],ignore_index=True)
    bars['frequency'],bars['tradable'] = '1m',True
    bars = bars.sort_values('timestamp').reset_index(drop=True)
    bundle = DataBundle(symbol=base.symbol,daily=daily,bars=bars,calendar=dates[1:],quality={'minute_ratio':1.})
    selection = select_at(config,base,bundle,dates[-1])
    assert selection['status'] == 'completed' and selection['recommendation_regions']
    d = selection['diagnostics']
    assert d['validation_unique_count'] > config.strategy.validation_budget
    for region in selection['recommendation_regions']:
        for role in ('stable_point','peak_point'):
            point = region[role]
            replay_base = base.model_copy(update={**point['params'],
                'start_date':pd.Timestamp(d['validation_start']).date(),'end_date':pd.Timestamp(d['validation_end']).date()})
            result = run_share_backtest(replay_base,bundle)
            actual = result['strategies']['grid']['metrics']['total_return']-result['strategies']['buy_hold']['metrics']['total_return']
            assert actual == pytest.approx(point['excess_return'],abs=1e-12)
