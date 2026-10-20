from datetime import date
from itertools import product
import json
import math

import numpy as np
import pandas as pd
import pytest

from stockregression.models import BacktestConfig, DataBundle, OptimizationConfig, ResearchRequestV4, ResearchStrategySpec
from stockregression.jobs import _research_protocol_hash
from stockregression.storage import Store
from stockregression.research_v4.neighborhoods import (
    analyze_platforms,
    find_regions,
    maximum_positive_box,
)
from stockregression.research_v4.search import axis_sweep_coordinates, box_coordinates, build_validation_plan, refine_coordinates
from stockregression.research_v4.selector import select_at
from stockregression.research_v4.space import ParameterSpace, value_to_ticks


def config(**updates):
    base = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date=date(2024, 1, 2),
        frequency="1d", initial_shares=1000, min_shares=0, max_shares=2000,
        buy_shares=100, sell_shares=100, grid_mode="geometric",
        buy_step=.03, sell_step=.03,
    )
    ranges = {
        "buy_step": {"low": 0.0001, "high": 0.0005},
        "sell_step": {"low": 0.0003, "high": 0.0003},
        "sell_shares": {"low": 100, "high": 300},
        "buy_shares": {"low": 100, "high": 100},
        "min_shares": {"low": 0, "high": 0},
        "max_shares": {"low": 2000, "high": 2000},
    }
    values = {
        "base": base.model_dump(mode="python"), "protocol_version": 4,
        "ranges": ranges, "strategy": {
            "strategy_version": 1,
            "search_method": "exhaustive", "ranking_mode": "max_volume",
            "training_budget": 100, "validation_budget": 100,
        },
    }
    values.update(updates)
    return OptimizationConfig.model_validate(values)


def test_v4_fixed_units_reject_off_lattice_values():
    assert value_to_ticks("buy_step", 0.0001) == 1
    assert value_to_ticks("buy_step", 0.0011) == 11
    assert value_to_ticks("sell_step", .0001) == 1
    assert value_to_ticks("buy_step", .001) == 10
    assert value_to_ticks("buy_step", 1) == 10000
    assert ResearchStrategySpec(fixed_half_widths={"buy_step": .0001, "sell_step": .0001})
    assert value_to_ticks("buy_shares", 300) == 3
    with pytest.raises(ValueError, match="0.01"):
        value_to_ticks("buy_step", 0.00015)
    with pytest.raises(ValueError, match="100"):
        value_to_ticks("buy_shares", 150)


def test_v4_neighborhood_fields_must_be_a_nonempty_unique_subset():
    assert ResearchStrategySpec().neighborhood_fields == ["buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares"]
    with pytest.raises(ValueError, match="至少需要一个"):
        ResearchStrategySpec(neighborhood_fields=[])
    with pytest.raises(ValueError, match="不能重复"):
        ResearchStrategySpec(neighborhood_fields=["buy_step", "buy_step"])
    with pytest.raises(ValueError):
        ResearchStrategySpec(neighborhood_fields=["not_a_field"])


def test_v4_holding_bounds_use_500_share_ticks_and_clip_to_initial_position():
    source = config().model_dump(mode="python")
    source["base"]["initial_shares"] = 1200
    source["research_context"]["initial_shares"] = 1200
    source["ranges"]["min_shares"] = {"low": 0, "high": 2000}
    source["ranges"]["max_shares"] = {"low": 500, "high": 2500}
    space = ParameterSpace(OptimizationConfig.model_validate(source))
    points = [space.params(point) for point in space.iterate()]
    assert {point["min_shares"] for point in points} == {0, 500, 1000}
    assert {point["max_shares"] for point in points} == {1500, 2000, 2500}
    assert space.size() == len(points)
    for point in points:
        assert space.params(space.coordinate(point)) == point
    assert value_to_ticks("min_shares", 1500) == 3
    with pytest.raises(ValueError, match="500"):
        value_to_ticks("max_shares", 1200)


def test_v4_zero_slippage_and_unlimited_holding_are_research_settings():
    value = config(strategy={
        "search_method": "auto", "ranking_mode": "max_volume",
        "max_shares_mode": "unlimited", "slippage_mode": "zero",
        "training_budget": 100, "validation_budget": 100,
    })
    assert value.base.max_shares is None
    assert value.base.fees.slippage_bps == 0
    space = ParameterSpace(value)
    assert "max_shares" not in space.active
    assert space.params(list(space.iterate())[0])["max_shares"] is None


def test_v4_public_request_builds_independent_geometric_context():
    source = config()
    request = ResearchRequestV4.model_validate({
        "research_context": {
            "symbol": "159611.SZ", "start_date": "2025-01-02", "frequency": "1d",
            "initial_shares": 1000, "grid_mode": "geometric",
        },
        "ranges": source.model_dump(mode="python")["ranges"],
        "strategy": source.strategy.model_dump(mode="python"),
    })
    optimized = request.to_optimization()
    assert optimized.protocol_version == 4
    assert optimized.base.grid_mode == "geometric"
    assert optimized.base.initial_shares == 1000


def test_v4_candidate_archives_are_split_and_hydrated(tmp_path):
    store = Store(tmp_path)
    result_id = store.save_result("optimization", {
        "protocol_version": 4,
        "evidence": {"candidate_archives": {"fixed": [{"candidate_id": 1}]}},
    }, "159611.SZ", "v4 artifact test")
    main = json.loads((tmp_path / "results" / f"{result_id}.json").read_text(encoding="utf-8"))
    assert "__archive__" in main["evidence"]["candidate_archives"]["fixed"]
    hydrated = store.result(result_id)
    assert hydrated["evidence"]["candidate_archives"]["fixed"] == [{"candidate_id": 1}]


def test_v4_source_grid_and_search_values_do_not_define_research_space():
    first = config()
    source = first.base.model_copy(update={
        "grid_mode": "trailing", "buy_step": .08, "sell_step": .09,
        "buy_shares": 900, "sell_shares": 700, "min_shares": 100, "max_shares": None,
    })
    second = OptimizationConfig.model_validate({**first.model_dump(mode="python"), "base": source.model_dump(mode="python")})
    assert second.base.grid_mode == "geometric"
    assert ParameterSpace(first).active == ParameterSpace(second).active
    assert list(ParameterSpace(first).iterate()) == list(ParameterSpace(second).iterate())
    assert _research_protocol_hash(first) == _research_protocol_hash(second)
    shifted = OptimizationConfig.model_validate({
        **first.model_dump(mode="python"),
        "base": first.base.model_copy(update={"start_date": date(2024, 2, 1), "end_date": date(2025, 2, 1)}).model_dump(mode="python"),
        "research_context": None,
    })
    assert _research_protocol_hash(first) == _research_protocol_hash(shifted)


def _space():
    return ParameterSpace(config())


def _rows(space, value):
    rows = {}
    for index, coordinate in enumerate(space.iterate(), start=1):
        item = value(coordinate)
        rows[coordinate] = {
            "key": coordinate, "params": space.params(coordinate),
            "validation_excess_original": item, "evaluation_status": "completed",
            "candidate_id": index, "max_drawdown": .1, "trade_count": 20,
            "buy_count": 10, "sell_count": 10,
        }
    return rows


def test_v4_local_peak_uses_joint_neighbors_and_closes_plateaus():
    space = _space()
    values = {}
    for item in space.iterate():
        by_name = dict(zip(space.fields, item))
        g, s = by_name["buy_step"], by_name["sell_shares"]
        values[item] = .20 if (g, s) in {(2, 2), (3, 2)} else .01
    rows = _rows(space, lambda item: values[item])
    platforms = analyze_platforms(space, rows)
    peaks = [item for item in platforms if item.local_peak]
    assert len(peaks) == 1
    assert peaks[0].value == pytest.approx(.20)
    assert len(peaks[0].members) == 2


def test_v4_maximum_box_matches_complete_radius_enumeration():
    space = _space()
    center = space.full_coordinate((3, 2))
    by_name = dict(zip(space.fields, center))
    # An asymmetric positive rectangle. Greedy one-axis expansion can miss it.
    positive = {}
    for item in space.iterate():
        values = dict(zip(space.fields, item))
        inside = (
            abs(values["buy_step"] - by_name["buy_step"]) <= 1
            and abs(values["sell_shares"] - by_name["sell_shares"]) <= 1
        )
        positive[item] = .1 if inside else -.01
    found = maximum_positive_box(space, center, positive)
    assert found is not None
    radii, points = found
    brute = []
    max_r = {
        "buy_step": min(by_name["buy_step"] - 1, 5 - by_name["buy_step"]),
        "sell_shares": min(by_name["sell_shares"] - 1, 3 - by_name["sell_shares"]),
    }
    for rg in range(1, max_r["buy_step"] + 1):
        for rs in range(1, max_r["sell_shares"] + 1):
            box = box_coordinates(space, center, {"buy_step": rg, "sell_shares": rs})
            if box and all(positive.get(point, -math.inf) > 1e-12 for point in box):
                brute.append((rg * rs, rg, rs, box))
    best = max(brute)
    assert radii == {"buy_step": best[1], "sell_shares": best[2]}
    assert len(points) == len(best[3])


def test_v4_minimum_box_rejects_deep_joint_negative_points():
    space = _space()
    center = space.full_coordinate((3, 2))
    values = {}
    for item in space.iterate():
        by_name = dict(zip(space.fields, item))
        bad = (
            by_name["buy_step"] == 4 and by_name["sell_shares"] == 3
        )
        values[item] = -.001 if bad else .1
    assert maximum_positive_box(space, center, values) is None


def test_v4_three_dimensional_internal_negative_point_breaks_the_box():
    ranges = {
        "buy_step": {"low": 0.0001, "high": 0.0005},
        "sell_step": {"low": 0.0003, "high": 0.0003},
        "sell_shares": {"low": 100, "high": 300},
        "buy_shares": {"low": 100, "high": 300},
        "min_shares": {"low": 0, "high": 0},
        "max_shares": {"low": 2000, "high": 2000},
    }
    space = ParameterSpace(config(ranges=ranges))
    center = space.full_coordinate((3, 2, 2))
    values = {}
    for item in space.iterate():
        values[item] = -.001 if item == space.full_coordinate((4, 3, 3)) else .1
    assert maximum_positive_box(space, center, values) is None


def test_v4_fixed_dimensions_do_not_inflate_volume():
    space = _space()
    center = space.full_coordinate((3, 2))
    radii = {"buy_step": 1, "sell_shares": 1}
    assert 0 < space.normalized_volume(radii) <= 1
    assert len(box_coordinates(space, center, radii) or []) == 9


def test_v4_neighborhood_fields_freeze_unchecked_dimensions_in_fine_search():
    space = _space()
    center = space.full_coordinate((3, 2))
    radii = {"buy_step": 1}
    points = box_coordinates(space, center, radii, movable_fields={"buy_step"}) or []
    assert len(points) == 3
    assert {point[space.fields.index("sell_shares")] for point in points} == {center[space.fields.index("sell_shares")]}

    values = {}
    center_sell = center[space.fields.index("sell_shares")]
    center_grid = center[space.fields.index("buy_step")]
    for point in space.iterate():
        by_name = dict(zip(space.fields, point))
        values[point] = (
            .1 if by_name["sell_shares"] == center_sell and abs(by_name["buy_step"] - center_grid) <= 1
            else -.01
        )
    found = maximum_positive_box(space, center, values, movable_fields={"buy_step"})
    assert found is not None
    found_radii, found_points = found
    assert found_radii == {"buy_step": 1}
    assert len(found_points) == 3


def test_v4_coarse_sweep_probes_every_tick_of_a_frozen_fine_search_field():
    space = _space()
    sweep = axis_sweep_coordinates(space, {"sell_shares"})
    sell_index = space.fields.index("sell_shares")
    assert {point[sell_index] for point in sweep} == set(space.coords("sell_shares"))

    rows = [{"key": space.full_coordinate((3, 2)), "training_score": 1.0, "evaluation_status": "completed"}]
    refined = refine_coordinates(
        space, rows, existing={tuple(rows[0]["key"])}, limit=20, recommendation_count=1,
        movable_fields={"buy_step"},
    )
    assert refined
    assert {point[sell_index] for point in refined} == {rows[0]["key"][sell_index]}


def test_v4_validation_plan_holds_unchecked_fields_at_the_center():
    space = _space()
    value = config(strategy={
        "search_method": "hierarchical", "ranking_mode": "max_volume",
        "training_budget": 100, "validation_budget": 100, "recommendation_count": 1,
        "neighborhood_fields": ["buy_step"],
    })
    center = space.full_coordinate((3, 2))
    plan = build_validation_plan(
        space, value,
        [{"key": center, "training_score": 1.0, "evaluation_status": "completed"}],
        asof="2024-01-01", exhaustive=False,
    )
    assert len(plan.coordinates) >= 3
    sell_index = space.fields.index("sell_shares")
    assert {point[sell_index] for point in plan.coordinates} == {center[sell_index]}


def test_v4_qualified_region_requires_all_points_complete_and_positive():
    space = _space()
    center = space.full_coordinate((3, 2))

    def value(item):
        values = dict(zip(space.fields, item))
        if 2 <= values["buy_step"] <= 4 and 1 <= values["sell_shares"] <= 3:
            return .10 - .01 * abs(values["buy_step"] - 3) - .01 * abs(values["sell_shares"] - 2)
        return -.02

    rows = _rows(space, value)
    regions, platforms = find_regions(
        space, rows, ranking_mode="max_volume", fixed_half_widths={"buy_step": 0.0001, "sell_shares": 100},
        max_drawdown=.2, min_trades=10, global_exhaustive=True,
    )
    assert len(regions) == 1
    assert tuple(regions[0]["center_coordinate"]) == center
    assert regions[0]["radii"] == {"buy_step": 1, "sell_shares": 1}
    assert regions[0]["combination_count"] == 9
    assert regions[0]["min_excess_return"] > 1e-12
    assert regions[0]["complete"] is True
    assert regions[0]["global_exhaustive"] is True


def test_v4_max_volume_and_fixed_mean_choose_different_peaks():
    ranges = {
        "buy_step": {"low": 0.0001, "high": 0.0009},
        "sell_step": {"low": 0.0003, "high": 0.0003},
        "sell_shares": {"low": 100, "high": 500},
        "buy_shares": {"low": 100, "high": 100},
        "min_shares": {"low": 0, "high": 0},
        "max_shares": {"low": 2000, "high": 2000},
    }
    space = ParameterSpace(config(ranges=ranges))

    def value(item):
        values = dict(zip(space.fields, item))
        g, s = values["buy_step"], values["sell_shares"]
        if 1 <= g <= 3 and 1 <= s <= 3:
            return .07
        if 5 <= g <= 9 and 1 <= s <= 5:
            return .06
        return -.02

    rows = _rows(space, value)
    volume, _ = find_regions(
        space, rows, ranking_mode="max_volume", fixed_half_widths={"buy_step": 0.0001, "sell_shares": 100},
        max_drawdown=.2, min_trades=10, global_exhaustive=True,
    )
    fixed, _ = find_regions(
        space, rows, ranking_mode="fixed_mean", fixed_half_widths={"buy_step": 0.0001, "sell_shares": 100},
        max_drawdown=.2, min_trades=10, global_exhaustive=True,
    )
    volume_center = dict(zip(space.fields, volume[0]["center_coordinate"]))
    fixed_center = dict(zip(space.fields, fixed[0]["center_coordinate"]))
    assert 5 <= volume_center["buy_step"] <= 9
    assert volume[0]["raw_volume"] == 4
    assert 1 <= fixed_center["buy_step"] <= 3
    assert fixed[0]["center_excess_return"] == pytest.approx(.07)


def test_v4_all_dimensions_fixed_is_a_single_diagnostic_point():
    ranges = {
        "buy_step": {"low": 0.003, "high": 0.003},
        "sell_step": {"low": 0.003, "high": 0.003},
        "sell_shares": {"low": 100, "high": 100},
        "buy_shares": {"low": 100, "high": 100},
        "min_shares": {"low": 0, "high": 0},
        "max_shares": {"low": 2000, "high": 2000},
    }
    space = ParameterSpace(config(ranges=ranges))
    assert space.active == ()
    assert space.size() == 1
    assert len(list(space.iterate())) == 1


def test_v4_selection_exports_reproducible_validation_context():
    from stockregression.share_engine import run_share_backtest

    request = config()
    dates = pd.bdate_range("2024-01-01", periods=169)
    prices = 10 + np.sin(np.arange(len(dates)) / 2) * .8
    daily = pd.DataFrame({
        "date": dates, "open": prices, "close": prices,
        "high": prices, "low": prices, "volume": 100000,
        "preclose": np.r_[prices[0], prices[:-1]], "tradestatus": 1,
    })
    bars = daily.iloc[1:].copy()
    bars["timestamp"] = bars.date + pd.Timedelta(hours=15)
    bars["frequency"], bars["tradable"] = "1d", True
    bundle = DataBundle(symbol=request.base.symbol, daily=daily,
        bars=bars, calendar=dates[1:], quality={"minute_ratio": 1.0})
    selected = select_at(request, request.base, bundle, dates[-1])
    diagnostics = selected["diagnostics"]
    assert diagnostics["validation_account_mode"] == "fresh_initial_config"
    assert diagnostics["validation_return_basis"] == "peak_invested_roi"
    assert diagnostics["validation_start"] == str(dates[85].date())
    assert diagnostics["validation_end"] == str(dates[-1].date())
    assert len(diagnostics["validation_folds"]) == 4
    assert len(selected["candidate_archive"]) == ParameterSpace(request).size()
    for candidate in selected["candidate_archive"]:
        replay = request.base.model_copy(update={
            **candidate["params"], "start_date": dates[85].date(),
            "end_date": dates[-1].date(),
        })
        result = run_share_backtest(replay, bundle)
        grid_return = result["strategies"]["grid"]["metrics"]["total_return"]
        hold_return = result["strategies"]["buy_hold"]["metrics"]["total_return"]
        assert candidate["validation_excess_original"] == pytest.approx(grid_return - hold_return)
