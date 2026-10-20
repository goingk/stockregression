"""Deterministic full and hierarchical training searches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from itertools import product
from typing import Any, Iterable, Iterator

import numpy as np

from ..models import OptimizationConfig
from .space import FIELDS, UNITS, ParameterSpace, value_to_ticks


def _stable_seed(config: OptimizationConfig, asof: str) -> int:
    payload = {
        "seed": int(config.seed), "asof": asof,
        "strategy": config.strategy.model_dump(mode="json"),
        "ranges": {key: value.model_dump(mode="json") for key, value in config.ranges.items() if key in FIELDS},
        "initial_shares": config.base.initial_shares,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _dedupe(coordinates: Iterable[tuple[int, ...]], limit: int) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    output = []
    for coordinate in coordinates:
        if coordinate in seen:
            continue
        seen.add(coordinate)
        output.append(coordinate)
        if len(output) >= limit:
            break
    return output


def coarse_coordinates(space: ParameterSpace, limit: int) -> list[tuple[int, ...]]:
    if limit <= 0:
        return []
    if not space.active:
        return [space.full_coordinate(())]
    levels = {name: sorted({space.dimensions[name].low, space.dimensions[name].high}) for name in space.active}

    def count() -> int:
        return math.prod(len(levels[name]) for name in space.active)

    while count() < limit:
        choices = []
        for name in space.active:
            values = levels[name]
            gaps = [(values[index + 1] - values[index], name, index) for index in range(len(values) - 1)]
            if gaps:
                gap, _, index = max(gaps)
                mid = (values[index] + values[index + 1]) // 2
                if mid not in (values[index], values[index + 1]):
                    choices.append((gap / max(1, space.dimensions[name].width), name, mid))
        if not choices:
            break
        _, name, mid = max(choices)
        levels[name].append(mid)
        levels[name].sort()
        if count() > limit:
            levels[name].remove(mid)
            break
    candidates = []
    for active in product(*(levels[name] for name in space.active)):
        if space.legal(active):
            candidates.append(space.full_coordinate(active))
    return sorted(set(candidates))[:limit]


def stratified_coordinates(space: ParameterSpace, limit: int, seed: int) -> list[tuple[int, ...]]:
    if limit <= 0 or not space.active:
        return []
    rng = np.random.default_rng(seed)
    dimensions = [space.dimensions[name] for name in space.active]
    strata = (limit + 1) ** (1.0 / len(dimensions))
    binned = max(2, int(math.ceil(strata)))
    permutations = np.asarray([rng.permutation(binned) for _ in dimensions])
    output = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(output) < limit and attempts < limit * 20 + 100:
        row = attempts % binned
        attempts += 1
        values = []
        for dim_index, dim in enumerate(dimensions):
            start = math.floor(permutations[dim_index, row] * dim.count / binned)
            end = math.floor((permutations[dim_index, row] + 1) * dim.count / binned)
            end = max(start + 1, end)
            jitter = int(rng.integers(0, max(1, end - start)))
            values.append(min(dim.high, max(dim.low, dim.low + start + jitter)))
        if not space.legal(values):
            continue
        coordinate = space.full_coordinate(values)
        if coordinate not in seen:
            seen.add(coordinate)
            output.append(coordinate)
    return output


def statistical_coordinates(space: ParameterSpace, profile: dict[str, Any]) -> list[tuple[int, ...]]:
    quantiles = profile.get("quantiles") if isinstance(profile.get("quantiles"), dict) else {}
    ranges = quantiles.get("range") if isinstance(quantiles, dict) else None
    if not isinstance(ranges, list):
        return []
    output: list[tuple[int, ...]] = []
    for raw in ranges[:3]:
        values = []
        for name in space.active:
            dim = space.dimensions[name]
            if name in {"buy_step", "sell_step"}:
                try:
                    tick = value_to_ticks(name, float(raw))
                except (TypeError, ValueError):
                    tick = dim.low
                values.append(min(dim.high, max(dim.low, tick)))
            else:
                values.append((dim.low + dim.high) // 2)
        if space.legal(values):
            coordinate = space.full_coordinate(values)
            if coordinate not in output:
                output.append(coordinate)
    return output


def axis_sweep_coordinates(space: ParameterSpace, fields: Iterable[str]) -> list[tuple[int, ...]]:
    """Probe every lattice value of fields that are fixed during fine search."""
    wanted = [name for name in space.active if name in set(fields)]
    if not wanted or not space.active:
        return []
    index_of = {name: index for index, name in enumerate(space.active)}
    anchor = [dim.low + dim.width // 2 for dim in (space.dimensions[name] for name in space.active)]
    if not space.legal(anchor):
        fallback = coarse_coordinates(space, 1000)
        if not fallback:
            return []
        anchor = list(space.active_coords(fallback[0]))
    output = []
    for name in wanted:
        position = index_of[name]
        for tick in space.coords(name):
            values = list(anchor)
            values[position] = tick
            if space.legal(values):
                output.append(space.full_coordinate(values))
    return list(dict.fromkeys(output))


def initial_training_coordinates(space: ParameterSpace, config: OptimizationConfig, asof: str) -> list[tuple[int, ...]]:
    budget = int(config.strategy.training_budget)
    total = space.size()
    if config.strategy.search_method == "exhaustive" or (
        config.strategy.search_method == "auto" and total <= budget
    ):
        return list(space.iterate(budget + 1)) if total <= budget + 1 else list(space.iterate(budget))
    seed = _stable_seed(config, asof)
    coarse_budget = max(1, budget // 4)
    sample_budget = max(1, budget // 4)
    coarse = coarse_coordinates(space, coarse_budget)
    sample = stratified_coordinates(space, sample_budget, seed)
    fixed_in_fine = [name for name in space.active if name not in set(config.strategy.neighborhood_fields)]
    sweep = axis_sweep_coordinates(space, fixed_in_fine)
    return _dedupe([*coarse, *sweep, *sample], budget)


def refine_coordinates(
    space: ParameterSpace,
    scored_rows: list[dict[str, Any]],
    *,
    existing: set[tuple[int, ...]],
    limit: int,
    recommendation_count: int,
    movable_fields: Iterable[str] | None = None,
) -> list[tuple[int, ...]]:
    if limit <= 0 or not space.active:
        return []
    completed = sorted(
        (row for row in scored_rows if row.get("evaluation_status") == "completed" and math.isfinite(float(row.get("training_score", math.nan)))),
        key=lambda row: (-float(row["training_score"]), tuple(row["key"])),
    )
    if not completed:
        return []
    region_limit = min(64, max(20, 3 * recommendation_count))
    regions: list[dict[str, Any]] = []
    for row in completed:
        coordinate = tuple(row["key"])
        if any(sum(abs(coordinate[index] - tuple(other["key"])[index]) for index in range(len(FIELDS))) <= 1 for other in regions):
            continue
        regions.append(row)
        if len(regions) >= region_limit:
            break
    output: list[tuple[int, ...]] = []
    used = set(existing)
    movable = set(space.active if movable_fields is None else movable_fields)
    movable = {name for name in space.active if name in movable}
    if not movable:
        return []
    max_width = max((space.dimensions[name].width for name in movable), default=0)
    stride = max(1, 1 << max(0, int(math.ceil(math.log2(max(1, max_width))))))
    while stride >= 1 and len(output) < limit:
        for region in regions:
            center = tuple(region["key"])
            axes = []
            for name in space.active:
                dim = space.dimensions[name]
                index = space.fields.index(name)
                if name not in movable:
                    axes.append((center[index],))
                    continue
                radius = max(1, stride) if dim.width >= 2 else 0
                axes.append(tuple(range(max(dim.low, center[index] - radius), min(dim.high, center[index] + radius) + 1)))
            for active in product(*axes):
                if not space.legal(active):
                    continue
                coordinate = space.full_coordinate(active)
                if coordinate in used:
                    continue
                used.add(coordinate)
                output.append(coordinate)
                if len(output) >= limit:
                    return output
        stride //= 2
    return output


@dataclass
class ValidationPlan:
    coordinates: list[tuple[int, ...]]
    centers: dict[tuple[int, ...], dict[str, Any]]
    plan_hash: str
    global_exhaustive: bool


def box_coordinates(
    space: ParameterSpace,
    center: tuple[int, ...],
    radii: dict[str, int],
    movable_fields: Iterable[str] | None = None,
) -> list[tuple[int, ...]] | None:
    axes: list[tuple[int, ...]] = []
    movable = set(space.active if movable_fields is None else movable_fields)
    for name in space.active:
        index = space._field_index[name]
        dim = space.dimensions[name]
        if name not in movable:
            axes.append((center[index],))
            continue
        radius = int(radii.get(name, 0))
        low, high = center[index] - radius, center[index] + radius
        if radius < 1 or low < dim.low or high > dim.high:
            return None
        axes.append(tuple(range(low, high + 1)))
    if not axes:
        return [center]
    output = []
    for active in product(*axes):
        if not space.legal(active):
            continue
        coordinate = space.full_coordinate(active)
        output.append(coordinate)
    return output


def build_validation_plan(
    space: ParameterSpace,
    config: OptimizationConfig,
    training_rows: list[dict[str, Any]],
    *,
    asof: str,
    exhaustive: bool,
) -> ValidationPlan:
    budget = int(config.strategy.validation_budget)
    if exhaustive:
        coordinates = list(space.iterate(budget + 1))
        if len(coordinates) > budget:
            raise ValueError(f"完整枚举需要 {space.size()} 个验证点，超过预算 {budget}")
        centers = {item: {"region_id": index + 1, "source_key": list(item)} for index, item in enumerate(coordinates)}
        plan_hash = hashlib.sha256(json.dumps([list(item) for item in coordinates]).encode()).hexdigest()
        return ValidationPlan(coordinates, centers, plan_hash, True)

    completed = sorted(
        (row for row in training_rows if row.get("evaluation_status") == "completed" and math.isfinite(float(row.get("training_score", math.nan)))),
        key=lambda row: (-float(row["training_score"]), tuple(row["key"])),
    )
    if not completed:
        return ValidationPlan([], {}, hashlib.sha256(b"empty").hexdigest(), False)
    region_limit = min(64, max(20, 3 * int(config.strategy.recommendation_count)))
    seeds: list[tuple[int, ...]] = []
    for row in completed:
        key = tuple(row["key"])
        if key not in seeds:
            seeds.append(key)
        if len(seeds) >= min(region_limit, len(completed)):
            break
    selected: dict[tuple[int, ...], int] = {}
    centers: dict[tuple[int, ...], dict[str, Any]] = {}
    movable = {name for name in space.active if name in set(config.strategy.neighborhood_fields)}
    base_radii = {
        name: (
            value_to_ticks(name, config.strategy.fixed_half_widths.get(name, UNITS[name]))
            if config.strategy.ranking_mode == "fixed_mean" else 1
        ) if name in movable else 0
        for name in space.active
    }
    for region_id, seed in enumerate(seeds, start=1):
        center = space.shift_inside(seed, movable)
        if not movable:
            continue
        points = box_coordinates(space, center, base_radii, movable_fields=movable)
        if points is None:
            continue
        if len(selected) + sum(point not in selected for point in points) > budget:
            continue
        centers[center] = {"region_id": region_id, "source_key": list(seed)}
        for point in points:
            selected.setdefault(point, region_id)

    # Expand complete shells in deterministic round-robin order. A partial shell
    # is never represented as a complete neighborhood.
    radii = {center: dict(base_radii) for center in centers}
    changed = True
    while changed and len(selected) < budget:
        changed = False
        for center in list(centers):
            for name in movable:
                candidate = dict(radii[center])
                candidate[name] += 1
                points = box_coordinates(space, center, candidate, movable_fields=movable)
                if points is None:
                    continue
                additions = [point for point in points if point not in selected]
                if additions and len(selected) + len(additions) <= budget:
                    for point in additions:
                        selected.setdefault(point, centers[center]["region_id"])
                    radii[center] = candidate
                    changed = True
                    if len(selected) >= budget:
                        break
            if len(selected) >= budget:
                break
    coordinates = sorted(selected)
    payload = {
        "asof": asof,
        "budget": budget,
        "method": config.strategy.search_method,
        "ranking": config.strategy.ranking_mode,
        "neighborhood_fields": sorted(movable),
        "coordinates": [list(item) for item in coordinates],
        "centers": [list(item) for item in centers],
        "radii": [[list(center), radii[center]] for center in centers],
    }
    plan_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return ValidationPlan(coordinates, centers, plan_hash, False)
