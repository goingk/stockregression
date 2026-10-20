"""Peak, plateau and complete positive-neighborhood analysis."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Callable, Iterable

import numpy as np

from ..performance import record_count, record_details

try:
    from numba import njit
except ImportError:  # pragma: no cover - Numba is a runtime dependency
    njit = None

from .space import UNITS, ParameterSpace, value_to_ticks
from .search import box_coordinates


TOLERANCE = 1e-12


if njit is None:
    def _find_root(parent, index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _component_labels(count, left, right, buckets):
        parent = list(range(count))
        for a, b, bucket in zip(left, right, buckets):
            if buckets[a] != bucket or buckets[b] != bucket:
                continue
            ra, rb = _find_root(parent, int(a)), _find_root(parent, int(b))
            if ra != rb:
                parent[rb] = ra
        return np.asarray([_find_root(parent, index) for index in range(count)], dtype=np.int64)
else:
    @njit(cache=True)
    def _find_root(parent, index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    @njit(cache=True)
    def _component_labels(count, left, right, buckets):
        parent = np.arange(count, dtype=np.int64)
        for edge in range(left.shape[0]):
            a, b = left[edge], right[edge]
            if buckets[a] != buckets[b]:
                continue
            ra, rb = _find_root(parent, a), _find_root(parent, b)
            if ra != rb:
                parent[rb] = ra
        labels = np.empty(count, dtype=np.int64)
        for index in range(count):
            labels[index] = _find_root(parent, index)
        return labels


@dataclass
class Platform:
    platform_id: int
    members: list[tuple[int, ...]]
    value: float
    complete: bool
    local_peak: bool
    reasons: list[str]


def _quantized(value: float) -> int:
    return int(round(float(value) / TOLERANCE))


def analyze_platforms(
    space: ParameterSpace,
    rows: dict[tuple[int, ...], dict[str, Any]],
    neighborhood_fields: Iterable[str] | None = None,
) -> list[Platform]:
    values = {
        key: float(row["validation_excess_original"])
        for key, row in rows.items()
        if row.get("evaluation_status") == "completed"
        and isinstance(row.get("validation_excess_original"), (int, float))
        and math.isfinite(float(row["validation_excess_original"]))
    }
    movable = tuple(name for name in space.active if neighborhood_fields is None or name in set(neighborhood_fields))
    offsets = tuple(space.neighbor_offsets(diagonal=True, fields=movable))
    # Follow each undirected adjacency once while building the equality graph.
    forward_offsets = tuple(
        offset for offset in offsets
        if next((value for value in offset if value), 0) > 0
    )
    points = sorted(values)
    index_of = {point: index for index, point in enumerate(points)}
    buckets = np.asarray([_quantized(values[point]) for point in points], dtype=np.int64)
    left, right = [], []
    for index, point in enumerate(points):
        for offset in forward_offsets:
            neighbor = space.neighbor(point, offset)
            neighbor_index = index_of.get(neighbor)
            if neighbor_index is None or buckets[index] != buckets[neighbor_index]:
                continue
            left.append(index)
            right.append(neighbor_index)
    labels = _component_labels(
        len(points),
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        buckets,
    )
    grouped: dict[int, list[tuple[int, ...]]] = {}
    for point, label in zip(points, labels):
        grouped.setdefault(int(label), []).append(point)
    components = [sorted(component) for component in grouped.values()]
    record_details(neighborhood_graph_edges=len(left))
    record_count("neighborhood_points", len(points))

    component_of = {key: index for index, comp in enumerate(components) for key in comp}
    platforms = []
    for component_index, component in enumerate(components):
        complete = True
        local_peak = True
        reasons = []
        value = values[component[0]]
        for member in component:
            for offset in offsets:
                neighbor = space.neighbor(member, offset)
                if neighbor is None or component_of.get(neighbor) == component_index:
                    continue
                if neighbor not in values:
                    complete = False
                    reasons.append("平台外边界存在未计算或失败点")
                    continue
                if values[neighbor] > value + TOLERANCE:
                    local_peak = False
                    reasons.append("平台外边界存在更高收益点")
        platforms.append(Platform(
            platform_id=component_index + 1,
            members=component,
            value=value,
            complete=complete,
            local_peak=local_peak,
            reasons=list(dict.fromkeys(reasons)),
        ))
    return platforms


def _all_positive(points: list[tuple[int, ...]] | None, values: dict[tuple[int, ...], float]) -> bool:
    if not points:
        return False
    for point in points:
        value = values.get(point)
        if value is None or not math.isfinite(value) or value <= TOLERANCE:
            return False
    return True


def maximum_positive_box(
    space: ParameterSpace,
    center: tuple[int, ...],
    values: dict[tuple[int, ...], float],
    movable_fields: Iterable[str] | None = None,
) -> tuple[dict[str, int], list[tuple[int, ...]]] | None:
    """Branch-and-bound for the largest symmetric all-positive box."""
    movable = [name for name in space.active if movable_fields is None or name in set(movable_fields)]
    if not space.active:
        if values.get(center, -math.inf) <= TOLERANCE:
            return None
        return {}, [center]
    if not movable:
        return None
    by_name = dict(zip(space.fields, center))
    upper = {}
    for name in movable:
        dim = space.dimensions[name]
        tick = by_name[name]
        upper[name] = min(tick - dim.low, dim.high - tick)
        if upper[name] < 1:
            return None
    one = {name: 1 for name in movable}
    minimum_points = box_coordinates(space, center, one, movable_fields=movable)
    if not _all_positive(minimum_points, values):
        return None
    high = tuple(upper[name] for name in movable)
    maximum_points = box_coordinates(space, center, dict(zip(movable, high)), movable_fields=movable)
    if _all_positive(maximum_points, values):
        return dict(zip(movable, high)), list(maximum_points or [])

    best_volume = math.prod(one[name] for name in movable)
    best_radii = one
    best_points = list(minimum_points or [])
    heap: list[tuple[int, tuple[int, ...]]] = [(-math.prod(high), high)]
    visited: set[tuple[int, ...]] = set()
    while heap:
        negative_volume, candidate = heapq.heappop(heap)
        bound = -negative_volume
        if bound < best_volume or candidate in visited:
            continue
        visited.add(candidate)
        radii = dict(zip(movable, candidate))
        points = box_coordinates(space, center, radii, movable_fields=movable)
        if _all_positive(points, values):
            best_volume = bound
            best_radii = radii
            best_points = list(points or [])
            continue
        splittable = [index for index, value in enumerate(candidate) if value > 1]
        if not splittable:
            continue
        index = max(splittable, key=lambda item: (candidate[item] - 1, -item))
        for lower in range(1, candidate[index]):
            child = list(candidate)
            child[index] = lower
            child_tuple = tuple(child)
            child_volume = math.prod(child_tuple)
            if child_volume >= best_volume and child_tuple not in visited:
                heapq.heappush(heap, (-child_volume, child_tuple))
    if best_points is None:
        return None
    return best_radii, best_points


def fixed_box(
    space: ParameterSpace,
    center: tuple[int, ...],
    half_widths: dict[str, int],
    values: dict[tuple[int, ...], float],
    movable_fields: Iterable[str] | None = None,
) -> tuple[dict[str, int], list[tuple[int, ...]]] | None:
    movable = [name for name in space.active if movable_fields is None or name in set(movable_fields)]
    if not movable:
        return None
    radii = {name: int(half_widths.get(name, 1)) for name in movable}
    if any(radius < 1 for radius in radii.values()):
        return None
    points = box_coordinates(space, center, radii, movable_fields=movable)
    return (radii, points) if _all_positive(points, values) else None


def region_record(
    space: ParameterSpace,
    row: dict[str, Any],
    radii: dict[str, int],
    points: list[tuple[int, ...]],
    rows: dict[tuple[int, ...], dict[str, Any]],
    *,
    platform_id: int | None,
    global_exhaustive: bool,
) -> dict[str, Any]:
    values = [float(rows[point]["validation_excess_original"]) for point in points]
    coordinate = tuple(row["key"])
    raw_volume = math.prod(int(radii.get(name, 0)) for name in space.active if int(radii.get(name, 0)) > 0) if space.active else 1
    return {
        "center_params": dict(row["params"]),
        "center_coordinate": list(coordinate),
        "candidate_id": int(row.get("candidate_id", 0)),
        "platform_id": platform_id,
        "bounds": space.bounds_for(coordinate, radii),
        "radii": dict(radii),
        "raw_volume": int(raw_volume),
        "normalized_volume": space.normalized_volume(radii),
        "combination_count": len(points),
        "completed_count": len(points),
        "center_excess_return": float(row["validation_excess_original"]),
        "mean_excess_return": float(sum(values) / len(values)),
        "min_excess_return": float(min(values)),
        "complete": True,
        "maximum_search_complete": True,
        "global_exhaustive": bool(global_exhaustive),
        "overlapping_region_ids": [],
        "stress": {},
        "region_points": [list(point) for point in points],
    }


def region_sort_key(region: dict[str, Any], ranking_mode: str) -> tuple[Any, ...]:
    base = (
        -float(region["mean_excess_return"]),
        -float(region["min_excess_return"]),
        -float(region["center_excess_return"]),
        int(region["candidate_id"]),
    )
    if ranking_mode == "max_volume":
        return (-int(region["raw_volume"]), *base)
    return base


def annotate_overlaps(regions: list[dict[str, Any]]) -> None:
    sets = [set(tuple(point) for point in region["region_points"]) for region in regions]
    for left_index, left in enumerate(regions):
        overlaps = []
        for right_index, right in enumerate(regions):
            if left_index == right_index:
                continue
            if sets[left_index] & sets[right_index]:
                overlaps.append(int(right.get("region_id", right_index + 1)))
        left["overlapping_region_ids"] = overlaps


def find_regions(
    space: ParameterSpace,
    rows: dict[tuple[int, ...], dict[str, Any]],
    *,
    ranking_mode: str,
    fixed_half_widths: dict[str, float],
    max_drawdown: float,
    min_trades: int,
    global_exhaustive: bool,
    neighborhood_fields: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ranked formal regions and all validated platform diagnostics."""
    values = {
        key: float(row["validation_excess_original"])
        for key, row in rows.items()
        if row.get("evaluation_status") == "completed"
        and isinstance(row.get("validation_excess_original"), (int, float))
        and math.isfinite(float(row["validation_excess_original"]))
    }
    fixed_ticks = {
        name: value_to_ticks(name, fixed_half_widths.get(name, UNITS[name]))
        for name in space.active
    }
    movable = tuple(name for name in space.active if neighborhood_fields is None or name in set(neighborhood_fields))
    platforms = analyze_platforms(space, rows, neighborhood_fields=movable)
    regions: list[dict[str, Any]] = []
    diagnostics = []
    for platform in platforms:
        diagnostics.append({
            "platform_id": platform.platform_id,
            "members": [list(item) for item in platform.members],
            "value": platform.value,
            "complete": platform.complete,
            "local_peak": platform.local_peak,
            "reasons": platform.reasons,
        })
        if not platform.complete or not platform.local_peak:
            continue
        best = None
        for member in platform.members:
            row = rows[member]
            if int(row.get("buy_count", 0) or 0) <= 0 or int(row.get("sell_count", 0) or 0) <= 0:
                continue
            if float(row.get("max_drawdown", math.inf)) > max_drawdown or int(row.get("trade_count", 0) or 0) < min_trades:
                continue
            if ranking_mode == "max_volume":
                found = maximum_positive_box(space, member, values, movable_fields=movable)
            else:
                found = fixed_box(space, member, fixed_ticks, values, movable_fields=movable)
            if found is None:
                continue
            radii, points = found
            region = region_record(
                space, row, radii, points, rows,
                platform_id=platform.platform_id,
                global_exhaustive=global_exhaustive,
            )
            if best is None or region_sort_key(region, ranking_mode) < region_sort_key(best, ranking_mode):
                best = region
        if best is not None:
            regions.append(best)
    regions.sort(key=lambda region: region_sort_key(region, ranking_mode))
    for index, region in enumerate(regions, start=1):
        region["region_id"] = index
    annotate_overlaps(regions)
    return regions, diagnostics
