"""Peak, plateau and complete positive-neighborhood analysis."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Callable, Iterable

from .space import UNITS, ParameterSpace, value_to_ticks
from .search import box_coordinates


TOLERANCE = 1e-12


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
) -> list[Platform]:
    values = {
        key: float(row["validation_excess_original"])
        for key, row in rows.items()
        if row.get("evaluation_status") == "completed"
        and isinstance(row.get("validation_excess_original"), (int, float))
        and math.isfinite(float(row["validation_excess_original"]))
    }
    offsets = tuple(space.neighbor_offsets(diagonal=True))
    components: list[list[tuple[int, ...]]] = []
    assigned: set[tuple[int, ...]] = set()
    for start in sorted(values):
        if start in assigned:
            continue
        queue = [start]
        assigned.add(start)
        component = []
        while queue:
            current = queue.pop()
            component.append(current)
            bucket = _quantized(values[current])
            for offset in offsets:
                neighbor = space.neighbor(current, offset)
                if neighbor is None or neighbor in assigned or neighbor not in values:
                    continue
                if _quantized(values[neighbor]) == bucket:
                    assigned.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))

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
) -> tuple[dict[str, int], list[tuple[int, ...]]] | None:
    """Branch-and-bound for the largest symmetric all-positive box."""
    if not space.active:
        if values.get(center, -math.inf) <= TOLERANCE:
            return None
        return {}, [center]
    by_name = dict(zip(space.fields, center))
    upper = {}
    for name in space.active:
        dim = space.dimensions[name]
        tick = by_name[name]
        upper[name] = min(tick - dim.low, dim.high - tick)
        if upper[name] < 1:
            return None
    one = {name: 1 for name in space.active}
    minimum_points = box_coordinates(space, center, one)
    if not _all_positive(minimum_points, values):
        return None
    high = tuple(upper[name] for name in space.active)
    maximum_points = box_coordinates(space, center, dict(zip(space.active, high)))
    if _all_positive(maximum_points, values):
        return dict(zip(space.active, high)), list(maximum_points or [])

    best_volume = math.prod(one[name] for name in space.active)
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
        radii = dict(zip(space.active, candidate))
        points = box_coordinates(space, center, radii)
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
) -> tuple[dict[str, int], list[tuple[int, ...]]] | None:
    radii = {name: int(half_widths.get(name, 1)) for name in space.active}
    if any(radius < 1 for radius in radii.values()):
        return None
    points = box_coordinates(space, center, radii)
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
    raw_volume = math.prod(int(radii[name]) for name in space.active) if space.active else 1
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
    platforms = analyze_platforms(space, rows)
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
                found = maximum_positive_box(space, member, values)
            else:
                found = fixed_box(space, member, fixed_ticks, values)
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
