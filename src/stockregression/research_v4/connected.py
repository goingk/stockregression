"""Complete positive components on the exact parameter lattice (strategy v2)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable

import numpy as np
from scipy.spatial import cKDTree

from ..performance import rebalance_cpu_budget, record_count, record_seconds
from .search import _stable_seed, stratified_coordinates
from .space import FIELDS, ParameterSpace, ticks_to_value


TOLERANCE = 1e-12
Coordinate = tuple[int, ...]


def completed(row: dict | None) -> bool:
    if not row or row.get("evaluation_status") != "completed":
        return False
    value = row.get("validation_excess_original")
    return isinstance(value, (int, float)) and math.isfinite(value)


def positive(row: dict | None) -> bool:
    return completed(row) and float(row["validation_excess_original"]) > TOLERANCE


def neighbors(space: ParameterSpace, point: Coordinate, fields: Iterable[str]):
    """At most 2d legal axial neighbors; full coordinates always use FIELDS."""
    for name in fields:
        index = FIELDS.index(name)
        dim = space.dimensions[name]
        for delta in (-1, 1):
            value = point[index] + delta
            if value < dim.low or value > dim.high:
                continue
            candidate = list(point)
            candidate[index] = value
            if candidate[5] >= 0 and candidate[4] > candidate[5]:
                continue
            yield tuple(candidate)


def at_search_edge(space: ParameterSpace, point: Coordinate, fields: Iterable[str]) -> bool:
    """Whether a point lies on the declared search range, excluding legal min/max edges."""
    for name in fields:
        index = FIELDS.index(name)
        dim = space.dimensions[name]
        if point[index] in (dim.low, dim.high):
            return True
    return False


def coordinate_hash(points: Iterable[Coordinate]) -> str:
    digest = hashlib.sha256()
    for point in sorted(points):
        digest.update((json.dumps(point, separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def seed_coordinates(space: ParameterSpace, config, rows: list[dict], asof: str) -> list[Coordinate]:
    limit = min(int(config.strategy.validation_budget), space.size())
    ranked = sorted(
        (row for row in rows if row.get("evaluation_status") == "completed"
         and math.isfinite(float(row.get("training_score", math.nan)))),
        key=lambda row: (-float(row["training_score"]), tuple(row["key"])),
    )
    points = list(dict.fromkeys(tuple(row["key"]) for row in ranked))[: (limit + 1) // 2]
    seen = set(points)
    if len(points) >= limit:
        return points
    for point in stratified_coordinates(space, limit, _stable_seed(config, asof) ^ 0xC011EC7):
        if point not in seen:
            points.append(point)
            seen.add(point)
        if len(points) >= limit:
            break
    if len(points) < limit:
        for point in space.iterate():
            if point not in seen:
                points.append(point)
                seen.add(point)
            if len(points) >= limit:
                break
    return points


def refine_training(space: ParameterSpace, rows: list[dict], *, existing: set[Coordinate],
                    limit: int, recommendation_count: int, movable_fields: Iterable[str]) -> list[Coordinate]:
    """Round-robin multi-source BFS, including traversal of previously scored points."""
    if limit <= 0:
        return []
    fields = tuple(movable_fields)
    ranked = sorted(
        (row for row in rows if row.get("evaluation_status") == "completed"
         and math.isfinite(float(row.get("training_score", math.nan)))),
        key=lambda row: (-float(row["training_score"]), tuple(row["key"])),
    )
    centers: list[Coordinate] = []
    for row in ranked:
        point = tuple(row["key"])
        if any(sum(abs(a-b) for a, b in zip(point, other)) <= 1 for other in centers):
            continue
        centers.append(point)
        if len(centers) >= min(64, max(20, 3 * recommendation_count)):
            break
    queues = [deque([point]) for point in centers]
    visited = set(centers)
    used = set(existing)
    output: list[Coordinate] = []
    while len(output) < limit and any(queues):
        for queue in queues:
            while queue:
                point = queue.popleft()
                for other in neighbors(space, point, fields):
                    if other not in visited:
                        visited.add(other)
                        queue.append(other)
                if point not in used:
                    used.add(point)
                    output.append(point)
                    break
            if len(output) >= limit:
                break
    return output


@dataclass
class Exploration:
    rows: dict[Coordinate, dict]
    status: str
    seed_count: int
    retry_count: int = 0


def explore(space: ParameterSpace, seeds: list[Coordinate], fields: Iterable[str],
            evaluate: Callable[[list[Coordinate]], list[dict]], *, chunk_size: int = 512,
            cancelled: Callable[[], bool] | None = None,
            progress: Callable[[str], None] | None = None) -> Exploration:
    """Finish every discovered component, even when the seed budget is exceeded."""
    fields = tuple(fields)
    rows: dict[Coordinate, dict] = {}
    frontier: deque[Coordinate] = deque()
    queued: set[Coordinate] = set()
    seed_index = retries = 0
    parent: dict[Coordinate, Coordinate] = {}
    root_open: dict[Coordinate, int] = {}
    unresolved: dict[Coordinate, int] = {}

    def root(point):
        while parent[point] != point:
            parent[point] = parent[parent[point]]
            point = parent[point]
        return point

    def merge(a, b):
        left, right = root(a), root(b)
        if left != right:
            parent[right] = left
            root_open[left] += root_open.pop(right)

    def run(points):
        try:
            returned = evaluate(points)
        except InterruptedError:
            raise
        except (MemoryError, OSError):
            raise
        except Exception as exc:
            returned = [{"key": p, "params": space.params(p), "evaluation_status": "failed",
                         "validation_excess_original": None, "evaluation_error": str(exc)} for p in points]
        lookup = {tuple(row["key"]): row for row in returned}
        output = {p: lookup.get(p, {"key": p, "params": space.params(p),
                  "evaluation_status": "failed", "validation_excess_original": None}) for p in points}
        for row in output.values():
            if not completed(row):
                row["evaluation_status"] = "failed"
        return output

    try:
        while frontier or seed_index < len(seeds):
            if cancelled and cancelled():
                raise InterruptedError("连通域探查已取消")
            stage = "boundary_expansion" if frontier else "seed_evaluation"
            batch = []
            if frontier:
                while frontier and len(batch) < chunk_size:
                    point = frontier.popleft()
                    if point not in rows:
                        batch.append(point)
            else:
                while seed_index < len(seeds) and len(batch) < chunk_size:
                    point = seeds[seed_index]
                    seed_index += 1
                    if point not in rows and point not in queued:
                        batch.append(point)
                        queued.add(point)
            if not batch:
                continue
            started = time.perf_counter()
            result = run(batch)
            failed = [p for p in batch if not completed(result[p])]
            if failed:
                result.update(run(failed))
                retries += len(failed)
                record_count("region_retry_evaluations", len(failed))
            record_seconds(stage, time.perf_counter() - started)
            record_count(stage + "_points", len(batch))
            # Resolve existing positive nodes before adding the current batch.
            for point in batch:
                if completed(result[point]):
                    for other in neighbors(space, point, fields):
                        if other in parent:
                            unresolved[other] -= 1
                            root_open[root(other)] -= 1
            rows.update(result)
            new_positive = [p for p in batch if positive(rows[p])]
            for point in new_positive:
                parent[point] = point
                adjacent = list(neighbors(space, point, fields))
                unresolved[point] = sum(not completed(rows.get(p)) for p in adjacent)
                root_open[point] = unresolved[point]
                for other in adjacent:
                    if other not in rows and other not in queued:
                        queued.add(other)
                        frontier.append(other)
            for point in new_positive:
                for other in neighbors(space, point, fields):
                    if other in parent:
                        merge(point, other)
            if progress:
                progress(f"已评价 {len(rows):,} 点 · 已发现 {len(root_open):,} 域 / 已闭合 "
                         f"{sum(value == 0 for value in root_open.values()):,} 域 · 待评边界 {len(frontier):,} 点"
                         f" · 超出入口预算 {max(0, len(rows)-len(seeds)):,} 点")
    except InterruptedError:
        return Exploration(rows, "cancelled", len(seeds), retries)
    failed = any(not completed(row) for row in rows.values())
    return Exploration(rows, "partial" if failed else "completed", len(seeds), retries)


def _slice_size(space: ParameterSpace, point: Coordinate, fields: tuple[str, ...]) -> int:
    counts = {name: space.dimensions[name].count for name in fields}
    if not counts:
        return 1
    # min/max holdings use the same 500-share unit.
    if {"min_shares", "max_shares"} <= set(fields):
        a, b = space.dimensions["min_shares"], space.dimensions["max_shares"]
        full = max(0, min(a.high, b.low) - a.low + 1) * b.count
        lo, hi = max(a.low, b.low + 1), min(a.high, b.high)
        n = max(0, hi - lo + 1)
        pairs = full + n * (b.high + 1) - (lo + hi) * n // 2
        return pairs * math.prod(v for k, v in counts.items() if k not in {"min_shares", "max_shares"})
    if "min_shares" in fields and point[5] >= 0:
        dim = space.dimensions["min_shares"]
        counts["min_shares"] = max(0, min(dim.high, point[5]) - dim.low + 1)
    if "max_shares" in fields:
        dim = space.dimensions["max_shares"]
        counts["max_shares"] = max(0, dim.high - max(dim.low, point[4]) + 1)
    return math.prod(counts.values())


def analyze_regions(space: ParameterSpace, rows: dict[Coordinate, dict], fields: Iterable[str], *,
                    global_exhaustive: bool = False, cancelled: Callable[[], bool] | None = None):
    fields = tuple(fields)
    remaining = {p for p, row in rows.items() if positive(row)}
    regions = []
    incomplete = []
    started = time.perf_counter()
    distance_seconds = 0.0
    for first in sorted(remaining):
        if first not in remaining:
            continue
        if cancelled and cancelled():
            raise InterruptedError("连通域分析已取消")
        remaining.remove(first)
        queue = [first]
        members = []
        boundary = []
        closed = True
        touches_range = False
        while queue:
            if len(members) % 2048 == 0 and cancelled and cancelled():
                raise InterruptedError("连通域分析已取消")
            point = queue.pop()
            members.append(point)
            adjacent = list(neighbors(space, point, fields))
            edge = len(adjacent) < 2 * len(fields)
            touches_range |= at_search_edge(space, point, fields)
            for other in adjacent:
                if positive(rows.get(other)):
                    if other in remaining:
                        remaining.remove(other)
                        queue.append(other)
                else:
                    edge = True
                    closed &= completed(rows.get(other))
            if edge or not fields:
                boundary.append(point)
        members.sort()
        if not closed:
            incomplete.append({"combination_count": len(members), "boundary_closed": False,
                               "member_hash": coordinate_hash(members)})
            continue
        distance_start = time.perf_counter()
        indices = [FIELDS.index(name) for name in fields]
        margins = np.zeros(len(members), dtype=np.float64)
        if indices and boundary and len(boundary) < len(members):
            widths = np.asarray([space.dimensions[name].width for name in fields], dtype=np.float64)
            lows = np.asarray([space.dimensions[name].low for name in fields], dtype=np.float64)
            edge_coords = (np.asarray(boundary, dtype=np.float64)[:, indices] - lows) / widths
            tree = cKDTree(edge_coords)
            for offset in range(0, len(members), 16384):
                if cancelled and cancelled():
                    raise InterruptedError("稳定距离计算已取消")
                block = (np.asarray(members[offset:offset+16384], dtype=np.float64)[:, indices] - lows) / widths
                margins[offset:offset+len(block)] = tree.query(block, k=1, eps=0, p=2,
                                                               workers=rebalance_cpu_budget())[0]
        distance_seconds += time.perf_counter() - distance_start
        for point, margin in zip(members, margins):
            rows[point]["stability_margin"] = float(margin)
        stable = min(members, key=lambda p: (-rows[p]["stability_margin"], -rows[p]["validation_excess_original"], p))
        peak = min(members, key=lambda p: (-rows[p]["validation_excess_original"], -rows[p]["stability_margin"], p))

        def representative(point):
            row = rows[point]
            return {"candidate_id": row["candidate_id"], "params": dict(row["params"]),
                    "coordinate": list(point), "excess_return": row["validation_excess_original"],
                    "stability_margin": row["stability_margin"],
                    "constraint_eligible": bool(row.get("constraint_eligible")),
                    "constraint_reasons": list(row.get("constraint_reasons", [])),
                    "max_drawdown": row.get("max_drawdown"), "trade_count": row.get("trade_count")}

        values = [float(rows[p]["validation_excess_original"]) for p in members]
        bounds = {name: {"low": ticks_to_value(name, min(p[FIELDS.index(name)] for p in members)),
                         "high": ticks_to_value(name, max(p[FIELDS.index(name)] for p in members))}
                  for name in space.dimensions}
        regions.append({
            "geometry": "connected_lattice", "connectivity": "axis_one_tick",
            "stable_point": representative(stable), "peak_point": representative(peak),
            "center_params": dict(rows[stable]["params"]), "center_coordinate": list(stable),
            "candidate_id": rows[stable]["candidate_id"],
            "combination_count": len(members), "completed_count": len(members),
            "raw_volume": len(members), "normalized_volume": len(members) / max(1, _slice_size(space, stable, fields)),
            "mean_excess_return": math.fsum(values) / len(values), "min_excess_return": min(values),
            "max_excess_return": max(values), "center_excess_return": rows[stable]["validation_excess_original"],
            "complete": True, "boundary_closed": True, "touches_search_boundary": bool(touches_range),
            "global_exhaustive": global_exhaustive, "bounds": bounds, "bounds_are_envelope": True,
            "member_hash": coordinate_hash(members), "region_points": members,
            "constraint_failure_count": sum(not rows[p].get("constraint_eligible", False) for p in members),
        })
    regions.sort(key=lambda r: (-r["combination_count"], -r["stable_point"]["stability_margin"],
                               -r["mean_excess_return"], -r["max_excess_return"], tuple(r["region_points"][0])))
    for index, region in enumerate(regions, 1):
        region["region_id"] = index
        region["member_reference"] = {"archive": "candidate_archive", "region_id": index}
        for point in region["region_points"]:
            rows[point]["region_ids"] = [index]
        rows[tuple(region["stable_point"]["coordinate"])].update(is_center=True, is_stable_point=True)
        rows[tuple(region["peak_point"]["coordinate"])]["is_peak_point"] = True
    record_seconds("connected_components", time.perf_counter()-started-distance_seconds)
    record_seconds("stability_distance", distance_seconds)
    record_count("connected_regions", len(regions))
    record_count("component_points", sum(r["combination_count"] for r in regions))
    return regions, incomplete
