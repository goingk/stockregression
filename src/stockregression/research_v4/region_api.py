"""Views of immutable connected-region archives, without rectangular interpolation."""
from __future__ import annotations

import math
from typing import Any

from .space import FIELDS


def decision_context(result: dict, decision: str, index: int = 0):
    if decision == "current":
        return result.get("fixed", {}).get("current", {})
    if decision == "fixed":
        return result.get("fixed", {})
    decisions = result.get("evidence", {}).get("decisions", [])
    if index >= len(decisions):
        raise LookupError("研究决策不存在")
    return decisions[index]


def decision_archive(result: dict, decision: str, index: int = 0):
    archive = result.get("evidence", {}).get("candidate_archives", {}).get(decision, [])
    if decision == "adaptive":
        if index >= len(archive):
            raise LookupError("研究决策不存在")
        archive = archive[index].get("candidate_archive", [])
    return [row for row in archive if isinstance(row, dict) and isinstance(row.get("params"), dict)]


def connected_landscape(result: dict, *, decision: str, decision_index: int, region_id: int | None,
                        view: str, x: str | None, y: str | None, anchor: str, offset: int, limit: int,
                        sample: str, cost: str, fold: int | None, point_factory) -> dict[str, Any]:
    context = decision_context(result, decision, decision_index)
    regions = context.get("recommendation_regions", [])
    region = next((r for r in regions if r.get("region_id") == region_id), None)
    if region_id is not None and region is None:
        raise LookupError("推荐参数域不存在")
    archive = decision_archive(result, decision, decision_index)
    if region_id is not None:
        archive = [r for r in archive if region_id in r.get("region_ids", [])]
    archive.sort(key=lambda row: row["candidate_id"])
    names = [name for name in FIELDS if any(r["params"].get(name) is not None for r in archive)]
    if x is not None and x not in names or y is not None and y not in names:
        raise ValueError("横纵轴必须为当前参数域中的参数")
    if x and y and x == y:
        raise ValueError("横纵轴参数必须不同")
    x = x or (names[0] if names else "")
    y = y or next((name for name in names if name != x), "")
    chosen = region.get("peak_point" if anchor == "peak" else "stable_point") if region else None
    if chosen is None and archive:
        chosen = next((r for r in archive if r.get("is_stable_point")), archive[0])
    focus_params = chosen.get("params", {}) if chosen else {}

    def point(row):
        value = point_factory(row, sample, cost, fold)
        value["is_focus"] = bool(chosen and row.get("candidate_id") == chosen.get("candidate_id"))
        return value

    projected = []
    selected = archive
    if view == "slice":
        selected = [r for r in archive if all(r["params"].get(name) == focus_params.get(name)
                                             for name in names if name not in {x, y})]
    elif view == "projection":
        cells = {}
        for row in archive:
            key = (row["params"].get(x), row["params"].get(y) if y else 0)
            cell = cells.setdefault(key, {"x": key[0], "y": key[1], "count": 0,
                                         "finite_count": 0, "sum_excess": 0.0, "sum_grid": 0.0,
                                         "grid_count": 0})
            cell["count"] += 1
            value = point(row)
            if isinstance(value["excess_return"], (int, float)) and math.isfinite(value["excess_return"]):
                cell["finite_count"] += 1
                cell["sum_excess"] += value["excess_return"]
            if isinstance(value["grid_return"], (int, float)) and math.isfinite(value["grid_return"]):
                cell["grid_count"] += 1
                cell["sum_grid"] += value["grid_return"]
        projected = [{"x": c["x"], "y": c["y"], "count": c["count"],
                      "excess_return": c["sum_excess"]/c["finite_count"] if c["finite_count"] else None,
                      "grid_return": c["sum_grid"]/c["grid_count"] if c["grid_count"] else None}
                     for _, c in sorted(cells.items())]
    viewed_total = len(projected) if view == "projection" else len(selected)
    page = selected[offset:offset+limit] if view != "projection" else []
    representatives = {}
    for role in ("stable_point", "peak_point"):
        representative = region.get(role) if region else None
        row = next((r for r in archive if representative and r["candidate_id"] == representative["candidate_id"]), None)
        representatives[role] = point(row) if row else None
    return {
        "geometry": "connected_lattice", "decision": decision, "decision_index": decision_index,
        "sample": sample, "cost": cost, "fold": fold, "view": view, "region_id": region_id,
        "regions": regions, "region": region, "diagnostics": context.get("diagnostics", {}),
        "as_of": context.get("as_of", context.get("decision_date")),
        "parameters": names, "x": x, "y": y, "anchor": anchor, "fixed_params": focus_params,
        "total": len(archive), "view_total": viewed_total, "offset": offset, "limit": limit,
        "points": [point(r) for r in page], "projection": projected[offset:offset+limit],
        "stable_point": representatives["stable_point"], "peak_point": representatives["peak_point"],
        "sampled": False, "paginated": viewed_total > offset+limit,
        "representative_basis": "original_cost_full_validation_interval",
    }
