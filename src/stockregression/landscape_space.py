"""Archive-only parameter-space views with bounded deterministic sampling."""
from __future__ import annotations

import hashlib
import heapq
import sqlite3

from .storage import _CANDIDATE_REFERENCE, _candidate_where

FIELDS = ("buy_step", "sell_step", "buy_shares", "sell_shares")


def landscape_space(store, result, result_id, *, decision, decision_index=0,
                    region_id=None, revision=None, limit=5000, x=None, y=None,
                    z=None, fixed_axis=None, fixed_tick=None, selected_candidate_id=None):
    from .api import _landscape_point, _robust_landscape_ranking, _research_decision_context

    config = result.get("config") or {}
    base = config.get("base") or {}
    links = {"spacing": False, "shares": False, **(base.get("parameter_links") or {})}
    groups = {name: [name] for name in FIELDS}
    for enabled, left, right in ((links["spacing"], "buy_step", "sell_step"),
                                  (links["shares"], "buy_shares", "sell_shares")):
        if enabled:
            groups[left] = [left, right]
            groups.pop(right)
    axes = list(groups)
    group = "search_candidate_archives"
    values = store.candidate_axis_values(result, group, decision, decision_index,
                                         names=axes, revision=revision)
    ranges = config.get("ranges") or {}
    granularity = (config.get("strategy") or {}).get("granularity") or {}
    units = {name: float(granularity.get(name, .0001 if "step" in name else 100)) for name in axes}
    bounds = {}
    for name in axes:
        observed = values.get(name) or []
        source = ranges.get(name) or {}
        low = source.get("low", min(observed) if observed else 0)
        high = source.get("high", max(observed) if observed else low)
        bounds[name] = {"min": low, "max": high}
    active = [name for name in axes if bounds[name]["max"] > bounds[name]["min"]]
    metadata = {"version": 1, "parameter_links": links,
                "independent_axes": [{"id": name, "members": members} for name, members in groups.items()],
                "active_axes": active}
    # Full new configurations can reproduce effective tick bounds (including
    # STAR-board minimums and ranges that collapse after tick alignment).
    # Minimal historical summaries remain readable via the archive fallback.
    try:
        from .models import OptimizationConfig
        from .research.space import ParameterSpace, ROBUST_FIELDS
        space = ParameterSpace(OptimizationConfig.model_validate(config), field_order=ROBUST_FIELDS)
        metadata = space.metadata()
        active = metadata["active_axes"]
        for name in axes:
            dim = space.dimensions[name]
            bounds[name] = {"min": space.value(name, dim.low), "max": space.value(name, dim.high)}
    except (ValueError, KeyError, TypeError):
        frozen = (result.get("protocol") or {}).get("parameter_space")
        if isinstance(frozen, dict) and isinstance(frozen.get("active_axes"), list):
            metadata = frozen
            active = [name for name in frozen["active_axes"] if name in axes]
    reference = store.candidate_reference(result, group, decision, decision_index, revision=revision)
    definition = {"parameter_space": metadata, "axis_values": values, "axis_bounds": bounds,
                  "axis_ticks": {name: [round(v / units[name]) for v in values.get(name, [])] for name in axes}}
    payload = {"result_id": result_id, "decision": decision, "decision_index": decision_index,
               "revision": revision, "sample": "training", "cost": "original", "parameters": axes,
               "total": 0, "points": [], "returned_count": 0, "sampled": False,
               "sample_strategy": "all", "focus": None, "focus_neighborhood": None,
               "top_candidates": [], "slices": {}, "bounds": None, "center_count": 0,
               "metric_bounds": {"excess": None, "grid": None}, **definition}
    if not reference:
        return payload
    path = (store.root / reference[_CANDIDATE_REFERENCE]).resolve()
    if store.root.resolve() not in path.parents or not path.is_file():
        raise LookupError("候选档案不存在")

    def query(**kwargs):
        return store.candidate_archive(result, group, decision, decision_index,
                                       revision=revision, **kwargs)["rows"]

    leader_rows = query(region_id=region_id, sort_by="training_excess_return", descending=True, limit=10)
    if not leader_rows:
        return payload
    leader = leader_rows[0]
    requested_axes = [name for name in (x, y, z) if name is not None]
    if any(name not in axes for name in requested_axes) or len(set(requested_axes)) != len(requested_axes):
        raise ValueError("视图坐标轴必须是不重复的独立参数")
    if fixed_axis is not None and fixed_axis not in axes:
        raise ValueError("固定轴必须是独立参数")
    if fixed_axis is not None and fixed_axis in requested_axes:
        raise ValueError("固定轴不能同时作为展示轴")
    if len(active) >= 4 and fixed_axis is None:
        fixed_axis = "sell_shares" if "sell_shares" in active else active[-1]
    filters = {}
    if fixed_axis is not None:
        available = values.get(fixed_axis) or []
        if fixed_tick is None:
            raw = leader.get("params", {}).get(fixed_axis, available[0] if available else 0)
            fixed_tick = round(float(raw) / units[fixed_axis])
        # Pick the original archived value after integer-tick validation so
        # SQLite float equality never depends on a reconstructed multiplication.
        matches = [v for v in available if round(v / units[fixed_axis]) == fixed_tick]
        if not matches:
            raise ValueError("固定刻度没有对应的归档候选")
        filters = {member: matches[0] for member in groups[fixed_axis]}
    # A two-axis request is an exact plane, even when the whole space has
    # three/four free axes. This also supplies the WebGL fallback correctly.
    if len(requested_axes) == 2:
        for name in axes:
            if name in requested_axes or name == fixed_axis:
                continue
            raw = leader.get("params", {}).get(name)
            if raw is not None:
                filters.update({member: raw for member in groups[name]})
    payload.update(fixed_axis=fixed_axis, fixed_tick=fixed_tick, fixed_parameters=filters)
    ranking_rows = query(region_id=region_id, filters={"score_complete": 1},
                         sort_by="j_score", descending=True, limit=800)
    selection = _research_decision_context(result, decision, decision_index)
    ranking = _robust_landscape_ranking(ranking_rows, config, selection)
    payload.update(ranking)
    payload["j_candidate_count"] = store.candidate_stats(
        result, group, decision, decision_index, region_id=region_id, revision=revision,
    ).get("j_complete", 0)
    payload["focus"] = {**_landscape_point(leader, "training", "original", None), "is_focus": True}
    payload["top_candidates"] = [_landscape_point(row, "training", "original", None) for row in leader_rows]

    connection = sqlite3.connect(path)
    try:
        global_clause, global_args = _candidate_where(connection, region_id=region_id)
        metric_row = connection.execute(
            "SELECT MIN(c.training_excess_return),MAX(c.training_excess_return),"
            "MIN(json_extract(c.row_json,'$.training_return')),"
            "MAX(json_extract(c.row_json,'$.training_return')) FROM candidates c" + global_clause,
            global_args,
        ).fetchone()
        for index, metric in ((0, "excess"), (2, "grid")):
            if metric_row[index] is not None:
                payload["metric_bounds"][metric] = {"min": metric_row[index], "max": metric_row[index + 1]}
        payload["bounds"] = payload["metric_bounds"]["excess"]
        clause, arguments = _candidate_where(connection, region_id=region_id, filters=filters)
        total = connection.execute("SELECT COUNT(*) FROM candidates c" + clause, arguments).fetchone()[0]
        payload["total"] = total
        mandatory = [leader]
        recommended = {row.get("candidate_id") for row in ranking.get("j_ranking", [])}
        mandatory.extend(row for row in ranking_rows if row.get("candidate_id") in recommended)
        if selected_candidate_id is not None:
            mandatory.extend(query(region_id=region_id, filters={"candidate_id": selected_candidate_id}, limit=1))
        required = {int(row["_archive_ordinal"]) for row in mandatory
                    if all(row.get("params", {}).get(name) == value for name, value in filters.items())}
        # The normal 5000-point view retains every recommended point. Tiny
        # explicit limits prioritize the leader and then selected/recommended.
        if len(required) > limit:
            ordered = [int(row["_archive_ordinal"]) for row in mandatory if int(row["_archive_ordinal"]) in required]
            required = set(dict.fromkeys(ordered[:limit]))
        selected = set(required)
        if total <= limit:
            selected.update(row[0] for row in connection.execute("SELECT c.ordinal FROM candidates c" + clause, arguments))
        else:
            display_axes = requested_axes or [name for name in active if name != fixed_axis][:3]
            slots = max(0, limit - len(required))
            bins = max(1, int(max(1, slots) ** (1 / max(1, len(display_axes)))))
            cells = {}
            reservoir = []
            sql = "SELECT c.ordinal,c.candidate_id," + ",".join("c." + name for name in FIELDS) + " FROM candidates c" + clause
            for row in connection.execute(sql, arguments):
                ordinal = int(row[0])
                if ordinal in required:
                    continue
                params = dict(zip(FIELDS, row[2:]))
                digest = hashlib.blake2b(repr(tuple(row[1:])).encode(), digest_size=8).digest()
                priority = int.from_bytes(digest, "big")
                cell = []
                for name in display_axes:
                    low, high = bounds[name]["min"], bounds[name]["max"]
                    value = params[name] if params[name] is not None else low
                    cell.append(min(bins - 1, max(0, int((value - low) / max(1e-20, high - low) * bins))))
                key = tuple(cell)
                if key not in cells or (priority, ordinal) < cells[key]:
                    cells[key] = (priority, ordinal)
                if slots:
                    entry = (-priority, -ordinal, ordinal)
                    if len(reservoir) < slots:
                        heapq.heappush(reservoir, entry)
                    elif entry > reservoir[0]:
                        heapq.heapreplace(reservoir, entry)
            selected.update(ordinal for _, ordinal in sorted(cells.values())[:slots])
            for _, _, ordinal in sorted(reservoir, reverse=True):
                if len(selected) >= limit:
                    break
                selected.add(ordinal)
        selected_rows = query(region_id=region_id, filters=filters, ordinals=sorted(selected),
                              sort_by="ordinal", limit=max(1, len(selected))) if selected else []
    finally:
        connection.close()
    points = [_landscape_point(row, "training", "original", None) for row in selected_rows]
    for point in points:
        point["is_focus"] = point["candidate_id"] == leader.get("candidate_id")
    payload.update(points=points, returned_count=len(points), sampled=len(points) < total,
                   sample_strategy="spatial_hash" if len(points) < total else "all",
                   center_count=sum(point["is_center"] for point in points),
                   retained_candidate_ids=[row["candidate_id"] for row in selected_rows if row["_archive_ordinal"] in required],
                   completion={"status": "available", "endpoint": f"/api/v1/results/{result_id}/completion"})
    return payload
