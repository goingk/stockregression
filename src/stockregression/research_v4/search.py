"""Deterministic coarse lattice seeding for the v5 robust-J search.

Only :func:`coarse_coordinates` survives from the old search module.  The
stratified, statistical, sweep, refinement and validation-plan helpers it used
to host belonged to the removed v1-v4 protocols and have no v5 caller.
"""
from __future__ import annotations

from itertools import product
import math

from .space import ParameterSpace


def coarse_coordinates(space: ParameterSpace, limit: int) -> list[tuple[int, ...]]:
    """Return a deterministic, evenly spread subset of the lattice."""
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
