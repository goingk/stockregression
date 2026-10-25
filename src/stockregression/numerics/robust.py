"""Pure mathematics for the robust-J neighbourhood objective."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from ..models import ResearchStrategySpec
if TYPE_CHECKING:
    from ..research.space import ParameterSpace


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def completed(row: Mapping[str, Any] | None) -> bool:
    return bool(row and row.get("evaluation_status") == "completed" and _finite(row.get("training_excess_return")))


@dataclass(frozen=True)
class NeighborhoodOperator:
    """A clipped fixed-edge lattice cube in integer-tick space."""

    space: ParameterSpace
    radius: int

    def points(self, center: tuple[int, ...], *, radius: int | None = None) -> list[tuple[int, ...]]:
        radius = self.radius if radius is None else int(radius)
        axes: list[tuple[int, ...]] = []
        for name in self.space.active:
            index = self.space._field_index[name]
            dim = self.space.dimensions[name]
            low = max(dim.low, center[index] - radius)
            high = min(dim.high, center[index] + radius)
            axes.append(tuple(range(low, high + 1)))
        if not axes:
            return [center]
        return [
            self.space.full_coordinate(values)
            for values in product(*axes)
            if self.space.legal(values)
        ]

    def distance_squared(self, left: tuple[int, ...], right: tuple[int, ...]) -> int:
        total = 0
        for name in self.space.active:
            index = self.space._field_index[name]
            total += (int(left[index]) - int(right[index])) ** 2
        return total

    def weights(self, center: tuple[int, ...], sigma: float, points: Iterable[tuple[int, ...]] | None = None) -> list[float]:
        values = list(points if points is not None else self.points(center))
        raw = [math.exp(-self.distance_squared(center, point) / (2.0 * float(sigma) ** 2)) for point in values]
        total = math.fsum(raw)
        if total <= 0 or not math.isfinite(total):
            return []
        return [value / total for value in raw]


@dataclass(frozen=True)
class RobustScore:
    center_return: float
    neighborhood_mean: float
    downside_risk: float
    position_penalty: float
    max_position_penalty: float
    min_position_penalty: float
    j_score: float
    neighborhood_size: int
    support_size: int
    positive_ratio: float
    safe_ratio: float
    neighborhood_max_eta: float | None
    complete: bool
    missing_count: int
    floor_episode_count: int | None = None
    floor_episode_grids: tuple[int, ...] = ()
    rows: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "center_excess_return": self.center_return,
            "neighborhood_mean": self.neighborhood_mean,
            "downside_risk_tau": self.downside_risk,
            "pos_penalty": self.position_penalty,
            "max_position_penalty": self.max_position_penalty,
            "min_position_penalty": self.min_position_penalty,
            "j_score": self.j_score,
            "neighborhood_size": self.neighborhood_size,
            "neighborhood_support_size": self.support_size,
            "pos_win_rate": self.positive_ratio,
            "safe_win_rate": self.safe_ratio,
            "neighborhood_max_eta": self.neighborhood_max_eta,
            "score_complete": self.complete,
            "score_missing_count": self.missing_count,
            "floor_episode_count": self.floor_episode_count,
            "floor_episode_grids": list(self.floor_episode_grids),
            "floor_penalty": self.min_position_penalty,
            "floor_penalty_basis": "episode_grids_v1",
        }


def episode_grids(row: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    if row is None:
        return None
    values = row.get("floor_episode_grids")
    if not isinstance(values, (list, tuple)):
        return None
    output = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = int(value)
        if number < 0 or float(value) != number:
            return None
        output.append(number)
    return tuple(output)


def position_penalty(row: Mapping[str, Any], strategy: ResearchStrategySpec) -> tuple[float, float, float]:
    maximum = float(row.get("actual_max_pos_ratio", math.inf))
    maximum_penalty = float(strategy.gamma_max) * max(0.0, maximum - float(strategy.eta_limit)) ** 2 if math.isfinite(maximum) else 0.0
    grids = episode_grids(row)
    minimum_penalty = (
        float(strategy.gamma_min) * math.fsum(
            (grid / float(strategy.floor_k_ref)) ** 2 for grid in grids
        ) if grids is not None else math.nan
    )
    return maximum_penalty + minimum_penalty, maximum_penalty, minimum_penalty


def robust_score(
    center: tuple[int, ...],
    rows: Mapping[tuple[int, ...], dict[str, Any]],
    space: ParameterSpace,
    strategy: ResearchStrategySpec,
) -> RobustScore:
    operator = NeighborhoodOperator(space, int(strategy.hypercube_edge) // 2)
    points = operator.points(center)
    weights = operator.weights(center, float(strategy.sigma), points)
    missing = [point for point in points if not completed(rows.get(point))]
    center_row = rows.get(center) if completed(rows.get(center)) else None
    center_return = float(center_row["training_excess_return"]) if center_row is not None else math.nan
    floors = episode_grids(center_row)
    if center_row is None or missing or len(weights) != len(points) or floors is None:
        return RobustScore(
            center_return=center_return, neighborhood_mean=math.nan,
            downside_risk=math.nan, position_penalty=math.nan,
            max_position_penalty=math.nan, min_position_penalty=math.nan,
            j_score=math.nan, neighborhood_size=len(points) - len(missing),
            support_size=len(points) - len(missing), positive_ratio=math.nan,
            safe_ratio=math.nan, neighborhood_max_eta=None, complete=False,
            missing_count=len(missing) + (0 if floors is not None else 1), rows={},
        )
    returns = [float(rows[point]["training_excess_return"]) for point in points]
    mean = math.fsum(weight * value for weight, value in zip(weights, returns))
    downside = math.sqrt(math.fsum(
        weight * min(0.0, value - float(strategy.tau)) ** 2
        for weight, value in zip(weights, returns)
    ))
    penalty, maximum_penalty, minimum_penalty = position_penalty(center_row, strategy)
    j_score = (
        float(strategy.alpha) * center_return
        + (1.0 - float(strategy.alpha)) * mean
        - float(strategy.beta) * downside
        - penalty
    )
    max_etas = [float(rows[point].get("actual_max_pos_ratio", math.nan)) for point in points]
    finite_etas = [value for value in max_etas if math.isfinite(value)]
    return RobustScore(
        center_return=center_return,
        neighborhood_mean=mean,
        downside_risk=downside,
        position_penalty=penalty,
        max_position_penalty=maximum_penalty,
        min_position_penalty=minimum_penalty,
        j_score=float(j_score),
        neighborhood_size=len(points),
        support_size=len(points),
        positive_ratio=float(sum(value > 0 for value in returns) / len(returns)) if returns else math.nan,
        safe_ratio=float(sum(value >= float(strategy.tau) for value in returns) / len(returns)) if returns else math.nan,
        neighborhood_max_eta=max(finite_etas) if finite_etas else None,
        complete=True,
        missing_count=0,
        floor_episode_count=len(floors),
        floor_episode_grids=floors,
        rows={str(point): dict(rows[point]) for point in points},
    )


def tensor_robust_scores(
    space: ParameterSpace,
    rows: Mapping[tuple[int, ...], dict[str, Any]],
    strategy: ResearchStrategySpec,
) -> dict[tuple[int, ...], RobustScore]:
    """Vectorised direct-form J values over one shared dense lattice buffer.

    The buffer is zero-filled for convenience, while a separate legality mask
    and observation mask prevent padding from being counted as a real return.
    """
    import numpy as np

    active = space.active
    if not active:
        center = space.full_coordinate(())
        return {center: robust_score(center, rows, space, strategy)}
    shape = tuple(space.dimensions[name].count for name in active)
    returns = np.full(shape, np.nan, dtype=np.float64)
    observed = np.zeros(shape, dtype=bool)
    etas = np.full(shape, np.nan, dtype=np.float64)
    for coordinate, row in rows.items():
        values = space.active_coords(coordinate)
        index = tuple(value - space.dimensions[name].low for value, name in zip(values, active))
        if any(value < 0 or value >= size for value, size in zip(index, shape)):
            continue
        if completed(row):
            returns[index] = float(row["training_excess_return"])
            observed[index] = True
        eta = row.get("actual_max_pos_ratio")
        if _finite(eta):
            etas[index] = float(eta)

    radius = int(strategy.hypercube_edge) // 2
    sigma = float(strategy.sigma)
    weighted_return = np.zeros(shape, dtype=np.float64)
    weighted_downside = np.zeros(shape, dtype=np.float64)
    legal_weight_mass = np.zeros(shape, dtype=np.float64)
    observed_count = np.zeros(shape, dtype=np.int64)
    positive_count = np.zeros(shape, dtype=np.int64)
    safe_count = np.zeros(shape, dtype=np.int64)
    maximum_eta = np.full(shape, np.nan, dtype=np.float64)
    legal_count = np.zeros(shape, dtype=np.int64)

    for offset in product(range(-radius, radius + 1), repeat=len(active)):
        weight = math.exp(-sum(value * value for value in offset) / (2.0 * sigma * sigma))
        source = tuple(slice(max(0, -shift), min(size, size - shift)) for shift, size in zip(offset, shape))
        target = tuple(slice(max(0, shift), min(size, size + shift)) for shift, size in zip(offset, shape))
        if not all(part.stop > part.start for part in source):
            continue
        source_return = returns[source]
        source_observed = observed[source]
        legal_weight_mass[target] += weight
        legal_count[target] += 1
        observed_count[target] += source_observed.astype(np.int64)
        weighted_return[target] += np.where(source_observed, source_return, 0.0) * weight
        weighted_downside[target] += np.where(
            source_observed, np.minimum(0.0, source_return - float(strategy.tau)) ** 2, 0.0,
        ) * weight
        positive_count[target] += (source_observed & (source_return > 0)).astype(np.int64)
        safe_count[target] += (source_observed & (source_return >= float(strategy.tau))).astype(np.int64)
        source_eta = etas[source]
        current_eta = maximum_eta[target]
        maximum_eta[target] = np.where(
            np.isfinite(source_eta),
            np.where(np.isfinite(current_eta), np.maximum(current_eta, source_eta), source_eta),
            current_eta,
        )

    output: dict[tuple[int, ...], RobustScore] = {}
    for coordinate in space.iterate():
        active_values = space.active_coords(coordinate)
        index = tuple(value - space.dimensions[name].low for value, name in zip(active_values, active))
        row = rows.get(coordinate)
        center_return = float(row["training_excess_return"]) if completed(row) else math.nan
        missing = int(legal_count[index] - observed_count[index])
        floors = episode_grids(row)
        if not completed(row) or missing or legal_weight_mass[index] <= 0 or floors is None:
            output[coordinate] = RobustScore(
                center_return=center_return, neighborhood_mean=math.nan,
                downside_risk=math.nan, position_penalty=math.nan,
                max_position_penalty=math.nan, min_position_penalty=math.nan,
                j_score=math.nan, neighborhood_size=int(legal_count[index]),
                support_size=int(observed_count[index]), positive_ratio=math.nan,
                safe_ratio=math.nan,
                neighborhood_max_eta=float(maximum_eta[index]) if np.isfinite(maximum_eta[index]) else None,
                complete=False, missing_count=missing + (0 if floors is not None else 1), rows={},
            )
            continue
        mass = float(legal_weight_mass[index])
        mean = float(weighted_return[index] / mass)
        downside = math.sqrt(float(weighted_downside[index] / mass))
        penalty, maximum_penalty, minimum_penalty = position_penalty(row, strategy)
        j_score = (
            float(strategy.alpha) * center_return
            + (1.0 - float(strategy.alpha)) * mean
            - float(strategy.beta) * downside
            - penalty
        )
        count = int(observed_count[index])
        output[coordinate] = RobustScore(
            center_return=center_return, neighborhood_mean=mean, downside_risk=downside,
            position_penalty=penalty, max_position_penalty=maximum_penalty,
            min_position_penalty=minimum_penalty, j_score=j_score,
            neighborhood_size=int(legal_count[index]), support_size=count,
            positive_ratio=float(positive_count[index] / count) if count else math.nan,
            safe_ratio=float(safe_count[index] / count) if count else math.nan,
            neighborhood_max_eta=float(maximum_eta[index]) if np.isfinite(maximum_eta[index]) else None,
            complete=True, missing_count=0, floor_episode_count=len(floors),
            floor_episode_grids=floors, rows={},
        )
    return output


def eligible(row: Mapping[str, Any] | None, score: RobustScore | None, *, max_principal_drawdown: float, min_trades: int, strategy: ResearchStrategySpec) -> bool:
    if not completed(row) or score is None or not score.complete or not math.isfinite(score.j_score) or score.j_score <= 0:
        return False
    drawdown = float(row.get("max_principal_drawdown", math.inf))
    if not math.isfinite(drawdown) or drawdown > float(max_principal_drawdown):
        return False
    if int(row.get("trade_count", 0) or 0) < int(min_trades):
        return False
    if int(row.get("buy_count", 0) or 0) <= 0 or int(row.get("sell_count", 0) or 0) <= 0:
        return False
    maximum = row.get("actual_max_pos_ratio")
    if strategy.eta_hard_cap is not None and (not _finite(maximum) or float(maximum) > float(strategy.eta_hard_cap)):
        return False
    return True


@dataclass(frozen=True)
class PeakAnalysis:
    state: str
    center: tuple[int, ...]
    representative: tuple[int, ...] | None = None
    plateau: tuple[tuple[int, ...], ...] = ()
    higher: tuple[int, ...] | None = None
    missing: tuple[tuple[int, ...], ...] = ()
    complete: bool = False
    local_peak: bool = False
    flat_region: bool = False


ScoreLookup = Callable[[tuple[int, ...]], RobustScore | None]
EligibilityLookup = Callable[[tuple[int, ...]], bool]


class PeakAnalyzer:
    """Closes an equal-J plateau and checks every one-tick exit."""

    def __init__(self, space: ParameterSpace, tolerance: float):
        self.space = space
        self.operator = NeighborhoodOperator(space, 1)
        self.tolerance = float(tolerance)

    def inspect(self, center: tuple[int, ...], score_of: ScoreLookup, eligible_of: EligibilityLookup) -> PeakAnalysis:
        base = score_of(center)
        if base is None or not base.complete:
            return PeakAnalysis("incomplete", center, missing=(center,))
        level = float(base.j_score)
        queue = [center]
        plateau: set[tuple[int, ...]] = set()
        missing: set[tuple[int, ...]] = set()
        higher: tuple[int, ...] | None = None
        while queue and higher is None:
            point = queue.pop(0)
            if point in plateau:
                continue
            plateau.add(point)
            for neighbor in self.operator.points(point):
                if neighbor == point:
                    continue
                candidate = score_of(neighbor)
                if candidate is None or not candidate.complete:
                    missing.add(neighbor)
                    continue
                if not eligible_of(neighbor):
                    continue
                delta = float(candidate.j_score) - level
                if delta > self.tolerance:
                    higher = neighbor
                    break
                if abs(delta) <= self.tolerance and neighbor not in plateau:
                    queue.append(neighbor)
        if higher is not None:
            return PeakAnalysis("slope", center, higher=higher, plateau=tuple(sorted(plateau)), missing=tuple(sorted(missing)))
        if missing:
            return PeakAnalysis("incomplete", center, plateau=tuple(sorted(plateau)), missing=tuple(sorted(missing)))
        representative = min(plateau)
        return PeakAnalysis(
            "peak", center, representative=representative, plateau=tuple(sorted(plateau)),
            complete=True, local_peak=True, flat_region=len(plateau) > 1,
        )
