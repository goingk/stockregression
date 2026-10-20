"""Exact integer-tick parameter space for v4 research."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import product
import math
from typing import Any, Iterable, Iterator

from ..models import BacktestConfig, OptimizationConfig, SearchRange


FIELDS = ("buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares")
LABELS = {
    "buy_step": "买入间距",
    "sell_step": "卖出间距",
    "sell_shares": "每次卖出",
    "buy_shares": "每次买入",
    "min_shares": "最低持股",
    "max_shares": "最高持股",
}
UNITS = {"buy_step": 0.0001, "sell_step": 0.0001, "sell_shares": 100.0, "buy_shares": 100.0, "min_shares": 500.0, "max_shares": 500.0}


def _strict_tick(value: float, unit: float, name: str) -> int:
    try:
        scaled = Decimal(str(value)) / Decimal(str(unit))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{LABELS[name]}不是有效数值") from exc
    nearest = int(scaled.to_integral_value())
    if abs(scaled - Decimal(nearest)) > Decimal("0.0000001"):
        requirement = "0.01 个百分点" if name in {"buy_step", "sell_step"} else f"{int(unit)} 股 / 份"
        raise ValueError(f"{LABELS[name]}必须以 {requirement} 为单位")
    return nearest


def value_to_ticks(name: str, value: float) -> int:
    return _strict_tick(float(value), UNITS[name], name)


def ticks_to_value(name: str, value: int) -> float | int:
    return float(value) * UNITS[name] if name in {"buy_step", "sell_step"} else int(value) * int(UNITS[name])


@dataclass(frozen=True)
class Dimension:
    name: str
    low: int
    high: int
    unit: float

    @property
    def fixed(self) -> bool:
        return self.low == self.high

    @property
    def count(self) -> int:
        return self.high - self.low + 1

    @property
    def width(self) -> int:
        return max(0, self.high - self.low)


class ParameterSpace:
    """Validated inclusive lattice; all algorithms operate on integer ticks."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.base: BacktestConfig = config.base
        if self.base.grid_mode != "geometric":
            raise ValueError("v4 参数研究只支持等比网格")
        self.initial_shares = int(self.base.initial_shares)
        self.strategy = config.strategy
        dimensions: dict[str, Dimension] = {}
        unlimited = self.strategy.max_shares_mode == "unlimited"
        for name in FIELDS:
            if name == "max_shares" and unlimited:
                continue
            raw = config.ranges.get(name)
            if raw is None:
                raise ValueError(f"v4 研究缺少 {LABELS[name]} 搜索范围")
            search_range = raw if isinstance(raw, SearchRange) else SearchRange.model_validate(raw)
            low, high = value_to_ticks(name, search_range.low), value_to_ticks(name, search_range.high)
            if name in {"buy_step", "sell_step"}:
                minimum, maximum = 1, 10000
            elif name in ("sell_shares", "buy_shares"):
                minimum, maximum = 1, 10**9
            else:
                minimum, maximum = 0, 10**9
            low, high = max(low, minimum), min(high, maximum)
            if low > high:
                raise ValueError(f"{LABELS[name]}搜索范围无合法值")
            if name == "min_shares":
                high = min(high, self.initial_shares // int(UNITS[name]))
            elif name == "max_shares":
                low = max(low, math.ceil(self.initial_shares / UNITS[name]))
            if low > high:
                raise ValueError(f"{LABELS[name]}与固定初始持股不相容")
            dimensions[name] = Dimension(name, low, high, UNITS[name])
        self.dimensions = {name: dimensions[name] for name in FIELDS if name in dimensions}
        self.max_unlimited = unlimited
        self.fixed = {name: dim.low for name, dim in self.dimensions.items() if dim.fixed}
        self.active = tuple(name for name, dim in self.dimensions.items() if not dim.fixed)
        self.fields = tuple(self.dimensions)
        if self.max_unlimited:
            self.fixed["max_shares"] = -1
        special_buy_min = 2 if self.base.symbol.startswith(("688", "689")) else 1
        if "buy_shares" in self.dimensions:
            dim = self.dimensions["buy_shares"]
            if dim.low < special_buy_min:
                object.__setattr__(self, "dimensions", {**self.dimensions, "buy_shares": Dimension(dim.name, special_buy_min, dim.high, dim.unit)})
                dim = self.dimensions["buy_shares"]
            if dim.low > dim.high:
                raise ValueError("每次买入数量低于该标的最低买入量")
        self.active = tuple(name for name, dim in self.dimensions.items() if not dim.fixed)
        self.fixed = {name: dim.low for name, dim in self.dimensions.items() if dim.fixed}
        self._field_index = {name: index for index, name in enumerate(self.fields)}
        self._active_indices = tuple(self._field_index[name] for name in self.active)
        self._active_lows = tuple(self.dimensions[name].low for name in self.active)
        self._active_highs = tuple(self.dimensions[name].high for name in self.active)
        self._min_index = self._field_index.get("min_shares")
        self._max_index = self._field_index.get("max_shares")
        self._check_fixed_values()

    def _check_fixed_values(self) -> None:
        for name, tick in self.fixed.items():
            if name == "max_shares" and tick == -1:
                continue
            value = ticks_to_value(name, tick)
            if name == "min_shares" and int(value) > self.initial_shares:
                raise ValueError("最低持股高于固定初始持股")
            if name == "max_shares" and int(value) < self.initial_shares:
                raise ValueError("最高持股低于固定初始持股")
            if name == "buy_shares" and self.base.symbol.startswith(("688", "689")) and int(value) < 200:
                raise ValueError("科创板每次买入至少为 200 股")
        if "min_shares" in self.dimensions and "max_shares" in self.dimensions:
            if self.dimensions["min_shares"].low > self.dimensions["max_shares"].high:
                raise ValueError("最低持股范围与最高持股范围没有合法组合")

    def coords(self, name: str) -> tuple[int, ...]:
        dim = self.dimensions[name]
        return tuple(range(dim.low, dim.high + 1))

    def full_coordinate(self, active_coords: Iterable[int]) -> tuple[int, ...]:
        values = dict(zip(self.active, (int(value) for value in active_coords)))
        output = []
        for name in FIELDS:
            if name in values:
                output.append(values[name])
            elif name in self.fixed and self.fixed[name] >= 0:
                output.append(self.fixed[name])
            elif name == "max_shares" and self.max_unlimited:
                output.append(-1)
            else:
                raise KeyError(name)
        return tuple(output)

    def active_coords(self, coordinate: tuple[int, ...]) -> tuple[int, ...]:
        by_name = dict(zip(FIELDS, coordinate))
        return tuple(by_name[name] for name in self.active)

    def params(self, coordinate: tuple[int, ...]) -> dict[str, Any]:
        by_name = dict(zip(FIELDS, coordinate))
        params = {name: ticks_to_value(name, by_name[name]) for name in FIELDS if name in by_name and by_name[name] >= 0}
        params["initial_shares"] = self.initial_shares
        if self.max_unlimited:
            params["max_shares"] = None
        return params

    def coordinate(self, params: dict[str, Any]) -> tuple[int, ...]:
        values = {name: value_to_ticks(name, params[name]) for name in FIELDS if name in params and params[name] is not None}
        if self.max_unlimited:
            values["max_shares"] = -1
        return tuple(values[name] for name in FIELDS)

    def legal(self, active_coords: Iterable[int]) -> bool:
        vals = dict(zip(self.active, (int(value) for value in active_coords)))
        for name in self.active:
            if not self.dimensions[name].low <= vals[name] <= self.dimensions[name].high:
                return False
        if "min_shares" in vals and "max_shares" in vals and vals["min_shares"] > vals["max_shares"]:
            return False
        if "min_shares" not in vals and "max_shares" in vals and vals["max_shares"] * UNITS["max_shares"] < self.initial_shares:
            return False
        return True

    def iterate(self, limit: int | None = None) -> Iterator[tuple[int, ...]]:
        if not self.active:
            yield self.full_coordinate(())
            return
        emitted = 0
        for values in product(*(self.coords(name) for name in self.active)):
            if self.legal(values):
                yield self.full_coordinate(values)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def size(self) -> int:
        if not self.active:
            return 1
        axes = [self.dimensions[name].count for name in self.active]
        total = math.prod(axes)
        if not ({"min_shares", "max_shares"} <= set(self.active)):
            return total
        lows, highs = self.dimensions["min_shares"].count, self.dimensions["max_shares"].count
        min_start, max_start = self.dimensions["min_shares"].low, self.dimensions["max_shares"].low
        valid_pairs = 0
        for left in range(lows):
            absolute_low = min_start + left
            valid_pairs += min(highs, max(0, max_start + highs - absolute_low))
        others = math.prod(self.dimensions[name].count for name in self.active if name not in {"min_shares", "max_shares"})
        return valid_pairs * others

    def min_box_size(self, fields: Iterable[str] | None = None) -> int:
        selected = self.active if fields is None else tuple(name for name in self.active if name in set(fields))
        return 3 ** len(selected) if selected else 1

    def center_with_radius(self, coordinate: tuple[int, ...], coordinate_index: int) -> int | None:
        if coordinate_index < 0 or self.max_unlimited and coordinate_index >= len(self.fields):
            return None
        name = self.fields[coordinate_index]
        return coordinate[coordinate_index] if name in self.active else None

    def shift_inside(self, coordinate: tuple[int, ...], fields: Iterable[str] | None = None) -> tuple[int, ...]:
        values = list(coordinate)
        selected = set(self.active if fields is None else fields)
        for index, name in enumerate(self.fields):
            if name not in self.active or name not in selected:
                continue
            dim = self.dimensions[name]
            if dim.width < 2:
                continue
            values[index] = min(dim.high - 1, max(dim.low + 1, values[index]))
        if {"min_shares", "max_shares"} <= selected:
            min_index = self.fields.index("min_shares")
            max_index = self.fields.index("max_shares")
            min_dim, max_dim = self.dimensions["min_shares"], self.dimensions["max_shares"]
            if values[min_index] > values[max_index]:
                values[min_index] = min(min_dim.high - 1, max(min_dim.low + 1, values[max_index]))
                values[max_index] = max(max_dim.low + 1, min(max_dim.high - 1, values[min_index]))
        return tuple(values)

    def normalize(self, coordinate: tuple[int, ...]) -> tuple[int, ...]:
        return self.full_coordinate(self.active_coords(coordinate))

    def bounds_for(self, coordinate: tuple[int, ...], radii: dict[str, int]) -> dict[str, dict[str, Any]]:
        by_name = dict(zip(FIELDS, coordinate))
        bounds = {}
        for name in self.dimensions:
            radius = int(radii.get(name, 0))
            dim = self.dimensions[name]
            low, high = by_name[name] - radius, by_name[name] + radius
            bounds[name] = {
                "low": ticks_to_value(name, low) if low >= 0 else None,
                "high": ticks_to_value(name, high) if high >= 0 else None,
                "low_ticks": low,
                "high_ticks": high,
            }
        return bounds

    def normalized_volume(self, radii: dict[str, int]) -> float:
        if not self.active:
            return 0.0
        volume = 1.0
        for name in self.active:
            dim = self.dimensions[name]
            if dim.width <= 0:
                continue
            radius = int(radii.get(name, 0))
            if radius <= 0:
                continue
            volume *= min(1.0, 2.0 * radius * dim.unit / (dim.width * dim.unit))
        return float(volume)

    def neighbor_offsets(self, diagonal: bool = True, fields: Iterable[str] | None = None) -> Iterator[tuple[int, ...]]:
        if not self.active:
            return
        selected = set(self.active if fields is None else fields)
        indices = [index for index, name in enumerate(self.active) if name in selected]
        if not indices:
            return
        choices = (-1, 0, 1) if diagonal else (1,)
        for selected_offset in product(choices, repeat=len(indices)):
            if not any(selected_offset):
                continue
            offset = [0] * len(self.active)
            for index, value in zip(indices, selected_offset):
                offset[index] = value
            yield tuple(offset)

    def neighbor(self, coordinate: tuple[int, ...], offset: tuple[int, ...]) -> tuple[int, ...] | None:
        values = list(coordinate)
        for active_index, index in enumerate(self._active_indices):
            value = values[index] + offset[active_index]
            if value < self._active_lows[active_index] or value > self._active_highs[active_index]:
                return None
            values[index] = value
        if self._min_index is not None and self._max_index is not None and values[self._min_index] > values[self._max_index]:
            return None
        return tuple(values)
