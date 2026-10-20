from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, model_serializer

Frequency = Literal["1m", "5m", "15m", "30m", "60m", "1d"]


def normalize_symbol(value: str) -> str:
    value = value.strip().upper().replace("SH.", "").replace("SZ.", "")
    if value.startswith(("SH", "SZ")) and len(value) == 8:
        value = value[2:] + "." + value[:2]
    if "." not in value and len(value) == 6 and value.isdigit():
        value += ".SH" if value.startswith(("5", "6")) else ".SZ"
    parts = value.split(".")
    if len(parts) != 2 or len(parts[0]) != 6 or not parts[0].isdigit():
        raise ValueError("股票代码格式应为 600000.SH 或 000001.SZ")
    if parts[1] not in ("SH", "SZ") or not parts[0].startswith(("0", "1", "3", "5", "6")):
        raise ValueError("仅支持沪深 A 股和场内 ETF")
    if (parts[0].startswith(("5", "6"))) != (parts[1] == "SH"):
        raise ValueError("股票代码与交易所不匹配")
    return value


class QuantityConfig(BaseModel):
    mode: Literal["shares", "cash", "equity_ratio"] = "equity_ratio"
    value: float = Field(default=0.1, gt=0)

    @model_validator(mode="after")
    def valid_quantity(self):
        if self.mode == "equity_ratio" and self.value > 1:
            raise ValueError("净资产比例必须在 0 到 1 之间")
        if self.mode == "shares" and self.value != int(self.value):
            raise ValueError("股数必须为整数")
        return self


class FeeConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    commission_rate: float = Field(default=0.0005, ge=0, le=0.05)
    min_commission: float = Field(default=0.1, ge=0)
    slippage_bps: float = Field(default=5, ge=0, le=1000)
    share_face_value: float = Field(default=1, gt=0)
    legacy_sh_transfer_rate: float = Field(default=0.001, ge=0, le=0.05)
    legacy_sh_min_transfer: float = Field(default=1, ge=0)
    legacy_sz_transfer_rate: float = Field(default=0.0000255, ge=0, le=0.05)


class DcaConfig(BaseModel):
    frequency: Literal["daily", "weekly", "monthly"] = "monthly"
    shares: int = Field(default=100, ge=100, multiple_of=100)
    initial_shares: int = Field(default=0, ge=0, multiple_of=100)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    # Unversioned records retain their original accounting. New clients send 3.
    schema_version: Literal[1, 2, 3] = 1
    symbol: str
    start_date: date
    end_date: date | None = None
    frequency: Frequency = "1m"
    initial_shares: int = Field(default=1000, ge=0, multiple_of=100)
    min_shares: int = Field(default=0, ge=0, multiple_of=100)
    max_shares: int | None = Field(default=2000, ge=0, multiple_of=100)
    buy_shares: int = Field(default=100, ge=100, multiple_of=100)
    sell_shares: int = Field(default=100, ge=100, multiple_of=100)
    dca: DcaConfig = Field(default_factory=DcaConfig)
    reference_price_mode: Literal["auto", "manual"] = "auto"
    initial_cash: float = Field(default=100_000, gt=0)
    # Kept for reading v1/v2 results and configurations. Version 3 derives
    # funding from successful buys and never accepts a cash budget.
    reserve_cash: float | None = Field(default=None, ge=0)
    selection_asof: date | None = None
    initial_position: float = Field(default=0.5, ge=0, le=1)
    min_position: float = Field(default=0, ge=0, le=1)
    max_position: float = Field(default=1, ge=0, le=1)
    grid_mode: Literal["trailing", "arithmetic", "geometric"] = "trailing"
    buy_step: float = Field(default=0.03, gt=0, le=1)
    sell_step: float = Field(default=0.03, gt=0, le=1)
    grid_step: float = Field(default=0.03, gt=0, le=1)
    reference_price: float | None = Field(default=None, gt=0)
    lower_price: float | None = Field(default=None, gt=0)
    upper_price: float | None = Field(default=None, gt=0)
    buy_quantity: QuantityConfig = Field(default_factory=QuantityConfig)
    sell_quantity: QuantityConfig = Field(default_factory=QuantityConfig)
    fees: FeeConfig = Field(default_factory=FeeConfig)
    dca_frequency: Literal["daily", "weekly", "monthly"] = "monthly"

    _symbol = field_validator("symbol")(normalize_symbol)

    @model_validator(mode="before")
    @classmethod
    def active_parameters(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            raw_symbol = str(value.get("symbol", "")).upper().replace("SH", "").replace("SZ", "").strip(".")
            is_etf = raw_symbol.startswith(("1", "5"))
            if "schema_version" not in value and (is_etf or any(key in value for key in ("initial_shares", "buy_shares", "dca"))):
                value["schema_version"] = 2
            if is_etf and value.get("schema_version") == 1:
                raise ValueError("ETF 请使用 schema_version=2 的股数配置")
        if isinstance(value, dict):
            value = dict(value)
            # ``grid_step`` was the single spacing used by older fixed grids.
            # New share-account configurations always carry both directions;
            # map the legacy value when an older record explicitly stored it.
            legacy = value.get("grid_step")
            if legacy is not None:
                if value.get("schema_version", 1) not in (2, 3) and value.get("grid_mode", "trailing") != "trailing":
                    value["buy_step"] = legacy
                    value["sell_step"] = legacy
                else:
                    value.setdefault("buy_step", legacy)
                    value.setdefault("sell_step", legacy)
        return value

    @model_serializer(mode="wrap")
    def serialize_config(self, handler):
        values = handler(self)
        legacy = {"initial_cash", "initial_position", "min_position", "max_position", "buy_quantity", "sell_quantity", "dca_frequency"}
        modern = {"initial_shares", "min_shares", "max_shares", "buy_shares", "sell_shares", "dca", "reference_price_mode", "reserve_cash", "selection_asof"}
        for key in legacy if self.schema_version in (2, 3) else modern:
            values.pop(key, None)
        if self.schema_version == 3:
            values.pop("reserve_cash", None)
        if self.schema_version in (2, 3):
            values.pop("grid_step", None)
        return values

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("结束日期不能早于开始日期")
        if not self.min_position <= self.initial_position <= self.max_position:
            raise ValueError("初始仓位必须位于仓位上下限之间")
        if self.schema_version in (2, 3) and self.min_shares > self.initial_shares:
            raise ValueError("持股数量必须满足：最低持股 ≤ 初始持股 ≤ 最高持股")
        if self.schema_version in (2, 3) and self.max_shares is not None and self.initial_shares > self.max_shares:
            raise ValueError("持股数量必须满足：最低持股 ≤ 初始持股 ≤ 最高持股")
        if self.schema_version in (2, 3) and self.symbol.startswith(("688", "689")) and (self.buy_shares < 200 or self.dca.shares < 200):
            raise ValueError("科创板每次买入及定投至少为 200 股")
        if self.schema_version == 3 and self.reserve_cash is not None:
            raise ValueError("schema_version=3 已取消固定可用现金；请通过配置转换生成 v3 草稿")
        if self.lower_price and self.upper_price and self.lower_price >= self.upper_price:
            raise ValueError("网格价格下限必须小于上限")
        return self


class SearchRange(BaseModel):
    low: float
    high: float
    step: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.high < self.low:
            raise ValueError("搜索区间上限必须不小于下限")
        return self


NeighborhoodField = Literal["buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares"]


class ResearchStrategySpec(BaseModel):
    """Versioned research algorithm configuration."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    strategy_id: Literal["geometric_excess_neighborhood"] = "geometric_excess_neighborhood"
    strategy_version: Literal[1, 2] = 2
    search_method: Literal["auto", "hierarchical", "exhaustive"] = "auto"
    ranking_mode: Literal["max_volume", "fixed_mean"] = "max_volume"
    training_budget: int = Field(default=20_000, ge=1, le=1_000_000)
    validation_budget: int = Field(default=100_000, ge=1, le=2_000_000)
    recommendation_count: int = Field(default=10, ge=1, le=100)
    max_shares_mode: Literal["search", "unlimited"] = "search"
    slippage_mode: Literal["configured", "zero"] = "configured"
    neighborhood_fields: list[NeighborhoodField] = Field(default_factory=lambda: [
        "buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares",
    ])
    fixed_half_widths: dict[str, float] = Field(default_factory=lambda: {
        "buy_step": 0.001,
        "sell_step": 0.001,
        "sell_shares": 100,
        "buy_shares": 100,
        "min_shares": 500,
        "max_shares": 500,
    })

    @model_validator(mode="before")
    @classmethod
    def migrate_single_spacing(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if "strategy_version" not in value and value.get("ranking_mode") == "fixed_mean":
            value["strategy_version"] = 1
        fields = value.get("neighborhood_fields")
        if isinstance(fields, list) and "grid_step" in fields:
            migrated: list[str] = []
            for name in fields:
                names = ("buy_step", "sell_step") if name == "grid_step" else (str(name),)
                for mapped in names:
                    if mapped not in migrated:
                        migrated.append(mapped)
            value["neighborhood_fields"] = migrated
        half_widths = value.get("fixed_half_widths")
        if isinstance(half_widths, dict):
            migrated_widths = dict(half_widths)
            legacy = migrated_widths.pop("grid_step", None)
            if legacy is not None:
                migrated_widths.setdefault("buy_step", legacy)
                migrated_widths.setdefault("sell_step", legacy)
            value["fixed_half_widths"] = migrated_widths
        return value

    @model_validator(mode="after")
    def valid_half_widths(self):
        if self.strategy_version == 2 and self.ranking_mode != "max_volume":
            raise ValueError("连通域策略 v2 仅支持最大正收益连通域；固定邻域请使用策略 v1")
        if not self.neighborhood_fields:
            raise ValueError("细搜索邻域至少需要一个可移动参数")
        if len(set(self.neighborhood_fields)) != len(self.neighborhood_fields):
            raise ValueError("细搜索邻域参数不能重复")
        allowed = {"buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares"}
        if set(self.fixed_half_widths) - allowed:
            raise ValueError("固定邻域半宽包含不支持的参数")
        for name, value in self.fixed_half_widths.items():
            if float(value) <= 0:
                raise ValueError("固定邻域半宽必须为正数")
            if name in {"buy_step", "sell_step"}:
                scaled = round(float(value) / 0.0001)
                if abs(float(value) / 0.0001 - scaled) > 1e-7:
                    raise ValueError("买卖间距半宽必须对齐 0.0001")
            else:
                unit = 500 if name in {"min_shares", "max_shares"} else 100
                if abs(float(value) / unit - round(float(value) / unit)) > 1e-7:
                    raise ValueError(f"{name} 邻域半宽必须对齐 {unit}")
        return self


class ResearchContext(BaseModel):
    """Public v4 research input independent of backtest search values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    symbol: str
    start_date: date
    end_date: date | None = None
    frequency: Frequency = "1m"
    initial_shares: int = Field(default=1000, ge=0, multiple_of=100)
    grid_mode: Literal["geometric"] = "geometric"
    fees: FeeConfig = Field(default_factory=FeeConfig)
    reference_price_mode: Literal["auto", "manual"] = "auto"
    reference_price: float | None = Field(default=None, gt=0)
    lower_price: float | None = Field(default=None, gt=0)
    upper_price: float | None = Field(default=None, gt=0)
    dca: DcaConfig = Field(default_factory=DcaConfig)

    _symbol = field_validator("symbol")(normalize_symbol)


class CandidatePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    plan_hash: str
    fields: list[str]
    units: dict[str, float]
    training_budget: int
    validation_budget: int
    validation_coordinates: list[list[int]]
    source_regions: list[int] = Field(default_factory=list)
    global_exhaustive: bool = False


class RecommendationRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    region_id: int | None = None
    center_params: dict[str, Any]
    center_coordinate: list[int] = Field(default_factory=list)
    candidate_id: int
    platform_id: int | None = None
    bounds: dict[str, dict[str, Any]]
    radii: dict[str, int]
    normalized_volume: float
    raw_volume: int = 0
    combination_count: int
    completed_count: int
    center_excess_return: float | None = None
    mean_excess_return: float | None = None
    min_excess_return: float | None = None
    complete: bool
    maximum_search_complete: bool
    global_exhaustive: bool
    overlapping_region_ids: list[int] = Field(default_factory=list)
    stress: dict[str, Any] = Field(default_factory=dict)


class ResearchRequestV4(BaseModel):
    """Standalone public request that does not carry backtest search defaults."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    research_context: ResearchContext
    ranges: dict[str, SearchRange]
    strategy: ResearchStrategySpec = Field(default_factory=ResearchStrategySpec)
    selection_target: Literal["fixed", "adaptive", "both"] = "both"
    rebalance_days: int = Field(default=63, ge=1)
    bootstrap_reps: int = Field(default=10_000, ge=1)
    seed: int = 42
    top_candidates: int = Field(default=20, ge=1, le=100)
    max_drawdown: float = Field(default=0.2, gt=0, le=1)
    min_trades: int = Field(default=10, ge=0)
    min_history_days: int = Field(default=504, ge=20)
    min_minute_ratio: float = Field(default=0.95, ge=0, le=1)

    def to_optimization(self) -> OptimizationConfig:
        context = self.research_context
        base = BacktestConfig(
            schema_version=3, symbol=context.symbol, start_date=context.start_date,
            end_date=context.end_date, frequency=context.frequency,
            initial_shares=context.initial_shares, min_shares=0,
            max_shares=None if self.strategy.max_shares_mode == "unlimited" else max(2000, context.initial_shares),
            buy_shares=100, sell_shares=100, grid_mode="geometric",
            buy_step=.03, sell_step=.03,
            fees=context.fees, reference_price_mode=context.reference_price_mode,
            reference_price=context.reference_price, lower_price=context.lower_price,
            upper_price=context.upper_price, dca=context.dca,
        )
        return OptimizationConfig(
            base=base, protocol_version=4, selection_target=self.selection_target,
            rebalance_days=self.rebalance_days, bootstrap_reps=self.bootstrap_reps,
            seed=self.seed, top_candidates=self.top_candidates, max_drawdown=self.max_drawdown,
            min_trades=self.min_trades, min_history_days=self.min_history_days,
            min_minute_ratio=self.min_minute_ratio, strategy=self.strategy,
            research_context=self.research_context, ranges=self.ranges,
        )


class OptimizationConfig(BaseModel):
    base: BacktestConfig
    protocol_version: Literal[1, 2, 3, 4] = 1
    selection_target: Literal["fixed", "adaptive", "both"] = "both"
    rebalance_days: int = Field(default=63, ge=1)
    bootstrap_reps: int = Field(default=10_000, ge=1)
    mode: Literal["historical", "robust", "both"] = "both"
    n_trials: int = Field(default=200, ge=1, le=5000)
    local_search_budget: int = Field(default=3000, ge=1, le=5000)
    seed: int = 42
    top_candidates: int = Field(default=20, ge=1, le=100)
    max_drawdown: float = Field(default=0.2, gt=0, le=1)
    min_trades: int = Field(default=10, ge=0)
    min_history_days: int = Field(default=504, ge=20)
    min_minute_ratio: float = Field(default=0.95, ge=0, le=1)
    strategy: ResearchStrategySpec = Field(default_factory=ResearchStrategySpec)
    research_context: ResearchContext | None = None
    ranges: dict[str, SearchRange] = Field(default_factory=lambda: {
        "buy_step": SearchRange(low=0.005, high=0.10, step=0.005),
        "sell_step": SearchRange(low=0.005, high=0.10, step=0.005),
        "buy_ratio": SearchRange(low=0.02, high=0.20, step=0.02),
        "sell_ratio": SearchRange(low=0.02, high=0.20, step=0.02),
        "initial_position": SearchRange(low=0.20, high=0.80, step=0.10),
        "min_position": SearchRange(low=0, high=0.40, step=0.10),
        "max_position": SearchRange(low=0.60, high=1, step=0.10),
    })

    @model_validator(mode="before")
    @classmethod
    def share_ranges(cls, values):
        if not isinstance(values, dict) or "base" not in values:
            return values
        values = dict(values)
        if values.get("protocol_version", 1) == 2:
            if "n_trials" not in values:
                values["n_trials"] = 1000
        if values.get("protocol_version", 1) == 3 and "local_search_budget" not in values:
            values["local_search_budget"] = 3000
        if values.get("protocol_version", 1) == 4:
            context = values.get("research_context")
            raw_base = values["base"]
            base_values = raw_base if isinstance(raw_base, dict) else raw_base.model_dump(mode="python")
            base_values = dict(base_values)
            if context is not None:
                context = context if isinstance(context, ResearchContext) else ResearchContext.model_validate(context)
                context_values = context.model_dump(mode="python")
                fees = context_values.pop("fees")
                base_values.update(context_values)
                base_values["fees"] = fees
            base_values["schema_version"] = 3
            base_values["grid_mode"] = "geometric"
            strategy_raw = values.get("strategy") or {}
            strategy = strategy_raw if isinstance(strategy_raw, ResearchStrategySpec) else ResearchStrategySpec.model_validate(strategy_raw)
            if strategy.max_shares_mode == "unlimited":
                base_values["max_shares"] = None
            elif base_values.get("max_shares") is None:
                raw_ranges = values.get("ranges") or {}
                maximum = raw_ranges.get("max_shares") if isinstance(raw_ranges, dict) else None
                high = maximum.get("high") if isinstance(maximum, dict) else getattr(maximum, "high", None)
                base_values["max_shares"] = int(high) if high is not None else int(base_values.get("initial_shares", 1000))
            if strategy.slippage_mode == "zero":
                fees = dict(base_values.get("fees") or {})
                fees["slippage_bps"] = 0
                base_values["fees"] = fees
            if values.get("research_context") is None:
                values["research_context"] = {
                    "symbol": base_values["symbol"], "start_date": base_values["start_date"],
                    "end_date": base_values.get("end_date"), "frequency": base_values.get("frequency", "1m"),
                    "initial_shares": base_values.get("initial_shares", 1000), "grid_mode": "geometric",
                    "fees": base_values.get("fees") or {},
                    "reference_price_mode": base_values.get("reference_price_mode", "auto"),
                    "reference_price": base_values.get("reference_price"),
                    "lower_price": base_values.get("lower_price"), "upper_price": base_values.get("upper_price"),
                    "dca": base_values.get("dca") or {},
                }
            values["base"] = base_values
        base = values["base"]
        base = base if isinstance(base, BacktestConfig) else BacktestConfig.model_validate(base)
        if values.get("protocol_version", 1) == 2 and base.schema_version != 2:
            raise ValueError("protocol_version=2 需要 base.schema_version=2")
        if values.get("protocol_version", 1) in (3, 4) and base.schema_version != 3:
            raise ValueError("protocol_version=3/4 需要 base.schema_version=3")
        if base.schema_version not in (2, 3):
            return values
        raw_ranges = values.get("ranges")
        if isinstance(raw_ranges, dict) and "grid_step" in raw_ranges:
            ranges = dict(raw_ranges)
            legacy = ranges.pop("grid_step")
            ranges.setdefault("buy_step", legacy)
            ranges.setdefault("sell_step", legacy)
            values["ranges"] = ranges
        if "ranges" not in values:
            values = {**values, "ranges": {
                "buy_step": {"low": .005, "high": .1, "step": .005},
                "sell_step": {"low": .005, "high": .1, "step": .005},
                "buy_shares": {"low": 100, "high": 1000, "step": 100},
                "sell_shares": {"low": 100, "high": 1000, "step": 100},
                "min_shares": {"low": 0, "high": 1000, "step": 100},
                "max_shares": {"low": 1000, "high": 5000, "step": 100},
            }}
        if values.get("protocol_version") == 4:
            inactive = {"buy_ratio", "sell_ratio", "initial_position", "min_position", "max_position", "initial_shares"}
        else:
            inactive: set[str] = set()
        if isinstance(values["ranges"], dict):
            values = {**values, "ranges": {key: value for key, value in values["ranges"].items() if key not in inactive}}
        return values


@dataclass
class DataBundle:
    """Canonical unadjusted data; timestamps are naive Asia/Shanghai bar ends.

    bars: timestamp, open, high, low, close, volume, turnover, frequency,
          tradable, is_synthetic, preclose, is_st, limit_up, limit_down.
    daily: date, open, high, low, close, volume, turnover, preclose,
           tradestatus, is_st. date is a pandas Timestamp.
    actions: record_date, ex_date, pay_date, stock_date, cash_per_share,
             bonus_ratio, implemented, unsupported_reason.
    calendar: expected exchange trading days within the requested interval.
    """
    symbol: str
    bars: pd.DataFrame
    daily: pd.DataFrame
    actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    calendar: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    quality: dict[str, Any] = field(default_factory=dict)
    version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
