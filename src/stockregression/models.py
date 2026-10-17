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
    # Unversioned records retain their original accounting. New clients send 2.
    schema_version: Literal[1, 2] = 1
    symbol: str
    start_date: date
    end_date: date | None = None
    frequency: Frequency = "1m"
    initial_shares: int = Field(default=1000, ge=0, multiple_of=100)
    min_shares: int = Field(default=0, ge=0, multiple_of=100)
    max_shares: int = Field(default=2000, ge=0, multiple_of=100)
    buy_shares: int = Field(default=100, ge=100, multiple_of=100)
    sell_shares: int = Field(default=100, ge=100, multiple_of=100)
    dca: DcaConfig = Field(default_factory=DcaConfig)
    reference_price_mode: Literal["auto", "manual"] = "auto"
    initial_cash: float = Field(default=100_000, gt=0)
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
        if isinstance(value, dict) and value.get("schema_version") == 2:
            value = dict(value)
            inactive = ("grid_step",) if value.get("grid_mode", "trailing") == "trailing" else ("buy_step", "sell_step")
            for key in inactive:
                value.pop(key, None)
        return value

    @model_serializer(mode="wrap")
    def serialize_config(self, handler):
        values = handler(self)
        legacy = {"initial_cash", "initial_position", "min_position", "max_position", "buy_quantity", "sell_quantity", "dca_frequency"}
        modern = {"initial_shares", "min_shares", "max_shares", "buy_shares", "sell_shares", "dca", "reference_price_mode", "reserve_cash", "selection_asof"}
        for key in legacy if self.schema_version == 2 else modern:
            values.pop(key, None)
        if self.schema_version == 2:
            for key in (("grid_step",) if self.grid_mode == "trailing" else ("buy_step", "sell_step")):
                values.pop(key, None)
        return values

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("结束日期不能早于开始日期")
        if not self.min_position <= self.initial_position <= self.max_position:
            raise ValueError("初始仓位必须位于仓位上下限之间")
        if self.schema_version == 2 and not self.min_shares <= self.initial_shares <= self.max_shares:
            raise ValueError("持股数量必须满足：最低持股 ≤ 初始持股 ≤ 最高持股")
        if self.schema_version == 2 and self.symbol.startswith(("688", "689")) and (self.buy_shares < 200 or self.dca.shares < 200):
            raise ValueError("科创板每次买入及定投至少为 200 股")
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


class OptimizationConfig(BaseModel):
    base: BacktestConfig
    protocol_version: Literal[1, 2] = 1
    selection_target: Literal["fixed", "adaptive", "both"] = "both"
    rebalance_days: int = Field(default=63, ge=1)
    bootstrap_reps: int = Field(default=10_000, ge=1)
    mode: Literal["historical", "robust", "both"] = "both"
    n_trials: int = Field(default=200, ge=1, le=5000)
    seed: int = 42
    top_candidates: int = Field(default=20, ge=1, le=100)
    max_drawdown: float = Field(default=0.2, gt=0, le=1)
    min_trades: int = Field(default=10, ge=0)
    min_history_days: int = Field(default=504, ge=20)
    min_minute_ratio: float = Field(default=0.95, ge=0, le=1)
    ranges: dict[str, SearchRange] = Field(default_factory=lambda: {
        "buy_step": SearchRange(low=0.005, high=0.10, step=0.005),
        "sell_step": SearchRange(low=0.005, high=0.10, step=0.005),
        "grid_step": SearchRange(low=0.005, high=0.10, step=0.005),
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
        base = values["base"]
        base = base if isinstance(base, BacktestConfig) else BacktestConfig.model_validate(base)
        if values.get("protocol_version", 1) == 2 and base.schema_version != 2:
            raise ValueError("protocol_version=2 需要 base.schema_version=2")
        if base.schema_version != 2:
            return values
        if "ranges" not in values:
            values = {**values, "ranges": {
                "buy_step": {"low": .005, "high": .1, "step": .005},
                "sell_step": {"low": .005, "high": .1, "step": .005},
                "grid_step": {"low": .005, "high": .1, "step": .005},
                "buy_shares": {"low": 100, "high": 1000, "step": 100},
                "sell_shares": {"low": 100, "high": 1000, "step": 100},
            }}
        inactive = {"grid_step"} if base.grid_mode == "trailing" else {"buy_step", "sell_step"}
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
