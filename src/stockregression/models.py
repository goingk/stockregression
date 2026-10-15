from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Frequency = Literal["1m", "5m", "15m", "30m", "60m", "1d"]


def normalize_symbol(value: str) -> str:
    value = value.strip().upper().replace("SH.", "").replace("SZ.", "")
    if value.startswith(("SH", "SZ")) and len(value) == 8:
        value = value[2:] + "." + value[:2]
    if "." not in value and len(value) == 6 and value.isdigit():
        value += ".SH" if value.startswith("6") else ".SZ"
    parts = value.split(".")
    if len(parts) != 2 or len(parts[0]) != 6 or not parts[0].isdigit():
        raise ValueError("股票代码格式应为 600000.SH 或 000001.SZ")
    if parts[1] not in ("SH", "SZ") or not parts[0].startswith(("0", "3", "6")):
        raise ValueError("首版仅支持沪深 A 股股票")
    if (parts[0].startswith("6")) != (parts[1] == "SH"):
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
    commission_rate: float = Field(default=0.0003, ge=0, le=0.05)
    min_commission: float = Field(default=5, ge=0)
    slippage_bps: float = Field(default=5, ge=0, le=1000)
    share_face_value: float = Field(default=1, gt=0)
    legacy_sh_transfer_rate: float = Field(default=0.001, ge=0, le=0.05)
    legacy_sh_min_transfer: float = Field(default=1, ge=0)
    legacy_sz_transfer_rate: float = Field(default=0.0000255, ge=0, le=0.05)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    start_date: date
    end_date: date | None = None
    frequency: Frequency = "1m"
    initial_cash: float = Field(default=100_000, gt=0)
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
    dca_frequency: Literal["weekly", "monthly"] = "monthly"

    _symbol = field_validator("symbol")(normalize_symbol)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("结束日期不能早于开始日期")
        if not self.min_position <= self.initial_position <= self.max_position:
            raise ValueError("初始仓位必须位于仓位上下限之间")
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
