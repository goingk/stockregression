"""Independent request contract for ETF diagnostics (not research protocol v5)."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import BacktestConfig, FeeConfig, normalize_symbol

METHOD_VERSION = "etf_min_grid_v5"
RESULT_SCHEMA_VERSION = 1
MIN_COMPLETE_DAY_RATIO = 0.90
MIN_OBSERVED_MINUTE_RATIO = 0.95


class EtfExecutionTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    initial_shares: int = Field(default=1000, ge=0, multiple_of=100)
    min_shares: int = Field(default=0, ge=0, multiple_of=100)
    max_shares: int | None = Field(default=2000, ge=0, multiple_of=100)
    buy_shares: int = Field(default=100, ge=100, multiple_of=100)
    sell_shares: int = Field(default=100, ge=100, multiple_of=100)
    fees: FeeConfig = Field(default_factory=FeeConfig)

    @model_validator(mode="after")
    def ordered(self):
        if self.min_shares > self.initial_shares:
            raise ValueError("最低持股不能超过初始持股")
        if self.max_shares is not None and self.initial_shares > self.max_shares:
            raise ValueError("初始持股不能超过最高持股")
        return self

    def backtest(self, symbol: str, start, end, **patch) -> BacktestConfig:
        return BacktestConfig.model_validate({
            "schema_version": 3, "symbol": symbol, "frequency": "1m",
            "start_date": start, "end_date": end, "grid_mode": "geometric",
            **self.model_dump(mode="json"), **patch,
        })


class EtfResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: Literal["watchlist", "local", "selected"] = "watchlist"
    symbols: list[str] = Field(default_factory=list, max_length=10000)
    as_of: date | None = None
    window_days: Literal[120, 250] = 120
    validation_top_k: int = Field(default=10, ge=0, le=100)
    execution: EtfExecutionTemplate = Field(default_factory=EtfExecutionTemplate)
    min_trades: int = Field(default=10, ge=0, le=1000000)
    max_principal_drawdown: float = Field(default=.2, gt=0, le=1)
    max_runtime_seconds: int = Field(default=3600, ge=1, le=86400)
    configuration_revision: str | None = None

    @field_validator("symbols")
    @classmethod
    def normalized(cls, values):
        symbols = list(dict.fromkeys(normalize_symbol(value) for value in values))
        if any(not symbol.startswith(("1", "5")) for symbol in symbols):
            raise ValueError("ETF 研究仅接受沪深场内 ETF")
        return symbols

    @model_validator(mode="after")
    def selected_symbols(self):
        if self.scope == "selected" and not self.symbols:
            raise ValueError("指定标的范围至少需要一只 ETF")
        if self.scope != "selected" and self.symbols:
            raise ValueError("显式证券代码须使用 selected 范围")
        return self
