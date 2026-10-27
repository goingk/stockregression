"""Versioned contracts for complete research validation, separate from v5 J."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import ResearchRequestV5, normalize_symbol


class Contribution(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    available_at: datetime
    amount: float = Field(gt=0)

    @field_validator("available_at")
    @classmethod
    def local_time(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            from zoneinfo import ZoneInfo
            value = value.astimezone(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
        return value


class QualificationAccount(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    total_budget: float = Field(gt=0)
    max_shares: int = Field(ge=100, multiple_of=100)
    contributions: list[Contribution] = Field(default_factory=list)
    participation_rate: float = Field(default=0.01, gt=0, le=1)
    ttl_bars: int = Field(default=3, ge=1, le=100)
    extra_delay_bars: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def funding_fits(self):
        if sum(item.amount for item in self.contributions) > self.total_budget:
            raise ValueError("计划补款总额不能超过总预算")
        self.contributions.sort(key=lambda item: item.available_at)
        return self


class ResearchValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    research: ResearchRequestV5
    symbols: list[str] = Field(default_factory=list, max_length=100)
    history_policy: Literal["auto", "explicit"] = "auto"
    accounts: dict[str, QualificationAccount]
    requested_data_versions: dict[str, str] = Field(default_factory=dict)
    suite_version: Literal["research_validation_v1"] = "research_validation_v1"
    primary_branch: Literal["adaptive"] = "adaptive"
    family_id: str | None = Field(default=None, max_length=128)
    parent_result_id: str | None = None
    plan_hash: str | None = None
    max_runtime_seconds: int = Field(default=86400, ge=1, le=2_592_000)
    total_evaluation_budget: int | None = Field(default=None, ge=1, le=1_000_000_000)

    @field_validator("symbols")
    @classmethod
    def canonical_symbols(cls, values):
        return list(dict.fromkeys(normalize_symbol(value) for value in values))

    @field_validator("accounts", "requested_data_versions")
    @classmethod
    def canonical_keys(cls, values):
        return {normalize_symbol(key): value for key, value in values.items()}

    @model_validator(mode="after")
    def complete_accounts(self):
        if not self.symbols:
            self.symbols = [self.research.research_context.symbol]
        source_version = self.research.requested_data_version
        source_symbol = self.research.research_context.symbol
        if source_version and source_symbol in self.symbols:
            if self.requested_data_versions.get(source_symbol, source_version) != source_version:
                raise ValueError("研究配置与验证请求的行情版本不一致")
            self.requested_data_versions[source_symbol] = source_version
        missing = [symbol for symbol in self.symbols if symbol not in self.accounts]
        if missing:
            raise ValueError("缺少标的预算设置：" + "、".join(missing))
        if self.history_policy == "explicit" and self.research.research_context.end_date is None:
            raise ValueError("显式历史方案需要明确结束日期")
        for symbol in self.symbols:
            if self.accounts[symbol].max_shares < self.research.research_context.initial_shares:
                raise ValueError(f"{symbol} 最高持仓不能低于初始持股")
        return self


class ValidationViewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section: str = Field(default="overview", min_length=1, max_length=100)


class ValidationResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_runtime_seconds: int | None = Field(default=None, ge=1, le=2_592_000)
    total_evaluation_budget: int | None = Field(default=None, ge=1, le=1_000_000_000)
