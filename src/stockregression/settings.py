from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(os.environ.get("STOCKREGRESSION_HOME", Path(__file__).resolve().parents[2])).resolve()
SETTINGS_FILE = PROJECT_ROOT / ".local" / "settings.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_DIR = DEFAULT_DATA_DIR / "dataset"
DEFAULT_ETF_PACKAGE_DIR = DEFAULT_DATASET_DIR / "etf"
DEFAULT_STOCK_PACKAGE_DIR = DEFAULT_DATASET_DIR / "stock"
DEFAULT_PACKAGE_DIR = DEFAULT_ETF_PACKAGE_DIR


class ScheduleSettings(BaseModel):
    enabled: bool = False
    daily_time: str = "18:30"
    weekly_day: str = "SAT"
    weekly_time: str = "10:00"


class FeeDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    commission_rate: float = Field(default=0.0005, ge=0, le=0.05)
    min_commission: float = Field(default=0.1, ge=0)


class _WorkbenchPreference(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GridDefaults(_WorkbenchPreference):
    grid_mode: Literal["trailing", "arithmetic", "geometric"] = "trailing"
    buy_step: float = Field(default=0.03, gt=0, le=1)
    sell_step: float = Field(default=0.03, gt=0, le=1)
    lower_price: float | None = Field(default=None, gt=0)
    upper_price: float | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_single_spacing(cls, values):
        if isinstance(values, dict) and "grid_step" in values:
            values = dict(values)
            legacy = values.pop("grid_step")
            if values.get("grid_mode", "trailing") != "trailing":
                values["buy_step"] = legacy
                values["sell_step"] = legacy
            else:
                values.setdefault("buy_step", legacy)
                values.setdefault("sell_step", legacy)
        return values

    @model_validator(mode="after")
    def ordered_prices(self):
        if self.lower_price is not None and self.upper_price is not None and self.lower_price >= self.upper_price:
            raise ValueError("网格价格下限必须小于上限")
        return self


class PositionDefaults(_WorkbenchPreference):
    initial_shares: int = Field(default=1000, ge=0, multiple_of=100)
    min_shares: int = Field(default=0, ge=0, multiple_of=100)
    max_shares: int | None = Field(default=2000, ge=0, multiple_of=100)
    buy_shares: int = Field(default=100, ge=100, multiple_of=100)
    sell_shares: int = Field(default=100, ge=100, multiple_of=100)

    @model_validator(mode="after")
    def ordered_positions(self):
        if self.max_shares is not None and not self.min_shares <= self.initial_shares <= self.max_shares:
            raise ValueError("最低持股 ≤ 初始持股 ≤ 最高持股")
        if self.min_shares > self.initial_shares:
            raise ValueError("最低持股 ≤ 初始持股")
        return self


class DcaDefaults(_WorkbenchPreference):
    frequency: Literal["daily", "weekly", "monthly"] = "monthly"
    shares: int = Field(default=100, ge=100, multiple_of=100)
    initial_shares: int = Field(default=0, ge=0, multiple_of=100)


class ResearchRangeDefaults(_WorkbenchPreference):
    low: float
    high: float
    step: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.high < self.low:
            raise ValueError("搜索区间上限必须不小于下限")
        return self


_RESEARCH_RANGE_DEFAULTS = {
    "buy_step": {"low": 0.005, "high": 0.1, "step": 0.005},
    "sell_step": {"low": 0.005, "high": 0.1, "step": 0.005},
    "buy_shares": {"low": 100, "high": 1000, "step": 100},
    "sell_shares": {"low": 100, "high": 1000, "step": 100},
    "initial_shares": {"low": 0, "high": 2000, "step": 100},
    "min_shares": {"low": 0, "high": 1000, "step": 100},
    "max_shares": {"low": 1000, "high": 5000, "step": 100},
}


class ResearchDefaults(_WorkbenchPreference):
    protocol_version: Literal[1, 2, 3, 4] = 4
    selection_target: Literal["fixed", "adaptive", "both"] = "both"
    rebalance_days: int = Field(default=63, ge=1)
    bootstrap_reps: int = Field(default=10_000, ge=1)
    mode: Literal["historical", "robust", "both"] = "both"
    n_trials: int = Field(default=1000, ge=1, le=5000)
    local_search_budget: int = Field(default=3000, ge=1, le=5000)
    seed: int = 42
    top_candidates: int = Field(default=20, ge=1, le=100)
    max_drawdown: float = Field(default=0.2, gt=0, le=1)
    min_trades: int = Field(default=10, ge=0)
    min_history_days: int = Field(default=504, ge=20)
    min_minute_ratio: float = Field(default=0.95, ge=0, le=1)
    ranking_mode: Literal["max_volume", "fixed_mean"] = "max_volume"
    search_method: Literal["auto", "hierarchical", "exhaustive"] = "auto"
    training_budget: int = Field(default=20_000, ge=1, le=1_000_000)
    validation_budget: int = Field(default=100_000, ge=1, le=2_000_000)
    recommendation_count: int = Field(default=10, ge=1, le=100)
    max_shares_mode: Literal["search", "unlimited"] = "search"
    slippage_mode: Literal["configured", "zero"] = "configured"
    neighborhood_fields: list[Literal[
        "buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares"
    ]] = Field(default_factory=lambda: [
        "buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares",
    ])
    fixed_half_widths: dict[str, float] = Field(default_factory=lambda: {
        "buy_step": 0.001, "sell_step": 0.001, "sell_shares": 100, "buy_shares": 100,
        "min_shares": 500, "max_shares": 500,
    })
    ranges: dict[str, ResearchRangeDefaults] = Field(default_factory=lambda: {
        key: ResearchRangeDefaults.model_validate(value) for key, value in _RESEARCH_RANGE_DEFAULTS.items()
    })

    @model_validator(mode="before")
    @classmethod
    def migrate_protocol(cls, values):
        if isinstance(values, dict):
            values = dict(values)
            if isinstance(values.get("ranges"), dict):
                ranges = dict(values["ranges"])
                legacy = ranges.pop("grid_step", None)
                if legacy is not None:
                    ranges["buy_step"] = legacy
                    ranges["sell_step"] = legacy
                values["ranges"] = ranges
            if isinstance(values.get("neighborhood_fields"), list):
                fields = values["neighborhood_fields"]
                if "grid_step" in fields:
                    migrated_fields = []
                    for name in fields:
                        for mapped in (("buy_step", "sell_step") if name == "grid_step" else (name,)):
                            if mapped not in migrated_fields:
                                migrated_fields.append(mapped)
                    values["neighborhood_fields"] = migrated_fields
            if isinstance(values.get("fixed_half_widths"), dict):
                saved_widths = dict(values["fixed_half_widths"])
                legacy = saved_widths.pop("grid_step", None)
                merged_widths = {
                    "buy_step": .001, "sell_step": .001, "sell_shares": 100, "buy_shares": 100,
                    "min_shares": 500, "max_shares": 500,
                    **saved_widths,
                }
                if legacy is not None:
                    merged_widths["buy_step"] = legacy
                    merged_widths["sell_step"] = legacy
                values["fixed_half_widths"] = merged_widths
            if isinstance(values.get("fixed_half_widths"), dict):
                for name in ("min_shares", "max_shares"):
                    raw = values["fixed_half_widths"].get(name, 500)
                    if isinstance(raw, (int, float)) and raw > 0:
                        values["fixed_half_widths"][name] = max(500, math.ceil(raw / 500) * 500)
            if values.get("protocol_version") in (1, 2, 3):
                values["protocol_version"] = 4
        return values

    @model_validator(mode="after")
    def valid_ranges(self):
        expected = set(_RESEARCH_RANGE_DEFAULTS)
        if set(self.ranges) != expected:
            raise ValueError("搜索范围必须包含全部七个支持参数")
        for name, value in self.ranges.items():
            if name.endswith("_shares"):
                minimum = 100 if name in {"buy_shares", "sell_shares"} else 0
                if value.low < minimum or any(number % 100 for number in (value.low, value.high, value.step)):
                    raise ValueError(f"股数搜索范围 {name} 必须以 100 为单位，交易量至少 100")
            else:
                if value.low <= 0 or value.high > 1:
                    raise ValueError(f"网格间距搜索范围 {name} 必须位于 (0, 1]")
        return self

    @model_validator(mode="after")
    def valid_fixed_half_widths(self):
        allowed = {"buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares"}
        if set(self.fixed_half_widths) != allowed:
            raise ValueError("固定邻域半宽必须包含六个 v4 参数")
        if any(float(value) <= 0 for value in self.fixed_half_widths.values()):
            raise ValueError("固定邻域半宽必须为正数")
        if any(abs(float(self.fixed_half_widths[name]) / 0.001 - round(float(self.fixed_half_widths[name]) / 0.001)) > 1e-7 for name in ("buy_step", "sell_step")):
            raise ValueError("买卖间距半宽必须对齐 0.001")
        if any(abs(float(self.fixed_half_widths[name]) / (500 if name in {"min_shares", "max_shares"} else 100) - round(float(self.fixed_half_widths[name]) / (500 if name in {"min_shares", "max_shares"} else 100))) > 1e-7 for name in ("sell_shares", "buy_shares", "min_shares", "max_shares")):
            raise ValueError("交易数量半宽必须对齐 100，持股边界半宽必须对齐 500")
        return self

    @model_validator(mode="after")
    def valid_neighborhood_fields(self):
        if not self.neighborhood_fields:
            raise ValueError("细搜索邻域至少需要一个可移动参数")
        if len(set(self.neighborhood_fields)) != len(self.neighborhood_fields):
            raise ValueError("细搜索邻域参数不能重复")
        return self


class WorkbenchDefaults(_WorkbenchPreference):
    grid: GridDefaults = Field(default_factory=GridDefaults)
    position: PositionDefaults = Field(default_factory=PositionDefaults)
    dca: DcaDefaults = Field(default_factory=DcaDefaults)
    research: ResearchDefaults = Field(default_factory=ResearchDefaults)


class PerformanceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_budget_mode: Literal["auto", "custom"] = "auto"
    cpu_budget: int | None = Field(default=None, ge=1, le=128)
    download_workers: int = Field(default=4, ge=1, le=8)
    validation_workers: int = Field(default=4, ge=1, le=4)
    blas_threads: int = Field(default=8, ge=1, le=16)
    workspace_mb: int = Field(default=8192, ge=256, le=65536)

    @model_validator(mode="after")
    def valid_custom_budget(self):
        if self.cpu_budget_mode == "custom" and self.cpu_budget is None:
            raise ValueError("指定计算预算时必须填写 CPU 线程数")
        return self


class Settings(BaseModel):
    data_dir: str = str(DEFAULT_DATA_DIR)
    package_dir: str | None = str(DEFAULT_PACKAGE_DIR)
    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings)
    fee_defaults: FeeDefaults = Field(default_factory=FeeDefaults)
    workbench_defaults: WorkbenchDefaults = Field(default_factory=WorkbenchDefaults)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    host: str = "127.0.0.1"
    port: int = 8000
    max_workers: int = Field(default=2, ge=1, le=8)
    market_cache_mb: int = Field(default=256, ge=0, le=16384)
    recent_minute_source: Literal["sina", "tdx"] = "sina"
    etf_daily_source: Literal["baostock", "eastmoney"] = "baostock"


def load_settings() -> Settings:
    values = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) if SETTINGS_FILE.exists() else {}
    if os.environ.get("STOCKREGRESSION_DATA_DIR"):
        values["data_dir"] = os.environ["STOCKREGRESSION_DATA_DIR"]
    if os.environ.get("STOCKREGRESSION_PACKAGE_DIR"):
        values["package_dir"] = os.environ["STOCKREGRESSION_PACKAGE_DIR"]
    settings = Settings.model_validate(values)
    settings.data_dir = str(Path(settings.data_dir).expanduser().resolve())
    if settings.package_dir:
        settings.package_dir = str(Path(settings.package_dir).expanduser().resolve())
    return settings


def save_settings(values: dict) -> Settings:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(SETTINGS_FILE.with_suffix(".lock")), timeout=10):
        previous = load_settings().model_dump()
        values = dict(values)
        for key in ("schedule", "fee_defaults", "performance"):
            if key in values and isinstance(values[key], dict):
                values[key] = {**previous[key], **values[key]}
        if isinstance(values.get("workbench_defaults"), dict):
            merged_defaults = dict(previous["workbench_defaults"])
            for group, patch in values["workbench_defaults"].items():
                if isinstance(patch, dict) and isinstance(merged_defaults.get(group), dict):
                    merged_group = {**merged_defaults[group], **patch}
                    if group == "research" and isinstance(patch.get("ranges"), dict):
                        merged_ranges = dict(merged_defaults[group]["ranges"])
                        for name, range_patch in patch["ranges"].items():
                            if isinstance(range_patch, dict) and isinstance(merged_ranges.get(name), dict):
                                merged_ranges[name] = {**merged_ranges[name], **range_patch}
                            else:
                                merged_ranges[name] = range_patch
                        merged_group["ranges"] = merged_ranges
                    merged_defaults[group] = merged_group
                else:
                    merged_defaults[group] = patch
            values["workbench_defaults"] = merged_defaults
        settings = Settings.model_validate({**previous, **values})
        settings.data_dir = str(Path(settings.data_dir).expanduser().resolve())
        if settings.package_dir:
            settings.package_dir = str(Path(settings.package_dir).expanduser().resolve())
        temporary = SETTINGS_FILE.with_suffix(".writing")
        temporary.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(SETTINGS_FILE)
    return settings
