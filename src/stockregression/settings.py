from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(os.environ.get("STOCKREGRESSION_HOME", Path(__file__).resolve().parents[2])).resolve()
SETTINGS_FILE = PROJECT_ROOT / ".local" / "settings.json"


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
    grid_step: float = Field(default=0.03, gt=0, le=1)
    lower_price: float | None = Field(default=None, gt=0)
    upper_price: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def ordered_prices(self):
        if self.lower_price is not None and self.upper_price is not None and self.lower_price >= self.upper_price:
            raise ValueError("网格价格下限必须小于上限")
        return self


class PositionDefaults(_WorkbenchPreference):
    initial_shares: int = Field(default=1000, ge=0, multiple_of=100)
    min_shares: int = Field(default=0, ge=0, multiple_of=100)
    max_shares: int = Field(default=2000, ge=0, multiple_of=100)
    buy_shares: int = Field(default=100, ge=100, multiple_of=100)
    sell_shares: int = Field(default=100, ge=100, multiple_of=100)

    @model_validator(mode="after")
    def ordered_positions(self):
        if not self.min_shares <= self.initial_shares <= self.max_shares:
            raise ValueError("最低持股 ≤ 初始持股 ≤ 最高持股")
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
    "grid_step": {"low": 0.005, "high": 0.1, "step": 0.005},
    "buy_shares": {"low": 100, "high": 1000, "step": 100},
    "sell_shares": {"low": 100, "high": 1000, "step": 100},
    "initial_shares": {"low": 0, "high": 2000, "step": 100},
    "min_shares": {"low": 0, "high": 1000, "step": 100},
    "max_shares": {"low": 1000, "high": 5000, "step": 100},
}


class ResearchDefaults(_WorkbenchPreference):
    protocol_version: Literal[1, 2] = 2
    selection_target: Literal["fixed", "adaptive", "both"] = "both"
    rebalance_days: int = Field(default=63, ge=1)
    bootstrap_reps: int = Field(default=10_000, ge=1)
    mode: Literal["historical", "robust", "both"] = "both"
    n_trials: int = Field(default=1000, ge=1, le=5000)
    seed: int = 42
    top_candidates: int = Field(default=20, ge=1, le=100)
    max_drawdown: float = Field(default=0.2, gt=0, le=1)
    min_trades: int = Field(default=10, ge=0)
    min_history_days: int = Field(default=504, ge=20)
    min_minute_ratio: float = Field(default=0.95, ge=0, le=1)
    ranges: dict[str, ResearchRangeDefaults] = Field(default_factory=lambda: {
        key: ResearchRangeDefaults.model_validate(value) for key, value in _RESEARCH_RANGE_DEFAULTS.items()
    })

    @model_validator(mode="after")
    def valid_ranges(self):
        expected = set(_RESEARCH_RANGE_DEFAULTS)
        if set(self.ranges) != expected:
            raise ValueError("搜索范围必须包含全部八个支持参数")
        for name, value in self.ranges.items():
            if name.endswith("_shares"):
                minimum = 100 if name in {"buy_shares", "sell_shares"} else 0
                if value.low < minimum or any(number % 100 for number in (value.low, value.high, value.step)):
                    raise ValueError(f"股数搜索范围 {name} 必须以 100 为单位，交易量至少 100")
            else:
                if value.low <= 0 or value.high > 1:
                    raise ValueError(f"网格步长搜索范围 {name} 必须位于 (0, 1]")
        return self


class WorkbenchDefaults(_WorkbenchPreference):
    grid: GridDefaults = Field(default_factory=GridDefaults)
    position: PositionDefaults = Field(default_factory=PositionDefaults)
    dca: DcaDefaults = Field(default_factory=DcaDefaults)
    research: ResearchDefaults = Field(default_factory=ResearchDefaults)


class Settings(BaseModel):
    data_dir: str = str(PROJECT_ROOT / "data")
    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings)
    fee_defaults: FeeDefaults = Field(default_factory=FeeDefaults)
    workbench_defaults: WorkbenchDefaults = Field(default_factory=WorkbenchDefaults)
    host: str = "127.0.0.1"
    port: int = 8000
    max_workers: int = Field(default=2, ge=1, le=8)
    hf_revision: str = "ba589a11534825044fe5a6b84838f50ba8d8d188"
    etf_hf_revision: str = "4b7e48c1e8265cc7d25baee55879dd5bbb24f74d"


def load_settings() -> Settings:
    values = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) if SETTINGS_FILE.exists() else {}
    if os.environ.get("STOCKREGRESSION_DATA_DIR"):
        values["data_dir"] = os.environ["STOCKREGRESSION_DATA_DIR"]
    settings = Settings.model_validate(values)
    settings.data_dir = str(Path(settings.data_dir).expanduser().resolve())
    return settings


def save_settings(values: dict) -> Settings:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(SETTINGS_FILE.with_suffix(".lock")), timeout=10):
        previous = load_settings().model_dump()
        values = dict(values)
        for key in ("schedule", "fee_defaults"):
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
        temporary = SETTINGS_FILE.with_suffix(".writing")
        temporary.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(SETTINGS_FILE)
    return settings
