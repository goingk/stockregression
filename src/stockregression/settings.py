from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import GRANULARITY_DEFAULTS, GRANULARITY_MINIMUMS, LEGACY_GRANULARITY_DEFAULTS, ROBUST_RANGE_DEFAULTS

PROJECT_ROOT = Path(os.environ.get("STOCKREGRESSION_HOME", Path(__file__).resolve().parents[2])).resolve()
PROJECT_CONFIG_FILE = PROJECT_ROOT / "config" / "application.json"
LEGACY_SETTINGS_FILE = PROJECT_ROOT / ".local" / "settings.json"
CONFIG_SCHEMA_VERSION = 4
# ``SETTINGS_FILE`` remains the single test/CLI override point.  Its default is
# now the project configuration file rather than the legacy hidden settings.
SETTINGS_FILE = PROJECT_CONFIG_FILE
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_DIR = DEFAULT_DATA_DIR / "dataset"
DEFAULT_ETF_PACKAGE_DIR = DEFAULT_DATASET_DIR / "etf"
DEFAULT_STOCK_PACKAGE_DIR = DEFAULT_DATASET_DIR / "stock"
DEFAULT_PACKAGE_DIR = DEFAULT_ETF_PACKAGE_DIR


def resolve_project_path(value: str | Path) -> Path:
    """Resolve configured paths relative to the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def portable_project_path(value: str | Path) -> str:
    """Keep project paths relative so the checkout can be moved."""
    resolved = resolve_project_path(value)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


class ScheduleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    daily_time: str = "18:30"
    weekly_day: str = "SAT"
    weekly_time: str = "10:00"
    subprocess_timeout_seconds: int = Field(default=45, ge=5, le=900)
    lock_timeout_seconds: int = Field(default=30, ge=1, le=300)


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
    step: float | None = Field(default=None, gt=0)

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
    strategy_id: Literal["geometric_excess_neighborhood", "geometric_robust_j"] = "geometric_robust_j"
    protocol_version: Literal[1, 2, 3, 4] = 4
    selection_target: Literal["fixed", "adaptive", "both"] = "both"
    rebalance_days: int = Field(default=63, ge=1)
    bootstrap_reps: int = Field(default=10_000, ge=1)
    bootstrap_block_lengths: list[int] = Field(default_factory=lambda: [20, 10, 40])
    significance_alpha: float = Field(default=.05, gt=0, lt=1)
    outer_validation_days: int = Field(default=126, ge=20, le=1000)
    mode: Literal["historical", "robust", "both"] = "both"
    n_trials: int = Field(default=1000, ge=1, le=5000)
    local_search_budget: int = Field(default=3000, ge=1, le=5000)
    seed: int = 42
    top_candidates: int = Field(default=20, ge=1, le=100)
    max_drawdown: float = Field(default=0.2, gt=0, le=1)
    min_trades: int = Field(default=10, ge=0)
    min_history_days: int = Field(default=250, ge=20)
    min_minute_ratio: float = Field(default=0.95, ge=0, le=1)
    ranking_mode: Literal["max_volume", "fixed_mean"] = "fixed_mean"
    strategy_version: Literal[1, 2, 3, 4] = 4
    hypercube_edge: int = Field(default=3, ge=1, le=99)
    granularity: dict[str, float] = Field(default_factory=lambda: dict(GRANULARITY_DEFAULTS))
    search_method: Literal["auto", "hierarchical", "exhaustive"] = "auto"
    training_budget: int = Field(default=20_000, ge=1, le=1_000_000)
    validation_budget: int = Field(default=100_000, ge=1, le=2_000_000)
    recommendation_count: int = Field(default=10, ge=1, le=100)
    max_shares_mode: Literal["search", "unlimited"] = "search"
    slippage_mode: Literal["configured", "zero"] = "configured"
    neighborhood_fields: list[Literal[
        "buy_step", "sell_step", "sell_shares", "buy_shares", "min_shares", "max_shares"
    ]] = Field(default_factory=lambda: [
        "buy_step", "sell_step", "buy_shares", "sell_shares",
    ])
    fixed_half_widths: dict[str, float] = Field(default_factory=lambda: {
        "buy_step": 0.001, "sell_step": 0.001, "sell_shares": 100, "buy_shares": 100,
        "min_shares": 500, "max_shares": 500,
    })
    ranges: dict[str, ResearchRangeDefaults] = Field(default_factory=lambda: {
        key: ResearchRangeDefaults.model_validate(value) for key, value in _RESEARCH_RANGE_DEFAULTS.items()
    })
    sigma: float = Field(default=1.0, gt=0, le=100)
    alpha: float = Field(default=.30, ge=0, le=1)
    beta: float = Field(default=1.20, ge=0, le=100)
    tau: float = Field(default=.01, ge=-1, le=10)
    eta_limit: float = Field(default=2.0, ge=0, le=1000)
    eta_hard_cap: float | None = Field(default=3.5, ge=0, le=1000)
    gamma_max: float = Field(default=.40, ge=0, le=1000)
    eta_floor: float = Field(default=0.0, ge=0, le=1)
    gamma_min: float = Field(default=0.0, ge=0, le=1000)
    training_min_days: int = Field(default=250, ge=20, le=5000)
    training_max_days: int = Field(default=500, ge=20, le=5000)
    peak_tolerance: float = Field(default=1e-12, gt=0, le=.1)
    start_min: int = Field(default=20, ge=1, le=1000)
    start_max: int = Field(default=64, ge=1, le=1000)
    starts_per_candidate: int = Field(default=4, ge=1, le=100)
    initial_strides: list[int] = Field(default_factory=lambda: [1, 2])
    stride_multiplier: int = Field(default=2, ge=2, le=16)
    max_stride_fraction: float = Field(default=.25, gt=0, le=1)
    validation_reserve_fraction: float = Field(default=.25, ge=0, le=1)
    exhaustive_memory_limit_mb: int = Field(default=512, ge=16, le=65536)
    estimated_bytes_per_point: int = Field(default=512, ge=64, le=65536)
    stress_cost_multiplier: float = Field(default=2.0, gt=1, le=20)
    stress_region_count: int = Field(default=3, ge=0, le=50)

    @model_validator(mode="before")
    @classmethod
    def migrate_protocol(cls, values):
        if isinstance(values, dict):
            values = dict(values)
            if "strategy_version" not in values:
                values["strategy_version"] = 4 if "ranking_mode" not in values else (1 if values.get("ranking_mode") == "fixed_mean" else 3)
            elif int(values.get("strategy_version")) == 2 and values.get("ranking_mode") == "max_volume":
                # The connected-component selector is no longer offered for new
                # studies.  Remembered v4 settings move to the hypercube mode.
                values["strategy_version"] = 3
            if int(values.get("strategy_version", 4)) in (3, 4):
                values["ranking_mode"] = "fixed_mean"
            elif int(values.get("strategy_version")) == 2:
                values["ranking_mode"] = "max_volume"
            values["strategy_id"] = (
                "geometric_robust_j" if int(values.get("strategy_version", 4)) == 4
                else "geometric_excess_neighborhood"
            )
            robust = int(values.get("strategy_version", 4)) == 4
            if "max_shares_mode" not in values:
                values["max_shares_mode"] = "unlimited" if robust else "search"
            granularity_defaults = GRANULARITY_DEFAULTS if robust else LEGACY_GRANULARITY_DEFAULTS
            if not isinstance(values.get("granularity"), dict):
                values["granularity"] = dict(granularity_defaults)
            else:
                saved_granularity = values["granularity"]
                unknown = set(saved_granularity) - set(LEGACY_GRANULARITY_DEFAULTS)
                if unknown:
                    raise ValueError("研究细粒度包含未知参数：" + "、".join(sorted(unknown)))
                granularity = {
                    name: saved_granularity.get(name, default)
                    for name, default in granularity_defaults.items()
                }
                for name, value in list(granularity.items()):
                    minimum = float(GRANULARITY_MINIMUMS[name])
                    steps = float(value) / minimum
                    if float(value) <= 0 or abs(steps - round(steps)) > 1e-7:
                        raise ValueError(
                            f"研究细粒度 {name} 必须是参数最小单位 "
                            f"{'0.0001' if minimum == 0.0001 else int(minimum)} 的正整数倍"
                        )
                values["granularity"] = granularity
            if isinstance(values.get("ranges"), dict):
                ranges = dict(values["ranges"])
                legacy = ranges.pop("grid_step", None)
                if legacy is not None:
                    ranges["buy_step"] = legacy
                    ranges["sell_step"] = legacy
                if robust:
                    unknown = set(ranges) - set(ROBUST_RANGE_DEFAULTS)
                    if unknown:
                        raise ValueError("稳健 J 研究范围包含非搜索轴：" + "、".join(sorted(unknown)))
                    if any(
                        isinstance(value, dict) and value.get("step") is not None
                        for value in ranges.values()
                    ):
                        raise ValueError("稳健 J 研究的步长只能配置在 granularity 中")
                    ranges = {
                        name: {**dict(ranges.get(name) or ROBUST_RANGE_DEFAULTS[name]), "step": None}
                        for name in ROBUST_RANGE_DEFAULTS
                    }
                values["ranges"] = ranges
            elif robust:
                values["ranges"] = {
                    name: {**value, "step": None}
                    for name, value in ROBUST_RANGE_DEFAULTS.items()
                }
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
        robust = self.strategy_version == 4
        expected = set(ROBUST_RANGE_DEFAULTS) if robust else set(_RESEARCH_RANGE_DEFAULTS)
        if set(self.ranges) != expected:
            raise ValueError("稳健 J 搜索范围必须包含四个搜索轴" if robust else "搜索范围必须包含全部七个支持参数")
        for name, value in self.ranges.items():
            if name.endswith("_shares"):
                minimum = 100 if name in {"buy_shares", "sell_shares"} else 0
                numbers = (value.low, value.high) + (() if value.step is None else (value.step,))
                if value.low < minimum or any(number % 100 for number in numbers):
                    raise ValueError(f"股数搜索范围 {name} 必须以 100 为单位，交易量至少 100")
            else:
                if value.low <= 0 or value.high > 1:
                    raise ValueError(f"网格间距搜索范围 {name} 必须位于 (0, 1]")
            if robust and value.step is not None:
                raise ValueError("稳健 J 研究的步长只能配置在 granularity 中")
            if not robust and value.step is None:
                raise ValueError("历史研究范围必须包含步长")
        if not self.bootstrap_block_lengths or any(int(value) < 1 for value in self.bootstrap_block_lengths):
            raise ValueError("Bootstrap 区块长度必须为正整数")
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

    @model_validator(mode="after")
    def valid_hypercube(self):
        if self.hypercube_edge < 1 or self.hypercube_edge % 2 != 1:
            raise ValueError("超正方体边长必须为正奇数")
        expected_granularity = set(GRANULARITY_DEFAULTS if self.strategy_version == 4 else LEGACY_GRANULARITY_DEFAULTS)
        if set(self.granularity) != expected_granularity:
            raise ValueError("研究细粒度包含的参数与策略版本不匹配")
        for name, value in self.granularity.items():
            minimum = float(GRANULARITY_MINIMUMS[name])
            steps = float(value) / minimum
            if float(value) <= 0 or abs(steps - round(steps)) > 1e-7:
                raise ValueError("研究细粒度必须是参数最小单位的正整数倍")
        if self.training_min_days > self.training_max_days:
            raise ValueError("研究训练时段最短日数不能大于最长日数")
        if self.start_min > self.start_max:
            raise ValueError("局部搜索起点数下限不能大于上限")
        if not self.initial_strides or any(int(value) < 1 for value in self.initial_strides):
            raise ValueError("局部搜索初始步长必须为正整数")
        if self.strategy_version == 4:
            if self.eta_floor > self.eta_limit:
                raise ValueError("底仓保护阈值不能高于持仓免罚阈值")
            if self.eta_hard_cap is not None and self.eta_hard_cap < self.eta_limit:
                raise ValueError("最大持仓淘汰阈值不能低于持仓免罚阈值")
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
    evaluation_chunk_size: int = Field(default=512, ge=16, le=8192)
    job_cancel_poll_ms: int = Field(default=300, ge=100, le=10_000)
    progress_update_ms: int = Field(default=500, ge=100, le=60_000)

    @model_validator(mode="after")
    def valid_custom_budget(self):
        if self.cpu_budget_mode == "custom" and self.cpu_budget is None:
            raise ValueError("指定计算预算时必须填写 CPU 线程数")
        return self


class UiSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    autosave_debounce_ms: int = Field(default=500, ge=100, le=10_000)
    polling_interval_ms: int = Field(default=2_000, ge=500, le=60_000)
    preview_debounce_ms: int = Field(default=300, ge=50, le=10_000)
    page_size: int = Field(default=50, ge=10, le=500)
    chart_points: int = Field(default=2_000, ge=100, le=5_000)
    default_ma: list[int] = Field(default_factory=lambda: [5, 20, 60])
    task_polling_interval_ms: int = Field(default=2500, ge=500, le=60_000)
    data_refresh_interval_ms: int = Field(default=5000, ge=1000, le=300_000)
    data_health_page_size: int = Field(default=100, ge=10, le=500)
    inventory_page_size: int = Field(default=50, ge=10, le=500)
    instrument_lookup_limit: int = Field(default=40, ge=5, le=200)
    instrument_lookup_debounce_ms: int = Field(default=250, ge=50, le=5000)

    @model_validator(mode="after")
    def valid_ma(self):
        if any(value < 1 or value > 1000 for value in self.default_ma):
            raise ValueError("默认均线周期必须位于 1 到 1000 之间")
        return self


class StatisticsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bootstrap_reps: int = Field(default=10_000, ge=100, le=1_000_000)
    block_lengths: list[int] = Field(default_factory=lambda: [20, 10, 40])
    alpha: float = Field(default=.05, gt=0, lt=1)
    outer_validation_days: int = Field(default=126, ge=20, le=1000)

    @model_validator(mode="after")
    def valid_blocks(self):
        if not self.block_lengths or any(value < 1 for value in self.block_lengths):
            raise ValueError("Bootstrap 区块长度必须为正整数")
        return self


class MarketDataSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache_mb: int = Field(default=256, ge=0, le=16384)
    recent_minute_source: Literal["sina", "tdx"] = "sina"
    etf_daily_source: Literal["baostock", "eastmoney"] = "baostock"
    network_timeout_seconds: float = Field(default=15, ge=1, le=300)
    request_retries: int = Field(default=1, ge=0, le=5)
    failure_cooldown_hours: int = Field(default=24, ge=0, le=720)
    retry_after_seconds: int = Field(default=60, ge=1, le=86400)
    source_lock_timeout_seconds: int = Field(default=30, ge=1, le=600)
    sync_lock_timeout_seconds: int = Field(default=60, ge=1, le=3600)
    prepare_lock_timeout_seconds: int = Field(default=300, ge=1, le=86400)
    package_lock_timeout_seconds: int = Field(default=600, ge=1, le=86400)
    sqlite_timeout_seconds: int = Field(default=60, ge=1, le=600)
    daily_batch_size: int = Field(default=128, ge=1, le=2048)
    service_batch_size: int = Field(default=128, ge=1, le=2048)
    scheduler_poll_seconds: float = Field(default=.5, gt=0, le=60)


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
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    statistics: StatisticsSettings = Field(default_factory=StatisticsSettings)
    ui: UiSettings = Field(default_factory=UiSettings)
    research_legacy: dict = Field(default_factory=dict)


class ApplicationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_workers: int = Field(default=2, ge=1, le=8)


class PathsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: str
    package_dir: str | None = None


class BacktestSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grid: GridDefaults = Field(default_factory=GridDefaults)
    position: PositionDefaults = Field(default_factory=PositionDefaults)
    dca: DcaDefaults = Field(default_factory=DcaDefaults)
    fees: FeeDefaults = Field(default_factory=FeeDefaults)


class ResearchSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defaults: ResearchDefaults = Field(default_factory=ResearchDefaults)
    legacy: dict = Field(default_factory=dict)


class ConfigurationDocument(BaseModel):
    """The complete validated project configuration file."""

    model_config = ConfigDict(extra="forbid")
    config_schema_version: Literal[1, 2, 3, 4] = CONFIG_SCHEMA_VERSION
    config_revision: str | None = None
    application: ApplicationSection = Field(default_factory=ApplicationSection)
    paths: PathsSection = Field(default_factory=lambda: PathsSection(data_dir=str(DEFAULT_DATA_DIR)))
    backtest: BacktestSection = Field(default_factory=BacktestSection)
    research: ResearchSection = Field(default_factory=ResearchSection)
    statistics: StatisticsSettings = Field(default_factory=StatisticsSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    runtime: PerformanceSettings = Field(default_factory=PerformanceSettings)
    ui: UiSettings = Field(default_factory=UiSettings)
    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings)


def configuration_document(value: Settings, *, portable: bool = True) -> dict:
    """Materialise the grouped, single-file configuration document."""
    data_dir = portable_project_path(value.data_dir) if portable else str(value.data_dir)
    package_dir = portable_project_path(value.package_dir) if portable and value.package_dir else value.package_dir if value.package_dir else None
    core = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "application": {"host": value.host, "port": value.port, "max_workers": value.max_workers},
        "paths": {"data_dir": data_dir, "package_dir": package_dir},
        "backtest": {
            "grid": value.workbench_defaults.grid.model_dump(mode="json"),
            "position": value.workbench_defaults.position.model_dump(mode="json"),
            "dca": value.workbench_defaults.dca.model_dump(mode="json"),
            "fees": value.fee_defaults.model_dump(mode="json"),
        },
        "research": {
            "defaults": value.workbench_defaults.research.model_dump(
                mode="json", exclude={"bootstrap_reps", "bootstrap_block_lengths", "significance_alpha", "outer_validation_days"},
            ),
            "legacy": deepcopy(value.research_legacy),
        },
        "statistics": value.statistics.model_dump(mode="json"),
        "market_data": {
            **value.market_data.model_dump(mode="json"),
            "cache_mb": value.market_cache_mb,
            "recent_minute_source": value.recent_minute_source,
            "etf_daily_source": value.etf_daily_source,
        },
        "runtime": value.performance.model_dump(mode="json"),
        "ui": value.ui.model_dump(mode="json"),
        "schedule": value.schedule.model_dump(mode="json"),
    }
    revision = configuration_revision(core)
    return {**core, "config_revision": revision}


def configuration_revision(document: dict) -> str:
    clean = {key: value for key, value in document.items() if key != "config_revision"}
    encoded = json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def settings_from_document(document: dict) -> Settings:
    validated = ConfigurationDocument.model_validate(document)
    return Settings(
        data_dir=validated.paths.data_dir,
        package_dir=validated.paths.package_dir,
        schedule=validated.schedule,
        fee_defaults=validated.backtest.fees,
        workbench_defaults=WorkbenchDefaults(
            grid=validated.backtest.grid,
            position=validated.backtest.position,
            dca=validated.backtest.dca,
            research=validated.research.defaults,
        ),
        performance=validated.runtime,
        host=validated.application.host,
        port=validated.application.port,
        max_workers=validated.application.max_workers,
        market_cache_mb=validated.market_data.cache_mb,
        recent_minute_source=validated.market_data.recent_minute_source,
        etf_daily_source=validated.market_data.etf_daily_source,
        market_data=validated.market_data,
        statistics=validated.statistics,
        ui=validated.ui,
        research_legacy=deepcopy(validated.research.legacy),
    )


def _flat_configuration_document(raw: dict) -> dict:
    """Convert the former flat settings object without changing its values first."""
    legacy = Settings.model_validate(raw)
    document = configuration_document(legacy)
    # ``configuration_document`` materialises the current schema.  Keep the
    # source schema here so the migration pipeline can still archive the
    # original research/backtest defaults before activating robust-J defaults.
    document["config_schema_version"] = int(raw.get("config_schema_version", 1))
    return document


def migrate_configuration_document(raw: dict) -> dict:
    """Apply deterministic configuration migrations in memory.

    Schema 2 archives the complete former research defaults and activates the
    robust-J research defaults.  Historical results remain bound to their own
    task snapshots; this migration only affects new studies.
    """
    if not isinstance(raw, dict):
        raise ValueError("配置文件根节点必须是对象")
    version = int(raw.get("config_schema_version", 1))
    # Schema v2 and later are always grouped documents.  In particular, a
    # damaged/truncated v4 file must never fall through to the flat legacy
    # adapter, where extra keys are ignored and defaults silently replace the
    # user's file.
    grouped = (
        version >= 2
        or "defaults" in (raw.get("research") or {})
        or ("application" in raw and "workbench_defaults" not in raw)
    )
    document = deepcopy(raw) if grouped else _flat_configuration_document(raw)
    if version > CONFIG_SCHEMA_VERSION:
        raise ValueError(f"配置架构 {version} 高于当前程序支持的 {CONFIG_SCHEMA_VERSION}")
    if version < 2:
        research = document.setdefault("research", {})
        defaults = deepcopy(research.get("defaults") or {})
        legacy = deepcopy(research.get("legacy") or {})
        if not legacy:
            legacy = {
                "source": "migration_v1",
                "research_defaults": deepcopy(defaults),
                "backtest_defaults": deepcopy(document.get("backtest") or {}),
            }
        # New studies use the natural-position robust-J family.  Compatible
        # user values (budgets, ranges, weights, fees) remain intact; the old
        # exact values are available in research.legacy for legacy tasks.
        defaults["strategy_id"] = "geometric_robust_j"
        defaults["strategy_version"] = 4
        defaults["max_shares_mode"] = "unlimited"
        defaults["neighborhood_fields"] = ["buy_step", "sell_step", "buy_shares", "sell_shares"]
        research["defaults"] = defaults
        research["legacy"] = legacy
        document["research"] = research
        if "statistics" not in document:
            document["statistics"] = {
                "bootstrap_reps": defaults.get("bootstrap_reps", 10_000),
                "block_lengths": defaults.get("bootstrap_block_lengths", [20, 10, 40]),
                "alpha": defaults.get("significance_alpha", .05),
                "outer_validation_days": defaults.get("outer_validation_days", 126),
            }
        document["config_schema_version"] = CONFIG_SCHEMA_VERSION
    if version < 3:
        research = document.setdefault("research", {})
        defaults = dict(research.get("defaults") or {})
        statistics = dict(document.get("statistics") or {})
        for key, target, fallback in (
            ("bootstrap_reps", "bootstrap_reps", 10_000),
            ("bootstrap_block_lengths", "block_lengths", [20, 10, 40]),
            ("significance_alpha", "alpha", .05),
            ("outer_validation_days", "outer_validation_days", 126),
        ):
            value = defaults.pop(key, None)
            statistics.setdefault(target, fallback if value is None else value)
        if int(defaults.get("strategy_version", 4)) == 4:
            ranges = defaults.get("ranges") or {}
            defaults["ranges"] = {
                name: {
                    "low": (ranges.get(name) or ROBUST_RANGE_DEFAULTS[name]).get("low", ROBUST_RANGE_DEFAULTS[name]["low"]),
                    "high": (ranges.get(name) or ROBUST_RANGE_DEFAULTS[name]).get("high", ROBUST_RANGE_DEFAULTS[name]["high"]),
                    "step": None,
                }
                for name in ROBUST_RANGE_DEFAULTS
            }
            granularity = defaults.get("granularity") or {}
            defaults["granularity"] = {
                name: granularity.get(name, value)
                for name, value in GRANULARITY_DEFAULTS.items()
            }
        research["defaults"] = defaults
        document["statistics"] = statistics
        document["config_schema_version"] = CONFIG_SCHEMA_VERSION
    if version < 4:
        document["config_schema_version"] = CONFIG_SCHEMA_VERSION
    document.pop("config_revision", None)
    validated = ConfigurationDocument.model_validate(document).model_dump(mode="python", exclude_none=False)
    # These fields are versioned under the top-level ``statistics`` section.
    # Keep the migration output in exactly the same canonical shape as
    # ``configuration_document``; otherwise every read adds hidden defaults and
    # therefore reports a different content hash from the one written to disk.
    for key in ("bootstrap_reps", "bootstrap_block_lengths", "significance_alpha", "outer_validation_days"):
        validated["research"]["defaults"].pop(key, None)
    return validated


def read_configuration_document(path: Path | None = None) -> tuple[dict, str]:
    target = path or SETTINGS_FILE
    raw = json.loads(target.read_text(encoding="utf-8"))
    document = migrate_configuration_document(raw)
    revision = configuration_revision(document)
    document["config_revision"] = revision
    return document, revision


def _resolve_settings_paths(settings: Settings) -> Settings:
    settings.data_dir = str(resolve_project_path(settings.data_dir))
    if settings.package_dir:
        settings.package_dir = str(resolve_project_path(settings.package_dir))
    return settings


def _write_configuration(settings: Settings) -> dict:
    document = configuration_document(settings, portable=True)
    target = SETTINGS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = target.parent / "backups"
    if target.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(target, backup_dir / "application.previous.json")
        except OSError:
            pass
    fd, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".writing", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass
    return document


def load_settings() -> Settings:
    target = SETTINGS_FILE
    is_main_target = target.resolve() != LEGACY_SETTINGS_FILE.resolve()
    if target.exists():
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("配置文件根节点必须是对象")
        document = migrate_configuration_document(raw)
        settings = settings_from_document(document)
        if is_main_target and (
            int(raw.get("config_schema_version", 1)) != CONFIG_SCHEMA_VERSION
            or "defaults" not in (raw.get("research") or {})
        ):
            _resolve_settings_paths(settings)
            with FileLock(str(target.with_suffix(".lock")), timeout=10):
                _write_configuration(settings)
    elif is_main_target and LEGACY_SETTINGS_FILE.exists():
        legacy = json.loads(LEGACY_SETTINGS_FILE.read_text(encoding="utf-8"))
        settings = settings_from_document(migrate_configuration_document(legacy))
        _resolve_settings_paths(settings)
        with FileLock(str(target.with_suffix(".lock")), timeout=10):
            _write_configuration(settings)
    else:
        settings = Settings()
        if is_main_target:
            _resolve_settings_paths(settings)
            target.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(target.with_suffix(".lock")), timeout=10):
                if not target.exists():
                    _write_configuration(settings)
    return _resolve_settings_paths(settings)


def save_settings(values: dict, *, base_revision: str | None = None) -> Settings:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(SETTINGS_FILE.with_suffix(".lock")), timeout=10):
        previous_document = None
        if SETTINGS_FILE.exists():
            previous_document, current_revision = read_configuration_document(SETTINGS_FILE)
            if base_revision is not None and base_revision != current_revision:
                raise RuntimeError("CONFIG_REVISION_CONFLICT")
        if previous_document is not None:
            previous = settings_from_document(previous_document).model_dump()
        elif SETTINGS_FILE.resolve() != LEGACY_SETTINGS_FILE.resolve() and LEGACY_SETTINGS_FILE.exists():
            previous = settings_from_document(
                migrate_configuration_document(json.loads(LEGACY_SETTINGS_FILE.read_text(encoding="utf-8")))
            ).model_dump()
        else:
            previous = Settings().model_dump()
        values = dict(values)
        for key in ("schedule", "fee_defaults", "performance", "market_data", "statistics", "ui"):
            if key in values and isinstance(values[key], dict):
                values[key] = {**previous[key], **values[key]}
        # Older clients submit source and cache choices as flat settings
        # fields.  Keep the grouped configuration coherent as well, so a
        # partial legacy save cannot reset the other market-data settings to
        # their defaults when the document is materialised.
        market_data = dict(values.get("market_data") or previous["market_data"])
        submitted_market_data = values.get("market_data") if isinstance(values.get("market_data"), dict) else {}
        for scalar, grouped in (
            ("market_cache_mb", "cache_mb"),
            ("recent_minute_source", "recent_minute_source"),
            ("etf_daily_source", "etf_daily_source"),
        ):
            if scalar in values:
                market_data[grouped] = values[scalar]
            elif grouped in submitted_market_data:
                values[scalar] = submitted_market_data[grouped]
        values["market_data"] = market_data
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
                    if group == "research" and isinstance(patch.get("granularity"), dict):
                        merged_group["granularity"] = {
                            **merged_defaults[group].get("granularity", GRANULARITY_DEFAULTS),
                            **patch["granularity"],
                        }
                    merged_defaults[group] = merged_group
                else:
                    merged_defaults[group] = patch
            values["workbench_defaults"] = merged_defaults
        settings = Settings.model_validate({**previous, **values})
        _resolve_settings_paths(settings)
        _write_configuration(settings)
    return settings


def save_configuration_document(document: dict, *, base_revision: str | None = None) -> tuple[dict, Settings]:
    target = SETTINGS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target.with_suffix(".lock")), timeout=10):
        if target.exists():
            _, current_revision = read_configuration_document(target)
            if base_revision is not None and base_revision != current_revision:
                raise RuntimeError("CONFIG_REVISION_CONFLICT")
        candidate = deepcopy(document)
        candidate.pop("config_revision", None)
        settings = settings_from_document(migrate_configuration_document(candidate))
        _resolve_settings_paths(settings)
        written = _write_configuration(settings)
    return written, settings
