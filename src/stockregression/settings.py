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
CONFIG_SCHEMA_VERSION = 5
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
    strategy_id: Literal["geometric_robust_j"] = "geometric_robust_j"
    protocol_version: Literal[5] = 5
    selection_target: Literal["fixed", "adaptive", "both"] = "both"
    rebalance_days: int = Field(default=63, ge=1)
    seed: int = 42
    top_candidates: int = Field(default=20, ge=1, le=100)
    max_principal_drawdown: float = Field(default=0.2, gt=0, le=1)
    min_trades: int = Field(default=10, ge=0)
    min_history_days: int = Field(default=250, ge=20)
    min_minute_ratio: float = Field(default=0.95, ge=0, le=1)
    ranking_mode: Literal["robust_j"] = "robust_j"
    strategy_version: Literal[5] = 5
    hypercube_edge: int = Field(default=3, ge=1, le=99)
    granularity: dict[str, float] = Field(default_factory=lambda: dict(GRANULARITY_DEFAULTS))
    search_method: Literal["auto", "hierarchical", "exhaustive"] = "auto"
    training_budget: int = Field(default=20_000, ge=1, le=1_000_000)
    validation_budget: int = Field(default=100_000, ge=1, le=2_000_000)
    reporting_budget: int = Field(default=1_000, ge=0, le=2_000_000)
    total_evaluation_budget: int | None = Field(default=None, ge=1, le=10_000_000)
    max_runtime_seconds: int = Field(default=3600, ge=1, le=86400)
    recommendation_count: int = Field(default=10, ge=1, le=100)
    max_shares_mode: Literal["unlimited"] = "unlimited"
    slippage_mode: Literal["configured", "zero"] = "configured"
    neighborhood_fields: list[Literal[
        "buy_step", "sell_step", "sell_shares", "buy_shares"
    ]] = Field(default_factory=lambda: [
        "buy_step", "sell_step", "buy_shares", "sell_shares",
    ])
    fixed_half_widths: dict[str, float] = Field(default_factory=lambda: {
        "buy_step": 0.001, "sell_step": 0.001, "sell_shares": 100, "buy_shares": 100,
    })
    ranges: dict[str, ResearchRangeDefaults] = Field(default_factory=lambda: {
        key: ResearchRangeDefaults.model_validate({**value, "step": None})
        for key, value in ROBUST_RANGE_DEFAULTS.items()
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
    floor_k_ref: float = Field(default=10.0, gt=0, le=1_000_000)
    training_min_days: int = Field(default=250, ge=20, le=5000)
    training_max_days: int = Field(default=500, ge=20, le=5000)
    training_window_mode: Literal["same_price", "fixed"] = "same_price"
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
        if not isinstance(values, dict):
            return values
        values = dict(values)
        if "max_principal_drawdown" not in values:
            # The former unit-NAV threshold has a different denominator and is
            # deliberately not promoted as an equivalent capital drawdown.
            values["max_principal_drawdown"] = 0.2
        # Only whole legacy research objects are migrated.  A current API
        # patch that submits a retired control must fail validation instead of
        # being silently accepted and discarded.
        previous_protocol = values.get("protocol_version")
        migrating_legacy = (
            previous_protocol is not None
            and str(previous_protocol).isdigit()
            and int(previous_protocol) < 5
        )
        if migrating_legacy:
            for key in (
                "bootstrap_reps", "bootstrap_block_lengths", "significance_alpha",
                "outer_validation_days", "mode", "n_trials", "local_search_budget",
                "max_drawdown",
            ):
                values.pop(key, None)
        versioned_defaults = {
            "strategy_id": "geometric_robust_j", "protocol_version": 5,
            "strategy_version": 5, "ranking_mode": "robust_j",
            "max_shares_mode": "unlimited",
            "neighborhood_fields": ["buy_step", "sell_step", "buy_shares", "sell_shares"],
        }
        if migrating_legacy:
            values.update(versioned_defaults)
        else:
            for name, default in versioned_defaults.items():
                values.setdefault(name, default)
        saved_granularity = values.get("granularity")
        if migrating_legacy:
            values["granularity"] = {
                name: (saved_granularity.get(name, default) if isinstance(saved_granularity, dict) else default)
                for name, default in GRANULARITY_DEFAULTS.items()
            }
        saved_ranges = values.get("ranges")
        if migrating_legacy:
            values["ranges"] = {
                name: {
                    **(dict(saved_ranges.get(name) or default) if isinstance(saved_ranges, dict) else default),
                    "step": None,
                }
                for name, default in ROBUST_RANGE_DEFAULTS.items()
            }
        saved_widths = values.get("fixed_half_widths")
        if migrating_legacy:
            values["fixed_half_widths"] = {
                name: (saved_widths.get(name, default) if isinstance(saved_widths, dict) else default)
                for name, default in {
                    "buy_step": .001, "sell_step": .001,
                    "sell_shares": 100, "buy_shares": 100,
                }.items()
            }
        values.setdefault("reporting_budget", 1_000)
        values.setdefault("total_evaluation_budget", None)
        values.setdefault("max_runtime_seconds", 3_600)
        values.setdefault("training_window_mode", "same_price")
        return values

    @model_validator(mode="after")
    def valid_ranges(self):
        if set(self.ranges) != set(ROBUST_RANGE_DEFAULTS):
            raise ValueError("稳健 J 搜索范围必须包含四个搜索轴")
        for name, value in self.ranges.items():
            if name.endswith("_shares"):
                if value.low < 100 or value.high % 100 or value.low % 100:
                    raise ValueError(f"股数搜索范围 {name} 必须以 100 为单位，交易量至少 100")
            else:
                if value.low <= 0 or value.high > 1:
                    raise ValueError(f"网格间距搜索范围 {name} 必须位于 (0, 1]")
            if value.step is not None:
                raise ValueError("稳健 J 研究的步长只能配置在 granularity 中")
        return self

    @model_validator(mode="after")
    def valid_fixed_half_widths(self):
        allowed = {"buy_step", "sell_step", "sell_shares", "buy_shares"}
        if set(self.fixed_half_widths) != allowed:
            raise ValueError("固定邻域半宽必须包含四个稳健 J 搜索轴")
        if any(float(value) <= 0 for value in self.fixed_half_widths.values()):
            raise ValueError("固定邻域半宽必须为正数")
        if any(abs(float(self.fixed_half_widths[name]) / 0.001 - round(float(self.fixed_half_widths[name]) / 0.001)) > 1e-7 for name in ("buy_step", "sell_step")):
            raise ValueError("买卖间距半宽必须对齐 0.001")
        if any(abs(float(self.fixed_half_widths[name]) / 100 - round(float(self.fixed_half_widths[name]) / 100)) > 1e-7 for name in ("sell_shares", "buy_shares")):
            raise ValueError("交易数量半宽必须对齐 100")
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
        expected_granularity = set(GRANULARITY_DEFAULTS)
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
        if self.eta_floor > self.eta_limit:
            raise ValueError("底仓保护阈值不能高于持仓免罚阈值")
        if self.eta_hard_cap is not None and self.eta_hard_cap < self.eta_limit:
            raise ValueError("最大持仓淘汰阈值不能低于持仓免罚阈值")
        if not math.isfinite(self.floor_k_ref) or self.floor_k_ref <= 0:
            raise ValueError("底仓参考格数必须为正数")
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
    config_schema_version: Literal[1, 2, 3, 4, 5] = CONFIG_SCHEMA_VERSION
    config_revision: str | None = None
    application: ApplicationSection = Field(default_factory=ApplicationSection)
    paths: PathsSection = Field(default_factory=lambda: PathsSection(data_dir=str(DEFAULT_DATA_DIR)))
    backtest: BacktestSection = Field(default_factory=BacktestSection)
    research: ResearchSection = Field(default_factory=ResearchSection)
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
            "defaults": value.workbench_defaults.research.model_dump(mode="json"),
            "legacy": deepcopy(value.research_legacy),
        },
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
    research_defaults = ((raw.get("workbench_defaults") or {}).get("research"))
    if isinstance(research_defaults, dict):
        # Settings validation normalises away fields from retired protocols.
        # Restore the exact source document here so the v5 migration archive is
        # an audit record of what the user actually had configured.
        document.setdefault("research", {})["defaults"] = deepcopy(research_defaults)
    return document


def migrate_configuration_document(raw: dict) -> dict:
    """Apply deterministic configuration migrations in memory.

    Schema 5 starts a new research protocol.  Earlier research defaults and
    statistics controls are archived under ``research.legacy`` so existing
    historical records can still be identified without pretending that their
    former unit-NAV metrics are equivalent to peak-principal metrics.
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
    if version < CONFIG_SCHEMA_VERSION:
        research = document.setdefault("research", {})
        previous_defaults = deepcopy(research.get("defaults") or {})
        previous_legacy = deepcopy(research.get("legacy") or {})
        previous_statistics = deepcopy(document.pop("statistics", None))

        # Keep every earlier archive.  Existing installations already carry a
        # v1 archive in ``legacy``; moving that object into ``history`` avoids
        # silently replacing it with the immediately preceding v4 defaults.
        if previous_legacy:
            history = list(previous_legacy.get("history") or [])
            history.append({key: value for key, value in previous_legacy.items() if key != "history"})
            previous_legacy = {**previous_legacy, "history": history}
        previous_legacy["source"] = f"migration_v{version}"
        previous_legacy["research_defaults"] = previous_defaults
        if previous_statistics is not None:
            previous_legacy["statistics"] = previous_statistics
        research["legacy"] = previous_legacy

        # ResearchDefaults owns the protocol upgrade.  It deliberately resets
        # the old unit-NAV drawdown limit and carries forward budgets, ranges,
        # weights and other still-valid search settings.
        research["defaults"] = ResearchDefaults.model_validate(previous_defaults).model_dump(mode="python")
        document["research"] = research
        document["config_schema_version"] = CONFIG_SCHEMA_VERSION
    document.pop("config_revision", None)
    validated = ConfigurationDocument.model_validate(document).model_dump(mode="python", exclude_none=False)
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
        for key in ("schedule", "fee_defaults", "performance", "market_data", "ui"):
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
