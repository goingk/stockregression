"""Single-file configuration service and public configuration contract."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import settings as settings_module
from .settings import (
    CONFIG_SCHEMA_VERSION,
    ConfigurationDocument,
    Settings,
    UiSettings,
    configuration_document,
    configuration_revision,
    load_settings,
    read_configuration_document,
    save_configuration_document,
    settings_from_document,
)


class ConfigurationConflict(ValueError):
    """The caller edited a stale revision of the shared configuration."""


_PATCH_GROUPS = {
    "application", "paths", "backtest", "research", "market_data", "runtime", "ui", "schedule",
}

_FIELD_LABELS = {
    "host": "监听地址", "port": "监听端口", "max_workers": "最大并发任务",
    "data_dir": "数据目录", "package_dir": "行情数据包目录",
    "grid_mode": "网格模式", "buy_step": "买入间距", "sell_step": "卖出间距",
    "lower_price": "价格下限", "upper_price": "价格上限",
    "initial_shares": "初始持仓", "min_shares": "最低持仓", "max_shares": "最高持仓",
    "buy_shares": "每次买入", "sell_shares": "每次卖出",
    "frequency": "定投频率", "shares": "定投数量",
    "commission_rate": "佣金费率", "min_commission": "最低佣金",
    "strategy_id": "策略标识", "strategy_version": "策略版本", "protocol_version": "研究协议",
    "selection_target": "研究方案", "rebalance_days": "自适应换参周期", "seed": "根随机种子",
    "top_candidates": "结果候选展示上限", "max_principal_drawdown": "最大本金回撤约束", "min_trades": "最少成交笔数",
    "min_history_days": "最少历史", "min_minute_ratio": "最小分钟覆盖率",
    "ranking_mode": "历史排名模式", "hypercube_edge": "超正方体边长", "granularity": "研究刻度",
    "search_method": "搜索方法", "training_budget": "训练评价预算", "validation_budget": "邻域评价预算",
    "recommendation_count": "Top-K 峰区", "max_shares_mode": "最高持股研究方式", "slippage_mode": "滑点处理",
    "neighborhood_fields": "研究搜索轴", "fixed_half_widths": "固定邻域半宽", "ranges": "搜索范围",
    "sigma": "高斯核宽度", "alpha": "中心收益权重或显著性水平", "beta": "下行风险系数",
    "tau": "安全垫阈值", "eta_limit": "持仓免罚上限", "eta_hard_cap": "最大持仓淘汰阈值",
    "gamma_max": "上界持仓惩罚", "eta_floor": "底仓保护下限", "gamma_min": "底仓整格惩罚",
    "floor_k_ref": "底仓参考格数",
    "training_min_days": "训练回看下限", "training_max_days": "训练回看上限",
    "peak_tolerance": "峰值比较容差", "start_min": "最少起点数", "start_max": "最多起点数",
    "starts_per_candidate": "每候选起点数", "initial_strides": "初始局部步长",
    "stride_multiplier": "大步长倍率", "max_stride_fraction": "大步长跨度比例",
    "validation_reserve_fraction": "峰值核验预算比例", "exhaustive_memory_limit_mb": "穷举内存上限",
    "estimated_bytes_per_point": "单点内存估计", "stress_cost_multiplier": "费用压力倍率",
    "stress_region_count": "费用压力峰区数", "cache_mb": "行情缓存",
    "recent_minute_source": "近期分钟来源", "etf_daily_source": "ETF 日线来源",
    "etf_daily_fallback_enabled": "ETF 日线备用来源回退",
    "network_timeout_seconds": "网络超时", "request_retries": "请求重试次数",
    "failure_cooldown_hours": "来源失败冷却", "retry_after_seconds": "准备重试等待",
    "source_lock_timeout_seconds": "来源锁超时", "sync_lock_timeout_seconds": "同步锁超时",
    "prepare_lock_timeout_seconds": "准备锁超时", "package_lock_timeout_seconds": "数据包锁超时",
    "sqlite_timeout_seconds": "SQLite 超时", "daily_batch_size": "日线同步批次",
    "service_batch_size": "行情服务批次", "scheduler_poll_seconds": "调度轮询间隔",
    "cpu_budget_mode": "CPU 预算模式", "cpu_budget": "CPU 线程预算", "download_workers": "下载并发",
    "validation_workers": "验证并发", "blas_threads": "BLAS 线程", "workspace_mb": "工作内存",
    "evaluation_chunk_size": "评价批大小", "job_cancel_poll_ms": "取消轮询间隔",
    "progress_update_ms": "进度更新间隔", "autosave_debounce_ms": "自动保存防抖",
    "polling_interval_ms": "配置轮询间隔", "preview_debounce_ms": "预览防抖", "page_size": "分页大小",
    "chart_points": "图表点数", "default_ma": "默认均线", "task_polling_interval_ms": "任务轮询间隔",
    "data_refresh_interval_ms": "数据刷新间隔", "data_health_page_size": "数据体检分页",
    "inventory_page_size": "库存分页", "instrument_lookup_limit": "证券搜索条数",
    "instrument_lookup_debounce_ms": "证券搜索防抖", "enabled": "启用状态",
    "daily_time": "每日更新时间", "weekly_day": "每周核对日", "weekly_time": "每周核对时间",
    "subprocess_timeout_seconds": "计划子进程超时", "lock_timeout_seconds": "计划任务锁超时",
}

_FIELD_UNITS = {
    "buy_step": "比例；界面显示百分比", "sell_step": "比例；界面显示百分比",
    "commission_rate": "比例；界面显示百分比", "alpha": "权重或比例；界面显示百分比",
    "tau": "训练区间累计超额收益率；界面显示百分比", "max_principal_drawdown": "比例；界面显示百分比",
    "min_minute_ratio": "比例；界面显示百分比", "eta_limit": "初始持仓 P0 的倍数",
    "eta_hard_cap": "初始持仓 P0 的倍数", "eta_floor": "初始持仓 P0 的倍数",
    "floor_k_ref": "卖出网格整格数",
    "training_budget": "唯一原始回测点", "validation_budget": "唯一原始回测点",
    "training_min_days": "交易日", "training_max_days": "交易日",
    "rebalance_days": "交易日",
    "network_timeout_seconds": "秒", "retry_after_seconds": "秒",
    "failure_cooldown_hours": "小时", "source_lock_timeout_seconds": "秒",
    "sync_lock_timeout_seconds": "秒", "prepare_lock_timeout_seconds": "秒",
    "package_lock_timeout_seconds": "秒", "sqlite_timeout_seconds": "秒",
    "scheduler_poll_seconds": "秒", "workspace_mb": "MiB",
    "cache_mb": "MiB", "exhaustive_memory_limit_mb": "MiB",
    "estimated_bytes_per_point": "字节 / 点", "autosave_debounce_ms": "毫秒",
    "polling_interval_ms": "毫秒", "preview_debounce_ms": "毫秒",
    "task_polling_interval_ms": "毫秒", "data_refresh_interval_ms": "毫秒",
    "instrument_lookup_debounce_ms": "毫秒", "job_cancel_poll_ms": "毫秒",
    "progress_update_ms": "毫秒", "buy_shares": "股或份", "sell_shares": "股或份",
    "initial_shares": "股或份", "min_shares": "股或份", "max_shares": "股或份",
}

_METHOD_GROUPS = {"backtest", "research"}


def _resolve_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    current = schema
    while isinstance(current, dict) and "$ref" in current:
        reference = str(current["$ref"])
        if not reference.startswith("#/$defs/"):
            break
        current = root.get("$defs", {}).get(reference.rsplit("/", 1)[-1], {})
    return current if isinstance(current, dict) else {}


def _value_at(document: dict[str, Any], parts: tuple[str, ...]) -> Any:
    current: Any = document
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _type_label(schema: dict[str, Any], root: dict[str, Any]) -> str:
    schema = _resolve_schema(schema, root)
    if "enum" in schema:
        return "枚举：" + " / ".join(str(value) for value in schema["enum"])
    type_name = schema.get("type")
    if type_name == "array":
        item = _type_label(schema.get("items", {}), root)
        return f"数组<{item}>"
    if type_name == "object":
        return "对象"
    if "anyOf" in schema:
        labels = [_type_label(item, root) for item in schema["anyOf"]]
        return " 或 ".join(dict.fromkeys(label for label in labels if label != "null"))
    return str(type_name or "任意值")


def _effect(path: tuple[str, ...]) -> str:
    if path[:1] == ("paths",) or path in {("application", "host"), ("application", "port")}:
        return "需重启"
    if path in {
        ("application", "max_workers"),
        ("market_data", "scheduler_poll_seconds"),
    }:
        return "需重启"
    if path[:1] in {("ui",), ("market_data",)}:
        return "后续读取或任务"
    if path[:1] == ("schedule",):
        return "重新安装系统计划"
    return "下次任务提交"


def _field_catalog(document: dict[str, Any], json_schema: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def visit(properties: dict[str, Any], prefix: tuple[str, ...]) -> None:
        for name, raw_schema in properties.items():
            path = (*prefix, name)
            schema = _resolve_schema(raw_schema, json_schema)
            nested = schema.get("properties")
            if isinstance(nested, dict) and nested:
                visit(nested, path)
                continue
            label = _FIELD_LABELS.get(name, name)
            output.append({
                "path": ".".join(path), "group": path[0], "label": label,
                "description": f"{label}；分组 {path[0]}。详细边界与消费者见 config/README.md。",
                "type": _type_label(schema, json_schema), "unit": _FIELD_UNITS.get(name),
                "default": _value_at(document, path), "effect": _effect(path),
                "method_impact": path[0] in _METHOD_GROUPS,
                "constraints": {
                    key: schema[key] for key in (
                        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                        "minLength", "maxLength", "minItems", "maxItems", "enum", "pattern",
                    ) if key in schema
                },
            })

    visit(json_schema.get("properties", {}), ())
    return output


def _merge(target: dict[str, Any], patch: dict[str, Any], path: tuple[str, ...] = ()) -> dict[str, Any]:
    merged = deepcopy(target)
    for key, value in patch.items():
        key = str(key)
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value, (*path, key))
        else:
            merged[key] = deepcopy(value)
    return merged


def current_configuration() -> dict[str, Any]:
    settings = load_settings()
    target = settings_module.SETTINGS_FILE
    if target.exists():
        document, _ = read_configuration_document(target)
        return document
    return configuration_document(settings)


def pending_restart_fields(
    document: dict[str, Any], *, active_data_dir: str, active_package_dir: str | None,
    active_host: str | None = None, active_port: int | None = None,
    active_max_workers: int | None = None,
    active_scheduler_poll_seconds: float | None = None,
) -> list[str]:
    pending: list[str] = []
    data_value = str(document.get("paths", {}).get("data_dir", ""))
    data_path = Path(data_value).expanduser()
    if not data_path.is_absolute():
        data_path = settings_module.PROJECT_ROOT / data_path
    if str(data_path.resolve()) != str(Path(active_data_dir).resolve()):
        pending.append("paths.data_dir")
    package = document.get("paths", {}).get("package_dir")
    package_path = Path(str(package)).expanduser() if package else None
    if package_path is not None and not package_path.is_absolute():
        package_path = settings_module.PROJECT_ROOT / package_path
    if (str(package_path.resolve()) if package_path is not None else None) != (
        str(Path(active_package_dir).resolve()) if active_package_dir else None
    ):
        pending.append("paths.package_dir")
    application = document.get("application") or {}
    if active_host is not None and str(application.get("host")) != str(active_host):
        pending.append("application.host")
    if active_port is not None and int(application.get("port", 0)) != int(active_port):
        pending.append("application.port")
    if active_max_workers is not None and int(application.get("max_workers", 0)) != int(active_max_workers):
        pending.append("application.max_workers")
    market_data = document.get("market_data") or {}
    if (
        active_scheduler_poll_seconds is not None
        and float(market_data.get("scheduler_poll_seconds", 0)) != float(active_scheduler_poll_seconds)
    ):
        pending.append("market_data.scheduler_poll_seconds")
    return pending


def configuration_schema() -> dict[str, Any]:
    settings_schema = ConfigurationDocument.model_json_schema()
    defaults = ConfigurationDocument().model_dump(mode="json")
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "revision": "sha256",
        "groups": {
            "application": {"label": "应用", "restart_required": True},
            "paths": {"label": "路径", "restart_required": True},
            "backtest": {"label": "回测默认值", "restart_required": False},
            "research": {"label": "研究", "restart_required": False},
            "market_data": {"label": "行情数据", "restart_required": False},
            "runtime": {"label": "运行资源", "restart_required": False},
            "ui": {"label": "界面交互", "restart_required": False},
            "schedule": {"label": "定时更新", "restart_required": False},
        },
        "json_schema": settings_schema,
        "fields": _field_catalog(defaults, settings_schema),
        "auxiliary_schemas": {"ui": UiSettings.model_json_schema()},
        "units": {
            "research.alpha": "小数；界面显示百分比",
            "research.tau": "训练区间累计超额收益率小数",
            "research.beta": "评分惩罚系数",
        },
    }


def validate_configuration_patch(patch: dict[str, Any], *, base_revision: str | None = None) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("配置补丁必须是对象")
    # The revision is derived from the document and returned alongside it.
    # It is never editable, but a browser draft saved while another autosave
    # was in flight can still carry the old value in its patch.
    patch = {key: value for key, value in patch.items() if key != "config_revision"}
    unknown = sorted(set(patch) - _PATCH_GROUPS)
    if unknown:
        raise ValueError("不支持的配置分组：" + "、".join(unknown))
    document = current_configuration()
    candidate = _merge(document, patch)
    candidate.pop("config_revision", None)
    try:
        settings = settings_from_document(candidate)
    except (ValidationError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    candidate = configuration_document(settings)
    return candidate


def patch_configuration(patch: dict[str, Any], *, base_revision: str) -> dict[str, Any]:
    if not base_revision:
        raise ValueError("保存配置必须提供 base_revision")
    patch = {key: value for key, value in patch.items() if key != "config_revision"}
    try:
        document, _ = save_configuration_document(
            _merge(current_configuration(), patch), base_revision=base_revision,
        )
    except RuntimeError as exc:
        if str(exc) == "CONFIG_REVISION_CONFLICT":
            raise ConfigurationConflict("配置文件已被其他页面或手工编辑修改，请刷新后重试") from exc
        raise
    return document


def task_configuration_snapshot(
    *,
    active_data_dir: str | None = None,
    active_package_dir: str | None = None,
    active_host: str | None = None,
    active_port: int | None = None,
) -> dict[str, Any]:
    document = current_configuration()
    settings = load_settings()
    # Restart-only paths and listeners are frozen from the values actually held
    # by the running application, not from a newer saved value that is still
    # waiting for restart.  The saved document and its revision remain attached
    # for provenance.
    if active_data_dir is not None:
        settings.data_dir = str(active_data_dir)
    if active_package_dir is not None:
        settings.package_dir = str(active_package_dir)
    if active_host is not None:
        settings.host = str(active_host)
    if active_port is not None:
        settings.port = int(active_port)
    return {
        "revision": document.get("config_revision"),
        "document": document,
        "settings": settings.model_dump(mode="json"),
        "active": {
            "data_dir": settings.data_dir,
            "package_dir": settings.package_dir,
            "host": settings.host,
            "port": settings.port,
        },
    }
