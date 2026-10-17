from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable

import optuna
import pandas as pd

from .models import BacktestConfig, DataBundle, OptimizationConfig, SearchRange


Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

_SUPPORTED_RANGES = {
    "initial_shares", "min_shares", "max_shares", "buy_shares", "sell_shares",
    "buy_step",
    "sell_step",
    "grid_step",
    "buy_ratio",
    "sell_ratio",
    "initial_position",
    "min_position",
    "max_position",
}
_POSITION_RANGES = {"initial_position", "min_position", "max_position"}
_STEP_RANGES = {"buy_step", "sell_step", "grid_step"}


def run_backtest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load the engine only when optimization is actually invoked."""
    from .engine import run_backtest as engine_run_backtest

    return engine_run_backtest(*args, **kwargs)


def _check_ranges(config: OptimizationConfig) -> None:
    unknown = sorted(set(config.ranges) - _SUPPORTED_RANGES)
    if unknown:
        raise ValueError(f"不支持的搜索参数: {', '.join(unknown)}")

    for name, search_range in config.ranges.items():
        if name in {"buy_step", "sell_step", "grid_step"} and name not in (("buy_step", "sell_step") if config.base.grid_mode == "trailing" else ("grid_step",)):
            continue
        low, high = float(search_range.low), float(search_range.high)
        step = search_range.step
        if name.endswith("_shares"):
            if config.base.schema_version != 2 or low < (100 if name in {"buy_shares", "sell_shares"} else 0) or any(value % 100 for value in [low, high, step or 100]):
                raise ValueError(f"股数搜索范围 {name} 必须以 100 为单位，交易量至少 100")
        elif config.base.schema_version == 2 and name in {"buy_ratio", "sell_ratio", "initial_position", "min_position", "max_position"}:
            raise ValueError("新版配置应搜索股数，不能搜索资金或仓位比例")
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(f"搜索区间 {name} 必须是有限数值")
        if step is not None and (not math.isfinite(float(step)) or float(step) <= 0):
            raise ValueError(f"搜索区间 {name} 的步长必须为正数")
        if name in _STEP_RANGES and (low <= 0 or high > 1):
            raise ValueError(f"搜索区间 {name} 必须位于 (0, 1]")
        if name in {"buy_ratio", "sell_ratio"} and (low <= 0 or high > 1):
            raise ValueError(f"搜索区间 {name} 必须位于 (0, 1]")
        if name in _POSITION_RANGES and (low < 0 or high > 1):
            raise ValueError(f"搜索区间 {name} 必须位于 [0, 1]")


def _as_range(value: SearchRange | dict[str, Any]) -> SearchRange:
    return value if isinstance(value, SearchRange) else SearchRange.model_validate(value)


def _sample(trial: optuna.Trial, name: str, search_range: SearchRange) -> float:
    r = _as_range(search_range)
    if r.step is None:
        return float(trial.suggest_float(name, r.low, r.high))
    return float(trial.suggest_float(name, r.low, r.high, step=r.step))


def _number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normalize_dates(values: Any) -> pd.Series:
    """Parse date-like values, including compact YYYYMMDD action dates."""
    source = pd.Series(values, copy=False)
    parsed = pd.to_datetime(source, errors="coerce")
    if not isinstance(parsed, pd.Series):
        parsed = pd.Series(parsed, index=source.index)

    # Some upstream feeds encode dates as eight-digit strings. pandas accepts
    # most of them directly, but explicitly retry values it could not parse.
    text_values = source.astype("string").str.strip()
    compact = text_values.str.fullmatch(r"\d{8}", na=False)
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(
            text_values.loc[compact], format="%Y%m%d", errors="coerce"
        )

    # DataBundle timestamps are normally naive; normalize aware inputs to
    # Shanghai wall time with the timezone marker removed for date comparison.
    try:
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
        return parsed.dt.normalize()
    except (AttributeError, TypeError):
        values_out: list[pd.Timestamp | pd.NaT] = []
        for item in parsed.tolist():
            if pd.isna(item):
                values_out.append(pd.NaT)
                continue
            timestamp = pd.Timestamp(item)
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_localize(None)
            values_out.append(timestamp.normalize())
        return pd.Series(values_out, index=source.index, dtype="datetime64[ns]")


def _dates_in_frame(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return _normalize_dates(frame[column])


def _calendar_dates(data: DataBundle) -> pd.DatetimeIndex:
    if len(data.calendar):
        dates = _normalize_dates(pd.Series(data.calendar)).dropna()
    elif "date" in data.daily.columns and not data.daily.empty:
        dates = _dates_in_frame(data.daily, "date").dropna()
    elif "timestamp" in data.bars.columns and not data.bars.empty:
        dates = _dates_in_frame(data.bars, "timestamp").dropna()
    else:
        dates = pd.Series([], dtype="datetime64[ns]")
    return pd.DatetimeIndex(sorted(pd.unique(dates)))


def _quality_days(data: DataBundle) -> list[dict[str, Any]]:
    day_rows = data.quality.get("days", []) if isinstance(data.quality, dict) else []
    if isinstance(day_rows, pd.DataFrame):
        return day_rows.to_dict(orient="records")
    if isinstance(day_rows, list):
        return [dict(row) for row in day_rows if isinstance(row, dict)]
    return []


def _slice_bundle(data: DataBundle, start: pd.Timestamp, end: pd.Timestamp) -> DataBundle:
    """Return an independent-account bundle bounded by inclusive trading dates.

    Corporate actions are retained only when their registration date falls in
    this segment. This prevents a validation/test account with no pre-segment
    holdings from receiving an earlier dividend. The segment's own registration
    dates remain available to the engine.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    bars = data.bars.copy()
    if not bars.empty and "timestamp" in bars.columns:
        bar_dates = _dates_in_frame(bars, "timestamp")
        bars = bars.loc[bar_dates.between(start, end)].copy()

    daily = data.daily.copy()
    if not daily.empty and "date" in daily.columns:
        daily_dates = _dates_in_frame(daily, "date")
        daily = daily.loc[daily_dates.between(start, end)].copy()

    calendar = pd.DatetimeIndex(data.calendar)
    if len(calendar):
        calendar_dates = _normalize_dates(pd.Series(calendar))
        calendar = pd.DatetimeIndex(calendar[calendar_dates.between(start, end).to_numpy()])
    else:
        calendar = pd.DatetimeIndex([])

    actions = data.actions.copy()
    if not actions.empty:
        registration_column = "record_date" if "record_date" in actions.columns else None
        if registration_column is None:
            registration_column = next(
                (column for column in ("ex_date", "stock_date", "pay_date") if column in actions.columns),
                None,
            )
        if registration_column is not None:
            action_dates = _dates_in_frame(actions, registration_column)
            if "prior_closes" in data.metadata and "ex_date" in actions:
                actions = actions.loc[(_dates_in_frame(actions, "ex_date") >= start) & (action_dates <= end)].copy()
            else:
                actions = actions.loc[action_dates.between(start, end)].copy()
        else:
            actions = actions.iloc[0:0].copy()

    quality = deepcopy(data.quality) if isinstance(data.quality, dict) else {}
    all_days = _quality_days(data)
    selected_days: list[dict[str, Any]] = []
    for report in all_days:
        report_date = _normalize_dates(pd.Series([report.get("date")])).iloc[0]
        if pd.notna(report_date) and start <= report_date <= end:
            selected_days.append(report)

    quality["days"] = selected_days
    modes = [str(row.get("mode", "")).lower() for row in selected_days]
    minute_days = sum(mode == "minute" for mode in modes)
    daily_days = sum(mode == "daily" for mode in modes)
    suspended_days = sum(mode == "suspended" for mode in modes)
    if selected_days:
        denominator = minute_days + daily_days
        quality["minute_ratio"] = minute_days / denominator if denominator else 0.0
        quality["minute_days"] = minute_days
        quality["daily_days"] = daily_days
        quality["suspended_days"] = suspended_days
        quality["total_days"] = len(selected_days)
    else:
        # Without per-day reports preserve a known aggregate ratio as the best
        # available quality estimate; otherwise the robust gate will fail shut.
        quality["minute_ratio"] = _number(quality.get("minute_ratio"), 0.0)
        quality["minute_days"] = int(_number(quality.get("minute_days"), 0))
        quality["daily_days"] = int(_number(quality.get("daily_days"), 0))
        quality["suspended_days"] = int(_number(quality.get("suspended_days"), 0))
        quality["total_days"] = len(calendar)

    if "is_synthetic" in bars.columns:
        quality["synthetic_bars"] = int(bars["is_synthetic"].fillna(False).astype(bool).sum())
    elif not selected_days:
        quality["synthetic_bars"] = int(_number(quality.get("synthetic_bars"), 0))

    if selected_days:
        report_warnings = [
            f"{row.get('date')}: {row.get('reason')}"
            for row in selected_days
            if row.get("reason")
        ]
        quality["warnings"] = list(dict.fromkeys(report_warnings))

    return DataBundle(
        symbol=data.symbol,
        bars=bars,
        daily=daily,
        actions=actions,
        calendar=calendar,
        quality=quality,
        version=data.version,
        metadata=deepcopy(data.metadata),
        warnings=list(data.warnings),
    )


def _minute_ratio(data: DataBundle) -> float:
    quality = data.quality if isinstance(data.quality, dict) else {}
    value = _number(quality.get("minute_ratio"))
    if math.isfinite(value):
        return max(0.0, min(1.0, value))
    day_rows = _quality_days(data)
    minute_days = sum(str(row.get("mode", "")).lower() == "minute" for row in day_rows)
    daily_days = sum(str(row.get("mode", "")).lower() == "daily" for row in day_rows)
    denominator = minute_days + daily_days
    return minute_days / denominator if denominator else 0.0


def _split_dates(data: DataBundle) -> dict[str, dict[str, Any]] | None:
    dates = _calendar_dates(data)
    if len(dates) < 3:
        return None
    train_end = max(1, int(len(dates) * 0.60))
    validation_end = max(train_end + 1, int(len(dates) * 0.80))
    validation_end = min(validation_end, len(dates) - 1)
    groups = {
        "train": dates[:train_end],
        "validation": dates[train_end:validation_end],
        "test": dates[validation_end:],
    }
    if any(len(group) == 0 for group in groups.values()):
        return None
    return {
        name: {
            "start": group[0].date().isoformat(),
            "end": group[-1].date().isoformat(),
            "days": int(len(group)),
        }
        for name, group in groups.items()
    }


def _group_dates(split: dict[str, dict[str, Any]]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        name: (pd.Timestamp(bounds["start"]), pd.Timestamp(bounds["end"]))
        for name, bounds in split.items()
    }


def _quantity_as_ratio(config: BacktestConfig, side: str, data: DataBundle) -> float:
    quantity = getattr(config, f"{side}_quantity")
    if quantity.mode == "equity_ratio":
        return min(1.0, max(0.000001, float(quantity.value)))
    if quantity.mode == "cash":
        amount = float(quantity.value) / float(config.initial_cash)
    else:
        reference = config.reference_price
        if reference is None and not data.daily.empty and "close" in data.daily.columns:
            reference = _number(data.daily.iloc[0]["close"], math.nan)
        if reference is None or not math.isfinite(float(reference)):
            if not data.bars.empty and "close" in data.bars.columns:
                reference = _number(data.bars.iloc[0]["close"], math.nan)
        amount = float(quantity.value) * float(reference or 0) / float(config.initial_cash)
    return min(1.0, max(0.000001, amount))


def _parameters_for_trial(
    trial: optuna.Trial, config: OptimizationConfig, data: DataBundle
) -> dict[str, Any]:
    base = config.base
    ranges = config.ranges
    params: dict[str, Any] = {}

    active_steps = ("buy_step", "sell_step") if base.grid_mode == "trailing" else ("grid_step",)
    for name in active_steps:
        params[name] = (
            _sample(trial, name, ranges[name]) if name in ranges else float(getattr(base, name))
        )

    if base.schema_version == 2:
        for name in ("initial_shares", "min_shares", "max_shares", "buy_shares", "sell_shares"):
            if name in ranges:
                r = ranges[name]
                params[name] = trial.suggest_int(name, int(r.low), int(r.high), step=int(r.step or 100))
            else:
                params[name] = int(getattr(base, name))
        return params

    for name in ("initial_position", "min_position", "max_position"):
        params[name] = (
            _sample(trial, name, ranges[name]) if name in ranges else float(getattr(base, name))
        )

    buy_ratio = _sample(trial, "buy_ratio", ranges["buy_ratio"]) if "buy_ratio" in ranges else _quantity_as_ratio(base, "buy", data)
    sell_ratio = _sample(trial, "sell_ratio", ranges["sell_ratio"]) if "sell_ratio" in ranges else _quantity_as_ratio(base, "sell", data)
    params["buy_quantity"] = {"mode": "equity_ratio", "value": buy_ratio}
    params["sell_quantity"] = {"mode": "equity_ratio", "value": sell_ratio}
    return params


def _config_with_params(base: BacktestConfig, params: dict[str, Any], start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> BacktestConfig:
    values = base.model_dump(mode="python")
    values.update(deepcopy(params))
    if start is not None:
        values["start_date"] = start.date()
    if end is not None:
        values["end_date"] = end.date()
    return BacktestConfig.model_validate(values)


def _grid_metrics(result: dict[str, Any]) -> dict[str, Any]:
    strategies = result.get("strategies", {}) if isinstance(result, dict) else {}
    grid = strategies.get("grid", {}) if isinstance(strategies, dict) else {}
    metrics = grid.get("metrics", {}) if isinstance(grid, dict) else {}
    return dict(metrics) if isinstance(metrics, dict) else {}


def _calmar(metrics: dict[str, Any]) -> float:
    cagr = _number(metrics.get("cagr"))
    drawdown = _number(metrics.get("max_drawdown"))
    if not math.isfinite(cagr) or not math.isfinite(drawdown):
        return math.nan
    return cagr / max(drawdown, 0.01)


def _is_cancelled(callback: Cancelled | None) -> bool:
    return bool(callback and callback())


def _emit(progress: Progress | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, float(fraction))), message)


def _run(
    backtest_config: BacktestConfig,
    data: DataBundle,
    *,
    summary_only: bool,
    progress: Progress | None,
    cancelled: Cancelled | None,
) -> dict[str, Any]:
    result = run_backtest(
        backtest_config,
        data,
        progress=progress,
        cancelled=cancelled,
        summary_only=summary_only,
    )
    if not isinstance(result, dict):
        raise TypeError("run_backtest 必须返回字典")
    return result


def _trial_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _base_result(config: OptimizationConfig, data: DataBundle) -> dict[str, Any]:
    warnings = list(data.warnings)
    quality = data.quality if isinstance(data.quality, dict) else {}
    warnings.extend(str(item) for item in quality.get("warnings", []) if item)
    warnings = list(dict.fromkeys(warnings))
    return {
        "config": config.model_dump(mode="json"),
        "quality": deepcopy(quality),
        "is_demo": bool(quality.get("is_demo") or data.metadata.get("is_demo")),
        "mode": config.mode,
        "seed": config.seed,
        "n_trials": config.n_trials,
        "data_version": data.version,
        "label": f"{config.base.symbol} 参数搜索（{config.mode}）",
        "warnings": warnings,
    }


def _new_study(seed: int) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )


def _valid_config(base: BacktestConfig, params: dict[str, Any], start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> tuple[BacktestConfig | None, str | None]:
    try:
        return _config_with_params(base, params, start, end), None
    except Exception as exc:
        return None, _trial_error(exc)


def _historical(
    config: OptimizationConfig,
    data: DataBundle,
    progress: Progress | None,
    cancelled: Cancelled | None,
    fraction_start: float,
    fraction_trials: float,
    fraction_finalize: float,
) -> dict[str, Any]:
    study = _new_study(config.seed)
    candidates: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    cancelled_run = False
    n = config.n_trials

    for index in range(n):
        if _is_cancelled(cancelled):
            cancelled_run = True
            break
        trial = study.ask()
        record: dict[str, Any] = {"trial": index + 1, "status": "failed", "params": {}}
        try:
            params = _parameters_for_trial(trial, config, data)
            record["params"] = params
            backtest_config, validation_error = _valid_config(config.base, params)
            if validation_error:
                record.update(status="invalid_constraints", reason=validation_error, score=-1e12)
                study.tell(trial, -1e12)
            else:
                start_fraction = fraction_start + fraction_trials * index / n
                slot = fraction_trials / n
                inner_progress = (
                    lambda fraction, message, base_fraction=start_fraction, width=slot: _emit(
                        progress, base_fraction + width * max(0.0, min(1.0, fraction)), message
                    )
                ) if progress else None
                result = _run(
                    backtest_config,
                    data,
                    summary_only=True,
                    progress=inner_progress,
                    cancelled=cancelled,
                )
                if result.get("status") == "cancelled" or result.get("cancelled"):
                    record.update(status="cancelled", reason="用户取消")
                    study.tell(trial, -1e12)
                    candidates.append(record)
                    cancelled_run = True
                    break
                metrics = _grid_metrics(result)
                net_return = _number(metrics.get("total_return"))
                if not math.isfinite(net_return):
                    raise ValueError("回测未返回有效的 total_return")
                record.update(status="completed", metrics=metrics, score=net_return)
                study.tell(trial, net_return)
                completed.append(record)
        except Exception as exc:
            if _is_cancelled(cancelled):
                record.update(status="cancelled", reason="用户取消", score=-1e12)
            else:
                record.update(status="failed", reason=_trial_error(exc), score=-1e12)
            study.tell(trial, -1e12)
            if _is_cancelled(cancelled):
                cancelled_run = True
                candidates.append(record)
                break
        candidates.append(record)
        _emit(progress, fraction_start + fraction_trials * (index + 1) / n, f"历史搜索：{index + 1}/{n}")

    best = max(completed, key=lambda candidate: (candidate["score"], -candidate["trial"])) if completed else None
    best_result: dict[str, Any] | None = None
    best_params = deepcopy(best["params"]) if best else None
    status = "cancelled" if cancelled_run else ("completed" if best else "failed")

    if best and not cancelled_run and not _is_cancelled(cancelled):
        best_config, validation_error = _valid_config(config.base, best["params"])
        if validation_error:
            status = "failed"
        else:
            try:
                best_result = _run(
                    best_config,
                    data,
                    summary_only=False,
                    progress=(lambda fraction, message: _emit(
                        progress, fraction_start + fraction_trials + fraction_finalize * max(0.0, min(1.0, fraction)), message
                    )) if progress else None,
                    cancelled=cancelled,
                )
                if best_result.get("status") == "cancelled" or best_result.get("cancelled"):
                    best_result = None
                    status = "cancelled"
            except Exception:
                if _is_cancelled(cancelled):
                    status = "cancelled"
                else:
                    status = "best_result_failed"
    elif best and not cancelled_run and _is_cancelled(cancelled):
        status = "cancelled"

    section: dict[str, Any] = {
        "status": status,
        "best_params": best_params,
        "candidates": candidates,
        "best_result": best_result,
    }
    return section


def _robust(
    config: OptimizationConfig,
    data: DataBundle,
    progress: Progress | None,
    cancelled: Cancelled | None,
    fraction_start: float,
    fraction_train: float,
    fraction_validation: float,
) -> dict[str, Any]:
    split = _split_dates(data)
    days = len(_calendar_dates(data))
    ratio = _minute_ratio(data)
    unavailable_reasons: list[str] = []
    if days < config.min_history_days:
        unavailable_reasons.append(f"历史交易日不足：{days} < {config.min_history_days}")
    if ratio < config.min_minute_ratio:
        unavailable_reasons.append(
            f"分钟数据保留率不足：{ratio:.1%} < {config.min_minute_ratio:.1%}"
        )
    if split is None:
        unavailable_reasons.append("历史日期不足，无法按时间切分训练、验证和测试区间")
    if unavailable_reasons:
        return {
            "status": "unavailable",
            "reason": "；".join(unavailable_reasons),
            **({"split": split} if split else {}),
        }

    assert split is not None
    ranges = _group_dates(split)
    train_data = _slice_bundle(data, *ranges["train"])
    validation_data = _slice_bundle(data, *ranges["validation"])
    test_data = _slice_bundle(data, *ranges["test"])

    study = _new_study(config.seed)
    candidates: list[dict[str, Any]] = []
    cancelled_run = False
    first_error: str | None = None
    n = config.n_trials

    for index in range(n):
        if _is_cancelled(cancelled):
            cancelled_run = True
            break
        trial = study.ask()
        record: dict[str, Any] = {"trial": index + 1, "status": "failed", "params": {}}
        try:
            params = _parameters_for_trial(trial, config, train_data)
            record["params"] = params
            train_config, validation_error = _valid_config(config.base, params, *ranges["train"])
            if validation_error:
                record.update(status="invalid_constraints", reason=validation_error, score=-1e12)
                study.tell(trial, -1e12)
            else:
                start_fraction = fraction_start + fraction_train * index / n
                slot = fraction_train / n
                inner_progress = (
                    lambda fraction, message, base_fraction=start_fraction, width=slot: _emit(
                        progress, base_fraction + width * max(0.0, min(1.0, fraction)), message
                    )
                ) if progress else None
                train_result = _run(
                    train_config,
                    train_data,
                    summary_only=True,
                    progress=inner_progress,
                    cancelled=cancelled,
                )
                if train_result.get("status") == "cancelled" or train_result.get("cancelled"):
                    record.update(status="cancelled", reason="用户取消")
                    study.tell(trial, -1e12)
                    candidates.append(record)
                    cancelled_run = True
                    break
                metrics = _grid_metrics(train_result)
                score = _calmar(metrics)
                if not math.isfinite(score):
                    raise ValueError("训练回测未返回有效的 cagr/max_drawdown")
                record.update(status="not_validated", train_metrics=metrics, score=score)
                study.tell(trial, score)
        except Exception as exc:
            if _is_cancelled(cancelled):
                record.update(status="cancelled", reason="用户取消", score=-1e12)
            else:
                record.update(status="failed", reason=_trial_error(exc), score=-1e12)
                if first_error is None:
                    first_error = record["reason"]
            study.tell(trial, -1e12)
            if _is_cancelled(cancelled):
                cancelled_run = True
                candidates.append(record)
                break
        candidates.append(record)
        _emit(progress, fraction_start + fraction_train * (index + 1) / n, f"稳健训练搜索：{index + 1}/{n}")

    if cancelled_run:
        return {
            "status": "cancelled",
            "reason": "用户取消，已保留完成的训练试验",
            "split": split,
            "candidates": candidates,
        }

    trained = [candidate for candidate in candidates if candidate.get("status") == "not_validated"]
    trained.sort(key=lambda c: (-c["score"], _number(c["train_metrics"].get("max_drawdown"), math.inf), _number(c["train_metrics"].get("turnover"), math.inf), c["trial"]))
    top = trained[: config.top_candidates]
    for index, candidate in enumerate(top):
        if _is_cancelled(cancelled):
            cancelled_run = True
            break
        params = candidate["params"]
        validation_config, validation_error = _valid_config(config.base, params, *ranges["validation"])
        if validation_error:
            candidate.update(status="validation_failed", validation_reason=validation_error)
            continue
        start_fraction = fraction_start + fraction_train + fraction_validation * index / max(1, len(top))
        slot = fraction_validation / max(1, len(top))
        inner_progress = (
            lambda fraction, message, base_fraction=start_fraction, width=slot: _emit(
                progress, base_fraction + width * max(0.0, min(1.0, fraction)), message
            )
        ) if progress else None
        try:
            validation_result = _run(
                validation_config,
                validation_data,
                summary_only=True,
                progress=inner_progress,
                cancelled=cancelled,
            )
            if validation_result.get("status") == "cancelled" or validation_result.get("cancelled"):
                candidate.update(status="cancelled", validation_reason="用户取消")
                cancelled_run = True
                break
            metrics = _grid_metrics(validation_result)
            validation_score = _calmar(metrics)
            candidate["validation_metrics"] = metrics
            candidate["validation_score"] = validation_score
            if not math.isfinite(validation_score):
                candidate.update(status="validation_failed", validation_reason="验证回测指标无效")
            elif (
                _number(metrics.get("max_drawdown"), math.inf) <= config.max_drawdown
                and int(_number(metrics.get("trade_count"), 0)) >= config.min_trades
                and int(_number(metrics.get("buy_count"), 0)) > 0
                and int(_number(metrics.get("sell_count"), 0)) > 0
            ):
                candidate["status"] = "eligible"
            else:
                candidate["status"] = "validation_rejected"
                candidate["validation_reason"] = "验证段未满足回撤、成交次数或买卖双向要求"
        except Exception as exc:
            if _is_cancelled(cancelled):
                candidate.update(status="cancelled", validation_reason="用户取消")
                cancelled_run = True
                break
            candidate.update(status="validation_failed", validation_reason=_trial_error(exc))
            if first_error is None:
                first_error = candidate["validation_reason"]
        _emit(progress, start_fraction + slot, f"稳健验证：{index + 1}/{len(top)}")

    if cancelled_run:
        return {
            "status": "cancelled",
            "reason": "用户取消，已保留已完成的训练和验证结果",
            "split": split,
            "candidates": candidates,
        }

    eligible = [candidate for candidate in top if candidate.get("status") == "eligible"]
    eligible.sort(key=lambda c: (
        -c["validation_score"],
        _number(c["validation_metrics"].get("max_drawdown"), math.inf),
        _number(c["validation_metrics"].get("turnover"), math.inf),
        -c["score"],
        c["trial"],
    ))
    if not eligible:
        return {
            "status": "no_qualifying_candidate",
            "reason": first_error or "前列训练候选均未通过验证段筛选",
            "split": split,
            "candidates": candidates,
        }

    selected = eligible[0]
    selected_params = deepcopy(selected["params"])
    test_config, validation_error = _valid_config(config.base, selected_params, *ranges["test"])
    if validation_error:
        return {
            "status": "failed",
            "reason": validation_error,
            "split": split,
            "candidates": candidates,
            "selected_params": selected_params,
            "train_metrics": deepcopy(selected["train_metrics"]),
            "validation_metrics": deepcopy(selected["validation_metrics"]),
        }
    if _is_cancelled(cancelled):
        return {
            "status": "cancelled",
            "reason": "用户取消，已保留选定参数和验证指标",
            "split": split,
            "candidates": candidates,
            "selected_params": selected_params,
            "train_metrics": deepcopy(selected["train_metrics"]),
            "validation_metrics": deepcopy(selected["validation_metrics"]),
        }

    try:
        test_result = _run(
            test_config,
            test_data,
            summary_only=False,
            progress=(lambda fraction, message: _emit(
                progress,
                fraction_start + fraction_train + fraction_validation + (1 - fraction_start - fraction_train - fraction_validation) * max(0.0, min(1.0, fraction)),
                message,
            )) if progress else None,
            cancelled=cancelled,
        )
        if test_result.get("status") == "cancelled" or test_result.get("cancelled"):
            return {
                "status": "cancelled",
                "reason": "用户取消，已保留选定参数和训练、验证指标",
                "split": split,
                "candidates": candidates,
                "selected_params": selected_params,
                "train_metrics": deepcopy(selected["train_metrics"]),
                "validation_metrics": deepcopy(selected["validation_metrics"]),
            }
        test_metrics = _grid_metrics(test_result)
        return {
            "status": "completed",
            "split": split,
            "selected_params": selected_params,
            "candidates": candidates,
            "train_metrics": deepcopy(selected["train_metrics"]),
            "validation_metrics": deepcopy(selected["validation_metrics"]),
            "test_metrics": test_metrics,
            "test_result": test_result,
        }
    except Exception as exc:
        if _is_cancelled(cancelled):
            return {
                "status": "cancelled",
                "reason": "用户取消，已保留选定参数和训练、验证指标",
                "split": split,
                "selected_params": selected_params,
                "candidates": candidates,
                "train_metrics": deepcopy(selected["train_metrics"]),
                "validation_metrics": deepcopy(selected["validation_metrics"]),
            }
        return {
            "status": "test_failed",
            "reason": _trial_error(exc),
            "split": split,
            "selected_params": selected_params,
            "candidates": candidates,
            "train_metrics": deepcopy(selected["train_metrics"]),
            "validation_metrics": deepcopy(selected["validation_metrics"]),
        }


def optimize(
    config: OptimizationConfig,
    data: DataBundle,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> dict[str, Any]:
    """Search grid parameters and optionally validate them out of sample.

    Historical mode chooses the highest net return among the sampled trials.
    Robust mode uses a chronological 60/20/20 split, ranks training trials by
    Calmar, validates the configured leading candidates, and evaluates the
    selected parameters once on the untouched test period.
    """
    _check_ranges(config)
    if config.protocol_version == 2:
        from .scientific import optimize_scientific

        return optimize_scientific(config, data, progress, cancelled)
    result = _base_result(config, data)
    if progress:
        _emit(progress, 0.0, "准备参数搜索")

    if config.mode == "historical":
        result["historical"] = _historical(
            config,
            data,
            progress,
            cancelled,
            fraction_start=0.0,
            fraction_trials=0.9,
            fraction_finalize=0.09,
        )
    elif config.mode == "robust":
        robust = _robust(
            config,
            data,
            progress,
            cancelled,
            fraction_start=0.0,
            fraction_train=0.45,
            fraction_validation=0.45,
        )
        result["robust"] = robust
    else:
        result["historical"] = _historical(
            config,
            data,
            progress,
            cancelled,
            fraction_start=0.0,
            fraction_trials=0.45,
            fraction_finalize=0.04,
        )
        result["robust"] = _robust(
            config,
            data,
            progress,
            cancelled,
            fraction_start=0.5,
            fraction_train=0.25,
            fraction_validation=0.2,
        )

    _emit(progress, 1.0, "参数搜索完成")
    return result
