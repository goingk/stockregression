"""Leakage-aware rolling parameter selection for share-based research."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from itertools import product
import math
from typing import Any, Callable
import zlib

import numpy as np
import optuna
import pandas as pd

from .models import BacktestConfig, DataBundle, OptimizationConfig, SearchRange
from . import optimization as legacy_optimization
from .research_statistics import paired_stationary_bootstrap

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

_ACTIVE_FIELDS = {
    "trailing": ("buy_step", "sell_step"),
    "arithmetic": ("buy_step", "sell_step"),
    "geometric": ("buy_step", "sell_step"),
}
_SHARE_FIELDS = ("min_shares", "max_shares", "buy_shares", "sell_shares")
_COST_MULTIPLIERS = (1.0, 2.0)
_LOOKBACK_DAYS = 504
_FULL_GENERATION_DAYS = 252
_FOLD_DAYS = 63
_MIN_SPLIT_DAYS = 168
_V3_POSITIVE_NEIGHBOR_RATIO = 0.8
_V3_MAX_VALIDATION_CENTERS = 10


def _cancelled(callback: Cancelled | None) -> bool:
    return bool(callback and callback())


def _emit(callback: Progress | None, fraction: float, message: str) -> None:
    if callback:
        callback(max(0.0, min(1.0, float(fraction))), message)


def _mapped_progress(callback: Progress | None, start: float, width: float) -> Progress | None:
    if callback is None:
        return None

    def mapped(fraction: float, message: str) -> None:
        _emit(callback, start + width * float(fraction), message)

    return mapped


def _date_key(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _calendar(data: DataBundle) -> pd.DatetimeIndex:
    return legacy_optimization._calendar_dates(data)


def _known_preceding_close(data: DataBundle, effective_date: pd.Timestamp | date | str) -> pd.Timestamp | None:
    effective = _date_key(effective_date)
    calendar = _calendar(data)
    candidates = list(calendar[calendar < effective])
    metadata = data.metadata if isinstance(data.metadata, dict) else {}
    reference_date = metadata.get("reference_date")
    if reference_date and float(metadata.get("valuation_price", 0) or 0) > 0:
        try:
            stamp = _date_key(reference_date)
            if stamp < effective:
                candidates.append(stamp)
        except (TypeError, ValueError):
            pass
    prior_closes = metadata.get("prior_closes", {})
    if isinstance(prior_closes, dict):
        for raw_date, raw_price in prior_closes.items():
            try:
                stamp = _date_key(raw_date)
                if stamp < effective and float(raw_price) > 0:
                    candidates.append(stamp)
            except (TypeError, ValueError):
                continue
    return max(candidates) if candidates else None


def _base_with(
    base: BacktestConfig,
    params: dict[str, Any] | None = None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    reserve_cash: float | None = None,
) -> BacktestConfig:
    values = base.model_dump(mode="python")
    if params:
        values.update(deepcopy(params))
    if start is not None:
        values["start_date"] = _date_key(start).date()
    if end is not None:
        values["end_date"] = _date_key(end).date()
    if reserve_cash is not None and "reserve_cash" in BacktestConfig.model_fields and base.schema_version != 3:
        values["reserve_cash"] = float(reserve_cash)
    return BacktestConfig.model_validate(values)


def _active_names(base: BacktestConfig) -> tuple[str, ...]:
    return (*_ACTIVE_FIELDS[base.grid_mode], *_SHARE_FIELDS)


def _range_for(config: OptimizationConfig, base: BacktestConfig, name: str) -> SearchRange:
    raw = config.ranges.get(name)
    if raw is not None:
        return raw if isinstance(raw, SearchRange) else SearchRange.model_validate(raw)
    value = float(getattr(base, name))
    return SearchRange(low=value, high=value)


def _share_step(search_range: SearchRange) -> int:
    return max(100, int(round(float(search_range.step or 100) / 100.0)) * 100)


def _suggest(trial: optuna.Trial, name: str, search_range: SearchRange) -> float | int:
    low, high = float(search_range.low), float(search_range.high)
    if name.endswith("_shares"):
        step = _share_step(search_range)
        return int(trial.suggest_int(name, int(round(low)), int(round(high)), step=step))
    if search_range.step is None:
        return float(trial.suggest_float(name, low, high))
    return float(trial.suggest_float(name, low, high, step=float(search_range.step)))


def _params_for_trial(
    trial: optuna.Trial,
    config: OptimizationConfig,
    base: BacktestConfig,
    names: tuple[str, ...],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name in names:
        search_range = _range_for(config, base, name)
        if search_range.low == search_range.high:
            params[name] = int(search_range.low) if name.endswith("_shares") else float(search_range.low)
        else:
            params[name] = _suggest(trial, name, search_range)
    # Initial holdings and the cash reserve are intentionally immutable.
    params["initial_shares"] = int(base.initial_shares)
    return params


def _params_key(params: dict[str, Any], names: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(int(params[name]) if name.endswith("_shares") else round(float(params[name]), 12) for name in names)


def _valid_params(params: dict[str, Any], base: BacktestConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    original = int(base.initial_shares)
    if int(params.get("min_shares", base.min_shares)) > original:
        reasons.append("min_shares 高于固定 initial_shares")
    raw_max = params.get("max_shares", base.max_shares)
    if raw_max is not None and int(raw_max) < original:
        reasons.append("max_shares 低于固定 initial_shares")
    if raw_max is not None and int(params.get("min_shares", base.min_shares)) > int(raw_max):
        reasons.append("min_shares 大于 max_shares")
    try:
        _base_with(base, params)
    except Exception as exc:
        reasons.append(f"参数配置无效：{type(exc).__name__}: {exc}"[:250])
    return not reasons, reasons


def _seed_offset_for_date(value: pd.Timestamp | date | str) -> int:
    """Stable seed component tied only to this decision's information date."""
    return int(zlib.crc32(str(_date_key(value).date()).encode("ascii")))


def _normalize_params(params: dict[str, Any], base: BacktestConfig) -> dict[str, Any]:
    normalized = {
        name: (None if name == "max_shares" and value is None else int(value) if name.endswith("_shares") else float(value))
        for name, value in params.items()
    }
    normalized["initial_shares"] = int(base.initial_shares)
    return normalized


def _trial_params(
    config: OptimizationConfig,
    base: BacktestConfig,
    names: tuple[str, ...],
    *,
    seed: int,
    generation_data: DataBundle,
    generation_dates: pd.DatetimeIndex,
    reserve_cash: float | None,
    progress: Progress | None,
    cancelled: Cancelled | None,
    progress_start: float,
    progress_width: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(seed) % (2**32)),
    )
    discovered: dict[tuple[Any, ...], dict[str, Any]] = {}
    trial_rows: list[dict[str, Any]] = []
    gen_start, gen_end = generation_dates[0], generation_dates[-1]
    generation_base = _base_with(base, start=gen_start, end=gen_end, reserve_cash=reserve_cash)
    cancelled_run = False

    pending: dict[tuple[Any, ...], list[tuple[optuna.Trial, dict[str, Any], dict[str, Any]]]] = {}
    generation_fold = [(str(gen_start.date()), str(gen_end.date()))]

    def emit_completed_rows() -> None:
        completed = len(trial_rows)
        _emit(progress, progress_start + progress_width * completed / max(1, config.n_trials), f"生成候选：{completed}/{config.n_trials}")

    def flush_pending() -> None:
        nonlocal cancelled_run
        if not pending:
            return
        keys = list(pending)
        params_batch = [pending[key][0][2] for key in keys]
        if _cancelled(cancelled):
            cancelled_run = True
            for key in keys:
                for trial, row, _params in pending[key]:
                    row.update(status="cancelled", reason="用户取消")
                    trial_rows.append(row)
            pending.clear()
            return
        try:
            result = _research_batch(
                generation_base,
                generation_data,
                params_batch,
                generation_fold,
                cost_multiplier=1.0,
                cancelled=cancelled,
            )
        except InterruptedError:
            cancelled_run = True
            for key in keys:
                for trial, row, _params in pending[key]:
                    row.update(status="cancelled", reason="用户取消")
                    trial_rows.append(row)
            pending.clear()
            return
        if result.get("status") == "cancelled" or result.get("cancelled"):
            cancelled_run = True
            for key in keys:
                for trial, row, _params in pending[key]:
                    row.update(status="cancelled", reason="用户取消")
                    trial_rows.append(row)
            pending.clear()
            return
        dates, daily_returns = _batch_returns(result, len(keys))
        exact_drawdowns = np.asarray(result.get("fold_drawdowns", []), dtype=np.float64)
        if exact_drawdowns.shape != (len(keys), 1):
            raise ValueError("generation research_batch.fold_drawdowns must have shape [candidates, 1]")
        scores = _window_scores(dates, daily_returns, [(gen_start, gen_end)], exact_drawdowns)
        for index, key in enumerate(keys):
            score = float(scores[index]) if index < len(scores) else -1e12
            if not math.isfinite(score):
                score = -1e12
            discovered[key] = {"params": params_batch[index], "generation_score": score}
            for trial, row, _params in pending[key]:
                row.update(status="completed", generation_score=score)
                study.tell(trial, score)
                trial_rows.append(row)
        pending.clear()
        emit_completed_rows()

    batch_limit = 64
    for index in range(int(config.n_trials)):
        if _cancelled(cancelled):
            cancelled_run = True
            break
        trial = study.ask()
        row: dict[str, Any] = {"trial": index + 1, "params": {}, "status": "failed", "generation_score": None}
        params = _normalize_params(_params_for_trial(trial, config, base, names), base)
        row["params"] = params
        valid, reasons = _valid_params(params, base)
        if not valid:
            row.update(status="invalid_constraints", reasons=reasons)
            study.tell(trial, -1e12)
            trial_rows.append(row)
            emit_completed_rows()
            continue
        key = _params_key(params, names)
        if key in discovered:
            score = discovered[key]["generation_score"]
            row.update(status="completed", generation_score=score)
            study.tell(trial, score)
            trial_rows.append(row)
            emit_completed_rows()
            continue
        if key not in pending:
            pending[key] = []
        pending[key].append((trial, row, params))
        # The kernel evaluates candidate families in parallel blocks of up to
        # 64.  Tell Optuna only after each measured block completes.
        if len(pending) >= batch_limit:
            flush_pending()
            if cancelled_run:
                break

    if not cancelled_run:
        flush_pending()
    elif pending:
        for group in pending.values():
            for _trial, row, _params in group:
                row.update(status="cancelled", reason="用户取消")
                trial_rows.append(row)
        pending.clear()

    generated = list(discovered.values())
    return [row["params"] for row in generated], trial_rows, cancelled_run


def _v3_names(base: BacktestConfig) -> tuple[str, ...]:
    names = list(_active_names(base))
    if base.schema_version == 3 and base.max_shares is None:
        names = [name for name in names if name != "max_shares"]
    return tuple(names)


def _range_distance(search_range: SearchRange, name: str) -> float:
    if search_range.low == search_range.high:
        return 0.0
    if name.endswith("_shares"):
        return float(_share_step(search_range))
    if search_range.step is not None:
        return float(search_range.step)
    return max((float(search_range.high) - float(search_range.low)) * 0.05, 1e-9)


def _snap_dimension(value: float, search_range: SearchRange, name: str) -> float:
    low, high = float(search_range.low), float(search_range.high)
    bounded = min(high, max(low, float(value)))
    if name.endswith("_shares"):
        return float(int(round(bounded / 100.0)) * 100)
    distance = _range_distance(search_range, name)
    if distance <= 0:
        return bounded
    return float(min(high, max(low, low + round((bounded - low) / distance) * distance)))


def _roundtrip_cost_rate(base: BacktestConfig, data: DataBundle, price: float, day: date) -> float:
    from .rules import commissions_and_taxes, round_money

    quantity = max(100, int(base.buy_shares))
    notional = round_money(quantity * price)
    kwargs = {
        "symbol": base.symbol,
        "quantity": quantity,
        "face_value": base.fees.share_face_value,
        "legacy_sh_transfer_rate": base.fees.legacy_sh_transfer_rate,
        "legacy_sh_min_transfer": base.fees.legacy_sh_min_transfer,
        "legacy_sz_transfer_rate": base.fees.legacy_sz_transfer_rate,
    }
    charges = 0.0
    for side in ("buy", "sell"):
        charge = commissions_and_taxes(
            notional, side, day, base.fees.commission_rate, base.fees.min_commission, **kwargs
        )
        charges += float(charge["total"])
    return (charges + notional * 2.0 * base.fees.slippage_bps / 10_000.0) / notional if notional > 0 else math.inf


def _statistical_profile(base: BacktestConfig, data: DataBundle, dates: pd.DatetimeIndex) -> dict[str, Any]:
    frame = data.daily.copy()
    if frame.empty or not len(dates):
        return {"status": "unavailable", "reason": "候选生成段没有日线行情", "quantiles": {}}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[frame["date"].between(dates[0], dates[-1])].sort_values("date")
    if frame.empty:
        return {"status": "unavailable", "reason": "候选生成段没有有效日线", "quantiles": {}}
    for name in ("open", "high", "low", "close", "preclose"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)].copy()
    if frame.empty:
        return {"status": "unavailable", "reason": "候选生成段没有有效价格", "quantiles": {}}
    if "preclose" not in frame or frame["preclose"].isna().any():
        frame["preclose"] = frame["preclose"].fillna(frame["close"].shift(1)).fillna(frame["open"])
    frame = frame[(frame["preclose"] > 0)]
    open_values = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    preclose = frame["preclose"].to_numpy(dtype=np.float64)
    returns = close / preclose - 1.0
    up = np.maximum(0.0, high - open_values) / open_values
    down = np.maximum(0.0, open_values - low) / open_values
    true_range = (np.maximum(high, preclose) - np.minimum(low, preclose)) / preclose
    repair = []
    for index in np.flatnonzero(returns < 0):
        if index + 5 >= len(returns):
            break
        repair.append(float(np.prod(1.0 + returns[index + 1:index + 6]) - 1.0))
    x = np.arange(len(close), dtype=np.float64)
    log_price = np.log(close)
    trend = float(np.polyfit(x, log_price, 1)[0] * 252.0) if len(close) > 2 else 0.0
    quantiles = {
        "up": [float(np.quantile(up, q)) for q in (0.25, 0.5, 0.75)],
        "down": [float(np.quantile(down, q)) for q in (0.25, 0.5, 0.75)],
        "range": [float(np.quantile(true_range, q)) for q in (0.25, 0.5, 0.75)],
    }
    price = float(np.median(close))
    tick = max(float(data.metadata.get("price_tick", 0.001) or 0.001), 1e-9)
    return {
        "status": "completed",
        "days": int(len(frame)),
        "start": frame["date"].iloc[0].date().isoformat(),
        "end": frame["date"].iloc[-1].date().isoformat(),
        "trend_strength": trend,
        "repair_ratio": float(np.median(repair)) if repair else None,
        "down_repair_samples": len(repair),
        "roundtrip_cost_rate": _roundtrip_cost_rate(base, data, price, frame["date"].iloc[-1].date()),
        "tick_excursion": 2.0 * tick / price,
        "quantile_labels": ["p25", "p50", "p75"],
        "quantiles": quantiles,
    }


def _statistical_centers(
    config: OptimizationConfig,
    base: BacktestConfig,
    data: DataBundle,
    generation_dates: pd.DatetimeIndex,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = _statistical_profile(base, data, generation_dates)
    names = _v3_names(base)
    baseline = {name: getattr(base, name) for name in names}
    baseline["initial_shares"] = int(base.initial_shares)
    if profile.get("status") != "completed":
        return [], {**profile, "centers": [], "rejected_centers": []}
    floor = max(2.0 * float(profile["roundtrip_cost_rate"]), float(profile["tick_excursion"]), 0.0)
    step_names = _ACTIVE_FIELDS[base.grid_mode]
    centers: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for quantile_index, label in enumerate(profile["quantile_labels"]):
        params = deepcopy(baseline)
        details: dict[str, float] = {}
        reasons: list[str] = []
        for name in step_names:
            if name == "buy_step":
                raw = profile["quantiles"]["down"][quantile_index]
            elif name == "sell_step":
                raw = profile["quantiles"]["up"][quantile_index]
            else:
                raw = profile["quantiles"]["range"][quantile_index]
            search_range = _range_for(config, base, name)
            target = max(float(raw), floor)
            value = _snap_dimension(target, search_range, name)
            distance = _range_distance(search_range, name)
            while value + 1e-12 < floor and distance > 0 and value + distance <= search_range.high + 1e-12:
                value = _snap_dimension(value + distance, search_range, name)
            if value < float(search_range.low) - 1e-12 or value > float(search_range.high) + 1e-12:
                reasons.append(f"{name} 成本下限 {floor:.6f} 超出搜索区间")
            elif value + 1e-12 < floor:
                reasons.append(f"{name} 合法刻度无法达到成本下限 {floor:.6f}")
            params[name] = value
            details[name] = value
        key = tuple(float(params[name]) for name in names)
        if reasons:
            rejected.append({"quantile": label, "params": params, "reasons": reasons, "steps": details})
            continue
        if any(tuple(float(item["params"][name]) for name in names) == key for item in centers):
            continue
        centers.append({
            "statistical_center_id": len(centers) + 1,
            "quantile": label,
            "params": params,
            "steps": details,
            "status": "completed",
        })
    profile.update({"floor": floor, "centers": deepcopy(centers), "rejected_centers": rejected})
    return centers, profile


def _local_grid_rows(
    config: OptimizationConfig,
    base: BacktestConfig,
    centers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, bool]:
    names = _v3_names(base)
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    total_expected = 0
    exceeded = False
    for center in centers:
        axes: list[list[float]] = []
        for name in names:
            current = float(center["params"][name])
            search_range = _range_for(config, base, name)
            if search_range.low == search_range.high:
                axes.append([_snap_dimension(float(search_range.low), search_range, name)])
                continue
            distance = _range_distance(search_range, name)
            values = {_snap_dimension(current + delta, search_range, name) for delta in (-distance, 0.0, distance)}
            axes.append(sorted(values))
        count = int(np.prod([len(values) for values in axes], dtype=np.int64)) if axes else 1
        total_expected += count
        if total_expected > int(config.local_search_budget):
            exceeded = True
            break
        for combination in product(*axes):
            params = {name: int(value) if name.endswith("_shares") else float(value) for name, value in zip(names, combination)}
            params["initial_shares"] = int(base.initial_shares)
            if base.max_shares is None:
                params["max_shares"] = None
            legal, reasons = _valid_params(params, base)
            if not legal:
                continue
            key = _params_key(params, names)
            if key not in rows_by_key:
                rows_by_key[key] = {"params": params, "source_center_ids": [], "training_status": "not_evaluated", "reasons": reasons}
            rows_by_key[key]["source_center_ids"] = sorted(set(rows_by_key[key]["source_center_ids"]) | {center["statistical_center_id"]})
    ordered = sorted(rows_by_key.items(), key=lambda item: item[0])
    rows = [{"search_id": index + 1, "key": key, **metadata} for index, (key, metadata) in enumerate(ordered)]
    return rows, total_expected, exceeded


def estimate_v3_search(config: OptimizationConfig, data: DataBundle) -> dict[str, Any]:
    """Return the statistical-center/local-grid size without running a backtest."""
    dates = _calendar(data)
    evaluation_start = _date_key(config.base.start_date)
    asof = _known_preceding_close(data, evaluation_start)
    history = dates[dates <= asof][-_LOOKBACK_DAYS:] if asof is not None else pd.DatetimeIndex([])
    generation, folds, has_oos, reasons = _folds_for_history(history)
    if not len(generation):
        return {"status": "insufficient_history", "expected": 0, "budget": int(config.local_search_budget), "reasons": reasons}
    centers, profile = _statistical_centers(config, config.base, data, generation)
    _, expected, exceeded = _local_grid_rows(config, config.base, centers)
    return {
        "status": "budget_exceeded" if exceeded else "ready",
        "expected": int(expected),
        "budget": int(config.local_search_budget),
        "statistical_center_count": len(centers),
        "generation_days": int(len(generation)),
        "validation_fold_count": int(len(folds)),
        "full_504_day_history": bool(has_oos and len(history) >= _LOOKBACK_DAYS),
        "profile": profile,
        "reasons": reasons,
    }


def _compound_daily(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    growth = 1.0 + values
    if not len(values) or np.any(growth <= 0) or not np.isfinite(growth).all():
        return math.nan
    return float(np.prod(growth) - 1.0)


def _buy_hold_returns(base: BacktestConfig, data: DataBundle) -> tuple[pd.DatetimeIndex, np.ndarray]:
    from .share_engine import research_buy_hold

    result = research_buy_hold(base, data)
    dates = pd.DatetimeIndex(pd.to_datetime(result["dates"])).normalize()
    values = np.asarray(result["daily_returns"], dtype=np.float64)
    if len(dates) != len(values):
        raise ValueError("buy_hold dates do not align with daily_returns")
    return dates, values


def _score_training_rows(
    config: OptimizationConfig,
    base: BacktestConfig,
    data: DataBundle,
    generation_dates: pd.DatetimeIndex,
    rows: list[dict[str, Any]],
    cancelled: Cancelled | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return rows, {"status": "empty", "evaluations": 0}
    start, end = pd.Timestamp(generation_dates[0]), pd.Timestamp(generation_dates[-1])
    training_base = _base_with(base, start=start, end=end)
    params = [row["params"] for row in rows]
    result = _research_batch(
        training_base,
        data,
        params,
        [(start.date().isoformat(), end.date().isoformat())],
        cost_multiplier=1.0,
        cancelled=cancelled,
    )
    fold_rows = _metric_fold_rows(result, len(rows), 1)
    baseline_dates, baseline_returns = _buy_hold_returns(training_base, data)
    mask = (baseline_dates >= start) & (baseline_dates <= end)
    baseline_return = _compound_daily(baseline_returns[mask])
    for index, row in enumerate(rows):
        evidence = fold_rows[index][0]
        principal = float(evidence.get("principal", 0.0) or 0.0)
        profit = float(evidence.get("profit", 0.0) or 0.0)
        drawdown = float(evidence.get("max_drawdown", math.inf) or math.inf)
        grid_return = profit / principal if principal > 0 else math.nan
        excess = grid_return - baseline_return if math.isfinite(grid_return) else math.nan
        score = excess - drawdown if math.isfinite(excess) else -math.inf
        row.update({
            "training_status": "completed",
            "training_return": grid_return,
            "training_baseline_return": baseline_return,
            "training_excess_return": excess,
            "training_drawdown": drawdown,
            "training_score": score,
            "training_trade_count": int(evidence.get("trade_count", 0)),
        })
    return rows, {
        "status": "completed",
        "evaluations": len(rows),
        "baseline_return": baseline_return,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
    }


def _center_distance(left: dict[str, Any], right: dict[str, Any], config: OptimizationConfig, base: BacktestConfig, names: tuple[str, ...]) -> float:
    squared = 0.0
    for name in names:
        distance = _range_distance(_range_for(config, base, name), name)
        if distance <= 0:
            continue
        squared += ((float(left[name]) - float(right[name])) / distance) ** 2
    return math.sqrt(squared)


def _training_centers(
    config: OptimizationConfig,
    base: BacktestConfig,
    rows: list[dict[str, Any]],
    statistical_centers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = _v3_names(base)
    completed = [row for row in rows if row.get("training_status") == "completed" and math.isfinite(float(row.get("training_score", -math.inf)))]
    selected: list[dict[str, Any]] = []
    by_center: dict[int, list[dict[str, Any]]] = {}
    for row in completed:
        for center_id in row.get("source_center_ids", []):
            by_center.setdefault(int(center_id), []).append(row)
    for center in statistical_centers:
        candidates = sorted(by_center.get(int(center["statistical_center_id"]), []), key=lambda row: (-float(row["training_score"]), row["search_id"]))
        if not candidates:
            continue
        best = candidates[0]
        selected.append({
            "params": deepcopy(best["params"]),
            "training_score": float(best["training_score"]),
            "training_excess_return": float(best.get("training_excess_return", math.nan)),
            "source_center_ids": list(best.get("source_center_ids", [])),
            "selection_kind": "statistical_best",
            "quantile": center.get("quantile"),
        })
    for row in sorted(completed, key=lambda item: (-float(item["training_score"]), item["search_id"])):
        if len(selected) >= _V3_MAX_VALIDATION_CENTERS:
            break
        if any(_center_distance(row["params"], item["params"], config, base, names) < 2.0 for item in selected):
            continue
        selected.append({
            "params": deepcopy(row["params"]),
            "training_score": float(row["training_score"]),
            "training_excess_return": float(row.get("training_excess_return", math.nan)),
            "source_center_ids": list(row.get("source_center_ids", [])),
            "selection_kind": "complementary",
            "quantile": None,
        })
    return selected


def _candidate_family(
    sampled: list[dict[str, Any]],
    config: OptimizationConfig,
    base: BacktestConfig,
    names: tuple[str, ...],
    source_trials: dict[tuple[Any, ...], list[int]] | None = None,
    pairwise: bool = False,
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], list[tuple[Any, ...]]]]:
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    clipped_neighbor_keys: set[tuple[Any, ...]] = set()
    domains = {name: _range_for(config, base, name) for name in names}
    center_keys: set[tuple[Any, ...]] = set()

    def in_domain(params: dict[str, Any]) -> bool:
        return all(domains[name].low <= float(params[name]) <= domains[name].high for name in names)

    def add(
        params: dict[str, Any],
        *,
        is_center: bool = False,
        is_baseline: bool = False,
    ) -> tuple[Any, ...] | None:
        params = _normalize_params(params, base)
        legal, _ = _valid_params(params, base)
        if not legal:
            return None
        key = _params_key(params, names)
        if key not in candidates:
            candidates[key] = {
                "params": params,
                "is_center": bool(is_center),
                "is_baseline": bool(is_baseline),
                "in_search_domain": in_domain(params),
                "source_trials": list((source_trials or {}).get(key, [])),
            }
        else:
            candidates[key]["is_center"] = candidates[key]["is_center"] or bool(is_center)
            candidates[key]["is_baseline"] = candidates[key]["is_baseline"] or bool(is_baseline)
            candidates[key]["in_search_domain"] = in_domain(params)
            candidates[key]["source_trials"] = sorted(set(candidates[key].get("source_trials", [])) | set((source_trials or {}).get(key, [])))
        if is_center:
            center_keys.add(key)
        return key

    baseline = {name: getattr(base, name) for name in names}
    baseline["initial_shares"] = int(base.initial_shares)
    for center in sampled:
        center_key = add(center, is_center=True)
        if center_key is None:
            continue
    baseline_key = add(baseline, is_center=True, is_baseline=True)

    # Freeze the selectable centers first, then add only their legal one-step
    # neighbors to the full validation family.  An out-of-range baseline is a
    # standalone reference and does not widen any configured search domain.
    for key in list(center_keys):
        center = candidates[key]["params"]
        if not candidates[key]["in_search_domain"]:
            continue
        for name in names:
            r = domains[name]
            if r.low == r.high:
                continue
            if name.endswith("_shares"):
                distance = _share_step(r) if r.step is not None else max(100, int(round((r.high - r.low) * 0.05 / 100)) * 100)
            else:
                distance = float(r.step) if r.step is not None else (r.high - r.low) * 0.05
                if distance <= 0:
                    continue
            for direction in (-1, 1):
                neighbor = deepcopy(center)
                requested_value = center[name] + direction * distance
                proposed = max(r.low, min(r.high, requested_value))
                if name.endswith("_shares"):
                    proposed = int(round(float(proposed) / 100) * 100)
                neighbor[name] = proposed
                neighbor_key = add(neighbor)
                if neighbor_key is not None and abs(float(proposed) - float(requested_value)) > 1e-12:
                    clipped_neighbor_keys.add(neighbor_key)
        if pairwise:
            for left_index, left_name in enumerate(names):
                for right_name in names[left_index + 1:]:
                    left_range = domains[left_name]
                    right_range = domains[right_name]
                    left_distance = _range_distance(left_range, left_name)
                    right_distance = _range_distance(right_range, right_name)
                    if left_distance <= 0 or right_distance <= 0:
                        continue
                    for left_direction, right_direction in product((-1, 1), repeat=2):
                        neighbor = deepcopy(center)
                        left_value = max(left_range.low, min(left_range.high, center[left_name] + left_direction * left_distance))
                        right_value = max(right_range.low, min(right_range.high, center[right_name] + right_direction * right_distance))
                        if left_name.endswith("_shares"):
                            left_value = int(round(float(left_value) / 100.0) * 100)
                        if right_name.endswith("_shares"):
                            right_value = int(round(float(right_value) / 100.0) * 100)
                        neighbor[left_name] = left_value
                        neighbor[right_name] = right_value
                        neighbor_key = add(neighbor)
                        if neighbor_key is not None and (abs(float(left_value) - float(center[left_name] + left_direction * left_distance)) > 1e-12 or abs(float(right_value) - float(center[right_name] + right_direction * right_distance)) > 1e-12):
                            clipped_neighbor_keys.add(neighbor_key)

    ordered = sorted(candidates.items(), key=lambda item: item[0])
    rows = [
        {"candidate_id": index + 1, "key": key, **deepcopy(metadata)}
        for index, (key, metadata) in enumerate(ordered)
    ]
    candidate_ids = {row["key"]: row["candidate_id"] for row in rows}
    known = set(candidates)
    links: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    for key in center_keys:
        center = candidates[key]["params"]
        adjacent = {key}
        if not candidates[key]["in_search_domain"]:
            links[key] = [key]
            continue
        for name in names:
            r = domains[name]
            if r.low == r.high:
                continue
            if name.endswith("_shares"):
                distance = _share_step(r) if r.step is not None else max(100, int(round((r.high - r.low) * 0.05 / 100)) * 100)
            else:
                distance = float(r.step) if r.step is not None else (r.high - r.low) * 0.05
                if distance <= 0:
                    continue
            for direction in (-1, 1):
                proposed = max(r.low, min(r.high, center[name] + direction * distance))
                if name.endswith("_shares"):
                    proposed = int(round(float(proposed) / 100) * 100)
                neighbor = deepcopy(center)
                neighbor[name] = proposed
                neighbor_key = _params_key(_normalize_params(neighbor, base), names)
                if neighbor_key in known:
                    adjacent.add(neighbor_key)
        if pairwise:
            for left_index, left_name in enumerate(names):
                for right_name in names[left_index + 1:]:
                    left_range = domains[left_name]
                    right_range = domains[right_name]
                    left_distance = _range_distance(left_range, left_name)
                    right_distance = _range_distance(right_range, right_name)
                    if left_distance <= 0 or right_distance <= 0:
                        continue
                    for left_direction, right_direction in product((-1, 1), repeat=2):
                        neighbor = deepcopy(center)
                        left_value = max(left_range.low, min(left_range.high, center[left_name] + left_direction * left_distance))
                        right_value = max(right_range.low, min(right_range.high, center[right_name] + right_direction * right_distance))
                        if left_name.endswith("_shares"):
                            left_value = int(round(float(left_value) / 100.0) * 100)
                        if right_name.endswith("_shares"):
                            right_value = int(round(float(right_value) / 100.0) * 100)
                        neighbor[left_name] = left_value
                        neighbor[right_name] = right_value
                        neighbor_key = _params_key(_normalize_params(neighbor, base), names)
                        if neighbor_key in known:
                            adjacent.add(neighbor_key)
        links[key] = sorted(adjacent)
    for row in rows:
        key = row["key"]
        if row.get("is_center"):
            variable_names = [name for name in names if domains[name].low != domains[name].high]
            expected_neighbors = 2 * len(variable_names) + (4 * len(variable_names) * (len(variable_names) - 1) // 2 if pairwise else 0)
            row["neighbor_expected_count"] = int(expected_neighbors)
        else:
            row["neighbor_expected_count"] = 0
        row["neighbor_candidate_ids"] = [candidate_ids[item] for item in links.get(key, []) if item in candidate_ids]
        row["neighbor_of_candidate_ids"] = [
            candidate_ids[center_key]
            for center_key, neighbor_keys in links.items()
            if key in neighbor_keys and center_key in candidate_ids and center_key != key
        ]
        row["neighbor_boundary_clipped"] = row["key"] in clipped_neighbor_keys
        row["neighbor_boundary_clipped_count"] = sum(item in clipped_neighbor_keys and item != row["key"] for item in links.get(row["key"], []))
    # Baseline is intentionally included even if no TPE trial sampled it.
    if baseline_key is not None:
        candidates[baseline_key]["is_baseline"] = True
    return rows, links


def _research_batch(base, data, params, fold_ranges, *, cost_multiplier=1.0, cancelled=None):
    from . import share_engine

    return share_engine.research_batch(
        base,
        data,
        params,
        fold_ranges,
        cost_multiplier=cost_multiplier,
        cancelled=cancelled,
    )


def _batch_returns(result: dict[str, Any], count: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    matrix = np.asarray(result["daily_returns"], dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] != count:
        raise ValueError("research_batch.daily_returns must have shape [days, candidates]")
    dates = pd.DatetimeIndex(pd.to_datetime(result["dates"])).normalize()
    if len(dates) != matrix.shape[0]:
        raise ValueError("research_batch dates do not align with daily_returns")
    if not np.isfinite(matrix).all() or np.any(matrix <= -1):
        raise ValueError("research_batch returned invalid daily returns")
    return dates, matrix


def _metric_fold_rows(result: dict[str, Any], candidate_count: int, fold_count: int) -> list[list[dict[str, Any]]]:
    rows = result.get("fold_metrics")
    if rows is None:
        raise ValueError("research_batch must return exact fold_metrics for constraint checks")
    if len(rows) != candidate_count:
        raise ValueError("research_batch.fold_metrics candidate dimension is invalid")
    normalized = []
    for candidate in rows:
        if len(candidate) != fold_count:
            raise ValueError("research_batch.fold_metrics fold dimension is invalid")
        normalized.append([dict(item) for item in candidate])
    return normalized


def _folds_for_history(
    history_dates: pd.DatetimeIndex,
) -> tuple[pd.DatetimeIndex, list[tuple[pd.Timestamp, pd.Timestamp]], bool, list[str]]:
    dates = history_dates[-_LOOKBACK_DAYS:]
    reasons: list[str] = []
    if len(dates) >= _LOOKBACK_DAYS:
        generation = dates[:_FULL_GENERATION_DAYS]
        validation = dates[_FULL_GENERATION_DAYS:]
        fold_arrays = np.array_split(validation, 4)
        folds = [(pd.Timestamp(part[0]), pd.Timestamp(part[-1])) for part in fold_arrays if len(part)]
        return generation, folds, True, reasons
    reasons.append(f"历史不足504个交易日（当前{len(dates)}日），排名可用但证据不足")
    if len(dates) >= _MIN_SPLIT_DAYS:
        midpoint = len(dates) // 2
        generation, validation = dates[:midpoint], dates[midpoint:]
        fold_arrays = np.array_split(validation, 4)
        folds = [(pd.Timestamp(part[0]), pd.Timestamp(part[-1])) for part in fold_arrays if len(part)]
        return generation, folds, True, reasons
    reasons.append(f"历史不足{_MIN_SPLIT_DAYS}日，使用样本内回看且不做显著性结论")
    generation = dates
    return generation, [], False, reasons


def _daily_window_score(values: np.ndarray) -> float:
    if len(values) == 0:
        return math.nan
    growths = 1.0 + np.asarray(values, dtype=np.float64)
    if np.any(growths <= 0) or not np.isfinite(growths).all():
        annualized = -1.0
    else:
        annualized = float(np.expm1(np.clip(np.log(growths).sum() * 252.0 / len(growths), -700, 700)))
    equity = np.r_[1.0, np.cumprod(growths)]
    peaks = np.maximum.accumulate(equity)
    drawdown = float(np.max((peaks - equity) / np.maximum(peaks, 1e-15)))
    return annualized - drawdown


def _window_scores(
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
    folds: list[tuple[pd.Timestamp, pd.Timestamp]],
    fold_drawdowns: np.ndarray | None,
) -> np.ndarray:
    scores = np.full(returns.shape[1], np.nan, dtype=np.float64)
    for candidate in range(returns.shape[1]):
        values = []
        for fold_index, (start, end) in enumerate(folds):
            mask = (dates >= start) & (dates <= end)
            sample = returns[mask, candidate]
            if len(sample) == 0:
                continue
            score = _daily_window_score(sample)
            if fold_drawdowns is not None and fold_index < fold_drawdowns.shape[1] and np.isfinite(fold_drawdowns[candidate, fold_index]):
                growth = 1.0 + sample
                annualized = float(np.expm1(np.clip(np.log(growth).sum() * 252.0 / len(growth), -700, 700))) if np.all(growth > 0) else -1.0
                score = annualized - float(fold_drawdowns[candidate, fold_index])
            values.append(score)
        if not values:
            mask = np.ones(len(dates), dtype=bool)
            values = [_daily_window_score(returns[mask, candidate])]
        score_values = np.asarray(values, dtype=np.float64)
        q25, q75 = np.quantile(score_values, [0.25, 0.75])
        scores[candidate] = float(np.median(score_values) - 0.5 * (q75 - q25))
    return scores


def _aggregate_validation(
    family: list[dict[str, Any]],
    folds: list[tuple[pd.Timestamp, pd.Timestamp]],
    result: dict[str, Any],
    base: BacktestConfig,
    config: OptimizationConfig,
) -> list[dict[str, Any]]:
    dates, returns = _batch_returns(result, len(family))
    raw_metrics = np.asarray(result.get("fold_drawdowns", []), dtype=np.float64)
    if raw_metrics.ndim != 2 or raw_metrics.shape != (len(family), len(folds)):
        raise ValueError("research_batch.fold_drawdowns must have shape [candidates, folds]")
    fold_rows = _metric_fold_rows(result, len(family), len(folds))
    scores = _window_scores(dates, returns, folds, raw_metrics)
    output = []
    for index, candidate in enumerate(family):
        metrics_by_fold = fold_rows[index]
        fold_scores_original = []
        fold_scores_stress = []
        fold_evidence = []
        maximum_drawdown = 0.0
        total_trades = total_buys = total_sells = 0
        turnover_values = []
        for fold_index, (start, end) in enumerate(folds):
            mask = (dates >= start) & (dates <= end)
            sample = returns[mask, index]
            row = metrics_by_fold[fold_index]
            drawdown = float(raw_metrics[index, fold_index])
            maximum_drawdown = max(maximum_drawdown, drawdown)
            trade_count = int(row.get("trade_count", int(row.get("buy_count", 0)) + int(row.get("sell_count", 0))))
            buy_count, sell_count = int(row.get("buy_count", 0)), int(row.get("sell_count", 0))
            total_trades += trade_count
            total_buys += buy_count
            total_sells += sell_count
            turnover = float(row.get("turnover", 0.0) or 0.0)
            turnover_values.append(turnover)
            annualized = float(np.expm1(np.clip(np.log1p(sample).sum() * 252 / len(sample), -700, 700))) if len(sample) and np.all(sample > -1) else -1.0
            fold_u = annualized - drawdown
            fold_scores_original.append(fold_u)
            fold_scores_stress.append(None)
            fold_evidence.append({
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "days": int(mask.sum()),
                "annualized_return": annualized,
                "max_drawdown": drawdown,
                "utility": fold_u,
                "trade_count": trade_count,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "total_fees": float(row.get("total_fees", 0.0) or 0.0),
                "turnover": turnover,
            })

        # Store per-cost utilities and precise fold evidence after the caller
        # supplies the second cost scenario.
        candidate["_fold_original"] = fold_scores_original
        candidate["_fold_rows"] = fold_evidence
        candidate["_max_drawdown"] = maximum_drawdown
        candidate["_trade_count"] = total_trades
        candidate["_buy_count"] = total_buys
        candidate["_sell_count"] = total_sells
        candidate["_turnover"] = float(np.median(turnover_values)) if turnover_values else 0.0
        candidate["_score_from_matrix"] = float(scores[index])
        candidate["_daily_returns"] = returns[:, index].copy()
        candidate["_dates"] = dates
        output.append(candidate)
    return output


def _robust_score(values: list[float]) -> float:
    if not values:
        return math.nan
    array = np.asarray(values, dtype=np.float64)
    q25, q75 = np.quantile(array, [0.25, 0.75])
    return float(np.median(array) - 0.5 * (q75 - q25))


def _fold_score_set(result: dict[str, Any], family: list[dict[str, Any]], folds: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[list[float]]:
    dates, returns = _batch_returns(result, len(family))
    drawdowns = np.asarray(result.get("fold_drawdowns", []), dtype=np.float64)
    if drawdowns.shape != (len(family), len(folds)):
        raise ValueError("research_batch.fold_drawdowns has invalid shape")
    scores: list[list[float]] = [[] for _ in family]
    for index in range(len(family)):
        for fold_index, (start, end) in enumerate(folds):
            mask = (dates >= start) & (dates <= end)
            sample = returns[mask, index]
            if not len(sample):
                raise ValueError("validation fold does not overlap research_batch dates")
            annualized = float(np.expm1(np.clip(np.log1p(sample).sum() * 252.0 / len(sample), -700, 700))) if np.all(sample > -1) else -1.0
            scores[index].append(annualized - float(drawdowns[index, fold_index]))
    return scores


def _evaluate_family(
    family_rows: list[dict[str, Any]],
    config: OptimizationConfig,
    base: BacktestConfig,
    scoring_data: DataBundle,
    scoring_dates: pd.DatetimeIndex,
    folds: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    reserve_cash: float | None,
    progress: Progress | None,
    cancelled: Cancelled | None,
    progress_start: float,
    progress_width: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    params = [row["params"] for row in family_rows]
    sample_start, sample_end = scoring_dates[0], scoring_dates[-1]
    scoring_base = _base_with(base, start=sample_start, end=sample_end, reserve_cash=reserve_cash)
    v3_scoring = int(config.protocol_version) in (3, 4)
    baseline_folds: list[float] = []
    if v3_scoring:
        baseline_dates, baseline_returns = _buy_hold_returns(scoring_base, scoring_data)
        for start, end in folds:
            mask = (baseline_dates >= start) & (baseline_dates <= end)
            baseline_folds.append(_compound_daily(baseline_returns[mask]))
    fold_bounds = [(start.date().isoformat(), end.date().isoformat()) for start, end in folds]
    fold_by_cost: dict[str, list[list[float]]] = {}
    excess_by_cost: dict[str, list[list[float]]] = {}
    evidence_by_cost: dict[str, list[list[dict[str, Any]]]] = {}
    peak_by_cost: dict[str, list[float]] = {}
    original_daily: tuple[pd.DatetimeIndex, np.ndarray] | None = None

    for cost_index, multiplier in enumerate(_COST_MULTIPLIERS):
        if _cancelled(cancelled):
            result = None
            interrupted = True
        else:
            interrupted = False
            try:
                result = _research_batch(
                    scoring_base,
                    scoring_data,
                    params,
                    fold_bounds,
                    cost_multiplier=multiplier,
                    cancelled=cancelled,
                )
            except InterruptedError:
                result = None
                interrupted = True
        if interrupted:
            partial_rows = []
            available_cost_scores = {
                cost_key: [_robust_score(values[index]) for index in range(len(params))]
                for cost_key, values in fold_by_cost.items()
            }
            for index, row in enumerate(family_rows):
                partial_scores = [values[index] for values in available_cost_scores.values() if math.isfinite(values[index])]
                row.update({
                    "raw_score": min(partial_scores) if partial_scores else None,
                    "eligible": False,
                    "constraint_reasons": ["双成本验证未完成，不具备候选资格"],
                    "max_drawdown": max(
                        [float(fold["max_drawdown"]) for values in evidence_by_cost.values() for fold in values[index]]
                        or [math.inf]
                    ),
                    "trade_count": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "turnover": math.inf,
                    "folds": evidence_by_cost.get("original_cost", [[] for _ in params])[index],
                    "stress_folds": evidence_by_cost.get("double_commission_minimum_and_slippage", [[] for _ in params])[index],
                    "cost_scores": available_cost_scores,
                    "evaluation_status": "cancelled_partial_cost",
                })
                partial_rows.append(row)
            return partial_rows, {
                "status": "cancelled",
                "candidate_count": len(params),
                "completed_costs": list(fold_by_cost),
                "incomplete_cost": "double_commission_minimum_and_slippage" if "original_cost" in fold_by_cost else "original_cost",
                "validation_evaluation_count": len(params) * len(fold_by_cost),
            }, True
        if result.get("status") == "cancelled" or result.get("cancelled"):
            partial_rows = []
            available_cost_scores = {
                cost_key: [_robust_score(values[index]) for index in range(len(params))]
                for cost_key, values in fold_by_cost.items()
            }
            for index, row in enumerate(family_rows):
                partial_scores = [values[index] for values in available_cost_scores.values() if math.isfinite(values[index])]
                row.update({
                    "raw_score": min(partial_scores) if partial_scores else None,
                    "eligible": False,
                    "constraint_reasons": ["双成本验证未完成，不具备候选资格"],
                    "max_drawdown": max(
                        [float(fold["max_drawdown"]) for values in evidence_by_cost.values() for fold in values[index]]
                        or [math.inf]
                    ),
                    "trade_count": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "turnover": math.inf,
                    "folds": evidence_by_cost.get("original_cost", [[] for _ in params])[index],
                    "stress_folds": evidence_by_cost.get("double_commission_minimum_and_slippage", [[] for _ in params])[index],
                    "cost_scores": available_cost_scores,
                    "evaluation_status": "cancelled_partial_cost",
                })
                partial_rows.append(row)
            return partial_rows, {
                "status": "cancelled",
                "candidate_count": len(params),
                "completed_costs": list(fold_by_cost),
                "validation_evaluation_count": len(params) * len(fold_by_cost),
            }, True
        dates, matrix = _batch_returns(result, len(params))
        drawdowns = np.asarray(result.get("fold_drawdowns", []), dtype=np.float64)
        if drawdowns.shape != (len(params), len(folds)):
            raise ValueError("research_batch.fold_drawdowns must have shape [candidates, folds]")
        fold_metrics = _metric_fold_rows(result, len(params), len(folds))
        scores: list[list[float]] = [[] for _ in params]
        excess_scores: list[list[float]] = [[] for _ in params]
        per_candidate_evidence: list[list[dict[str, Any]]] = [[] for _ in params]
        for candidate_index in range(len(params)):
            for fold_index, (start, end) in enumerate(folds):
                mask = (dates >= start) & (dates <= end)
                sample = matrix[mask, candidate_index]
                if not len(sample):
                    raise ValueError("validation fold does not overlap research_batch dates")
                annualized = float(np.expm1(np.clip(np.log1p(sample).sum() * 252.0 / len(sample), -700, 700))) if np.all(sample > -1) else -1.0
                drawdown = float(drawdowns[candidate_index, fold_index])
                row = fold_metrics[candidate_index][fold_index]
                if v3_scoring:
                    principal = float(row.get("principal", 0.0) or 0.0)
                    profit = float(row.get("profit", 0.0) or 0.0)
                    grid_return = profit / principal if principal > 0 else -1.0
                    baseline_return = baseline_folds[fold_index] if fold_index < len(baseline_folds) else math.nan
                    excess = grid_return - baseline_return if math.isfinite(baseline_return) else math.nan
                    score = excess - drawdown if math.isfinite(excess) else -1e12
                    excess_scores[candidate_index].append(excess)
                else:
                    grid_return = annualized
                    baseline_return = math.nan
                    excess = math.nan
                    score = annualized - drawdown
                scores[candidate_index].append(score)
                per_candidate_evidence[candidate_index].append({
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "days": int(mask.sum()),
                    "annualized_return": annualized,
                    "grid_return": grid_return,
                    "baseline_return": baseline_return if math.isfinite(baseline_return) else None,
                    "excess_return": excess if math.isfinite(excess) else None,
                    "max_drawdown": drawdown,
                    "utility": score,
                    "trade_count": int(row.get("trade_count", int(row.get("buy_count", 0)) + int(row.get("sell_count", 0)))),
                    "buy_count": int(row.get("buy_count", 0)),
                    "sell_count": int(row.get("sell_count", 0)),
                    "total_fees": float(row.get("total_fees", 0.0) or 0.0),
                    "turnover": float(row.get("turnover", 0.0) or 0.0),
                })
        key = "original_cost" if multiplier == 1.0 else "double_commission_minimum_and_slippage"
        fold_by_cost[key] = scores
        excess_by_cost[key] = excess_scores
        evidence_by_cost[key] = per_candidate_evidence
        stats = np.asarray(result.get("funding_stats", []), dtype=np.float64)
        peak_by_cost[key] = [float(stats[index, 2]) if len(stats) > index and np.isfinite(stats[index, 2]) else math.inf for index in range(len(params))]
        if multiplier == 1.0:
            original_daily = (dates, matrix)
        _emit(progress, progress_start + progress_width * (cost_index + 1) / len(_COST_MULTIPLIERS), f"双成本验证：{cost_index + 1}/{len(_COST_MULTIPLIERS)}")

    candidate_scores_by_key: dict[tuple[Any, ...], list[float]] = {}
    for cost_key, candidate_scores in fold_by_cost.items():
        for index, scores in enumerate(candidate_scores):
            candidate_scores_by_key.setdefault(family_rows[index]["key"], []).append(_robust_score(scores))

    result_rows: list[dict[str, Any]] = []
    max_drawdowns: dict[tuple[Any, ...], float] = {}
    turn_overs: dict[tuple[Any, ...], float] = {}
    constraints: dict[tuple[Any, ...], tuple[bool, list[str]]] = {}
    for index, family_row in enumerate(family_rows):
        original_folds = evidence_by_cost["original_cost"][index]
        stress_folds = evidence_by_cost["double_commission_minimum_and_slippage"][index]
        scores = candidate_scores_by_key[family_row["key"]]
        if v3_scoring:
            excess_scores = [_robust_score(values[index]) for values in excess_by_cost.values()]
            raw_score = min(excess_scores) if excess_scores else math.nan
        else:
            raw_score = min(scores) if scores else math.nan
        drawdown = max(
            [float(row["max_drawdown"]) for row in original_folds + stress_folds] or [math.inf]
        )
        trade_count = min(
            sum(int(row["trade_count"]) for row in original_folds),
            sum(int(row["trade_count"]) for row in stress_folds),
        )
        buy_count = min(sum(int(row["buy_count"]) for row in original_folds), sum(int(row["buy_count"]) for row in stress_folds))
        sell_count = min(sum(int(row["sell_count"]) for row in original_folds), sum(int(row["sell_count"]) for row in stress_folds))
        turnover = float(np.median([row["turnover"] for row in original_folds + stress_folds])) if original_folds else math.inf
        reasons = []
        if drawdown > float(config.max_drawdown):
            reasons.append("验证折最大回撤超过上限")
        if trade_count < int(config.min_trades):
            reasons.append("验证折成交次数不足")
        if buy_count <= 0 or sell_count <= 0:
            reasons.append("验证折未同时出现买入和卖出")
        eligible = not reasons
        original_excess = [float(row["excess_return"]) for row in original_folds if row.get("excess_return") is not None]
        stress_excess = [float(row["excess_return"]) for row in stress_folds if row.get("excess_return") is not None]
        family_row.update({
            "raw_score": float(raw_score),
            "eligible": eligible,
            "constraint_reasons": reasons,
            "max_drawdown": float(drawdown),
            "trade_count": int(trade_count),
            "buy_count": int(buy_count),
            "sell_count": int(sell_count),
            "turnover": turnover,
            "folds": original_folds,
            "stress_folds": stress_folds,
            "cost_scores": {key: _robust_score(values[index]) for key, values in fold_by_cost.items()},
            "validation_excess_original": _robust_score(original_excess),
            "validation_excess_stress": _robust_score(stress_excess),
            "validation_return_original": _robust_score([float(row["grid_return"]) for row in original_folds if row.get("grid_return") is not None]),
            "validation_return_stress": _robust_score([float(row["grid_return"]) for row in stress_folds if row.get("grid_return") is not None]),
            "peak_invested": min([values[index] for values in peak_by_cost.values() if index < len(values)] or [math.inf]),
            "evaluation_status": "completed",
        })
        if original_daily is not None:
            family_row["_current_validation_dates"] = original_daily[0]
            family_row["_current_validation_returns"] = original_daily[1][:, index]
        max_drawdowns[family_row["key"]] = drawdown
        turn_overs[family_row["key"]] = turnover
        constraints[family_row["key"]] = (eligible, reasons)
        result_rows.append(family_row)
    diagnostic = {
        "status": "completed",
        "costs": {key: {"fold_utility": values} for key, values in fold_by_cost.items()},
        "fold_count": len(folds),
        "candidate_count": len(params),
        "validation_evaluation_count": len(params) * len(_COST_MULTIPLIERS),
        "days": int(len(scoring_dates)),
        "date_start": scoring_dates[0].date().isoformat(),
        "date_end": scoring_dates[-1].date().isoformat(),
    }
    return result_rows, diagnostic, False


def _rank_family(
    rows: list[dict[str, Any]],
    neighborhoods: dict[tuple[Any, ...], list[tuple[Any, ...]]],
    protocol_version: int = 1,
    full_history: bool = True,
) -> list[dict[str, Any]]:
    by_key = {row["key"]: row for row in rows}
    if int(protocol_version) == 3:
        for row in rows:
            row["constraint_eligible"] = bool(row.get("eligible", False))
            if not row.get("is_center"):
                row["neighbor_scores"] = []
                row["neighbor_count"] = 0
                row["selection_score"] = None
                row["score"] = None
                row["recommendation_eligible"] = False
                row["eligible"] = False
                continue
            expected = list(dict.fromkeys(neighborhoods.get(row["key"], [row["key"]])))
            neighbor_keys = [key for key in expected if key != row["key"]]
            neighbor_rows = [by_key[key] for key in neighbor_keys if key in by_key]
            values_original = [float(item["validation_excess_original"]) for item in neighbor_rows if item.get("evaluation_status") == "completed" and item.get("validation_excess_original") is not None and math.isfinite(float(item["validation_excess_original"]))]
            values_stress = [float(item["validation_excess_stress"]) for item in neighbor_rows if item.get("evaluation_status") == "completed" and item.get("validation_excess_stress") is not None and math.isfinite(float(item["validation_excess_stress"]))]
            complete = len(neighbor_rows) == len(neighbor_keys) and len(values_original) == len(neighbor_keys) and len(values_stress) == len(neighbor_keys)
            positive_ratio = float(np.mean(np.asarray(values_original) > 0.0)) if values_original else 0.0
            stress_positive_ratio = float(np.mean(np.asarray(values_stress) > 0.0)) if values_stress else 0.0
            q25_original = float(np.quantile(values_original, 0.25)) if values_original else math.nan
            q25_stress = float(np.quantile(values_stress, 0.25)) if values_stress else math.nan
            center_excess_original = row.get("validation_excess_original")
            center_excess_stress = row.get("validation_excess_stress")
            center_positive = center_excess_original is not None and center_excess_stress is not None and float(center_excess_original) > 0 and float(center_excess_stress) > 0
            constraints_ok = bool(row.get("constraint_eligible", row.get("eligible", False)))
            has_neighbors = bool(neighbor_keys)
            robust = bool(
                complete and has_neighbors and full_history and constraints_ok and center_positive and not row.get("is_baseline")
                and positive_ratio >= _V3_POSITIVE_NEIGHBOR_RATIO and stress_positive_ratio >= _V3_POSITIVE_NEIGHBOR_RATIO
                and math.isfinite(q25_original) and q25_original > 0 and math.isfinite(q25_stress) and q25_stress > 0
            )
            row.update({
                "neighbor_count": len(neighbor_keys),
                "neighbor_scores": values_original,
                "neighbor_scores_stress": values_stress,
                "neighbor_positive_ratio": positive_ratio,
                "neighbor_positive_ratio_stress": stress_positive_ratio,
                "neighbor_q25_excess": q25_original,
                "neighbor_q25_excess_stress": q25_stress,
                "neighbor_median_excess": float(np.median(values_original)) if values_original else math.nan,
                "neighbor_worst_excess": float(np.min(values_original)) if values_original else math.nan,
                "negative_neighbor_count": int(np.sum(np.asarray(values_original) <= 0.0)) if values_original else 0,
                "neighbor_evaluation_complete": complete,
                "robustness_status": "passed" if robust else "failed" if complete and has_neighbors else "unavailable",
                "robustness_reasons": [] if robust else [
                    reason for reason, failed in (
                        ("完整历史不足", not full_history),
                        ("邻域未完整计算", not complete),
                        ("没有真实邻点", not has_neighbors),
                        ("中心参数未满足硬约束", not constraints_ok),
                        ("中心超额收益未同时为正", not center_positive),
                        ("正超额邻点占比不足 80%", positive_ratio < _V3_POSITIVE_NEIGHBOR_RATIO or stress_positive_ratio < _V3_POSITIVE_NEIGHBOR_RATIO),
                        ("邻域第 25 百分位未同时为正", not (math.isfinite(q25_original) and q25_original > 0 and math.isfinite(q25_stress) and q25_stress > 0)),
                    ) if failed
                ],
                "island_suspect": bool(center_positive and complete and 0.0 < positive_ratio < 0.5),
                "recommendation_eligible": robust,
                "selection_score": min(q25_original, q25_stress) if math.isfinite(q25_original) and math.isfinite(q25_stress) else math.nan,
                "score": row.get("raw_score"),
                "eligible": robust,
            })
        return sorted(
            [row for row in rows if row.get("is_center") and row.get("evaluation_status", "completed") == "completed"],
            key=lambda row: (
                -int(bool(row.get("recommendation_eligible"))),
                -float(row["selection_score"]) if math.isfinite(float(row["selection_score"])) else math.inf,
                -float(row["raw_score"]) if row.get("raw_score") is not None and math.isfinite(float(row["raw_score"])) else math.inf,
                float(row["max_drawdown"]),
                float(row.get("peak_invested", math.inf)),
                int(row["candidate_id"]),
            ),
        )
    for row in rows:
        if not row.get("is_center"):
            row["neighbor_scores"] = []
            row["neighbor_count"] = 0
            row["selection_score"] = None
            row["score"] = None
            continue
        neighborhood_keys = neighborhoods.get(row["key"], [row["key"]])
        values = []
        for key in neighborhood_keys:
            if key not in by_key:
                continue
            score = by_key[key].get("raw_score")
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and by_key[key].get("evaluation_status", "completed") == "completed":
                values.append(value)
        row["neighbor_scores"] = values
        row["neighbor_count"] = len(values)
        row["selection_score"] = float(np.quantile(values, 0.25)) if values else math.nan
        row["score"] = row["selection_score"]
    return sorted(
        [row for row in rows if row.get("is_center") and row.get("evaluation_status", "completed") == "completed"],
        key=lambda row: (
            -int(bool(row["eligible"])),
            -float(row["selection_score"]) if math.isfinite(float(row["selection_score"])) else math.inf,
            float(row["max_drawdown"]),
            float(row["turnover"]),
            int(row["candidate_id"]),
        ),
    )


def _public_candidate(row: dict[str, Any]) -> dict[str, Any]:
    def finite_or_none(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    return {
        "candidate_id": int(row["candidate_id"]),
        "params": deepcopy(row["params"]),
        "score": finite_or_none(row.get("score")),
        "raw_score": finite_or_none(row.get("raw_score")),
        "eligible": bool(row.get("eligible", False)),
        "constraint_reasons": list(row.get("constraint_reasons", [])),
        "max_drawdown": finite_or_none(row.get("max_drawdown")),
        "trade_count": int(row.get("trade_count", 0)),
        "buy_count": int(row.get("buy_count", 0)),
        "sell_count": int(row.get("sell_count", 0)),
        "turnover": finite_or_none(row.get("turnover")),
        "folds": deepcopy(row.get("folds", [])),
        "stress_folds": deepcopy(row.get("stress_folds", [])),
        "cost_scores": deepcopy(row.get("cost_scores", {})),
        "validation_excess_original": finite_or_none(row.get("validation_excess_original")),
        "validation_excess_stress": finite_or_none(row.get("validation_excess_stress")),
        "peak_invested": finite_or_none(row.get("peak_invested")),
        "training_score": finite_or_none(row.get("training_score")),
        "training_excess_return": finite_or_none(row.get("training_excess_return")),
        "statistical_center_id": list(row.get("statistical_center_id", [])),
        "center_selection_kind": row.get("center_selection_kind"),
        "constraint_eligible": bool(row.get("constraint_eligible", row.get("eligible", False))),
        "recommendation_eligible": bool(row.get("recommendation_eligible", False)),
        "robustness_status": row.get("robustness_status", "not_available"),
        "robustness_reasons": list(row.get("robustness_reasons", [])),
        "neighbor_positive_ratio": finite_or_none(row.get("neighbor_positive_ratio")),
        "neighbor_positive_ratio_stress": finite_or_none(row.get("neighbor_positive_ratio_stress")),
        "neighbor_q25_excess": finite_or_none(row.get("neighbor_q25_excess")),
        "neighbor_q25_excess_stress": finite_or_none(row.get("neighbor_q25_excess_stress")),
        "neighbor_median_excess": finite_or_none(row.get("neighbor_median_excess")),
        "neighbor_worst_excess": finite_or_none(row.get("neighbor_worst_excess")),
        "negative_neighbor_count": int(row.get("negative_neighbor_count", 0)),
        "neighbor_evaluation_complete": bool(row.get("neighbor_evaluation_complete", False)),
        "island_suspect": bool(row.get("island_suspect", False)),
        "neighbor_boundary_clipped": bool(row.get("neighbor_boundary_clipped", False)),
        "neighbor_boundary_clipped_count": int(row.get("neighbor_boundary_clipped_count", 0)),
        "neighbor_count": int(row.get("neighbor_count", 0)),
        "neighbor_expected_count": int(row.get("neighbor_expected_count", 0)),
        "neighbor_scores": list(row.get("neighbor_scores", [])),
        "is_center": bool(row.get("is_center", False)),
        "is_baseline": bool(row.get("is_baseline", False)),
        "in_search_domain": bool(row.get("in_search_domain", False)),
        "neighbor_candidate_ids": list(row.get("neighbor_candidate_ids", [])),
        "neighbor_of_candidate_ids": list(row.get("neighbor_of_candidate_ids", [])),
        "source_trials": list(row.get("source_trials", [])),
        "evaluation_status": row.get("evaluation_status", "not_evaluated"),
    }


def _select_at_v3(
    config: OptimizationConfig,
    base: BacktestConfig,
    data: DataBundle,
    asof: pd.Timestamp | None,
    *,
    seed_offset: int,
    progress: Progress | None,
    cancelled: Cancelled | None,
    progress_start: float = 0.0,
    progress_width: float = 0.0,
) -> dict[str, Any]:
    dates = _calendar(data)
    history_dates = dates[dates <= asof] if asof is not None else pd.DatetimeIndex([])
    history_dates = history_dates[-_LOOKBACK_DAYS:]
    as_of = history_dates[-1] if len(history_dates) else (_date_key(asof) if asof is not None else None)
    names = _v3_names(base)
    reasons: list[str] = []
    generation_dates, folds, has_oos_split, window_reasons = _folds_for_history(history_dates)
    reasons.extend(window_reasons)
    fallback = {name: getattr(base, name) for name in names}
    fallback["initial_shares"] = int(base.initial_shares)
    if base.max_shares is None:
        fallback["max_shares"] = None
    if len(history_dates) == 0:
        return {
            "status": "insufficient_history", "recommendation_status": "no_qualified",
            "params": fallback, "recommended_params": None, "as_of": as_of.date().isoformat() if as_of is not None else None,
            "score": None, "raw_score": None, "eligible": False,
            "reasons": list(dict.fromkeys([*reasons, "没有可用于统计画像的历史行情"])),
            "candidates": [], "candidate_archive": [], "search_candidate_archive": [], "trial_archive": [],
            "diagnostics": {"status": "insufficient_history", "candidate_count": 0, "eligible_count": 0, "validation_folds": []},
            "_ranked_rows": [], "_family_rows": [], "_neighborhoods": {}, "_search_rows": [],
            "_fold_ranges": [], "_selection_data": None, "_selection_dates": history_dates,
        }
    if len(generation_dates) == 0:
        generation_dates = history_dates
    gen_start, gen_end = pd.Timestamp(generation_dates[0]), pd.Timestamp(generation_dates[-1])
    gen_data = legacy_optimization._slice_bundle(data, gen_start, gen_end)
    _emit(progress, progress_start, "统计画像：波动、趋势、修复与交易成本")
    statistical_centers, profile = _statistical_centers(config, base, data, generation_dates)
    local_rows, expected_count, budget_exceeded = _local_grid_rows(config, base, statistical_centers)
    if budget_exceeded:
        diagnostic = {
            "status": "budget_exceeded", "local_search_expected": expected_count,
            "local_search_budget": int(config.local_search_budget), "statistical_profile": profile,
        }
        return {
            "status": "budget_exceeded", "recommendation_status": "no_qualified",
            "params": fallback, "recommended_params": None, "as_of": as_of.date().isoformat() if as_of is not None else None,
            "score": None, "raw_score": None, "eligible": False,
            "reasons": [*reasons, f"局部网格预计 {expected_count} 组，超过预算 {config.local_search_budget}；请调整范围或预算"],
            "candidates": [], "candidate_archive": [], "search_candidate_archive": [], "trial_archive": [],
            "diagnostics": diagnostic,
            "_ranked_rows": [], "_family_rows": [], "_neighborhoods": {}, "_search_rows": [],
            "_fold_ranges": folds, "_selection_data": gen_data, "_selection_dates": history_dates,
        }
    if _cancelled(cancelled):
        raise InterruptedError("研究已取消")
    training_rows, training = _score_training_rows(config, base, gen_data, generation_dates, local_rows, cancelled)
    selected_centers = _training_centers(config, base, training_rows, statistical_centers)
    if not selected_centers:
        selected_centers = [{
            "params": deepcopy(fallback), "training_score": None,
            "source_center_ids": [], "selection_kind": "baseline_only", "quantile": None,
        }]
        reasons.append("统计中心没有合法局部组合，仅保留原始配置作为研究对照")
    family_rows, neighborhoods = _candidate_family(
        [item["params"] for item in selected_centers], config, base, names, pairwise=True
    )
    center_meta = {_params_key(_normalize_params(item["params"], base), names): item for item in selected_centers}
    for row in family_rows:
        meta = center_meta.get(row["key"])
        row["statistical_center_id"] = meta.get("source_center_ids") if meta else []
        row["center_selection_kind"] = meta.get("selection_kind") if meta else None
        row["training_score"] = meta.get("training_score") if meta else None
        row["training_excess_return"] = meta.get("training_excess_return") if meta else None
    scoring_start, scoring_end = history_dates[0], history_dates[-1]
    scoring_data = legacy_optimization._slice_bundle(data, scoring_start, scoring_end)
    folds_for_score = folds or [(generation_dates[0], generation_dates[-1])]
    rows, validation_diagnostic, validation_cancelled = _evaluate_family(
        family_rows, config, base, scoring_data, history_dates, folds_for_score,
        reserve_cash=None,
        progress=progress,
        cancelled=cancelled,
        progress_start=progress_start + progress_width * 0.55,
        progress_width=progress_width * 0.45,
    )
    full_history = len(history_dates) >= _LOOKBACK_DAYS
    ranked = _rank_family(rows, neighborhoods, protocol_version=3, full_history=full_history)
    if validation_cancelled:
        reasons.append("双成本验证期间用户取消")
    recommended = next((row for row in ranked if row.get("recommendation_eligible")), None)
    relative_best = ranked[0] if ranked else None
    if recommended is None:
        reasons.append("暂无合格推荐：没有参数同时满足时间约束、80% 正超额邻点、邻域第 25 百分位为正及双成本条件")
        recommended_params = None
    else:
        recommended_params = deepcopy(recommended["params"])
    search_archive = []
    for row in training_rows:
        search_archive.append({
            "search_id": int(row["search_id"]),
            "params": deepcopy(row["params"]),
            "source_center_ids": list(row.get("source_center_ids", [])),
            "status": row.get("training_status"),
            "training_return": row.get("training_return"),
            "training_baseline_return": row.get("training_baseline_return"),
            "training_excess_return": row.get("training_excess_return"),
            "training_drawdown": row.get("training_drawdown"),
            "training_score": row.get("training_score"),
        })
    diagnostic = {
        **validation_diagnostic,
        "status": "completed" if not validation_cancelled else "cancelled",
        "protocol_version": 3,
        "statistical_profile": profile,
        "statistical_center_count": len(statistical_centers),
        "validation_center_count": len(selected_centers),
        "local_search_expected": expected_count,
        "local_search_unique": len(local_rows),
        "local_search_budget": int(config.local_search_budget),
        "training": training,
        "candidate_count": len(family_rows),
        "neighbor_count": sum(max(0, len(value) - 1) for value in neighborhoods.values()),
        "selectable_center_count": sum(bool(row.get("is_center")) for row in family_rows),
        "eligible_count": sum(bool(row.get("recommendation_eligible")) for row in ranked),
        "recommendation_status": "qualified" if recommended_params is not None else "no_qualified",
        "robustness_enabled": True,
        "positive_neighbor_ratio": _V3_POSITIVE_NEIGHBOR_RATIO,
        "neighbor_quantile": 0.25,
        "full_history": full_history,
        "validation_folds": [
            {"start": start.date().isoformat(), "end": end.date().isoformat(), "days": int(((history_dates >= start) & (history_dates <= end)).sum())}
            for start, end in folds
        ],
        "constraints": {"max_drawdown": float(config.max_drawdown), "min_trades": int(config.min_trades)},
    }
    best_for_display = recommended or relative_best
    return {
        "status": "cancelled" if validation_cancelled else "completed",
        "recommendation_status": "qualified" if recommended_params is not None else "no_qualified",
        "params": deepcopy(relative_best["params"]) if relative_best else fallback,
        "recommended_params": recommended_params,
        "as_of": as_of.date().isoformat() if as_of is not None else None,
        "score": float(best_for_display["selection_score"]) if best_for_display and best_for_display.get("selection_score") is not None and math.isfinite(float(best_for_display["selection_score"])) else None,
        "raw_score": float(best_for_display["raw_score"]) if best_for_display and best_for_display.get("raw_score") is not None else None,
        "eligible": bool(recommended is not None),
        "reasons": list(dict.fromkeys(reasons)),
        "candidates": [_public_candidate(row) for row in ranked],
        "candidate_archive": [_public_candidate(row) for row in family_rows],
        "search_candidate_archive": search_archive,
        "trial_archive": search_archive,
        "diagnostics": diagnostic,
        "_ranked_rows": ranked,
        "_family_rows": family_rows,
        "_neighborhoods": neighborhoods,
        "_fold_ranges": folds,
        "_selection_data": scoring_data,
        "_selection_dates": history_dates,
        "_search_rows": training_rows,
    }


def _select_at(
    config: OptimizationConfig,
    base: BacktestConfig,
    data: DataBundle,
    asof: pd.Timestamp | None,
    *,
    reserve_cash: float | None,
    seed_offset: int,
    progress: Progress | None,
    cancelled: Cancelled | None,
    progress_start: float = 0.0,
    progress_width: float = 0.0,
) -> dict[str, Any]:
    if int(config.protocol_version) == 4:
        from .research_v4 import select_at as select_at_v4

        return select_at_v4(
            config,
            base,
            data,
            asof,
            seed_offset=seed_offset,
            progress=progress,
            cancelled=cancelled,
            progress_start=progress_start,
            progress_width=progress_width,
        )
    if int(config.protocol_version) == 3:
        return _select_at_v3(
            config, base, data, asof, seed_offset=seed_offset, progress=progress,
            cancelled=cancelled, progress_start=progress_start, progress_width=progress_width,
        )
    dates = _calendar(data)
    history_dates = dates[dates <= asof] if asof is not None else pd.DatetimeIndex([])
    history_dates = history_dates[-_LOOKBACK_DAYS:]
    n_history = len(history_dates)
    as_of = history_dates[-1] if len(history_dates) else (_date_key(asof) if asof is not None else None)
    names = _active_names(base)
    reasons: list[str] = []
    generation_dates, folds, has_oos_split, window_reasons = _folds_for_history(history_dates)
    reasons.extend(window_reasons)
    if asof is None:
        reasons.append("估值日前缺少历史收盘数据，使用固定输入附近参数作相对排名")
    elif n_history == 0:
        reasons.append("仅有参考估值日期，没有可供搜索与验证的历史行情；保留原始参数，评分为空")

    if n_history == 0:
        # Even with no observations, return a deterministic relative choice
        # based only on fixed inputs and configured range bounds.
        fallback = {name: getattr(base, name) for name in names}
        fallback["initial_shares"] = int(base.initial_shares)
        sampled = [fallback]
        trial_rows = []
        tpe_cancelled = False
        gen_range = None
        gen_data = None
    else:
        if len(generation_dates) == 0:
            generation_dates = history_dates
        gen_range = (generation_dates[0], generation_dates[-1])
        gen_data = legacy_optimization._slice_bundle(data, *gen_range)
        sampled, trial_rows, tpe_cancelled = _trial_params(
            config,
            base,
            names,
            seed=(int(config.seed) + int(seed_offset)) % (2**32),
            generation_data=gen_data,
            generation_dates=generation_dates,
            reserve_cash=reserve_cash,
            progress=progress,
            cancelled=cancelled,
            progress_start=progress_start,
            progress_width=progress_width * 0.45,
        )
        if tpe_cancelled:
            reasons.append("候选生成期间用户取消")

    source_trials: dict[tuple[Any, ...], list[int]] = {}
    for trial in trial_rows:
        if trial.get("status") == "completed" and isinstance(trial.get("params"), dict):
            key = _params_key(_normalize_params(trial["params"], base), names)
            source_trials.setdefault(key, []).append(int(trial.get("trial", 0)))
    family_rows, neighborhoods = _candidate_family(sampled, config, base, names)
    for row in family_rows:
        row["source_trials"] = sorted(set(row.get("source_trials", [])) | set(source_trials.get(row["key"], [])))
    if not family_rows:
        fallback = {name: getattr(base, name) for name in names}
        fallback["initial_shares"] = int(base.initial_shares)
        family_rows, neighborhoods = _candidate_family([fallback], config, base, names)
        for row in family_rows:
            row["source_trials"] = sorted(set(row.get("source_trials", [])) | set(source_trials.get(row["key"], [])))
        reasons.append("配置范围内没有合法候选，保留基础参数作相对排名")

    # When a genuine validation split exists, run the frozen candidate family
    # continuously from the start of the full lookback and score only its
    # four chronological validation folds.  The generation and validation
    # segments are never used together to suggest or expand candidates.
    validation_cancelled = False
    if n_history and not tpe_cancelled:
        scoring_start, scoring_end = history_dates[0], history_dates[-1]
        scoring_data = legacy_optimization._slice_bundle(data, scoring_start, scoring_end)
        if not folds:
            folds_for_score = [(generation_dates[0], generation_dates[-1])]
        else:
            folds_for_score = folds
        rows, diagnostic, validation_cancelled = _evaluate_family(
            family_rows,
            config,
            base,
            scoring_data,
            history_dates,
            folds_for_score,
            reserve_cash=reserve_cash,
            progress=progress,
            cancelled=cancelled,
            progress_start=progress_start + progress_width * 0.45,
            progress_width=progress_width * 0.45,
        )
        if validation_cancelled:
            reasons.append("双成本验证期间用户取消")
    elif n_history and tpe_cancelled:
        rows = []
        diagnostic = {"status": "cancelled", "candidate_count": len(family_rows), "fold_count": len(folds)}
        for row in family_rows:
            row.update({
                "raw_score": None,
                "eligible": False,
                "constraint_reasons": ["候选生成取消，未进入双成本验证"],
                "max_drawdown": math.inf,
                "trade_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "turnover": math.inf,
                "folds": [],
                "stress_folds": [],
                "cost_scores": {},
                "evaluation_status": "not_evaluated_cancelled",
            })
        rows = family_rows
    else:
        rows = []
        diagnostic = {"status": "insufficient_history", "candidate_count": len(family_rows), "fold_count": 0}
        for row in family_rows:
            row.update({
                "raw_score": 0.0,
                "eligible": False,
                "constraint_reasons": ["无可用历史收益供评分"],
                "max_drawdown": math.inf,
                "trade_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "turnover": math.inf,
                "folds": [],
                "stress_folds": [],
                "cost_scores": {},
                "evaluation_status": "insufficient_history_relative",
            })
        rows = family_rows

    if rows:
        ranked = _rank_family(rows, neighborhoods)
    else:
        ranked = []
    best = ranked[0] if ranked else None
    if best:
        selected = deepcopy(best["params"])
        status = "completed"
        if not best["eligible"]:
            reasons.append("没有通过回撤与交易约束的候选，返回相对得分最高项")
    else:
        selected = {name: getattr(base, name) for name in names}
        selected["initial_shares"] = int(base.initial_shares)
        status = "cancelled" if (_cancelled(cancelled) or tpe_cancelled or validation_cancelled) else "insufficient_history"
    if len(generation_dates) and n_history:
        generation_start = generation_dates[0].date().isoformat()
        generation_end = generation_dates[-1].date().isoformat()
    else:
        generation_start = generation_end = None
    return {
        "status": status,
        "params": selected,
        "as_of": as_of.date().isoformat() if as_of is not None else None,
        "score": float(best["selection_score"]) if best else None,
        "raw_score": float(best["raw_score"]) if best else None,
        "eligible": bool(best["eligible"]) if best else False,
        "reasons": list(dict.fromkeys(reasons)),
        "candidates": [_public_candidate(row) for row in ranked],
        "candidate_archive": [_public_candidate(row) for row in family_rows],
        "trial_archive": [
            {
                "trial": int(row.get("trial", 0)),
                "status": row.get("status"),
                "params": deepcopy(row.get("params", {})),
                "generation_score": row.get("generation_score"),
                "reason": row.get("reason"),
                "reasons": deepcopy(row.get("reasons", [])),
            }
            for row in trial_rows
        ],
        "diagnostics": {
            **diagnostic,
            "generated_trial_count": len(trial_rows),
            "unique_generated_count": len(sampled),
            "candidate_count": len(family_rows),
            "selectable_center_count": sum(bool(row.get("is_center")) for row in family_rows),
            "neighbor_count": sum(len(value) for value in neighborhoods.values()),
            "eligible_count": sum(bool(row.get("eligible")) for row in ranked),
        "generation_evaluation_count": len(sampled) if n_history else 0,
        "validation_evaluation_count": int(diagnostic.get("validation_evaluation_count", 0)),
        "selected_constraint_reasons": list(best.get("constraint_reasons", [])) if best else (["候选选择未完成"] if status == "cancelled" else ["无已验证合格候选"]),
            "generation_start": generation_start,
            "generation_end": generation_end,
            "generation_days": int(len(generation_dates)),
            "selection_history_days": int(n_history),
            "selection_history_start": history_dates[0].date().isoformat() if n_history else None,
            "selection_history_end": history_dates[-1].date().isoformat() if n_history else None,
            "validation_folds": [
                {"start": start.date().isoformat(), "end": end.date().isoformat(), "days": int(((history_dates >= start) & (history_dates <= end)).sum())}
                for start, end in folds
            ],
            "trial_status_counts": {
                state: sum(row.get("status") == state for row in trial_rows)
                for state in ("completed", "invalid_constraints", "failed", "cancelled")
            },
            "constraints": {"max_drawdown": float(config.max_drawdown), "min_trades": int(config.min_trades)},
        },
        "_ranked_rows": ranked,
        "_family_rows": family_rows,
        "_neighborhoods": neighborhoods,
        "_fold_ranges": folds,
        "_selection_data": legacy_optimization._slice_bundle(data, history_dates[0], history_dates[-1]) if n_history else None,
        "_selection_dates": history_dates,
    }


def _funding_reserve(base: BacktestConfig, data: DataBundle, evaluation_start: pd.Timestamp) -> dict[str, Any]:
    try:
        from .share_engine import funding

        locked_base = _base_with(base, start=evaluation_start)
        result = funding(locked_base, data)
        return {"status": "locked", **deepcopy(result)}
    except (KeyError, TypeError):
        # Let the engine surface hard funding/data failures when the actual
        # backtest starts.  This fallback exists only for old model fixtures.
        configured = getattr(base, "reserve_cash", None)
        return {
            "status": "configured" if configured is not None else "unavailable",
            "reserve_cash": None if configured is None else float(configured),
            "reason": "funding details unavailable for legacy fixture",
        }


def _funding_amount(lock: dict[str, Any] | None) -> float | None:
    if not isinstance(lock, dict) or lock.get("reserve_cash") is None:
        return None
    return float(lock["reserve_cash"])


def _run_share_backtest(base, data, progress=None, cancelled=None, parameter_schedule=None):
    from .share_engine import run_share_backtest

    return run_share_backtest(
        base,
        data,
        progress=progress,
        cancelled=cancelled,
        parameter_schedule=parameter_schedule,
        daily_details=True,
    )


def _run_passive_backtest(base, data, progress=None, cancelled=None):
    from .share_engine import run_passive_backtest

    return run_passive_backtest(base, data, progress=progress, cancelled=cancelled, daily_details=True)


def _attach_passive_strategy(result: dict[str, Any] | None, passive_result: dict[str, Any] | None) -> None:
    if not isinstance(result, dict) or not isinstance(passive_result, dict):
        return
    passive_strategy = passive_result.get("strategies", {}).get("grid")
    if not isinstance(passive_strategy, dict):
        return
    result.setdefault("strategies", {})["same_account_passive"] = deepcopy(passive_strategy)


def _returns_from_backtest(result: dict[str, Any], strategy: str = "grid") -> tuple[pd.DatetimeIndex, np.ndarray]:
    strategies = result.get("strategies", {}) if isinstance(result, dict) else {}
    section = strategies.get(strategy, {}) if isinstance(strategies, dict) else {}
    rows = section.get("period_returns", []) if isinstance(section, dict) else []
    if not rows:
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)
    dates = pd.DatetimeIndex(pd.to_datetime([row.get("timestamp") for row in rows])).normalize()
    returns = np.asarray([float(row.get("return", 0.0)) for row in rows], dtype=np.float64)
    return dates, returns


def _returns_from_strategy_result(section: dict[str, Any]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    rows = section.get("period_returns", []) if isinstance(section, dict) else []
    if not rows:
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)
    dates = pd.DatetimeIndex(pd.to_datetime([row.get("timestamp") for row in rows])).normalize()
    returns = np.asarray([float(row.get("return", 0.0)) for row in rows], dtype=np.float64)
    return dates, returns


def _passive_returns(base: BacktestConfig, data: DataBundle, buy_hold: bool = False) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if buy_hold:
        return _buy_hold_returns(base, data)
    from .share_engine import research_passive

    result = research_passive(base, data)
    dates = pd.DatetimeIndex(pd.to_datetime(result["dates"])).normalize()
    returns = np.asarray(result["daily_returns"], dtype=np.float64).reshape(-1)
    if len(dates) != len(returns):
        raise ValueError("research_passive dates do not align with daily_returns")
    return dates, returns


def evaluate_current_family(
    config: OptimizationConfig,
    data: DataBundle,
    current: dict[str, Any],
    archive: list[dict[str, Any]],
    reserve_cash: float | None,
    cancelled: Cancelled | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Bootstrap the frozen current candidate family on its internal folds.

    This is an internal validation diagnostic, not independent OOS evidence,
    and its result never changes the selected current parameters.
    """
    diagnostics = current.get("diagnostics", {}) if isinstance(current, dict) else {}
    if diagnostics.get("adaptive_validation_search"):
        return {
            "status": "not_applicable", "sample_role": "internal_validation_used_for_selection",
            "independent_oos": False, "family_count": len(archive or []),
            "reasons": ["连通域按内部验证收益自适应扩张，未预先冻结完整候选族；不报告固定候选族显著性"],
        }
    history_days = int(diagnostics.get("selection_history_days", 0) or 0)
    folds_raw = diagnostics.get("validation_folds", [])
    result: dict[str, Any] = {
        "status": "unavailable",
        "sample_role": "internal_validation_used_for_selection",
        "independent_oos": False,
        "family_count": 0,
        "reasons": [],
    }
    if history_days < _MIN_SPLIT_DAYS or not folds_raw:
        result.update(
            status="insufficient_history",
            reasons=["当前参数族没有至少168日且与生成区间分离的验证样本，不报告显著性"],
        )
        return result
    if _cancelled(cancelled):
        result.update(status="cancelled", reasons=["当前参数族bootstrap前用户取消"])
        return result
    asof_raw = current.get("as_of")
    if not asof_raw:
        result.update(status="unavailable", reasons=["当前选择缺少as_of日期"])
        return result
    asof = _date_key(asof_raw)
    all_dates = _calendar(data)
    history_dates = all_dates[all_dates <= asof][- _LOOKBACK_DAYS:]
    if len(history_dates) < _MIN_SPLIT_DAYS:
        result.update(status="insufficient_history", reasons=["可用行情不足168个交易日"])
        return result
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for fold in folds_raw:
        try:
            start, end = _date_key(fold["start"]), _date_key(fold["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= asof and start <= end:
            folds.append((start, end))
    if not folds:
        result.update(status="unavailable", reasons=["当前参数族没有有效的历史验证折"])
        return result
    candidates = [
        row for row in (archive or [])
        if isinstance(row, dict)
        and isinstance(row.get("params"), dict)
        and row.get("evaluation_status", "completed") == "completed"
    ]
    if not candidates:
        result.update(status="unavailable", reasons=["当前参数族没有完整验证候选"])
        return result
    names = _v3_names(config.base) if int(config.protocol_version) in (3, 4) else _active_names(config.base)
    selection_base = _base_with(
        config.base,
        start=history_dates[0],
        end=history_dates[-1],
        reserve_cash=reserve_cash,
    )
    selection_data = legacy_optimization._slice_bundle(data, history_dates[0], history_dates[-1])
    internal_rows = current.get("_family_rows", []) if isinstance(current, dict) else []
    internal_by_id = {row.get("candidate_id"): row for row in internal_rows if isinstance(row, dict)}
    internal_dates = None
    internal_values = []
    can_reuse = len(internal_by_id) > 0
    for candidate in candidates:
        internal = internal_by_id.get(candidate.get("candidate_id"))
        candidate_dates = internal.get("_current_validation_dates") if internal else None
        candidate_returns = internal.get("_current_validation_returns") if internal else None
        if candidate_dates is None or candidate_returns is None:
            can_reuse = False
            break
        dates = pd.DatetimeIndex(candidate_dates).normalize()
        values = np.asarray(candidate_returns, dtype=np.float64).reshape(-1)
        if len(values) != len(dates) or (internal_dates is not None and not dates.equals(internal_dates)):
            can_reuse = False
            break
        internal_dates = dates
        internal_values.append(values)
    if can_reuse and internal_dates is not None:
        dates = internal_dates
        matrix = np.column_stack(internal_values)
        result["matrix_source"] = "reused_original_cost_validation_matrix"
    else:
        try:
            batch = _research_batch(
                selection_base,
                selection_data,
                [row["params"] for row in candidates],
                [(start.date().isoformat(), end.date().isoformat()) for start, end in folds],
                cost_multiplier=1.0,
                cancelled=cancelled,
            )
        except InterruptedError:
            result.update(status="cancelled", family_count=len(candidates), reasons=["当前参数族原成本验证期间用户取消"])
            return result
        if batch.get("status") == "cancelled" or batch.get("cancelled"):
            result.update(status="cancelled", family_count=len(candidates), reasons=["当前参数族原成本验证期间用户取消"])
            return result
        dates, matrix = _batch_returns(batch, len(candidates))
        result["matrix_source"] = "recomputed_original_cost_validation_matrix"
    if int(config.protocol_version) in (3, 4):
        passive_dates, passive_values = _passive_returns(selection_base, selection_data, buy_hold=True)
    else:
        passive_dates, passive_values = _passive_returns(selection_base, selection_data)
    passive_series = pd.Series(passive_values, index=passive_dates).groupby(level=0).last()
    active_frame = pd.DataFrame(matrix, index=dates, columns=np.arange(matrix.shape[1])).groupby(level=0).last()
    validation_mask = np.zeros(len(active_frame.index), dtype=bool)
    for start, end in folds:
        validation_mask |= (active_frame.index >= start) & (active_frame.index <= end)
    validation_dates = active_frame.index[validation_mask]
    common_dates = validation_dates.intersection(passive_series.index).sort_values()
    if len(common_dates) == 0:
        result.update(status="unavailable", family_count=len(candidates), reasons=["当前验证折与被动持有收益没有共同日期"])
        return result
    candidate_matrix = active_frame.loc[common_dates].to_numpy(dtype=np.float64)
    passive_aligned = passive_series.loc[common_dates].to_numpy(dtype=np.float64)
    selected_params = current.get("params", {})
    selected_key = _params_key(_normalize_params(selected_params, config.base), names) if selected_params else None
    selected_index = next((i for i, row in enumerate(candidates) if _params_key(_normalize_params(row["params"], config.base), names) == selected_key), 0)
    bootstrap = paired_stationary_bootstrap(
        candidate_matrix,
        passive_aligned,
        selected_index=selected_index,
        reps=int(getattr(config, "bootstrap_reps", 10_000)),
        block_lengths=(20, 10, 40),
        seed=(int(config.seed) + _seed_offset_for_date(asof)) % (2**32),
        cancelled=cancelled,
        progress=lambda fraction: _emit(progress, fraction, "当前参数族内部验证bootstrap"),
    )
    result.update({
        "status": "cancelled" if bootstrap.get("cancelled") else "completed",
        "as_of": asof.date().isoformat(),
        "training_start": history_dates[0].date().isoformat(),
        "training_end": history_dates[-1].date().isoformat(),
        "validation_folds": [{"start": start.date().isoformat(), "end": end.date().isoformat()} for start, end in folds],
        "paired_start": common_dates[0].date().isoformat(),
        "paired_end": common_dates[-1].date().isoformat(),
        "paired_days": int(len(common_dates)),
        "candidate_count": len(candidates),
        "family_count": len(candidates),
        "candidate_ids": [int(row.get("candidate_id", index + 1)) for index, row in enumerate(candidates)],
        "selected_candidate_id": int(candidates[selected_index].get("candidate_id", selected_index + 1)),
        "bootstrap": bootstrap,
        "reasons": [],
    })
    return result


def _align_pair(
    left_dates: pd.DatetimeIndex,
    left_returns: np.ndarray,
    right_dates: pd.DatetimeIndex,
    right_returns: np.ndarray,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    left = pd.Series(left_returns, index=left_dates).groupby(level=0).last()
    right = pd.Series(right_returns, index=right_dates).groupby(level=0).last()
    common = left.index.intersection(right.index).sort_values()
    return pd.DatetimeIndex(common), left.loc[common].to_numpy(), right.loc[common].to_numpy()


def _evaluate_final_126(
    config: OptimizationConfig,
    base: BacktestConfig,
    data: DataBundle,
    eval_dates: pd.DatetimeIndex,
    fixed: dict[str, Any],
    adaptive: dict[str, Any],
    passive_result: dict[str, Any] | None = None,
    *,
    progress: Progress | None,
    cancelled: Cancelled | None,
) -> dict[str, Any]:
    tail = eval_dates[-126:]
    short = len(tail) < 126
    output: dict[str, Any] = {
        "status": "insufficient_history" if len(tail) == 0 else "completed",
        "start": tail[0].date().isoformat() if len(tail) else None,
        "end": tail[-1].date().isoformat() if len(tail) else None,
        "days": int(len(tail)),
        "required_days": 126,
        "short_term_insufficient": bool(short),
        "source": "actual_continuous_fixed_and_adaptive_oos_runs",
        "procedures": [],
        "reasons": ["外层核验不足126个交易日，区间较短"] if short and len(tail) else ([] if len(tail) else ["评估期没有交易日"]),
    }
    if len(tail) == 0:
        return output
    if _cancelled(cancelled):
        output.update(status="cancelled", reasons=[*output["reasons"], "外层配对bootstrap期间用户取消"])
        return output
    requested: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    if isinstance(fixed, dict) and fixed.get("status") not in ("not_requested", "skipped"):
        requested.append(("fixed", fixed, fixed.get("test_result")))
    if isinstance(adaptive, dict) and adaptive.get("status") not in ("not_requested", "skipped"):
        requested.append(("adaptive", adaptive, adaptive.get("test_result")))
    if not requested:
        output.update(status="unavailable", reasons=[*output["reasons"], "没有已声明的历史OOS择参流程"])
        return output
    if isinstance(passive_result, dict) and passive_result.get("status") == "cancelled":
        output.update(status="cancelled", procedures={}, reasons=[*output["reasons"], "同现金被动持有回测期间用户取消"])
        return output
    passive_section = None
    if isinstance(passive_result, dict):
        passive_section = passive_result.get("strategies", {}).get("buy_hold" if int(config.protocol_version) in (3, 4) else "grid")
    if not isinstance(passive_section, dict):
        for _, section, test_result in requested:
            if isinstance(test_result, dict):
                baseline_key = "buy_hold" if int(config.protocol_version) in (3, 4) else "same_account_passive"
                passive_section = test_result.get("strategies", {}).get(baseline_key)
                if isinstance(passive_section, dict):
                    break
    if isinstance(passive_section, dict):
        passive_dates, passive = _returns_from_strategy_result(passive_section)
    else:
        try:
            if int(config.protocol_version) in (3, 4):
                passive_dates, passive = _passive_returns(base, data, buy_hold=True)
            else:
                passive_dates, passive = _passive_returns(base, data)
        except InterruptedError:
            output.update(status="cancelled", procedures={}, reasons=[*output["reasons"], "同现金被动收益期间用户取消"])
            return output
    passive_series = pd.Series(passive, index=passive_dates).reindex(tail)
    procedure_series: dict[str, pd.Series] = {}
    procedure_details: dict[str, Any] = {}
    for name, section, test_result in requested:
        if test_result is None:
            procedure_details[name] = {
                "status": section.get("status", "unavailable"),
                "params": section.get("params") if name == "fixed" else None,
                "reason": "连续评估回测未完成，无法从实际轨迹核验",
            }
            continue
        dates, values = _returns_from_backtest(test_result)
        series = pd.Series(values, index=dates).groupby(level=0).last().reindex(tail)
        procedure_series[name] = series
        procedure_details[name] = {
            "status": section.get("status", "completed"),
            "params": deepcopy(section.get("params")) if name == "fixed" else None,
            "schedule": deepcopy(section.get("schedule")) if name == "adaptive" else None,
        }
    names = [name for name, _, _ in requested if name in procedure_series]
    if not names:
        output.update(status="unavailable", procedures=procedure_details, reasons=[*output["reasons"], "没有完成的连续OOS轨迹"])
        return output
    all_series = [passive_series, *(procedure_series[name] for name in names)]
    common_mask = np.ones(len(tail), dtype=bool)
    for series in all_series:
        common_mask &= series.notna().to_numpy()
    aligned_dates = tail[common_mask]
    passive_values = passive_series.loc[aligned_dates].to_numpy(dtype=np.float64)
    matrix = np.column_stack([procedure_series[name].loc[aligned_dates].to_numpy(dtype=np.float64) for name in names])
    if len(aligned_dates) == 0:
        output.update(status="unavailable", procedures=procedure_details, reasons=[*output["reasons"], "固定/动态连续轨迹与被动持有没有共同估值日期"])
        return output
    if len(aligned_dates) != len(tail):
        output["reasons"].append(f"配对行情覆盖{len(aligned_dates)}/{len(tail)}日")
    log_excess = np.log1p(matrix) - np.log1p(passive_values[:, None])
    selected_index = int(np.argmax(log_excess.mean(axis=0)))
    bootstrap = paired_stationary_bootstrap(
        matrix,
        passive_values,
        selected_index=selected_index,
        reps=int(getattr(config, "bootstrap_reps", 10_000)),
        block_lengths=(20, 10, 40),
        seed=(int(config.seed) + _seed_offset_for_date(tail[-1])) % (2**32),
        cancelled=cancelled,
        progress=lambda fraction: _emit(progress, 0.94 + 0.05 * fraction, "末126日配对bootstrap"),
    )
    for index, name in enumerate(names):
        process_returns = matrix[:, index]
        process_total = float(np.prod(1.0 + process_returns) - 1.0)
        passive_total = float(np.prod(1.0 + passive_values) - 1.0)
        procedure_details[name].update({
            "paired_days": int(len(aligned_dates)),
            "total_return": process_total,
            "passive_total_return": passive_total,
            "total_excess_return": float((1.0 + process_total) / (1.0 + passive_total) - 1.0) if passive_total > -1 else None,
            "mean_log_excess": float(log_excess[:, index].mean()),
        })
    output.update({
        "status": "cancelled" if bootstrap.get("cancelled") else "completed",
        "paired_days": int(len(aligned_dates)),
        "paired_start": aligned_dates[0].date().isoformat() if len(aligned_dates) else None,
        "paired_end": aligned_dates[-1].date().isoformat() if len(aligned_dates) else None,
        "procedures": procedure_details,
        "bootstrap_family": names,
        "bootstrap_selected_procedure": names[selected_index],
        "bootstrap": bootstrap,
        "interpretation": "末126日使用固定/自适应流程真实连续OOS轨迹与同现金被动持有配对；current参数未参与该核验。",
    })
    _emit(progress, 0.99, "末126日配对核验完成")
    return output


def optimize_scientific(
    config: OptimizationConfig,
    data: DataBundle,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> dict[str, Any]:
    """Run research selection using only information available at each close."""
    if config.base.schema_version not in (2, 3):
        raise ValueError("scientific protocol requires a v2 or v3 share config")
    if config.protocol_version in (3, 4) and config.base.schema_version != 3:
        raise ValueError("protocol_version=3/4 requires schema_version=3")
    if config.protocol_version == 2 and config.base.schema_version != 2:
        raise ValueError("protocol_version=2 requires schema_version=2")
    if not isinstance(data, DataBundle):
        raise TypeError("data must be a DataBundle")
    legacy_optimization._check_ranges(config)
    available_dates = _calendar(data)
    eval_start = _date_key(config.base.start_date)
    eval_dates = available_dates[available_dates >= eval_start]
    if config.base.end_date is not None:
        eval_dates = eval_dates[eval_dates <= _date_key(config.base.end_date)]
    selection_target = getattr(config, "selection_target", "both")
    rebalance_days = max(1, int(getattr(config, "rebalance_days", 63)))
    bootstrap_reps = int(getattr(config, "bootstrap_reps", 10_000))
    quality = deepcopy(data.quality if isinstance(data.quality, dict) else {})
    if not len(eval_dates):
        # No OOS observations means no evaluation result.  We may still show a
        # parameter choice based on closes strictly before the requested start,
        # but never move the evaluation window forward to later snapshot data.
        asof = _known_preceding_close(data, eval_start)
        try:
            lock = _funding_reserve(config.base, data, eval_start)
        except (ValueError, KeyError, TypeError) as exc:
            lock = {"status": "unavailable", "reserve_cash": getattr(config.base, "reserve_cash", None), "reason": f"{type(exc).__name__}: {exc}"}
        reserve_cash = _funding_amount(lock)
        fallback_base = _base_with(config.base, start=eval_start, end=config.base.end_date, reserve_cash=reserve_cash)
        if selection_target in ("fixed", "both"):
            selection = _select_at(
                config, fallback_base, data, asof,
                reserve_cash=reserve_cash,
                seed_offset=_seed_offset_for_date(asof or eval_start),
                progress=progress, cancelled=cancelled,
                progress_start=0.0, progress_width=0.75,
            )
        else:
            selection = {
                "status": "insufficient_history", "params": {name: getattr(fallback_base, name) for name in _active_names(fallback_base)} | {"initial_shares": int(fallback_base.initial_shares)},
                "as_of": asof.date().isoformat() if asof is not None else None, "score": None, "raw_score": None,
                "eligible": False, "reasons": ["评估区间没有交易日，返回固定输入参数"],
                "candidates": [], "candidate_archive": [], "trial_archive": [],
                "diagnostics": {"status": "insufficient_history", "candidate_count": 0, "eligible_count": 0, "validation_folds": []},
            }
        no_oos = "评估区间没有可回测交易日；没有使用结束日期之后的行情代替样本外数据"
        all_reasons = list(dict.fromkeys([*selection.get("reasons", []), no_oos]))
        protocol_v3 = int(config.protocol_version) in (3, 4)
        recommended = deepcopy(selection.get("recommended_params")) if protocol_v3 else deepcopy(selection["params"])
        selected = deepcopy(recommended if protocol_v3 else selection["params"])
        fixed = {
            "status": "insufficient_history" if selected is not None else "no_qualified", "params": selected if selection_target in ("fixed", "both") else None,
            "diagnostic_params": deepcopy(selection["params"]),
            "as_of": selection.get("as_of"), "score": selection.get("score"), "eligible": bool(selection.get("eligible", False)),
            "reasons": all_reasons, "test_result": None,
            "candidates": selection.get("candidates", [])[: int(config.top_candidates)],
            "diagnostics": selection.get("diagnostics", {}),
            "current": {
                "params": selected, "as_of": selection.get("as_of"), "score": selection.get("score"),
                "status": "insufficient_history", "eligible": bool(selection.get("eligible", False)),
                "reasons": all_reasons, "candidates": selection.get("candidates", [])[: int(config.top_candidates)],
                "diagnostics": selection.get("diagnostics", {}),
            },
        }
        if selection_target == "adaptive":
            fixed["status"] = "not_requested"
            fixed["params"] = None
        adaptive = {"status": "insufficient_history" if selection_target in ("adaptive", "both") else "not_requested", "schedule": [], "test_result": None, "latest_params": None}
        if selection_target in ("adaptive", "both"):
            adaptive.update({"latest_params": selected, "latest_as_of": selection.get("as_of"), "active_params": None, "active_effective_date": None})
        return {
            "status": "cancelled" if selection.get("status") == "cancelled" or _cancelled(cancelled) else "partial",
            "protocol_version": int(config.protocol_version),
            "config": config.model_dump(mode="json"),
            "protocol": {
                "selection_target": selection_target, "lookback_trading_days": _LOOKBACK_DAYS,
                "generation_days_full": _FULL_GENERATION_DAYS, "validation_fold_days": _FOLD_DAYS,
                "validation_folds": 4, "rebalance_days": rebalance_days,
                "cost_multipliers": list(_COST_MULTIPLIERS), "bootstrap_reps": bootstrap_reps,
                "bootstrap_block_lengths": [20, 10, 40], "reserve_cash": reserve_cash,
                "funding_mode": "top_up_on_fill" if protocol_v3 else "configured_reserve",
                "positive_neighbor_ratio": _V3_POSITIVE_NEIGHBOR_RATIO if protocol_v3 else None,
                "neighbor_quantile": 0.25 if protocol_v3 else None,
                "local_search_budget": None if int(config.protocol_version) == 4 else int(config.local_search_budget) if protocol_v3 else None,
                "training_start": selection.get("diagnostics", {}).get("selection_history_start"),
                "training_end": selection.get("diagnostics", {}).get("selection_history_end"),
                "evaluation_start": eval_start.date().isoformat(),
                "evaluation_end": _date_key(config.base.end_date).date().isoformat() if config.base.end_date else None,
            },
            "fixed": fixed, "adaptive": adaptive,
            "locked_funding": lock,
            "evidence": {
                "fixed_decision": {"decision_date": selection.get("as_of"), "status": fixed["status"], "selected_params": fixed.get("params"), "diagnostic_params": fixed.get("diagnostic_params"), "reasons": all_reasons},
                "current_decision": {"decision_date": selection.get("as_of"), "status": "insufficient_history", "selected_params": selected, "reasons": all_reasons},
                "decisions": [], "candidate_archives": {"fixed": selection.get("candidate_archive", []), "current": selection.get("candidate_archive", []), "adaptive": []},
                "trial_archives": {"fixed": selection.get("trial_archive", []), "current": selection.get("trial_archive", []), "adaptive": []},
                "final_126_day_check": {"status": "insufficient_history", "start": None, "end": None, "days": 0, "required_days": 126, "short_term_insufficient": True, "procedures": {}, "reasons": [no_oos]},
                "data_days": int(len(available_dates)), "evaluation_days": 0,
                "evaluation_start": eval_start.date().isoformat(),
                "evaluation_end": _date_key(config.base.end_date).date().isoformat() if config.base.end_date else None,
                "minute_coverage": legacy_optimization._minute_ratio(data), "selection_target": selection_target,
            },
            "warnings": [*data.warnings, *all_reasons], "data_version": data.version, "quality": quality,
            "is_demo": bool(quality.get("is_demo") or data.metadata.get("is_demo")),
            "computation": {"n_trials_per_decision": int(config.n_trials) if not protocol_v3 else 0, "local_search_budget": None if int(config.protocol_version) == 4 else int(config.local_search_budget) if protocol_v3 else None, "rebalance_days": rebalance_days, "decision_count": 0,
                "candidate_count": int(selection.get("diagnostics", {}).get("candidate_count", 0)), "cost_evaluations": 0,
                "bootstrap_reps": bootstrap_reps, "cancelled": bool(_cancelled(cancelled))},
        }

    # Compute the cash reserve from the actual evaluation-start funding date
    # once; every candidate and every rolling window carries this same amount.
    locked_funding = _funding_reserve(config.base, data, eval_dates[0])
    reserve_cash = _funding_amount(locked_funding)
    base = _base_with(config.base, start=eval_dates[0], end=eval_dates[-1], reserve_cash=reserve_cash)
    protocol_v3 = int(config.protocol_version) in (3, 4)
    reasons = []
    if len(eval_dates) < 126:
        reasons.append(f"评估期仅{len(eval_dates)}个交易日，末126日核验不足")
    minute_ratio = legacy_optimization._minute_ratio(data)
    if minute_ratio < float(config.min_minute_ratio):
        reasons.append(f"分钟覆盖率{minute_ratio:.3f}低于要求{config.min_minute_ratio:.3f}；仍提供相对排名")

    _emit(progress, 0.0, "准备滚动科学择参")
    fixed: dict[str, Any] = {"status": "skipped", "params": None, "as_of": None, "test_result": None}
    fixed_selection: dict[str, Any] | None = None
    fixed_replay_cancelled = False
    candidate_archives: dict[str, Any] = {"fixed": [], "current": [], "adaptive": []}
    trial_archives: dict[str, Any] = {"fixed": [], "current": [], "adaptive": []}
    search_candidate_archives: dict[str, Any] = {"fixed": [], "current": [], "adaptive": []}
    if selection_target in ("fixed", "both"):
        initial_asof = _known_preceding_close(data, eval_dates[0])
        selected = _select_at(
            config,
            base,
            data,
            initial_asof,
            reserve_cash=reserve_cash,
            seed_offset=_seed_offset_for_date(initial_asof or eval_dates[0]),
            progress=progress,
            cancelled=cancelled,
            progress_start=0.02,
            progress_width=0.23,
        )
        fixed_selection = selected
        candidate_archives["fixed"] = selected["candidate_archive"]
        trial_archives["fixed"] = selected["trial_archive"]
        search_candidate_archives["fixed"] = selected.get("search_candidate_archive", [])
        protocol_v3 = int(config.protocol_version) in (3, 4)
        recommended = deepcopy(selected.get("recommended_params")) if protocol_v3 else deepcopy(selected["params"])
        if protocol_v3 and recommended is None:
            fixed_config = base.model_copy(update={
                "min_shares": int(base.initial_shares),
                "max_shares": int(base.initial_shares),
            })
        else:
            fixed_config = _base_with(base, recommended or selected["params"], start=eval_dates[0], end=eval_dates[-1], reserve_cash=reserve_cash)
        test_result = None
        if not _cancelled(cancelled):
            try:
                test_result = _run_share_backtest(fixed_config, data, _mapped_progress(progress, 0.25, 0.03), cancelled)
            except InterruptedError:
                fixed_replay_cancelled = True
        fixed = {
            "status": ("no_qualified" if protocol_v3 and recommended is None else selected["status"]) if test_result is not None else ("cancelled" if selected["status"] == "cancelled" or fixed_replay_cancelled or _cancelled(cancelled) else "partial"),
            "params": recommended,
            "recommended_params": recommended,
            "diagnostic_params": selected["params"],
            "as_of": selected["as_of"],
            "score": selected["score"],
            "eligible": selected["eligible"],
            "reasons": selected["reasons"],
            "test_result": test_result,
            "candidates": selected["candidates"][: int(config.top_candidates)],
            "diagnostics": selected["diagnostics"],
            "recommendation_regions": deepcopy(selected.get("recommendation_regions", [])),
            "recommendation_status": selected.get("recommendation_status", "qualified" if recommended is not None else "no_qualified"),
        }
    elif selection_target == "adaptive":
        fixed["status"] = "not_requested"

    adaptive: dict[str, Any] = {"status": "skipped", "schedule": [], "test_result": None, "latest_params": None}
    decisions: list[dict[str, Any]] = []
    if selection_target in ("adaptive", "both"):
        schedule: list[dict[str, Any]] = []
        step_indices = list(range(0, len(eval_dates), rebalance_days))
        for step_number, effective_index in enumerate(step_indices):
            if _cancelled(cancelled):
                break
            effective_date = eval_dates[effective_index]
            if effective_index == 0:
                asof = _known_preceding_close(data, effective_date)
            else:
                asof = eval_dates[effective_index - 1]
            reused_from_fixed = bool(
                step_number == 0
                and fixed_selection is not None
                and selection_target == "both"
                and fixed_selection.get("as_of") == (asof.date().isoformat() if asof is not None else None)
            )
            selected = deepcopy(fixed_selection) if reused_from_fixed else _select_at(
                config,
                base,
                data,
                asof,
                reserve_cash=reserve_cash,
                seed_offset=_seed_offset_for_date(asof or effective_date),
                progress=progress,
                cancelled=cancelled,
                progress_start=0.28 + 0.5 * step_number / max(1, len(step_indices)),
                progress_width=0.5 / max(1, len(step_indices)),
            )
            candidate_archives["adaptive"].append({
                "effective_date": effective_date.date().isoformat(),
                "decision_date": selected["as_of"],
                "candidate_archive": selected["candidate_archive"],
                "search_candidate_archive": selected.get("search_candidate_archive", []),
            })
            trial_archives["adaptive"].append({
                "effective_date": effective_date.date().isoformat(),
                "decision_date": selected["as_of"],
                "trial_archive": selected["trial_archive"],
            })
            if selected["status"] != "cancelled":
                protocol_v3 = int(config.protocol_version) in (3, 4)
                recommended = deepcopy(selected.get("recommended_params")) if protocol_v3 else deepcopy(selected["params"])
                paused = bool(protocol_v3 and recommended is None)
                schedule.append({
                    "effective_date": effective_date.date().isoformat(),
                    "decision_date": selected["as_of"],
                    "params": deepcopy(selected["params"] if paused else recommended),
                    "paused": paused,
                    "recommendation_status": selected.get("recommendation_status", "qualified" if not paused else "no_qualified"),
                })
            decisions.append({
                "effective_date": effective_date.date().isoformat(),
                "decision_date": selected["as_of"],
                "status": selected["status"],
                "reused_from_fixed": reused_from_fixed,
                "selected_params": None if int(config.protocol_version) in (3, 4) and selected.get("recommended_params") is None else deepcopy(selected.get("recommended_params") if int(config.protocol_version) in (3, 4) else selected["params"]),
                "diagnostic_params": deepcopy(selected["params"]),
                "paused": bool(schedule[-1].get("paused")) if schedule else False,
                "score": selected["score"],
                "reasons": selected["reasons"],
                "candidate_count": selected["diagnostics"].get("candidate_count", 0),
                "evaluated_count": selected["diagnostics"].get("candidate_count", 0),
                "eligible_count": selected["diagnostics"].get("eligible_count", 0),
                "selected_eligible": bool(selected["eligible"]),
                "constraint_reasons": selected["diagnostics"].get("selected_constraint_reasons", selected["reasons"]),
                "neighbor_count": selected["diagnostics"].get("neighbor_count", 0),
                "windows": selected["diagnostics"].get("validation_folds", []),
                "diagnostics": selected["diagnostics"],
                "recommendation_regions": deepcopy(selected.get("recommendation_regions", [])),
                "candidates": selected["candidates"][: int(config.top_candidates)],
            })
            _emit(progress, 0.28 + 0.5 * (step_number + 1) / max(1, len(step_indices)), f"滚动择参：{step_number + 1}/{len(step_indices)}")
            if selected["status"] == "cancelled":
                break

        adaptive_result = None
        adaptive_replay_cancelled = False
        if schedule and not _cancelled(cancelled):
            try:
                adaptive_result = _run_share_backtest(base, data, _mapped_progress(progress, 0.78, 0.03), cancelled, parameter_schedule=schedule)
            except InterruptedError:
                adaptive_replay_cancelled = True
        adaptive = {
            "status": "completed" if adaptive_result is not None else ("cancelled" if _cancelled(cancelled) or adaptive_replay_cancelled or len(schedule) < len(step_indices) else "partial"),
            "schedule": schedule,
            "test_result": adaptive_result,
            "latest_params": deepcopy(schedule[-1]["params"]) if schedule else None,
        }
    elif selection_target == "fixed":
        adaptive["status"] = "not_requested"

    passive_result = None
    passive_run_cancelled = False
    if not _cancelled(cancelled) and (fixed.get("test_result") is not None or adaptive.get("test_result") is not None):
        if protocol_v3:
            try:
                baseline_dates, baseline_values = _buy_hold_returns(base, data)
                passive_result = {"strategies": {"buy_hold": {"period_returns": [
                    {"timestamp": (day + pd.Timedelta(hours=15)).isoformat(), "return": float(value)}
                    for day, value in zip(baseline_dates, baseline_values)
                ]}}}
            except InterruptedError:
                passive_run_cancelled = True
        else:
            try:
                passive_result = _run_passive_backtest(base, data, _mapped_progress(progress, 0.81, 0.02), cancelled)
            except InterruptedError:
                passive_run_cancelled = True
            _attach_passive_strategy(fixed.get("test_result"), passive_result)
            _attach_passive_strategy(adaptive.get("test_result"), passive_result)

    # Current parameters are selected as of the final available evaluation
    # close.  This research is separate from the historical fixed OOS result.
    last_asof = eval_dates[-1]
    current = _select_at(
        config,
        base,
        data,
        last_asof,
        reserve_cash=reserve_cash,
        seed_offset=_seed_offset_for_date(last_asof),
        progress=progress,
        cancelled=cancelled,
        progress_start=0.83,
        progress_width=0.05,
    )
    candidate_archives["current"] = current["candidate_archive"]
    trial_archives["current"] = current["trial_archive"]
    search_candidate_archives["current"] = current.get("search_candidate_archive", [])
    protocol_v3 = int(config.protocol_version) in (3, 4)
    current_recommended = deepcopy(current.get("recommended_params")) if protocol_v3 else deepcopy(current["params"])
    fixed["current"] = {
        "params": current_recommended,
        "recommended_params": current_recommended,
        "diagnostic_params": deepcopy(current["params"]),
        "as_of": current["as_of"],
        "score": current["score"],
        "status": "no_qualified" if protocol_v3 and current_recommended is None else current["status"],
        "eligible": current["eligible"],
        "reasons": current["reasons"],
        "candidates": current["candidates"][: int(config.top_candidates)],
        "diagnostics": current["diagnostics"],
        "recommendation_regions": deepcopy(current.get("recommendation_regions", [])),
    }
    if selection_target in ("adaptive", "both"):
        adaptive.update({
            "latest_params": current_recommended,
            "latest_diagnostic_params": deepcopy(current["params"]),
            "latest_as_of": current["as_of"],
            "active_params": deepcopy(adaptive["schedule"][-1]["params"]) if adaptive.get("schedule") and not adaptive["schedule"][-1].get("paused") else None,
            "active_effective_date": adaptive["schedule"][-1]["effective_date"] if adaptive.get("schedule") and not adaptive["schedule"][-1].get("paused") else None,
        })
    current_family_bootstrap = evaluate_current_family(
        config,
        data,
        current,
        candidate_archives["current"],
        reserve_cash,
        cancelled=cancelled,
        progress=_mapped_progress(progress, 0.88, 0.06),
    )
    fixed["current"]["diagnostics"]["candidate_family_bootstrap"] = current_family_bootstrap

    final_126 = _evaluate_final_126(config, base, data, eval_dates, fixed, adaptive,
                                    {"status": "cancelled"} if passive_run_cancelled else passive_result,
                                    progress=progress, cancelled=cancelled)
    if final_126.get("short_term_insufficient"):
        reasons.append("末126日外层检验区间不足，显著性证据应谨慎解读")
    if current["reasons"]:
        reasons.extend(current["reasons"])
    if fixed.get("reasons"):
        reasons.extend(fixed["reasons"])
    warnings = list(dict.fromkeys([*data.warnings, *[str(value) for value in quality.get("warnings", []) if value], *reasons]))
    fixed_decision = {
        "effective_date": eval_dates[0].date().isoformat(),
        "decision_date": fixed.get("as_of"),
        "status": fixed.get("status"),
        "selected_params": fixed.get("params"),
        "diagnostic_params": fixed.get("diagnostic_params"),
        "recommendation_status": fixed.get("recommendation_status"),
        "score": fixed.get("score"),
        "candidate_count": fixed.get("diagnostics", {}).get("candidate_count", 0),
        "eligible_count": fixed.get("diagnostics", {}).get("eligible_count", 0),
        "selected_eligible": fixed.get("eligible", False),
        "constraint_reasons": fixed.get("reasons", []),
        "windows": fixed.get("diagnostics", {}).get("validation_folds", []),
        "recommendation_regions": fixed.get("recommendation_regions", []),
        "reasons": fixed.get("reasons", []),
    }
    current_decision = {
        "effective_date": None,
        "decision_date": current.get("as_of"),
        "status": current.get("status"),
        "selected_params": current_recommended,
        "diagnostic_params": current.get("params"),
        "recommendation_status": current.get("recommendation_status", "qualified" if current_recommended is not None else "no_qualified"),
        "score": current.get("score"),
        "candidate_count": current.get("diagnostics", {}).get("candidate_count", 0),
        "eligible_count": current.get("diagnostics", {}).get("eligible_count", 0),
        "selected_eligible": current.get("eligible", False),
        "constraint_reasons": current.get("reasons", []),
        "windows": current.get("diagnostics", {}).get("validation_folds", []),
        "recommendation_regions": current.get("recommendation_regions", []),
        "reasons": current.get("reasons", []),
    }
    all_selections = [item for item in [fixed.get("diagnostics", {}), current.get("diagnostics", {}), *[
        item.get("diagnostics", {}) for item in decisions if not item.get("reused_from_fixed")
    ]] if item]
    unique_archives = [candidate_archives["fixed"], candidate_archives["current"], *[
        archive.get("candidate_archive", [])
        for decision, archive in zip(decisions, candidate_archives["adaptive"])
        if not decision.get("reused_from_fixed")
    ]]
    total_candidates = sum(int(item.get("candidate_count", 0) or 0) for item in all_selections)
    total_generated = sum(int(item.get("unique_generated_count", 0) or 0) for item in all_selections)
    total_evaluations = sum(int(item.get("generation_evaluation_count", 0) or 0) + int(item.get("validation_evaluation_count", 0) or 0) for item in all_selections)
    if fixed.get("diagnostics", {}).get("selection_history_days"):
        training_source = fixed["diagnostics"]
    elif decisions:
        training_source = decisions[0].get("diagnostics", {})
    else:
        training_source = current.get("diagnostics", {})
    training_start = training_source.get("selection_history_start")
    training_end = training_source.get("selection_history_end")
    constraint_failures = sum(not bool(item.get("eligible")) for item in [fixed, current, *[{
        "eligible": row.get("selected_eligible", False)
    } for row in decisions]] if item.get("status") not in ("skipped", "not_requested"))
    cancelled_state = _cancelled(cancelled) or fixed_replay_cancelled or passive_run_cancelled or fixed.get("status") == "cancelled" or current.get("status") == "cancelled" or adaptive.get("status") == "cancelled" or current_family_bootstrap.get("status") == "cancelled" or final_126.get("status") == "cancelled"
    expected_decisions = len(range(0, len(eval_dates), rebalance_days)) if selection_target in ("adaptive", "both") else 0
    incomplete = (
        (selection_target in ("adaptive", "both") and len(decisions) < expected_decisions)
        or (selection_target in ("adaptive", "both") and adaptive.get("test_result") is None and not cancelled_state)
        or (selection_target in ("fixed", "both") and fixed.get("test_result") is None and not cancelled_state)
        or any(item.get("strategy_version") == 2 and item.get("status") in ("partial", "failed")
               for item in all_selections)
    )
    result_status = "cancelled" if cancelled_state else "partial" if incomplete else "completed"
    _emit(progress, 1.0, "科学参数选择完成" if result_status == "completed" else "科学参数选择未完成")
    computation = {
        "n_trials_per_decision": int(config.n_trials) if not protocol_v3 else 0,
        "local_search_budget": None if int(config.protocol_version) == 4 else int(config.local_search_budget) if protocol_v3 else None,
        "rebalance_days": rebalance_days,
        "decision_count": len(decisions),
        "candidate_count": total_candidates,
        "generation_evaluations": sum(int(item.get("training_unique_count", 0) or 0) for item in all_selections) if int(config.protocol_version) == 4 else total_generated if not protocol_v3 else sum(int(item.get("local_search_unique", 0) or 0) for item in all_selections),
        "cost_evaluations": total_evaluations,
        "constraint_failures": int(constraint_failures),
        "neighborhood_count": sum(int(item.get("neighbor_count", 0) or 0) for item in all_selections) if int(config.protocol_version) == 4 else sum(1 for archive in unique_archives for candidate in archive if not candidate.get("is_center")),
        "unique_selection_calls": len(all_selections),
        "outer_backtest_runs": int(fixed.get("test_result") is not None) + int(adaptive.get("test_result") is not None) + int(passive_result is not None),
        "bootstrap_reps": bootstrap_reps,
        "lookback_days": _LOOKBACK_DAYS,
        "fold_count_per_full_window": 4,
        "fold_days_target": _FOLD_DAYS,
        "cancelled": bool(_cancelled(cancelled)),
        "recommendation_status": "qualified" if (fixed.get("recommendation_status") == "qualified" or current.get("recommendation_status") == "qualified") else "no_qualified",
    }
    return {
        "status": result_status,
        "protocol_version": int(config.protocol_version),
        "config": config.model_dump(mode="json"),
        "protocol": {
            "selection_target": selection_target,
            "training_start": training_start,
            "training_end": training_end,
            "evaluation_start": eval_dates[0].date().isoformat(),
            "evaluation_end": eval_dates[-1].date().isoformat(),
            "lookback_trading_days": _LOOKBACK_DAYS,
            "generation_days_full": _FULL_GENERATION_DAYS,
            "validation_fold_days": _FOLD_DAYS,
            "validation_folds": 4,
            "rebalance_days": rebalance_days,
            "cost_multipliers": list(_COST_MULTIPLIERS),
            "score": ("原成本完整验证段 f=网格收益率-全仓收益率；按完整正收益连通域成员数排名，默认使用域内参数最稳定点" if int(config.protocol_version) == 4 and config.strategy.strategy_version == 2 else "等比网格完整区间超额收益 f=网格收益率-全仓收益率；最大正邻域体积排名，邻域内全部合法点均须为正" if int(config.protocol_version) == 4 else "邻域超额收益第25百分位与跨窗稳健分；分窗 U=超额收益-资金流调整回撤，R=median(U)-0.5*IQR(U)，取双成本中较差结果" if protocol_v3 else "25%四分位邻域稳健分；分窗U=252日年化收益-maxDD，R=median(U)-0.5*IQR(U)，取原成本与双佣金/最低佣金/滑点压力中较差R"),
            "bootstrap_reps": bootstrap_reps,
            "bootstrap_block_lengths": [20, 10, 40],
            "reserve_cash": reserve_cash,
            "funding_mode": "top_up_on_fill" if protocol_v3 else "configured_reserve",
            "local_search_budget": None if int(config.protocol_version) == 4 else int(config.local_search_budget) if protocol_v3 else None,
            "strategy": config.strategy.model_dump(mode="json") if int(config.protocol_version) == 4 else None,
            "grid_mode": "geometric" if int(config.protocol_version) == 4 else config.base.grid_mode,
            "spacing_unit": 0.0001 if int(config.protocol_version) == 4 else None,
            "share_unit": 100 if int(config.protocol_version) == 4 else None,
            "positive_neighbor_ratio": None if int(config.protocol_version) == 4 else _V3_POSITIVE_NEIGHBOR_RATIO if protocol_v3 else None,
            "neighbor_quantile": None if int(config.protocol_version) == 4 else 0.25 if protocol_v3 else None,
            "neighbor_perturbations": "axis_one_tick_complete_components" if int(config.protocol_version) == 4 and config.strategy.strategy_version == 2 else "all_joint_neighbors_and_complete_boxes" if int(config.protocol_version) == 4 else "single_and_pairwise_one_step" if protocol_v3 else "single_factor_one_step",
            "ranking_baseline": "independent_buy_hold" if protocol_v3 else "same_cash_passive",
        },
        "fixed": fixed,
        "adaptive": adaptive,
        "locked_funding": locked_funding,
        "evidence": {
            "fixed_decision": fixed_decision,
            "current_decision": current_decision,
            "decisions": decisions,
            "candidate_archives": candidate_archives,
            "trial_archives": trial_archives,
            "search_candidate_archives": search_candidate_archives,
            "final_126_day_check": final_126,
            "current_candidate_family_bootstrap": current_family_bootstrap,
            "data_days": int(len(available_dates)),
            "evaluation_days": int(len(eval_dates)),
            "evaluation_start": eval_dates[0].date().isoformat(),
            "evaluation_end": eval_dates[-1].date().isoformat(),
            "minute_coverage": minute_ratio,
            "selection_target": selection_target,
        },
        "warnings": warnings,
        "data_version": data.version,
        "quality": quality,
        "is_demo": bool(quality.get("is_demo") or data.metadata.get("is_demo")),
        "computation": computation,
    }

