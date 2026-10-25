"""Local, snapshot-bound ETF screening and finite in-sample candidate replay.

This is deliberately separate from robust-J protocol v5. Preflight reads catalog
and validation metadata only; price arrays are loaded by background workers.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
import time
from typing import Callable

import pandas as pd

from .etf_models import EtfResearchRequest, METHOD_VERSION, RESULT_SCHEMA_VERSION, MIN_COMPLETE_DAY_RATIO, MIN_OBSERVED_MINUTE_RATIO
from .market_data import MarketDataService
from .market_data.models import MarketDataError
from .models import BacktestConfig, DataBundle
from .providers import latest_closed_date
from .performance import current_process_rss_bytes, current_workspace_bytes, rebalance_cpu_budget
from .storage import Store, json_safe
from .snapshot_quality import frozen_quality_state


def _date(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return stamp.normalize()


def _gap(kind: str, message: str, timeframe: str | None = None) -> dict:
    return {"kind": kind, "message": message, **({"timeframe": timeframe} if timeframe else {})}


def _identity(market: MarketDataService) -> tuple[str | None, str | None]:
    manifest = market.repository.package.manifest()
    return manifest.get("package_id"), manifest.get("generation")


def _check_identity(market: MarketDataService, universe: dict) -> None:
    if _identity(market) != (universe.get("package_id"), universe.get("generation")):
        raise MarketDataError("DATA_GENERATION_CHANGED: 数据包代际已改变，请重新预检和提交")


def _symbols(store: Store, request: EtfResearchRequest, market: MarketDataService) -> list[dict]:
    if request.scope == "watchlist":
        values = store.watches()
    elif request.scope == "selected":
        values = [store.instrument(symbol) or {"symbol": symbol, "name": symbol} for symbol in request.symbols]
    else:
        # A local pool contains only instruments actually present in this package.
        keys = market.repository.package.manifest().get("datasets", {})
        present = {key.split("|", 1)[0] for key in keys}
        catalog = {row["symbol"]: row for row in store.instruments(limit=None, asset_type="etf")}
        values = [catalog.get(symbol, {"symbol": symbol, "name": symbol}) for symbol in sorted(present)]
    output = {}
    for item in values:
        symbol = str(item.get("symbol", ""))
        if symbol.startswith(("1", "5")) and item.get("asset_type", "etf") != "stock":
            output.setdefault(symbol, item)
    return list(output.values())


def _calendar(market: MarketDataService, symbol: str, manifest: dict) -> pd.DatetimeIndex:
    version = manifest.get("auxiliary_versions", {}).get("calendar")
    if version is None:
        return pd.DatetimeIndex([])
    saved = market.repository.auxiliary(symbol, "calendar", version=version)
    payload = (saved or {}).get("payload") or []
    dates = pd.DatetimeIndex(pd.to_datetime(
        [row.get("date") for row in payload], errors="coerce",
    )).dropna().normalize().unique().sort_values()
    daily = manifest.get("market_versions", {}).get("1day", {})
    if not len(dates) or not daily.get("first") or not daily.get("last"):
        return dates
    # Incremental auxiliary calendars may contain only the latest sync tail.
    # Read the frozen daily date index, never minute/OHLC bodies, to recover
    # the complete local trading clock just as the bundle consumer does.
    return market.repository.trading_days(
        symbol, min(dates.min(), _date(daily["first"])),
        max(dates.max(), _date(daily["last"])),
        calendar_version=version, daily_version=int(daily["version"]),
    )


def _validation_state(market, symbol, timeframe, manifest, start, end) -> dict:
    reference = manifest["market_versions"][timeframe]
    return market.repository.validation_state(
        symbol, timeframe, start=start, end=end,
        version=int(reference["version"]),
        calendar_version=manifest.get("auxiliary_versions", {}).get("calendar"),
        daily_version=int(manifest["market_versions"]["1day"]["version"]),
    )


def _validation_gaps(market, symbol, timeframe, manifest, start, end, *, state=None) -> list[dict]:
    state = state if state is not None else _validation_state(market, symbol, timeframe, manifest, start, end)
    gaps = []
    if state.get("failed"):
        gaps.append(_gap("quality_failed", f"{timeframe} 有 {len(state['failed'])} 个校验失败交易日：" + "、".join(state["failed"][:8]), timeframe))
    if state.get("available") is False or state.get("pending") or state.get("pending_count") or not state.get("verified_from") or not state.get("verified_through"):
        gaps.append(_gap("pending_validation", f"{timeframe} 在请求区间尚有待本地校验的交易日", timeframe))
    if state.get("verified_from") and _date(state["verified_from"]) > start:
        gaps.append(_gap("pending_validation", f"{timeframe} 质量覆盖未达到窗口起点", timeframe))
    if state.get("verified_through") and _date(state["verified_through"]) < end:
        gaps.append(_gap("pending_validation", f"{timeframe} 质量覆盖未达到窗口终点", timeframe))
    return gaps


def _limited_minute_days(market, symbol, window, state) -> list[str] | None:
    """Only known missing bars may be omitted; pending or invalid data cannot."""
    failed = sorted(set(state.get("failed") or []))
    days = {day.date().isoformat() for day in window}
    if (
        state.get("available") is False
        or state.get("pending") or state.get("pending_count")
        or not set(failed).issubset(days)
        or int(state.get("failed_count", len(failed))) != len(failed)
        or (failed and int(state.get("validated_days") or 0) < len(days) - len(failed))
        or (len(days) - len(failed)) / len(days) + 1e-12 < MIN_COMPLETE_DAY_RATIO
    ):
        return None
    records = {
        row["trade_date"]: row for row in market.repository.day_validations(
            symbol, "1min", start=window[0], end=window[-1],
        )
    }
    for day in failed:
        record = records.get(day) or {}
        report = record.get("report") or {}
        reason = str(report.get("reason") or "")
        messages = [reason, *(report.get("pending_validation") or [])]
        if (
            record.get("status") != "failed" or report.get("price_anomalies")
            or report.get("blocked_price_rows") or report.get("blocked_timestamps")
            or not all(
                re.fullmatch(r"存在 \d+ 个未补齐的分钟缺口", str(message))
                or str(message) == "交易日没有行情记录"
                for message in messages
            )
        ):
            return None
    excluded = set(failed)
    missing_minutes = 0
    for day, record in records.items():
        if day not in days:
            continue
        report = record.get("report") or {}
        if record.get("status") == "suspended" or report.get("no_trades") or report.get("non_trading"):
            continue
        missing = max(0, 240 - int(record.get("row_count", report.get("observed_rows") or 0)) - int(report.get("auction_placeholders") or 0))
        missing_minutes += missing
        # A passed ledger may include short forward fills. Such a day is not
        # a fully observed statistical sample, even though backtesting accepts it.
        if missing or report.get("synthetic_rows"):
            excluded.add(day)
    if (
        (len(days) - len(excluded)) / len(days) + 1e-12 < MIN_COMPLETE_DAY_RATIO
        or 1 - missing_minutes / (240 * len(days)) + 1e-12 < MIN_OBSERVED_MINUTE_RATIO
    ):
        return None
    return sorted(excluded)


def _sampling(window, excluded) -> dict:
    valid = len(window) - len(excluded)
    return {
        "expected_days": len(window), "valid_days": valid,
        "coverage_ratio": valid / len(window), "excluded_days": list(excluded),
        "statistics_excluded_days": list(excluded), "statistics_days": valid,
        "comparable": True,
    }


def _align_sampling(rows, windows) -> None:
    usable = [row for row in rows if row["status"] in {"ready", "limited"}]
    if not usable:
        return
    samples = [
        {day.date().isoformat() for day in windows[row["symbol"]]} - set(row["sampling"]["excluded_days"])
        for row in usable
    ]
    common = set.intersection(*samples)
    aligned = all(len(common) / row["sampling"]["expected_days"] + 1e-12 >= MIN_COMPLETE_DAY_RATIO for row in usable)
    for row in usable:
        sampling = row["sampling"]
        if aligned:
            expected = {day.date().isoformat() for day in windows[row["symbol"]]}
            sampling.update(statistics_excluded_days=sorted(expected - common), statistics_days=len(common))
            if sampling["statistics_excluded_days"]:
                row["warnings"].append(f"横截面统一使用 {len(common)}/{sampling['expected_days']} 个完整分钟交易日；缺口日期整日排除，不拼接缺失路径")
        elif row["status"] == "limited":
            sampling["comparable"] = False
            row["warnings"].append("范围内公共完整日不足90%，此标的只显示有效日统计，不参与GSS；完整标的继续按完整窗口评分")


def _freeze_quality(market, manifest, start, end, daily_start, *, sampling=None, window=None) -> None:
    """Bind successful ledger intervals to exact immutable data dependencies."""
    from .market_data.repository import VALIDATION_RULES_VERSION

    intervals = {
        timeframe: [{"start": start.date().isoformat(), "end": end.date().isoformat(), "status": "complete"}]
        for timeframe in ("1day", "1min")
    }
    if sampling is not None:
        manifest["etf_sampling"] = sampling.copy()
        excluded = set(sampling["excluded_days"])
        if excluded:
            intervals["1min"] = []
            previous = None
            for day in window:
                if day.date().isoformat() in excluded:
                    previous = None
                    continue
                stamp = day.date().isoformat()
                if previous is None:
                    intervals["1min"].append({"start": stamp, "end": stamp, "status": "complete"})
                else:
                    intervals["1min"][-1]["end"] = stamp
                previous = day
    daily_start = _date(daily_start)
    if daily_start < start and not _validation_gaps(market, manifest["symbol"], "1day", manifest, daily_start, end):
        intervals["1day"][0]["start"] = daily_start.date().isoformat()
    manifest["quality_evidence"] = {
        "schema_version": 1, "rules_version": VALIDATION_RULES_VERSION,
        "market_versions": {key: int(value["version"]) for key, value in manifest["market_versions"].items()},
        "auxiliary_versions": dict(manifest.get("auxiliary_versions", {})),
        "intervals": intervals,
    }
    content = {key: value for key, value in manifest.items() if key not in {"version", "created_at", "files", "profiles"}}
    manifest["version"] = "mkt-" + hashlib.sha256(
        json.dumps(json_safe(content), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(),
    ).hexdigest()[:24]


def preflight(
    store: Store, request: EtfResearchRequest, *, market: MarketDataService | None = None,
    persist: bool = False,
) -> dict:
    """Resolve a cohort without reading minute/daily price bodies or networking."""
    market = market or MarketDataService(store)
    package_id, generation = _identity(market)
    instruments = _symbols(store, request, market)
    manifests = {}
    for item in instruments:
        symbol = item["symbol"]
        # Reuse the existing repository instead of constructing/indexing a new
        # one through Store.snapshot for every symbol in a market-wide cohort.
        snapshot = market.repository.active_manifest(
            symbol, metadata={**item, "is_demo": False, "warnings": []},
        )
        if snapshot:
            manifests[symbol] = snapshot
    closed = _date(latest_closed_date())
    latest = [
        min(closed, _date(manifest["market_versions"]["1min"]["last"]))
        for manifest in manifests.values()
        if manifest.get("market_versions", {}).get("1min", {}).get("last")
    ]
    as_of = _date(request.as_of) if request.as_of else max(latest, default=closed)
    versions = {}
    rows = []
    windows = {}
    for instrument in instruments:
        symbol = instrument["symbol"]
        manifest = manifests.get(symbol)
        row = {
            "symbol": symbol, "symbol_name": instrument.get("name") or symbol,
            "status": "data_not_ready", "version": (manifest or {}).get("version"),
            "window_start": None, "window_end": as_of.date().isoformat(),
            "gaps": [], "warnings": [],
        }
        rows.append(row)
        if not manifest or not manifest.get("market_data"):
            row["gaps"].append(_gap("local", "本地数据包没有该 ETF 的行情快照"))
            continue
        if (manifest.get("package_id"), manifest.get("generation")) != (package_id, generation):
            row["gaps"].append(_gap("version", "行情快照不属于当前数据包代际"))
            continue
        versions[symbol] = {
            "version": manifest["version"],
            "market_versions": {
                timeframe: {key: value for key, value in reference.items() if key in {"version", "first", "last", "rows", "status"}}
                for timeframe, reference in manifest.get("market_versions", {}).items()
            },
            "auxiliary_versions": manifest.get("auxiliary_versions", {}),
        }
        references = manifest.get("market_versions", {})
        for timeframe in ("1day", "1min"):
            if timeframe not in references:
                row["gaps"].append(_gap("bars", f"缺少本地 {timeframe} 行情", timeframe))
        if row["gaps"]:
            continue
        try:
            calendar = _calendar(market, symbol, manifest)
        except (ValueError, MarketDataError) as exc:
            row["gaps"].append(_gap("calendar", f"冻结的交易日期索引不可用：{exc}"))
            continue
        if not len(calendar):
            row["gaps"].append(_gap("calendar", "该快照缺少真实交易日历"))
            continue
        if as_of > closed:
            row["gaps"].append(_gap("range", "评估截止日尚未收盘"))
            continue
        days = calendar[calendar <= as_of]
        if not len(days) or calendar.max() < as_of and as_of.weekday() < 5:
            row["gaps"].append(_gap("calendar", "交易日历未覆盖评估截止日"))
            continue
        if len(days) < request.window_days + 1:
            row["status"] = "insufficient_history"
            row["gaps"].append(_gap("history", f"需要 {request.window_days} 个交易日及前置估值收盘"))
            continue
        window = days[-request.window_days:]
        start, end = window[0], window[-1]
        windows[symbol] = window
        row.update(window_start=start.date().isoformat(), window_end=end.date().isoformat())
        # An independent daily window is used for MDDD, without requiring 250
        # days of minute prices when the selected minute window is 120 days.
        row["daily_start"] = days[max(0, len(days) - max(251, request.window_days + 1))].date().isoformat()
        for timeframe in ("1day", "1min"):
            reference = references[timeframe]
            first, last = reference.get("first"), reference.get("last")
            if not first or _date(first) > start or (timeframe == "1day" and _date(first) >= start):
                row["status"] = "insufficient_history"
                row["gaps"].append(_gap("history", f"{timeframe} 未覆盖完整窗口及必要前置行情", timeframe))
            if not last or _date(last) < end:
                row["gaps"].append(_gap("range", f"{timeframe} 截止未达到 {end.date()}", timeframe))
        actions_version = manifest.get("auxiliary_versions", {}).get("actions")
        actions = market.repository.auxiliary(symbol, "actions", version=actions_version) if actions_version is not None else None
        from .corporate_actions import action_issues, resolve_actions
        action_frame = pd.DataFrame((actions or {}).get("payload") or [])
        tick = float(instrument.get("price_tick") or 0.001)
        normalized_actions = resolve_actions(action_frame, pd.DataFrame(), tick)
        # Ordinary preflight remains catalog-only. A vendor fold date with a
        # usable ratio needs just its nearby daily reference prices to resolve.
        unresolved = normalized_actions.get("date_unresolved", pd.Series(False, index=normalized_actions.index)).fillna(False)
        ratios = pd.to_numeric(normalized_actions.get("bonus_ratio"), errors="coerce")
        usable_ratio = ratios.notna() & ratios.ne(0) & ratios.gt(-1)
        nearby_daily = pd.DataFrame()
        if (unresolved & usable_ratio).any():
            folds = pd.to_datetime(normalized_actions.loc[unresolved & usable_ratio, "ex_date"], errors="coerce").dropna()
            if len(folds):
                nearby_daily = market.repository.bars(
                    symbol, "1day", version=int(references["1day"]["version"]),
                    start=folds.min() - pd.Timedelta(days=30), end=folds.max() + pd.Timedelta(days=10),
                )
        for reason in action_issues(action_frame, nearby_daily, start, end, price_tick=tick):
            row["gaps"].append(_gap("actions", reason))
        if row["gaps"]:
            continue
        states = {timeframe: _validation_state(market, symbol, timeframe, manifest, start, end) for timeframe in ("1day", "1min")}
        daily_gaps = _validation_gaps(market, symbol, "1day", manifest, start, end, state=states["1day"])
        row["gaps"].extend(daily_gaps)
        minute_gaps = _validation_gaps(market, symbol, "1min", manifest, start, end, state=states["1min"])
        row["gaps"].extend(minute_gaps)
        can_sample = not daily_gaps and (not minute_gaps or states["1min"].get("failed"))
        excluded = _limited_minute_days(market, symbol, window, states["1min"]) if can_sample else None
        if excluded:
            row["status"] = "limited"
            row["gaps"] = [gap for gap in row["gaps"] if gap["kind"] != "pending_validation"]
            row["sampling"] = _sampling(window, excluded)
            row["warnings"].append(f"排除 {len(excluded)} 个分钟缺口日后可统计；参数回放需先补齐分钟行情")
            if not any(gap["kind"] == "quality_failed" for gap in row["gaps"]):
                row["gaps"].append(_gap("observed_minutes", "部分已校验日期含少量补线，统计按整日排除：" + "、".join(excluded[:12]), "1min"))
            continue
        if any(gap["kind"] == "quality_failed" for gap in row["gaps"]):
            row["status"] = "data_not_ready"
        elif row["gaps"]:
            row["status"] = "pending_validation"
        elif excluded == []:
            row["status"] = "ready"
            row["sampling"] = _sampling(window, [])
        else:
            row["gaps"].append(_gap("coverage", "真实完整日不足90%或真实分钟不足95%，请补齐数据后重试", "1min"))
    _align_sampling(rows, windows)
    for row in rows:
        if row["status"] not in {"ready", "limited"}:
            continue
        manifest = manifests[row["symbol"]]
        _freeze_quality(market, manifest, _date(row["window_start"]), _date(row["window_end"]), row["daily_start"], sampling=row["sampling"], window=windows[row["symbol"]])
        row["version"] = manifest["version"]
        versions[row["symbol"]]["version"] = manifest["version"]
    universe = {
        "scope": request.scope, "symbols": [item["symbol"] for item in instruments],
        "package_id": package_id, "generation": generation,
        "as_of": as_of.date().isoformat(), "versions": versions,
    }
    _check_identity(market, universe)
    if persist:
        store.remember_market_snapshots(list(manifests.values()))
        _check_identity(market, universe)
    counts = Counter(row["status"] for row in rows)
    return json_safe({
        "as_of": as_of.date().isoformat(),
        "requested_as_of": request.as_of.isoformat() if request.as_of else None,
        "window_days": request.window_days, "scope": request.scope,
        "symbols": universe["symbols"], "universe": universe, "rows": rows,
        "counts": {"total": len(rows), **{name: counts[name] for name in ("ready", "limited", "pending_validation", "insufficient_history", "data_not_ready")}},
    })


def _result(request, plan, kind="etf_research") -> dict:
    return {
        "kind": kind, "method_version": METHOD_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed", "as_of": plan["as_of"],
        "request": request.model_dump(mode="json"), "universe": plan["universe"],
        "ranking_complete": False, "rows": [], "warnings": [],
        "evaluation_kind": "in_sample", "message": "ETF 研究完成",
    }


def _stop(request, started, cancelled) -> str | None:
    if time.monotonic() - started >= request.max_runtime_seconds:
        return "runtime_limit"
    if current_process_rss_bytes() > current_workspace_bytes():
        return "workspace_limit"
    if cancelled and cancelled():
        return "cancelled"
    return None


def _set_stop(result, reason):
    result["status"] = "cancelled" if reason == "cancelled" else "partial"
    result["stop_reason"] = reason
    result["message"] = "已停止，已完成的指标和候选保留"


def _bundle(store, market, request, row, universe, *, statistics_only=False) -> DataBundle:
    _check_identity(market, universe)
    manifest = store.snapshot(row["symbol"], row["version"], persist=False)
    if manifest is None:
        raise MarketDataError("DATA_VERSION_NOT_FOUND: 冻结的行情版本已不可用")
    sampling = row.get("sampling")
    if sampling and manifest.get("etf_sampling") != sampling:
        raise MarketDataError("冻结的统计样本与研究窗口不一致，请重新预检")
    if sampling and sampling.get("excluded_days") and not statistics_only:
        raise MarketDataError("此快照仅供完整日统计；参数回放需补齐分钟行情")
    config = request.execution.backtest(row["symbol"], row["window_start"], row["window_end"])
    if sampling and sampling.get("excluded_days"):
        # The full-window certificate intentionally does not cover failed days.
        # Check the frozen complete-day intervals, never today's mutable ledger.
        actions = market.repository.auxiliary(
            row["symbol"], "actions", version=manifest.get("auxiliary_versions", {}).get("actions"),
        )
        if actions is None:
            raise MarketDataError("冻结的公司行动资料已不可用，请重新预检")
        calendar = _calendar(market, row["symbol"], manifest)
        window = calendar[(calendar >= _date(row["window_start"])) & (calendar <= _date(row["window_end"]))]
        excluded = set(sampling["excluded_days"])
        good = [day for day in window if day.date().isoformat() not in excluded]
        if (
            len(window) != request.window_days
            or len(good) != sampling.get("valid_days")
            or len(good) / len(window) + 1e-12 < MIN_COMPLETE_DAY_RATIO
            or not frozen_quality_state(manifest, "1day", window[0], window[-1])
            or any(not frozen_quality_state(manifest, "1min", day, day) for day in good)
        ):
            raise MarketDataError("冻结的完整分钟交易日质量凭证不可用，请重新预检")
    else:
        readiness = market.readiness(config, row["version"], settlement=not statistics_only)
        if not readiness["ready"] or readiness.get("degraded_days"):
            raise MarketDataError("请求窗口不满足严格分钟计算条件：" + "；".join(gap.get("message", "") for gap in readiness.get("gaps", [])))
    if statistics_only:
        from .market_data.consumer import load_native_bundle
        data = load_native_bundle(market, config, row["version"], statistics_only=True)
    else:
        data = market.load_backtest_bundle(config, row["version"])
    if sampling:
        data.metadata["etf_sampling"] = sampling.copy()
    # Retain the independent daily risk lookback. The replay engine itself
    # still obeys the selected minute window's start and end dates.
    daily_version = int(manifest["market_versions"]["1day"]["version"])
    daily = market.repository.bars(
        row["symbol"], "1day", version=daily_version,
        start=row.get("daily_start"), end=row["window_end"],
    )
    data.daily = daily.rename(columns={"timestamp": "date"})
    if "date" in data.daily:
        data.daily["date"] = pd.to_datetime(data.daily.date).dt.normalize()
    daily_state = frozen_quality_state(
        manifest, "1day", row.get("daily_start", row["window_start"]), row["window_end"],
    ) or market.repository.validation_state(
        row["symbol"], "1day", version=daily_version,
        calendar_version=manifest.get("auxiliary_versions", {}).get("calendar"),
        start=row.get("daily_start", row["window_start"]), end=row["window_end"],
    )
    data.metadata["mddd_history_verified"] = bool(
        daily_state.get("verified_from") and daily_state.get("verified_through")
        and not daily_state.get("pending") and not daily_state.get("failed")
    )
    _check_identity(market, universe)
    return data


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _candidate_order(candidate):
    metrics = candidate["metrics"]
    return (
        -(metrics.get("excess_return") if metrics.get("excess_return") is not None else -math.inf),
        metrics.get("max_principal_drawdown") if metrics.get("max_principal_drawdown") is not None else math.inf,
        metrics.get("total_fees") if metrics.get("total_fees") is not None else math.inf,
        candidate["buy_step"], candidate["sell_step"],
    )


def validate_candidates(
    request: EtfResearchRequest, row: dict, data: DataBundle,
    *, stopped: Callable[[], str | None], progress: Callable[[float, str], None] | None = None,
) -> dict:
    from .engine import run_backtest
    from .etf_statistics import generate_candidates
    from .price_basis import price_factor_at

    candidates = generate_candidates(row.get("grid_basis") or {})
    validation = {"status": "unavailable", "candidates": [], "recommendation": None, "forward_config": None}
    if not candidates:
        validation["reason"] = "价格区间、步长范围或 Tick 约束没有形成有效候选"
        return validation
    start = _date(row["window_start"])
    prior = data.daily[pd.to_datetime(data.daily.date).dt.normalize() < start].sort_values("date")
    prior = prior[(pd.to_numeric(prior.close, errors="coerce") > 0) & (pd.to_numeric(prior.get("tradestatus", 1), errors="coerce") == 1) & (pd.to_numeric(prior.volume, errors="coerce") > 0)]
    if prior.empty:
        validation["reason"] = "缺少回放开始前的有效估值收盘"
        return validation
    price_factor = price_factor_at(row.get("actions_applied", []), _date(prior.iloc[-1]["date"]))
    if not math.isfinite(price_factor) or price_factor <= 0:
        validation["reason"] = "回放起始价格基准转换无效"
        return validation
    for index, parameters in enumerate(candidates):
        rebalance_cpu_budget()
        reason = stopped()
        if reason:
            validation.update(status="cancelled" if reason == "cancelled" else "partial", stop_reason=reason)
            return validation
        patch = {
            "buy_step": parameters["buy_step"], "sell_step": parameters["sell_step"],
            "reference_price_mode": "manual", "selection_asof": row["window_end"],
            **{key: parameters[key] / price_factor for key in ("reference_price", "lower_price", "upper_price")},
        }
        config = request.execution.backtest(row["symbol"], row["window_start"], row["window_end"], **patch)
        candidate = {
            **parameters, "id": index + 1, "qualified": False, "reasons": [],
            "replay_config": config.model_dump(mode="json"),
            "metrics": {key: None for key in (
                "peak_principal_return", "matched_funding_hold_return", "excess_return",
                "max_principal_drawdown", "total_fees", "trade_count", "buy_count", "sell_count",
            )},
        }
        try:
            replay = run_backtest(config, data, summary_only=True, cancelled=lambda: bool(stopped()))
            grid = replay["strategies"]["grid"]["metrics"]
            matched = replay["strategies"]["matched_funding_hold"]["metrics"]
            net, baseline = _finite(grid.get("peak_principal_return")), _finite(matched.get("peak_principal_return"))
            candidate["metrics"] = {
                "peak_principal_return": net, "matched_funding_hold_return": baseline,
                "excess_return": net - baseline if net is not None and baseline is not None else None,
                **{key: _finite(grid.get(key)) for key in ("max_principal_drawdown", "total_fees")},
                **{key: int(grid.get(key) or 0) for key in ("trade_count", "buy_count", "sell_count")},
            }
            metrics = candidate["metrics"]
            if net is None or baseline is None:
                candidate["reasons"].append("本金或收益不可评价")
            elif net <= 0 or metrics["excess_return"] <= 0:
                candidate["reasons"].append("净收益或同资金配对超额不为正")
            if metrics["trade_count"] < request.min_trades:
                candidate["reasons"].append("成交次数不足")
            if not metrics["buy_count"] or not metrics["sell_count"]:
                candidate["reasons"].append("缺少买入或卖出成交")
            if metrics["max_principal_drawdown"] is None or metrics["max_principal_drawdown"] > request.max_principal_drawdown:
                candidate["reasons"].append("本金回撤超过门槛或不可评价")
            candidate["qualified"] = not candidate["reasons"]
        except InterruptedError:
            reason = stopped() or "cancelled"
            validation.update(status="cancelled" if reason == "cancelled" else "partial", stop_reason=reason)
            return validation
        except (ValueError, MarketDataError) as exc:
            candidate["reasons"].append(str(exc))
        validation["candidates"].append(candidate)
        if progress:
            progress((index + 1) / len(candidates), f"{row['symbol']} 核验候选 {index + 1}/{len(candidates)}")
    reason = stopped()
    if reason:
        validation.update(status="cancelled" if reason == "cancelled" else "partial", stop_reason=reason)
        return validation
    validation["candidates"].sort(key=_candidate_order)
    qualified = [candidate for candidate in validation["candidates"] if candidate["qualified"]]
    if not qualified:
        validation["status"] = "no_qualified"
        validation["reason"] = "有限候选中没有同时通过收益、成交和风险门槛的参数"
        return validation
    winner = qualified[0]
    validation.update(status="completed", recommendation=winner)
    forward_start = _date(row["window_end"]) + pd.Timedelta(days=1)
    forward_end = max(forward_start, _date(latest_closed_date()))
    validation["forward_config"] = request.execution.backtest(
        row["symbol"], forward_start.date(), forward_end.date(),
        selection_asof=row["window_end"], reference_price_mode="manual",
        **{key: winner[key] for key in ("buy_step", "sell_step", "reference_price", "lower_price", "upper_price")},
    ).model_dump(mode="json")
    return validation


def run_etf_research(
    store: Store, request: EtfResearchRequest, plan: dict,
    *, market: MarketDataService | None = None, progress=None, cancelled=None,
) -> dict:
    from .etf_statistics import compute_etf_statistics, score_etfs

    market = market or MarketDataService(store)
    result = _result(request, plan)
    result["rows"] = [{
        **item, "data_status": item["status"], "status": "pending" if item["status"] in {"ready", "limited"} else item["status"],
        "metrics": {}, "metric_details": {}, "grid_basis": {}, "diagnostics": {},
        "market_state": None,
        "eligible": False, "filter_reasons": [], "gss_score": None,
        "rank": None, "score_cohort_size": 0, "score_contributions": {},
        "validation": {
            "status": "unavailable" if item["status"] == "limited" else "not_selected",
            "candidates": [], "recommendation": None, "forward_config": None,
            **({"reason": "有效完整日统计可用；参数回放需先补齐窗口内分钟行情，再创建新的研究"} if item["status"] == "limited" else {}),
        },
    } for item in plan["rows"]]
    started = time.monotonic()
    stopped = lambda: _stop(request, started, cancelled)
    for index, row in enumerate(result["rows"]):
        rebalance_cpu_budget()
        reason = stopped()
        if reason:
            _set_stop(result, reason)
            break
        if row["data_status"] in {"ready", "limited"}:
            try:
                data = _bundle(store, market, request, row, plan["universe"], statistics_only=True)
                statistics = compute_etf_statistics(data, row["window_end"], request.window_days)
                statistics["warnings"] = list(dict.fromkeys([*row["warnings"], *statistics.get("warnings", [])]))
                row.update(statistics)
                if row["status"] != "completed":
                    row["data_status"] = row["status"]
                    if statistics.get("warnings"):
                        row["gaps"].append(_gap("statistics", statistics["warnings"][-1]))
                del data
            except (ValueError, MarketDataError) as exc:
                if "DATA_GENERATION_CHANGED" in str(exc):
                    _set_stop(result, "data_generation_changed")
                    break
                row.update(status="data_not_ready", data_status="data_not_ready")
                row["gaps"] = [*row["gaps"], _gap("quality", str(exc))]
        if progress:
            progress(.02 + .48 * (index + 1) / max(1, len(result["rows"])), f"统计 ETF {index + 1}/{len(result['rows'])} · {row['symbol']}")
    else:
        reason = stopped()
        if reason:
            _set_stop(result, reason)
        else:
            score_etfs(result["rows"])
            result["ranking_complete"] = True
    if result["ranking_complete"]:
        eligible = sorted(
            [row for row in result["rows"] if row.get("eligible") and row["status"] == "completed" and row["data_status"] == "ready"
             and (row.get("gss_score") is not None or len(result["rows"]) == 1)],
            key=lambda row: (-(row.get("gss_score") or 0), row["symbol"]),
        )[:request.validation_top_k]
        for index, row in enumerate(eligible):
            reason = stopped()
            if reason:
                _set_stop(result, reason)
                break
            try:
                data = _bundle(store, market, request, row, plan["universe"])
                row["validation"] = validate_candidates(
                    request, row, data, stopped=stopped,
                    progress=(lambda fraction, message: progress(.50 + .49 * (index + fraction) / max(1, len(eligible)), message)) if progress else None,
                )
                del data
                if row["validation"].get("stop_reason"):
                    _set_stop(result, row["validation"]["stop_reason"])
                    break
            except (ValueError, MarketDataError) as exc:
                if "DATA_GENERATION_CHANGED" in str(exc):
                    _set_stop(result, "data_generation_changed")
                    break
                row["validation"] = {"status": "unavailable", "reason": str(exc), "candidates": [], "recommendation": None, "forward_config": None}
    result["rows"].sort(key=lambda row: (row.get("rank") is None, row.get("rank") or 0, row["symbol"]))
    result["rows_count"] = len(result["rows"])
    result["counts"] = {
        "total": len(result["rows"]),
        "scored": sum(row.get("gss_score") is not None for row in result["rows"]),
        "eligible": sum(bool(row.get("eligible")) for row in result["rows"]),
        "validated": sum(row["validation"]["status"] in {"completed", "no_qualified"} for row in result["rows"]),
        "recommended": sum(row["validation"].get("recommendation") is not None for row in result["rows"]),
        **{name: 0 for name in ("ready", "limited", "pending_validation", "insufficient_history", "data_not_ready")},
        **dict(Counter(row["data_status"] for row in result["rows"])),
    }
    if result["status"] == "completed":
        result["message"] = f"ETF 研究完成：{result['counts']['total']} 个标的，{result['counts']['scored']} 个可评分，{result['counts']['recommended']} 个有合格的样本内参数"
    return json_safe(result)


def run_local_validation(store, request, plan, *, market=None, progress=None, cancelled=None):
    """Explicit local-only validation. Never calls preparation/download APIs."""
    market = market or MarketDataService(store)
    result = _result(request, plan, "etf_validation")
    result["ranking_complete"] = False
    started = time.monotonic()
    for index, row in enumerate(plan["rows"]):
        rebalance_cpu_budget()
        reason = _stop(request, started, cancelled)
        if reason:
            _set_stop(result, reason)
            break
        try:
            _check_identity(market, plan["universe"])
        except MarketDataError:
            _set_stop(result, "data_generation_changed")
            break
        entry = {**row, "reports": [], "status": "completed"}
        try:
            if not row.get("window_start"):
                raise ValueError("缺少可校验的完整本地窗口")
            frozen = plan["universe"].get("versions", {}).get(row["symbol"], {})
            current = market.repository.active_manifest(row["symbol"]) or {}
            if (
                any(current.get("market_versions", {}).get(timeframe, {}).get("version")
                    != frozen.get("market_versions", {}).get(timeframe, {}).get("version")
                    for timeframe in ("1day", "1min"))
                or current.get("auxiliary_versions", {}) != frozen.get("auxiliary_versions", {})
            ):
                raise MarketDataError("DATA_VERSION_CHANGED: 排队后本地行情或辅助资料版本已变化，请重新预检后校验")
            for timeframe_index, timeframe in enumerate(("1day", "1min")):
                reason = _stop(request, started, cancelled)
                if reason:
                    raise InterruptedError(reason)
                entry["reports"].append(market.revalidate(
                    row["symbol"], timeframe, mode="incremental",
                    start=row.get("daily_start", row["window_start"]) if timeframe == "1day" else row["window_start"],
                    end=row["window_end"],
                    cancelled=lambda: bool(_stop(request, started, cancelled)),
                    progress=(lambda fraction, message: progress(
                        (index + (timeframe_index + fraction) / 2) / max(1, len(plan["rows"])), message,
                    )) if progress else (lambda *_: None),
                ))
                _check_identity(market, plan["universe"])
                reason = _stop(request, started, cancelled)
                if reason:
                    raise InterruptedError(reason)
            if any(report.get("status") not in {"complete", "ready", "completed"} for report in entry["reports"]):
                entry.update(status="partial", reason="部分交易日仍未通过本地质量校验，请查看报告并修正数据")
        except InterruptedError as exc:
            reason = _stop(request, started, cancelled) or "cancelled"
            entry.update(status="cancelled" if reason == "cancelled" else "partial", reason=str(exc))
            result["rows"].append(entry)
            _set_stop(result, reason)
            break
        except (ValueError, MarketDataError) as exc:
            if "DATA_GENERATION_CHANGED" in str(exc):
                entry.update(status="partial", reason=str(exc))
                result["rows"].append(entry)
                _set_stop(result, "data_generation_changed")
                break
            entry.update(status="failed", reason=str(exc))
        result["rows"].append(entry)
        if progress:
            progress((index + 1) / max(1, len(plan["rows"])), f"本地校验 {index + 1}/{len(plan['rows'])} · {row['symbol']}")
    result["rows_count"] = len(result["rows"])
    result["counts"] = {"total": len(plan["rows"]), **dict(Counter(row["status"] for row in result["rows"]))}
    if result["status"] == "completed":
        if result["counts"].get("failed") or result["counts"].get("partial"):
            result["status"] = "partial"
        result["message"] = "本地校验任务完成；请重新预检研究窗口，无需下载行情"
    return json_safe(result)


def request_fingerprint(request, plan) -> str:
    return hashlib.sha256(json.dumps(
        {"method_version": METHOD_VERSION, "request": request.model_dump(mode="json"), "universe": plan["universe"]},
        sort_keys=True, ensure_ascii=False, default=str,
    ).encode()).hexdigest()
