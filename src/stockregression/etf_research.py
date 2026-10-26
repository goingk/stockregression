"""Local, snapshot-bound ETF screening and finite in-sample candidate replay.

This is deliberately separate from robust-J protocol v5. Preflight reads catalog
and validation metadata only; price arrays are loaded by background workers.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import time
from typing import Callable

import pandas as pd

from .etf_models import EtfResearchRequest, METHOD_VERSION, RESULT_SCHEMA_VERSION, MIN_COMPLETE_DAY_RATIO
from .etf_guidance import GUIDANCE_VERSION, parameter_guidance
from .market_data import MarketDataService
from .market_data.models import MarketDataError
from .models import BacktestConfig, DataBundle
from .providers import latest_closed_date
from .performance import current_process_rss_bytes, current_workspace_bytes, rebalance_cpu_budget
from .storage import Store, json_safe
from .snapshot_quality import frozen_quality_state
from .data_resolution import (
    POLICY as ETF_DATA_POLICY, build_data_resolution, resolution_intervals,
    resolution_sampling, resolution_snapshot_version, validate_data_resolution,
)


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


def _align_sampling(rows, windows) -> None:
    usable = [row for row in rows if row["status"] in {"ready", "limited"}]
    if not usable:
        return
    scoring_group = []
    for row in usable:
        sampling = row["sampling"]
        sampling["comparable"] = True
        own = {day.date().isoformat() for day in windows[row["symbol"]]} - set(sampling["excluded_days"])
        if len(own) / max(1, sampling["expected_days"]) + 1e-12 >= MIN_COMPLETE_DAY_RATIO:
            scoring_group.append((row, own))
        else:
            sampling["comparable"] = False
            row["warnings"].append(
                f"自有完整分钟样本 {len(own)}/{sampling['expected_days']} 日不足90%；保留统计和试算，不纳入GSS共同样本组",
            )
    if not scoring_group:
        return

    def common_for(group):
        if len(group) < 2:
            return None
        common = set.intersection(*(sample for _, sample in group))
        ratio = min(
            len(common) / max(1, row["sampling"]["expected_days"])
            for row, _ in group
        )
        return common if ratio + 1e-12 >= MIN_COMPLETE_DAY_RATIO else None

    active = list(scoring_group)
    excluded = []
    common = common_for(active)
    while common is None and len(active) > 2:
        best = None
        for index, (row, own) in enumerate(active):
            trial = active[:index] + active[index + 1:]
            trial_common = set.intersection(*(sample for _, sample in trial))
            ratio = min(
                len(trial_common) / max(1, item["sampling"]["expected_days"])
                for item, _ in trial
            )
            # Prefer the largest remaining common sample. Ties remove the
            # least complete row first, then preserve input order.
            candidate = (ratio, len(trial_common), -len(own), -index, index)
            if best is None or candidate > best[0]:
                best = (candidate, index)
        if best is None:
            break
        excluded.append(active.pop(best[1]))
        common = common_for(active)

    if common is None:
        for row, _ in scoring_group:
            sampling = row["sampling"]
            sampling["comparable"] = False
            row["warnings"].append(
                "高覆盖标的之间无法形成至少两只、共同完整分钟样本达到90%的GSS样本组；保留本标的统计和试算，不参与GSS横向评分",
            )
        return

    for row, _ in excluded:
        sampling = row["sampling"]
        sampling["comparable"] = False
        row["warnings"].append(
            f"GSS样本对齐后剩余共同完整分钟 {len(common)}/{sampling['expected_days']} 日，"
            "本标的与主样本组不兼容，已从横向评分组排除",
        )
    for row, _ in active:
        sampling = row["sampling"]
        expected = {day.date().isoformat() for day in windows[row["symbol"]]}
        sampling.update(statistics_excluded_days=sorted(expected - common), statistics_days=len(common))
        if sampling["statistics_excluded_days"]:
            row["warnings"].append(
                f"GSS共同样本使用 {len(common)}/{sampling['expected_days']} 个完整分钟交易日；跨缺口窗口整段排除",
            )


def _freeze_quality(market, manifest, start, end, daily_start, *, sampling=None, window=None, data_resolution=None) -> None:
    """Bind successful ledger intervals to exact immutable data dependencies."""
    from .market_data.repository import VALIDATION_RULES_VERSION, validation_rules_version

    intervals = {
        timeframe: [{"start": start.date().isoformat(), "end": end.date().isoformat(), "status": "complete"}]
        for timeframe in ("1day", "1min")
    }
    if data_resolution is not None:
        manifest["data_resolution"] = data_resolution
        intervals = {
            timeframe: resolution_intervals(data_resolution, timeframe)
            for timeframe in ("1day", "1min")
        }
        intervals["1day"] = [
            *(
                {"start": day, "end": day, "status": "complete"}
                for day in data_resolution.get("history_daily_days", data_resolution.get("prior_daily_days", []))
                if day not in {entry["start"] for entry in intervals["1day"]}
            ),
            *intervals["1day"],
        ]
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
    if data_resolution is None and daily_start < start and not _validation_gaps(market, manifest["symbol"], "1day", manifest, daily_start, end):
        intervals["1day"][0]["start"] = daily_start.date().isoformat()
    manifest["quality_evidence"] = {
        "schema_version": 1, "rules_version": VALIDATION_RULES_VERSION,
        "rules_versions": {
            timeframe: validation_rules_version(timeframe)
            for timeframe in ("1day", "1min")
        },
        "market_versions": {key: int(value["version"]) for key, value in manifest["market_versions"].items()},
        "auxiliary_versions": dict(manifest.get("auxiliary_versions", {})),
        "intervals": intervals,
    }
    manifest["version"] = resolution_snapshot_version(manifest)


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
        min(closed, _date((manifest.get("market_versions", {}).get("1day")
                           or manifest.get("market_versions", {}).get("1min", {}))["last"]))
        for manifest in manifests.values()
        if (manifest.get("market_versions", {}).get("1day")
            or manifest.get("market_versions", {}).get("1min", {})).get("last")
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
        if "1day" not in references:
            row["gaps"].append(_gap("bars", "缺少本地 1day 行情；无法建立日线统计与估值基准", "1day"))
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
        for timeframe in ("1day",):
            reference = references[timeframe]
            first, last = reference.get("first"), reference.get("last")
            if not first or _date(first) >= start:
                row["status"] = "insufficient_history"
                row["gaps"].append(_gap("history", f"{timeframe} 未覆盖完整窗口及必要前置行情", timeframe))
            if not last or _date(last) < end:
                # Preserve the requested window. An internal or trailing
                # daily hole is resolved per session and appears in the plan.
                pass
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
        try:
            data_resolution = build_data_resolution(market, symbol, manifest, window)
        except (ValueError, MarketDataError) as exc:
            row["status"] = "data_not_ready"
            row["gaps"].append(_gap("data_resolution", str(exc)))
            continue
        pending_validation = data_resolution.get("pending_validation") or {}
        if any(pending_validation.values()):
            row.update(status="pending_validation", replay_allowed=False)
            for timeframe, pending_days in pending_validation.items():
                if pending_days:
                    row["gaps"].append(_gap(
                        "pending_validation",
                        f"{len(pending_days)} 个交易日需先本地校验"
                        f"（{pending_days[0]} 至 {pending_days[-1]}）；"
                        "请点击“校验待处理数据”后重新预检",
                        timeframe,
                    ))
            continue
        row["data_resolution"] = data_resolution
        row["sampling"] = resolution_sampling(data_resolution)
        row["replay_allowed"] = bool(
            data_resolution["counts"].get("minute", 0) + data_resolution["counts"].get("daily", 0) > 0
            and data_resolution.get("first_evaluable_date")
        )
        # A row with a usable day is researchable even when other dates are
        # degraded or skipped. Those dates remain visible in the frozen plan.
        if row["replay_allowed"]:
            row["status"] = "ready"
            degraded = data_resolution["counts"].get("daily", 0)
            skipped = data_resolution["counts"].get("skip", 0)
            if degraded or skipped:
                row["warnings"].append(
                    f"逐日数据：分钟 {data_resolution['counts'].get('minute', 0)} 日，日线降级 {degraded} 日，跳过 {skipped} 日",
                )
        else:
            all_unverified = all(not item.get("evidence", {}).get("1day") for item in data_resolution["days"])
            row["status"] = "pending_validation" if all_unverified else "data_not_ready"
            row["gaps"].append(_gap(
                "pending_validation" if all_unverified else "no_evaluable_days",
                "本地日线尚无冻结质量凭证，请先校验本地行情" if all_unverified
                else "窗口内没有可用于研究或回放的分钟/日线交易日",
            ))
    _align_sampling(rows, windows)
    for row in rows:
        if row["status"] not in {"ready", "limited"}:
            continue
        manifest = manifests[row["symbol"]]
        _freeze_quality(market, manifest, _date(row["window_start"]), _date(row["window_end"]), row["daily_start"], sampling=row["sampling"], window=windows[row["symbol"]], data_resolution=row["data_resolution"])
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
        "method_version": METHOD_VERSION,
        "as_of": as_of.date().isoformat(),
        "requested_as_of": request.as_of.isoformat() if request.as_of else None,
        "window_days": request.window_days, "scope": request.scope,
        "symbols": universe["symbols"], "universe": universe, "rows": rows,
        "counts": {"total": len(rows), **{name: counts[name] for name in ("ready", "limited", "pending_validation", "insufficient_history", "data_not_ready")}},
    })


def _result(request, plan, kind="etf_research") -> dict:
    return {
        "kind": kind, "method_version": plan.get("method_version", METHOD_VERSION),
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
    data_resolution = manifest.get("data_resolution")
    v5 = data_resolution is not None
    if v5:
        data_resolution = validate_data_resolution(manifest, row["window_start"], row["window_end"])
        if row.get("data_resolution", {}).get("plan_hash") != data_resolution["plan_hash"]:
            raise MarketDataError("DATA_RESOLUTION_MISMATCH: 行结果与冻结逐日计划不一致")
        sampling = row.get("sampling") or resolution_sampling(data_resolution)
        if manifest.get("etf_sampling") != sampling:
            raise MarketDataError("冻结的统计样本与研究窗口不一致，请重新预检")
        config = request.execution.backtest(
            row["symbol"], row["window_start"], row["window_end"], data_policy=ETF_DATA_POLICY,
        )
    else:
        # Old ready snapshots remain reproducible through their strict v1-v4
        # path. A limited snapshot can never silently gain v5 semantics.
        if row.get("data_status") == "limited" or (row.get("sampling") or {}).get("excluded_days"):
            raise MarketDataError("旧版 limited 结果不能自动升级为 v5 回放；请按新规则重新研究")
        sampling = row.get("sampling")
        config = request.execution.backtest(row["symbol"], row["window_start"], row["window_end"])
        readiness = market.readiness(config, row["version"], settlement=not statistics_only)
        if not readiness["ready"] or readiness.get("degraded_days"):
            raise MarketDataError("旧版快照不满足严格分钟计算条件：" + "；".join(gap.get("message", "") for gap in readiness.get("gaps", [])))
    if statistics_only:
        from .market_data.consumer import load_native_bundle
        data = load_native_bundle(market, config, row["version"], statistics_only=True)
    else:
        data = market.load_backtest_bundle(config, row["version"])
    if v5:
        data.metadata["etf_sampling"] = sampling.copy()
        data.metadata["data_resolution"] = data_resolution
        # The v5 consumer returns only plan-accepted sessions and frozen
        # history_daily_days. Verify the full 250-session MDDD window against
        # that frozen list and the returned frame; never consult today's ledger.
        history = data.daily.copy()
        if not history.empty:
            history["date"] = pd.to_datetime(history.date, errors="coerce").dt.normalize()
            valid_dates = set(history.loc[
                pd.to_numeric(history.close, errors="coerce").gt(0)
                & pd.to_numeric(history.volume, errors="coerce").gt(0)
                & pd.to_numeric(history.tradestatus, errors="coerce").eq(1), "date",
            ].dt.strftime("%Y-%m-%d"))
            calendar = _calendar(market, row["symbol"], manifest)
            expected = {day.date().isoformat() for day in calendar[calendar <= _date(row["window_end"])][-250:]}
            accepted_dates = {
                item["date"] for item in data_resolution["days"]
                if item["mode"] in {"minute", "daily"}
            }
            frozen_daily_dates = set(data_resolution.get("history_daily_days") or []) | accepted_dates
            data.metadata["mddd_history_verified"] = bool(
                len(expected) == 250 and expected.issubset(frozen_daily_dates)
                and expected.issubset(valid_dates)
            )
        else:
            data.metadata["mddd_history_verified"] = False
    else:
        # Legacy snapshots have no resolution plan. Keep their strict historic
        # behavior while deriving MDDD trust only from frozen certificates.
        daily_version = int(manifest["market_versions"]["1day"]["version"])
        daily = market.repository.bars(
            row["symbol"], "1day", version=daily_version,
            start=row.get("daily_start"), end=row["window_end"],
        )
        if not daily.empty:
            daily = daily.rename(columns={"timestamp": "date"})
            daily["date"] = pd.to_datetime(daily.date).dt.normalize()
            daily = daily.loc[daily.date.map(
                lambda day: bool(frozen_quality_state(manifest, "1day", day, day)),
            )].copy()
            data.daily = daily
        data.metadata["mddd_history_verified"] = len(data.daily) >= 250
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
    from .etf_statistics import generate_candidates, generate_candidates_with_diagnostics
    from .price_basis import price_factor_at

    basis = row.get("grid_basis") or {}
    basis = dict(basis)
    grid_basis = dict(basis.get("grid_basis") or basis)
    current = _finite(grid_basis.get("current_raw_price"))
    if current is not None and grid_basis.get("current_price_basis") != "as_of_adjusted":
        # Grid bounds are calculated in the as-of adjusted price basis.
        # Convert the final raw close into that same basis before comparing.
        price_date = grid_basis.get("current_price_date") or row["window_end"]
        current_factor = price_factor_at(row.get("actions_applied", []), _date(price_date))
        if math.isfinite(current_factor) and current_factor > 0:
            grid_basis["current_raw_price"] = current * current_factor
    grid_basis["price_basis"] = "as_of_adjusted"
    basis["grid_basis"] = grid_basis
    candidates = generate_candidates(basis)
    generated = (generate_candidates_with_diagnostics(basis) if not candidates
                 else {"candidates": candidates, "reason_code": None, "constraints": {}})
    validation = {"status": "unavailable", "candidates": [], "recommendation": None, "forward_config": None}
    if not candidates:
        validation.update(status="no_candidates", reason_code=generated["reason_code"],
                          constraints=generated["constraints"])
        source_date = grid_basis.get("current_price_date")
        source_name = {"minute": "分钟线", "daily": "日线"}.get(
            grid_basis.get("current_price_source"), "可信行情",
        )
        observed_at = f"截至 {source_date} 的最近有效收盘" if source_date else "现有可信收盘"
        if generated["reason_code"] == "current_at_or_below_lower":
            validation["reason"] = (
                f"{observed_at}（{source_name}）{grid_basis['current_raw_price']:.4f} 元低于建议区间下界 "
                f"{grid_basis['lower_price']:.7f} 元，因此未生成网格"
            )
        elif generated["reason_code"] == "current_at_or_above_upper":
            validation["reason"] = (
                f"{observed_at}（{source_name}）{grid_basis['current_raw_price']:.4f} 元高于建议区间上界 "
                f"{grid_basis['upper_price']:.7f} 元，因此未生成网格"
            )
        else:
            validation["reason"] = f"未生成有效网格：{generated['reason_code']}"
        return validation
    start = _date(row["window_start"])
    prior = data.daily[pd.to_datetime(data.daily.date).dt.normalize() < start].sort_values("date")
    prior = prior[(pd.to_numeric(prior.close, errors="coerce") > 0) & (pd.to_numeric(prior.get("tradestatus", 1), errors="coerce") == 1) & (pd.to_numeric(prior.volume, errors="coerce") > 0)]
    if prior.empty:
        validation["reason_code"] = "initial_valuation_missing"
        validation["reason"] = "缺少回放开始前的有效估值收盘，无法初始化本金和持仓"
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
        if data.metadata.get("data_resolution"):
            config.data_policy = ETF_DATA_POLICY
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
    forward = request.execution.backtest(
        row["symbol"], forward_start.date(), forward_end.date(),
        selection_asof=row["window_end"], reference_price_mode="manual",
        **{key: winner[key] for key in ("buy_step", "sell_step", "reference_price", "lower_price", "upper_price")},
    )
    if data.metadata.get("data_resolution"):
        forward.data_policy = ETF_DATA_POLICY
    validation["forward_config"] = forward.model_dump(mode="json")
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
        "market_state": None, "parameter_guidance": None,
        "eligible": False, "filter_reasons": [], "gss_score": None,
        "rank": None, "score_cohort_size": 0, "score_contributions": {},
        "replay_allowed": bool(item.get(
            "replay_allowed",
            item.get("status") == "ready" and not (item.get("sampling") or {}).get("excluded_days"),
        )),
        "validation": {
            "status": "not_selected", "candidates": [],
            "recommendation": None, "forward_config": None,
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
                statistics["warnings"] = list(dict.fromkeys([
                    *row["warnings"], *getattr(data, "warnings", []), *statistics.get("warnings", []),
                ]))
                row.update(statistics)
                row["parameter_guidance"] = parameter_guidance(row, request)
                if row["status"] != "completed":
                    row["data_status"] = row["status"]
                    row["replay_allowed"] = False
                    if statistics.get("warnings"):
                        row["gaps"].append(_gap("statistics", statistics["warnings"][-1]))
                del data
            except (ValueError, MarketDataError) as exc:
                if "DATA_GENERATION_CHANGED" in str(exc):
                    _set_stop(result, "data_generation_changed")
                    break
                row.update(status="data_not_ready", data_status="data_not_ready")
                row["replay_allowed"] = False
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
        eligible_rows = [row for row in result["rows"]
                         if row.get("eligible") and row.get("replay_allowed")]
        ranked = sorted(
            (row for row in eligible_rows if row.get("gss_score") is not None),
            key=lambda row: (-(row["gss_score"]), row["symbol"]),
        )
        scored_symbols = {row["symbol"] for row in ranked}
        unscored = sorted(
            (row for row in eligible_rows if row["symbol"] not in scored_symbols),
            key=lambda row: row["symbol"],
        )
        eligible = (ranked + unscored)[:request.validation_top_k]
        for index, row in enumerate(eligible):
            reason = stopped()
            if reason:
                _set_stop(result, reason)
                break
            try:
                data = _bundle(store, market, request, row, plan["universe"])
                row["warnings"] = list(dict.fromkeys([
                    *row.get("warnings", []), *getattr(data, "warnings", []),
                ]))
                row["validation"] = validate_candidates(
                    request, row, data, stopped=stopped,
                    progress=(lambda fraction, message: progress(.50 + .49 * (index + fraction) / max(1, len(eligible)), message)) if progress else None,
                )
                if row["validation"].get("reason_code") == "initial_valuation_missing":
                    row["replay_allowed"] = False
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
        "no_candidates": sum(row["validation"]["status"] == "no_candidates" for row in result["rows"]),
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
        {"method_version": plan.get("method_version", METHOD_VERSION), "guidance_version": GUIDANCE_VERSION,
         "request": request.model_dump(mode="json"), "universe": plan["universe"]},
        sort_keys=True, ensure_ascii=False, default=str,
    ).encode()).hexdigest()
