"""Read-only, price-performance-independent plans for complete validation."""
from __future__ import annotations

import hashlib
import json
import math

import pandas as pd

from .models import BacktestConfig
from .snapshot_quality import frozen_quality_state
from .validation_models import ResearchValidationRequest


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def history_plan(calendar, verified_days, request: ResearchValidationRequest, *, symbol: str) -> dict:
    """Only timestamps and certification determine these boundaries."""
    context, strategy = request.research.research_context, request.research.strategy
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    dates = dates[dates.weekday < 5]
    if context.end_date is not None:
        dates = dates[dates <= pd.Timestamp(context.end_date)]
    verified = set(pd.DatetimeIndex(pd.to_datetime(list(verified_days))).normalize())
    result = {"symbol": symbol, "status": "data_issue", "blocks": [], "gaps": [],
              "history_policy": request.history_policy, "available_days": 0}
    if not len(dates):
        result["gaps"] = [{"kind": "calendar", "message": "没有冻结日历覆盖请求区间"}]
        return result
    if request.history_policy == "explicit":
        evaluation = dates[(dates >= pd.Timestamp(context.start_date)) &
                           (dates <= pd.Timestamp(context.end_date))]
        previous = dates[dates < pd.Timestamp(context.start_date)]
        t = int(strategy.training_max_days)
        selected = previous[-(t + 1):].append(evaluation)
        if len(previous) < t + 1 or len(evaluation) < 63:
            result["gaps"] = [{"kind": "history", "message": "显式方案需要完整训练、前置估值及至少63个样本外交易日"}]
            return result
        missing = [day for day in selected if day not in verified]
        if missing:
            result["gaps"] = [{"kind": "quality", "date": str(day.date()), "message": "请求区间存在未核验或缺失交易日"} for day in missing]
            return result
        selected_start = selected[0]
    else:
        latest = len(dates) - 1
        while latest >= 0 and dates[latest] not in verified:
            latest -= 1
        if latest < 0:
            result["gaps"] = [{"kind": "quality", "message": "没有连续可验证的本地历史"}]
            return result
        first = latest
        while first > 0 and dates[first - 1] in verified:
            first -= 1
        # One extra certified day is reserved for initial valuation/previous close.
        continuous = dates[first:latest + 1]
        n = max(0, len(continuous) - 1)
        result["available_days"] = n
        result["available_start"] = str(continuous[0].date())
        result["available_end"] = str(continuous[-1].date())
        t = min(int(strategy.training_max_days), max(int(strategy.training_min_days), n // 2))
        o = min(504, n - t)
        result["excluded_tail_days"] = [str(day.date()) for day in dates[latest + 1:]]
        if o < 63:
            shortfall = max(0, t + 63 - n)
            result["gaps"] = [{"kind": "history", "additional_verified_days": shortfall,
                "start": str(dates[max(0, first - shortfall)].date()),
                "end": str(dates[first - 1].date()) if first > 0 else None,
                "message": f"连续可用历史{n}日，训练需{t}日，样本外不足63日；未知缺口不拼接"}]
            return result
        evaluation = continuous[-o:]
        selected_start = continuous[-(o + t + 1)]
    count = min(4, len(evaluation) // 63)
    width, extra = divmod(len(evaluation), count)
    blocks, offset = [], 0
    for index in range(count):
        days = evaluation[offset:offset + width + (index < extra)]
        blocks.append({"index": index, "start": str(days[0].date()),
                       "end": str(days[-1].date()), "days": len(days)})
        offset += len(days)
    result.update(status="ready", training_days=t, training_min_days=int(strategy.training_min_days),
                  prepared_start=str(selected_start.date()), evaluation_start=str(evaluation[0].date()),
                  evaluation_end=str(evaluation[-1].date()), evaluation_days=len(evaluation), blocks=blocks,
                  decision_count=math.ceil(len(evaluation) / request.research.rebalance_days))
    return result


def _verified_days(market, manifest, calendar, timeframe):
    from .market_data.coverage_index import CoverageArchive
    from .market_data.repository import validation_rules_version, minute_grid_report_stale
    symbol = manifest["symbol"]
    market_versions = manifest.get("market_versions", {})
    selected = market_versions.get(timeframe)
    if not selected or not len(calendar):
        return set()
    first, last = calendar[0], calendar[-1]
    if frozen_quality_state(manifest, timeframe, first, last):
        return set(calendar)
    state = market.repository.validation_state(
        symbol, timeframe, start=first, end=last, version=int(selected["version"]),
        calendar_version=manifest.get("auxiliary_versions", {}).get("calendar"),
        daily_version=int(market_versions["1day"]["version"]),
    )
    if state.get("available") is False:
        # Historical certificates may consist of several separate intervals.
        return {day for day in calendar if frozen_quality_state(manifest, timeframe, day, day)}
    if state.get("status") == "complete" and not state.get("pending_count") and not state.get("failed_count"):
        return set(calendar)
    rows = CoverageArchive(market.repository).rows(symbol, timeframe, start=first, end=last)
    if not rows:
        rows = market.repository.day_validations(symbol, timeframe, start=first, end=last)
    accepted = {
        pd.Timestamp(row["trade_date"]).normalize() for row in rows
        if row.get("status") in {"passed", "suspended"} and row.get("scanned", True)
        and not row.get("dirty_reason") and int(row.get("rules_version") or 0) >= validation_rules_version(timeframe)
        and not minute_grid_report_stale(timeframe, row)
    }
    accepted.update(day for day in calendar if frozen_quality_state(manifest, timeframe, day, day))
    return accepted


def validation_base(request: ResearchValidationRequest, item: dict) -> BacktestConfig:
    base = request.research.to_optimization().base
    return base.model_copy(update={"symbol": item["symbol"], "start_date": pd.Timestamp(item["evaluation_start"]).date(),
                                   "end_date": pd.Timestamp(item["evaluation_end"]).date(), "data_policy": None})


def plan_validation(store, request: ResearchValidationRequest, *, market=None, settings=None) -> dict:
    from .market_data import MarketDataService
    from .market_data.timeframes import normalize_timeframe
    from .research.robust import estimate_robust_search
    from .providers import latest_closed_date
    market = market or MarketDataService(store, settings=settings)
    items, versions, universe = [], {}, []
    total_evaluations, total_memory, total_candidate_evaluations = 0, 0.0, 0
    context = request.research.research_context
    for symbol in request.symbols:
        manifest = store.snapshot(symbol, request.requested_data_versions.get(symbol), persist=False)
        instrument = store.instrument(symbol) or {"symbol": symbol}
        universe.append({key: instrument.get(key) for key in ("symbol", "name", "listing_date", "delisting_date", "asset_type")})
        if not manifest or not manifest.get("market_data"):
            items.append({"symbol": symbol, "status": "data_issue", "blocks": [],
                          "gaps": [{"kind": "version", "message": "指定冻结版本不存在或当前没有本地行情"}]})
            continue
        version = versions[symbol] = manifest["version"]
        try:
            daily_version = int(manifest["market_versions"]["1day"]["version"])
            daily = market.repository.bars(symbol, "1day", version=daily_version)
            if daily.empty:
                raise ValueError("缺少日线与前置估值")
            daily["timestamp"] = pd.to_datetime(daily.timestamp)
            end = min(pd.Timestamp(context.end_date or latest_closed_date()), pd.Timestamp(latest_closed_date()))
            calendar = market.repository.trading_days(symbol, daily.timestamp.min(), end,
                calendar_version=manifest.get("auxiliary_versions", {}).get("calendar"), daily_version=daily_version)
            # Incremental auxiliary payloads can contain only the recent tail.
            # The selected daily version is also immutable evidence of trading dates.
            calendar = pd.DatetimeIndex(calendar).append(pd.DatetimeIndex(daily.timestamp)).normalize().unique().sort_values()
            calendar = calendar[(calendar <= end) & (calendar.weekday < 5)]
            daily_good = _verified_days(market, manifest, calendar, "1day")
            target = normalize_timeframe(context.frequency)
            target_good = daily_good if target == "1day" else _verified_days(market, manifest, calendar, target)
            item = history_plan(calendar, daily_good.intersection(target_good), request, symbol=symbol)
            item["data_version"] = version
            item["instrument"] = universe[-1]
            if item["status"] == "ready":
                base = validation_base(request, item)
                prepared = base.model_copy(update={"start_date": pd.Timestamp(item["prepared_start"]).date()})
                if request.history_policy == "explicit":
                    if pd.Timestamp(context.end_date) > pd.Timestamp(latest_closed_date()):
                        raise ValueError("显式评价截止日尚未收盘，不能自动裁剪日期")
                    prepared = prepared.model_copy(update={"end_date": context.end_date})
                origin = request.research.research_origin or {}
                if origin.get("applied_parameter_guidance") and origin.get("source_as_of"):
                    if pd.Timestamp(origin["source_as_of"]) >= pd.Timestamp(item["evaluation_start"]):
                        raise ValueError("ETF来源参数范围建议晚于本方案选参时点，不能用于该历史验证")
                ready = market.readiness(prepared, version)
                if not ready.get("ready") or ready.get("degraded_days"):
                    item.update(status="data_issue", gaps=ready.get("gaps") or [{"kind": "quality", "message": "严格验证不接受分钟降级"}])
                else:
                    prior = daily[(daily.timestamp < pd.Timestamp(item["evaluation_start"])) & (daily.close > 0)]
                    if "tradestatus" in prior:
                        prior = prior[prior.tradestatus == 1]
                    if prior.empty:
                        raise ValueError("评估开始前缺少真实估值收盘")
                    reference = float(prior.iloc[-1].close)
                    account = request.accounts[symbol]
                    initial = round(account.total_budget - reference * context.initial_shares - sum(c.amount for c in account.contributions), 2)
                    if initial < 0:
                        raise ValueError("总预算不足以覆盖初始持仓和全部计划补款")
                    lower, upper = pd.Timestamp(item["evaluation_start"]), pd.Timestamp(item["evaluation_end"]) + pd.Timedelta(days=1)
                    if any(not lower <= pd.Timestamp(c.available_at) < upper for c in account.contributions):
                        raise ValueError("计划补款到账时间必须位于本标的实际评价区间")
                    item.update(initial_cash=initial, valuation_price=reference, valuation_date=str(prior.iloc[-1].timestamp.date()))
                    bundle = market.load_backtest_bundle(prepared, version, summary_only=True)
                    optimization = request.research.to_optimization().model_copy(update={
                        "base": base, "selection_target": "both", "requested_data_version": version,
                        "min_history_days": min(request.research.min_history_days, item["training_days"]),
                        "strategy": request.research.strategy.model_copy(update={"training_max_days": item["training_days"]})})
                    estimate = estimate_robust_search(optimization, bundle)
                    if estimate.get("status") not in {None, "ready"}:
                        item["status"] = "budget_exceeded" if estimate.get("status") == "budget_exceeded" else "data_issue"
                        item["gaps"] = [{"kind": "search_budget", "message": str(reason)}
                                        for reason in estimate.get("reasons", [])] or [
                            {"kind": "search_budget", "message": "选参预算或训练前置条件不足"}]
                    per_run = int(estimate.get("task_evaluation_upper_bound") or estimate.get("effective_total_evaluation_budget") or 1)
                    upper_bound = 4 * per_run
                    # Calls, not ledger rows: each candidate/kernel call runs its paired accounts.
                    replay_upper_bound = 8 + 8 * 6 + 4 * (1 + item["decision_count"]) * 4
                    series = manifest.get("market_versions", {}).get(target, {})
                    expected_rows = math.ceil(int(series.get("rows") or 0) *
                        (item["training_days"] + item["evaluation_days"] + 1) / max(1, len(daily)))
                    # Numerical arrays plus one pair of full Python ledgers/serialization.
                    bar_memory_mb = expected_rows * 2048 / (1024 * 1024)
                    item_memory = float(estimate.get("estimated_task_memory_mb") or 0) + bar_memory_mb
                    item["estimated_work"] = {"selection_runs": 4, "arms": 8,
                        "candidate_evaluations_upper_bound": upper_bound,
                        "selection_evaluations_per_run_upper_bound": per_run,
                        "replay_evaluations_upper_bound": replay_upper_bound,
                        "qualification_runs": 8, "execution_pressure_scenarios_per_arm": 6,
                        "price_pressure_scenarios_per_decision": 4,
                        "estimated_bar_rows": expected_rows, "estimated_memory_mb": item_memory}
                    total_evaluations += upper_bound + replay_upper_bound
                    total_candidate_evaluations += upper_bound
                    total_memory += item_memory
            items.append(item)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            items.append({"symbol": symbol, "data_version": version, "status": "data_issue", "blocks": [],
                          "gaps": [{"kind": "data", "message": str(exc)}]})
    ready = bool(items) and all(item["status"] == "ready" for item in items)
    budget = request.total_evaluation_budget or max(1, total_evaluations)
    workspace = float(getattr(getattr(settings, "performance", None), "workspace_mb", 8192))
    plan = {"suite_version": request.suite_version,
            "status": "ready" if ready else "budget_exceeded" if any(item["status"] == "budget_exceeded" for item in items) else "data_issue",
            "symbols": items, "data_versions": versions,
            "primary_arm": f"{request.research.strategy.training_window_mode}:{request.research.prior_policy}:adaptive",
            "universe": {"members": universe, "basis": "current_local_subset", "historical_universe_verified": False},
            "estimated_work": {"candidate_evaluations_upper_bound": total_candidate_evaluations,
                "total_evaluations_upper_bound": total_evaluations,
                "total_evaluation_budget": budget, "estimated_memory_mb": total_memory,
                "workspace_budget_mb": workspace, "max_runtime_seconds": request.max_runtime_seconds},
            "statistics": {"status": "descriptive_only", "human_untouched_verified": False}}
    etf_count = sum(str(row["symbol"]).startswith(("1", "5")) for row in items)
    plan["warnings"] = (["ETF候选不足3只；单标的账户可评价，ETF筛选对照将缺少完整共同样本"]
                        if etf_count < 3 else [])
    if ready and (budget < total_evaluations or total_memory > workspace):
        plan["status"] = "budget_exceeded"
        plan["reason"] = "完整套件的冻结工作量或内存超过预算，不能删减项目后标为完整"
    plan["plan_hash"] = plan_hash(request, plan)
    return plan


def plan_hash(request: ResearchValidationRequest, plan: dict) -> str:
    config = request.model_dump(mode="json", exclude={"plan_hash"})
    config["requested_data_versions"] = plan.get("data_versions", {})
    return stable_hash({"request": config, "plan": {k: v for k, v in plan.items() if k != "plan_hash"}})


def load_validation_bundles(store, request, plan, *, settings=None, cancelled=lambda: False):
    """Load exactly the ranges and versions already frozen by the preview."""
    from .data import DataService
    service, bundles = DataService(store, settings=settings), {}
    if plan_hash(request, plan) != plan.get("plan_hash"):
        raise ValueError("冻结验证方案哈希不一致")
    for item in plan["symbols"]:
        if cancelled():
            break
        if item.get("status") != "ready":
            continue
        base = validation_base(request, item).model_copy(update={"start_date": pd.Timestamp(item["prepared_start"]).date()})
        bundles[item["symbol"]] = service.ensure_bundle(base, item["data_version"])
    return bundles
