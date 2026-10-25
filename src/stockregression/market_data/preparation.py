from __future__ import annotations

from typing import Callable

import pandas as pd

from ..models import BacktestConfig
from ..providers import fetch_market_actions, fetch_market_calendar, latest_closed_date
from .models import MarketDataError
from .service import MarketDataService
from .sync import MarketDataSyncService
from .timeframes import normalize_timeframe


class MarketPreparationService:
    """Build and execute an explicit, package-local data preparation plan."""

    def __init__(self, service: MarketDataService):
        self.service = service
        self.repository = service.repository

    def plan(self, config: BacktestConfig) -> dict:
        readiness = self.service.readiness(config)
        target = normalize_timeframe(config.frequency)
        start = pd.Timestamp(config.start_date).normalize()
        history_start = start - pd.Timedelta(days=10)
        end = pd.Timestamp(config.end_date).normalize() if config.end_date else pd.Timestamp(latest_closed_date()).normalize()
        gaps = list(readiness.get("gaps", []))
        kinds = {str(gap.get("kind")) for gap in gaps}
        steps: list[dict] = []

        datasets = {
            timeframe: self.repository.dataset(config.symbol, timeframe)
            for timeframe in dict.fromkeys(("1day", target))
        }
        calendar = self.repository.auxiliary(config.symbol, "calendar")
        calendar_available = bool(calendar and calendar.get("payload"))
        try:
            session_dates = (
                self.repository.trading_days(config.symbol, history_start, end)
                if calendar_available else None
            )
        except (OSError, ValueError, TypeError):
            session_dates = None

        def first_local_date(timeframe: str) -> str | None:
            dataset = datasets.get(timeframe)
            value = dataset.get("first_time") if dataset else None
            return pd.Timestamp(value).date().isoformat() if value is not None else None

        def missing_leading_range(timeframe: str, lower, upper) -> dict | None:
            first = first_local_date(timeframe)
            missing_start = pd.Timestamp(lower).normalize()
            missing_end = pd.Timestamp(upper).normalize()
            if first:
                first_day = pd.Timestamp(first)
                missing_end = min(missing_end, first_day - pd.Timedelta(days=1))
            if missing_start > missing_end:
                return None
            count = None
            if session_dates is not None:
                selected = session_dates[(session_dates >= missing_start) & (session_dates <= missing_end)]
                if timeframe == "1week":
                    count = int(selected.to_period("W-FRI").nunique())
                elif timeframe == "1month":
                    count = int(selected.to_period("M").nunique())
                else:
                    count = int(len(selected))
            return {
                "start": missing_start.date().isoformat(),
                "end": missing_end.date().isoformat(),
                "count": count,
                "unit": "周线周期" if timeframe == "1week" else "月线周期" if timeframe == "1month" else "交易日",
            }

        history_coverage = {
            "request_start": start.date().isoformat(),
            "request_end": end.date().isoformat(),
            "daily_first": first_local_date("1day"),
            "target_first": first_local_date(target),
            "preparation_start": history_start.date().isoformat(),
            "preparation_end": (start - pd.Timedelta(days=1)).date().isoformat(),
            "preparation_lookback_calendar_days": 10,
            "daily_validation_missing": missing_leading_range("1day", history_start, end),
            "daily_request_missing": missing_leading_range("1day", start, end),
            "daily_prestart_missing": missing_leading_range(
                "1day", history_start, start - pd.Timedelta(days=1),
            ),
            "target_request_missing": (
                missing_leading_range(target, start, end) if target != "1day" else None
            ),
            "previous_close_missing": any(gap.get("kind") == "reference_price" for gap in gaps),
            "trading_calendar_available": calendar_available,
        }

        if "calendar" in kinds or "actions" in kinds:
            steps.append({
                "kind": "auxiliary",
                "symbol": config.symbol,
                "start": str(history_start.date()),
                "end": str(end.date()),
                "reason": "补齐交易日历和公司行动资料",
            })
        daily_gap = any(
            (str(gap.get("timeframe") or "") == "1day" and gap.get("kind") not in {"quality", "calendar", "actions"})
            or str(gap.get("kind")) in {"daily", "reference_price"}
            for gap in gaps
        )
        if daily_gap:
            steps.append({
                "kind": "bars", "timeframe": "1day", "mode": "history",
                "symbol": config.symbol, "start": str(history_start.date()),
                "end": str(end.date()),
                "reason": (
                    f"补齐日线；校验从请求起点前 10 个自然日 {history_start.date()} 开始，"
                    f"用于查找 {start.date()} 前最近的有效收盘价"
                ),
            })
        target_gap = any(str(gap.get("timeframe") or "") == target and gap.get("kind") not in {"quality", "calendar", "actions"} for gap in gaps)
        if target != "1day" and target_gap:
            dataset = self.repository.dataset(config.symbol, target)
            if target == "1min" and dataset is None:
                steps.append({
                    "kind": "import", "timeframe": target, "symbol": config.symbol,
                    "reason": "1 分钟历史行情只从本地数据包读取；请登记数据包或导入文件后再运行",
                })
            else:
                steps.append({
                    "kind": "bars", "timeframe": target,
                    "mode": "repair" if dataset else "history",
                    "symbol": config.symbol, "start": str(history_start.date()),
                    "end": str(end.date()),
                    "reason": (
                        "修补近期 1min 行情；更早的分钟历史必须从本地数据包或文件导入"
                        if target == "1min" else f"准备 {target} 行情"
                    ),
                })
        for timeframe in dict.fromkeys(("1day", target)):
            steps.append({
                "kind": "validate", "symbol": config.symbol,
                "timeframe": timeframe,
                "start": str(history_start.date()) if timeframe == "1day" else str(start.date()),
                "end": str(end.date()),
                "reason": (
                    f"核验日线和前置收盘；检查范围从请求起点前 10 个自然日 "
                    f"{history_start.date()} 开始"
                    if timeframe == "1day" else "核验本次请求区间内的包内行情"
                ),
            })
        return {
            "symbol": config.symbol,
            "frequency": config.frequency,
            "asset_type": readiness.get("asset_type"),
            "ready": bool(readiness.get("ready")),
            "checked_start": readiness.get("checked_start"),
            "checked_end": readiness.get("checked_end"),
            "package_id": self.repository.package_id,
            "package_version": int(self.repository.package.manifest().get("package_version", 0)),
            "gaps": readiness.get("gaps", []),
            "history_coverage": history_coverage,
            "warnings": readiness.get("warnings", []),
            "degraded_days": readiness.get("degraded_days", []),
            "steps": steps,
            "config": config.model_dump(mode="json"),
        }

    def execute(
        self, config: BacktestConfig, *,
        progress: Callable[[float, str], None] = lambda *_: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> dict:
        plan = self.plan(config)
        steps = list(plan.get("steps", []))
        blocked = [step for step in steps if step.get("kind") == "import"]
        if blocked:
            raise MarketDataError(blocked[0]["reason"])
        if plan["ready"]:
            warnings = list(plan.get("warnings") or [])
            message = "本次请求已有可用数据包数据"
            if warnings:
                message += "；" + "；".join(warnings)
            return {
                "status": "completed", "mode": "already_ready", "symbol": config.symbol,
                "steps": [], "remaining_gaps": [], "readiness": {"ready": True},
                "package_id": plan["package_id"], "warnings": warnings,
                "message": message,
            }
        results = []
        auxiliary_done = False
        total = max(1, len(steps))
        for index, step in enumerate(steps):
            if cancelled():
                raise InterruptedError("任务已取消")
            progress(0.95 * index / total, f"准备 {step.get('kind')} {step.get('timeframe', '')}")
            try:
                if step["kind"] == "import":
                    raise MarketDataError(step["reason"])
                if step["kind"] == "auxiliary":
                    if auxiliary_done:
                        results.append({**step, "status": "skipped"})
                        continue
                    result = self._refresh_auxiliary(
                        config.symbol, step["start"], step["end"], cancelled
                    )
                    auxiliary_done = True
                elif step["kind"] == "bars":
                    result = MarketDataSyncService(self.service).sync(
                        [config.symbol], step["timeframe"], step.get("mode", "history"),
                        start=step.get("start"), end=step.get("end"), resume=False,
                        workers=1,
                        progress=lambda fraction, message, offset=index: progress(
                            0.95 * (offset + float(fraction)) / total, message
                        ),
                        cancelled=cancelled,
                    )
                elif step["kind"] == "validate":
                    result = self.service.revalidate(
                        config.symbol, step["timeframe"], start=step["start"], end=step["end"],
                        progress=lambda fraction, message, offset=index: progress(
                            0.95 * (offset + float(fraction)) / total, message
                        ),
                        cancelled=cancelled,
                    )
                else:
                    raise MarketDataError("未知的数据准备步骤：" + str(step.get("kind")))
                step_status = "completed"
                reason = step["reason"]
                if step["kind"] == "validate":
                    report = result.get("report") or {}
                    if result.get("status") != "complete":
                        step_status = "partial"
                    reason += (
                        f"；结果：新检查 {int(report.get('checked_days') or 0)} 日，"
                        f"复用 {int(report.get('reused_days') or 0)} 日，"
                        f"待核验 {len(report.get('pending_days') or [])} 日，"
                        f"失败 {len(report.get('failed') or [])} 日"
                    )
                results.append({**step, "reason": reason, "status": step_status, "result": result})
            except InterruptedError:
                raise
            except Exception as exc:
                results.append({**step, "status": "failed", "error": str(exc)})
        progress(0.99, "重新检查本次请求")
        readiness = self.service.readiness(config)
        failed = [row for row in results if row.get("status") == "failed"]
        remaining = readiness.get("gaps", [])
        status = "completed" if not failed and not remaining else "partial" if any(
            row.get("status") in {"completed", "partial", "skipped"} for row in results
        ) else "failed"
        progress(1.0, "数据准备结束")
        if not remaining:
            message = "本次请求已可运行"
        elif status == "partial":
            message = f"准备步骤已完成，但仍有 {len(remaining)} 项数据缺口，暂不可用"
        else:
            message = f"准备失败，仍有 {len(remaining)} 项数据缺口"
        return {
            "status": status, "mode": "preparation", "symbol": config.symbol,
            "frequency": config.frequency, "steps": results,
            "readiness": readiness, "remaining_gaps": remaining,
            "package_id": self.repository.package_id,
            "package_version": int(self.repository.package.manifest().get("package_version", 0)),
            "message": message,
        }

    def _refresh_auxiliary(self, symbol, start, end, cancelled) -> dict:
        calendar = fetch_market_calendar(start, end, cancelled)
        actions, metadata = fetch_market_actions(
            symbol, start, end, cancelled,
            network_timeout=self.service.settings.market_data.network_timeout_seconds,
        )
        if calendar is None or calendar.empty:
            raise MarketDataError("交易日历来源未返回数据")
        # Preparation and ordinary synchronization share event normalization,
        # history merging and content-based versions. A short preparation
        # window must not discard earlier events or reintroduce placeholders.
        from .sync_auxiliary import persist_sync_auxiliary
        with self.repository.package.transaction_guard():
            persist_sync_auxiliary(self.service, symbol, {
                **metadata, "calendar": calendar, "actions": actions,
            })
        if metadata.get("name"):
            self.service.store.put_instrument(
                symbol, metadata["name"],
                **{key: value for key, value in metadata.items() if key not in {"name"}},
            )
        return {
            "calendar_rows": len(calendar),
            "action_rows": len(actions),
            "actions_verified": bool(metadata.get("actions_verified", False)),
            "daily_rows": 0,
        }
