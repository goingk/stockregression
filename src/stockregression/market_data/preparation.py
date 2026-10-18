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

        if "calendar" in kinds or "actions" in kinds:
            steps.append({
                "kind": "auxiliary",
                "symbol": config.symbol,
                "start": str(history_start.date()),
                "end": str(end.date()),
                "reason": "补齐交易日历和公司行动资料",
            })
        daily_gap = any(
            str(gap.get("timeframe") or "") == "1day"
            or str(gap.get("kind")) in {"daily", "reference_price"}
            for gap in gaps
        )
        if daily_gap:
            steps.append({
                "kind": "bars", "timeframe": "1day", "mode": "history",
                "symbol": config.symbol, "start": str(history_start.date()),
                "end": str(end.date()), "reason": "补齐日线及开始日前置收盘",
            })
        target_gap = any(str(gap.get("timeframe") or "") == target for gap in gaps)
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
                    "end": str(end.date()), "reason": f"准备 {target} 行情",
                })
        steps.append({
            "kind": "validate", "symbol": config.symbol,
            "timeframe": target, "reason": "发布包内行情并重新判断本次请求是否可用",
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
            return {
                "status": "completed", "mode": "already_ready", "symbol": config.symbol,
                "steps": [], "remaining_gaps": [], "readiness": {"ready": True},
                "package_id": plan["package_id"], "message": "本次请求已有可用数据包数据",
            }
        results = []
        auxiliary_done = False
        total = max(1, len(steps))
        for index, step in enumerate(steps):
            if cancelled():
                raise InterruptedError("任务已取消")
            progress(index / total, f"准备 {step.get('kind')} {step.get('timeframe', '')}")
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
                            (offset + float(fraction)) / total, message
                        ),
                        cancelled=cancelled,
                    )
                elif step["kind"] == "validate":
                    result = self.service.revalidate(
                        config.symbol, step["timeframe"]
                    )
                else:
                    raise MarketDataError("未知的数据准备步骤：" + str(step.get("kind")))
                results.append({**step, "status": "completed", "result": result})
            except InterruptedError:
                raise
            except Exception as exc:
                results.append({**step, "status": "failed", "error": str(exc)})
        progress(0.99, "重新检查本次请求")
        readiness = self.service.readiness(config)
        failed = [row for row in results if row.get("status") == "failed"]
        remaining = readiness.get("gaps", [])
        status = "completed" if not failed and not remaining else "partial" if any(
            row.get("status") in {"completed", "skipped"} for row in results
        ) else "failed"
        progress(1.0, "数据准备结束")
        return {
            "status": status, "mode": "preparation", "symbol": config.symbol,
            "frequency": config.frequency, "steps": results,
            "readiness": readiness, "remaining_gaps": remaining,
            "package_id": self.repository.package_id,
            "package_version": int(self.repository.package.manifest().get("package_version", 0)),
            "message": "本次请求已可运行" if not remaining else f"仍有 {len(remaining)} 项缺口",
        }

    def _refresh_auxiliary(self, symbol, start, end, cancelled) -> dict:
        calendar = fetch_market_calendar(start, end, cancelled)
        actions, metadata = fetch_market_actions(symbol, start, end, cancelled)
        if calendar is None or calendar.empty:
            raise MarketDataError("交易日历来源未返回数据")
        self.repository.save_auxiliary(
            symbol, "calendar", calendar.to_dict("records"),
            version="preparation-v1", verified=True,
        )
        self.repository.save_auxiliary(
            symbol, "actions", actions.to_dict("records"),
            version="preparation-v1",
            verified=bool(metadata.get("actions_verified", False)),
        )
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
