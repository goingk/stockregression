from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import threading
import time
import traceback
from pathlib import Path

from .data import DataService
from .models import BacktestConfig, OptimizationConfig
from .market_data import MarketDataService
from .market_data.aggregation import AggregationService
from .market_data.package import import_etf_package, index_etf_package_direct
from .market_data.maintenance import rebuild_state, update_rebuild_state
from .market_data.preparation import MarketPreparationService
from .market_data.sync import MarketDataSyncService
from .research_preparation import prepare_research_bundle
from .source_health import probe_source
from .storage import Store


def _cleanup_uploaded_input(store: Store, job_id: str):
    job = store.job(job_id) or {}
    payload = job.get("payload") or {}
    directories = payload.get("upload_dirs") or ([payload["upload_dir"]] if payload.get("upload_dir") else [])
    allowed = [
        (store.root / "imports").resolve(),
        MarketDataService(store).repository.package.staging_dir.resolve(),
    ]
    for relative in directories:
        path = Path(relative)
        target = path.resolve() if path.is_absolute() else (store.root / path).resolve()
        if any(target.is_relative_to(root) for root in allowed) and target.is_dir():
            shutil.rmtree(target, ignore_errors=True)


def _research_protocol_hash(config: OptimizationConfig) -> str:
    payload = config.model_dump(mode="json")
    if config.protocol_version == 3:
        payload["protocol_thresholds"] = {
            "positive_neighbor_ratio": 0.8,
            "neighbor_quantile": 0.25,
            "cost_multipliers": [1.0, 2.0],
            "neighbor_perturbations": "single_and_pairwise_one_step",
            "funding_mode": "top_up_on_fill",
        }
    elif config.protocol_version == 4:
        payload["protocol_thresholds"] = {
            "strategy": config.strategy.model_dump(mode="json"),
            "grid_mode": "geometric",
            "grid_step_unit": 0.001,
            "share_unit": 100,
            "negative_tolerance": 1e-12,
            "funding_mode": "top_up_on_fill",
            "pressure_is_diagnostic_only": True,
            "validation_account_mode": "fresh_initial_config",
            "validation_return_basis": "peak_invested_roi",
        }
    base = dict(payload.get("base", {}))
    for key in ("symbol", "start_date", "end_date"):
        base.pop(key, None)
    if config.protocol_version == 4:
        for key in ("grid_step", "buy_step", "sell_step", "buy_shares", "sell_shares", "min_shares", "max_shares"):
            base.pop(key, None)
    payload["base"] = base
    if config.protocol_version == 4 and isinstance(payload.get("research_context"), dict):
        context = dict(payload["research_context"])
        for key in ("symbol", "start_date", "end_date"):
            context.pop(key, None)
        payload["research_context"] = context
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validation_intervals(result: dict, start: str, end: str | None) -> list[dict]:
    intervals: list[dict] = [{"start": start, "end": end or start, "kind": "input_evaluation"}]

    def add(item, kind):
        if not isinstance(item, dict):
            return
        left = item.get("start", item.get("start_date"))
        right = item.get("end", item.get("end_date"))
        if left is None or right is None:
            return
        intervals.append({"start": str(left)[:10], "end": str(right)[:10], "kind": kind})

    def visit(node, path=()):
        if isinstance(node, dict):
            if path and path[-1] in {"final_126_day_check", "test_window", "validation_interval"}:
                add(node, ".".join(path))
            for key, value in node.items():
                if key in {"windows", "validation_windows", "evaluation_windows"} and isinstance(value, list):
                    for item in value:
                        add(item, ".".join((*path, key)))
                visit(value, (*path, key))
        elif isinstance(node, list):
            for value in node:
                visit(value, path)

    visit(result.get("evidence", {}), ("evidence",))
    return intervals


def _scientific_failed(result: dict) -> bool:
    if result.get("status") in {"failed", "error"}:
        return True
    branches = [result.get(name, {}) for name in ("fixed", "adaptive") if isinstance(result.get(name), dict)]
    statuses = [branch.get("status") for branch in branches if branch.get("status") is not None]
    return bool(statuses) and all(status in {"failed", "error"} for status in statuses)


def execute_job(root: str, job_id: str, observer=None):
    store = Store(root)
    job = store.job(job_id)
    if not job:
        return
    store.update_job(job_id, status="running", message="正在执行", pid=os.getpid())
    last_progress = 0.0
    last_cancel_check = 0.0
    cancel_state = False

    def cancelled():
        nonlocal last_cancel_check, cancel_state
        tick = time.monotonic()
        if tick - last_cancel_check > 0.3:
            cancel_state = bool((store.job(job_id) or {}).get("cancel_requested"))
            last_cancel_check = tick
        return cancel_state

    def progress(fraction, message):
        nonlocal last_progress
        tick = time.monotonic()
        if tick - last_progress > 0.4 or fraction <= 0 or fraction >= 1:
            store.update_job(job_id, progress=max(0, min(1, float(fraction))), message=str(message))
            if observer:
                observer(fraction, str(message))
            last_progress = tick
    try:
        service = DataService(store)
        payload, kind = job["payload"], job["kind"]
        if cancelled():
            raise InterruptedError("任务已取消")
        if kind == "backtest":
            from .engine import run_backtest
            config = BacktestConfig.model_validate(payload["config"])
            progress(0.01, "检查本地行情并准备回测数据")
            bundle = service.ensure_bundle(config, payload.get("version"), lambda fraction, message: progress(0.02 + 0.30 * fraction, message), cancelled)
            payload["version"] = bundle.version
            store.update_job(job_id, payload=payload)
            progress(0.33, "行情准备完成，开始回测")
            result = run_backtest(config, bundle, lambda fraction, message: progress(0.33 + 0.67 * fraction, message), cancelled)
            symbol, title = config.symbol, "现金补足网格 / 定投 / 买入持有" if config.schema_version == 3 else "股数网格 / 定投 / 买入持有" if config.schema_version == 2 else "网格 / 定投 / 初始全仓"
        elif kind == "optimization":
            from .optimization import optimize
            config = OptimizationConfig.model_validate(payload["config"])
            scientific = config.protocol_version in (2, 3, 4)
            progress(0.01, "检查本地行情并准备研究数据" if not scientific else "准备评估区间及前置 504 个交易日")
            if scientific:
                bundle = prepare_research_bundle(
                    service,
                    config.base,
                    payload.get("version"),
                    lambda fraction, message: progress(0.02 + 0.30 * fraction, message),
                    cancelled,
                )
            else:
                bundle = service.ensure_bundle(config.base, payload.get("version"), lambda fraction, message: progress(0.02 + 0.30 * fraction, message), cancelled)
            payload["version"] = bundle.version
            store.update_job(job_id, payload=payload)
            progress(0.33, "行情准备完成，开始科学参数研究" if scientific else "行情准备完成，开始参数搜索")
            result = optimize(config, bundle, lambda fraction, message: progress(0.33 + 0.67 * fraction, message), cancelled)
            result["diagnostics"] = bundle.metadata.get("diagnostics", {})
            result["source_events"] = bundle.metadata.get("source_events", [])
            symbol = config.base.symbol
            if scientific:
                audit = store.register_research(
                    registry_id=job_id,
                    protocol_hash=_research_protocol_hash(config),
                    symbol=symbol,
                    input_start=str(config.base.start_date),
                    input_end=str(config.base.end_date) if config.base.end_date else str(bundle.calendar.max())[:10] if len(bundle.calendar) else None,
                    data_version=bundle.version,
                    validation_intervals=_validation_intervals(
                        result,
                        str(config.base.start_date),
                        str(config.base.end_date) if config.base.end_date else str(bundle.calendar.max())[:10] if len(bundle.calendar) else None,
                    ),
                )
                result["evidence"] = {**(result.get("evidence") if isinstance(result.get("evidence"), dict) else {}), "local_audit": audit}
                result["protocol_hash"] = audit["protocol_hash"]
                title = ("等比网格多峰研究" if config.protocol_version == 4 else "科学参数研究") + (" · 固定与自适应" if config.selection_target == "both" else f" · {config.selection_target}")
            else:
                title = "历史搜索 / 样本外验证"
        elif kind == "market_import":
            market = MarketDataService(store)
            result = market.import_file(
                payload["symbol"], payload["timeframe"], payload["path"],
                label_convention=payload.get("label_convention", "auto"),
                adjustment=payload.get("adjustment", "none"), etf_category=payload.get("etf_category"),
                calendar_path=payload.get("calendar_path"), actions_path=payload.get("actions_path"),
                skip_invalid_rows=payload.get("skip_invalid_rows", False),
                import_mode=payload.get("import_mode", "merge"),
                progress=progress, cancelled=cancelled,
            )
            symbol, title = payload["symbol"], f"导入 {payload['timeframe']} 行情"
        elif kind == "market_import_batch":
            market = MarketDataService(store)
            result = market.import_folder(
                payload["folder"], payload["timeframe"], recursive=payload.get("recursive", False),
                label_convention=payload.get("label_convention", "auto"),
                adjustment=payload.get("adjustment", "none"), etf_category=payload.get("etf_category"),
                resume=payload.get("resume", True), calendar_path=payload.get("calendar_path"),
                workers=payload.get("workers", 2), skip_invalid_rows=payload.get("skip_invalid_rows", False),
                import_mode=payload.get("import_mode", "merge"),
                progress=progress, cancelled=cancelled,
            )
            symbol, title = "", f"批量导入 {payload['timeframe']}"
        elif kind == "market_import_package":
            market = MarketDataService(store)
            update_rebuild_state(store, stage="importing")
            result = import_etf_package(
                market, payload["folder"], timeframes=payload.get("timeframes"),
                symbols=payload.get("symbols"),
                label_convention=payload.get("label_convention", "auto"),
                resume=payload.get("resume", True), workers=payload.get("workers", 2),
                skip_invalid_rows=payload.get("skip_invalid_rows", False),
                progress=progress, cancelled=cancelled,
            )
            update_rebuild_state(store, stage="finished", result=result)
            symbol, title = "", "ETF 数据包导入"
        elif kind == "market_index_package":
            update_rebuild_state(store, stage="indexing")
            result = index_etf_package_direct(
                MarketDataService(store), payload["folder"],
                timeframes=payload.get("timeframes"), symbols=payload.get("symbols"),
                progress=progress, cancelled=cancelled,
            )
            update_rebuild_state(store, stage="finished", result=result)
            symbol, title = "", "登记 ETF 原始 Parquet 数据包"
        elif kind == "market_sync":
            timeframes = payload.get("timeframes") or [payload["timeframe"]]
            symbols = payload.get("symbols", [])
            scope = payload.get("scope")
            if "1min" in timeframes and payload.get("mode") == "history":
                raise ValueError("1 分钟历史行情只从本地数据包读取；请先登记数据包或导入文件，近期缺口使用缺口修补。")
            if "1min" in timeframes and scope == "market" and payload.get("initialize"):
                raise ValueError("1 分钟行情不支持全市场在线初始化；请登记本地数据包或导入文件。")
            combined = []
            failures = []
            for timeframe in sorted(set(timeframes), key=lambda value: 0 if value == "1day" else 1):
                selected = list(symbols)
                if scope == "market":
                    if payload.get("initialize"):
                        service.refresh_instruments(progress, cancelled)
                        selected = [row["symbol"] for row in store.instruments(limit=None)]
                    else:
                        selected = MarketDataService(store).repository.symbols_for_timeframe(timeframe)
                elif scope == "watchlist":
                    selected = [row["symbol"] for row in store.watches()]
                one = MarketDataSyncService(MarketDataService(store)).sync(
                    selected, timeframe, payload.get("mode", "incremental"),
                    start=payload.get("start_date"), end=payload.get("end_date"),
                    resume=payload.get("resume", False), retry_failed=payload.get("retry_failed", False),
                    workers=payload.get("workers", 2),
                    progress=progress, cancelled=cancelled,
                )
                combined.append(one)
                failures.extend(one.get("failures", []))
            result = {
                "status": "completed" if not failures else "partial",
                "timeframes": sorted(set(timeframes)),
                "runs": combined, "failures": failures,
                "message": f"完成 {len(combined)} 个基础周期更新",
            }
            symbol, title = ",".join(symbols), f"基础行情同步 · {', '.join(sorted(set(timeframes)))}"
        elif kind == "market_source_probe":
            result = probe_source(
                store, payload["source"], payload["capability"],
                payload.get("market", "all"), cancelled,
            )
            symbol, title = "", f"数据源探测 · {payload['source']}"
        elif kind == "market_validate":
            result = MarketDataService(store).revalidate(
                payload["symbol"], payload["timeframe"],
                label_convention=payload.get("label_convention"),
                mode=payload.get("mode", "incremental"),
                start=payload.get("start_date"), end=payload.get("end_date"),
            )
            symbol, title = payload["symbol"], f"校验 {payload['timeframe']} 行情"
        elif kind == "market_archive":
            market = MarketDataService(store)
            values = []
            for symbol in payload.get("symbols", []):
                values.append(market.repository.archive_dataset(
                    symbol, payload["timeframe"], progress=progress, cancelled=cancelled,
                ))
            result = {
                "status": "completed", "timeframe": payload["timeframe"],
                "symbols": values,
            }
            symbol, title = ",".join(payload.get("symbols", [])), "保护历史行情快照"
        elif kind == "market_prepare":
            market = MarketDataService(store)
            config = BacktestConfig.model_validate(payload["config"])
            result = MarketPreparationService(market).execute(
                config, progress=progress, cancelled=cancelled,
            )
            symbol, title = config.symbol, f"准备 {config.frequency} 行情"
        elif kind == "market_aggregate":
            market = MarketDataService(store)
            aggregated = AggregationService(market)
            values = []
            aggregate_failures = []
            for symbol in payload["symbols"]:
                if cancelled():
                    raise InterruptedError("任务已取消")
                try:
                    values.append(aggregated.aggregate(
                        symbol, payload["source_timeframe"], payload["target_timeframe"],
                        start=payload.get("start_date"), end=payload.get("end_date"),
                        progress=progress, cancelled=cancelled,
                    ))
                except InterruptedError:
                    raise
                except Exception as exc:
                    aggregate_failures.append({"symbol": symbol, "reason": str(exc)})
            failures = [*aggregate_failures, *[item for item in values if item.get("status") == "failed"]]
            result = {"status": "completed" if not failures else "partial", "results": values,
                      "targets": len(payload["symbols"]), "completed": len(values) - len(failures),
                      "failures": failures, "message": f"聚合完成 {len(values) - len(failures)} 个标的"}
            symbol, title = ",".join(payload["symbols"]), f"本地聚合 {payload['source_timeframe']} → {payload['target_timeframe']}"
        elif kind == "instruments":
            result = service.refresh_instruments(progress, cancelled)
            symbol, title = "", "证券清单更新"
        else:
            raise ValueError("未知任务类型：" + kind)
        if cancelled() and kind != "optimization":
            raise InterruptedError("任务已取消")
        was_cancelled = cancelled()
        failed_sync = kind in {"market_sync", "market_source_probe", "market_import", "market_import_batch", "market_import_package", "market_index_package", "market_aggregate", "market_prepare", "market_archive"} and result.get("status") == "failed"
        failed_research = kind == "optimization" and payload.get("config", {}).get("protocol_version") in (2, 3, 4) and _scientific_failed(result)
        result_cancelled = kind == "optimization" and result.get("status") == "cancelled"
        cancelled_job = was_cancelled or result_cancelled
        if kind == "optimization" and payload.get("config", {}).get("protocol_version") in (2, 3, 4):
            if failed_research:
                title += " · 失败"
            elif cancelled_job:
                title += " · 已取消"
            elif result.get("status") == "partial":
                title += " · 部分完成"
        result_id = store.save_result(kind, result, symbol, title)
        sync_error = None
        if failed_sync:
            failures = result.get("failures", [])
            sync_error = "; ".join(f"{item.get('symbol', '标的')}: {item.get('reason', '未知错误')}" for item in failures) or result.get("message") or "所有同步目标失败"
        if cancelled_job:
            message = "已取消，已完成的搜索候选保留"
        elif failed_sync:
            message = f"同步失败：{sync_error}"
        elif failed_research:
            message = "科学参数研究失败：" + str(result.get("reason") or result.get("message") or "查看结果中的失败原因")
        elif kind == "optimization" and payload.get("config", {}).get("protocol_version") in (2, 3, 4) and result.get("status") == "partial":
            message = "科学参数研究部分完成；已保留已完成的候选与结果"
        else:
            message = result.get("message", "完成")
        partial = result.get("status") == "partial" and kind != "optimization"
        store.update_job(job_id, status="cancelled" if cancelled_job else "failed" if failed_sync or failed_research else "partial" if partial else "completed", progress=1, message=message, error=sync_error or (result.get("reason") if failed_research else None), result_id=result_id)
    except InterruptedError as exc:
        store.update_job(job_id, status="cancelled", message=str(exc))
    except Exception as exc:
        folder = store.root / "logs"
        folder.mkdir(exist_ok=True)
        (folder / f"{job_id}.log").write_text(traceback.format_exc(), encoding="utf-8")
        store.update_job(job_id, status="failed", message="任务失败，查看原因后可重试", error=str(exc))
    finally:
        _cleanup_uploaded_input(store, job_id)


class JobManager:
    def __init__(self, root: str, workers=2):
        self.root, self.workers = root, workers
        self.stop_event = threading.Event()
        self.active: dict[str, multiprocessing.Process] = {}
        self.thread = threading.Thread(target=self._loop, daemon=True, name="grid-job-dispatcher")

    def start(self):
        # Previously interrupted application jobs are reviewable and retryable.
        store = Store(self.root)
        for job in store.jobs(1000):
            if job["status"] == "running" and (not job.get("pid") or not _pid_alive(job["pid"])):
                store.update_job(job["id"], status="failed", error="应用或计算进程中断；原数据版本保留，请重新运行", message="执行中断")
        self.thread.start()

    def _loop(self):
        context = multiprocessing.get_context("spawn")
        store = Store(self.root)
        while not self.stop_event.wait(0.5):
            for job_id, process in list(self.active.items()):
                if not process.is_alive():
                    process.join()
                    self.active.pop(job_id)
                    job = store.job(job_id)
                    if job and job["status"] == "running":
                        store.update_job(job_id, status="failed", error="计算进程异常退出", message="进程中断")
            if len(self.active) >= self.workers:
                continue
            queued = [job for job in reversed(store.jobs(1000)) if job["status"] == "queued"]
            for job in queued[:self.workers - len(self.active)]:
                if job["cancel_requested"]:
                    store.update_job(job["id"], status="cancelled", message="排队期间已取消")
                    continue
                # Claim before spawn so two dispatcher ticks cannot duplicate a job.
                store.update_job(job["id"], status="running", message="正在启动计算进程")
                process = context.Process(target=execute_job, args=(self.root, job["id"]), daemon=True)
                try:
                    process.start()
                    store.update_job(job["id"], pid=process.pid)
                    self.active[job["id"]] = process
                except Exception as exc:
                    store.update_job(job["id"], status="failed", error="无法启动计算进程：" + str(exc), message="进程启动失败")

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        store = Store(self.root)
        for job_id, process in self.active.items():
            store.update_job(job_id, cancel_requested=1)
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                store.update_job(job_id, status="cancelled", message="应用已退出；计算任务已停止")


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
