from __future__ import annotations

import multiprocessing
import os
import threading
import time
import traceback

from .data import DataService
from .models import BacktestConfig, OptimizationConfig
from .storage import Store


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
            progress(0.01, "读取锁定的数据版本并进行质量检查")
            bundle = service.load_bundle(config, payload.get("version"))
            result = run_backtest(config, bundle, lambda fraction, message: progress(0.1 + 0.9 * fraction, message), cancelled)
            symbol, title = config.symbol, "股数网格 / 定投 / 买入持有" if config.schema_version == 2 else "网格 / 定投 / 初始全仓"
        elif kind == "optimization":
            from .optimization import optimize
            config = OptimizationConfig.model_validate(payload["config"])
            progress(0.01, "载入锁定版本和时间切分数据")
            bundle = service.load_bundle(config.base, payload.get("version"))
            result = optimize(config, bundle, lambda fraction, message: progress(0.1 + 0.9 * fraction, message), cancelled)
            symbol, title = config.base.symbol, "历史搜索 / 样本外验证"
        elif kind == "sync":
            result = service.sync(payload.get("symbols", []), payload.get("mode", "daily"), progress, cancelled, scope=payload.get("scope"), resume=payload.get("resume", False), retry_failed=payload.get("retry_failed", False), workers=payload.get("workers", 2))
            symbol, title = ",".join(payload.get("symbols", [])), "数据同步"
        elif kind == "demo":
            result = service.demo(payload.get("symbol", "000001.SZ"), payload.get("days", 780), progress, cancelled)
            symbol, title = result["symbol"], "模拟行情演示"
        elif kind == "import":
            result = service.import_files(payload, progress, cancelled)
            symbol, title = payload["symbol"], "本地数据导入"
        elif kind == "instruments":
            result = service.refresh_instruments(progress, cancelled)
            symbol, title = "", "证券清单更新"
        else:
            raise ValueError("未知任务类型：" + kind)
        if cancelled() and kind != "optimization":
            raise InterruptedError("任务已取消")
        result_id = store.save_result(kind, result, symbol, title)
        was_cancelled = cancelled()
        failed_sync = kind == "sync" and result.get("status") == "failed"
        store.update_job(job_id, status="cancelled" if was_cancelled else "failed" if failed_sync else "completed", progress=1, message="已取消，已完成的搜索候选保留" if was_cancelled else result.get("message", "完成"), error="所有同步目标失败，逐标的原因已保存在结果报告，可重试失败项" if failed_sync else None, result_id=result_id)
    except InterruptedError as exc:
        store.update_job(job_id, status="cancelled", message=str(exc))
    except Exception as exc:
        folder = store.root / "logs"
        folder.mkdir(exist_ok=True)
        (folder / f"{job_id}.log").write_text(traceback.format_exc(), encoding="utf-8")
        store.update_job(job_id, status="failed", message="任务失败，查看原因后可重试", error=str(exc))


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
