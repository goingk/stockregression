from __future__ import annotations

import hashlib
import io
import multiprocessing
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import (
    FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait,
)
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager, redirect_stdout

import pandas as pd

from ..models import normalize_symbol
from ..performance import SharedSlots, effective_cpu_budget, record_count, record_details, record_seconds
from ..providers import (
    _baostock_reset, _baostock_session, fetch_sina, fetch_tdx, fetch_ths,
    latest_closed_date, serialized_baostock,
)
from ..settings import Settings
from ..source_health import SourceFailure, SourceRouter, classify_source_error, fetch_auxiliary_routed, market_for_symbol
from ..storage import Store
from ..rules import instrument_terms
from ..quality import minute_slots
from .models import MarketDataError, MarketSyncCancelled
from .quality import validate_and_fill
from .reader import normalize_frame
from .service import MarketDataService
from .repository import VALIDATION_RULES_VERSION
from .timeframes import MINUTE_TIMEFRAMES, normalize_timeframe, timeframe_spec


SOURCE_VERSION = "market-sync-v2-package"
DAILY_BATCH_SIZE = 128
_DAILY_WORKER_CONTEXT = None


def _initialize_daily_worker(root, package_dir, settings_json, cutoff_date):
    global _DAILY_WORKER_CONTEXT
    configured = Settings.model_validate_json(settings_json)
    store = Store(root, settings=configured)
    service = MarketDataService(store, package_dir=package_dir, settings=configured, read_only=True)
    _DAILY_WORKER_CONTEXT = MarketDataSyncService(service, cutoff_date=cutoff_date)


def _longest_true_run(mask) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _group_stamps_by_day(stamps: pd.DatetimeIndex) -> dict:
    """Group timestamps by their trading day in a single pass.

    Filtering the whole frame once per day costs one full scan per day;
    the grouped view keeps the same membership for every session.
    """
    if not len(stamps):
        return {}
    return {
        pd.Timestamp(day).normalize(): pd.DatetimeIndex(part.to_numpy())
        for day, part in pd.Series(stamps).groupby(stamps.normalize())
    }


def _recent_minute_incomplete(frame, symbol, start, end) -> str | None:
    """Describe a materially incomplete response from a recent-minute source.

    A provider can succeed at the HTTP level while returning only a tail slice
    of an otherwise available session. That is the failure mode that left
    recent gaps undetected, so assess only sessions the response actually
    starts returning; wholly absent earlier days remain a separate coverage
    gap rather than a reason to discard the usable tail.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "未返回行情"
    stamps = pd.DatetimeIndex(pd.to_datetime(frame.get("timestamp"), errors="coerce").dropna())
    if not len(stamps):
        return "时间字段无效"
    # Group the response by session once instead of filtering the whole
    # frame again for every trading day.
    observed_by_day = _group_stamps_by_day(stamps)
    for day in sorted(observed_by_day):
        day = pd.Timestamp(day).normalize()
        expected = minute_slots(day)
        expected = expected[(expected >= pd.Timestamp(start)) & (expected <= pd.Timestamp(end))]
        if not len(expected):
            continue
        observed = observed_by_day[day]
        missing = expected.difference(observed)
        if not len(missing):
            continue
        if symbol.endswith(".SZ") and day >= pd.Timestamp("2023-02-17"):
            placeholders = pd.DatetimeIndex([
                day + pd.Timedelta(hours=14, minutes=58),
                day + pd.Timedelta(hours=14, minutes=59),
            ])
            anchors = pd.DatetimeIndex([
                day + pd.Timedelta(hours=14, minutes=57),
                day + pd.Timedelta(hours=15),
            ])
            if missing.equals(placeholders) and anchors.isin(observed).all():
                continue
        longest = _longest_true_run(expected.isin(missing))
        if len(missing) > 5 or longest > 3 or missing[0] == expected[0] or missing[-1] == expected[-1]:
            return f"{day.date()} 缺 {len(missing)} 根，最长连续 {longest} 根"
    return None


def _prepare_daily_worker(
    root: str,
    package_dir: str | None,
    symbol: str,
    mode: str,
    start,
    end,
    settings_json: str | None = None,
    cutoff_date: str | None = None,
):
    global _DAILY_WORKER_CONTEXT
    if _DAILY_WORKER_CONTEXT is None:
        configured = Settings.model_validate_json(settings_json) if settings_json else Settings()
        _initialize_daily_worker(root, package_dir, configured.model_dump_json(), cutoff_date)
    return _DAILY_WORKER_CONTEXT._prepare_daily(
        symbol, mode, start, end, lambda: False
    )


def _terminate_executor(executor) -> None:
    if isinstance(executor, ProcessPoolExecutor):
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            try:
                process.terminate()
                process.join(timeout=1)
            except Exception:
                pass
    executor.shutdown(wait=False, cancel_futures=True)


class MarketDataSyncService:
    def __init__(self, service: MarketDataService, *, job_id=None, job_claim_token=None,
                 detail=None, cutoff_date=None):
        self.service = service
        self.store = service.store
        self.repository = service.repository
        self.job_id = job_id
        self.job_claim_token = job_claim_token
        self.detail = detail or (lambda _: None)
        self.cutoff_date = str(pd.Timestamp(cutoff_date if cutoff_date is not None else latest_closed_date()).date())
        self._attempts = {}
        self._attempt_versions = {}
        self._target_ranges = {}
        self._downloaded_symbols = set()
        self._resume = False
        self._run_key = None
        settings = service.settings
        configured = settings.performance
        self.market_settings = settings.market_data
        self.daily_batch_size = int(self.market_settings.daily_batch_size)
        self.scheduler_poll_seconds = float(self.market_settings.scheduler_poll_seconds)
        self.download_capacity = min(
            int(configured.download_workers),
            effective_cpu_budget(configured),
        )
        record_details(download_workers=self.download_capacity)

    def _cutoff(self):
        return pd.Timestamp(self.cutoff_date or latest_closed_date()).normalize()

    def _claim(self, run_key, symbol, mode, start, end, timeframe="1day"):
        with self.repository.package.transaction_guard():
            with self.repository._connect() as db:
                row = self.repository._dataset_row(db, symbol, timeframe, "none")
            self._attempt_versions[symbol] = int(row["current_version"]) if row else 0
            target_start = start
            target_end = min(pd.Timestamp(end), self._cutoff()) if end else self._cutoff()
            if mode == "incremental" and row:
                target_start, target_end = self._increment_range(dict(row), timeframe)
            self._target_ranges[symbol] = (target_start, target_end)
            if not self.job_id:
                self.store.sync_item(run_key, symbol, status="running", mode=mode, error=None)
                return True
            claimed = self.store.claim_sync_item(
                run_key, symbol, job_id=self.job_id, job_claim_token=self.job_claim_token,
                attempt_token=uuid.uuid4().hex, mode=mode,
                target_start=str(target_start) if target_start is not None else None,
                target_end=str(target_end) if target_end is not None else None,
            )
            if claimed:
                self._attempts[symbol] = claimed["attempt_token"]
            return bool(claimed)

    def _assert_attempt(self, symbol):
        if self.job_id and not self.store.sync_attempt_current(
            self._run_key, symbol, job_id=self.job_id, job_claim_token=self.job_claim_token,
            attempt_token=self._attempts.get(symbol),
        ):
            raise InterruptedError("任务已取消或同步检查点所有权已失效")

    @contextmanager
    def _publication_guard(self, symbol):
        with self.repository.package.transaction_guard():
            with self.repository.package.publication_fence(lambda: self._assert_attempt(symbol)):
                yield

    def _finish(self, run_key, symbol, **values):
        if not self.job_id:
            self.store.sync_item(run_key, symbol, **values)
            return
        if not self.store.finish_sync_item(
            run_key, symbol, job_id=self.job_id, job_claim_token=self.job_claim_token,
            attempt_token=self._attempts[symbol], **values,
        ):
            raise InterruptedError("任务已取消或同步检查点所有权已失效")

    def _failure(self, symbol, exc, start=None, end=None, stage=None):
        start, end = self._target_ranges.get(symbol, (start, end))
        category = classify_source_error(exc)
        if isinstance(exc, sqlite3.Error) or (
            isinstance(exc, OSError) and category not in {"network", "timeout"}
            and stage in {"merge", "history_validation", "ledger_publication", "snapshot"}
        ):
            category = "storage"
        route = getattr(exc, "etf_daily_route", {}) or {}
        return {
            "symbol": symbol, "reason": str(exc), "stage": stage,
            "category": "empty_unresolved" if category == "empty" else category,
            "requested_start": str(pd.Timestamp(start).date()) if start is not None else None,
            "requested_end": str(pd.Timestamp(end).date()) if end is not None else str(self._cutoff().date()),
            "source_attempts": getattr(exc, "source_attempts", []),
            "product_type": getattr(exc, "product_type", None) or route.get("product_type"),
            "fallback_status": getattr(exc, "fallback_status", None) or route.get("fallback_status"),
            "fallback_reason": getattr(exc, "fallback_reason", None) or route.get("fallback_reason"),
            "source_route": route,
        }

    def _daily_checkpoint_action(self, symbol, state, start=None, end=None):
        if not self._resume or not state or state.get("status") not in {"completed", "running"}:
            return "download"
        dataset = self.repository.dataset(symbol, "1day")
        if not dataset or dataset.get("file_status") not in {None, "readable"}:
            return "download"
        target_end = min(pd.Timestamp(end).normalize(), self._cutoff()) if end else self._cutoff()
        if pd.Timestamp(dataset["last_time"]).normalize() < target_end:
            return "download"
        from .validation import validation_targets
        selection = validation_targets(self.service, symbol, "1day", start=start or dataset["first_time"], end=target_end)
        return "validate" if selection["targets"] else "reuse"

    def _local_daily_validation(self, symbol, progress, cancelled):
        from .validation import validate_incremental
        dataset = self.repository.dataset(symbol, "1day")
        started = time.perf_counter()
        ledger = validate_incremental(
            self.service, symbol, "1day", mode="incremental",
            start=dataset["first_time"], end=self._cutoff(),
            progress=progress, cancelled=cancelled,
        )
        validation_seconds = max(0.0, time.perf_counter() - started - float(ledger.get("ledger_publication_seconds") or 0))
        record_seconds("history_validation", validation_seconds)
        snapshot_started = time.perf_counter()
        if ledger["checked_days"]:
            self.service.refresh_snapshot(symbol)
        snapshot_seconds = time.perf_counter() - snapshot_started
        record_seconds("snapshot", snapshot_seconds)
        return {
            "symbol": symbol, "timeframe": "1day", "version": dataset["current_version"],
            "last_time": dataset["last_time"], "quality_status": ledger["status"],
            "quality_revision": ledger.get("archive_revision"), "local_validation": True,
            "metrics": {"validated_days": ledger["checked_days"], "reused_days": ledger["reused_days"],
                        "history_validation_seconds": validation_seconds, "snapshot_seconds": snapshot_seconds,
                        "ledger_publication_seconds": float(ledger.get("ledger_publication_seconds") or 0)},
        }

    def sync(
        self, symbols: list[str], timeframe: str, mode: str, *, start=None, end=None,
        resume=False, retry_failed=False, workers=4, progress=lambda *_: None, cancelled=lambda: False,
    ) -> dict:
        timeframe = normalize_timeframe(timeframe)
        if timeframe not in {"1day", "1min"}:
            raise MarketDataError("只有日线和 1 分钟可以从线上更新；其他周期由本地基础行情按需聚合")
        if mode == "daily":
            mode = "incremental"
        if mode not in {"history", "incremental", "repair"}:
            raise MarketDataError("同步模式应为 history/incremental/repair")
        if not symbols:
            raise MarketDataError("同步必须指定标的")
        symbols = sorted({normalize_symbol(symbol) for symbol in symbols})
        checkpoint_scope = str(self._cutoff().date()) if mode == "incremental" else f"{start or ''}:{end or ''}"
        run_key = (
            f"market-sync:{self.repository.package_id}:{timeframe}:{mode}:"
            f"{checkpoint_scope}:{SOURCE_VERSION}"
        )
        if timeframe == "1day":
            return self._sync_daily_batches(
                symbols, mode, start=start, end=end, resume=resume,
                retry_failed=retry_failed, workers=workers,
                progress=progress, cancelled=cancelled, run_key=run_key,
            )
        self._run_key, self._resume = run_key, resume
        missing_initialized = []
        if mode == "incremental":
            ready = []
            for symbol in symbols:
                if self.repository.dataset(symbol, timeframe):
                    ready.append(symbol)
                else:
                    missing_initialized.append(symbol)
            symbols = ready
        results, failures, skipped = [], [], []
        skipped.extend({
            "symbol": symbol, "timeframe": timeframe, "skipped": True,
            "message": "尚无该周期初始化数据，增量更新已跳过",
        } for symbol in missing_initialized)
        if not symbols:
            return {
                "status": "failed", "mode": mode, "timeframe": timeframe,
                "symbols": [], "skipped": skipped,
                "failures": [{
                    "symbol": item["symbol"],
                    "reason": "尚无该周期初始化数据；增量同步不会自动扩大历史范围",
                } for item in skipped],
                "message": f"{len(skipped)} 个标的尚无初始化数据",
            }
        result_lock = threading.Lock()
        completed = 0

        def one(symbol):
            state = self.store.sync_item(run_key, symbol)
            if retry_failed and (not state or state.get("status") != "failed"):
                return "skipped", {"symbol": symbol, "timeframe": timeframe, "skipped": True,
                                   "message": "不属于失败重试范围"}
            if cancelled():
                raise InterruptedError("任务已取消")
            if not self._claim(run_key, symbol, mode, start, end, timeframe):
                raise InterruptedError("同步检查点由其他活动任务持有")
            try:
                result = self._one(symbol, timeframe, mode, start, end, cancelled)
                with self._publication_guard(symbol):
                    current = self.repository.dataset(symbol, timeframe)
                    actual_through = current.get("last_time") if current else None
                    self._finish(
                        run_key, symbol, status="completed", mode=mode, error=None, error_category=None,
                        daily_through=actual_through if timeframe == "1day" else None,
                        minute_through=actual_through if timeframe in MINUTE_TIMEFRAMES else None,
                        data_version=result.get("version"), quality_revision=result.get("quality_revision"),
                        quality_status=result.get("quality_status"))
                return ("skipped" if result.get("skipped") else "result"), result
            except InterruptedError:
                raise
            except Exception as exc:
                value = self._failure(symbol, exc, start, end)
                with self._publication_guard(symbol):
                    self._finish(run_key, symbol, status="failed", mode=mode, error=str(exc),
                                 error_category=value["category"], details=value)
                return "failure", value

        cursor = 0
        worker_count = self._worker_count(workers)
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                while cursor < len(symbols):
                    if cancelled():
                        raise InterruptedError("任务已取消")
                    batch = symbols[cursor:cursor + worker_count * 2]
                    cursor += len(batch)
                    futures = {executor.submit(one, symbol): symbol for symbol in batch}
                    for future in as_completed(futures):
                        category, value = future.result()
                        with result_lock:
                            if category == "result":
                                results.append(value)
                            elif category == "skipped":
                                skipped.append(value)
                            else:
                                failures.append(value)
                            completed += 1
                            progress(completed / len(symbols), f"{completed}/{len(symbols)} · {value.get('symbol')} {timeframe}")
        except InterruptedError as exc:
            settled_symbols = {item["symbol"] for item in results + failures + skipped}
            raise MarketSyncCancelled({"status": "cancelled", "mode": mode, "timeframe": timeframe,
                                       "symbols": results, "failures": failures, "skipped": skipped,
                                       "unprocessed": [{"symbol": symbol} for symbol in symbols if symbol not in settled_symbols],
                                       "run_key": run_key, "cutoff_date": str(self._cutoff().date())}) from exc
        noncompliant = any(x.get("quality_status") in {"partial", "failed"} for x in results + skipped)
        status = "partial" if noncompliant or (failures and results) else "failed" if failures else "completed"
        return {"status": status, "mode": mode, "timeframe": timeframe,
                "symbols": results, "failures": failures, "skipped": skipped,
                "message": f"完成 {len(results)} 个，跳过 {len(skipped)} 个，失败 {len(failures)} 个"}

    def _sync_daily_batches(
        self, symbols, mode, *, start, end, resume, retry_failed, workers,
        progress, cancelled, run_key,
    ) -> dict:
        self._run_key, self._resume = run_key, resume
        results, failures, skipped = [], [], []
        total = len(symbols)
        downloaded = written = 0
        settled, prepared_symbols, quality_started = set(), set(), set()
        worker_count = self._worker_count(workers)
        process_pool_healthy = worker_count >= 4
        executor = None
        using_processes = False
        interrupted = False
        timings = {"download": 0.0, "merge": 0.0, "history_validation": 0.0,
                   "ledger_publication": 0.0, "snapshot": 0.0}
        active_detail = {}
        resources = {"download_workers": min(2, worker_count) if process_pool_healthy else worker_count,
                     "baostock_sessions": min(2, worker_count), "write_workers": 1}

        def counts():
            return {"total": total, "processed": len(settled), "succeeded": len(results),
                    "failed": len(failures), "skipped": len(skipped),
                    "reused": sum(bool(x.get("checkpoint_reused")) for x in skipped),
                    "unprocessed": total - len(settled)}

        def report(message, *, stage=None, symbol=None, fraction=0):
            if stage:
                active_detail.clear()
                active_detail.update(stage=stage, symbol=symbol)
            # Include the actual day counters from long running validation phases.
            match = re.search(r"(\d+)/(\d+) 日", message)
            if match:
                active_detail.update(checked_days=int(match[1]), total_days=int(match[2]))
            self.detail({**active_detail, "counts": counts(), "resources": resources,
                         "cutoff_date": str(self._cutoff().date())})
            progress(min(1.0, (downloaded + written + fraction) / (2 * max(1, total))), message)

        def checked_cancel(symbol):
            if cancelled():
                return True
            self._assert_attempt(symbol)
            return False

        def finish(symbol, value):
            with self._publication_guard(symbol):
                self._finish(run_key, symbol, status="completed", mode=mode, error=None,
                             error_category=None, daily_through=value.get("last_time"),
                             data_version=value.get("version"),
                             quality_revision=value.get("quality_revision"),
                             quality_status=value.get("quality_status"),
                             details={"source_attempts": value.get("source_attempts", []),
                                      "operation_complete": True,
                                      "quality_compliant": value.get("quality_status") in {"complete", "ready"}})

        def fail(symbol, exc):
            item = self._failure(symbol, exc, start, end, stage=active_detail.get("stage"))
            with self._publication_guard(symbol):
                self._finish(run_key, symbol, status="failed", mode=mode,
                             error=str(exc), error_category=item["category"], details=item)
            failures.append(item)
            settled.add(symbol)

        def partial_report(status):
            completed_symbols = {x["symbol"] for x in results + skipped}
            failed_symbols = {x["symbol"] for x in failures}
            published_pending, downloaded_pending, unprocessed = [], [], []
            if status == "cancelled":
                for symbol in symbols:
                    if symbol in completed_symbols or symbol in failed_symbols:
                        continue
                    # Read only: cancellation cannot mutate a checkpoint or package.
                    with self.repository._connect() as db:
                        row = self.repository._dataset_row(db, symbol, "1day", "none")
                    head = self.repository.package.manifest().get("datasets", {}).get(
                        self.repository.package.dataset_key(symbol, "1day", "none"), {})
                    version = int(head.get("version") or (row["current_version"] if row else 0))
                    if (symbol in self._attempt_versions and version > self._attempt_versions[symbol]) or (version and symbol in quality_started):
                        published_pending.append({"symbol": symbol, "version": version,
                                                  "state": "market_published_quality_pending"})
                    elif symbol in prepared_symbols:
                        downloaded_pending.append({"symbol": symbol, "state": "downloaded_uncommitted"})
                    else:
                        unprocessed.append({"symbol": symbol, "state": "unprocessed"})
            categories = {}
            for failure in failures:
                key = failure.get("category", "unknown")
                categories[key] = categories.get(key, 0) + 1
            return {"status": status, "mode": mode, "timeframe": "1day", "run_key": run_key,
                    "cutoff_date": str(self._cutoff().date()), "symbols": results,
                    "failures": failures, "skipped": skipped, "counts": counts(),
                    "failure_categories": categories, "published_quality_pending": published_pending,
                    "downloaded_uncommitted": downloaded_pending, "unprocessed": unprocessed,
                    "resources": resources, "timings": timings,
                    "batches": (total + self.daily_batch_size - 1) // self.daily_batch_size,
                    "batch_size": self.daily_batch_size,
                    "message": f"完成 {len(results)} 个，复用/跳过 {len(skipped)} 个，失败 {len(failures)} 个，未处理 {total-len(settled)} 个"}

        def local(symbol, action):
            nonlocal downloaded, written
            downloaded += 1
            report(f"{symbol} · 核对本地质量账本", stage="history_validation", symbol=symbol)
            with self._publication_guard(symbol):
                if checked_cancel(symbol):
                    raise InterruptedError("任务已取消")
                quality_started.add(symbol)
                value = self._local_daily_validation(
                    symbol, lambda f, m: report(m, fraction=f), lambda: checked_cancel(symbol))
                finish(symbol, value)
            if action == "reuse":
                skipped.append({**value, "skipped": True, "checkpoint_reused": True,
                                "message": "数据版本、质量依赖与目标日期已核对"})
            else:
                results.append(value)
            for key in timings:
                if key != "download":
                    timings[key] += (value.get("metrics") or {}).get(key + "_seconds", 0)
            settled.add(symbol)
            written += 1
            report(f"{symbol} · 本地校验完成", stage="completed", symbol=symbol)

        try:
            for batch_start in range(0, total, self.daily_batch_size):
                if cancelled():
                    raise InterruptedError("任务已取消")
                active = []
                for symbol in symbols[batch_start:batch_start + self.daily_batch_size]:
                    state = self.store.sync_item(run_key, symbol)
                    if retry_failed and (not state or state.get("status") != "failed"):
                        skipped.append({"symbol": symbol, "timeframe": "1day", "skipped": True,
                                        "message": "不属于失败重试范围"})
                        downloaded += 1
                        written += 1
                        settled.add(symbol)
                        continue
                    if not self._claim(run_key, symbol, mode, start, end):
                        raise InterruptedError("同步检查点正由其他活动任务持有，请在其结束后续跑")
                    try:
                        action = self._daily_checkpoint_action(symbol, state, start, end)
                        if action != "download":
                            local(symbol, action)
                            continue
                        # Repair planning may publish local validation, so it belongs in the parent.
                        if mode in {"history", "repair"} and self.repository.dataset(symbol, "1day") and self.repository.auxiliary(symbol, "calendar"):
                            with self._publication_guard(symbol):
                                self._repair_windows(symbol, "1day", pd.Timestamp(start),
                                                     min(pd.Timestamp(end), self._cutoff()) if end else self._cutoff(),
                                                     cancelled=lambda: checked_cancel(symbol))
                        active.append(symbol)
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        downloaded += 1
                        written += 1
                        fail(symbol, exc)
                if not active:
                    report(f"{len(settled)}/{total} · 日线检查点已核对", stage="checkpoint")
                    continue
                if executor is None:
                    using_processes = len(active) > 1 and process_pool_healthy
                    if using_processes:
                        executor = ProcessPoolExecutor(
                            max_workers=min(2, worker_count), mp_context=multiprocessing.get_context("spawn"),
                            initializer=_initialize_daily_worker,
                            initargs=(str(self.store.root), str(self.repository.package_dir),
                                      self.service.settings.model_dump_json(), self.cutoff_date))
                    else:
                        executor = ThreadPoolExecutor(max_workers=min(2, worker_count))
                        resources["download_workers"] = min(2, worker_count)
                prepared = []
                download_started = time.perf_counter()
                futures = {
                    (executor.submit(_prepare_daily_worker, str(self.store.root), str(self.repository.package_dir),
                                     symbol, mode, start, end, self.service.settings.model_dump_json(), self.cutoff_date)
                     if using_processes else executor.submit(self._prepare_daily, symbol, mode, start, end, cancelled)): symbol
                    for symbol in active
                }
                pending = set(futures)
                while pending:
                    if cancelled():
                        raise InterruptedError("任务已取消")
                    done, pending = wait(pending, timeout=self.scheduler_poll_seconds, return_when=FIRST_COMPLETED)
                    for future in done:
                        symbol = futures[future]
                        downloaded += 1
                        try:
                            try:
                                value = future.result()
                            except BrokenProcessPool:
                                if process_pool_healthy:
                                    _terminate_executor(executor)
                                    executor = ThreadPoolExecutor(max_workers=min(2, worker_count))
                                process_pool_healthy = False
                                value = executor.submit(
                                    self._prepare_daily, symbol, mode, start, end, cancelled,
                                ).result()
                            if value.get("skipped"):
                                # A range skip is valid only after local archive validation.
                                if self.repository.dataset(symbol, "1day"):
                                    with self._publication_guard(symbol):
                                        quality = self._local_daily_validation(symbol, lambda f, m: report(m, fraction=f),
                                                                               lambda: checked_cancel(symbol))
                                        value.update(quality)
                                finish(symbol, value)
                                skipped.append(value)
                                settled.add(symbol)
                                written += 1
                            else:
                                prepared.append(value)
                                prepared_symbols.add(symbol)
                        except InterruptedError:
                            raise
                        except Exception as exc:
                            written += 1
                            fail(symbol, exc)
                        report(f"{downloaded}/{total} · {symbol} 日线下载", stage="download", symbol=symbol)
                elapsed = time.perf_counter() - download_started
                timings["download"] += elapsed
                record_seconds("download", elapsed)
                if not process_pool_healthy and using_processes:
                    using_processes = False
                    resources["download_workers"] = min(2, worker_count)
                for source in prepared:
                    symbol = source["symbol"]
                    try:
                        report(f"{symbol} · 合并行情", stage="merge", symbol=symbol)
                        with self._publication_guard(symbol):
                            if checked_cancel(symbol):
                                raise InterruptedError("任务已取消")
                            auxiliary_status = self._persist_auxiliary(symbol, source.get("auxiliary") or {})
                            source.setdefault("metadata", {}).update(auxiliary_status)
                            quality_started.add(symbol)
                            result = self.service._commit_frame(
                                symbol, "1day", source["frame"], fingerprint=source.get("fingerprint"),
                                file_name=f"sync:{source.get('source')}:{mode}", params_hash=source.get("params_hash"),
                                pending=list(source.get("pending") or []), source=source.get("source") or "sync",
                                adjustment="none", etf_category=None, calendar_path=None, actions_path=None,
                                progress=lambda f, m: report(m, stage="history_validation" if "日" in m and "/" in m else None,
                                                             symbol=symbol, fraction=f),
                                cancelled=lambda: checked_cancel(symbol), source_metadata=source.get("metadata") or {},
                                coverage_start=source.get("coverage_start"), coverage_end=source.get("coverage_end"),
                                validation_start=source.get("request_start"), validation_end=source.get("request_end"),
                                import_mode="merge")
                            result.update(mode=mode, requested_start=str(source["request_start"].date()),
                                          requested_end=str(source["request_end"].date()), source=source["source"],
                                          source_attempts=(source.get("metadata") or {}).get("source_attempts", []))
                            finish(symbol, result)
                        results.append(result)
                        for key in timings:
                            if key != "download":
                                timings[key] += (result.get("metrics") or {}).get(key + "_seconds", 0)
                        settled.add(symbol)
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        fail(symbol, exc)
                    written += 1
                    report(f"{len(settled)}/{total} · {symbol} 日线写入", stage="completed", symbol=symbol)
            noncompliant = any(x.get("quality_status") in {"partial", "failed"} for x in results + skipped)
            status = "partial" if noncompliant or (failures and (results or skipped)) else "failed" if failures else "completed"
            return partial_report(status)
        except InterruptedError as exc:
            interrupted = True
            raise MarketSyncCancelled(partial_report("cancelled"), str(exc)) from exc
        finally:
            if executor is not None:
                if interrupted or cancelled():
                    _terminate_executor(executor)
                else:
                    executor.shutdown(wait=True, cancel_futures=True)

    def _persist_auxiliary(self, symbol, metadata):
        from .sync_auxiliary import persist_sync_auxiliary
        return persist_sync_auxiliary(self.service, symbol, metadata)

    def _prepare_daily(self, symbol, mode, start, end, cancelled):
        dataset = self.repository.dataset(symbol, "1day")
        repair_windows = None
        if mode == "incremental":
            if not dataset:
                raise MarketDataError("该周期尚无初始化数据；增量同步不会自动执行全量历史导入，请先明确选择历史区间")
            request_start, request_end = self._increment_range(dataset, "1day")
            if request_start is None:
                return {"symbol": symbol, "timeframe": "1day", "skipped": True,
                        "message": "已更新至最新已结束周期"}
        else:
            if start is None:
                raise MarketDataError("历史导入或修补必须明确指定开始日期")
            request_start = pd.Timestamp(start).normalize()
            closed = self._cutoff()
            request_end = min(pd.Timestamp(end).normalize(), closed) if end else closed
            if mode == "repair":
                repair_windows = self._repair_windows(symbol, "1day", request_start, request_end, publish_validation=False)
                if not repair_windows:
                    return {"symbol": symbol, "timeframe": "1day", "skipped": True,
                            "message": "所选区间未发现需要修补的缺口",
                            "metrics": {"archive_hit": True, "checked_days": 0,
                                        "reused_days": 0, "scanned_shards": 0}}
            elif dataset and self.repository.auxiliary(symbol, "calendar"):
                repair_windows = self._repair_windows(symbol, "1day", request_start, request_end, publish_validation=False)
                if not repair_windows:
                    return {"symbol": symbol, "timeframe": "1day", "skipped": True,
                            "message": "所选区间已有完整本地数据，未调用行情来源",
                            "metrics": {"archive_hit": True, "checked_days": 0,
                                        "reused_days": 0, "scanned_shards": 0}}
        if request_start > request_end:
            raise MarketDataError("开始日期不能晚于结束日期")
        frames, source, metadata = [], "", {}
        for window_start, window_end in (repair_windows or [(request_start, request_end)]):
            if cancelled():
                raise InterruptedError("任务已取消")
            fetched, current_source, current_metadata = self._fetch(
                symbol, "1day", mode, window_start, window_end, cancelled
            )
            if not fetched.empty:
                frames.append(fetched)
            source = current_source or source
            metadata.update(current_metadata)
        if not frames:
            raise MarketDataError("来源没有返回行情；覆盖水位未推进")
        raw = pd.concat(frames, ignore_index=True)
        normalized, pending = normalize_frame(
            raw, "1day", label_convention=metadata.pop("label_convention", "auto"), source=source,
            skip_invalid_rows=True,
        )
        normalized = normalized[
            (normalized.timestamp >= request_start)
            & (normalized.timestamp <= request_end + pd.Timedelta(days=1))
        ]
        if normalized.empty:
            raise MarketDataError("来源返回的行情不在请求区间内")
        validation = validate_and_fill(
            normalized, "1day", calendar=(metadata.get("calendar") if metadata.get("calendar") is not None else self.service._calendar(symbol)), pending=pending,
            coverage_start=normalized.timestamp.min(),
            coverage_end=normalized.timestamp.max(), symbol=symbol,
        )
        instrument = self.store.instrument(symbol) or instrument_terms(symbol, {"name": symbol})
        report = {
            **validation.report, "pending_validation": validation.pending,
            "reason": "; ".join(validation.pending) or None,
        }
        item_metadata = {
            **instrument, **{key: value for key, value in metadata.items() if key not in {"calendar", "actions"}},
            "is_demo": False, "imported": True, "validation": validation.report,
        }
        item_metadata = instrument_terms(symbol, item_metadata)
        fingerprint = hashlib.sha256(
            f"{symbol}|1day|{source}|{normalized.timestamp.min()}|{normalized.timestamp.max()}|{len(normalized)}".encode()
        ).hexdigest()
        return {
            "symbol": symbol, "timeframe": "1day", "frame": validation.frame,
            "pending": validation.pending, "source": source, "metadata": item_metadata,
            "auxiliary": {key: value for key, value in metadata.items()
                          if key in {"calendar", "actions"} or key.startswith("actions_")},
            "report": report, "fingerprint": fingerprint, "request_start": request_start,
            "request_end": request_end, "coverage_start": normalized.timestamp.min(),
            "coverage_end": normalized.timestamp.max(),
            "params_hash": hashlib.sha256(f"{SOURCE_VERSION}|{mode}|{source}".encode()).hexdigest(),
        }

    def _one(self, symbol, timeframe, mode, start, end, cancelled):
        if timeframe == "1min" and mode == "history":
            raise MarketDataError(
                "1 分钟历史行情只从本地数据包读取；请先登记数据包或导入文件，近期缺口使用“缺口修补”。"
            )
        dataset = self.repository.dataset(symbol, timeframe)
        repair_windows = None
        if mode == "incremental":
            if not dataset:
                raise MarketDataError("该周期尚无初始化数据；增量同步不会自动执行全量历史导入，请先明确选择历史区间")
            request_start, request_end = self._increment_range(dataset, timeframe)
            if request_start is None:
                return {"symbol": symbol, "timeframe": timeframe, "skipped": True,
                        "message": "已更新至最新已结束周期"}
        else:
            if start is None:
                raise MarketDataError("历史导入或修补必须明确指定开始日期")
            request_start = pd.Timestamp(start).normalize()
            closed = self._cutoff()
            request_end = min(pd.Timestamp(end).normalize(), closed) if end else closed
            if mode == "repair":
                repair_windows = self._repair_windows(symbol, timeframe, request_start, request_end)
                if not repair_windows:
                    return {"symbol": symbol, "timeframe": timeframe, "skipped": True,
                            "message": "所选区间已有完整本地数据，未调用行情来源" if mode == "history" else "所选区间未发现需要修补的缺口",
                            "metrics": {"archive_hit": True, "checked_days": 0,
                                        "reused_days": 0, "scanned_shards": 0}}
            elif mode == "history" and dataset and self.repository.auxiliary(symbol, "calendar"):
                repair_windows = self._repair_windows(symbol, timeframe, request_start, request_end)
                if not repair_windows:
                    return {"symbol": symbol, "timeframe": timeframe, "skipped": True,
                            "message": "所选区间已有完整本地数据，未调用行情来源",
                            "metrics": {"archive_hit": True, "checked_days": 0,
                                        "reused_days": 0, "scanned_shards": 0}}
        if request_start > request_end:
            raise MarketDataError("开始日期不能晚于结束日期")
        if cancelled():
            raise InterruptedError("任务已取消")
        if timeframe == "1min" and repair_windows:
            # The recent-minute API returns one bounded tail window. Fetch it
            # once per symbol and let the missing-slot calculation decide which
            # returned bars are new, rather than calling it once per gap.
            request_start = min(window[0] for window in repair_windows)
            request_end = max(window[1] for window in repair_windows)
            repair_windows = None
        frames = []
        source = ""
        metadata: dict = {}
        for window_start, window_end in (repair_windows or [(request_start, request_end)]):
            fetched = None
            current_source, current_metadata = "", {}
            if cancelled():
                raise InterruptedError("任务已取消")
            fetched, current_source, current_metadata = self._fetch(
                symbol, timeframe, mode, window_start, window_end, cancelled
            )
            if not fetched.empty:
                frames.append(fetched)
            source = current_source or source
            metadata.update(current_metadata)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            raise MarketDataError("来源没有返回行情；覆盖水位未推进")
        normalized, pending = normalize_frame(
            frame, timeframe, label_convention=metadata.pop("label_convention", "auto"), source=source,
            skip_invalid_rows=True,
        )
        normalized = normalized[(normalized.timestamp >= request_start) & (normalized.timestamp <= request_end + pd.Timedelta(days=1))]
        if normalized.empty:
            raise MarketDataError("来源返回的行情不在请求区间内")
        frame_hash = hashlib.sha256(
            (str(symbol) + timeframe + source + normalized.timestamp.min().isoformat() + normalized.timestamp.max().isoformat() + str(len(normalized))).encode()
        ).hexdigest()
        with self._publication_guard(symbol):
            if cancelled():
                raise InterruptedError("任务已取消")
            metadata.update(self._persist_auxiliary(symbol, metadata))
            result = self.service._commit_frame(
                symbol, timeframe, normalized, fingerprint=frame_hash,
                file_name=f"sync:{source}:{mode}", params_hash=hashlib.sha256(f"{SOURCE_VERSION}|{mode}|{source}".encode()).hexdigest(),
                pending=pending, source=source, adjustment="none", etf_category=metadata.get("etf_category"),
                calendar_path=None, actions_path=None, progress=lambda *_: None, cancelled=cancelled,
                coverage_start=normalized.timestamp.min(), coverage_end=normalized.timestamp.max(),
                validation_start=request_start, validation_end=request_end,
                source_metadata={key: value for key, value in metadata.items() if key not in {"calendar", "actions"}},
            )
        result.update({"source": source, "requested_start": str(request_start.date()),
                       "requested_end": str(request_end.date()), "mode": mode})
        return result

    def _repair_windows(self, symbol, timeframe, start, end, cancelled=lambda: False, publish_validation=True):
        calendar_saved = self.repository.auxiliary(symbol, "calendar")
        if not calendar_saved:
            raise MarketDataError("缺口修补需要本地交易日历；请先导入或同步交易日历资料")
        calendar = self.repository._calendar_dates(calendar_saved.get("payload") or [])
        calendar = calendar[
            (calendar >= pd.Timestamp(start).normalize())
            & (calendar <= pd.Timestamp(end).normalize())
        ]
        if not len(calendar):
            return []
        from .coverage_index import CoverageArchive
        from .validation import validate_incremental

        archive = CoverageArchive(self.repository)
        head = archive.trusted_head(symbol, timeframe, "none", start=start, end=end)
        file_state = self.repository.market_file_state(symbol, timeframe, "none")
        invalid_ranges = []
        for start_value, end_value in file_state.get("invalid_ranges") or []:
            try:
                invalid_ranges.append((
                    pd.Timestamp(start_value).normalize(),
                    pd.Timestamp(end_value).normalize(),
                ))
            except (TypeError, ValueError):
                continue
        requested_calendar = set(calendar)
        rows = archive.rows(symbol, timeframe, "none", start=start, end=end)
        scanned = {
            pd.Timestamp(row["trade_date"]).normalize() for row in rows
            if row.get("scanned") and not row.get("dirty_reason")
            and int(row.get("rules_version") or 0) >= VALIDATION_RULES_VERSION
            and not any(
                start_value <= pd.Timestamp(row["trade_date"]).normalize() <= end_value
                for start_value, end_value in invalid_ranges
            )
        }
        pending = archive.pending_dates(head, start, end)
        unscanned = [
            day for day in sorted(requested_calendar)
            if day not in scanned or day in pending
        ]
        if head is None and publish_validation and not self.repository._read_only:
            # The first repair is the explicit archive-building pass. It reads
            # local shards only; no provider request is issued until gaps exist.
            with self._publication_guard(symbol):
                validate_incremental(
                    self.service, symbol, timeframe, mode="incremental",
                    start=start, end=end, cancelled=cancelled,
                )
        elif unscanned and publish_validation and not self.repository._read_only:
            with self._publication_guard(symbol):
                validate_incremental(
                    self.service, symbol, timeframe, mode="incremental",
                    start=start, end=end, dates=unscanned, force_dates=True, cancelled=cancelled,
                )
        result = CoverageArchive(self.repository).coverage_gaps(
            symbol, timeframe, start, end,
        )
        gap_dates = {
            pd.Timestamp(gap.get("trade_date") or gap.get("start")).normalize()
            for gap in result.get("gaps") or []
            if gap.get("trade_date") or gap.get("start")
        }
        gap_dates.update(
            pd.Timestamp(value).normalize()
            for value in result.get("unknown_dates") or []
            if pd.Timestamp(value).normalize() in requested_calendar
        )
        if not gap_dates:
            return []
        interval = timeframe_spec(timeframe).minutes or 1
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for day in sorted(gap_dates):
            window = (
                max(pd.Timestamp(start).normalize(), day - pd.Timedelta(days=10)),
                min(
                    pd.Timestamp(end).normalize() + pd.Timedelta(days=1),
                    day + pd.Timedelta(days=1, minutes=interval),
                ),
            )
            if windows and window[0] <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(windows[-1][1], window[1]))
            else:
                windows.append(window)
        return windows

    def _increment_range(self, dataset, timeframe):
        # Market publication and quality publication have separate watermarks.
        # A missing quality archive must never expand the network request.
        last = pd.Timestamp(dataset["last_time"])
        cutoff = self._cutoff()
        spec = timeframe_spec(timeframe)
        if spec.kind == "minute":
            start = last - pd.Timedelta(minutes=(spec.minutes or 1) * 2)
            end = cutoff + pd.Timedelta(days=1)
        elif timeframe == "1day":
            start = last - pd.Timedelta(days=3)
            end = cutoff
        elif timeframe == "1week":
            start = last - pd.Timedelta(days=14)
            end = cutoff
        else:
            start = last - pd.Timedelta(days=62)
            end = cutoff
        if last.normalize() > cutoff:
            return None, None
        return start, end

    def _fetch(self, symbol, timeframe, mode, start, end, cancelled):
        metadata: dict = {}
        if timeframe == "1day":
            record_count("download_requests", 1)
            with SharedSlots(
                self.store.root, "daily-source", self.download_capacity
            ).acquire(cancelled):
                daily, actions, calendar, source_metadata = fetch_auxiliary_routed(
                    self.store, symbol, start, end, cancelled
                )
            metadata = {**source_metadata, "actions": actions, "calendar": calendar}
            frame, source = daily, metadata.get("daily_source", "baostock")
            metadata.setdefault("label_convention", "end")
            return frame, source, metadata
        if timeframe == "1min":
            record_count("download_requests", 1)
            with SharedSlots(
                self.store.root, "recent-minute-source", self.download_capacity
            ).acquire(cancelled):
                def accept_recent(value, source):
                    if source == "ths":
                        return
                    reason = _recent_minute_incomplete(value, symbol, start, end)
                    if reason:
                        raise SourceFailure(
                            source, "recent_minute", "empty", "分钟窗口不完整：" + reason,
                        )

                frame, source, route = SourceRouter(
                    self.store, settings=self.service.settings,
                ).call(
                    "recent_minute", market_for_symbol(symbol),
                    {
                        "sina": lambda: fetch_sina(
                            symbol,
                            network_timeout=self.market_settings.network_timeout_seconds,
                        ),
                        "tdx": lambda: fetch_tdx(
                            symbol, earliest=start, cancelled=cancelled,
                            network_timeout=self.market_settings.network_timeout_seconds,
                        ),
                        "ths": lambda: fetch_ths(
                            symbol,
                            network_timeout=self.market_settings.network_timeout_seconds,
                            daily=self.repository.active_rows(
                                symbol, "1day", start=start, end=end,
                            ),
                        ),
                    },
                    fallbacks=("ths",),
                    accept=accept_recent,
                )
            metadata["label_convention"] = "end"
            metadata["recent_minute_route"] = route
            frame = frame[(frame.timestamp >= start - pd.Timedelta(days=1)) & (frame.timestamp <= end + pd.Timedelta(days=1))].copy()
            metadata["calendar"] = self._trade_calendar(start, end)
            return frame, source, metadata
        frame = self._baostock_history(symbol, timeframe, start, end)
        metadata["label_convention"] = "end"
        metadata["calendar"] = self._trade_calendar(start, end)
        return frame, "baostock", metadata

    def _worker_count(self, workers) -> int:
        configured = self.service.settings.performance
        requested = max(1, int(workers or 1))
        return max(1, min(
            requested, int(configured.download_workers),
            effective_cpu_budget(configured), 4,
        ))

    @staticmethod
    @serialized_baostock
    def _trade_calendar(start, end):
        try:
            bs = _baostock_session()
            result = bs.query_trade_dates(
                start_date=str(pd.Timestamp(start).date()), end_date=str(pd.Timestamp(end).date())
            )
            if result.error_code != "0":
                raise MarketDataError("BaoStock 日历失败：" + result.error_msg)
            rows = []
            while result.next():
                rows.append(result.get_row_data())
        except Exception:
            _baostock_reset()
            raise
        frame = pd.DataFrame(rows, columns=result.fields)
        if frame.empty:
            return frame
        return pd.DataFrame({"date": pd.to_datetime(frame.loc[frame.is_trading_day == "1", "calendar_date"])})

    @staticmethod
    @serialized_baostock
    def _baostock_history(symbol, timeframe, start, end):
        code, exchange = normalize_symbol(symbol).split(".")
        frequency = {"5min": "5", "15min": "15", "30min": "30", "60min": "60", "1week": "w", "1month": "m"}[timeframe]
        if timeframe in MINUTE_TIMEFRAMES:
            fields = "date,time,open,high,low,close,volume,amount"
        elif timeframe == "1day":
            fields = "date,open,high,low,close,preclose,volume,amount,tradestatus,isST"
        else:
            fields = "date,open,high,low,close,volume,amount,pctChg"
        try:
            bs = _baostock_session()
            result = bs.query_history_k_data_plus(
                exchange.lower() + "." + code, fields,
                start_date=str(pd.Timestamp(start).date()), end_date=str(pd.Timestamp(end).date()),
                frequency=frequency, adjustflag="3",
            )
            if result.error_code != "0":
                raise MarketDataError("BaoStock 行情失败：" + result.error_msg)
            rows = []
            while result.next():
                rows.append(result.get_row_data())
        except Exception:
            _baostock_reset()
            raise
        frame = pd.DataFrame(rows, columns=result.fields)
        if frame.empty:
            return frame
        if "time" in frame and frame.time.astype(str).str.len().gt(0).any():
            frame["timestamp"] = pd.to_datetime(frame.time, format="%Y%m%d%H%M%S%f", errors="coerce")
        else:
            frame["timestamp"] = pd.to_datetime(frame.date)
        return frame
