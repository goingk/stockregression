from __future__ import annotations

import hashlib
import io
import multiprocessing
import threading
from concurrent.futures import (
    FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait,
)
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stdout

import pandas as pd

from ..models import normalize_symbol
from ..performance import SharedSlots, effective_cpu_budget, record_count, record_details
from ..providers import (
    _baostock_reset, _baostock_session, fetch_sina, fetch_tdx, fetch_ths,
    latest_closed_date, serialized_baostock,
)
from ..settings import Settings
from ..source_health import SourceFailure, SourceRouter, fetch_auxiliary_routed, market_for_symbol
from ..storage import Store
from ..rules import instrument_terms
from ..quality import minute_slots
from .models import MarketDataError
from .quality import validate_and_fill
from .reader import normalize_frame
from .service import MarketDataService
from .repository import VALIDATION_RULES_VERSION
from .timeframes import MINUTE_TIMEFRAMES, normalize_timeframe, timeframe_spec


SOURCE_VERSION = "market-sync-v2-package"
DAILY_BATCH_SIZE = 128


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
):
    configured = Settings.model_validate_json(settings_json) if settings_json else None
    store = Store(root, settings=configured)
    service = MarketDataService(store, package_dir=package_dir, settings=configured)
    return MarketDataSyncService(service)._prepare_daily(
        symbol, mode, start, end, lambda: False
    )


def _terminate_executor(executor) -> None:
    if isinstance(executor, ProcessPoolExecutor):
        for process in list(getattr(executor, "_processes", {}).values()):
            try:
                process.terminate()
            except Exception:
                pass
    executor.shutdown(wait=False, cancel_futures=True)


class MarketDataSyncService:
    def __init__(self, service: MarketDataService):
        self.service = service
        self.store = service.store
        self.repository = service.repository
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
        checkpoint_scope = str(pd.Timestamp(latest_closed_date()).date()) if mode == "incremental" else f"{start or ''}:{end or ''}"
        run_key = (
            f"market-sync:{self.repository.package_id}:{timeframe}:{mode}:"
            f"{checkpoint_scope}:{SOURCE_VERSION}"
        )
        if timeframe == "1day" and len(symbols) > 1:
            return self._sync_daily_batches(
                symbols, mode, start=start, end=end, resume=resume,
                retry_failed=retry_failed, workers=workers,
                progress=progress, cancelled=cancelled, run_key=run_key,
            )
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
            if resume and state and state.get("status") == "completed":
                return "skipped", {"symbol": symbol, "timeframe": timeframe, "skipped": True,
                                   "message": "该同步检查点已完成"}
            if retry_failed and (not state or state.get("status") != "failed"):
                return "skipped", {"symbol": symbol, "timeframe": timeframe, "skipped": True,
                                   "message": "不属于失败重试范围"}
            if cancelled():
                raise InterruptedError("任务已取消")
            self.store.sync_item(run_key, symbol, status="running", mode=mode)
            try:
                result = self._one(symbol, timeframe, mode, start, end, cancelled)
                current = self.repository.dataset(symbol, timeframe)
                actual_through = current.get("last_time") if current else None
                self.store.sync_item(
                    run_key, symbol, status="completed", mode=mode,
                    daily_through=actual_through if timeframe == "1day" else None,
                    minute_through=actual_through if timeframe in MINUTE_TIMEFRAMES else None,
                )
                return ("skipped" if result.get("skipped") else "result"), result
            except InterruptedError:
                raise
            except Exception as exc:
                self.store.sync_item(run_key, symbol, status="failed", mode=mode, error=str(exc))
                return "failure", {"symbol": symbol, "reason": str(exc)}

        cursor = 0
        worker_count = self._worker_count(workers)
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
        status = "completed" if not failures else "partial" if results else "failed"
        return {"status": status, "mode": mode, "timeframe": timeframe,
                "symbols": results, "failures": failures, "skipped": skipped,
                "message": f"完成 {len(results)} 个，跳过 {len(skipped)} 个，失败 {len(failures)} 个"}

    def _sync_daily_batches(
        self, symbols, mode, *, start, end, resume, retry_failed, workers,
        progress, cancelled, run_key,
    ) -> dict:
        results, failures, skipped = [], [], []
        total = len(symbols)
        # Every symbol earns two credits: one when its bars are downloaded and
        # one when they are written into the package.  A whole batch used to be
        # worth a single credit, so the bar stayed at 0% for the entire
        # download and the job looked frozen.
        downloaded = 0
        written = 0
        worker_count = self._worker_count(workers)
        process_pool_healthy = True

        def report(message: str) -> None:
            progress((downloaded + written) / (2 * max(1, total)), message)

        for batch_start in range(0, total, self.daily_batch_size):
            if cancelled():
                raise InterruptedError("任务已取消")
            batch = symbols[batch_start:batch_start + self.daily_batch_size]
            active = []
            for symbol in batch:
                state = self.store.sync_item(run_key, symbol)
                if resume and state and state.get("status") == "completed":
                    skipped.append({"symbol": symbol, "timeframe": "1day", "skipped": True,
                                    "message": "该同步检查点已完成"})
                    # A settled checkpoint needs no download at all.
                    downloaded += 1
                    written += 1
                    continue
                if retry_failed and (not state or state.get("status") != "failed"):
                    skipped.append({"symbol": symbol, "timeframe": "1day", "skipped": True,
                                    "message": "不属于失败重试范围"})
                    downloaded += 1
                    written += 1
                    continue
                active.append(symbol)
                self.store.sync_item(run_key, symbol, status="running", mode=mode)
            prepared = []
            executor = None
            if len(active) > 1 and worker_count >= 4 and process_pool_healthy:
                executor = ProcessPoolExecutor(
                    max_workers=min(2, worker_count),
                    mp_context=multiprocessing.get_context("spawn"),
                )
                futures = {
                    executor.submit(
                        _prepare_daily_worker,
                        str(self.store.root), str(self.repository.package_dir),
                        symbol, mode, start, end, self.service.settings.model_dump_json(),
                    ): symbol for symbol in active
                }
            else:
                executor = ThreadPoolExecutor(max_workers=worker_count)
                futures = {
                    executor.submit(
                        self._prepare_daily, symbol, mode, start, end, cancelled
                    ): symbol for symbol in active
                }
            try:
                pending = set(futures)
                while pending:
                    if cancelled():
                        raise InterruptedError("任务已取消")
                    done, pending = wait(
                        pending, timeout=self.scheduler_poll_seconds, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        symbol = futures[future]
                        downloaded += 1
                        try:
                            try:
                                value = future.result()
                            except BrokenProcessPool:
                                # A dead worker must not turn every queued symbol
                                # into a failure. Retry only the unfinished read,
                                # then use threads for subsequent batches.
                                process_pool_healthy = False
                                value = self._prepare_daily(symbol, mode, start, end, cancelled)
                            if value.get("skipped"):
                                skipped.append(value)
                                self.store.sync_item(
                                    run_key, symbol, status="completed", mode=mode,
                                    message=value.get("message"),
                                )
                                written += 1
                            else:
                                prepared.append(value)
                        except InterruptedError:
                            raise
                        except Exception as exc:
                            written += 1
                            failures.append({"symbol": symbol, "reason": str(exc)})
                            self.store.sync_item(
                                run_key, symbol, status="failed", mode=mode, error=str(exc)
                            )
                        report(f"{downloaded}/{total} · {symbol} 1day")
            except InterruptedError:
                _terminate_executor(executor)
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            if prepared:
                for source in prepared:
                    symbol = source["symbol"]
                    try:
                        result = self.service._commit_frame(
                            symbol, "1day", source["frame"],
                            fingerprint=source.get("fingerprint"),
                            file_name=f"sync:{source.get('source')}:{mode}",
                            params_hash=source.get("params_hash"),
                            pending=list(source.get("pending") or []),
                            source=source.get("source") or "sync",
                            adjustment="none", etf_category=None,
                            calendar_path=None, actions_path=None,
                            progress=lambda *_: None, cancelled=cancelled,
                            source_metadata=source.get("metadata") or {},
                            coverage_start=source.get("coverage_start"),
                            coverage_end=source.get("coverage_end"),
                            validation_start=source.get("request_start"),
                            validation_end=source.get("request_end"),
                            import_mode="merge",
                        )
                        results.append({
                            **result, "mode": mode,
                            "requested_start": str(source["request_start"].date()),
                            "requested_end": str(source["request_end"].date()),
                            "source": source["source"],
                        })
                        self.store.sync_item(
                            run_key, symbol, status="completed", mode=mode,
                            daily_through=result.get("last_time"),
                        )
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        failures.append({"symbol": symbol, "reason": str(exc)})
                        self.store.sync_item(
                            run_key, symbol, status="failed", mode=mode, error=str(exc)
                        )
                    written += 1
                    report(f"{downloaded}/{total} · {symbol} 1day 写入")
            report(f"{downloaded}/{total} · 日线批次 {batch_start // self.daily_batch_size + 1}")
        status = "completed" if not failures else "partial" if results else "failed"
        return {
            "status": status, "mode": mode, "timeframe": "1day",
            "symbols": results, "failures": failures, "skipped": skipped,
            "batches": (total + self.daily_batch_size - 1) // self.daily_batch_size,
            "batch_size": self.daily_batch_size,
            "message": f"完成 {len(results)} 个，跳过 {len(skipped)} 个，失败 {len(failures)} 个",
        }

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
            closed = pd.Timestamp(latest_closed_date()).normalize()
            request_end = min(pd.Timestamp(end).normalize(), closed) if end else closed
            if mode == "repair":
                repair_windows = self._repair_windows(symbol, "1day", request_start, request_end)
                if not repair_windows:
                    return {"symbol": symbol, "timeframe": "1day", "skipped": True,
                            "message": "所选区间未发现需要修补的缺口",
                            "metrics": {"archive_hit": True, "checked_days": 0,
                                        "reused_days": 0, "scanned_shards": 0}}
            elif dataset and self.repository.auxiliary(symbol, "calendar"):
                repair_windows = self._repair_windows(symbol, "1day", request_start, request_end)
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
        if metadata.get("calendar") is not None:
            self.repository.save_auxiliary(
                symbol, "calendar", metadata["calendar"].to_dict("records"),
                version=SOURCE_VERSION, verified=True,
            )
        if metadata.get("actions") is not None:
            self.repository.save_auxiliary(
                symbol, "actions", metadata["actions"].to_dict("records"),
                version=SOURCE_VERSION, verified=bool(metadata.get("actions_verified")),
            )
        validation = validate_and_fill(
            normalized, "1day", calendar=self.service._calendar(symbol), pending=pending,
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
        if metadata.get("name"):
            self.store.put_instrument(
                symbol, metadata["name"],
                **{key: value for key, value in item_metadata.items() if key not in {"symbol", "name"}},
            )
        fingerprint = hashlib.sha256(
            f"{symbol}|1day|{source}|{normalized.timestamp.min()}|{normalized.timestamp.max()}|{len(normalized)}".encode()
        ).hexdigest()
        return {
            "symbol": symbol, "timeframe": "1day", "frame": validation.frame,
            "pending": validation.pending, "source": source, "metadata": item_metadata,
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
            closed = pd.Timestamp(latest_closed_date()).normalize()
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
        if metadata.get("calendar") is not None:
            self.repository.save_auxiliary(symbol, "calendar", metadata["calendar"].to_dict("records"), version=SOURCE_VERSION, verified=True)
        if metadata.get("actions") is not None:
            self.repository.save_auxiliary(symbol, "actions", metadata["actions"].to_dict("records"), version=SOURCE_VERSION, verified=bool(metadata.get("actions_verified")))
        frame_hash = hashlib.sha256(
            (str(symbol) + timeframe + source + normalized.timestamp.min().isoformat() + normalized.timestamp.max().isoformat() + str(len(normalized))).encode()
        ).hexdigest()
        result = self.service._commit_frame(
            symbol, timeframe, normalized, fingerprint=frame_hash,
            file_name=f"sync:{source}:{mode}", params_hash=hashlib.sha256(f"{SOURCE_VERSION}|{mode}|{source}".encode()).hexdigest(),
            pending=pending, source=source, adjustment="none", etf_category=metadata.get("etf_category"),
            calendar_path=None, actions_path=None, progress=lambda *_: None, cancelled=cancelled,
            coverage_start=normalized.timestamp.min(), coverage_end=normalized.timestamp.max(),
            validation_start=request_start, validation_end=request_end,
            source_metadata={key: value for key, value in metadata.items() if key not in {"calendar", "actions"}},
        )
        # Persist auxiliary material only when the same source request supplied it.
        if metadata.get("calendar") is not None:
            self.repository.save_auxiliary(symbol, "calendar", metadata["calendar"].to_dict("records"), version=SOURCE_VERSION, verified=True)
        if metadata.get("actions") is not None:
            self.repository.save_auxiliary(symbol, "actions", metadata["actions"].to_dict("records"), version=SOURCE_VERSION, verified=bool(metadata.get("actions_verified")))
        result.update({"source": source, "requested_start": str(request_start.date()),
                       "requested_end": str(request_end.date()), "mode": mode})
        return result

    def _repair_windows(self, symbol, timeframe, start, end):
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
        if head is None:
            # The first repair is the explicit archive-building pass. It reads
            # local shards only; no provider request is issued until gaps exist.
            validate_incremental(
                self.service, symbol, timeframe, mode="incremental",
                start=start, end=end,
            )
        elif unscanned:
            validate_incremental(
                self.service, symbol, timeframe, mode="incremental",
                start=start, end=end, dates=unscanned, force_dates=True,
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
        from .coverage_index import CoverageArchive

        archive_head = CoverageArchive(self.repository).load_head(
            dataset["symbol"], timeframe, dataset.get("adjustment", "none")
        )
        # The archive records the tail that has actually reached the quality
        # index.  If publication stopped between market and archive heads, the
        # overlap is intentionally fetched again; unchanged rows are discarded.
        archive_last = pd.Timestamp(
            (archive_head or {}).get("last_time") or dataset["last_time"]
        ).normalize()
        dataset_last = pd.Timestamp(dataset["last_time"]).normalize()
        last = min(archive_last, dataset_last)
        cutoff = pd.Timestamp(latest_closed_date()).normalize()
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
