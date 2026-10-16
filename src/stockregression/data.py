from __future__ import annotations

import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import shutil
from filelock import FileLock, Timeout

import numpy as np
import pandas as pd

from .models import BacktestConfig, normalize_symbol
from .providers import fetch_auxiliary, fetch_hf, fetch_instruments, fetch_tdx, fetch_sina, fetch_reference_5m, fetch_latest_trading_date, latest_closed_date, normalize_actions
from .quality import DataQualityError, align_minutes, canonical_frame, infer_profile, minute_slots, prepare_bundle
from .settings import load_settings
from .storage import Store
from .rules import instrument_terms


def merge_frames(old: pd.DataFrame, new: pd.DataFrame, key: str) -> pd.DataFrame:
    parts = [part for part in (old, new) if not part.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(key, kind="stable").drop_duplicates(key, keep="last").reset_index(drop=True)


class DataService:
    def __init__(self, store: Store):
        self.store = store
        self._sync_context = threading.local()

    def _sync_stage(self, symbol, **values):
        run_key = getattr(self._sync_context, "run_key", None)
        if run_key:
            self.store.sync_item(run_key, symbol, **values)

    def refresh_instruments(self, progress=lambda *_: None, cancelled=lambda: False):
        progress(0.1, "下载固定版本的证券覆盖清单")
        records = fetch_instruments(load_settings().hf_revision)
        for record in records:
            if cancelled():
                raise InterruptedError("任务已取消")
            self.store.put_instrument(**record)
        warnings = sorted({warning for record in records for warning in record.get("catalog_warnings", [])})
        self.store.state("catalog:latest", {"count": len(records), "warnings": warnings})
        return {"count": len(records), "warnings": warnings, "message": f"已更新 {len(records)} 只股票及 ETF；保留来源覆盖的 ST 与退市历史"}

    def sync(self, symbols: list[str], mode: str, progress=lambda *_: None, cancelled=lambda: False, scope: str | None = None, resume: bool = False, retry_failed: bool = False, workers: int = 2):
        scope = scope or ("selected" if symbols else "watchlist")
        if scope == "market":
            if symbols:
                raise ValueError("全市场范围不能同时指定代码")
            if not retry_failed or not self.store.instruments(limit=1):
                self.refresh_instruments(progress, cancelled)
            symbols = [row["symbol"] for row in self.store.instruments(limit=None)]
        elif not symbols:
            symbols = [row["symbol"] for row in self.store.watches()]
        if not symbols:
            return {"symbols": [], "message": "自选股为空，本次无需更新", "warnings": []}
        symbols = sorted({normalize_symbol(symbol) for symbol in symbols})
        cutoff = latest_closed_date()
        calendar_warning = None
        if mode != "history":
            try:
                cutoff = fetch_latest_trading_date(str(cutoff.date()))
            except Exception as exc:
                calendar_warning = "最新交易日预检暂不可用，各标的仍须获取真实交易日历：" + str(exc)
        estimate = None
        estimate_warning = None
        if scope == "market" and mode == "history":
            try:
                from .providers import estimate_history_bytes
                estimate = estimate_history_bytes(symbols, load_settings())
            except Exception as exc:
                estimate_warning = "无法取得精确下载量估算：" + str(exc)
        free = shutil.disk_usage(self.store.root).free
        preflight = {"target_count": len(symbols), "source_bytes": estimate, "free_bytes": free, "warning": estimate_warning, "closed_through": str(cutoff.date()), "calendar_warning": calendar_warning}
        self.store.state("sync:preflight", preflight)
        progress(0.0, f"目标 {len(symbols)} 个标的 · 历史源约 {estimate / 1024 ** 3:.2f} GiB · 可用 {free / 1024 ** 3:.1f} GiB" if estimate is not None else f"目标 {len(symbols)} 个标的 · 可用空间 {free / 1024 ** 3:.1f} GiB；历史源需数十 GB，另预留缓存及分区空间")
        run_key = f"{scope}:{mode}:" + (f"{load_settings().hf_revision}:{load_settings().etf_hf_revision}" if mode == "history" else str(cutoff.date()))
        results, failures, skipped = [], [], []
        targets = []
        for symbol in symbols:
            state = self.store.sync_item(run_key, symbol)
            if (resume and state and state["status"] == "completed") or (retry_failed and (not state or state["status"] != "failed")):
                skipped.append(symbol)
            else:
                targets.append(symbol)
        report_lock = threading.Lock()
        completed = 0

        def one(symbol):
            nonlocal completed
            self._sync_context.run_key = run_key
            self._sync_context.cutoff = cutoff
            self.store.sync_item(run_key, symbol, status="running", mode=mode, daily_status="running", minute_status="running", error=None)
            try:
                if cancelled():
                    raise InterruptedError("任务已取消")
                current = self.store.snapshot(symbol)
                if mode != "history" and current and current["metadata"].get("is_demo"):
                    self.store.sync_item(run_key, symbol, status="completed", daily_status="skipped_demo", minute_status="skipped_demo", error=None)
                    return {"symbol": symbol, "skipped": True, "message": "演示数据不参与自动真实行情更新"}
                def report(fraction, message):
                    with report_lock:
                        progress((completed + fraction) / max(1, len(targets)), f"{completed}/{len(targets)} · {symbol}：{message}")
                with FileLock(str(self.store.root / f"sync-{symbol}.lock"), timeout=60):
                    for attempt in range(3):
                        try:
                            result = self._sync_one(symbol, mode, report, cancelled)
                            break
                        except InterruptedError:
                            raise
                        except Exception:
                            if attempt == 2:
                                raise
                            for _ in range(5 * (attempt + 1)):
                                if cancelled():
                                    raise InterruptedError("任务已取消")
                                time.sleep(0.2)
                self.store.sync_item(run_key, symbol, status="completed", daily_status="completed", minute_status=result.get("minute_status", "available" if result.get("minute_through") else "unavailable"), daily_through=result.get("last_date"), minute_through=result.get("minute_through"), error=None)
                self.store.state(f"sync:{symbol}", {"mode": mode, "result": result, "error": None})
                return result
            except InterruptedError:
                self.store.sync_item(run_key, symbol, status="pending", error="任务中断，可恢复")
                raise
            except Exception as exc:
                state = self.store.sync_item(run_key, symbol) or {}
                self.store.sync_item(run_key, symbol, status="failed", daily_status="failed" if state.get("daily_status") == "running" else state.get("daily_status", "failed"), minute_status="failed", error=str(exc))
                return {"symbol": symbol, "error": str(exc)}
            finally:
                with report_lock:
                    completed += 1

        # Bound the queue as well as worker count, so cancellation does not leave
        # thousands of requests waiting to execute.
        with ThreadPoolExecutor(max_workers=max(1, min(4, workers))) as executor:
            for offset in range(0, len(targets), max(1, workers)):
                if cancelled():
                    raise InterruptedError("任务已取消")
                for future in as_completed([executor.submit(one, symbol) for symbol in targets[offset:offset + max(1, workers)]]):
                    result = future.result()
                    if "error" in result:
                        failures.append({"symbol": result["symbol"], "reason": result["error"]})
                    else:
                        results.append(result)
        warnings = (self.store.state("catalog:latest") or {}).get("warnings", []) if scope == "market" else []
        if calendar_warning:
            warnings = [*warnings, calendar_warning]
        return {"status": "failed" if failures and not results else "completed", "run_key": run_key, "scope": scope, "preflight": preflight, "warnings": warnings, "symbols": results, "failures": failures, "skipped": skipped, "message": f"完成 {len(results)} 只，失败 {len(failures)} 只，已完成跳过 {len(skipped)} 只" + (f"；证券目录来源警告 {len(warnings)} 条，请查看报告" if warnings else "")}

    def current_version_if_ready(self, config: BacktestConfig) -> str | None:
        """Return an immutable submission-time version only when it needs no refresh."""
        manifest = self.store.snapshot(config.symbol)
        if not manifest:
            return None
        request, context_key = self._preparation_request(config, check_minutes=True)
        cache_key = "ready:" + context_key
        cached = self.store.state(cache_key)
        if cached and cached.get("version") == manifest["version"]:
            return manifest["version"] if cached.get("ready") else None
        try:
            bundle, gaps, load_error = self._probe_bundle(config, manifest["version"], check_minutes=True)
        except DataQualityError as exc:
            bundle, gaps, load_error = None, [], str(exc)
        gaps = self._ignore_demo_gaps(manifest, gaps)
        ready = bundle is not None and not gaps and not load_error
        self.store.state(cache_key, {"version": manifest["version"], "ready": ready, "error": load_error, "gaps": gaps})
        return manifest["version"] if ready else None

    def ensure_bundle(
        self,
        config: BacktestConfig,
        version: str | None = None,
        progress=lambda *_: None,
        cancelled=lambda: False,
        check_minutes: bool = True,
        cached_bundle=None,
    ):
        """Prepare and verify local data, sharing one cross-process attempt per symbol/range.

        Explicit versions are immutable inputs from an already-submitted job and are
        never replaced by a newer active snapshot. New requests pass ``None`` when
        their current snapshot needs preparation, then lock the exact resolved version.
        """
        if version is not None:
            progress(0.02, "读取已锁定的数据版本")
            if cancelled():
                raise InterruptedError("任务已取消")
            if cached_bundle is not None and cached_bundle.version == version:
                bundle = cached_bundle
            else:
                try:
                    bundle = self.load_bundle(config, version)
                except DataQualityError as exc:
                    raise DataQualityError(f"锁定的数据版本 {version} 不满足本次配置：{exc}") from exc
            return self._add_preparation_warnings(bundle, [])

        progress(0.01, "检查本地数据覆盖")
        if cancelled():
            raise InterruptedError("任务已取消")
        request, context_key = self._preparation_request(config, check_minutes)
        target_cutoff = pd.Timestamp(request["end_date"])
        source_versions = request["source_versions"]
        key = "prepare:" + context_key

        # This lock is deliberately distinct from sync-{symbol}.lock. sync() owns
        # that lock around _sync_one; nesting the same lock here would deadlock.
        prepare_lock = FileLock(str(self.store.root / f"prepare-{config.symbol}.lock"))
        self._acquire_prepare_lock(prepare_lock, cancelled, timeout=300)
        try:
            if cancelled():
                raise InterruptedError("任务已取消")
            manifest = self.store.snapshot(config.symbol)
            active_version = manifest["version"] if manifest else None
            if cached_bundle is not None and cached_bundle.version == active_version:
                bundle = cached_bundle
                gaps = self._bundle_gaps(config, manifest, bundle, check_minutes)
                load_error = None
            else:
                bundle, gaps, load_error = self._probe_bundle(config, active_version, check_minutes)
            gaps = self._ignore_demo_gaps(manifest, gaps)
            mode = self._sync_mode(config, manifest, bundle, gaps, load_error, target_cutoff, check_minutes)
            record = self.store.state(key) or {}
            same_attempt = (
                record.get("source_versions") == source_versions
                and record.get("target_cutoff") == str(target_cutoff.date())
                and record.get("request") == request
            )
            suppress = False
            if same_attempt and record.get("output_version") == active_version:
                if record.get("status") in {"prepared", "insufficient"} and not record.get("error"):
                    suppress = True
                elif record.get("error"):
                    retry_after = float(record.get("retry_after", 0) or 0)
                    suppress = time.time() < retry_after
            elif same_attempt and record.get("input_version") == active_version:
                retry_after = float(record.get("retry_after", 0) or 0)
                suppress = record.get("status") == "failed" and time.time() < retry_after

            if mode is None or suppress:
                progress(0.06, "复用已检查的本地数据" if mode is None else "复用最近一次补数检查结果")
                if bundle is None:
                    detail = record.get("error") if suppress else load_error
                    raise DataQualityError(self._preparation_error(config.symbol, detail, gaps))
                warnings = self._preparation_messages(gaps)
                if suppress and record.get("error"):
                    warnings.append(f"最近一次自动补数未完成：{record['error']}；60 秒后可重试")
                return self._add_preparation_warnings(bundle, warnings)

            input_version = active_version
            gap_signature = hashlib.sha256(json.dumps(gaps, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            self.store.state(key, {
                "status": "running",
                "source_versions": source_versions,
                "request": request,
                "target_cutoff": str(target_cutoff.date()),
                "input_version": input_version,
                "gap_signature": gap_signature,
                "remaining_gaps": gaps,
                "attempted_at": time.time(),
            })
            progress(0.08, self._sync_message(mode, gaps))
            if cancelled():
                raise InterruptedError("任务已取消")
            try:
                report = self.sync([config.symbol], mode, lambda fraction, message: progress(0.08 + 0.55 * max(0, min(1, fraction)), message), cancelled, scope="selected", workers=1)
                failures = report.get("failures", [])
                sync_error = "; ".join(f"{item.get('symbol')}: {item.get('reason')}" for item in failures) or None
                synced = next((item for item in report.get("symbols", []) if item.get("symbol") == config.symbol), None)
                candidate_version = (synced or {}).get("version")
                source_errors = (synced or {}).get("source_errors", [])
                if source_errors and not sync_error:
                    sync_error = "; ".join(source_errors)
                if report.get("status") == "failed" and sync_error is None:
                    sync_error = report.get("message") or "数据源未返回可用数据"
            except InterruptedError:
                self.store.state(key, {"status": "cancelled", "source_versions": source_versions, "request": request, "target_cutoff": str(target_cutoff.date()), "input_version": input_version, "error": "任务已取消"})
                raise
            except Exception as exc:
                sync_error = str(exc)
                candidate_version = None

            if cancelled():
                raise InterruptedError("任务已取消")
            progress(0.68, "校验补数结果并锁定数据版本")
            resolved_version = candidate_version or ((self.store.snapshot(config.symbol) or {}).get("version"))
            try:
                bundle, remaining, load_error = self._probe_bundle(config, resolved_version, check_minutes)
            except Exception as exc:
                bundle, remaining, load_error = None, gaps, str(exc)
            status = "failed" if bundle is None else "insufficient" if remaining else "prepared"
            state = {
                "status": status,
                "source_versions": source_versions,
                "request": request,
                "target_cutoff": str(target_cutoff.date()),
                "input_version": input_version,
                "output_version": resolved_version if bundle is not None else active_version,
                "gap_signature": gap_signature,
                "remaining_gaps": remaining,
                "attempted_at": time.time(),
                "retry_after": time.time() + 60 if sync_error or bundle is None else None,
                "error": sync_error or load_error,
            }
            self.store.state(key, state)
            if bundle is None:
                raise DataQualityError(self._preparation_error(config.symbol, sync_error or load_error, remaining))
            warnings = self._preparation_messages(remaining)
            if sync_error:
                warnings.append(f"自动补数未完全成功：{sync_error}")
            progress(0.72, "本地行情已校验；开始计算")
            return self._add_preparation_warnings(bundle, warnings)
        finally:
            prepare_lock.release()

    @staticmethod
    def _acquire_prepare_lock(lock, cancelled, timeout=300):
        """Acquire the preparation lock in short waits so queued jobs can cancel."""
        deadline = time.monotonic() + timeout
        while True:
            if cancelled():
                raise InterruptedError("任务已取消")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Timeout(f"等待行情准备锁超过 {timeout} 秒")
            try:
                lock.acquire(timeout=min(0.25, remaining))
                return
            except Timeout:
                continue

    @staticmethod
    def _preparation_request(config, check_minutes):
        target_cutoff = min(
            pd.Timestamp(config.end_date or latest_closed_date()).normalize(),
            pd.Timestamp(latest_closed_date()).normalize(),
        )
        settings = load_settings()
        source_versions = {
            "stock_history": settings.hf_revision,
            "etf_history": settings.etf_hf_revision,
            "preparation": 1,
        }
        request = {
            "symbol": config.symbol,
            "start_date": str(pd.Timestamp(config.start_date).date()),
            "end_date": str(target_cutoff.date()),
            "frequency": config.frequency if check_minutes else "1d",
            "schema_version": config.schema_version,
            "check_minutes": bool(check_minutes),
            "source_versions": source_versions,
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        return request, key

    @staticmethod
    def _ignore_demo_gaps(manifest, gaps):
        if not manifest or not manifest.get("metadata", {}).get("is_demo"):
            return gaps
        return [gap for gap in gaps if not gap.startswith(("日线尾部缺失", "分钟数据待补"))]

    def _probe_bundle(self, config, version, check_minutes):
        manifest = self.store.snapshot(config.symbol, version)
        if not manifest:
            return None, ["尚无本地快照"], "尚无本地行情"
        daily = self.store.read_files(manifest["files"].get("daily", []))
        if daily.empty:
            return None, ["缺少日线"], "本地版本没有日线"
        try:
            bundle = self.load_bundle(config, manifest["version"])
        except DataQualityError as exc:
            # V2 historically permits a request end beyond local coverage by
            # clipping to the last available closed bar. Keep that behavior for
            # a failed tail refresh; all other hard quality failures still stop.
            message = str(exc)
            target_end = min(pd.Timestamp(config.end_date or latest_closed_date()).normalize(), pd.Timestamp(latest_closed_date()).normalize())
            if config.schema_version == 2 and target_end > daily.date.max() and not any(token in message for token in ("ETF 分红", "参考价发生变化", "没有可回测交易日")):
                clipped = config.model_copy(update={"end_date": daily.date.max().date()})
                try:
                    bundle = self.load_bundle(clipped, manifest["version"])
                except DataQualityError:
                    return None, [message], message
            else:
                return None, [message], message
        gaps = self._bundle_gaps(config, manifest, bundle, check_minutes, daily)
        return bundle, gaps, None

    def _bundle_gaps(self, config, manifest, bundle, check_minutes, daily=None):
        daily = daily if daily is not None else bundle.daily
        gaps = []
        start = pd.Timestamp(config.start_date).normalize()
        target_end = min(pd.Timestamp(config.end_date or latest_closed_date()).normalize(), pd.Timestamp(latest_closed_date()).normalize())
        if daily.date.min().normalize() > start:
            gaps.append(f"历史日线起点缺失：请求 {start.date()}，本地 {daily.date.min().date()}")
        if daily.date.max().normalize() < target_end:
            gaps.append(f"日线尾部缺失：请求至 {target_end.date()}，本地至 {daily.date.max().date()}")
        if check_minutes and config.frequency != "1d":
            daily_days = [row for row in bundle.quality.get("days", []) if row.get("mode") == "daily"]
            if daily_days:
                dates = [row.get("date") for row in daily_days]
                gaps.append("分钟数据待补：" + ", ".join(str(value) for value in dates[:12]) + ("…" if len(dates) > 12 else ""))
        if check_minutes and config.frequency != "1d":
            source = "etf_history" if manifest.get("metadata", {}).get("asset_type") == "etf" else "stock_history"
            settings = load_settings()
            current_source = getattr(settings, "etf_hf_revision" if source == "etf_history" else "hf_revision")
            saved_source = manifest.get("metadata", {}).get("hf_revision")
            if saved_source and saved_source != current_source:
                gaps.append(f"历史分钟源版本已变化：本地 {saved_source}，当前 {current_source}")
        return gaps

    def _preparation_gaps(self, config, manifest, bundle, check_minutes):
        _, gaps, _ = self._probe_bundle(config, manifest["version"], check_minutes)
        # A demo snapshot is an explicit local fixture; preserve its existing
        # behavior instead of silently replacing it during parameter edits.
        if manifest.get("metadata", {}).get("is_demo"):
            gaps = [gap for gap in gaps if not gap.startswith("日线尾部缺失") and not gap.startswith("分钟数据待补")]
        return gaps

    def _sync_mode(self, config, manifest, bundle, gaps, load_error, target_cutoff, check_minutes):
        if not gaps:
            return None
        if not manifest or any(gap.startswith(("尚无", "缺少日线", "历史日线起点缺失", "历史分钟源版本已变化")) for gap in gaps):
            return "history"
        if load_error and not bundle:
            if "ETF 分红" in load_error or "公司行动" in load_error or "除权" in load_error:
                return "history"
            if "日线完整截止" in load_error or "请求至" in load_error:
                return "daily"
            return "history"
        daily = self.store.read_files(manifest["files"].get("daily", []))
        has_tail = daily.empty or daily.date.max().normalize() < target_cutoff
        minute_gap = any(gap.startswith("分钟数据待补") for gap in gaps)
        if has_tail and not minute_gap:
            return "daily"
        if minute_gap:
            rows = [row for row in (bundle.quality.get("days", []) if bundle else []) if row.get("mode") == "daily"]
            gap_dates = [pd.Timestamp(row["date"]).normalize() for row in rows if row.get("date")]
            if gap_dates and min(gap_dates) >= target_cutoff - pd.Timedelta(days=100):
                return "repair"
            return "history"
        return "daily"

    @staticmethod
    def _preparation_messages(gaps):
        messages = []
        for gap in gaps:
            if gap.startswith("分钟数据待补"):
                messages.append("分钟数据源补数后仍存在缺口，相关交易日按已校验日线降级；稳健研究分钟覆盖率门槛保持不变。" + gap)
            elif gap.startswith("日线尾部缺失"):
                messages.append(gap + "；本次结果截至本地最后一个已收盘日。")
            elif gap.startswith("历史日线起点缺失"):
                messages.append(gap)
            elif gap != "尚无本地快照":
                messages.append(gap)
        return messages

    @staticmethod
    def _preparation_error(symbol, detail, gaps):
        reason = detail or ("；".join(gaps) if gaps else "数据源未返回可用数据")
        return f"{symbol} 自动准备行情失败：{reason}。请检查数据源，或在行情页导入本地数据后重试。"

    @staticmethod
    def _sync_message(mode, gaps):
        label = {"history": "补齐历史行情", "daily": "更新近期日线和分钟行情", "repair": "修复近期分钟缺口"}.get(mode, "更新行情")
        return f"{label}：" + ("；".join(gaps[:2]) if gaps else "发现数据覆盖不足")

    @staticmethod
    def _add_preparation_warnings(bundle, warnings):
        if warnings:
            bundle.warnings = list(dict.fromkeys([*(bundle.warnings or []), *warnings]))
            bundle.metadata = {**bundle.metadata, "warnings": list(dict.fromkeys([*bundle.metadata.get("warnings", []), *warnings]))}
        return bundle

    def _sync_one(self, symbol, mode, progress, cancelled):
        previous = self.store.snapshot(symbol)
        if previous and previous["metadata"].get("is_demo"):
            previous = None  # Never mix simulated records into a real-data snapshot.
        old_files = previous["files"] if previous else {}
        # Only the tail is read for calibration; publish merges affected years.
        end = getattr(self._sync_context, "cutoff", latest_closed_date())
        old_daily = self.store.read_files(old_files.get("daily", []), "date", end - pd.Timedelta(days=45))
        warnings = []
        source_errors = []
        hf = pd.DataFrame()
        profiles = dict(previous.get("profiles", {})) if previous else {}
        if mode == "history" or previous is None:
            progress(0.05, "按股票下载历史分钟 Parquet")
            try:
                hf = fetch_hf(symbol, self.store.root, load_settings().hf_revision)
            except Exception as exc:
                warnings.append("历史分钟下载失败：" + str(exc))
                source_errors.append("历史分钟下载失败：" + str(exc))
        known = self.store.instrument(symbol) or {}
        if old_daily.empty and previous:
            # A long-offline database must resume from its own watermark.
            paths = old_files.get("daily", [])
            old_daily = self.store.read_files(paths[-1:] if paths else [])
        start = (old_daily.date.max() - pd.Timedelta(days=30)) if mode != "history" and not old_daily.empty else pd.Timestamp(known.get("listing_date") or "1990-01-01")
        progress(0.25, "获取不复权日线、交易日历和公司行动")
        daily, actions, calendar, metadata = fetch_auxiliary(symbol, start, end, cancelled)
        self._sync_stage(symbol, daily_status="fetched", daily_through=str(daily.date.max().date()))
        incoming_daily = daily
        # A history calibration needs the daily history, an increment needs only
        # its recent overlap. Neither reads the old full minute archive.
        daily = merge_frames(old_daily, daily, "date")
        if metadata.get("actions_checked_years"):
            checked = set(metadata["actions_checked_years"])
            earlier = (previous or {}).get("metadata", {}).get("actions_unverified_years", [])
            metadata["actions_unverified_years"] = sorted((set(earlier) - checked) | set(metadata.get("actions_unverified_years", [])))
            metadata["actions_verified"] = not metadata["actions_unverified_years"]
        metadata = instrument_terms(symbol, {**known, **metadata})
        if not metadata.get("listing_date") and start.year <= 1990:
            metadata["listing_date"] = str(daily.date.min().date())
        metadata["hf_revision"] = load_settings().etf_hf_revision if metadata["asset_type"] == "etf" else load_settings().hf_revision
        metadata["is_demo"] = False
        if metadata.get("listing_date"):
            calendar = calendar[calendar.date >= pd.Timestamp(metadata["listing_date"])].reset_index(drop=True)
        if metadata.get("delisting_date"):
            calendar = calendar[calendar.date < pd.Timestamp(metadata["delisting_date"])].reset_index(drop=True)
        progress(0.55, "采集通达信分钟窗口并核对标签与单位")
        recent = pd.DataFrame()
        raw_cache = []
        earliest = end - pd.Timedelta(days=14) if mode == "daily" and previous else None
        try:
            recent_raw = fetch_tdx(symbol, earliest, cancelled)
            profiles["tdx"] = infer_profile(recent_raw, daily, "tdx", price_tick=metadata.get("price_tick", .01))
            recent = align_minutes(recent_raw, profiles["tdx"]["convention"], profiles["tdx"]["volume_multiplier"])
            recent["source_verified"] = profiles["tdx"]["verified"]
            raw_cache.append(("tdx", recent_raw))
        except InterruptedError:
            raise
        except Exception as exc:
            warnings.append("近期分钟源不可用：" + str(exc))
            source_errors.append("近期分钟源不可用：" + str(exc))
            progress(0.62, "通达信不可用，尝试新浪近期分钟备源")
            try:
                recent_raw = fetch_sina(symbol)
                reference = fetch_reference_5m(symbol, recent_raw.timestamp.max().normalize() - pd.Timedelta(days=21), recent_raw.timestamp.max().normalize())
                profiles["sina"] = infer_profile(recent_raw, daily, "sina", reference=reference, reference_frequency="5m", price_tick=metadata.get("price_tick", .01))
                recent = align_minutes(recent_raw, profiles["sina"]["convention"], profiles["sina"]["volume_multiplier"])
                recent["source_verified"] = profiles["sina"]["verified"]
                raw_cache.append(("sina", recent_raw))
            except Exception as backup_error:
                warnings.append("新浪分钟备源不可用或未校准：" + str(backup_error))
                source_errors.append("新浪分钟备源不可用或未校准：" + str(backup_error))
        if not hf.empty:
            profiles["hf"] = infer_profile(hf, daily, "hf", recent if not recent.empty and profiles.get("tdx", {}).get("verified") else None, price_tick=metadata.get("price_tick", .01))
            if not profiles["hf"]["verified"]:
                progress(0.72, "用独立历史5分钟行情核验历史分钟时间标签")
                try:
                    reference_end = hf.timestamp.max().normalize() - pd.Timedelta(days=1)
                    reference = fetch_reference_5m(symbol, reference_end - pd.Timedelta(days=21), reference_end)
                    # Only the bounded reference window is needed for calibration.
                    sample = hf[hf.timestamp.between(reference.timestamp.min() - pd.Timedelta(minutes=5), reference.timestamp.max())]
                    profiles["hf"] = infer_profile(sample, daily, "hf", reference=reference, reference_frequency="5m", price_tick=metadata.get("price_tick", .01))
                except Exception as exc:
                    warnings.append("历史分钟独立校验失败：" + str(exc))
                    source_errors.append("历史分钟独立校验失败：" + str(exc))
            hf = align_minutes(hf, profiles["hf"]["convention"], profiles["hf"]["volume_multiplier"])
            hf["source_verified"] = profiles["hf"]["verified"]
        # Prefer the explicitly calibrated recent provider in its overlap window.
        minute = merge_frames(hf, recent, "timestamp")
        if not minute.empty:
            minute = minute[minute.timestamp < end + pd.Timedelta(days=1)].copy()
        warnings.extend(metadata.get("warnings", []))
        if metadata.get("asset_type") == "etf":
            warnings.append("ETF 历史分钟库仅覆盖部分标的；新浪 1 分钟备源仅覆盖近期约 8 个交易日，其他缺口需导入补齐")
        metadata["warnings"] = list(dict.fromkeys(warnings))
        metadata["source_errors"] = list(dict.fromkeys(source_errors))
        metadata["daily_through"] = str(daily.date.max().date())
        metadata["minute_through"] = str(minute.timestamp.max()) if not minute.empty else (previous or {}).get("metadata", {}).get("minute_through")
        minute_status = "available" if any(profile.get("verified") for profile in profiles.values()) else "unverified" if metadata["minute_through"] else "unavailable"
        self._sync_stage(symbol, minute_status=minute_status, minute_through=metadata["minute_through"])
        if cancelled():
            raise InterruptedError("任务已取消")
        progress(0.80, "检查日线、日历、公司行动和分钟质量")
        previous_actions = self.store.read_files(old_files.get("actions", [])) if previous else pd.DataFrame()
        validation_actions = merge_frames(previous_actions, actions, "ex_date")
        candidate_quality = self._validate_candidate(symbol, daily, minute, validation_actions, calendar, metadata, profiles)
        metadata["quality_summary"] = {
            key: candidate_quality.get(key)
            for key in ("minute_days", "daily_days", "suspended_days", "total_days", "minute_ratio", "synthetic_bars")
        }
        metadata["quality_warnings"] = list(candidate_quality.get("warnings", []))
        for source, raw_frame in raw_cache:
            raw_folder = self.store.root / "raw" / source / symbol
            raw_folder.mkdir(parents=True, exist_ok=True)
            raw_frame.to_parquet(raw_folder / "latest.parquet", index=False)
        progress(0.85, "发布已校验的不可变分区和数据版本")
        manifest = self.store.publish(symbol, minute, incoming_daily, actions, calendar, metadata, profiles, incremental=previous is not None)
        self.store.put_instrument(symbol, **{key: value for key, value in metadata.items() if key != "symbol"})
        progress(1, "完成；质量报告已可查看")
        return {"symbol": symbol, "version": manifest["version"], "last_date": str(daily.date.max().date()), "daily_status": "completed", "minute_status": minute_status, "minute_through": metadata["minute_through"], "rows": len(minute), "warnings": metadata["warnings"], "source_errors": metadata["source_errors"], "profiles": profiles}

    def _validate_candidate(self, symbol, daily, minute, actions, calendar, metadata, profiles):
        """Reject hard-invalid candidate data before it can become an active snapshot.

        Minute coverage failures remain valid, reportable data: prepare_bundle
        intentionally falls back to checked daily bars for those days.
        """
        if daily.empty:
            raise DataQualityError("候选行情没有日线")
        if calendar.empty:
            raise DataQualityError("候选行情没有真实交易日历")
        start = pd.Timestamp(calendar.date.min()).normalize()
        end = pd.Timestamp(calendar.date.max()).normalize()
        candidate = prepare_bundle(
            symbol,
            minute,
            daily,
            actions,
            calendar,
            start,
            end,
            "1m",
            "candidate",
            {**metadata, "delisting_date": None},
            profiles,
        )
        observed = daily[(daily.tradestatus == 1) & (daily.volume > 0)].sort_values("date").copy()
        previous_close = observed.close.shift(1)
        unexplained = (observed.preclose - previous_close).abs() > float(metadata.get("price_tick", 0.01)) * 1.1
        action_days = set(actions.ex_date.dt.normalize()) if not actions.empty and "ex_date" in actions else set()
        unexplained &= observed.date.between(start, end)
        unknown = [str(day.date()) for day in observed.loc[unexplained, "date"] if day not in action_days]
        if unknown:
            raise DataQualityError("候选行情存在未解释的除权参考价变化：" + ", ".join(unknown[:20]))
        return candidate.quality

    def load_bundle(self, config: BacktestConfig, version: str | None = None):
        manifest = self.store.snapshot(config.symbol, version)
        if not manifest:
            raise DataQualityError("该股票尚无本地数据，请先下载、导入或生成演示数据")
        files, metadata = manifest["files"], manifest["metadata"]
        daily_all = self.store.read_files(files["daily"])
        if daily_all.empty:
            raise DataQualityError("本地版本没有日线")
        end = pd.Timestamp(config.end_date) if config.end_date else daily_all.date.max()
        if config.schema_version == 2:
            requested_end = end
            end = min(end, daily_all.date.max(), latest_closed_date())
            if end < pd.Timestamp(config.start_date):
                raise DataQualityError("所选区间没有本地已收盘数据")
            context = self.reference_price(config.symbol, config.start_date, manifest["version"])
            metadata = {**metadata, **context, "requested_end_date": str(requested_end.date()), "effective_end_date": str(end.date()), "prior_closes": {str(row.date.date()): float(row.close) for row in daily_all.itertuples() if row.tradestatus == 1 and row.volume > 0}}
            if end < requested_end:
                metadata["warnings"] = [*metadata.get("warnings", []), f"请求截至 {requested_end.date()}，实际本地已收盘数据截至 {end.date()}"]
            unverified = metadata.get("actions_unverified_years")
            missing_actions = any(config.start_date.year <= year <= end.year for year in unverified) if unverified is not None else metadata.get("actions_verified") is False
            if metadata.get("asset_type") == "etf" and missing_actions:
                raise DataQualityError("ETF 分红或拆分资料未完成核验，请补齐资料后回测；行情浏览仍可使用")
        elif end > daily_all.date.max():
            raise DataQualityError(f"日线完整截止到 {daily_all.date.max().date()}，请补数或缩短结束日期")
        minute = self.store.read_files(files["minute"], "timestamp", config.start_date, end)
        actions = self.store.read_files(files["actions"])
        calendar = self.store.read_files(files["calendar"])
        # A changed exchange reference price requires a modelled action, not qfq.
        observed = daily_all[(daily_all.tradestatus == 1) & (daily_all.volume > 0)].copy()
        previous_close = observed.close.shift(1)
        unexplained = (observed.preclose - previous_close).abs() > float(metadata.get("price_tick", 0.01)) * 1.1
        action_days = set(actions.ex_date.dt.normalize()) if not actions.empty else set()
        unexplained &= observed.date.between(pd.Timestamp(config.start_date), end)
        unknown = [str(day.date()) for day in observed.loc[unexplained, "date"] if day not in action_days]
        if unknown:
            raise DataQualityError("除权参考价发生变化但缺少公司行动资料：" + ", ".join(unknown[:20]))
        return prepare_bundle(config.symbol, minute, daily_all, actions, calendar, config.start_date, end, config.frequency, manifest["version"], metadata, manifest["profiles"])

    def reference_price(self, symbol, start_date, version=None):
        manifest = self.store.snapshot(normalize_symbol(symbol), version)
        if not manifest:
            raise DataQualityError("尚无本地数据，请先同步该标的")
        daily = self.store.read_files(manifest["files"]["daily"], "date", end=pd.Timestamp(start_date) - pd.Timedelta(days=1))
        available = daily[(daily.tradestatus == 1) & (daily.volume > 0) & (daily.close > 0)].sort_values("date") if not daily.empty else daily
        if available.empty:
            raise DataQualityError("开始日期前缺少有效交易日收盘价，请补齐历史或调整开始日期")
        row = available.iloc[-1]
        return {"reference_price": float(row.close), "valuation_price": float(row.close), "reference_date": str(row.date.date()), "data_version": manifest["version"], "asset_type": instrument_terms(symbol, manifest["metadata"])["asset_type"]}

    def market_bars(self, symbol, frequency="1d", start_date=None, end_date=None, before=None, limit=600, ma_periods=(5, 10, 20, 60)):
        symbol = normalize_symbol(symbol)
        manifest = self.store.snapshot(symbol)
        if not manifest:
            return {"symbol": symbol, "bars": [], "warnings": ["尚无本地行情"]}
        if frequency not in ("1d", "1m", "5m", "15m", "30m", "60m"):
            raise ValueError("不支持的行情周期")
        periods = sorted({int(p) for p in ma_periods if 1 <= int(p) <= 500})
        column = "date" if frequency == "1d" else "timestamp"
        files = manifest["files"]["daily" if frequency == "1d" else "minute"]
        # Read newest partitions first. MA lookback is included before viewport
        # slicing, while an older page never scans the entire minute archive.
        minutes = 1 if frequency == "1d" else int(frequency[:-1])
        wanted = (limit + max(periods, default=1) + 2) * minutes + (240 if frequency != "1d" else 0)
        upper = min(pd.Timestamp(end_date), pd.Timestamp(before)) if end_date and before else end_date or before
        parts, count = [], 0
        for path in sorted(files, reverse=True):
            part = self.store.read_files([path], column, end=upper)
            if part.empty:
                continue
            parts.append(part.tail(wanted))
            count += len(parts[-1])
            if count >= wanted:
                break
        frame = pd.concat(parts, ignore_index=True).sort_values(column).tail(wanted) if parts else pd.DataFrame()
        if frame.empty:
            return {"symbol": symbol, "version": manifest["version"], "bars": [], "warnings": ["该周期尚无行情，请同步或导入分钟数据"]}
        frame = frame.sort_values(column).rename(columns={column: "timestamp"})
        if frequency != "1d":
            frame["source_verified"] = frame.get("source_verified", pd.Series(any(profile.get("verified") for profile in manifest.get("profiles", {}).values()), index=frame.index)).fillna(False).astype(bool)
            if frequency == "1m":
                stamps = pd.to_datetime(frame.timestamp)
                same_session = stamps.dt.normalize().eq(stamps.shift().dt.normalize()) & stamps.dt.hour.lt(12).eq(stamps.shift().dt.hour.lt(12))
                frame["missing_minutes"] = (stamps.diff().dt.total_seconds().div(60).sub(1).where(same_session, 0).fillna(0).clip(lower=0)).astype(int)
        if frequency not in ("1d", "1m"):
            size = int(frequency[:-1])
            stamps = pd.to_datetime(frame.timestamp)
            minute = stamps.dt.hour * 60 + stamps.dt.minute
            session = np.where(minute <= 690, 570, 780)
            bucket = np.maximum(0, (minute.to_numpy() - session - 1) // size)
            group = stamps.dt.strftime("%Y-%m-%d").astype(str) + ":" + pd.Series(session, index=frame.index).astype(str) + ":" + pd.Series(bucket, index=frame.index).astype(str)
            frame = frame.groupby(group, sort=False).agg(timestamp=("timestamp", "last"), open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), source_verified=("source_verified", "all"), minute_count=("timestamp", "count")).sort_values("timestamp")
            frame["missing_minutes"] = (size - frame.minute_count).clip(lower=0)
        for period in periods:
            frame[f"ma{period}"] = frame.close.rolling(period, min_periods=period).mean()
        if start_date:
            frame = frame[frame.timestamp >= pd.Timestamp(start_date)]
        if before:
            frame = frame[frame.timestamp < pd.Timestamp(before)]
        more = len(frame) > limit
        frame = frame.tail(limit).copy()
        frame["frequency"] = frequency
        quote_frame = self.store.read_files(manifest["files"]["daily"], "date", end=frame.timestamp.iloc[-1] if not frame.empty else upper)
        quote = quote_frame.sort_values("date").iloc[-1].to_dict() if not quote_frame.empty else None
        flags = []
        if not frame.empty:
            days = pd.to_datetime(frame.timestamp).dt.normalize()
            states = dict(zip(quote_frame.date, quote_frame.tradestatus)) if not quote_frame.empty else {}
            suspended = days.map(states).eq(0)
            frame["quality_note"] = np.where(suspended, "确认停牌", "")
            if suspended.any():
                flags.append(f"显示区间含 {days[suspended].nunique()} 个确认停牌日")
            if frequency != "1d":
                unverified = ~frame.source_verified
                frame.loc[unverified, "quality_note"] += " 分钟来源未核验"
                if unverified.any():
                    flags.append(f"{int(unverified.sum())} 根 K 线的分钟来源未核验，严格回测可能按日线处理")
                if "missing_minutes" in frame:
                    incomplete = frame.missing_minutes.gt(0)
                    frame.loc[incomplete, "quality_note"] += " 此前分钟有缺口" if frequency == "1m" else " 周期内分钟不完整"
                    if incomplete.any():
                        flags.append(f"{int(incomplete.sum())} 根 K 线内缺少分钟，未补造价格")
            calendar = self.store.read_files(manifest["files"]["calendar"], "date", days.min(), days.max())
            missing_days = set(calendar.date) - set(days) if not calendar.empty else set()
            if missing_days:
                flags.append(f"显示区间缺少 {len(missing_days)} 个交易日的所选周期行情")
        if manifest["metadata"].get("actions_verified") is False:
            flags.append("公司行动资料未完成核验，补齐前仅浏览价格")
        return {"symbol": symbol, "version": manifest["version"], "frequency": frequency, "bars": frame.to_dict("records"), "quote": quote, "has_more": more, "next_before": str(frame.timestamp.iloc[0]) if more else None, "is_demo": bool(manifest["metadata"].get("is_demo")), "warnings": list(dict.fromkeys([*manifest["metadata"].get("warnings", []), *flags])), "quality_flags": flags, "instrument": instrument_terms(symbol, manifest["metadata"])}

    def coverage(self, symbol: str):
        symbol = normalize_symbol(symbol)
        manifest = self.store.snapshot(symbol)
        if not manifest:
            return {"symbol": symbol, "version": None, "rows": 0, "days": [], "profiles": {}, "warnings": ["尚未下载本地数据"], "bytes": 0}
        cache = self.store.state("coverage:" + manifest["version"])
        if cache:
            return cache
        daily = self.store.read_files(manifest["files"]["daily"])
        minute = self.store.read_files(manifest["files"]["minute"])
        calendar = self.store.read_files(manifest["files"]["calendar"])
        actions = self.store.read_files(manifest["files"]["actions"])
        if daily.empty:
            return {"symbol": symbol, "version": manifest["version"], "rows": len(minute), "days": [], "warnings": ["日线缺失"], "profiles": manifest["profiles"], "bytes": 0}
        try:
            bundle = prepare_bundle(symbol, minute, daily, actions, calendar, daily.date.min(), daily.date.max(), "1m", manifest["version"], {**manifest["metadata"], "delisting_date": None}, manifest["profiles"])
            quality = bundle.quality
            warnings = bundle.warnings
        except DataQualityError as exc:
            quality = {"days": [], "minute_days": 0, "daily_days": len(daily), "minute_ratio": 0}
            warnings = [str(exc)]
        result = {**quality, "symbol": symbol, "version": manifest["version"], "rows": len(minute), "first_date": str(daily.date.min().date()), "last_date": str(daily.date.max().date()), "profiles": manifest["profiles"], "warnings": warnings + manifest["metadata"].get("warnings", []), "is_demo": manifest["metadata"].get("is_demo", False), "bytes": sum((self.store.root / path).stat().st_size for paths in manifest["files"].values() for path in paths)}
        self.store.state("coverage:" + manifest["version"], result)
        return result

    def import_files(self, payload: dict, progress=lambda *_: None, cancelled=lambda: False):
        symbol = normalize_symbol(payload["symbol"])
        def read(path):
            path = Path(path).expanduser().resolve()
            if not path.is_file():
                raise ValueError("导入文件不存在：" + str(path))
            return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        with FileLock(str(self.store.root / f"sync-{symbol}.lock"), timeout=60), self.store.lock:
            previous = self.store.snapshot(symbol)
            files = previous["files"] if previous else {}
            if payload.get("daily_path"):
                raw_daily = read(payload["daily_path"])
                csv_listing = None
                if "listing_date" in raw_daily:
                    listings = pd.to_datetime(raw_daily.listing_date.dropna(), errors="raise").dt.normalize().unique()
                    if len(listings) > 1:
                        raise DataQualityError("单只股票的上市日期 listing_date 必须一致")
                    if len(listings):
                        csv_listing = str(pd.Timestamp(listings[0]).date())
                required = {"preclose", "tradestatus"}
                if not required.issubset(raw_daily.columns) or (instrument_terms(symbol)["asset_type"] == "stock" and not ({"is_st", "isST"} & set(raw_daily.columns))):
                    raise DataQualityError("准确导入的日线需包含 preclose、tradestatus 和 is_st/isST 历史状态字段")
                daily = canonical_frame(raw_daily, daily=True)
            else:
                daily = self.store.read_files(files.get("daily", []))
            if daily.empty:
                raise DataQualityError("首次导入必须提供不复权日线 CSV/Parquet")
            minute = self.store.read_files(files.get("minute", []))
            profiles = dict(previous.get("profiles", {})) if previous else {}
            if payload.get("minute_path"):
                raw = canonical_frame(read(payload["minute_path"]), source="import")
                declared = payload.get("label_convention")
                if declared not in (None, "auto", "start", "end"):
                    raise DataQualityError("时间标签应为 auto/start/end")
                profiles["import"] = infer_profile(raw, daily, "import", declared=declared if declared in ("start", "end") else None, price_tick=instrument_terms(symbol)["price_tick"])
                imported = align_minutes(raw, profiles["import"]["convention"], profiles["import"]["volume_multiplier"])
                imported["source_verified"] = profiles["import"]["verified"]
                minute = merge_frames(minute, imported, "timestamp")
            actions = normalize_actions(read(payload["actions_path"])) if payload.get("actions_path") else self.store.read_files(files.get("actions", []))
            metadata = dict(previous.get("metadata", {})) if previous else {}
            known = self.store.instruments(symbol, limit=1)
            metadata = {**(known[0] if known else {}), **metadata}
            if payload.get("etf_category"):
                if payload["etf_category"] not in {"stock", "bond", "crossborder", "commodity", "money"}:
                    raise DataQualityError("无效的 ETF 类别")
                metadata["etf_category"] = payload["etf_category"]
                metadata.pop("settlement", None)
            metadata = instrument_terms(symbol, metadata)
            metadata["listing_date"] = payload.get("listing_date") or (csv_listing if payload.get("daily_path") else None) or metadata.get("listing_date") or (known[0].get("listing_date") if known else None)
            # Imported files supply their own explicit calendar; an optional calendar
            # file can include expected no-bar days so missing daily rows are detected.
            calendar = read(payload["calendar_path"]) if payload.get("calendar_path") else pd.DataFrame({"date": pd.bdate_range(daily.date.min(), daily.date.max())})
            if not payload.get("calendar_path"):
                try:
                    import baostock as bs
                    import contextlib, io
                    with contextlib.redirect_stdout(io.StringIO()):
                        bs.login()
                        result = bs.query_trade_dates(str(daily.date.min().date()), str(daily.date.max().date()))
                        rows = []
                        while result.next():
                            rows.append(result.get_row_data())
                        bs.logout()
                    observed = pd.DataFrame(rows, columns=result.fields)
                    if observed.empty:
                        raise RuntimeError("未获取到日历")
                    calendar = pd.DataFrame({"date": pd.to_datetime(observed.loc[observed.is_trading_day == "1", "calendar_date"])})
                except Exception as exc:
                    raise DataQualityError("导入必须提供真实交易日历文件(calendar_path)，自动日历获取失败：" + str(exc))
            calendar["date"] = pd.to_datetime(calendar["date"]).dt.normalize()
            metadata.update({"actions_verified": bool(payload.get("actions_path")) or bool(metadata.get("actions_verified")), "is_demo": False, "imported": True})
            if payload.get("actions_path"):
                metadata["actions_unverified_years"] = []
            self.store.put_instrument(symbol, **{key: value for key, value in metadata.items() if key != "symbol"})
            self.store.add_watch(symbol)
            if cancelled():
                raise InterruptedError("任务已取消")
            progress(0.8, "发布导入数据版本")
            manifest = self.store.publish(symbol, minute, daily, actions, calendar, metadata, profiles)
            return {"symbol": symbol, "version": manifest["version"], "profiles": profiles}

    def demo(self, symbol="000001.SZ", days=780, progress=lambda *_: None, cancelled=lambda: False):
        symbol = normalize_symbol(symbol)
        if self.store.snapshot(symbol) and not self.store.snapshot(symbol)["metadata"].get("is_demo"):
            raise ValueError("该股票已有真实数据，不能用演示数据覆盖；请选择无数据股票或独立数据目录")
        days = max(20, min(int(days), 1600))
        calendar_days = pd.bdate_range(end=latest_closed_date(), periods=days)
        count = days * 240
        rng = np.random.default_rng(7)
        minute_index = np.arange(count)
        log_price = np.log(12) + 0.000001 * minute_index + 0.11 * np.sin(minute_index / 1100) + 0.055 * np.sin(minute_index / 160) + np.cumsum(rng.normal(0, 0.00025, count))
        close = np.round(np.exp(log_price), 2)
        opening = np.r_[close[0], close[:-1]]
        volume = rng.integers(2000, 80_000, count)
        timestamps = pd.DatetimeIndex(np.concatenate([minute_slots(day).to_numpy() for day in calendar_days]))
        minute = pd.DataFrame({"timestamp": timestamps, "open": opening, "high": np.maximum(opening, close) + 0.01, "low": np.minimum(opening, close) - 0.01, "close": close, "volume": volume, "turnover": volume * close, "source": "demo", "source_verified": True})
        daily = minute.groupby(minute.timestamp.dt.normalize()).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), turnover=("turnover", "sum")).reset_index(names="date")
        daily["preclose"] = daily.close.shift(1).fillna(daily.iloc[0].open)
        daily["tradestatus"], daily["is_st"] = 1, False
        if days >= 100:
            # Demonstrate both bounded padding and whole-day fallback explicitly.
            minute = minute.drop(index=[50 * 240 + 90, 50 * 240 + 91, *range(90 * 240 + 40, 90 * 240 + 100)]).reset_index(drop=True)
        calendar = pd.DataFrame({"date": calendar_days})
        actions = normalize_actions(pd.DataFrame())
        metadata = {"name": "演示 · " + symbol, "listing_date": str(calendar_days[0].date()), "board": "star" if symbol.startswith("688") else "chinext" if symbol.startswith("3") else "main", "actions_verified": True, "is_demo": True}
        profiles = {"demo": {"verified": True, "convention": "end", "volume_multiplier": 1, "reason": "确定性生成的模拟数据；不是历史行情"}}
        progress(0.6, "保存已标记的演示分钟与日线数据")
        with FileLock(str(self.store.root / f"sync-{symbol}.lock"), timeout=60), self.store.lock:
            if self.store.snapshot(symbol) and not self.store.snapshot(symbol)["metadata"].get("is_demo"):
                raise ValueError("该标的已有真实数据，不能用演示数据覆盖")
            manifest = self.store.publish(symbol, minute, daily, actions, calendar, metadata, profiles)
            self.store.add_watch(symbol, metadata["name"])
        return {"symbol": symbol, "version": manifest["version"], "is_demo": True, "days": days, "message": "演示数据已生成；包含短缺口与日线降级示例"}
