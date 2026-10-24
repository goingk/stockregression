from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from filelock import FileLock, Timeout

import numpy as np
import pandas as pd

from .models import BacktestConfig, normalize_symbol
from .providers import fetch_instruments, fetch_tdx, fetch_sina, fetch_reference_5m, latest_closed_date, normalize_actions
from .source_health import SOURCE_POLICY_VERSION, SourceRouter, fetch_auxiliary_routed, market_for_symbol
from .quality import QUALITY_RULES_VERSION, DataQualityError, _closing_auction_placeholders, align_minutes, canonical_frame, infer_profile, minute_slots, prepare_bundle
from .data_diagnostics import add_source_events, derive_source_spans, project_data_diagnostics
from .storage import Store
from .rules import instrument_terms
from .settings import Settings, load_settings


def merge_frames(old: pd.DataFrame, new: pd.DataFrame, key: str) -> pd.DataFrame:
    parts = [part for part in (old, new) if not part.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(key, kind="stable").drop_duplicates(key, keep="last").reset_index(drop=True)


def _verified_closing_auction_slots(symbol: str, minute: pd.DataFrame) -> set[pd.Timestamp]:
    """Return only missing 14:58/14:59 slots proven by verified close anchors."""
    if minute is None or minute.empty or "timestamp" not in minute or "close" not in minute:
        return set()
    placeholders: set[pd.Timestamp] = set()
    frame = minute.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="coerce")
    frame = frame[frame.timestamp.notna()]
    for day, rows in frame.groupby(frame.timestamp.dt.normalize(), sort=False):
        aligned = rows.drop_duplicates("timestamp", keep="last").set_index("timestamp").reindex(minute_slots(day))
        missing_mask = aligned.close.isna().to_numpy()
        mask = _closing_auction_placeholders(symbol, day, aligned, missing_mask)
        placeholders.update(pd.Timestamp(slot) for slot in aligned.index[mask])
    return placeholders


def _market_bar_group_keys(stamps: pd.DatetimeIndex, size: int) -> pd.Series:
    minute = stamps.hour * 60 + stamps.minute
    session = np.where(minute <= 690, 570, 780)
    bucket = np.maximum(0, (minute - session - 1) // size)
    return pd.Series(
        stamps.strftime("%Y-%m-%d").astype(str)
        + ":"
        + pd.Series(session, index=range(len(stamps))).astype(str).to_numpy()
        + ":"
        + pd.Series(bucket, index=range(len(stamps))).astype(str).to_numpy()
    )


def _minute_gap_counts(stamps: pd.Series, auction_placeholders: set[pd.Timestamp]) -> np.ndarray:
    values = pd.DatetimeIndex(pd.to_datetime(stamps))
    gaps = np.zeros(len(values), dtype=int)
    for index in range(1, len(values)):
        previous, current = values[index - 1], values[index]
        if previous.normalize() != current.normalize() or (previous.hour < 12) != (current.hour < 12):
            continue
        gap = max(0, int((current - previous).total_seconds() // 60) - 1)
        if gap:
            for minute in (58, 59):
                slot = previous.normalize() + pd.Timedelta(hours=14, minutes=minute)
                if previous < slot < current and slot in auction_placeholders:
                    gap -= 1
        gaps[index] = max(0, gap)
    return gaps


def _verified_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "verified"}
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return False


class DataService:
    def __init__(self, store: Store, settings: Settings | None = None):
        self.store = store
        self.settings = settings or load_settings()
        self.market_settings = self.settings.market_data
        self._sync_context = threading.local()

    def _sync_stage(self, symbol, **values):
        run_key = getattr(self._sync_context, "run_key", None)
        if run_key:
            self.store.sync_item(run_key, symbol, **values)

    def refresh_instruments(self, progress=lambda *_: None, cancelled=lambda: False):
        progress(0.1, "刷新 AKShare 股票与新浪 ETF 证券目录")
        records = fetch_instruments(
            network_timeout=self.market_settings.network_timeout_seconds,
        )
        excluded = 0
        for record in records:
            if cancelled():
                raise InterruptedError("任务已取消")
            existing = self.store.instrument(record["symbol"])
            if existing and existing.get("delisting_date"):
                excluded += 1
                continue
            self.store.put_instrument(**record)
        warnings = sorted({warning for record in records for warning in record.get("catalog_warnings", [])})
        self.store.state("catalog:latest", {"count": len(records), "warnings": warnings})
        return {
            "count": len(records) - excluded,
            "excluded": excluded,
            "warnings": warnings,
            "message": (
                f"已更新 {len(records) - excluded} 只股票及 ETF；"
                f"跳过 {excluded} 只已退市或永久排除的标的"
            ),
        }

    def sync(self, symbols: list[str], mode: str, progress=lambda *_: None, cancelled=lambda: False, scope: str | None = None, resume: bool = False, retry_failed: bool = False, workers: int = 2):
        """Compatibility entry point that always writes the managed ETF package."""
        from .market_data import MarketDataService
        from .market_data.sync import MarketDataSyncService

        scope = scope or ("selected" if symbols else "watchlist")
        if scope == "market":
            if symbols:
                raise ValueError("全市场范围不能同时指定代码")
            symbols = MarketDataService(self.store, settings=self.settings).repository.symbols_for_timeframe("1day")
        elif not symbols:
            symbols = [row["symbol"] for row in self.store.watches()]
        if not symbols:
            return {"status": "completed", "symbols": [], "failures": [], "skipped": [],
                    "message": "没有需要同步的标的"}
        timeframe = "1day" if mode == "history" else mode
        selected = sorted(set(normalize_symbol(value) for value in symbols))
        runs = []
        for offset, value in enumerate((["1day"] if mode == "history" else ["1day", "1min"])):
            runs.append(MarketDataSyncService(MarketDataService(self.store, settings=self.settings)).sync(
                selected, value, "incremental" if mode == "daily" else mode,
                resume=resume, retry_failed=retry_failed, workers=workers,
                progress=lambda fraction, message, offset=offset: progress(
                    (offset + float(fraction)) / (1 if mode == "history" else 2), message
                ), cancelled=cancelled,
            ))
        failures = [failure for run in runs for failure in run.get("failures", [])]
        results = [result for run in runs for result in run.get("symbols", [])]
        skipped = [result for run in runs for result in run.get("skipped", [])]
        return {
            "status": "completed" if not failures else "partial" if results else "failed",
            "runs": runs, "symbols": results, "failures": failures, "skipped": skipped,
            "message": f"完成 {len(results)} 项，失败 {len(failures)} 项",
        }


    def current_version_if_ready(self, config: BacktestConfig) -> str | None:
        """Return an immutable submission-time version only when it needs no refresh."""
        manifest = self.store.snapshot(config.symbol)
        if not manifest or not manifest.get("market_data"):
            return None
        if manifest.get("market_data"):
            from .market_data import MarketDataService
            readiness = MarketDataService(self.store, settings=self.settings).readiness(config)
            return manifest["version"] if readiness.get("ready") else None
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
        manifest = self.store.snapshot(config.symbol, version)
        if not manifest or not manifest.get("market_data"):
            raise DataQualityError("DATA_VERSION_NOT_FOUND: 当前没有该行情数据版本；旧格式快照已停止兼容")
        if manifest and manifest.get("market_data"):
            from .market_data import MarketDataService
            market = MarketDataService(self.store, settings=self.settings)
            readiness = market.readiness(config, manifest["version"])
            if not readiness.get("ready"):
                details = "；".join(item.get("message", "") for item in readiness.get("gaps", []))
                raise DataQualityError("本地数据尚不能用于回测：" + (details or "存在待核验事项"))
            progress(0.1, "读取本地行情数据库并锁定版本")
            return market.load_backtest_bundle(config, manifest["version"])
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

        # Calculation paths are strictly local. Missing data is reported with
        # actionable ranges; preparation is an explicit data-manager task.
        try:
            local_bundle, local_gaps, local_error = self._probe_bundle(
                config, (manifest or {}).get("version"), check_minutes
            )
        except DataQualityError as exc:
            local_bundle, local_gaps, local_error = None, [], str(exc)
        local_gaps = self._ignore_demo_gaps(manifest or {}, local_gaps)
        if local_bundle is not None and not local_gaps and not local_error:
            return self._add_preparation_warnings(local_bundle, [])
        details = [str(item.get("reason") or item.get("kind") or item) if isinstance(item, dict) else str(item) for item in local_gaps]
        if local_error:
            details.insert(0, local_error)
        raise DataQualityError(
            "所选区间缺少本地行情或质量尚待核验；请先在“数据管理”中准备对应周期："
            + ("；".join(details[:8]) if details else "没有可用的本地数据版本")
        )

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
        self._acquire_prepare_lock(prepare_lock, cancelled, timeout=self.market_settings.prepare_lock_timeout_seconds)
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
                events = []
                if suppress and record.get("error"):
                    events.append({"kind": "source_error", "message": str(record["error"]), "status": "historical"})
                return self._add_preparation_warnings(bundle, warnings, events)

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
                "retry_after": time.time() + self.market_settings.retry_after_seconds if sync_error or bundle is None else None,
                "error": sync_error or load_error,
            }
            self.store.state(key, state)
            if bundle is None:
                raise DataQualityError(self._preparation_error(config.symbol, sync_error or load_error, remaining))
            warnings = self._preparation_messages(remaining)
            events = [{"kind": "source_error", "message": sync_error, "status": "historical"}] if sync_error else []
            progress(0.72, "本地行情已校验；开始计算")
            return self._add_preparation_warnings(bundle, warnings, events)
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
        source_versions = {
            "source_policy": SOURCE_POLICY_VERSION,
            "preparation": 3,
            "quality_rules": QUALITY_RULES_VERSION,
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
            if config.schema_version in (2, 3) and target_end > daily.date.max() and not any(token in message for token in ("ETF 分红", "参考价发生变化", "没有可回测交易日")):
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
        listing_date = manifest.get("metadata", {}).get("listing_date") if manifest else None
        if listing_date:
            start = max(start, pd.Timestamp(listing_date).normalize())
        target_end = min(pd.Timestamp(config.end_date or latest_closed_date()).normalize(), pd.Timestamp(latest_closed_date()).normalize())
        if daily.date.min().normalize() > start:
            gaps.append(f"历史日线起点缺失：请求 {start.date()}，本地 {daily.date.min().date()}")
        if daily.date.max().normalize() < target_end:
            gaps.append(f"日线尾部缺失：请求至 {target_end.date()}，本地至 {daily.date.max().date()}")
        # A stale metadata watermark is not evidence of a gap. Validate against
        # the selected exchange calendar and the current prepare_bundle report.
        if not bundle.calendar.empty and not daily.empty:
            expected = pd.DatetimeIndex(bundle.calendar).normalize().unique()
            observed = pd.DatetimeIndex(pd.to_datetime(daily.date)).normalize().unique()
            missing = expected.difference(observed)
            if len(missing):
                gaps.append("日线交易日缺失：" + ", ".join(str(value.date()) for value in missing[:12]) + ("…" if len(missing) > 12 else ""))
        if check_minutes and config.frequency != "1d":
            daily_days = [row for row in bundle.quality.get("days", []) if row.get("mode") == "daily" and row.get("download_needed", True)]
            if daily_days:
                dates = [row.get("date") for row in daily_days]
                gaps.append("分钟数据待补：" + ", ".join(str(value) for value in dates[:12]) + ("…" if len(dates) > 12 else ""))
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
    def _add_preparation_warnings(bundle, warnings, source_events=()):
        metadata = dict(bundle.metadata or {})
        if source_events:
            metadata = add_source_events(metadata, source_events)
        dates = pd.DatetimeIndex(bundle.calendar) if len(bundle.calendar) else pd.DatetimeIndex([])
        start = dates.min() if len(dates) else metadata.get("effective_start_date")
        end = dates.max() if len(dates) else metadata.get("effective_end_date")
        projection = project_data_diagnostics(metadata, bundle.quality, start, end)
        current = list(projection["current_warnings"])
        current.extend(str(message) for message in warnings if message)
        current = list(dict.fromkeys(current))
        metadata["diagnostics"] = projection["diagnostics"]
        metadata["source_events"] = projection["source_events"]
        metadata["warnings"] = current
        bundle.metadata = metadata
        bundle.warnings = current
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
        profiles = dict(previous.get("profiles", {})) if previous else {}
        known = self.store.instrument(symbol) or {}
        if old_daily.empty and previous:
            # A long-offline database must resume from its own watermark.
            paths = old_files.get("daily", [])
            old_daily = self.store.read_files(paths[-1:] if paths else [])
        start = (old_daily.date.max() - pd.Timedelta(days=30)) if mode != "history" and not old_daily.empty else pd.Timestamp(known.get("listing_date") or "1990-01-01")
        progress(0.25, "获取不复权日线、交易日历和公司行动")
        daily, actions, calendar, metadata = fetch_auxiliary_routed(
            self.store, symbol, start, end, cancelled
        )
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
        metadata["source_policy_version"] = SOURCE_POLICY_VERSION
        metadata["is_demo"] = False
        if metadata.get("listing_date"):
            calendar = calendar[calendar.date >= pd.Timestamp(metadata["listing_date"])].reset_index(drop=True)
        if metadata.get("delisting_date"):
            calendar = calendar[calendar.date < pd.Timestamp(metadata["delisting_date"])].reset_index(drop=True)
        progress(0.55, "采集近期分钟窗口并核对标签与单位")
        recent = pd.DataFrame()
        raw_cache = []
        earliest = end - pd.Timedelta(days=14) if mode == "daily" and previous else None
        try:
            recent_raw, recent_source, route = SourceRouter(self.store, settings=self.settings).call(
                "recent_minute", market_for_symbol(symbol),
                {
                    "sina": lambda: fetch_sina(
                        symbol,
                        network_timeout=self.market_settings.network_timeout_seconds,
                    ),
                    "tdx": lambda: fetch_tdx(
                        symbol, earliest, cancelled,
                        network_timeout=self.market_settings.network_timeout_seconds,
                    ),
                },
            )
            if recent_source == "sina":
                reference = fetch_reference_5m(symbol, recent_raw.timestamp.max().normalize() - pd.Timedelta(days=21), recent_raw.timestamp.max().normalize())
                profiles["sina"] = infer_profile(recent_raw, daily, "sina", reference=reference, reference_frequency="5m", price_tick=metadata.get("price_tick", .01))
                recent = align_minutes(recent_raw, profiles["sina"]["convention"], profiles["sina"]["volume_multiplier"])
                recent["source_verified"] = profiles["sina"]["verified"]
            else:
                profiles["tdx"] = infer_profile(recent_raw, daily, "tdx", price_tick=metadata.get("price_tick", .01))
                recent = align_minutes(recent_raw, profiles["tdx"]["convention"], profiles["tdx"]["volume_multiplier"])
                recent["source_verified"] = profiles["tdx"]["verified"]
            raw_cache.append((recent_source, recent_raw))
            metadata["recent_minute_route"] = route
            if route.get("fallback"):
                warnings.append(route.get("reason") or "可选分钟来源不可用，已回退默认来源")
        except InterruptedError:
            raise
        except Exception as exc:
            warnings.append("近期分钟来源不可用：" + str(exc))
            source_errors.append("近期分钟来源不可用：" + str(exc))
        minute = recent
        if not minute.empty:
            minute = minute[minute.timestamp < end + pd.Timedelta(days=1)].copy()
        source_warnings = [*warnings, *(metadata.get("warnings") or [])]
        source_warnings.append("新浪 1 分钟只覆盖近期窗口；更早历史分钟须从本地数据包登记或导入")
        if cancelled():
            raise InterruptedError("任务已取消")
        progress(0.80, "检查日线、日历、公司行动和分钟质量")
        # The increment being fetched is not the full local dataset. Read and
        # merge the active snapshot so diagnostics and watermarks reflect the
        # actual import+provider mixture that will be published.
        previous_daily = self.store.read_files(old_files.get("daily", [])) if previous else pd.DataFrame()
        previous_minute = self.store.read_files(old_files.get("minute", [])) if previous else pd.DataFrame()
        previous_actions = self.store.read_files(old_files.get("actions", [])) if previous else pd.DataFrame()
        previous_calendar = self.store.read_files(old_files.get("calendar", [])) if previous else pd.DataFrame()
        full_daily = merge_frames(previous_daily, incoming_daily, "date")
        full_minute = merge_frames(previous_minute, minute, "timestamp")
        full_actions = merge_frames(previous_actions, actions, "ex_date")
        full_calendar = merge_frames(previous_calendar, calendar, "date")
        diagnostic_metadata = dict(metadata)
        prior_metadata = (previous or {}).get("metadata", {})
        diagnostic_metadata["source_events"] = list(prior_metadata.get("source_events") or [])
        diagnostic_metadata["warnings"] = [*(prior_metadata.get("warnings") or []), *source_warnings]
        diagnostic_metadata["source_errors"] = list(dict.fromkeys([*(prior_metadata.get("source_errors") or []), *source_errors]))
        candidate_quality = self._validate_candidate(symbol, full_daily, full_minute, full_actions, full_calendar, diagnostic_metadata, profiles)
        diagnostic_metadata["diagnostics"] = {
            "source_spans": derive_source_spans(full_minute, profiles),
        }
        projection = project_data_diagnostics(
            diagnostic_metadata,
            candidate_quality,
            full_calendar.date.min() if not full_calendar.empty else None,
            full_calendar.date.max() if not full_calendar.empty else None,
            profiles,
        )
        metadata["diagnostics"] = projection["diagnostics"]
        metadata["source_events"] = projection["source_events"]
        metadata["warnings"] = projection["current_warnings"]
        metadata.pop("source_errors", None)
        metadata["daily_through"] = str(full_daily.date.max().date()) if not full_daily.empty else None
        metadata["minute_through"] = str(full_minute.timestamp.max()) if not full_minute.empty else None
        metadata["minute_status"] = self._minute_status(full_minute, candidate_quality)
        metadata["quality_summary"] = self._quality_summary(candidate_quality)
        metadata["quality_warnings"] = list(projection["current_warnings"])
        minute_status = metadata["minute_status"]
        self._sync_stage(symbol, minute_status=minute_status, minute_through=metadata["minute_through"])
        for source, raw_frame in raw_cache:
            raw_folder = self.store.root / "raw" / source / symbol
            raw_folder.mkdir(parents=True, exist_ok=True)
            raw_frame.to_parquet(raw_folder / "latest.parquet", index=False)
        progress(0.85, "发布已校验的不可变分区和数据版本")
        manifest = self.store.publish(symbol, minute, incoming_daily, actions, calendar, metadata, profiles, incremental=previous is not None)
        self.store.put_instrument(symbol, **{key: value for key, value in metadata.items() if key != "symbol"})
        progress(1, "完成；质量报告已可查看")
        return {"symbol": symbol, "version": manifest["version"], "last_date": metadata["daily_through"], "daily_status": "completed", "minute_status": minute_status, "minute_through": metadata["minute_through"], "rows": len(minute), "warnings": metadata["warnings"], "source_errors": list(dict.fromkeys(source_errors)), "source_events": metadata["source_events"], "diagnostics": metadata["diagnostics"], "profiles": profiles}

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
        from .corporate_actions import action_issues
        issues = action_issues(
            actions, daily, start, end,
            price_tick=float(metadata.get("price_tick", 0.01)),
        )
        if issues:
            raise DataQualityError("候选行情公司行动异常：" + "；".join(issues[:8]))
        return candidate.quality

    def load_bundle(
        self, config: BacktestConfig, version: str | None = None, *,
        summary_only: bool = False,
    ):
        manifest = self.store.snapshot(config.symbol, version)
        if not manifest:
            raise DataQualityError("DATA_VERSION_NOT_FOUND: 该数据版本不存在或已清理")
        if manifest.get("market_data"):
            from .market_data import MarketDataService
            return MarketDataService(self.store, settings=self.settings).load_backtest_bundle(
                config, version or manifest["version"], summary_only=summary_only,
            )
        raise DataQualityError("旧 Parquet 快照已取消兼容，请从当前 ETF 数据包读取行情")
        files, metadata = manifest["files"], manifest["metadata"]
        daily_all = self.store.read_files(files["daily"])
        if daily_all.empty:
            raise DataQualityError("本地版本没有日线")
        end = pd.Timestamp(config.end_date) if config.end_date else daily_all.date.max()
        if config.schema_version in (2, 3):
            requested_end = end
            closed_through = pd.Timestamp(latest_closed_date()).normalize()
            expected_end = min(end.normalize(), closed_through)
            end = min(end, daily_all.date.max(), closed_through)
            if end < pd.Timestamp(config.start_date):
                raise DataQualityError("所选区间没有本地已收盘数据")
            context = self.reference_price(config.symbol, config.start_date, manifest["version"])
            metadata = {**metadata, **context, "requested_end_date": str(requested_end.date()), "effective_end_date": str(end.date()), "prior_closes": {str(row.date.date()): float(row.close) for row in daily_all.itertuples() if row.tradestatus == 1 and row.volume > 0}}
            # Old request-scoped messages can remain in immutable snapshots.
            # Replace them from today's effective range; a weekend after the
            # latest closed session is not a missing-data warning.
            metadata["warnings"] = [warning for warning in (metadata.get("warnings") or []) if not str(warning).startswith("请求截至 ")]
            if end < expected_end:
                metadata["warnings"] = [*metadata.get("warnings", []), f"请求截至 {requested_end.date()}，实际本地已收盘数据截至 {end.date()}"]
        elif end > daily_all.date.max():
            raise DataQualityError(f"日线完整截止到 {daily_all.date.max().date()}，请补数或缩短结束日期")
        minute = self.store.read_files(files["minute"], "timestamp", config.start_date, end)
        actions = self.store.read_files(files["actions"])
        calendar = self.store.read_files(files["calendar"])
        # A changed exchange reference price requires a modelled action, not qfq.
        observed = daily_all[(daily_all.tradestatus == 1) & (daily_all.volume > 0)].copy()
        previous_close = observed.close.shift(1)
        unexplained = (observed.preclose - previous_close).abs() > float(metadata.get("price_tick", 0.01)) * 1.1
        from .corporate_actions import action_issues
        unknown = action_issues(actions, daily_all, config.start_date, end, settlement=True, price_tick=float(metadata.get("price_tick", 0.01)))
        if unknown:
            raise DataQualityError("除权参考价发生变化但缺少公司行动资料：" + ", ".join(unknown[:20]))
        bundle = prepare_bundle(config.symbol, minute, daily_all, actions, calendar, config.start_date, end, config.frequency, manifest["version"], metadata, manifest["profiles"])
        projected_metadata = dict(bundle.metadata or {})
        projected_metadata["daily_through"] = str(daily_all.date.max().date()) if not daily_all.empty else None
        projected_metadata["minute_through"] = str(minute.timestamp.max()) if not minute.empty else None
        projected_metadata["minute_status"] = self._minute_status(minute, bundle.quality)
        projected_metadata["quality_summary"] = self._quality_summary(bundle.quality)
        diagnostic_metadata = dict(projected_metadata)
        stored_diagnostics = dict(diagnostic_metadata.get("diagnostics") or {})
        stored_diagnostics["source_spans"] = derive_source_spans(minute, manifest.get("profiles"), config.start_date, end)
        diagnostic_metadata["diagnostics"] = stored_diagnostics
        projection = project_data_diagnostics(diagnostic_metadata, bundle.quality, config.start_date, end, manifest.get("profiles"))
        projected_metadata["diagnostics"] = projection["diagnostics"]
        projected_metadata["source_events"] = projection["source_events"]
        projected_metadata["warnings"] = projection["current_warnings"]
        bundle.metadata = projected_metadata
        bundle.warnings = list(projection["current_warnings"])
        return bundle

    @staticmethod
    def _quality_summary(quality):
        return {key: quality.get(key) for key in ("minute_days", "daily_days", "suspended_days", "total_days", "minute_ratio", "pending_calibration_days", "synthetic_bars", "auction_placeholders")}

    @staticmethod
    def _minute_status(minute, quality=None):
        if minute is None or minute.empty:
            return "unavailable"
        if "source_verified" in minute and minute.source_verified.fillna(False).map(lambda value: str(value).lower() in {"true", "1"} if isinstance(value, str) else bool(value)).any():
            return "available"
        if quality and int(quality.get("minute_days", 0) or 0) > 0:
            return "available"
        return "unverified"

    def reference_price(self, symbol, start_date, version=None):
        manifest = self.store.snapshot(normalize_symbol(symbol), version, persist=False)
        if not manifest:
            raise DataQualityError("尚无本地数据，请先同步该标的")
        if manifest.get("market_data"):
            from .market_data.repository import MarketDataRepository
            daily_version = int((manifest.get("market_versions", {}).get("1day") or {}).get("version"))
            daily = MarketDataRepository(self.store.root).bars(
                symbol, "1day",
                end=pd.Timestamp(start_date) - pd.Timedelta(days=1), version=daily_version,
            ).rename(columns={"timestamp": "date"})
        else:
            raise DataQualityError("旧 Parquet 快照已取消兼容，不能作为参考价来源")
        available = daily[(daily.tradestatus == 1) & (daily.volume > 0) & (daily.close > 0)].sort_values("date") if not daily.empty else daily
        if available.empty:
            raise DataQualityError("开始日期前缺少有效交易日收盘价，请补齐历史或调整开始日期")
        row = available.iloc[-1]
        return {"reference_price": float(row.close), "valuation_price": float(row.close), "reference_date": str(row.date.date()), "data_version": manifest["version"], "asset_type": instrument_terms(symbol, manifest["metadata"])["asset_type"]}

    def market_bars(self, symbol, frequency="1d", start_date=None, end_date=None, before=None, limit=600, ma_periods=(5, 10, 20, 60)):
        from .market_data import MarketDataService
        from .market_data.timeframes import normalize_timeframe
        return MarketDataService(self.store, settings=self.settings).market_bars(
            symbol, normalize_timeframe(frequency), start=start_date, end=end_date,
            before=before, limit=limit, ma_periods=ma_periods,
        )
        symbol = normalize_symbol(symbol)
        manifest = self.store.snapshot(symbol)
        if not manifest:
            projection = project_data_diagnostics({}, {}, start_date, end_date)
            return {"symbol": symbol, "bars": [], "warnings": ["尚无本地行情"], "diagnostics": projection["diagnostics"], "source_events": []}
        snapshot_metadata = dict(manifest["metadata"])
        saved_diagnostics = snapshot_metadata.get("diagnostics") or {}
        if saved_diagnostics.get("quality_rules_version") != QUALITY_RULES_VERSION:
            # Legacy snapshots may still carry a structurally valid diagnostics
            # block computed under older rules. Rebuild the one-per-version
            # coverage projection before showing its status in the chart.
            coverage = self.coverage(symbol)
            snapshot_metadata["diagnostics"] = coverage.get("diagnostics") or {}
            snapshot_metadata["source_events"] = coverage.get("source_events") or []
            snapshot_metadata["warnings"] = coverage.get("warnings") or []
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
            existing = snapshot_metadata.get("diagnostics") or {}
            projection = project_data_diagnostics(
                snapshot_metadata, existing,
                existing.get("checked_start"), existing.get("checked_end"), manifest.get("profiles"),
            )
            warnings = list(dict.fromkeys([*projection["current_warnings"], "该周期尚无行情，请同步或导入分钟数据"]))
            return {"symbol": symbol, "version": manifest["version"], "bars": [], "warnings": warnings, "diagnostics": projection["diagnostics"], "source_events": projection["source_events"]}
        frame = frame.sort_values(column).rename(columns={column: "timestamp"})
        auction_placeholders: set[pd.Timestamp] = set()
        if frequency != "1d":
            frame["source_verified"] = frame.get("source_verified", pd.Series(any(profile.get("verified") for profile in manifest.get("profiles", {}).values()), index=frame.index)).map(_verified_flag)
            auction_placeholders = _verified_closing_auction_slots(symbol, frame)
            if frequency == "1m":
                frame["missing_minutes"] = _minute_gap_counts(frame.timestamp, auction_placeholders)
        if frequency not in ("1d", "1m"):
            size = int(frequency[:-1])
            stamps = pd.to_datetime(frame.timestamp)
            group = _market_bar_group_keys(pd.DatetimeIndex(stamps), size)
            frame = frame.groupby(group.to_numpy(), sort=False).agg(timestamp=("timestamp", "last"), open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), source_verified=("source_verified", "all"), minute_count=("timestamp", "count")).sort_values("timestamp")
            if auction_placeholders:
                placeholder_keys = _market_bar_group_keys(pd.DatetimeIndex(sorted(auction_placeholders)), size)
                placeholder_counts = pd.Series(1, index=placeholder_keys.to_numpy()).groupby(level=0).sum()
                frame["auction_placeholder_count"] = frame.index.map(placeholder_counts).fillna(0).astype(int)
            else:
                frame["auction_placeholder_count"] = 0
            frame["missing_minutes"] = (size - frame.minute_count - frame.auction_placeholder_count).clip(lower=0)
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
        saved_diagnostics = snapshot_metadata.get("diagnostics") or {}
        projection = project_data_diagnostics(
            snapshot_metadata, saved_diagnostics,
            saved_diagnostics.get("checked_start"), saved_diagnostics.get("checked_end"), manifest.get("profiles"),
        )
        warnings = list(dict.fromkeys([*projection["current_warnings"], *flags]))
        return {"symbol": symbol, "version": manifest["version"], "frequency": frequency, "bars": frame.to_dict("records"), "quote": quote, "has_more": more, "next_before": str(frame.timestamp.iloc[0]) if more else None, "is_demo": bool(manifest["metadata"].get("is_demo")), "warnings": warnings, "quality_flags": flags, "diagnostics": projection["diagnostics"], "source_events": projection["source_events"], "instrument": instrument_terms(symbol, manifest["metadata"])}

    def coverage(self, symbol: str):
        from .market_data import MarketDataService
        return MarketDataService(self.store, settings=self.settings).compatibility_coverage(symbol)
        symbol = normalize_symbol(symbol)
        manifest = self.store.snapshot(symbol)
        if not manifest:
            projection = project_data_diagnostics({}, {}, None, None)
            return {"symbol": symbol, "version": None, "rows": 0, "days": [], "profiles": {}, "warnings": ["尚未下载本地数据"], "diagnostics": projection["diagnostics"], "source_events": [], "bytes": 0}
        if manifest.get("market_data"):
            from .market_data import MarketDataService
            return MarketDataService(self.store, settings=self.settings).compatibility_coverage(symbol)
        cache_key = f"coverage:{manifest['version']}:q{QUALITY_RULES_VERSION}"
        cache = self.store.state(cache_key)
        if cache and cache.get("quality_rules_version") == QUALITY_RULES_VERSION and (cache.get("diagnostics") or {}).get("schema_version") == 1 and "source_events" in cache:
            return cache
        daily = self.store.read_files(manifest["files"]["daily"])
        minute = self.store.read_files(manifest["files"]["minute"])
        calendar = self.store.read_files(manifest["files"]["calendar"])
        actions = self.store.read_files(manifest["files"]["actions"])
        if daily.empty:
            projection = project_data_diagnostics(manifest["metadata"], {}, None, None, manifest["profiles"])
            return {"symbol": symbol, "version": manifest["version"], "rows": len(minute), "days": [], "warnings": list(dict.fromkeys(["日线缺失", *projection["current_warnings"]])), "diagnostics": projection["diagnostics"], "source_events": projection["source_events"], "profiles": manifest["profiles"], "bytes": 0}
        try:
            bundle = prepare_bundle(symbol, minute, daily, actions, calendar, daily.date.min(), daily.date.max(), "1m", manifest["version"], {**manifest["metadata"], "delisting_date": None}, manifest["profiles"])
            quality = bundle.quality
        except DataQualityError as exc:
            quality = {"days": [], "minute_days": 0, "daily_days": len(daily), "suspended_days": 0, "total_days": len(calendar), "pending_calibration_days": 0, "minute_ratio": 0, "synthetic_bars": 0, "warnings": [str(exc)]}
        diagnostic_metadata = dict(manifest["metadata"])
        diagnostic_metadata["daily_through"] = str(daily.date.max().date()) if not daily.empty else None
        diagnostic_metadata["minute_through"] = str(minute.timestamp.max()) if not minute.empty else None
        diagnostic_metadata["minute_status"] = self._minute_status(minute, quality)
        diagnostic_metadata["quality_summary"] = self._quality_summary(quality)
        diagnostic_metadata["diagnostics"] = {**(diagnostic_metadata.get("diagnostics") or {}), "source_spans": derive_source_spans(minute, manifest.get("profiles"))}
        projection = project_data_diagnostics(diagnostic_metadata, quality, daily.date.min(), daily.date.max(), manifest["profiles"])
        result = {**quality, "symbol": symbol, "version": manifest["version"], "rows": len(minute), "first_date": str(daily.date.min().date()), "last_date": str(daily.date.max().date()), "profiles": manifest["profiles"], "warnings": projection["current_warnings"], "diagnostics": projection["diagnostics"], "source_events": projection["source_events"], "quality_rules_version": QUALITY_RULES_VERSION, "is_demo": manifest["metadata"].get("is_demo", False), "bytes": sum((self.store.root / path).stat().st_size for paths in manifest["files"].values() for path in paths)}
        self.store.state(cache_key, result)
        return result

    def import_files(self, payload: dict, progress=lambda *_: None, cancelled=lambda: False):
        """Keep the historical import entry point while routing every file to the managed package."""
        from .market_data import MarketDataService

        market = MarketDataService(self.store, settings=self.settings)
        symbol = normalize_symbol(payload["symbol"])
        results = []
        for timeframe, key, convention in (
            ("1day", "daily_path", "end"), ("1min", "minute_path", payload.get("label_convention", "auto")),
        ):
            if not payload.get(key):
                continue
            results.append(market.import_file(
                symbol, timeframe, payload[key], label_convention=convention,
                etf_category=payload.get("etf_category"),
                calendar_path=payload.get("calendar_path"),
                actions_path=payload.get("actions_path") if timeframe == "1day" else None,
                skip_invalid_rows=bool(payload.get("skip_invalid_rows", False)),
                progress=progress, cancelled=cancelled,
            ))
        if not results:
            raise DataQualityError("请至少提供一个行情文件")
        return results[0] if len(results) == 1 else {
            "symbol": symbol, "results": results,
            "message": f"已导入 {len(results)} 个基础周期",
        }
