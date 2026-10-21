from __future__ import annotations

import hashlib
import math
import multiprocessing
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..models import BacktestConfig, normalize_symbol
from ..performance import (
    blas_thread_limit, effective_cpu_budget, effective_setting, record_count,
    record_seconds,
)
from ..settings import Settings, load_settings
from ..quality import QUALITY_RULES_VERSION
from ..price_validation import (
    PRICE_VALIDATION_VERSION, anomalies_in_range, anomaly_records, anomaly_summary,
    details_to_rows, price_issue_details,
)
from ..rules import instrument_terms
from ..storage import Store
from .models import DataNotReady, FileImportResult, MarketDataError
from .quality import validate_and_fill
from .reader import (
    check_declared_adjustment, check_declared_identity, check_declared_timeframe, file_fingerprint,
    normalize_frame, read_file, scan_folder,
)
from .repository import MarketDataRepository, PARSER_VERSION
from .timeframes import MINUTE_TIMEFRAMES, TIMEFRAMES, normalize_timeframe, timeframe_spec
from .validation import validate_incremental


BASE_TIMEFRAMES = ("1day", "1min")
# Mirrors the default research minute-coverage gate.  A small fraction of
# sessions may use verified daily OHLC when the intraday path is unavailable;
# the research layer still enforces its own min_minute_ratio setting.
MAX_DAILY_FALLBACK_RATIO = 0.05
DERIVED_SOURCES = {
    "5min": "1min", "15min": "1min", "30min": "1min", "60min": "1min",
    "1week": "1day", "1month": "1day",
}
SOURCE_TO_TARGETS = {
    "1min": ("5min", "15min", "30min", "60min"),
    "1day": ("1week", "1month"),
}


def _price_anomaly_rows(frame: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    anomalies = anomalies_in_range(frame, start, end)
    if anomalies.empty:
        return anomalies
    return anomalies


def _summary_from_anomalies(
    anomalies: pd.DataFrame, symbol: str, timeframe: str, version=None,
) -> dict:
    holder = pd.DataFrame()
    holder.attrs["price_anomaly_rows"] = details_to_rows(anomalies)
    return anomaly_summary(holder, symbol=symbol, timeframe=timeframe, version=version)


def _raise_price_not_ready(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    version=None,
    *,
    start=None,
    end=None,
    context: str = "行情",
) -> None:
    anomalies = _price_anomaly_rows(frame, start, end)
    if anomalies.empty:
        return
    samples = anomaly_records(
        anomalies, symbol=symbol, timeframe=timeframe, version=version, limit=12
    )
    stamps = [str(item["timestamp"]) for item in samples[:6] if item.get("timestamp")]
    raise DataNotReady(
        f"{context}仍有 {len(anomalies)} 条无效价格（{', '.join(stamps)}）；请修正源记录后重新校验",
        gaps=[{"kind": "price_quality", "timeframe": timeframe, "anomalies": samples}],
    )


def _validate_month_worker(
    root: str, package_dir: str | None, symbol: str, timeframe: str,
    adjustment: str, mode: str, start: str, end: str,
):
    """Validate one independent month without sharing pandas or SQLite state."""
    service = MarketDataService(Store(root), package_dir=package_dir)
    with blas_thread_limit(1):
        return validate_incremental(
            service, symbol, timeframe, mode=mode, start=start, end=end,
            progress=lambda *_: None, cancelled=lambda: False,
        )


class MarketDataService:
    def __init__(
        self,
        store_or_root: Store | str | Path,
        package_dir: str | Path | None = None,
        *,
        settings: Settings | None = None,
    ):
        self.settings = settings or load_settings()
        self.store = store_or_root if isinstance(store_or_root, Store) else Store(store_or_root)
        self.repository = MarketDataRepository(
            self.store.root, package_dir=package_dir,
            market_settings=self.settings.market_data,
        )

    def frequency_catalog(self) -> list[dict]:
        sources = {
            "1min": ["local", "sina", "tdx"],
            "5min": ["aggregate:1min"],
            "15min": ["aggregate:1min"],
            "30min": ["aggregate:1min"],
            "60min": ["aggregate:1min"],
            "1day": ["local", "baostock", "eastmoney"],
            "1week": ["aggregate:1day"],
            "1month": ["aggregate:1day"],
        }
        return [
            {"name": name, "label": timeframe_spec(name).label, "kind": timeframe_spec(name).kind,
             "minutes": timeframe_spec(name).minutes, "sources": sources[name],
             "local_history_only": name == "1min",
             "online_history_supported": name == "1day",
             "derived": name not in BASE_TIMEFRAMES,
             "derived_from": DERIVED_SOURCES.get(name),
             "legacy": {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "60min": "60m", "1day": "1d"}.get(name)}
            for name in TIMEFRAMES
        ]

    def datasets(self, timeframe=None, query="", status=None, limit=200, offset=0):
        normalized = normalize_timeframe(timeframe) if timeframe else None
        return self.repository.datasets(normalized, query, status, limit, offset)

    def package_info(self) -> dict:
        return self.repository.package_info()

    def inventory(self, *, query="", asset_type=None, timeframe=None, status=None, limit=50, offset=0) -> dict:
        return self.repository.inventory(
            query=query, asset_type=asset_type, timeframe=timeframe,
            status=status, limit=limit, offset=offset,
        )

    def coverage(self, symbol: str, timeframe: str) -> dict:
        symbol, timeframe = normalize_symbol(symbol), normalize_timeframe(timeframe)
        current = self.repository.coverage(symbol, timeframe)
        if current:
            return current
        raise MarketDataError(f"{symbol} 尚无 {timeframe} 数据")

    def compatibility_coverage(self, symbol: str) -> dict:
        """Project the managed package into the single coverage response used by the UI."""
        symbol = normalize_symbol(symbol)
        daily = self.repository.coverage(symbol, "1day")
        minute = self.repository.coverage(symbol, "1min")
        if not daily and not minute:
            raise MarketDataError(f"{symbol} 尚无本地 Parquet 行情")
        primary = daily or minute
        cache_key = (
            f"market-coverage:{self.repository.package_id}:{self.repository.generation}:"
            f"{symbol}:{(daily or {}).get('current_version')}:{(minute or {}).get('current_version')}:"
            f"{primary.get('status')}:{(daily or {}).get('verified_through')}:"
            f"{(minute or {}).get('verified_through')}:q{QUALITY_RULES_VERSION}:p{PRICE_VALIDATION_VERSION}"
        )
        cached_coverage = self.store.state(cache_key)
        if isinstance(cached_coverage, dict) and cached_coverage.get("price_validation_version") == PRICE_VALIDATION_VERSION:
            return cached_coverage
        quality_fields = ("timestamp", "open", "high", "low", "close", "volume")
        daily_frame = (
            self.repository.read_frame(symbol, "1day", fields=quality_fields)
            if daily else pd.DataFrame()
        )
        minute_frame = (
            self.repository.read_frame(symbol, "1min", fields=quality_fields)
            if minute else pd.DataFrame()
        )
        daily_quality = anomaly_summary(
            daily_frame, symbol=symbol, timeframe="1day",
            version=(daily or {}).get("current_version"),
        )
        minute_quality = anomaly_summary(
            minute_frame, symbol=symbol, timeframe="1min",
            version=(minute or {}).get("current_version"),
        )
        days = []
        for segment in (daily or primary).get("segments", []):
            report = segment.get("report") or {}
            days.append({
                "date": str(segment.get("start_time", ""))[:10], "mode": "daily" if daily else "minute",
                "missing_minutes": report.get("missing_rows", 0), "missing_data_minutes": report.get("unfillable", 0),
                "auction_placeholders": 0, "reason": f"检查至 {str(segment.get('end_time', ''))[:10]}",
            })
        anomaly_by_day: dict[str, dict[str, Any]] = {}
        for timeframe, quality in (("1day", daily_quality), ("1min", minute_quality)):
            for item in quality.get("samples", []):
                day = str(item.get("timestamp") or "")[:10]
                if not day:
                    continue
                current = anomaly_by_day.setdefault(day, {"count": 0, "reasons": set(), "timeframes": set()})
                current["count"] += 1
                current["reasons"].add(str(item.get("reason") or "价格无效"))
                current["timeframes"].add(timeframe)
        for day, item in sorted(anomaly_by_day.items()):
            days.append({
                "date": day, "mode": "price_invalid",
                "missing_minutes": 0, "missing_data_minutes": int(item["count"]),
                "auction_placeholders": 0,
                "reason": "无效价格：" + "；".join(sorted(item["reasons"])),
            })
        bytes_used = 0
        for timeframe in ("1day", "1min"):
            dataset = self.repository.dataset(symbol, timeframe)
            if dataset:
                bytes_used += int(dataset.get("current_bytes") or 0)
        daily_days = int(daily_frame.timestamp.dt.normalize().nunique()) if not daily_frame.empty else 0
        minute_days = int(minute_frame.timestamp.dt.normalize().nunique()) if not minute_frame.empty else 0
        price_anomaly_count = daily_quality["count"] + minute_quality["count"]
        warnings = [primary.get("reason")] if primary.get("reason") else []
        if price_anomaly_count:
            warnings.append(
                f"隔离了 {price_anomaly_count} 条无效价格；受影响区间不能用于回测或研究"
            )
        diagnostics = {
            "schema_version": 1,
            "status": "failed" if price_anomaly_count else "complete" if primary.get("status") == "ready" else "partial",
            "requested_frequency": "1m" if minute else "1d", "minute_quality_applicable": bool(minute),
            "daily_days": daily_days, "minute_days": minute_days,
            "total_days": daily_days, "minute_ratio": 1 if minute and minute.get("status") == "ready" else 0,
            "synthetic_bars": (minute or {}).get("synthetic_rows", 0), "auction_placeholders": 0,
            "price_anomaly_count": price_anomaly_count,
            "price_quality": {"daily": daily_quality, "minute": minute_quality},
            "current_warnings": warnings,
        }
        result = {
            "symbol": symbol, "version": primary.get("current_version"), "rows": (minute or daily).get("row_count", 0),
            "first_date": str((daily or minute).get("first_time", ""))[:10], "last_date": str((daily or minute).get("last_time", ""))[:10],
            "days": days, "profiles": {}, "warnings": warnings,
            "diagnostics": diagnostics, "source_events": [], "quality_rules_version": QUALITY_RULES_VERSION,
            "is_demo": False, "bytes": bytes_used, "storage": primary.get("storage", "parquet"),
            "minute_days": diagnostics["minute_days"], "daily_days": diagnostics["daily_days"],
            "total_days": diagnostics["total_days"], "minute_ratio": diagnostics["minute_ratio"],
            "synthetic_bars": diagnostics["synthetic_bars"], "auction_placeholders": 0,
            "price_anomaly_count": price_anomaly_count,
            "price_validation_version": PRICE_VALIDATION_VERSION,
        }
        self.store.state(cache_key, result)
        return result

    def ensure_timeframe(
        self, symbol: str, timeframe: str, *, start=None, end=None,
        progress=lambda *_: None, cancelled=lambda: False,
    ) -> dict | None:
        """Materialize a derived timeframe lazily from its single base source."""
        symbol, timeframe = normalize_symbol(symbol), normalize_timeframe(timeframe)
        if timeframe not in DERIVED_SOURCES:
            return self.repository.dataset(symbol, timeframe)
        source = DERIVED_SOURCES[timeframe]
        target = self.repository.dataset(symbol, timeframe)
        derived = target.get("derived") if target else None
        source_dataset = self.repository.dataset(symbol, source)
        if not source_dataset:
            raise MarketDataError(
                f"{timeframe} 只能由本地 {source} 数据生成；请先补齐 {source} 行情"
            )
        request_start, request_end = start, end
        source_last = source_dataset.get("last_time")
        if request_end is None:
            request_end = source_last
        if request_start is None and request_end is not None:
            lookback_days = 45 if source == "1min" else 400
            request_start = (
                pd.Timestamp(request_end).normalize() - pd.Timedelta(days=lookback_days)
            )
        fresh_range = bool(
            target and derived
            and int(derived.get("source_version") or -1)
            == int(source_dataset.get("current_version") or 0)
            and derived.get("start_time") and derived.get("end_time")
            and (request_start is None or pd.Timestamp(request_start) >= pd.Timestamp(derived["start_time"]))
            and (request_end is None or pd.Timestamp(request_end) <= pd.Timestamp(derived["end_time"]))
        )
        covers_request = bool(
            target
            and (request_start is None or (
                target.get("first_time")
                and pd.Timestamp(target["first_time"]).normalize() <= pd.Timestamp(request_start).normalize()
            ))
            and (request_end is None or (
                target.get("last_time")
                and pd.Timestamp(target["last_time"]).normalize() >= pd.Timestamp(request_end).normalize()
            ))
        )
        if (
            target and derived
            and str(derived.get("source_timeframe")) == source
            and target.get("file_status") == "readable"
            and ((not target.get("derived_stale") and covers_request) or fresh_range)
        ):
            probe = self.repository.read_frame(
                symbol, timeframe, start=request_start, end=request_end,
                fields=("timestamp", "open", "high", "low", "close", "volume"),
            )
            _raise_price_not_ready(
                probe, symbol, timeframe, target.get("current_version"),
                start=request_start, end=request_end, context="派生周期",
            )
            return target
        full_target_range = bool(
            target
            and request_start is not None and request_end is not None
            and target.get("first_time") and target.get("last_time")
            and pd.Timestamp(request_start).normalize() <= pd.Timestamp(target["first_time"]).normalize()
            and pd.Timestamp(request_end).normalize() >= pd.Timestamp(target["last_time"]).normalize()
        )
        snapshot = bool(target and target.get("derived_stale") and full_target_range)
        from .aggregation import AggregationService

        source_probe = self.repository.read_frame(
            symbol, source, start=request_start, end=request_end,
            fields=("timestamp", "open", "high", "low", "close", "volume"),
        )
        _raise_price_not_ready(
            source_probe, symbol, source, source_dataset.get("current_version"),
            start=request_start, end=request_end, context="聚合源行情",
        )

        result = AggregationService(self).aggregate(
            symbol, source, timeframe, start=request_start, end=request_end,
            snapshot=snapshot, progress=progress, cancelled=cancelled,
        )
        if target and target.get("derived_stale") and not snapshot:
            self.repository.mark_derived_target_stale(symbol, timeframe)
        return self.repository.dataset(symbol, timeframe) or result

    def bars(self, symbol: str, timeframe: str, *, start=None, end=None, version=None, limit=None) -> dict:
        symbol, timeframe = normalize_symbol(symbol), normalize_timeframe(timeframe)
        if version is None and timeframe in DERIVED_SOURCES:
            self.ensure_timeframe(symbol, timeframe, start=start, end=end)
        dataset = self.repository.dataset(symbol, timeframe)
        if not dataset:
            raise MarketDataError(f"{symbol} 尚无 {timeframe} 数据")
        if timeframe in DERIVED_SOURCES and version is not None and not dataset.get("derived"):
            raise MarketDataError(
                f"历史 {timeframe} 文件仅用于追溯；派生周期须由对应基础行情生成"
            )
        frame = self.repository.bars(symbol, timeframe, start=start, end=end, version=version, limit=limit)
        selected = int(version) if version is not None else int(dataset["current_version"])
        detail = self.repository.version_info(symbol, timeframe, selected) if selected != int(dataset["current_version"]) else None
        quality = anomaly_summary(frame, symbol=symbol, timeframe=timeframe, version=selected)
        return {
            "symbol": symbol, "timeframe": timeframe, "version": selected,
            "rows": frame.to_dict("records"), "count": len(frame),
            "first_time": (detail or dataset)["first_time"], "last_time": (detail or dataset)["last_time"],
            "status": (detail or dataset)["status"], "storage": dataset.get("storage", "parquet"),
            "quality": quality,
        }

    def market_bars(self, symbol: str, timeframe: str, *, start=None, end=None, before=None,
                    limit=600, ma_periods=(5, 10, 20, 60)) -> dict:
        symbol, timeframe = normalize_symbol(symbol), normalize_timeframe(timeframe)
        if timeframe in DERIVED_SOURCES:
            self.ensure_timeframe(symbol, timeframe, start=start, end=end or before)
        dataset = self.repository.dataset(symbol, timeframe)
        if not dataset:
            raise MarketDataError(f"{symbol} 尚无 {timeframe} 数据")
        periods = sorted({int(value) for value in ma_periods if 1 <= int(value) <= 500})
        warmup = max(periods, default=1) - 1
        upper = before or end
        upper_exclusive = (
            pd.Timestamp(before) if before is not None
            else pd.Timestamp(end).normalize() + pd.Timedelta(days=1) if end is not None else None
        )
        fetched = self.repository.last_bars(
            symbol, timeframe, before=upper_exclusive,
            limit=limit + warmup + 1,
        )
        price_anomalies = _price_anomaly_rows(fetched, start=start, end=upper)
        frame = fetched
        if start:
            frame = frame[frame.timestamp >= pd.Timestamp(start)]
        calculation = frame.copy()
        if not price_anomalies.empty:
            gaps = price_anomalies[[column for column in ("timestamp",) if column in price_anomalies]].copy()
            gaps["close"] = np.nan
            gaps["open"] = np.nan
            gaps["high"] = np.nan
            gaps["low"] = np.nan
            calculation = (
                gaps if calculation.empty else
                pd.concat([calculation, gaps], ignore_index=True, sort=False)
            ).sort_values("timestamp")
        for period in periods:
            calculation[f"ma{period}"] = calculation.close.rolling(period, min_periods=period).mean()
        if not price_anomalies.empty:
            moving_columns = [f"ma{period}" for period in periods]
            frame = frame.drop(columns=[column for column in moving_columns if column in frame], errors="ignore").merge(
                calculation[["timestamp", *moving_columns]], on="timestamp", how="left"
            )
        more = len(frame) > limit
        frame = frame.tail(limit).copy()
        daily_upper = (
            pd.Timestamp(upper).normalize() + pd.Timedelta(days=1)
            if upper is not None and before is None else upper
        )
        daily = self.repository.last_bars(symbol, "1day", before=daily_upper, limit=1)
        quote = daily.iloc[-1].to_dict() if not daily.empty else None
        warnings = []
        if dataset["status"] != "ready":
            warnings.append(dataset.get("reason") or f"{timeframe} 数据有待核验事项")
        quality = _summary_from_anomalies(
            price_anomalies, symbol, timeframe, dataset["current_version"]
        )
        if quality["count"]:
            warnings.append(
                f"显示区间隔离了 {quality['count']} 条无效价格记录；均线在该缺口处断开，不能用于计算"
            )
        frame["frequency"] = timeframe
        if quote:
            quote["timestamp"] = pd.Timestamp(quote["timestamp"]).isoformat()
            quote["date"] = pd.Timestamp(quote["timestamp"]).date().isoformat()
        return {
            "symbol": symbol, "version": dataset["current_version"], "frequency": timeframe,
            "bars": frame.to_dict("records"), "quote": quote, "has_more": more,
            "next_before": str(frame.timestamp.iloc[0]) if more and len(frame) else None,
            "is_demo": False, "warnings": warnings, "quality_flags": warnings,
            "price_quality": quality,
            "diagnostics": {}, "source_events": [], "storage": dataset.get("storage", "parquet"),
            "instrument": self.store.instrument(symbol) or instrument_terms(symbol, {"name": symbol}),
        }

    def import_file(
        self,
        symbol: str,
        timeframe: str,
        path: str | Path,
        *,
        label_convention: str = "auto",
        adjustment: str = "none",
        source: str = "import",
        etf_category: str | None = None,
        calendar_path: str | Path | None = None,
        actions_path: str | Path | None = None,
        skip_invalid_rows: bool = False,
        import_mode: str = "merge",
        progress=lambda *_: None,
        cancelled=lambda: False,
    ) -> dict:
        symbol, timeframe = normalize_symbol(symbol), normalize_timeframe(timeframe)
        if timeframe not in BASE_TIMEFRAMES:
            source = DERIVED_SOURCES.get(timeframe)
            raise MarketDataError(
                f"{timeframe} 属于派生周期；请导入或更新 {source} 基础行情，查询时会按需聚合"
            )
        if str(adjustment).lower() not in {"none", "raw", "unadjusted"}:
            raise MarketDataError("当前版本仅支持不复权行情导入")
        adjustment = "none"
        if cancelled():
            raise InterruptedError("任务已取消")
        progress(0.05, f"读取 {Path(path).name}")
        frame, fingerprint, encoding, batches = read_file(path)
        params_hash = hashlib.sha256(
            f"{timeframe}|{label_convention}|{adjustment}|{source}|{PARSER_VERSION}|{import_mode}".encode()
        ).hexdigest()
        old_import = self.repository.fingerprint_seen(symbol, timeframe, adjustment, fingerprint, params_hash)
        if old_import and old_import.get("status") == "completed":
            return FileImportResult(
                symbol, timeframe, "skipped", version=old_import.get("version"),
                rows_read=old_import.get("rows_read", 0), unchanged=old_import.get("unchanged", 0),
                message="相同文件内容和参数已经成功导入，已跳过", fingerprint=fingerprint,
            ).as_dict()
        check_declared_identity(frame, symbol)
        check_declared_timeframe(frame, timeframe)
        check_declared_adjustment(frame, adjustment)
        normalized, pending = normalize_frame(
            frame, timeframe, label_convention=label_convention, source=source,
            skip_invalid_rows=skip_invalid_rows,
        )
        return self._commit_frame(
            symbol, timeframe, normalized, fingerprint=fingerprint, file_name=Path(path).name,
            params_hash=params_hash, pending=pending, source=source, adjustment=adjustment,
            etf_category=etf_category, calendar_path=calendar_path, actions_path=actions_path,
            progress=progress, cancelled=cancelled, encoding=encoding, batches=batches,
            import_mode=import_mode,
        )

    def _commit_frame(
        self, symbol, timeframe, frame, *, fingerprint, file_name, params_hash, pending,
        source, adjustment, etf_category, calendar_path, actions_path, progress, cancelled,
        encoding=None, batches=1, derived=None,
        coverage_start=None, coverage_end=None,
        source_metadata: dict | None = None,
        calendar_frame: pd.DataFrame | None = None,
        import_mode: str = "merge",
    ) -> dict:
        started_at = time.perf_counter()
        if import_mode not in {"merge", "snapshot"}:
            raise MarketDataError("导入模式应为 merge 或 snapshot")
        if cancelled():
            raise InterruptedError("任务已取消")
        progress(0.35, "校验时间、价格与交易日覆盖")
        rejected_prices = price_issue_details(frame)
        if not rejected_prices.empty:
            pending.append(
                f"已隔离 {len(rejected_prices)} 条无效 OHLCV 源记录，未写入可计算行情"
            )
            source_metadata = {
                **(source_metadata or {}),
                "price_validation": {
                    "rule_version": PRICE_VALIDATION_VERSION,
                    "rejected_rows": len(rejected_prices),
                    "samples": anomaly_records(
                        rejected_prices, symbol=symbol, timeframe=timeframe, limit=50
                    ),
                },
            }
            frame = frame.drop(index=rejected_prices.index).copy()
            if frame.empty:
                raise MarketDataError("提交内容仅包含无效价格，未写入任何行情")
        calendar = (
            pd.DataFrame(calendar_frame).copy()
            if calendar_frame is not None
            else self._calendar(symbol, path=calendar_path)
        )
        if timeframe in MINUTE_TIMEFRAMES:
            frame, volume_pending = self._verify_volume_units(symbol, timeframe, frame)
            if volume_pending:
                pending.append(volume_pending)
        if timeframe == "1day":
            required = ["preclose", "tradestatus"]
            missing = [field for field in required if field not in frame or frame[field].isna().any()]
            if instrument_terms(symbol, {"name": symbol}).get("asset_type") == "stock":
                if "is_st" not in frame or frame["is_st"].isna().any():
                    missing.append("is_st")
            if missing:
                pending.append("日线缺少严格回测字段：" + ", ".join(missing))
        validation = validate_and_fill(
            frame, timeframe, calendar=calendar, pending=pending,
            coverage_start=coverage_start, coverage_end=coverage_end,
            symbol=symbol,
        )
        if cancelled():
            raise InterruptedError("任务已取消")
        progress(0.65, "合并本地记录并提交数据库版本")
        instrument = self.store.instrument(symbol) or instrument_terms(symbol, {"name": symbol})
        metadata = {**instrument, **(source_metadata or {}), "is_demo": False, "imported": True, "encoding": encoding, "batches": batches,
                    "validation": validation.report}
        if getattr(validation.frame, "attrs", None):
            metadata.update(validation.frame.attrs)
        if etf_category:
            if etf_category not in {"stock", "bond", "crossborder", "commodity", "money"}:
                raise MarketDataError("无效的 ETF 类别")
            metadata["etf_category"] = etf_category
        metadata = instrument_terms(symbol, metadata)
        report = {**validation.report, "pending_validation": validation.pending,
                  "reason": "; ".join(validation.pending) or None}
        result = self.repository.commit_bars(
            symbol, timeframe, validation.frame, adjustment=adjustment, source=source,
            file_name=file_name, fingerprint=fingerprint, params_hash=params_hash,
            report=report, metadata=metadata, derived=derived,
            advance_completed=not (
                timeframe in {"1week", "1month"}
                and any("尚未结束" in item for item in validation.pending)
            ),
            import_mode=import_mode,
        )
        self.repository.save_coverage(
            symbol, timeframe, adjustment, min(validation.frame.timestamp.min(), pd.Timestamp(coverage_start) if coverage_start is not None else validation.frame.timestamp.min()),
            max(validation.frame.timestamp.max(), pd.Timestamp(coverage_end) if coverage_end is not None else validation.frame.timestamp.max()), validation.report,
        )
        auxiliary_changed = False
        if actions_path:
            actions, _, _, _ = read_file(actions_path)
            existing_actions = self.repository.auxiliary(symbol, "actions")
            auxiliary_changed = not existing_actions or existing_actions.get("payload") != actions.to_dict("records")
            self.repository.save_auxiliary(symbol, "actions", actions.to_dict("records"), version=fingerprint, verified=True)
        if calendar_path:
            calendar_frame, _, _, _ = read_file(calendar_path)
            if "date" not in calendar_frame:
                raise MarketDataError("交易日历必须包含 date 列")
            current_calendar = self.repository.auxiliary(symbol, "calendar")
            new_dates = sorted(pd.to_datetime(calendar_frame.date, errors="coerce").dropna().dt.normalize().astype(str).unique())
            old_dates = sorted(pd.to_datetime(
                [row.get("date") for row in ((current_calendar or {}).get("payload") or [])],
                errors="coerce",
            ).dropna().normalize().astype(str).unique()) if current_calendar else []
            auxiliary_changed = auxiliary_changed or old_dates != new_dates
            self.repository.save_auxiliary(symbol, "calendar", calendar_frame[["date"]].to_dict("records"), version=fingerprint, verified=True)
        if auxiliary_changed:
            self.repository.invalidate_all_validations(symbol, adjustment)
        changed_dates = list(result.get("changed_dates") or [])
        if not changed_dates:
            changed_dates = sorted({
                str(pd.Timestamp(value).date()) for value in validation.frame.timestamp
            })
        if timeframe == "1day":
            for minute_timeframe in MINUTE_TIMEFRAMES:
                self.repository.invalidate_day_validations(
                    symbol, minute_timeframe, changed_dates, adjustment=adjustment
                )
        validation_start = min(pd.Timestamp(value) for value in changed_dates) if changed_dates else validation.frame.timestamp.min()
        validation_end = max(pd.Timestamp(value) for value in changed_dates) if changed_dates else validation.frame.timestamp.max()
        ledger = validate_incremental(
            self, symbol, timeframe, mode="incremental",
            start=validation_start, end=validation_end,
            progress=lambda fraction, message: progress(0.8 + 0.15 * fraction, message),
            cancelled=cancelled,
        )
        metrics = {
            "seconds": round(time.perf_counter() - started_at, 4),
            "rows_read": int(result.get("rows_read") or 0),
            "bytes_written": int(result.get("bytes") or 0),
            "base_rewrites": 1 if result.get("layout") == "immutable-base" else 0,
            "increment_files": int(result.get("increment_count") or 0),
            "validated_days": int(ledger.get("checked_days") or 0),
        }
        self.store.put_instrument(symbol, **{key: value for key, value in metadata.items() if key not in {"symbol", "name"}})
        self.repository.save_basic_info(symbol, metadata)
        self.refresh_snapshot(symbol)
        progress(1.0, "行情已写入本地 Parquet 行情库")
        return FileImportResult(
            symbol, timeframe, result["status"], version=result["version"],
            rows_read=result["rows_read"], inserted=result["inserted"], updated=result["updated"],
            unchanged=result["unchanged"], synthetic=result["synthetic"],
            gap_count=validation.report.get("gap_count", 0), pending_validation=validation.pending,
            message=(f"新增 {result['inserted']} 行，修正 {result['updated']} 行，未变化 {result['unchanged']} 行"
                     + ("；有待核验事项" if validation.pending else "")
                     + (f"；合规截至 {ledger.get('verified_through')}" if ledger.get("verified_through") else "")),
            fingerprint=fingerprint,
            import_mode=import_mode, changed_dates=changed_dates,
            increment_count=int(result.get("increment_count") or result.get("file_count") or 0),
            layout=result.get("layout"), compaction=result.get("compaction"),
            metrics=metrics,
        ).as_dict()

    def _verify_volume_units(self, symbol, timeframe, frame):
        daily = self.repository.active_rows(
            symbol, "1day", start=frame.timestamp.min().normalize(),
            end=frame.timestamp.max().normalize() + pd.Timedelta(days=1),
        )
        if daily.empty:
            return frame, "分钟成交量单位缺少日线核验依据"
        daily = daily[(pd.to_numeric(daily.tradestatus, errors="coerce") == 1) & (pd.to_numeric(daily.volume, errors="coerce") > 0)].copy()
        if daily.empty:
            return frame, "日线没有可用成交量，无法核验分钟成交量单位"
        daily["day"] = pd.to_datetime(daily.timestamp).dt.normalize()
        observed_days = pd.DatetimeIndex(frame.timestamp.dt.normalize().unique())
        matched_daily = daily.set_index("day").reindex(observed_days)
        result = frame.copy()
        pending = []
        convertible_days = matched_daily.index[
            pd.to_numeric(matched_daily.volume, errors="coerce").gt(0)
        ]
        if not len(convertible_days):
            return frame, "分钟记录日期没有对应日线成交量"

        def choose_factor(column, candidates):
            totals = frame.loc[
                frame.timestamp.dt.normalize().isin(convertible_days)
            ].groupby(frame.timestamp.dt.normalize()).agg(
                total=(column, "sum"),
            )
            reference = pd.to_numeric(
                matched_daily.reindex(totals.index)[column], errors="coerce"
            )
            valid = totals.total.gt(0) & reference.gt(0)
            ratios = (totals.loc[valid, "total"] / reference.loc[valid]).to_numpy()
            if not len(ratios):
                return 1.0, 0, 0.0
            best_factor, best_matches = 1.0, 0
            for factor in candidates:
                matches = int(np.sum(np.abs(ratios * factor - 1) <= 0.01))
                if matches > best_matches:
                    best_factor, best_matches = float(factor), matches
            required = max(1, math.ceil(len(ratios) * 0.95))
            if len(ratios) >= 3:
                required = max(3, required)
            if best_matches < required:
                return None, best_matches, len(ratios)
            return best_factor, best_matches, len(ratios)

        volume_factor, volume_matches, volume_total = choose_factor("volume", (1, 100))
        if volume_factor is None:
            return frame, (
                f"分钟成交量聚合与日线偏差超过 1%（{volume_matches}/{volume_total} 日一致）"
            )
        if volume_factor != 1:
            result["volume"] = pd.to_numeric(result.volume, errors="coerce") * volume_factor
        if "turnover" in frame and frame.turnover.notna().any():
            turnover_factor, turnover_matches, turnover_total = choose_factor(
                "turnover", (1, 100, 1000)
            )
            if turnover_factor is None:
                pending.append(
                    f"分钟成交额聚合与日线偏差超过 1%（{turnover_matches}/{turnover_total} 日一致）"
                )
            elif turnover_factor not in (None, 1):
                result["turnover"] = pd.to_numeric(result.turnover, errors="coerce") * turnover_factor
        if len(convertible_days) < math.ceil(len(observed_days) * 0.95):
            pending.append(
                f"分钟日期缺少对应日线成交量（{len(convertible_days)}/{len(observed_days)} 日可核验）"
            )
        tick = float(instrument_terms(symbol).get("price_tick", 0.01))
        expected_bars = 240 // (timeframe_spec(timeframe).minutes or 1)
        price_checked = price_matched = 0
        price_mismatches = []
        for day, part in result.groupby(result.timestamp.dt.normalize()):
            day = pd.Timestamp(day).normalize()
            minimum_bars = max(1, expected_bars - (2 if timeframe == "1min" else 1))
            if day not in matched_daily.index or len(part) < minimum_bars:
                continue
            row = matched_daily.loc[day]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            status = pd.to_numeric(row.get("tradestatus"), errors="coerce")
            daily_volume = pd.to_numeric(row.get("volume"), errors="coerce")
            if pd.isna(status) or int(status) != 1 or pd.isna(daily_volume) or float(daily_volume) <= 0:
                continue
            ordered = part.sort_values("timestamp")
            if pd.Timestamp(ordered.timestamp.max()).time() != pd.Timestamp("15:00").time():
                continue
            price_checked += 1
            errors = (
                abs(float(ordered.iloc[-1].close) - float(row["close"])),
                abs(float(ordered.high.max()) - float(row["high"])),
                abs(float(ordered.low.min()) - float(row["low"])),
            )
            if max(errors) <= tick * 1.1:
                price_matched += 1
            else:
                price_mismatches.append(str(day.date()))
        if price_mismatches:
            pending.append(
                f"分钟价格聚合与日线不一致（{len(price_mismatches)}/{price_checked} 日）："
                + ", ".join(price_mismatches[:8])
            )
        elif len(convertible_days) and price_checked == 0:
            pending.append("没有完整交易日可用于核验分钟价格聚合")
        metadata = {
            "unit_verification": {
                "volume_factor": volume_factor,
                "volume_verified_days": volume_matches,
                "turnover_factor": turnover_factor if "turnover_factor" in locals() else 1,
                "turnover_verified_days": locals().get("turnover_matches", 0),
                "price_checked_days": price_checked,
                "price_verified_days": price_matched,
            }
        }
        result.attrs.update(metadata)
        return result, "; ".join(pending) or None

    def preview_folder(self, folder, timeframe, *, recursive=False) -> dict:
        timeframe = normalize_timeframe(timeframe)
        entries = scan_folder(folder, timeframe, recursive=recursive)
        by_symbol: dict[str, list[dict]] = defaultdict(list)
        seen: dict[tuple[str | None, str], str] = {}
        for item in entries:
            if item["symbol"]:
                by_symbol[item["symbol"]].append(item)
            try:
                fingerprint, _ = file_fingerprint(Path(item["path"]))
                item["fingerprint"] = fingerprint
                duplicate_key = (item.get("symbol"), fingerprint)
                previous = seen.get(duplicate_key)
                if previous:
                    item["duplicate"] = True
                    item["duplicate_of"] = previous
                else:
                    seen[duplicate_key] = item["relative_path"]
            except Exception as exc:
                item["duplicate"] = False
                item["error"] = item.get("error") or str(exc)
        return {
            "folder": str(Path(folder).expanduser().resolve()), "timeframe": timeframe,
            "items": entries, "symbols": sorted(by_symbol),
            "target_count": len(by_symbol), "file_count": len(entries),
            "invalid_count": sum(bool(item.get("error")) for item in entries),
        }

    def import_folder(
        self, folder, timeframe, *, recursive=False, label_convention="auto", adjustment="none",
        etf_category=None, resume=True, calendar_path=None, workers=4, skip_invalid_rows=False,
        import_mode="merge",
        progress=lambda *_: None, cancelled=lambda: False,
    ) -> dict:
        timeframe = normalize_timeframe(timeframe)
        if timeframe not in BASE_TIMEFRAMES:
            raise MarketDataError(
                f"{timeframe} 属于派生周期；请导入 {DERIVED_SOURCES.get(timeframe)} 基础行情"
            )
        if str(adjustment).lower() not in {"none", "raw", "unadjusted"}:
            raise MarketDataError("当前版本仅支持不复权行情导入")
        adjustment = "none"
        preview = self.preview_folder(folder, timeframe, recursive=recursive)
        groups: dict[str, list[dict]] = defaultdict(list)
        failures = []
        for item in preview["items"]:
            if item.get("symbol"):
                groups[item["symbol"]].append(item)
            else:
                failures.append({"file": item["relative_path"], "symbol": None, "reason": item.get("error") or "无法识别证券代码"})
        results, skipped = [], []
        symbols = sorted(groups)
        existing_daily = any(
            self.repository.dataset(symbol, "1day") for symbol in symbols
        )
        if timeframe == "1day" and symbols and not existing_daily and import_mode == "merge":
            return self._import_daily_folder_batches(
                groups, failures, symbols, folder=preview["folder"], label_convention=label_convention,
                etf_category=etf_category, resume=resume, calendar_path=calendar_path,
                skip_invalid_rows=skip_invalid_rows, progress=progress,
                cancelled=cancelled,
            )
        state_lock = threading.Lock()
        completed = 0

        def process(symbol, position):
            nonlocal completed
            if cancelled():
                raise InterruptedError("任务已取消")
            files = [item for item in groups[symbol] if not item.get("duplicate")]
            active = [item for item in groups[symbol] if item.get("duplicate")]
            if not files and active:
                with state_lock:
                    skipped.append({"symbol": symbol, "reason": "目录内只有重复文件"})
                    completed += 1
                    progress(completed / max(1, len(groups)), f"{symbol}：重复文件，已跳过")
                return
            try:
                frames = []
                fingerprints = []
                pending = []
                for item in files:
                    frame, fingerprint, _, _ = read_file(item["path"])
                    check_declared_identity(frame, symbol)
                    check_declared_timeframe(frame, timeframe)
                    check_declared_adjustment(frame, adjustment)
                    normalized, current_pending = normalize_frame(
                        frame, timeframe, label_convention=label_convention,
                        skip_invalid_rows=skip_invalid_rows,
                    )
                    frames.append(normalized)
                    pending.extend(current_pending)
                    fingerprints.append((item, fingerprint))
                combined = _merge_input_frames(frames)
                fingerprint = hashlib.sha256("".join(item[1] for item in fingerprints).encode()).hexdigest()
                params_hash = hashlib.sha256(
                    f"{timeframe}|{label_convention}|{adjustment}|import|{PARSER_VERSION}|{import_mode}".encode()
                ).hexdigest()
                old = self.repository.fingerprint_seen(symbol, timeframe, adjustment, fingerprint, params_hash)
                if resume and old and old.get("status") == "completed":
                    with state_lock:
                        skipped.append({"symbol": symbol, "reason": "相同文件组已导入", "version": old.get("version")})
                        completed += 1
                        progress(completed / max(1, len(groups)), f"{symbol}：相同文件已跳过")
                    return
                result = self._commit_frame(
                    symbol, timeframe, combined, fingerprint=fingerprint,
                    file_name=", ".join(item["relative_path"] for item, _ in fingerprints),
                    params_hash=params_hash, pending=pending, source="import", adjustment=adjustment,
                    etf_category=etf_category, calendar_path=calendar_path, actions_path=None,
                    progress=lambda fraction, message: progress((position + fraction) / max(1, len(groups)), f"{symbol}：{message}"),
                    cancelled=cancelled,
                    import_mode=import_mode,
                )
                with state_lock:
                    results.append(result)
                    completed += 1
                    progress(completed / max(1, len(groups)), f"{symbol}：完成 {result['inserted']} 行")
            except InterruptedError:
                raise
            except Exception as exc:
                with state_lock:
                    failures.append({"symbol": symbol, "file": ", ".join(item["relative_path"] for item in files), "reason": str(exc)})
                    completed += 1
                    progress(completed / max(1, len(groups)), f"{symbol}：失败 {exc}")

        worker_count = max(1, min(4, int(workers or 1)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            cursor = 0
            while cursor < len(symbols):
                if cancelled():
                    raise InterruptedError("任务已取消")
                batch = symbols[cursor:cursor + worker_count * 2]
                futures = {executor.submit(process, symbol, cursor + offset): symbol for offset, symbol in enumerate(batch)}
                cursor += len(batch)
                for future in as_completed(futures):
                    future.result()
        status = "completed" if not failures else "partial" if results else "failed"
        return {
            "status": status, "timeframe": timeframe, "folder": preview["folder"],
            "files": preview["file_count"], "targets": len(groups),
            "completed": len(results), "skipped": len(skipped), "failed": len(failures),
            "results": results, "skipped_items": skipped, "failures": failures,
            "message": f"完成 {len(results)} 个标的，跳过 {len(skipped)} 个，失败 {len(failures)} 个",
        }

    def _import_daily_folder_batches(
        self, groups, failures, symbols, *, folder, label_convention, etf_category,
        resume, calendar_path, skip_invalid_rows, progress, cancelled,
    ) -> dict:
        results, skipped = [], []
        market_settings = self.settings.market_data
        batch_size = int(market_settings.service_batch_size)
        for batch_start in range(0, len(symbols), batch_size):
            if cancelled():
                raise InterruptedError("任务已取消")
            prepared = []
            for symbol in symbols[batch_start:batch_start + batch_size]:
                files = [item for item in groups[symbol] if not item.get("duplicate")]
                if not files:
                    skipped.append({"symbol": symbol, "reason": "目录内只有重复文件"})
                    continue
                try:
                    frames, fingerprints, pending = [], [], []
                    for item in files:
                        frame, fingerprint, _, _ = read_file(item["path"])
                        check_declared_identity(frame, symbol)
                        check_declared_timeframe(frame, "1day")
                        check_declared_adjustment(frame, "none")
                        normalized, current_pending = normalize_frame(
                            frame, "1day", label_convention=label_convention,
                            skip_invalid_rows=skip_invalid_rows,
                        )
                        frames.append(normalized)
                        pending.extend(current_pending)
                        fingerprints.append((item, fingerprint))
                    combined = _merge_input_frames(frames)
                    fingerprint = hashlib.sha256(
                        "".join(item[1] for item in fingerprints).encode()
                    ).hexdigest()
                    params_hash = hashlib.sha256(
                        f"1day|{label_convention}|none|import|{PARSER_VERSION}|merge".encode()
                    ).hexdigest()
                    old = self.repository.fingerprint_seen(
                        symbol, "1day", "none", fingerprint, params_hash
                    )
                    if resume and old and old.get("status") == "completed":
                        skipped.append({
                            "symbol": symbol, "reason": "相同文件组已导入",
                            "version": old.get("version"),
                        })
                        continue
                    calendar = self._calendar(symbol, path=calendar_path)
                    validation = validate_and_fill(
                        combined, "1day", calendar=calendar, pending=pending,
                        coverage_start=combined.timestamp.min(),
                        coverage_end=combined.timestamp.max(), symbol=symbol,
                    )
                    instrument = self.store.instrument(symbol) or instrument_terms(
                        symbol, {"name": symbol}
                    )
                    metadata = {
                        **instrument, "is_demo": False, "imported": True,
                        "validation": validation.report,
                    }
                    if etf_category:
                        if etf_category not in {"stock", "bond", "crossborder", "commodity", "money"}:
                            raise MarketDataError("无效的 ETF 类别")
                        metadata["etf_category"] = etf_category
                    metadata = instrument_terms(symbol, metadata)
                    report = {
                        **validation.report,
                        "pending_validation": validation.pending,
                        "reason": "; ".join(validation.pending) or None,
                    }
                    prepared.append({
                        "symbol": symbol, "timeframe": "1day", "frame": validation.frame,
                        "pending": validation.pending, "source": "import",
                        "metadata": metadata, "report": report,
                        "fingerprint": fingerprint, "params_hash": params_hash,
                        "file_name": ", ".join(item["relative_path"] for item, _ in fingerprints),
                        "coverage_start": combined.timestamp.min(),
                        "coverage_end": combined.timestamp.max(),
                    })
                except InterruptedError:
                    raise
                except Exception as exc:
                    failures.append({
                        "symbol": symbol,
                        "file": ", ".join(item["relative_path"] for item in files),
                        "reason": str(exc),
                    })
            if prepared:
                committed = self.repository.commit_daily_batch(prepared)
                by_symbol = {item["symbol"]: item for item in prepared}
                for item in committed:
                    prepared_item = by_symbol[item["symbol"]]
                    if item.get("status") == "failed":
                        failures.append({"symbol": item["symbol"], "reason": item.get("reason")})
                        continue
                    self.repository.save_coverage(
                        item["symbol"], "1day", "none",
                        prepared_item["coverage_start"], prepared_item["coverage_end"],
                        prepared_item["report"],
                    )
                    if calendar_path:
                        calendar_frame, _, _, _ = read_file(calendar_path)
                        if "date" not in calendar_frame:
                            raise MarketDataError("交易日历必须包含 date 列")
                        self.repository.save_auxiliary(
                            item["symbol"], "calendar",
                            calendar_frame[["date"]].to_dict("records"),
                            version=prepared_item["fingerprint"][:16], verified=True,
                        )
                    self.store.put_instrument(
                        item["symbol"],
                        **{
                            key: value for key, value in prepared_item["metadata"].items()
                            if key not in {"symbol", "name"}
                        },
                    )
                    self.repository.save_basic_info(item["symbol"], prepared_item["metadata"])
                    self.revalidate(item["symbol"], "1day")
                    self.refresh_snapshot(item["symbol"])
                    results.append({
                        "symbol": item["symbol"], "timeframe": "1day",
                        "status": item["status"], "version": item["version"],
                        "rows_read": item["rows_read"], "inserted": item["inserted"],
                        "updated": item["updated"], "unchanged": item["unchanged"],
                        "synthetic": 0, "gap_count": prepared_item["report"].get("gap_count", 0),
                        "pending_validation": prepared_item["pending"],
                        "fingerprint": prepared_item["fingerprint"],
                        "message": item["message"],
                    })
            progress(
                min(1.0, (batch_start + batch_size) / max(1, len(symbols))),
                f"已处理 {min(batch_start + batch_size, len(symbols))}/{len(symbols)} 个日线标的",
            )
        status = "completed" if not failures else "partial" if results else "failed"
        return {
            "status": status, "timeframe": "1day", "folder": folder,
            "files": sum(len(groups[symbol]) for symbol in symbols),
            "targets": len(groups), "completed": len(results),
            "skipped": len(skipped), "failed": len(failures),
            "results": results, "skipped_items": skipped, "failures": failures,
            "message": f"完成 {len(results)} 个标的，跳过 {len(skipped)} 个，失败 {len(failures)} 个",
        }

    def revalidate(
        self, symbol: str, timeframe: str, *, adjustment="none",
        label_convention: str | None = None, mode: str = "incremental",
        start=None, end=None,
        progress: Callable[[float, str], None] = lambda *_: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> dict:
        if cancelled():
            raise InterruptedError("任务已取消")
        symbol, timeframe = normalize_symbol(symbol), normalize_timeframe(timeframe)
        dataset = self.repository.dataset(symbol, timeframe, adjustment)
        if not dataset:
            raise MarketDataError("没有可校验的本地行情")
        if mode not in {"incremental", "full"}:
            raise MarketDataError("校验模式应为 incremental 或 full")
        requested_start = start or dataset.get("first_time")
        requested_end = end or dataset.get("last_time")
        anomaly_cache_key = (
            f"validation-anomalies:{self.repository.package_id}:{symbol}:{timeframe}:"
            f"{dataset.get('current_version')}:p{PRICE_VALIDATION_VERSION}"
        )
        cached_anomaly_count = self.store.state(anomaly_cache_key)
        existing_anomalies = pd.DataFrame()
        quality_probe = pd.DataFrame()
        cache_hit = False
        if (
            mode == "incremental" and label_convention is None
            and self._validation_covers(dataset, requested_start, requested_end)
            and (cached_anomaly_count is None or int(cached_anomaly_count) == 0)
        ):
            state = self.repository.validation_state(
                symbol, timeframe, adjustment=adjustment,
                start=requested_start, end=requested_end,
            )
            if not state.get("failed") and not state.get("pending"):
                if cached_anomaly_count is None:
                    quality_probe = self.repository.bars(
                        symbol, timeframe, adjustment=adjustment,
                        start=requested_start, end=requested_end,
                    )
                    existing_anomalies = _price_anomaly_rows(
                        quality_probe, requested_start, requested_end
                    )
                    cached_anomaly_count = len(existing_anomalies)
                    self.store.state(anomaly_cache_key, int(cached_anomaly_count))
                cache_hit = int(cached_anomaly_count) == 0
        if cache_hit:
            state = self.repository.validation_state(
                symbol, timeframe, adjustment=adjustment,
                start=requested_start, end=requested_end,
            )
            if not state.get("failed") and not state.get("pending"):
                record_count("validation_cache_hits", 1)
                progress(1.0, f"{timeframe} 已有有效校验，复用 {state.get('validated_days', 0)} 日")
                return {
                    "symbol": symbol, "timeframe": timeframe, "status": "complete",
                    "pending_validation": [],
                    "report": {
                        "status": "complete", "mode": mode, "checked_days": 0,
                        "skipped_days": state.get("validated_days", 0),
                        "verified_from": state.get("verified_from"),
                        "verified_through": state.get("verified_through"),
                        "pending_validation": [], "failed": [], "cached": True,
                    },
                    "verified_from": state.get("verified_from"),
                    "verified_through": state.get("verified_through"),
                }
        if quality_probe.empty:
            record_count("validation_cache_misses", 1)
            quality_probe = self.repository.bars(
                symbol, timeframe, adjustment=adjustment,
                start=requested_start, end=requested_end,
            )
            existing_anomalies = _price_anomaly_rows(
                quality_probe, requested_start, requested_end
            )
            self.store.state(anomaly_cache_key, int(len(existing_anomalies)))
        frame = self.repository.active_rows(
            symbol, timeframe, adjustment,
            start=None if label_convention not in (None, "auto") else requested_start,
            end=None if label_convention not in (None, "auto") else requested_end,
        )
        if frame.empty:
            raise MarketDataError("没有可校验的本地行情")
        ledger_changed = False
        if label_convention is not None and timeframe in MINUTE_TIMEFRAMES:
            if label_convention not in {"auto", "start", "end"}:
                raise MarketDataError("时间标签应为 auto/start/end")
            if label_convention != "auto":
                unverified = ~frame.source_verified.astype(bool)
                if label_convention == "start":
                    frame.loc[unverified, "timestamp"] = frame.loc[unverified, "timestamp"] + pd.Timedelta(minutes=timeframe_spec(timeframe).minutes or 1)
                frame.loc[unverified, "source_verified"] = True
                if not unverified.any():
                    frame["source_verified"] = True
                params_hash = _params_hash(timeframe, label_convention, adjustment, "revalidate", PARSER_VERSION)
                fingerprint = hashlib.sha256(
                    f"revalidate|{symbol}|{timeframe}|{label_convention}|{frame.timestamp.min()}|{frame.timestamp.max()}|{len(frame)}".encode()
                ).hexdigest()
                self.repository.commit_bars(
                    symbol, timeframe, frame, adjustment=adjustment, source="revalidate",
                    file_name="local-revalidation", fingerprint=fingerprint, params_hash=params_hash,
                    report={"status": "ready", "pending_validation": []}, metadata={},
                    replace_range=label_convention == "start",
                )
                self.repository.invalidate_all_validations(symbol, adjustment)
                ledger_changed = True
                frame = self.repository.active_rows(symbol, timeframe, adjustment)
                self.repository.save_coverage(symbol, timeframe, adjustment, frame.timestamp.min(), frame.timestamp.max(), {
                    "status": "complete", "expected_rows": len(frame), "observed_rows": len(frame),
                    "synthetic_rows": int(frame.is_synthetic.sum()), "gap_count": 0,
                })
                self.refresh_snapshot(symbol)
        if mode == "full" and timeframe in MINUTE_TIMEFRAMES:
            verified_frame, volume_pending = self._verify_volume_units(symbol, timeframe, frame)
            if not verified_frame[['volume']].equals(frame[['volume']]):
                fingerprint = hashlib.sha256(
                    f"volume-revalidate|{symbol}|{timeframe}|{len(verified_frame)}".encode()
                ).hexdigest()
                self.repository.commit_bars(
                    symbol, timeframe, verified_frame, adjustment=adjustment, source="revalidate",
                    file_name="volume-revalidation", fingerprint=fingerprint,
                    params_hash=_params_hash(timeframe, "volume", adjustment, "revalidate", PARSER_VERSION),
                    report={"status": "ready", "pending_validation": []}, metadata={},
                )
                frame = verified_frame
                self.refresh_snapshot(symbol)
                self.repository.invalidate_all_validations(symbol, adjustment)
                ledger_changed = True
        validation_started = time.perf_counter()
        ledger = self._validate_ledger(
            symbol, timeframe, mode=mode, adjustment=adjustment,
            start=start, end=end, progress=progress, cancelled=cancelled,
        )
        record_seconds("validation", time.perf_counter() - validation_started)
        state = self.repository.validation_state(
            symbol, timeframe, adjustment=adjustment,
            start=requested_start, end=requested_end,
        )
        status = "complete" if state.get("verified_through") and not state.get("failed") and not state.get("pending") else "partial"
        pending = list(state.get("pending") or [])
        if state.get("failed"):
            pending.append(f"{len(state['failed'])} 个交易日未通过校验")
        if not existing_anomalies.empty:
            pending.append(f"存在 {len(existing_anomalies)} 条未解决的无效价格记录")
            status = "partial"
        validation_report = {
            "status": status,
            "mode": mode,
            "checked_days": ledger.get("checked_days", 0),
            "skipped_days": ledger.get("skipped_days", 0),
            "verified_from": state.get("verified_from"),
            "verified_through": state.get("verified_through"),
            "failed": state.get("failed", []),
            "pending_validation": pending,
            "price_anomalies": anomaly_records(
                existing_anomalies, symbol=symbol, timeframe=timeframe, limit=200
            ),
            "ledger_changed": ledger_changed,
        }
        return {
            "symbol": symbol, "timeframe": timeframe, "status": status,
            "pending_validation": pending, "report": validation_report,
            "verified_from": state.get("verified_from"),
            "verified_through": state.get("verified_through"),
        }

    def _validate_ledger(
        self, symbol: str, timeframe: str, *, mode: str, adjustment: str,
        start=None, end=None, progress: Callable[[float, str], None] = lambda *_: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> dict:
        """Validate independent months in parallel and commit from the parent."""
        dataset = self.repository.dataset(symbol, timeframe, adjustment)
        if not dataset:
            raise MarketDataError("没有可校验的本地行情")
        first = pd.Timestamp(start or dataset["first_time"]).normalize()
        last = pd.Timestamp(end or dataset["last_time"]).normalize()
        months: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        cursor = first.to_period("M").start_time.normalize()
        while cursor <= last:
            month_end = (cursor + pd.offsets.MonthBegin(1)).normalize() - pd.Timedelta(days=1)
            months.append((max(first, cursor), min(last, month_end)))
            cursor += pd.offsets.MonthBegin(1)
        configured = self.settings.performance
        workers = min(
            effective_setting(configured, "validation_workers", 4, 4),
            effective_cpu_budget(configured), max(1, len(months)),
            max(1, int(configured.workspace_mb) // 256),
        )
        if len(months) <= 1 or workers <= 1:
            return validate_incremental(
                self, symbol, timeframe, mode=mode, start=first, end=last,
                progress=progress, cancelled=cancelled,
            )
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures = {
            executor.submit(
                _validate_month_worker,
                str(self.store.root), str(self.repository.package_dir),
                symbol, timeframe, adjustment, mode,
                str(month_start.date()), str(month_end.date()),
            ): (month_start, month_end)
            for month_start, month_end in months
        }
        checked = skipped = failed = 0
        completed = 0
        scheduler_poll = float(self.settings.market_data.scheduler_poll_seconds)
        try:
            pending = set(futures)
            while pending:
                if cancelled():
                    raise InterruptedError("任务已取消")
                done, pending = wait(
                    pending, timeout=scheduler_poll, return_when=FIRST_COMPLETED
                )
                for future in done:
                    result = future.result()
                    checked += int(result.get("checked_days", 0) or 0)
                    skipped += int(result.get("skipped_days", 0) or 0)
                    failed += int(result.get("failed_days", 0) or 0)
                    completed += 1
                    record_count("validation_months", 1)
                    progress(
                        completed / max(1, len(months)),
                        f"校验 {timeframe}：{completed}/{len(months)} 个月完成 · {workers} 进程",
                    )
        except BaseException:
            for process in list(getattr(executor, "_processes", {}).values()):
                try:
                    process.terminate()
                except Exception:
                    pass
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=True)
        return {
            "symbol": symbol, "timeframe": timeframe, "mode": mode,
            "checked_days": checked, "skipped_days": skipped,
            "failed_days": failed, "parallel_workers": workers,
        }

    def readiness(self, config: BacktestConfig, version: str | None = None) -> dict:
        symbol = normalize_symbol(config.symbol)
        gaps = []
        instrument = self.store.instrument(symbol) or {}
        asset_type = str(instrument.get("asset_type") or "")
        manifest = self.store.snapshot(symbol, version) if version else self.store.snapshot(symbol)
        if version and not manifest:
            return {
                "ready": False, "symbol": symbol, "frequency": config.frequency,
                "gaps": [{"kind": "version", "message": "DATA_VERSION_NOT_FOUND: 原数据版本已清理"}],
            }
        market_versions = (manifest or {}).get("market_versions", {})
        calendar = self.repository.auxiliary(symbol, "calendar")
        if not calendar or not calendar.get("payload"):
            gaps.append({
                "kind": "calendar", "timeframe": "1day",
                "message": "缺少真实交易日历，无法核验交易日和覆盖完整性",
            })
        daily = (
            self.repository.version_info(symbol, "1day", int(market_versions["1day"]["version"]))
            if version and "1day" in market_versions
            else self.repository.dataset(symbol, "1day")
        )
        if not daily:
            gaps.append({"kind": "daily", "timeframe": "1day", "message": "缺少日线数据，无法确定交易状态和参考价"})
        target = normalize_timeframe(config.frequency)
        requested_end = pd.Timestamp(config.end_date).normalize() if config.end_date else (
            pd.Timestamp(daily.get("last_time")).normalize() if daily and daily.get("last_time") else pd.Timestamp(config.start_date).normalize()
        )
        requested_start = pd.Timestamp(config.start_date).normalize()
        degraded_days: list[dict] = []
        if target in DERIVED_SOURCES and not version:
            try:
                self.ensure_timeframe(
                    symbol, target, start=requested_start, end=requested_end,
                )
            except MarketDataError as exc:
                gaps.append({"kind": "bars", "timeframe": target, "message": str(exc)})
        if target != "1day":
            minute = (
                self.repository.version_info(symbol, target, int(market_versions[target]["version"]))
                if version and target in market_versions
                else self.repository.dataset(symbol, target)
            )
            if not minute:
                gaps.append({"kind": "bars", "timeframe": target, "message": f"缺少 {target} 行情"})
            elif minute and minute.get("file_status") != "readable":
                message = f"{target} 行情文件缺失" if minute.get("file_status") == "missing" else f"{target} 行情文件已被外部修改，需要重新登记和校验"
                gaps.append({"kind": "bars", "timeframe": target, "message": message})
            elif minute:
                contract = self.repository.dataset(symbol, DERIVED_SOURCES.get(target, target))
                if not self._validation_covers(contract, requested_start, requested_end):
                    degradable = self._degradable_minute_days(
                        contract, requested_start, requested_end,
                    )
                    if degradable is None:
                        gaps.append({"kind": "quality", "timeframe": target,
                                     "message": minute.get("reason") or f"{target} 在本次请求区间尚未完成质量核验"})
                    else:
                        degraded_days.extend(degradable)
            if minute and minute.get("file_status") == "readable":
                selected_version = (
                    int(market_versions[target]["version"])
                    if version and target in market_versions
                    else int(minute.get("current_version") or 0) or None
                )
                minute_probe = self.repository.read_frame(
                    symbol, target,
                    version=selected_version, start=requested_start, end=requested_end,
                    fields=("timestamp", "open", "high", "low", "close", "volume"),
                )
                price_issues = _price_anomaly_rows(
                    minute_probe, requested_start, requested_end
                )
                if not price_issues.empty:
                    gaps.append({
                        "kind": "price_quality", "timeframe": target,
                        "message": f"{target} 在请求区间存在 {len(price_issues)} 条无效价格，修正前不能用于计算",
                        "anomalies": anomaly_records(
                            price_issues, symbol=symbol, timeframe=target,
                            version=selected_version, limit=50,
                        ),
                    })
        if daily:
            if daily.get("file_status") != "readable":
                message = "日线行情文件缺失" if daily.get("file_status") == "missing" else "日线行情文件已被外部修改，需要重新登记和校验"
                gaps.append({"kind": "daily", "timeframe": "1day", "message": message})
            if daily.get("first_time") and pd.Timestamp(config.start_date) < pd.Timestamp(daily["first_time"]):
                gaps.append({"kind": "range", "timeframe": "1day", "message": "开始日期早于本地日线覆盖"})
            if config.end_date and daily.get("last_time") and pd.Timestamp(config.end_date) > pd.Timestamp(daily["last_time"]):
                gaps.append({"kind": "range", "timeframe": "1day", "message": "结束日期晚于本地日线覆盖"})
            if not self._validation_covers(daily, requested_start, requested_end):
                gaps.append({"kind": "quality", "timeframe": "1day", "message": daily.get("reason") or "日线在本次请求区间尚未完成质量核验"})
            daily_version = (
                int(market_versions["1day"]["version"])
                if version and "1day" in market_versions
                else int(daily.get("current_version") or 0) or None
            )
            daily_rows = self.repository.bars(
                symbol, "1day", version=daily_version,
                start=requested_start - pd.Timedelta(days=30), end=requested_end,
            )
            daily_price_issues = _price_anomaly_rows(
                daily_rows, requested_start, requested_end
            )
            if not daily_price_issues.empty:
                gaps.append({
                    "kind": "price_quality", "timeframe": "1day",
                    "message": f"日线在请求区间存在 {len(daily_price_issues)} 条无效价格，修正前不能用于计算",
                    "anomalies": anomaly_records(
                        daily_price_issues, symbol=symbol, timeframe="1day",
                        version=daily_version, limit=50,
                    ),
                })
            required = ["open", "high", "low", "close", "volume", "preclose", "tradestatus"]
            missing_fields = [field for field in required if field not in daily_rows or daily_rows[field].isna().any()]
            if missing_fields:
                gaps.append({"kind": "fields", "timeframe": "1day",
                             "message": "日线缺少回测字段或空值：" + ", ".join(missing_fields)})
            if asset_type == "stock":
                st_state = daily_rows.get("is_st")
                if st_state is None or st_state.isna().any():
                    gaps.append({"kind": "fields", "timeframe": "1day",
                                 "message": "日线缺少完整 ST 历史状态，无法用于严格回测"})
            auxiliary_versions = (manifest or {}).get("auxiliary_versions", {})
            actions = self.repository.auxiliary(
                symbol, "actions", version=auxiliary_versions.get("actions")
            )
            action_days = set()
            if actions:
                action_frame = pd.DataFrame(actions["payload"])
                if not action_frame.empty and "ex_date" in action_frame:
                    action_days = set(pd.to_datetime(action_frame.ex_date, errors="coerce").dropna().dt.normalize())
            observed = daily_rows[(pd.to_numeric(daily_rows.tradestatus, errors="coerce") == 1) & (pd.to_numeric(daily_rows.volume, errors="coerce") > 0)].copy()
            if not observed.empty:
                observed["date"] = pd.to_datetime(observed.timestamp).dt.normalize()
                prior = observed[observed.date < pd.Timestamp(config.start_date)]
                if prior.empty:
                    gaps.append({"kind": "reference_price", "timeframe": "1day",
                                 "message": "开始日期前缺少有效交易日收盘价，请补足前置历史"})
                observed["previous_close"] = observed.close.shift(1)
                tick = float(self.store.instrument(symbol).get("price_tick", 0.01)) if self.store.instrument(symbol) else 0.01
                unexplained = (observed.preclose - observed.previous_close).abs() > tick * 1.1
                unknown = [str(day.date()) for day in observed.loc[unexplained.fillna(False), "date"] if day not in action_days]
                if unknown:
                    gaps.append({"kind": "actions", "timeframe": "1day",
                                 "message": "除权参考价发生变化但缺少公司行动资料：" + ", ".join(unknown[:8])})
            if asset_type == "etf" and not (actions and actions.get("verified")):
                gaps.append({"kind": "actions", "timeframe": "1day", "message": "ETF 分红或拆分资料未完成核验"})
        if target != "1day":
            target_dataset = self.repository.dataset(symbol, target)
            if target_dataset:
                if target_dataset.get("first_time") and pd.Timestamp(config.start_date) < pd.Timestamp(target_dataset["first_time"]):
                    gaps.append({"kind": "range", "timeframe": target, "message": f"开始日期早于本地 {target} 覆盖"})
                if config.end_date and target_dataset.get("last_time") and pd.Timestamp(config.end_date) > pd.Timestamp(target_dataset["last_time"]):
                    gaps.append({"kind": "range", "timeframe": target, "message": f"结束日期晚于本地 {target} 覆盖"})
        warnings = []
        if degraded_days:
            dates = ", ".join(item["date"] for item in degraded_days[:8])
            total_days = len(self.repository.trading_days(symbol, requested_start, requested_end))
            ratio = 1 - len(degraded_days) / max(1, total_days)
            warnings.append(
                f"{len(degraded_days)} 个交易日将按已校验日线降级：{dates}；"
                f"分钟覆盖率 {ratio:.2%}"
            )
        return {"ready": not gaps, "symbol": symbol, "frequency": config.frequency,
                "asset_type": asset_type, "gaps": gaps,
                "degraded_days": degraded_days, "warnings": warnings,
                "checked_start": str(config.start_date),
                "checked_end": str(config.end_date) if config.end_date else None,
                "package_id": self.repository.package_id,
                "preparation": {"symbol": symbol, "frequency": target,
                "start": str(config.start_date), "end": str(config.end_date) if config.end_date else None}}

    def _validation_covers(self, dataset: dict | None, start, end) -> bool:
        if not dataset:
            return False
        if dataset.get("file_status") not in {None, "readable"}:
            return False
        state = self.repository.validation_state(
            str(dataset.get("symbol")), str(dataset.get("timeframe")),
            adjustment=str(dataset.get("adjustment") or "none"),
            start=start, end=end,
        )
        verified_from = state.get("verified_from")
        verified_through = state.get("verified_through")
        if not verified_from or not verified_through:
            return False
        # Weekend boundaries may sit outside the last trading date. A weekday
        # beyond both the source and the known calendar is still uncovered.
        known_start = min(pd.Timestamp(dataset["first_time"]).normalize(), pd.Timestamp(state.get("calendar_start") or dataset["first_time"]).normalize())
        known_end = max(pd.Timestamp(dataset["last_time"]).normalize(), pd.Timestamp(state.get("calendar_end") or dataset["last_time"]).normalize())
        if pd.offsets.BDay().rollforward(pd.Timestamp(start).normalize()) < known_start:
            return False
        if pd.offsets.BDay().rollback(pd.Timestamp(end).normalize()) > known_end:
            return False
        return (
            pd.Timestamp(verified_from).normalize() <= pd.Timestamp(state.get("expected_start") or start).normalize()
            and pd.Timestamp(verified_through).normalize() >= pd.Timestamp(state.get("expected_end") or end).normalize()
            and not state.get("failed") and not state.get("pending")
        )

    def _degradable_minute_days(self, dataset: dict | None, start, end) -> list[dict] | None:
        """Return failed minute sessions that can use a verified daily row.

        The engine already supports mixed daily/minute bundles.  Readiness was
        rejecting every failed minute day before that fallback could be used.
        Only bounded, genuine intraday gaps are accepted here; invalid prices,
        missing daily bars and unvalidated pending days remain hard failures.
        """
        if not dataset:
            return None
        symbol = str(dataset.get("symbol"))
        timeframe = str(dataset.get("timeframe"))
        adjustment = str(dataset.get("adjustment") or "none")
        state = self.repository.validation_state(
            symbol, timeframe, adjustment=adjustment, start=start, end=end,
        )
        if not state.get("calendar_available") or state.get("pending"):
            return None
        failed = [str(day) for day in (state.get("failed") or [])]
        if not failed:
            return None
        expected = self.repository.trading_days(symbol, start, end)
        expected_start = state.get("expected_start")
        expected_end = state.get("expected_end")
        if (
            not len(expected) or not expected_start or not expected_end
            or pd.Timestamp(expected_start).normalize() != expected[0].normalize()
            or pd.Timestamp(expected_end).normalize() != expected[-1].normalize()
            or len(failed) / len(expected) > MAX_DAILY_FALLBACK_RATIO
        ):
            return None
        records = {
            str(row["trade_date"]): row
            for row in self.repository.day_validations(
                symbol, timeframe, adjustment=adjustment, start=start, end=end,
            )
        }
        daily = self.repository.bars(symbol, "1day", start=start, end=end)
        if daily.empty:
            return None
        daily["_day"] = pd.to_datetime(daily["timestamp"]).dt.normalize()
        daily_by_day = {
            pd.Timestamp(day).normalize(): part.iloc[-1]
            for day, part in daily.groupby("_day", sort=True)
        }
        degraded = []
        for day in failed:
            record = records.get(day) or {}
            report = record.get("report") or {}
            reason = str(report.get("reason") or "")
            unfillable = int(report.get("unfillable") or 0)
            if int(report.get("blocked_price_rows") or 0) > 0 or report.get("price_anomalies"):
                return None
            if unfillable <= 0 and "缺口" not in reason and "未补齐" not in reason:
                return None
            row = daily_by_day.get(pd.Timestamp(day).normalize())
            if row is None:
                return None
            try:
                status = int(row.get("tradestatus"))
                volume = float(row.get("volume"))
                prices = [float(row.get(field)) for field in ("open", "high", "low", "close")]
            except (TypeError, ValueError):
                return None
            if status not in (0, 1) or volume <= 0 or any(
                not math.isfinite(price) or price <= 0 for price in prices
            ):
                return None
            degraded.append({
                "date": day, "timeframe": timeframe,
                "reason": reason or f"{unfillable} 根分钟缺口",
            })
        return degraded

    def load_backtest_bundle(self, config: BacktestConfig, version: str | None = None):
        target = normalize_timeframe(config.frequency)
        if version is None and target in DERIVED_SOURCES:
            self.ensure_timeframe(
                config.symbol, target, start=config.start_date, end=config.end_date,
            )
        from .consumer import load_native_bundle
        return load_native_bundle(self, config, version)

    def refresh_snapshot(self, symbol: str) -> dict | None:
        market = self.repository.active_manifest(symbol)
        if not market:
            return None
        instrument = self.store.instrument(symbol) or instrument_terms(symbol, {"name": symbol})
        versions = market["market_versions"]
        daily = versions.get("1day", {})
        metadata = {
            **instrument, "market_data": True, "is_demo": False,
            "daily_through": daily.get("last", "")[0:10] if daily else None,
            "minute_through": max((item.get("last") or "" for name, item in versions.items() if name.endswith("min")), default=None),
            "actions_verified": self.repository.auxiliary(symbol, "actions") is not None,
            "warnings": [], "quality_warnings": [],
        }
        market["metadata"] = metadata
        # The version was computed before metadata was attached; retain its
        # identity because the referenced market database versions are stable.
        self.store.put_market_snapshot(market, symbol)
        self.repository.package.write_version_manifest(market["version"], market)
        return market

    def _calendar(self, symbol: str, *, path=None):
        if path:
            frame, _, _, _ = read_file(path)
            if "date" not in frame:
                raise MarketDataError("交易日历必须包含 date 列")
            return frame[['date']]
        saved = self.repository.auxiliary(symbol, "calendar")
        return pd.DataFrame(saved["payload"]) if saved else None


def _params_hash(timeframe, label_convention, adjustment, source, parser_version) -> str:
    return hashlib.sha256(
        f"{timeframe}|{label_convention}|{adjustment}|{source}|{parser_version}|price-q{PRICE_VALIDATION_VERSION}".encode()
    ).hexdigest()


def _merge_input_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise MarketDataError("没有可导入的行情文件")
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    duplicates = combined[combined.duplicated("timestamp", keep=False)]
    if not duplicates.empty:
        conflicts = []
        for stamp, group in duplicates.groupby("timestamp"):
            values = group[["open", "high", "low", "close", "volume", "turnover"]].round(10).drop_duplicates()
            if len(values) > 1:
                conflicts.append(str(stamp))
        if conflicts:
            raise MarketDataError("同一标的的多个文件在重叠时间存在不同行情：" + ", ".join(conflicts[:8]))
    return combined.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
