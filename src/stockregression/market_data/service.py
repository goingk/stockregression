from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import BacktestConfig, normalize_symbol
from ..rules import instrument_terms
from ..storage import Store
from .models import FileImportResult, MarketDataError
from .quality import validate_and_fill
from .reader import (
    check_declared_adjustment, check_declared_identity, check_declared_timeframe, file_fingerprint,
    normalize_frame, read_file, scan_folder,
)
from .repository import MarketDataRepository, PARSER_VERSION
from .timeframes import MINUTE_TIMEFRAMES, TIMEFRAMES, normalize_timeframe, timeframe_spec
from .validation import validate_incremental


BASE_TIMEFRAMES = ("1day", "1min")
DERIVED_SOURCES = {
    "5min": "1min", "15min": "1min", "30min": "1min", "60min": "1min",
    "1week": "1day", "1month": "1day",
}
SOURCE_TO_TARGETS = {
    "1min": ("5min", "15min", "30min", "60min"),
    "1day": ("1week", "1month"),
}


class MarketDataService:
    def __init__(self, store_or_root: Store | str | Path, package_dir: str | Path | None = None):
        self.store = store_or_root if isinstance(store_or_root, Store) else Store(store_or_root)
        self.repository = MarketDataRepository(self.store.root, package_dir=package_dir)

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
        # Existing snapshots remain visible until explicitly migrated.
        legacy_key = "1min" if timeframe == "1min" else "1day" if timeframe == "1day" else None
        if legacy_key:
            manifest = self.store.snapshot(symbol)
            if manifest and manifest.get("market_data") is not True:
                files = manifest.get("files", {})
                if files.get("minute" if legacy_key == "1min" else "daily"):
                    return {
                        "symbol": symbol, "timeframe": timeframe, "row_count": None,
                        "status": "legacy", "ready": False, "legacy": True,
                        "reason": "旧 Parquet 快照；可浏览，迁移后进入受管 Parquet 行情库",
                        "version": manifest.get("version"), "storage": "legacy_parquet",
                    }
        raise MarketDataError(f"{symbol} 尚无 {timeframe} 数据")

    def compatibility_coverage(self, symbol: str) -> dict:
        """Project frequency-database state into the legacy coverage response."""
        symbol = normalize_symbol(symbol)
        daily = self.repository.coverage(symbol, "1day")
        minute = self.repository.coverage(symbol, "1min")
        if not daily and not minute:
            raise MarketDataError(f"{symbol} 尚无本地 Parquet 行情")
        primary = daily or minute
        days = []
        for segment in (daily or primary).get("segments", []):
            report = segment.get("report") or {}
            days.append({
                "date": str(segment.get("start_time", ""))[:10], "mode": "daily" if daily else "minute",
                "missing_minutes": report.get("missing_rows", 0), "missing_data_minutes": report.get("unfillable", 0),
                "auction_placeholders": 0, "reason": f"检查至 {str(segment.get('end_time', ''))[:10]}",
            })
        bytes_used = 0
        for timeframe in ("1day", "1min"):
            dataset = self.repository.dataset(symbol, timeframe)
            if dataset:
                bytes_used += int(dataset.get("current_bytes") or 0)
        daily_days = self.repository.observed_days(symbol, "1day") if daily else 0
        minute_days = self.repository.observed_days(symbol, "1min") if minute else 0
        diagnostics = {
            "schema_version": 1, "status": "complete" if primary.get("status") == "ready" else "partial",
            "requested_frequency": "1m" if minute else "1d", "minute_quality_applicable": bool(minute),
            "daily_days": daily_days, "minute_days": minute_days,
            "total_days": daily_days, "minute_ratio": 1 if minute and minute.get("status") == "ready" else 0,
            "synthetic_bars": (minute or {}).get("synthetic_rows", 0), "auction_placeholders": 0,
            "current_warnings": [primary.get("reason")] if primary.get("reason") else [],
        }
        return {
            "symbol": symbol, "version": primary.get("current_version"), "rows": (minute or daily).get("row_count", 0),
            "first_date": str((daily or minute).get("first_time", ""))[:10], "last_date": str((daily or minute).get("last_time", ""))[:10],
            "days": days, "profiles": {}, "warnings": diagnostics["current_warnings"],
            "diagnostics": diagnostics, "source_events": [], "quality_rules_version": 3,
            "is_demo": False, "bytes": bytes_used, "storage": primary.get("storage", "parquet"),
            "minute_days": diagnostics["minute_days"], "daily_days": diagnostics["daily_days"],
            "total_days": diagnostics["total_days"], "minute_ratio": diagnostics["minute_ratio"],
            "synthetic_bars": diagnostics["synthetic_bars"], "auction_placeholders": 0,
        }

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
            return target
        legacy = bool(target and not derived)
        full_target_range = bool(
            target
            and request_start is not None and request_end is not None
            and target.get("first_time") and target.get("last_time")
            and pd.Timestamp(request_start).normalize() <= pd.Timestamp(target["first_time"]).normalize()
            and pd.Timestamp(request_end).normalize() >= pd.Timestamp(target["last_time"]).normalize()
        )
        snapshot = legacy or bool(target and target.get("derived_stale") and full_target_range)
        from .aggregation import AggregationService

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
            manifest = self.store.snapshot(symbol)
            if manifest and manifest.get("market_data") is not True and timeframe in {"1min", "1day"}:
                return self._legacy_bars(symbol, timeframe, start, end, limit)
            raise MarketDataError(f"{symbol} 尚无 {timeframe} 数据")
        if timeframe in DERIVED_SOURCES and version is not None and not dataset.get("derived"):
            raise MarketDataError(
                f"历史 {timeframe} 文件仅用于追溯；派生周期须由对应基础行情生成"
            )
        frame = self.repository.bars(symbol, timeframe, start=start, end=end, version=version, limit=limit)
        selected = int(version) if version is not None else int(dataset["current_version"])
        detail = self.repository.version_info(symbol, timeframe, selected) if selected != int(dataset["current_version"]) else None
        return {
            "symbol": symbol, "timeframe": timeframe, "version": selected,
            "rows": frame.to_dict("records"), "count": len(frame),
            "first_time": (detail or dataset)["first_time"], "last_time": (detail or dataset)["last_time"],
            "status": (detail or dataset)["status"], "storage": dataset.get("storage", "parquet"),
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
            else pd.Timestamp(end) + pd.Timedelta(nanoseconds=1) if end is not None else None
        )
        frame = self.repository.last_bars(
            symbol, timeframe, before=upper_exclusive,
            limit=limit + warmup + 1,
        )
        if start:
            frame = frame[frame.timestamp >= pd.Timestamp(start)]
        for period in periods:
            frame[f"ma{period}"] = frame.close.rolling(period, min_periods=period).mean()
        more = len(frame) > limit
        frame = frame.tail(limit).copy()
        daily_upper = (
            pd.Timestamp(upper) + pd.Timedelta(nanoseconds=1)
            if upper is not None and before is None else upper
        )
        daily = self.repository.last_bars(symbol, "1day", before=daily_upper, limit=1)
        quote = daily.iloc[-1].to_dict() if not daily.empty else None
        warnings = []
        if dataset["status"] != "ready":
            warnings.append(dataset.get("reason") or f"{timeframe} 数据有待核验事项")
        frame["frequency"] = timeframe
        if quote:
            quote["timestamp"] = pd.Timestamp(quote["timestamp"]).isoformat()
            quote["date"] = pd.Timestamp(quote["timestamp"]).date().isoformat()
        return {
            "symbol": symbol, "version": dataset["current_version"], "frequency": timeframe,
            "bars": frame.to_dict("records"), "quote": quote, "has_more": more,
            "next_before": str(frame.timestamp.iloc[0]) if more and len(frame) else None,
            "is_demo": False, "warnings": warnings, "quality_flags": warnings,
            "diagnostics": {}, "source_events": [], "storage": dataset.get("storage", "parquet"),
            "instrument": self.store.instrument(symbol) or instrument_terms(symbol, {"name": symbol}),
        }

    def _legacy_bars(self, symbol, timeframe, start, end, limit):
        manifest = self.store.snapshot(symbol)
        files = manifest["files"].get("minute" if timeframe == "1min" else "daily", [])
        frame = self.store.read_files(files, "timestamp" if timeframe == "1min" else "date", start, end)
        if limit:
            frame = frame.tail(limit)
        column = "timestamp" if timeframe == "1min" else "date"
        if column in frame:
            frame["timestamp"] = pd.to_datetime(frame[column])
        return {"symbol": symbol, "timeframe": timeframe, "version": manifest["version"],
                "rows": frame.to_dict("records"), "count": len(frame), "status": "legacy", "storage": "legacy_parquet"}

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
        etf_category=None, resume=True, calendar_path=None, workers=2, skip_invalid_rows=False,
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
        batch_size = 128
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
    ) -> dict:
        symbol, timeframe = normalize_symbol(symbol), normalize_timeframe(timeframe)
        dataset = self.repository.dataset(symbol, timeframe, adjustment)
        if not dataset:
            raise MarketDataError("没有可校验的本地行情")
        requested_start = start or dataset.get("first_time")
        requested_end = end or dataset.get("last_time")
        if (
            mode == "incremental" and label_convention is None
            and self._validation_covers(dataset, requested_start, requested_end)
        ):
            state = self.repository.validation_state(
                symbol, timeframe, adjustment=adjustment,
                start=requested_start, end=requested_end,
            )
            if not state.get("failed") and not state.get("pending"):
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
        frame = self.repository.active_rows(symbol, timeframe, adjustment)
        if frame.empty:
            raise MarketDataError("没有可校验的本地行情")
        if mode not in {"incremental", "full"}:
            raise MarketDataError("校验模式应为 incremental 或 full")
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
        ledger = validate_incremental(
            self, symbol, timeframe, mode=mode, start=start, end=end,
        )
        state = self.repository.validation_state(symbol, timeframe, adjustment=adjustment)
        status = "complete" if state.get("verified_through") and not state.get("failed") else "partial"
        pending = list(state.get("pending") or [])
        if state.get("failed"):
            pending.append(f"{len(state['failed'])} 个交易日未通过校验")
        validation_report = {
            "status": status,
            "mode": mode,
            "checked_days": ledger.get("checked_days", 0),
            "skipped_days": ledger.get("skipped_days", 0),
            "verified_from": state.get("verified_from"),
            "verified_through": state.get("verified_through"),
            "failed": state.get("failed", []),
            "pending_validation": pending,
            "ledger_changed": ledger_changed,
        }
        return {
            "symbol": symbol, "timeframe": timeframe, "status": status,
            "pending_validation": pending, "report": validation_report,
            "verified_from": state.get("verified_from"),
            "verified_through": state.get("verified_through"),
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
        legacy = None if version else manifest
        if not daily and not (legacy and not legacy.get("market_data") and legacy.get("files", {}).get("daily")):
            gaps.append({"kind": "daily", "timeframe": "1day", "message": "缺少日线数据，无法确定交易状态和参考价"})
        target = normalize_timeframe(config.frequency)
        requested_end = pd.Timestamp(config.end_date).normalize() if config.end_date else (
            pd.Timestamp(daily.get("last_time")).normalize() if daily and daily.get("last_time") else pd.Timestamp(config.start_date).normalize()
        )
        requested_start = pd.Timestamp(config.start_date).normalize()
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
            if not minute and not (legacy and legacy.get("files", {}).get("minute")):
                gaps.append({"kind": "bars", "timeframe": target, "message": f"缺少 {target} 行情"})
            elif minute and minute.get("file_status") != "readable":
                message = f"{target} 行情文件缺失" if minute.get("file_status") == "missing" else f"{target} 行情文件已被外部修改，需要重新登记和校验"
                gaps.append({"kind": "bars", "timeframe": target, "message": message})
            elif minute and not self._validation_covers(
                self.repository.dataset(symbol, DERIVED_SOURCES.get(target, target)),
                requested_start, requested_end,
            ):
                gaps.append({"kind": "quality", "timeframe": target,
                             "message": minute.get("reason") or f"{target} 在本次请求区间尚未完成质量核验"})
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
                if version and "1day" in market_versions else None
            )
            daily_rows = self.repository.bars(
                symbol, "1day", version=daily_version
            )
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
        return {"ready": not gaps, "symbol": symbol, "frequency": config.frequency,
                "asset_type": asset_type, "gaps": gaps,
                "checked_start": str(config.start_date),
                "checked_end": str(config.end_date) if config.end_date else None,
                "package_id": self.repository.package_id,
                "preparation": {"symbol": symbol, "frequency": target,
                "start": str(config.start_date), "end": str(config.end_date) if config.end_date else None}}

    @staticmethod
    def _validation_covers(dataset: dict | None, start, end) -> bool:
        if not dataset:
            return False
        verified_from = dataset.get("verified_from")
        verified_through = dataset.get("verified_through")
        if not verified_from or not verified_through:
            return False
        return (
            pd.Timestamp(verified_from).normalize() <= pd.Timestamp(start).normalize()
            and pd.Timestamp(verified_through).normalize() >= pd.Timestamp(end).normalize()
        )

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
        legacy = self.store.snapshot(symbol)
        if legacy and legacy.get("files", {}).get("calendar"):
            return self.store.read_files(legacy["files"]["calendar"])
        saved = self.repository.auxiliary(symbol, "calendar")
        return pd.DataFrame(saved["payload"]) if saved else None


def _params_hash(timeframe, label_convention, adjustment, source, parser_version) -> str:
    return hashlib.sha256(f"{timeframe}|{label_convention}|{adjustment}|{source}|{parser_version}".encode()).hexdigest()


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
