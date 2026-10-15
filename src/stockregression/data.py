from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import BacktestConfig, normalize_symbol
from .providers import fetch_auxiliary, fetch_hf, fetch_instruments, fetch_tdx, fetch_sina, fetch_reference_5m, latest_closed_date, normalize_actions
from .quality import DataQualityError, align_minutes, canonical_frame, infer_profile, minute_slots, prepare_bundle
from .settings import load_settings
from .storage import Store


def merge_frames(old: pd.DataFrame, new: pd.DataFrame, key: str) -> pd.DataFrame:
    parts = [part for part in (old, new) if not part.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(key, kind="stable").drop_duplicates(key, keep="last").reset_index(drop=True)


class DataService:
    def __init__(self, store: Store):
        self.store = store

    def refresh_instruments(self, progress=lambda *_: None, cancelled=lambda: False):
        progress(0.1, "下载固定版本的证券覆盖清单")
        records = fetch_instruments(load_settings().hf_revision)
        for record in records:
            if cancelled():
                raise InterruptedError("任务已取消")
            self.store.put_instrument(**record)
        return {"count": len(records), "message": f"已更新 {len(records)} 只沪深股票；未过滤 ST 或退市历史"}

    def sync(self, symbols: list[str], mode: str, progress=lambda *_: None, cancelled=lambda: False):
        if not symbols:
            symbols = [row["symbol"] for row in self.store.watches()]
        if not symbols:
            return {"symbols": [], "message": "自选股为空，本次无需更新", "warnings": []}
        results, failures = [], []
        with self.store.lock:
            for index, raw_symbol in enumerate(symbols):
                if cancelled():
                    raise InterruptedError("任务已取消")
                symbol = normalize_symbol(raw_symbol)
                current = self.store.snapshot(symbol)
                if mode != "history" and current and current["metadata"].get("is_demo"):
                    results.append({"symbol": symbol, "skipped": True, "message": "演示数据不参与自动真实行情更新"})
                    continue
                prefix = index / len(symbols)
                def report(fraction, message):
                    progress(prefix + fraction / len(symbols), f"{symbol}：{message}")
                try:
                    results.append(self._sync_one(symbol, mode, report, cancelled))
                except InterruptedError:
                    raise
                except Exception as exc:
                    failures.append({"symbol": symbol, "reason": str(exc)})
                self.store.state(f"sync:{symbol}", {"mode": mode, "result": results[-1] if results and results[-1].get("symbol") == symbol else None, "error": next((r["reason"] for r in failures if r["symbol"] == symbol), None)})
        if failures and not results:
            raise RuntimeError("；".join(row["symbol"] + " " + row["reason"] for row in failures))
        return {"symbols": results, "failures": failures, "message": f"完成 {len(results)} 只，失败 {len(failures)} 只"}

    def _sync_one(self, symbol, mode, progress, cancelled):
        previous = self.store.snapshot(symbol)
        if previous and previous["metadata"].get("is_demo"):
            previous = None  # Never mix simulated records into a real-data snapshot.
        old_files = previous["files"] if previous else {}
        old_minute = self.store.read_files(old_files.get("minute", []))
        old_daily = self.store.read_files(old_files.get("daily", []))
        warnings = []
        hf = pd.DataFrame()
        profiles = dict(previous.get("profiles", {})) if previous else {}
        if mode == "history" or previous is None:
            progress(0.05, "按股票下载历史分钟 Parquet")
            try:
                hf = fetch_hf(symbol, self.store.root, load_settings().hf_revision)
            except Exception as exc:
                warnings.append("历史分钟下载失败：" + str(exc))
        start = (old_daily.date.max() - pd.Timedelta(days=30)) if mode != "history" and not old_daily.empty else pd.Timestamp("2010-01-01")
        end = latest_closed_date()
        progress(0.25, "获取不复权日线、交易日历和公司行动")
        daily, actions, calendar, metadata = fetch_auxiliary(symbol, start, end, cancelled)
        daily = merge_frames(old_daily, daily, "date")
        actions = merge_frames(self.store.read_files(old_files.get("actions", [])), actions, "ex_date")
        calendar = merge_frames(self.store.read_files(old_files.get("calendar", [])), calendar, "date")
        metadata["board"] = "star" if symbol.startswith("688") else "chinext" if symbol.startswith("3") else "main"
        metadata["hf_revision"] = load_settings().hf_revision
        metadata["is_demo"] = False
        self.store.put_instrument(symbol, **{key: metadata.get(key) for key in ("name", "listing_date", "delisting_date")})
        self.store.add_watch(symbol, metadata.get("name", symbol))
        if metadata.get("listing_date"):
            calendar = calendar[calendar.date >= pd.Timestamp(metadata["listing_date"])].reset_index(drop=True)
        if metadata.get("delisting_date"):
            calendar = calendar[calendar.date < pd.Timestamp(metadata["delisting_date"])].reset_index(drop=True)
        progress(0.55, "采集通达信分钟窗口并核对标签与单位")
        recent = pd.DataFrame()
        earliest = end - pd.Timedelta(days=14) if mode == "daily" and previous else None
        try:
            recent_raw = fetch_tdx(symbol, earliest, cancelled)
            profiles["tdx"] = infer_profile(recent_raw, daily, "tdx")
            recent = align_minutes(recent_raw, profiles["tdx"]["convention"], profiles["tdx"]["volume_multiplier"])
            recent["source_verified"] = profiles["tdx"]["verified"]
            raw_folder = self.store.root / "raw" / "tdx" / symbol
            raw_folder.mkdir(parents=True, exist_ok=True)
            recent_raw.to_parquet(raw_folder / "latest.parquet", index=False)
        except InterruptedError:
            raise
        except Exception as exc:
            warnings.append("近期分钟源不可用：" + str(exc))
            progress(0.62, "通达信不可用，尝试新浪近期分钟备源")
            try:
                recent_raw = fetch_sina(symbol)
                reference = fetch_reference_5m(symbol, recent_raw.timestamp.max().normalize() - pd.Timedelta(days=21), recent_raw.timestamp.max().normalize())
                profiles["sina"] = infer_profile(recent_raw, daily, "sina", reference=reference, reference_frequency="5m")
                recent = align_minutes(recent_raw, profiles["sina"]["convention"], profiles["sina"]["volume_multiplier"])
                recent["source_verified"] = profiles["sina"]["verified"]
                raw_folder = self.store.root / "raw" / "sina" / symbol
                raw_folder.mkdir(parents=True, exist_ok=True)
                recent_raw.to_parquet(raw_folder / "latest.parquet", index=False)
            except Exception as backup_error:
                warnings.append("新浪分钟备源不可用或未校准：" + str(backup_error))
        if not hf.empty:
            profiles["hf"] = infer_profile(hf, daily, "hf", recent if not recent.empty and profiles.get("tdx", {}).get("verified") else None)
            if not profiles["hf"]["verified"]:
                progress(0.72, "用独立历史5分钟行情核验历史分钟时间标签")
                try:
                    reference_end = hf.timestamp.max().normalize() - pd.Timedelta(days=1)
                    reference = fetch_reference_5m(symbol, reference_end - pd.Timedelta(days=21), reference_end)
                    # Only the bounded reference window is needed for calibration.
                    sample = hf[hf.timestamp.between(reference.timestamp.min() - pd.Timedelta(minutes=5), reference.timestamp.max())]
                    profiles["hf"] = infer_profile(sample, daily, "hf", reference=reference, reference_frequency="5m")
                except Exception as exc:
                    warnings.append("历史分钟独立校验失败：" + str(exc))
            hf = align_minutes(hf, profiles["hf"]["convention"], profiles["hf"]["volume_multiplier"])
            hf["source_verified"] = profiles["hf"]["verified"]
        # Prefer the explicitly calibrated recent provider in its overlap window.
        minute = merge_frames(merge_frames(old_minute, hf, "timestamp"), recent, "timestamp")
        if not minute.empty:
            minute = minute[minute.timestamp < end + pd.Timedelta(days=1)].copy()
        metadata["warnings"] = warnings
        progress(0.85, "发布不可变分区和数据版本")
        manifest = self.store.publish(symbol, minute, daily, actions, calendar, metadata, profiles)
        progress(1, "完成；质量报告已可查看")
        return {"symbol": symbol, "version": manifest["version"], "last_date": str(daily.date.max().date()), "rows": len(minute), "warnings": warnings, "profiles": profiles}

    def load_bundle(self, config: BacktestConfig, version: str | None = None):
        manifest = self.store.snapshot(config.symbol, version)
        if not manifest:
            raise DataQualityError("该股票尚无本地数据，请先下载、导入或生成演示数据")
        files, metadata = manifest["files"], manifest["metadata"]
        daily_all = self.store.read_files(files["daily"])
        if daily_all.empty:
            raise DataQualityError("本地版本没有日线")
        end = pd.Timestamp(config.end_date) if config.end_date else daily_all.date.max()
        if end > daily_all.date.max():
            raise DataQualityError(f"日线完整截止到 {daily_all.date.max().date()}，请补数或缩短结束日期")
        minute = self.store.read_files(files["minute"], "timestamp", config.start_date, end)
        actions = self.store.read_files(files["actions"])
        calendar = self.store.read_files(files["calendar"])
        # A changed exchange reference price requires a modelled action, not qfq.
        observed = daily_all[(daily_all.tradestatus == 1) & (daily_all.volume > 0)].copy()
        previous_close = observed.close.shift(1)
        unexplained = (observed.preclose - previous_close).abs() > 0.011
        action_days = set(actions.ex_date.dt.normalize()) if not actions.empty else set()
        unexplained &= observed.date.between(pd.Timestamp(config.start_date), end)
        unknown = [str(day.date()) for day in observed.loc[unexplained, "date"] if day not in action_days]
        if unknown:
            raise DataQualityError("除权参考价发生变化但缺少公司行动资料：" + ", ".join(unknown[:20]))
        return prepare_bundle(config.symbol, minute, daily_all, actions, calendar, config.start_date, end, config.frequency, manifest["version"], metadata, manifest["profiles"])

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
        with self.store.lock:
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
                if not required.issubset(raw_daily.columns) or not ({"is_st", "isST"} & set(raw_daily.columns)):
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
                profiles["import"] = infer_profile(raw, daily, "import", declared=declared if declared in ("start", "end") else None)
                imported = align_minutes(raw, profiles["import"]["convention"], profiles["import"]["volume_multiplier"])
                imported["source_verified"] = profiles["import"]["verified"]
                minute = merge_frames(minute, imported, "timestamp")
            actions = normalize_actions(read(payload["actions_path"])) if payload.get("actions_path") else self.store.read_files(files.get("actions", []))
            metadata = dict(previous.get("metadata", {})) if previous else {}
            known = self.store.instruments(symbol, limit=1)
            metadata["board"] = "star" if symbol.startswith("688") else "chinext" if symbol.startswith("3") else "main"
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
            self.store.add_watch(symbol)
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
        with self.store.lock:
            manifest = self.store.publish(symbol, minute, daily, actions, calendar, metadata, profiles)
            self.store.add_watch(symbol, metadata["name"])
        return {"symbol": symbol, "version": manifest["version"], "is_demo": True, "days": days, "message": "演示数据已生成；包含短缺口与日线降级示例"}
