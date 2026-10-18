from __future__ import annotations

import hashlib

import pandas as pd

from ..models import normalize_symbol
from .models import MarketDataError
from .reader import normalize_frame
from .service import MarketDataService
from .timeframes import MINUTE_TIMEFRAMES, normalize_timeframe, timeframe_spec


class AggregationService:
    def __init__(self, service: MarketDataService):
        self.service = service
        self.repository = service.repository

    def aggregate(self, symbol, source_timeframe, target_timeframe, *, start=None, end=None,
                  snapshot: bool = False, progress=lambda *_: None, cancelled=lambda: False):
        symbol = normalize_symbol(symbol)
        source_timeframe = normalize_timeframe(source_timeframe)
        target_timeframe = normalize_timeframe(target_timeframe)
        self._validate_pair(source_timeframe, target_timeframe)
        existing = self.repository.dataset(symbol, target_timeframe)
        if existing and not existing.get("derived"):
            snapshot = True
        source = self.repository.dataset(symbol, source_timeframe)
        if not source:
            raise MarketDataError(f"没有本地 {source_timeframe} 数据可供聚合")
        frame = self.repository.active_rows(symbol, source_timeframe, start=start, end=end)
        if frame.empty:
            raise MarketDataError("聚合区间没有源行情")
        if cancelled():
            raise InterruptedError("任务已取消")
        progress(0.25, f"读取 {len(frame)} 行 {source_timeframe} 数据")
        aggregated = self._aggregate_frame(frame, source_timeframe, target_timeframe)
        if aggregated.empty:
            raise MarketDataError("没有可生成的粗周期记录")
        progress(0.65, f"生成 {len(aggregated)} 行 {target_timeframe} 数据")
        normalized, pending = normalize_frame(aggregated, target_timeframe, label_convention="end", source="aggregate")
        pending.append(f"派生自本地 {source_timeframe} 版本 {source['current_version']}")
        fingerprint = hashlib.sha256(
            f"{symbol}|{source_timeframe}|{source['current_version']}|{target_timeframe}|{normalized.timestamp.min()}|{normalized.timestamp.max()}".encode()
        ).hexdigest()
        result = self.service._commit_frame(
            symbol, target_timeframe, normalized, fingerprint=fingerprint,
            file_name=f"aggregate:{source_timeframe}->{target_timeframe}",
            params_hash=hashlib.sha256(f"aggregate-v1|{source_timeframe}|{target_timeframe}".encode()).hexdigest(),
            pending=pending, source="aggregate", adjustment="none", etf_category=None,
            calendar_path=None, actions_path=None, progress=progress, cancelled=cancelled,
            derived={"source_timeframe": source_timeframe, "source_version": source["current_version"],
                     "start_time": str(normalized.timestamp.min()), "end_time": str(normalized.timestamp.max())},
            import_mode="snapshot" if snapshot else "merge",
        )
        return {**result, "source_timeframe": source_timeframe, "target_timeframe": target_timeframe,
                "source_version": source["current_version"],
                "warnings": ["派生数据不会自动刷新；源行情修正后请重新执行聚合"]}

    @staticmethod
    def _validate_pair(source, target):
        if source == target:
            raise MarketDataError("源周期和目标周期不能相同")
        if source == "1min" and target in {"5min", "15min", "30min", "60min"}:
            return
        if source == "1day" and target in {"1week", "1month"}:
            return
        raise MarketDataError(f"不支持从 {source} 聚合为 {target}")

    @staticmethod
    def _aggregate_frame(frame: pd.DataFrame, source, target):
        data = frame.copy().sort_values("timestamp").reset_index(drop=True)
        if target in MINUTE_TIMEFRAMES:
            size = timeframe_spec(target).minutes
            clock = data.timestamp.dt.hour * 60 + data.timestamp.dt.minute
            session = (clock > 690).astype(int)
            index = data.timestamp.dt.strftime("%Y-%m-%d") + ":" + session.astype(str) + ":" + ((clock - session * 210 - 571) // size).astype(str)
        elif target == "1day":
            index = data.timestamp.dt.strftime("%Y-%m-%d")
        elif target == "1week":
            index = data.timestamp.dt.to_period("W-FRI").astype(str)
        elif target == "1month":
            index = data.timestamp.dt.to_period("M").astype(str)
        else:
            raise MarketDataError("不支持的聚合目标")
        grouped = data.groupby(index, sort=True)
        result = grouped.agg(
            timestamp=("timestamp", "max"), open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
            turnover=("turnover", "sum"), source_verified=("source_verified", "all"),
            is_synthetic=("is_synthetic", "any"),
            is_auction_placeholder=("is_auction_placeholder", "any"),
            opening_auction_volume=("opening_auction_volume", "sum"),
            preclose=("preclose", "first"), tradestatus=("tradestatus", "max"),
            is_st=("is_st", "any"), adj_factor=("adj_factor", "last"),
        ).reset_index(drop=True)
        result["source"] = "aggregate"
        return result
