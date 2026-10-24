from __future__ import annotations

import numpy as np
import pandas as pd

from .models import DataBundle
from .price_validation import price_issue_details

QUALITY_RULES_VERSION = 4

MINUTE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover", "source"]


class DataQualityError(ValueError):
    pass


def canonical_frame(frame: pd.DataFrame, daily: bool = False, source: str = "import") -> pd.DataFrame:
    names = {"datetime": "timestamp", "day": "timestamp", "dt": "timestamp", "time": "timestamp",
             "trade_time": "timestamp", "trade_date": "date",
             "vol": "volume", "amount": "turnover", "日期": "date", "时间": "timestamp",
             "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
             "成交量": "volume", "成交额": "turnover", "isST": "is_st"}
    column = "date" if daily else "timestamp"
    frame = frame.copy().rename(columns={k: v for k, v in names.items() if k in frame and v not in frame})
    levels = frame.index.levels if isinstance(frame.index, pd.MultiIndex) else [frame.index]
    if column not in frame and any(pd.api.types.is_datetime64_any_dtype(level.dtype) for level in levels):
        # Tushare-style exports keep the bar clock in the index instead of a
        # named column; promote the levels so the aliases above can match them.
        frame = frame.reset_index()
        if column not in frame and "index" in frame and pd.api.types.is_datetime64_any_dtype(frame["index"]):
            frame = frame.rename(columns={"index": column})
        frame = frame.rename(columns={k: v for k, v in names.items() if k in frame and v not in frame})
    if daily and "date" not in frame and "timestamp" in frame:
        frame = frame.rename(columns={"timestamp": "date"})
    if not daily and "timestamp" not in frame and "date" in frame:
        frame = frame.rename(columns={"date": "timestamp"})
    required = [column, "open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise DataQualityError("行情缺少字段：" + ", ".join(missing))
    frame[column] = pd.to_datetime(frame[column], errors="raise")
    if frame[column].dt.tz is not None:
        frame[column] = frame[column].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if daily:
        frame[column] = frame[column].dt.normalize()
    for name in ["open", "high", "low", "close", "volume", "turnover", "preclose"]:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if "turnover" not in frame:
        frame["turnover"] = np.nan
    if daily:
        if "tradestatus" not in frame:
            # The presence of a daily bar is a trading-state fact. A zero
            # volume bar is a no-trade session, not an inferred suspension.
            frame["tradestatus"] = 1
        frame["tradestatus"] = pd.to_numeric(frame["tradestatus"], errors="coerce").fillna(1).astype(int)
        if "is_st" not in frame:
            frame["is_st"] = False
        frame["is_st"] = frame["is_st"].astype(str).isin(["1", "True", "true"])
        if "preclose" not in frame:
            frame["preclose"] = frame.sort_values(column)["close"].shift(1)
    else:
        frame["source"] = source if "source" not in frame else frame["source"].fillna(source)
    result = frame.sort_values(column, kind="stable").drop_duplicates(column, keep="last").reset_index(drop=True)
    invalid = price_issue_details(result)
    if not invalid.empty:
        first = invalid.iloc[0]
        stamp = first.get("timestamp", first.get("date"))
        raise DataQualityError(
            f"行情包含 {len(invalid)} 条无效价格（非正价格或 OHLC 关系错误）；{stamp}: {first['quality_reason']}"
        )
    return result


def minute_slots(day) -> pd.DatetimeIndex:
    day = pd.Timestamp(day).normalize()
    return pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min").append(
        pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min"))


def align_minutes(frame: pd.DataFrame, convention: str, multiplier: float = 1) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    result["raw_timestamp"] = result.get("raw_timestamp", result["timestamp"])
    result["raw_volume"] = result.get("raw_volume", result["volume"])
    if convention == "start":
        result["timestamp"] = result["timestamp"] + pd.Timedelta(minutes=1)
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    hour, minute = result["timestamp"].dt.hour, result["timestamp"].dt.minute
    clock = hour * 60 + minute
    if convention == "end":
        # Some exports separate the opening auction at 09:30. Include its real
        # prices and volume in the first end-labelled bar rather than losing it.
        auction = result[clock == 570].set_index(result.loc[clock == 570, "timestamp"].dt.normalize())
        first = result.index[clock == 571]
        days = result.loc[first, "timestamp"].dt.normalize()
        matched = days.isin(auction.index)
        first, days = first[matched], days[matched]
        if len(first):
            opening = auction.reindex(days)
            result.loc[first, "open"] = opening.open.to_numpy()
            result.loc[first, "high"] = np.maximum(result.loc[first, "high"].to_numpy(), opening.high.to_numpy())
            result.loc[first, "low"] = np.minimum(result.loc[first, "low"].to_numpy(), opening.low.to_numpy())
            for field in ("volume", "turnover", "raw_volume"):
                if field in result and field in opening:
                    result.loc[first, field] = result.loc[first, field].to_numpy() + opening[field].to_numpy()
            result.loc[first, "opening_auction_volume"] = opening.volume.to_numpy() * multiplier
    # Keep only continuous-session end slots after allocating the auction.
    result = result[((clock >= 571) & (clock <= 690)) | ((clock >= 781) & (clock <= 900))].copy()
    result["volume"] = (result["volume"] * multiplier).round().astype("int64")
    return result.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def infer_profile(frame: pd.DataFrame, daily: pd.DataFrame, source: str, reference: pd.DataFrame | None = None, declared: str | None = None, reference_frequency: str = "1m", price_tick: float = 0.01) -> dict:
    """Calibrate volume using daily totals and labels using independently aligned bars.

    Daily totals alone never establish timestamp semantics. TDX's published code
    explicitly defines its 240 end-labelled slots; imports may declare semantics.
    """
    if frame.empty:
        return {"verified": False, "reason": "没有分钟数据", "source": source}
    daily_lookup = daily.set_index("date") if not daily.empty else pd.DataFrame()
    possibilities = []
    for convention in ([declared] if declared else ["end", "start"]):
        for multiplier in (1, 100):
            aligned = align_minutes(frame, convention, multiplier)
            good = 0
            volume_matches = 0
            for day, part in aligned.groupby(aligned["timestamp"].dt.normalize()):
                if not 235 <= len(part) <= 240 or daily_lookup.empty or day not in daily_lookup.index:
                    continue
                row = daily_lookup.loc[day]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                if row["volume"] <= 0:
                    continue
                vol_error = abs(part.volume.sum() / row.volume - 1)
                if vol_error <= 0.01:
                    volume_matches += 1
                close_error = abs(float(part.iloc[-1].close) - float(row.close))
                high_error = abs(float(part.high.max()) - float(row.high))
                low_error = abs(float(part.low.min()) - float(row.low))
                if max(close_error, high_error, low_error) <= price_tick * 1.1:
                    good += 1
            score = None
            volume_score = None
            overlap = 0
            if reference is not None and not reference.empty:
                comparable = aligned
                if reference_frequency != "1m":
                    size = int(reference_frequency[:-1])
                    pieces = []
                    reference_start = reference.timestamp.min().normalize()
                    reference_end = reference.timestamp.max().normalize() + pd.Timedelta(days=1)
                    sample = aligned[aligned.timestamp.between(reference_start, reference_end, inclusive="left")]
                    for day, group in sample.groupby(sample.timestamp.dt.normalize()):
                        for session in (minute_slots(day)[:120], minute_slots(day)[120:]):
                            observed = group.set_index("timestamp").reindex(session)
                            for offset in range(0, 120, size):
                                bucket = observed.iloc[offset:offset + size]
                                if len(bucket) == size and bucket.close.notna().all():
                                    pieces.append({"timestamp": session[offset + size - 1], "open": bucket.iloc[0].open, "high": bucket.high.max(), "low": bucket.low.min(), "close": bucket.iloc[-1].close, "volume": bucket.volume.sum()})
                    comparable = pd.DataFrame(pieces, columns=["timestamp", "open", "high", "low", "close", "volume"])
                joined = comparable.merge(reference, on="timestamp", suffixes=("", "_ref"))
                overlap = len(joined)
                if overlap:
                    price_ok = np.ones(overlap, dtype=bool)
                    for field in ("open", "high", "low", "close"):
                        price_ok &= (joined[field] - joined[field + "_ref"]).abs().to_numpy() <= price_tick * 1.1
                    volume_ok = (joined.volume - joined.volume_ref).abs() <= np.maximum(1, joined.volume_ref * 0.005)
                    # Labels are established by price paths; volume similarity
                    # is retained as a diagnostic only. Auction allocation may
                    # differ between independent vendors at the five-minute edge.
                    score = float(price_ok.mean())
                    volume_score = float(volume_ok.mean())
            possibilities.append({"convention": convention, "volume_multiplier": multiplier, "verified_days": good, "volume_verified_days": volume_matches, "match_score": score, "reference_volume_score": volume_score, "overlap_bars": overlap})
    possibilities.sort(key=lambda p: (p["match_score"] or 0, p["verified_days"], p["volume_verified_days"]), reverse=True)
    best = possibilities[0]
    credible_labels = bool(declared) or (source == "tdx" and best["convention"] == "end")
    if reference is not None and best["overlap_bars"] >= (120 if reference_frequency == "5m" else 600):
        alternate = next((p for p in possibilities if p["convention"] != best["convention"]), None)
        margin = (best["match_score"] or 0) - ((alternate or {}).get("match_score") or 0)
        # A declared/known timestamp convention is authoritative; independent
        # vendor disagreement is diagnostic and cannot revoke that declaration.
        if not declared and source != "tdx":
            credible_labels = (best["match_score"] or 0) >= 0.9 and margin >= 0.03
    # Daily volume is useful for diagnosing a likely 100x unit mismatch, but
    # cannot establish timestamp labels or veto otherwise sound price bars.
    verified = credible_labels
    return {**best, "source": source, "verified": verified,
            "reason": "时间标签及价格路径核验通过" if verified else "分钟时间标签未校准，按日线处理",
            "declared_convention": declared, "reference_frequency": reference_frequency}


def _max_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _closing_auction_placeholders(symbol: str, day, part: pd.DataFrame, missing_mask: np.ndarray) -> np.ndarray:
    """Identify empty valuation slots inside a verified SZ closing auction.

    SZSE rules (2023 and 2026), 2.3.2 / 3.4.1: continuous matching ends
    at 14:57 and the closing auction matches once at 15:00. Trade-only
    vendors can omit 14:58/14:59. Existing observations are never removed.
    Earlier dates and other markets retain ordinary gap handling.
    """
    placeholders = np.zeros(len(missing_mask), dtype=bool)
    if not symbol.endswith(".SZ") or pd.Timestamp(day) < pd.Timestamp("2023-02-17") or len(part) != 240:
        return placeholders
    anchors = part.iloc[[236, 239]]  # End-labelled 14:57 and 15:00.
    if anchors.close.isna().any() or not anchors.get("source_verified", pd.Series(False, index=anchors.index)).eq(True).all():
        return placeholders
    placeholders[237:239] = missing_mask[237:239]
    return placeholders


def apply_price_limits(daily: pd.DataFrame, calendar: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    if metadata.get("asset_type") == "etf":
        from .rules import round_limit_price, etf_price_limit_rate
        result = daily.copy()
        tick = metadata.get("price_tick", 0.001)
        rates = result.date.map(lambda day: etf_price_limit_rate(metadata, day))
        for column, multiplier in (("limit_up", 1 + rates), ("limit_down", 1 - rates)):
            calculated = (result.preclose * multiplier).map(lambda price: round_limit_price(price, tick) if pd.notna(price) and price > 0 else np.nan)
            result[column] = result[column].fillna(calculated) if column in result else calculated
        result["no_price_limit"] = result.get("no_price_limit", pd.Series(False, index=result.index)).fillna(False).astype(bool)
        result.loc[result.no_price_limit, ["limit_up", "limit_down"]] = np.nan
        return result
    from .rules import calculated_price_limits
    result = daily.copy()
    board = metadata.get("board", "main")
    listing = pd.to_datetime(metadata.get("listing_date"), errors="coerce")
    actual_calendar = pd.DatetimeIndex(pd.to_datetime(calendar.date)).normalize().sort_values().unique()
    listing_indices = {}
    if pd.notna(listing) and listing in actual_calendar:
        listing_indices = {day: index for index, day in enumerate(actual_calendar[actual_calendar >= listing])}
    updates = []
    for row in result.itertuples(index=False):
        day = pd.Timestamp(row.date).normalize()
        age = listing_indices.get(day)
        no_limit = bool(getattr(row, "no_price_limit", False))
        up, down = getattr(row, "limit_up", np.nan), getattr(row, "limit_down", np.nan)
        modern = (board == "star") or (board == "chinext" and pd.notna(listing) and listing >= pd.Timestamp("2020-08-24")) or (board == "main" and pd.notna(listing) and listing >= pd.Timestamp("2023-04-10"))
        if age is not None and age < 5 and modern:
            no_limit, up, down = True, np.nan, np.nan
        elif age == 0:
            if listing < pd.Timestamp("2013-12-13"):
                no_limit, up, down = True, np.nan, np.nan
            elif float(row.preclose) > 0:
                # Exchange preclose is the issue price for an IPO first day.
                from .rules import round_limit_price
                up, down = round_limit_price(float(row.preclose) * 1.44), round_limit_price(float(row.preclose) * 0.64)
        elif not no_limit and (not np.isfinite(up) or not np.isfinite(down)):
            up, down = calculated_price_limits(row.preclose, board, day, bool(row.is_st))
        updates.append((up, down, no_limit))
    if updates:
        result[["limit_up", "limit_down", "no_price_limit"]] = pd.DataFrame(updates, index=result.index)
        result["no_price_limit"] = result.no_price_limit.astype(bool)
    return result


def prepare_bundle(symbol: str, minute: pd.DataFrame, daily: pd.DataFrame, actions: pd.DataFrame, calendar: pd.DataFrame, start, end, frequency: str, version: str, metadata: dict, profiles: dict) -> DataBundle:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if daily.empty:
        raise DataQualityError("缺少不复权日线，请先下载或导入日线数据")
    daily = daily.copy().sort_values("date").drop_duplicates("date", keep="last")
    if metadata.get("listing_date"):
        start = max(start, pd.Timestamp(metadata["listing_date"]))
    if metadata.get("delisting_date") and end >= pd.Timestamp(metadata["delisting_date"]):
        raise DataQualityError("所选区间跨越退市日期，缺少退市结算模型；请缩短回测区间")
    if calendar.empty:
        raise DataQualityError("缺少交易日历，不能将没有行情的工作日静默跳过")
    daily = apply_price_limits(daily, calendar, metadata)
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.date.between(start, end), "date"])).sort_values().unique()
    daily = daily[daily.date.between(start, end)].copy()
    day_lookup = daily.set_index("date")
    missing_days = [d.date().isoformat() for d in days if d not in day_lookup.index]
    if missing_days:
        raise DataQualityError("以下交易日没有日线且无法确认停牌：" + ", ".join(missing_days[:20]))
    if not len(days):
        raise DataQualityError("所选区间没有可回测交易日")
    groups = {day: group.copy() for day, group in minute.groupby(minute.timestamp.dt.normalize())} if not minute.empty else {}
    output, reports, warnings = [], [], []
    for day in days:
        row = day_lookup.loc[day]
        status = int(row.tradestatus)
        if status not in (0, 1):
            raise DataQualityError(f"{day.date()} 的交易状态未知")
        if status == 0:
            reports.append({"date": str(day.date()), "mode": "suspended", "missing_minutes": 0, "missing_data_minutes": 0, "auction_placeholders": 0, "reason": "确认停牌"})
            continue
        if float(row.volume) == 0:
            reports.append({"date": str(day.date()), "mode": "no_trades", "missing_minutes": 0, "missing_data_minutes": 0, "auction_placeholders": 0, "reason": "交易状态正常但全天无成交量"})
            continue
        prices = np.array([row.open, row.high, row.low, row.close], dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any() or not np.isfinite(row.volume) or row.volume < 0 or row.high + 0.001 < max(row.open, row.close) or row.low - 0.001 > min(row.open, row.close) or row.high < row.low:
            raise DataQualityError(f"{day.date()} 日线价格无效")
        part = groups.get(day, pd.DataFrame())
        slots = minute_slots(day)
        reason = ""
        volume_ratio_error = None
        missing = len(slots)
        missing_data = missing
        auction_count = 0
        filled = None
        download_needed = frequency != "1d" and part.empty
        if frequency != "1d" and not part.empty:
            part = part.drop_duplicates("timestamp", keep="last").set_index("timestamp").reindex(slots)
            missing_mask = part.close.isna().to_numpy()
            missing = int(missing_mask.sum())
            auction_mask = _closing_auction_placeholders(symbol, day, part, missing_mask)
            auction_count = int(auction_mask.sum())
            data_missing_mask = missing_mask & ~auction_mask
            missing_data = int(data_missing_mask.sum())
            download_needed = bool(missing_data > 5 or _max_run(data_missing_mask[:120]) > 3 or _max_run(data_missing_mask[120:]) > 3 or missing_mask[0] or missing_mask[120])
            source_verified = part.get("source_verified", pd.Series(False, index=slots)).eq(True)
            observed = part[~part.close.isna()]
            price_issues = price_issue_details(observed)
            if not source_verified.loc[observed.index].all():
                reason = "来源分钟时间标签未校准"
            elif not price_issues.empty:
                reason = "分钟 OHLC/量额校验失败"
            elif missing_data > 5 or _max_run(data_missing_mask[:120]) > 3 or _max_run(data_missing_mask[120:]) > 3:
                reason = "分钟缺口超过连续3根或全日5根阈值"
            elif missing_mask[0] or missing_mask[120]:
                reason = "交易时段首分钟缺失，无法因果补齐"
            elif max(abs(float(observed.high.max()) - row.high), abs(float(observed.low.min()) - row.low), abs(float(observed.iloc[-1].close) - row.close)) > float(metadata.get("price_tick", 0.01)) * 1.1 and missing_data == 0:
                reason = "完整分钟价格聚合与日线不符"
            else:
                if missing_data == 0 and np.isfinite(row.volume) and row.volume > 0:
                    volume_ratio_error = abs(float(observed.volume.sum()) / float(row.volume) - 1)
                previous = part.close.groupby(np.repeat([0, 1], 120)).ffill()
                for price in ("open", "high", "low", "close"):
                    part.loc[missing_mask, price] = previous.loc[missing_mask]
                part.loc[missing_mask, ["volume", "turnover"]] = 0
                part["is_synthetic"] = missing_mask
                part["is_auction_placeholder"] = auction_mask
                part["tradable"] = (~missing_mask) & (part.volume > 0)
                part["frequency"] = "1m"
                part["timestamp"] = slots
                filled = part.reset_index(drop=True)
        elif frequency != "1d":
            reason = "没有分钟数据"
        if filled is None:
            filled = pd.DataFrame([{**row.to_dict(), "timestamp": day + pd.Timedelta(hours=15), "frequency": "1d", "tradable": True, "is_synthetic": False}])
            mode = "daily"
        else:
            mode = "minute"
            if frequency != "1m":
                size = int(frequency[:-1])
                resampled = []
                for offset in (0, 120):
                    session = filled.iloc[offset:offset + 120].copy()
                    # A known non-matching auction slot does not invalidate the
                    # OHLCV of an otherwise observed five-minute closing bar.
                    session["is_synthetic"] &= ~session.is_auction_placeholder
                    bins = np.arange(120) // size
                    grouped = session.groupby(bins).agg(timestamp=("timestamp", "last"), open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), turnover=("turnover", "sum"), is_synthetic=("is_synthetic", "any"), is_auction_placeholder=("is_auction_placeholder", "all"))
                    grouped["frequency"] = frequency
                    grouped["tradable"] = (grouped.volume > 0) & ~grouped.is_synthetic
                    resampled.append(grouped)
                filled = pd.concat(resampled, ignore_index=True)
        for field in ("preclose", "is_st", "limit_up", "limit_down", "no_price_limit"):
            filled[field] = row.get(field, False if field == "is_st" else np.nan)
        output.append(filled)
        report_reason = reason
        if not report_reason:
            if missing_data and mode == "minute":
                report_reason = "少量缺口因果补齐"
            elif auction_count and mode == "minute":
                report_reason = "收盘集合竞价无逐分钟成交，占位保留"
            else:
                report_reason = "完整"
        volume_diagnostic = None
        if volume_ratio_error is not None and volume_ratio_error > 0.01:
            volume_diagnostic = f"分钟与日线成交量比差异 {volume_ratio_error:.1%}；已保留分钟数据"
        reports.append({"date": str(day.date()), "mode": mode, "missing_minutes": missing if frequency != "1d" else 0,
                        "missing_data_minutes": missing_data if frequency != "1d" else 0,
                        "auction_placeholders": auction_count, "download_needed": download_needed,
                        "volume_ratio_error": volume_ratio_error,
                        "volume_diagnostic": volume_diagnostic, "reason": report_reason})
    bars = pd.concat(output, ignore_index=True).sort_values("timestamp").reset_index(drop=True) if output else pd.DataFrame(columns=MINUTE_COLUMNS + ["frequency", "tradable", "is_synthetic"])
    minute_days = sum(report["mode"] == "minute" for report in reports)
    daily_days = sum(report["mode"] == "daily" for report in reports)
    suspended_days = sum(report["mode"] == "suspended" for report in reports)
    no_trade_days = sum(report["mode"] == "no_trades" for report in reports)
    pending_calibration_days = sum(report["mode"] == "daily" and report["reason"] == "来源分钟时间标签未校准" and not report.get("download_needed", False) for report in reports)
    quality = {"pending_calibration_days": pending_calibration_days, "days": reports, "minute_days": minute_days, "daily_days": daily_days, "suspended_days": suspended_days, "no_trade_days": no_trade_days, "total_days": len(days), "minute_ratio": minute_days / max(1, minute_days + daily_days),
               "synthetic_bars": sum(report["missing_minutes"] for report in reports if report["mode"] == "minute"),
               "auction_placeholders": sum(report.get("auction_placeholders", 0) for report in reports if report["mode"] == "minute"),
               "is_demo": bool(metadata.get("is_demo")), "requested_frequency": frequency, "warnings": warnings}
    if pending_calibration_days:
        warnings.append(f"{pending_calibration_days} 个交易日已有本地分钟数据，但时间标签尚未校准；重复下载不能解决此状态。请核对数据源说明，确认开始或结束时间标签后重新导入，或提供独立参考行情核验。")
    volume_mismatch_days = sum(report.get("volume_ratio_error") is not None and report["volume_ratio_error"] > 0.01 for report in reports)
    if volume_mismatch_days:
        warnings.append(f"{volume_mismatch_days} 个交易日分钟与日线成交量比存在偏差；分钟数据已保留，请核对来源单位或成交量口径。")
    if daily_days and frequency != "1d":
        warnings.append(f"{daily_days} 个交易日已整日降级为日线；不还原盘中路径")
    if metadata.get("is_demo"):
        warnings.append("演示数据为确定性生成的模拟行情，不能作为投资或历史数据结论")
    return DataBundle(symbol=symbol, bars=bars, daily=daily, actions=actions.copy(), calendar=days, quality=quality, version=version, metadata=metadata, warnings=warnings)
