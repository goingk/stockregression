from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .models import MarketDataError
from .timeframes import MINUTE_TIMEFRAMES, normalize_timeframe, timeframe_spec


CHUNK_SIZE = 50_000
FILENAME_CODE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
ALIASES = {
    "datetime": "timestamp", "day": "timestamp", "dt": "timestamp", "time": "timestamp",
    "trade_time": "timestamp", "trade_date": "date", "vol": "volume", "amount": "turnover",
    "日期": "date", "时间": "timestamp", "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "turnover", "isST": "is_st",
}
REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


def file_fingerprint(path: Path) -> tuple[str, str | None]:
    digest = hashlib.sha256()
    encoding = None
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    if path.suffix.lower() == ".csv":
        encoding = detect_encoding(path)
    return digest.hexdigest(), encoding


def detect_encoding(path: Path) -> str:
    sample = path.open("rb").read(4096)
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for encoding in ("utf-8", "gb18030"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise MarketDataError(f"无法识别 CSV 编码：{path.name}")


def symbol_from_filename(path: Path) -> str | None:
    matches = sorted(set(FILENAME_CODE.findall(path.stem)))
    return matches[0] if len(matches) == 1 else None


def filename_candidates(path: Path) -> list[str]:
    return sorted(set(FILENAME_CODE.findall(path.stem)))


def _promote_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "timestamp" not in result and "date" in result:
        result["timestamp"] = result.date
    if "timestamp" not in result and isinstance(result.index, pd.MultiIndex):
        result = result.reset_index()
    if "timestamp" not in result and isinstance(result.index, pd.DatetimeIndex):
        result = result.reset_index()
    result = result.rename(columns={key: value for key, value in ALIASES.items() if key in result and value not in result})
    if "timestamp" not in result and "index" in result and pd.api.types.is_datetime64_any_dtype(result["index"]):
        result = result.rename(columns={"index": "timestamp"})
    return result


def normalize_frame(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    label_convention: str = "auto",
    source: str = "import",
    skip_invalid_rows: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    timeframe = normalize_timeframe(timeframe)
    result = _promote_time_columns(frame)
    missing = [field for field in REQUIRED if field not in result]
    if missing:
        raise MarketDataError("行情缺少字段：" + ", ".join(missing))
    result["timestamp"] = pd.to_datetime(result.timestamp, errors="raise")
    if result.timestamp.dt.tz is not None:
        result["timestamp"] = result.timestamp.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    for field in ("open", "high", "low", "close", "volume", "turnover", "preclose"):
        if field in result:
            result[field] = pd.to_numeric(result[field], errors="coerce")
    if "turnover" not in result:
        result["turnover"] = np.nan
    if "source" not in result:
        result["source"] = source
    else:
        result["source"] = result.source.fillna(source).astype(str)
    result["is_st"] = result.get("is_st", pd.Series(np.nan, index=result.index)).map(
        lambda value: pd.NA if pd.isna(value) else _bool(value)
    )
    result["tradestatus"] = pd.to_numeric(result.get("tradestatus", pd.Series(np.nan, index=result.index)), errors="coerce")
    result["is_synthetic"] = False
    result["is_auction_placeholder"] = False

    pending: list[str] = []
    spec = timeframe_spec(timeframe)
    if spec.kind == "minute":
        invalid = (result[["open", "high", "low", "close"]] <= 0).any(axis=1) | result[["open", "high", "low", "close"]].isna().any(axis=1)
        if invalid.any():
            if not skip_invalid_rows:
                raise MarketDataError(f"分钟行情包含 {int(invalid.sum())} 条非正或空价格，开盘/最高/最低/收盘必须为正")
            pending.append(f"已跳过 {int(invalid.sum())} 条非正或空价格源记录")
            result = result.loc[~invalid].copy()
            if result.empty:
                raise MarketDataError("文件仅包含无效价格记录")
    if spec.kind == "minute":
        result["timestamp"] = result.timestamp.dt.floor("min")
        if (result.timestamp.dt.second != 0).any():
            raise MarketDataError("分钟行情时间必须对齐到整分钟")
        if label_convention not in {"auto", "start", "end"}:
            raise MarketDataError("时间标签应为 auto/start/end")
        if label_convention == "start":
            result["timestamp"] = result.timestamp + pd.Timedelta(minutes=spec.minutes or 1)
        result["source_verified"] = label_convention == "end"
        if label_convention == "auto":
            pending.append("分钟时间标签未确认；请选择开始或结束标签后重新校验")
        if label_convention == "end":
            result = _fold_opening_auction(result, minutes=spec.minutes or 1)
        if label_convention != "auto" and timeframe == "1min":
            clock = result.timestamp.dt.hour * 60 + result.timestamp.dt.minute
            outside = ~(((clock >= 571) & (clock <= 690)) | ((clock >= 781) & (clock <= 900)))
            if outside.any():
                sample = result.loc[outside, "timestamp"].head(5).astype(str).tolist()
                raise MarketDataError("1 分钟时间超出连续交易时段：" + ", ".join(sample))
    else:
        result["timestamp"] = result.timestamp.dt.normalize()
        result["source_verified"] = True
        if spec.kind in {"week", "month"}:
            period = result.timestamp.dt.to_period("W-FRI" if spec.kind == "week" else "M")
            repeated = result.assign(_period=period).groupby("_period").timestamp.nunique()
            if repeated.gt(1).any():
                labels = [str(value) for value in repeated[repeated.gt(1)].index]
                raise MarketDataError(f"文件在同一{'周' if spec.kind == 'week' else '月'}内包含多个日期：" + ", ".join(labels[:8]))
    if frame_is_daily_only(result) and spec.kind == "minute" and result.timestamp.dt.normalize().nunique() > 1:
        raise MarketDataError("文件的日期没有分钟时间，与所选分钟周期不一致")

    duplicate = result[result.duplicated("timestamp", keep=False)].sort_values("timestamp")
    if not duplicate.empty:
        conflicting = []
        for stamp, group in duplicate.groupby("timestamp"):
            keys = group[["open", "high", "low", "close", "volume"]].round(10).drop_duplicates()
            if len(keys) > 1:
                conflicting.append(str(stamp))
        if conflicting:
            raise MarketDataError("文件内同一时间存在不同的 OHLCV 记录：" + ", ".join(conflicting[:8]))
    result = result.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if result.empty:
        raise MarketDataError("文件没有可导入的行情记录")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise MarketDataError("开盘、最高、最低、收盘必须为正数")
    if (result.volume < 0).any():
        raise MarketDataError("成交量不能为负数")
    if result.turnover.notna().any() and (result.turnover.fillna(0) < 0).any():
        raise MarketDataError("成交额不能为负数")
    if ((result.low > result.high + 1e-8) | (result.open > result.high + 1e-8) | (result.open < result.low - 1e-8) |
            (result.close > result.high + 1e-8) | (result.close < result.low - 1e-8)).any():
        raise MarketDataError("OHLC 高低价关系不合法")
    if result[list(REQUIRED[1:])].isna().any().any():
        raise MarketDataError("OHLCV 含空值")
    return result, pending


def _fold_opening_auction(frame: pd.DataFrame, minutes: int = 1) -> pd.DataFrame:
    """Merge an end-labelled 09:30 auction bar into the first regular bar."""
    result = frame.copy().sort_values("timestamp", kind="stable")
    clock = result.timestamp.dt.hour * 60 + result.timestamp.dt.minute
    auction_mask = clock.eq(570)
    if not auction_mask.any():
        return result
    auction = result.loc[auction_mask].copy()
    auction["day"] = auction.timestamp.dt.normalize()
    auction = auction.set_index("day")
    first_mask = clock.eq(570 + max(1, int(minutes)))
    first_index = result.index[first_mask]
    first_days = pd.DatetimeIndex(result.loc[first_mask, "timestamp"].dt.normalize())
    matched = first_days.isin(auction.index)
    matched_index = first_index[matched]
    matched_auction = auction.reindex(first_days[matched])
    output = result.copy()
    if len(matched_index):
        output.loc[matched_index, "open"] = matched_auction.open.to_numpy()
        output.loc[matched_index, "high"] = np.maximum(
            output.loc[matched_index, "high"].to_numpy(dtype=float), matched_auction.high.to_numpy(dtype=float)
        )
        output.loc[matched_index, "low"] = np.minimum(
            output.loc[matched_index, "low"].to_numpy(dtype=float), matched_auction.low.to_numpy(dtype=float)
        )
        output.loc[matched_index, "volume"] = (
            output.loc[matched_index, "volume"].to_numpy(dtype=float) + matched_auction.volume.to_numpy(dtype=float)
        )
        output.loc[matched_index, "turnover"] = (
            output.loc[matched_index, "turnover"].fillna(0).to_numpy(dtype=float) + matched_auction.turnover.fillna(0).to_numpy(dtype=float)
        )
        output.loc[matched_index, "opening_auction_volume"] = matched_auction.volume.to_numpy(dtype=float)
    output = output.loc[~auction_mask].copy()
    return output.sort_values("timestamp", kind="stable").reset_index(drop=True)


def frame_is_daily_only(frame: pd.DataFrame) -> bool:
    return bool(len(frame)) and bool((frame.timestamp.dt.hour == 0).all() and (frame.timestamp.dt.minute == 0).all())


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "是"}
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return False


def check_declared_identity(frame: pd.DataFrame, symbol: str) -> None:
    for column in ("ts_code", "symbol", "code", "证券代码"):
        if column not in frame:
            continue
        values = set(frame[column].dropna().astype(str).str.strip())
        values = {value for value in values if value and value.lower() not in {"nan", "none"}}
        if not values:
            continue
        from ..models import normalize_symbol
        try:
            normalized = {normalize_symbol(value) for value in values}
        except ValueError as exc:
            raise MarketDataError(f"文件证券代码无法解析：{', '.join(sorted(values))}") from exc
        if normalized != {symbol}:
            raise MarketDataError(
                f"目标证券代码 {symbol} 与文件 {', '.join(sorted(normalized))} 不一致"
            )


def check_declared_timeframe(frame: pd.DataFrame, timeframe: str) -> None:
    for column in ("frequency", "周期", "interval"):
        if column not in frame:
            continue
        aliases = {str(value).strip().lower() for value in frame[column].dropna().unique()}
        if not aliases:
            continue
        accepted = {timeframe.lower(), timeframe.replace("min", "m").lower(), timeframe.replace("1day", "1d").lower()}
        if aliases.isdisjoint(accepted):
            raise MarketDataError(f"文件声明的周期 {', '.join(sorted(aliases))} 与所选 {timeframe} 不一致")


def check_declared_adjustment(frame: pd.DataFrame, adjustment: str) -> None:
    for column in ("adjustflag", "adjust", "复权口径"):
        if column not in frame:
            continue
        values = {str(value).strip().lower() for value in frame[column].dropna().unique()}
        if not values:
            continue
        raw = {"none", "raw", "unadjusted", "不复权", "3", ""}
        if adjustment == "none" and not values.issubset(raw):
            raise MarketDataError("文件声明的复权口径与本次不复权导入不一致")


def read_file(path: str | Path) -> tuple[pd.DataFrame, str, str | None, int]:
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise MarketDataError("导入文件不存在：" + str(file))
    if file.suffix.lower() not in {".csv", ".parquet"}:
        raise MarketDataError("仅支持 CSV 和 Parquet 行情文件：" + file.name)
    fingerprint, encoding = file_fingerprint(file)
    try:
        if file.suffix.lower() == ".parquet":
            frame = pd.read_parquet(file)
            batches = 1
        else:
            parts = []
            reader = pd.read_csv(file, encoding=encoding or "utf-8", chunksize=CHUNK_SIZE, low_memory=False)
            batches = 0
            for part in reader:
                parts.append(part)
                batches += 1
            frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    except Exception as exc:
        raise MarketDataError(f"读取 {file.name} 失败：{exc}") from exc
    if frame.empty:
        raise MarketDataError("行情文件为空：" + file.name)
    return frame, fingerprint, encoding, batches


def scan_folder(folder: str | Path, timeframe: str, *, recursive: bool = False) -> list[dict]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise MarketDataError("导入文件夹不存在：" + str(root))
    iterator = root.rglob("*") if recursive else root.glob("*")
    items = []
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        candidates = filename_candidates(path)
        error = None
        symbol = None
        if path.stat().st_size == 0:
            error = "文件为空"
        if len(candidates) == 1:
            from ..models import normalize_symbol
            try:
                symbol = normalize_symbol(candidates[0])
            except ValueError as exc:
                error = str(exc)
        elif not candidates:
            error = "文件名中没有六位证券代码"
        else:
            error = "文件名包含多个不同的六位代码：" + ", ".join(candidates)
        items.append({
            "path": str(path), "relative_path": path.relative_to(root).as_posix(),
            "symbol": symbol, "timeframe": normalize_timeframe(timeframe),
            "size": path.stat().st_size, "error": error,
        })
    return items
