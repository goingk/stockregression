"""Numerical ETF suitability statistics and grid parameter suggestions.

The functions in this module operate only on an already loaded
``DataBundle``. They never read market files, call a provider, or fill future
rows. Minute statistics use unadjusted observed closes after applying the
shared :mod:`stockregression.price_basis` factors in the decision-day basis.
Rows marked synthetic are missing-data fill lines and are excluded. A verified
closing-auction placeholder is retained as a no-trade valuation point, as are
ordinary zero-volume minutes.

The variance ratios are descriptive statistics. No testing p-value is
produced, and overlapping observations are reported as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import copy
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import rules
from . import price_basis
from .models import DataBundle
from .etf_models import MIN_COMPLETE_DAY_RATIO
from .market_state import METRIC_KEYS as MARKET_STATE_METRIC_KEYS, daily_market_state


ANNUAL_TRADING_DAYS = 252.0
MINUTES_PER_DAY = 240
PROFILE_BINS = 200

METRIC_KEYS = (
    "rv_annualized",
    "path_redundancy_daily_mean",
    "path_redundancy_5d_mean",
    "path_redundancy_20d_mean",
    "path_redundancy_daily_mean_1m",
    "path_redundancy_5d_mean_1m",
    "path_redundancy_20d_mean_1m",
    "iovr",
    "variance_ratio_60m",
    "variance_ratio_240m",
    "variance_ratio_1200m",
    "ou_half_life_days",
    "zero_crossing_rate",
    "jump_ratio",
    "max_drawdown_duration_days",
    "tick_to_price_ratio",
    "rv_5m_annualized",
    "mean_raw_price",
)

GRID_BASIS_KEYS = (
    "reference_price",
    "lower_price",
    "upper_price",
    "current_raw_price",
    "current_price_date",
    "current_price_source",
    "current_price_basis",
    "atr30_pct",
    "svr",
    "realized_skewness",
    "realized_kurtosis",
    "price_tick",
)

_SLOT_SECONDS = np.array(
    [hour * 3600 + minute * 60 for hour, minute in (
        [(9, minute) for minute in range(31, 60)]
        + [(10, minute) for minute in range(60)]
        + [(11, minute) for minute in range(31)]
        + [(13, minute) for minute in range(1, 60)]
        + [(14, minute) for minute in range(60)]
        + [(15, 0)]
    )],
    dtype=np.int64,
)


class _StatsError(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class _Context:
    as_of: pd.Timestamp
    requested_window_days: int
    calendar_days: pd.DatetimeIndex
    days: pd.DatetimeIndex
    coverage_complete: bool
    raw_bars: pd.DataFrame
    raw_daily: pd.DataFrame
    adjusted_bars: pd.DataFrame
    adjusted_daily: pd.DataFrame
    profile_bars: pd.DataFrame
    factors: list[dict[str, Any]]
    warnings: list[str]
    minute_kind: str | None
    points_1m: pd.DataFrame
    points_5m: pd.DataFrame
    points_30m: pd.DataFrame
    sampling: dict | None = None
    data_resolution: dict | None = None
    minute_integrity: dict | None = None
    mddd_history_verified: bool = False
    mddd_unavailable_reason: str | None = None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _bool_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是"}
    return bool(value)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_bool_value).astype(bool)


def _verified(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _timestamp(value: Any) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _StatsError("unavailable", "as_of 不是有效时间") from exc
    if stamp is pd.NaT:
        raise _StatsError("unavailable", "as_of 不是有效时间")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return stamp


def _normalise_bars(data: DataBundle, as_of: pd.Timestamp) -> pd.DataFrame:
    frame = data.bars if isinstance(data.bars, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame()
    if "timestamp" not in frame:
        raise _StatsError("data_not_ready", "分钟行情缺少 timestamp 字段")
    bars = frame.copy()
    stamps = pd.to_datetime(bars["timestamp"], errors="coerce")
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    bars["timestamp"] = stamps
    bars = bars[bars.timestamp.notna()].copy()
    if as_of == as_of.normalize():
        bars = bars[bars.timestamp < as_of.normalize() + pd.Timedelta(days=1)].copy()
    else:
        bars = bars[bars.timestamp <= as_of].copy()
    for column in ("open", "high", "low", "close", "preclose", "volume", "turnover", "amount"):
        if column in bars:
            bars[column] = pd.to_numeric(bars[column], errors="coerce")
    if "frequency" not in bars:
        bars["frequency"] = "1m"
    bars["frequency"] = bars.frequency.fillna("").astype(str).str.strip().str.lower()
    return bars.sort_values("timestamp", kind="stable").drop_duplicates(
        "timestamp", keep="last",
    ).reset_index(drop=True)


def _normalise_daily(data: DataBundle, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.daily if isinstance(data.daily, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "date" not in frame:
        raise _StatsError("data_not_ready", "日线行情缺少 date 字段")
    daily = frame.copy()
    dates = pd.to_datetime(daily["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    daily["date"] = dates.dt.normalize()
    daily = daily[daily.date.notna() & (daily.date <= as_of.normalize())].copy()
    for column in ("open", "high", "low", "close", "preclose", "volume", "turnover", "amount"):
        if column in daily:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily = daily.sort_values("date", kind="stable").drop_duplicates(
        "date", keep="last",
    )

    metadata = data.metadata if isinstance(data.metadata, dict) else {}
    prior_closes = metadata.get("prior_closes") if isinstance(metadata.get("prior_closes"), Mapping) else {}
    effective = daily.preclose.copy() if "preclose" in daily else pd.Series(np.nan, index=daily.index)
    previous_close = daily.get("close", pd.Series(np.nan, index=daily.index)).shift(1)
    for index, row in daily.iterrows():
        if _positive(effective.loc[index]) is not None:
            continue
        day = pd.Timestamp(row.date).date().isoformat()
        candidates = (
            prior_closes.get(day), prior_closes.get(str(day)),
            prior_closes.get(pd.Timestamp(row.date).normalize()),
            metadata.get("prior_close"),
        )
        replacement = next((_positive(value) for value in candidates if _positive(value) is not None), None)
        if replacement is None:
            replacement = _positive(previous_close.loc[index])
        if replacement is not None:
            effective.loc[index] = replacement
    daily["preclose"] = effective
    columns = [name for name in ("close", "preclose", "high", "low") if name in daily]
    prices = daily.set_index("date")[columns].copy()
    return daily.reset_index(drop=True), prices


def _calendar_days(data: DataBundle, daily: pd.DataFrame, as_of: pd.Timestamp) -> pd.DatetimeIndex:
    if isinstance(data.calendar, pd.DataFrame):
        values = data.calendar["date"] if "date" in data.calendar else data.calendar.index
    elif isinstance(data.calendar, pd.DatetimeIndex):
        values = data.calendar
    else:
        try:
            values = list(data.calendar or [])
        except TypeError:
            values = []
    stamps = pd.to_datetime(list(values), errors="coerce")
    days = pd.DatetimeIndex(stamps)
    if days.tz is not None:
        days = days.tz_convert("Asia/Shanghai").tz_localize(None)
    days = days.dropna().normalize().unique().sort_values()
    days = days[days <= as_of.normalize()]
    if not len(days) and not daily.empty:
        days = pd.DatetimeIndex(daily.date).dropna().normalize().unique().sort_values()
    return pd.DatetimeIndex(days, name="date")


def _slot_table(days: pd.DatetimeIndex) -> pd.DataFrame:
    if not len(days):
        return pd.DataFrame(columns=["timestamp", "day", "session", "slot"])
    parts = []
    for day in days:
        parts.append(pd.DataFrame({
            "timestamp": pd.DatetimeIndex(day + pd.to_timedelta(_SLOT_SECONDS, unit="s")),
            "day": day,
            "session": np.repeat([0, 1], 120),
            "slot": np.tile(np.arange(120), 2),
        }))
    return pd.concat(parts, ignore_index=True)


def _bar_slots(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    values = bars.copy()
    seconds = (
        values.timestamp.dt.hour * 3600
        + values.timestamp.dt.minute * 60
        + values.timestamp.dt.second
    ).astype("int64")
    lookup = {int(second): position for position, second in enumerate(_SLOT_SECONDS)}
    values["slot_position"] = seconds.map(lookup)
    values = values[values.slot_position.notna()].copy()
    values["slot_position"] = values.slot_position.astype(int)
    values["day"] = values.timestamp.dt.normalize()
    values["session"] = (values.slot_position >= 120).astype(int)
    values["slot"] = np.where(values.session.eq(0), values.slot_position, values.slot_position - 120)
    return values


def _valid_bar_mask(frame: pd.DataFrame, *, require_ohlc: bool = False) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    columns = ("open", "high", "low", "close") if require_ohlc else ("close",)
    valid = pd.Series(True, index=frame.index)
    for column in columns:
        values = pd.to_numeric(
            frame[column] if column in frame else pd.Series(np.nan, index=frame.index),
            errors="coerce",
        )
        valid &= np.isfinite(values.to_numpy(dtype=float)) & values.gt(0).to_numpy()
    synthetic = _bool_series(frame, "is_synthetic")
    placeholder = _bool_series(frame, "is_auction_placeholder")
    return valid & (~synthetic | placeholder)


def _expected_minutes(
    bars: pd.DataFrame, days: pd.DatetimeIndex, raw_bars: pd.DataFrame | None = None,
) -> pd.DataFrame:
    expected = _slot_table(days)
    if expected.empty:
        return expected
    actual = _bar_slots(bars)
    actual = actual[actual.day.isin(days)].copy()
    if actual.empty:
        for column, default in (
            ("open", np.nan), ("high", np.nan), ("low", np.nan), ("close_adj", np.nan),
            ("raw_close", np.nan), ("volume", 0.0), ("turnover", np.nan),
            ("is_synthetic", False), ("is_auction_placeholder", False),
        ):
            expected[column] = default
        expected["valid"] = False
        expected["valid_ohlc"] = False
        return expected
    actual["valid"] = _valid_bar_mask(actual)
    actual["valid_ohlc"] = _valid_bar_mask(actual, require_ohlc=True)
    actual = actual.rename(columns={"close": "close_adj"})
    keep = [
        "timestamp", "open", "high", "low", "close_adj", "volume", "turnover",
        "is_synthetic", "is_auction_placeholder", "source_verified", "valid", "valid_ohlc",
    ]
    actual = actual[[column for column in keep if column in actual]]
    merged = expected.merge(actual, on="timestamp", how="left", sort=False)
    merged["valid"] = merged.valid.eq(True)
    merged["valid_ohlc"] = merged.valid_ohlc.eq(True)
    raw_source = bars if raw_bars is None else raw_bars
    raw_source = raw_source[raw_source.frequency.eq("1m")][["timestamp", "close"]].copy()
    raw_source = raw_source.rename(columns={"close": "raw_close"}).drop_duplicates(
        "timestamp", keep="last",
    )
    merged = merged.merge(raw_source, on="timestamp", how="left", sort=False)
    merged["day"] = pd.to_datetime(merged.day).dt.normalize()
    return merged.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _expected_five_minutes(points_1m: pd.DataFrame) -> pd.DataFrame:
    if points_1m.empty:
        return pd.DataFrame()
    source = points_1m.copy()
    source["bucket"] = source.slot // 5
    rows = []
    for (day, session, bucket), part in source.groupby(
        ["day", "session", "bucket"], sort=True, observed=True,
    ):
        complete = len(part) == 5
        close_valid = complete and bool(part.valid.all())
        ohlc_valid = complete and bool(part.valid_ohlc.all())
        rows.append({
            "bucket": int(bucket), "timestamp": part.timestamp.iloc[-1],
            "day": pd.Timestamp(day).normalize(), "session": int(session), "slot": int(part.slot.iloc[-1]),
            "valid": close_valid, "valid_ohlc": ohlc_valid,
            "close": float(part.close_adj.iloc[-1]) if close_valid else np.nan,
            "raw_close": float(part.raw_close.iloc[-1]) if close_valid else np.nan,
            "open": float(part.open.iloc[0]) if ohlc_valid else np.nan,
            "high": float(part.high.max()) if ohlc_valid else np.nan,
            "low": float(part.low.min()) if ohlc_valid else np.nan,
            "volume": float(part.volume.fillna(0).sum()),
            "turnover": float(part.turnover.fillna(0).sum()) if "turnover" in part else np.nan,
        })
    return pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)


def _direct_five_minutes(bars: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    expected = _slot_table(days)
    if expected.empty:
        return pd.DataFrame()
    expected["bucket"] = expected.slot // 5
    expected = expected[expected.slot.mod(5).eq(4)].copy()
    actual = _bar_slots(bars[bars.frequency.eq("5m")])
    actual["bucket"] = actual.slot // 5
    actual = actual[actual.slot.mod(5).eq(4)].copy()
    actual["valid"] = _valid_bar_mask(actual)
    actual["valid_ohlc"] = _valid_bar_mask(actual, require_ohlc=True)
    actual = actual.rename(columns={"close": "close"})
    keep = [
        "timestamp", "valid", "valid_ohlc", "close", "open", "high", "low", "volume", "turnover",
    ]
    actual = actual[[column for column in keep if column in actual]]
    merged = expected[["timestamp", "day", "session", "bucket"]].merge(
        actual, on="timestamp", how="left", sort=False,
    )
    merged["valid"] = merged.valid.eq(True)
    merged["valid_ohlc"] = merged.valid_ohlc.eq(True)
    return merged.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _expected_thirty_minutes(points_1m: pd.DataFrame) -> pd.DataFrame:
    if points_1m.empty:
        return pd.DataFrame()
    source = points_1m.copy()
    source["bucket"] = source.slot // 30
    rows = []
    for (day, session, bucket), part in source.groupby(
        ["day", "session", "bucket"], sort=True, observed=True,
    ):
        complete = len(part) == 30
        close_valid = complete and bool(part.valid.all())
        ohlc_valid = complete and bool(part.valid_ohlc.all())
        rows.append({
            "bucket": int(bucket), "timestamp": part.timestamp.iloc[-1],
            "day": pd.Timestamp(day).normalize(), "session": int(session), "slot": int(part.slot.iloc[-1]),
            "valid": close_valid, "valid_ohlc": ohlc_valid,
            "close": float(part.close_adj.iloc[-1]) if close_valid else np.nan,
            "open": float(part.open.iloc[0]) if ohlc_valid else np.nan,
            "high": float(part.high.max()) if ohlc_valid else np.nan,
            "low": float(part.low.min()) if ohlc_valid else np.nan,
            "volume": float(part.volume.fillna(0).sum()),
        })
    return pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)


def _clock_link(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    day = frame.day.reset_index(drop=True)
    session = frame.session.reset_index(drop=True)
    slot = frame.slot.reset_index(drop=True)
    within = day.eq(day.shift()) & session.eq(session.shift())
    differences = slot - slot.shift()
    positive = differences[within & differences.gt(0)]
    step = int(positive.min()) if not positive.empty else 1
    same_session = within & differences.eq(step)
    first_slot = step - 1
    lunch = day.eq(day.shift()) & session.eq(1) & session.shift().eq(0) & slot.eq(first_slot) & slot.shift().eq(119)
    day_codes = pd.Series(pd.factorize(day)[0])
    overnight = (
        day_codes.eq(day_codes.shift() + 1) & session.eq(0) & slot.eq(first_slot)
        & session.shift().eq(1) & slot.shift().eq(119)
    )
    return (same_session | lunch | overnight).fillna(False).reset_index(drop=True)


def _within_session_returns(frame: pd.DataFrame) -> pd.DataFrame:
    close_column = "close" if "close" in frame else "close_adj" if "close_adj" in frame else None
    if frame.empty or close_column is None or "valid" not in frame:
        return pd.DataFrame(columns=["day", "session", "start_slot", "end_slot", "r"])
    close = pd.to_numeric(frame[close_column], errors="coerce")
    previous = close.shift(1)
    within = frame.day.eq(frame.day.shift()) & frame.session.eq(frame.session.shift())
    differences = frame.slot - frame.slot.shift()
    positive = differences[within & differences.gt(0)]
    step = int(positive.min()) if not positive.empty else 1
    same = within & differences.eq(step)
    valid = frame.valid.fillna(False).astype(bool) & frame.valid.shift(1, fill_value=False).astype(bool) & same
    values = np.log(close / previous)
    session_start = frame.slot.eq(step - 1)
    first_values = np.log(
        close / pd.to_numeric(frame.get("open", pd.Series(np.nan, index=frame.index)), errors="coerce")
    )
    values = values.mask(session_start, first_values)
    start_slots = frame.slot.where(session_start, frame.slot - step)
    mask = (
        frame.valid.fillna(False).astype(bool)
        & (valid | session_start)
        & np.isfinite(values)
    )
    return pd.DataFrame({
        "day": frame.day[mask].to_numpy(),
        "session": frame.session[mask].astype(int).to_numpy(),
        "start_slot": start_slots[mask].astype(int).to_numpy(),
        "end_slot": frame.slot[mask].astype(int).to_numpy(),
        "r": values[mask].astype(float).to_numpy(),
    })


def _daily_rv(returns: pd.DataFrame) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    grouped = returns.assign(r2=returns.r.astype(float) ** 2).groupby(
        "day", sort=True,
    ).agg(rv=("r2", "sum"), n=("r2", "size"))
    return grouped.loc[grouped.n.gt(0), "rv"].astype(float)


def _annualized(series: pd.Series) -> tuple[float | None, int, str | None]:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values) & values.ge(0)]
    if values.empty:
        return None, 0, "没有可用的日内平方收益样本"
    value = math.sqrt(ANNUAL_TRADING_DAYS * float(values.mean()))
    return _finite(value), int(len(values)), None


def _path_redundancy(points: pd.DataFrame) -> tuple[float | None, int, str | None]:
    if points.empty:
        return None, 0, "没有分钟路径"
    valid = points.valid.fillna(False).astype(bool)
    close = pd.to_numeric(points.close_adj, errors="coerce")
    link = _clock_link(points) & valid & valid.shift(1, fill_value=False)
    differences = close.diff().abs()
    usable = link & np.isfinite(differences)
    length = float(differences[usable].sum())
    positions = np.flatnonzero(valid.to_numpy())
    if not len(positions) or length <= 1e-15:
        return None, int(usable.sum()), "全路径长度为零或没有相邻有效价格"
    # Missing sessions break the path. Each continuous segment contributes its
    # own net displacement; never connect prices across an omitted trading day.
    starts = valid & ~usable
    ends = valid & ~usable.shift(-1, fill_value=False)
    net = float(np.abs(close[ends].to_numpy() - close[starts].to_numpy()).sum())
    return _finite(max(0.0, min(1.0, 1.0 - net / length))), int(usable.sum()), None


def _path_redundancy_means(
    points: pd.DataFrame, days: pd.DatetimeIndex, *, sampling_minutes: int = 5,
) -> dict[str, tuple[float | None, int, str | None]]:
    """Equal-weight means of independently computed 1/5/20-day path ratios.

    The caller supplies the full selected trading calendar, including excluded
    days. A missing day invalidates every window containing it. Each daily path
    starts at its first sampled bar's open and visits the sampled closes; a
    multi-day path also includes the true close-to-next-open displacement.
    Flat days remain valid inside longer windows, but zero-length windows have
    no defined ratio and do not contribute to the mean.
    """
    windows = (
        ("path_redundancy_daily_mean", 1),
        ("path_redundancy_5d_mean", 5),
        ("path_redundancy_20d_mean", 20),
    )
    calendar = pd.DatetimeIndex(days).normalize().unique().sort_values()
    close_column = "close" if "close" in points else "close_adj"
    required = {"day", "session", "slot", "valid", "open", close_column}
    if sampling_minutes not in (1, 5):
        raise ValueError("路径冗余仅支持1分钟或5分钟采样")
    if points.empty or not required.issubset(points.columns) or not len(calendar):
        return {key: (None, 0, "没有完整的采样价格路径") for key, _ in windows}

    frame = points.copy()
    frame["day"] = pd.to_datetime(frame.day, errors="coerce").dt.normalize()
    grouped = {day: part for day, part in frame.groupby("day", sort=False)}
    expected_slots = np.tile(np.arange(sampling_minutes - 1, 120, sampling_minutes), 2)
    expected_sessions = np.repeat([0, 1], 120 // sampling_minutes)
    opening = np.full(len(calendar), np.nan)
    closing = np.full(len(calendar), np.nan)
    lengths = np.full(len(calendar), np.nan)
    complete = np.zeros(len(calendar), dtype=bool)
    for index, day in enumerate(calendar):
        part = grouped.get(day)
        if part is None or len(part) != len(expected_slots):
            continue
        part = part.sort_values(["session", "slot"], kind="stable")
        if (
            not np.array_equal(part.slot.to_numpy(), expected_slots)
            or not np.array_equal(part.session.to_numpy(), expected_sessions)
            or not bool(part.valid.fillna(False).astype(bool).all())
        ):
            continue
        prices = pd.to_numeric(part[close_column], errors="coerce").to_numpy(dtype=float)
        first_open = _positive(part.open.iloc[0])
        if first_open is None or not np.all(np.isfinite(prices) & (prices > 0)):
            continue
        opening[index], closing[index] = first_open, prices[-1]
        lengths[index] = abs(prices[0] - first_open) + float(np.abs(np.diff(prices)).sum())
        complete[index] = True

    result = {}
    for key, width in windows:
        ratios = []
        complete_windows = 0
        for start in range(len(calendar) - width + 1):
            stop = start + width
            if not complete[start:stop].all():
                continue
            complete_windows += 1
            length = float(lengths[start:stop].sum())
            if width > 1:
                length += float(np.abs(opening[start + 1:stop] - closing[start:stop - 1]).sum())
            if length <= 1e-15:
                continue
            net = abs(closing[stop - 1] - opening[start])
            ratios.append(max(0.0, min(1.0, 1.0 - net / length)))
        if ratios:
            result[key] = (_finite(float(np.mean(ratios))), len(ratios), None)
        elif complete_windows:
            result[key] = (None, 0, f"连续{width}日窗口的有效总路径均为零")
        else:
            result[key] = (None, 0, f"没有连续完整的{width}个交易日窗口")
    return result


def _official_iovr(
    daily: pd.DataFrame, returns_1m: pd.DataFrame, days: pd.DatetimeIndex,
) -> tuple[float | None, int, str | None]:
    intraday = _daily_rv(returns_1m)
    by_day = daily.drop_duplicates("date", keep="last").set_index("date") if not daily.empty else pd.DataFrame()
    intraday_values: list[float] = []
    overnight_values: list[float] = []
    for day in days:
        if day not in intraday.index or day not in by_day.index:
            continue
        row = by_day.loc[day]
        open_price = _positive(row.get("open"))
        preclose = _positive(row.get("preclose"))
        rv = _finite(intraday.loc[day])
        if open_price is None or preclose is None or rv is None or rv < 0:
            continue
        intraday_values.append(rv)
        overnight_values.append(math.log(open_price / preclose) ** 2)
    if not intraday_values:
        return None, 0, "没有同时具备有效日内RV和官方开盘/前收盘的交易日"
    denominator = float(np.mean(overnight_values))
    if denominator <= 1e-18:
        return None, len(intraday_values), "隔夜平方收益均值为零，IOVR分母无效"
    return _finite(float(np.mean(intraday_values)) / denominator), len(intraday_values), None


def _sample_variance(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    value = float(np.var(values, ddof=1))
    return value if math.isfinite(value) else None


def _variance_ratio(points: pd.DataFrame, q: int) -> tuple[float | None, int, str | None]:
    if points.empty or q < 2:
        return None, 0, "没有连续交易时钟"
    close = pd.to_numeric(points.close, errors="coerce")
    valid = points.valid.fillna(False).astype(bool)
    link = _clock_link(points) & valid & valid.shift(1, fill_value=False)
    one_step = np.log(close / close.shift(1))
    link &= np.isfinite(one_step)
    q_step = np.log(close / close.shift(q))
    contiguous_q = link.astype(int).rolling(q, min_periods=q).sum().eq(q)
    choices = valid & contiguous_q.fillna(False) & np.isfinite(q_step)
    base = one_step[link].to_numpy(dtype=float)
    aggregated = q_step[choices].to_numpy(dtype=float)
    base_var = _sample_variance(base)
    q_var = _sample_variance(aggregated)
    if base_var is None or base_var <= 1e-24:
        return None, int(aggregated.size), "5分钟单步收益方差为零，方差比无可定义"
    if q_var is None:
        return None, int(aggregated.size), "聚合收益样本不足"
    return _finite(q_var / (q * base_var)), int(aggregated.size), None


def _zero_crossing(points: pd.DataFrame) -> tuple[float | None, int, str | None]:
    if len(points) < 240:
        return None, 0, "不足240个5分钟点，无法形成过去5日移动中枢"
    close = pd.to_numeric(points.close, errors="coerce")
    valid_close = points.valid.fillna(False).astype(bool) & np.isfinite(close) & close.gt(0)
    observed_close = close.where(valid_close)
    moving = observed_close.rolling(240, min_periods=240).mean()
    centered = observed_close - moving
    pairs = _clock_link(points) & np.isfinite(centered) & np.isfinite(centered.shift(1))
    product = centered * centered.shift(1)
    crossings = pairs & product.lt(0)
    count = int(pairs.sum())
    if count == 0:
        return None, 0, "移动中枢形成后没有有效相邻价格"
    return _finite(float(crossings.sum()) / count), count, None


def _jump_and_semivariance(
    returns: pd.DataFrame,
) -> tuple[
    float | None, int, str | None,
    float | None, int, str | None,
    float | None, float | None, int, str | None,
]:
    if returns.empty:
        missing = (None, 0, "没有5分钟日内收益")
        return (*missing, *missing, None, None, 0, "没有5分钟日内收益")
    positive = returns.r.gt(0)
    negative = returns.r.lt(0)
    up_variance = float((returns.loc[positive, "r"].astype(float) ** 2).sum())
    down_variance = float((returns.loc[negative, "r"].astype(float) ** 2).sum())
    svr = None if down_variance <= 1e-24 else _finite(up_variance / down_variance)
    jumps: list[float] = []
    skews: list[float] = []
    kurtses: list[float] = []
    has_slots = {"start_slot", "end_slot"}.issubset(returns.columns)
    for _, same_day in returns.groupby("day", sort=True):
        values = same_day.r.astype(float).to_numpy()
        if not values.size:
            continue
        rv = float(np.dot(values, values))
        if rv <= 1e-24:
            continue
        bv_parts = []
        for _, session in same_day.groupby("session", sort=True):
            if has_slots:
                ordered = session.sort_values(
                    ["start_slot", "end_slot"], kind="stable",
                ).reset_index(drop=True)
                segment = (
                    ordered.start_slot.ne(ordered.end_slot.shift())
                    .fillna(True).cumsum()
                )
                segments = ordered.groupby(segment, sort=False)
            else:
                # Direct callers using the legacy frame have no clock
                # metadata; keep their single-session sample intact.
                segments = [(0, session)]
            for _, contiguous in segments:
                local = pd.to_numeric(contiguous.r, errors="coerce").to_numpy(
                    dtype=float,
                )
                local = local[np.isfinite(local)]
                if local.size < 2:
                    continue
                products = np.abs(local[1:]) * np.abs(local[:-1])
                pair_count = int(products.size)
                # A contiguous segment with p valid adjacent pairs contains
                # p + 1 returns, giving the standard finite-sample factor
                # (p + 1) / p without borrowing a pair across a real gap.
                correction = (pair_count + 1.0) / pair_count
                bv_parts.append(0.5 * math.pi * correction * float(products.sum()))
        if bv_parts:
            bv = float(sum(bv_parts))
            jumps.append(max(rv - bv, 0.0) / rv)
        squares = float(np.dot(values, values))
        if values.size >= 3:
            skews.append(math.sqrt(values.size) * float((values**3).sum()) / (squares**1.5))
        if values.size >= 4:
            kurtses.append(values.size * float((values**4).sum()) / squares**2 - 3.0)
    jump = _finite(float(np.mean(jumps))) if jumps else None
    skew = _finite(float(np.mean(skews))) if skews else None
    kurt = _finite(float(np.mean(kurtses))) if kurtses else None
    jump_reason = None if jump is not None else "没有包含至少两个相邻5分钟收益的有效交易日"
    moment_reason = None if skew is not None and kurt is not None else "每日有效5分钟收益不足4个"
    svr_reason = None if svr is not None else "下行半方差为零"
    return svr, len(returns), svr_reason, jump, len(jumps), jump_reason, skew, kurt, len(skews), moment_reason


def _ou_fit(points_30m: pd.DataFrame) -> tuple[dict[str, Any], str | None]:
    if points_30m.empty:
        return {}, "没有30分钟价格"
    close = pd.to_numeric(points_30m.close, errors="coerce")
    values = np.log(close.where(close > 0))
    same = (
        points_30m.day.eq(points_30m.day.shift())
        & points_30m.session.eq(points_30m.session.shift())
        & points_30m.bucket.eq(points_30m.bucket.shift() + 1)
    )
    pair = (
        same & points_30m.valid.fillna(False).astype(bool)
        & points_30m.valid.shift(1, fill_value=False).astype(bool)
        & np.isfinite(values) & np.isfinite(values.shift(1))
    )
    x = values.shift(1)[pair].to_numpy(dtype=float)
    y = values[pair].to_numpy(dtype=float)
    if x.size < 10:
        return {"sample_count": int(x.size)}, "同日同连续时段的有效30分钟相邻配对不足10个"
    x_mean, y_mean = float(x.mean()), float(y.mean())
    variance = float(np.dot(x - x_mean, x - x_mean) / x.size)
    if variance <= 1e-24:
        return {"sample_count": int(x.size)}, "30分钟对数价格没有变化，AR(1)不可拟合"
    phi = float(np.dot(x - x_mean, y - y_mean) / x.size / variance)
    alpha = y_mean - phi * x_mean
    residual = y - alpha - phi * x
    residual_sd = math.sqrt(float(np.dot(residual, residual) / max(1, x.size - 2)))
    details = {
        "sample_count": int(x.size), "phi": _finite(phi), "alpha": _finite(alpha),
        "residual_sd": _finite(residual_sd),
    }
    if not (0.0 < phi < 1.0):
        return details, f"30分钟AR(1)系数phi={phi:.6g}不满足0<phi<1"
    steady_sd = residual_sd / math.sqrt(max(1e-18, 1.0 - phi * phi))
    mean = alpha / (1.0 - phi)
    half_life = math.log(2.0) / (8.0 * -math.log(phi))
    details.update({
        "steady_sd": _finite(steady_sd),
        "mean_log_price": _finite(mean),
        "half_life_days": _finite(half_life),
    })
    if not all(math.isfinite(value) for value in (mean, steady_sd, half_life)):
        return details, "OU稳态参数包含非有限值"
    try:
        details["lower_price"] = _positive(math.exp(mean - 1.96 * steady_sd))
        details["upper_price"] = _positive(math.exp(mean + 1.96 * steady_sd))
        details["center_price"] = _positive(math.exp(mean))
    except OverflowError:
        return details, "OU稳态边界溢出"
    return details, None


def _atr30(
    points_30m: pd.DataFrame, days: pd.DatetimeIndex,
    adjusted_daily: pd.DataFrame | None = None,
) -> tuple[float | None, int, int, str | None]:
    if points_30m.empty:
        return None, 0, 0, "没有可聚合的30分钟OHLC"
    recent_days = set(days[-60:])
    frame = points_30m.sort_values(
        ["day", "session", "bucket"], kind="stable",
    ).reset_index(drop=True)
    valid = frame.valid_ohlc.fillna(False).astype(bool) & frame.day.isin(recent_days)
    usable_close = frame.valid.fillna(False).astype(bool)
    daily_close = pd.Series(dtype=float)
    if isinstance(adjusted_daily, pd.DataFrame) and not adjusted_daily.empty:
        history = adjusted_daily.copy()
        if {"date", "close"}.issubset(history.columns):
            history["date"] = pd.to_datetime(history.date, errors="coerce").dt.normalize()
            history["close"] = pd.to_numeric(history.close, errors="coerce")
            history = history[
                history.date.notna() & np.isfinite(history.close) & history.close.gt(0)
            ].sort_values("date", kind="stable").drop_duplicates("date", keep="last")
            daily_close = history.set_index("date").close.astype(float)
    ratios: list[float] = []
    valid_days: set[pd.Timestamp] = set()
    for position, row in frame.iterrows():
        if not bool(valid.loc[position]):
            continue
        day = pd.Timestamp(row.day).normalize()
        high, low = _positive(row.high), _positive(row.low)
        if high is None or low is None:
            continue
        previous_close = None
        if position > 0:
            previous = frame.iloc[position - 1]
            previous_day = pd.Timestamp(previous.day).normalize()
            previous_valid = bool(usable_close.loc[position - 1])
            if previous_valid and previous_day == day:
                previous_bucket = int(previous.bucket)
                current_bucket = int(row.bucket)
                same_session = (
                    int(previous.session) == int(row.session)
                    and previous_bucket == current_bucket - 1
                )
                lunch_continuation = (
                    int(previous.session) == 0 and previous_bucket == 3
                    and int(row.session) == 1 and current_bucket == 0
                )
                if same_session or lunch_continuation:
                    previous_close = _positive(previous.close)
        if (
            previous_close is None and int(row.session) == 0
            and int(row.bucket) == 0 and not daily_close.empty
        ):
            prior = daily_close[daily_close.index < day]
            if not prior.empty:
                previous_close = _positive(prior.iloc[-1])
        if previous_close is None:
            continue
        true_range = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close),
        )
        ratio = true_range / previous_close
        if math.isfinite(ratio) and ratio >= 0:
            ratios.append(float(ratio))
            valid_days.add(day)
    if len(valid_days) < 5 or len(ratios) < 15:
        return None, len(ratios), len(valid_days), "最近60日有效30分钟TR不足5日/15根"
    return _finite(float(np.mean(ratios))), len(ratios), len(valid_days), None


def _volume_profile(
    profile_bars: pd.DataFrame, days: pd.DatetimeIndex,
) -> tuple[dict[str, float], str | None]:
    if profile_bars.empty:
        return {}, "没有可用于成交量分布的分钟线"
    frame = _bar_slots(profile_bars)
    frame = frame[frame.day.isin(days)].copy()
    if "frequency" in frame:
        frame = frame[frame.frequency.eq("1m")].copy()
    if frame.empty:
        return {}, "没有原始的1分钟OHLC用于成交量分布"
    valid = _valid_bar_mask(frame, require_ohlc=True)
    volume = pd.to_numeric(frame.volume, errors="coerce") if "volume" in frame else pd.Series(np.nan, index=frame.index)
    price = (frame.open + frame.high + frame.low + frame.close) / 4.0
    usable = valid & np.isfinite(price) & price.gt(0) & np.isfinite(volume) & volume.gt(0)
    prices = price[usable].to_numpy(dtype=float)
    weights = volume[usable].to_numpy(dtype=float)
    if prices.size < 10:
        return {}, "有效加权分钟样本不足10个"
    low, high = float(prices.min()), float(prices.max())
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-15):
        bins = np.array([low - 0.5, high + 0.5])
        weights_by_bin = np.array([float(weights.sum())])
        centers = np.array([high])
    else:
        bins = np.linspace(low, high, PROFILE_BINS + 1)
        indices = np.clip(np.searchsorted(bins, prices, side="right") - 1, 0, PROFILE_BINS - 1)
        weights_by_bin = np.bincount(indices, weights=weights, minlength=PROFILE_BINS)
        centers = (bins[:-1] + bins[1:]) / 2.0
    total = float(weights_by_bin.sum())
    if total <= 0:
        return {}, "调整后成交量为零"
    cumulative = np.cumsum(weights_by_bin)
    lower = float(centers[min(int(np.searchsorted(cumulative, total * 0.05, side="left")), centers.size - 1)])
    upper = float(centers[min(int(np.searchsorted(cumulative, total * 0.95, side="left")), centers.size - 1)])
    peak = float(centers[int(np.argmax(weights_by_bin))])
    return {"peak": peak, "lower": lower, "upper": upper}, None


def _mdDD(adjusted_daily: pd.DataFrame) -> tuple[float | None, dict[str, Any], str | None]:
    if adjusted_daily.empty or "close" not in adjusted_daily:
        return None, {"mddd_censored": None}, "没有日线收盘历史"
    frame = adjusted_daily.copy()
    frame["close"] = pd.to_numeric(frame.close, errors="coerce")
    frame = frame[np.isfinite(frame.close) & frame.close.gt(0)].sort_values("date").drop_duplicates(
        "date", keep="last",
    )
    if len(frame) < 250:
        diagnostics = {
            "mddd_censored": None,
            "mddd_window_start": str(frame.date.iloc[0].date()) if not frame.empty else None,
            "mddd_window_end": str(frame.date.iloc[-1].date()) if not frame.empty else None,
            "mddd_days": len(frame),
        }
        return None, diagnostics, "独立250日日线窗口历史不足"
    window = frame.iloc[-250:].reset_index(drop=True)
    close = window.close.to_numpy(dtype=float)
    peak_value = close[0]
    peak_index = 0
    high_water_date = pd.Timestamp(window.date.iloc[0])
    in_drawdown = False
    trough_ratio = 1.0
    episodes: list[tuple[int, int, int, float, pd.Timestamp, pd.Timestamp | None, bool]] = []
    for index in range(1, len(close)):
        value = close[index]
        if value >= peak_value:
            if in_drawdown:
                episodes.append((
                    peak_index, index, index - peak_index, trough_ratio,
                    high_water_date, pd.Timestamp(window.date.iloc[index]), False,
                ))
                in_drawdown = False
            peak_value = value
            peak_index = index
            high_water_date = pd.Timestamp(window.date.iloc[index])
            trough_ratio = 1.0
        else:
            drawdown = (peak_value - value) / peak_value
            if not in_drawdown:
                in_drawdown = True
                trough_ratio = drawdown
            elif drawdown > trough_ratio:
                trough_ratio = drawdown
    censored = bool(in_drawdown)
    if in_drawdown:
        episodes.append((
            peak_index, len(close) - 1, len(close) - 1 - peak_index, trough_ratio,
            high_water_date, None, True,
        ))
    if not episodes:
        diagnostics = {
            "mddd_censored": False,
            "mddd_window_start": str(window.date.iloc[0].date()),
            "mddd_window_end": str(window.date.iloc[-1].date()),
            "mddd_days": 250,
            "mddd_peak_date": str(window.date.iloc[-1].date()),
            "mddd_trough_date": None,
            "mddd_recovery_date": None,
            "mddd_max_drawdown": 0.0,
            "mddd_episode_count": 0,
        }
        return 0.0, diagnostics, None
    peak_idx, end_idx, duration, max_drawdown, peak_day, recovery_day, episode_censored = max(
        episodes, key=lambda item: (item[2], item[1]),
    )
    trough_index = int(np.argmin(close[peak_idx:end_idx + 1])) + peak_idx
    diagnostics = {
        "mddd_censored": bool(censored),
        "mddd_selected_episode_censored": bool(episode_censored),
        "mddd_window_start": str(window.date.iloc[0].date()),
        "mddd_window_end": str(window.date.iloc[-1].date()),
        "mddd_days": 250,
        "mddd_peak_date": str(peak_day.date()),
        "mddd_trough_date": str(window.date.iloc[trough_index].date()),
        "mddd_recovery_date": str(recovery_day.date()) if recovery_day is not None else None,
        "mddd_max_drawdown": _finite(max_drawdown),
        "mddd_duration_days": int(duration),
        "mddd_episode_count": len(episodes),
        "mddd_duration_unit": "交易日，峰值到恢复或窗口末端",
    }
    return _finite(float(duration)), diagnostics, None


def _mdDD_unverified(adjusted_daily: pd.DataFrame) -> tuple[float | None, dict[str, Any], str | None]:
    frame = adjusted_daily.copy()
    if not frame.empty and "close" in frame:
        frame = frame[np.isfinite(pd.to_numeric(frame.close, errors="coerce"))]
    start = str(frame.date.iloc[0].date()) if not frame.empty and "date" in frame else None
    end = str(frame.date.iloc[-1].date()) if not frame.empty and "date" in frame else None
    diagnostics = {
        "mddd_censored": None,
        "mddd_history_verified": False,
        "mddd_window_start": start,
        "mddd_window_end": end,
        "mddd_days": int(len(frame)),
    }
    return None, diagnostics, "MDDD所需的250日独立日线缺少版本化校验账本"


def _price_tick(data: DataBundle) -> float:
    metadata = data.metadata if isinstance(data.metadata, dict) else {}
    tick = _positive(metadata.get("price_tick"))
    if tick is not None:
        return tick
    try:
        tick = _positive(rules.instrument_terms(data.symbol, metadata).get("price_tick"))
    except Exception:
        tick = None
    return tick or 0.001


def _source_quality(
    bars: pd.DataFrame, days: pd.DatetimeIndex, daily: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if bars.empty:
        return {
            "minute_rows": 0, "minute_days": 0, "synthetic_missing_minutes": 0,
            "auction_placeholder_minutes": 0, "zero_volume_retained_minutes": 0,
            "amount_field": None, "amount_missing_minutes": 0,
            "amount_zero_with_volume_minutes": 0, "amount_valid_ratio": None,
            "volume_coverage": None, "turnover_source": "unknown",
            "vwap_quality": "unknown", "turnover_confirmed": False,
            "source_names": [],
        }
    frame = bars[bars.timestamp.dt.normalize().isin(days)].copy()
    if "frequency" in frame:
        frame = frame[~frame.frequency.eq("1d")].copy()
    synthetic = _bool_series(frame, "is_synthetic")
    placeholder = _bool_series(frame, "is_auction_placeholder")
    volume = pd.to_numeric(
        frame.get("volume", pd.Series(np.nan, index=frame.index)), errors="coerce",
    )
    amount_field = "turnover" if "turnover" in frame else "amount" if "amount" in frame else None
    amount = (
        pd.to_numeric(frame[amount_field], errors="coerce")
        if amount_field else pd.Series(np.nan, index=frame.index)
    )
    missing = amount.isna() | ~np.isfinite(amount)
    valid_amount = ~missing & amount.ge(0)
    zero_with_volume = valid_amount & amount.eq(0) & volume.gt(0)
    valid_bar = _valid_bar_mask(frame)
    volume_valid = valid_bar & np.isfinite(volume) & volume.ge(0)
    observed_days = int(frame.timestamp.dt.normalize().nunique())
    expected_volume_slots = observed_days * MINUTES_PER_DAY
    source_text = (
        frame.source.fillna("").astype(str).str.strip().str.lower()
        if "source" in frame else pd.Series("", index=frame.index)
    )
    economic = ~placeholder & source_text.ne("")
    source_names = sorted(source_text[economic].unique().tolist())
    non_placeholder = ~placeholder
    all_ths = bool(non_placeholder.any()) and bool(source_text[non_placeholder].eq("ths").all())
    turnover_source = "unknown"
    vwap_quality = "unknown"
    if all_ths:
        daily_scaled = False
        if isinstance(daily, pd.DataFrame) and not daily.empty and "date" in daily:
            daily_turnover_field = "turnover" if "turnover" in daily else "amount" if "amount" in daily else None
            if daily_turnover_field is not None:
                official = daily.copy()
                official["date"] = pd.to_datetime(official.date, errors="coerce").dt.normalize()
                official[daily_turnover_field] = pd.to_numeric(
                    official[daily_turnover_field], errors="coerce",
                )
                official = official.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date")
                minute_turnover = pd.to_numeric(
                    frame.turnover if "turnover" in frame else frame.get("amount", pd.Series(np.nan, index=frame.index)),
                    errors="coerce",
                )
                by_day = frame.assign(
                    _day=frame.timestamp.dt.normalize(), _amount=minute_turnover,
                ).groupby("_day", sort=True)._amount.sum(min_count=1)
                comparisons = []
                for day, estimated in by_day.items():
                    reference = _positive(official[daily_turnover_field].get(day))
                    comparisons.append(
                        reference is not None and _finite(estimated) is not None
                        and math.isclose(float(estimated), reference, rel_tol=1e-6, abs_tol=1e-6)
                    )
                daily_scaled = bool(comparisons) and all(comparisons)
        turnover_source = "estimated_ohlc4_daily_scaled" if daily_scaled else "estimated"
        vwap_quality = turnover_source
    return {
        "minute_rows": int(len(frame)),
        "minute_days": int(frame.timestamp.dt.normalize().nunique()),
        "synthetic_missing_minutes": int((synthetic & ~placeholder).sum()),
        "auction_placeholder_minutes": int(placeholder.sum()),
        "zero_volume_retained_minutes": int((~synthetic & volume.eq(0)).sum()),
        "amount_field": amount_field,
        "amount_missing_minutes": int(missing.sum()),
        "amount_zero_with_volume_minutes": int(zero_with_volume.sum()),
        "amount_valid_ratio": _finite(float(valid_amount.sum()) / len(frame)) if len(frame) else None,
        "volume_coverage": (
            _finite(float(volume_valid.sum()) / expected_volume_slots)
            if expected_volume_slots else None
        ),
        "turnover_source": turnover_source,
        "vwap_quality": vwap_quality,
        "turnover_confirmed": False,
        "source_names": source_names,
    }


def _minute_integrity(
    points: pd.DataFrame, days: pd.DatetimeIndex, ignored_days: set[pd.Timestamp],
) -> dict[str, Any]:
    """Validate all 240 expected slots, with calibrated OHLC or a legal placeholder."""
    required = set(days) - ignored_days
    complete: set[pd.Timestamp] = set()
    failures: list[dict[str, Any]] = []
    expected_slots = 0
    accepted_slots = 0
    placeholder_slots = 0
    unverified_slots = 0
    invalid_ohlc_slots = 0
    missing_slots = 0
    if points.empty:
        by_day = {}
    else:
        by_day = {
            pd.Timestamp(day).normalize(): part
            for day, part in points.groupby("day", sort=False)
        }
    for day in sorted(required):
        expected_slots += 240
        part = by_day.get(day)
        if part is None or part.empty:
            missing_slots += 240
            failures.append({
                "date": day.date().isoformat(), "valid_slots": 0,
                "missing_slots": 240, "unverified_slots": 0,
                "invalid_ohlc_slots": 0, "placeholder_slots": 0,
                "missing_examples": [],
            })
            continue
        valid_ohlc = part.valid_ohlc.fillna(False).astype(bool) if "valid_ohlc" in part else pd.Series(False, index=part.index)
        volume = pd.to_numeric(
            part.get("volume", pd.Series(np.nan, index=part.index)), errors="coerce",
        )
        placeholder = (
            _bool_series(part, "is_auction_placeholder")
            & np.isfinite(volume) & volume.eq(0)
        )
        calibrated = part.source_verified.eq(True) if "source_verified" in part else pd.Series(False, index=part.index)
        accepted = placeholder | (valid_ohlc & calibrated)
        count = int(accepted.sum())
        accepted_slots += count
        placeholder_count = int(placeholder.sum())
        placeholder_slots += placeholder_count
        unverified_count = int((valid_ohlc & ~calibrated & ~placeholder).sum())
        unverified_slots += unverified_count
        invalid_count = int((~valid_ohlc & ~placeholder).sum())
        invalid_ohlc_slots += invalid_count
        missing_slots += max(0, 240 - count)
        complete_day = len(part) == 240 and count == 240
        if complete_day:
            complete.add(day)
            continue
        examples = [] if "timestamp" not in part else [
            pd.Timestamp(value).isoformat() for value in part.loc[~accepted, "timestamp"].head(5)
        ]
        failures.append({
            "date": day.date().isoformat(), "valid_slots": count,
            "missing_slots": int(240 - count), "unverified_slots": unverified_count,
            "invalid_ohlc_slots": invalid_count, "placeholder_slots": placeholder_count,
            "missing_examples": examples,
        })
    return {
        "complete_days": complete,
        "incomplete_days": [item["date"] for item in failures],
        "failures": failures,
        "expected_slots": expected_slots,
        "accepted_slots": accepted_slots,
        "placeholder_slots": placeholder_slots,
        "unverified_slots": unverified_slots,
        "invalid_ohlc_slots": invalid_ohlc_slots,
        "missing_slots": missing_slots,
    }


def _repair_action_reference_precloses(
    daily: pd.DataFrame, factors: list[dict[str, Any]],
) -> pd.DataFrame:
    """Replace stale ex-day precloses with the reference used by each factor."""
    if daily.empty or "date" not in daily or "preclose" not in daily or not factors:
        return daily.copy()
    repaired = daily.copy()
    dates = pd.to_datetime(repaired.date, errors="coerce").dt.normalize()
    for item in factors:
        try:
            ex_date = pd.Timestamp(item.get("ex_date")).normalize()
        except (TypeError, ValueError, OverflowError):
            continue
        if pd.isna(ex_date):
            continue
        source = str(item.get("source") or "")
        if source == "verified_preclose":
            reference = _positive(item.get("preclose"))
        else:
            reference = (
                _positive(item.get("implied_reference"))
                or _positive(item.get("preclose"))
            )
        if reference is None:
            continue
        repaired.loc[dates.eq(ex_date), "preclose"] = reference
    return repaired


def _prepare(data: DataBundle, as_of: pd.Timestamp, window_days: int) -> _Context:
    raw_bars = _normalise_bars(data, as_of)
    raw_daily, prices = _normalise_daily(data, as_of)
    calendar_days = _calendar_days(data, raw_daily, as_of)
    if window_days <= 0:
        raise _StatsError("unavailable", "window_days 必须为正整数")
    selected = pd.DatetimeIndex(calendar_days[-window_days:]) if len(calendar_days) else pd.DatetimeIndex([])
    coverage_complete = len(calendar_days) >= window_days
    if not len(selected):
        raise _StatsError("data_not_ready", "as_of 之前没有可用交易日历")
    metadata = data.metadata if isinstance(data.metadata, dict) else {}
    mddd_verified = _verified(metadata.get("mddd_history_verified"))
    mddd_start = (
        pd.Timestamp(raw_daily.date.iloc[-250]).normalize()
        if mddd_verified and len(raw_daily) >= 250 else selected[0]
    )
    action_start = min(selected[0], mddd_start)
    daily_before = raw_daily[raw_daily.date < action_start].tail(1)
    prices_before = prices[prices.index < action_start].tail(1) if not prices.empty else prices
    action_daily = pd.concat([
        daily_before, raw_daily[raw_daily.date >= action_start],
    ], ignore_index=True).copy()
    action_prices = (
        pd.concat([prices_before, prices[prices.index >= action_start]]).copy()
        if not prices.empty else prices
    )
    action_bars = raw_bars[raw_bars.timestamp.dt.normalize() >= action_start].copy()
    action_data = copy(data)
    action_data.daily = action_daily
    action_data.bars = action_bars
    if isinstance(data.actions, pd.DataFrame) and "ex_date" in data.actions:
        ex_dates = pd.to_datetime(data.actions.ex_date, errors="coerce")
        if getattr(ex_dates.dt, "tz", None) is not None:
            ex_dates = ex_dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        ex_dates = ex_dates.dt.normalize()
        action_data.actions = data.actions.loc[
            ex_dates.notna() & ex_dates.ge(action_start) & ex_dates.le(as_of.normalize())
        ].copy()
    else:
        action_data.actions = data.actions.copy() if isinstance(data.actions, pd.DataFrame) else pd.DataFrame()
    factors, factor_error = price_basis.action_factors(
        action_data, as_of, prices=action_prices if not action_prices.empty else None,
    )
    mddd_warning = None
    if factor_error and action_start < selected[0]:
        # An unsupported older split can invalidate the independent 250-day
        # risk metric without invalidating a later, already ex-action window.
        short_data = copy(action_data)
        short_data.daily = pd.concat([
            action_daily[action_daily.date < selected[0]].tail(1),
            action_daily[action_daily.date >= selected[0]],
        ], ignore_index=True)
        short_data.bars = action_bars[action_bars.timestamp.dt.normalize() >= selected[0]].copy()
        if "ex_date" in short_data.actions:
            short_data.actions = short_data.actions[pd.to_datetime(short_data.actions.ex_date).dt.normalize() >= selected[0]].copy()
        short_prices = pd.concat([action_prices[action_prices.index < selected[0]].tail(1), action_prices[action_prices.index >= selected[0]]]) if not action_prices.empty else None
        short_factors, short_error = price_basis.action_factors(short_data, as_of, prices=short_prices)
        if not short_error:
            mddd_warning = "独立250日日线存在无法处理的较早公司行动，MDDD不可用；当前分钟窗口不跨越该行动，保留分钟统计"
            mddd_verified = False
            factors, factor_error = short_factors, None
    if factor_error:
        raise _StatsError("data_not_ready", str(factor_error))
    if factors is None:
        factors = []
    elif not isinstance(factors, list):
        factors = list(factors)
    statistics_daily = _repair_action_reference_precloses(action_daily, factors)
    try:
        adjusted_bars = price_basis.adjust_frame(
            raw_bars, factors, timestamp_column="timestamp", adjust_volume=False,
        )
        adjusted_daily = price_basis.adjust_frame(
            statistics_daily, factors, timestamp_column="date", adjust_volume=False,
        )
        profile_bars = price_basis.adjust_frame(
            raw_bars, factors, timestamp_column="timestamp", adjust_volume=True,
        )
    except Exception as exc:
        raise _StatsError("data_not_ready", f"公司行动价格调整失败：{exc}") from exc

    bars_window = raw_bars[raw_bars.timestamp.dt.normalize().isin(selected)].copy()
    minute = bars_window[~bars_window.frequency.eq("1d")].copy()
    minute_kind = None
    sampling = metadata.get("etf_sampling")
    integrity = None
    if not minute.empty:
        if minute.frequency.eq("1m").any():
            minute_kind = "1m"
        elif minute.frequency.eq("5m").any():
            minute_kind = "5m"
    if minute_kind == "1m":
        points_1m = _expected_minutes(
            adjusted_bars[adjusted_bars.frequency.eq("1m")], selected, raw_bars,
        )
        if sampling:
            excluded = set(sampling.get("statistics_excluded_days") or [])
            own_excluded = set(sampling.get("excluded_days") or [])
            expected = {day.date().isoformat() for day in selected}
            count = len(expected - excluded)
            if (
                sampling.get("expected_days") != window_days
                or not own_excluded.issubset(excluded) or not excluded.issubset(expected)
                or sampling.get("statistics_days") != count
            ):
                raise _StatsError("data_not_ready", "冻结的分钟统计日与交易日历不一致")
            ignored = {
                pd.Timestamp(item["date"]).normalize()
                for item in (data.quality or {}).get("days", [])
                if item.get("mode") in {"suspended", "no_trades"}
            }
            integrity = _minute_integrity(points_1m, selected, ignored)
            omitted = points_1m.day.isin(pd.to_datetime(sorted(excluded)))
            points_1m.loc[omitted, ["valid", "valid_ohlc"]] = False
            # Keep the full trading clock, including invalid slots, so rolling
            # returns cannot join the two sides of an omitted session.
            profile_bars = profile_bars[~profile_bars.timestamp.dt.normalize().isin(pd.to_datetime(sorted(excluded)))].copy()
        points_5m = _expected_five_minutes(points_1m)
        points_30m = _expected_thirty_minutes(points_1m)
    elif minute_kind == "5m":
        points_1m = pd.DataFrame()
        points_5m = _direct_five_minutes(adjusted_bars, selected)
        points_30m = pd.DataFrame()
    else:
        points_1m = pd.DataFrame()
        points_5m = pd.DataFrame()
        points_30m = pd.DataFrame()

    warnings: list[str] = []
    if mddd_warning:
        warnings.append(mddd_warning)
    quality = data.quality if isinstance(data.quality, dict) else {}
    if quality.get("is_demo"):
        warnings.append("演示数据不可作为真实市场统计结论")
    if minute_kind != "1m":
        warnings.append("窗口没有完整1分钟数据；1分钟RV、路径冗余和成交量分布不可用")
    daily_only = bars_window[bars_window.frequency.eq("1d")]
    if not daily_only.empty:
        warnings.append(
            f"窗口有{int(daily_only.timestamp.dt.normalize().nunique())}个整日降级行，未还原盘中路径",
        )
    if sampling and sampling.get("statistics_excluded_days"):
        warnings.append(f"分钟指标使用 {sampling['statistics_days']}/{window_days} 个完整交易日；缺口日整日排除，未用日线或合成分钟替代")
    return _Context(
        as_of=as_of, requested_window_days=window_days,
        calendar_days=calendar_days, days=selected, coverage_complete=coverage_complete,
        raw_bars=raw_bars, raw_daily=raw_daily, adjusted_bars=adjusted_bars, adjusted_daily=adjusted_daily,
        profile_bars=profile_bars, factors=factors, warnings=warnings,
        minute_kind=minute_kind, points_1m=points_1m, points_5m=points_5m,
        points_30m=points_30m,
        sampling=sampling, data_resolution=metadata.get("data_resolution"), minute_integrity=integrity,
        mddd_history_verified=mddd_verified,
        mddd_unavailable_reason=mddd_warning,
    )


def _empty_grid_basis(
    price_tick: float | None = None, current_raw: float | None = None,
) -> dict[str, Any]:
    return {
        "reference_price": None, "lower_price": None, "upper_price": None,
        "current_raw_price": current_raw, "current_price_date": None,
        "current_price_source": None, "current_price_basis": "as_of_adjusted",
        "current_raw_price_unadjusted": None,
        "atr30_pct": None, "svr": None,
        "realized_skewness": None, "realized_kurtosis": None, "price_tick": price_tick,
    }


def _empty_result(
    status: str, *, reason: str = "", diagnostics: dict[str, Any] | None = None,
    price_tick: float | None = None, current_raw: float | None = None,
    market_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {key: None for key in METRIC_KEYS}
    details = {
        key: {"status": "unavailable", "reason": reason or "未完成计算", "sample_count": 0}
        for key in METRIC_KEYS
    }
    return {
        "status": status, "metrics": metrics, "metric_details": details,
        "grid_basis": _empty_grid_basis(price_tick, current_raw),
        "market_state": market_state or _empty_market_state(reason),
        "actions_applied": [], "eligible": False,
        "filter_reasons": [] if status == "completed" else [f"status_{status}"],
        "warnings": [reason] if reason else [], "diagnostics": diagnostics or {},
    }


def _empty_market_state(reason: str = "市场状态未计算") -> dict[str, Any]:
    return {
        "method_version": "daily_state_v1", "as_of": None,
        "status": "unavailable",
        "metrics": {key: None for key in MARKET_STATE_METRIC_KEYS},
        "metric_details": {},
        "samples": {}, "warmup": {},
        "price_basis": {"error": reason}, "input_fingerprint": None,
        "message": reason,
    }


def _market_state_diagnostic(data: DataBundle, as_of: Any) -> dict[str, Any]:
    try:
        return daily_market_state(data, as_of)
    except Exception as exc:
        return _empty_market_state(
            f"日线市场状态计算失败：{type(exc).__name__}: {exc}",
        )


def _safe_action(item: Mapping[str, Any]) -> dict[str, Any]:
    def safe_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): safe_value(nested) for key, nested in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe_value(nested) for nested in value]
        if value is None or isinstance(value, (str, bool)):
            return value
        number = _finite(value)
        return number if number is not None else None

    result: dict[str, Any] = {}
    for key, value in item.items():
        if key == "ex_date":
            try:
                result[key] = pd.Timestamp(value).date().isoformat()
            except (TypeError, ValueError):
                result[key] = None
        else:
            result[key] = safe_value(value)
    return result


def _set_metric(
    metrics: dict[str, Any], details: dict[str, Any], key: str, value: Any,
    sample_count: int, *, reason: str, null_reason: str | None = None,
) -> None:
    number = _finite(value)
    if number is None:
        details[key] = {
            "status": "insufficient_samples",
            "reason": null_reason or reason,
            "sample_count": int(max(0, sample_count)),
        }
        metrics[key] = None
        return
    metrics[key] = number
    details[key] = {
        "status": "available", "reason": reason,
        "sample_count": int(max(0, sample_count)),
    }


def _current_trusted_close(context: _Context) -> tuple[float | None, str | None, str | None, float | None]:
    """Return the latest plan-approved close, adjusted to the decision basis."""
    plan = context.data_resolution
    modes = {item.get("date"): item.get("mode") for item in (plan or {}).get("days", [])}
    if plan:
        candidates = [
            item["date"] for item in plan["days"]
            if item.get("mode") in {"minute", "daily"}
        ]
        for day in reversed(candidates):
            stamp = pd.Timestamp(day).normalize()
            mode = modes[day]
            if mode == "minute" and not context.raw_bars.empty:
                part = context.raw_bars[
                    context.raw_bars.timestamp.dt.normalize().eq(stamp)
                    & ~context.raw_bars.frequency.eq("1d")
                ]
                if not part.empty:
                    values = pd.to_numeric(part.close, errors="coerce")
                    valid = np.isfinite(values) & values.gt(0)
                    if valid.any():
                        raw = _finite(values[valid].iloc[-1])
                        source = "minute"
                    else:
                        raw = None
                        source = None
                else:
                    raw, source = None, None
            else:
                raw, source = None, None
            if raw is None and not context.raw_daily.empty:
                row = context.raw_daily[context.raw_daily.date.eq(stamp)]
                if not row.empty:
                    raw = _positive(row.sort_values("date").iloc[-1].get("close"))
                    if raw is not None:
                        source = "daily"
            if raw is not None:
                factor = price_basis.price_factor_at(context.factors, stamp)
                adjusted = raw * factor if _finite(factor) is not None and factor > 0 else None
                if adjusted is not None and math.isfinite(adjusted) and adjusted > 0:
                    return float(adjusted), stamp.date().isoformat(), source, float(raw)
        return None, None, None, None
    if context.points_1m.empty:
        return None, None, None, None
    raw = pd.to_numeric(context.points_1m.raw_close, errors="coerce")
    usable = context.points_1m.valid.fillna(False).astype(bool) & np.isfinite(raw) & raw.gt(0)
    if not usable.any():
        return None, None, None, None
    index = usable[usable].index[-1]
    stamp = pd.Timestamp(context.points_1m.loc[index, "day"]).normalize()
    raw_value = _finite(raw.loc[index])
    factor = price_basis.price_factor_at(context.factors, stamp)
    adjusted = raw_value * factor if raw_value is not None and _finite(factor) is not None and factor > 0 else None
    return (adjusted, stamp.date().isoformat(), "minute", raw_value)


def _recent_raw_price(context: _Context) -> tuple[float | None, int, int]:
    if context.points_1m.empty:
        return None, 0, 0
    raw = pd.to_numeric(context.points_1m.raw_close, errors="coerce")
    usable = context.points_1m.valid.fillna(False).astype(bool) & np.isfinite(raw) & raw.gt(0)
    frame = context.points_1m.loc[usable, ["day", "raw_close"]].copy()
    if frame.empty:
        return None, 0, 0
    recent_days = sorted(frame.day.unique())[-20:]
    frame = frame[frame.day.isin(recent_days)]
    value = _finite(float(pd.to_numeric(frame.raw_close, errors="coerce").mean()))
    return value, int(len(frame)), int(len(recent_days))


def compute_etf_statistics(
    data: DataBundle, as_of: str | pd.Timestamp, window_days: int,
) -> dict[str, Any]:
    """Compute ETF suitability metrics using data no later than ``as_of``.

    All returned numbers are finite JSON-safe values or ``None``. The
    corresponding ``metric_details`` entry explains every null result.
    """
    market_state = _market_state_diagnostic(data, as_of)
    try:
        stamp = _timestamp(as_of)
        if not isinstance(window_days, (int, np.integer)) or isinstance(window_days, bool):
            raise _StatsError("unavailable", "window_days 必须为正整数")
        context = _prepare(data, stamp, int(window_days))
        metrics: dict[str, Any] = {key: None for key in METRIC_KEYS}
        details: dict[str, Any] = {key: {} for key in METRIC_KEYS}
        tick = _price_tick(data)
        current_raw, current_price_date, current_price_source, current_raw_unadjusted = _current_trusted_close(context)
        if context.sampling and stamp.normalize().date().isoformat() in context.sampling.get("statistics_excluded_days", []):
            official = data.daily[pd.to_datetime(data.daily.date).dt.normalize().le(stamp.normalize())]
            if not official.empty:
                current_raw = _positive(official.sort_values("date").close.iloc[-1])
        mean_raw, raw_count, raw_days = _recent_raw_price(context)

        returns_1m = _within_session_returns(context.points_1m)
        rv_daily_1m = _daily_rv(returns_1m)
        rv_1m, rv_1m_days, rv_1m_error = _annualized(rv_daily_1m)
        _set_metric(
            metrics, details, "rv_annualized", rv_1m, rv_1m_days,
            reason="1分钟日内平方收益按日均值×252年化；不跨午休、不跨隔夜",
            null_reason=rv_1m_error,
        )

        for frequency, points, suffix in (
            (5, context.points_5m, ""), (1, context.points_1m, "_1m"),
        ):
            path_means = _path_redundancy_means(points, context.days, sampling_minutes=frequency)
            for key, (path_value, path_count, path_error) in path_means.items():
                _set_metric(
                    metrics, details, key + suffix, path_value, path_count,
                    reason=f"{frequency}分钟采样；先独立计算每个连续完整交易日窗口的1-|末收-首开|/累计价格路程，再等权平均；跨日含真实隔夜位移，缺口整窗剔除，零路程不计入，样本数为有效窗口数",
                    null_reason=path_error,
                )

        iovr_value, iovr_count, iovr_error = _official_iovr(
            context.adjusted_daily, returns_1m, context.days,
        )
        _set_metric(
            metrics, details, "iovr", iovr_value, iovr_count,
            reason="日内1分钟RV均值/官方开盘对前收盘的隔夜平方收益均值；零分母返回null",
            null_reason=iovr_error,
        )

        for key, q in (
            ("variance_ratio_60m", 12),
            ("variance_ratio_240m", 48),
            ("variance_ratio_1200m", 240),
        ):
            value, count, error = _variance_ratio(context.points_5m, q)
            _set_metric(
                metrics, details, key, value, count,
                reason=f"5分钟连续交易时钟q={q}轴聚合方差比；含真实隔夜位移，描述性且不输出p值",
                null_reason=error,
            )

        returns_5m = _within_session_returns(context.points_5m)
        rv5_daily = _daily_rv(returns_5m)
        rv5, rv5_days, rv5_error = _annualized(rv5_daily)
        _set_metric(
            metrics, details, "rv_5m_annualized", rv5, rv5_days,
            reason="5分钟日内平方收益按日均值×252年化；不跨午休、不跨隔夜",
            null_reason=rv5_error,
        )

        mddd_verified = context.mddd_history_verified
        if mddd_verified:
            mddd, mddd_details, mddd_error = _mdDD(context.adjusted_daily)
        else:
            mddd, mddd_details, mddd_error = _mdDD_unverified(context.adjusted_daily)
            mddd_error = context.mddd_unavailable_reason or mddd_error
        _set_metric(
            metrics, details, "max_drawdown_duration_days", mddd,
            int(mddd_details.get("mddd_days", 0)),
            reason="独立最近250个日线收盘；峰值至恢复前高的最长交易日数，末端未恢复会计入并censor",
            null_reason=mddd_error,
        )

        zcr_value, zcr_count, zcr_error = _zero_crossing(context.points_5m)
        _set_metric(
            metrics, details, "zero_crossing_rate", zcr_value, zcr_count,
            reason="5分钟close对过去240点（5日）算术滚动均线的严格变号数/有效相邻价格对",
            null_reason=zcr_error,
        )

        (
            svr, svr_count, svr_error, jump, jump_count, jump_error,
            skew, kurt, statistic_days, moment_error,
        ) = _jump_and_semivariance(returns_5m)
        _set_metric(
            metrics, details, "jump_ratio", jump, jump_count,
            reason="5分钟按日max(RV-BV,0)/RV；BV只在同段相邻配对并乘N/(N-1)，不跨午休或隔夜",
            null_reason=jump_error,
        )

        tpr = tick / mean_raw if mean_raw and mean_raw > 0 else None
        _set_metric(
            metrics, details, "tick_to_price_ratio", tpr, raw_count,
            reason=f"最近{raw_days}个有分钟数据的交易日raw分钟收盘均价；tick/mean_raw",
            null_reason="没有最近20个有效raw分钟收盘样本" if tpr is None else None,
        )
        _set_metric(
            metrics, details, "mean_raw_price", mean_raw, raw_count,
            reason="最近20个有分钟数据的交易日的raw分钟收盘均价",
            null_reason="没有最近20个有效raw分钟收盘样本" if mean_raw is None else None,
        )

        ou_details, ou_error = _ou_fit(context.points_30m)
        _set_metric(
            metrics, details, "ou_half_life_days", ou_details.get("half_life_days"),
            int(ou_details.get("sample_count", 0)),
            reason="30分钟同日同连续时段AR(1)，仅0<phi<1；半衰期=ln2/(8*-lnphi)交易日",
            null_reason=ou_error,
        )

        atr, atr_count, atr_days, atr_error = _atr30(
            context.points_30m, context.days, context.adjusted_daily,
        )
        if data.metadata.get("etf_volume_verified") is False:
            context.warnings.append(
                "分钟成交量未通过日线一致性核验；成交量分布仍按分钟成交量及份额折算计算"
            )
        profile, profile_error = _volume_profile(context.profile_bars, context.days)
        ou_center = _positive(ou_details.get("center_price"))
        ou_lower = _positive(ou_details.get("lower_price"))
        ou_upper = _positive(ou_details.get("upper_price"))
        profile_peak = _positive(profile.get("peak"))
        profile_lower = _positive(profile.get("lower"))
        profile_upper = _positive(profile.get("upper"))
        reference = current_raw
        if profile_peak is not None and ou_center is not None:
            reference = 0.6 * profile_peak + 0.4 * ou_center
        elif profile_peak is not None:
            reference = profile_peak
        elif ou_center is not None:
            reference = ou_center
        lower_price = (
            0.5 * profile_lower + 0.5 * ou_lower
            if profile_lower is not None and ou_lower is not None
            else profile_lower if profile_lower is not None else ou_lower
        )
        upper_price = (
            0.5 * profile_upper + 0.5 * ou_upper
            if profile_upper is not None and ou_upper is not None
            else profile_upper if profile_upper is not None else ou_upper
        )
        if reference is not None and lower_price is not None and upper_price is not None:
            if not lower_price < reference < upper_price:
                context.warnings.append("成交量分布与OU融合后的中枢不在上下边界内")
                lower_price = None
                upper_price = None

        grid_basis = _empty_grid_basis(tick, current_raw)
        grid_basis.update({
            "current_price_date": current_price_date,
            "current_price_source": current_price_source,
            "current_price_basis": "as_of_adjusted",
            "current_raw_price_unadjusted": current_raw_unadjusted,
            "reference_price": _finite(reference),
            "lower_price": _finite(lower_price),
            "upper_price": _finite(upper_price),
            "atr30_pct": _finite(atr),
            "svr": _finite(svr),
            "realized_skewness": _finite(skew),
            "realized_kurtosis": _finite(kurt),
        })
        if profile_error:
            context.warnings.append(f"成交量分布未生成：{profile_error}")
        if atr_error:
            context.warnings.append(f"ATR30未生成：{atr_error}")
        if ou_error:
            context.warnings.append(f"OU边界未生成：{ou_error}")

        quality = data.quality if isinstance(data.quality, dict) else {}
        ignored_minute_days: set[pd.Timestamp] = set()
        for report in quality.get("days", []) if isinstance(quality.get("days"), list) else []:
            if not isinstance(report, Mapping) or report.get("mode") not in {"suspended", "no_trades"}:
                continue
            try:
                ignored_minute_days.add(pd.Timestamp(report.get("date")).normalize())
            except (TypeError, ValueError):
                continue
        integrity = context.minute_integrity or _minute_integrity(
            context.points_1m, context.days, ignored_minute_days,
        )
        complete_minute_days = integrity["complete_days"]
        missing_minute_days = integrity["incomplete_days"]
        excluded_days = set((context.sampling or {}).get("statistics_excluded_days") or [])
        unexpected_missing = set(missing_minute_days) - excluded_days

        status = "completed"
        if not context.coverage_complete:
            status = "insufficient_history"
            context.warnings.append(
                f"请求{context.requested_window_days}个交易日，交易日历仅覆盖{len(context.days)}个",
            )
        elif context.adjusted_daily.empty or unexpected_missing:
            status = "data_not_ready"
            if context.adjusted_daily.empty:
                context.warnings.append("窗口缺少可信日线历史；依赖日线的指标不可评价")
            if unexpected_missing:
                preview = "、".join(sorted(unexpected_missing)[:5])
                context.warnings.append(
                    f"{len(unexpected_missing)}个未排除日期未满足240槽有效OHLC/校准要求：{preview}",
                )
        elif context.minute_kind != "1m":
            # A v5 sample may contain only verified daily bars. Keep the ETF
            # and its daily metrics; minute-only metrics remain individually
            # null with their own sample diagnostics.
            status = "completed"
        filter_reasons: list[str] = []
        if status != "completed":
            filter_reasons.append(f"status_{status}")
        if metrics.get("ou_half_life_days") is None:
            filter_reasons.append("ou_invalid")
        if metrics.get("variance_ratio_240m") is None:
            filter_reasons.append("variance_ratio_240m_unavailable")
        elif float(metrics["variance_ratio_240m"]) > 1.10 + 1e-12:
            filter_reasons.append("variance_ratio_240m_gt_1.10")
        if metrics.get("mean_raw_price") is None:
            filter_reasons.append("mean_raw_price_unavailable")
        elif float(metrics["mean_raw_price"]) < 0.5 - 1e-12:
            filter_reasons.append("mean_raw_price_lt_0.5")
        eligible = status == "completed" and not filter_reasons
        source_quality = _source_quality(
            context.raw_bars, context.days, context.adjusted_daily,
        )
        source_quality["volume_verified"] = data.metadata.get("etf_volume_verified")
        source_quality["volume_unverified_days"] = data.metadata.get("volume_unverified_days", [])

        diagnostics = {
            "as_of": stamp.date().isoformat(),
            "window_start": context.days[0].date().isoformat() if len(context.days) else None,
            "window_end": context.days[-1].date().isoformat() if len(context.days) else None,
            "days": int(len(context.days)),
            "requested_window_days": context.requested_window_days,
            "window_complete": context.coverage_complete,
            "missing_window_days": int(max(0, context.requested_window_days - len(context.days))),
            "minute_days_covered": int(len(complete_minute_days)),
            "minute_missing_days": missing_minute_days,
            "sampling": context.sampling,
            "minute_integrity": {
                "expected_slots": integrity["expected_slots"],
                "accepted_slots": integrity["accepted_slots"],
                "placeholder_slots": integrity["placeholder_slots"],
                "unverified_slots": integrity["unverified_slots"],
                "invalid_ohlc_slots": integrity["invalid_ohlc_slots"],
                "unaccepted_slots": integrity["missing_slots"],
                "incomplete_days": integrity["failures"],
            },
            "sampling_frequencies": {
                "rv": "1m", "path_redundancy": "5m", "path_redundancy_comparison": "1m", "volume_profile": "1m OHLC4",
                "variance_ratio": "5m", "zero_crossing": "5m", "jump_ratio": "5m",
                "svr": "5m", "realized_moments": "5m", "ou": "30m",
                "atr": "30m", "tick_to_price": "20-day raw 1m close", "mddd": "1d",
            },
            "source_quality": source_quality,
            "path_redundancy": {
                "window_days": [1, 5, 20], "sampling": "5m", "comparison_sampling": "1m",
                "averaging": "equal_weight_valid_window_ratios", "rolling_step_days": 1,
                "includes_overnight": True, "daily_includes_overnight": False,
                "start_price": "first_sampled_bar_open",
                "missing_day_policy": "exclude_whole_window", "zero_path_policy": "exclude_window",
                "gss_weights": {"path_redundancy_daily_mean": 0.10, "path_redundancy_5d_mean": 0.10},
            },
            "profile": {
                "bins": PROFILE_BINS, "price_basis": "adjusted OHLC4",
                "price_proxy": "1m OHLC4 arithmetic proxy; not traded VWAP",
                "price_quality": "ohlc4_approximation",
                "amount_source": source_quality["turnover_source"],
                "vwap_quality": source_quality["vwap_quality"],
                "weight": "split-adjusted volume", **profile,
            },
            "ou": ou_details,
            "atr30": {
                "mean_tr_ratio": _finite(atr), "bar_count": atr_count,
                "day_count": atr_days, "window_days": min(60, len(context.days)),
                "method": "最近60日内1m聚合成30m；前收取同连续时钟上一根，午休首根取上午末根，日首根取前置日线收盘，真实缺失后无相邻前收则跳过；TR按前收取价格归一后求均值",
            },
            "mddd": mddd_details,
            "mddd_history_verified": mddd_verified,
            "svr": {
                "sample_count": svr_count,
                "positive_semivariance_over_negative": _finite(svr),
            },
            "realized_moments": {
                "day_count": statistic_days, "skewness": _finite(skew),
                "excess_kurtosis": _finite(kurt), "reason": moment_error,
            },
            "actions_error": None,
        }
        for key in ("mddd_censored", "mddd_window_start", "mddd_window_end", "mddd_recovery_date"):
            if key in mddd_details:
                diagnostics[key] = mddd_details[key]
        return {
            "status": status, "metrics": metrics, "metric_details": details,
            "grid_basis": grid_basis,
            "market_state": market_state,
            "actions_applied": [_safe_action(item) for item in context.factors],
            "eligible": bool(eligible), "filter_reasons": filter_reasons,
            "warnings": list(dict.fromkeys(context.warnings)), "diagnostics": diagnostics,
        }
    except _StatsError as exc:
        try:
            tick = _price_tick(data)
        except Exception:
            tick = None
        return _empty_result(
            exc.status, reason=exc.message, price_tick=tick, market_state=market_state,
        )
    except Exception as exc:
        try:
            tick = _price_tick(data)
        except Exception:
            tick = None
        return _empty_result(
            "unavailable", reason=f"ETF统计计算失败：{type(exc).__name__}: {exc}",
            price_tick=tick, market_state=market_state,
        )


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks, assigning exact ties a shared rank."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and math.isclose(
            values[order[end]], values[order[position]],
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            end += 1
        average = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average
        position = end
    return ranks


def score_etfs(rows: list[dict]) -> list[dict]:
    """Write cross-sectional GSS fields into ``rows`` (completed rows only).

    Eligibility is independent of scoring: a completed row with all seven
    statistical inputs receives a score even if a hard filter makes it
    ineligible. A one-row cohort has no meaningful percentile and gets null.
    """
    score_specs = (
        ("rv_annualized", 0.25, 1.0),
        ("path_redundancy_daily_mean", 0.10, 1.0),
        ("path_redundancy_5d_mean", 0.10, 1.0),
        ("variance_ratio_240m", 0.20, -1.0),
        ("iovr", 0.15, 1.0),
        ("zero_crossing_rate", 0.10, 1.0),
        ("jump_ratio", 0.10, -1.0),
    )
    cohort: list[tuple[int, dict[str, float]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if (row.get("sampling") or {}).get("comparable") is False:
            continue
        status = row.get("status")
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else row
        values: dict[str, float] = {}
        for key, _, _ in score_specs:
            value = _finite(metrics.get(key))
            if value is not None:
                values[key] = value
        if status == "completed" and len(values) == len(score_specs):
            cohort.append((index, values))

    n = len(cohort)
    for row in rows:
        if isinstance(row, dict):
            row["gss_score"] = None
            row["score_contributions"] = {}
            row["rank"] = None
            row["score_cohort_size"] = n
    if n < 2:
        return rows

    percentiles: dict[str, list[float]] = {}
    for key, _, sign in score_specs:
        ranks = _average_ranks([sign * item[1][key] for item in cohort])
        percentiles[key] = [(rank - 1.0) / (n - 1.0) for rank in ranks]

    scored = []
    for cohort_index, (row_index, _) in enumerate(cohort):
        raw_contributions = {
            key: 100.0 * weight * percentiles[key][cohort_index]
            for key, weight, _ in score_specs
        }
        contributions = {key: float(value) for key, value in raw_contributions.items()}
        gss = float(sum(raw_contributions.values()))
        row = rows[row_index]
        row["gss_score"] = gss
        row["score_contributions"] = contributions
        scored.append((gss, str(row.get("symbol", "")), row_index, row))
    ordered = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
    for rank, (_, _, _, row) in enumerate(ordered, start=1):
        row["rank"] = rank
    return rows


def _candidate_for_steps(
    *, base_step: float, buy_step: float, sell_step: float,
    reference: float, lower: float, upper: float, current: float, tick: float,
) -> dict[str, Any] | None:
    if not (0.0 < buy_step <= 0.05 + 1e-12 and 0.0 < sell_step <= 0.05 + 1e-12):
        return None
    mid = rules.round_to_tick(reference, tick)
    lower_tick = rules.round_to_tick(lower, tick)
    upper_tick = rules.round_to_tick(upper, tick)
    if not (
        math.isfinite(mid) and math.isfinite(lower_tick) and math.isfinite(upper_tick)
        and 0 < lower_tick < mid < upper_tick
    ):
        return None
    if not lower_tick < current < upper_tick:
        return None
    try:
        theoretical_down = int(math.floor(math.log(lower_tick / mid) / math.log1p(-buy_step) + 1e-12))
        theoretical_up = int(math.floor(math.log(upper_tick / mid) / math.log1p(sell_step) + 1e-12))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if theoretical_down < 1 or theoretical_up < 1:
        return None

    actual_down: list[float] = []
    previous = mid
    for k in range(1, theoretical_down + 1):
        line = rules.round_to_tick(mid * (1.0 - buy_step) ** k, tick)
        if not math.isfinite(line) or line < lower_tick or line >= previous - 1e-12:
            return None
        actual_down.append(line)
        previous = line
    actual_up: list[float] = []
    previous = mid
    for k in range(1, theoretical_up + 1):
        line = rules.round_to_tick(mid * (1.0 + sell_step) ** k, tick)
        if not math.isfinite(line) or line > upper_tick or line <= previous + 1e-12:
            return None
        actual_up.append(line)
        previous = line
    if (
        not actual_down or not actual_up
        or len(actual_down) != theoretical_down
        or len(actual_up) != theoretical_up
    ):
        return None
    if any(not lower_tick <= line < mid for line in actual_down):
        return None
    if any(not mid < line <= upper_tick for line in actual_up):
        return None
    return {
        "base_step": float(base_step),
        "buy_step": float(buy_step),
        "sell_step": float(sell_step),
        "reference_price": _finite(mid),
        "lower_price": _finite(lower_tick),
        "upper_price": _finite(upper_tick),
        "grids_down_count": len(actual_down),
        "grids_up_count": len(actual_up),
        "asymmetric": not math.isclose(
            buy_step, sell_step, rel_tol=1e-12, abs_tol=1e-12,
        ),
    }


def generate_candidates(basis: dict) -> list[dict]:
    """Generate tick-safe symmetric and directionally adjusted candidates."""
    if not isinstance(basis, Mapping):
        return []
    source = basis.get("grid_basis") if isinstance(basis.get("grid_basis"), Mapping) else basis
    reference = _positive(source.get("reference_price"))
    lower = _positive(source.get("lower_price"))
    upper = _positive(source.get("upper_price"))
    current = _positive(source.get("current_raw_price"))
    tick = _positive(source.get("price_tick")) or 0.001
    atr = _finite(source.get("atr30_pct"))
    metric_values = basis.get("metrics") if isinstance(basis.get("metrics"), Mapping) else {}
    mean_raw = (
        _positive(source.get("mean_raw_price"))
        or _positive(metric_values.get("mean_raw_price"))
        or current
    )
    if reference is None or lower is None or upper is None or current is None:
        return []
    if not (lower < current < upper and lower < reference < upper):
        return []
    if mean_raw is None or mean_raw <= 0 or atr is None or atr < 0:
        return []
    minimum = max(1.2 * atr, 5.0 * tick / mean_raw)
    if not math.isfinite(minimum) or minimum <= 0:
        return []
    first_units = max(1, int(math.ceil(minimum * 1000.0 - 1e-12)))
    if first_units > 50:
        return []

    result: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()

    def append(base_step: float, buy_step: float, sell_step: float) -> bool:
        key = (round(buy_step, 12), round(sell_step, 12))
        if key in seen or len(result) >= 100:
            return False
        if buy_step + 1e-12 < minimum or sell_step + 1e-12 < minimum:
            return False
        candidate = _candidate_for_steps(
            base_step=base_step, buy_step=buy_step, sell_step=sell_step,
            reference=reference, lower=lower, upper=upper, current=current, tick=tick,
        )
        if candidate is not None:
            seen.add(key)
            result.append(candidate)
            return True
        return False

    bases = [units / 1000.0 for units in range(first_units, 51)]
    symmetric_bases = [base for base in bases if append(base, base, base)]

    svr = _finite(source.get("svr"))
    skew = _finite(source.get("realized_skewness"))
    direction = 0
    ratio = None
    if svr is not None and svr > 0 and skew is not None:
        if skew > 0.20 and svr > 1.10:
            direction = 1
            ratio = min(svr, 1.60)
        elif skew < -0.20 and svr < 0.90:
            direction = -1
            ratio = min(1.0 / svr, 1.60)
    if direction and ratio is not None and ratio > 1.0 + 1e-12:
        scale = math.sqrt(ratio)
        for base in symmetric_bases:
            buy = base / scale if direction > 0 else base * scale
            sell = base * scale if direction > 0 else base / scale
            append(base, buy, sell)
            if len(result) >= 100:
                break
    return result


def generate_candidates_with_diagnostics(basis: dict) -> dict[str, Any]:
    """Generate candidates and retain the first concrete reason when none fit."""
    if not isinstance(basis, Mapping):
        return {"candidates": [], "reason_code": "grid_basis_missing", "constraints": {}}
    source = basis.get("grid_basis") if isinstance(basis.get("grid_basis"), Mapping) else basis
    reference = _positive(source.get("reference_price"))
    lower = _positive(source.get("lower_price"))
    upper = _positive(source.get("upper_price"))
    current = _positive(source.get("current_raw_price"))
    tick = _positive(source.get("price_tick")) or 0.001
    atr = _finite(source.get("atr30_pct"))
    mean = _positive(source.get("mean_raw_price")) or current
    reason_code = None
    if None in (reference, lower, upper, current):
        reason_code = "grid_basis_missing"
    elif current <= lower:
        reason_code = "current_at_or_below_lower"
    elif current >= upper:
        reason_code = "current_at_or_above_upper"
    elif reference <= lower or reference >= upper:
        reason_code = "reference_outside_band"
    elif atr is None or atr < 0 or mean is None:
        reason_code = "step_inputs_invalid"
    elif max(1.2 * atr, 5.0 * tick / mean) > 0.05 + 1e-12:
        reason_code = "minimum_step_exceeds_maximum"
    candidates = generate_candidates(basis)
    if candidates:
        return {"candidates": candidates, "reason_code": None, "constraints": {}}
    if reason_code is None:
        reason_code = "tick_grid_invalid"
    return {
        "candidates": [], "reason_code": reason_code,
        "constraints": {
            "reference_price": reference, "lower_price": lower,
            "upper_price": upper, "current_price": current,
            "current_price_date": source.get("current_price_date"),
            "current_price_source": source.get("current_price_source"),
            "current_price_basis": source.get("current_price_basis", "as_of_adjusted"),
            "current_raw_price_unadjusted": _finite(source.get("current_raw_price_unadjusted")),
            "price_tick": tick, "atr30_pct": atr,
            "minimum_step": max(1.2 * atr, 5.0 * tick / mean) if atr is not None and mean else None,
            "maximum_step": 0.05,
            "inequalities": ["lower < current_price < upper", "lower < reference_price < upper"],
            "price_basis": source.get("price_basis", "as_of_adjusted"),
        },
    }


__all__ = [
    "compute_etf_statistics",
    "score_etfs",
    "generate_candidates",
    "generate_candidates_with_diagnostics",
]
