from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRICE_VALIDATION_VERSION = 1
OHLC_FIELDS = ("open", "high", "low", "close")


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[field], errors="coerce")


def price_issue_details(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one explanatory record for every invalid OHLCV/preclose row.

    A zero open is a source defect, not a trade that can be repaired from the
    close. Missing volume is invalid; a finite zero volume is valid and is
    deliberately not interpreted as a suspended trading day here.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["quality_reason"])

    result = frame.copy()
    reasons: list[list[str]] = [[] for _ in range(len(result))]
    positions = np.arange(len(result))

    prices = pd.DataFrame(
        {field: _numeric(result, field) for field in OHLC_FIELDS}, index=result.index
    )
    for field in OHLC_FIELDS:
        values = prices[field].to_numpy(dtype=float)
        bad = ~np.isfinite(values) | (values <= 0)
        for row_number in positions[bad]:
            reasons[int(row_number)].append(f"{field} 缺失、非有限或非正")

    price_values = prices.to_numpy(dtype=float)
    price_finite = np.isfinite(price_values).all(axis=1)
    for row_number in positions[price_finite]:
        opening, high, low, close = (float(value) for value in price_values[row_number])
        if high < low - 1e-9:
            reasons[int(row_number)].append("最高价低于最低价")
        if high < max(opening, close, low) - 1e-9:
            reasons[int(row_number)].append("最高价低于开盘价、收盘价或最低价")
        if low > min(opening, close, high) + 1e-9:
            reasons[int(row_number)].append("最低价高于开盘价、收盘价或最高价")

    volume = _numeric(result, "volume").to_numpy(dtype=float)
    for row_number in positions[~np.isfinite(volume) | (volume < 0)]:
        reasons[int(row_number)].append("volume 缺失、非有限或为负")

    if "turnover" in result:
        turnover = _numeric(result, "turnover")
        values = turnover.to_numpy(dtype=float)
        invalid = turnover.notna().to_numpy() & (~np.isfinite(values) | (values < 0))
        for row_number in positions[invalid]:
            reasons[int(row_number)].append("turnover 非有限或为负")

    if "preclose" in result:
        preclose = _numeric(result, "preclose")
        values = preclose.to_numpy(dtype=float)
        invalid = preclose.notna().to_numpy() & (~np.isfinite(values) | (values <= 0))
        for row_number in positions[invalid]:
            reasons[int(row_number)].append("preclose 缺失、非有限或非正")

    failed = np.array([bool(items) for items in reasons], dtype=bool)
    if not failed.any():
        return pd.DataFrame(columns=["quality_reason"])
    details = result.loc[failed].copy()
    kept_reasons = [items for items, keep in zip(reasons, failed) if keep]
    details["quality_fields"] = ["、".join(items) for items in kept_reasons]
    details["quality_reason"] = ["; ".join(items) for items in kept_reasons]
    return details


def split_invalid_prices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return usable rows plus an explanatory anomaly frame."""
    details = price_issue_details(frame)
    if details.empty:
        return frame, details
    return frame.drop(index=details.index).copy(), details


def details_to_rows(details: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert invalid rows into attrs-safe values (DataFrames are not safe in pandas attrs)."""
    if details is None or details.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in details.iterrows():
        stamp = row.get("timestamp") if "timestamp" in row else row.get("date")
        source = row.get("source")
        rows.append({
            "timestamp": pd.Timestamp(stamp).isoformat() if pd.notna(stamp) else None,
            "source": None if pd.isna(source) else str(source),
            "quality_fields": str(row.get("quality_fields") or "OHLC"),
            "quality_reason": str(row.get("quality_reason") or "价格无效"),
        })
    return rows


def anomalies_from_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.attrs.get("price_anomaly_rows") if frame is not None else None
    if not rows:
        return pd.DataFrame(columns=["timestamp", "source", "quality_fields", "quality_reason"])
    result = pd.DataFrame(rows)
    if "timestamp" in result:
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    return result


def validate_price_frame(frame: pd.DataFrame, *, context: str = "行情") -> pd.DataFrame:
    """Raise on any invalid price row instead of silently substituting a price."""
    details = price_issue_details(frame)
    if details.empty:
        return frame
    sample = details.head(8)
    labels = []
    for _, row in sample.iterrows():
        stamp = row.get("timestamp") if "timestamp" in row else row.get("date")
        labels.append(f"{stamp}: {row['quality_reason']}")
    suffix = "" if len(details) <= len(sample) else f"；另有 {len(details) - len(sample)} 条"
    raise ValueError(f"{context}包含 {len(details)} 条无效价格：" + "；".join(labels) + suffix)


def anomaly_records(
    anomalies: pd.DataFrame | None,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    version: Any = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Serialize anomaly rows without NaN/inf values leaking into JSON."""
    if anomalies is None or anomalies.empty:
        return []
    output: list[dict[str, Any]] = []
    for _, row in anomalies.head(max(0, int(limit))).iterrows():
        stamp = row.get("timestamp") if "timestamp" in row else row.get("date")
        source = row.get("source")
        output.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
            "timestamp": pd.Timestamp(stamp).isoformat() if pd.notna(stamp) else None,
            "source": None if pd.isna(source) else str(source),
            "fields": str(row.get("quality_fields") or "OHLC"),
            "reason": str(row.get("quality_reason") or "价格无效"),
        })
    return output


def anomaly_summary(
    frame: pd.DataFrame,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    version: Any = None,
    sample_limit: int = 200,
) -> dict[str, Any]:
    anomalies = anomalies_from_attrs(frame)
    if anomalies.empty:
        return {
            "rule_version": PRICE_VALIDATION_VERSION,
            "status": "passed",
            "count": 0,
            "days": 0,
            "first_time": None,
            "last_time": None,
            "samples": [],
        }
    stamps = pd.to_datetime(anomalies["timestamp"], errors="coerce")
    valid_stamps = stamps.dropna()
    return {
        "rule_version": PRICE_VALIDATION_VERSION,
        "status": "failed",
        "count": int(len(anomalies)),
        "days": int(stamps.dt.normalize().nunique()),
        "first_time": valid_stamps.min().isoformat() if len(valid_stamps) else None,
        "last_time": valid_stamps.max().isoformat() if len(valid_stamps) else None,
        "samples": anomaly_records(
            anomalies, symbol=symbol, timeframe=timeframe, version=version, limit=sample_limit
        ),
    }


def anomalies_in_range(frame: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    anomalies = anomalies_from_attrs(frame)
    if anomalies.empty:
        return pd.DataFrame()
    result = anomalies.copy()
    stamps = pd.to_datetime(result["timestamp"], errors="coerce")
    mask = stamps.notna()
    if start is not None:
        mask &= stamps >= pd.Timestamp(start)
    if end is not None:
        mask &= stamps < pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    return result.loc[mask].copy()
