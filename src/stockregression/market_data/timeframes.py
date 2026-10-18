from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Timeframe = Literal[
    "1min", "5min", "15min", "30min", "60min",
    "1day", "1week", "1month",
]

TIMEFRAMES: tuple[str, ...] = (
    "1min", "5min", "15min", "30min", "60min", "1day", "1week", "1month",
)

ALIASES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "1d": "1day",
    "daily": "1day",
    "1w": "1week",
    "weekly": "1week",
    "1mo": "1month",
    "monthly": "1month",
}

MINUTE_TIMEFRAMES = {"1min", "5min", "15min", "30min", "60min"}
LEGACY_TIMEFRAMES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min", "1d": "1day"}


@dataclass(frozen=True)
class TimeframeSpec:
    name: str
    label: str
    minutes: int | None
    kind: str


SPECS = {
    "1min": TimeframeSpec("1min", "1 分钟", 1, "minute"),
    "5min": TimeframeSpec("5min", "5 分钟", 5, "minute"),
    "15min": TimeframeSpec("15min", "15 分钟", 15, "minute"),
    "30min": TimeframeSpec("30min", "30 分钟", 30, "minute"),
    "60min": TimeframeSpec("60min", "60 分钟", 60, "minute"),
    "1day": TimeframeSpec("1day", "日线", None, "day"),
    "1week": TimeframeSpec("1week", "周线", None, "week"),
    "1month": TimeframeSpec("1month", "月线", None, "month"),
}


def normalize_timeframe(value: str | None, *, default: str | None = None) -> str:
    if value is None or not str(value).strip():
        if default is None:
            raise ValueError("必须明确指定数据周期")
        value = default
    normalized = ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if normalized not in TIMEFRAMES:
        raise ValueError(f"不支持的数据周期：{value}")
    return normalized


def legacy_timeframe(value: str) -> str:
    return LEGACY_TIMEFRAMES.get(value, value)


def timeframe_spec(value: str) -> TimeframeSpec:
    return SPECS[normalize_timeframe(value)]


def public_timeframe(value: str) -> str:
    """Return the API/CLI spelling used by established endpoints."""
    return {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "60min": "60m", "1day": "1d"}.get(
        normalize_timeframe(value), normalize_timeframe(value)
    )


def timeframe_from_legacy_frequency(value: str) -> str:
    return normalize_timeframe(value)
