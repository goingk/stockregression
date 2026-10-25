"""Small date-control projection; never reads minute prices or prepares data."""
from __future__ import annotations

import pandas as pd

from .market_data import MarketDataService
from .market_data.timeframes import normalize_timeframe
from .models import normalize_symbol


def workbench_calendar(store, symbol: str, frequency: str, data_version: str | None = None) -> dict:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(frequency)
    source = "1day" if timeframe == "1day" else "1min"
    market = MarketDataService(store)
    repository = market.repository
    manifest = store.snapshot(symbol, data_version, persist=False, repository=repository)
    if data_version and manifest is None:
        raise LookupError("指定行情版本不可读取，不能改用当前数据")
    info = ((manifest or {}).get("market_versions") or {}).get(source) or {}
    # The catalog row contains bounds; dataset() would also walk coverage and
    # validation metadata, which is unnecessary for a date selector.
    if not data_version:
        with repository._connect() as db:
            row = repository._dataset_row(db, symbol, source, "none")
        if row:
            info = {"first": row["first_time"], "last": row["last_time"],
                    "version": row["current_version"], "rows": row["row_count"]}
    first, last = info.get("first"), info.get("last")
    dates = pd.DatetimeIndex([])
    calendar_version = None
    daily_reference = {}
    if manifest and manifest.get("market_data"):
        calendar_version = (manifest.get("auxiliary_versions") or {}).get("calendar")
        daily_reference = (manifest.get("market_versions") or {}).get("1day") or {}
    elif manifest:
        calendar = store.read_files((manifest.get("files") or {}).get("calendar", []))
        if not calendar.empty and "date" in calendar:
            dates = pd.DatetimeIndex(pd.to_datetime(calendar["date"], errors="coerce")).dropna()
        first = first or manifest.get("first_date") or manifest.get("start_date")
        last = last or manifest.get("last_date") or manifest.get("end_date")
    first_text = str(pd.Timestamp(first).date()) if first else None
    last_text = str(pd.Timestamp(last).date()) if last else None
    if manifest and manifest.get("market_data") and first_text and last_text:
        daily_version = (
            int(daily_reference["version"])
            if daily_reference.get("version") is not None else None
        )
        # Incremental auxiliary calendars can contain only the latest sync
        # tail. Recover the full local trading clock from the frozen daily date
        # index without reading minute/OHLC bodies.
        dates = repository.trading_days(
            symbol, first_text, last_text,
            calendar_version=calendar_version, daily_version=daily_version,
        )
    dates = dates.normalize().unique().sort_values()
    if first_text and last_text:
        dates = dates[(dates >= pd.Timestamp(first_text)) & (dates <= pd.Timestamp(last_text))]
    else:
        dates = pd.DatetimeIndex([])
    return {
        "symbol": symbol, "frequency": frequency, "source_timeframe": source,
        "data_version": (manifest or {}).get("version"),
        "first_date": first_text, "last_date": last_text,
        "calendar_available": bool(len(dates)),
        "trading_dates": [str(day.date()) for day in dates],
        "message": "" if first_text and last_text else "该周期没有本地数据，请先准备数据。",
    }
