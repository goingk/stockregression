from __future__ import annotations

import pandas as pd

from .data import DataService
from .models import BacktestConfig, DataBundle
from .storage import Store


_RESEARCH_LOOKBACK_TRADING_DAYS = 504
_CALENDAR_DAYS_PER_TRADING_DAY = 1.6


def preparation_config(store: Store, base: BacktestConfig, version: str | None = None) -> BacktestConfig:
    """Return a copy whose start includes the selection lookback.

    The user-selected start/end stay on ``base`` for the out-of-sample period.
    The local exchange calendar is preferred; a conservative calendar-day
    estimate gives automatic history sync a useful target when the local
    snapshot does not yet reach far enough back.
    """
    anchor = pd.Timestamp(base.start_date).normalize()
    if base.selection_asof is not None:
        anchor = min(anchor, pd.Timestamp(base.selection_asof).normalize())

    manifest = store.snapshot(base.symbol, version)
    dates = pd.DatetimeIndex([])
    if manifest:
        calendar = store.read_files(manifest["files"].get("calendar", []))
        if not calendar.empty and "date" in calendar:
            dates = pd.DatetimeIndex(pd.to_datetime(calendar["date"], errors="coerce").dropna().dt.normalize().unique()).sort_values()
    first_selection_day = int(dates.searchsorted(anchor, side="left")) if len(dates) else 0
    if first_selection_day >= _RESEARCH_LOOKBACK_TRADING_DAYS + 1:
        start = dates[first_selection_day - _RESEARCH_LOOKBACK_TRADING_DAYS - 1]
    else:
        calendar_days = int(_RESEARCH_LOOKBACK_TRADING_DAYS * _CALENDAR_DAYS_PER_TRADING_DAY)
        start = anchor - pd.Timedelta(days=calendar_days)
    return base.model_copy(update={"start_date": start.date()})


def prepare_research_bundle(
    service: DataService,
    base: BacktestConfig,
    version: str | None,
    progress=lambda *_: None,
    cancelled=lambda: False,
) -> DataBundle:
    """Prepare selection lookback and evaluation data from one immutable snapshot."""
    store = service.store
    request = preparation_config(store, base, version)
    if cancelled():
        raise InterruptedError("任务已取消")
    try:
        bundle = service.ensure_bundle(request, version, progress, cancelled)
    except InterruptedError:
        raise
    except Exception:
        # A frozen or partially available history is still useful evidence. Try
        # to load all usable local history so the selector can report a weak
        # sample with the exact same selected OOS dates.
        manifest = store.snapshot(base.symbol, version)
        if not manifest:
            raise
        if manifest.get("market_data"):
            fallback = base.model_copy(update={"start_date": request.start_date, "end_date": base.end_date})
            bundle = service.load_bundle(fallback, manifest["version"])
            message = "自动 504 日准备未完成；已使用锁定快照中可用历史，证据强度可能较弱。"
            bundle.warnings = list(dict.fromkeys([*(bundle.warnings or []), message]))
            bundle.metadata = {
                **bundle.metadata,
                "warnings": list(dict.fromkeys([*bundle.metadata.get("warnings", []), message])),
            }
        else:
            daily = store.read_files(manifest["files"].get("daily", []))
            if daily.empty or "date" not in daily:
                raise
            valid = daily.copy()
            if "tradestatus" in valid:
                valid = valid[valid.tradestatus == 1]
            if "volume" in valid:
                valid = valid[valid.volume > 0]
            valid_dates = pd.DatetimeIndex(pd.to_datetime(valid.date, errors="coerce").dropna().dt.normalize().unique()).sort_values()
            all_dates = pd.DatetimeIndex(pd.to_datetime(daily.date, errors="coerce").dropna().dt.normalize().unique()).sort_values()
            available_start = valid_dates[1] if len(valid_dates) > 1 else all_dates[1] if len(all_dates) > 1 else all_dates[0]
            latest_available = all_dates.max()
            if request.start_date < available_start.date() or request.start_date > latest_available.date():
                start = available_start.date()
            else:
                start = request.start_date
            end = base.end_date
            if end is not None and end < start:
                start = available_start.date()
            fallback = base.model_copy(update={"start_date": start, "end_date": end})
            bundle = service.load_bundle(fallback, manifest["version"])
            message = "自动补齐 504 个前置交易日未完成；已使用锁定快照中可用历史，证据强度可能较弱。"
            bundle.warnings = list(dict.fromkeys([*(bundle.warnings or []), message]))
            bundle.metadata = {
                **bundle.metadata,
                "warnings": list(dict.fromkeys([*bundle.metadata.get("warnings", []), message])),
            }

    if cancelled():
        raise InterruptedError("任务已取消")

    # Funding derives its reference from dates before base.start_date. Restrict
    # this convenience map to the prepared bundle so later snapshot rows cannot
    # accidentally become an input to early selection decisions.
    bundle.metadata = dict(bundle.metadata)
    prepared_days = pd.DatetimeIndex(pd.to_datetime(bundle.calendar)).normalize() if len(bundle.calendar) else pd.DatetimeIndex([])
    lower = prepared_days.min() if len(prepared_days) else pd.Timestamp(request.start_date)
    upper = pd.Timestamp(base.end_date) if base.end_date is not None else (prepared_days.max() if len(prepared_days) else pd.Timestamp.max.normalize())
    prior_closes = bundle.metadata.get("prior_closes")
    if isinstance(prior_closes, dict):
        bundle.metadata["prior_closes"] = {
            str(day): price for day, price in prior_closes.items()
            if lower <= pd.Timestamp(day).normalize() <= upper
        }
    bundle.metadata["research_preparation"] = {
        "requested_lookback_trading_days": _RESEARCH_LOOKBACK_TRADING_DAYS,
        "evaluation_start": str(base.start_date),
        "evaluation_end": str(base.end_date) if base.end_date else None,
        "prepared_start": str(lower.date()),
        "data_version": bundle.version,
    }
    return bundle
