from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import BacktestConfig, DataBundle
from ..quality import (
    DataQualityError, _closing_auction_placeholders, apply_price_limits,
    minute_slots, prepare_bundle,
)
from ..price_validation import anomaly_records, anomalies_in_range, price_issue_details
from .models import DataNotReady
from .timeframes import MINUTE_TIMEFRAMES, normalize_timeframe, timeframe_spec


def load_native_bundle(
    service, config: BacktestConfig, version: str | None = None, *,
    statistics_only: bool = False,
    summary_only: bool = False,
) -> DataBundle:
    """Read an immutable snapshot from the frequency databases.

    Native coarse-minute datasets are consumed directly.  The legacy
    one-minute path continues to use the established gap and auction rules.
    """
    manifest = service.store.snapshot(config.symbol, version)
    if not manifest or not manifest.get("market_data"):
        raise DataQualityError("该数据版本不是行情数据库快照")
    if manifest.get("data_resolution") or getattr(config, "data_policy", None):
        manifest = service.resolution_manifest(config, version, manifest=manifest)
        return _load_resolved_bundle(service, config, manifest,
            statistics_only=statistics_only, summary_only=summary_only)
    versions = manifest.get("market_versions", {})
    if "1day" not in versions:
        raise DataQualityError("该版本没有日线，无法进行回测")
    daily_version = int(versions["1day"]["version"])
    _reject_reported_price_anomalies(
        service.repository, config.symbol, "1day", daily_version,
        start=config.start_date, end=config.end_date, context="日线",
    )
    daily = service.repository.bars(config.symbol, "1day", version=daily_version)
    if daily.empty:
        raise DataQualityError("本地版本没有日线")
    _reject_invalid_prices(
        daily, config.symbol, "1day", daily_version,
        start=config.start_date, end=config.end_date, context="日线",
    )
    daily = daily.rename(columns={"timestamp": "date"})
    daily["date"] = pd.to_datetime(daily.date).dt.normalize()
    for field, default in (("preclose", np.nan), ("tradestatus", np.nan), ("is_st", pd.NA), ("turnover", np.nan)):
        if field not in daily:
            daily[field] = default
    if daily[['open', 'high', 'low', 'close', 'volume', 'preclose', 'tradestatus']].isna().any().any():
        raise DataQualityError("日线缺少 preclose、交易状态或有效 OHLCV，当前版本不能用于严格回测")
    daily["tradestatus"] = pd.to_numeric(daily.tradestatus, errors="coerce").fillna(-1).astype(int)
    asset_type = str((manifest.get("metadata") or {}).get("asset_type") or "")
    if daily.is_st.isna().any():
        if asset_type == "etf":
            daily["is_st"] = daily.is_st.fillna(False)
        else:
            raise DataQualityError("日线缺少 ST 历史状态，当前版本不能用于严格回测")
    daily["is_st"] = daily.is_st.map(bool)
    daily = daily.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    auxiliary_versions = manifest.get("auxiliary_versions", {})
    calendar = _calendar(service, config.symbol, daily, auxiliary_versions.get("calendar"))
    daily = apply_price_limits(daily, calendar, manifest.get("metadata", {}))
    actions = _actions(service, config.symbol, auxiliary_versions.get("actions"))
    end = pd.Timestamp(config.end_date) if config.end_date else daily.date.max()
    end = min(end, daily.date.max())
    start = pd.Timestamp(config.start_date)
    if not statistics_only:
        from ..corporate_actions import action_issues
        issues = action_issues(
            actions, daily, start, end, settlement=True,
            price_tick=float((manifest.get("metadata") or {}).get("price_tick", 0.01)),
        )
        if issues:
            raise DataQualityError("公司行动资料无法用于结算：" + "；".join(issues[:8]))
    if end < start:
        raise DataQualityError("所选区间没有本地已收盘数据")
    target = normalize_timeframe(config.frequency)
    if target != "1day" and (target not in MINUTE_TIMEFRAMES or target not in versions):
        raise DataQualityError(f"当前版本没有 {target} 行情")
    if summary_only:
        return _preflight_bundle(
            config, daily, actions, calendar, manifest, target, start, end,
        )
    if target == "1day":
        return prepare_bundle(
            config.symbol, pd.DataFrame(), daily, actions, calendar, start, end,
            "1d", manifest["version"], _metadata(manifest, daily, target), manifest.get("profiles", {}),
        )
    _reject_reported_price_anomalies(
        service.repository, config.symbol, target, int(versions[target]["version"]),
        start=start, end=end, context=target,
    )
    minute = service.repository.bars(
        config.symbol, target, start=start, end=end + pd.Timedelta(days=1),
        version=int(versions[target]["version"]),
    )
    _reject_invalid_prices(
        minute, config.symbol, target, int(versions[target]["version"]),
        start=start, end=end, context=f"{target}",
    )
    if minute.empty:
        raise DataQualityError(f"所选区间没有 {target} 行情")
    if target == "1min":
        if statistics_only:
            return _statistics_bundle(
                config, minute, daily, actions, calendar, manifest,
            )
        return prepare_bundle(
            config.symbol, minute, daily, actions, calendar, start, end,
            config.frequency, manifest["version"], _metadata(manifest, daily, target), manifest.get("profiles", {}),
        )
    return _native_coarse_bundle(
        config, target, minute, daily, actions, calendar, manifest
    )


def _load_resolved_bundle(service, config, manifest, *, statistics_only=False, summary_only=False):
    """Execute only the immutable, independently certified per-day inputs."""
    from ..data_resolution import POLICY, validate_data_resolution, resolution_sampling
    from ..snapshot_quality import frozen_quality_state

    plan = validate_data_resolution(manifest, config.start_date, config.end_date)
    if getattr(config, "data_policy", None) not in (None, POLICY):
        raise DataQualityError("DATA_RESOLUTION_MISMATCH: 请求策略与冻结版本不一致")
    target = normalize_timeframe(config.frequency)
    if target != "1min":
        raise DataQualityError("逐日分钟／日线策略只适用于 1 分钟回放")
    versions = manifest.get("market_versions", {})
    daily_version = int(versions["1day"]["version"])
    daily = service.repository.bars(config.symbol, "1day", version=daily_version)
    if daily.empty:
        raise DataQualityError("没有可信日线可供初始化持仓或核实交易状态")
    daily = daily.rename(columns={"timestamp": "date"}).copy()
    daily["date"] = pd.to_datetime(daily.date).dt.normalize()
    daily = daily.sort_values("date").drop_duplicates("date", keep="last")
    modes = {item["date"]: item["mode"] for item in plan["days"]}
    accepted = {day for day, mode in modes.items() if mode != "skip"}
    minute_days = {day for day, mode in modes.items() if mode == "minute"}
    # Historical rows are useful for initial valuation and daily diagnostics,
    # but a current live ledger must never bless this frozen replay.
    prior = set(plan.get("prior_daily_days") or []) | set(plan.get("history_daily_days") or [])
    history = {day.date().isoformat() for day in daily.date
               if day < pd.Timestamp(plan["start"])
               and (day.date().isoformat() in prior
                    or frozen_quality_state(manifest, "1day", day, day))}
    daily = daily[daily.date.dt.date.astype(str).isin(accepted | history)].copy()
    required = ["open", "high", "low", "close", "volume", "preclose", "tradestatus"]
    if any(field not in daily for field in required):
        raise DataQualityError("RESOLUTION_PLAN_MISMATCH: 已校验日线缺少必要回放字段")
    for field in required:
        daily[field] = pd.to_numeric(daily[field], errors="coerce")
    window = daily[daily.date.dt.date.astype(str).isin(accepted)]
    missing = accepted - set(window.date.dt.date.astype(str))
    invalid = price_issue_details(window)
    if (missing or not invalid.empty or window[required].isna().any().any()
            or (~window.tradestatus.isin([0, 1])).any()):
        raise DataQualityError("RESOLUTION_PLAN_MISMATCH: 冻结策略指定的日线与实际文件不一致")
    _reject_reported_price_anomalies(service.repository, config.symbol, "1day", daily_version,
        start=plan["start"], end=plan["end"], context="日线", accepted_days=accepted)
    daily["is_st"] = daily.get("is_st", pd.Series(False, index=daily.index)).fillna(False).astype(bool)
    daily["tradestatus"] = daily.tradestatus.astype(int)
    calendar = _calendar(service, config.symbol, daily, manifest.get("auxiliary_versions", {}).get("calendar"))
    days = pd.DatetimeIndex([pd.Timestamp(item["date"]) for item in plan["days"]])
    # Preserve every planned session, including missing boundary dates.
    calendar = pd.concat([calendar, pd.DataFrame({"date": days})], ignore_index=True).drop_duplicates("date").sort_values("date")
    actions_version = manifest.get("auxiliary_versions", {}).get("actions")
    if actions_version is None or service.repository.auxiliary(config.symbol, "actions", version=actions_version) is None:
        raise DataQualityError("冻结的公司行动资料不可用，不能确认持仓结算")
    actions = _actions(service, config.symbol, actions_version)
    from ..corporate_actions import action_issues
    issues = action_issues(actions, daily, pd.Timestamp(plan["start"]), pd.Timestamp(plan["end"]),
        settlement=not statistics_only, price_tick=float((manifest.get("metadata") or {}).get("price_tick", .001)))
    if issues:
        raise DataQualityError("公司行动资料无法用于该区间：" + "；".join(issues[:8]))
    daily = apply_price_limits(daily, calendar, manifest.get("metadata", {}))
    metadata = _metadata(manifest, daily, target)
    metadata["data_resolution"] = plan
    metadata["data_policy"] = POLICY
    metadata["etf_sampling"] = manifest.get("etf_sampling") or resolution_sampling(plan)
    metadata["valuation_as_of"] = plan.get("last_evaluable_date")
    reports = [{"date": item["date"], "mode": item["mode"], "reason": item["reason"],
                "reason_code": item["reason_code"]} for item in plan["days"]]
    counts = plan["counts"]
    warnings = []
    if counts["daily"] or counts["skip"]:
        warnings.append(f"分钟 {counts['minute']} 日，日线降级 {counts['daily']} 日，跳过 {counts['skip']} 日；日线不还原盘中路径")
    quality = {"days": reports, "minute_days": counts["minute"], "daily_days": counts["daily"],
        "skipped_days": counts["skip"], "skip_days": [item["date"] for item in reports if item["mode"] == "skip"],
        "suspended_days": counts["suspended"], "no_trade_days": counts["no_trades"],
        "total_days": len(days), "usable_days": counts["minute"] + counts["daily"],
        "minute_ratio": counts["minute"] / max(1, counts["minute"] + counts["daily"]),
        "data_resolution_hash": plan["plan_hash"], "warnings": warnings, "requested_frequency": target}
    if summary_only:
        return DataBundle(symbol=config.symbol, bars=pd.DataFrame(), daily=daily,
            actions=actions, calendar=days, quality=quality, version=manifest["version"], metadata=metadata, warnings=warnings)
    minute = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    if minute_days:
        if "1min" not in versions:
            raise DataQualityError("RESOLUTION_PLAN_MISMATCH: 冻结版本缺少计划指定的分钟线")
        minute_version = int(versions["1min"]["version"])
        minute = service.repository.bars(config.symbol, "1min", version=minute_version,
            start=plan["start"], end=pd.Timestamp(plan["end"]) + pd.Timedelta(days=1))
        minute = minute.copy()
        minute["timestamp"] = pd.to_datetime(minute.timestamp)
        minute = minute[minute.timestamp.dt.date.astype(str).isin(minute_days)].copy()
        _reject_invalid_prices(minute, config.symbol, "1min", minute_version,
            start=plan["start"], end=plan["end"], context="1min")
        _reject_reported_price_anomalies(service.repository, config.symbol, "1min", minute_version,
            start=plan["start"], end=plan["end"], context="1min", accepted_days=minute_days)
    if statistics_only:
        if minute_days:
            sample_calendar = pd.DataFrame({"date": sorted(pd.to_datetime(list(minute_days)))})
            data = _statistics_bundle(config, minute, daily, actions, sample_calendar, manifest)
            metadata.update({key: value for key, value in data.metadata.items()
                             if key in {"etf_volume_verified", "volume_unverified_days"}})
            bars = data.bars
        else:
            bars = minute.copy()
            bars["frequency"] = "1m"
        return DataBundle(symbol=config.symbol, bars=bars, daily=daily,
            actions=actions, calendar=days, quality=quality, version=manifest["version"], metadata=metadata, warnings=warnings)
    data = prepare_bundle(config.symbol, minute, daily, actions, calendar,
        plan["start"], plan["end"], config.frequency, manifest["version"], metadata, manifest.get("profiles", {}))
    data.metadata.update(metadata)
    data.daily = daily
    data.quality.update({"data_resolution_hash": plan["plan_hash"], "skipped_days": counts["skip"],
                         "skip_days": quality["skip_days"]})
    # The engine needs the trusted prior close, not only in-window daily rows.
    return data


def _preflight_bundle(config, daily, actions, calendar, manifest, target, start, end):
    """Return the daily evidence needed for a read-only search preflight.

    The production bundle materializes every minute bar for replay.  Budget,
    training-window and memory estimates only need the locked daily series,
    actions and calendar; callers supply the minute-frame bytes separately.
    """
    days = pd.DatetimeIndex(
        pd.to_datetime(calendar.date, errors="coerce")
    ).dropna().normalize().unique().sort_values()
    days = days[(days >= start) & (days <= end)]
    daily_window = daily[daily.date.between(start, end)].copy().reset_index(drop=True)
    metadata = _metadata(manifest, daily, target)
    metadata["preflight_summary"] = True
    warnings = list((manifest.get("metadata") or {}).get("warnings") or [])
    return DataBundle(
        symbol=config.symbol, bars=pd.DataFrame(), daily=daily_window,
        actions=actions, calendar=days,
        quality={
            "days": [], "requested_frequency": target,
            "summary_only": True, "is_demo": bool(metadata.get("is_demo")),
            "warnings": warnings,
        },
        version=manifest["version"], metadata=metadata, warnings=warnings,
    )


def _statistics_bundle(config, minute, daily, actions, calendar, manifest):
    """Keep frozen raw minute prices intact for statistics without replay prep."""
    invalid = price_issue_details(minute)
    if not invalid.empty:
        raise DataNotReady(
            f"1min存在 {len(invalid)} 条无效价格，当前版本不能用于统计",
            gaps=[{"kind": "price_quality", "timeframe": "1min", "anomalies": anomaly_records(
                invalid, symbol=config.symbol, timeframe="1min",
                version=int(manifest["market_versions"]["1min"]["version"]), limit=50,
            )}],
        )
    start = pd.Timestamp(config.start_date).normalize()
    end = pd.Timestamp(config.end_date).normalize() if config.end_date else daily.date.max()
    end = min(end, daily.date.max())
    from ..corporate_actions import action_issues
    issues = action_issues(
        actions, daily, start, end, settlement=False,
        price_tick=float((manifest.get("metadata") or {}).get("price_tick", 0.01)),
    )
    if issues:
        raise DataQualityError("公司行动资料存在窗口内问题：" + "；".join(issues[:8]))
    days = pd.DatetimeIndex(pd.to_datetime(calendar.date)).normalize().sort_values().unique()
    days = days[(days >= start) & (days <= end)]
    daily_window = daily[daily.date.isin(days)].copy()
    missing_daily = days.difference(pd.DatetimeIndex(daily_window.date))
    if len(missing_daily):
        raise DataQualityError(
            "所选交易日缺少日线，无法确认交易状态："
            + ", ".join(str(day.date()) for day in missing_daily[:20])
        )
    daily_lookup = daily_window.set_index("date")
    minute = minute.copy()
    minute["timestamp"] = pd.to_datetime(minute.timestamp)
    # The repository's target is known to be 1min. Set the consumer label on
    # observed rows before concatenating auction placeholders, which would
    # otherwise create a frequency column only on those two placeholder rows.
    minute["frequency"] = "1m"
    groups = {
        pd.Timestamp(day).normalize(): group.copy()
        for day, group in minute.groupby(minute.timestamp.dt.normalize(), sort=False)
    }
    placeholders = []
    quality_days = []
    volume_unverified_days = []
    volume_comparable_days = 0
    for day in days:
        row = daily_lookup.loc[day] if day in daily_lookup.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if row is None:
            continue
        status = int(row.tradestatus)
        daily_volume = float(row.volume)
        if status not in (0, 1):
            raise DataQualityError(f"{day.date()} 的交易状态未知")
        if status == 0:
            quality_days.append({"date": str(day.date()), "mode": "suspended"})
            continue
        if daily_volume == 0:
            quality_days.append({"date": str(day.date()), "mode": "no_trades"})
            continue

        part = groups.get(day, minute.iloc[0:0]).drop_duplicates("timestamp", keep="last")
        slots = minute_slots(day)
        indexed = part.set_index("timestamp").reindex(slots)
        missing = indexed.close.isna().to_numpy()
        auction_mask = _closing_auction_placeholders(
            config.symbol, day, indexed, missing,
        )
        for position in np.flatnonzero(auction_mask):
            earlier = indexed.close.iloc[:position].dropna()
            if earlier.empty:
                continue
            previous_close = earlier.iloc[-1]
            if pd.isna(previous_close) or not np.isfinite(float(previous_close)) or float(previous_close) <= 0:
                continue
            stamp = slots[position]
            placeholders.append({
                "timestamp": stamp, "open": float(previous_close),
                "high": float(previous_close), "low": float(previous_close),
                "close": float(previous_close), "volume": 0.0, "turnover": 0.0,
                "source": "statistics_auction_placeholder",
                "source_verified": True, "is_synthetic": False,
                "is_auction_placeholder": True, "frequency": "1m",
            })
        if int(indexed.close.notna().sum()) + int(auction_mask.sum()) == 240:
            try:
                minute_volume = float(pd.to_numeric(part.volume, errors="coerce").sum())
                comparable = np.isfinite(minute_volume) and np.isfinite(daily_volume) and daily_volume > 0
            except (TypeError, ValueError):
                comparable = False
            if comparable:
                volume_comparable_days += 1
                if abs(minute_volume / daily_volume - 1.0) > 0.01:
                    volume_unverified_days.append(str(day.date()))

    if placeholders:
        minute = pd.concat([minute, pd.DataFrame(placeholders)], ignore_index=True, sort=False)
    if not minute.empty:
        minute = minute[minute.timestamp.dt.normalize().isin(days)]
        minute = minute.sort_values("timestamp", kind="stable").reset_index(drop=True)
    metadata = _metadata(manifest, daily, "1min")
    metadata["etf_volume_verified"] = volume_comparable_days > 0 and not volume_unverified_days
    metadata["volume_unverified_days"] = volume_unverified_days
    # Older snapshots used volume disagreement as a replay veto. Keep the
    # evidence, but ordinary research assumes the supplied minute weights.
    metadata.pop("etf_replay_blocked_reason", None)
    warnings = []
    if volume_unverified_days:
        warnings.append(
            "分钟聚合成交量与日线不一致；按分钟成交量继续理想化统计和参数回放"
        )
    quality = {
        "days": quality_days, "is_demo": bool(metadata.get("is_demo")),
        "requested_frequency": "1min", "warnings": warnings,
    }
    return DataBundle(
        symbol=config.symbol, bars=minute, daily=daily_window, actions=actions,
        calendar=days, quality=quality, version=manifest["version"],
        metadata=metadata, warnings=warnings,
    )


def _reject_invalid_prices(frame, symbol, timeframe, version, *, start, end, context):
    anomalies = anomalies_in_range(frame, start, end)
    if anomalies.empty:
        return
    raise DataNotReady(
        f"{context}存在 {len(anomalies)} 条无效价格，当前版本不能用于计算",
        gaps=[{
            "kind": "price_quality", "timeframe": timeframe,
            "anomalies": anomaly_records(
                anomalies, symbol=symbol, timeframe=timeframe,
                version=version, limit=50,
            ),
        }],
    )


def _reject_reported_price_anomalies(
    repository, symbol, timeframe, version, *, start, end, context, accepted_days=None,
):
    """Keep rejected source prices attached to the exact immutable data version."""
    detail = repository.version_info(symbol, timeframe, version)
    if not detail:
        return
    report = detail.get("report") or {}
    metadata = detail.get("metadata") or {}
    validation = metadata.get("validation") or {}
    reported = report.get("price_anomalies") or validation.get("price_anomalies") or []
    lower = pd.Timestamp(start).normalize() if start is not None else None
    upper = pd.Timestamp(end).normalize() + pd.Timedelta(days=1) if end is not None else None
    matching = []
    for row in reported:
        try:
            stamp = pd.Timestamp(row.get("timestamp") or row.get("date"))
        except (TypeError, ValueError):
            continue
        if lower is not None and stamp < lower:
            continue
        if upper is not None and stamp >= upper:
            continue
        if accepted_days is not None and stamp.date().isoformat() not in accepted_days:
            continue
        matching.append({
            **row, "symbol": symbol, "timeframe": timeframe,
            "version": int(version), "timestamp": stamp.isoformat(),
        })
    if matching:
        raise DataNotReady(
            f"{context}存在 {len(matching)} 条已隔离的无效价格记录，当前版本不能用于计算",
            gaps=[{
                "kind": "price_quality", "timeframe": timeframe,
                "anomalies": matching[:50],
            }],
        )


def _calendar(service, symbol, daily, version=None):
    saved = service.repository.auxiliary(symbol, "calendar", version=version) if version else service.repository.auxiliary(symbol, "calendar")
    if saved:
        frame = pd.DataFrame(saved["payload"])
        if not frame.empty and "date" in frame:
            frame = frame[['date']].copy()
            frame["date"] = pd.to_datetime(frame.date, errors="raise").dt.normalize()
            # Packaged auxiliary calendars may contain only the incremental
            # tail. Daily rows remain the authoritative local trading series;
            # union them so research can reconstruct the full lookback window.
            observed = pd.DataFrame({"date": pd.DatetimeIndex(daily.date).normalize().unique().sort_values()})
            return pd.concat([frame, observed], ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return pd.DataFrame({"date": pd.DatetimeIndex(daily.date).normalize().unique().sort_values()})


def _actions(service, symbol, version=None):
    saved = service.repository.auxiliary(symbol, "actions", version=version) if version else service.repository.auxiliary(symbol, "actions")
    if not saved:
        return pd.DataFrame(columns=["record_date", "ex_date", "pay_date", "stock_date", "cash_per_share", "bonus_ratio", "implemented", "unsupported_reason"])
    frame = pd.DataFrame(saved["payload"])
    for field in ("record_date", "ex_date", "pay_date", "stock_date"):
        if field in frame:
            frame[field] = pd.to_datetime(frame[field], errors="coerce")
    return frame


def _metadata(manifest, daily, target):
    prior_closes = {}
    for row in daily.itertuples():
        try:
            if float(row.close) > 0 and int(row.tradestatus) == 1 and float(row.volume) > 0:
                prior_closes[str(pd.Timestamp(row.date).date())] = float(row.close)
        except (TypeError, ValueError):
            continue
    return {**manifest.get("metadata", {}), "daily_through": str(daily.date.max().date()),
            "requested_frequency": target, "market_versions": manifest.get("market_versions", {}),
            "prior_closes": prior_closes}


def _native_coarse_bundle(config, target, minute, daily, actions, calendar, manifest):
    start, end = pd.Timestamp(config.start_date).normalize(), pd.Timestamp(config.end_date).normalize() if config.end_date else daily.date.max()
    end = min(end, daily.date.max())
    days = pd.DatetimeIndex(pd.to_datetime(calendar.date)).normalize().sort_values().unique()
    days = days[(days >= start) & (days <= end)]
    lookup = daily.set_index("date")
    expected = 240 // (timeframe_spec(target).minutes or 1)
    output, reports = [], []
    groups = {day: part.copy() for day, part in minute.groupby(minute.timestamp.dt.normalize())}
    for day in days:
        row = lookup.loc[day] if day in lookup.index else None
        if row is None:
            raise DataQualityError(f"{day.date()} 缺少日线，无法确认交易状态")
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if int(row.tradestatus) == 0:
            reports.append({"date": str(day.date()), "mode": "suspended", "missing_minutes": 0, "missing_data_minutes": 0, "auction_placeholders": 0, "reason": "确认停牌"})
            continue
        if float(row.volume) == 0:
            reports.append({"date": str(day.date()), "mode": "no_trades", "missing_minutes": 0, "missing_data_minutes": 0, "auction_placeholders": 0, "reason": "交易状态正常但全天无成交量"})
            continue
        part = groups.get(day, pd.DataFrame()).sort_values("timestamp")
        if len(part) != expected:
            fallback = pd.DataFrame([{**row.to_dict(), "timestamp": day + pd.Timedelta(hours=15), "frequency": "1d", "tradable": True, "is_synthetic": False}])
            output.append(fallback)
            reports.append({"date": str(day.date()), "mode": "daily", "missing_minutes": max(0, expected - len(part)),
                            "missing_data_minutes": max(0, expected - len(part)), "auction_placeholders": 0,
                            "reason": f"原生 {target} 周期不完整，整日降级为日线"})
            continue
        part = part.copy()
        part["frequency"] = target
        part["tradable"] = ~part.is_synthetic.astype(bool) & part.volume.gt(0)
        part["preclose"] = row.preclose
        part["is_st"] = bool(row.is_st)
        for field in ("limit_up", "limit_down", "no_price_limit"):
            part[field] = row.get(field, False if field == "no_price_limit" else np.nan)
        output.append(part)
        reports.append({"date": str(day.date()), "mode": "minute", "missing_minutes": 0,
                        "missing_data_minutes": 0, "auction_placeholders": 0, "reason": "完整原生周期"})
    bars = pd.concat(output, ignore_index=True).sort_values("timestamp") if output else pd.DataFrame()
    mode_minute = sum(item["mode"] == "minute" for item in reports)
    mode_daily = sum(item["mode"] == "daily" for item in reports)
    suspended = sum(item["mode"] == "suspended" for item in reports)
    no_trades = sum(item["mode"] == "no_trades" for item in reports)
    quality = {"days": reports, "minute_days": mode_minute, "daily_days": mode_daily,
               "suspended_days": suspended, "no_trade_days": no_trades, "total_days": len(reports),
               "minute_ratio": mode_minute / max(1, mode_minute + mode_daily),
               "synthetic_bars": 0, "auction_placeholders": 0, "is_demo": False,
               "requested_frequency": target, "warnings": []}
    warnings = [f"{mode_daily} 个交易日原生 {target} 行情不完整，已整日降级为日线"] if mode_daily else []
    return DataBundle(
        symbol=config.symbol, bars=bars, daily=daily, actions=actions, calendar=days,
        quality=quality, version=manifest["version"],
        metadata=_metadata(manifest, daily, target), warnings=warnings,
    )
