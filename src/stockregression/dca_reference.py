"""Daily reference preparation for smart DCA, separate from tradable instruments.

Index identifiers are deliberately not passed to normalize_symbol: 000001.SH
is the Shanghai index, whereas 000001.SZ is a stock.
"""
from __future__ import annotations

import hashlib
import json
import pandas as pd

INDEX_REFERENCES = {
    "000001.SH": "上证指数", "000300.SH": "沪深300", "000905.SH": "中证500",
    "000016.SH": "上证50", "399001.SZ": "深证成指", "399006.SZ": "创业板指",
}


def _cache_key(symbol):
    return f"smart_dca_index_v1:{symbol}"


def fetch_index_daily(symbol, start, end):
    # Reuse the application's serialized, paced BaoStock connection.
    from .providers import serialized_baostock, _baostock_session, _baostock_reset, paced_api_request

    @serialized_baostock
    def fetch():
        try:
            code, exchange = symbol.split(".")
            bs = _baostock_session()
            with paced_api_request():
                result = bs.query_history_k_data_plus(
                    f"{exchange.lower()}.{code}", "date,close,preclose,volume",
                    start_date=str(pd.Timestamp(start).date()), end_date=str(pd.Timestamp(end).date()),
                    frequency="d", adjustflag="3",
                )
            if result.error_code != "0":
                raise ValueError(f"指数参考数据读取失败：{result.error_msg}")
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            if result.error_code != "0":
                raise ValueError(f"指数参考数据读取中断：{result.error_msg}")
            return pd.DataFrame(rows, columns=result.fields)
        except Exception:
            _baostock_reset()
            raise
    return fetch()


def prepare_index_reference(store, config):
    from .quality import DataQualityError
    from .data import latest_closed_date
    dca = config.dca
    if dca.reference != "index" or dca.reference_symbol not in INDEX_REFERENCES:
        raise ValueError("请选择有效的智能定投参考指数")
    # Calendar padding is only a fetch bound; signals still require N actual closes.
    window = dca.ma_window if dca.mode == "ma" else dca.change_window + 1
    start = pd.Timestamp(config.start_date) - pd.Timedelta(days=window * 3 + 30)
    end = min(pd.Timestamp(config.end_date or latest_closed_date()), pd.Timestamp(latest_closed_date()))
    if end < start:
        raise ValueError("参考数据日期区间无效")
    symbol = dca.reference_symbol
    frame = fetch_index_daily(symbol, start, end)
    if frame.empty or not {"date", "close"}.issubset(frame.columns):
        raise DataQualityError(f"指数 {symbol} 没有返回历史收盘数据")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame.close, errors="coerce")
    frame = frame[frame.date.between(start, end) & frame.close.gt(0)].sort_values("date").drop_duplicates("date")
    if frame.empty:
        raise DataQualityError(f"指数 {symbol} 没有有效历史收盘数据")
    rows = [{"date": str(row.date.date()), "close": float(row.close)} for row in frame.itertuples()]
    version = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:20]
    payload = {"symbol": symbol, "kind": "index", "source": "baostock", "version": version,
               "rows": rows, "first_date": rows[0]["date"], "last_date": rows[-1]["date"]}
    # The immutable version is retained for audit; the pointer selects new requests.
    store.state(f"{_cache_key(symbol)}:{version}", payload)
    store.state(_cache_key(symbol), payload)
    return {key: value for key, value in payload.items() if key != "rows"} | {"row_count": len(rows)}


def freeze_dca_reference(store, config):
    """Resolve the latest pointer once, before a request is enqueued.

    Already frozen submissions/resumes validate the exact version and never
    fall back. Target references are bound by the main backtest snapshot.
    """
    from .quality import DataQualityError
    dca = config.dca
    if dca.mode == "fixed_shares" or dca.reference == "target":
        return config
    symbol = dca.reference_symbol
    if dca.reference == "index":
        key = _cache_key(symbol) + (f":{dca.reference_version}" if dca.reference_version else "")
        payload = store.state(key)
        if not payload or payload.get("symbol") != symbol or (dca.reference_version and payload.get("version") != dca.reference_version):
            raise DataQualityError(f"智能定投指数 {symbol} 的参考版本不存在，请先准备指数参考数据")
        version = payload["version"]
        if not store.state(f"{_cache_key(symbol)}:{version}"):
            raise DataQualityError(f"智能定投指数 {symbol} 的冻结版本已缺失，请重新准备参考数据")
    else:
        manifest = store.snapshot(symbol, dca.reference_version)
        if not manifest or "1day" not in manifest.get("market_versions", {}):
            raise DataQualityError(f"智能定投参考证券 {symbol} 的日线版本不存在，请在数据管理中同步该证券")
        version = manifest["version"]
    return config.model_copy(update={"dca": dca.model_copy(update={"reference_version": str(version)})})


def freeze_research_dca_reference(store, config):
    """Freeze both representations consumed by OptimizationConfig validators.

    The worker rebuilds ``base`` from ``research_context`` when validating a
    serialized job. Updating base alone would silently discard the lock.
    """
    base = freeze_dca_reference(store, config.base)
    updates = {"base": base}
    if config.research_context is not None:
        updates["research_context"] = config.research_context.model_copy(update={"dca": base.dca})
    return config.model_copy(update=updates)


def attach_dca_reference(service, config, bundle):
    """Load all warmup rows from the chosen snapshot, including pre-start dates."""
    from .quality import DataQualityError
    dca = config.dca
    if dca.mode == "fixed_shares":
        return bundle
    symbol = config.symbol if dca.reference == "target" else dca.reference_symbol
    if dca.reference != "target" and not dca.reference_version:
        raise DataQualityError("智能定投外部参考版本未锁定；请重新提交任务，不允许执行时读取最新参考")
    if dca.reference == "index":
        payload = service.store.state(f"{_cache_key(symbol)}:{dca.reference_version}")
        if not payload or payload.get("symbol") != symbol or payload.get("version") != dca.reference_version:
            raise DataQualityError(f"智能定投参考指数 {symbol} 的锁定版本不存在，不能改用最新参考")
        daily = pd.DataFrame(payload["rows"])
        daily["date"] = pd.to_datetime(daily.date)
        reference = {"kind": "index", "symbol": symbol, "version": payload["version"],
                     "source": payload["source"], "daily": daily, "actions": pd.DataFrame(), "metadata": {}}
    else:
        manifest = service.store.snapshot(symbol, bundle.version if dca.reference == "target" else dca.reference_version)
        if not manifest or "1day" not in manifest.get("market_versions", {}):
            raise DataQualityError(f"智能定投参考证券 {symbol} 缺少本地日线，请在数据管理中同步该证券")
        daily = service.repository.bars(symbol, "1day", version=int(manifest["market_versions"]["1day"]["version"]))
        from .market_data.consumer import _actions
        action_version = manifest.get("auxiliary_versions", {}).get("actions")
        # Missing in this snapshot means missing; never read the newest actions.
        actions = _actions(service, symbol, action_version) if action_version else pd.DataFrame()
        reference = {"kind": dca.reference, "symbol": symbol, "version": manifest["version"],
                     "source": "local_snapshot", "daily": daily.rename(columns={"timestamp": "date"}),
                     "actions": actions, "metadata": manifest.get("metadata", {})}
    bundle.dca_reference = reference
    return bundle
