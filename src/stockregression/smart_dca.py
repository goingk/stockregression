"""Causal, explainable DCA signals. These are project rules, not a vendor replica."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .models import DataBundle
from .price_basis import action_factors


def smart_dca_signals(config, data, days):
    """Return daily amount caps and audit evidence, using only rows before each day.

    A failed execution may retry on the next day of the same period, with a new
    causal signal. Unused allocations never carry into another period.
    """
    dca = config.dca
    if dca.mode == "fixed_shares":
        return None, {}
    symbol = config.symbol if dca.reference == "target" else dca.reference_symbol
    source = data.dca_reference
    if source is None:
        if dca.reference != "target":
            raise ValueError(f"智能定投参考 {symbol} 未准备，不能使用交易标的替代")
        source = {"symbol": data.symbol, "kind": "target", "daily": data.daily,
                  "actions": data.actions, "metadata": data.metadata, "version": data.version}
    if source.get("symbol") != symbol or source.get("kind") != dca.reference:
        raise ValueError("智能定投参考证券/指数与冻结数据不匹配")
    if dca.reference_version and source.get("version") != dca.reference_version:
        raise ValueError("智能定投参考数据版本与配置中的冻结版本不匹配")
    raw = source["daily"].copy().rename(columns={"timestamp": "date"})
    if not {"date", "close"}.issubset(raw.columns):
        raise ValueError(f"智能定投参考 {symbol} 缺少日期或收盘字段")
    raw["date"] = pd.to_datetime(raw.date, errors="coerce").dt.normalize()
    raw = raw[raw.date.notna()].sort_values("date")
    if raw.date.duplicated().any():
        raise ValueError(f"智能定投参考 {symbol} 存在重复日线")
    raw["close"] = pd.to_numeric(raw.close, errors="coerce")
    if "preclose" in raw:
        raw["preclose"] = pd.to_numeric(raw.preclose, errors="coerce")
    target_days = pd.DatetimeIndex(data.calendar).union(pd.DatetimeIndex(pd.to_datetime(data.daily.date)))
    target_days = target_days.union(pd.DatetimeIndex(pd.to_datetime(list(data.metadata.get("prior_closes", {})))))
    target_days = target_days.normalize().unique().sort_values()
    required = dca.ma_window if dca.mode == "ma" else dca.change_window + 1
    amounts = np.full(len(days), np.nan, dtype=np.float64)
    evidence = {}
    for index, day in enumerate(days):
        day = pd.Timestamp(day).normalize()
        detail = {"mode": dca.mode, "reference": dca.reference, "reference_symbol": symbol,
                  "reference_version": source.get("version"), "execution_date": str(day.date()),
                  "signal_date": None, "window": required, "base_amount": dca.base_amount}
        def skip(code, message):
            detail.update(status="skipped", reason=code, message=message)
            evidence[str(day.date())] = detail
        # Slice first. No future close, preclose, or action is allowed into any
        # subsequent validation, adjustment or aggregation.
        prior = raw[raw.date < day].copy()
        expected = target_days[target_days < day]
        if prior.empty or (len(expected) and prior.date.iloc[-1] < expected[-1]):
            skip("dca_reference_stale", "参考数据缺少执行日前一交易日的收盘价")
            continue
        window = prior.tail(required)
        if len(window) < required:
            skip("dca_insufficient_history", f"参考历史不足，需要 {required} 条此前收盘，现有 {len(window)} 条")
            continue
        valid = np.isfinite(window.close) & window.close.gt(0)
        if "tradestatus" in window:
            valid &= pd.to_numeric(window.tradestatus, errors="coerce").eq(1)
        if not valid.all():
            skip("dca_reference_invalid", "参考窗口含无效收盘价或停牌，未补造信号")
            continue
        known_days = expected[(expected >= window.date.iloc[0]) & (expected <= window.date.iloc[-1])]
        if len(known_days.difference(pd.DatetimeIndex(window.date))):
            skip("dca_reference_gap", "参考窗口缺少交易日收盘数据")
            continue
        signal_day = window.date.iloc[-1]
        detail["signal_date"] = str(signal_day.date())
        prices = window.close.to_numpy(dtype=float).copy()
        if dca.reference != "index":
            actions = source.get("actions", pd.DataFrame()).copy()
            if not actions.empty:
                actions = actions[pd.to_datetime(actions.ex_date, errors="coerce") <= signal_day]
            known = DataBundle(symbol=symbol, bars=pd.DataFrame(), daily=prior, actions=actions,
                               metadata=source.get("metadata", {}))
            factors, issue = action_factors(known, signal_day, start=window.date.iloc[0] + pd.Timedelta(days=1))
            if issue:
                skip("dca_reference_action", issue)
                continue
            # An unexplained ex-rights jump is not a market drawdown.
            if "preclose" in window:
                changes = (window.preclose - window.close.shift(1)).abs()
                explained = {pd.Timestamp(factor["ex_date"]) for factor in factors}
                tick = float(known.metadata.get("price_tick", .001 if symbol.startswith(("1", "5")) else .01))
                if any(stamp not in explained for stamp in window.loc[changes > tick * 1.1, "date"]):
                    skip("dca_reference_action", "参考价格发生除权跳变，但缺少对应公司行动")
                    continue
            for factor in factors:
                prices[(window.date < factor["ex_date"]).to_numpy()] *= factor["factor"]
        baseline = float(prices.mean()) if dca.mode == "ma" else float(prices[0])
        signal = float(prices[-1] / baseline - 1)
        multiplier = min(dca.max_multiplier, max(dca.min_multiplier, 1 - dca.sensitivity * signal))
        amount = round(dca.base_amount * multiplier, 2)
        if not math.isfinite(amount):
            skip("dca_reference_invalid", "参考信号计算产生无效金额")
            continue
        amounts[index] = amount
        detail.update(status="ready", close=float(prices[-1]), baseline=baseline, signal=signal,
                      multiplier=multiplier, planned_amount=amount)
        evidence[str(day.date())] = detail
    return amounts, evidence
