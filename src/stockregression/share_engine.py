"""Version-two share accounting. Only market preparation is cached across trials."""
from __future__ import annotations
from collections import OrderedDict
from datetime import date
from dataclasses import replace
import math
import numpy as np
import pandas as pd
from .rules import instrument_terms, commissions_and_taxes, round_to_tick, round_money, fee_rates, transfer_fee_terms, calculated_price_limits, etf_price_limit_rate

_CACHE = OrderedDict()
BUY_HOLD_INITIAL_SHARES = 1000


def period_key(day, frequency):
    day = pd.Timestamp(day)
    if frequency == "daily":
        return str(day.date())
    if frequency == "weekly":
        iso = day.isocalendar()
        return f"{iso.year}-{iso.week}"
    return str(day.date())[:7]


def funding(config, data):
    prices = dict(data.metadata.get("prior_closes", {}))
    prices.update({str(pd.Timestamp(row.date).date()): float(row.close) for row in data.daily.itertuples() if row.close > 0 and getattr(row, "tradestatus", 1) == 1})
    previous = sorted((day, price) for day, price in prices.items() if day < str(config.start_date))
    if previous:
        day, price = previous[-1]
    elif data.metadata.get("reference_date", "9999") < str(config.start_date) and data.metadata.get("valuation_price", 0) > 0:
        day, price = data.metadata["reference_date"], float(data.metadata["valuation_price"])
    else:
        raise ValueError("开始日期前缺少收盘价，请补齐历史数据")
    reference = config.reference_price if config.reference_price_mode == "manual" and config.reference_price else price
    terms, fees = instrument_terms(config.symbol, data.metadata), config.fees
    execution = round_to_tick(reference * (1 + fees.slippage_bps / 10000), terms["price_tick"], side="buy")
    def cost(quantity):
        if not quantity:
            return 0.0
        charge = commissions_and_taxes(round_money(quantity * execution), "buy", config.start_date, fees.commission_rate, fees.min_commission, symbol=config.symbol, quantity=quantity, face_value=fees.share_face_value, legacy_sh_transfer_rate=fees.legacy_sh_transfer_rate, legacy_sh_min_transfer=fees.legacy_sh_min_transfer, legacy_sz_transfer_rate=fees.legacy_sz_transfer_rate)
        return round_money(quantity * execution + charge["total"])
    count, remainder = divmod(config.max_shares - config.initial_shares, config.buy_shares)
    reserve = round_money(count * cost(config.buy_shares) + cost(remainder))
    return {
        "reserve_cash": reserve,
        "initial_equity": round_money(config.initial_shares * price + reserve),
        "dca_initial_equity": round_money(config.dca.initial_shares * price),
        "buy_hold_initial_shares": BUY_HOLD_INITIAL_SHARES,
        "buy_hold_initial_equity": round_money(BUY_HOLD_INITIAL_SHARES * price),
        "valuation_price": price,
        "reference_price": reference,
        "reference_date": day,
        "initial_shares": config.initial_shares,
        "max_shares": config.max_shares,
    }


def _inputs(config, data):
    from .engine import _prepare_actions, open_timestamp
    fees = config.fees
    key = (id(data), data.version, config.start_date, config.end_date, config.frequency, config.dca.frequency, fees.share_face_value, fees.legacy_sh_transfer_rate, fees.legacy_sh_min_transfer, fees.legacy_sz_transfer_rate)
    if key in _CACHE and _CACHE[key]["data"] is data:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    terms = instrument_terms(config.symbol, data.metadata)
    start = pd.Timestamp(config.start_date)
    end = min(pd.Timestamp(config.end_date) if config.end_date else pd.Timestamp.max.normalize(), pd.Timestamp(data.calendar.max()) if len(data.calendar) else pd.Timestamp(data.daily.date.max()))
    daily = data.daily[data.daily.date.between(start, end)].sort_values("date")
    frame = data.bars.copy()
    if not frame.empty:
        frame = frame[frame.timestamp.between(start, end + pd.Timedelta(days=1), inclusive="left")]
        if "is_synthetic" in frame:
            frame = frame[~frame.is_synthetic.fillna(False)]
    observed = set(pd.to_datetime(frame.timestamp).dt.normalize()) if not frame.empty else set()
    additions = []
    for row in daily.to_dict("records"):
        day = pd.Timestamp(row["date"])
        suspended = row.get("tradestatus") == 0 or row.get("volume", 0) == 0
        if suspended and not frame.empty:
            frame = frame[frame.timestamp.dt.normalize() != day]
        if suspended or day not in observed:
            if not suspended and config.frequency != "1d" and not any(x.get("date") == str(day.date()) and x.get("mode") == "daily" for x in data.quality.get("days", [])):
                raise ValueError(f"{day.date()} 缺少可回测行情")
            additions.append({**row, "timestamp": day + pd.Timedelta(hours=15), "frequency": "1d", "tradable": True, "volume": 0 if suspended else row["volume"]})
    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    if frame.empty:
        raise ValueError("所选区间没有可回测行情")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    stamps = pd.DatetimeIndex(frame.timestamp)
    days = stamps.normalize().unique().sort_values()
    codes = pd.factorize(stamps.normalize(), sort=True)[0].astype(np.int64)
    n = len(frame)
    first = np.r_[True, codes[1:] != codes[:-1]]
    tradable = frame.get("tradable", pd.Series(True, index=frame.index)).fillna(False).to_numpy(dtype=np.bool_)
    last = np.zeros(n, dtype=np.bool_)
    valid = np.flatnonzero(tradable)
    if len(valid):
        last[valid[np.r_[codes[valid][:-1] != codes[valid][1:], True]]] = True
    keys = [period_key(day, config.dca.frequency) for day in days]
    new_period = np.r_[True, np.asarray(keys[1:]) != np.asarray(keys[:-1])]
    lookup = {pd.Timestamp(row["date"]): row for row in daily.to_dict("records")}
    ups, downs = np.full(n, np.nan), np.full(n, np.nan)
    preclose, stamp, transfer, minimum = (np.full(len(days), np.nan) for _ in range(4))
    face, bilateral = np.zeros(len(days), dtype=np.bool_), np.zeros(len(days), dtype=np.bool_)
    for day_id, day in enumerate(days):
        row = lookup.get(day, {})
        previous = float(row.get("preclose", np.nan))
        preclose[day_id] = previous
        up, down = row.get("limit_up", np.nan), row.get("limit_down", np.nan)
        if not bool(row.get("no_price_limit", False)):
            if terms["asset_type"] == "etf":
                rate = etf_price_limit_rate(terms, day.date())
                calc = (round_to_tick(previous * (1 + rate), terms["price_tick"]), round_to_tick(previous * (1 - rate), terms["price_tick"])) if previous > 0 else (np.nan, np.nan)
            else:
                calc = calculated_price_limits(previous if previous > 0 else None, terms["board"], day.date(), bool(row.get("is_st", False)))
            up, down = up if pd.notna(up) else calc[0], down if pd.notna(down) else calc[1]
            ups[codes == day_id], downs[codes == day_id] = up or np.nan, down or np.nan
        stamp[day_id] = fee_rates(day.date(), config.symbol)[0]
        transfer[day_id], face[day_id], minimum[day_id] = transfer_fee_terms(day.date(), config.symbol, legacy_sh_transfer_rate=fees.legacy_sh_transfer_rate, legacy_sh_min_transfer=fees.legacy_sh_min_transfer, legacy_sz_transfer_rate=fees.legacy_sz_transfer_rate)
        bilateral[day_id] = day.date() < date(2008, 9, 19)
    for column, output in [("limit_up", ups), ("limit_down", downs)]:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            mask = np.isfinite(values) & np.isfinite(output)
            output[mask] = values[mask]
    opens = pd.to_numeric(frame.open, errors="coerce").to_numpy(dtype=np.float64)
    closes = pd.to_numeric(frame.close, errors="coerce").to_numpy(dtype=np.float64)
    if np.any(tradable & (~np.isfinite(opens) | ~np.isfinite(closes) | (opens <= 0) | (closes <= 0))):
        raise ValueError("行情价格无效")
    action_data = data
    if not data.actions.empty:
        record_dates = pd.to_datetime(data.actions.record_date, errors="coerce")
        ex_dates = pd.to_datetime(data.actions.ex_date, errors="coerce")
        relevant = (record_dates.isna() | (record_dates <= end)) & (ex_dates.isna() | (ex_dates >= start))
        action_data = replace(data, actions=data.actions.loc[relevant])
    actions = _prepare_actions(action_data, config.start_date, end.date())
    day_ids = {day.date(): index for index, day in enumerate(days)}
    arrays = (opens, closes, pd.to_numeric(frame.volume, errors="coerce").fillna(0).to_numpy(dtype=np.float64), tradable, ups, downs, stamps.asi8, np.asarray([pd.Timestamp(open_timestamp(ts, str(freq))).value for ts, freq in zip(stamps, frame.get("frequency", pd.Series(config.frequency, index=frame.index)))], dtype=np.int64), codes, first & new_period[codes], last, stamp[codes], transfer[codes], face[codes], minimum[codes], bilateral[codes], preclose,
              np.asarray([-1 if a.record_date < config.start_date else day_ids.get(a.record_date, -2) for a in actions], dtype=np.int64), np.asarray([day_ids.get(a.ex_date, -2) for a in actions], dtype=np.int64), np.asarray([day_ids.get(a.pay_date, -2) for a in actions], dtype=np.int64), np.asarray([day_ids.get(a.stock_date, -2) for a in actions], dtype=np.int64), np.asarray([a.cash_per_share for a in actions], dtype=np.float64), np.asarray([a.bonus_ratio for a in actions], dtype=np.float64))
    result = {"arrays": arrays, "days": days, "stamps": stamps, "tradable": tradable, "opens": arrays[7], "period_keys": keys, "terms": terms, "data": data, "end": str(end.date())}
    _CACHE[key] = result
    while len(_CACHE) > 2:
        _CACHE.popitem(last=False)
    return result


def run_share_backtest(config, data, progress=None, cancelled=None, summary_only=False, preview=False):
    from . import engine
    from .share_kernel import simulate
    if cancelled and cancelled():
        raise InterruptedError("回测已取消")
    prepared, money = _inputs(config, data), funding(config, data)
    terms, fees = prepared["terms"], config.fees
    if progress:
        progress(0.2, "已载入股数配置与锁定行情")
    mode = 0 if summary_only else 1 if preview else 2
    options = (money["initial_equity"], money["dca_initial_equity"], money["buy_hold_initial_equity"], float(config.min_shares), float(config.max_shares), {"trailing": 0, "arithmetic": 1, "geometric": 2}[config.grid_mode], config.buy_step, config.sell_step, config.grid_step, money["reference_price"], config.lower_price or 0.0, config.upper_price or 0.0, config.buy_shares, config.sell_shares, 2 if terms["board"] == "star" else 0, fees.commission_rate, fees.min_commission, fees.slippage_bps / 10000, fees.share_face_value, config.initial_shares, money["reserve_cash"], config.dca.initial_shares, config.dca.shares, money["buy_hold_initial_shares"], money["valuation_price"], terms["price_tick"], terms["settlement"] == "T0", mode)
    runner = simulate if engine._NUMBA_FASTPATH_ENABLED else getattr(simulate, "py_func", simulate)
    output = runner(*prepared["arrays"], *options)
    if cancelled and cancelled():
        raise InterruptedError("回测已取消")
    metrics, eq, cash, shares, drawdown, periods, tb, ta, ts, tq, tp, tf, tc, th, tr, rb, rt, ra, rs, rr, rq, invested = output
    strategies = {}
    points = prepared["days"] + pd.Timedelta(hours=15) if preview else prepared["stamps"]
    indices = np.arange(len(points)) if preview else np.flatnonzero(prepared["tradable"])
    openings = pd.to_datetime(prepared["opens"]).strftime("%Y-%m-%dT%H:%M:%S").tolist() if not summary_only else []
    initial_equities = (money["initial_equity"], money["dca_initial_equity"], money["buy_hold_initial_equity"])
    for account, name in enumerate(("grid", "dca", "buy_hold")):
        metric = {key: (int(round(metrics[account, i])) if key in {"trade_count", "buy_count", "sell_count"} else float(metrics[account, i]) if math.isfinite(metrics[account, i]) else None) for i, key in enumerate(engine._METRIC_KEYS)}
        metric["initial_equity"] = initial_equities[account]
        if account != 1 and initial_equities[account] <= 0:
            metric.update({key: None for key in ("total_return", "cagr", "max_drawdown", "sharpe", "calmar", "turnover")})
        if account == 1:
            metric.update(total_invested=float(metrics[account, 10]), skipped_periods=len(set(prepared["period_keys"])) - metric["buy_count"])
        equity = []
        if not summary_only:
            for i in indices:
                base = invested[i] if account == 1 else initial_equities[account]
                equity.append({"timestamp": points[i].isoformat(), "equity": float(eq[account, i]), "cash": float(cash[account, i]), "position": float(shares[account, i]), "drawdown": None if account == 1 else float(drawdown[account, i]), "normalized_equity": float(eq[account, i] / base) if base > 0 else None, "total_invested": float(invested[i]) if account == 1 else initial_equities[account]})
        trades = [{"timestamp": openings[int(tb[j])], "side": "buy" if ts[j] == 1 else "sell", "quantity": int(tq[j]), "price": float(tp[j]), "fees": float(tf[j]), "cash": float(tc[j]), "shares": int(th[j]), "reason": engine._TRADE_REASONS[int(tr[j])]} for j in range(len(tb)) if ta[j] == account]
        rejections = [{"timestamp": pd.Timestamp(rt[j]).isoformat(), "side": "buy" if rs[j] == 1 else "sell", "reason": engine._REJECTION_REASONS[int(rr[j])], "requested_quantity": int(rq[j])} for j in range(len(rb)) if ra[j] == account]
        if account == 1 and not summary_only and not preview:
            bought = {period_key(t["timestamp"], config.dca.frequency) for t in trades}
            for key in dict.fromkeys(prepared["period_keys"]):
                if key not in bought:
                    day = [d for d, p in zip(prepared["days"], prepared["period_keys"]) if p == key][-1]
                    rejections.append({"timestamp": (day + pd.Timedelta(hours=15)).isoformat(), "side": "buy", "reason": "dca_period_skipped", "requested_quantity": config.dca.shares})
        returns = [] if account == 1 or summary_only else [{"timestamp": (day + pd.Timedelta(hours=15)).isoformat(), "return": float(periods[account, i])} for i, day in enumerate(prepared["days"])]
        strategies[name] = {"metrics": metric, "equity": equity, "equity_frequency": "1d" if preview else config.frequency, "trades": trades, "rejections": rejections, "period_returns": returns, "return_basis": "cost" if account == 1 else "initial_equity"}
    if progress:
        progress(1.0, "股数回测完成")
    return {"schema_version": 2, "accounting_basis": "independent_contributions_v1", "symbol": config.symbol, "config": config.model_dump(mode="json"), "data_version": data.version, "quality": data.quality, "is_demo": bool(data.metadata.get("is_demo")), "funding": money, "requested_end_date": str(config.end_date) if config.end_date else None, "effective_end_date": prepared["end"], "warnings": list(dict.fromkeys([*data.warnings, *data.metadata.get("warnings", [])])), "strategies": strategies}
