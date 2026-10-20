"""Share accounting for version two and the cash-flow-aware version three."""
from __future__ import annotations
from collections import OrderedDict
from datetime import date
from dataclasses import replace
import math
import numpy as np
import pandas as pd
from .rules import instrument_terms, commissions_and_taxes, round_to_tick, round_money, fee_rates, transfer_fee_terms, calculated_price_limits, etf_price_limit_rate
from .performance import (
    blas_thread_limit, current_workspace_bytes, rebalance_cpu_budget,
    record_count, record_details,
)

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
    if config.schema_version == 3:
        reserve = 0.0
        initial_equity = round_money(config.initial_shares * price)
        funding_mode = "top_up_on_fill"
    else:
        maximum = config.max_shares if config.max_shares is not None else config.initial_shares
        count, remainder = divmod(maximum - config.initial_shares, config.buy_shares)
        reserve = round_money(count * cost(config.buy_shares) + cost(remainder))
        if getattr(config, "reserve_cash", None) is not None:
            reserve = round_money(config.reserve_cash)
        initial_equity = round_money(config.initial_shares * price + reserve)
    return {
        "reserve_cash": reserve,
        "initial_equity": initial_equity,
        "initial_principal": round_money(config.initial_shares * price),
        "initial_cash": 0.0 if config.schema_version == 3 else reserve,
        "funding_mode": "top_up_on_fill" if config.schema_version == 3 else "configured_reserve",
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


def _parameter_arrays(config, prepared, parameter_schedule):
    days, values, normalized = [], [], []
    for row in parameter_schedule or []:
        effective = pd.Timestamp(row["effective_date"]).normalize()
        decision = pd.Timestamp(row["decision_date"]).normalize()
        if decision >= effective:
            raise ValueError("换参决策必须早于生效交易日")
        index = int(prepared["days"].searchsorted(effective))
        if index >= len(prepared["days"]):
            continue
        if days and index <= days[-1]:
            raise ValueError("换参生效日期必须严格递增")
        params = dict(row["params"])
        if "grid_step" in params:
            params.setdefault("buy_step", params["grid_step"])
            params.setdefault("sell_step", params["grid_step"])
            params.pop("grid_step", None)
        if "initial_shares" in params and params["initial_shares"] != config.initial_shares:
            raise ValueError("换参不能改变初始持股")
        from .models import BacktestConfig
        candidate = BacktestConfig.model_validate({**config.model_dump(mode="python"), **params})
        days.append(index)
        values.append([candidate.min_shares, np.inf if candidate.max_shares is None else candidate.max_shares, candidate.buy_step,
                       candidate.sell_step, candidate.grid_step, candidate.buy_shares, candidate.sell_shares,
                       1.0 if row.get("paused") else 0.0])
        normalized.append({**row, "effective_date": str(prepared["days"][index].date())})
    return np.asarray(days, dtype=np.int64), np.asarray(values, dtype=np.float64).reshape((-1, 8)), normalized


def _options(config, prepared, money, mode=0, cost_multiplier=1.0):
    terms, fees = prepared["terms"], config.fees
    return (money["initial_equity"], money["dca_initial_equity"], money["buy_hold_initial_equity"],
        float(config.min_shares), np.inf if config.max_shares is None else float(config.max_shares),
        {"trailing": 0, "arithmetic": 1, "geometric": 2}[config.grid_mode],
        config.buy_step, config.sell_step, config.grid_step, money["reference_price"],
        config.lower_price or 0.0, config.upper_price or 0.0, config.buy_shares, config.sell_shares,
        2 if terms["board"] == "star" else 0, fees.commission_rate * cost_multiplier,
        fees.min_commission * cost_multiplier, fees.slippage_bps / 10000 * cost_multiplier,
        fees.share_face_value, config.initial_shares, money["reserve_cash"], config.dca.initial_shares,
        config.dca.shares, money["buy_hold_initial_shares"], money["valuation_price"],
        terms["price_tick"], terms["settlement"] == "T0", mode,
        1 if config.schema_version == 3 else 0)


def _return_days(prepared):
    return prepared["days"][np.unique(prepared["arrays"][8][prepared["arrays"][10]])]


def run_share_backtest(config, data, progress=None, cancelled=None, summary_only=False, preview=False,
                       parameter_schedule=None, passive_grid=False, daily_details=False):
    from . import engine
    from .share_kernel import simulate
    if cancelled and cancelled():
        raise InterruptedError("回测已取消")
    prepared, money = _inputs(config, data), funding(config, data)
    terms, fees = prepared["terms"], config.fees
    if progress:
        progress(0.2, "已载入股数配置与锁定行情")
    mode = 0 if summary_only else 1 if preview else 3 if daily_details else 2
    options = _options(config, prepared, money, mode)
    change_days, change_values, schedule = _parameter_arrays(config, prepared, parameter_schedule)
    runner = simulate if engine._NUMBA_FASTPATH_ENABLED else getattr(simulate, "py_func", simulate)
    output = runner(*prepared["arrays"], *options, change_days, change_values,
                    np.full(len(prepared["days"]), -1, dtype=np.int64), 0, False, passive_grid)
    if cancelled and cancelled():
        raise InterruptedError("回测已取消")
    metrics, eq, cash, shares, drawdown, periods, tb, ta, ts, tq, tp, tf, tc, th, tr, ttop, rb, rt, ra, rs, rr, rq, invested, _, funding_stats = output
    strategies = {}
    return_days = _return_days(prepared)
    points = return_days + pd.Timedelta(hours=15) if preview or daily_details else prepared["stamps"]
    indices = np.arange(len(points)) if preview or daily_details else np.flatnonzero(prepared["tradable"])
    openings = pd.to_datetime(prepared["opens"]).strftime("%Y-%m-%dT%H:%M:%S").tolist() if not summary_only else []
    initial_equities = (money["initial_equity"], money["dca_initial_equity"], money["buy_hold_initial_equity"])
    flow_aware = config.schema_version == 3
    peak_invested = float(funding_stats[0, 2]) if flow_aware else initial_equities[0]
    peak_day = int(round(funding_stats[0, 3])) if flow_aware and math.isfinite(funding_stats[0, 3]) else -1
    funding_flows = [
        {"timestamp": openings[int(tb[j])], "amount": float(ttop[j])}
        for j in range(len(tb)) if ta[j] == 0 and ttop[j] > 0
    ]
    if flow_aware:
        money.update({
            "topups": float(funding_stats[0, 1]),
            "peak_invested": peak_invested,
            "peak_invested_date": str(prepared["days"][peak_day].date()) if 0 <= peak_day < len(prepared["days"]) else None,
            "final_invested": float(funding_stats[0, 4]),
            "funding_flows": funding_flows,
        })
    for account, name in enumerate(("grid", "dca", "buy_hold")):
        metric = {key: (int(round(metrics[account, i])) if key in {"trade_count", "buy_count", "sell_count"} else float(metrics[account, i]) if math.isfinite(metrics[account, i]) else None) for i, key in enumerate(engine._METRIC_KEYS)}
        metric["initial_equity"] = initial_equities[account]
        if account != 1 and initial_equities[account] <= 0 and not (flow_aware and account == 0 and peak_invested > 0):
            metric.update({key: None for key in ("total_return", "cagr", "max_drawdown", "sharpe", "calmar", "turnover")})
        if account == 1:
            metric.update(total_invested=float(metrics[account, 10]), skipped_periods=len(set(prepared["period_keys"])) - metric["buy_count"])
        if flow_aware and account == 0:
            metric.update({
                "initial_equity": float(funding_stats[0, 0]),
                "peak_invested": peak_invested,
                "total_invested": float(funding_stats[0, 4]),
                "external_topups": float(funding_stats[0, 1]),
                "total_profit": float(funding_stats[0, 6]),
                "time_weighted_total_return": None if not math.isfinite(funding_stats[0, 7]) else float(funding_stats[0, 7]),
                "return_basis": "peak_invested_roi",
                "risk_return_basis": "cash_flow_adjusted_unit_nav",
            })
        equity = []
        if not summary_only:
            for i in indices:
                if account == 0 and flow_aware:
                    invested_at_point = float(invested[0, i])
                    profit = float(eq[account, i]) - invested_at_point
                    normalized = 1.0 + profit / peak_invested if peak_invested > 0 else None
                else:
                    invested_at_point = float(invested[1, i]) if account == 1 else initial_equities[account]
                    normalized = float(eq[account, i] / invested_at_point) if invested_at_point > 0 else None
                equity.append({"timestamp": points[i].isoformat(), "equity": float(eq[account, i]), "cash": float(cash[account, i]), "position": float(shares[account, i]), "drawdown": None if account == 1 else float(drawdown[account, i]), "normalized_equity": normalized, "total_invested": invested_at_point, "cumulative_profit": (float(eq[account, i]) - invested_at_point) if account == 0 and flow_aware else None})
        trades = [{"timestamp": openings[int(tb[j])], "side": "buy" if ts[j] == 1 else "sell", "quantity": int(tq[j]), "price": float(tp[j]), "fees": float(tf[j]), "cash": float(tc[j]), "shares": int(th[j]), "reason": engine._TRADE_REASONS[int(tr[j])], "external_topup": float(ttop[j]) if account == 0 else 0.0} for j in range(len(tb)) if ta[j] == account]
        rejections = [{"timestamp": pd.Timestamp(rt[j]).isoformat(), "side": "buy" if rs[j] == 1 else "sell", "reason": "parameter_update" if int(rr[j]) == 12 else engine._REJECTION_REASONS[int(rr[j])], "requested_quantity": int(rq[j])} for j in range(len(rb)) if ra[j] == account]
        if account == 1 and not summary_only and not preview:
            bought = {period_key(t["timestamp"], config.dca.frequency) for t in trades}
            for key in dict.fromkeys(prepared["period_keys"]):
                if key not in bought:
                    day = [d for d, p in zip(prepared["days"], prepared["period_keys"]) if p == key][-1]
                    rejections.append({"timestamp": (day + pd.Timedelta(hours=15)).isoformat(), "side": "buy", "reason": "dca_period_skipped", "requested_quantity": config.dca.shares})
        returns = [] if account == 1 or summary_only else [{"timestamp": (day + pd.Timedelta(hours=15)).isoformat(), "return": float(periods[account, i])} for i, day in enumerate(return_days)]
        strategy_basis = "peak_invested" if account == 0 and flow_aware else "cost" if account == 1 else "initial_equity"
        strategies[name] = {"metrics": metric, "equity": equity, "equity_frequency": "1d" if preview or daily_details else config.frequency, "trades": trades, "rejections": rejections, "period_returns": returns, "return_basis": strategy_basis}
    if progress:
        progress(1.0, "股数回测完成")
    warnings = list(dict.fromkeys([*data.warnings, *data.metadata.get("warnings", [])]))
    selection_asof = getattr(config, "selection_asof", None)
    retrospective = bool(selection_asof and config.start_date <= selection_asof)
    if retrospective:
        warnings.append(f"参数选定于 {selection_asof}；选参截止日及以前的历史重跑属于样本内回看，不是样本外成绩。")
    accounting_basis = "cash_flow_adjusted_peak_contributions_v1" if flow_aware else "independent_contributions_v1"
    result = {"schema_version": config.schema_version, "accounting_basis": accounting_basis, "symbol": config.symbol, "config": config.model_dump(mode="json"), "data_version": data.version, "quality": data.quality, "is_demo": bool(data.metadata.get("is_demo")), "funding": money, "requested_end_date": str(config.end_date) if config.end_date else None, "effective_end_date": prepared["end"], "warnings": warnings, "strategies": strategies}
    result["diagnostics"] = data.metadata.get("diagnostics", {})
    result["source_events"] = data.metadata.get("source_events", [])
    if schedule:
        result["parameter_schedule"] = schedule
    if selection_asof:
        result["parameter_selection"] = {"as_of": str(selection_asof), "retrospective": retrospective}
    if passive_grid:
        result["strategies"]["grid"]["label"] = "同资金同初始持股被动持有"
    return result


def research_batch(base, data, params, fold_ranges, cost_multiplier=1.0,
                   cancelled=None, parallel=True):
    """Compact day-by-candidate returns with exact intrabar fold risk.

    The cash budget is determined once from ``base`` and shared by every row.
    All folds are measurements on the same continuous candidate ledger.
    """
    from . import engine
    from .models import BacktestConfig
    from .share_kernel import simulate, simulate_research_batch
    if base.schema_version not in (2, 3):
        raise ValueError("科学研究需要股数配置")
    if not math.isfinite(cost_multiplier) or cost_multiplier <= 0:
        raise ValueError("成本倍数必须为正数")
    prepared, money = _inputs(base, data), funding(base, data)
    if money["initial_equity"] <= 0:
        raise ValueError("研究账户起始资产必须大于零")
    fold_ids = np.full(len(prepared["days"]), -1, dtype=np.int64)
    for index, (start, end) in enumerate(fold_ranges):
        mask = (prepared["days"] >= pd.Timestamp(start)) & (prepared["days"] <= pd.Timestamp(end))
        if np.any(fold_ids[mask] >= 0):
            raise ValueError("验证窗口不能重叠")
        fold_ids[mask] = index
    rows = []
    for patch in params:
        if patch.get("initial_shares", base.initial_shares) != base.initial_shares:
            raise ValueError("科学研究不能搜索初始持股")
        if base.schema_version == 2 and patch.get("reserve_cash", money["reserve_cash"]) != money["reserve_cash"]:
            raise ValueError("科学研究不能改变现金预算")
        candidate = BacktestConfig.model_validate({**base.model_dump(mode="python"), **patch})
        rows.append([candidate.min_shares, np.inf if candidate.max_shares is None else candidate.max_shares, candidate.buy_step,
                     candidate.sell_step, candidate.grid_step, candidate.buy_shares, candidate.sell_shares])
    values = np.asarray(rows, dtype=np.float64).reshape((-1, 7))
    dates = _return_days(prepared)
    returns = np.empty((len(dates), len(values)), dtype=np.float64)
    metric_values = np.empty((len(values), 11), dtype=np.float64)
    fold_values = np.empty((len(values), len(fold_ranges), 9), dtype=np.float64)
    funding_stats = np.empty((len(values), 9), dtype=np.float64)
    options = _options(base, prepared, money, cost_multiplier=cost_multiplier)
    empty_days, empty_values = np.zeros(0, dtype=np.int64), np.zeros((0, 7), dtype=np.float64)
    old_threads = None
    active_threads = 1
    if engine._NUMBA_FASTPATH_ENABLED and parallel and len(values) > 1:
        from numba import get_num_threads, set_num_threads
        old_threads = get_num_threads()
        active_threads = min(old_threads, rebalance_cpu_budget())
        set_num_threads(active_threads)
    bytes_per_candidate = max(
        8192,
        len(dates) * 8 + max(1, len(fold_ranges)) * 72 + 128,
    )
    chunk_size = max(1, min(256, int(current_workspace_bytes() // (bytes_per_candidate * 4))))
    record_details(
        research_threads=active_threads,
        research_chunk_size=chunk_size,
    )
    try:
        for begin in range(0, len(values), chunk_size):
            if cancelled and cancelled():
                raise InterruptedError("研究已取消")
            end = min(begin + chunk_size, len(values))
            if old_threads is not None:
                refreshed = min(old_threads, rebalance_cpu_budget())
                if refreshed != active_threads:
                    active_threads = refreshed
                    set_num_threads(active_threads)
                with blas_thread_limit(1):
                    r, m, f, s = simulate_research_batch(
                        prepared["arrays"], options, values[begin:end],
                        fold_ids, len(fold_ranges),
                    )
                returns[:, begin:end], metric_values[begin:end], fold_values[begin:end], funding_stats[begin:end] = r, m, f, s
            else:
                runner = simulate if engine._NUMBA_FASTPATH_ENABLED else getattr(simulate, "py_func", simulate)
                for index in range(begin, end):
                    p = values[index]
                    individual = (*options[:3], p[0], p[1], options[5], p[2], p[3], p[4],
                                  *options[9:12], int(p[5]), int(p[6]), *options[14:])
                    output = runner(*prepared["arrays"], *individual, empty_days, empty_values,
                                    fold_ids, len(fold_ranges), True, False)
                    returns[:, index], metric_values[index], fold_values[index], funding_stats[index] = output[5][0], output[0][0], output[-2], output[-1][0]
            record_count("research_candidates", end - begin)
        if cancelled and cancelled():
            raise InterruptedError("研究已取消")
    finally:
        if old_threads is not None:
            from numba import set_num_threads
            set_num_threads(old_threads)
    metrics = []
    for row in metric_values:
        metric = {key: int(round(row[i])) if key in {"trade_count", "buy_count", "sell_count"} else float(row[i]) for i, key in enumerate(engine._METRIC_KEYS)}
        metric["initial_equity"] = money["initial_equity"]
        metrics.append(metric)
    keys = ("max_drawdown", "trade_count", "buy_count", "sell_count", "total_fees", "turnover", "profit", "principal", "end_assets")
    fold_metrics = [[{key: int(round(row[j])) if key.endswith("count") else float(row[j]) for j, key in enumerate(keys)} for row in candidate] for candidate in fold_values]
    return {"dates": dates, "daily_returns": returns, "fold_drawdowns": fold_values[:, :, 0],
            "fold_metrics": fold_metrics, "metrics": metrics, "funding": money,
            "funding_stats": funding_stats,
            "backend": "numba_batch" if old_threads is not None else "serial"}


def research_passive(config, data):
    from . import engine
    from .share_kernel import simulate
    prepared, money = _inputs(config, data), funding(config, data)
    runner = simulate if engine._NUMBA_FASTPATH_ENABLED else getattr(simulate, "py_func", simulate)
    output = runner(*prepared["arrays"], *_options(config, prepared, money),
                    np.zeros(0, dtype=np.int64), np.zeros((0, 7), dtype=np.float64),
                    np.full(len(prepared["days"]), -1, dtype=np.int64), 0, True, True)
    return {"dates": _return_days(prepared), "daily_returns": output[5][0], "funding": money}


def research_buy_hold(config, data):
    """Daily returns for the independent 1,000-share buy-and-hold account."""
    from . import engine
    from .share_kernel import simulate

    prepared, money = _inputs(config, data), funding(config, data)
    runner = simulate if engine._NUMBA_FASTPATH_ENABLED else getattr(simulate, "py_func", simulate)
    output = runner(*prepared["arrays"], *_options(config, prepared, money),
                    np.zeros(0, dtype=np.int64), np.zeros((0, 7), dtype=np.float64),
                    np.full(len(prepared["days"]), -1, dtype=np.int64), 0, False, True)
    return {"dates": _return_days(prepared), "daily_returns": output[5][2], "funding": money}


def run_passive_backtest(config, data, progress=None, cancelled=None, **kwargs):
    return run_share_backtest(config, data, progress=progress, cancelled=cancelled,
                              passive_grid=True, **kwargs)
