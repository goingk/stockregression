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
from .validation_models import QualificationAccount

_CACHE = OrderedDict()
BUY_HOLD_INITIAL_SHARES = 1000

# Minutes covered by one intraday bar.  This mirrors the engine's own
# ``open_timestamp`` rule table; the vectorised helper below applies exactly
# that rule to a whole bar frame instead of calling it once per bar.
_FREQUENCY_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60}
_NS_PER_MINUTE = 60_000_000_000
_NS_PER_DAY = 86_400_000_000_000

_QUALIFICATION_TRADE_REASONS = {
    4: "grid_buy",
    5: "grid_sell",
    16: "qualification_initial_cash",
    17: "qualification_scheduled_funding",
}

_QUALIFICATION_REJECTION_REASONS = {
    1: "limit_up_locked",
    2: "limit_down_locked",
    3: "below_lot",
    4: "position_limit",
    5: "insufficient_cash",
    6: "t_plus_1",
    7: "insufficient_sellable_shares",
    8: "participation_capacity_unavailable",
    9: "invalid_execution_price",
    10: "position_limit_or_below_lot",
    11: "non_tradable_bar",
    12: "parameter_update",
    13: "unavailable_day",
    16: "reverse_signal",
    18: "ttl_expired",
    19: "day_end_cancel",
}


def _open_timestamps_ns(stamps: pd.DatetimeIndex, frequencies: np.ndarray) -> np.ndarray:
    """Vectorised twin of ``engine.open_timestamp`` for a whole bar frame.

    The rule is the engine's: a daily bar opens at 09:30, an intraday bar
    opens at the start of its own session, and the bar that closes a morning
    or afternoon session opens at that session's boundary.  Every branch of
    the scalar rule is reproduced here, so the result is bit-identical to
    ``pd.Timestamp(engine.open_timestamp(ts, frequency)).value``.
    """
    timestamps_ns = stamps.asi8.astype(np.int64)
    count = len(timestamps_ns)
    day_ns = timestamps_ns // _NS_PER_DAY * _NS_PER_DAY
    time_ns = timestamps_ns - day_ns
    minutes = np.fromiter(
        (0 if value == "1d" else _FREQUENCY_MINUTES.get(value, 1) for value in frequencies),
        dtype=np.int64, count=count,
    )
    start_ns = timestamps_ns - minutes * _NS_PER_MINUTE
    start_day_ns = start_ns // _NS_PER_DAY * _NS_PER_DAY
    start_time_ns = start_ns - start_day_ns
    daily = minutes == 0
    morning_open = day_ns + (9 * 60 + 30) * _NS_PER_MINUTE
    afternoon_open = day_ns + 13 * 60 * _NS_PER_MINUTE
    morning_end = (11 * 60 + 30) * _NS_PER_MINUTE
    afternoon_end = 15 * 60 * _NS_PER_MINUTE
    open_timestamps_ns = start_ns.copy()
    open_timestamps_ns[daily] = morning_open[daily]
    morning_mask = (
        (~daily) & (time_ns <= morning_end)
        & (start_time_ns < (9 * 60 + 30) * _NS_PER_MINUTE)
    )
    open_timestamps_ns[morning_mask] = morning_open[morning_mask]
    afternoon_mask = (
        (~daily) & (time_ns > morning_end) & (time_ns <= afternoon_end)
        & (start_time_ns < 13 * 60 * _NS_PER_MINUTE)
    )
    open_timestamps_ns[afternoon_mask] = afternoon_open[afternoon_mask]
    return open_timestamps_ns


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
    from .engine import _prepare_actions
    fees = config.fees
    resolution = data.metadata.get("data_resolution")
    if resolution is not None:
        from .data_resolution import POLICY, _digest
        if (resolution.get("policy") != POLICY or resolution.get("symbol") != config.symbol
                or resolution.get("plan_hash") != _digest({key: value for key, value in resolution.items() if key != "plan_hash"})):
            raise ValueError("RESOLUTION_PLAN_MISMATCH: 冻结逐日计划校验失败")
    resolution_hash = resolution.get("plan_hash") if resolution else None
    key = (id(data), data.version, resolution_hash, config.start_date, config.end_date, config.frequency, config.data_policy, config.dca.frequency, fees.share_face_value, fees.legacy_sh_transfer_rate, fees.legacy_sh_min_transfer, fees.legacy_sz_transfer_rate)
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
    resolution = data.metadata.get("data_resolution")
    frozen = resolution is not None
    if config.data_policy and (not frozen or config.data_policy != resolution.get("policy")):
        raise ValueError("RESOLUTION_PLAN_MISMATCH: 配置要求的逐日策略与冻结快照不一致")
    if frozen and (resolution.get("policy") != "minute_verified_daily_then_skip_v1" or not resolution.get("plan_hash")):
        raise ValueError("RESOLUTION_PLAN_MISMATCH: 冻结逐日计划无效")
    resolution_by_day = {pd.Timestamp(item["date"]).normalize(): item for item in resolution.get("days", [])} if frozen else {}
    if frozen:
        from .data_resolution import MODES, POLICY, _digest
        if (resolution.get("policy") != POLICY or resolution.get("symbol") != config.symbol
                or resolution.get("plan_hash") != _digest({key: value for key, value in resolution.items() if key != "plan_hash"})
                or any(item.get("mode") not in MODES for item in resolution_by_day.values())):
            raise ValueError("RESOLUTION_PLAN_MISMATCH: 冻结逐日计划校验失败")
    observed = set(pd.to_datetime(frame.timestamp).dt.normalize()) if not frame.empty else set()
    additions = []
    for row in daily.to_dict("records"):
        day = pd.Timestamp(row["date"])
        plan = resolution_by_day.get(day)
        if frozen and plan is None:
            raise ValueError(f"RESOLUTION_PLAN_MISMATCH: {day.date()} 不在冻结逐日计划中")
        if plan and plan.get("mode") == "skip":
            continue
        if plan and plan.get("mode") == "daily":
            frame = frame[frame.timestamp.dt.normalize() != day]
            observed.discard(day)
        suspended = row.get("tradestatus") == 0 or row.get("volume", 0) == 0
        if suspended and not frame.empty:
            frame = frame[frame.timestamp.dt.normalize() != day]
        if suspended or day not in observed:
            expected_mode = plan.get("mode") if plan else None
            if not suspended and config.frequency != "1d" and expected_mode not in ("daily", "skip") and not any(x.get("date") == str(day.date()) and x.get("mode") == "daily" for x in data.quality.get("days", [])):
                raise ValueError(f"{day.date()} 缺少可回测行情")
            if expected_mode == "minute" and not suspended and day not in observed:
                raise ValueError(f"RESOLUTION_PLAN_MISMATCH: {day.date()} 冻结分钟行情缺失")
            if expected_mode == "daily" and not suspended:
                additions.append({**row, "timestamp": day + pd.Timedelta(hours=15), "frequency": "1d", "tradable": True, "volume": row["volume"], "is_skip": False})
            elif suspended:
                additions.append({**row, "timestamp": day + pd.Timedelta(hours=15), "frequency": "1d", "tradable": False, "volume": 0, "is_skip": False})
    if frozen:
        for day, plan in resolution_by_day.items():
            if plan.get("mode") != "skip":
                continue
            frame = frame[frame.timestamp.dt.normalize() != day]
            additions.append({"timestamp": day + pd.Timedelta(hours=15), "frequency": "1d", "open": np.nan,
                              "high": np.nan, "low": np.nan, "close": np.nan, "volume": 0.0,
                              "tradable": False, "is_synthetic": False, "is_skip": True})
    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    if frame.empty:
        raise ValueError("所选区间没有可回测行情")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    stamps = pd.DatetimeIndex(frame.timestamp)
    calendar_days = pd.DatetimeIndex(pd.to_datetime(data.calendar)).normalize()
    calendar_days = calendar_days[(calendar_days >= start) & (calendar_days <= end)].unique().sort_values()
    days = calendar_days if len(calendar_days) else stamps.normalize().unique().sort_values()
    codes = pd.factorize(stamps.normalize(), sort=True)[0].astype(np.int64)
    n = len(frame)
    first = np.r_[True, codes[1:] != codes[:-1]]
    tradable = frame.get("tradable", pd.Series(True, index=frame.index)).fillna(False).to_numpy(dtype=np.bool_)
    skip = frame.get("is_skip", pd.Series(False, index=frame.index, dtype=bool)).eq(True).to_numpy(dtype=np.bool_)
    last = np.zeros(n, dtype=np.bool_)
    valid = np.flatnonzero(tradable)
    if len(valid):
        last[valid[np.r_[codes[valid][:-1] != codes[valid][1:], True]]] = True
    keys = [period_key(day, config.dca.frequency) for day in days]
    new_period = np.r_[True, np.asarray(keys[1:]) != np.asarray(keys[:-1])]
    lookup = {pd.Timestamp(row["date"]): row for row in daily.to_dict("records")}
    daily_ups, daily_downs = np.full(len(days), np.nan), np.full(len(days), np.nan)
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
            daily_ups[day_id] = up if up else np.nan
            daily_downs[day_id] = down if down else np.nan
        stamp[day_id] = fee_rates(day.date(), config.symbol)[0]
        transfer[day_id], face[day_id], minimum[day_id] = transfer_fee_terms(day.date(), config.symbol, legacy_sh_transfer_rate=fees.legacy_sh_transfer_rate, legacy_sh_min_transfer=fees.legacy_sh_min_transfer, legacy_sz_transfer_rate=fees.legacy_sz_transfer_rate)
        bilateral[day_id] = day.date() < date(2008, 9, 19)
    # The daily bands are expanded onto the bars with one index gather.  The
    # previous per-day mask assignment rescanned the whole bar array once for
    # every trading day.  A session without a price limit keeps its NaN.
    ups, downs = daily_ups[codes], daily_downs[codes]
    for column, output in [("limit_up", ups), ("limit_down", downs)]:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            mask = np.isfinite(values) & np.isfinite(output)
            output[mask] = values[mask]
    opens = pd.to_numeric(frame.open, errors="coerce").to_numpy(dtype=np.float64)
    closes = pd.to_numeric(frame.close, errors="coerce").to_numpy(dtype=np.float64)
    if np.any(tradable & (~np.isfinite(opens) | ~np.isfinite(closes) | (opens <= 0) | (closes <= 0))):
        raise ValueError("行情价格无效")
    # Resolve split fold dates before window filtering: the vendor fold can
    # precede the actual ex-date while the requested window starts in between.
    actions = _prepare_actions(data, config.start_date, end.date())
    day_ids = {day.date(): index for index, day in enumerate(days)}
    frequency_column = frame.get("frequency")
    frequencies = (
        np.full(n, str(config.frequency), dtype=object) if frequency_column is None
        else frequency_column.astype(str).to_numpy()
    )
    policy_volume = pd.to_numeric(frame.volume, errors="coerce").to_numpy(dtype=np.float64)
    volume_unavailable = np.isnan(policy_volume)
    filled_volume = policy_volume.copy()
    filled_volume[volume_unavailable] = 0.0
    # Extend the array tuple at its end: existing research/preview consumers
    # use the stable price-band, opening-time and corporate-action positions.
    arrays = (opens, closes, filled_volume, tradable, ups, downs, stamps.asi8, _open_timestamps_ns(stamps, frequencies), codes, first & new_period[codes], last, stamp[codes], transfer[codes], face[codes], minimum[codes], bilateral[codes], preclose,
              np.asarray([-1 if a.record_date < config.start_date else day_ids.get(a.record_date, -2) for a in actions], dtype=np.int64), np.asarray([day_ids.get(a.ex_date, -2) for a in actions], dtype=np.int64), np.asarray([day_ids.get(a.pay_date, -2) for a in actions], dtype=np.int64), np.asarray([day_ids.get(a.stock_date, -2) for a in actions], dtype=np.int64), np.asarray([a.cash_per_share for a in actions], dtype=np.float64), np.asarray([a.bonus_ratio for a in actions], dtype=np.float64), skip)
    result = {"arrays": arrays, "days": days, "stamps": stamps, "tradable": tradable, "skip": skip, "opens": arrays[7], "period_keys": keys, "terms": terms, "data": data, "end": str(end.date()), "volume_unavailable": volume_unavailable}
    _CACHE[key] = result
    while len(_CACHE) > 2:
        _CACHE.popitem(last=False)
    return result


def _parameter_arrays(config, prepared, parameter_schedule):
    days, values, normalized = [], [], []
    ex_days = prepared["arrays"][18]
    action_bonus = prepared["arrays"][22]
    ex_groups = {}
    for action_index, ex_index in enumerate(ex_days):
        if ex_index >= 0:
            ex_groups.setdefault(int(ex_index), []).append(action_index)
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
        share_scale, price_scale = 1.0, 1.0
        first_post_decision = decision + pd.Timedelta(days=1)
        last_pre_effective = prepared["days"][index] - pd.Timedelta(days=1)
        action_price_factors = {}
        if first_post_decision <= last_pre_effective:
            from .price_basis import action_factors
            action_price_factors, issue = action_factors(
                prepared["data"], last_pre_effective,
                start=first_post_decision,
            )
            if issue:
                raise ValueError(f"换参日期之间的公司行动无效：{issue}")
            action_price_factors = {
                pd.Timestamp(item["ex_date"]).normalize(): float(item["factor"])
                for item in action_price_factors
            }
        for ex_index, action_indices in ex_groups.items():
            ex_date = prepared["days"][ex_index]
            # The effective day's conversion is still applied in the kernel
            # after schedule activation, so only intervening sessions count.
            if decision < ex_date < prepared["days"][index]:
                ratio = 1.0 + math.fsum(float(action_bonus[i]) for i in action_indices)
                share_scale *= ratio
                price_scale *= action_price_factors.get(ex_date, 1.0)
        min_value = candidate.min_shares * share_scale
        max_value = np.inf if candidate.max_shares is None else candidate.max_shares * share_scale
        days.append(index)
        values.append([min_value, max_value, candidate.buy_step,
                       candidate.sell_step, candidate.grid_step * price_scale,
                       candidate.buy_shares * share_scale,
                       candidate.sell_shares * share_scale,
                       1.0 if row.get("paused") else 0.0])
        normalized.append({**row, "effective_date": str(prepared["days"][index].date())})
    return np.asarray(days, dtype=np.int64), np.asarray(values, dtype=np.float64).reshape((-1, 8)), normalized


def _options(config, prepared, money, mode=0, cost_multiplier=1.0, allow_odd_lot_exit=False):
    terms, fees = prepared["terms"], config.fees
    floor = config.floor_penalty_config
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
        1 if config.schema_version == 3 else 0, bool(allow_odd_lot_exit),
        float(floor.eta_floor) if floor is not None else -1.0,
        float(floor.gamma_min) if floor is not None else 0.0,
        float(floor.floor_k_ref) if floor is not None else 1.0)


def _return_days(prepared):
    return prepared["days"][np.unique(prepared["arrays"][8][prepared["arrays"][10]])]


def run_share_backtest(config, data, progress=None, cancelled=None, summary_only=False, preview=False,
                       parameter_schedule=None, passive_grid=False, daily_details=False,
                       cancel_flag=None):
    from . import engine
    from .share_kernel import ACCOUNT_MODE_FULL, simulate
    if cancelled and cancelled():
        raise InterruptedError("回测已取消")
    prepared, money = _inputs(config, data), funding(config, data)
    terms, fees = prepared["terms"], config.fees
    if progress:
        progress(0.2, "已载入股数配置与锁定行情")
    mode = 0 if summary_only else 1 if preview else 3 if daily_details else 2
    options = _options(config, prepared, money, mode)
    change_days, change_values, schedule = _parameter_arrays(config, prepared, parameter_schedule)
    from .smart_dca import smart_dca_signals
    dca_budgets, dca_signals = smart_dca_signals(config, data, prepared["days"])
    runner = simulate if engine._NUMBA_FASTPATH_ENABLED else getattr(simulate, "py_func", simulate)
    output = runner(*prepared["arrays"], *options, change_days, change_values,
                    np.full(len(prepared["days"]), -1, dtype=np.int64), 0, False, passive_grid,
                    ACCOUNT_MODE_FULL, cancel_flag if preview else None, dca_budgets)
    if cancelled and cancelled():
        raise InterruptedError("回测已取消")
    metrics, eq, cash, shares, drawdown, periods, tb, ta, ts, tq, tp, tf, tc, th, tr, ttop, rb, rt, ra, rs, rr, rq, invested, _, floor_episode_grids, funding_stats = output
    strategies = {}
    return_days = _return_days(prepared)
    points = return_days + pd.Timedelta(hours=15) if preview or daily_details else prepared["stamps"]
    indices = np.arange(len(points)) if preview or daily_details else np.flatnonzero(prepared["tradable"])
    openings = pd.to_datetime(prepared["opens"]).strftime("%Y-%m-%dT%H:%M:%S").tolist() if not summary_only else []
    initial_equities = (
        money["initial_equity"], money["dca_initial_equity"],
        money["buy_hold_initial_equity"], money["initial_equity"],
    )
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
    for account, name in enumerate(("grid", "dca", "buy_hold", "matched_funding_hold")):
        metric = {
            "peak_principal_return": float(metrics[account, 0]) if math.isfinite(metrics[account, 0]) else None,
            "max_principal_drawdown": float(metrics[account, 2]) if math.isfinite(metrics[account, 2]) else None,
            "trade_count": int(round(metrics[account, 5])),
            "total_fees": float(metrics[account, 6]) if math.isfinite(metrics[account, 6]) else None,
            "turnover": float(metrics[account, 7]) if math.isfinite(metrics[account, 7]) else None,
            "buy_count": int(round(metrics[account, 8])),
            "sell_count": int(round(metrics[account, 9])),
            "return_basis": "peak_principal",
            "risk_basis": "principal_profit_drawdown",
        }
        metric["initial_equity"] = initial_equities[account]
        if metric.get("peak_principal_return") is None or metric.get("max_principal_drawdown") is None:
            metric["unevaluable_reason"] = "评价区间外部本金峰值为零"
        if account == 0:
            metric.update({
                "matched_funding_hold_return": float(funding_stats[0, 16]) if math.isfinite(funding_stats[0, 16]) else None,
                "matched_funding_hold_drawdown": float(funding_stats[0, 17]) if math.isfinite(funding_stats[0, 17]) else None,
                "total_invested": float(funding_stats[0, 4]),
                "external_topups": float(funding_stats[0, 1]),
                "total_profit": float(funding_stats[0, 6]),
                "matched_funding_hold_invested": float(funding_stats[0, 21]),
            })
        if account == 0:
            metric.update(
                actual_min_pos_ratio=float(funding_stats[0, 9]),
                actual_max_pos_ratio=float(funding_stats[0, 10]),
                stockout_count=int(round(funding_stats[0, 11])),
                tplus1_block_count=int(round(funding_stats[0, 12])),
            )
            if config.floor_penalty_config is not None:
                metric.update({
                    "floor_episode_count": int(round(funding_stats[0, 13])),
                    "floor_penalty": float(funding_stats[0, 14]),
                })
        if account == 1:
            metric.update(total_invested=float(metrics[account, 10]), skipped_periods=len(set(prepared["period_keys"])) - metric["buy_count"])
        if flow_aware and account == 0:
            metric.update({
                "initial_equity": float(funding_stats[0, 0]),
                "peak_invested": peak_invested,
                "total_invested": float(funding_stats[0, 4]),
                "external_topups": float(funding_stats[0, 1]),
                "total_profit": float(funding_stats[0, 6]),
                "peak_principal_return": float(metrics[account, 0]) if math.isfinite(metrics[account, 0]) else None,
                "max_principal_drawdown": float(metrics[account, 2]) if math.isfinite(metrics[account, 2]) else None,
                "return_basis": "peak_principal",
                "risk_basis": "principal_profit_drawdown",
            })
        if flow_aware and account == 3:
            metric.update({
                "peak_principal_return": float(metrics[account, 0]) if math.isfinite(metrics[account, 0]) else None,
                "max_principal_drawdown": float(metrics[account, 2]) if math.isfinite(metrics[account, 2]) else None,
                "total_invested": float(funding_stats[0, 21]) if funding_stats.shape[1] > 21 else None,
                "external_topups": float(funding_stats[0, 22]) if funding_stats.shape[1] > 22 else None,
                "return_basis": "peak_principal",
                "risk_basis": "principal_profit_drawdown",
            })
        equity = []
        if not summary_only:
            drawdown_denominator = {
                0: peak_invested,
                1: float(metrics[1, 10]),
                2: initial_equities[2],
                3: float(funding_stats[0, 23]),
            }[account]
            current_peak_principal = 0.0
            for i in indices:
                invested_at_point = float(invested[account, i])
                # Use the high-water mark reached by this point.  Funding
                # that arrives later must not dilute the return already
                # earned before it was contributed.
                current_peak_principal = max(current_peak_principal, invested_at_point)
                profit = float(eq[account, i]) - invested_at_point
                principal_curve = 1.0 + profit / current_peak_principal if current_peak_principal > 0 else None
                equity.append({
                    "timestamp": points[i].isoformat(), "equity": float(eq[account, i]),
                    "cash": float(cash[account, i]), "position": float(shares[account, i]),
                    "total_invested": invested_at_point, "peak_principal": current_peak_principal,
                    "cumulative_profit": profit, "principal_curve": principal_curve,
                    "principal_drawdown": float(drawdown[account, i]) if drawdown_denominator > 0 else None,
                })
        trades = [{"timestamp": openings[int(tb[j])], "side": "buy" if ts[j] == 1 else "sell", "quantity": int(tq[j]), "price": float(tp[j]), "fees": float(tf[j]), "cash": float(tc[j]), "shares": int(th[j]), "reason": engine._TRADE_REASONS[int(tr[j])], "external_topup": float(ttop[j]) if account == 0 else 0.0} for j in range(len(tb)) if ta[j] == account]
        rejection_names = {12: "parameter_update", 13: "unavailable_day", 14: "dca_signal_unavailable", 15: "dca_amount_below_lot"}
        rejections = [{"timestamp": pd.Timestamp(rt[j]).isoformat(), "side": "buy" if rs[j] == 1 else "sell", "reason": rejection_names.get(int(rr[j])) or engine._REJECTION_REASONS[int(rr[j])], "requested_quantity": int(rq[j])} for j in range(len(rb)) if ra[j] == account]
        if account == 1 and dca_signals:
            for row in [*trades, *rejections]:
                signal = dca_signals.get(row["timestamp"][:10])
                if signal:
                    row["dca_signal"] = signal
                    if row["reason"] == "dca_signal_unavailable":
                        row["reason"] = signal.get("reason", row["reason"])
                        row["message"] = signal.get("message")
            metric["signal_unavailable_days"] = sum(item["status"] == "skipped" for item in dca_signals.values())
        if account == 1 and not summary_only and not preview:
            bought = {period_key(t["timestamp"], config.dca.frequency) for t in trades}
            for key in dict.fromkeys(prepared["period_keys"]):
                if key not in bought:
                    day = [d for d, p in zip(prepared["days"], prepared["period_keys"]) if p == key][-1]
                    rejections.append({"timestamp": (day + pd.Timedelta(hours=15)).isoformat(), "side": "buy", "reason": "dca_period_skipped", "requested_quantity": config.dca.shares})
        strategy_basis = "peak_principal" if account in (0, 3) else "cost" if account == 1 else "initial_equity"
        strategy_result = {
            "metrics": metric, "equity": equity,
            "equity_frequency": "1d" if preview or daily_details else config.frequency,
            "trades": trades, "rejections": rejections, "return_basis": strategy_basis,
        }
        if account == 1 and dca_signals:
            strategy_result["dca_rule"] = "contrarian_linear_v1"
            strategy_result["signals"] = list(dca_signals.values()) if not summary_only else []
        if account == 0 and config.floor_penalty_config is not None:
            strategy_result.update({
                "floor_episode_count": int(round(funding_stats[0, 13])),
                "floor_episode_grids": [int(value) for value in floor_episode_grids],
                "floor_penalty": float(funding_stats[0, 14]),
                "floor_penalty_basis": "episode_grids_v1",
            })
        strategies[name] = strategy_result
    if progress:
        progress(1.0, "股数回测完成")
    warnings = list(dict.fromkeys([*data.warnings, *data.metadata.get("warnings", [])]))
    missing_signals = sum(item["status"] == "skipped" for item in dca_signals.values())
    if missing_signals:
        warnings.append(f"智能定投有 {missing_signals} 个交易日缺少可用的此前参考信号；到期时跳过并记录原因，不替换参考标的。")
    selection_asof = getattr(config, "selection_asof", None)
    retrospective = bool(selection_asof and config.start_date <= selection_asof)
    if retrospective:
        warnings.append(f"参数选定于 {selection_asof}；选参截止日及以前的历史重跑属于样本内回看，不是样本外成绩。")
    accounting_basis = "peak_principal_v2" if flow_aware else "independent_contributions_v1"
    result = {"schema_version": config.schema_version, "result_schema_version": 2, "accounting_basis": accounting_basis, "symbol": config.symbol, "config": config.model_dump(mode="json"), "data_version": data.version, "quality": data.quality, "is_demo": bool(data.metadata.get("is_demo")), "funding": money, "requested_end_date": str(config.end_date) if config.end_date else None, "effective_end_date": prepared["end"], "warnings": warnings, "strategies": strategies}
    if config.floor_penalty_config is not None:
        grid = strategies["grid"]
        result.update({
            "floor_episode_count": grid["floor_episode_count"],
            "floor_episode_grids": grid["floor_episode_grids"],
            "floor_penalty": grid["floor_penalty"],
            "floor_penalty_basis": grid["floor_penalty_basis"],
        })
    result["diagnostics"] = data.metadata.get("diagnostics", {})
    result["source_events"] = data.metadata.get("source_events", [])
    if schedule:
        result["parameter_schedule"] = schedule
    if selection_asof:
        result["parameter_selection"] = {"as_of": str(selection_asof), "retrospective": retrospective}
    if passive_grid:
        result["strategies"]["grid"]["label"] = "同资金同初始持股被动持有"
    return result


def run_qualification(
    config: BacktestConfig,
    data: DataBundle,
    account: QualificationAccount | dict,
    parameter_schedule: list | None = None,
    passive_grid: bool = False,
    cancelled=None,
    cost_multiplier: float = 1.0,
    volume_multiplier: float = 1.0,
    extra_delay_bars: int = 0,
    omitted_contribution_index: int | None = None,
    max_principal_drawdown: float | None = None,
) -> dict:
    """Run the qualification policy on the shared share-account kernel.

    The grid and the scheduled-funding hold are two independent ledgers inside
    the same native simulation.  They receive the same initial cash and the
    same dated contributions, while each has its own per-bar participation
    capacity.  Orders evaluate a close and execute at the next eligible bar's
    close with adverse slippage and tick rounding.
    """
    import threading

    from . import engine
    from .share_kernel import ACCOUNT_MODE_RESEARCH, simulate

    profile = account if isinstance(account, QualificationAccount) else QualificationAccount.model_validate(account)
    cost_multiplier = float(cost_multiplier)
    volume_multiplier = float(volume_multiplier)
    extra_delay_bars = int(extra_delay_bars)
    if not math.isfinite(cost_multiplier) or cost_multiplier <= 0:
        raise ValueError("成本倍率必须为正数")
    if not math.isfinite(volume_multiplier) or volume_multiplier <= 0:
        raise ValueError("成交量倍率必须为正数")
    if extra_delay_bars < 0:
        raise ValueError("额外延迟 bar 数不能为负")
    risk_limit = float(max_principal_drawdown if max_principal_drawdown is not None else getattr(config, "max_principal_drawdown", 0.2))
    if not math.isfinite(risk_limit) or not 0 < risk_limit <= 1:
        raise ValueError("资格回撤门槛必须大于0且不超过1")
    if cancelled and cancelled():
        raise InterruptedError("资格回测已取消")

    prepared = _inputs(config, data)
    money = funding(config, data)
    terms = prepared["terms"]
    initial_holdings_value = round_money(float(config.initial_shares) * float(money["valuation_price"]))
    all_contributions = list(profile.contributions)
    scheduled_total = round_money(math.fsum(item.amount for item in all_contributions))
    contributions = list(all_contributions)
    contribution_indices = list(range(len(all_contributions)))
    omitted_contribution_amount = 0.0
    if omitted_contribution_index is not None:
        if (isinstance(omitted_contribution_index, bool)
                or not isinstance(omitted_contribution_index, (int, np.integer))
                or not 0 <= int(omitted_contribution_index) < len(contributions)):
            raise ValueError("待忽略补款序号超出计划补款范围")
        omitted_contribution_index = int(omitted_contribution_index)
        omitted_contribution_amount = round_money(all_contributions[omitted_contribution_index].amount)
        contributions.pop(omitted_contribution_index)
        contribution_indices.pop(omitted_contribution_index)
    eligible_scheduled_total = round_money(scheduled_total - omitted_contribution_amount)
    initial_cash = round_money(profile.total_budget - initial_holdings_value - scheduled_total)
    if initial_cash < -0.005:
        raise ValueError("总预算不足以覆盖初始持仓和全部计划补款")
    initial_cash = max(0.0, initial_cash)

    stamps_ns = prepared["stamps"].asi8
    contribution_bars = []
    contribution_amounts = []
    for item in contributions:
        available_ns = pd.Timestamp(item.available_at).value
        bar = int(np.searchsorted(stamps_ns, available_ns, side="left"))
        contribution_bars.append(bar if bar < len(stamps_ns) else len(stamps_ns))
        contribution_amounts.append(float(item.amount))

    arrays = list(prepared["arrays"])
    arrays[2] = np.ascontiguousarray(arrays[2] * volume_multiplier, dtype=np.float64)
    volume_unavailable = np.ascontiguousarray(prepared.get("volume_unavailable", np.zeros(len(arrays[2]), dtype=np.bool_)), dtype=np.bool_)
    policy_config = config.model_copy(update={"max_shares": profile.max_shares})
    change_days, change_values, schedule = _parameter_arrays(policy_config, prepared, parameter_schedule)
    options = list(_options(config, prepared, money, 2, cost_multiplier=cost_multiplier))
    options[4] = float(profile.max_shares)
    fold_ids = np.full(len(prepared["days"]), -1, dtype=np.int64)
    effective_delay = int(profile.extra_delay_bars) + extra_delay_bars
    runner = simulate if engine._NUMBA_FASTPATH_ENABLED else getattr(simulate, "py_func", simulate)
    cancel_flag = np.zeros(1, dtype=np.int8) if cancelled else None
    stop_monitor = threading.Event() if cancelled else None
    monitor = None
    if cancelled:
        def watch_cancellation():
            while not stop_monitor.wait(.05):
                if cancelled():
                    cancel_flag[0] = 1
                    return
        monitor = threading.Thread(target=watch_cancellation, daemon=True)
        monitor.start()
    try:
        output = runner(
            *tuple(arrays), *tuple(options), change_days, change_values,
            fold_ids, 0, False, bool(passive_grid), ACCOUNT_MODE_RESEARCH,
            cancel_flag, None, True, float(initial_cash), float(profile.participation_rate),
            int(profile.ttl_bars), effective_delay,
            np.ascontiguousarray(contribution_bars, dtype=np.int64),
            np.ascontiguousarray(contribution_amounts, dtype=np.float64),
            volume_unavailable,
        )
    finally:
        if stop_monitor is not None:
            stop_monitor.set()
        if monitor is not None:
            monitor.join(timeout=1.0)
    if cancelled and (cancel_flag[0] or cancelled()):
        raise InterruptedError("资格回测已取消")
    (metrics, eq, cash_values, share_values, drawdown, _periods, trade_bar,
     trade_strategy, trade_side, trade_quantity, trade_price, trade_fees,
     trade_cash, trade_shares, trade_reason, _trade_topup, rejection_bar,
     rejection_ts, rejection_strategy, rejection_side, rejection_reason,
     rejection_quantity, invested_values, _folds, _episodes, _funding_stats) = output

    tradable = np.asarray(prepared["tradable"], dtype=np.bool_)
    skip = np.asarray(prepared["skip"], dtype=np.bool_)
    visible = np.flatnonzero(tradable & ~skip)
    bar_timestamps = prepared["stamps"]
    contribution_rows = []
    for index, item, bar, amount in zip(contribution_indices, contributions, contribution_bars, contribution_amounts):
        credited = bar < len(bar_timestamps)
        contribution_rows.append({
            "index": index,
            "available_at": pd.Timestamp(item.available_at).isoformat(),
            "amount": float(amount),
            "status": "credited" if credited else "outside_backtest",
            "credited_at": bar_timestamps[bar].isoformat() if credited else None,
            "bar": int(bar) if credited else None,
        })
    received_total = round_money(math.fsum(
        float(item["amount"]) for item in contribution_rows if item["status"] == "credited"
    ))
    start_equity = round_money(initial_holdings_value + initial_cash)
    volume_nan_count = int(np.count_nonzero(volume_unavailable))

    strategies = {}
    for account_id, name in ((0, "grid"), (3, "scheduled_funding_hold")):
        account_trades = []
        for offset in np.flatnonzero(trade_strategy == account_id):
            bar = int(trade_bar[offset])
            side = "buy" if int(trade_side[offset]) == 1 else "sell"
            account_trades.append({
                "timestamp": bar_timestamps[bar].isoformat(),
                "side": side,
                "quantity": int(trade_quantity[offset]),
                "price": float(trade_price[offset]),
                "fees": float(trade_fees[offset]),
                "notional": round_money(float(trade_quantity[offset]) * float(trade_price[offset])),
                "cash": float(trade_cash[offset]),
                "shares": int(trade_shares[offset]),
                "reason": _QUALIFICATION_TRADE_REASONS.get(int(trade_reason[offset]), "qualification_trade"),
                "execution": "next_bar_end",
            })
        account_rejections = []
        for offset in np.flatnonzero(rejection_strategy == account_id):
            reason = int(rejection_reason[offset])
            account_rejections.append({
                "timestamp": pd.Timestamp(int(rejection_ts[offset])).isoformat(),
                "side": "buy" if int(rejection_side[offset]) == 1 else "sell",
                "reason": _QUALIFICATION_REJECTION_REASONS.get(reason, f"qualification_rejection_{reason}"),
                "requested_quantity": int(rejection_quantity[offset]),
            })

        equity = []
        peak_principal = 0.0
        for bar in visible:
            invested = float(invested_values[account_id, bar])
            peak_principal = max(peak_principal, invested)
            profit = float(eq[account_id, bar]) - invested
            equity.append({
                "timestamp": bar_timestamps[bar].isoformat(),
                "equity": float(eq[account_id, bar]),
                "cash": float(cash_values[account_id, bar]),
                "position": float(share_values[account_id, bar]),
                "total_invested": invested,
                "peak_principal": peak_principal,
                "cumulative_profit": profit,
                "principal_curve": 1.0 + profit / peak_principal if peak_principal > 0 else None,
                "principal_drawdown": float(drawdown[account_id, bar]) if math.isfinite(drawdown[account_id, bar]) else None,
            })
        final_equity = float(eq[account_id, visible[-1]]) if len(visible) else start_equity
        final_invested = round_money(start_equity + received_total)
        total_profit = round_money(final_equity - final_invested)
        peak_metric = float(metrics[account_id, 0]) if math.isfinite(metrics[account_id, 0]) else None
        drawdown_metric = float(metrics[account_id, 2]) if math.isfinite(metrics[account_id, 2]) else None
        reason_counts = {}
        for item in account_rejections:
            reason_counts[item["reason"]] = reason_counts.get(item["reason"], 0) + 1
        metric = {
            "peak_principal_return": peak_metric,
            "max_principal_drawdown": drawdown_metric,
            "total_profit": total_profit,
            "trade_count": int(metrics[account_id, 5]),
            "external_topups": received_total,
            "return_on_committed_budget": total_profit / profile.total_budget if profile.total_budget > 0 else None,
            "total_budget": float(profile.total_budget),
            "initial_equity": start_equity,
            "initial_holdings_value": initial_holdings_value,
            "initial_cash": initial_cash,
            "scheduled_contributions": scheduled_total,
            "eligible_scheduled_contributions": eligible_scheduled_total,
            "omitted_contribution_amount": omitted_contribution_amount,
            "unreceived_contributions": round_money(eligible_scheduled_total - received_total),
            "total_invested": final_invested,
            "peak_invested": round_money(peak_principal),
            "final_equity": final_equity,
            "final_cash": float(cash_values[account_id, visible[-1]]) if len(visible) else initial_cash,
            "final_position": float(share_values[account_id, visible[-1]]) if len(visible) else float(config.initial_shares),
            "total_fees": float(metrics[account_id, 6]) if math.isfinite(metrics[account_id, 6]) else None,
            "turnover": float(metrics[account_id, 7]) if math.isfinite(metrics[account_id, 7]) else None,
            "buy_count": int(metrics[account_id, 8]),
            "sell_count": int(metrics[account_id, 9]),
            "return_basis": "peak_principal",
            "risk_basis": "principal_profit_drawdown",
            "funding_diagnostics": {
                "policy": "budget_includes_initial_holdings",
                "initial_cash_formula": "total_budget - initial_holdings_value - all_scheduled_contributions",
                "scheduled_receipts_linked_to_grid_fills": False,
                "omitted_contribution_index": omitted_contribution_index,
                "initial_shares_sellable_at_start": True,
                "cash_shortfall_action": "cancel_remainder",
                "automatic_top_up_on_fill": False,
                "unspent_cash": float(cash_values[account_id, visible[-1]]) if len(visible) else initial_cash,
            },
            "execution_diagnostics": {
                "model": "next_bar_end_close_participation_ttl_v1",
                "capacity_scope": "account_per_bar",
                "participation_rate": float(profile.participation_rate),
                "volume_multiplier": volume_multiplier,
                "effective_extra_delay_bars": effective_delay,
                "ttl_bars": int(profile.ttl_bars),
                "partial_fill_remainder_ttl_bars": int(profile.ttl_bars),
                "day_end_cancel": True,
                "fees_charged_per_independent_fill": True,
                "trade_fill_count": len(account_trades),
                "rejection_count": len(account_rejections),
                "rejections_by_reason": reason_counts,
                "volume_nan_bars": volume_nan_count,
            },
        }
        strategies[name] = {
            "metrics": metric,
            "equity": equity,
            "equity_frequency": config.frequency,
            "trades": account_trades,
            "rejections": account_rejections,
            "contributions": [dict(item) for item in contribution_rows],
        }

    grid_drawdown = strategies["grid"]["metrics"]["max_principal_drawdown"]
    funding_rejections_by_strategy = {
        name: int(strategy["metrics"]["execution_diagnostics"]
                  ["rejections_by_reason"].get("insufficient_cash", 0))
        for name, strategy in strategies.items()
    }
    grid_funding_rejections = funding_rejections_by_strategy["grid"]
    risk_breached = grid_drawdown is not None and grid_drawdown > risk_limit
    unevaluable_reasons = []
    if volume_nan_count:
        unevaluable_reasons.append(
            "存在成交量为 NaN 的行情 bar；保留原始成交量掩码，不能按零成交量推断资格"
        )
    if grid_drawdown is None:
        unevaluable_reasons.append("grid 最大本金回撤不可评价")
    unevaluable_reason = "；".join(unevaluable_reasons) or None
    if unevaluable_reason:
        qualification_status = "unevaluable"
    elif risk_breached:
        qualification_status = "risk_breached"
    elif grid_funding_rejections:
        qualification_status = "execution_limited"
    else:
        qualification_status = "qualified"
    qualification_basis = {
        "priority": ["unevaluable", "risk_breached", "execution_limited", "qualified"],
        "risk_limit": risk_limit,
        "grid_max_principal_drawdown": grid_drawdown,
        "risk_breached": risk_breached,
        "grid_insufficient_cash_rejections": grid_funding_rejections,
        "funding_rejections_by_strategy": funding_rejections_by_strategy,
        "volume_nan_bars": volume_nan_count,
    }
    if unevaluable_reason:
        for strategy in strategies.values():
            strategy["metrics"]["unevaluable_reason"] = unevaluable_reason

    execution_profile = {
        "version": "qualification_execution_v1",
        "decision_price": "bar_close",
        "execution": "next_bar_end",
        "execution_price": "next_bar_close",
        "slippage": "adverse",
        "price_tick": float(terms["price_tick"]),
        "participation_rate": float(profile.participation_rate),
        "volume_multiplier": volume_multiplier,
        "cost_multiplier": cost_multiplier,
        "ttl_bars": int(profile.ttl_bars),
        "extra_delay_bars": effective_delay,
        "partial_fill_remainder": "ttl_then_day_end_cancel_v1",
        "day_end_cancel": True,
        "same_account_capacity_shared": True,
        "independent_strategy_capacity": True,
        "insufficient_funds": "cancel_remainder_without_cash_resize",
        "automatic_top_up_on_fill": False,
        "fees_per_independent_fill": True,
        "passive_grid": bool(passive_grid),
        "funding_policy": "budget_includes_initial_holdings",
        "scheduled_receipts_independent_of_grid_fills": True,
    }
    funding_summary = {
        "policy": "budget_includes_initial_holdings",
        "total_budget": float(profile.total_budget),
        "initial_holdings_value": initial_holdings_value,
        "initial_cash": initial_cash,
        "scheduled_contributions": scheduled_total,
        "eligible_scheduled_contributions": eligible_scheduled_total,
        "omitted_contribution_amount": omitted_contribution_amount,
        "received_contributions": received_total,
        "external_topups": received_total,
        "valuation_price": float(money["valuation_price"]),
        "valuation_date": str(money.get("reference_date")),
        "initial_shares": int(config.initial_shares),
        "initial_cash_formula": "total_budget - initial_holdings_value - all_scheduled_contributions",
        "automatic_top_up_on_fill": False,
    }
    return {
        "status": "completed",
        "qualification_status": qualification_status,
        "unevaluable_reason": unevaluable_reason,
        "risk_limit": risk_limit,
        "qualification_basis": qualification_basis,
        "execution_profile": execution_profile,
        "config": config.model_dump(mode="json"),
        "account": profile.model_dump(mode="json"),
        "initial_cash": initial_cash,
        "initial_holdings_value": initial_holdings_value,
        "external_topups": received_total,
        "funding": funding_summary,
        "contributions": [dict(item) for item in contribution_rows],
        "volume_unavailable_bars": [
            bar_timestamps[index].isoformat()
            for index in np.flatnonzero(volume_unavailable)
        ],
        "parameter_schedule": schedule,
        "strategies": strategies,
    }


def research_batch(base, data, params, fold_ranges, cost_multiplier=1.0,
                   cancelled=None, parallel=True, allow_odd_lot_exit=False,
                   account_mode=None):
    """Evaluate candidates with exact intrabar fold risk.

    All folds are measurements on the same continuous candidate ledger.  The
    result contains peak-principal summaries only; no daily-return matrix is
    allocated.

    Only the grid ledger and its matched-funding baseline are scored here, so
    the kernel runs those two ledgers and skips the DCA and buy-and-hold
    accounts.  ``account_mode`` overrides that for equivalence checks; the
    default is the research mode of the kernel.
    """
    from . import engine
    from .models import BacktestConfig
    from .share_kernel import ACCOUNT_MODE_RESEARCH, simulate, simulate_research_batch
    if base.schema_version != 3:
        raise ValueError("科学研究需要 schema_version=3 的股数配置")
    if not math.isfinite(cost_multiplier) or cost_multiplier <= 0:
        raise ValueError("成本倍数必须为正数")
    mode = ACCOUNT_MODE_RESEARCH if account_mode is None else int(account_mode)
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
        candidate = BacktestConfig.model_validate({**base.model_dump(mode="python"), **patch})
        rows.append([candidate.min_shares, np.inf if candidate.max_shares is None else candidate.max_shares, candidate.buy_step,
                     candidate.sell_step, candidate.grid_step, candidate.buy_shares, candidate.sell_shares])
    values = np.asarray(rows, dtype=np.float64).reshape((-1, 7))
    dates = _return_days(prepared)
    metric_values = np.empty((len(values), 11), dtype=np.float64)
    fold_values = np.empty((len(values), len(fold_ranges), 9), dtype=np.float64)
    funding_stats = np.empty((len(values), 24), dtype=np.float64)
    episode_values = [np.empty(0, dtype=np.int64) for _ in values]
    options = _options(
        base, prepared, money, cost_multiplier=cost_multiplier,
        allow_odd_lot_exit=allow_odd_lot_exit,
    )
    empty_days, empty_values = np.zeros(0, dtype=np.int64), np.zeros((0, 7), dtype=np.float64)
    old_threads = None
    active_threads = 1
    if engine._NUMBA_FASTPATH_ENABLED and parallel and len(values) > 1:
        from numba import get_num_threads, set_num_threads
        old_threads = get_num_threads()
        active_threads = min(old_threads, rebalance_cpu_budget())
        set_num_threads(active_threads)
    bytes_per_candidate = max(8192, max(1, len(fold_ranges)) * 72 + 128)
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
                    m, f, e, s = simulate_research_batch(
                        prepared["arrays"], options, values[begin:end],
                        fold_ids, len(fold_ranges), mode,
                    )
                metric_values[begin:end], fold_values[begin:end], funding_stats[begin:end] = m, f, s
                for offset in range(end - begin):
                    episode_values[begin + offset] = e[offset]
            else:
                runner = simulate if engine._NUMBA_FASTPATH_ENABLED else getattr(simulate, "py_func", simulate)
                for index in range(begin, end):
                    p = values[index]
                    individual = (*options[:3], p[0], p[1], options[5], p[2], p[3], p[4],
                                  *options[9:12], int(p[5]), int(p[6]), *options[14:])
                    output = runner(*prepared["arrays"], *individual, empty_days, empty_values,
                                    fold_ids, len(fold_ranges), False, False, mode)
                    metric_values[index], fold_values[index], funding_stats[index] = output[0][0], output[-3], output[-1][0]
                    episode_values[index] = output[-2]
            record_count("research_candidates", end - begin)
        if cancelled and cancelled():
            raise InterruptedError("研究已取消")
    finally:
        if old_threads is not None:
            from numba import set_num_threads
            set_num_threads(old_threads)
    metrics = []
    for index, row in enumerate(metric_values):
        metric = {
            "peak_principal_return": float(funding_stats[index, 18]),
            "max_principal_drawdown": float(funding_stats[index, 19]),
            "matched_funding_hold_return": float(funding_stats[index, 16]),
            "matched_funding_hold_drawdown": float(funding_stats[index, 17]),
            "trade_count": int(round(row[5])),
            "total_fees": float(row[6]),
            "turnover": float(row[7]),
            "buy_count": int(round(row[8])),
            "sell_count": int(round(row[9])),
        }
        metric["initial_equity"] = money["initial_equity"]
        metric["total_invested"] = float(funding_stats[index, 4])
        metric["external_topups"] = float(funding_stats[index, 1])
        metric["total_profit"] = float(funding_stats[index, 6])
        metric["matched_funding_hold_invested"] = float(funding_stats[index, 21])
        metric["return_basis"] = "peak_principal"
        metric["risk_basis"] = "principal_profit_drawdown"
        metric.update({
            "actual_min_pos_ratio": float(funding_stats[index, 9]),
            "actual_max_pos_ratio": float(funding_stats[index, 10]),
            "stockout_count": int(round(funding_stats[index, 11])),
            "tplus1_block_count": int(round(funding_stats[index, 12])),
        })
        if base.floor_penalty_config is not None:
            metric.update({
                "floor_episode_count": int(round(funding_stats[index, 13])),
                "floor_penalty": float(funding_stats[index, 14]),
            })
        metrics.append(metric)
    keys = ("principal_drawdown", "trade_count", "buy_count", "sell_count", "total_fees", "turnover", "profit", "principal", "end_assets")
    fold_metrics = [[{key: int(round(row[j])) if key.endswith("count") else float(row[j]) for j, key in enumerate(keys)} for row in candidate] for candidate in fold_values]
    position_stats = [{
        "actual_min_pos_ratio": float(row[9]),
        "actual_max_pos_ratio": float(row[10]),
        "stockout_count": int(round(row[11])),
        "tplus1_block_count": int(round(row[12])),
        **({
            "floor_episode_count": int(round(row[13])),
            "floor_episode_grids": [int(value) for value in episode_values[index]],
            "floor_penalty": float(row[14]),
            "floor_penalty_basis": "episode_grids_v1",
        } if base.floor_penalty_config is not None else {}),
    } for index, row in enumerate(funding_stats)]
    return {"dates": dates, "fold_drawdowns": fold_values[:, :, 0],
            "fold_metrics": fold_metrics, "metrics": metrics, "funding": money,
            "funding_stats": funding_stats, "position_stats": position_stats,
            "matched_funding_returns": funding_stats[:, 16],
            "backend": "numba_batch" if old_threads is not None else "serial"}


def run_passive_backtest(config, data, progress=None, cancelled=None, **kwargs):
    return run_share_backtest(config, data, progress=progress, cancelled=cancelled,
                              passive_grid=True, **kwargs)
