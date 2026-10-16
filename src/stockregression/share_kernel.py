"""Share-based ledger kernel; the same function runs interpreted and with Numba.

Version-one cash/ratio results retain their original engine. This kernel seeds
existing holdings and funds fixed-share DCA only on successful executions.
"""
from __future__ import annotations
import math
import numpy as np
from .engine import _jit, _nb_round_money

@_jit(cache=True)
def _nb_buy_lot(quantity, board_code):
    result = max(0, int(math.floor(quantity / 100))) * 100
    return 0 if board_code == 2 and result < 200 else result

@_jit(cache=True)
def _nb_sell_lot(requested, available, board_code):
    return max(0, min(requested, available) // 100 * 100)

@_jit(cache=True)
def _nb_cell(price, anchor, grid_mode, step):
    if price <= 0 or anchor <= 0:
        return 0
    if grid_mode == 1:
        return int(math.floor((price - anchor) / max(anchor * step, 0.001) + 1e-10))
    return int(math.floor(math.log(price / anchor) / math.log(1 + step) + 1e-10))

@_jit(cache=True)
def _nb_attempt(
    side: int, requested: int, opening: float, volume: float,
    limit_up: float, limit_down: float, stamp_rate: float, transfer_rate: float,
    transfer_is_face_value: bool, transfer_minimum: float,
    stamp_bilateral: bool, day_id: int, cash: float, cash_receivable: float,
    share_receivable: float, shares: int, sellable: int,
    max_shares: float, min_shares: float, commission_rate: float,
    min_commission: float, slippage: float, board_code: int, face_value: float, tick: float, t0: bool,
) -> tuple:
    """Return fill state and a compact rejection code; side 1=buy, 2=sell."""
    if volume <= 0 or not math.isfinite(volume):
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 8
    if side == 1 and math.isfinite(limit_up) and opening >= limit_up - tick / 2:
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 1
    if side == 2 and math.isfinite(limit_down) and opening <= limit_down + tick / 2:
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 2

    raw_price = opening * (1.0 + slippage if side == 1 else 1.0 - slippage)
    if side == 1:
        price = math.ceil(raw_price / tick - 1e-10) * tick
    else:
        price = math.floor(raw_price / tick + 1e-10) * tick
    price = round(price, 6)
    if math.isfinite(limit_up):
        price = min(price, limit_up)
    if math.isfinite(limit_down):
        price = max(price, limit_down)
    if price <= 0 or not math.isfinite(price):
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 9

    position_shares = shares + share_receivable
    equity_open = max(0.0, cash + cash_receivable + position_shares * opening)
    if side == 1:
        normalized_request = _nb_buy_lot(min(max(0, requested), 2147483647), board_code)
        max_by_position = max(0.0, max_shares - position_shares)
        position_quantity = _nb_buy_lot(max_by_position, board_code)
        effective_transfer_rate = transfer_rate * face_value / price if transfer_is_face_value else transfer_rate
        max_by_cash = max(0.0, cash - min_commission - transfer_minimum) / (
            price * (1.0 + commission_rate + effective_transfer_rate)
        )
        affordable_quantity = _nb_buy_lot(max_by_cash, board_code)
        quantity = min(normalized_request, position_quantity, affordable_quantity)
        if quantity <= 0:
            if normalized_request <= 0:
                reason = 3
            elif position_quantity <= 0:
                reason = 4
            else:
                reason = 5
            return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, normalized_request, reason
    else:
        normalized_request = max(0, requested)
        min_shares_shares = max(0.0, min_shares)
        position_capacity = max(0, int(math.floor(position_shares - min_shares_shares + 1e-9)))
        sellable_capacity = min(sellable, position_capacity)
        quantity = _nb_sell_lot(min(normalized_request, sellable_capacity), sellable_capacity, board_code)
        if quantity <= 0:
            if sellable <= 0 and shares > 0:
                reason = 6
            elif position_capacity <= 0:
                reason = 4
            elif normalized_request <= 0:
                reason = 3
            else:
                reason = 7
            return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, normalized_request, reason

    # Execution prices are tick-rounded and quantities are integer shares, so
    # the exchange notional is exact cents even when the binary product is not.
    notional = 0.0
    commission = 0.0
    stamp = 0.0
    transfer = 0.0
    fee = 0.0
    cash_after = cash
    while quantity > 0:
        notional = _nb_round_money(quantity * price)
        commission = _nb_round_money(max(min_commission, notional * commission_rate))
        stamp = _nb_round_money(notional * stamp_rate) if side == 2 or stamp_bilateral else 0.0
        transfer_basis = quantity * face_value if transfer_is_face_value else notional
        transfer = _nb_round_money(max(transfer_minimum, transfer_basis * transfer_rate))
        fee = _nb_round_money(commission + stamp + transfer)
        cash_after = _nb_round_money(cash - notional - fee) if side == 1 else cash
        if side != 1 or cash_after >= -0.005:
            break
        quantity -= 100
        if board_code == 2 and quantity < 200:
            quantity = 0
    if quantity <= 0:
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, normalized_request, 5
    if side == 1:
        return True, max(0.0, cash_after), shares + quantity, sellable + (quantity if t0 else 0), quantity, price, fee, notional, normalized_request, 0
    cash_after = _nb_round_money(cash + notional - fee)
    return True, cash_after, shares - quantity, sellable - quantity, quantity, price, fee, notional, normalized_request, 0


@_jit(cache=True)
def simulate(
    opens: np.ndarray, closes: np.ndarray, volumes: np.ndarray,
    tradable_mask: np.ndarray,
    limit_ups: np.ndarray, limit_downs: np.ndarray,
    timestamps_ns: np.ndarray, open_timestamps_ns: np.ndarray,
    day_ids: np.ndarray, first_dca_day: np.ndarray, last_bar_day: np.ndarray,
    stamp_rates: np.ndarray, transfer_rates: np.ndarray,
    transfer_is_face_value: np.ndarray, transfer_minimums: np.ndarray,
    stamp_bilateral: np.ndarray,
    preclose_by_day: np.ndarray, action_record_day: np.ndarray,
    action_ex_day: np.ndarray, action_pay_day: np.ndarray, action_stock_day: np.ndarray,
    action_cash_per_share: np.ndarray, action_bonus_ratio: np.ndarray,
    grid_initial_equity: float, dca_initial_equity: float,
    buy_hold_initial_equity: float, min_shares: float, max_shares: float,
    grid_mode: int, buy_step: float, sell_step: float, grid_step: float,
    reference_price: float, lower_price: float, upper_price: float,
    buy_shares: int, sell_shares: int,
    board_code: int, commission_rate: float,
    min_commission: float, slippage: float, face_value: float, seed_shares: int,
    reserve_cash: float, dca_initial_shares: int, dca_shares: int,
    buy_hold_initial_shares: int, valuation_price: float, tick: float,
    t0: bool, output_mode: int,
) -> tuple:
    keep_details = output_mode == 2
    n = len(opens)
    day_count = 0
    for flag in last_bar_day:
        if flag:
            day_count += 1
    initial_equities = np.array([grid_initial_equity, dca_initial_equity, buy_hold_initial_equity], dtype=np.float64)
    cash = initial_equities.copy()
    shares = np.zeros(3, dtype=np.int64)
    sellable = np.zeros(3, dtype=np.int64)
    cash_receivable = np.zeros(3, dtype=np.float64)
    share_receivable = np.zeros(3, dtype=np.float64)
    started = np.zeros(3, dtype=np.bool_)
    reference = np.zeros(3, dtype=np.float64)
    fixed_anchor = np.zeros(3, dtype=np.float64)
    active_cell = np.zeros(3, dtype=np.int64)
    has_active_cell = np.zeros(3, dtype=np.bool_)
    pending_side = np.zeros(3, dtype=np.int64)
    pending_quantity = np.zeros(3, dtype=np.int64)
    pending_reason = np.zeros(3, dtype=np.int64)
    dca_due = False
    dca_last_day = -1
    action_count = len(action_record_day)
    entitlement_cash = np.zeros((3, action_count), dtype=np.float64)
    entitlement_bonus = np.zeros((3, action_count), dtype=np.float64)
    accrued_cash = np.zeros((3, action_count), dtype=np.float64)
    accrued_bonus = np.zeros((3, action_count), dtype=np.float64)
    accrued_exists = np.zeros((3, action_count), dtype=np.bool_)
    action_paid = np.zeros((3, action_count), dtype=np.bool_)
    action_stock_issued = np.zeros((3, action_count), dtype=np.bool_)
    prior_close = 0.0
    trade_count = np.zeros(3, dtype=np.int64)
    buy_count = np.zeros(3, dtype=np.int64)
    sell_count = np.zeros(3, dtype=np.int64)
    total_fees = np.zeros(3, dtype=np.float64)
    gross_turnover = np.zeros(3, dtype=np.float64)
    peak_equity = initial_equities.copy()
    max_drawdown = np.zeros(3, dtype=np.float64)
    last_equity = initial_equities.copy()
    daily_return_sum = np.zeros(3, dtype=np.float64)
    daily_return_sumsq = np.zeros(3, dtype=np.float64)
    prior_day_equity = initial_equities.copy()
    daily_return_count = 0
    first_timestamp = 0
    last_timestamp = 0
    for index in range(n):
        if tradable_mask[index]:
            if first_timestamp == 0:
                first_timestamp = timestamps_ns[index]
            last_timestamp = timestamps_ns[index]

    if keep_details:
        period_values = np.zeros((3, day_count), dtype=np.float64)
        trade_bar = np.zeros(3 * n, dtype=np.int64)
        trade_strategy = np.zeros(3 * n, dtype=np.int8)
        trade_side = np.zeros(3 * n, dtype=np.int8)
        trade_quantity = np.zeros(3 * n, dtype=np.int64)
        trade_price = np.zeros(3 * n, dtype=np.float64)
        trade_fees = np.zeros(3 * n, dtype=np.float64)
        trade_cash = np.zeros(3 * n, dtype=np.float64)
        trade_shares = np.zeros(3 * n, dtype=np.int64)
        trade_reason = np.zeros(3 * n, dtype=np.int8)
        trade_n = 0
        rejection_bar = np.zeros(5 * n, dtype=np.int64)
        rejection_ts = np.zeros(5 * n, dtype=np.int64)
        rejection_strategy = np.zeros(5 * n, dtype=np.int8)
        rejection_side = np.zeros(5 * n, dtype=np.int8)
        rejection_reason = np.zeros(5 * n, dtype=np.int8)
        rejection_quantity = np.zeros(5 * n, dtype=np.int64)
        rejection_n = 0
    else:
        # Only a few hundred/thousand session returns are required to preserve
        # the exact Sharpe calculation in summary-only optimization runs.
        period_values = np.zeros((3, day_count), dtype=np.float64)
        trade_bar = np.zeros(0, dtype=np.int64)
        trade_strategy = np.zeros(0, dtype=np.int8)
        trade_side = np.zeros(0, dtype=np.int8)
        trade_quantity = np.zeros(0, dtype=np.int64)
        trade_price = np.zeros(0, dtype=np.float64)
        trade_fees = np.zeros(0, dtype=np.float64)
        trade_cash = np.zeros(0, dtype=np.float64)
        trade_shares = np.zeros(0, dtype=np.int64)
        trade_reason = np.zeros(0, dtype=np.int8)
        trade_n = 0
        rejection_bar = np.zeros(0, dtype=np.int64)
        rejection_ts = np.zeros(0, dtype=np.int64)
        rejection_strategy = np.zeros(0, dtype=np.int8)
        rejection_side = np.zeros(0, dtype=np.int8)
        rejection_reason = np.zeros(0, dtype=np.int8)
        rejection_quantity = np.zeros(0, dtype=np.int64)
        rejection_n = 0

    cash[0] = reserve_cash
    shares[0] = seed_shares
    sellable[0] = seed_shares
    started[0] = True
    reference[0] = reference_price
    fixed_anchor[0] = reference_price
    has_active_cell[0] = True
    cash[1] = 0.0
    shares[1] = dca_initial_shares
    sellable[1] = dca_initial_shares
    cash[2] = 0.0
    shares[2] = buy_hold_initial_shares
    sellable[2] = buy_hold_initial_shares
    started[2] = True
    dca_invested = dca_initial_equity
    prior_close = valuation_price
    initial_holdings = (seed_shares, dca_initial_shares, buy_hold_initial_shares)
    for action in range(action_count):
        if action_record_day[action] == -1 and action_ex_day[action] >= 0:
            for account in range(3):
                entitlement_cash[account, action] = initial_holdings[account] * action_cash_per_share[action]
                entitlement_bonus[account, action] = initial_holdings[account] * action_bonus_ratio[action]
    output_count = n if output_mode == 2 else (day_count if output_mode == 1 else 0)
    equity_values = np.zeros((3, output_count), dtype=np.float64)
    equity_cash = np.zeros((3, output_count), dtype=np.float64)
    equity_shares = np.zeros((3, output_count), dtype=np.float64)
    equity_drawdown = np.zeros((3, output_count), dtype=np.float64)
    invested_values = np.zeros(output_count, dtype=np.float64)
    current_day_id = -1
    dca_day_index = 0
    for i in range(n):
        day_id = day_ids[i]
        if day_id != current_day_id:
            current_day_id = day_id
            for account in range(3):
                sellable[account] = shares[account]
            cash_delta = 0.0
            bonus_delta = 0.0
            has_ex_action = False
            for action in range(action_count):
                if action_ex_day[action] == day_id:
                    has_ex_action = True
                    cash_delta += action_cash_per_share[action]
                    bonus_delta += action_bonus_ratio[action]
            factor = 1.0
            if has_ex_action and prior_close > 0:
                if math.isfinite(preclose_by_day[day_id]) and preclose_by_day[day_id] > 0:
                    factor = preclose_by_day[day_id] / prior_close
                else:
                    factor = max(0.0, (prior_close - cash_delta) / (prior_close * (1.0 + bonus_delta)))
                if factor <= 0 or not math.isfinite(factor):
                    raise ValueError("公司行动除权参考价无效")
            for account in range(3):
                if has_ex_action and factor != 1.0 and account == 0:
                    if reference[account] > 0:
                        reference[account] *= factor
                    if fixed_anchor[account] > 0:
                        fixed_anchor[account] *= factor
                for action in range(action_count):
                    if action_ex_day[action] == day_id:
                        cash_claim = entitlement_cash[account, action]
                        bonus_claim = entitlement_bonus[account, action]
                        cash_receivable[account] += cash_claim
                        share_receivable[account] += bonus_claim
                        accrued_cash[account, action] = cash_claim
                        accrued_bonus[account, action] = bonus_claim
                        accrued_exists[account, action] = True
                        entitlement_cash[account, action] = 0.0
                        entitlement_bonus[account, action] = 0.0
                for action in range(action_count):
                    if action_pay_day[action] == day_id and accrued_exists[account, action] and not action_paid[account, action]:
                        amount = accrued_cash[account, action]
                        cash_receivable[account] = max(0.0, cash_receivable[account] - amount)
                        cash[account] = _nb_round_money(cash[account] + amount)
                        action_paid[account, action] = True
                    if action_stock_day[action] == day_id and accrued_exists[account, action] and not action_stock_issued[account, action]:
                        amount = accrued_bonus[account, action]
                        share_receivable[account] = max(0.0, share_receivable[account] - amount)
                        issued = int(round(amount))
                        shares[account] += issued
                        sellable[account] += issued
                        action_stock_issued[account, action] = True
        if first_dca_day[i]:
            dca_due = True

        # Observed non-tradable bars are real opening events: an order waiting
        # for this open expires here, while the bar itself must not generate a
        # trade, a close signal, an equity mark, or a period-return observation.
        if not tradable_mask[i]:
            for account in range(3):
                if pending_side[account] != 0:
                    if keep_details:
                        rejection_bar[rejection_n] = i
                        rejection_ts[rejection_n] = open_timestamps_ns[i]
                        rejection_strategy[rejection_n] = account
                        rejection_side[rejection_n] = pending_side[account]
                        rejection_reason[rejection_n] = 11
                        rejection_quantity[rejection_n] = pending_quantity[account]
                        rejection_n += 1
                    pending_side[account] = 0
                    pending_quantity[account] = 0
                    pending_reason[account] = 0
            continue

        opening = opens[i]
        close = closes[i]
        volume = volumes[i]
        for account in range(3):
            traded = False
            if pending_side[account] != 0:
                side = pending_side[account]
                quantity = pending_quantity[account]
                reason_code = pending_reason[account]
                pending_side[account] = 0
                result = _nb_attempt(
                    side, quantity, opening, volume, limit_ups[i], limit_downs[i],
                    stamp_rates[i], transfer_rates[i], transfer_is_face_value[i],
                    transfer_minimums[i], stamp_bilateral[i], day_id, cash[account],
                    cash_receivable[account], share_receivable[account],
                    shares[account], sellable[account], max_shares, min_shares,
                    commission_rate, min_commission, slippage, board_code, face_value, tick, t0,
                )
                filled = result[0]
                if not filled:
                    if keep_details:
                        rejection_bar[rejection_n] = i
                        rejection_ts[rejection_n] = open_timestamps_ns[i]
                        rejection_strategy[rejection_n] = account
                        rejection_side[rejection_n] = side
                        rejection_reason[rejection_n] = result[9]
                        rejection_quantity[rejection_n] = result[8]
                        rejection_n += 1
                else:
                    cash[account], shares[account], sellable[account] = result[1], result[2], result[3]
                    trade_count[account] += 1
                    buy_count[account] += side == 1
                    sell_count[account] += side == 2
                    total_fees[account] += result[6]
                    gross_turnover[account] += result[7]
                    if keep_details:
                        trade_bar[trade_n] = i
                        trade_strategy[trade_n] = account
                        trade_side[trade_n] = side
                        trade_quantity[trade_n] = result[4]
                        trade_price[trade_n] = result[5]
                        trade_fees[trade_n] = result[6]
                        trade_cash[trade_n] = cash[account]
                        trade_shares[trade_n] = shares[account]
                        trade_reason[trade_n] = reason_code
                        trade_n += 1
                    traded = True
                    if account == 0:
                        if grid_mode == 0:
                            reference[account] = result[5]
                        else:
                            if fixed_anchor[account] == 0:
                                fixed_anchor[account] = reference_price if reference_price > 0 else result[5]
                            if reference[account] == 0:
                                reference[account] = fixed_anchor[account]
                            active_cell[account] = _nb_cell(result[5], fixed_anchor[account], grid_mode, grid_step)
                            has_active_cell[account] = True

            if not traded and account == 1 and dca_due and dca_last_day != day_id:
                dca_last_day = day_id
                if opening > 0:
                    quantity = _nb_buy_lot(dca_shares, board_code)
                    result = _nb_attempt(
                        1, quantity, opening, volume, limit_ups[i], limit_downs[i],
                        stamp_rates[i], transfer_rates[i], transfer_is_face_value[i],
                        transfer_minimums[i], stamp_bilateral[i], day_id, 1e16,
                        cash_receivable[account], share_receivable[account],
                        shares[account], sellable[account], 2147483647.0, 0.0,
                        commission_rate, min_commission, slippage, board_code, face_value, tick, t0,
                    )
                    if not result[0]:
                        if keep_details:
                            rejection_bar[rejection_n] = i
                            rejection_ts[rejection_n] = open_timestamps_ns[i]
                            rejection_strategy[rejection_n] = account
                            rejection_side[rejection_n] = 1
                            rejection_reason[rejection_n] = result[9]
                            rejection_quantity[rejection_n] = result[8]
                            rejection_n += 1
                    else:
                        shares[account], sellable[account] = result[2], result[3]
                        dca_invested += result[7] + result[6]
                        dca_due = False
                        trade_count[account] += 1
                        buy_count[account] += 1
                        total_fees[account] += result[6]
                        gross_turnover[account] += result[7]
                        if keep_details:
                            trade_bar[trade_n] = i
                            trade_strategy[trade_n] = account
                            trade_side[trade_n] = 1
                            trade_quantity[trade_n] = result[4]
                            trade_price[trade_n] = result[5]
                            trade_fees[trade_n] = result[6]
                            trade_cash[trade_n] = cash[account]
                            trade_shares[trade_n] = shares[account]
                            trade_reason[trade_n] = 2
                            trade_n += 1
                        traded = True

        # Submit at most one grid order from the real close; it becomes eligible
        # at the next real bar's open.  Position sizing is frozen at this close.
        if started[0] and volumes[i] > 0:
            position_close = shares[0] + share_receivable[0]
            equity_close = cash[0] + cash_receivable[0] + position_close * close
            intent_side = 0
            intent_quantity = 0
            intent_reason = 0
            if grid_mode == 0:
                if reference[0] > 0 and close <= reference[0] * (1.0 - buy_step) and position_close < max_shares:
                    intent_side = 1
                    intent_quantity = buy_shares
                    intent_reason = 4
                elif reference[0] > 0 and close >= reference[0] * (1.0 + sell_step) and position_close > min_shares:
                    intent_side = 2
                    intent_quantity = sell_shares
                    intent_reason = 5
            else:
                anchor = fixed_anchor[0] if fixed_anchor[0] > 0 else reference[0]
                index = _nb_cell(close, anchor, grid_mode, grid_step)
                if not has_active_cell[0]:
                    active_cell[0] = _nb_cell(reference[0] if reference[0] > 0 else close, anchor, grid_mode, grid_step)
                    has_active_cell[0] = True
                lower_ok = lower_price <= 0 or close >= lower_price
                upper_ok = upper_price <= 0 or close <= upper_price
                if index < active_cell[0] and lower_ok and position_close < max_shares:
                    intent_side = 1
                    intent_quantity = buy_shares
                    intent_reason = 4
                elif index > active_cell[0] and upper_ok and position_close > min_shares:
                    intent_side = 2
                    intent_quantity = sell_shares
                    intent_reason = 5
            if intent_side == 1 and lower_price > 0 and close < lower_price:
                intent_side = 0
            if intent_side == 2 and upper_price > 0 and close > upper_price:
                intent_side = 0
            if intent_side != 0:
                clipped_quantity = intent_quantity
                if intent_side == 1:
                    maximum = max(0.0, max_shares - position_close)
                    clipped_quantity = min(intent_quantity, _nb_buy_lot(maximum, board_code))
                    clipped_quantity = _nb_buy_lot(clipped_quantity, board_code)
                else:
                    max_sell = max(0.0, position_close - min_shares)
                    clipped_quantity = min(intent_quantity, int(max_sell) // 100 * 100)
                if clipped_quantity <= 0:
                    if keep_details:
                        rejection_bar[rejection_n] = i
                        rejection_ts[rejection_n] = timestamps_ns[i]
                        rejection_strategy[rejection_n] = 0
                        rejection_side[rejection_n] = intent_side
                        rejection_reason[rejection_n] = 10
                        rejection_quantity[rejection_n] = intent_quantity
                        rejection_n += 1
                else:
                    pending_side[0] = intent_side
                    pending_quantity[0] = clipped_quantity
                    pending_reason[0] = intent_reason

        for account in range(3):
            equity = cash[account] + cash_receivable[account] + (shares[account] + share_receivable[account]) * close
            last_equity[account] = equity
            if equity > peak_equity[account]:
                peak_equity[account] = equity
            drawdown = (peak_equity[account] - equity) / peak_equity[account] if peak_equity[account] > 0 else 0.0
            if drawdown > max_drawdown[account]:
                max_drawdown[account] = drawdown
            if output_mode == 2 or (output_mode == 1 and last_bar_day[i]):
                out = i if output_mode == 2 else dca_day_index
                equity_values[account, out] = equity
                equity_cash[account, out] = cash[account]
                equity_shares[account, out] = shares[account] + share_receivable[account]
                equity_drawdown[account, out] = drawdown
                invested_values[out] = dca_invested

        if last_bar_day[i]:
            daily_return_count += 1
            for account in range(3):
                # ``prior_day_equity`` starts at independent initial principal and
                # then uses the previous session's closing mark.
                daily_return = last_equity[account] / prior_day_equity[account] - 1.0 if prior_day_equity[account] > 0 else 0.0
                prior_day_equity[account] = last_equity[account]
                daily_return_sum[account] += daily_return
                daily_return_sumsq[account] += daily_return * daily_return
                period_values[account, dca_day_index] = daily_return
            dca_day_index += 1
            for action in range(action_count):
                if action_record_day[action] == day_id:
                    for account in range(3):
                        entitlement_cash[account, action] = shares[account] * action_cash_per_share[action]
                        entitlement_bonus[account, action] = shares[account] * action_bonus_ratio[action]
        prior_close = close

    metrics = np.zeros((3, 11), dtype=np.float64)
    elapsed = (last_timestamp - first_timestamp) / 1_000_000_000.0 / (365.25 * 24.0 * 3600.0) if n else 0.0
    elapsed = max(elapsed, 1.0 / 252.0)
    for account in range(3):
        total_return = last_equity[account] / initial_equities[account] - 1.0 if initial_equities[account] > 0 else 0.0
        cagr = 0.0
        if last_equity[account] > 0 and initial_equities[account] > 0:
            log_growth = math.log(last_equity[account] / initial_equities[account]) / elapsed
            cagr = math.expm1(min(700.0, max(-700.0, log_growth)))
        sharpe = 0.0
        if daily_return_count > 1:
            mean = daily_return_sum[account] / daily_return_count
            squared_deviation = 0.0
            for session in range(daily_return_count):
                delta = period_values[account, session] - mean
                squared_deviation += delta * delta
            variance = squared_deviation / (daily_return_count - 1)
            deviation = math.sqrt(max(0.0, variance))
            if deviation > 0:
                sharpe = mean / deviation * math.sqrt(252.0)
        calmar = cagr / max_drawdown[account] if max_drawdown[account] > 0 else 0.0
        metrics[account, 0] = total_return
        metrics[account, 1] = cagr
        metrics[account, 2] = max_drawdown[account]
        metrics[account, 3] = sharpe
        metrics[account, 4] = calmar
        metrics[account, 5] = buy_count[account] + sell_count[account]
        metrics[account, 6] = total_fees[account]
        metrics[account, 7] = gross_turnover[account] / initial_equities[account] if initial_equities[account] > 0 else 0.0
        metrics[account, 8] = buy_count[account]
        metrics[account, 9] = sell_count[account]

    metrics[1, 0] = last_equity[1] / dca_invested - 1.0 if dca_invested > 0 else np.nan
    metrics[1, 10] = dca_invested
    for index in (1, 2, 3, 4, 7):
        metrics[1, index] = np.nan
    return (
        metrics, equity_values, equity_cash, equity_shares, equity_drawdown,
        period_values, trade_bar[:trade_n], trade_strategy[:trade_n],
        trade_side[:trade_n], trade_quantity[:trade_n], trade_price[:trade_n],
        trade_fees[:trade_n], trade_cash[:trade_n], trade_shares[:trade_n],
        trade_reason[:trade_n], rejection_bar[:rejection_n], rejection_ts[:rejection_n],
        rejection_strategy[:rejection_n], rejection_side[:rejection_n],
        rejection_reason[:rejection_n], rejection_quantity[:rejection_n], invested_values,
    )

