"""Share-based ledger kernel; the same function runs interpreted and with Numba.

Version-one cash/ratio results retain their original engine. This kernel seeds
existing holdings and funds fixed-share DCA only on successful executions.
"""
from __future__ import annotations
import math
import numpy as np
from .engine import _jit, _nb_round_money
try:
    from numba import prange
except ImportError:  # pragma: no cover
    prange = range

# Which ledgers a simulation actually runs.  Every output array keeps its four
# rows in the original account order (0 grid, 1 DCA, 2 buy-and-hold, 3 matched
# funding hold), so a caller reads row 0 and row 3 without remapping.
ACCOUNT_MODE_FULL = 0
ACCOUNT_MODE_RESEARCH = 1
ACCOUNT_MODE_GRID = 2

@_jit(cache=True)
def _nb_buy_lot(quantity, board_code):
    result = max(0, int(math.floor(quantity / 100))) * 100
    return 0 if board_code == 2 and result < 200 else result

@_jit(cache=True)
def _nb_sell_lot(requested, available, board_code):
    return max(0, min(requested, available) // 100 * 100)

@_jit(cache=True)
def _nb_cell(price, anchor, grid_mode, buy_step, sell_step):
    if price <= 0 or anchor <= 0:
        return 0
    if grid_mode == 1:
        if price < anchor:
            if buy_step >= 1.0:
                return -1
            width = max(anchor * buy_step, 0.001)
            return -int(math.floor((anchor - price) / width + 1e-10))
        width = max(anchor * sell_step, 0.001)
        return int(math.floor((price - anchor) / width + 1e-10))
    if price < anchor:
        if buy_step >= 1.0:
            return -1
        return -int(math.floor(math.log(anchor / price) / math.log(1.0 / (1.0 - buy_step)) + 1e-10))
    return int(math.floor(math.log(price / anchor) / math.log(1.0 + sell_step) + 1e-10))


@_jit(cache=True)
def _nb_floor_grid(anchor, high, sell_step):
    if anchor <= 0 or high <= anchor or sell_step <= 0:
        return 0
    strides = math.log(high / anchor) / math.log1p(sell_step) + 1e-10
    return max(0, int(math.floor(strides)))


@_jit(cache=True)
def _nb_close_floor(active, anchor, high, sell_step, grids, count, squares):
    if not active:
        return active, anchor, high, grids, count, squares
    grid = _nb_floor_grid(anchor, high, sell_step)
    grids = np.append(grids, np.int64(grid))
    return False, 0.0, 0.0, grids, count + 1, squares + float(grid) * float(grid)


@_jit(cache=True)
def _nb_grid_intent(close, position, reference, anchor, active_cell,
                    grid_mode, buy_step, sell_step, grid_step,
                    lower_price, upper_price, min_shares, max_shares,
                    buy_shares, sell_shares, last_fill_price, board_code):
    """A close-only signal; never reads the next opening price or volume."""
    side, requested = 0, 0
    if grid_mode == 0:
        if reference > 0 and close <= reference * (1.0 - buy_step) and position < max_shares:
            side, requested = 1, buy_shares
        elif reference > 0 and close >= reference * (1.0 + sell_step) and position > min_shares:
            side, requested = 2, sell_shares
    else:
        index = _nb_cell(close, anchor, grid_mode, buy_step, sell_step)
        if index < active_cell and position < max_shares:
            side, requested = 1, buy_shares
        elif index > active_cell and position > min_shares:
            side, requested = 2, sell_shares
    if (side == 1 and lower_price > 0 and close < lower_price) or (side == 2 and upper_price > 0 and close > upper_price):
        side = 0
    if side != 0 and grid_mode != 0 and last_fill_price > 0:
        step = buy_step if side == 1 else sell_step
        if grid_mode == 1:
            width = max(anchor * step, 0.001)
            if side == 1:
                spacing_ok = close <= last_fill_price - width + 1e-10
            else:
                spacing_ok = close >= last_fill_price + width - 1e-10
        elif side == 1:
            spacing_ok = step < 1.0 and close <= last_fill_price * (1.0 - step) + 1e-10
        else:
            spacing_ok = close >= last_fill_price * (1.0 + step) - 1e-10
        if not spacing_ok:
            side = 0
    quantity = requested
    if side == 1:
        if math.isfinite(max_shares):
            quantity = _nb_buy_lot(min(requested, _nb_buy_lot(max(0.0, max_shares - position), board_code)), board_code)
        else:
            quantity = _nb_buy_lot(requested, board_code)
    elif side == 2:
        quantity = min(requested, int(max(0.0, position - min_shares)) // 100 * 100)
    return side, quantity, (4 if side == 1 else 5), requested

@_jit(cache=True)
def _nb_attempt(
    side: int, requested: int, opening: float, volume: float,
    limit_up: float, limit_down: float, stamp_rate: float, transfer_rate: float,
    transfer_is_face_value: bool, transfer_minimum: float,
    stamp_bilateral: bool, day_id: int, cash: float, cash_receivable: float,
    share_receivable: float, shares: int, sellable: int,
    max_shares: float, min_shares: float, commission_rate: float,
    min_commission: float, slippage: float, board_code: int, face_value: float, tick: float, t0: bool,
    top_up_on_fill: bool, allow_odd_lot_exit: bool,
) -> tuple:
    """Return fill state and a compact rejection code; side 1=buy, 2=sell."""
    if volume <= 0 or not math.isfinite(volume):
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 8, 0.0
    if side == 1 and math.isfinite(limit_up) and opening >= limit_up - tick / 2:
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 1, 0.0
    if side == 2 and math.isfinite(limit_down) and opening <= limit_down + tick / 2:
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 2, 0.0

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
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 9, 0.0

    position_shares = shares + share_receivable
    equity_open = max(0.0, cash + cash_receivable + position_shares * opening)
    if side == 1:
        normalized_request = _nb_buy_lot(min(max(0, requested), 2147483647), board_code)
        max_by_position = max(0.0, max_shares - position_shares)
        position_quantity = _nb_buy_lot(max_by_position, board_code) if math.isfinite(max_shares) else normalized_request
        affordable_quantity = normalized_request
        if not top_up_on_fill:
            affordable_quantity = 0
            candidate_quantity = _nb_buy_lot(min(normalized_request, max_by_position), board_code)
            while candidate_quantity > 0:
                notional = _nb_round_money(candidate_quantity * price)
                commission = _nb_round_money(max(min_commission, notional * commission_rate))
                transfer_basis = candidate_quantity * face_value if transfer_is_face_value else notional
                transfer = _nb_round_money(max(transfer_minimum, transfer_basis * transfer_rate))
                total = _nb_round_money(notional + commission + transfer)
                if total <= cash + 0.005:
                    affordable_quantity = candidate_quantity
                    break
                candidate_quantity -= 100
        quantity = min(normalized_request, position_quantity, affordable_quantity)
        if quantity <= 0:
            if normalized_request <= 0:
                reason = 3
            elif position_quantity <= 0:
                reason = 4
            else:
                reason = 5
            return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, normalized_request, reason, 0.0
    else:
        normalized_request = max(0, requested)
        min_shares_shares = max(0.0, min_shares)
        position_capacity = max(0, int(math.floor(position_shares - min_shares_shares + 1e-9)))
        sellable_capacity = min(sellable, position_capacity)
        requested_available = min(normalized_request, sellable_capacity)
        if allow_odd_lot_exit and normalized_request >= position_shares and requested_available > 0:
            # A closing sell may dispose of the complete remaining odd lot.  The
            # legacy path retains lot-sized sells for byte-for-byte reproduction.
            quantity = requested_available
        else:
            quantity = _nb_sell_lot(requested_available, sellable_capacity, board_code)
        if quantity <= 0:
            if sellable <= 0 and shares > 0:
                reason = 6
            elif position_capacity <= 0:
                reason = 4
            elif normalized_request <= 0:
                reason = 3
            else:
                reason = 7
            return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, normalized_request, reason, 0.0

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
        if side != 1 or cash_after >= -0.005 or top_up_on_fill:
            break
        quantity -= 100
        if board_code == 2 and quantity < 200:
            quantity = 0
    if quantity <= 0:
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, normalized_request, 5, 0.0
    if side == 1:
        top_up = 0.0
        if cash_after < 0:
            if top_up_on_fill:
                top_up = _nb_round_money(-cash_after)
                cash_after = 0.0
            else:
                return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, normalized_request, 5, 0.0
        return True, max(0.0, cash_after), shares + quantity, sellable + (quantity if t0 else 0), quantity, price, fee, notional, normalized_request, 0, top_up
    cash_after = _nb_round_money(cash + notional - fee)
    return True, cash_after, shares - quantity, sellable - quantity, quantity, price, fee, notional, normalized_request, 0, 0.0


@_jit(cache=True, nogil=True)
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
    skip_mask: np.ndarray,
    grid_initial_equity: float, dca_initial_equity: float,
    buy_hold_initial_equity: float, min_shares: float, max_shares: float,
    grid_mode: int, buy_step: float, sell_step: float, grid_step: float,
    reference_price: float, lower_price: float, upper_price: float,
    buy_shares: int, sell_shares: int,
    board_code: int, commission_rate: float,
    min_commission: float, slippage: float, face_value: float, seed_shares: int,
    reserve_cash: float, dca_initial_shares: int, dca_shares: int,
    buy_hold_initial_shares: int, valuation_price: float, tick: float,
    t0: bool, output_mode: int, funding_mode: int,
    allow_odd_lot_exit: bool,
    floor_eta_floor: float, floor_gamma_min: float, floor_k_ref: float,
    parameter_days: np.ndarray, parameter_values: np.ndarray,
    fold_ids: np.ndarray, fold_count: int,
    grid_only: bool, passive_grid: bool, account_mode: int,
    cancel_flag: np.ndarray | None = None,
    dca_budgets_by_day: np.ndarray | None = None,
) -> tuple:
    keep_details = output_mode >= 2
    top_up_mode = funding_mode == 1
    # The matched-funding account mirrors the grid's external contributions in
    # the same bar.  It is kept in the native ledger so research batches and
    # ordinary backtests use exactly the same execution and corporate-action
    # rules.
    #
    # ``account_mode`` selects the ledgers that are simulated at all:
    #   0 - grid, DCA, buy-and-hold and matched funding hold (ordinary backtest)
    #   1 - grid and matched funding hold only; research batches never read the
    #       DCA or buy-and-hold ledgers
    #   2 - grid only
    # The surviving ledgers keep their original account index, so every output
    # array stays four rows wide and row 3 is still the matched funding hold.
    mode = int(account_mode)
    if grid_only:
        mode = ACCOUNT_MODE_GRID
    active_accounts = np.zeros(4, dtype=np.int64)
    if mode == ACCOUNT_MODE_RESEARCH:
        active_accounts[0] = 0
        active_accounts[1] = 3
        active_count = 2
    elif mode == ACCOUNT_MODE_GRID:
        active_accounts[0] = 0
        active_count = 1
    else:
        active_accounts[0] = 0
        active_accounts[1] = 1
        active_accounts[2] = 2
        active_accounts[3] = 3
        active_count = 4
    account_active = np.zeros(4, dtype=np.bool_)
    for slot in range(active_count):
        account_active[active_accounts[slot]] = True
    matched_enabled = bool(account_active[3])
    n = len(opens)
    day_count = 0
    for flag in last_bar_day:
        if flag:
            day_count += 1
    initial_equities = np.array([grid_initial_equity, dca_initial_equity, buy_hold_initial_equity, grid_initial_equity], dtype=np.float64)
    cash = initial_equities.copy()
    shares = np.zeros(4, dtype=np.int64)
    sellable = np.zeros(4, dtype=np.int64)
    share_locked = np.zeros(4, dtype=np.int64)
    cash_receivable = np.zeros(4, dtype=np.float64)
    share_receivable = np.zeros(4, dtype=np.float64)
    started = np.zeros(4, dtype=np.bool_)
    reference = np.zeros(4, dtype=np.float64)
    fixed_anchor = np.zeros(4, dtype=np.float64)
    active_cell = np.zeros(4, dtype=np.int64)
    has_active_cell = np.zeros(4, dtype=np.bool_)
    pending_side = np.zeros(4, dtype=np.int64)
    pending_quantity = np.zeros(4, dtype=np.int64)
    pending_reason = np.zeros(4, dtype=np.int64)
    dca_due = False
    dca_last_day = -1
    action_count = len(action_record_day)
    entitlement_cash = np.zeros((4, action_count), dtype=np.float64)
    entitlement_bonus = np.zeros((4, action_count), dtype=np.float64)
    accrued_cash = np.zeros((4, action_count), dtype=np.float64)
    accrued_bonus = np.zeros((4, action_count), dtype=np.float64)
    accrued_exists = np.zeros((4, action_count), dtype=np.bool_)
    action_paid = np.zeros((4, action_count), dtype=np.bool_)
    action_stock_issued = np.zeros((4, action_count), dtype=np.bool_)
    action_locked_quantity = np.zeros((4, action_count), dtype=np.int64)
    prior_close = 0.0
    trade_count = np.zeros(4, dtype=np.int64)
    buy_count = np.zeros(4, dtype=np.int64)
    sell_count = np.zeros(4, dtype=np.int64)
    actual_min_position = 0.0
    actual_max_position = 0.0
    floor_enabled = floor_eta_floor >= 0 and floor_k_ref > 0
    floor_limit = floor_eta_floor * float(seed_shares)
    floor_active = floor_enabled and float(seed_shares) <= floor_limit + 1e-9
    floor_anchor = float(valuation_price) if floor_active else 0.0
    floor_high = floor_anchor
    floor_sell_step = sell_step if floor_active else 0.0
    floor_episode_grids = np.empty(0, dtype=np.int64)
    floor_episode_count = 0
    floor_grid_squares = 0.0
    stockout_count = 0.0
    stockout_signal_active = False
    tplus1_block_count = 0.0
    tplus1_signal_active = False
    total_fees = np.zeros(4, dtype=np.float64)
    gross_turnover = np.zeros(4, dtype=np.float64)
    interval_peak_principal = initial_equities.copy()
    max_drawdown_raw = np.zeros(4, dtype=np.float64)
    last_equity = initial_equities.copy()
    fold_stats = np.zeros((fold_count, 9), dtype=np.float64)
    fold_peaks = np.zeros(fold_count, dtype=np.float64)
    fold_drawdown_raw = np.zeros(fold_count, dtype=np.float64)
    fold_peak_principal = np.zeros(fold_count, dtype=np.float64)
    fold_initials = np.zeros(fold_count, dtype=np.float64)
    fold_snapshots = np.zeros((fold_count, 5), dtype=np.float64)
    fold_start_assets = np.zeros(fold_count, dtype=np.float64)
    fold_topups = np.zeros(fold_count, dtype=np.float64)
    grid_invested = float(grid_initial_equity)
    grid_topups = 0.0
    grid_peak_invested = float(grid_initial_equity)
    grid_peak_day = -1
    matched_invested = float(grid_initial_equity)
    matched_peak_invested = float(grid_initial_equity)
    matched_pending_cash = 0.0
    peak_profit = np.zeros(4, dtype=np.float64)
    day_external_flow = 0.0
    parameter_index = 0
    params_paused = False
    buy_shares_exact = float(buy_shares)
    sell_shares_exact = float(sell_shares)
    dca_shares_exact = float(dca_shares)
    last_grid_price = reference_price
    prior_signal_allowed = False
    if keep_details:
        period_values = np.zeros((0, 0), dtype=np.float64)
        trade_bar = np.zeros(5 * n, dtype=np.int64)
        trade_strategy = np.zeros(5 * n, dtype=np.int8)
        trade_side = np.zeros(5 * n, dtype=np.int8)
        trade_quantity = np.zeros(5 * n, dtype=np.int64)
        trade_price = np.zeros(5 * n, dtype=np.float64)
        trade_fees = np.zeros(5 * n, dtype=np.float64)
        trade_cash = np.zeros(5 * n, dtype=np.float64)
        trade_shares = np.zeros(5 * n, dtype=np.int64)
        trade_reason = np.zeros(5 * n, dtype=np.int8)
        trade_topup = np.zeros(5 * n, dtype=np.float64)
        trade_n = 0
        rejection_bar = np.zeros(5 * n, dtype=np.int64)
        rejection_ts = np.zeros(5 * n, dtype=np.int64)
        rejection_strategy = np.zeros(5 * n, dtype=np.int8)
        rejection_side = np.zeros(5 * n, dtype=np.int8)
        rejection_reason = np.zeros(5 * n, dtype=np.int8)
        rejection_quantity = np.zeros(5 * n, dtype=np.int64)
        rejection_n = 0
    else:
        period_values = np.zeros((0, 0), dtype=np.float64)
        trade_bar = np.zeros(0, dtype=np.int64)
        trade_strategy = np.zeros(0, dtype=np.int8)
        trade_side = np.zeros(0, dtype=np.int8)
        trade_quantity = np.zeros(0, dtype=np.int64)
        trade_price = np.zeros(0, dtype=np.float64)
        trade_fees = np.zeros(0, dtype=np.float64)
        trade_cash = np.zeros(0, dtype=np.float64)
        trade_shares = np.zeros(0, dtype=np.int64)
        trade_reason = np.zeros(0, dtype=np.int8)
        trade_topup = np.zeros(0, dtype=np.float64)
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
    cash[3] = reserve_cash
    shares[3] = seed_shares
    sellable[3] = seed_shares
    started[3] = True
    dca_invested = dca_initial_equity
    prior_close = valuation_price
    initial_holdings = (seed_shares, dca_initial_shares, buy_hold_initial_shares, seed_shares)
    actual_min_position = float(seed_shares)
    actual_max_position = float(seed_shares)
    for action in range(action_count):
        if action_record_day[action] == -1 and action_ex_day[action] >= 0:
            for slot in range(active_count):
                account = active_accounts[slot]
                entitlement_cash[account, action] = initial_holdings[account] * action_cash_per_share[action]
                entitlement_bonus[account, action] = initial_holdings[account] * action_bonus_ratio[action]
    output_count = n if output_mode == 2 else (day_count if output_mode in (1, 3) else 0)
    equity_values = np.zeros((4, output_count), dtype=np.float64)
    equity_cash = np.zeros((4, output_count), dtype=np.float64)
    equity_shares = np.zeros((4, output_count), dtype=np.float64)
    equity_drawdown = np.zeros((4, output_count), dtype=np.float64)
    invested_values = np.zeros((4, output_count), dtype=np.float64)
    current_day_id = -1
    dca_day_index = 0
    for i in range(n):
        if cancel_flag is not None and (i & 511) == 0 and cancel_flag[0] != 0:
            raise InterruptedError("回测已取消")
        floor_position_changed = False
        day_id = day_ids[i]
        if day_id != current_day_id:
            current_day_id = day_id
            day_external_flow = 0.0
            fold_id = fold_ids[day_id]
            if fold_id >= 0 and fold_initials[fold_id] == 0.0:
                # Fold profit starts at the principal already committed when
                # the evaluation interval begins.  Equity may contain retained
                # profits; treating that equity as principal would dilute the
                # fold's peak-principal return and drawdown.
                fold_initials[fold_id] = grid_invested
                fold_start_assets[fold_id] = grid_invested
                fold_peak_principal[fold_id] = grid_invested
                fold_snapshots[fold_id, 0] = trade_count[0]
                fold_snapshots[fold_id, 1] = buy_count[0]
                fold_snapshots[fold_id, 2] = sell_count[0]
                fold_snapshots[fold_id, 3] = total_fees[0]
                fold_snapshots[fold_id, 4] = gross_turnover[0]

            # Rules become effective at this day's first opening event.  Only
            # the previous real close may be used to replace a pending signal.
            while parameter_index < len(parameter_days) and parameter_days[parameter_index] <= day_id:
                values = parameter_values[parameter_index]
                next_paused = bool(values[7]) if len(values) > 7 else False
                changed = (min_shares != values[0] or max_shares != values[1]
                           or buy_step != values[2] or sell_step != values[3]
                           or grid_step != values[4] or buy_shares != int(values[5])
                           or sell_shares != int(values[6]) or params_paused != next_paused)
                min_shares, max_shares = values[0], values[1]
                buy_step, sell_step, grid_step = values[2], values[3], values[4]
                buy_shares_exact, sell_shares_exact = float(values[5]), float(values[6])
                buy_shares, sell_shares = int(values[5]), int(values[6])
                params_paused = next_paused
                if changed and not passive_grid:
                    if pending_side[0] and keep_details:
                        rejection_bar[rejection_n] = i
                        rejection_ts[rejection_n] = open_timestamps_ns[i]
                        rejection_strategy[rejection_n] = 0
                        rejection_side[rejection_n] = pending_side[0]
                        rejection_reason[rejection_n] = 12
                        rejection_quantity[rejection_n] = pending_quantity[0]
                        rejection_n += 1
                    pending_side[0], pending_quantity[0], pending_reason[0] = 0, 0, 0
                    if grid_mode != 0:
                        active_cell[0] = _nb_cell(last_grid_price, fixed_anchor[0], grid_mode, buy_step, sell_step)
                    if i > 0 and prior_signal_allowed and not params_paused:
                        intent = _nb_grid_intent(prior_close, shares[0] + share_receivable[0],
                            reference[0], fixed_anchor[0], active_cell[0], grid_mode,
                            buy_step, sell_step, grid_step, lower_price, upper_price,
                            min_shares, max_shares, buy_shares, sell_shares, last_grid_price, board_code)
                        if intent[0] != 0 and intent[1] > 0:
                            pending_side[0], pending_quantity[0], pending_reason[0] = intent[0], intent[1], intent[2]
                parameter_index += 1
            for slot in range(active_count):
                account = active_accounts[slot]
                sellable[account] = max(0, shares[account] - share_locked[account])
            cash_delta = 0.0
            bonus_delta = 0.0
            has_ex_action = False
            for action in range(action_count):
                if action_ex_day[action] == day_id:
                    has_ex_action = True
                    cash_delta += action_cash_per_share[action]
                    bonus_delta += action_bonus_ratio[action]
            # Same-date rows are components of one corporate action; their
            # bonus ratios add against the same record-date share basis.
            # Multipliers compound only as separate ex-dates are processed.
            bonus_multiplier = 1.0 + bonus_delta
            factor = 1.0
            if has_ex_action and prior_close > 0:
                implied_close = prior_close * bonus_multiplier
                implied_factor = max(
                    0.0,
                    (prior_close - cash_delta) / implied_close
                    if implied_close > 0 else 0.0,
                )
                official = preclose_by_day[day_id]
                if (
                    math.isfinite(official) and official > 0
                    and abs(official - prior_close * implied_factor) <= tick * 1.1
                ):
                    factor = official / prior_close
                else:
                    # A stale or unrelated preclose is not a verified ex-rights
                    # reference.  Use the action-implied factor, matching the
                    # training-window and statistics path.
                    factor = implied_factor
                if factor <= 0 or not math.isfinite(factor):
                    raise ValueError("公司行动除权参考价无效")
            for slot in range(active_count):
                account = active_accounts[slot]
                if has_ex_action and account == 0:
                    # Convert share-denominated strategy settings from the
                    # pre-action basis exactly once, retaining fractional
                    # precision until the kernel's existing order rules apply.
                    ratio = bonus_multiplier
                    min_shares *= ratio
                    if math.isfinite(max_shares):
                        max_shares *= ratio
                    floor_limit *= ratio
                    buy_shares_exact *= ratio
                    sell_shares_exact *= ratio
                    dca_shares_exact *= ratio
                    buy_shares = int(math.floor(buy_shares_exact + 1e-10))
                    sell_shares = int(math.floor(sell_shares_exact + 1e-10))
                    dca_shares = int(math.floor(dca_shares_exact + 1e-10))
                    grid_step *= factor
                    if pending_side[0] != 0:
                        pending_quantity[0] = int(math.floor(pending_quantity[0] * ratio + 1e-10))
                        if pending_quantity[0] <= 0:
                            pending_side[0], pending_reason[0] = 0, 0
                    if factor != 1.0 and reference[account] > 0:
                        reference[account] *= factor
                    if factor != 1.0 and fixed_anchor[account] > 0:
                        fixed_anchor[account] *= factor
                    if factor != 1.0 and lower_price > 0:
                        lower_price *= factor
                    if factor != 1.0 and upper_price > 0:
                        upper_price *= factor
                    if factor != 1.0:
                        last_grid_price *= factor
                    if floor_active:
                        floor_anchor *= factor
                        floor_high *= factor
                    # A floor episode may not have started yet; its price
                    # threshold still follows the adjusted price basis.
                    if not floor_active and floor_anchor > 0:
                        floor_anchor *= factor
                        floor_high *= factor
                for action in range(action_count):
                    if action_ex_day[action] == day_id:
                        first_same_ex_date = True
                        for prior_action in range(action):
                            if action_ex_day[prior_action] == day_id:
                                first_same_ex_date = False
                                break
                        if first_same_ex_date:
                            total_cash_claim = 0.0
                            total_bonus_claim = 0.0
                            total_bonus_ratio = 0.0
                            for grouped_action in range(action_count):
                                if action_ex_day[grouped_action] == day_id:
                                    cash_claim = entitlement_cash[account, grouped_action]
                                    bonus_claim = entitlement_bonus[account, grouped_action]
                                    total_cash_claim += cash_claim
                                    total_bonus_claim += bonus_claim
                                    total_bonus_ratio += action_bonus_ratio[grouped_action]
                                    accrued_cash[account, grouped_action] = cash_claim
                                    accrued_bonus[account, grouped_action] = bonus_claim
                                    accrued_exists[account, grouped_action] = True
                                    entitlement_cash[account, grouped_action] = 0.0
                                    entitlement_bonus[account, grouped_action] = 0.0
                            cash_receivable[account] += total_cash_claim
                            if total_bonus_claim >= 0:
                                share_receivable[account] += total_bonus_claim
                            else:
                                old_shares = shares[account]
                                ratio = max(0.0, 1.0 + total_bonus_ratio)
                                shares[account] = int(math.floor(old_shares * ratio + 1e-10))
                                converted = shares[account]
                                unlock_day = action_stock_day[action]
                                if unlock_day > day_id:
                                    # The reverse-converted position is not
                                    # sellable until the exchange delivers it.
                                    share_locked[account] = converted
                                    action_locked_quantity[account, action] = converted
                                    sellable[account] = 0
                                else:
                                    share_locked[account] = 0
                                    sellable[account] = converted
                for action in range(action_count):
                    if action_pay_day[action] == day_id and accrued_exists[account, action] and not action_paid[account, action]:
                        amount = accrued_cash[account, action]
                        cash_receivable[account] = max(0.0, cash_receivable[account] - amount)
                        cash[account] = _nb_round_money(cash[account] + amount)
                        action_paid[account, action] = True
                    if action_stock_day[action] == day_id and accrued_exists[account, action] and not action_stock_issued[account, action]:
                        first_same_issue = True
                        total_issue = 0.0
                        for other_action in range(action_count):
                            if (action_stock_day[other_action] == day_id
                                    and action_ex_day[other_action] == action_ex_day[action]
                                    and accrued_exists[account, other_action]
                                    and not action_stock_issued[account, other_action]):
                                if other_action < action:
                                    first_same_issue = False
                                total_issue += accrued_bonus[account, other_action]
                        if first_same_issue:
                            if total_issue > 0:
                                issued = int(math.floor(total_issue + 1e-10))
                                share_receivable[account] = max(0.0, share_receivable[account] - total_issue)
                                if share_receivable[account] < 1e-9:
                                    share_receivable[account] = 0.0
                                shares[account] += issued
                                sellable[account] += issued
                            locked_release = 0
                            for other_action in range(action_count):
                                if (action_stock_day[other_action] == day_id
                                        and action_ex_day[other_action] == action_ex_day[action]
                                        and accrued_exists[account, other_action]):
                                    locked_release += action_locked_quantity[account, other_action]
                                    action_locked_quantity[account, other_action] = 0
                                    action_stock_issued[account, other_action] = True
                            if locked_release > 0:
                                share_locked[account] = max(0, share_locked[account] - locked_release)
                                sellable[account] = min(shares[account], sellable[account] + locked_release)
                if account == 0:
                    current_position = float(shares[0] + share_receivable[0])
                    actual_min_position = min(actual_min_position, current_position)
                    actual_max_position = max(actual_max_position, current_position)
                    floor_position_changed = True
        if first_dca_day[i]:
            dca_due = True

        # Observed non-tradable bars are real opening events: an order waiting
        # for this open expires here, while the bar itself must not generate a
        # trade, a close signal, an equity mark, or a period-return observation.
        if skip_mask[i]:
            prior_signal_allowed = False
            for slot in range(active_count):
                account = active_accounts[slot]
                if pending_side[account] != 0:
                    if keep_details:
                        rejection_bar[rejection_n] = i
                        rejection_ts[rejection_n] = open_timestamps_ns[i]
                        rejection_strategy[rejection_n] = account
                        rejection_side[rejection_n] = pending_side[account]
                        rejection_reason[rejection_n] = 13
                        rejection_quantity[rejection_n] = pending_quantity[account]
                        rejection_n += 1
                    pending_side[account] = 0
                    pending_quantity[account] = 0
                    pending_reason[account] = 0
            for action in range(action_count):
                if action_record_day[action] == day_id:
                    for slot in range(active_count):
                        account = active_accounts[slot]
                        entitlement_cash[account, action] = shares[account] * action_cash_per_share[action]
                        entitlement_bonus[account, action] = shares[account] * action_bonus_ratio[action]
            continue

        if not tradable_mask[i]:
            prior_signal_allowed = False
            for slot in range(active_count):
                account = active_accounts[slot]
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
        if floor_active:
            floor_high = max(floor_high, opening)
        if floor_position_changed and floor_enabled:
            position = float(shares[0] + share_receivable[0])
            if floor_active and position > floor_limit + 1e-9:
                floor_active, floor_anchor, floor_high, floor_episode_grids, floor_episode_count, floor_grid_squares = _nb_close_floor(
                    floor_active, floor_anchor, floor_high, floor_sell_step,
                    floor_episode_grids, floor_episode_count, floor_grid_squares,
                )
            elif not floor_active and position <= floor_limit + 1e-9:
                floor_active, floor_anchor, floor_high, floor_sell_step = True, opening, opening, sell_step
        for slot in range(active_count):
            account = active_accounts[slot]
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
                    top_up_mode and account == 0, allow_odd_lot_exit,
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
                    if account == 0:
                        current_position = float(shares[0] + share_receivable[0])
                        actual_min_position = min(actual_min_position, current_position)
                        actual_max_position = max(actual_max_position, current_position)
                        if floor_enabled:
                            if floor_active and current_position > floor_limit + 1e-9:
                                floor_active, floor_anchor, floor_high, floor_episode_grids, floor_episode_count, floor_grid_squares = _nb_close_floor(
                                    floor_active, floor_anchor, floor_high, floor_sell_step,
                                    floor_episode_grids, floor_episode_count, floor_grid_squares,
                                )
                            elif not floor_active and current_position <= floor_limit + 1e-9:
                                floor_active, floor_anchor, floor_high, floor_sell_step = True, result[5], result[5], sell_step
                    if account == 0 and result[10] > 0:
                        top_up = float(result[10])
                        grid_invested = _nb_round_money(grid_invested + top_up)
                        grid_topups = _nb_round_money(grid_topups + top_up)
                        grid_peak_invested = grid_invested
                        grid_peak_day = day_id
                        if matched_enabled:
                            # Mirror the real contribution in the same fill event.
                            cash[3] = _nb_round_money(cash[3] + top_up)
                            matched_invested = _nb_round_money(matched_invested + top_up)
                            matched_peak_invested = max(matched_peak_invested, matched_invested)
                            matched_pending_cash = _nb_round_money(matched_pending_cash + top_up)
                        day_external_flow = _nb_round_money(day_external_flow + top_up)
                        if fold_id >= 0:
                            fold_topups[fold_id] = _nb_round_money(fold_topups[fold_id] + top_up)
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
                        trade_topup[trade_n] = result[10]
                        trade_n += 1
                    traded = True
                    if account == 0:
                        last_grid_price = result[5]
                        if grid_mode == 0:
                            reference[account] = result[5]
                        else:
                            if fixed_anchor[account] == 0:
                                fixed_anchor[account] = reference_price if reference_price > 0 else result[5]
                            if reference[account] == 0:
                                reference[account] = fixed_anchor[account]
                            active_cell[account] = _nb_cell(result[5], fixed_anchor[account], grid_mode, buy_step, sell_step)
                            has_active_cell[account] = True

            if matched_enabled and account == 0 and matched_pending_cash > 0 and opening > 0:
                available_pool = min(matched_pending_cash, cash[3])
                matched_quantity = _nb_buy_lot(available_pool / opening, board_code)
                if matched_quantity > 0:
                    matched = _nb_attempt(
                        1, matched_quantity, opening, volume, limit_ups[i], limit_downs[i],
                        stamp_rates[i], transfer_rates[i], transfer_is_face_value[i],
                        transfer_minimums[i], stamp_bilateral[i], day_id, available_pool,
                        cash_receivable[3], share_receivable[3], shares[3], sellable[3],
                        2147483647.0, 0.0, commission_rate, min_commission, slippage,
                        board_code, face_value, tick, t0, False, False,
                    )
                    if matched[0]:
                        # Dividends and other retained account cash are not part
                        # of the contribution pool.  Only the mirrored top-up
                        # and its uninvested lot remainder may fund this buy.
                        matched_cost = _nb_round_money(matched[7] + matched[6])
                        cash[3] = max(0.0, _nb_round_money(cash[3] - matched_cost))
                        shares[3], sellable[3] = matched[2], matched[3]
                        matched_pending_cash = max(0.0, _nb_round_money(matched_pending_cash - matched_cost))
                        trade_count[3] += 1
                        buy_count[3] += 1
                        total_fees[3] += matched[6]
                        gross_turnover[3] += matched[7]
                        if keep_details:
                            trade_bar[trade_n] = i
                            trade_strategy[trade_n] = 3
                            trade_side[trade_n] = 1
                            trade_quantity[trade_n] = matched[4]
                            trade_price[trade_n] = matched[5]
                            trade_fees[trade_n] = matched[6]
                            trade_cash[trade_n] = cash[3]
                            trade_shares[trade_n] = shares[3]
                            trade_reason[trade_n] = 3
                            trade_topup[trade_n] = 0.0
                            trade_n += 1
                    elif keep_details:
                        rejection_bar[rejection_n] = i
                        rejection_ts[rejection_n] = open_timestamps_ns[i]
                        rejection_strategy[rejection_n] = 3
                        rejection_side[rejection_n] = 1
                        rejection_reason[rejection_n] = matched[9]
                        rejection_quantity[rejection_n] = matched[8]
                        rejection_n += 1

            if not traded and account == 1 and dca_due and dca_last_day != day_id:
                dca_last_day = day_id
                if opening > 0:
                    quantity = _nb_buy_lot(dca_shares, board_code)
                    if dca_budgets_by_day is not None:
                        budget = dca_budgets_by_day[day_id]
                        if not math.isfinite(budget):
                            if keep_details:
                                rejection_bar[rejection_n] = i
                                rejection_ts[rejection_n] = open_timestamps_ns[i]
                                rejection_strategy[rejection_n] = account
                                rejection_side[rejection_n] = 1
                                rejection_reason[rejection_n] = 14
                                rejection_quantity[rejection_n] = 0
                                rejection_n += 1
                            continue
                        execution = round(math.ceil(opening * (1.0 + slippage) / tick - 1e-10) * tick, 6)
                        if math.isfinite(limit_ups[i]):
                            execution = min(execution, limit_ups[i])
                        if math.isfinite(limit_downs[i]):
                            execution = max(execution, limit_downs[i])
                        quantity = _nb_buy_lot(int(math.floor(budget / execution)), board_code)
                        if quantity <= 0:
                            if keep_details:
                                rejection_bar[rejection_n] = i
                                rejection_ts[rejection_n] = open_timestamps_ns[i]
                                rejection_strategy[rejection_n] = account
                                rejection_side[rejection_n] = 1
                                rejection_reason[rejection_n] = 15
                                rejection_quantity[rejection_n] = 0
                                rejection_n += 1
                            continue
                    result = _nb_attempt(
                        1, quantity, opening, volume, limit_ups[i], limit_downs[i],
                        stamp_rates[i], transfer_rates[i], transfer_is_face_value[i],
                        transfer_minimums[i], stamp_bilateral[i], day_id, 1e16,
                        cash_receivable[account], share_receivable[account],
                        shares[account], sellable[account], 2147483647.0, 0.0,
                        commission_rate, min_commission, slippage, board_code, face_value, tick, t0, False,
                        False,
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
                            trade_topup[trade_n] = 0.0
                            trade_n += 1
                        traded = True

        # Submit at most one grid order from the real close; it becomes eligible
        # at the next real bar's open.  Position sizing is frozen at this close.
        if started[0] and volumes[i] > 0 and not passive_grid and not params_paused:
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
                index = _nb_cell(close, anchor, grid_mode, buy_step, sell_step)
                if not has_active_cell[0]:
                    active_cell[0] = _nb_cell(reference[0] if reference[0] > 0 else close, anchor, grid_mode, buy_step, sell_step)
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
            if intent_side != 0 and last_grid_price > 0:
                step = buy_step if intent_side == 1 else sell_step
                if grid_mode == 1:
                    width = max(anchor * step, 0.001)
                    if intent_side == 1:
                        spacing_ok = close <= last_grid_price - width + 1e-10
                    else:
                        spacing_ok = close >= last_grid_price + width - 1e-10
                elif intent_side == 1:
                    spacing_ok = step < 1.0 and close <= last_grid_price * (1.0 - step) + 1e-10
                else:
                    spacing_ok = close >= last_grid_price * (1.0 + step) - 1e-10
                if not spacing_ok:
                    intent_side = 0
            if intent_side != 0:
                if intent_side == 2 and position_close < intent_quantity:
                    if not stockout_signal_active:
                        stockout_count += 1.0
                        stockout_signal_active = True
                    tplus1_signal_active = False
                elif intent_side != 2 or position_close >= intent_quantity:
                    stockout_signal_active = False
                if intent_side == 2 and position_close >= intent_quantity:
                    requested_capacity = max(0, min(intent_quantity, int(position_close - min_shares)))
                    if max(0, sellable[0]) < requested_capacity:
                        if not tplus1_signal_active:
                            tplus1_block_count += 1.0
                            tplus1_signal_active = True
                    else:
                        tplus1_signal_active = False
                elif intent_side != 2:
                    tplus1_signal_active = False
                clipped_quantity = intent_quantity
                if intent_side == 1:
                    maximum = max(0.0, max_shares - position_close)
                    clipped_quantity = min(intent_quantity, _nb_buy_lot(maximum, board_code)) if math.isfinite(max_shares) else _nb_buy_lot(intent_quantity, board_code)
                    clipped_quantity = _nb_buy_lot(clipped_quantity, board_code)
                else:
                    max_sell = max(0.0, position_close - min_shares)
                    if allow_odd_lot_exit and intent_quantity >= position_close:
                        clipped_quantity = int(max_sell)
                    else:
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

        for slot in range(active_count):
            account = active_accounts[slot]
            equity = cash[account] + cash_receivable[account] + (shares[account] + share_receivable[account]) * close
            if account == 0:
                current_position = float(shares[0] + share_receivable[0])
                actual_min_position = min(actual_min_position, current_position)
                actual_max_position = max(actual_max_position, current_position)
            last_equity[account] = equity
            principal = initial_equities[account]
            if account == 0:
                principal = grid_invested
            elif account == 1:
                principal = dca_invested
            elif account == 3:
                principal = matched_invested
            profit = equity - principal
            peak_profit[account] = max(peak_profit[account], profit)
            interval_peak_principal[account] = max(interval_peak_principal[account], principal)
            drawdown_raw = max(0.0, peak_profit[account] - profit)
            if drawdown_raw > max_drawdown_raw[account]:
                max_drawdown_raw[account] = drawdown_raw
            if output_mode == 2 or (output_mode in (1, 3) and last_bar_day[i]):
                out = i if output_mode == 2 else dca_day_index
                equity_values[account, out] = equity
                equity_cash[account, out] = cash[account]
                equity_shares[account, out] = shares[account] + share_receivable[account]
                equity_drawdown[account, out] = drawdown_raw

        if output_mode == 2 or (output_mode in (1, 3) and last_bar_day[i]):
            out = i if output_mode == 2 else dca_day_index
            invested_values[0, out] = grid_invested if top_up_mode else initial_equities[0]
            invested_values[1, out] = dca_invested
            invested_values[2, out] = initial_equities[2]
            invested_values[3, out] = matched_invested if matched_enabled else initial_equities[3]

        if floor_active:
            floor_high = max(floor_high, close)

        fold_id = fold_ids[day_id]
        if fold_id >= 0:
            fold_principal = fold_start_assets[fold_id] + fold_topups[fold_id]
            fold_peak_principal[fold_id] = max(fold_peak_principal[fold_id], fold_principal)
            fold_profit = last_equity[0] - fold_principal
            fold_peaks[fold_id] = max(fold_peaks[fold_id], fold_profit)
            fold_drawdown_raw[fold_id] = max(
                fold_drawdown_raw[fold_id], fold_peaks[fold_id] - fold_profit,
            )
            fold_stats[fold_id, 1] = trade_count[0] - fold_snapshots[fold_id, 0]
            fold_stats[fold_id, 2] = buy_count[0] - fold_snapshots[fold_id, 1]
            fold_stats[fold_id, 3] = sell_count[0] - fold_snapshots[fold_id, 2]
            fold_stats[fold_id, 4] = total_fees[0] - fold_snapshots[fold_id, 3]
            fold_stats[fold_id, 5] = (gross_turnover[0] - fold_snapshots[fold_id, 4]) / fold_initials[fold_id] if fold_initials[fold_id] > 0 else 0.0
            fold_stats[fold_id, 6] = last_equity[0] - fold_start_assets[fold_id] - fold_topups[fold_id]
            fold_stats[fold_id, 7] = max(0.0, fold_start_assets[fold_id] + fold_topups[fold_id])
            fold_stats[fold_id, 8] = last_equity[0]

        if last_bar_day[i]:
            dca_day_index += 1
            for action in range(action_count):
                if action_record_day[action] == day_id:
                    for slot in range(active_count):
                        account = active_accounts[slot]
                        entitlement_cash[account, action] = shares[account] * action_cash_per_share[action]
                        entitlement_bonus[account, action] = shares[account] * action_bonus_ratio[action]
        prior_close = close
        prior_signal_allowed = volumes[i] > 0

    metrics = np.zeros((4, 11), dtype=np.float64)
    for slot in range(active_count):
        account = active_accounts[slot]
        principal = initial_equities[account]
        peak_principal = initial_equities[account]
        if account == 0:
            principal = grid_invested
            peak_principal = grid_peak_invested
        elif account == 1:
            principal = dca_invested
            peak_principal = max(dca_initial_equity, dca_invested)
        elif account == 3:
            principal = matched_invested
            peak_principal = matched_peak_invested
        peak_principal = interval_peak_principal[account]
        total_return = (last_equity[account] - principal) / peak_principal if peak_principal > 0 else np.nan
        metrics[account, 0] = total_return
        metrics[account, 2] = max_drawdown_raw[account] / peak_principal if peak_principal > 0 else np.nan
        metrics[account, 5] = buy_count[account] + sell_count[account]
        metrics[account, 6] = total_fees[account]
        turnover_base = grid_peak_invested if top_up_mode and account == 0 else initial_equities[account]
        metrics[account, 7] = gross_turnover[account] / turnover_base if turnover_base > 0 else 0.0
        metrics[account, 8] = buy_count[account]
        metrics[account, 9] = sell_count[account]

    if account_active[1]:
        metrics[1, 10] = dca_invested
        for index in (1, 3, 4, 7):
            metrics[1, index] = np.nan
    floor_active, floor_anchor, floor_high, floor_episode_grids, floor_episode_count, floor_grid_squares = _nb_close_floor(
        floor_active, floor_anchor, floor_high, floor_sell_step,
        floor_episode_grids, floor_episode_count, floor_grid_squares,
    )
    for slot in range(active_count):
        account = active_accounts[slot]
        if interval_peak_principal[account] > 0:
            equity_drawdown[account, :] = equity_drawdown[account, :] / interval_peak_principal[account]
    # The ledgers that were not simulated keep their row and their place in
    # the original account order, but report nothing as if they had run.
    for account in range(4):
        if not account_active[account]:
            metrics[account, :] = np.nan
            equity_values[account, :] = np.nan
            equity_cash[account, :] = np.nan
            equity_shares[account, :] = np.nan
            equity_drawdown[account, :] = np.nan
            invested_values[account, :] = np.nan
    for fold_id in range(fold_count):
        if fold_peak_principal[fold_id] > 0:
            fold_stats[fold_id, 0] = fold_drawdown_raw[fold_id] / fold_peak_principal[fold_id]
        else:
            fold_stats[fold_id, 0] = np.nan
    funding_stats = np.zeros((4, 24), dtype=np.float64)
    funding_stats[0, 0] = initial_equities[0]
    funding_stats[0, 1] = grid_topups
    funding_stats[0, 2] = grid_peak_invested
    funding_stats[0, 3] = grid_peak_day
    funding_stats[0, 4] = grid_invested
    funding_stats[0, 5] = last_equity[0]
    funding_stats[0, 6] = last_equity[0] - grid_invested
    funding_stats[0, 7] = np.nan
    funding_stats[0, 8] = 1.0 if top_up_mode else 0.0
    baseline_position = max(1.0, float(seed_shares))
    funding_stats[0, 9] = actual_min_position / baseline_position
    funding_stats[0, 10] = actual_max_position / baseline_position
    funding_stats[0, 11] = stockout_count
    funding_stats[0, 12] = tplus1_block_count
    funding_stats[0, 13] = floor_episode_count
    funding_stats[0, 14] = floor_gamma_min * floor_grid_squares / (floor_k_ref * floor_k_ref) if floor_enabled else np.nan
    funding_stats[0, 15] = floor_grid_squares
    if grid_peak_invested > 0:
        metrics[0, 0] = (last_equity[0] - grid_invested) / grid_peak_invested
    if matched_enabled:
        funding_stats[0, 16] = metrics[3, 0]
        funding_stats[0, 17] = metrics[3, 2]
    funding_stats[0, 18] = metrics[0, 0]
    funding_stats[0, 19] = metrics[0, 2]
    funding_stats[0, 20] = initial_equities[3]
    funding_stats[0, 21] = matched_invested
    funding_stats[0, 22] = matched_peak_invested - initial_equities[3]
    funding_stats[0, 23] = matched_peak_invested
    return (
        metrics, equity_values, equity_cash, equity_shares, equity_drawdown,
        period_values, trade_bar[:trade_n], trade_strategy[:trade_n],
        trade_side[:trade_n], trade_quantity[:trade_n], trade_price[:trade_n],
        trade_fees[:trade_n], trade_cash[:trade_n], trade_shares[:trade_n],
        trade_reason[:trade_n], trade_topup[:trade_n],
        rejection_bar[:rejection_n], rejection_ts[:rejection_n],
        rejection_strategy[:rejection_n], rejection_side[:rejection_n],
        rejection_reason[:rejection_n], rejection_quantity[:rejection_n], invested_values,
        fold_stats, floor_episode_grids, funding_stats,
    )


@_jit(cache=True, parallel=True)
def simulate_research_batch(arrays, options, parameters, fold_ids, fold_count, account_mode):
    """Parallel independent ledgers; each ledger's time axis stays serial.

    ``account_mode`` reaches the kernel unchanged, so a research batch runs the
    grid and its matched-funding baseline without the DCA and buy-and-hold
    ledgers.
    """
    count = len(parameters)
    # Scoring consumes peak-principal summaries only; no day-by-candidate
    # return tensor is allocated.
    metrics = np.zeros((count, 11), dtype=np.float64)
    folds = np.zeros((count, fold_count, 9), dtype=np.float64)
    funding = np.zeros((count, 24), dtype=np.float64)
    episodes = [np.empty(0, dtype=np.int64) for _ in range(count)]
    empty_days = np.zeros(0, dtype=np.int64)
    empty_values = np.zeros((0, 7), dtype=np.float64)
    for index in prange(count):
        p = parameters[index]
        individual = (options[0], options[1], options[2], p[0], p[1], options[5],
            p[2], p[3], p[4], options[9], options[10], options[11], int(p[5]), int(p[6]),
            options[14], options[15], options[16], options[17], options[18], options[19],
            options[20], options[21], options[22], options[23], options[24], options[25],
            options[26], options[27], options[28], options[29], options[30], options[31], options[32],
            empty_days, empty_values,
            fold_ids, fold_count, False, False, account_mode)
        output = simulate(*(arrays + individual))
        metrics[index] = output[0][0]
        folds[index] = output[-3]
        episodes[index] = output[-2]
        funding[index] = output[-1][0]
    return metrics, folds, episodes, funding

