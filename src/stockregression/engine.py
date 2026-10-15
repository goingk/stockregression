"""Event-driven A-share backtest engine.

Signals are evaluated from real bar closes and discretionary orders fill at the
next real bar open.  Scheduled first-position and DCA orders are submitted at
their scheduled opening events.  All account state is private to one strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

try:  # Numba is declared by the project; keep the reference engine usable without it.
    from numba import njit
except ImportError:  # pragma: no cover - exercised only by minimal installations
    njit = None

from .metrics import summarize_metrics
from .models import BacktestConfig, DataBundle
from .rules import (
    as_date,
    calculated_price_limits,
    commissions_and_taxes,
    fee_rates,
    floor_buy_quantity,
    infer_board,
    legal_sell_quantity,
    lot_size,
    round_to_tick,
    round_money,
    transfer_fee_terms,
)


_NUMBA_FASTPATH_ENABLED = True
_FAST_INPUT_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


@dataclass
class _Order:
    side: str
    quantity: int
    signal_timestamp: Any
    reason: str


@dataclass
class _Ledger:
    name: str
    cash: float
    keep_details: bool = True
    shares: int = 0
    lots: list[tuple[date, int, bool]] = field(default_factory=list)
    cash_receivable: float = 0.0
    share_receivable: float = 0.0
    entitlements: dict[int, tuple[float, float]] = field(default_factory=dict)
    accrued_actions: dict[int, tuple[float, float, bool, bool]] = field(default_factory=dict)
    pending: _Order | None = None
    started: bool = False
    reference: float | None = None
    fixed_anchor: float | None = None
    active_cell: int | None = None
    dca_budget: float = 0.0
    dca_due: bool = False
    dca_last_attempt: date | None = None
    equity: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    equity_values: list[float] = field(default_factory=list)
    equity_timestamps: list[Any] = field(default_factory=list)
    daily_returns: list[dict[str, Any]] = field(default_factory=list)
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_fees: float = 0.0
    gross_turnover: float = 0.0
    last_trade_price: float = 0.0
    last_trade_notional: float = 0.0
    last_trade_fee: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    first_timestamp: Any = None
    last_timestamp: Any = None
    last_equity: float = 0.0
    last_day_equity: float | None = None


@dataclass(frozen=True)
class _Action:
    index: int
    record_date: date
    ex_date: date
    pay_date: date | None
    stock_date: date | None
    cash_per_share: float
    bonus_ratio: float


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    return default if _is_missing(value) else value


def _date_or_none(value: Any) -> date | None:
    return None if _is_missing(value) else as_date(value)


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return datetime.fromisoformat(str(value))


def _iso(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def open_timestamp(timestamp: Any, frequency: str) -> datetime:
    """Estimate the session opening timestamp of a bar labelled by its end."""
    end = _to_datetime(timestamp)
    day = end.date()
    if frequency == "1d":
        return datetime.combine(day, time(9, 30))
    minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60}.get(frequency, 1)
    start = end - timedelta(minutes=minutes)
    morning = time(9, 30)
    afternoon = time(13, 0)
    if end.time() <= time(11, 30) and start.time() < morning:
        return datetime.combine(day, morning)
    if end.time() > time(11, 30) and end.time() <= time(15, 0) and start.time() < afternoon:
        return datetime.combine(day, afternoon)
    return start


def _board_name(config: BacktestConfig, data: DataBundle) -> str:
    return infer_board(config.symbol, data.metadata)


def _fee_settings(config: BacktestConfig) -> tuple[float, float, float, float]:
    fees = config.fees
    # ``getattr`` preserves compatibility while older saved configs and callers
    # are upgraded to the exchange-history fee fields.
    return (
        float(getattr(fees, "share_face_value", 1.0)),
        float(getattr(fees, "legacy_sh_transfer_rate", 0.001)),
        float(getattr(fees, "legacy_sh_min_transfer", 1.0)),
        float(getattr(fees, "legacy_sz_transfer_rate", 0.0000255)),
    )


def _legacy_fee_warnings(config: BacktestConfig, end: date) -> list[str]:
    if config.symbol.endswith(".SH") and config.start_date < date(2012, 6, 1) and end >= config.start_date:
        face_value, rate, minimum, _ = _fee_settings(config)
        return [
            "上海市场2012-06-01前过户费历史沿革未完全复原；按可用的2009年上交所规则估算，"
            f"默认每边按每股面值{face_value:g}元×成交股数×{rate:g}且最低{minimum:g}元。"
            "可在手续费配置中调整面值、费率和最低额。"
        ]
    return []


def _prepare_actions(data: DataBundle, start: date, end: date) -> list[_Action]:
    actions = data.actions
    if actions is None or actions.empty:
        return []
    required = {"record_date", "ex_date", "cash_per_share", "bonus_ratio"}
    missing = required.difference(actions.columns)
    if missing:
        raise ValueError(f"公司行动数据缺少字段：{', '.join(sorted(missing))}")
    result: list[_Action] = []
    for i, row in enumerate(actions.to_dict("records")):
        reason = str(_value(row, "unsupported_reason", "") or "").strip()
        if reason:
            raise ValueError(f"公司行动第 {i + 1} 条暂不支持：{reason}")
        implemented = _value(row, "implemented", True)
        if not bool(implemented):
            raise ValueError(f"公司行动第 {i + 1} 条未确认已实施，无法准确回测")
        record_date = _date_or_none(_value(row, "record_date"))
        ex_date = _date_or_none(_value(row, "ex_date"))
        cash = float(_value(row, "cash_per_share", 0.0) or 0.0)
        bonus = float(_value(row, "bonus_ratio", 0.0) or 0.0)
        pay_date = _date_or_none(_value(row, "pay_date"))
        stock_date = _date_or_none(_value(row, "stock_date"))
        if record_date is None or ex_date is None:
            raise ValueError(f"公司行动第 {i + 1} 条缺少登记日或除权除息日")
        if cash > 0 and pay_date is None:
            raise ValueError(f"公司行动第 {i + 1} 条有现金红利但缺少派息日")
        if bonus > 0 and stock_date is None:
            raise ValueError(f"公司行动第 {i + 1} 条有送转股份但缺少可交易日")
        if pay_date is not None and pay_date < ex_date:
            raise ValueError(f"公司行动第 {i + 1} 条派息日早于除权除息日")
        if stock_date is not None and stock_date < ex_date:
            raise ValueError(f"公司行动第 {i + 1} 条股份可交易日早于除权除息日")
        if record_date <= end and ex_date >= start:
            result.append(_Action(i, record_date, ex_date, pay_date, stock_date, cash, bonus))
    return result


def _prepare_market(config: BacktestConfig, data: DataBundle):
    start = config.start_date
    end = config.end_date or (max(data.calendar.date) if len(data.calendar) else None)
    if end is None:
        if data.bars is not None and not data.bars.empty:
            end = max(as_date(value) for value in data.bars["timestamp"])
        elif data.daily is not None and not data.daily.empty:
            end = max(as_date(value) for value in data.daily["date"])
        else:
            end = start

    bars_by_day: dict[date, list[dict[str, Any]]] = {}
    if data.bars is not None and not data.bars.empty:
        records = data.bars.to_dict("records")
        for row in records:
            ts = _value(row, "timestamp")
            if ts is None:
                continue
            day = as_date(ts)
            if day < start or day > end or bool(_value(row, "is_synthetic", False)):
                continue
            frequency = str(_value(row, "frequency", config.frequency))
            row["_timestamp"] = _to_datetime(ts)
            row["_date"] = day
            row["_frequency"] = frequency
            row["_open_timestamp"] = open_timestamp(ts, frequency)
            bars_by_day.setdefault(day, []).append(row)
        for rows in bars_by_day.values():
            rows.sort(key=lambda x: x["_timestamp"])

    daily_by_day: dict[date, dict[str, Any]] = {}
    if data.daily is not None and not data.daily.empty:
        for row in data.daily.to_dict("records"):
            raw_date = _value(row, "date")
            if raw_date is None:
                continue
            day = as_date(raw_date)
            if start <= day <= end:
                daily_by_day[day] = row

    calendar_days: set[date] = set()
    if len(data.calendar):
        calendar_days.update(day for day in (as_date(value) for value in data.calendar) if start <= day <= end)
    calendar_days.update(bars_by_day)
    calendar_days.update(daily_by_day)
    quality_days = {}
    for row in (data.quality or {}).get("days", []):
        try:
            quality_days[as_date(row["date"])] = row
        except (KeyError, TypeError, ValueError):
            continue

    # The root data layer emits a daily fallback row for daily-mode sessions.  Keep
    # an explicit fallback here for direct engine callers with only a daily frame.
    for day in list(calendar_days):
        if day in bars_by_day or day not in daily_by_day:
            continue
        day_quality = quality_days.get(day, {})
        if config.frequency == "1d" or day_quality.get("mode") == "daily":
            daily = daily_by_day[day]
            ts = datetime.combine(day, time(15, 0))
            bars_by_day[day] = [{
                "timestamp": ts, "open": _value(daily, "open"),
                "high": _value(daily, "high"), "low": _value(daily, "low"),
                "close": _value(daily, "close"), "volume": _value(daily, "volume", 0),
                "turnover": _value(daily, "turnover", 0), "frequency": "1d",
                "tradable": True, "is_synthetic": False, "preclose": _value(daily, "preclose"),
                "is_st": _value(daily, "is_st", False), "limit_up": _value(daily, "limit_up"),
                "limit_down": _value(daily, "limit_down"), "_timestamp": ts, "_date": day,
                "_frequency": "1d", "_open_timestamp": datetime.combine(day, time(9, 30)),
            }]

    return end, sorted(calendar_days), bars_by_day, daily_by_day, quality_days


# The general event loop below is intentionally kept as the readable source of
# truth for mixed-frequency data, suspensions, synthetic bars, and corporate
# actions.  Optimization repeatedly evaluates ordinary, all-minute histories;
# this compiled array kernel handles that common shape without constructing a
# Python dictionary for every bar or retaining summary-only equity paths.
if njit is None:  # pragma: no cover - Numba is a declared runtime dependency
    def _jit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate
else:
    _jit = njit


@_jit(cache=True)
def _nb_round_money(value: float) -> float:
    """Half-up to cents while absorbing binary error from exact cent ties."""
    if value >= 0:
        return math.floor(value * 100.0 + 0.5 + 1e-10) / 100.0
    return -math.floor(-value * 100.0 + 0.5 + 1e-10) / 100.0


@_jit(cache=True)
def _nb_buy_lot(quantity: float, board_code: int) -> int:
    if not math.isfinite(quantity) or quantity <= 0:
        return 0
    if board_code == 2:
        result = int(math.floor(quantity + 1e-9))
        return result if result >= 200 else 0
    return int(math.floor(quantity / 100.0 + 1e-9)) * 100


@_jit(cache=True)
def _nb_sell_lot(requested: int, available: int, board_code: int) -> int:
    quantity = min(max(0, requested), max(0, available))
    if quantity == available and (available % 100 or (board_code == 2 and available < 200)):
        return quantity
    if board_code == 2:
        return quantity if quantity >= 200 else 0
    return (quantity // 100) * 100


@_jit(cache=True)
def _nb_cell(price: float, anchor: float, grid_mode: int, step: float) -> int:
    if price <= 0 or anchor <= 0:
        return 0
    if grid_mode == 1:
        return int(math.floor((price - anchor) / max(anchor * step, 0.01) + 1e-10))
    return int(math.floor(math.log(price / anchor) / math.log(1.0 + step) + 1e-10))


@_jit(cache=True)
def _nb_quantity(mode: int, value: float, equity: float, price: float) -> int:
    if price <= 0:
        return 0
    if mode == 0:
        return int(value)
    if mode == 1:
        return max(0, int(math.floor(value / price)))
    return max(0, int(math.floor(value * equity / price)))


@_jit(cache=True)
def _nb_attempt(
    side: int, requested: int, opening: float, volume: float,
    limit_up: float, limit_down: float, stamp_rate: float, transfer_rate: float,
    transfer_is_face_value: bool, transfer_minimum: float,
    stamp_bilateral: bool, day_id: int, cash: float, cash_receivable: float,
    share_receivable: float, shares: int, sellable: int,
    max_position: float, min_position: float, commission_rate: float,
    min_commission: float, slippage: float, board_code: int, face_value: float,
) -> tuple:
    """Return fill state and a compact rejection code; side 1=buy, 2=sell."""
    if volume <= 0 or not math.isfinite(volume):
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 8
    if side == 1 and math.isfinite(limit_up) and opening >= limit_up - 0.005:
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 1
    if side == 2 and math.isfinite(limit_down) and opening <= limit_down + 0.005:
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, requested, 2

    raw_price = opening * (1.0 + slippage if side == 1 else 1.0 - slippage)
    if side == 1:
        price = math.ceil(raw_price / 0.01 - 1e-10) * 0.01
    else:
        price = math.floor(raw_price / 0.01 + 1e-10) * 0.01
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
        max_by_position = max(0.0, max_position * equity_open - position_shares * opening) / price
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
        min_position_shares = max(0.0, min_position * equity_open / price)
        position_capacity = max(0, int(math.floor(position_shares - min_position_shares + 1e-9)))
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
        quantity -= 1 if board_code == 2 else 100
        if board_code == 2 and quantity < 200:
            quantity = 0
    if quantity <= 0:
        return False, cash, shares, sellable, 0, 0.0, 0.0, 0.0, normalized_request, 5
    if side == 1:
        return True, max(0.0, cash_after), shares + quantity, sellable, quantity, price, fee, notional, normalized_request, 0
    cash_after = _nb_round_money(cash + notional - fee)
    return True, cash_after, shares - quantity, sellable - quantity, quantity, price, fee, notional, normalized_request, 0


@_jit(cache=True)
def _nb_simulate(
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
    initial_cash: float, initial_position: float, min_position: float, max_position: float,
    grid_mode: int, buy_step: float, sell_step: float, grid_step: float,
    reference_price: float, lower_price: float, upper_price: float,
    buy_mode: int, buy_value: float, sell_mode: int, sell_value: float,
    dca_allowance: float, board_code: int, commission_rate: float,
    min_commission: float, slippage: float, face_value: float, keep_details: bool,
) -> tuple:
    n = len(opens)
    day_count = 0
    for flag in last_bar_day:
        if flag:
            day_count += 1
    cash = np.full(3, initial_cash, dtype=np.float64)
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
    dca_budget = 0.0
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
    peak_equity = np.full(3, initial_cash, dtype=np.float64)
    max_drawdown = np.zeros(3, dtype=np.float64)
    last_equity = np.full(3, initial_cash, dtype=np.float64)
    daily_return_sum = np.zeros(3, dtype=np.float64)
    daily_return_sumsq = np.zeros(3, dtype=np.float64)
    prior_day_equity = np.full(3, initial_cash, dtype=np.float64)
    daily_return_count = 0
    first_timestamp = 0
    last_timestamp = 0
    for index in range(n):
        if tradable_mask[index]:
            if first_timestamp == 0:
                first_timestamp = timestamps_ns[index]
            last_timestamp = timestamps_ns[index]

    if keep_details:
        equity_values = np.zeros((3, n), dtype=np.float64)
        equity_cash = np.zeros((3, n), dtype=np.float64)
        equity_shares = np.zeros((3, n), dtype=np.float64)
        equity_drawdown = np.zeros((3, n), dtype=np.float64)
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
        equity_values = np.zeros((3, 0), dtype=np.float64)
        equity_cash = np.zeros((3, 0), dtype=np.float64)
        equity_shares = np.zeros((3, 0), dtype=np.float64)
        equity_drawdown = np.zeros((3, 0), dtype=np.float64)
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
            dca_budget += dca_allowance
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
                    shares[account], sellable[account], max_position, min_position,
                    commission_rate, min_commission, slippage, board_code, face_value,
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

            if not traded and account == 0 and not started[account]:
                if initial_position == 0:
                    started[account] = True
                    reference[account] = reference_price if reference_price > 0 else opening
                    fixed_anchor[account] = reference_price if reference_price > 0 else opening
                    active_cell[account] = _nb_cell(opening, fixed_anchor[account], grid_mode, grid_step)
                    has_active_cell[account] = True
                else:
                    raw_quantity = int(math.floor(initial_cash * initial_position / opening)) if opening > 0 else 0
                    quantity = _nb_buy_lot(raw_quantity, board_code)
                    result = _nb_attempt(
                        1, quantity, opening, volume, limit_ups[i], limit_downs[i],
                        stamp_rates[i], transfer_rates[i], transfer_is_face_value[i],
                        transfer_minimums[i], stamp_bilateral[i], day_id, cash[account],
                        cash_receivable[account], share_receivable[account],
                        shares[account], sellable[account], max_position, min_position,
                        commission_rate, min_commission, slippage, board_code, face_value,
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
                        cash[account], shares[account], sellable[account] = result[1], result[2], result[3]
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
                            trade_reason[trade_n] = 1
                            trade_n += 1
                        started[account] = True
                        reference[account] = result[5]
                        fixed_anchor[account] = reference_price if reference_price > 0 else result[5]
                        active_cell[account] = _nb_cell(result[5], fixed_anchor[account], grid_mode, grid_step)
                        has_active_cell[account] = True
                        traded = True

            if not traded and account == 1 and dca_due and dca_last_day != day_id:
                dca_last_day = day_id
                if dca_budget > 0 and opening > 0:
                    quantity = _nb_buy_lot(int(math.floor(dca_budget / opening)), board_code)
                    result = _nb_attempt(
                        1, quantity, opening, volume, limit_ups[i], limit_downs[i],
                        stamp_rates[i], transfer_rates[i], transfer_is_face_value[i],
                        transfer_minimums[i], stamp_bilateral[i], day_id, cash[account],
                        cash_receivable[account], share_receivable[account],
                        shares[account], sellable[account], max_position, min_position,
                        commission_rate, min_commission, slippage, board_code, face_value,
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
                        cash[account], shares[account], sellable[account] = result[1], result[2], result[3]
                        dca_budget = max(0.0, dca_budget - result[7] - result[6])
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

            if not traded and account == 2 and not started[account]:
                result = _nb_attempt(
                    1, 2147483647, opening, volume, limit_ups[i], limit_downs[i],
                    stamp_rates[i], transfer_rates[i], transfer_is_face_value[i],
                    transfer_minimums[i], stamp_bilateral[i], day_id, cash[account],
                    cash_receivable[account], share_receivable[account],
                    shares[account], sellable[account], max_position, min_position,
                    commission_rate, min_commission, slippage, board_code, face_value,
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
                    cash[account], shares[account], sellable[account] = result[1], result[2], result[3]
                    started[account] = True
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
                        trade_reason[trade_n] = 3
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
                if reference[0] > 0 and close <= reference[0] * (1.0 - buy_step) and position_close * close < max_position * equity_close:
                    intent_side = 1
                    intent_quantity = _nb_quantity(buy_mode, buy_value, equity_close, close)
                    intent_reason = 4
                elif reference[0] > 0 and close >= reference[0] * (1.0 + sell_step) and position_close * close > min_position * equity_close:
                    intent_side = 2
                    intent_quantity = _nb_quantity(sell_mode, sell_value, equity_close, close)
                    intent_reason = 5
            else:
                anchor = fixed_anchor[0] if fixed_anchor[0] > 0 else reference[0]
                index = _nb_cell(close, anchor, grid_mode, grid_step)
                if not has_active_cell[0]:
                    active_cell[0] = _nb_cell(reference[0] if reference[0] > 0 else close, anchor, grid_mode, grid_step)
                    has_active_cell[0] = True
                lower_ok = lower_price <= 0 or close >= lower_price
                upper_ok = upper_price <= 0 or close <= upper_price
                if index < active_cell[0] and lower_ok and position_close * close < max_position * equity_close:
                    intent_side = 1
                    intent_quantity = _nb_quantity(buy_mode, buy_value, equity_close, close)
                    intent_reason = 4
                elif index > active_cell[0] and upper_ok and position_close * close > min_position * equity_close:
                    intent_side = 2
                    intent_quantity = _nb_quantity(sell_mode, sell_value, equity_close, close)
                    intent_reason = 5
            if intent_side != 0:
                clipped_quantity = intent_quantity
                if intent_side == 1:
                    maximum = max(0.0, max_position * equity_close - position_close * close) / close if close > 0 else 0.0
                    clipped_quantity = min(intent_quantity, _nb_buy_lot(maximum, board_code))
                    clipped_quantity = _nb_buy_lot(clipped_quantity, board_code)
                else:
                    max_sell = max(0.0, position_close - min_position * equity_close / close) if close > 0 else 0.0
                    clipped_quantity = min(intent_quantity, max(0, int(math.floor(max_sell + 1e-9))))
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
            if keep_details:
                equity_values[account, i] = equity
                equity_cash[account, i] = cash[account]
                equity_shares[account, i] = shares[account] + share_receivable[account]
                equity_drawdown[account, i] = drawdown

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

    metrics = np.zeros((3, 10), dtype=np.float64)
    elapsed = (last_timestamp - first_timestamp) / 1_000_000_000.0 / (365.25 * 24.0 * 3600.0) if n else 0.0
    elapsed = max(elapsed, 1.0 / 252.0)
    for account in range(3):
        total_return = last_equity[account] / initial_cash - 1.0 if initial_cash > 0 else 0.0
        cagr = 0.0
        if last_equity[account] > 0 and initial_cash > 0:
            log_growth = math.log(last_equity[account] / initial_cash) / elapsed
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
        metrics[account, 7] = gross_turnover[account] / initial_cash if initial_cash > 0 else 0.0
        metrics[account, 8] = buy_count[account]
        metrics[account, 9] = sell_count[account]

    return (
        metrics, equity_values, equity_cash, equity_shares, equity_drawdown,
        period_values, trade_bar[:trade_n], trade_strategy[:trade_n],
        trade_side[:trade_n], trade_quantity[:trade_n], trade_price[:trade_n],
        trade_fees[:trade_n], trade_cash[:trade_n], trade_shares[:trade_n],
        trade_reason[:trade_n], rejection_bar[:rejection_n], rejection_ts[:rejection_n],
        rejection_strategy[:rejection_n], rejection_side[:rejection_n],
        rejection_reason[:rejection_n], rejection_quantity[:rejection_n],
    )


def _cell(price: float, anchor: float, config: BacktestConfig) -> int:
    if price <= 0 or anchor <= 0:
        return 0
    if config.grid_mode == "arithmetic":
        step = max(anchor * config.grid_step, 0.01)
        return int(math.floor((price - anchor) / step + 1e-10))
    ratio = 1 + config.grid_step
    return int(math.floor(math.log(price / anchor) / math.log(ratio) + 1e-10))


def _position_shares(ledger: _Ledger) -> float:
    return ledger.shares + ledger.share_receivable


def _position_ratio(shares: float, price: float, equity: float) -> float:
    """Return marked stock exposure as a share of total account equity."""
    if price <= 0 or equity <= 0:
        return 0.0
    return shares * price / equity


def _equity_at(ledger: _Ledger, price: float) -> float:
    return ledger.cash + ledger.cash_receivable + _position_shares(ledger) * price


def _quantity_from_config(quantity: Any, equity: float, price: float) -> int:
    if price <= 0:
        return 0
    if quantity.mode == "shares":
        return int(quantity.value)
    if quantity.mode == "cash":
        return max(0, int(math.floor(quantity.value / price)))
    return max(0, int(math.floor(quantity.value * equity / price)))


def _available_to_sell(ledger: _Ledger, day: date) -> int:
    return sum(quantity for acquired, quantity, same_day_allowed in ledger.lots
               if acquired < day or (acquired == day and same_day_allowed))


def _consume_sellable_lots(ledger: _Ledger, day: date, quantity: int) -> None:
    remaining = quantity
    kept: list[tuple[date, int, bool]] = []
    for acquired, lots_quantity, same_day_allowed in ledger.lots:
        sellable = acquired < day or (acquired == day and same_day_allowed)
        if sellable and remaining:
            used = min(lots_quantity, remaining)
            lots_quantity -= used
            remaining -= used
        if lots_quantity:
            kept.append((acquired, lots_quantity, same_day_allowed))
    ledger.lots = kept


def _limits(row: dict[str, Any], daily: dict[str, Any] | None, board: str, day: date, is_st: bool):
    no_price_limit = bool(_value(row, "no_price_limit", False)) or bool(
        _value(daily or {}, "no_price_limit", False)
    )
    if no_price_limit:
        return None, None
    up = _value(row, "limit_up")
    down = _value(row, "limit_down")
    if (up is None or down is None) and daily:
        up = up if up is not None else _value(daily, "limit_up")
        down = down if down is not None else _value(daily, "limit_down")
    preclose = _value(row, "preclose")
    if preclose is None and daily:
        preclose = _value(daily, "preclose")
    if up is None or down is None:
        calculated_up, calculated_down = calculated_price_limits(
            float(preclose) if preclose is not None else None, board, day, is_st
        )
        up = calculated_up if up is None else float(up)
        down = calculated_down if down is None else float(down)
    return (float(up) if up is not None else None, float(down) if down is not None else None)


def _record_rejection(ledger: _Ledger, timestamp: Any, side: str, reason: str, requested: int) -> None:
    if ledger.keep_details:
        ledger.rejections.append({
            "timestamp": _iso(timestamp), "side": side, "reason": reason,
            "requested_quantity": int(requested),
        })


def _attempt_trade(
    ledger: _Ledger,
    side: str,
    requested: int,
    row: dict[str, Any],
    day: date,
    board: str,
    config: BacktestConfig,
    daily: dict[str, Any] | None,
    reason: str,
) -> tuple[bool, str | None]:
    open_price = float(_value(row, "open", 0) or 0)
    timestamp = row["_open_timestamp"]
    volume = float(_value(row, "volume", 0) or 0)
    tradable = bool(_value(row, "tradable", True))
    if not tradable:
        _record_rejection(ledger, timestamp, side, "non_tradable_bar", requested)
        return False, "non_tradable_bar"
    if not math.isfinite(open_price) or open_price <= 0:
        _record_rejection(ledger, timestamp, side, "invalid_open", requested)
        return False, "invalid_open"
    if not math.isfinite(volume) or volume <= 0:
        _record_rejection(ledger, timestamp, side, "zero_volume", requested)
        return False, "zero_volume"
    is_st = bool(_value(row, "is_st", _value(daily or {}, "is_st", False)))
    limit_up, limit_down = _limits(row, daily, board, day, is_st)
    tick = 0.01
    # At an opening limit price the queue position is unknowable from OHLC bars;
    # use the conservative assumption that a locked order does not fill.
    if side == "buy" and limit_up is not None and open_price >= limit_up - tick / 2:
        _record_rejection(ledger, timestamp, side, "limit_up_locked", requested)
        return False, "limit_up_locked"
    if side == "sell" and limit_down is not None and open_price <= limit_down + tick / 2:
        _record_rejection(ledger, timestamp, side, "limit_down_locked", requested)
        return False, "limit_down_locked"

    slip = config.fees.slippage_bps / 10_000
    price = round_to_tick(open_price * (1 + slip if side == "buy" else 1 - slip), tick, side=side)
    if limit_up is not None:
        price = min(price, limit_up)
    if limit_down is not None:
        price = max(price, limit_down)
    if price <= 0:
        _record_rejection(ledger, timestamp, side, "invalid_execution_price", requested)
        return False, "invalid_execution_price"

    equity_open = max(0.0, _equity_at(ledger, open_price))
    held_value = _position_shares(ledger) * open_price
    face_value, legacy_sh_rate, legacy_sh_minimum, legacy_sz_rate = _fee_settings(config)
    transfer_rate, transfer_is_face_value, transfer_minimum = transfer_fee_terms(
        day,
        config.symbol,
        legacy_sh_transfer_rate=legacy_sh_rate,
        legacy_sh_min_transfer=legacy_sh_minimum,
        legacy_sz_transfer_rate=legacy_sz_rate,
    )
    if side == "buy":
        requested = floor_buy_quantity(min(max(0, int(requested)), 2**31 - 1), board)
        max_by_position = max(0.0, config.max_position * equity_open - held_value) / price
        position_quantity = floor_buy_quantity(max_by_position, board)
        effective_transfer_rate = transfer_rate * face_value / price if transfer_is_face_value else transfer_rate
        max_by_cash = max(0.0, ledger.cash - config.fees.min_commission - transfer_minimum) / (
            price * (1 + config.fees.commission_rate + effective_transfer_rate)
        )
        affordable_quantity = floor_buy_quantity(max_by_cash, board)
        quantity = min(requested, position_quantity, affordable_quantity)
        if quantity <= 0:
            if requested <= 0:
                failure = "below_lot"
            elif position_quantity <= 0:
                failure = "position_limit"
            else:
                failure = "insufficient_cash"
            _record_rejection(ledger, timestamp, side, failure, requested)
            return False, failure
        while quantity > 0:
            notional = round_money(quantity * price)
            charges = commissions_and_taxes(
                notional, side, day, config.fees.commission_rate, config.fees.min_commission,
                symbol=config.symbol, quantity=quantity, face_value=face_value,
                legacy_sh_transfer_rate=legacy_sh_rate,
                legacy_sh_min_transfer=legacy_sh_minimum,
                legacy_sz_transfer_rate=legacy_sz_rate,
            )
            cash_after = round_money(ledger.cash - notional - charges["total"])
            if cash_after >= -0.005:
                break
            quantity -= 1 if board == "star" else 100
            if board == "star" and quantity < 200:
                quantity = 0
        if quantity <= 0 or cash_after < -0.005:
            _record_rejection(ledger, timestamp, side, "insufficient_cash", requested)
            return False, "insufficient_cash"
        ledger.cash = max(0.0, cash_after)
        ledger.shares += quantity
        ledger.lots.append((day, quantity, False))
    else:
        available = _available_to_sell(ledger, day)
        requested = max(0, int(requested))
        min_position_shares = max(0.0, config.min_position * equity_open / price)
        position_capacity = max(0, int(math.floor(_position_shares(ledger) - min_position_shares + 1e-9)))
        sellable_capacity = min(available, position_capacity)
        quantity = legal_sell_quantity(min(requested, sellable_capacity), sellable_capacity, board)
        if quantity <= 0:
            if available <= 0 and ledger.shares > 0:
                failure = "t_plus_1"
            elif position_capacity <= 0:
                failure = "position_limit"
            elif requested <= 0:
                failure = "below_lot"
            else:
                failure = "insufficient_sellable_shares"
            _record_rejection(ledger, timestamp, side, failure, requested)
            return False, failure
        notional = round_money(quantity * price)
        charges = commissions_and_taxes(
            notional, side, day, config.fees.commission_rate, config.fees.min_commission,
            symbol=config.symbol, quantity=quantity, face_value=face_value,
            legacy_sh_transfer_rate=legacy_sh_rate,
            legacy_sh_min_transfer=legacy_sh_minimum,
            legacy_sz_transfer_rate=legacy_sz_rate,
        )
        ledger.cash = round_money(ledger.cash + notional - charges["total"])
        ledger.shares -= quantity
        _consume_sellable_lots(ledger, day, quantity)

    fee = charges["total"]
    ledger.trade_count += 1
    ledger.buy_count += side == "buy"
    ledger.sell_count += side == "sell"
    ledger.total_fees += fee
    ledger.gross_turnover += notional
    ledger.last_trade_price = float(price)
    ledger.last_trade_notional = float(notional)
    ledger.last_trade_fee = float(fee)
    if ledger.keep_details:
        ledger.trades.append({
            "timestamp": _iso(timestamp), "side": side, "quantity": int(quantity),
            "price": float(price), "fees": float(fee), "cash": float(ledger.cash),
            "shares": int(ledger.shares), "reason": reason,
        })
    return True, None


def _grid_intent(ledger: _Ledger, close: float, equity: float, config: BacktestConfig) -> _Order | None:
    if not ledger.started or close <= 0:
        return None
    if config.grid_mode == "trailing":
        reference = ledger.reference or ledger.fixed_anchor
        if not reference:
            return None
        if close <= reference * (1 - config.buy_step) and ledger.shares * close + ledger.share_receivable * close < config.max_position * equity:
            quantity = _quantity_from_config(config.buy_quantity, equity, close)
            return _Order("buy", quantity, None, "grid_buy")
        if close >= reference * (1 + config.sell_step) and _position_shares(ledger) * close > config.min_position * equity:
            quantity = _quantity_from_config(config.sell_quantity, equity, close)
            return _Order("sell", quantity, None, "grid_sell")
        return None

    anchor = ledger.fixed_anchor or ledger.reference
    if not anchor:
        return None
    index = _cell(close, anchor, config)
    if ledger.active_cell is None:
        ledger.active_cell = _cell(ledger.reference or close, anchor, config)
    lower_ok = config.lower_price is None or close >= config.lower_price
    upper_ok = config.upper_price is None or close <= config.upper_price
    if index < ledger.active_cell and lower_ok and _position_shares(ledger) * close < config.max_position * equity:
        return _Order("buy", _quantity_from_config(config.buy_quantity, equity, close), None, "grid_buy")
    if index > ledger.active_cell and upper_ok and _position_shares(ledger) * close > config.min_position * equity:
        return _Order("sell", _quantity_from_config(config.sell_quantity, equity, close), None, "grid_sell")
    return None


def _update_reference_after_fill(ledger: _Ledger, price: float, config: BacktestConfig) -> None:
    if config.grid_mode == "trailing":
        ledger.reference = price
        return
    anchor = ledger.fixed_anchor or config.reference_price or price
    ledger.fixed_anchor = float(anchor)
    ledger.reference = ledger.reference or float(anchor)
    ledger.active_cell = _cell(price, ledger.fixed_anchor, config)


def _accept_grid_intent(
    ledger: _Ledger,
    intent: _Order,
    close: float,
    timestamp: Any,
    board: str,
    config: BacktestConfig,
) -> None:
    quantity = intent.quantity
    equity = _equity_at(ledger, close)
    if intent.side == "buy":
        held = _position_shares(ledger) * close
        maximum = max(0.0, config.max_position * equity - held) / close if close > 0 else 0
        quantity = min(quantity, floor_buy_quantity(maximum, board))
        quantity = floor_buy_quantity(quantity, board)
    else:
        max_sell = max(0.0, _position_shares(ledger) - config.min_position * equity / close) if close > 0 else 0
        quantity = min(quantity, int(math.floor(max_sell + 1e-9)))
        quantity = max(0, int(quantity))
    if quantity <= 0:
        _record_rejection(ledger, timestamp, intent.side, "position_limit_or_below_lot", intent.quantity)
        return
    ledger.pending = _Order(intent.side, quantity, timestamp, intent.reason)


def _dca_period_key(day: date, frequency: str):
    if frequency == "weekly":
        return day.isocalendar().year, day.isocalendar().week
    return day.year, day.month


def _actions_by_date(actions: list[_Action]):
    result: dict[date, dict[str, list[int]]] = {}
    for action in actions:
        for key, day in (("record", action.record_date), ("ex", action.ex_date),
                         ("pay", action.pay_date), ("stock", action.stock_date)):
            if day is not None:
                result.setdefault(day, {}).setdefault(key, []).append(action.index)
    return result


def _daily_status(daily: dict[str, Any] | None) -> bool | None:
    if not daily or _value(daily, "tradestatus") is None:
        return None
    try:
        return int(float(_value(daily, "tradestatus"))) != 0
    except (TypeError, ValueError):
        return bool(_value(daily, "tradestatus"))


def _apply_ex_adjustment(
    day: date,
    daily: dict[str, Any] | None,
    prior_close: float | None,
    accounts: list[_Ledger],
    ids: list[int],
    action_map: dict[int, _Action],
    board: str,
    config: BacktestConfig,
) -> float | None:
    # Apply the official exchange reference ratio once per ex-date, including
    # cash-only events; it prevents a false price-grid crossing.
    factor = 1.0
    if ids and daily is not None and prior_close and prior_close > 0:
        ex_preclose = _value(daily, "preclose")
        if ex_preclose is not None and float(ex_preclose) > 0:
            factor = float(ex_preclose) / prior_close
        else:
            cash = sum(action_map[index].cash_per_share for index in ids)
            bonus = sum(action_map[index].bonus_ratio for index in ids)
            factor = max(0.0, (prior_close - cash) / (prior_close * (1 + bonus)))
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"{day.isoformat()} 公司行动的除权参考价无效")
    for account in accounts:
        if ids and factor != 1.0 and account.name == "grid":
            if account.reference is not None:
                account.reference *= factor
            if account.fixed_anchor is not None:
                account.fixed_anchor *= factor
        for index in ids:
            cash, bonus = account.entitlements.pop(index, (0.0, 0.0))
            account.cash_receivable += cash
            account.share_receivable += bonus
            account.accrued_actions[index] = (cash, bonus, False, False)
    return factor


def _settle_actions(day: date, accounts: list[_Ledger], events: dict[str, list[int]]) -> None:
    for account in accounts:
        for index in events.get("pay", []):
            state = account.accrued_actions.get(index)
            if state is None:
                continue
            cash, bonus, paid, stock_issued = state
            if not paid:
                account.cash_receivable = max(0.0, account.cash_receivable - cash)
                account.cash = round_money(account.cash + cash)
                account.accrued_actions[index] = (cash, bonus, True, stock_issued)
        for index in events.get("stock", []):
            state = account.accrued_actions.get(index)
            if state is None:
                continue
            cash, bonus, paid, stock_issued = state
            if not stock_issued:
                account.share_receivable = max(0.0, account.share_receivable - bonus)
                quantity = int(round(bonus))
                account.shares += quantity
                if quantity:
                    # Corporate-action shares are deliverable before this session opens.
                    account.lots.append((day, quantity, True))
                account.accrued_actions[index] = (cash, bonus, paid, True)


def _metric_summary(ledger: _Ledger, initial_cash: float) -> dict[str, Any]:
    return summarize_metrics(
        initial_cash=initial_cash, final_equity=ledger.last_equity,
        first_timestamp=ledger.first_timestamp, last_timestamp=ledger.last_timestamp,
        max_drawdown=ledger.max_drawdown, daily_returns=ledger.daily_returns,
        buy_count=ledger.buy_count, sell_count=ledger.sell_count,
        total_fees=ledger.total_fees, gross_turnover=ledger.gross_turnover,
    )


_METRIC_KEYS = (
    "total_return", "cagr", "max_drawdown", "sharpe", "calmar",
    "trade_count", "total_fees", "turnover", "buy_count", "sell_count",
)
_STRATEGY_NAMES = ("grid", "dca", "buy_hold")
_TRADE_REASONS = {1: "initial_position", 2: "dca", 3: "buy_hold", 4: "grid_buy", 5: "grid_sell"}
_REJECTION_REASONS = {
    1: "limit_up_locked", 2: "limit_down_locked", 3: "below_lot",
    4: "position_limit", 5: "insufficient_cash", 6: "t_plus_1",
    7: "insufficient_sellable_shares", 8: "zero_volume",
    9: "invalid_execution_price", 10: "position_limit_or_below_lot",
    11: "non_tradable_bar",
}


def _numeric_column(frame: pd.DataFrame, name: str, default: float = math.nan) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64, copy=False)


def _fast_options(config: BacktestConfig, board_code: int, period_count: int) -> tuple:
    face_value, _, _, _ = _fee_settings(config)
    return (
        float(config.initial_cash), float(config.initial_position),
        float(config.min_position), float(config.max_position),
        {"trailing": 0, "arithmetic": 1, "geometric": 2}[config.grid_mode],
        float(config.buy_step), float(config.sell_step), float(config.grid_step),
        float(config.reference_price or 0.0), float(config.lower_price or 0.0),
        float(config.upper_price or 0.0),
        {"shares": 0, "cash": 1, "equity_ratio": 2}[config.buy_quantity.mode],
        float(config.buy_quantity.value),
        {"shares": 0, "cash": 1, "equity_ratio": 2}[config.sell_quantity.mode],
        float(config.sell_quantity.value), float(config.initial_cash / max(1, period_count)),
        board_code, float(config.fees.commission_rate), float(config.fees.min_commission),
        float(config.fees.slippage_bps) / 10_000.0,
        face_value,
    )


def _fast_market_inputs(config: BacktestConfig, data: DataBundle):
    """Build typed arrays when the compiled ordinary-bar path is safe.

    Confirmed suspended days and company actions deliberately fall through to
    the readable general event loop.  Observed non-tradable bars stay in the
    event stream so their pending-order cancellations retain their timing.
    A reported daily fallback is represented by its existing one-row bar in the
    same kernel.
    """
    if not _NUMBA_FASTPATH_ENABLED or njit is None:
        return None
    if config.frequency == "1d" or data.bars is None or data.bars.empty:
        return None
    if not {"timestamp", "open", "close", "volume"}.issubset(data.bars.columns):
        return None
    cache_end = config.end_date
    if cache_end is None and data.calendar is not None and len(data.calendar):
        cache_end = max(data.calendar.date)
    elif cache_end is None and data.daily is not None and not data.daily.empty:
        cache_end = pd.to_datetime(data.daily["date"], errors="coerce").max().date()
    face_value, legacy_sh_rate, legacy_sh_minimum, legacy_sz_rate = _fee_settings(config)
    cache_key = (
        id(data), data.version, config.start_date, cache_end,
        config.frequency, config.dca_frequency, face_value,
        legacy_sh_rate, legacy_sh_minimum, legacy_sz_rate,
    )
    cached = _FAST_INPUT_CACHE.get(cache_key)
    if cached is not None and cached.get("_data") is data:
        # Most optimizer parameters do not affect market arrays.  Reuse the
        # prepared arrays across trials and only repack scalar trial settings.
        base = {key: value for key, value in cached.items() if not key.startswith("_")}
        base["options"] = _fast_options(config, cached["board_code"], cached["period_count"])
        del _FAST_INPUT_CACHE[cache_key]
        _FAST_INPUT_CACHE[cache_key] = cached
        return base

    raw_ts = pd.to_datetime(data.bars["timestamp"], errors="coerce")
    if getattr(raw_ts.dt, "tz", None) is not None:
        raw_ts = raw_ts.dt.tz_localize(None)
    if raw_ts.isna().any():
        return None
    bar_days = raw_ts.dt.normalize()
    if config.end_date is not None:
        end_day = pd.Timestamp(config.end_date)
    elif data.calendar is not None and len(data.calendar):
        end_day = pd.Timestamp(max(data.calendar.date))
    elif data.daily is not None and not data.daily.empty:
        end_day = pd.to_datetime(data.daily["date"], errors="coerce").max().normalize()
    else:
        end_day = bar_days.max()
    start_day = pd.Timestamp(config.start_date)
    actions = _prepare_actions(data, config.start_date, end_day.date())
    mask = (bar_days >= start_day) & (bar_days <= end_day)
    if not bool(mask.any()):
        return None
    frame = data.bars.loc[mask].copy()
    frame["_ts"] = raw_ts.loc[mask].to_numpy()
    # The canonical event loop removes synthetic bars before signal evaluation.
    # Dropping them here is equivalent and lets the next real bar receive orders.
    if "is_synthetic" in frame:
        frame = frame.loc[~frame["is_synthetic"].fillna(False).astype(bool)].copy()
    frame = frame.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    timestamps = pd.DatetimeIndex(frame["_ts"])
    if len(timestamps) == 0:
        return None
    frequencies = (
        frame["frequency"].dropna().astype(str).unique()
        if "frequency" in frame else np.array([config.frequency])
    )
    if len(frequencies) == 0 or any(value not in {config.frequency, "1d"} for value in frequencies):
        return None
    tradable_mask = (
        frame["tradable"].fillna(True).astype(bool).to_numpy()
        if "tradable" in frame else np.ones(len(frame), dtype=np.bool_)
    )

    opens = _numeric_column(frame, "open")
    closes = _numeric_column(frame, "close")
    volumes = _numeric_column(frame, "volume", 0.0)
    valid_prices = np.isfinite(opens) & np.isfinite(closes) & (opens > 0) & (closes > 0)
    if np.any(tradable_mask & ~valid_prices):
        return None
    if np.any(tradable_mask & ~np.isfinite(volumes)):
        return None
    # Invalid prices on a non-tradable event are immaterial: the reference
    # engine cancels its pending order and skips the bar before reading OHLC.
    opens = np.where(valid_prices, opens, 1.0)
    closes = np.where(valid_prices, closes, 1.0)
    volumes = np.where(tradable_mask, volumes, 0.0)

    day_index = pd.DatetimeIndex(timestamps.normalize())
    unique_days = day_index.unique().sort_values()
    expected: set[pd.Timestamp] = set(unique_days)
    if data.calendar is not None and len(data.calendar):
        calendar_days = pd.DatetimeIndex(pd.to_datetime(data.calendar)).normalize()
        expected.update(day for day in calendar_days if start_day <= day <= end_day)
    if data.daily is not None and not data.daily.empty:
        daily_days = pd.to_datetime(data.daily["date"], errors="coerce").dt.normalize()
        expected.update(
            day for day in pd.DatetimeIndex(daily_days.dropna().unique())
            if start_day <= day <= end_day
        )
        if "tradestatus" in data.daily:
            status = pd.to_numeric(data.daily["tradestatus"], errors="coerce")
            relevant = daily_days.between(start_day, end_day)
            if status.loc[relevant].isna().any() or (status.loc[relevant] == 0).any():
                return None
            if not status.loc[relevant].isin([1]).all():
                return None
    if expected != set(unique_days):
        return None

    for report in (data.quality or {}).get("days", []):
        try:
            report_day = pd.Timestamp(report["date"]).normalize()
        except (KeyError, TypeError, ValueError):
            return None
        if start_day <= report_day <= end_day and report.get("mode") not in {"minute", "daily"}:
            return None

    day_codes = pd.factorize(day_index, sort=True)[0].astype(np.int64)
    n = len(frame)
    first_day = np.zeros(n, dtype=np.bool_)
    last_bar_day = np.zeros(n, dtype=np.bool_)
    first_day[0] = True
    if n > 1:
        changed = day_codes[1:] != day_codes[:-1]
        first_day[1:] = changed
    tradable_indices = np.flatnonzero(tradable_mask)
    if not len(tradable_indices):
        return None
    tradable_day_codes = day_codes[tradable_indices]
    # A fully non-tradable day is handled by the reference loop's daily-close
    # mark path.  All ordinary minute/daily fast-path days must have at least
    # one observed trading event.
    if len(np.unique(tradable_day_codes)) != len(unique_days):
        return None
    last_in_day = np.ones(len(tradable_indices), dtype=np.bool_)
    if len(tradable_indices) > 1:
        last_in_day[:-1] = tradable_day_codes[:-1] != tradable_day_codes[1:]
    last_bar_day[tradable_indices[last_in_day]] = True

    day_dates = list(unique_days.to_pydatetime())
    period_keys = [_dca_period_key(value.date(), config.dca_frequency) for value in day_dates]
    seen_periods: set[tuple[int, int]] = set()
    first_period_days: set[int] = set()
    for index, key in enumerate(period_keys):
        if key not in seen_periods:
            seen_periods.add(key)
            first_period_days.add(index)
    first_dca_day = first_day & np.isin(day_codes, np.asarray(sorted(first_period_days), dtype=np.int64))
    dca_allowance = config.initial_cash / max(1, len(seen_periods))

    board = _board_name(config, data)
    board_code = 2 if board == "star" else (1 if board == "chinext" else 0)
    daily_lookup: dict[pd.Timestamp, dict[str, Any]] = {}
    if data.daily is not None and not data.daily.empty:
        daily_dates = pd.to_datetime(data.daily["date"], errors="coerce").dt.normalize()
        daily_mask = daily_dates.between(start_day, end_day)
        relevant_daily = data.daily.loc[daily_mask]
        relevant_dates = daily_dates.loc[daily_mask]
        daily_lookup = {
            day: row for day, row in zip(relevant_dates, relevant_daily.to_dict("records"))
        }

    limit_ups = _numeric_column(frame, "limit_up")
    limit_downs = _numeric_column(frame, "limit_down")
    bar_preclose = _numeric_column(frame, "preclose")
    if "is_st" in frame:
        bar_is_st = frame["is_st"].fillna(False).astype(bool).to_numpy()
        bar_is_st_missing = frame["is_st"].isna().to_numpy()
    else:
        bar_is_st = np.zeros(n, dtype=np.bool_)
        bar_is_st_missing = np.ones(n, dtype=np.bool_)
    if "no_price_limit" in frame:
        bar_no_price_limit = frame["no_price_limit"].fillna(False).astype(bool).to_numpy()
    else:
        bar_no_price_limit = np.zeros(n, dtype=np.bool_)
    missing_limits = False
    stamp_by_day = np.empty(len(unique_days), dtype=np.float64)
    transfer_by_day = np.empty(len(unique_days), dtype=np.float64)
    transfer_face_by_day = np.zeros(len(unique_days), dtype=np.bool_)
    transfer_minimum_by_day = np.zeros(len(unique_days), dtype=np.float64)
    stamp_bilateral_by_day = np.empty(len(unique_days), dtype=np.bool_)
    preclose_by_day = np.full(len(unique_days), np.nan, dtype=np.float64)
    for day_id, stamp in enumerate(unique_days):
        day_date = stamp.date()
        daily = daily_lookup.get(stamp, {})
        stamp_by_day[day_id] = fee_rates(day_date, config.symbol)[0]
        (
            transfer_by_day[day_id], transfer_face_by_day[day_id],
            transfer_minimum_by_day[day_id],
        ) = transfer_fee_terms(
            day_date, config.symbol,
            legacy_sh_transfer_rate=legacy_sh_rate,
            legacy_sh_min_transfer=legacy_sh_minimum,
            legacy_sz_transfer_rate=legacy_sz_rate,
        )
        stamp_bilateral_by_day[day_id] = day_date < date(2008, 9, 19)
        daily_up = _value(daily, "limit_up")
        daily_down = _value(daily, "limit_down")
        daily_preclose = _value(daily, "preclose")
        daily_no_price_limit = bool(_value(daily, "no_price_limit", False))
        if daily_preclose is not None and math.isfinite(float(daily_preclose)):
            preclose_by_day[day_id] = float(daily_preclose)
        daily_is_st = bool(_value(daily, "is_st", False))
        first = int(np.searchsorted(day_codes, day_id, side="left"))
        stop = int(np.searchsorted(day_codes, day_id, side="right"))
        for position in range(first, stop):
            if daily_no_price_limit or bar_no_price_limit[position]:
                # An exchange no-limit session (for example an IPO's first
                # sessions) must not inherit stale values or a regular board
                # limit inferred from preclose.
                limit_ups[position] = np.nan
                limit_downs[position] = np.nan
                continue
            if not np.isfinite(limit_ups[position]) and daily_up is not None:
                limit_ups[position] = float(daily_up)
            if not np.isfinite(limit_downs[position]) and daily_down is not None:
                limit_downs[position] = float(daily_down)
            preclose = bar_preclose[position]
            if not np.isfinite(preclose) and daily_preclose is not None:
                preclose = float(daily_preclose)
            is_st = daily_is_st if bar_is_st_missing[position] else bool(bar_is_st[position])
            if not np.isfinite(limit_ups[position]) or not np.isfinite(limit_downs[position]):
                calculated_up, calculated_down = calculated_price_limits(
                    float(preclose) if np.isfinite(preclose) else None,
                    board, day_date, is_st,
                )
                if not np.isfinite(limit_ups[position]) and calculated_up is not None:
                    limit_ups[position] = calculated_up
                if not np.isfinite(limit_downs[position]) and calculated_down is not None:
                    limit_downs[position] = calculated_down
            if (
                (not np.isfinite(limit_ups[position]) or not np.isfinite(limit_downs[position]))
                and not np.isfinite(preclose)
            ):
                missing_limits = True

    stamp_rates = stamp_by_day[day_codes]
    transfer_rates = transfer_by_day[day_codes]
    transfer_is_face_value = transfer_face_by_day[day_codes]
    transfer_minimums = transfer_minimum_by_day[day_codes]
    stamp_bilateral = stamp_bilateral_by_day[day_codes]
    timestamps_ns = timestamps.asi8.astype(np.int64)
    ns_day = 86_400_000_000_000
    day_ns = timestamps_ns // ns_day * ns_day
    time_ns = timestamps_ns - day_ns
    frequency_minutes_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60}
    if config.frequency not in frequency_minutes_map:
        return None
    if "frequency" in frame:
        bar_frequencies = frame["frequency"].fillna(config.frequency).astype(str).to_numpy()
    else:
        bar_frequencies = np.full(n, config.frequency, dtype=object)
    daily_bar_mask = bar_frequencies == "1d"
    bar_minutes = np.fromiter(
        (0 if value == "1d" else frequency_minutes_map.get(value, -1) for value in bar_frequencies),
        dtype=np.int64, count=n,
    )
    if (bar_minutes < 0).any():
        return None
    candidate = timestamps_ns - bar_minutes * 60_000_000_000
    morning_open = day_ns + (9 * 60 + 30) * 60_000_000_000
    afternoon_open = day_ns + 13 * 60 * 60_000_000_000
    open_timestamps_ns = candidate.copy()
    open_timestamps_ns[daily_bar_mask] = morning_open[daily_bar_mask]
    morning_mask = (~daily_bar_mask) & (time_ns <= (11 * 60 + 30) * 60_000_000_000) & (candidate < morning_open)
    open_timestamps_ns[morning_mask] = morning_open[morning_mask]
    afternoon_mask = (
        (~daily_bar_mask)
        & (time_ns > (11 * 60 + 30) * 60_000_000_000)
        & (time_ns <= 15 * 60 * 60 * 1_000_000_000)
        & (candidate < afternoon_open)
    )
    open_timestamps_ns[afternoon_mask] = afternoon_open[afternoon_mask]

    arrays = (
        opens, closes, volumes, tradable_mask, limit_ups, limit_downs, timestamps_ns,
        open_timestamps_ns, day_codes, first_dca_day, last_bar_day,
        stamp_rates, transfer_rates, transfer_is_face_value,
        transfer_minimums, stamp_bilateral, preclose_by_day,
    )
    day_id_by_date = {stamp.date(): index for index, stamp in enumerate(unique_days)}
    action_record_day = np.asarray([
        day_id_by_date.get(action.record_date, -1) for action in actions
    ], dtype=np.int64)
    action_ex_day = np.asarray([
        day_id_by_date.get(action.ex_date, -1) for action in actions
    ], dtype=np.int64)
    action_pay_day = np.asarray([
        day_id_by_date.get(action.pay_date, -1) if action.pay_date is not None else -1
        for action in actions
    ], dtype=np.int64)
    action_stock_day = np.asarray([
        day_id_by_date.get(action.stock_date, -1) if action.stock_date is not None else -1
        for action in actions
    ], dtype=np.int64)
    action_cash_per_share = np.asarray([action.cash_per_share for action in actions], dtype=np.float64)
    action_bonus_ratio = np.asarray([action.bonus_ratio for action in actions], dtype=np.float64)
    arrays = arrays + (
        action_record_day, action_ex_day, action_pay_day, action_stock_day,
        action_cash_per_share, action_bonus_ratio,
    )
    day_close_ns = unique_days.asi8.astype(np.int64) + 15 * 60 * 60 * 1_000_000_000
    cache_value = {
        "arrays": arrays, "day_dates": unique_days,
        "day_close_ns": day_close_ns, "timestamp_ns": timestamps_ns,
        "tradable_mask": tradable_mask,
        "open_timestamp_ns": open_timestamps_ns, "missing_limits": missing_limits,
        "end": end_day.date(), "n": n, "board_code": board_code,
        "period_count": len(seen_periods), "_data": data,
    }
    _FAST_INPUT_CACHE[cache_key] = cache_value
    while len(_FAST_INPUT_CACHE) > 2:
        _FAST_INPUT_CACHE.pop(next(iter(_FAST_INPUT_CACHE)))
    return {
        **{key: value for key, value in cache_value.items() if not key.startswith("_") and key not in {"board_code", "period_count"}},
        "options": _fast_options(config, board_code, len(seen_periods)),
    }


def _run_fastpath(
    config: BacktestConfig,
    data: DataBundle,
    inputs: dict[str, Any],
    progress: Callable[[float, str], Any] | None,
    cancelled: Callable[[], bool] | None,
    summary_only: bool,
) -> dict[str, Any]:
    if cancelled and cancelled():
        raise InterruptedError("回测已取消")
    if progress:
        progress(0.0, "准备回测数据")
    result = _nb_simulate(*inputs["arrays"], *inputs["options"], not summary_only)
    if cancelled and cancelled():
        raise InterruptedError("回测已取消")
    if progress:
        progress(1.0, "回测完成")

    (
        metrics_array, equity_values, equity_cash, equity_shares, equity_drawdown,
        period_values, trade_bar, trade_strategy, trade_side, trade_quantity,
        trade_price, trade_fees, trade_cash, trade_shares, trade_reason,
        rejection_bar, rejection_ts, rejection_strategy, rejection_side,
        rejection_reason, rejection_quantity,
    ) = result
    timestamp_strings = (
        pd.to_datetime(inputs["timestamp_ns"]).strftime("%Y-%m-%dT%H:%M:%S").tolist()
        if not summary_only else []
    )
    open_timestamp_strings = (
        pd.to_datetime(inputs["open_timestamp_ns"]).strftime("%Y-%m-%dT%H:%M:%S").tolist()
        if not summary_only else []
    )
    period_strings = (
        pd.to_datetime(inputs["day_close_ns"]).strftime("%Y-%m-%dT%H:%M:%S").tolist()
        if not summary_only else []
    )
    rejection_strings = (
        pd.to_datetime(rejection_ts).strftime("%Y-%m-%dT%H:%M:%S").tolist()
        if not summary_only and len(rejection_ts) else []
    )

    strategies: dict[str, Any] = {}
    for account, name in enumerate(_STRATEGY_NAMES):
        metric = {
            key: (
                int(round(metrics_array[account, index]))
                if key in {"trade_count", "buy_count", "sell_count"}
                else float(metrics_array[account, index])
            )
            for index, key in enumerate(_METRIC_KEYS)
        }
        if summary_only:
            equity: list[dict[str, Any]] = []
            trades: list[dict[str, Any]] = []
            rejections: list[dict[str, Any]] = []
            period_returns: list[dict[str, Any]] = []
        else:
            closes = inputs["arrays"][1]
            equity = [
                {
                    "timestamp": timestamp_strings[i],
                    "equity": float(equity_values[account, i]),
                    "cash": float(equity_cash[account, i]),
                    "position": float(equity_shares[account, i]),
                    "position_ratio": float(_position_ratio(
                        equity_shares[account, i], closes[i], equity_values[account, i],
                    )),
                    "drawdown": float(equity_drawdown[account, i]),
                }
                for i in range(inputs["n"]) if inputs["tradable_mask"][i]
            ]
            trades = [
                {
                    "timestamp": open_timestamp_strings[int(trade_bar[j])],
                    "side": "buy" if trade_side[j] == 1 else "sell",
                    "quantity": int(trade_quantity[j]),
                    "price": float(trade_price[j]),
                    "fees": float(trade_fees[j]),
                    "cash": float(trade_cash[j]),
                    "shares": int(trade_shares[j]),
                    "reason": _TRADE_REASONS[int(trade_reason[j])],
                }
                for j in range(len(trade_bar)) if trade_strategy[j] == account
            ]
            rejections = [
                {
                    "timestamp": rejection_strings[j],
                    "side": "buy" if rejection_side[j] == 1 else "sell",
                    "reason": _REJECTION_REASONS[int(rejection_reason[j])],
                    "requested_quantity": int(rejection_quantity[j]),
                }
                for j in range(len(rejection_bar)) if rejection_strategy[j] == account
            ]
            period_returns = [
                {"timestamp": period_strings[j], "return": float(period_values[account, j])}
                for j in range(len(period_strings))
            ]
        strategies[name] = {
            "metrics": metric, "equity": equity, "trades": trades,
            "rejections": rejections, "period_returns": period_returns,
        }

    warnings = list(dict.fromkeys([
        *(data.warnings or []), *((data.quality or {}).get("warnings", []) or []),
    ]))
    warnings.extend(_legacy_fee_warnings(config, inputs["end"]))
    if inputs["missing_limits"]:
        warnings.append("部分K线缺少涨跌停价和前收盘价，相关开盘成交无法进行涨跌停封板校验。")
    return {
        "symbol": config.symbol, "config": config.model_dump(mode="json"),
        "data_version": data.version, "quality": data.quality or {},
        "warnings": list(dict.fromkeys(warnings)), "strategies": strategies,
    }


def run_backtest(
    config: BacktestConfig,
    data: DataBundle,
    progress: Callable[[float, str], Any] | None = None,
    cancelled: Callable[[], bool] | None = None,
    summary_only: bool = False,
) -> dict[str, Any]:
    """Run grid, DCA and buy-and-hold ledgers against one canonical data bundle."""
    if config.symbol != data.symbol:
        raise ValueError(f"回测标的 {config.symbol} 与数据标的 {data.symbol} 不一致")
    fast_inputs = _fast_market_inputs(config, data)
    if fast_inputs is not None:
        return _run_fastpath(config, data, fast_inputs, progress, cancelled, summary_only)
    end, days, bars_by_day, daily_by_day, quality_days = _prepare_market(config, data)
    actions = _prepare_actions(data, config.start_date, end)
    action_map = {action.index: action for action in actions}
    events_by_day = _actions_by_date(actions)
    if progress:
        progress(0.0, "准备回测数据")

    accounts = [
        _Ledger("grid", config.initial_cash, keep_details=not summary_only, peak_equity=config.initial_cash, last_equity=config.initial_cash),
        _Ledger("dca", config.initial_cash, keep_details=not summary_only, peak_equity=config.initial_cash, last_equity=config.initial_cash),
        _Ledger("buy_hold", config.initial_cash, keep_details=not summary_only, peak_equity=config.initial_cash, last_equity=config.initial_cash),
    ]
    board = _board_name(config, data)
    # Only known trading calendar periods are used to divide the fixed DCA principal.
    period_first_days: dict[tuple[int, int], date] = {}
    dca_calendar_days = [day for day in days if day in bars_by_day or day in daily_by_day or day in quality_days]
    for day in dca_calendar_days:
        key = _dca_period_key(day, config.dca_frequency)
        period_first_days.setdefault(key, day)
    dca_allowance = config.initial_cash / max(1, len(period_first_days))

    prior_close = None
    last_close = None
    prior_day_equity: dict[str, float] = {account.name: config.initial_cash for account in accounts}
    total_bars = max(1, sum(len(rows) for rows in bars_by_day.values()))
    processed_bars = 0
    scheduled_periods: set[tuple[int, int]] = set()
    warned_missing_limits = False
    warnings = list(dict.fromkeys([*(data.warnings or []), *((data.quality or {}).get("warnings", []) or [])]))
    warnings.extend(_legacy_fee_warnings(config, end))

    for day_index, day in enumerate(days):
        if cancelled and cancelled():
            raise InterruptedError("回测已取消")
        daily = daily_by_day.get(day)
        day_events = events_by_day.get(day, {})

        # Ex-date adjustments/claims are activated before the first opening event.
        _apply_ex_adjustment(day, daily, prior_close, accounts, day_events.get("ex", []), action_map, board, config)
        _settle_actions(day, accounts, day_events)

        period_key = _dca_period_key(day, config.dca_frequency)
        if period_key in period_first_days and period_first_days[period_key] == day and period_key not in scheduled_periods:
            accounts[1].dca_budget += dca_allowance
            accounts[1].dca_due = True
            scheduled_periods.add(period_key)

        suspended = (_daily_status(daily) is False or quality_days.get(day, {}).get("mode") == "suspended")
        if suspended:
            for account in accounts:
                if account.pending is not None:
                    _record_rejection(account, datetime.combine(day, time(9, 30)), account.pending.side, "suspended_day", account.pending.quantity)
                    account.pending = None

        day_bars = bars_by_day.get(day, [])
        # A daily status of suspension wins over stale or erroneous intraday rows.
        if suspended:
            day_bars = []
        processed_day_bar = False
        for bar_index, row in enumerate(day_bars):
            processed_bars += 1
            tradable = bool(_value(row, "tradable", True))
            if not tradable:
                for account in accounts:
                    if account.pending is not None:
                        _record_rejection(account, row["_open_timestamp"], account.pending.side, "non_tradable_bar", account.pending.quantity)
                        account.pending = None
                continue
            open_price = float(_value(row, "open", 0) or 0)
            close = float(_value(row, "close", 0) or 0)
            if close <= 0 or not math.isfinite(close):
                continue
            processed_day_bar = True
            # If supplied limits are absent, a derived limit still applies where a
            # valid exchange preclose exists.  Keep one warning rather than flood UI.
            no_price_limit = bool(_value(row, "no_price_limit", False)) or bool(
                _value(daily or {}, "no_price_limit", False)
            )
            if (
                not no_price_limit
                and (_value(row, "limit_up") is None or _value(row, "limit_down") is None)
                and _value(row, "preclose") is None
                and not warned_missing_limits
            ):
                warnings.append("部分K线缺少涨跌停价和前收盘价，相关开盘成交无法进行涨跌停封板校验。")
                warned_missing_limits = True

            grid, dca, hold = accounts
            for account in accounts:
                traded = False
                if account.pending is not None:
                    order = account.pending
                    account.pending = None  # every opening order expires at this opening
                    filled, _ = _attempt_trade(account, order.side, order.quantity, row, day, board, config, daily, order.reason)
                    if filled:
                        traded = True
                        if account.name == "grid":
                            _update_reference_after_fill(account, account.last_trade_price, config)
                if not traded and account.name == "grid" and not account.started:
                    # Seed the grid to the configured initial exposure, then signal
                    # only after a successful real opening fill.
                    if config.initial_position == 0:
                        account.started = True
                        account.reference = float(config.reference_price or open_price)
                        account.fixed_anchor = float(config.reference_price or open_price)
                        account.active_cell = _cell(open_price, account.fixed_anchor, config)
                    else:
                        raw_quantity = int(math.floor(config.initial_cash * config.initial_position / open_price)) if open_price > 0 else 0
                        quantity = floor_buy_quantity(raw_quantity, board)
                        filled, _ = _attempt_trade(account, "buy", quantity, row, day, board, config, daily, "initial_position")
                        if filled:
                            traded = True
                            account.started = True
                            fill_price = account.last_trade_price
                            account.reference = float(fill_price)
                            account.fixed_anchor = float(config.reference_price or fill_price)
                            account.active_cell = _cell(fill_price, account.fixed_anchor, config)
                if not traded and account.name == "dca" and account.dca_due and account.dca_last_attempt != day:
                    account.dca_last_attempt = day
                    if account.dca_budget > 0 and open_price > 0:
                        wanted = int(math.floor(account.dca_budget / open_price))
                        quantity = floor_buy_quantity(wanted, board)
                        filled, _ = _attempt_trade(account, "buy", quantity, row, day, board, config, daily, "dca")
                        if filled:
                            spent = account.last_trade_notional + account.last_trade_fee
                            account.dca_budget = max(0.0, account.dca_budget - spent)
                            account.dca_due = False
                            traded = True
                if not traded and account.name == "buy_hold" and not account.started:
                    quantity = 2**31 - 1
                    filled, _ = _attempt_trade(account, "buy", quantity, row, day, board, config, daily, "buy_hold")
                    if filled:
                        account.started = True
                        traded = True

            # The signal uses this real bar's close.  Its order can only reach the
            # next real bar opening because the pending order is visited above.
            if grid.started and grid.pending is None and float(_value(row, "volume", 0) or 0) > 0:
                equity_at_close = _equity_at(grid, close)
                intent = _grid_intent(grid, close, equity_at_close, config)
                if intent is not None:
                    intent.signal_timestamp = row["_timestamp"]
                    _accept_grid_intent(grid, intent, close, row["_timestamp"], board, config)

            timestamp = row["_timestamp"]
            for account in accounts:
                equity = _equity_at(account, close)
                if account.first_timestamp is None:
                    account.first_timestamp = timestamp
                account.last_timestamp = timestamp
                account.last_equity = equity
                if not summary_only:
                    account.equity_values.append(equity)
                    account.equity_timestamps.append(timestamp)
                account.peak_equity = max(account.peak_equity, equity)
                drawdown = (account.peak_equity - equity) / account.peak_equity if account.peak_equity > 0 else 0.0
                account.max_drawdown = max(account.max_drawdown, drawdown)
                if not summary_only:
                    account.equity.append({
                        "timestamp": _iso(timestamp), "equity": float(equity),
                        "cash": float(account.cash), "position": float(_position_shares(account)),
                        "position_ratio": float(_position_ratio(
                            _position_shares(account), close, equity,
                        )),
                        "drawdown": float(drawdown),
                    })
            last_close = close
            prior_close = close

            if progress and (processed_bars % 512 == 0 or processed_bars == total_bars):
                progress(min(0.99, processed_bars / total_bars), f"正在回测：{processed_bars}/{total_bars} 根K线")
            if cancelled and processed_bars % 512 == 0 and cancelled():
                raise InterruptedError("回测已取消")

        if not processed_day_bar:
            # Mark a suspension day at its available daily close, or at the last
            # known close when no reliable bar exists.  This never creates signals.
            mark = _value(daily or {}, "close")
            mark = float(mark) if mark is not None and float(mark) > 0 else last_close
            if mark is not None and math.isfinite(float(mark)):
                timestamp = datetime.combine(day, time(15, 0))
                for account in accounts:
                    equity = _equity_at(account, float(mark))
                    account.first_timestamp = account.first_timestamp or timestamp
                    account.last_timestamp = timestamp
                    account.last_equity = equity
                    if not summary_only:
                        account.equity_values.append(equity)
                        account.equity_timestamps.append(timestamp)
                    account.peak_equity = max(account.peak_equity, equity)
                    drawdown = (account.peak_equity - equity) / account.peak_equity if account.peak_equity > 0 else 0.0
                    account.max_drawdown = max(account.max_drawdown, drawdown)
                    if not summary_only:
                        account.equity.append({
                            "timestamp": _iso(timestamp), "equity": float(equity),
                            "cash": float(account.cash), "position": float(_position_shares(account)),
                            "position_ratio": float(_position_ratio(
                                _position_shares(account), float(mark), equity,
                            )),
                            "drawdown": float(drawdown),
                        })
                last_close = float(mark)
        # End-of-day returns include any receivable recognized from the previous
        # record date and are the input to annualized Sharpe.
        day_mark = last_close
        if day_mark is not None:
            for account in accounts:
                day_equity = _equity_at(account, float(day_mark))
                previous = prior_day_equity[account.name]
                daily_return = day_equity / previous - 1 if previous > 0 else 0.0
                account.daily_returns.append({"timestamp": datetime.combine(day, time(15, 0)).isoformat(), "return": float(daily_return)})
                prior_day_equity[account.name] = day_equity
                account.last_day_equity = day_equity
        for index in day_events.get("record", []):
            action = action_map[index]
            for account in accounts:
                quantity = float(account.shares)
                account.entitlements[index] = (
                    quantity * action.cash_per_share,
                    quantity * action.bonus_ratio,
                )
        if progress and (day_index + 1) % max(1, len(days) // 100 or 1) == 0:
            progress(min(0.99, (day_index + 1) / max(1, len(days))), f"处理交易日 {day.isoformat()}")

    if progress:
        progress(1.0, "回测完成")
    result_strategies: dict[str, Any] = {}
    names = {"grid": "grid", "dca": "dca", "buy_hold": "buy_hold"}
    for account in accounts:
        metrics = _metric_summary(account, config.initial_cash)
        result_strategies[names[account.name]] = {
            "metrics": metrics,
            "equity": account.equity if not summary_only else [],
            "trades": account.trades if not summary_only else [],
            "rejections": account.rejections if not summary_only else [],
            "period_returns": account.daily_returns if not summary_only else [],
        }
    return {
        "symbol": config.symbol,
        "config": config.model_dump(mode="json"),
        "data_version": data.version,
        "quality": data.quality or {},
        "warnings": list(dict.fromkeys(warnings)),
        "strategies": result_strategies,
    }

