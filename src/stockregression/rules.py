"""Date-aware A-share trading conventions used by the backtest engine.

The functions in this module intentionally accept already-normalized values.  They
do not mutate data bundles and keep exchange-rule decisions easy to unit test.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Any
import warnings


def instrument_terms(symbol: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Canonical exchange terms. Provider-supplied product terms take precedence."""
    supplied = {key: value for key, value in (metadata or {}).items() if value is not None}
    code = symbol.split(".")[0]
    etf = supplied.get("asset_type", "etf" if code.startswith(("1", "5")) else "stock") == "etf"
    name = str(supplied.get("name", ""))
    category = "stock"
    for token, kind in [("债", "bond"), ("货币", "money"), ("日利", "money"), ("黄金", "commodity"), ("期货", "commodity"), ("纳指", "crossborder"), ("纳斯达克", "crossborder"), ("标普", "crossborder"), ("恒生", "crossborder"), ("港股", "crossborder"), ("日经", "crossborder"), ("德国", "crossborder"), ("法国", "crossborder")]:
        if token in name:
            category = kind
            break
    if any(token in name.upper() for token in ("H股", "香港", "美国", "日本", "沙特", "越南", "韩国", "东南亚", "印度", "海外", "全球", "英国", "中概")):
        category = "crossborder"
    if code.startswith(("513", "520", "521")):
        category = "crossborder"
    if code.startswith("511"):
        category = "money" if code.startswith(("5116", "5117", "5118", "5119")) else "bond"
    if code.startswith("518") or code in {"159980", "159981", "159985"}:
        category = "commodity"
    category = supplied.get("etf_category", category)
    profile = "star" if code.startswith(("588", "589")) or "科创" in name else "chinext" if code == "159915" or "创业" in name or "双创" in name else "standard"
    board = "etf" if etf else ("star" if code.startswith(("688", "689")) else "chinext" if code.startswith("3") else "main")
    return {
        "asset_type": "etf" if etf else "stock", "board": board,
        "etf_category": category if etf else None, "price_tick": 0.001 if etf else 0.01,
        "lot_size": 100, "settlement": "T0" if etf and category != "stock" else "T1",
        "etf_limit_profile": profile if etf else None,
        "price_limit_rate": 0.20 if etf and (code.startswith(("588", "589")) or any(x in name for x in ("科创", "创业", "双创"))) else 0.10,
        **supplied,
    }


def etf_price_limit_rate(metadata: dict[str, Any], on_date: Any) -> float:
    profile = metadata.get("etf_limit_profile")
    if profile == "chinext":
        return 0.20 if as_date(on_date) >= date(2020, 8, 24) else 0.10
    if profile == "star":
        return 0.20
    return float(metadata.get("price_limit_rate", 0.10))


def as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date"):
        result = value.date()
        if isinstance(result, date):
            return result
    return date.fromisoformat(str(value)[:10])


def infer_board(symbol: str, metadata: dict[str, Any] | None = None) -> str:
    board = (metadata or {}).get("board")
    if board in {"main", "chinext", "star"}:
        return board
    code = str(symbol).split(".")[0]
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("300", "301")):
        return "chinext"
    return "main"


def lot_size(board: str, side: str = "buy") -> int:
    """Return the minimum normal order unit; STAR purchases start at 200."""
    return 200 if board == "star" and side == "buy" else 100


def round_to_tick(price: float, tick: float = 0.01, *, side: str | None = None) -> float:
    """Round to a tick, using adverse rounding for simulated executions."""
    if not math.isfinite(price) or price <= 0:
        return 0.0
    scaled = price / tick
    if side == "buy":
        ticks = math.ceil(scaled - 1e-10)
    elif side == "sell":
        ticks = math.floor(scaled + 1e-10)
    else:
        ticks = int(Decimal(str(scaled)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return round(ticks * tick, 6)


def round_limit_price(price: float, tick: float = 0.01) -> float:
    """Exchange price limits are rounded to the nearest tick (half up)."""
    return round_to_tick(price, tick)


def price_limit_rate(board: str, on_date: Any, is_st: bool = False) -> float | None:
    """Return the ordinary daily price band for supported board history.

    IPO no-limit sessions require listing-age data and are handled separately by
    caller-supplied limit fields.  Rates reflect the long-standing bands and the
    2020 ChiNext and July 2026 main-board risk-warning changes.
    """
    day = as_date(on_date)
    if board == "star":
        return 0.20
    if board == "chinext":
        if day >= date(2020, 8, 24):
            return 0.20
        return 0.05 if is_st else 0.10
    if is_st and day < date(2026, 7, 6):
        return 0.05
    return 0.10


def calculated_price_limits(
    preclose: float | None, board: str, on_date: Any, is_st: bool = False,
    tick: float = 0.01,
) -> tuple[float | None, float | None]:
    if preclose is None or not math.isfinite(float(preclose)) or float(preclose) <= 0:
        return None, None
    rate = price_limit_rate(board, on_date, is_st)
    if rate is None:
        return None, None
    close = float(preclose)
    return round_limit_price(close * (1 + rate), tick), round_limit_price(close * (1 - rate), tick)


def _stamp_duty_rate(on_date: Any) -> float:
    day = as_date(on_date)
    if day <= date(2008, 4, 23):
        return 0.003
    if day <= date(2008, 9, 18):
        return 0.001
    return 0.0005 if day >= date(2023, 8, 28) else 0.001


def transfer_fee_terms(
    on_date: Any,
    symbol: str | None = None,
    *,
    legacy_sh_transfer_rate: float = 0.001,
    legacy_sh_min_transfer: float = 1.0,
    legacy_sz_transfer_rate: float = 0.0000255,
) -> tuple[float, bool, float]:
    """Return (rate, uses face value, minimum) for one order's transfer fee.

    Shanghai's pre-2015 schedule was based on share face value, while the
    Shenzhen schedule and later CSDC schedules use transaction value.  The
    caller supplies quantity and share face value when applying this rate.
    """
    if str(symbol or "").split(".")[0].startswith(("1", "5")):
        return 0.0, False, 0.0
    day = as_date(on_date)
    exchange = str(symbol or "").upper().rsplit(".", 1)[-1]
    if day >= date(2022, 4, 29):
        return 0.00001, False, 0.0
    if day >= date(2015, 8, 1):
        return 0.00002, False, 0.0
    if date(2012, 6, 1) <= day < date(2015, 8, 1):
        if exchange == "SH":
            rate = 0.000375 if day < date(2012, 9, 1) else 0.0003
            return rate, True, 0.0
        if exchange == "SZ":
            return 0.0000255, False, 0.0
        return 0.00002, False, 0.0
    if day < date(2012, 6, 1) and exchange == "SH":
        return float(legacy_sh_transfer_rate), True, float(legacy_sh_min_transfer)
    if day < date(2015, 8, 1) and exchange == "SZ":
        return float(legacy_sz_transfer_rate), False, 0.0
    # Backward-compatible fallback for old callers that lack an exchange.
    return 0.00002, False, 0.0


def fee_rates(on_date: Any, symbol: str | None = None) -> tuple[float, float]:
    """Return stamp-duty rate and transfer-fee rate for legacy callers.

    The second result is a rate only; use :func:`transfer_fee_terms` to learn
    whether a historical exchange rate applies to notional or face value.
    Calls without a symbol retain the historical 0.02‰ notional fallback.
    """
    stamp = 0.0 if str(symbol or "").split(".")[0].startswith(("1", "5")) else _stamp_duty_rate(on_date)
    return stamp, transfer_fee_terms(on_date, symbol)[0]


def round_money(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def commissions_and_taxes(
    notional: float,
    side: str,
    on_date: Any,
    commission_rate: float,
    min_commission: float,
    *,
    symbol: str | None = None,
    quantity: int | None = None,
    face_value: float = 1.0,
    legacy_sh_transfer_rate: float = 0.001,
    legacy_sh_min_transfer: float = 1.0,
    legacy_sz_transfer_rate: float = 0.0000255,
) -> dict[str, float]:
    if notional <= 0:
        return {"commission": 0.0, "stamp_duty": 0.0, "transfer_fee": 0.0, "total": 0.0}
    day = as_date(on_date)
    stamp_rate = fee_rates(day, symbol)[0]
    # Notional and rates represent decimal exchange/account values.  Multiply
    # those decimal inputs before conversion to float so binary artifacts such
    # as 44_450 * 0.0003 becoming 13.334999999999999 do not change a cent.
    decimal_notional = Decimal(str(notional))
    commission = max(
        Decimal(str(min_commission)),
        decimal_notional * Decimal(str(commission_rate)),
    )
    stamp_is_bilateral = as_date(on_date) < date(2008, 9, 19)
    stamp = decimal_notional * Decimal(str(stamp_rate)) if side == "sell" or stamp_is_bilateral else Decimal(0)
    is_etf = str(symbol or "").split(".")[0].startswith(("1", "5"))
    if is_etf:
        transfer = Decimal(0)
    elif day < date(2015, 8, 1) and (quantity is None or symbol is None):
        warnings.warn(
            "2015-08-01前的过户费规则依交易所、成交数量和面值区分；本次旧调用缺少symbol或quantity，按0.02‰成交金额估算。",
            UserWarning,
            stacklevel=2,
        )
        transfer = decimal_notional * Decimal("0.00002")
    else:
        transfer_rate, uses_face_value, minimum_transfer = transfer_fee_terms(
            day,
            symbol,
            legacy_sh_transfer_rate=legacy_sh_transfer_rate,
            legacy_sh_min_transfer=legacy_sh_min_transfer,
            legacy_sz_transfer_rate=legacy_sz_transfer_rate,
        )
        if day < date(2015, 8, 1) and str(symbol or "").upper().rsplit(".", 1)[-1] not in {"SH", "SZ"}:
            warnings.warn(
                "2015-08-01前的过户费规则依交易所区分；无法识别交易所，本次按0.02‰成交金额估算。",
                UserWarning,
                stacklevel=2,
            )
            transfer = decimal_notional * Decimal("0.00002")
        else:
            decimal_basis = (
                Decimal(str(quantity)) * Decimal(str(face_value))
                if uses_face_value else decimal_notional
            )
            transfer = max(
                Decimal(str(minimum_transfer)),
                decimal_basis * Decimal(str(transfer_rate)),
            )
    commission = round_money(float(commission))
    stamp = round_money(float(stamp))
    transfer = round_money(float(transfer))
    return {
        "commission": commission,
        "stamp_duty": stamp,
        "transfer_fee": transfer,
        "total": round_money(commission + stamp + transfer),
    }


def floor_buy_quantity(quantity: float, board: str) -> int:
    if not math.isfinite(quantity) or quantity <= 0:
        return 0
    if board == "star":
        # STAR accepts any integer quantity once its 200-share initial minimum is met.
        result = int(math.floor(quantity + 1e-9))
        return result if result >= 200 else 0
    return int(math.floor(quantity / 100 + 1e-9)) * 100


def legal_sell_quantity(requested: float, available: int, board: str) -> int:
    """Clip to holdings and A-share sell units, allowing a final odd lot."""
    requested = max(0, int(math.floor(requested + 1e-9)))
    available = max(0, int(available))
    quantity = min(requested, available)
    if quantity == available and (
        available % 100 or (board == "star" and available < 200)
    ):
        return quantity
    if board == "star":
        return quantity if quantity >= 200 else 0
    return (quantity // 100) * 100

