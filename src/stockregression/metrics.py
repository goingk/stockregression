"""Portfolio performance summaries for backtest ledgers."""
from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any, Iterable


def _year_fraction(first: Any, last: Any) -> float:
    def as_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    try:
        delta = as_dt(last) - as_dt(first)
        return max(0.0, delta.total_seconds() / (365.25 * 24 * 3600))
    except (TypeError, ValueError):
        return 0.0


def _annualized_return(initial_cash: float, final_equity: float, elapsed: float) -> float:
    if initial_cash <= 0 or final_equity <= 0 or elapsed <= 0:
        return 0.0
    # Treat any sub-session sample as one trading day; minute-scale annualization
    # otherwise overflows and exaggerates a result that has almost no history.
    elapsed = max(elapsed, 1 / 252)
    log_growth = math.log(final_equity / initial_cash) / elapsed
    return math.expm1(min(700.0, max(-700.0, log_growth)))


def summarize_metrics(
    *,
    initial_cash: float,
    final_equity: float,
    first_timestamp: Any,
    last_timestamp: Any,
    max_drawdown: float,
    daily_returns: list[dict[str, Any]],
    buy_count: int,
    sell_count: int,
    total_fees: float,
    gross_turnover: float,
) -> dict[str, float | int]:
    total_return = final_equity / initial_cash - 1 if initial_cash else 0.0
    elapsed = _year_fraction(first_timestamp, last_timestamp) if first_timestamp is not None and last_timestamp is not None else 0.0
    cagr = _annualized_return(initial_cash, final_equity, elapsed)
    period_values = [float(item.get("return", 0.0)) for item in daily_returns]
    if len(period_values) > 1:
        mean = sum(period_values) / len(period_values)
        variance = sum((value - mean) ** 2 for value in period_values) / (len(period_values) - 1)
        deviation = math.sqrt(max(0.0, variance))
        sharpe = mean / deviation * math.sqrt(252) if deviation > 0 else 0.0
    else:
        sharpe = 0.0
    calmar = cagr / max_drawdown if max_drawdown > 0 else 0.0
    return {
        "total_return": float(total_return), "cagr": float(cagr),
        "max_drawdown": float(max_drawdown), "sharpe": float(sharpe),
        "calmar": float(calmar), "trade_count": int(buy_count + sell_count),
        "total_fees": float(total_fees),
        "turnover": float(gross_turnover / initial_cash if initial_cash else 0.0),
        "buy_count": int(buy_count), "sell_count": int(sell_count),
    }


def compute_metrics(
    initial_cash: float,
    equity_values: Iterable[float],
    equity_timestamps: Iterable[Any],
    trades: list[dict[str, Any]],
    daily_returns: list[dict[str, Any]],
    *,
    gross_turnover: float | None = None,
) -> dict[str, float | int]:
    values = [float(value) for value in equity_values if math.isfinite(float(value))]
    timestamps = list(equity_timestamps)
    final_equity = values[-1] if values else float(initial_cash)
    total_return = final_equity / initial_cash - 1 if initial_cash else 0.0
    first_timestamp = timestamps[0] if timestamps else None
    last_timestamp = timestamps[-1] if timestamps else None

    peak = float(initial_cash)
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)
    buy_count = sum(1 for trade in trades if trade.get("side") == "buy")
    sell_count = sum(1 for trade in trades if trade.get("side") == "sell")
    total_fees = sum(float(trade.get("fees", 0.0)) for trade in trades)
    turnover = (gross_turnover if gross_turnover is not None else sum(
        float(trade.get("quantity", 0.0)) * float(trade.get("price", 0.0)) for trade in trades
    )) / initial_cash if initial_cash else 0.0
    return summarize_metrics(
        initial_cash=initial_cash, final_equity=final_equity,
        first_timestamp=first_timestamp, last_timestamp=last_timestamp,
        max_drawdown=max_drawdown, daily_returns=daily_returns,
        buy_count=buy_count, sell_count=sell_count, total_fees=total_fees,
        gross_turnover=turnover * initial_cash,
    )

