"""K3 本金账本回归测试：峰值本金口径、注资配对与逐 bar 回撤。

这些断言锁定 v5 研究协议的本金账本契约：

* ``matched_funding_hold`` 与网格共享同一份初始资产和初始持股；
* 只有网格**实际成交**才产生补款，注资与成交发生在同一时间、同一金额；
* 补款计入本金，永远不产生利润；
* 配对账户只能用整手成交，且整手数量必须含费用上限，余款保留在账上；
* 本金为零时收益必须是 ``None`` 并给出原因，而不是 0 或 NaN；
* 本金回撤逐 bar 统计，日线抽样不得漏掉盘中回撤；
* 现金分红与送转按账本计入应收与可交易日；
* 解释执行与 Numba 执行的结果必须完全一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockregression import engine
from stockregression.share_kernel import _nb_attempt
from stockregression.models import BacktestConfig, DataBundle, FeeConfig
from stockregression.rules import instrument_terms


SYMBOL = "600000.SH"
FEES = FeeConfig(commission_rate=0.0003, min_commission=5.0, slippage_bps=0)


def build_bundle(
    paths: list[list[float]],
    *,
    symbol: str = SYMBOL,
    fees: FeeConfig | None = None,
    actions: pd.DataFrame | None = None,
    initial_shares: int = 1000,
    buy_shares: int = 100,
    buy_step: float = 0.02,
    reference: float = 10.0,
    limit_ups: list[float] | None = None,
):
    """Build a deterministic share ledger fixture.

    ``paths`` holds one list of bar closes per session; a single value per
    session produces one daily bar, several values produce minute bars.  The
    first calendar day is a pre-start reference day so the ledger can price
    the initial holding.
    """
    closes = [values[-1] for values in paths]
    dates = pd.bdate_range("2024-01-02", periods=len(paths) + 1)
    frames = []
    for index, values in enumerate(paths):
        stamps = pd.date_range(
            dates[index + 1] + pd.Timedelta(hours=9, minutes=30), periods=len(values), freq="1min"
        )
        frames.append(pd.DataFrame({
            "timestamp": stamps, "open": values, "high": [value + 0.01 for value in values],
            "low": [value - 0.01 for value in values], "close": values, "volume": 100000,
            "turnover": [value * 100000 for value in values], "frequency": "1m",
            "tradable": True, "is_synthetic": False,
        }))
    bars = pd.concat(frames, ignore_index=True)
    daily = pd.DataFrame({
        "date": dates, "open": np.r_[closes[0], closes], "high": np.r_[closes[0], closes] + 0.01,
        "low": np.r_[closes[0], closes] - 0.01, "close": np.r_[closes[0], closes],
        "preclose": np.r_[np.nan, closes[0], closes[:-1]], "volume": 100000,
        "turnover": np.r_[closes[0], closes] * 100000, "tradestatus": 1, "is_st": False,
        "limit_up": np.r_[100.0, (limit_ups or [100.0] * len(paths))],
        "limit_down": np.r_[0.01, [0.01] * len(paths)],
    })
    config = BacktestConfig(
        schema_version=3, symbol=symbol, start_date=dates[1].date(), end_date=dates[-1].date(),
        frequency="1m" if len(paths[0]) > 1 else "1d",
        initial_shares=initial_shares, min_shares=0, max_shares=None,
        buy_shares=buy_shares, sell_shares=100,
        grid_mode="geometric", buy_step=buy_step, sell_step=buy_step,
        reference_price_mode="manual", reference_price=reference,
        fees=fees or FEES,
    )
    bundle = DataBundle(
        symbol=symbol, bars=bars, daily=daily,
        actions=actions if actions is not None else pd.DataFrame(),
        calendar=pd.DatetimeIndex(dates[1:]), metadata=instrument_terms(symbol, {}),
        version="fixture-v5-ledger",
    )
    return config, bundle


def grid_trade(result, *, external_topup=None):
    trades = result["strategies"]["grid"]["trades"]
    if external_topup is None:
        return trades
    return [trade for trade in trades if trade["external_topup"] > 0]


def test_matched_account_starts_with_the_same_assets_and_holdings_as_the_grid():
    config, bundle = build_bundle([[10.0], [10.0], [9.5], [9.4]])
    result = engine.run_backtest(config, bundle)

    grid = result["strategies"]["grid"]
    matched = result["strategies"]["matched_funding_hold"]
    assert grid["metrics"]["initial_equity"] == pytest.approx(10000.0)
    assert matched["metrics"]["initial_equity"] == pytest.approx(grid["metrics"]["initial_equity"])
    assert result["funding"]["initial_principal"] == pytest.approx(10000.0)

    first_grid, first_matched = grid["equity"][0], matched["equity"][0]
    assert first_grid["position"] == pytest.approx(1000.0)
    assert first_matched["position"] == pytest.approx(first_grid["position"])
    assert first_matched["cash"] == pytest.approx(first_grid["cash"])
    assert first_matched["total_invested"] == pytest.approx(first_grid["total_invested"])


def test_only_filled_grid_orders_contribute_matching_funding_in_the_same_bar():
    config, bundle = build_bundle([[10.0], [10.0], [9.5], [9.0]])
    result = engine.run_backtest(config, bundle)

    flows = result["funding"]["funding_flows"]
    fills = grid_trade(result, external_topup=True)
    assert len(fills) == 1 and len(flows) == 1
    assert fills[0]["timestamp"] == flows[0]["timestamp"]
    assert fills[0]["external_topup"] == pytest.approx(flows[0]["amount"])

    grid = result["strategies"]["grid"]["metrics"]
    matched = result["strategies"]["matched_funding_hold"]["metrics"]
    assert matched["external_topups"] == pytest.approx(grid["external_topups"])
    assert matched["total_invested"] == pytest.approx(grid["total_invested"])
    assert matched["total_invested"] == pytest.approx(10000.0 + flows[0]["amount"])

    # The contribution lands in the matched account on exactly the same bar and
    # is deployed into whole lots; only the remainder may stay as cash.
    row = next(
        item for item in result["strategies"]["matched_funding_hold"]["equity"]
        if item["timestamp"] == flows[0]["timestamp"]
    )
    assert row["total_invested"] == pytest.approx(10000.0 + flows[0]["amount"])
    spent = sum(
        trade["quantity"] * trade["price"] + trade["fees"]
        for trade in result["strategies"]["matched_funding_hold"]["trades"]
        if trade["timestamp"] == flows[0]["timestamp"]
    )
    assert row["cash"] == pytest.approx(flows[0]["amount"] - spent)
    assert row["cash"] >= -0.005


def test_rejected_grid_orders_do_not_contribute_any_funding():
    # The signal fires on day three, but the next session opens locked at the
    # limit-up price, so the grid order is rejected instead of filled.
    config, bundle = build_bundle(
        [[10.0], [10.0], [9.5], [9.0]], limit_ups=[100.0, 100.0, 100.0, 9.0]
    )
    result = engine.run_backtest(config, bundle)

    assert result["funding"]["topups"] == pytest.approx(0.0)
    assert result["funding"]["funding_flows"] == []
    assert grid_trade(result) == []
    rejections = result["strategies"]["grid"]["rejections"]
    assert any(item["reason"] == "limit_up_locked" for item in rejections)

    matched = result["strategies"]["matched_funding_hold"]
    assert matched["trades"] == []
    assert matched["metrics"]["total_invested"] == pytest.approx(10000.0)
    assert matched["metrics"]["external_topups"] == pytest.approx(0.0)


def test_top_up_contributions_are_principal_and_never_profit():
    config, bundle = build_bundle([[10.0], [10.0], [9.5], [9.0]])
    result = engine.run_backtest(config, bundle)

    flows = result["funding"]["funding_flows"]
    contributed = sum(flow["amount"] for flow in flows)
    matched = result["strategies"]["matched_funding_hold"]
    metrics = matched["metrics"]
    last = matched["equity"][-1]

    assert metrics["total_invested"] == pytest.approx(10000.0 + contributed)
    assert metrics["external_topups"] == pytest.approx(contributed)
    # The contribution is principal, so the reported profit stays the change
    # of the position value only.
    assert last["cumulative_profit"] == pytest.approx(last["equity"] - last["total_invested"])
    assert last["total_invested"] == pytest.approx(metrics["total_invested"])

    # Every contributed yuan is either a position cost, a fee or leftover
    # cash; none of it is booked as a gain.
    spent = sum(trade["quantity"] * trade["price"] + trade["fees"] for trade in matched["trades"])
    assert contributed == pytest.approx(spent + last["cash"] - 0.0, abs=0.02)


def test_matched_account_buys_the_fee_inclusive_max_whole_lot_and_keeps_the_remainder():
    # A 200 share grid fill contributes 1605.02, which buys exactly two whole
    # lots after the commission is paid.  The remainder must stay on the
    # ledger and the account must never spend more than it received.
    config, bundle = build_bundle(
        [[10.0], [10.0], [9.5], [8.0], [8.0]], buy_shares=200
    )
    result = engine.run_backtest(config, bundle)

    matched = result["strategies"]["matched_funding_hold"]
    trades = matched["trades"]
    assert len(trades) == 1
    trade = trades[0]
    assert trade["quantity"] % 100 == 0
    assert trade["quantity"] == 200

    leftover = trade["cash"]
    assert leftover >= -0.005, "配对账户不得透支"
    # The remainder is a partial lot: it cannot cover one more lot plus fees.
    assert leftover < trade["price"] * 100 + trade["fees"]
    assert matched["metrics"]["total_invested"] == pytest.approx(
        10000.0 + result["funding"]["topups"]
    )


def test_ledger_reports_a_reason_when_no_whole_lot_is_affordable():
    # K=0: the cash cannot cover a single whole lot including the fees.  The
    # attempt must return no fill and a machine readable reason.
    result = _nb_attempt(
        1, 100, 10.0, 100000.0, float("nan"), float("nan"), 0.0, 0.00001,
        False, 0.0, False, 0, 500.0, 0.0, 0.0, 0, 0, 2147483647.0, 0.0,
        0.0003, 5.0, 0.0, 0, 1.0, 0.01, True, False, False,
    )

    assert result[0] is False, "K=0 时不得成交"
    assert result[9] == 5, "K=0 必须返回拒单原因码（insufficient_cash）"
    assert engine._REJECTION_REASONS[result[9]] == "insufficient_cash"


def test_ledger_sizes_a_buy_by_the_fee_inclusive_maximum_whole_lot():
    # 1600.02 buys 100 shares plus a partial lot; the fee inclusive maximum is
    # therefore one whole lot, not two.
    result = _nb_attempt(
        1, 200, 8.0, 100000.0, float("nan"), float("nan"), 0.0, 0.00001,
        False, 0.0, False, 0, 1600.02, 0.0, 0.0, 0, 0, 2147483647.0, 0.0,
        0.0003, 5.0, 0.0, 0, 1.0, 0.01, True, False, False,
    )

    assert result[0] is True
    assert result[4] == 100, "含费用最大整手：1600.02 只够一手"
    assert result[1] == pytest.approx(1600.02 - 800.0 - 5.01, abs=0.02)
    assert result[1] >= -0.005


def test_matched_account_can_deploy_the_contribution_into_the_grid_position():
    # The paired baseline is only comparable to the grid when the same money,
    # at the earliest tradable open, buys the same position with the
    # fee inclusive maximum whole lot.
    config, bundle = build_bundle([[10.0], [10.0], [9.5], [9.0]])
    result = engine.run_backtest(config, bundle)

    grid = result["strategies"]["grid"]
    matched = result["strategies"]["matched_funding_hold"]
    fill = grid_trade(result, external_topup=True)[0]
    matched_trades = matched["trades"]
    assert len(matched_trades) == 1, "配对账户必须把注资变成与网格一致的持仓"
    assert matched_trades[0]["timestamp"] == fill["timestamp"]
    assert matched_trades[0]["price"] == pytest.approx(fill["price"])
    assert matched_trades[0]["quantity"] == fill["quantity"]


def test_zero_principal_returns_none_with_an_explicit_reason():
    # Without initial holding and without any fill the peak principal stays
    # zero: the return must be None plus a reason, never 0 or NaN.
    config, bundle = build_bundle([[10.0], [10.0], [10.0], [10.0]], initial_shares=0)
    result = engine.run_backtest(config, bundle)

    for name in ("grid", "matched_funding_hold"):
        metrics = result["strategies"][name]["metrics"]
        assert metrics["peak_principal_return"] is None, f"{name} 的收益必须是 None"
        assert metrics["max_principal_drawdown"] is None, f"{name} 的回撤必须是 None"
        assert metrics.get("unevaluable_reason"), f"{name} 必须给出无法评价的原因"


def test_principal_drawdown_is_measured_on_every_bar_not_only_at_the_session_close():
    # The price dips inside the session and recovers before the close, so a
    # daily sampled drawdown would report zero.
    config, bundle = build_bundle([[10.0], [10.0], [10.0, 9.0, 8.0, 9.0, 10.0], [10.0]])
    result = engine.run_backtest(config, bundle)

    grid = result["strategies"]["grid"]
    rows = grid["equity"]
    per_bar = max(row["principal_drawdown"] for row in rows)
    last_of_day = {}
    for row in rows:
        last_of_day[row["timestamp"][:10]] = row
    per_day = max(row["principal_drawdown"] for row in last_of_day.values())

    assert per_bar > 0.0
    assert per_day < per_bar, "日线抽样不得漏掉盘中本金回撤"
    assert grid["metrics"]["max_principal_drawdown"] == pytest.approx(per_bar)


def test_cash_dividend_and_bonus_shares_follow_the_ledger():
    actions = pd.DataFrame([{
        "record_date": pd.Timestamp("2024-01-02"), "ex_date": pd.Timestamp("2024-01-03"),
        "pay_date": pd.Timestamp("2024-01-04"), "stock_date": pd.Timestamp("2024-01-04"),
        "cash_per_share": 0.5, "bonus_ratio": 0.1, "implemented": True, "unsupported_reason": "",
    }])
    config, bundle = build_bundle(
        [[10.0], [10.0], [10.0], [10.0]], actions=actions,
        # This fixture keeps all raw prices at 10 while the action implies an
        # ex-right reference near 8.636.  Wide boundaries isolate settlement
        # from the intentional re-anchoring that would otherwise sell a grid
        # lot before the dividend cash is paid.
        buy_step=0.5,
    )
    result = engine.run_backtest(config, bundle)

    assert grid_trade(result) == [], "权益结算测试不得混入网格自主成交"
    assert result["funding"]["topups"] == pytest.approx(0.0)

    for name in ("grid", "matched_funding_hold"):
        rows = result["strategies"][name]["equity"]
        ex_date, pay_date = rows[0], rows[1]
        # Ex date: 1000 shares earn 0.5 cash and 0.1 bonus shares.
        assert ex_date["position"] == pytest.approx(1100.0)
        assert ex_date["cash"] == pytest.approx(0.0)
        assert ex_date["equity"] == pytest.approx(1100.0 * 10.0 + 500.0)
        # Pay date: the dividend is settled and the bonus shares are issued.
        assert pay_date["cash"] == pytest.approx(500.0)
        assert pay_date["position"] == pytest.approx(1100.0)
        assert pay_date["equity"] == pytest.approx(1100.0 * 10.0 + 500.0)


def test_interpreted_and_numba_full_backtests_agree(monkeypatch):
    config, bundle = build_bundle([[10.0], [10.0], [9.5], [9.0], [8.8]])
    fast = engine.run_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    slow = engine.run_backtest(config, bundle)

    assert slow["funding"] == fast["funding"]
    assert slow["strategies"] == fast["strategies"]
    assert slow["strategies"]["grid"]["trades"] == fast["strategies"]["grid"]["trades"]
    assert (
        slow["strategies"]["matched_funding_hold"]["metrics"]
        == fast["strategies"]["matched_funding_hold"]["metrics"]
    )
