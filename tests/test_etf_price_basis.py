from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import stockregression.engine as engine
from stockregression.models import BacktestConfig, DataBundle
from stockregression.price_basis import action_factors, adjust_frame, price_factor_at
from stockregression.share_engine import run_share_backtest


def _daily_frame(days, closes, precloses):
    return pd.DataFrame({
        "date": days,
        "open": closes,
        "high": [value * 1.02 for value in closes],
        "low": [value * 0.98 for value in closes],
        "close": closes,
        "preclose": precloses,
    })


def _actions(*rows):
    return pd.DataFrame([
        {
            "record_date": row["record_date"],
            "ex_date": row["ex_date"],
            "pay_date": row.get("pay_date", row["ex_date"]),
            "stock_date": row.get("stock_date", row["ex_date"]),
            "cash_per_share": row.get("cash_per_share", 0.0),
            "bonus_ratio": row.get("bonus_ratio", 0.0),
            "implemented": True,
            "unsupported_reason": "",
        }
        for row in rows
    ])


def test_action_factors_adjust_prices_and_only_share_bonus_for_volume():
    days = pd.bdate_range("2024-01-02", periods=6)
    daily = _daily_frame(
        days,
        [20.0, 20.0, 19.0, 19.0, 9.5, 9.5],
        [20.0, 20.0, 19.0, 19.0, 9.5, 9.5],
    )
    actions = _actions(
        {"record_date": days[0], "ex_date": days[2], "cash_per_share": 1.0},
        {"record_date": days[2], "ex_date": days[4], "bonus_ratio": 1.0},
    )
    bundle = DataBundle(
        symbol="510050.SH", bars=pd.DataFrame(), daily=daily, actions=actions,
        calendar=days, version="fixture", metadata={"asset_type": "etf"},
    )

    factors, issue = action_factors(bundle, days[-1])

    assert issue is None
    assert [pd.Timestamp(item["ex_date"]) for item in factors] == [days[2], days[4]]
    assert [item["factor"] for item in factors] == pytest.approx([0.95, 0.5])
    assert price_factor_at(factors, days[0]) == pytest.approx(0.475)
    assert price_factor_at(factors, days[2]) == pytest.approx(0.5)
    assert price_factor_at(factors, days[4]) == pytest.approx(1.0)

    frame = pd.DataFrame({
        "timestamp": [
            days[0] + pd.Timedelta(hours=9, minutes=31),
            days[2] + pd.Timedelta(hours=9, minutes=31),
            days[4] + pd.Timedelta(hours=9, minutes=31),
        ],
        "open": [20.0, 19.0, 9.5],
        "high": [21.0, 20.0, 10.0],
        "low": [19.0, 18.0, 9.0],
        "close": [20.0, 19.0, 9.5],
        "preclose": [20.0, 19.0, 9.5],
        "volume": [100.0, 100.0, 100.0],
        "turnover": [2000.0, 1900.0, 950.0],
    })
    before = frame.copy(deep=True)

    adjusted = adjust_frame(frame, factors, adjust_volume=True)

    pd.testing.assert_frame_equal(frame, before)
    assert adjusted.loc[0, "close"] == pytest.approx(9.5)
    assert adjusted.loc[0, "preclose"] == pytest.approx(9.5)
    assert adjusted.loc[0, "volume"] == pytest.approx(200.0)
    assert adjusted.loc[1, "close"] == pytest.approx(9.5)
    # The second bonus is still in the future on this row's ex-date.
    assert adjusted.loc[1, "volume"] == pytest.approx(200.0)
    assert adjusted.loc[2, "close"] == pytest.approx(9.5)
    assert adjusted.loc[2, "volume"] == pytest.approx(100.0)
    # Cash adjustment must never multiply the original transaction amount.
    pd.testing.assert_series_equal(adjusted["turnover"], before["turnover"])


@pytest.mark.parametrize("share_ratio", [2.0, 0.5])
def test_volume_diagnostics_do_not_skip_split_or_consolidation_adjustment(share_ratio):
    days = pd.bdate_range("2024-01-02", periods=2)
    daily = _daily_frame(days, [10.0, 10.0 / share_ratio], [10.0, 10.0 / share_ratio])
    bundle = DataBundle(
        symbol="510050.SH", bars=pd.DataFrame(), daily=daily,
        actions=_actions({
            "record_date": days[0], "ex_date": days[1],
            "bonus_ratio": share_ratio - 1.0,
        }),
        calendar=days, version="fixture",
        metadata={"asset_type": "etf", "etf_volume_verified": False},
    )
    factors, issue = action_factors(bundle, days[-1])
    assert issue is None
    frame = daily.rename(columns={"date": "timestamp"}).assign(volume=100.0)
    adjusted = adjust_frame(frame, factors, adjust_volume=True)
    assert adjusted.volume.tolist() == pytest.approx([100.0 * share_ratio, 100.0])
    assert adjusted.close.tolist() == pytest.approx([10.0 / share_ratio] * 2)


def _share_bundle(days, closes, precloses, action):
    bars = []
    daily = []
    for day, close, preclose in zip(days, closes, precloses):
        bars.append({
            "timestamp": day + pd.Timedelta(hours=15),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 100_000.0,
            "turnover": close * 100_000.0,
            "frequency": "1d",
            "tradable": True,
            "is_synthetic": False,
            "preclose": preclose,
            "is_st": False,
            "limit_up": float("nan"),
            "limit_down": float("nan"),
            "no_price_limit": True,
        })
        daily.append({
            "date": day,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 100_000.0,
            "turnover": close * 100_000.0,
            "preclose": preclose,
            "tradestatus": 1,
            "is_st": False,
            "no_price_limit": True,
        })
    return DataBundle(
        symbol="510050.SH", bars=pd.DataFrame(bars), daily=pd.DataFrame(daily),
        actions=_actions(action), calendar=pd.DatetimeIndex(days),
        version="share-boundary-fixture",
        metadata={
            "asset_type": "etf", "board": "main", "price_tick": 0.001,
            "prior_closes": {str((days[0] - pd.offsets.BDay(1)).date()): 10.0},
        },
    )


def _share_config(days, *, lower, upper):
    return BacktestConfig(
        schema_version=3, symbol="510050.SH", start_date=days[0].date(),
        end_date=days[-1].date(), frequency="1d", initial_shares=1000,
        min_shares=0, max_shares=2000, buy_shares=100, sell_shares=100,
        grid_mode="geometric", buy_step=0.03, sell_step=0.03,
        reference_price_mode="manual", reference_price=10.0,
        lower_price=lower, upper_price=upper,
        fees={"commission_rate": 0, "min_commission": 0, "slippage_bps": 0},
    )


@pytest.mark.parametrize(
    ("action", "ex_preclose", "ex_close"),
    [
        ({"cash_per_share": 1.0}, 9.0, 8.6),
        ({"bonus_ratio": 1.0}, 5.0, 4.8),
    ],
)
def test_corporate_action_adjusts_lower_before_grid_buy(action, ex_preclose, ex_close):
    days = pd.bdate_range("2024-01-02", periods=3)
    action = {"record_date": days[0], "ex_date": days[1], **action}
    bundle = _share_bundle(
        days, [10.0, ex_close, ex_close], [10.0, ex_preclose, ex_close], action,
    )
    config = _share_config(days, lower=9.5, upper=12.0).model_copy(
        update={"max_shares": 3000}
    )
    result = run_share_backtest(config, bundle)
    grid = result["strategies"]["grid"]

    assert [trade["reason"] for trade in grid["trades"]].count("grid_buy") == 1


@pytest.mark.parametrize(
    ("action", "ex_preclose", "ex_close"),
    [
        ({"cash_per_share": 1.0}, 9.0, 9.6),
        ({"bonus_ratio": 1.0}, 5.0, 5.4),
    ],
)
def test_corporate_action_adjusts_upper_before_grid_sell(action, ex_preclose, ex_close):
    days = pd.bdate_range("2024-02-01", periods=3)
    action = {"record_date": days[0], "ex_date": days[1], **action}
    bundle = _share_bundle(
        days, [10.0, ex_close, ex_close], [10.0, ex_preclose, ex_close], action,
    )
    config = _share_config(days, lower=9.5, upper=10.5).model_copy(
        update={"max_shares": 3000}
    )
    result = run_share_backtest(config, bundle)
    grid = result["strategies"]["grid"]

    assert "grid_sell" not in {trade["reason"] for trade in grid["trades"]}


def test_lower_boundary_adjustment_is_identical_in_compiled_and_python_kernel(monkeypatch):
    days = pd.bdate_range("2024-03-01", periods=3)
    action = {"record_date": days[0], "ex_date": days[1], "cash_per_share": 1.0}
    bundle = _share_bundle(
        days, [10.0, 8.6, 8.6], [10.0, 9.0, 8.6], action,
    )
    config = _share_config(days, lower=9.5, upper=12.0)

    compiled = run_share_backtest(config, bundle)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    interpreted = run_share_backtest(config, bundle)

    compiled_grid = compiled["strategies"]["grid"]
    interpreted_grid = interpreted["strategies"]["grid"]
    assert compiled_grid["trades"] == interpreted_grid["trades"]
    assert compiled_grid["rejections"] == interpreted_grid["rejections"]


@pytest.mark.parametrize(
    ("ex_close", "expected_reason"),
    [(8.2, None), (9.3, "grid_sell")],
)
def test_price_bands_are_adjusted_once_per_ex_date(ex_close, expected_reason):
    days = pd.bdate_range("2024-04-01", periods=3)
    action = {"record_date": days[0], "ex_date": days[1], "cash_per_share": 1.0}
    bundle = _share_bundle(
        days, [10.0, ex_close, ex_close], [10.0, 9.0, ex_close], action,
    )
    result = run_share_backtest(_share_config(days, lower=9.5, upper=10.5), bundle)
    reasons = [trade["reason"] for trade in result["strategies"]["grid"]["trades"]]

    if expected_reason is None:
        assert "grid_buy" not in reasons
    else:
        assert expected_reason in reasons


def test_stale_preclose_uses_the_shared_cash_and_bonus_factor():
    days = pd.bdate_range("2024-05-06", periods=3)
    action = {"record_date": days[0], "ex_date": days[1], "cash_per_share": 1.0}
    # The daily preclose is stale and still says 10.0.  The verified action
    # factor is 0.90, so the adjusted lower band is 8.55 and an 8.6 close may
    # buy.  Trusting the stale preclose would leave the band at 9.5 and reject.
    bundle = _share_bundle(
        days, [10.0, 8.6, 8.6], [10.0, 10.0, 8.6], action,
    )
    result = run_share_backtest(_share_config(days, lower=9.5, upper=12.0), bundle)

    assert "grid_buy" in {
        trade["reason"] for trade in result["strategies"]["grid"]["trades"]
    }
