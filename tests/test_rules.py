from datetime import date

import pytest

from stockregression.rules import (
    calculated_price_limits,
    commissions_and_taxes,
    fee_rates,
    floor_buy_quantity,
    legal_sell_quantity,
    lot_size,
    price_limit_rate,
    round_to_tick,
)


def test_board_price_limits_follow_historical_st_rules():
    assert price_limit_rate("main", date(2026, 7, 5), is_st=True) == 0.05
    assert price_limit_rate("main", date(2026, 7, 6), is_st=True) == 0.10
    assert price_limit_rate("main", date(2026, 7, 6), is_st=False) == 0.10
    assert price_limit_rate("chinext", date(2020, 8, 23), is_st=True) == 0.05
    assert price_limit_rate("chinext", date(2020, 8, 24), is_st=True) == 0.20
    assert price_limit_rate("star", date(2024, 1, 1), is_st=True) == 0.20


def test_limit_prices_round_to_tick_and_execution_rounding_is_adverse():
    assert calculated_price_limits(10, "main", date(2024, 1, 1)) == (11.0, 9.0)
    assert round_to_tick(10.001, side="buy") == 10.01
    assert round_to_tick(10.009, side="sell") == 10.0


def test_board_lots_and_final_odd_lot_sales():
    assert lot_size("main", "buy") == 100
    assert lot_size("star", "buy") == 200
    assert floor_buy_quantity(199, "star") == 0
    assert floor_buy_quantity(205, "star") == 205
    assert floor_buy_quantity(299, "main") == 200
    assert legal_sell_quantity(90, 90, "main") == 90
    assert legal_sell_quantity(150, 250, "main") == 100
    assert legal_sell_quantity(190, 190, "star") == 190
    assert legal_sell_quantity(150, 500, "star") == 0
    assert legal_sell_quantity(100, 100, "star") == 100


def test_stamp_duty_and_transfer_fee_change_on_effective_dates():
    assert fee_rates(date(2023, 8, 27)) == pytest.approx((0.001, 0.00001))
    assert fee_rates(date(2023, 8, 28)) == pytest.approx((0.0005, 0.00001))
    assert fee_rates(date(2022, 4, 28))[1] == pytest.approx(0.00002)
    assert fee_rates(date(2022, 4, 29))[1] == pytest.approx(0.00001)
    assert fee_rates(date(2008, 4, 23))[0] == pytest.approx(0.003)
    assert fee_rates(date(2008, 4, 24))[0] == pytest.approx(0.001)


def test_fee_floor_and_seller_only_stamp_after_2008():
    small_buy = commissions_and_taxes(1000, "buy", date(2024, 1, 2), 0.0003, 5)
    assert small_buy["commission"] == 5
    assert small_buy["stamp_duty"] == 0
    large_sell = commissions_and_taxes(1_000_000, "sell", date(2024, 1, 2), 0.0003, 5)
    assert large_sell["commission"] == 300
    assert large_sell["stamp_duty"] == 500
    assert large_sell["transfer_fee"] == 10


def test_fee_components_round_the_decimal_rate_product_half_up():
    # 44,450 * 0.0003 is exactly 13.335 in the fee schedule's decimal values.
    # Float multiplication can produce 13.334999999999999 and lose a cent.
    charges = commissions_and_taxes(44_450, "sell", date(2026, 1, 27), 0.0003, 5)
    assert charges == {
        "commission": 13.34,
        "stamp_duty": 22.23,
        "transfer_fee": 0.44,
        "total": 36.01,
    }


def test_stamp_duty_was_bilateral_before_september_2008():
    buy = commissions_and_taxes(
        1_000, "buy", date(2008, 9, 18), 0, 0,
        symbol="600000.SH", quantity=1_000,
    )
    sell = commissions_and_taxes(
        1_000, "sell", date(2008, 9, 18), 0, 0,
        symbol="600000.SH", quantity=1_000,
    )
    assert buy["stamp_duty"] == sell["stamp_duty"] == 1


def test_historical_sh_transfer_fee_uses_share_face_value_not_notional():
    # Both orders have the same notional. Before August 2015 SH charged based
    # on share face value (normally one yuan), so the share counts matter.
    small_quantity = commissions_and_taxes(
        40_000, "buy", date(2014, 6, 2), 0, 0,
        symbol="600000.SH", quantity=2_000,
    )
    large_quantity = commissions_and_taxes(
        40_000, "buy", date(2014, 6, 2), 0, 0,
        symbol="600000.SH", quantity=4_000,
    )
    assert small_quantity["transfer_fee"] == 0.60
    assert large_quantity["transfer_fee"] == 1.20


def test_historical_sz_transfer_fee_uses_notional():
    fewer_shares = commissions_and_taxes(
        40_000, "buy", date(2014, 6, 2), 0, 0,
        symbol="000001.SZ", quantity=2_000,
    )
    more_shares = commissions_and_taxes(
        40_000, "buy", date(2014, 6, 2), 0, 0,
        symbol="000001.SZ", quantity=4_000,
    )
    assert fewer_shares["transfer_fee"] == more_shares["transfer_fee"] == 1.02


def test_transfer_fee_historical_rate_change_boundaries():
    def transfer(day, symbol, quantity, notional):
        return commissions_and_taxes(
            notional, "buy", day, 0, 0,
            symbol=symbol, quantity=quantity,
        )["transfer_fee"]

    assert transfer(date(2012, 6, 1), "600000.SH", 1_000, 10_000) == 0.38
    assert transfer(date(2012, 8, 31), "600000.SH", 1_000, 10_000) == 0.38
    assert transfer(date(2012, 9, 1), "600000.SH", 1_000, 10_000) == 0.30
    assert transfer(date(2015, 7, 31), "600000.SH", 1_000, 10_000) == 0.30
    assert transfer(date(2015, 8, 1), "600000.SH", 40_000, 40_000) == 0.80


def test_legacy_sh_transfer_fee_uses_configured_override_and_minimum():
    charges = commissions_and_taxes(
        1_000, "buy", date(2011, 5, 3), 0, 0,
        symbol="600000.SH", quantity=100, face_value=1.5,
        legacy_sh_transfer_rate=0.002,
        legacy_sh_min_transfer=0.5,
    )
    assert charges["transfer_fee"] == 0.50


def test_pre_2015_call_without_exchange_or_quantity_warns_and_uses_fallback():
    with pytest.warns(UserWarning, match="缺少symbol或quantity"):
        charges = commissions_and_taxes(1_000, "buy", date(2014, 6, 2), 0, 0)
    assert charges["transfer_fee"] == 0.02
