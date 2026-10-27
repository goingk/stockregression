from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockregression.models import DataBundle
from stockregression.validation_scenarios import (
    STRESS_CASES,
    SYNTHETIC_CASES,
    largest_contribution_index,
    stress_cases,
    synthetic_bundle,
    synthetic_closes,
)


def _template(symbol: str, metadata: dict, *, is_st: bool = False) -> DataBundle:
    dates = pd.bdate_range("2024-01-02", periods=3)
    closes = [10.0, 10.2, 999.0]
    daily = pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1000.0] * 3,
        "turnover": [10000.0] * 3,
        "preclose": closes,
        "tradestatus": 1,
        "is_st": is_st,
        "no_price_limit": False,
    })
    return DataBundle(
        symbol=symbol, bars=pd.DataFrame(), daily=daily,
        calendar=pd.DatetimeIndex(dates), metadata=metadata,
    )


def test_six_execution_pressure_cases_and_unfunded_omission_marker():
    funded = stress_cases({
        "contributions": [
            {"amount": 20, "available_at": "2024-01-03"},
            {"amount": 50, "available_at": "2024-01-02"},
        ],
    })
    assert tuple(item["case"] for item in funded) == STRESS_CASES
    assert largest_contribution_index({
        "contributions": [
            {"amount": 20, "available_at": "2024-01-03"},
            {"amount": 50, "available_at": "2024-01-02"},
        ],
    }) == 0
    assert funded[-2]["applicable"] is True
    assert funded[-2]["kwargs"] == {"omitted_contribution_index": 0}
    assert funded[-1]["kwargs"] == {
        "cost_multiplier": 2.0, "volume_multiplier": 0.5,
        "extra_delay_bars": 1, "omitted_contribution_index": 0,
    }

    unfunded = stress_cases({"contributions": []})
    by_name = {item["case"]: item for item in unfunded}
    assert by_name["max_contribution_omit"]["applicable"] is False
    assert by_name["max_contribution_omit"]["reason"] == "no_scheduled_contributions"
    assert by_name["combined"]["applicable"] is False


def test_synthetic_paths_are_deterministic_and_keep_legal_limits():
    assert synthetic_closes("down30").min() == 7.0
    assert synthetic_closes("down30")[-1] == 7.0
    assert synthetic_closes("up30")[-1] == 13.0
    low = synthetic_closes("low_volatility")
    assert low.min() >= 9.9
    assert low.max() <= 10.1
    gap = synthetic_closes("gapdown", gap_limit_rate=.10)
    gap_index = 63 // 2
    assert gap[gap_index] / gap[gap_index - 1] - 1 == pytest.approx(-.1)

    bundle = synthetic_bundle("159611.SZ", "gapdown", start="2024-01-02")
    assert len(bundle.daily) == 63
    assert bundle.metadata["market_store_written"] is False
    assert bundle.metadata["synthetic_validation"] is True
    assert bundle.metadata["no_real_future_schedule"] is True
    assert bundle.bars["validation_synthetic"].all()
    assert not bundle.bars["is_synthetic"].any()
    gap_row = bundle.daily.iloc[gap_index]
    assert gap_row.close == gap_row.preclose * .90
    assert gap_row.limit_down == gap_row.preclose * .90
    assert set(SYNTHETIC_CASES) == {"down30", "up30", "low_volatility", "gapdown"}


def test_synthetic_limits_follow_anchor_st_and_twenty_percent_etf_rules():
    st = _template("600000.SH", {"asset_type": "stock", "board": "main", "price_tick": .01}, is_st=True)
    st_bundle = synthetic_bundle(
        "600000.SH", "gapdown", template=st, decision_date="2024-01-03",
    )
    assert st_bundle.metadata["valuation_price"] == pytest.approx(10.2)
    assert st_bundle.metadata["price_limit_rate"] == pytest.approx(.05)
    st_gap = st_bundle.daily.iloc[63 // 2]
    assert st_gap.close == pytest.approx(st_gap.preclose * .95)
    assert st_gap.limit_down == pytest.approx(st_gap.preclose * .95)
    assert st_gap.limit_up == pytest.approx(st_gap.preclose * 1.05)

    etf = _template("588000.SH", {
        "asset_type": "etf", "board": "etf", "price_tick": .001,
        "name": "科创ETF",
    })
    etf_bundle = synthetic_bundle(
        "588000.SH", "gapdown", template=etf, decision_date="2024-01-03",
    )
    assert etf_bundle.metadata["price_limit_rate"] == pytest.approx(.20)
    etf_gap = etf_bundle.daily.iloc[63 // 2]
    assert etf_gap.close == pytest.approx(etf_gap.preclose * .90)
    assert etf_gap.limit_up == pytest.approx(etf_gap.preclose * 1.20)
    assert etf_gap.limit_down == pytest.approx(etf_gap.preclose * .80)
    for bundle, tick in ((st_bundle, .01), (etf_bundle, .001)):
        for column in ("open", "high", "low", "close", "preclose"):
            values = bundle.daily[column].to_numpy() / tick
            assert max(abs(values - values.round())) < 1e-7
        assert (bundle.daily.high >= bundle.daily[["open", "close"]].max(axis=1)).all()
        assert (bundle.daily.low <= bundle.daily[["open", "close"]].min(axis=1)).all()


def test_synthetic_origin_ignores_future_template_rows_and_wrong_start_price():
    template = _template("510300.SH", {
        "asset_type": "etf", "board": "etf", "price_tick": .001,
    })
    first = synthetic_bundle(
        "510300.SH", "up30", template=template, decision_date="2024-01-03",
    )
    template.daily.loc[template.daily.index[-1], ["open", "high", "low", "close"]] = 1.0
    template.metadata["valuation_price"] = 500.0
    second = synthetic_bundle(
        "510300.SH", "up30", template=template, decision_date="2024-01-03",
    )
    assert first.metadata["valuation_price"] == pytest.approx(10.2)
    assert first.metadata["anchor_price_date"] == "2024-01-03"
    assert first.version == second.version
    pd.testing.assert_frame_equal(first.daily, second.daily)
    with pytest.raises(ValueError, match="decision_date"):
        synthetic_bundle(
            "510300.SH", "up30", template=template,
            decision_date="2024-01-03", start_price=500,
        )


def test_all_four_synthetic_bundles_are_independent_in_memory_values():
    from stockregression.validation_scenarios import synthetic_cases

    bundles = synthetic_cases("512880.SH", start=pd.Timestamp("2024-02-01"))
    assert [name for name, _ in bundles] == list(SYNTHETIC_CASES)
    assert len({bundle.version for _, bundle in bundles}) == 4
    first = bundles[0][1]
    original = first.daily.close.iloc[0]
    first.daily.loc[0, "close"] = 999
    assert bundles[1][1].daily.close.iloc[0] == original

