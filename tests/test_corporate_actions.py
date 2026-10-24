from __future__ import annotations

import pandas as pd
import pytest

from stockregression.corporate_actions import (
    LEGACY_SPLIT_REASON, action_issues, resolve_actions, split_record,
)
from stockregression.models import DataBundle
from stockregression.price_basis import action_factors, adjust_frame
from stockregression.providers import normalize_actions


def prices(ratio=2.0):
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-03-26", "2026-03-27", "2026-03-30", "2026-03-31"]),
        "close": [10., 10., 10. / ratio, 10. / ratio],
        "preclose": [10., 10., 10. / ratio, 10. / ratio],
    })


@pytest.mark.parametrize("ratio", [2.0, 0.5, 2.5])
def test_vendor_fold_day_resolves_next_real_session_without_network(ratio):
    actions = normalize_actions(pd.DataFrame([split_record({
        "拆分折算日": "2026-03-27", "拆分折算": ratio,
    })]))
    result = resolve_actions(actions, prices(ratio))
    assert result.iloc[0].ex_date == pd.Timestamp("2026-03-30")
    assert result.iloc[0].record_date == pd.Timestamp("2026-03-27")
    assert result.iloc[0].stock_date == pd.Timestamp("2026-03-30")
    assert result.iloc[0].bonus_ratio == ratio - 1
    assert not result.iloc[0].unsupported_reason
    pd.testing.assert_frame_equal(result, resolve_actions(result, prices(ratio)))


def test_split_on_fold_day_uses_previous_observed_session():
    daily = prices()
    daily.loc[1, ["close", "preclose"]] = 5.
    result = resolve_actions(pd.DataFrame([split_record({
        "拆分折算日": "2026-03-27", "拆分折算": 2.,
    })]), daily)
    assert result.iloc[0].ex_date == pd.Timestamp("2026-03-27")
    assert result.iloc[0].record_date == pd.Timestamp("2026-03-26")


def test_missing_ratio_is_not_silently_converted_to_zero():
    actions = normalize_actions(pd.DataFrame([split_record({"拆分折算日": "2026-03-27"})]))
    resolved = resolve_actions(actions, prices())
    assert pd.isna(resolved.iloc[0].bonus_ratio)
    assert action_issues(resolved, prices(), "2026-03-26", "2026-03-31")
    assert action_issues(resolved, prices(), "2026-03-31", "2026-03-31") == []


def test_complete_legacy_split_no_longer_requires_audit_flag():
    action = pd.DataFrame([{
        "record_date": "2026-03-27", "ex_date": "2026-03-30", "stock_date": "2026-03-30",
        "cash_per_share": 0., "bonus_ratio": 1., "unsupported_reason": LEGACY_SPLIT_REASON,
    }])
    assert action_issues(action, prices(), "2026-03-26", "2026-03-31", settlement=True) == []


def test_settlement_dates_only_required_for_accounting():
    action = pd.DataFrame([{"ex_date": "2026-03-30", "cash_per_share": 5., "bonus_ratio": 0.}])
    assert action_issues(action, prices(), "2026-03-26", "2026-03-31") == []
    assert action_issues(action, prices(), "2026-03-26", "2026-03-31", settlement=True)


def test_display_uses_reference_without_inventing_share_entitlements():
    daily = prices()
    bundle = DataBundle(symbol="159995.SZ", bars=pd.DataFrame(), daily=daily,
                        actions=pd.DataFrame(), calendar=pd.DatetimeIndex(daily.date),
                        version="test", metadata={"asset_type": "etf"})
    factors, issue = action_factors(bundle, pd.Timestamp("2026-03-31"), allow_reference_only=True)
    assert issue is None
    adjusted = adjust_frame(daily, factors, timestamp_column="date")
    assert adjusted.close.tolist() == [5.] * 4
    assert action_issues(pd.DataFrame(), daily, daily.date.min(), daily.date.max())


def test_display_unresolved_fold_placeholder_does_not_shift_real_jump():
    daily = prices()
    action = pd.DataFrame([{
        "record_date": "2026-03-27", "ex_date": "2026-03-27", "stock_date": "2026-03-27",
        "cash_per_share": 0., "bonus_ratio": 0., "unsupported_reason": LEGACY_SPLIT_REASON,
    }])
    bundle = DataBundle(symbol="159995.SZ", bars=pd.DataFrame(), daily=daily,
                        actions=action, calendar=pd.DatetimeIndex(daily.date), version="test")
    factors, issue = action_factors(bundle, daily.date.max(), allow_reference_only=True)
    assert issue is None
    assert [item["ex_date"] for item in factors] == [pd.Timestamp("2026-03-30")]
    assert adjust_frame(daily, factors, timestamp_column="date").close.tolist() == [5.] * 4
