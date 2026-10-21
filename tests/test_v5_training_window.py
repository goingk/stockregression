"""v5 研究协议 K2「选窗」契约测试。

覆盖 stockregression.research_v4.window 的 TrainingWindow /
TrainingWindowPolicy：

* 锚点（anchor）定义与 daily 行情取值规则；
* same_price 模式的搜索区间、严格不等号命中与「取最远命中日」；
* 公司行动复权因子的优先级、同日多行合并、多日累计与 data_issue；
* fixed 模式的最大回看窗口；
* diagnostic() 暴露的完整契约字段。

只断言 v5 新契约，不包含 protocol v4 / bootstrap 兼容行为。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from stockregression.models import DataBundle
from stockregression.research_v4.window import (
    TrainingWindow,
    TrainingWindowPolicy,
    training_window,
)


DAILY_COLUMNS = (
    "date", "open", "high", "low", "close", "volume", "turnover",
    "preclose", "tradestatus", "is_st", "limit_up", "limit_down",
)

ACTION_COLUMNS = (
    "record_date", "ex_date", "pay_date", "stock_date",
    "cash_per_share", "bonus_ratio", "implemented", "unsupported_reason",
)


def _session(day, close, *, high=None, low=None, preclose=None, open_=None) -> dict[str, Any]:
    """One daily row; omitted high/low/open fall back to the close."""
    return {
        "date": pd.Timestamp(day),
        "open": close if open_ is None else open_,
        "high": close if high is None else high,
        "low": close if low is None else low,
        "close": close,
        "preclose": np.nan if preclose is None else preclose,
    }


def _bundle(sessions, *, actions=None, calendar=None, symbol="600000.SH", metadata=None) -> DataBundle:
    """A bundle whose only price source is the daily frame."""
    rows = []
    for session in sessions:
        rows.append({
            "date": session["date"],
            "open": session["open"],
            "high": session["high"],
            "low": session["low"],
            "close": session["close"],
            "volume": 1000.0,
            "turnover": 1000.0,
            "preclose": session["preclose"],
            "tradestatus": 1,
            "is_st": 0,
            "limit_up": np.nan,
            "limit_down": np.nan,
        })
    daily = pd.DataFrame(rows, columns=list(DAILY_COLUMNS))
    index = (pd.DatetimeIndex(calendar) if calendar is not None
             else pd.DatetimeIndex([row["date"] for row in rows]))
    return DataBundle(
        symbol=symbol,
        bars=pd.DataFrame(),
        daily=daily,
        actions=pd.DataFrame() if actions is None else actions,
        calendar=index,
        quality={},
        version="v5-window-test",
        metadata={} if metadata is None else metadata,
        warnings=[],
    )


def _actions(*entries) -> pd.DataFrame:
    rows = []
    for entry in entries:
        ex_date = pd.Timestamp(entry["ex_date"])
        rows.append({
            "record_date": entry.get("record_date", ex_date),
            "ex_date": ex_date,
            "pay_date": entry.get("pay_date", ex_date),
            "stock_date": entry.get("stock_date", ex_date),
            "cash_per_share": entry.get("cash_per_share", 0.0),
            "bonus_ratio": entry.get("bonus_ratio", 0.0),
            "implemented": entry.get("implemented", True),
            "unsupported_reason": entry.get("unsupported_reason", ""),
        })
    return pd.DataFrame(rows, columns=list(ACTION_COLUMNS))


def _row(bundle: DataBundle, day) -> pd.Series:
    mask = bundle.daily["date"] == pd.Timestamp(day)
    return bundle.daily.loc[mask].iloc[0]


def _with_daily(bundle: DataBundle, daily: pd.DataFrame) -> DataBundle:
    return DataBundle(
        symbol=bundle.symbol,
        bars=bundle.bars,
        daily=daily,
        actions=bundle.actions,
        calendar=bundle.calendar,
        quality=bundle.quality,
        version=bundle.version,
        metadata=bundle.metadata,
        warnings=bundle.warnings,
    )


def test_split_adjusted_window_matches_the_unadjusted_equivalent_path():
    """2 送 1 拆分复权后，必须与无拆分的平价路径选出完全相同的训练窗口。"""
    dates = pd.bdate_range("2024-01-02", periods=12)
    anchor = dates[-1]
    ex_date = dates[10]

    split_sessions = []
    for index, day in enumerate(dates):
        if index < 10:
            # 拆分前原始价是拆分后价格的 2 倍。
            split_sessions.append(_session(
                day, 40.0, high=42.0, low=38.0,
                preclose=np.nan if index == 0 else 40.0,
            ))
        else:
            # 除权日与锚点日的 preclose 都是拆分后价格（20.0），构成有效 preclose。
            split_sessions.append(_session(day, 20.0, high=21.0, low=19.0, preclose=20.0))
    split_bundle = _bundle(
        split_sessions, actions=_actions({"ex_date": ex_date, "bonus_ratio": 1.0}),
    )

    plain_sessions = [_session(day, 20.0, high=21.0, low=19.0) for day in dates]
    plain_bundle = _bundle(plain_sessions)

    split_window = training_window(split_bundle, anchor, min_offset=3, max_offset=6)
    plain_window = training_window(plain_bundle, anchor, min_offset=3, max_offset=6)

    assert split_window.matched is True
    assert plain_window.matched is True
    pd.testing.assert_index_equal(split_window.dates, plain_window.dates)
    assert split_window.matched_date == plain_window.matched_date == dates[6]
    assert split_window.start == plain_window.start == dates[6]
    assert split_window.end == plain_window.end == anchor
    assert split_window.days == plain_window.days == 6
    assert split_window.anchor_date == plain_window.anchor_date == anchor
    assert split_window.anchor_price == plain_window.anchor_price == 20.0
    assert split_window.fallback is False and plain_window.fallback is False
    assert split_window.start_reason == plain_window.start_reason == "same_price_farthest"

    # 命中日位于复权生效的历史区间内，且未复权时该区间并不包含锚点收盘价。
    assert split_window.search_end < ex_date
    raw = _row(split_bundle, dates[6])
    factor = split_window.actions_applied[0]["factor"]
    assert float(raw["low"]) > split_window.anchor_price
    assert float(raw["low"]) * factor < split_window.anchor_price
    assert float(raw["high"]) * factor > split_window.anchor_price

    assert len(split_window.actions_applied) == 1
    assert split_window.actions_applied[0]["factor"] == pytest.approx(0.5)
    assert split_window.actions_applied[0]["source"] == "verified_preclose"
    assert split_window.actions_applied[0]["action_count"] == 1
    assert split_window.actions_applied[0]["bonus_ratio"] == 1.0
    assert split_window.actions_applied[0]["ex_date"] == ex_date.date().isoformat()
    assert plain_window.actions_applied == []


def test_cash_dividend_shrinks_history_and_breaks_the_same_price_match():
    """只有现金分红、没有有效 preclose 时，历史 high/low 按 (前收盘-现金)/前收盘 缩小。"""
    dates = pd.bdate_range("2024-03-01", periods=20)
    anchor = dates[-1]
    ex_date = dates[12]

    sessions = [_session(day, 20.0, high=20.5, low=19.5) for day in dates]
    plain = training_window(_bundle(sessions), anchor, min_offset=10, max_offset=12)
    assert plain.matched is True
    assert plain.matched_date == dates[8]
    assert plain.start == dates[8]

    dividend_bundle = _bundle(
        sessions, actions=_actions({"ex_date": ex_date, "cash_per_share": 1.0}),
    )
    dividend = training_window(dividend_bundle, anchor, min_offset=10, max_offset=12)

    assert dividend.matched is False
    assert dividend.fallback is True
    assert dividend.matched_date is None
    assert dividend.start_reason == "five_hundred_day_fallback"
    assert dividend.data_issue is None
    assert len(dividend.actions_applied) == 1
    action = dividend.actions_applied[0]
    assert action["factor"] == pytest.approx(0.95)
    assert action["source"] == "cash_and_bonus"
    assert action["cash_per_share"] == 1.0
    assert action["bonus_ratio"] == 0.0
    assert action["action_count"] == 1
    assert action["ex_date"] == ex_date.date().isoformat()
    # 复权后候选日最高价跌到锚点收盘价之下，因此不再包住锚点价。
    assert 20.5 * action["factor"] < dividend.anchor_price

    # 完全没有 preclose 列，与 preclose 为 NaN 等价：同样退回 cash_and_bonus。
    stripped = _with_daily(dividend_bundle, dividend_bundle.daily.drop(columns=["preclose"]))
    stripped_window = training_window(stripped, anchor, min_offset=10, max_offset=12)
    assert stripped_window.matched is False
    assert stripped_window.actions_applied[0]["source"] == "cash_and_bonus"
    assert stripped_window.actions_applied[0]["factor"] == pytest.approx(0.95)


def test_actions_on_the_same_ex_date_are_merged_into_one_factor():
    """同一 ex_date 的现金 + 送转两行合并成一组，除权因子按合并后的金额计算。"""
    dates = pd.bdate_range("2024-04-01", periods=12)
    anchor = dates[-1]
    ex_date = dates[10]

    sessions = []
    for index, day in enumerate(dates):
        if index < 10:
            high, low = (22.0, 18.0) if index == 6 else (21.0, 19.0)
            sessions.append(_session(day, 20.0, high=high, low=low))
        else:
            sessions.append(_session(day, 13.0, high=13.5, low=12.5))
    bundle = _bundle(
        sessions,
        actions=_actions(
            {"ex_date": ex_date, "cash_per_share": 0.5, "bonus_ratio": 0.0},
            {"ex_date": ex_date, "cash_per_share": 0.0, "bonus_ratio": 0.5},
        ),
    )

    window = training_window(bundle, anchor, min_offset=3, max_offset=6)

    assert len(window.actions_applied) == 1
    action = window.actions_applied[0]
    assert action["action_count"] == 2
    assert action["cash_per_share"] == pytest.approx(0.5)
    assert action["bonus_ratio"] == pytest.approx(0.5)
    assert action["factor"] == pytest.approx((20.0 - 0.5) / (20.0 * (1.0 + 0.5)))
    assert action["factor"] == pytest.approx(0.65)
    assert action["source"] == "cash_and_bonus"
    assert action["ex_date"] == ex_date.date().isoformat()

    # 未复权时候选日并不包住锚点收盘价，合并后的因子才让它包住。
    raw = _row(bundle, dates[6])
    assert not (float(raw["low"]) < window.anchor_price < float(raw["high"]))
    assert float(raw["low"]) * action["factor"] < window.anchor_price
    assert float(raw["high"]) * action["factor"] > window.anchor_price
    assert window.matched is True
    assert window.matched_date == dates[6]


def test_multiple_ex_dates_compound_factors_cumulatively():
    """两个 ex_date 的因子依次作用，更早的日期承受的是两个因子的乘积。"""
    dates = pd.bdate_range("2024-05-06", periods=20)
    anchor = dates[-1]
    cash_ex_date, bonus_ex_date = dates[5], dates[10]

    sessions = []
    for index, day in enumerate(dates):
        if index < 5:
            close = 20.0
            high, low = (22.0, 18.0) if index == 2 else (21.0, 19.0)
        elif index < 10:
            close, high, low = 18.0, 19.0, 17.0
        else:
            close, high, low = 9.0, 9.5, 8.5
        sessions.append(_session(day, close, high=high, low=low))
    bundle = _bundle(
        sessions,
        actions=_actions(
            {"ex_date": cash_ex_date, "cash_per_share": 2.0},
            {"ex_date": bonus_ex_date, "bonus_ratio": 1.0},
        ),
    )

    window = training_window(bundle, anchor, min_offset=12, max_offset=18)

    assert len(window.actions_applied) == 2
    first, second = window.actions_applied
    assert first["ex_date"] == cash_ex_date.date().isoformat()
    assert first["factor"] == pytest.approx(0.9)
    assert first["source"] == "cash_and_bonus"
    assert first["cash_per_share"] == 2.0
    assert first["bonus_ratio"] == 0.0
    assert first["action_count"] == 1
    assert second["ex_date"] == bonus_ex_date.date().isoformat()
    assert second["factor"] == pytest.approx(0.5)
    assert second["source"] == "cash_and_bonus"
    assert second["cash_per_share"] == 0.0
    assert second["bonus_ratio"] == 1.0
    assert second["action_count"] == 1

    # dates[2] 早于两个 ex_date，同时承受 0.9 与 0.5：累计因子为 0.45。
    compounded = first["factor"] * second["factor"]
    assert compounded == pytest.approx(0.45)
    raw = _row(bundle, dates[2])
    assert not (float(raw["low"]) < window.anchor_price < float(raw["high"]))
    assert float(raw["low"]) * compounded < window.anchor_price
    assert float(raw["high"]) * compounded > window.anchor_price
    # 只施加第一个因子时该日不会命中，因此命中「最远日」即证明因子是累计的。
    assert window.search_start == dates[2]
    assert window.matched is True
    assert window.matched_date == dates[2]
    assert window.start == dates[2]
    assert window.end == anchor
    assert window.anchor_price == 9.0


def test_same_price_match_requires_strict_low_and_high_bracketing():
    """low 必须严格小于、high 必须严格大于锚点收盘价，相等即不命中。"""
    dates = pd.bdate_range("2024-06-03", periods=6)
    anchor = dates[-1]
    candidate = dates[-2]

    def bundle_for(high, low):
        sessions = []
        for index, day in enumerate(dates):
            if index == len(dates) - 2:
                sessions.append(_session(day, 20.0, high=high, low=low))
            else:
                sessions.append(_session(day, 20.0, high=21.0, low=19.0))
        return _bundle(sessions)

    # min_offset == max_offset == 2 时搜索区间收敛到锚点的前一个交易日。
    low_equal = training_window(bundle_for(21.0, 20.0), anchor, min_offset=2, max_offset=2)
    assert low_equal.matched is False
    assert low_equal.fallback is True
    assert low_equal.matched_date is None
    assert low_equal.start_reason == "five_hundred_day_fallback"

    high_equal = training_window(bundle_for(20.0, 19.0), anchor, min_offset=2, max_offset=2)
    assert high_equal.matched is False
    assert high_equal.fallback is True
    assert high_equal.matched_date is None

    bracketed = training_window(bundle_for(21.0, 19.0), anchor, min_offset=2, max_offset=2)
    assert bracketed.matched is True
    assert bracketed.matched_date == candidate
    assert bracketed.start == candidate
    assert bracketed.end == anchor
    assert bracketed.days == 2
    assert bracketed.fallback is False
    assert bracketed.start_reason == "same_price_farthest"


def test_no_same_price_day_falls_back_to_the_max_offset_window():
    """区间内没有同价日时回退到评估日前 max_offset 个交易日。"""
    days = 520
    dates = pd.bdate_range("2023-01-02", periods=days)
    anchor = dates[-1]
    anchor_index = days - 1
    evaluation_index = anchor_index + 1

    sessions = []
    for index, day in enumerate(dates):
        close = 20.0 + 0.01 * (days - 1 - index)
        sessions.append(_session(day, close, high=close + 0.05, low=close - 0.05))
    bundle = _bundle(sessions)

    window = training_window(bundle, anchor)

    assert window.matched is False
    assert window.fallback is True
    assert window.matched_date is None
    assert window.start_reason == "five_hundred_day_fallback"
    assert window.data_issue is None
    assert window.match_rule == "daily_low_lt_anchor_close_lt_daily_high"
    assert window.mode == "same_price"
    assert window.anchor_date == anchor
    # 回退起点 = evaluation_index - max_offset，窗口覆盖到锚点日共 max_offset 个交易日。
    assert window.start == dates[evaluation_index - 500]
    assert window.end == anchor
    assert window.days == 500
    assert window.search_start == dates[evaluation_index - 500]
    assert window.search_end == dates[evaluation_index - 250]
    assert "没有同价日" in window.reason


def test_future_sessions_and_future_actions_do_not_change_the_window():
    """asof 之后的交易日与 ex_date > anchor 的行动都不改变已选定的窗口。"""
    base_dates = pd.bdate_range("2024-07-01", periods=12)
    anchor = base_dates[-1]
    ex_date = base_dates[10]

    base_sessions = []
    for index, day in enumerate(base_dates):
        if index < 10:
            base_sessions.append(_session(
                day, 40.0, high=42.0, low=38.0,
                preclose=np.nan if index == 0 else 40.0,
            ))
        else:
            base_sessions.append(_session(day, 20.0, high=21.0, low=19.0, preclose=20.0))
    base_actions = _actions({"ex_date": ex_date, "bonus_ratio": 1.0})
    base_bundle = _bundle(base_sessions, actions=base_actions)
    base_window = training_window(base_bundle, anchor, min_offset=3, max_offset=6)

    extended_dates = pd.bdate_range("2024-07-01", periods=15)
    extended_sessions = list(base_sessions) + [
        _session(day, 20.0, high=21.0, low=19.0, preclose=20.0) for day in extended_dates[12:14]
    ] + [
        _session(extended_dates[14], 20.0, high=21.0, low=19.0, preclose=10.0)
    ]
    extended_actions = pd.concat(
        [base_actions, _actions({"ex_date": extended_dates[13], "cash_per_share": 5.0})],
        ignore_index=True,
    )
    extended_bundle = _bundle(
        extended_sessions, actions=extended_actions, calendar=extended_dates,
    )
    extended_window = training_window(extended_bundle, anchor, min_offset=3, max_offset=6)

    pd.testing.assert_index_equal(extended_window.dates, base_window.dates)
    assert extended_window.start == base_window.start
    assert extended_window.end == base_window.end
    assert extended_window.matched == base_window.matched
    assert extended_window.matched_date == base_window.matched_date
    assert extended_window.anchor_date == base_window.anchor_date
    assert extended_window.anchor_price == base_window.anchor_price
    assert (extended_window.diagnostic()["action_factor_fingerprint"]
            == base_window.diagnostic()["action_factor_fingerprint"])
    assert base_window.diagnostic()["action_factor_fingerprint"] is not None
    assert len(extended_window.actions_applied) == 1
    assert extended_window.actions_applied[0]["ex_date"] == ex_date.date().isoformat()


def test_missing_prior_close_with_a_valid_preclose_is_reported_as_a_data_issue():
    """除权日前一交易日缺少收盘价时，除权因子无效，窗口退化为 data_issue。"""
    dates = pd.bdate_range("2024-08-01", periods=12)
    anchor = dates[-1]
    ex_date = dates[10]

    sessions = []
    for index, day in enumerate(dates):
        close = np.nan if index == 9 else 20.0
        sessions.append(_session(day, close, high=21.0, low=19.0, preclose=20.0))
    bundle = _bundle(
        sessions, actions=_actions({"ex_date": ex_date, "bonus_ratio": 1.0}),
    )

    window = training_window(bundle, anchor, min_offset=3, max_offset=6)

    assert window.data_issue is not None
    assert "除权因子无效" in window.data_issue
    assert ex_date.date().isoformat() in window.data_issue
    assert len(window.dates) == 0
    assert window.start is None
    assert window.end is None
    assert window.anchor_date == anchor
    assert window.anchor_price == 20.0
    assert window.reason == window.data_issue
    assert window.mode == "same_price"
    assert window.matched is False


def test_dividend_not_smaller_than_prior_close_is_reported_as_a_data_issue():
    """现金分红额不小于前收盘且无有效 preclose 时，除权因子无效。"""
    dates = pd.bdate_range("2024-09-02", periods=12)
    anchor = dates[-1]
    ex_date = dates[10]
    sessions = [_session(day, 20.0, high=21.0, low=19.0) for day in dates]

    for cash in (20.0, 25.0):
        bundle = _bundle(
            sessions, actions=_actions({"ex_date": ex_date, "cash_per_share": cash}),
        )
        window = training_window(bundle, anchor, min_offset=3, max_offset=6)

        assert window.data_issue is not None
        assert "除权因子无效" in window.data_issue
        assert ex_date.date().isoformat() in window.data_issue
        assert len(window.dates) == 0
        assert window.start is None
        assert window.end is None
        assert window.anchor_date == anchor
        assert window.anchor_price == 20.0
        assert window.reason == window.data_issue
        assert window.mode == "same_price"
        assert window.matched is False


def test_unsupported_corporate_action_is_reported_as_a_data_issue():
    """implemented 且带 unsupported_reason 的行动直接触发 data_issue。"""
    dates = pd.bdate_range("2024-10-01", periods=12)
    anchor = dates[-1]
    ex_date = dates[10]
    sessions = [_session(day, 20.0, high=21.0, low=19.0) for day in dates]
    bundle = _bundle(
        sessions,
        actions=_actions({
            "ex_date": ex_date, "implemented": True, "unsupported_reason": "配股未支持",
        }),
    )

    window = training_window(bundle, anchor, min_offset=3, max_offset=6)

    assert window.data_issue is not None
    assert "未支持" in window.data_issue
    assert ex_date.date().isoformat() in window.data_issue
    assert "配股未支持" in window.data_issue
    assert len(window.dates) == 0
    assert window.start is None
    assert window.end is None
    assert window.anchor_date == anchor
    assert window.anchor_price == 20.0
    assert window.reason == window.data_issue
    assert window.mode == "same_price"
    assert window.matched is False


def test_unimplemented_corporate_action_is_reported_as_a_data_issue():
    """明确标记未实施的公司行动必须报 data_issue，不得静默跳过复权。"""
    dates = pd.bdate_range("2024-10-21", periods=12)
    anchor = dates[-1]
    ex_date = dates[10]
    sessions = [_session(day, 20.0, high=21.0, low=19.0) for day in dates]

    bundle = _bundle(
        sessions,
        actions=_actions({"ex_date": ex_date, "cash_per_share": 1.0, "implemented": False}),
    )

    window = training_window(bundle, anchor, min_offset=3, max_offset=6)

    assert window.data_issue is not None
    assert "未确认已实施" in window.data_issue
    assert ex_date.date().isoformat() in window.data_issue
    assert len(window.dates) == 0
    assert window.start is None
    assert window.end is None
    assert window.matched is False
    assert window.fallback is False
    assert window.reason == window.data_issue
    assert window.mode == "same_price"
    assert window.actions_applied == [], "未确认的行动不得进入复权因子"

    # 字符串标记同样是明确的未实施。
    flagged = _bundle(
        sessions,
        actions=_actions({"ex_date": ex_date, "cash_per_share": 1.0, "implemented": "false"}),
    )
    assert training_window(flagged, anchor, min_offset=3, max_offset=6).data_issue is not None

    # 同一份行情在行动确认已实施时必须正常选窗，两种结果不得混同。
    verified = _bundle(
        sessions,
        actions=_actions({"ex_date": ex_date, "cash_per_share": 1.0, "implemented": True}),
    )
    confirmed = training_window(verified, anchor, min_offset=3, max_offset=6)
    assert confirmed.data_issue is None
    assert confirmed.actions_applied[0]["factor"] == pytest.approx(0.95)


def test_stale_preclose_is_not_trusted_and_falls_back_to_the_action_factor():
    """preclose 与公司行动不一致时不得当作已核验参考价，必须改用行动因子并记录差异。"""
    dates = pd.bdate_range("2024-03-01", periods=20)
    anchor = dates[-1]
    ex_date = dates[12]

    def sessions_with(preclose: float):
        rows = []
        for index, day in enumerate(dates):
            rows.append(_session(
                day, 20.0, high=20.5, low=19.5,
                preclose=np.nan if index == 0 else preclose,
            ))
        return rows

    stale = _bundle(
        sessions_with(20.0), actions=_actions({"ex_date": ex_date, "cash_per_share": 1.0}),
    )
    window = training_window(stale, anchor, min_offset=10, max_offset=12)

    assert window.data_issue is None
    action = window.actions_applied[0]
    assert action["source"] == "cash_and_bonus"
    assert action["factor"] == pytest.approx(0.95)
    assert action["preclose"] == pytest.approx(20.0)
    assert action["implied_reference"] == pytest.approx(19.0)
    assert action["preclose_mismatch"] is not None
    assert action["preclose_mismatch"]["difference"] == pytest.approx(1.0)
    assert action["preclose_mismatch"]["observed_reference"] == pytest.approx(20.0)
    # 复权后候选日的高低价不再包住锚点收盘价，因此不再命中同价日。
    assert window.matched is False
    assert window.fallback is True

    # 与行动一致的 preclose 才是已核验参考价。
    verified = _bundle(
        sessions_with(19.0), actions=_actions({"ex_date": ex_date, "cash_per_share": 1.0}),
    )
    good = training_window(verified, anchor, min_offset=10, max_offset=12)
    assert good.data_issue is None
    assert good.actions_applied[0]["source"] == "verified_preclose"
    assert good.actions_applied[0]["preclose_mismatch"] is None
    assert good.actions_applied[0]["factor"] == pytest.approx(0.95)


def test_window_verification_does_not_depend_on_the_data_package_descriptor():
    """除权验证只依赖价格与行动本身，不得依赖数据包描述符的写法。"""
    dates = pd.bdate_range("2024-06-03", periods=16)
    anchor = dates[-1]
    ex_date = dates[10]
    sessions = []
    for index, day in enumerate(dates):
        sessions.append(_session(
            day, 20.0, high=21.0, low=19.0,
            preclose=np.nan if index == 0 else (19.0 if index == 10 else 20.0),
        ))
    actions = _actions({"ex_date": ex_date, "cash_per_share": 1.0})

    reference = training_window(_bundle(sessions, actions=actions), anchor, min_offset=3, max_offset=6)
    assert reference.data_issue is None
    assert reference.actions_applied[0]["source"] == "verified_preclose"

    for metadata in ({}, {"price_tick": 0.001}, {"price_tick": 0.01, "asset_type": "stock"}):
        window = training_window(
            _bundle(sessions, actions=actions, metadata=metadata), anchor, min_offset=3, max_offset=6,
        )
        assert window.diagnostic() == reference.diagnostic(), (
            "同一份价格与行动在不同数据包描述符下必须得到同一个窗口"
        )

    # 行动表缺少 implemented 描述符时按已实施处理，不得误报数据问题。
    stripped = actions.drop(columns=["implemented"])
    without_flag = training_window(
        _bundle(sessions, actions=stripped), anchor, min_offset=3, max_offset=6,
    )
    assert without_flag.data_issue is None
    assert without_flag.actions_applied[0]["action_count"] == 1
    assert without_flag.actions_applied[0]["factor"] == pytest.approx(0.95)


def test_fixed_mode_takes_exactly_max_offset_sessions():
    """fixed 模式历史充足时，窗口长度恰好等于 max_offset 且没有 data_issue。"""
    days = 30
    dates = pd.bdate_range("2024-11-01", periods=days)
    anchor = dates[-1]
    sessions = [_session(day, 20.0, high=21.0, low=19.0) for day in dates]
    bundle = _bundle(
        sessions, actions=_actions({"ex_date": dates[5], "cash_per_share": 1.0}),
    )

    window = training_window(bundle, anchor, min_offset=3, max_offset=10, mode="fixed")

    assert window.mode == "fixed"
    assert window.days == 10
    assert len(window.dates) == 10
    assert window.data_issue is None
    assert window.start == dates[days - 10]
    assert window.end == anchor
    assert window.dates[-1] == anchor
    assert window.anchor_date == anchor
    assert window.anchor_price == 20.0
    assert window.matched is False
    assert window.fallback is False
    assert window.match_rule == "fixed_training_max_days"
    assert window.start_reason == "fixed_training_max_days", (
        "fixed 模式必须报告固定窗口规则，不得报告同价回退原因"
    )
    # fixed 模式不做公司行动复权。
    assert not window.actions_applied


def test_fixed_mode_reports_insufficient_history():
    """fixed 模式历史不足 max_offset 个交易日时给出明确的 data_issue。"""
    days = 12
    dates = pd.bdate_range("2024-11-01", periods=days)
    anchor = dates[-1]
    sessions = [_session(day, 20.0, high=21.0, low=19.0) for day in dates]
    bundle = _bundle(sessions)

    window = training_window(bundle, anchor, min_offset=3, max_offset=20, mode="fixed")

    assert window.mode == "fixed"
    assert window.days == 0
    assert len(window.dates) == 0
    assert window.start is None
    assert window.end is None
    assert window.data_issue is not None
    assert "固定训练窗口需要 20 个交易日" in window.data_issue
    assert "仅有 12 个交易日" in window.data_issue
    assert window.reason == window.data_issue
    assert window.fallback is True
    assert window.match_rule == "fixed_training_max_days"
    assert window.start_reason == "fixed_training_max_days"
    assert window.anchor_date == anchor
    assert window.anchor_price == 20.0


def test_the_farthest_same_price_day_inside_the_offset_range_wins():
    """搜索区间内存在多个同价日时，取离评估日最远的那一天。"""
    days = 20
    dates = pd.bdate_range("2025-01-02", periods=days)
    anchor = dates[-1]

    sessions = []
    for index, day in enumerate(dates):
        # dates[1] 在 min/max_offset 之外（太远），dates[9] 太近，都不得被选中。
        high, low = (21.0, 19.0) if index in (1, 2, 4, 8, 9) else (20.0, 20.0)
        sessions.append(_session(day, 20.0, high=high, low=low))
    bundle = _bundle(sessions)

    window = training_window(bundle, anchor, min_offset=12, max_offset=18)

    assert window.search_start == dates[2]
    assert window.search_end == dates[8]
    assert window.matched is True
    assert window.matched_date == dates[2]
    assert window.start == dates[2]
    assert window.end == anchor
    assert window.days == days - 2
    assert window.fallback is False
    assert window.start_reason == "same_price_farthest"
    assert "最远" in window.reason


def test_anchor_is_the_last_session_with_a_known_close():
    """锚点是 asof 之前最后一个收盘价有限且为正的交易日。"""
    dates = pd.bdate_range("2025-02-03", periods=6)
    anchor = dates[-1]

    def bundle_with(anchor_close):
        sessions = []
        for index, day in enumerate(dates):
            close = anchor_close if index == len(dates) - 1 else 20.0
            sessions.append(_session(day, close, high=21.0, low=19.0))
        return _bundle(sessions)

    for bad_close in (np.nan, 0.0, -5.0, float("inf")):
        window = training_window(bundle_with(bad_close), anchor, min_offset=1, max_offset=1)
        assert window.anchor_date == dates[-2]
        assert window.anchor_price == 20.0
        assert window.matched is True
        assert window.matched_date == dates[-2]


def test_policy_window_matches_the_functional_api_and_validates_arguments():
    """TrainingWindowPolicy 与函数式 API 等价，并拒绝非法参数组合。"""
    dates = pd.bdate_range("2025-03-03", periods=12)
    anchor = dates[-1]
    sessions = [_session(day, 20.0, high=21.0, low=19.0) for day in dates]
    bundle = _bundle(sessions)

    policy = TrainingWindowPolicy(min_offset=3, max_offset=6, mode="same_price")
    from_policy = policy.window(bundle, anchor)
    from_function = training_window(bundle, anchor, min_offset=3, max_offset=6, mode="same_price")

    assert isinstance(from_policy, TrainingWindow)
    assert from_policy.dates.equals(from_function.dates)
    assert from_policy.start == from_function.start
    assert from_policy.end == from_function.end
    assert from_policy.matched == from_function.matched
    assert from_policy.matched_date == from_function.matched_date
    assert from_policy.match_rule == from_function.match_rule
    assert from_policy.mode == from_function.mode
    assert policy.diagnostic() == {
        "min_offset_trading_days": 3, "max_offset_trading_days": 6, "mode": "same_price",
    }
    assert TrainingWindowPolicy(min_offset=3, max_offset=6, mode="fixed").window(bundle, anchor).mode == "fixed"

    with pytest.raises(ValueError):
        TrainingWindowPolicy(min_offset=0)
    with pytest.raises(ValueError):
        TrainingWindowPolicy(min_offset=10, max_offset=5)
    with pytest.raises(ValueError):
        TrainingWindowPolicy(mode="bogus")


def test_diagnostic_reports_the_full_window_contract():
    """diagnostic() 必须暴露完整的窗口契约字段。"""
    dates = pd.bdate_range("2025-04-01", periods=12)
    anchor = dates[-1]
    ex_date = dates[10]

    sessions = []
    for index, day in enumerate(dates):
        if index < 10:
            sessions.append(_session(
                day, 40.0, high=42.0, low=38.0,
                preclose=np.nan if index == 0 else 40.0,
            ))
        else:
            sessions.append(_session(day, 20.0, high=21.0, low=19.0, preclose=20.0))
    bundle = _bundle(
        sessions, actions=_actions({"ex_date": ex_date, "bonus_ratio": 1.0}),
    )

    window = training_window(bundle, anchor, min_offset=3, max_offset=6)
    diagnostic = window.diagnostic()

    assert set(diagnostic) == {
        "start", "end", "days", "anchor_date", "anchor_price", "matched_date", "matched",
        "fallback", "start_reason", "search_start", "search_end", "min_offset_trading_days",
        "max_offset_trading_days", "match_rule", "reason", "mode", "price_basis",
        "actions_applied", "action_factor_fingerprint", "data_issue",
    }
    assert diagnostic["start"] == dates[6].date().isoformat()
    assert diagnostic["end"] == anchor.date().isoformat()
    assert diagnostic["days"] == window.days == len(window.dates)
    assert diagnostic["anchor_date"] == anchor.date().isoformat()
    assert diagnostic["anchor_price"] == 20.0
    assert diagnostic["matched_date"] == dates[6].date().isoformat()
    assert diagnostic["matched"] is True
    assert diagnostic["fallback"] is False
    assert diagnostic["start_reason"] == "same_price_farthest"
    assert diagnostic["search_start"] == dates[6].date().isoformat()
    assert diagnostic["search_end"] == dates[9].date().isoformat()
    assert diagnostic["min_offset_trading_days"] == 3
    assert diagnostic["max_offset_trading_days"] == 6
    assert diagnostic["match_rule"] == "daily_low_lt_anchor_close_lt_daily_high"
    assert diagnostic["mode"] == "same_price"
    assert diagnostic["price_basis"] == "decision_date_adjusted"
    assert diagnostic["data_issue"] is None
    assert diagnostic["actions_applied"] == window.actions_applied
    fingerprint = diagnostic["action_factor_fingerprint"]
    assert isinstance(fingerprint, str) and len(fingerprint) == 64


def test_daily_high_and_low_must_be_finite_and_positive():
    """非有限或非正的 high/low 一律视为缺失，该交易日不可能命中。"""
    dates = pd.bdate_range("2025-05-05", periods=6)
    anchor = dates[-1]

    def bundle_for(high, low):
        sessions = []
        for index, day in enumerate(dates):
            if index == len(dates) - 2:
                sessions.append(_session(day, 20.0, high=high, low=low))
            else:
                sessions.append(_session(day, 20.0, high=21.0, low=19.0))
        return _bundle(sessions)

    for high, low in ((0.0, 19.0), (float("inf"), 19.0), (21.0, 0.0), (21.0, float("-inf"))):
        window = training_window(bundle_for(high, low), anchor, min_offset=2, max_offset=2)
        assert window.matched is False
        assert window.fallback is True
        assert window.matched_date is None

    ok = training_window(bundle_for(21.0, 19.0), anchor, min_offset=2, max_offset=2)
    assert ok.matched is True
    assert ok.matched_date == dates[-2]
