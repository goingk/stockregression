from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from stockregression import providers
from stockregression.quality import DataQualityError


def sina_frame(open_values=None):
    return pd.DataFrame({
        "day": pd.date_range("2024-01-02 09:31", periods=3, freq="min"),
        "open": open_values if open_values is not None else [10.0, 10.1, 10.2],
        "high": [10.2, 10.3, 10.4],
        "low": [9.9, 10.0, 10.1],
        "close": [10.1, 10.2, 10.3],
        "volume": [100, 200, 300],
        "amount": [1000, 2000, 3000],
    })


def fake_akshare(monkeypatch, frame):
    calls = []

    def stock_zh_a_minute(symbol, period, adjust):
        calls.append(symbol)
        return frame

    def fund_etf_hist_min_em(**_kwargs):
        raise AssertionError("ETF 近期分钟不再使用东财 trends2 接口（历史交易日开盘价为 0）")

    monkeypatch.setitem(sys.modules, "akshare", types.SimpleNamespace(
        stock_zh_a_minute=stock_zh_a_minute,
        fund_etf_hist_min_em=fund_etf_hist_min_em,
    ))
    return calls


def test_fetch_sina_uses_sina_minute_feed_for_etf(monkeypatch):
    calls = fake_akshare(monkeypatch, sina_frame())

    frame = providers.fetch_sina("159611.SZ")

    assert calls == ["sz159611"]
    assert frame.timestamp.tolist() == list(pd.date_range("2024-01-02 09:31", periods=3, freq="min"))
    assert (frame.open > 0).all()
    assert frame.source.tolist() == ["sina"] * 3


def test_fetch_sina_rejects_non_positive_prices(monkeypatch):
    fake_akshare(monkeypatch, sina_frame([0.0, 10.1, 10.2]))

    with pytest.raises(DataQualityError, match="非正价格"):
        providers.fetch_sina("159611.SZ")


class FakeBaostockResult:
    error_code = "0"
    error_msg = ""

    def __init__(self, fields, rows):
        self.fields = fields.split(",")
        self._rows = iter(rows)
        self._current = None

    def next(self):
        self._current = next(self._rows, None)
        return self._current is not None

    def get_row_data(self):
        return self._current


def fake_baostock(monkeypatch, daily_rows=(), calendar_rows=()):
    queries = []

    def query_history_k_data_plus(code, fields, start_date, end_date, frequency, adjustflag):
        queries.append((code, fields, frequency, adjustflag))
        return FakeBaostockResult(fields, daily_rows)

    monkeypatch.setitem(sys.modules, "baostock", types.SimpleNamespace(
        login=lambda: types.SimpleNamespace(error_code="0", error_msg=""),
        logout=lambda: None,
        query_history_k_data_plus=query_history_k_data_plus,
        query_trade_dates=lambda start_date, end_date: FakeBaostockResult("calendar_date,is_trading_day", calendar_rows),
    ))
    return queries


def test_fetch_reference_5m_uses_baostock_for_etf(monkeypatch):
    rows = [["2024-01-02", "20240102093500000", "1.030", "1.035", "1.028", "1.033", "120000", "123000"]]
    queries = fake_baostock(monkeypatch, daily_rows=rows)

    frame = providers.fetch_reference_5m("159611.SZ", "2024-01-01", "2024-01-05")

    assert queries[0][0] == "sz.159611"
    assert queries[0][2] == "5" and queries[0][3] == "3"
    assert frame.timestamp.tolist() == [pd.Timestamp("2024-01-02 09:35")]
    assert frame.volume.tolist() == [120000]
    assert frame.source.tolist() == ["baostock_5m"]


def fake_failing_eastmoney_daily(monkeypatch):
    def fund_etf_hist_em(**_kwargs):
        raise RuntimeError("mock EastMoney kline refusal")

    def fund_fh_em(year):
        return pd.DataFrame(columns=["基金代码", "权益登记日", "除息日期", "分红发放日", "分红"])

    def fund_cf_em(year):
        return pd.DataFrame(columns=["基金代码", "拆分折算日"])

    monkeypatch.setitem(sys.modules, "akshare", types.SimpleNamespace(
        fund_etf_hist_em=fund_etf_hist_em, fund_fh_em=fund_fh_em, fund_cf_em=fund_cf_em,
    ))


def test_fetch_etf_auxiliary_falls_back_to_baostock_daily(monkeypatch):
    providers._fund_actions.cache_clear()
    fake_failing_eastmoney_daily(monkeypatch)
    queries = fake_baostock(
        monkeypatch,
        daily_rows=[["2024-01-02", "1.030", "1.035", "1.028", "1.033", "1.030", "120000", "123000", "1"]],
        calendar_rows=[["2024-01-02", "1"], ["2024-01-03", "0"]],
    )

    daily, actions, calendar, metadata = providers.fetch_etf_auxiliary("159611.SZ", "2024-01-02", "2024-01-02")

    assert queries[0][0] == "sz.159611" and queries[0][2] == "d" and queries[0][3] == "3"
    assert daily.volume.tolist() == [120000]
    assert daily.close.tolist() == [1.033]
    assert calendar.date.tolist() == [pd.Timestamp("2024-01-02")]
    assert metadata["actions_verified"] is True
    assert any("BaoStock" in warning for warning in metadata["warnings"])


def test_fetch_etf_auxiliary_prefers_eastmoney_daily(monkeypatch):
    providers._fund_actions.cache_clear()
    eastmoney = pd.DataFrame({"日期": ["2024-01-02"], "开盘": [1.030], "收盘": [1.033], "最高": [1.035],
                              "最低": [1.028], "成交量": [1200], "成交额": [123000.0], "涨跌额": [0.003]})
    monkeypatch.setitem(sys.modules, "akshare", types.SimpleNamespace(
        fund_etf_hist_em=lambda **_kwargs: eastmoney,
        fund_fh_em=lambda year: pd.DataFrame(columns=["基金代码", "权益登记日", "除息日期", "分红发放日", "分红"]),
        fund_cf_em=lambda year: pd.DataFrame(columns=["基金代码", "拆分折算日"]),
    ))
    queries = fake_baostock(monkeypatch, calendar_rows=[["2024-01-02", "1"]])

    daily, actions, calendar, metadata = providers.fetch_etf_auxiliary("159611.SZ", "2024-01-02", "2024-01-02")

    assert queries == []
    assert daily.volume.tolist() == [120000]  # Eastmoney lots converted to units
    assert daily.preclose.tolist() == [1.030]
    assert metadata["warnings"] == []
