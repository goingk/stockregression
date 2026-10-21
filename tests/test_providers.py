from __future__ import annotations

import json
import struct
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


def fake_sina_minute(monkeypatch, frame):
    calls = []

    def fetch(symbol):
        calls.append(symbol)
        return frame

    monkeypatch.setattr(providers, "_fetch_sina_minute", fetch)
    return calls


def test_fetch_sina_uses_sina_minute_feed_for_etf(monkeypatch):
    calls = fake_sina_minute(monkeypatch, sina_frame())

    frame = providers.fetch_sina("159611.SZ")

    assert calls == ["sz159611"]
    assert frame.timestamp.tolist() == list(pd.date_range("2024-01-02 09:31", periods=3, freq="min"))
    assert (frame.open > 0).all()
    assert frame.source.tolist() == ["sina"] * 3


def test_fetch_sina_rejects_non_positive_prices(monkeypatch):
    fake_sina_minute(monkeypatch, sina_frame([0.0, 10.1, 10.2]))

    with pytest.raises(DataQualityError, match="非正价格"):
        providers.fetch_sina("159611.SZ")


def test_sina_minute_adapter_reuses_session_and_makes_one_request(monkeypatch):
    calls = []

    class Response:
        text = 'var _=(([{"day":"2026-10-16 15:00:00","open":"1.0","high":"1.1","low":"0.9","close":"1.05","volume":"100","amount":"105"}]));'

        def raise_for_status(self):
            return None

    class Session:
        headers = {}

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    session = Session()
    monkeypatch.setattr(providers, "_SINA_LOCAL", types.SimpleNamespace(session=session))
    monkeypatch.setattr(providers, "latest_closed_date", lambda: pd.Timestamp("2026-10-19"))

    first = providers.fetch_sina("159611.SZ")
    second = providers.fetch_sina("159611.SZ")

    assert len(calls) == 2
    assert calls[0][0] == providers._SINA_MINUTE_URL
    assert calls[0][1]["params"] == {"symbol": "sz159611", "scale": "1", "ma": "no", "datalen": "1970"}
    assert calls[0][1]["timeout"] == 15
    assert len(first) == len(second) == 1


def test_fetch_ths_decodes_prices_volume_and_estimates_turnover(monkeypatch):
    payload = {
        "total": "3",
        "start": "202610080930",
        "priceFactor": 1000,
        "dates": "10080930,10080931,10081500",
        "price": "1824,16,21,11,1826,9,20,18,1820,1,5,3",
        "volumn": "1000,2000,3000",
    }
    calls = []

    class Response:
        text = "quotebridge_v6_line_hs_517520_61_all(" + json.dumps(payload) + ")"

        def raise_for_status(self):
            return None

    class Session:
        headers = {}

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    monkeypatch.setattr(providers, "_THS_LOCAL", types.SimpleNamespace(session=Session()))
    monkeypatch.setattr(providers, "latest_closed_date", lambda: pd.Timestamp("2026-10-20"))
    daily = pd.DataFrame({
        "date": [pd.Timestamp("2026-10-08")],
        "open": [1.840], "high": [1.846], "low": [1.820], "close": [1.823],
        "volume": [6000], "turnover": [12000.0],
    })

    frame = providers.fetch_ths("517520.SH", network_timeout=7, daily=daily)

    assert calls[0][0] == "https://d.10jqka.com.cn/v6/line/hs_517520/61/all.js"
    assert calls[0][1]["timeout"] == 7
    assert frame.timestamp.tolist() == [
        pd.Timestamp("2026-10-08 09:30"),
        pd.Timestamp("2026-10-08 09:31"),
        pd.Timestamp("2026-10-08 15:00"),
    ]
    assert frame.iloc[0][["open", "high", "low", "close"]].tolist() == [1.84, 1.845, 1.824, 1.835]
    assert frame.volume.tolist() == [1000, 2000, 3000]
    assert abs(frame.turnover.sum() - 12000.0) < 1e-9
    assert frame.source.tolist() == ["ths"] * 3


def test_ths_timestamps_roll_over_new_year():
    stamps = providers._ths_timestamps(["12311459", "01020931"], "202512310930")

    assert stamps == [pd.Timestamp("2025-12-31 14:59"), pd.Timestamp("2026-01-02 09:31")]


def test_fetch_ths_drops_an_incomplete_current_session(monkeypatch):
    payload = {
        "total": "6",
        "start": "202610200930",
        "priceFactor": 1000,
        "dates": "10200930,10200931,10201500,10210930,10210931,10211500",
        "price": "1824,16,21,11,1826,9,20,18,1820,1,5,3,1800,10,30,0,1801,9,29,1,1802,8,28,2",
        "volumn": "1000,2000,3000,500,600,700",
    }

    class Response:
        text = "quotebridge_v6_line_hs_517520_61_all(" + json.dumps(payload) + ")"

        def raise_for_status(self):
            return None

    class Session:
        headers = {}

        def get(self, url, **kwargs):
            return Response()

    class FixedDateTime:
        @classmethod
        def now(cls, tz):
            return pd.Timestamp("2026-10-21 17:30", tz="Asia/Shanghai")

    monkeypatch.setattr(providers, "_THS_LOCAL", types.SimpleNamespace(session=Session()))
    monkeypatch.setattr(providers, "latest_closed_date", lambda: pd.Timestamp("2026-10-21"))
    monkeypatch.setattr(providers, "datetime", FixedDateTime)
    daily = pd.DataFrame({
        "date": [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-21")],
        "open": [1.824, 1.810], "high": [1.845, 1.900], "low": [1.820, 1.800],
        "close": [1.823, 1.900], "volume": [6000, 5000], "turnover": [12000.0, 9000.0],
    })

    frame = providers.fetch_ths("517520.SH", daily=daily)

    assert frame.timestamp.dt.normalize().unique().tolist() == [pd.Timestamp("2026-10-20")]


def test_baostock_session_is_reused_within_worker(monkeypatch):
    calls = {"login": 0, "logout": 0}

    def login():
        calls["login"] += 1
        return types.SimpleNamespace(error_code="0", error_msg="")

    def logout():
        calls["logout"] += 1

    basic_fields = "code,code_name,ipoDate,outDate"
    daily_fields = "date,open,high,low,close,preclose,volume,amount,tradestatus,isST"
    basic_rows = [["sh.600000", "浦发银行", "1999-11-10", ""]]
    daily_rows = [["2026-10-16", "10", "10.2", "9.9", "10.1", "10", "100", "1000", "1", "0"]]
    monkeypatch.setitem(sys.modules, "baostock", types.SimpleNamespace(
        login=login,
        logout=logout,
        query_stock_basic=lambda **kwargs: FakeBaostockResult(basic_fields, basic_rows),
        query_history_k_data_plus=lambda *args, **kwargs: FakeBaostockResult(daily_fields, daily_rows),
    ))
    monkeypatch.setattr(providers, "_BAOSTOCK_LOCAL", types.SimpleNamespace())

    providers.fetch_stock_daily("600000.SH", "2026-10-01", "2026-10-16")
    providers.fetch_stock_daily("600000.SH", "2026-10-01", "2026-10-16")

    assert calls == {"login": 1, "logout": 0}


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


def test_tdx_protocol_error_preserves_original_parser_message(monkeypatch):
    from pytdx.errors import TdxFunctionCallError

    class FakeApi:
        def __init__(self, **_kwargs):
            pass

        def connect(self, *args, **kwargs):
            return True

        def get_security_bars(self, *args):
            error = TdxFunctionCallError("calling function error")
            error.original_exception = struct.error("unpack requires a buffer of 4 bytes")
            raise error

        def disconnect(self):
            pass

    monkeypatch.setattr("pytdx.hq.TdxHq_API", FakeApi)
    with pytest.raises(RuntimeError, match="unpack requires a buffer of 4 bytes"):
        providers.fetch_tdx("600000.SH")


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
    assert not any("东方财富" in warning for warning in metadata["warnings"])


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

    daily, actions, calendar, metadata = providers.fetch_etf_auxiliary(
        "159611.SZ", "2024-01-02", "2024-01-02", daily_source="eastmoney"
    )

    assert queries == []
    assert daily.volume.tolist() == [120000]  # Eastmoney lots converted to units
    assert daily.preclose.tolist() == [1.030]
    assert metadata["warnings"] == []
