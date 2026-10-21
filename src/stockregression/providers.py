"""Side-effect-free adapters inspired by astock-data-toolkit (MIT).

Upstream CLI modules are deliberately never imported. All network activity is
explicit, bounded, and performed by a data job. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import threading
from functools import wraps, lru_cache
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .models import normalize_symbol
from .quality import canonical_frame
from .settings import load_settings

_BAOSTOCK_LOCK = threading.RLock()
_AKSHARE_HTTP_LOCK = threading.RLock()
_SINA_LOCAL = threading.local()
_BAOSTOCK_LOCAL = threading.local()
_THS_LOCAL = threading.local()
_MARKET_CALENDAR_CACHE: dict[tuple[int, str, str], pd.DataFrame] = {}
_SINA_MINUTE_URL = "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/CN_MarketDataService.getKLineData"
_THS_MINUTE_URL = "https://d.10jqka.com.cn/v6/line/hs_{code}/61/all.js"


def serialized_baostock(function):
    @wraps(function)
    def call(*args, **kwargs):
        with _BAOSTOCK_LOCK:
            return function(*args, **kwargs)
    return call


def _baostock_session():
    """Keep one authenticated BaoStock session per worker process."""
    import baostock as bs

    active = getattr(_BAOSTOCK_LOCAL, "module", None)
    if active is not bs or not getattr(_BAOSTOCK_LOCAL, "logged_in", False):
        with contextlib.redirect_stdout(io.StringIO()):
            login = bs.login()
        if login.error_code != "0":
            raise RuntimeError("BaoStock 登录失败：" + login.error_msg)
        _BAOSTOCK_LOCAL.module = bs
        _BAOSTOCK_LOCAL.logged_in = True
    return bs


def _baostock_reset() -> None:
    _BAOSTOCK_LOCAL.logged_in = False
    _BAOSTOCK_LOCAL.module = None
TDX_SERVERS = [("180.153.18.170", 7709), ("180.153.18.172", 80), ("202.108.253.139", 80), ("60.191.117.167", 7709)]


def _network_timeout(timeout: float | None = None) -> float:
    if timeout is not None:
        return float(timeout)
    return float(load_settings().market_data.network_timeout_seconds)


@contextlib.contextmanager
def bounded_akshare_http(timeout: float | None = None):
    """Give AKShare's network calls a bounded timeout without global process settings."""
    import requests

    timeout = _network_timeout() if timeout is None else float(timeout)
    original = requests.sessions.Session.request
    with _AKSHARE_HTTP_LOCK:
        def request_with_timeout(session, *args, **kwargs):
            kwargs.setdefault("timeout", timeout)
            return original(session, *args, **kwargs)

        requests.sessions.Session.request = request_with_timeout
        try:
            yield
        finally:
            requests.sessions.Session.request = original


def latest_closed_date():
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    day = pd.Timestamp(now.date())
    if now.hour < 16:
        day -= pd.Timedelta(days=1)
    return day


@lru_cache(maxsize=32)
@serialized_baostock
def fetch_latest_trading_date(as_of: str):
    """Resolve a closed calendar date through the exchange calendar, not weekdays."""
    end = pd.Timestamp(as_of)
    try:
        bs = _baostock_session()
        result = bs.query_trade_dates(str((end - pd.Timedelta(days=40)).date()), str(end.date()))
        if result.error_code != "0":
            raise RuntimeError("交易日历查询失败：" + result.error_msg)
        dates = []
        while result.next():
            row = dict(zip(result.fields, result.get_row_data()))
            if row.get("is_trading_day") == "1":
                dates.append(pd.Timestamp(row["calendar_date"]))
        if not dates:
            raise RuntimeError("最近 40 天未返回有效交易日")
        return max(dates)
    except Exception:
        _baostock_reset()
        raise


def fetch_instruments(
    _legacy_revision: str | None = None,
    *,
    network_timeout: float | None = None,
) -> list[dict]:
    """Refresh names from AKShare; history comes exclusively from the local package."""
    from .rules import instrument_terms
    records, failures = {}, []
    try:
        import akshare as ak
        with bounded_akshare_http(network_timeout):
            names = ak.stock_info_a_code_name()
        for item in names.to_dict("records"):
            try:
                symbol = normalize_symbol(str(item["code"]).zfill(6))
                records[symbol] = {**records.get(symbol, {}), "symbol": symbol, "name": item["name"], "asset_type": "stock"}
            except ValueError:
                continue
    except Exception as exc:
        failures.append(f"最新股票清单: {exc}")
    try:
        import akshare as ak
        with bounded_akshare_http(network_timeout):
            etf_frame = ak.fund_etf_category_sina(symbol="ETF基金")
        for item in etf_frame.to_dict("records"):
            symbol = normalize_symbol(str(item["代码"]))
            records[symbol] = {**records.get(symbol, {}), "symbol": symbol, "name": item["名称"], "asset_type": "etf"}
    except Exception as exc:
        failures.append(f"新浪 ETF 清单不可用: {exc}")
    if not records:
        raise RuntimeError("证券清单获取失败：" + "；".join(failures))
    return [instrument_terms(symbol, {**item, "catalog_warnings": failures}) for symbol, item in sorted(records.items())]


def fetch_tdx(
    symbol: str,
    earliest=None,
    cancelled=lambda: False,
    network_timeout: float | None = None,
) -> pd.DataFrame:
    from pytdx.hq import TdxHq_API
    code, exchange = normalize_symbol(symbol).split(".")
    market = 1 if exchange == "SH" else 0
    failures = []
    for host, port in TDX_SERVERS:
        if cancelled():
            raise InterruptedError("任务已取消")
        api = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=True)
        try:
            if not api.connect(host, port, time_out=int(max(1, round(_network_timeout(network_timeout))))):
                failures.append(f"{host}:{port}连接失败")
                continue
            bars = []
            for offset in range(0, 24800, 800):
                if cancelled():
                    raise InterruptedError("任务已取消")
                batch = api.get_security_bars(8, market, code, offset, 800)
                if not batch:
                    break
                bars.extend(batch)
                if earliest is not None and pd.Timestamp(min(r["datetime"] for r in batch)) < pd.Timestamp(earliest):
                    break
            if not bars:
                raise RuntimeError("通达信没有返回分钟数据")
            frame = canonical_frame(pd.DataFrame(bars), source="tdx")
            closed = latest_closed_date()
            return frame[frame.timestamp < closed + pd.Timedelta(days=1)].copy()
        except InterruptedError:
            raise
        except Exception as exc:
            original = getattr(exc, "original_exception", None)
            detail = f"{type(exc).__name__}: {exc}"
            if original is not None:
                detail += f"；{type(original).__name__}: {original}"
            failures.append(f"{host}:{port} {detail}")
        finally:
            api.disconnect()
    raise RuntimeError("通达信节点不可用：" + "；".join(failures))


def fetch_sina(symbol: str, network_timeout: float | None = None) -> pd.DataFrame:
    code, exchange = normalize_symbol(symbol).split(".")
    # AKShare's unadjusted helper performs extra adjusted-daily requests. Use
    # the documented Sina payload directly so one security costs one request.
    if network_timeout is None:
        frame = _fetch_sina_minute(exchange.lower() + code)
    else:
        frame = _fetch_sina_minute(
            exchange.lower() + code, network_timeout=network_timeout,
        )
    if frame is None or frame.empty:
        raise RuntimeError("新浪未返回近期分钟行情")
    frame = canonical_frame(frame, source="sina")
    return frame[frame.timestamp < latest_closed_date() + pd.Timedelta(days=1)].copy()


def _sina_session():
    import requests

    session = getattr(_SINA_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 stockregression/0.1",
            "Accept": "text/javascript,application/json,*/*",
        })
        _SINA_LOCAL.session = session
    return session


def _decode_sina_payload(text: str) -> list[dict]:
    match = re.search(r"=\s*\((.*)\)\s*;?\s*$", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("新浪分钟响应不是记录列表")
    return payload


def _fetch_sina_minute(sina_symbol: str, network_timeout: float | None = None) -> pd.DataFrame:
    response = _sina_session().get(
        _SINA_MINUTE_URL,
        params={"symbol": sina_symbol, "scale": "1", "ma": "no", "datalen": "1970"},
        timeout=_network_timeout(network_timeout),
    )
    response.raise_for_status()
    payload = _decode_sina_payload(response.text)
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    return frame.iloc[:, :7].copy()


def _ths_session():
    import requests

    session = getattr(_THS_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "text/javascript,application/json,*/*",
            "Referer": "https://stockpage.10jqka.com.cn/",
        })
        _THS_LOCAL.session = session
    return session


def _decode_ths_payload(text: str) -> dict:
    match = re.search(r"\((.*)\)\s*;?\s*$", text, flags=re.DOTALL)
    if not match:
        raise ValueError("同花顺分钟响应不是 JSONP")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("同花顺分钟响应不是记录对象")
    return payload


def _ths_timestamps(dates: list[str], start: str | None) -> list[pd.Timestamp]:
    """Resolve MMDDHHMM labels with the year supplied by the payload start."""
    year = None
    if start and len(start) >= 4 and str(start)[:4].isdigit():
        year = int(str(start)[:4])
    if year is None:
        year = int(datetime.now(ZoneInfo("Asia/Shanghai")).year)
    output = []
    previous_month = None
    for value in dates:
        raw = str(value).strip()
        if len(raw) != 8 or not raw.isdigit():
            raise ValueError("同花顺分钟时间标签格式无效：" + raw)
        month = int(raw[:2])
        if previous_month is not None and month < previous_month - 6:
            year += 1
        previous_month = month
        output.append(pd.Timestamp(
            year=year, month=month, day=int(raw[2:4]),
            hour=int(raw[4:6]), minute=int(raw[6:8]),
        ))
    return output


def fetch_ths(
    symbol: str,
    network_timeout: float | None = None,
    *,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read THS's rolling 1-minute feed as an independent fallback.

    THS publishes low/open-low/high-low/close-low tuples and share volume but
    no per-minute turnover. Turnover is estimated from volume and OHLC4, then
    scaled to the official daily turnover when local daily bars are supplied.
    The estimate never changes OHLC or volume and is not presented as a vendor
    amount field.
    """
    code, _exchange = normalize_symbol(symbol).split(".")
    response = _ths_session().get(
        _THS_MINUTE_URL.format(code=code),
        timeout=_network_timeout(network_timeout),
    )
    response.raise_for_status()
    payload = _decode_ths_payload(response.text)
    dates = [value for value in str(payload.get("dates") or "").split(",") if value]
    price_parts = [value for value in str(payload.get("price") or "").split(",") if value]
    volume_parts = [value for value in str(payload.get("volumn") or "").split(",") if value]
    if not dates:
        raise RuntimeError("同花顺未返回近期分钟行情")
    if len(price_parts) != len(dates) * 4 or len(volume_parts) != len(dates):
        raise ValueError(
            "同花顺分钟字段长度不一致："
            f"{len(dates)} 根、{len(price_parts)} 个价格字段、{len(volume_parts)} 个成交量"
        )
    factor = float(payload.get("priceFactor") or 1)
    if not factor > 0:
        raise ValueError("同花顺分钟 priceFactor 无效")
    prices = [float(value) for value in price_parts]
    volumes = [int(round(float(value))) for value in volume_parts]
    rows = []
    for index, stamp in enumerate(_ths_timestamps(dates, payload.get("start"))):
        low_raw, open_raw, high_raw, close_raw = prices[index * 4:index * 4 + 4]
        low = low_raw / factor
        opening = (low_raw + open_raw) / factor
        high = (low_raw + high_raw) / factor
        close = (low_raw + close_raw) / factor
        volume = volumes[index]
        rows.append({
            "timestamp": stamp, "open": opening, "high": high,
            "low": low, "close": close, "volume": volume,
            "turnover": volume * (opening + high + low + close) / 4.0,
        })
    frame = canonical_frame(pd.DataFrame(rows), source="ths")
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai")).date())
    verifiable = (
        daily is not None and not daily.empty
        and ("date" in daily.columns or "timestamp" in daily.columns)
        and {"open", "high", "low", "close", "volume", "turnover"}.issubset(daily.columns)
    )
    if not verifiable:
        # Without an official daily bar, a same-day session may still be in
        # progress in the vendor's rolling feed. Wait for the next sync.
        frame = frame[frame.timestamp < today].copy()
    else:
        day_column = "date" if "date" in daily else "timestamp"
        reference = daily.copy()
        reference["day"] = pd.to_datetime(reference[day_column], errors="coerce").dt.normalize()
        reference["turnover"] = pd.to_numeric(reference["turnover"], errors="coerce")
        reference = reference.dropna(subset=["day"]).drop_duplicates("day", keep="last").set_index("day")
        frame["day"] = frame.timestamp.dt.normalize()
        estimated = frame.groupby("day").turnover.transform("sum")
        official = frame.day.map(reference.turnover)
        usable = estimated.gt(0) & official.gt(0)
        frame.loc[usable, "turnover"] = (
            frame.loc[usable, "turnover"] * (official[usable] / estimated[usable])
        )
        if today in set(frame.day):
            # The rolling feed can expose a partial current session after the
            # local 16:00 cutoff. Keep it only when the official daily bar
            # already matches its volume and OHLC; otherwise the next sync will
            # pick it up after the vendor finalizes the day.
            part = frame[frame.day == today]
            row = reference.loc[today] if today in reference.index else None
            complete = False
            if row is not None and all(name in reference for name in ("open", "high", "low", "close", "volume")):
                daily_volume = pd.to_numeric(row.get("volume"), errors="coerce")
                volume_ok = bool(pd.notna(daily_volume) and float(daily_volume) > 0 and abs(
                    float(part.volume.sum()) / float(daily_volume) - 1
                ) <= 0.01)
                close_ok = abs(float(part.close.iloc[-1]) - float(row.get("close"))) <= 0.0011
                high_ok = abs(float(part.high.max()) - float(row.get("high"))) <= 0.0011
                low_ok = abs(float(part.low.min()) - float(row.get("low"))) <= 0.0011
                complete = volume_ok and close_ok and high_ok and low_ok
            if not complete:
                frame = frame[frame.day != today].copy()
        frame = frame.drop(columns=["day"])
    frame = frame[frame.timestamp < latest_closed_date() + pd.Timedelta(days=1)].copy()
    if frame.empty:
        raise RuntimeError("同花顺未返回已收盘的分钟行情")
    return frame


@serialized_baostock
def fetch_reference_5m(symbol: str, start, end) -> pd.DataFrame:
    # BaoStock serves unadjusted 5-minute bars for A shares and listed ETFs
    # alike, and is independent of the Sina minute feed used for recent bars.
    code, exchange = normalize_symbol(symbol).split(".")
    try:
        bs = _baostock_session()
        result = bs.query_history_k_data_plus(exchange.lower() + "." + code,
            "date,time,open,high,low,close,volume,amount", start_date=str(pd.Timestamp(start).date()),
            end_date=str(pd.Timestamp(end).date()), frequency="5", adjustflag="3")
        if result.error_code != "0":
            raise RuntimeError("BaoStock 参考分钟失败：" + result.error_msg)
        output = []
        while result.next():
            output.append(result.get_row_data())
        frame = pd.DataFrame(output, columns=result.fields)
        if frame.empty:
            raise RuntimeError("没有独立5分钟参考数据")
        frame["timestamp"] = pd.to_datetime(frame.time, format="%Y%m%d%H%M%S%f")
        frame = frame.drop(columns=["time", "date"])
        return canonical_frame(frame, source="baostock_5m")
    except Exception:
        _baostock_reset()
        raise


@serialized_baostock
def fetch_stock_daily(symbol: str, start, end, cancelled=lambda: False):
    code, exchange = normalize_symbol(symbol).split(".")
    bs_code = exchange.lower() + "." + code
    start, end = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())

    def rows(result):
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock {result.error_code}: {result.error_msg}")
        output = []
        while result.next():
            if cancelled():
                raise InterruptedError("任务已取消")
            output.append(result.get_row_data())
        return pd.DataFrame(output, columns=result.fields)

    try:
        bs = _baostock_session()
        basic = rows(bs.query_stock_basic(code=bs_code))
        metadata = {}
        if not basic.empty:
            row = basic.iloc[0]
            metadata = {"name": row.get("code_name", symbol), "listing_date": row.get("ipoDate") or None, "delisting_date": row.get("outDate") or None}
            if metadata.get("listing_date"):
                start = max(start, str(metadata["listing_date"]))
        daily = rows(bs.query_history_k_data_plus(bs_code,
            "date,open,high,low,close,preclose,volume,amount,tradestatus,isST",
            start_date=start, end_date=end, frequency="d", adjustflag="3"))
        if daily.empty:
            raise RuntimeError("BaoStock 未返回所选股票的日线")
        daily = canonical_frame(daily, daily=True)
        return daily, metadata
    except Exception:
        _baostock_reset()
        raise


@serialized_baostock
def fetch_etf_daily(
    symbol: str,
    start,
    end,
    source: str = "baostock",
    cancelled=lambda: False,
    network_timeout: float | None = None,
) -> pd.DataFrame:
    code, exchange = normalize_symbol(symbol).split(".")
    if source == "eastmoney":
        import akshare as ak
        with bounded_akshare_http(network_timeout):
            frame = ak.fund_etf_hist_em(
                symbol=code, period="daily",
                start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                end_date=pd.Timestamp(end).strftime("%Y%m%d"), adjust="",
            )
        if frame is None or frame.empty:
            raise RuntimeError("东方财富 ETF 日线未返回所选区间数据")
        if "涨跌额" in frame:
            frame["preclose"] = (pd.to_numeric(frame["收盘"]) - pd.to_numeric(frame["涨跌额"])).round(3)
        daily = canonical_frame(frame, daily=True)
        daily["volume"] *= 100  # Eastmoney reports lots.
        return daily
    if source != "baostock":
        raise ValueError("未知 ETF 日线来源：" + source)
    try:
        bs = _baostock_session()
        result = bs.query_history_k_data_plus(
            exchange.lower() + "." + code,
            "date,open,high,low,close,preclose,volume,amount,tradestatus",
            start_date=str(pd.Timestamp(start).date()), end_date=str(pd.Timestamp(end).date()),
            frequency="d", adjustflag="3",
        )
        if result.error_code != "0":
            raise RuntimeError("BaoStock ETF 日线失败：" + result.error_msg)
        rows = []
        while result.next():
            if cancelled():
                raise InterruptedError("任务已取消")
            rows.append(result.get_row_data())
    except Exception:
        _baostock_reset()
        raise
    if not rows:
        raise RuntimeError("BaoStock ETF 日线未返回所选区间数据")
    return canonical_frame(pd.DataFrame(rows, columns=result.fields), daily=True)


def fetch_auxiliary(
    symbol: str,
    start,
    end,
    cancelled=lambda: False,
    network_timeout: float | None = None,
):
    """Compatibility adapter for callers that need all auxiliary files at once."""
    if symbol.startswith(("1", "5")):
        daily = fetch_etf_daily(
            symbol, start, end, "baostock", cancelled,
            network_timeout=network_timeout,
        )
        calendar = fetch_market_calendar(start, end, cancelled)
        actions, metadata = fetch_market_actions(
            symbol, start, end, cancelled, network_timeout=network_timeout,
        )
        return daily, actions, calendar, metadata
    daily, metadata = fetch_stock_daily(symbol, start, end, cancelled)
    calendar = fetch_market_calendar(start, end, cancelled)
    actions, action_metadata = fetch_market_actions(
        symbol, start, end, cancelled, network_timeout=network_timeout,
    )
    metadata = {**metadata, **action_metadata}
    metadata["actions_verified"] = bool(action_metadata.get("actions_verified")) and not action_metadata.get("actions_unverified_years")
    return daily, actions, calendar, metadata


@lru_cache(maxsize=48)
def _fund_actions(year: int, as_of: str, network_timeout: float | None = None):
    import akshare as ak
    with bounded_akshare_http(network_timeout):
        return ak.fund_fh_em(year=str(year)), ak.fund_cf_em(year=str(year))


def fetch_etf_auxiliary(
    symbol,
    start,
    end,
    cancelled=lambda: False,
    daily_source="baostock",
    network_timeout: float | None = None,
):
    code, exchange = normalize_symbol(symbol).split(".")
    daily_notes = []
    try:
        daily = fetch_etf_daily(
            symbol, start, end, daily_source, cancelled,
            network_timeout=network_timeout,
        )
    except Exception as exc:
        if daily_source == "baostock":
            raise
        daily = fetch_etf_daily(
            symbol, start, end, "baostock", cancelled,
            network_timeout=network_timeout,
        )
        daily_notes.append("东方财富 ETF 日线不可用，已改用 BaoStock：" + str(exc))
    calendar = fetch_market_calendar(start, end, cancelled)
    records, warnings, unverified = [], [], []
    for year in range(daily.date.min().year, pd.Timestamp(end).year + 1):
        if cancelled():
            raise InterruptedError("任务已取消")
        try:
            dividends, splits = _fund_actions(
                year, str(pd.Timestamp(end).date()), network_timeout,
            )
            for row in dividends.loc[dividends["基金代码"].astype(str).str.zfill(6) == code].to_dict("records"):
                records.append({"record_date": row["权益登记日"], "ex_date": row["除息日期"], "pay_date": row["分红发放日"], "cash_per_share": row["分红"], "bonus_ratio": 0})
            for row in splits.loc[splits["基金代码"].astype(str).str.zfill(6) == code].to_dict("records"):
                records.append({"record_date": row["拆分折算日"], "ex_date": row["拆分折算日"], "stock_date": row["拆分折算日"], "cash_per_share": 0, "bonus_ratio": 0, "unsupported_reason": "ETF 拆分资料缺少登记日及精确份额处理，请导入核实后的公司行动"})
        except Exception as exc:
            warnings.append(f"{year} ETF 分红/拆分资料未核验：{exc}")
            unverified.append(year)
    metadata = {"actions_verified": not warnings, "actions_checked_years": list(range(daily.date.min().year, pd.Timestamp(end).year + 1)), "actions_unverified_years": unverified, "warnings": [*daily_notes, *warnings]}
    return daily, normalize_actions(pd.DataFrame(records)), calendar, metadata


@serialized_baostock
def fetch_market_calendar(start, end, cancelled=lambda: False) -> pd.DataFrame:
    """Fetch the exchange calendar without requesting a security's daily bars."""
    try:
        bs = _baostock_session()
    except Exception:
        _baostock_reset()
        raise
    cache_key = (
        id(bs),
        str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()),
    )
    cached = _MARKET_CALENDAR_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()
    try:
        result = bs.query_trade_dates(
            start_date=str(pd.Timestamp(start).date()),
            end_date=str(pd.Timestamp(end).date()),
        )
        if result.error_code != "0":
            raise RuntimeError("交易日历失败：" + result.error_msg)
        records = []
        while result.next():
            if cancelled():
                raise InterruptedError("任务已取消")
            records.append(result.get_row_data())
    except Exception:
        _baostock_reset()
        raise
    frame = pd.DataFrame(records, columns=result.fields)
    if frame.empty:
        return pd.DataFrame({"date": pd.to_datetime([])})
    result = pd.DataFrame({
        "date": pd.to_datetime(
            frame.loc[frame.is_trading_day == "1", "calendar_date"]
        )
    })
    if len(_MARKET_CALENDAR_CACHE) >= 64:
        _MARKET_CALENDAR_CACHE.clear()
    _MARKET_CALENDAR_CACHE[cache_key] = result.copy()
    return result


@serialized_baostock
def fetch_market_actions(
    symbol,
    start,
    end,
    cancelled=lambda: False,
    network_timeout: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch only corporate/fund actions for an already selected date range."""
    symbol = normalize_symbol(symbol)
    code, exchange = symbol.split(".")
    years = list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    records = []
    warnings = []
    if symbol.startswith(("1", "5")):
        for year in years:
            if cancelled():
                raise InterruptedError("任务已取消")
            try:
                dividends, splits = _fund_actions(
                    year, str(pd.Timestamp(end).date()), network_timeout,
                )
                for row in dividends.loc[
                    dividends["基金代码"].astype(str).str.zfill(6) == code
                ].to_dict("records"):
                    records.append({
                        "record_date": row["权益登记日"], "ex_date": row["除息日期"],
                        "pay_date": row["分红发放日"], "cash_per_share": row["分红"],
                        "bonus_ratio": 0,
                    })
                for row in splits.loc[
                    splits["基金代码"].astype(str).str.zfill(6) == code
                ].to_dict("records"):
                    records.append({
                        "record_date": row["拆分折算日"], "ex_date": row["拆分折算日"],
                        "stock_date": row["拆分折算日"], "cash_per_share": 0,
                        "bonus_ratio": 0,
                        "unsupported_reason": "ETF 拆分资料缺少登记日及精确份额处理，请导入核实后的公司行动",
                    })
            except Exception as exc:
                warnings.append(f"{year} ETF 分红/拆分资料未核验：{exc}")
    else:
        bs_code = exchange.lower() + "." + code
        try:
            bs = _baostock_session()
            for year in years:
                if cancelled():
                    raise InterruptedError("任务已取消")
                result = bs.query_dividend_data(
                    bs_code, year=str(year), yearType="operate"
                )
                if result.error_code != "0":
                    warnings.append(f"{year} 公司行动失败：{result.error_msg}")
                    continue
                while result.next():
                    records.append(result.get_row_data())
        except Exception:
            _baostock_reset()
            raise
    metadata = {
        "actions_verified": not warnings,
        "actions_checked_years": years,
        "actions_unverified_years": [
            year for year in years
            if any(str(year) in warning for warning in warnings)
        ],
        "warnings": warnings,
    }
    return normalize_actions(pd.DataFrame(records)), metadata


def normalize_actions(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {"dividRegistDate": "record_date", "dividOperateDate": "ex_date", "dividPayDate": "pay_date",
              "dividStockMarketDate": "stock_date", "dividCashPsBeforeTax": "cash_per_share"}
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame and v not in frame}).copy()
    columns = ["record_date", "ex_date", "pay_date", "stock_date", "cash_per_share", "bonus_ratio", "implemented", "unsupported_reason"]
    for name in columns[:4]:
        frame[name] = pd.to_datetime(frame.get(name, pd.Series(index=frame.index, dtype="object")), errors="coerce")
    if "cash_per_share" not in frame:
        frame["cash_per_share"] = 0.0
    frame["cash_per_share"] = pd.to_numeric(frame.cash_per_share, errors="coerce").fillna(0)
    if "bonus_ratio" not in frame:
        frame["bonus_ratio"] = pd.to_numeric(frame.get("dividStocksPs", 0), errors="coerce") + pd.to_numeric(frame.get("dividReserveToStockPs", 0), errors="coerce")
    frame["bonus_ratio"] = pd.to_numeric(frame.bonus_ratio, errors="coerce").fillna(0)
    frame["implemented"] = frame.get("implemented", frame.ex_date.notna())
    if "unsupported_reason" not in frame:
        frame["unsupported_reason"] = ""
    description = frame.get("dividCashStock", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame.loc[description.str.contains("配股|合并|拆股", regex=True), "unsupported_reason"] = "存在首版未建模的特殊公司行动"
    return frame[columns].drop_duplicates(["ex_date", "cash_per_share", "bonus_ratio"]).sort_values("ex_date").reset_index(drop=True)
