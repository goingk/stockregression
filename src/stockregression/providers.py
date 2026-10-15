"""Side-effect-free adapters inspired by astock-data-toolkit (MIT).

Upstream CLI modules are deliberately never imported. All network activity is
explicit, bounded, and performed by a data job. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import contextlib
import io
import time
import threading
from functools import wraps, lru_cache
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from .models import normalize_symbol
from .quality import canonical_frame

HF_REPO = "neigezhu/china-a-share-1min-ohlcv"
ETF_HF_REPO = "neigezhu/china-etf-1min-ohlcv"
_BAOSTOCK_LOCK = threading.RLock()


def serialized_baostock(function):
    @wraps(function)
    def call(*args, **kwargs):
        with _BAOSTOCK_LOCK:
            return function(*args, **kwargs)
    return call
TDX_SERVERS = [("180.153.18.170", 7709), ("180.153.18.172", 80), ("202.108.253.139", 80), ("60.191.117.167", 7709)]


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
    import baostock as bs
    end = pd.Timestamp(as_of)
    with contextlib.redirect_stdout(io.StringIO()):
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError("交易日历登录失败：" + login.error_msg)
        try:
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
        finally:
            bs.logout()


def fetch_hf(symbol: str, root: Path, revision: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    code, exchange = normalize_symbol(symbol).split(".")
    etf = code.startswith(("1", "5"))
    if etf:
        from .settings import load_settings
        revision = load_settings().etf_hf_revision
    repo = ETF_HF_REPO if etf else HF_REPO
    inventory = fetch_history_instruments(revision, repo)
    if symbol not in {item["symbol"] for item in inventory}:
        raise RuntimeError("固定历史库未覆盖此标的，将使用近期分钟来源并明确报告历史缺口")
    filename = f"data/{'etf' if etf else 'stock'}_1m/{exchange}/{code}.parquet"
    path = hf_hub_download(repo_id=ETF_HF_REPO if etf else HF_REPO, repo_type="dataset", filename=filename,
                           revision=revision, cache_dir=str(root / "raw" / "huggingface"))
    return canonical_frame(pd.read_parquet(path), source="hf")


@lru_cache(maxsize=4)
def fetch_history_instruments(revision: str, repo: str = HF_REPO) -> list[dict]:
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/metadata/coverage_by_instrument.csv"
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text), dtype=str)
    records = []
    for _, row in frame.iterrows():
        code = row.get("symbol", row.get("stock_code", ""))
        exchange = row.get("exchange", "")
        if exchange not in ("SH", "SZ"):
            continue
        try:
            symbol = normalize_symbol(f"{code}.{exchange}")
        except ValueError:
            continue
        records.append({"symbol": symbol, "name": str(row.get("name", row.get("stock_name", symbol))),
                        "listing_date": row.get("listing_date") if pd.notna(row.get("listing_date")) else None, "delisting_date": row.get("delisting_date") if pd.notna(row.get("delisting_date")) else None})
    return records


def estimate_history_bytes(symbols, settings):
    from huggingface_hub import HfApi
    wanted = set(symbols)
    total = 0
    for repo, revision, folder in [(HF_REPO, settings.hf_revision, "stock_1m"), (ETF_HF_REPO, settings.etf_hf_revision, "etf_1m")]:
        info = HfApi().dataset_info(repo, revision=revision, files_metadata=True, timeout=15)
        for file in info.siblings:
            parts = file.rfilename.split("/")
            if len(parts) == 4 and parts[0] == "data" and parts[1] == folder and f"{parts[3].removesuffix('.parquet')}.{parts[2]}" in wanted:
                total += file.size or 0
    return total


def fetch_instruments(revision: str) -> list[dict]:
    from .rules import instrument_terms
    from .settings import load_settings
    records, failures = {}, []
    for repo, pin in [(HF_REPO, revision), (ETF_HF_REPO, load_settings().etf_hf_revision)]:
        try:
            for item in fetch_history_instruments(pin, repo):
                records[item["symbol"]] = item
        except Exception as exc:
            failures.append(f"{repo}: {exc}")
    try:
        import akshare as ak
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
        for item in ak.fund_etf_spot_em().to_dict("records"):
            symbol = normalize_symbol(str(item["代码"]).zfill(6))
            records[symbol] = {**records.get(symbol, {}), "symbol": symbol, "name": item["名称"], "asset_type": "etf"}
    except Exception as exc:
        failures.append(f"最新 ETF 清单: {exc}")
        try:
            import akshare as ak
            for item in ak.fund_etf_category_sina(symbol="ETF基金").to_dict("records"):
                symbol = normalize_symbol(str(item["代码"]))
                records[symbol] = {**records.get(symbol, {}), "symbol": symbol, "name": item["名称"], "asset_type": "etf"}
            failures.append("ETF 清单已使用新浪备源；历史覆盖仍按逐标的报告核对")
        except Exception as backup_error:
            failures.append(f"ETF 清单备源也不可用，本次证券范围可能不完整: {backup_error}")
    if not records:
        raise RuntimeError("证券清单获取失败：" + "；".join(failures))
    return [instrument_terms(symbol, {**item, "catalog_warnings": failures}) for symbol, item in sorted(records.items())]


def fetch_tdx(symbol: str, earliest=None, cancelled=lambda: False) -> pd.DataFrame:
    from pytdx.hq import TdxHq_API
    code, exchange = normalize_symbol(symbol).split(".")
    market = 1 if exchange == "SH" else 0
    failures = []
    for host, port in TDX_SERVERS:
        if cancelled():
            raise InterruptedError("任务已取消")
        api = TdxHq_API(heartbeat=False, auto_retry=False)
        try:
            if not api.connect(host, port, time_out=4):
                failures.append(f"{host}:{port}连接失败")
                continue
            bars = []
            for offset in range(0, 24800, 800):
                if cancelled():
                    raise InterruptedError("任务已取消")
                batch = None
                for attempt in range(3):
                    try:
                        batch = api.get_security_bars(8, market, code, offset, 800)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(0.3 * (attempt + 1))
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
            failures.append(f"{host}:{port} {exc}")
        finally:
            api.disconnect()
    raise RuntimeError("通达信节点不可用：" + "；".join(failures))


def fetch_sina(symbol: str) -> pd.DataFrame:
    import akshare as ak
    code, exchange = normalize_symbol(symbol).split(".")
    if code.startswith(("1", "5")):
        frame = ak.fund_etf_hist_min_em(symbol=code, period="1", adjust="")
    else:
        frame = ak.stock_zh_a_minute(symbol=exchange.lower() + code, period="1", adjust="")
    if frame is None or frame.empty:
        raise RuntimeError("新浪未返回近期分钟行情")
    frame = canonical_frame(frame, source="sina")
    return frame[frame.timestamp < latest_closed_date() + pd.Timedelta(days=1)].copy()


@serialized_baostock
def fetch_reference_5m(symbol: str, start, end) -> pd.DataFrame:
    if symbol.startswith(("1", "5")):
        import akshare as ak
        frame = ak.fund_etf_hist_min_em(symbol=symbol.split(".")[0], period="5", adjust="", start_date=str(pd.Timestamp(start)), end_date=str(pd.Timestamp(end) + pd.Timedelta(days=1)))
        frame = canonical_frame(frame, source="etf_reference_5m")
        frame["volume"] *= 100
        return frame
    import baostock as bs
    code, exchange = normalize_symbol(symbol).split(".")
    with contextlib.redirect_stdout(io.StringIO()):
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError("BaoStock 参考分钟登录失败")
        try:
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
        finally:
            bs.logout()


@serialized_baostock
def fetch_auxiliary(symbol: str, start, end, cancelled=lambda: False):
    if symbol.startswith(("1", "5")):
        return fetch_etf_auxiliary(symbol, start, end, cancelled)
    import baostock as bs
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

    with contextlib.redirect_stdout(io.StringIO()):
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError("BaoStock 登录失败：" + login.error_msg)
        try:
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
            calendar = rows(bs.query_trade_dates(start_date=start, end_date=end))
            calendar = pd.DataFrame({"date": pd.to_datetime(calendar.loc[calendar.is_trading_day == "1", "calendar_date"])})
            action_frames = []
            for year in range(daily.date.min().year, pd.Timestamp(end).year + 1):
                action_frames.append(rows(bs.query_dividend_data(bs_code, year=str(year), yearType="operate")))
            actions = normalize_actions(pd.concat(action_frames, ignore_index=True)) if action_frames else normalize_actions(pd.DataFrame())
            metadata["actions_verified"] = True
            return daily, actions, calendar, metadata
        finally:
            bs.logout()


@lru_cache(maxsize=48)
def _fund_actions(year: int, as_of: str):
    import akshare as ak
    return ak.fund_fh_em(year=str(year)), ak.fund_cf_em(year=str(year))


def fetch_etf_auxiliary(symbol, start, end, cancelled=lambda: False):
    import akshare as ak
    import baostock as bs
    from .rules import instrument_terms
    code = symbol.split(".")[0]
    frame = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=pd.Timestamp(start).strftime("%Y%m%d"), end_date=pd.Timestamp(end).strftime("%Y%m%d"), adjust="")
    if frame is None or frame.empty:
        raise RuntimeError("ETF 日线来源未返回数据")
    if "涨跌额" in frame:
        frame["preclose"] = (pd.to_numeric(frame["收盘"]) - pd.to_numeric(frame["涨跌额"])).round(3)
    daily = canonical_frame(frame, daily=True)
    daily["volume"] *= 100  # Eastmoney reports lots; canonical ETF volume is units.
    with contextlib.redirect_stdout(io.StringIO()):
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError("交易日历登录失败")
        try:
            result = bs.query_trade_dates(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
            if result.error_code != "0":
                raise RuntimeError(result.error_msg)
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            dates = pd.DataFrame(rows, columns=result.fields)
            calendar = pd.DataFrame({"date": pd.to_datetime(dates.loc[dates.is_trading_day == "1", "calendar_date"])})
        finally:
            bs.logout()
    records, warnings, unverified = [], [], []
    for year in range(daily.date.min().year, pd.Timestamp(end).year + 1):
        if cancelled():
            raise InterruptedError("任务已取消")
        try:
            dividends, splits = _fund_actions(year, str(pd.Timestamp(end).date()))
            for row in dividends.loc[dividends["基金代码"].astype(str).str.zfill(6) == code].to_dict("records"):
                records.append({"record_date": row["权益登记日"], "ex_date": row["除息日期"], "pay_date": row["分红发放日"], "cash_per_share": row["分红"], "bonus_ratio": 0})
            for row in splits.loc[splits["基金代码"].astype(str).str.zfill(6) == code].to_dict("records"):
                records.append({"record_date": row["拆分折算日"], "ex_date": row["拆分折算日"], "stock_date": row["拆分折算日"], "cash_per_share": 0, "bonus_ratio": 0, "unsupported_reason": "ETF 拆分资料缺少登记日及精确份额处理，请导入核实后的公司行动"})
        except Exception as exc:
            warnings.append(f"{year} ETF 分红/拆分资料未核验：{exc}")
            unverified.append(year)
    metadata = {"actions_verified": not warnings, "actions_checked_years": list(range(daily.date.min().year, pd.Timestamp(end).year + 1)), "actions_unverified_years": unverified, "warnings": warnings}
    return daily, normalize_actions(pd.DataFrame(records)), calendar, metadata


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
