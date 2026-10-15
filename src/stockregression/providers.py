"""Side-effect-free adapters inspired by astock-data-toolkit (MIT).

Upstream CLI modules are deliberately never imported. All network activity is
explicit, bounded, and performed by a data job. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import contextlib
import io
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from .models import normalize_symbol
from .quality import canonical_frame

HF_REPO = "neigezhu/china-a-share-1min-ohlcv"
TDX_SERVERS = [("180.153.18.170", 7709), ("180.153.18.172", 80), ("202.108.253.139", 80), ("60.191.117.167", 7709)]


def latest_closed_date():
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    day = pd.Timestamp(now.date())
    if now.hour < 16:
        day -= pd.Timedelta(days=1)
    return day


def fetch_hf(symbol: str, root: Path, revision: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    code, exchange = normalize_symbol(symbol).split(".")
    filename = f"data/stock_1m/{exchange}/{code}.parquet"
    path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=filename,
                           revision=revision, cache_dir=str(root / "raw" / "huggingface"))
    return canonical_frame(pd.read_parquet(path), source="hf")


def fetch_instruments(revision: str) -> list[dict]:
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/metadata/coverage_by_instrument.csv"
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
                        "listing_date": row.get("listing_date"), "delisting_date": row.get("delisting_date")})
    # Metadata may not carry display names; this optional lookup never filters ST.
    try:
        import akshare as ak
        names = ak.stock_info_a_code_name()
        mapping = dict(zip(names["code"].astype(str).str.zfill(6), names["name"]))
        for record in records:
            record["name"] = mapping.get(record["symbol"].split(".")[0], record["name"])
    except Exception:
        pass
    return records


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
    frame = ak.stock_zh_a_minute(symbol=exchange.lower() + code, period="1", adjust="")
    if frame is None or frame.empty:
        raise RuntimeError("新浪未返回近期分钟行情")
    frame = canonical_frame(frame, source="sina")
    return frame[frame.timestamp < latest_closed_date() + pd.Timedelta(days=1)].copy()


def fetch_reference_5m(symbol: str, start, end) -> pd.DataFrame:
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


def fetch_auxiliary(symbol: str, start, end, cancelled=lambda: False):
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
            daily = rows(bs.query_history_k_data_plus(bs_code,
                "date,open,high,low,close,preclose,volume,amount,tradestatus,isST",
                start_date=start, end_date=end, frequency="d", adjustflag="3"))
            if daily.empty:
                raise RuntimeError("BaoStock 未返回所选股票的日线")
            daily = canonical_frame(daily, daily=True)
            calendar = rows(bs.query_trade_dates(start_date=start, end_date=end))
            calendar = pd.DataFrame({"date": pd.to_datetime(calendar.loc[calendar.is_trading_day == "1", "calendar_date"])})
            action_frames = []
            for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
                action_frames.append(rows(bs.query_dividend_data(bs_code, year=str(year), yearType="operate")))
            actions = normalize_actions(pd.concat(action_frames, ignore_index=True)) if action_frames else normalize_actions(pd.DataFrame())
            metadata["actions_verified"] = True
            return daily, actions, calendar, metadata
        finally:
            bs.logout()


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
