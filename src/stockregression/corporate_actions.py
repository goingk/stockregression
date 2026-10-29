"""Local, calculation-oriented normalization of corporate actions."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


DATE_FIELDS = ("record_date", "ex_date", "pay_date", "stock_date")
LEGACY_SPLIT_REASON = "ETF 拆分资料缺少登记日及精确份额处理，请导入核实后的公司行动"
UNRESOLVED_SPLIT = "份额折算缺少有效比例或无法从本地前收盘定位除权日"


def day(value: Any) -> pd.Timestamp | None:
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            return None
        return result.tz_localize(None).normalize() if result.tzinfo else result.normalize()
    except (ValueError, TypeError, OverflowError):
        return None


def number(value: Any, default: float = 0.0) -> float:
    try:
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return math.nan


def daily_prices(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    if values.empty:
        return pd.DataFrame(columns=["close", "preclose"])
    key = "date" if "date" in values else "timestamp" if "timestamp" in values else None
    values.index = pd.DatetimeIndex([day(value) for value in (values.pop(key) if key else values.index)])
    values = values[values.index.notna()].sort_index()
    if "tradestatus" in values:
        values = values[pd.to_numeric(values.tradestatus, errors="coerce").eq(1)]
    return values[~values.index.duplicated(keep="last")]


def split_record(row: dict) -> dict:
    """Keep the vendor's fold date distinct from an actual ex-date."""
    ratio = number(row.get("拆分折算", row.get("拆分折算比例")), math.nan)
    fold = row.get("拆分折算日")
    return {
        "record_date": row.get("权益登记日"),
        "ex_date": row.get("除权日"),
        "stock_date": row.get("份额可交易日"),
        "cash_per_share": 0.0, "bonus_ratio": ratio - 1.0,
        "action_type": "split", "split_date": fold,
        "implemented": True, "unsupported_reason": "",
    }


def resolve_actions(actions: pd.DataFrame, daily: pd.DataFrame, price_tick: float = 0.001) -> pd.DataFrame:
    """Resolve vendor fold dates using only that day and the next local session.

    Explicit dates are authoritative. No reference-price jump is used to invent
    a share ratio: the ratio must come from the event itself.
    """
    if actions is None or actions.empty:
        return pd.DataFrame(columns=[*DATE_FIELDS, "cash_per_share", "bonus_ratio", "implemented", "unsupported_reason"])
    prices = daily_prices(daily)
    rows = []
    for original in actions.to_dict("records"):
        row = dict(original)
        for key in DATE_FIELDS:
            row[key] = day(row.get(key))
        cash = number(row.get("cash_per_share"))
        bonus = number(row.get("bonus_ratio"), math.nan if row.get("action_type") == "split" else 0.0)
        row.update(cash_per_share=cash, bonus_ratio=bonus)
        reason_value = row.get("unsupported_reason")
        reason = "" if reason_value is None or pd.isna(reason_value) else str(reason_value).strip()
        legacy = reason == LEGACY_SPLIT_REASON
        split = row.get("action_type") == "split" or day(row.get("split_date")) is not None or legacy
        row["date_unresolved"] = str(row.get("date_unresolved", False)).lower() in {"true", "1"}
        if split:
            fold = day(row.get("split_date")) or row["record_date"] or row["ex_date"]
            row["split_date"] = fold
            row["action_type"] = "split"
            # Old placeholders explicitly copied the fold date into all dates.
            explicit = row["ex_date"] is not None and (not legacy or (row["record_date"] is not None and row["record_date"] < row["ex_date"])) and not row.get("date_unresolved", False)
            if not explicit:
                row["ex_date"] = None
                matches = []
                if fold is not None and math.isfinite(bonus) and 1 + bonus > 0 and bonus != 0:
                    candidates = ([fold] if fold in prices.index else []) + list(prices.index[prices.index > fold][:1])
                    for candidate in candidates:
                        before = prices.index[prices.index < candidate]
                        if not len(before):
                            continue
                        previous = number(prices.loc[before[-1]].get("close"), math.nan)
                        reference = number(prices.loc[candidate].get("preclose"), math.nan)
                        if previous > 0 and reference > 0 and abs(reference - (previous - cash) / (1 + bonus)) <= price_tick * 1.1:
                            matches.append(candidate)
                if len(matches) == 1:
                    row["ex_date"] = matches[0]
                    row["date_unresolved"] = False
                    before = prices.index[prices.index < matches[0]]
                    row["record_date"] = before[-1] if len(before) else None
                    row["stock_date"] = matches[0]
                else:
                    # Retain the fold day for window scoping, but never pretend
                    # this unresolved boundary can be used in a calculation.
                    row["ex_date"] = fold
                    row["date_unresolved"] = True
            valid = math.isfinite(bonus) and 1 + bonus > 0 and bonus != 0 and row["ex_date"] is not None and not row.get("date_unresolved", False)
            if valid:
                row["date_unresolved"] = False
                if row["stock_date"] is None:
                    row["stock_date"] = row["ex_date"]
                if reason in {LEGACY_SPLIT_REASON, UNRESOLVED_SPLIT}:
                    reason = ""
            else:
                reason = UNRESOLVED_SPLIT
        # Complete numeric legacy events are supported without an audit flag.
        row["unsupported_reason"] = reason
        implemented = row.get("implemented", True)
        if implemented is None or pd.isna(implemented):
            implemented = True
        elif isinstance(implemented, str):
            implemented = implemented.strip().lower() not in {"false", "0", "0.0", "no", "n", "否"}
        row["implemented"] = bool(implemented)
        rows.append(row)
    result = pd.DataFrame(rows)
    for key in DATE_FIELDS:
        result[key] = pd.to_datetime(result[key], errors="coerce")
    return result.drop_duplicates().sort_values("ex_date", na_position="last").reset_index(drop=True)


def action_issues(actions: pd.DataFrame, daily: pd.DataFrame, start, end, *, settlement: bool = False, price_tick: float = 0.001) -> list[str]:
    """Return only issues needed by this interval and this kind of calculation."""
    lower, upper = day(start), day(end)
    rows = resolve_actions(actions, daily, price_tick)
    prices = daily_prices(daily)
    issues, action_days = [], set()
    for row in rows.to_dict("records"):
        ex = day(row.get("ex_date"))
        if ex is not None and ((lower is not None and ex < lower) or (upper is not None and ex > upper)):
            continue
        if ex is None:
            record = day(row.get("record_date"))
            if record is not None and upper is not None and record > upper:
                continue
            issues.append("公司行动缺少除权日，无法计算相关区间")
            continue
        action_days.add(ex)
        reason = str(row.get("unsupported_reason") or "").strip()
        bonus, cash = number(row.get("bonus_ratio")), number(row.get("cash_per_share"))
        if reason:
            issues.append(f"{ex.date()}：{reason}")
        elif not math.isfinite(bonus) or 1 + bonus <= 0 or not math.isfinite(cash) or cash < 0:
            issues.append(f"{ex.date()}：公司行动比例或现金额无效")
        elif str(row.get("implemented", True)).lower() in {"false", "0", "no", "否"}:
            issues.append(f"{ex.date()}：公司行动尚未实施")
        if settlement:
            record, stock, pay = (day(row.get(key)) for key in ("record_date", "stock_date", "pay_date"))
            if record is None or record > ex:
                issues.append(f"{ex.date()}：缺少有效的除权前登记日")
            if bonus != 0 and (stock is None or stock < ex):
                issues.append(f"{ex.date()}：份额折算缺少有效可交易日")
            if cash > 0 and (pay is None or pay < ex):
                issues.append(f"{ex.date()}：现金分红缺少有效到账日")
    for ex, group in rows.groupby("ex_date"):
        if (lower is not None and ex < lower) or (upper is not None and ex > upper):
            continue
        ratios = pd.to_numeric(group.bonus_ratio, errors="coerce")
        if ratios.notna().all() and 1 + math.fsum(ratios) <= 0:
            issues.append(f"{ex.date()}：同日合计份额倍率必须大于零")
        if settlement and ratios.lt(0).any():
            share_dates = group.loc[ratios.ne(0), "stock_date"].drop_duplicates()
            if len(share_dates) > 1:
                issues.append(f"{ex.date()}：同日合并份额的可交易日不一致，无法结算")
    if not prices.empty and "preclose" in prices and "close" in prices:
        previous = pd.to_numeric(prices.close, errors="coerce").shift(1)
        mismatch = (pd.to_numeric(prices.preclose, errors="coerce") - previous).abs() > price_tick * 1.1
        for stamp in prices.index[mismatch.fillna(False)]:
            if (lower is None or stamp >= lower) and (upper is None or stamp <= upper) and stamp not in action_days:
                issues.append(f"{stamp.date()}：除权参考价变化缺少份额或分红数据")
    return list(dict.fromkeys(issues))
