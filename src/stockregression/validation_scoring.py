"""Descriptive scoring for continuous-account research validation.

Segment returns are level differences on one continuous account.  They are
never obtained by summing daily or per-rebalance return rates.  The module
also exposes the four GSS screens and deliberately emits no p values or
portfolio net-asset series.
"""
from __future__ import annotations

import math
import random
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .etf_statistics import score_etfs


GSS_SCREENS = ("random_all", "score_only", "filter_only", "score_filter")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp


def _normalized_equity(equity: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    rows = []
    for raw in equity or []:
        if not isinstance(raw, Mapping):
            continue
        stamp = _timestamp(raw.get("timestamp", raw.get("date")))
        if stamp is None:
            continue
        equity_value = _finite(raw.get("equity"))
        if equity_value is None or equity_value < 0:
            continue
        invested = _finite(raw.get("total_invested", raw.get("principal")))
        if invested is None:
            invested = equity_value
        rows.append({
            "date": stamp,
            "equity": equity_value,
            "total_invested": max(0.0, invested),
        })
    rows.sort(key=lambda item: item["date"])
    return rows


def _date(value: Any) -> pd.Timestamp | None:
    stamp = _timestamp(value)
    return stamp.normalize() if stamp is not None else None


def segment_principal_returns(
    equity: Iterable[Mapping[str, Any]] | None,
    blocks: Iterable[Mapping[str, Any]] | None,
    *, initial_equity: Any = None, initial_principal: Any = None,
    initial_point: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute each block from one continuous account's profit levels.

    For segment ``[start, end]`` (both days inclusive), the capital basis of a
    point is the account's equity at the segment baseline plus all new
    principal added since that baseline.  Profit is measured against that
    basis, so a previous block's gain or loss remains inside the starting
    assets without inheriting the whole-horizon cumulative-principal peak.
    Drawdown is rebuilt from the segment-local profit peak.  No segment rate is
    summed.
    """
    points = _normalized_equity(equity)
    opening_equity = _finite(initial_equity)
    opening_principal = _finite(initial_principal)
    if isinstance(initial_point, Mapping):
        opening_equity = _finite(initial_point.get("equity"))
        opening_principal = _finite(
            initial_point.get("total_invested", initial_point.get("principal"))
        )
    if opening_equity is not None and opening_equity < 0:
        opening_equity = None
    if opening_principal is not None and opening_principal < 0:
        opening_principal = None
    if opening_principal is None and opening_equity is not None:
        opening_principal = opening_equity
    segments: list[dict[str, Any]] = []
    for number, block in enumerate(blocks or [], start=1):
        if not isinstance(block, Mapping):
            continue
        start = _date(block.get("start"))
        end = _date(block.get("end"))
        if start is None or end is None or end < start:
            segments.append({
                "block": number, "start": block.get("start"), "end": block.get("end"),
                "status": "invalid_block", "principal_return": None,
            })
            continue
        end_exclusive = end + pd.Timedelta(days=1)
        inside = [point for point in points if start <= point["date"] < end_exclusive]
        before = [point for point in points if point["date"] < start]
        baseline = before[-1] if before else None
        if not inside:
            segments.append({
                "block": number, "start": start.date().isoformat(),
                "end": end.date().isoformat(), "status": "missing_equity",
                "principal_return": None, "days": int(block.get("days", 0) or 0),
            })
            continue
        if baseline is not None:
            start_point = baseline
            initial_source = "prior_equity_point"
        elif opening_equity is not None:
            start_point = {
                "equity": opening_equity,
                "total_invested": opening_principal,
            }
            initial_source = "explicit_initial_equity"
        else:
            start_point = inside[0]
            initial_source = "first_observed_equity"
        starting_equity = start_point["equity"]
        starting_invested = start_point["total_invested"]
        ending = inside[-1]
        ending_invested = ending["total_invested"]
        bases, profits = [], []
        for point in inside:
            capital_added = max(0.0, point["total_invested"] - starting_invested)
            basis = starting_equity + capital_added
            bases.append(basis)
            profits.append(point["equity"] - basis)
        peak_principal = max(bases, default=0.0)
        end_profit = profits[-1]
        running_peak = 0.0
        drawdown_raw = 0.0
        for profit in profits:
            running_peak = max(running_peak, profit)
            drawdown_raw = max(drawdown_raw, running_peak - profit)
        principal_return = end_profit / peak_principal if peak_principal > 0 else None
        segments.append({
            "block": number,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "days": int(block.get("days", len({item["date"] for item in inside})) or 0),
            "status": "completed" if peak_principal > 0 else "unevaluable_principal",
            "starting_equity": starting_equity,
            "initial_source": initial_source,
            "ending_equity": ending["equity"],
            "starting_profit": 0.0,
            "ending_profit": end_profit,
            "profit_delta": end_profit,
            "segment_peak_principal": peak_principal,
            "starting_invested": starting_invested,
            "ending_invested": ending_invested,
            "capital_added": ending_invested - starting_invested,
            "principal_return": principal_return,
            "max_principal_drawdown": (
                drawdown_raw / peak_principal if peak_principal > 0 else None
            ),
        })

    completed = [item for item in segments if item.get("principal_return") is not None]
    total = None
    if points:
        baseline = {
            "equity": opening_equity,
            "total_invested": opening_principal,
        } if opening_equity is not None else points[0]
        final = points[-1]
        bases = [
            baseline["equity"] + max(0.0, item["total_invested"] - baseline["total_invested"])
            for item in points
        ]
        principal = max(bases)
        if principal > 0:
            final_basis = baseline["equity"] + max(
                0.0, final["total_invested"] - baseline["total_invested"],
            )
            total = (final["equity"] - final_basis) / principal
    return {
        "method": "continuous_account_profit_delta_over_segment_peak_principal",
        "rates_are_summed": False,
        "segments": segments,
        "completed_segments": len(completed),
        "all_segments_completed": len(completed) == len(segments) and bool(segments),
        "continuous_account_return": total,
    }


def descriptive_summary(values: Iterable[Any]) -> dict[str, Any]:
    """Summarise independent observations without a significance test."""
    values = list(values)
    clean = [number for value in values if (number := _finite(value)) is not None]
    return {
        "count": len(clean),
        "missing": sum(1 for value in values if _finite(value) is None),
        "mean": mean(clean) if clean else None,
        "median": median(clean) if clean else None,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "evidence_kind": "descriptive",
    }


def independent_account_summary(accounts: Mapping[str, Any]) -> dict[str, Any]:
    """Return per-account summaries; no pooled portfolio NAV is constructed."""
    rows: dict[str, dict[str, Any]] = {}
    for symbol, raw in sorted(accounts.items()):
        if not isinstance(raw, Mapping):
            continue
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            metrics = raw.get("strategy_metrics")
        if not isinstance(metrics, Mapping):
            metrics = raw
        rows[str(symbol)] = {
            "peak_principal_return": _finite(metrics.get("peak_principal_return")),
            "max_principal_drawdown": _finite(metrics.get("max_principal_drawdown")),
            "total_profit": _finite(metrics.get("total_profit")),
            "peak_invested": _finite(metrics.get("peak_invested", metrics.get("total_invested"))),
        }
    return {
        "accounts": rows,
        "account_count": len(rows),
        "portfolio_nav_constructed": False,
        "evidence_kind": "descriptive_independent_accounts",
    }


def common_decision_anchors(
    dates_by_symbol: Mapping[str, Iterable[Any]], *, horizon_days: int,
) -> dict[str, Any]:
    """Build fixed, non-overlapping forward-test anchors from common dates.

    The common calendar is the intersection of actual evaluation dates, not an
    intersection of adaptive parameter-change dates.  Each anchor uses exactly
    ``horizon_days`` following common observations; an incomplete final horizon
    is retained as an anchor with no fabricated replacement.
    """
    horizon = int(horizon_days)
    if horizon < 1:
        raise ValueError("GSS 共同评价 horizon 必须为正整数")
    symbol_dates: dict[str, set[pd.Timestamp]] = {}
    for symbol, values in dates_by_symbol.items():
        symbol_dates[str(symbol)] = {
            value for raw in values or []
            if (value := _date(raw)) is not None
        }
    if not symbol_dates:
        common: list[pd.Timestamp] = []
    else:
        ordered = list(symbol_dates.values())
        common = sorted(set.intersection(*ordered))
    anchors: list[dict[str, Any]] = []
    index = 0
    while index + horizon < len(common):
        anchors.append({
            "decision_date": common[index].date().isoformat(),
            "outcome_start": common[index + 1].date().isoformat(),
            "outcome_end": common[index + horizon].date().isoformat(),
            "decision_index": index,
            "outcome_observations": horizon,
        })
        # Keep adjacent forward intervals disjoint.
        index += horizon + 1
    return {
        "horizon_days": horizon,
        "common_dates": [value.date().isoformat() for value in common],
        "common_date_count": len(common),
        "anchors": anchors,
        "status": "completed" if anchors else "insufficient_common_horizon",
        "symbols": sorted(symbol_dates),
    }


def paired_forward_outcome(
    grid_equity: Iterable[Mapping[str, Any]] | None,
    matched_funding_equity: Iterable[Mapping[str, Any]] | None,
    *, start: Any, end: Any, days: int | None = None,
    grid_initial_equity: Any = None, grid_initial_principal: Any = None,
    matched_funding_initial_equity: Any = None,
    matched_funding_initial_principal: Any = None,
) -> dict[str, Any]:
    """Measure one selected ETF over one exact common forward interval."""
    block = {"start": start, "end": end}
    if days is not None:
        block["days"] = int(days)
    grid = segment_principal_returns(
        grid_equity, [block], initial_equity=grid_initial_equity,
        initial_principal=grid_initial_principal,
    )["segments"][0]
    matched = segment_principal_returns(
        matched_funding_equity, [block],
        initial_equity=matched_funding_initial_equity,
        initial_principal=matched_funding_initial_principal,
    )["segments"][0]
    grid_return = _finite(grid.get("principal_return"))
    matched_return = _finite(matched.get("principal_return"))
    paired = (
        grid_return - matched_return
        if grid_return is not None and matched_return is not None else None
    )
    return {
        "start": str(start), "end": str(end),
        "status": (
            "completed" if paired is not None else
            "partial_pair" if grid_return is not None else "missing_outcome"
        ),
        "grid_principal_return": grid_return,
        "matched_funding_return": matched_return,
        "paired_excess": paired,
        "max_principal_drawdown": _finite(grid.get("max_principal_drawdown")),
        "segment_peak_principal": _finite(grid.get("segment_peak_principal")),
        "grid_segment_status": grid.get("status"),
        "matched_segment_status": matched.get("status"),
    }


def _information(row: Mapping[str, Any]) -> dict[str, Any]:
    score = _finite(row.get("gss_score"))
    symbol = str(row.get("symbol", ""))
    return {
        "symbol": symbol,
        "gss_score": score,
        "rank": int(row["rank"]) if _finite(row.get("rank")) is not None else None,
        "eligible": bool(row.get("eligible")),
        "status": row.get("status"),
        "filter_reasons": list(row.get("filter_reasons") or []),
    }


def _quality_member(row: Mapping[str, Any]) -> bool:
    """Apply the same visible-data gate before every GSS screening pool."""
    status = str(row.get("status") or "").strip().lower()
    sampling = row.get("sampling")
    comparable = not isinstance(sampling, Mapping) or sampling.get("comparable") is not False
    return status in {"completed", "complete"} and comparable


def _selection(members: Sequence[dict[str, Any]], top_n: int) -> dict[str, Any]:
    chosen = list(members[:top_n])
    count = len(chosen)
    return {
        "members": chosen,
        "symbols": [item["symbol"] for item in chosen],
        "count": count,
        "target_count": top_n,
        "sufficient": count >= top_n,
        "shortfall": max(0, top_n - count),
        "insufficient_marked": count < top_n,
    }


def gss_screens(
    rows: Sequence[Mapping[str, Any]], *, top_n: int = 3, seed: int = 42,
    decision_date: Any = None,
) -> dict[str, Any]:
    """Compute GSS once and expose the four required screening rules.

    Missing future outcomes are irrelevant to membership: only information
    available at ``decision_date`` participates.  When a screen cannot find
    three members, its shortfall is explicit and no later observation is used
    as a replacement.
    """
    target = int(top_n)
    if target < 1:
        raise ValueError("GSS 筛选目标数必须为正整数")
    copies: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        symbol = str(row.get("symbol", ""))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else None
        if metrics is not None:
            row.setdefault("status", "completed")
            row.setdefault("sampling", (row.get("diagnostics") or {}).get("sampling"))
        copies.append(row)
    score_etfs(copies)
    by_symbol = {str(row["symbol"]): _information(row) for row in copies}
    all_members = sorted(by_symbol.values(), key=lambda item: item["symbol"])
    quality_rows = [row for row in copies if _quality_member(row)]
    quality_members = sorted(
        (by_symbol[str(row["symbol"])] for row in quality_rows),
        key=lambda item: item["symbol"],
    )
    scored = sorted(
        (item for item in quality_members if item["gss_score"] is not None),
        key=lambda item: (-item["gss_score"], item["symbol"]),
    )
    eligible = [item for item in quality_members if item["eligible"]]
    eligible_scored = sorted(
        (item for item in eligible if item["gss_score"] is not None),
        key=lambda item: (-item["gss_score"], item["symbol"]),
    )
    count = len(all_members)
    quality_count = len(quality_members)
    take = min(target, quality_count)
    rng = random.Random(f"{seed}:{decision_date}:random_all")
    random_members = rng.sample(quality_members, take) if take else []
    filter_rng = random.Random(f"{seed}:{decision_date}:filter_only")
    filter_take = min(target, len(eligible))
    filter_members = filter_rng.sample(eligible, filter_take) if filter_take else []
    screens = {
        "random_all": _selection(random_members, target),
        "score_only": _selection(scored, target),
        "filter_only": _selection(filter_members, target),
        "score_filter": _selection(eligible_scored, target),
    }
    for name, selection in screens.items():
        selection["rule"] = {
            "random_all": "deterministic_random_from_all_available",
            "score_only": "highest_gss_regardless_of_filter",
            "filter_only": "deterministic_random_from_filter_eligible",
            "score_filter": "filter_eligible_then_highest_gss",
        }[name]
    return {
        "decision_date": str(decision_date) if decision_date is not None else None,
        "available_count": count,
        "cohort_size": quality_count,
        "quality_cohort_size": quality_count,
        "unavailable_count": count - quality_count,
        "score_cohort_size": len(scored),
        "eligible_count": len(eligible),
        "screens": screens,
        "future_missing_policy": "retain_at_decision_no_future_imputation",
        "historical_universe_verified": False,
        "evidence_kind": "descriptive_gss_screens",
    }


def gss_rows_for_date(
    statistics_by_symbol: Mapping[str, Mapping[str, Any] | None],
    *, available_by_symbol: Mapping[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Convert as-of statistics into rows accepted by :func:`gss_screens`."""
    rows = []
    for symbol, raw in sorted(statistics_by_symbol.items()):
        if available_by_symbol is not None and not available_by_symbol.get(symbol, False):
            continue
        if not isinstance(raw, Mapping):
            continue
        diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), Mapping) else {}
        rows.append({
            **dict(raw), "symbol": str(symbol),
            "status": raw.get("status"),
            "metrics": dict(raw.get("metrics") or {}),
            "eligible": bool(raw.get("eligible")),
            "filter_reasons": list(raw.get("filter_reasons") or []),
            "sampling": raw.get("sampling", diagnostics.get("sampling")),
        })
    return rows


__all__ = [
    "GSS_SCREENS", "segment_principal_returns", "descriptive_summary",
    "independent_account_summary", "common_decision_anchors",
    "paired_forward_outcome", "gss_screens", "gss_rows_for_date",
]
