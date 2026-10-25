"""Read immutable quality certificates captured with a market snapshot."""
from __future__ import annotations

import pandas as pd


def frozen_quality_state(manifest: dict | None, timeframe: str, start, end) -> dict | None:
    """Accept a certificate only for its exact dependencies, rules and interval.

    Certificates are generated from the server-side validation ledger and are
    hashed into ETF snapshot versions. They do not certify newly appended bars
    and never replace actual price/field checks performed by readiness.
    """
    from .market_data.repository import validation_rules_version
    from .market_data.timeframes import normalize_timeframe

    if not isinstance(manifest, dict):
        return None
    evidence = manifest.get("quality_evidence")
    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        return None
    timeframe = normalize_timeframe(timeframe)
    rules_version = validation_rules_version(timeframe)
    rules_versions = evidence.get("rules_versions") or {}
    if not isinstance(rules_versions, dict):
        return None
    evidence_version = rules_versions.get(timeframe, evidence.get("rules_version"))
    try:
        if evidence_version is None or int(evidence_version) < rules_version:
            return None
    except (TypeError, ValueError):
        return None
    try:
        versions = {
            key: int(value["version"])
            for key, value in manifest.get("market_versions", {}).items()
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if evidence.get("market_versions") != versions:
        return None
    if evidence.get("auxiliary_versions") != manifest.get("auxiliary_versions", {}):
        return None
    try:
        lower, upper = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    except (ValueError, TypeError):
        return None
    if pd.isna(lower) or pd.isna(upper) or lower > upper:
        return None
    groups = evidence.get("intervals", {})
    if not isinstance(groups, dict):
        return None
    intervals = groups.get(timeframe, [])
    if not isinstance(intervals, list):
        return None
    for interval in intervals:
        if not isinstance(interval, dict) or interval.get("status") != "complete":
            continue
        try:
            first = pd.Timestamp(interval["start"]).normalize()
            last = pd.Timestamp(interval["end"]).normalize()
        except (KeyError, ValueError, TypeError):
            continue
        if first <= lower <= upper <= last:
            return {
                "symbol": manifest.get("symbol"), "timeframe": timeframe,
                "version": versions.get(timeframe), "rules_version": rules_version,
                "verified_from": first.date().isoformat(), "verified_through": last.date().isoformat(),
                "expected_start": lower.date().isoformat(), "expected_end": upper.date().isoformat(),
                "range_start": lower.date().isoformat(), "range_end": upper.date().isoformat(),
                "failed": [], "pending": [], "available": True, "status": "complete",
                "quality_source": "frozen_snapshot", "calendar_available": True,
            }
    return None
