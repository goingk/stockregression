"""Safe persistence and coverage merging for sync auxiliary data."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


_SYNC_AUXILIARY_VERSION = "market-sync-v2-package"
_DATE_FIELDS = ("record_date", "ex_date", "pay_date", "stock_date")


def _records(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        rows = value.to_dict("records")
    elif isinstance(value, dict):
        rows = [value]
    else:
        rows = list(value)
    return [dict(row) for row in rows if isinstance(row, dict)]


def _clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _row_key(row: dict) -> str:
    return json.dumps(
        {str(key): _clean(value) for key, value in row.items()},
        sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"),
    )


def _dedupe(rows: list[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        clean = {str(key): _clean(value) for key, value in row.items()}
        for field in _DATE_FIELDS:
            if clean.get(field) is None:
                continue
            try:
                stamp = pd.Timestamp(clean[field])
                if pd.notna(stamp):
                    clean[field] = stamp.normalize().strftime("%Y-%m-%d")
            except (TypeError, ValueError, OverflowError):
                pass
        unique.setdefault(_row_key(clean), clean)
    return [unique[key] for key in sorted(unique)]


def _year(value: Any) -> int | None:
    if value is None or value is pd.NA:
        return None
    try:
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else int(stamp.year)
    except (TypeError, ValueError, OverflowError):
        return None


def _years(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (str, int, np.integer)):
        value = [value]
    result = set()
    for item in value:
        try:
            year = int(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if 1800 <= year <= 9999:
            result.add(year)
    return result


def _calendar_dates(rows: list[dict]) -> set[str]:
    dates = set()
    for row in rows:
        try:
            stamp = pd.Timestamp(row.get("date"))
            if pd.notna(stamp):
                dates.add(stamp.normalize().strftime("%Y-%m-%d"))
        except (TypeError, ValueError, OverflowError):
            continue
    return dates


def _expected_dataset_years(service, symbol: str) -> set[int]:
    try:
        dataset = service.repository.dataset(symbol, "1day") or {}
        first, last = _year(dataset.get("first_time")), _year(dataset.get("last_time"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return set()
    if first is None or last is None or last < first:
        return set()
    return set(range(first, last + 1))


def _merge_actions(
    service, symbol: str, metadata: dict, incoming_rows: list[dict],
) -> dict:
    repository = service.repository
    existing = repository.auxiliary(symbol, "actions") or {}
    old_rows = _records(existing.get("payload"))
    instrument = service.store.instrument(symbol) or {}

    checked = _years(metadata.get("actions_checked_years"))
    unverified_now = _years(metadata.get("actions_unverified_years"))
    explicit_unverified = "actions_unverified_years" in metadata
    reported_verified = metadata.get("actions_verified")
    if checked and reported_verified is not True and (
        not explicit_unverified or not unverified_now
    ):
        # A false/unknown global result without failure-year detail cannot
        # authorize replacement for an apparently successful checked year.
        unverified_now.update(checked)
    successful = checked - unverified_now

    # Only explicitly successful years authorize deletion. Unknown ex_date rows
    # cannot be assigned to a year and therefore remain untouched.
    retained = []
    for row in old_rows:
        action_year = _year(row.get("ex_date"))
        if action_year is None or action_year not in successful:
            retained.append(row)
    accepted_incoming = [
        row for row in incoming_rows
        if _year(row.get("ex_date")) not in unverified_now
    ]
    merged_rows = _dedupe([*retained, *accepted_incoming])

    old_checked = _years(instrument.get("actions_checked_years"))
    old_unverified = _years(instrument.get("actions_unverified_years"))
    merged_checked = old_checked | checked
    merged_unverified = (old_unverified - successful) | unverified_now

    old_verified = instrument.get("actions_verified")
    if old_verified is None:
        old_verified = existing.get("verified", False)
    old_verified = bool(old_verified)

    # A previously complete ledger stays complete after successful partial
    # refreshes. An incomplete ledger becomes complete only when all known
    # historical years, or at least every previously unresolved year, are now
    # covered successfully. A first partial probe has no such evidence.
    known_years = old_checked | old_unverified
    expected_years = _expected_dataset_years(service, symbol) or known_years
    required_recheck = expected_years - old_checked
    all_old_unknown_rechecked = (
        bool(old_unverified) and old_unverified.issubset(successful)
        and (not expected_years or expected_years.issubset(known_years))
    )
    full_known_range_covered = (
        bool(required_recheck) and required_recheck.issubset(successful)
        and expected_years.issubset(old_checked | successful)
    )
    no_new_failure = not unverified_now and reported_verified is not False
    if merged_unverified or not no_new_failure:
        actions_verified = False
    elif old_verified:
        actions_verified = True
    elif checked and (full_known_range_covered or all_old_unknown_rechecked):
        actions_verified = True
    else:
        actions_verified = False

    actions_persisted = False
    if "actions" in metadata:
        same_payload = _dedupe(old_rows) == merged_rows
        target_verified = actions_verified
        has_evidence = bool(
            incoming_rows or checked or unverified_now
            or metadata.get("actions_verified") is not None
        )
        if has_evidence and (
            not same_payload or bool(existing.get("verified", False)) != target_verified
        ):
            version = str(metadata.get("auxiliary_version") or _SYNC_AUXILIARY_VERSION)
            repository.save_auxiliary(
                symbol, "actions", merged_rows,
                version=version, verified=target_verified,
            )
            actions_persisted = True
            existing = repository.auxiliary(symbol, "actions") or existing

    changes = {
        "actions_verified": actions_verified,
        "actions_checked_years": sorted(merged_checked),
        "actions_unverified_years": sorted(merged_unverified),
    }
    # Keep the prior instrument coverage available to the next incremental run.
    # Do not synthesize an instrument row when no explicit listing row exists.
    if instrument and (checked or unverified_now or any(
        instrument.get(key) is not None
        for key in ("actions_verified", "actions_checked_years", "actions_unverified_years")
    )) and any(instrument.get(key) != value for key, value in changes.items()):
        service.store.put_instrument(
            symbol, instrument.get("name") or symbol, **changes,
        )
    changes["actions_persisted"] = actions_persisted
    changes["actions_auxiliary_version"] = existing.get("version")
    return changes


def persist_sync_auxiliary(service, symbol: str, metadata: dict) -> dict:
    """Persist sync calendar/actions without shrinking auxiliary history.

    Returns the actions coverage fields to merge into the sync's source metadata.
    The caller is responsible for holding its package lock and publication fence.
    """
    metadata = dict(metadata or {})
    repository = service.repository
    output: dict[str, Any] = {}

    if "calendar" in metadata and metadata["calendar"] is not None:
        incoming = _records(metadata["calendar"])
        if incoming:
            existing = repository.auxiliary(symbol, "calendar") or {}
            old_rows = _records(existing.get("payload"))
            old_dates, incoming_dates = _calendar_dates(old_rows), _calendar_dates(incoming)
            if incoming_dates:
                merged_dates = old_dates | incoming_dates
                merged_payload = [{"date": day} for day in sorted(merged_dates)]
                old_verified = bool(existing.get("verified", False))
                # A partial re-fetch cannot upgrade an unverified historical tail.
                calendar_verified = old_verified or old_dates.issubset(incoming_dates)
                same_payload = old_dates == merged_dates
                if not (same_payload and old_verified == calendar_verified):
                    version = str(metadata.get("auxiliary_version") or _SYNC_AUXILIARY_VERSION)
                    repository.save_auxiliary(
                        symbol, "calendar", merged_payload,
                        version=version, verified=calendar_verified,
                    )
                    persisted = repository.auxiliary(symbol, "calendar") or {}
                else:
                    persisted = existing
                output["calendar_auxiliary_version"] = persisted.get("version")
                output["calendar_persisted"] = not (
                    same_payload and old_verified == calendar_verified
                )

    if "actions" in metadata and metadata["actions"] is not None:
        incoming_actions = _records(metadata["actions"])
        output.update(_merge_actions(service, symbol, metadata, incoming_actions))

    return output
