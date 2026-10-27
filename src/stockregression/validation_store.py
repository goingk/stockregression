"""Durable audit state for complete research validation runs.

The validation workflow deliberately lives beside the legacy v5 result store.
This module owns its SQLite tables so the existing ``research_registry`` rows
remain untouched and can still be used as an independent history source.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from filelock import FileLock
from pydantic import BaseModel

from .models import normalize_symbol
from .storage import Store, json_safe, now_iso


__all__ = [
    "ValidationCheckpointIntegrityError",
    "ValidationLedgerIntegrityError",
    "ValidationRegistrationConflict",
    "ValidationRevisionConflict",
    "ValidationRunNotFound",
    "ValidationStore",
]


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_FROZEN_CHECKPOINT_REF = re.compile(
    r"validation-checkpoints/immutable/([0-9a-f]{64})\.json\Z"
)
_LEDGER_REF = re.compile(
    r"validation-ledgers/immutable/([0-9a-f]{64})\.json\.gz\Z"
)
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_INTERVAL_CONTAINER_KEYS = (
    "evaluation_intervals", "validation_intervals",
    "symbol_evaluation_intervals", "evaluation_windows",
    "validation_windows", "evaluation_periods", "validation_periods",
)
_WINDOW_KEYS = (
    "evaluation_window", "validation_window", "oos_window",
    "out_of_sample_window", "holdout_window", "evaluation", "validation",
)
_SYMBOL_CONTAINER_KEYS = (
    "symbol_plans", "symbol_evaluations", "by_symbol", "symbols",
)
_START_KEYS = (
    "evaluation_start", "evaluation_start_date", "validation_start",
    "start", "start_date", "from", "begin",
)
_END_KEYS = (
    "evaluation_end", "evaluation_end_date", "validation_end",
    "end", "end_date", "to", "finish",
)

_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY: set[str] = set()


class ValidationRegistrationConflict(ValueError):
    """A run id was reused for a different immutable research contract."""


class ValidationCheckpointIntegrityError(ValueError):
    """A frozen checkpoint is missing its content hash or is corrupted."""


class ValidationLedgerIntegrityError(ValueError):
    """A referenced validation ledger is missing or fails verification."""


class ValidationRevisionConflict(ValueError):
    """A revision attempted to lower a registered task limit."""


class ValidationRunNotFound(LookupError):
    """The requested validation run has not been registered."""


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_json_value(item) for item in value),
            key=lambda item: _canonical_json(item),
        )
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    converted = json_safe(value)
    if isinstance(converted, Mapping):
        return _json_value(converted)
    if isinstance(converted, (list, tuple)):
        return [_json_value(item) for item in converted]
    if isinstance(converted, (str, int, float, bool)) or converted is None:
        return converted
    return str(converted)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = _json_value(value)
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a mapping")
    return result


def _validate_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    value = run_id.strip()
    if value != run_id or not _RUN_ID_PATTERN.fullmatch(value):
        raise ValueError("run_id must be a safe file identifier")
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("run_id is reserved on Windows")
    return value


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _day(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        return timestamp.date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _endpoint(item: Mapping[str, Any]) -> tuple[str | None, str | None]:
    start = next((item[key] for key in _START_KEYS if key in item), None)
    end = next((item[key] for key in _END_KEYS if key in item), None)
    return _day(start), _day(end)


def _append_interval(
    target: dict[tuple[str, str, str, str], dict[str, Any]],
    symbol: Any,
    item: Mapping[str, Any],
    *,
    default_kind: str = "evaluation",
) -> None:
    start, end = _endpoint(item)
    if start is None:
        return
    end = end or start
    if end < start:
        return
    try:
        normalized = normalize_symbol(str(symbol))
    except (TypeError, ValueError):
        return
    kind = str(item.get("kind") or default_kind)
    key = (normalized, start, end, kind)
    target[key] = {
        "symbol": normalized,
        "start": start,
        "end": end,
        "kind": kind,
        "metadata": _json_value(item),
    }


def _collect_intervals(
    value: Any,
    target: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    default_symbol: str | None = None,
    depth: int = 0,
) -> None:
    """Read the common planner interval shapes without guessing dates.

    An explicit interval needs a start and an end (or an end equal to a
    single-day start).  If a symbol's plan contains a named evaluation window,
    that window is preferred over any other top-level date fields.
    """
    if depth > 8 or value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, Mapping):
                symbol = item.get("symbol") or default_symbol
                if symbol is None and any(key in item for key in _START_KEYS):
                    continue
                for key in _INTERVAL_CONTAINER_KEYS:
                    if key in item:
                        _collect_intervals(
                            item[key], target,
                            default_symbol=symbol, depth=depth + 1,
                        )
                if any(key in item for key in _START_KEYS):
                    if symbol is not None:
                        _append_interval(target, symbol, item)
                else:
                    for key in _WINDOW_KEYS:
                        if key in item:
                            _collect_intervals(
                                item[key], target,
                                default_symbol=symbol, depth=depth + 1,
                            )
        return
    if not isinstance(value, Mapping):
        return

    # A concrete interval, optionally naming its own symbol.
    item_symbol = value.get("symbol") or default_symbol
    has_explicit_window = any(key in value for key in _INTERVAL_CONTAINER_KEYS + _WINDOW_KEYS)
    if item_symbol is not None and any(key in value for key in _START_KEYS) and not has_explicit_window:
        _append_interval(target, item_symbol, value)
        return

    handled = False
    for key in _INTERVAL_CONTAINER_KEYS:
        if key in value:
            handled = True
            _collect_intervals(
                value[key], target, default_symbol=item_symbol, depth=depth + 1,
            )
    for key in _WINDOW_KEYS:
        if key in value:
            handled = True
            _collect_intervals(
                value[key], target, default_symbol=item_symbol, depth=depth + 1,
            )
    if handled:
        return

    # A mapping whose keys are symbols is common in multi-symbol plans.
    if item_symbol is None:
        for symbol, details in value.items():
            try:
                normalized = normalize_symbol(str(symbol))
            except (TypeError, ValueError):
                continue
            _collect_intervals(
                details, target, default_symbol=normalized, depth=depth + 1,
            )


def _extract_evaluation_intervals(
    request: Mapping[str, Any], plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    collected: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    # Plan data is authoritative.  The request is checked as a compatibility
    # fallback for callers that persist a manually authored plan.
    for source in (plan, request):
        for key in _INTERVAL_CONTAINER_KEYS:
            if key in source:
                _collect_intervals(source[key], collected)

    for source in (plan, request):
        symbol = source.get("symbol")
        _collect_intervals(source, collected, default_symbol=symbol)
        for key in _SYMBOL_CONTAINER_KEYS:
            details = source.get(key)
            if isinstance(details, Mapping):
                for symbol_key, symbol_plan in details.items():
                    try:
                        normalized = normalize_symbol(str(symbol_key))
                    except (TypeError, ValueError):
                        continue
                    _collect_intervals(
                        symbol_plan, collected, default_symbol=normalized,
                    )
            elif isinstance(details, (list, tuple)):
                _collect_intervals(details, collected)

    return [collected[key] for key in sorted(collected)]


def _request_symbols(request: Mapping[str, Any]) -> list[str]:
    values = request.get("symbols")
    if not isinstance(values, (list, tuple)):
        values = []
    if not values:
        research = request.get("research")
        context = research.get("research_context") if isinstance(research, Mapping) else None
        if isinstance(context, Mapping):
            values = [context.get("symbol")]
    symbols: list[str] = []
    for value in values:
        try:
            symbol = normalize_symbol(str(value))
        except (TypeError, ValueError):
            continue
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _payload_flag(payload: Mapping[str, Any], *names: str) -> bool:
    return any(bool(payload.get(name)) for name in names)


def _event_flags(event: str, payload: Mapping[str, Any]) -> tuple[bool, bool, int, str | None]:
    normalized = event.strip().lower()
    status = str(payload.get("status") or "").strip().lower()
    failed = (
        normalized in {"failed", "failure", "error", "interrupted"}
        or normalized.endswith("_failed")
        or normalized.endswith("_error")
        or status in {"failed", "failure", "error"}
        or bool(payload.get("error"))
        or bool(payload.get("failed"))
    )
    cache_hit = (
        normalized in {
            "cache_hit", "cache-hit", "cached", "cache_reused", "cache_reuse",
        }
        or ("cache" in normalized and any(word in normalized for word in ("hit", "reuse")))
        or _payload_flag(
            payload, "cache_hit", "cache_hit_event", "cache_reused",
            "cache_reuse", "cached", "cache",
        )
        or str(payload.get("source") or "").strip().lower() == "cache"
        or str(payload.get("cache_status") or "").strip().lower() in {"hit", "reused"}
        or bool(payload.get("from_cache"))
    )
    evaluations = 0
    if normalized == "completed" and not cache_hit and not failed:
        raw = payload.get("candidate_evaluations", payload.get("actual_evaluations"))
        if raw is None:
            for key in ("computation", "result", "summary"):
                nested = payload.get(key)
                if not isinstance(nested, Mapping):
                    continue
                raw = nested.get("candidate_evaluations", nested.get("actual_evaluations"))
                if raw is not None:
                    break
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            evaluations = max(0, int(raw))
    category = _first_text(
        payload.get("failure_category"), payload.get("error_category"),
        "unknown" if failed else None,
    )
    return failed, cache_hit, evaluations, category


def _row_payload(row: sqlite3.Row, field: str = "payload_json") -> Any:
    try:
        return json.loads(row[field] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _initial_limits(request: Mapping[str, Any], plan: Mapping[str, Any]) -> tuple[int | None, int | None]:
    budget = _first_number(
        request.get("total_evaluation_budget"), plan.get("total_evaluation_budget"),
        _nested(plan, "estimated_work", "total_evaluation_budget"),
        _nested(plan, "budget", "total_evaluation_budget"),
        _nested(request, "strategy", "total_evaluation_budget"),
        _nested(request, "research", "strategy", "total_evaluation_budget"),
    )
    runtime = _first_number(
        request.get("max_runtime_seconds"), plan.get("max_runtime_seconds"),
        _nested(plan, "estimated_work", "max_runtime_seconds"),
        _nested(plan, "budget", "max_runtime_seconds"),
    )
    return budget, runtime


def _revision_limit(
    payload: Mapping[str, Any], field: str,
) -> tuple[bool, int | None]:
    updates = payload.get("updates")
    updates = updates if isinstance(updates, Mapping) else {}
    aliases = (field,)
    if field == "total_evaluation_budget":
        aliases = (field, "effective_total_evaluation_budget")
    value: Any = None
    present = False
    for key in aliases:
        if key in payload:
            value, present = payload.get(key), True
            break
    if not present and field in updates:
        value, present = updates.get(field), True
    if not present or value is None:
        return False, None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not float(value).is_integer()
        or int(value) < 1
    ):
        raise ValidationRevisionConflict(f"{field} must be a positive integer")
    return True, int(value)


def _revision_plan_hash(payload: Mapping[str, Any]) -> str | None:
    updates = payload.get("updates")
    updates = updates if isinstance(updates, Mapping) else {}
    return _first_text(payload.get("plan_hash"), updates.get("plan_hash"))


def _interval_overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["symbol"] == right["symbol"]
        and left["start"] <= right["end"]
        and right["start"] <= left["end"]
    )


def _replace_atomic(source: Path, target: Path) -> None:
    """Replace with bounded retries for transient Windows sharing failures."""
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02 * 2 ** attempt)


class ValidationStore:
    """Persist immutable run contracts and append-only validation evidence."""

    def __init__(self, store: Store):
        if not isinstance(store, Store):
            raise TypeError("store must be a Store")
        self.store = store
        self.root = store.root
        self._schema_key = str(store.db_path.resolve())
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        if self._schema_key in _SCHEMA_READY:
            return
        with _SCHEMA_LOCK:
            if self._schema_key in _SCHEMA_READY:
                return
            with self.store.db() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_runs(
                        run_id TEXT PRIMARY KEY,
                        family_id TEXT,
                        parent_result_id TEXT,
                        parent_run_id TEXT,
                        suite_version TEXT,
                        declared_plan_hash TEXT,
                        request_json TEXT NOT NULL,
                        plan_json TEXT NOT NULL,
                        provenance_json TEXT NOT NULL,
                        request_sha256 TEXT NOT NULL,
                        plan_sha256 TEXT NOT NULL,
                        provenance_sha256 TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                db.execute(
                    "CREATE INDEX IF NOT EXISTS validation_runs_family "
                    "ON validation_runs(family_id,created_at)"
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_family_parents(
                        run_id TEXT PRIMARY KEY,
                        family_id TEXT,
                        parent_result_id TEXT,
                        parent_run_id TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                db.execute(
                    "CREATE INDEX IF NOT EXISTS validation_family_parents_family "
                    "ON validation_family_parents(family_id,created_at)"
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_symbol_intervals(
                        run_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        PRIMARY KEY(run_id,symbol,start_date,end_date,kind)
                    )
                    """
                )
                db.execute(
                    "CREATE INDEX IF NOT EXISTS validation_symbol_intervals_date "
                    "ON validation_symbol_intervals(symbol,start_date,end_date)"
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_events(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        unit_id TEXT NOT NULL,
                        event TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        failed INTEGER NOT NULL DEFAULT 0,
                        failure_category TEXT,
                        cache_hit INTEGER NOT NULL DEFAULT 0,
                        actual_evaluations INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                db.execute(
                    "CREATE INDEX IF NOT EXISTS validation_events_run "
                    "ON validation_events(run_id,id)"
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_views(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        result_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        section TEXT NOT NULL,
                        view_ordinal INTEGER NOT NULL,
                        viewed_at TEXT NOT NULL,
                        UNIQUE(result_id,section,view_ordinal)
                    )
                    """
                )
                db.execute(
                    "CREATE INDEX IF NOT EXISTS validation_views_run "
                    "ON validation_views(run_id,id)"
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_revisions(
                        run_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(run_id,revision)
                    )
                    """
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_checkpoints(
                        run_id TEXT PRIMARY KEY,
                        relative_path TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_store_state(
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                db.execute(
                    "INSERT INTO validation_store_state(key,value) VALUES('schema_version','1') "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
                )
            _SCHEMA_READY.add(self._schema_key)

    def _require_run(self, db, run_id: str) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM validation_runs WHERE run_id=?", (run_id,),
        ).fetchone()
        if row is None:
            raise ValidationRunNotFound(run_id)
        return row

    def register_plan(
        self,
        run_id: str,
        request_dict: Mapping[str, Any],
        plan_dict: Mapping[str, Any],
        provenance_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Register an immutable plan, or replay an identical registration."""
        run_id = _validate_run_id(run_id)
        request = _as_dict(request_dict, "request_dict")
        plan = _as_dict(plan_dict, "plan_dict")
        provenance = _as_dict(provenance_dict, "provenance_dict")
        request_json = _canonical_json(request)
        plan_json = _canonical_json(plan)
        provenance_json = _canonical_json(provenance)
        request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        plan_hash = hashlib.sha256(plan_json.encode("utf-8")).hexdigest()
        provenance_hash = hashlib.sha256(provenance_json.encode("utf-8")).hexdigest()
        family_id = _first_text(
            request.get("family_id"), plan.get("family_id"), provenance.get("family_id"),
        )
        parent_result_id = _first_text(
            request.get("parent_result_id"), plan.get("parent_result_id"),
            provenance.get("parent_result_id"),
        )
        parent_run_id = _first_text(
            plan.get("parent_run_id"), provenance.get("parent_run_id"),
        )
        suite_version = _first_text(
            request.get("suite_version"), plan.get("suite_version"),
        )
        declared_plan_hash = _first_text(
            request.get("plan_hash"), plan.get("plan_hash"),
            provenance.get("plan_hash"),
        )
        intervals = _extract_evaluation_intervals(request, plan)
        created_at = now_iso()
        replay = False

        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT request_sha256,plan_sha256,provenance_sha256 "
                "FROM validation_runs WHERE run_id=?", (run_id,),
            ).fetchone()
            if existing is not None:
                actual = (
                    existing["request_sha256"], existing["plan_sha256"],
                    existing["provenance_sha256"],
                )
                expected = (request_hash, plan_hash, provenance_hash)
                if actual != expected:
                    raise ValidationRegistrationConflict(
                        "run_id is already registered with a different request, plan, or provenance"
                    )
                replay = True
            else:
                db.execute(
                    "INSERT INTO validation_runs("
                    "run_id,family_id,parent_result_id,parent_run_id,suite_version,"
                    "declared_plan_hash,request_json,plan_json,provenance_json,"
                    "request_sha256,plan_sha256,provenance_sha256,created_at,updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id, family_id, parent_result_id, parent_run_id,
                        suite_version, declared_plan_hash, request_json, plan_json,
                        provenance_json, request_hash, plan_hash, provenance_hash,
                        created_at, created_at,
                    ),
                )
                db.execute(
                    "INSERT INTO validation_family_parents("
                    "run_id,family_id,parent_result_id,parent_run_id,created_at"
                    ") VALUES(?,?,?,?,?)",
                    (run_id, family_id, parent_result_id, parent_run_id, created_at),
                )
                for interval in intervals:
                    db.execute(
                        "INSERT INTO validation_symbol_intervals("
                        "run_id,symbol,start_date,end_date,kind,metadata_json"
                        ") VALUES(?,?,?,?,?,?)",
                        (
                            run_id, interval["symbol"], interval["start"],
                            interval["end"], interval["kind"],
                            _canonical_json(interval["metadata"]),
                        ),
                    )

        summary = self.audit_summary(run_id)
        summary["idempotent_replay"] = replay
        return summary

    def append_event(
        self,
        run_id: str,
        unit_id: str,
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_id = _validate_run_id(run_id)
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError("unit_id must be a non-empty string")
        if not isinstance(event, str) or not event.strip():
            raise ValueError("event must be a non-empty string")
        event = event.strip()
        if len(event) > 128:
            raise ValueError("event is too long")
        payload_value = {} if payload is None else _as_dict(payload, "payload")
        failed, cache_hit, evaluations, category = _event_flags(event, payload_value)
        created_at = now_iso()
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_run(db, run_id)
            cursor = db.execute(
                "INSERT INTO validation_events("
                "run_id,unit_id,event,payload_json,created_at,failed,"
                "failure_category,cache_hit,actual_evaluations"
                ") VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    run_id, unit_id, event, _canonical_json(payload_value),
                    created_at, int(failed), category, int(cache_hit), evaluations,
                ),
            )
            event_id = int(cursor.lastrowid)
        return {
            "id": event_id,
            "run_id": run_id,
            "unit_id": unit_id,
            "event": event,
            "payload": payload_value,
            "created_at": created_at,
            "failed": failed,
            "failure_category": category,
            "cache_hit": cache_hit,
            "actual_evaluations": evaluations,
        }

    def _checkpoint_path(self, run_id: str) -> Path:
        run_id = _validate_run_id(run_id)
        root = self.root.resolve()
        expected_folder = root / "validation-checkpoints"
        folder = expected_folder.resolve()
        if folder != expected_folder:
            raise ValueError("checkpoint directory escapes the validation data root")
        candidate = folder / f"{run_id}.json"
        if candidate.is_symlink():
            raise ValueError("checkpoint path must not be a symbolic link")
        target = candidate.resolve()
        if target != candidate or target.parent != folder:
            raise ValueError("checkpoint path escapes the validation data root")
        return target

    def _frozen_checkpoint_path(self, ref: str) -> tuple[Path, str]:
        if not isinstance(ref, str):
            raise TypeError("frozen checkpoint ref must be a string")
        matched = _FROZEN_CHECKPOINT_REF.fullmatch(ref)
        if matched is None:
            raise ValueError("frozen checkpoint ref is not a relative content hash")
        digest = matched.group(1)
        root = self.root.resolve()
        expected_folder = root / "validation-checkpoints" / "immutable"
        folder = expected_folder.resolve()
        if folder != expected_folder:
            raise ValueError("frozen checkpoint directory escapes the validation data root")
        candidate = folder / f"{digest}.json"
        if candidate.is_symlink():
            raise ValueError("frozen checkpoint path must not be a symbolic link")
        target = candidate.resolve()
        if target != candidate or target.parent != folder:
            raise ValueError("frozen checkpoint path escapes the validation data root")
        return target, digest

    def _ledger_path(self, ref: Mapping[str, Any]) -> tuple[Path, str]:
        if not isinstance(ref, Mapping):
            raise TypeError("ledger reference must be a mapping")
        relative = ref.get("__validation_ledger__")
        digest = ref.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValidationLedgerIntegrityError(
                "ledger reference must contain a relative path and sha256"
            )
        matched = _LEDGER_REF.fullmatch(relative)
        if matched is None or not _SHA256_PATTERN.fullmatch(digest):
            raise ValidationLedgerIntegrityError(
                "ledger reference is not a relative content-addressed gzip path"
            )
        if matched.group(1) != digest:
            raise ValidationLedgerIntegrityError(
                "ledger reference path does not match its sha256"
            )
        root = self.root.resolve()
        expected_folder = root / "validation-ledgers" / "immutable"
        folder = expected_folder.resolve()
        if folder != expected_folder:
            raise ValidationLedgerIntegrityError(
                "validation ledger directory escapes the data root"
            )
        candidate = folder / f"{digest}.json.gz"
        if candidate.is_symlink():
            raise ValidationLedgerIntegrityError(
                "validation ledger path must not be a symbolic link"
            )
        target = candidate.resolve()
        if target != candidate or target.parent != folder:
            raise ValidationLedgerIntegrityError(
                "validation ledger path escapes the data root"
            )
        return target, digest

    def save_ledger(
        self, run_id: str, unit_id: str, value: Mapping[str, Any],
    ) -> dict[str, str]:
        """Write one unit ledger as an immutable, content-addressed gzip file.

        The returned reference contains no ledger data.  A checkpoint can keep
        this small mapping in place of the full minute-level ledger object.
        """
        run_id = _validate_run_id(run_id)
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError("unit_id must be a non-empty string")
        ledger = _as_dict(value, "value")
        with self.store.db() as db:
            self._require_run(db, run_id)
        encoded = _canonical_json(ledger).encode("utf-8")
        compressed = gzip.compress(encoded, compresslevel=6, mtime=0)
        digest = hashlib.sha256(compressed).hexdigest()
        ref = {
            "__validation_ledger__": (
                f"validation-ledgers/immutable/{digest}.json.gz"
            ),
            "sha256": digest,
        }
        target, expected_digest = self._ledger_path(ref)
        if expected_digest != digest:
            raise ValidationLedgerIntegrityError("validation ledger hash mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(target) + ".lock", timeout=60)
        with lock:
            if target.is_file():
                self._verify_ledger_file(target, digest)
                return ref
            temporary = target.parent / f".{digest}.{uuid.uuid4().hex}.writing"
            try:
                with temporary.open("wb") as stream:
                    stream.write(compressed)
                    stream.flush()
                    os.fsync(stream.fileno())
                _replace_atomic(temporary, target)
            finally:
                if temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
        return ref

    @staticmethod
    def _verify_ledger_file(target: Path, digest: str) -> None:
        if not target.is_file() or target.is_symlink():
            raise ValidationLedgerIntegrityError(
                "validation ledger is missing or is not a regular file"
            )
        checksum = hashlib.sha256()
        try:
            with target.open("rb") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    checksum.update(chunk)
        except OSError as exc:
            raise ValidationLedgerIntegrityError(
                "validation ledger cannot be read"
            ) from exc
        if checksum.hexdigest() != digest:
            raise ValidationLedgerIntegrityError(
                "validation ledger content does not match its reference"
            )

    def load_ledger(self, ref: Mapping[str, Any]) -> dict[str, Any]:
        """Read and verify one immutable ledger reference."""
        target, digest = self._ledger_path(ref)
        self._verify_ledger_file(target, digest)
        try:
            with gzip.open(target, "rt", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidationLedgerIntegrityError(
                "validation ledger is not valid gzip JSON"
            ) from exc
        if not isinstance(value, dict):
            raise ValidationLedgerIntegrityError(
                "validation ledger must contain a JSON object"
            )
        return value

    def iter_ledger_json(self, ref: Mapping[str, Any]) -> Iterator[str]:
        """Yield verified decompressed UTF-8 JSON in bounded text chunks."""
        target, digest = self._ledger_path(ref)
        self._verify_ledger_file(target, digest)
        return self._iter_ledger_text(target)

    @staticmethod
    def _iter_ledger_text(target: Path) -> Iterator[str]:
        try:
            with gzip.open(target, "rt", encoding="utf-8") as stream:
                while True:
                    chunk = stream.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk
        except (OSError, UnicodeError) as exc:
            raise ValidationLedgerIntegrityError(
                "validation ledger stream is not valid gzip UTF-8"
            ) from exc

    def save_checkpoint(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        run_id = _validate_run_id(run_id)
        payload_value = _as_dict(payload, "payload")
        target = self._checkpoint_path(run_id)
        encoded = _canonical_json(payload_value)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with self.store.db() as db:
            self._require_run(db, run_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(target) + ".lock", timeout=60)
        with lock:
            temporary = target.parent / f".{run_id}.{uuid.uuid4().hex}.writing"
            try:
                with temporary.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                _replace_atomic(temporary, target)
            finally:
                if temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
        updated_at = now_iso()
        relative_path = target.relative_to(self.root.resolve()).as_posix()
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO validation_checkpoints("
                "run_id,relative_path,sha256,payload_json,updated_at"
                ") VALUES(?,?,?,?,?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "relative_path=excluded.relative_path,sha256=excluded.sha256,"
                "payload_json=excluded.payload_json,updated_at=excluded.updated_at",
                (run_id, relative_path, digest, encoded, updated_at),
            )
        return {
            "run_id": run_id,
            "path": str(target),
            "relative_path": relative_path,
            "sha256": digest,
            "updated_at": updated_at,
        }

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """Read a checkpoint without creating or changing any filesystem state."""
        target = self._checkpoint_path(run_id)
        if not target.is_file() or target.is_symlink():
            return None
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def freeze_checkpoint(self, run_id: str) -> str | None:
        """Archive the latest saved checkpoint under its content hash.

        Returns a data-root-relative POSIX reference that callers store beside
        the result.  An absent checkpoint returns ``None``.  The payload is
        copied from SQLite so a concurrent mutable-file update cannot change
        the content represented by the returned hash reference.
        """
        run_id = _validate_run_id(run_id)
        with self.store.db() as db:
            self._require_run(db, run_id)
            row = db.execute(
                "SELECT payload_json,sha256 FROM validation_checkpoints "
                "WHERE run_id=?", (run_id,),
            ).fetchone()
        if row is None:
            return None
        encoded = str(row["payload_json"])
        digest = str(row["sha256"])
        if (
            _SHA256_PATTERN.fullmatch(digest) is None
            or hashlib.sha256(encoded.encode("utf-8")).hexdigest() != digest
        ):
            raise ValidationCheckpointIntegrityError(
                "saved checkpoint payload does not match its recorded sha256"
            )
        try:
            payload = json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidationCheckpointIntegrityError(
                "saved checkpoint payload is not valid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ValidationCheckpointIntegrityError(
                "saved checkpoint payload must be a JSON object"
            )

        ref = f"validation-checkpoints/immutable/{digest}.json"
        target, expected_digest = self._frozen_checkpoint_path(ref)
        if expected_digest != digest:
            raise ValidationCheckpointIntegrityError("frozen checkpoint hash mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(target) + ".lock", timeout=60)
        with lock:
            if target.is_file():
                self._verify_frozen_checkpoint(target, digest)
                return ref
            if target.is_symlink():
                raise ValidationCheckpointIntegrityError(
                    "frozen checkpoint path must not be a symbolic link"
                )
            temporary = target.parent / f".{digest}.{uuid.uuid4().hex}.writing"
            try:
                with temporary.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                _replace_atomic(temporary, target)
            finally:
                if temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
        return ref

    def _verify_frozen_checkpoint(self, target: Path, digest: str) -> dict[str, Any]:
        if not target.is_file() or target.is_symlink():
            raise ValidationCheckpointIntegrityError(
                "frozen checkpoint is missing or is not a regular file"
            )
        try:
            encoded = target.read_bytes()
        except OSError as exc:
            raise ValidationCheckpointIntegrityError(
                "frozen checkpoint cannot be read"
            ) from exc
        if hashlib.sha256(encoded).hexdigest() != digest:
            raise ValidationCheckpointIntegrityError(
                "frozen checkpoint content does not match its reference"
            )
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidationCheckpointIntegrityError(
                "frozen checkpoint is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ValidationCheckpointIntegrityError(
                "frozen checkpoint must contain a JSON object"
            )
        return payload

    def load_frozen_checkpoint(self, ref: str | None) -> dict[str, Any] | None:
        """Read a content-addressed checkpoint; never create files or DB rows."""
        if ref is None:
            return None
        target, digest = self._frozen_checkpoint_path(ref)
        if not target.is_file():
            return None
        return self._verify_frozen_checkpoint(target, digest)

    def record_view(
        self,
        result_id: str,
        run_id: str | None = None,
        section: str = "overview",
    ) -> dict[str, Any]:
        if not isinstance(result_id, str) or not result_id.strip():
            raise ValueError("result_id must be a non-empty string")
        if not isinstance(section, str) or not section.strip():
            raise ValueError("section must be a non-empty string")
        section = section.strip()
        if len(section) > 100:
            raise ValueError("section is too long")

        resolved_run_id = _first_text(run_id)
        if resolved_run_id is None:
            try:
                result = self.store.result(result_id)
            except (OSError, ValueError, KeyError, TypeError):
                result = None
            if isinstance(result, dict):
                evidence = result.get("evidence")
                local_audit = evidence.get("local_audit") if isinstance(evidence, Mapping) else None
                resolved_run_id = _first_text(
                    result.get("run_id"), result.get("validation_run_id"),
                    local_audit.get("run_id") if isinstance(local_audit, Mapping) else None,
                    result.get("id"),
                )
        resolved_run_id = _first_text(resolved_run_id, result_id)
        if resolved_run_id is None:
            raise ValueError("run_id could not be resolved")
        resolved_run_id = _validate_run_id(resolved_run_id)
        viewed_at = now_iso()
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT COALESCE(MAX(view_ordinal),0)+1 AS ordinal "
                "FROM validation_views WHERE result_id=? AND section=?",
                (result_id, section),
            ).fetchone()
            ordinal = int(row["ordinal"])
            first_viewed_at = viewed_at
            if ordinal > 1:
                first = db.execute(
                    "SELECT viewed_at FROM validation_views "
                    "WHERE result_id=? AND section=? AND view_ordinal=1",
                    (result_id, section),
                ).fetchone()
                if first is not None:
                    first_viewed_at = first["viewed_at"]
            cursor = db.execute(
                "INSERT INTO validation_views("
                "result_id,run_id,section,view_ordinal,viewed_at"
                ") VALUES(?,?,?,?,?)",
                (result_id, resolved_run_id, section, ordinal, viewed_at),
            )
            view_id = int(cursor.lastrowid)
        return {
            "id": view_id,
            "result_id": result_id,
            "run_id": resolved_run_id,
            "section": section,
            "view_ordinal": ordinal,
            "view_count": ordinal,
            "first_view": ordinal == 1,
            "first_viewed_at": first_viewed_at,
            "viewed_at": viewed_at,
        }

    def register_revision(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        run_id = _validate_run_id(run_id)
        payload_value = _as_dict(payload, "payload")
        created_at = now_iso()
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            run = self._require_run(db, run_id)
            revision_rows = db.execute(
                "SELECT payload_json FROM validation_revisions "
                "WHERE run_id=? ORDER BY revision", (run_id,),
            ).fetchall()
            request = json.loads(run["request_json"])
            plan = json.loads(run["plan_json"])
            current_budget, current_runtime = _initial_limits(request, plan)
            for previous in revision_rows:
                previous_payload = _row_payload(previous)
                if not isinstance(previous_payload, Mapping):
                    continue
                _, previous_budget = _revision_limit(
                    previous_payload, "total_evaluation_budget",
                )
                _, previous_runtime = _revision_limit(
                    previous_payload, "max_runtime_seconds",
                )
                if previous_budget is not None:
                    current_budget = previous_budget
                if previous_runtime is not None:
                    current_runtime = previous_runtime
            present, proposed_budget = _revision_limit(
                payload_value, "total_evaluation_budget",
            )
            if present and current_budget is not None and proposed_budget < current_budget:
                raise ValidationRevisionConflict(
                    "total_evaluation_budget cannot be lowered by a revision"
                )
            present, proposed_runtime = _revision_limit(
                payload_value, "max_runtime_seconds",
            )
            if present and current_runtime is not None and proposed_runtime < current_runtime:
                raise ValidationRevisionConflict(
                    "max_runtime_seconds cannot be lowered by a revision"
                )
            row = db.execute(
                "SELECT COALESCE(MAX(revision),0)+1 AS revision "
                "FROM validation_revisions WHERE run_id=?", (run_id,),
            ).fetchone()
            revision = int(row["revision"])
            db.execute(
                "INSERT INTO validation_revisions(run_id,revision,payload_json,created_at) "
                "VALUES(?,?,?,?)",
                (run_id, revision, _canonical_json(payload_value), created_at),
            )
        return {
            **payload_value,
            "payload": payload_value,
            "run_id": run_id,
            "revision": revision,
            "created_at": created_at,
        }

    def _overlap_summary(
        self, db, run_id: str, current_family_id: str | None,
        intervals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not intervals:
            return {
                "detected": False,
                "recorded_registry_studies": 0,
                "exact_registry_studies": 0,
                "validation_runs": 0,
                "exact_validation_runs": 0,
                "different_family_runs": 0,
                "families": [],
                "pairs": [],
            }
        pairs: list[dict[str, Any]] = []
        registry_ids: set[str] = set()
        exact_registry_ids: set[str] = set()
        prior_runs: set[str] = set()
        exact_prior_runs: set[str] = set()
        different_family_runs: set[str] = set()
        prior_families: set[str] = set()

        for row in db.execute(
            "SELECT registry_id,protocol_hash,symbol,evaluation_intervals "
            "FROM research_registry"
        ).fetchall():
            try:
                prior_intervals = json.loads(row["evaluation_intervals"] or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(prior_intervals, list):
                continue
            for prior in prior_intervals:
                if not isinstance(prior, Mapping):
                    continue
                candidate = {
                    "symbol": row["symbol"],
                    "start": _day(prior.get("start")),
                    "end": _day(prior.get("end") or prior.get("start")),
                }
                if candidate["start"] is None or candidate["end"] is None:
                    continue
                for current in intervals:
                    if _interval_overlaps(current, candidate):
                        registry_ids.add(str(row["registry_id"]))
                        if (
                            current["start"] == candidate["start"]
                            and current["end"] == candidate["end"]
                        ):
                            exact_registry_ids.add(str(row["registry_id"]))
                        pairs.append({
                            "source": "research_registry",
                            "id": str(row["registry_id"]),
                            "protocol_hash": row["protocol_hash"],
                            "symbol": current["symbol"],
                            "current_interval": {
                                "start": current["start"], "end": current["end"],
                            },
                            "prior_interval": {
                                "start": candidate["start"], "end": candidate["end"],
                            },
                        })

        for row in db.execute(
            "SELECT i.run_id,i.symbol,i.start_date,i.end_date,r.family_id "
            "FROM validation_symbol_intervals i "
            "JOIN validation_runs r ON r.run_id=i.run_id WHERE i.run_id<>?",
            (run_id,),
        ).fetchall():
            candidate = {
                "symbol": row["symbol"], "start": row["start_date"],
                "end": row["end_date"],
            }
            for current in intervals:
                if _interval_overlaps(current, candidate):
                    prior_runs.add(str(row["run_id"]))
                    if (
                        current["start"] == candidate["start"]
                        and current["end"] == candidate["end"]
                    ):
                        exact_prior_runs.add(str(row["run_id"]))
                    if current_family_id is not None and row["family_id"] != current_family_id:
                        different_family_runs.add(str(row["run_id"]))
                    elif current_family_id is None:
                        different_family_runs.add(str(row["run_id"]))
                    if row["family_id"]:
                        prior_families.add(str(row["family_id"]))
                    pairs.append({
                        "source": "validation_run",
                        "id": str(row["run_id"]),
                        "family_id": row["family_id"],
                        "symbol": current["symbol"],
                        "current_interval": {
                            "start": current["start"], "end": current["end"],
                        },
                        "prior_interval": {
                            "start": candidate["start"], "end": candidate["end"],
                        },
                    })

        return {
            "detected": bool(registry_ids or prior_runs),
            "recorded_registry_studies": len(registry_ids),
            "exact_registry_studies": len(exact_registry_ids),
            "validation_runs": len(prior_runs),
            "exact_validation_runs": len(exact_prior_runs),
            "different_family_runs": len(different_family_runs),
            "families": sorted(prior_families),
            "pairs": pairs,
        }

    def _audit_summary(self, run_id: str) -> dict[str, Any]:
        with self.store.db() as db:
            run = self._require_run(db, run_id)
            interval_rows = db.execute(
                "SELECT symbol,start_date,end_date,kind,metadata_json "
                "FROM validation_symbol_intervals WHERE run_id=? "
                "ORDER BY symbol,start_date,end_date,kind", (run_id,),
            ).fetchall()
            intervals = [
                {
                    "symbol": row["symbol"], "start": row["start_date"],
                    "end": row["end_date"], "kind": row["kind"],
                    "metadata": _row_payload(row, "metadata_json"),
                }
                for row in interval_rows
            ]
            events = db.execute(
                "SELECT * FROM validation_events WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
            views = db.execute(
                "SELECT * FROM validation_views WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
            revisions = db.execute(
                "SELECT revision,payload_json,created_at FROM validation_revisions "
                "WHERE run_id=? ORDER BY revision", (run_id,),
            ).fetchall()
            checkpoint = db.execute(
                "SELECT relative_path,sha256,updated_at FROM validation_checkpoints "
                "WHERE run_id=?", (run_id,),
            ).fetchone()
            if run["family_id"]:
                family_rows = db.execute(
                    "SELECT run_id,family_id,parent_result_id,parent_run_id,created_at "
                    "FROM validation_family_parents WHERE family_id=? "
                    "ORDER BY created_at,run_id", (run["family_id"],),
                ).fetchall()
            else:
                family_rows = db.execute(
                    "SELECT run_id,family_id,parent_result_id,parent_run_id,created_at "
                    "FROM validation_family_parents WHERE run_id=?", (run_id,),
                ).fetchall()
            overlap = self._overlap_summary(
                db, run_id, run["family_id"], intervals,
            )

        request = json.loads(run["request_json"])
        plan = json.loads(run["plan_json"])
        provenance = json.loads(run["provenance_json"])

        failure_categories: dict[str, int] = {}
        event_log: list[dict[str, Any]] = []
        failed_events = 0
        cache_hits = 0
        actual_evaluations = 0
        completed_events = 0
        for row in events:
            failed = bool(row["failed"])
            cache_hit = bool(row["cache_hit"])
            evaluations = int(row["actual_evaluations"])
            failed_events += int(failed)
            cache_hits += int(cache_hit)
            actual_evaluations += evaluations
            completed_events += int(str(row["event"]).lower() == "completed")
            if failed:
                category = str(row["failure_category"] or "unknown")
                failure_categories[category] = failure_categories.get(category, 0) + 1
            event_log.append({
                "id": row["id"], "unit_id": row["unit_id"],
                "event": row["event"], "payload": _row_payload(row),
                "created_at": row["created_at"], "failed": failed,
                "failure_category": row["failure_category"],
                "cache_hit": cache_hit, "actual_evaluations": evaluations,
            })

        view_log: list[dict[str, Any]] = []
        sections: dict[str, dict[str, Any]] = {}
        for row in views:
            section = str(row["section"])
            entry = sections.setdefault(section, {
                "first_viewed_at": row["viewed_at"],
                "last_viewed_at": row["viewed_at"],
                "count": 0,
            })
            entry["last_viewed_at"] = row["viewed_at"]
            entry["count"] += 1
            view_log.append({
                "id": row["id"], "result_id": row["result_id"],
                "run_id": row["run_id"], "section": section,
                "view_ordinal": row["view_ordinal"], "viewed_at": row["viewed_at"],
            })

        revision_log: list[dict[str, Any]] = []
        initial_budget = _first_number(
            request.get("total_evaluation_budget"), plan.get("total_evaluation_budget"),
            _nested(plan, "estimated_work", "total_evaluation_budget"),
            _nested(plan, "budget", "total_evaluation_budget"),
            _nested(request, "strategy", "total_evaluation_budget"),
            _nested(request, "research", "strategy", "total_evaluation_budget"),
        )
        initial_runtime = _first_number(
            request.get("max_runtime_seconds"), plan.get("max_runtime_seconds"),
            _nested(plan, "estimated_work", "max_runtime_seconds"),
            _nested(plan, "budget", "max_runtime_seconds"),
        )
        current_budget = initial_budget
        current_runtime = initial_runtime
        current_plan_hash = run["declared_plan_hash"]
        for row in revisions:
            payload = _row_payload(row)
            if not isinstance(payload, dict):
                payload = {}
            updates = payload.get("updates")
            updates = updates if isinstance(updates, Mapping) else {}
            updated_budget = (
                payload.get("total_evaluation_budget")
                if "total_evaluation_budget" in payload
                else payload.get("effective_total_evaluation_budget")
                if "effective_total_evaluation_budget" in payload
                else updates.get("total_evaluation_budget")
            )
            updated_runtime = (
                payload.get("max_runtime_seconds")
                if "max_runtime_seconds" in payload
                else updates.get("max_runtime_seconds")
            )
            if _number(updated_budget) is not None:
                current_budget = _number(updated_budget)
            if _number(updated_runtime) is not None:
                current_runtime = _number(updated_runtime)
            revised_plan_hash = _first_text(
                payload.get("plan_hash"), updates.get("plan_hash"),
            )
            if revised_plan_hash is not None:
                current_plan_hash = revised_plan_hash
            revision_log.append({
                **payload, "payload": payload, "run_id": run_id,
                "revision": row["revision"], "created_at": row["created_at"],
            })

        symbol_intervals: dict[str, list[dict[str, Any]]] = {
            symbol: [] for symbol in _request_symbols(request)
        }
        for interval in intervals:
            symbol_intervals.setdefault(interval["symbol"], []).append(interval)
        symbol_intervals = {symbol: symbol_intervals[symbol] for symbol in sorted(symbol_intervals)}

        checkpoint_data = None
        if checkpoint is not None:
            checkpoint_data = {
                "path": str(self.root / checkpoint["relative_path"]),
                "relative_path": checkpoint["relative_path"],
                "sha256": checkpoint["sha256"],
                "updated_at": checkpoint["updated_at"],
            }

        overlap_status = (
            "overlap_detected" if overlap["detected"] else "no_recorded_overlap"
        )
        return {
            "run_id": run_id,
            "family_id": run["family_id"],
            "parent_result_id": run["parent_result_id"],
            "parent_run_id": run["parent_run_id"],
            "suite_version": run["suite_version"],
            "plan_hash": run["declared_plan_hash"],
            "current_plan_hash": current_plan_hash,
            "plan_content_sha256": run["plan_sha256"],
            "request_sha256": run["request_sha256"],
            "provenance_sha256": run["provenance_sha256"],
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "request": request,
            "plan": plan,
            "provenance": provenance,
            "family": {
                "family_id": run["family_id"],
                "parent_result_id": run["parent_result_id"],
                "parent_run_id": run["parent_run_id"],
                "members": [
                    {
                        "run_id": item["run_id"],
                        "family_id": item["family_id"],
                        "parent_result_id": item["parent_result_id"],
                        "parent_run_id": item["parent_run_id"],
                        "created_at": item["created_at"],
                    }
                    for item in family_rows
                ],
            },
            "symbol_intervals": symbol_intervals,
            "evaluation_intervals": intervals,
            "events": {
                "total": len(event_log),
                "completed": completed_events,
                "failed": failed_events,
                "cache_hits": cache_hits,
                "actual_evaluations": actual_evaluations,
                "failure_categories": failure_categories,
                "log": event_log,
            },
            "event_log": event_log,
            "failed_events": [item for item in event_log if item["failed"]],
            "event_count": len(event_log),
            "failed_event_count": failed_events,
            "cache_hit_count": cache_hits,
            "cache_hits": cache_hits,
            "actual_trial_evaluations": actual_evaluations,
            "candidate_evaluations": actual_evaluations,
            "views": {
                "total": len(view_log),
                "sections": sections,
                "log": view_log,
            },
            "view_events": view_log,
            "evidence_views": view_log,
            "overview_viewed": "overview" in sections,
            "overview_first_viewed_at": (
                sections["overview"]["first_viewed_at"]
                if "overview" in sections else None
            ),
            "revisions": revision_log,
            "budget": {
                "initial_total_evaluation_budget": initial_budget,
                "current_total_evaluation_budget": current_budget,
                "initial_max_runtime_seconds": initial_runtime,
                "current_max_runtime_seconds": current_runtime,
                "revision_count": len(revision_log),
                "actual_trial_evaluations": actual_evaluations,
                "remaining_evaluations": (
                    max(0, int(current_budget) - actual_evaluations)
                    if current_budget is not None else None
                ),
            },
            "checkpoint": checkpoint_data,
            "recorded_overlap": overlap,
            "overlapping_prior_studies": overlap["recorded_registry_studies"],
            "overlapping_validation_runs": overlap["validation_runs"],
            "overlapping_families": overlap["families"],
            "exact_repeat_prior_studies": overlap["exact_registry_studies"],
            "exact_repeat_validation_runs": overlap["exact_validation_runs"],
            "recorded_overlap_status": overlap_status,
            "contamination_assessed": False,
            "contamination_status": "not_assessed",
            "contaminated": None,
            "overlap_implies_contamination": False,
            "human_untouched_verified": False,
        }

    def audit_summary(self, run_id: str) -> dict[str, Any]:
        """Return audit evidence from reads only; no migration or metadata write."""
        return self._audit_summary(_validate_run_id(run_id))


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _first_number(*values: Any) -> int | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None
