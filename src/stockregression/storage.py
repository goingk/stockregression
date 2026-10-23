from __future__ import annotations

import hashlib
import gzip
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from .models import normalize_symbol


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat() if not pd.isna(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dumps(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)


_V4_ARCHIVE_FIELDS = (
    "candidate_archive", "search_candidate_archive", "trial_archive",
)

_CANDIDATE_REFERENCE = "__candidate_archive__"


class CompletionRevisionError(Exception):
    """A completion revision cannot serve the requested decision branch.

    A revision only covers the decision and decision index it was computed
    for.  Reading any other branch with that revision would silently show the
    pre-completion archive as if it had been revised, so the caller has to
    fail loudly instead.
    """

    def __init__(self, message: str, *, code: str, revision: int,
                 scope: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.revision = int(revision)
        self.scope = scope


_CANDIDATE_FIELDS = {
    "ordinal", "candidate_id", "buy_step", "sell_step", "buy_shares",
    "sell_shares", "j_score", "training_excess_return", "max_principal_drawdown",
    "validation_excess_original", "validation_return_original",
    "validation_excess_stress", "validation_return_stress",
    "neighborhood_mean", "downside_risk_tau", "pos_penalty", "score_complete",
}

_CANDIDATE_AXIS_FIELDS = {"buy_step", "sell_step", "buy_shares", "sell_shares"}


def _checkpoint_points(values: Any) -> set[tuple[int, ...]]:
    """Read a checkpoint coordinate list into a comparable set."""
    points: set[tuple[int, ...]] = set()
    for value in values or []:
        if isinstance(value, (list, tuple)):
            try:
                points.add(tuple(int(item) for item in value))
            except (TypeError, ValueError):
                continue
    return points


def _checkpoint_rows(*groups: Any) -> list[dict[str, Any]]:
    """Merge checkpoint rows, keeping one row per evaluated coordinate."""
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for row in group or []:
            if not isinstance(row, dict):
                continue
            params = row.get("params")
            key = dumps(params) if isinstance(params, dict) else dumps(row)
            merged[key] = row
    return list(merged.values())


def _merge_completion_checkpoint(existing: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Merge two concurrent checkpoint writes without losing paid work.

    Two workers may write the same checkpoint file.  The merge keeps the
    highest recorded consumption and the union of the processed points, so a
    later reader never believes that already paid evaluations are still
    pending and never spends the same budget twice.
    """
    if not isinstance(existing, dict) or not existing:
        return dict(payload)
    merged = {**existing, **payload}
    merged["consumed"] = max(
        int(existing.get("consumed", 0) or 0), int(payload.get("consumed", 0) or 0),
    )
    processed = _checkpoint_points(existing.get("processed")) | _checkpoint_points(payload.get("processed"))
    merged["processed"] = [list(point) for point in sorted(processed)]
    # A reservation stays charged until it is either processed or explicitly
    # released, so the in-flight set is the union minus the processed points.
    in_flight = _checkpoint_points(existing.get("in_flight")) | _checkpoint_points(payload.get("in_flight"))
    merged["in_flight"] = [list(point) for point in sorted(in_flight - processed)]
    merged["new_rows"] = _checkpoint_rows(existing.get("new_rows"), payload.get("new_rows"))
    merged["targets"] = existing.get("targets") or payload.get("targets")
    remaining = [
        int(value) for value in (existing.get("remaining_count"), payload.get("remaining_count"))
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    if remaining:
        merged["remaining_count"] = min(remaining)
    terminal = {"completed", "failed", "cancelled", "partial", "budget_exhausted"}
    if str(existing.get("status") or "") in terminal and str(payload.get("status") or "") == "running":
        merged["status"] = existing.get("status")
        merged["stop_reason"] = existing.get("stop_reason")
    return merged


def _write_candidate_archive(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = sorted(
        rows,
        key=lambda row: (
            int(row.get("candidate_id")) if isinstance(row.get("candidate_id"), (int, float)) else 2**63,
            dumps(row),
        ),
    )
    encoded = dumps(canonical)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    folder = root / "result_archives"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{digest}.sqlite"
    if not target.exists():
        lock = FileLock(str(target) + ".lock")
        with lock:
            if not target.exists():
                temporary = folder / f".{digest}.{uuid.uuid4().hex}.writing"
                connection = sqlite3.connect(temporary)
                try:
                    connection.executescript(
                        """
                        CREATE TABLE candidates(
                            ordinal INTEGER PRIMARY KEY,
                            candidate_id INTEGER,
                            buy_step REAL, sell_step REAL, buy_shares REAL, sell_shares REAL,
                            j_score REAL, training_excess_return REAL, max_principal_drawdown REAL,
                            validation_excess_original REAL, validation_return_original REAL,
                            validation_excess_stress REAL, validation_return_stress REAL,
                            neighborhood_mean REAL, downside_risk_tau REAL, pos_penalty REAL,
                            score_complete INTEGER,
                            row_json TEXT NOT NULL
                        );
                        CREATE TABLE candidate_regions(ordinal INTEGER NOT NULL, region_id INTEGER NOT NULL);
                        CREATE INDEX candidate_order ON candidates(ordinal);
                        CREATE INDEX candidate_j ON candidates(j_score DESC);
                        CREATE INDEX candidate_return ON candidates(training_excess_return DESC);
                        CREATE INDEX candidate_validation_return ON candidates(validation_excess_original DESC);
                        CREATE INDEX candidate_j_robust ON candidates(
                            j_score DESC, neighborhood_mean DESC, downside_risk_tau ASC,
                            pos_penalty ASC, training_excess_return DESC
                        );
                        CREATE INDEX candidate_region ON candidate_regions(region_id,ordinal);
                        """
                    )
                    connection.execute("BEGIN")
                    for ordinal, row in enumerate(canonical):
                        params = row.get("params") if isinstance(row.get("params"), dict) else {}
                        connection.execute(
                            "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                ordinal, row.get("candidate_id"),
                                params.get("buy_step"), params.get("sell_step"),
                                params.get("buy_shares"), params.get("sell_shares"),
                                row.get("j_score"), row.get("training_excess_return"),
                                row.get("max_principal_drawdown"),
                                row.get("validation_excess_original"), row.get("validation_return_original"),
                                row.get("validation_excess_stress"), row.get("validation_return_stress"),
                                row.get("neighborhood_mean"), row.get("downside_risk_tau"),
                                row.get("pos_penalty"),
                                None if row.get("score_complete") is None else int(bool(row.get("score_complete"))),
                                dumps(row),
                            ),
                        )
                        region_ids = row.get("region_ids") or []
                        if not isinstance(region_ids, list):
                            region_ids = []
                        connection.executemany(
                            "INSERT INTO candidate_regions VALUES(?,?)",
                            [(ordinal, int(region_id)) for region_id in region_ids if isinstance(region_id, int)],
                        )
                    connection.commit()
                finally:
                    connection.close()
                temporary.replace(target)
    return {_CANDIDATE_REFERENCE: target.relative_to(root).as_posix(), "archive_hash": digest, "row_count": len(canonical)}


def _merge_candidate_archives(root: Path, base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Create an immutable overlay view without materialising rows in memory."""
    base_relative = base.get(_CANDIDATE_REFERENCE)
    overlay_relative = overlay.get(_CANDIDATE_REFERENCE)
    if not isinstance(base_relative, str) or not isinstance(overlay_relative, str):
        return overlay
    identity = dumps({
        "base": base.get("archive_hash"), "overlay": overlay.get("archive_hash"),
    })
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    folder = root / "result_archives"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{digest}.sqlite"
    reference = {
        _CANDIDATE_REFERENCE: target.relative_to(root).as_posix(),
        "archive_hash": digest,
        "base_archive_hash": base.get("archive_hash"),
        "overlay_archive_hash": overlay.get("archive_hash"),
    }
    if target.exists():
        return {**reference, "row_count": _archive_row_count(target)}
    base_path = (root / base_relative).resolve()
    overlay_path = (root / overlay_relative).resolve()
    for path in (base_path, overlay_path):
        if root.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError("候选档案不存在")
    with FileLock(str(target) + ".lock"):
        if target.exists():
            return {**reference, "row_count": _archive_row_count(target)}
        temporary = folder / f".{digest}.{uuid.uuid4().hex}.writing"
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(
                """
                CREATE TABLE candidates(
                    ordinal INTEGER PRIMARY KEY, candidate_id INTEGER,
                    buy_step REAL, sell_step REAL, buy_shares REAL, sell_shares REAL,
                    j_score REAL, training_excess_return REAL, max_principal_drawdown REAL,
                    validation_excess_original REAL, validation_return_original REAL,
                    validation_excess_stress REAL, validation_return_stress REAL,
                    neighborhood_mean REAL, downside_risk_tau REAL, pos_penalty REAL,
                    score_complete INTEGER, row_json TEXT NOT NULL
                );
                CREATE TABLE candidate_regions(ordinal INTEGER NOT NULL, region_id INTEGER NOT NULL);
                CREATE INDEX candidate_j_robust ON candidates(
                    j_score DESC, neighborhood_mean DESC, downside_risk_tau ASC,
                    pos_penalty ASC, training_excess_return DESC
                );
                CREATE INDEX candidate_return ON candidates(training_excess_return DESC);
                CREATE INDEX candidate_validation_return ON candidates(validation_excess_original DESC);
                CREATE INDEX candidate_region ON candidate_regions(region_id,ordinal);
                """
            )
            # SQLite cannot attach a table alias with the same table name, so
            # collect the four-axis overlay rows (reporting overlays are
            # capped) and stream the much larger base archive row by row.
            overlay_connection = sqlite3.connect(overlay_path)
            overlay_connection.row_factory = sqlite3.Row
            overlay_rows: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
            overlay_order: list[tuple[Any, Any, Any, Any]] = []
            try:
                for row in overlay_connection.execute(
                    "SELECT buy_step,sell_step,buy_shares,sell_shares,row_json FROM candidates ORDER BY ordinal"
                ):
                    axis = (row["buy_step"], row["sell_step"], row["buy_shares"], row["sell_shares"])
                    overlay_rows[axis] = json.loads(row["row_json"])
                    overlay_order.append(axis)
            finally:
                overlay_connection.close()
            base_connection = sqlite3.connect(base_path)
            base_connection.row_factory = sqlite3.Row
            ordinal = 0
            replaced: set[tuple[Any, Any, Any, Any]] = set()
            next_candidate_id = 1
            try:
                for row in base_connection.execute(
                    "SELECT buy_step,sell_step,buy_shares,sell_shares,row_json FROM candidates ORDER BY ordinal"
                ):
                    axis = (row["buy_step"], row["sell_step"], row["buy_shares"], row["sell_shares"])
                    base_payload = json.loads(row["row_json"])
                    candidate_id = base_payload.get("candidate_id")
                    if isinstance(candidate_id, (int, float)) and not isinstance(candidate_id, bool):
                        next_candidate_id = max(next_candidate_id, int(candidate_id) + 1)
                    overlay = overlay_rows.get(axis)
                    if overlay is None:
                        _insert_archive_row(connection, ordinal, row)
                    else:
                        # The overlay replaces the base row in place and keeps
                        # the base identity and region membership, so a region
                        # filter still returns the same members after a merge.
                        payload = dict(overlay)
                        regions = list(dict.fromkeys([
                            *(payload.get("region_ids") or []),
                            *(base_payload.get("region_ids") or []),
                        ]))
                        if regions:
                            payload["region_ids"] = regions
                        if payload.get("candidate_id") is None and candidate_id is not None:
                            payload["candidate_id"] = candidate_id
                        _insert_archive_payload(connection, ordinal, payload, axis)
                        replaced.add(axis)
                    ordinal += 1
                for axis in overlay_order:
                    if axis in replaced:
                        continue
                    payload = dict(overlay_rows[axis])
                    if payload.get("candidate_id") is None:
                        payload["candidate_id"] = next_candidate_id
                        next_candidate_id += 1
                    _insert_archive_payload(connection, ordinal, payload, axis)
                    ordinal += 1
            finally:
                base_connection.close()
            connection.commit()
        finally:
            connection.close()
        temporary.replace(target)
    return {
        **reference, "row_count": ordinal,
    }


def _archive_row_count(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    finally:
        connection.close()


def _insert_archive_payload(connection: sqlite3.Connection, ordinal: int, payload: dict[str, Any],
                             axis: tuple[Any, Any, Any, Any], row_json: str | None = None) -> None:
    """Insert one candidate row from its JSON payload and its four-axis values."""
    connection.execute(
        "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            ordinal, payload.get("candidate_id"),
            axis[0], axis[1], axis[2], axis[3],
            payload.get("j_score"),
            payload.get("training_excess_return"),
            payload.get("max_principal_drawdown"),
            payload.get("validation_excess_original"), payload.get("validation_return_original"),
            payload.get("validation_excess_stress"), payload.get("validation_return_stress"),
            payload.get("neighborhood_mean"),
            payload.get("downside_risk_tau"),
            payload.get("pos_penalty"),
            None if payload.get("score_complete") is None else int(bool(payload.get("score_complete"))),
            row_json if row_json is not None else dumps(payload),
        ),
    )
    region_ids = payload.get("region_ids") or []
    if isinstance(region_ids, list):
        connection.executemany(
            "INSERT INTO candidate_regions VALUES(?,?)",
            [(ordinal, int(region_id)) for region_id in region_ids if isinstance(region_id, int)],
        )


def _insert_archive_row(connection: sqlite3.Connection, ordinal: int, row: sqlite3.Row) -> None:
    payload = json.loads(row["row_json"])
    axis = (row["buy_step"], row["sell_step"], row["buy_shares"], row["sell_shares"])
    _insert_archive_payload(connection, ordinal, payload, axis, row_json=row["row_json"])


def _detach_candidate_archives(root: Path, result: dict) -> dict:
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        return result
    for group_name in ("candidate_archives", "trial_archives", "search_candidate_archives"):
        group = evidence.get(group_name)
        if not isinstance(group, dict):
            continue
        for branch_name, value in list(group.items()):
            if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
                # Adaptive branches contain one decision entry per rebalance,
                # each with its own candidate/search archives.  Only flat lists
                # without archive-bearing decision metadata are candidate rows.
                decision_entries = any(
                    any(field in row for field in _V4_ARCHIVE_FIELDS)
                    for row in value
                )
                if not decision_entries:
                    group[branch_name] = _write_candidate_archive(root, value)
                else:
                    for item in value:
                        for field in _V4_ARCHIVE_FIELDS:
                            rows = item.get(field)
                            if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
                                item[field] = _write_candidate_archive(root, rows)
            elif isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    for field in _V4_ARCHIVE_FIELDS:
                        rows = item.get(field)
                        if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
                            item[field] = _write_candidate_archive(root, rows)
    return result


def _candidate_rows(root: Path, reference: dict[str, Any], *, offset: int = 0,
                    limit: int | None = None, sort_by: str = "ordinal",
                    descending: bool = False, region_id: int | None = None,
                    filters: dict[str, Any] | None = None,
                    in_filters: dict[str, list[Any]] | None = None,
                    ordinals: list[int] | tuple[int, ...] | None = None) -> dict[str, Any]:
    relative = reference.get(_CANDIDATE_REFERENCE)
    if not isinstance(relative, str):
        return {"rows": [], "total": 0, "offset": offset, "limit": limit}
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError("候选档案不存在")
    if sort_by not in _CANDIDATE_FIELDS:
        raise ValueError("不支持的候选排序字段")
    if ordinals is not None and not [int(value) for value in ordinals]:
        return {"rows": [], "total": 0, "offset": offset, "limit": limit}
    connection = sqlite3.connect(path)
    try:
        columns = _candidate_columns(connection)
        clause, arguments = _candidate_where(
            connection, region_id=region_id, filters=filters,
            in_filters=in_filters, ordinals=ordinals,
        )
        total = int(connection.execute("SELECT COUNT(*) FROM candidates c" + clause, arguments).fetchone()[0])
        sql = (
            f"SELECT c.ordinal,c.row_json FROM candidates c{clause} "
            f"ORDER BY {_candidate_expression(columns, sort_by)} {'DESC' if descending else 'ASC'},c.ordinal"
        )
        query_arguments = list(arguments)
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            query_arguments.extend([int(limit), int(offset)])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            query_arguments.append(int(offset))
        rows = []
        for row in connection.execute(sql, query_arguments):
            item = json.loads(row[1])
            item.setdefault("_archive_ordinal", int(row[0]))
            rows.append(item)
    finally:
        connection.close()
    return {"rows": rows, "total": total, "offset": offset, "limit": limit}


def _candidate_columns(connection: sqlite3.Connection) -> set[str]:
    return {row[1] for row in connection.execute("PRAGMA table_info(candidates)")}


def _candidate_expression(columns: set[str], name: str) -> str:
    if name in columns:
        return f"c.{name}"
    if name in _CANDIDATE_FIELDS:
        # Results written before the S4 index extension remain readable; the
        # expression fallback reads only rows selected by the surrounding
        # WHERE and LIMIT clauses.
        return f"json_extract(c.row_json, '$.{name}')"
    raise ValueError("不支持的候选字段")


def _candidate_where(connection: sqlite3.Connection, *, region_id: int | None = None,
                     filters: dict[str, Any] | None = None,
                     in_filters: dict[str, list[Any]] | None = None,
                     ordinals: list[int] | tuple[int, ...] | None = None) -> tuple[str, list[Any]]:
    """Build the shared WHERE clause used by archive reads and counts."""
    columns = _candidate_columns(connection)
    where: list[str] = []
    arguments: list[Any] = []
    if region_id is not None:
        where.append("c.ordinal IN (SELECT ordinal FROM candidate_regions WHERE region_id=?)")
        arguments.append(int(region_id))
    if ordinals is not None:
        values = [int(value) for value in ordinals]
        if not values:
            return " WHERE 0", []
        where.append("c.ordinal IN (" + ",".join("?" for _ in values) + ")")
        arguments.extend(values)
    for name, value in (filters or {}).items():
        if name not in _CANDIDATE_FIELDS:
            continue
        where.append(f"{_candidate_expression(columns, name)}=?")
        arguments.append(value)
    for name, values in (in_filters or {}).items():
        if name not in _CANDIDATE_AXIS_FIELDS or not values:
            continue
        where.append(f"c.{name} IN (" + ",".join("?" for _ in values) + ")")
        arguments.extend(values)
    clause = " WHERE " + " AND ".join(where) if where else ""
    return clause, arguments


def _candidate_total(root: Path, reference: dict[str, Any], *, region_id: int | None = None,
                     filters: dict[str, Any] | None = None,
                     in_filters: dict[str, list[Any]] | None = None) -> int:
    """Count the rows an archive query would return, without materialising them."""
    relative = reference.get(_CANDIDATE_REFERENCE)
    if not isinstance(relative, str):
        return 0
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError("候选档案不存在")
    connection = sqlite3.connect(path)
    try:
        clause, arguments = _candidate_where(
            connection, region_id=region_id, filters=filters, in_filters=in_filters,
        )
        return int(connection.execute("SELECT COUNT(*) FROM candidates c" + clause, arguments).fetchone()[0])
    finally:
        connection.close()


def _candidate_axis_values(root: Path, reference: dict[str, Any], names: list[str]) -> dict[str, list[float]]:
    relative = reference.get(_CANDIDATE_REFERENCE)
    if not isinstance(relative, str):
        return {}
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError("候选档案不存在")
    allowed = {"buy_step", "sell_step", "buy_shares", "sell_shares"}
    connection = sqlite3.connect(path)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(candidates)")}
        output = {}
        for name in names:
            if name not in allowed or name not in columns:
                continue
            output[name] = sorted(
                float(row[0]) for row in connection.execute(
                    f"SELECT DISTINCT {name} FROM candidates WHERE {name} IS NOT NULL ORDER BY {name}"
                )
            )
        return output
    finally:
        connection.close()


def _candidate_ordinals(root: Path, reference: dict[str, Any], *,
                        region_id: int | None = None, sort_by: str = "ordinal",
                        descending: bool = False) -> list[int]:
    relative = reference.get(_CANDIDATE_REFERENCE)
    if not isinstance(relative, str):
        return []
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError("候选档案不存在")
    connection = sqlite3.connect(path)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(candidates)")}
        allowed = {
            "ordinal", "candidate_id", "buy_step", "sell_step", "buy_shares",
            "sell_shares", "j_score", "training_excess_return", "max_principal_drawdown",
            "validation_excess_original", "validation_return_original",
            "validation_excess_stress", "validation_return_stress",
            "neighborhood_mean", "downside_risk_tau", "pos_penalty", "score_complete",
        }
        if sort_by not in allowed:
            raise ValueError("不支持的候选排序字段")
        order = f"c.{sort_by}" if sort_by in columns else f"json_extract(c.row_json, '$.{sort_by}')"
        if region_id is None:
            rows = connection.execute(
                f"SELECT ordinal FROM candidates c ORDER BY {order} {'DESC' if descending else 'ASC'},ordinal"
            )
        else:
            rows = connection.execute(
                f"SELECT ordinal FROM candidates c WHERE ordinal IN "
                f"(SELECT ordinal FROM candidate_regions WHERE region_id=?) "
                f"ORDER BY {order} {'DESC' if descending else 'ASC'},ordinal",
                (int(region_id),),
            )
        return [int(row[0]) for row in rows]
    finally:
        connection.close()


def _candidate_stats(root: Path, reference: dict[str, Any], *,
                     region_id: int | None = None) -> dict[str, int]:
    relative = reference.get(_CANDIDATE_REFERENCE)
    if not isinstance(relative, str):
        return {"total": 0, "j_complete": 0}
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError("候选档案不存在")
    where = ""
    arguments: list[Any] = []
    if region_id is not None:
        where = " WHERE ordinal IN (SELECT ordinal FROM candidate_regions WHERE region_id=?)"
        arguments.append(int(region_id))
    connection = sqlite3.connect(path)
    try:
        total = int(connection.execute("SELECT COUNT(*) FROM candidates" + where, arguments).fetchone()[0])
        columns = {row[1] for row in connection.execute("PRAGMA table_info(candidates)")}
        if "score_complete" in columns:
            j_where = where + (" AND " if where else " WHERE ") + "score_complete != 0 AND j_score IS NOT NULL"
        else:
            j_where = where + (" AND " if where else " WHERE ") + "j_score IS NOT NULL"
        j_complete = int(connection.execute("SELECT COUNT(*) FROM candidates" + j_where, arguments).fetchone()[0])
        return {"total": total, "j_complete": j_complete}
    finally:
        connection.close()


class Store:
    def __init__(self, root: str | Path, *, settings=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "catalog.sqlite"
        from .settings import load_settings

        configured = settings or load_settings()
        self.settings = configured
        self.market_settings = configured.market_data
        self.lock = FileLock(
            str(self.root / "sync.lock"),
            timeout=self.market_settings.source_lock_timeout_seconds,
        )
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS instruments(symbol TEXT PRIMARY KEY,name TEXT,exchange TEXT,board TEXT,listing_date TEXT,delisting_date TEXT);
                CREATE TABLE IF NOT EXISTS watchlist(symbol TEXT PRIMARY KEY,created_at TEXT);
                CREATE TABLE IF NOT EXISTS snapshots(version TEXT PRIMARY KEY,symbol TEXT,created_at TEXT,manifest TEXT);
                CREATE TABLE IF NOT EXISTS active_snapshots(symbol TEXT PRIMARY KEY,version TEXT);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT,status TEXT,progress REAL,message TEXT,error TEXT,result_id TEXT,payload TEXT,created_at TEXT,updated_at TEXT,cancel_requested INTEGER DEFAULT 0,pid INTEGER,claim_token TEXT,claimed_at TEXT,dedupe_key TEXT);
                CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,kind TEXT,symbol TEXT,created_at TEXT,title TEXT,path TEXT,protocol_version INTEGER,result_schema_version INTEGER,reproducible INTEGER);
                CREATE TABLE IF NOT EXISTS configs(id TEXT PRIMARY KEY,name TEXT,config TEXT,created_at TEXT);
                CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT);
                CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status,created_at);
            """)
            if "metadata" not in {row[1] for row in db.execute("PRAGMA table_info(instruments)")}:
                db.execute("ALTER TABLE instruments ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            job_columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            for column in ("claim_token", "claimed_at", "dedupe_key"):
                if column not in job_columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS jobs_active_dedupe ON jobs(dedupe_key) "
                "WHERE dedupe_key IS NOT NULL AND status IN ('queued','running')"
            )
            result_columns = {row[1] for row in db.execute("PRAGMA table_info(results)")}
            for column, kind in (
                ("protocol_version", "INTEGER"), ("result_schema_version", "INTEGER"),
                ("reproducible", "INTEGER"),
            ):
                if column not in result_columns:
                    db.execute(f"ALTER TABLE results ADD COLUMN {column} {kind}")
            # A NULL metadata tuple is valid for ordinary backtest and data
            # result files.  A persistent marker distinguishes those legitimate
            # values from rows that have not yet been inspected, preventing a
            # full reread of every historical result on each Store creation.
            metadata_migration = "results_metadata_v1"
            migrated = db.execute(
                "SELECT 1 FROM state WHERE key=?", (metadata_migration,),
            ).fetchone()
            if migrated is None:
                legacy_results = db.execute(
                    "SELECT id,path FROM results WHERE protocol_version IS NULL "
                    "AND result_schema_version IS NULL AND reproducible IS NULL"
                ).fetchall()
                for result_id, relative in legacy_results:
                    try:
                        payload = json.loads((self.root / relative).read_text(encoding="utf-8"))
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
                    provenance = payload.get("computation_provenance") if isinstance(payload, dict) else None
                    reproducible = provenance.get("reproducible") if isinstance(provenance, dict) else None
                    db.execute(
                        "UPDATE results SET protocol_version=?,result_schema_version=?,reproducible=? WHERE id=?",
                        (
                            payload.get("protocol_version") if isinstance(payload, dict) else None,
                            payload.get("result_schema_version") if isinstance(payload, dict) else None,
                            None if reproducible is None else int(bool(reproducible)), result_id,
                        ),
                    )
                db.execute(
                    "INSERT INTO state(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (metadata_migration, now_iso()),
                )
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sync_items(run_key TEXT,symbol TEXT,status TEXT,mode TEXT,daily_through TEXT,minute_through TEXT,error TEXT,updated_at TEXT,PRIMARY KEY(run_key,symbol));
                CREATE INDEX IF NOT EXISTS sync_items_status ON sync_items(run_key,status);
                CREATE TABLE IF NOT EXISTS market_auxiliary(
                    symbol TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    version TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL, PRIMARY KEY(symbol,kind)
                );
            """)
            db.executescript("""
                CREATE TABLE IF NOT EXISTS resource_leases(
                    job_id TEXT PRIMARY KEY, pid INTEGER NOT NULL, kind TEXT NOT NULL,
                    units INTEGER NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS resource_leases_kind ON resource_leases(kind);
            """)
            db.executescript("""
                CREATE TABLE IF NOT EXISTS research_registry(
                    registry_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    input_start TEXT,
                    input_end TEXT,
                    data_version TEXT,
                    evaluation_intervals TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    overlapping_prior_studies INTEGER NOT NULL DEFAULT 0,
                    exact_repeat_prior_studies INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS research_registry_symbol ON research_registry(symbol,created_at);
            """)
            registry_columns = {row[1] for row in db.execute("PRAGMA table_info(research_registry)")}
            for column in ("overlapping_prior_studies", "exact_repeat_prior_studies"):
                if column not in registry_columns:
                    db.execute(f"ALTER TABLE research_registry ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            sync_columns = {row[1] for row in db.execute("PRAGMA table_info(sync_items)")}
            for column in ("daily_status", "minute_status"):
                if column not in sync_columns:
                    db.execute(f"ALTER TABLE sync_items ADD COLUMN {column} TEXT NOT NULL DEFAULT 'pending'")

    @contextmanager
    def db(self):
        sqlite_timeout = int(self.market_settings.sqlite_timeout_seconds)
        connection = sqlite3.connect(self.db_path, timeout=sqlite_timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(f"PRAGMA busy_timeout={sqlite_timeout * 1000}")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def put_instrument(self, symbol: str, name: str = "", **metadata):
        from .rules import instrument_terms
        symbol = normalize_symbol(symbol)
        exchange = symbol.split(".")[1]
        board = "star" if symbol.startswith(("688", "689")) else "chinext" if symbol.startswith("3") else "main"
        with self.db() as db:
            old = db.execute("SELECT * FROM instruments WHERE symbol=?", (symbol,)).fetchone()
            merged = {**(json.loads(old["metadata"] or "{}") if old else {}), **{key: value for key, value in metadata.items() if value is not None}}
            merged = instrument_terms(symbol, {**merged, "name": name or (old["name"] if old else symbol)})
            db.execute("INSERT OR REPLACE INTO instruments(symbol,name,exchange,board,listing_date,delisting_date,metadata) VALUES(?,?,?,?,?,?,?)", (
                symbol, merged["name"], exchange, merged["board"],
                metadata.get("listing_date") or (old["listing_date"] if old else None),
                metadata.get("delisting_date") or (old["delisting_date"] if old else None),
                dumps(merged),
            ))

    def instruments(self, query: str = "", limit: int | None = 100, asset_type: str | None = None, offset: int = 0) -> list[dict]:
        with self.db() as db:
            rows = db.execute("SELECT * FROM instruments WHERE symbol LIKE ? OR name LIKE ? ORDER BY symbol", (f"%{query}%", f"%{query}%")).fetchall()
        items = [self._instrument(row) for row in rows]
        if asset_type:
            items = [item for item in items if item["asset_type"] == asset_type]
        return items[offset:] if limit is None else items[offset:offset + limit]

    @staticmethod
    def _instrument(row):
        from .rules import instrument_terms
        item = dict(row)
        if "has_data" in item:
            item["has_data"] = bool(item["has_data"])
        metadata = json.loads(item.pop("metadata", "{}") or "{}")
        return instrument_terms(item["symbol"], {**item, **metadata})

    def instrument(self, symbol):
        with self.db() as db:
            row = db.execute("SELECT * FROM instruments WHERE symbol=?", (normalize_symbol(symbol),)).fetchone()
        return self._instrument(row) if row else None

    def add_watch(self, symbol: str, name: str = ""):
        self.put_instrument(symbol, name)
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO watchlist VALUES(?,?)", (normalize_symbol(symbol), now_iso()))

    def watches(self):
        with self.db() as db:
            market_index = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='parquet_datasets'"
            ).fetchone()
            if market_index:
                rows = db.execute(
                    "SELECT i.*, EXISTS(SELECT 1 FROM parquet_datasets d "
                    "WHERE d.symbol=i.symbol AND d.row_count>0) AS has_data "
                    "FROM watchlist w JOIN instruments i USING(symbol) ORDER BY w.created_at"
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT i.* FROM watchlist w JOIN instruments i USING(symbol) ORDER BY w.created_at"
                ).fetchall()
            return [self._instrument(r) for r in rows]

    def sync_item(self, run_key, symbol, **values):
        with self.db() as db:
            row = db.execute("SELECT * FROM sync_items WHERE run_key=? AND symbol=?", (run_key, symbol)).fetchone()
            if not values:
                return dict(row) if row else None
            item = {"status": "pending", "mode": "history", "daily_through": None, "minute_through": None, "error": None, "daily_status": "pending", "minute_status": "pending", **(dict(row) if row else {}), **values}
            db.execute("INSERT OR REPLACE INTO sync_items(run_key,symbol,status,mode,daily_through,minute_through,error,updated_at,daily_status,minute_status) VALUES(?,?,?,?,?,?,?,?,?,?)", (run_key, symbol, item["status"], item["mode"], item["daily_through"], item["minute_through"], item["error"], now_iso(), item["daily_status"], item["minute_status"]))

    def sync_status(self, run_key=None, status=None, limit=100, offset=0):
        conditions, arguments = [], []
        for column, value in (("run_key", run_key), ("status", status)):
            if value:
                conditions.append(column + "=?")
                arguments.append(value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.db() as db:
            counts = {row[0]: row[1] for row in db.execute("SELECT status,COUNT(*) FROM sync_items" + where + " GROUP BY status", arguments)}
            rows = db.execute("SELECT * FROM sync_items" + where + " ORDER BY updated_at DESC,symbol LIMIT ? OFFSET ?", [*arguments, limit, offset]).fetchall()
        return {"items": [dict(row) for row in rows], "counts": counts, "total": sum(counts.values())}

    def state(self, key: str, value: Any = ...):
        with self.db() as db:
            if value is not ...:
                db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, dumps(value)))
                return value
            row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def save_market_auxiliary(self, symbol: str, kind: str, payload: Any, *, version="local", verified=False):
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO market_auxiliary(symbol,kind,payload,version,verified,updated_at) VALUES(?,?,?,?,?,?)",
                (normalize_symbol(symbol), kind, dumps(payload), version, int(verified), now_iso()),
            )

    def market_auxiliary(self, symbol: str, kind: str):
        with self.db() as db:
            row = db.execute(
                "SELECT * FROM market_auxiliary WHERE symbol=? AND kind=?",
                (normalize_symbol(symbol), kind),
            ).fetchone()
        return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    def write_frame(self, symbol: str, kind: str, frame: pd.DataFrame, date_column: str) -> list[str]:
        if frame.empty:
            return []
        frame = frame.copy().sort_values(date_column, kind="stable").reset_index(drop=True)
        paths = []
        years = pd.to_datetime(frame[date_column]).dt.year.fillna(0).astype(int)
        for year, part in frame.groupby(years, sort=True):
            folder = self.root / "parquet" / symbol / kind / str(year)
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / f"{uuid.uuid4().hex}.writing"
            table = pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False)
            pq.write_table(table, temporary, compression="zstd")
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            target = folder / f"{digest}.parquet"
            if target.exists():
                temporary.unlink()
            else:
                temporary.replace(target)
            paths.append(target.relative_to(self.root).as_posix())
        return paths

    def read_files(self, files: list[str], column: str | None = None, start=None, end=None) -> pd.DataFrame:
        if not files:
            return pd.DataFrame()
        virtual = [name for name in files if str(name).startswith("market:")]
        if virtual:
            parts = []
            for name in virtual:
                pieces = str(name).split(":")
                if len(pieces) != 3:
                    raise ValueError("无效的市场数据库引用：" + str(name))
                timeframe, symbol = pieces[1], pieces[2]
                from .market_data.repository import MarketDataRepository
                part = MarketDataRepository(self.root).bars(
                    symbol, timeframe, start=start, end=end
                )
                if not part.empty:
                    if column == "date" or timeframe in {"1day", "1week", "1month"}:
                        part["date"] = pd.to_datetime(part.timestamp).dt.normalize()
                    parts.append(part)
            return pd.concat(parts, ignore_index=True).sort_values(column or "timestamp").reset_index(drop=True) if parts else pd.DataFrame()
        paths = [str(self.root / name) for name in files]
        with duckdb.connect() as connection:
            relation = connection.read_parquet(paths, union_by_name=True)
            if column and start is not None:
                relation = relation.filter(f'"{column}" >= TIMESTAMP \'{pd.Timestamp(start).isoformat()}\'')
            if column and end is not None:
                relation = relation.filter(f'"{column}" < TIMESTAMP \'{(pd.Timestamp(end).normalize() + pd.Timedelta(days=1)).isoformat()}\'')
            return relation.df()

    def publish(self, symbol: str, minute: pd.DataFrame, daily: pd.DataFrame, actions: pd.DataFrame, calendar: pd.DataFrame, metadata: dict, profiles: dict, incremental: bool = False) -> dict:
        symbol = normalize_symbol(symbol)
        previous = self.snapshot(symbol) if incremental else None
        files = {}
        for key, kind, frame, column in [("minute", "1m", minute, "timestamp"), ("daily", "1d", daily, "date"), ("actions", "actions", actions, "ex_date"), ("calendar", "calendar", calendar, "date")]:
            old_paths = previous["files"].get(key, []) if previous else []
            if previous:
                if frame.empty:
                    files[key] = old_paths
                    continue
                years = {str(year) for year in pd.to_datetime(frame[column]).dt.year.unique()}
                touched = [path for path in old_paths if path.split("/")[3] in years]
                frame = pd.concat([self.read_files(touched), frame], ignore_index=True).sort_values(column, kind="stable").drop_duplicates(column, keep="last")
                untouched = [path for path in old_paths if path not in touched]
            else:
                untouched = []
            files[key] = sorted(untouched + self.write_frame(symbol, kind, frame, column))
        content = {"symbol": symbol, "files": files, "metadata": metadata, "profiles": profiles}
        version = hashlib.sha256(dumps(content).encode()).hexdigest()[:24]
        manifest = {**content, "version": version, "created_at": now_iso()}
        folder = self.root / "manifests"
        folder.mkdir(exist_ok=True)
        target = folder / f"{version}.json"
        if not target.exists():
            temporary = target.with_suffix(".writing")
            temporary.write_text(dumps(manifest), encoding="utf-8")
            temporary.replace(target)
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?)", (version, symbol, manifest["created_at"], dumps(manifest)))
            db.execute("INSERT OR REPLACE INTO active_snapshots VALUES(?,?)", (symbol, version))
        return self.snapshot(symbol, version)

    def put_market_snapshot(self, manifest: dict, symbol: str):
        """Register a snapshot whose bar payload lives in frequency databases."""
        symbol = normalize_symbol(symbol)
        version = manifest["version"]
        created = manifest.get("created_at") or now_iso()
        manifest = {**manifest, "created_at": created}
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?)", (version, symbol, created, dumps(manifest)))
            db.execute("INSERT OR REPLACE INTO active_snapshots VALUES(?,?)", (symbol, version))
        return self.snapshot(symbol, version)

    def snapshot(self, symbol: str, version: str | None = None, *, persist: bool = True) -> dict | None:
        symbol = normalize_symbol(symbol)
        with self.db() as db:
            if version:
                row = db.execute("SELECT manifest FROM snapshots WHERE symbol=? AND version=?", (symbol, version)).fetchone()
            else:
                row = db.execute("SELECT s.manifest FROM snapshots s JOIN active_snapshots a USING(version) WHERE a.symbol=?", (symbol,)).fetchone()
        if row:
            manifest = json.loads(row[0])
            if version is not None and manifest.get("market_data"):
                try:
                    from .market_data.repository import MarketDataRepository
                    repository = MarketDataRepository(self.root)
                    package = repository.package.manifest()
                    if (
                        manifest.get("package_id") != package.get("package_id")
                        or manifest.get("generation") != package.get("generation")
                    ):
                        return None
                    for timeframe, info in (manifest.get("market_versions") or {}).items():
                        dataset_manifest = repository.package.load_dataset_manifest(
                            symbol, timeframe, "none", int(info.get("version") or 0)
                        )
                        if dataset_manifest is None:
                            return None
                except (OSError, ValueError, KeyError, TypeError):
                    # A saved result remains viewable even when its exact source
                    # generation has been retired; it simply cannot be rerun.
                    return None
            if version is None and manifest.get("market_data"):
                from .market_data.repository import MarketDataRepository
                live = MarketDataRepository(self.root).active_manifest(symbol)
                if live and live.get("version") != manifest.get("version"):
                    instrument = self.instrument(symbol) or {"symbol": symbol, "name": symbol}
                    live["metadata"] = {**instrument, "is_demo": False, "warnings": []}
                    if persist:
                        self.put_market_snapshot(live, symbol)
                    return live
            return manifest
        if version is None:
            from .market_data.repository import MarketDataRepository
            live = MarketDataRepository(self.root).active_manifest(symbol)
            if live:
                # A live manifest can be returned to an API caller and then
                # stored in a job payload. Persist it before handing it out so
                # ``snapshot(symbol, version)`` can resolve the same version in
                # the worker process.
                instrument = self.instrument(symbol) or {"symbol": symbol, "name": symbol}
                live["metadata"] = {**instrument, "is_demo": False, "warnings": []}
                if persist:
                    self.put_market_snapshot(live, symbol)
            return live
        return None

    def new_job_info(self, kind: str, payload: dict, dedupe_key: str | None = None) -> tuple[str, bool]:
        """Create a job and report whether this call created a new row."""
        if (self.root / ".market-data-migration.lock").exists():
            raise RuntimeError("行情数据迁移期间暂停提交任务，请稍后重试")
        job_id = uuid.uuid4().hex
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if dedupe_key:
                existing = db.execute(
                    "SELECT id FROM jobs WHERE dedupe_key=? AND status IN ('queued','running') "
                    "ORDER BY created_at DESC LIMIT 1",
                    (dedupe_key,),
                ).fetchone()
                if existing:
                    return str(existing[0]), False
            stored_payload = dict(payload)
            if kind == "research_completion":
                # Initialise the checkpoint in the same transaction as the
                # queue row.  A later payload rewrite would erase the frozen
                # configuration snapshot attached by API submission.
                stored_payload.setdefault("checkpoint_id", job_id)
            # The queue is ordered by created_at and id. Keep the timestamp
            # strictly increasing inside the creation transaction so two jobs
            # created in the same clock tick still have an insertion order.
            created_at = now_iso()
            latest = db.execute("SELECT MAX(created_at) FROM jobs").fetchone()[0]
            if latest:
                stamp = datetime.fromisoformat(created_at)
                previous = datetime.fromisoformat(str(latest))
                if previous.tzinfo is None:
                    previous = previous.replace(tzinfo=stamp.tzinfo)
                if stamp <= previous:
                    created_at = (previous + timedelta(microseconds=1)).isoformat()
            db.execute(
                "INSERT INTO jobs(id,kind,status,progress,message,payload,created_at,updated_at,dedupe_key) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (job_id, kind, "queued", 0, "等待执行", dumps(stored_payload), created_at, created_at, dedupe_key),
            )
        return job_id, True

    def new_job(self, kind: str, payload: dict, dedupe_key: str | None = None) -> str:
        return self.new_job_info(kind, payload, dedupe_key=dedupe_key)[0]

    def update_job(self, job_id: str, *, require_claim: bool = False,
                   expected_claim_token: str | None = None, **values) -> bool:
        allowed = {
            "status", "progress", "message", "error", "result_id",
            "cancel_requested", "pid", "payload", "claim_token", "claimed_at",
            "dedupe_key",
        }
        values = {k: v for k, v in values.items() if k in allowed}
        if "payload" in values:
            values["payload"] = dumps(values["payload"])
        if not values:
            return False
        values["updated_at"] = now_iso()
        condition = " WHERE id=?"
        arguments = [*values.values(), job_id]
        if require_claim:
            condition += " AND claim_token=? AND status='running'"
            arguments.append(expected_claim_token)
        with self.db() as db:
            cursor = db.execute(
                f"UPDATE jobs SET {','.join(k+'=?' for k in values)}{condition}",
                arguments,
            )
            return cursor.rowcount > 0

    def claim_jobs(self, claim_token: str, max_workers: int, limit: int | None = None,
                   cancelled_ids: list[str] | None = None) -> list[str]:
        """Atomically cancel queued cancellations and claim oldest queued jobs."""
        stamp = now_iso()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if cancelled_ids is not None:
                cancelled_ids.extend(
                    str(row[0]) for row in db.execute(
                        "SELECT id FROM jobs WHERE status='queued' AND cancel_requested=1"
                    ).fetchall()
                )
            db.execute(
                "UPDATE jobs SET status='cancelled',message='排队期间已取消',updated_at=? "
                "WHERE status='queued' AND cancel_requested=1",
                (stamp,),
            )
            running = int(db.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0])
            capacity = max(0, int(max_workers) - running)
            if limit is not None:
                capacity = min(capacity, max(0, int(limit)))
            if capacity <= 0:
                return []
            rows = db.execute(
                "SELECT id FROM jobs WHERE status='queued' AND cancel_requested=0 "
                "ORDER BY created_at,id LIMIT ?",
                (capacity,),
            ).fetchall()
            claimed: list[str] = []
            for row in rows:
                job_id = row[0]
                cursor = db.execute(
                    "UPDATE jobs SET status='running',claim_token=?,claimed_at=?,updated_at=?,"
                    "message='正在启动计算进程',error=NULL WHERE id=? AND status='queued'",
                    (claim_token, stamp, stamp, job_id),
                )
                if cursor.rowcount:
                    claimed.append(job_id)
            return claimed

    def cancel_queued_job(self, job_id: str) -> bool:
        """Cancel only an unclaimed queue row, avoiding upload cleanup races."""
        with self.db() as db:
            cursor = db.execute(
                "UPDATE jobs SET status='cancelled',cancel_requested=1,"
                "message='排队期间已取消',updated_at=? "
                "WHERE id=? AND status='queued'",
                (now_iso(), job_id),
            )
            return cursor.rowcount > 0

    def claim_job(self, job_id: str, claim_token: str) -> bool:
        """Claim one known queued job for an in-process CLI worker."""
        stamp = now_iso()
        with self.db() as db:
            cursor = db.execute(
                "UPDATE jobs SET status='running',claim_token=?,claimed_at=?,updated_at=?,"
                "message='正在启动计算进程',error=NULL "
                "WHERE id=? AND status='queued' AND cancel_requested=0",
                (claim_token, stamp, stamp, job_id),
            )
            return cursor.rowcount > 0

    def active_job_count(self) -> int:
        with self.db() as db:
            return int(db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0])

    def jobs_by_status(self, status: str, *, after: tuple[str, str] | None = None,
                       limit: int = 200) -> list[dict]:
        """Cursor-paginated status query used by recovery and operations."""
        condition = "status=?"
        arguments: list[Any] = [status]
        if after is not None:
            condition += " AND (created_at>? OR (created_at=? AND id>?))"
            arguments.extend([after[0], after[0], after[1]])
        arguments.append(max(1, int(limit)))
        with self.db() as db:
            rows = db.execute(
                f"SELECT * FROM jobs WHERE {condition} ORDER BY created_at,id LIMIT ?",
                arguments,
            ).fetchall()
        values = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            item["cancel_requested"] = bool(item["cancel_requested"])
            values.append(item)
        return values

    def job(self, job_id: str) -> dict | None:
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["cancel_requested"] = bool(result["cancel_requested"])
        return result

    def jobs(self, limit=100, *, status: str | None = None,
             kind: str | None = None, offset: int = 0):
        arguments: list[Any] = []
        filters: list[str] = []
        if status:
            filters.append("status=?")
            arguments.append(status)
        if kind:
            filters.append("kind=?")
            arguments.append(kind)
        where = (" WHERE " + " AND ".join(filters)) if filters else ""
        arguments.extend([max(1, int(limit)), max(0, int(offset))])
        with self.db() as db:
            rows = db.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                arguments,
            ).fetchall()
        values = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            item["cancel_requested"] = bool(item["cancel_requested"])
            values.append(item)
        return values

    def job_history(self, limit: int = 50, *, offset: int = 0,
                    status: str | None = None, kind: str | None = None) -> dict:
        """Return one filtered history page without changing the polling contract."""
        filters: list[str] = []
        arguments: list[Any] = []
        if status:
            filters.append("status=?")
            arguments.append(status)
        if kind:
            filters.append("kind=?")
            arguments.append(kind)
        where = (" WHERE " + " AND ".join(filters)) if filters else ""
        page_limit = max(1, min(500, int(limit)))
        page_offset = max(0, int(offset))
        with self.db() as db:
            total = int(db.execute(f"SELECT COUNT(*) FROM jobs{where}", arguments).fetchone()[0])
            kind_rows = db.execute(
                "SELECT kind,COUNT(*) FROM jobs GROUP BY kind ORDER BY kind"
            ).fetchall()
            rows = db.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                [*arguments, page_limit, page_offset],
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            item["cancel_requested"] = bool(item["cancel_requested"])
            items.append(item)
        return {
            "items": items, "total": total, "offset": page_offset,
            "limit": page_limit, "status": status, "kind": kind,
            "kinds": [{"kind": str(row[0]), "count": int(row[1])} for row in kind_rows],
        }

    def active_jobs(self) -> list[dict]:
        """Return every queued/running job for lightweight polling."""
        with self.db() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','running') "
                "ORDER BY created_at DESC,id DESC"
            ).fetchall()
        values = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            item["cancel_requested"] = bool(item["cancel_requested"])
            values.append(item)
        return values

    def recent_submitted_symbols(self, limit: int = 10) -> list[dict[str, str]]:
        """Return the most recently submitted symbols from formal backtest jobs."""
        if limit <= 0:
            return []
        accepted_kinds = {"backtest", "optimization", "optimize", "batch", "batch_backtest"}
        with self.db() as db:
            rows = db.execute(
                "SELECT kind,payload,created_at FROM jobs ORDER BY created_at DESC,rowid DESC"
            ).fetchall()

        recent: dict[str, str] = {}

        def add(symbol: Any, submitted_at: Any) -> None:
            if not isinstance(symbol, str) or not isinstance(submitted_at, str) or symbol in ("",):
                return
            try:
                normalized = normalize_symbol(symbol)
            except (TypeError, ValueError):
                return
            recent.setdefault(normalized, submitted_at)

        def config_symbol(value: Any) -> str | None:
            if not isinstance(value, dict):
                return None
            return value.get("symbol") if isinstance(value.get("symbol"), str) else None

        for row in rows:
            if row["kind"] not in accepted_kinds:
                continue
            try:
                payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue

            # Older batch records kept the original submitted configs together.
            configs = payload.get("configs")
            if isinstance(configs, list):
                for config in configs:
                    if isinstance(config, dict) and isinstance(config.get("config"), dict):
                        config = config["config"]
                    add(config_symbol(config), row["created_at"])

            config = payload.get("config")
            if not isinstance(config, dict):
                config = payload
            base = config.get("base")
            symbol = config_symbol(base) if isinstance(base, dict) else config_symbol(config)
            add(symbol, row["created_at"])

            if len(recent) >= limit:
                break

        return [
            {"symbol": symbol, "last_submitted_at": submitted_at}
            for symbol, submitted_at in list(recent.items())[:limit]
        ]

    def save_result(self, kind: str, result: dict, symbol: str, title: str) -> str:
        result_id = uuid.uuid4().hex
        folder = self.root / "results"
        folder.mkdir(exist_ok=True)
        target = folder / f"{result_id}.json"
        payload = {**result, "id": result_id}
        if result.get("protocol_version") == 5:
            payload = _detach_candidate_archives(self.root, payload)
        target.write_text(dumps(payload), encoding="utf-8")
        provenance = result.get("computation_provenance") if isinstance(result.get("computation_provenance"), dict) else {}
        reproducible = provenance.get("reproducible") if provenance else None
        with self.db() as db:
            db.execute(
                "INSERT INTO results(id,kind,symbol,created_at,title,path,protocol_version,result_schema_version,reproducible) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    result_id, kind, symbol, now_iso(), title, str(target.relative_to(self.root)),
                    result.get("protocol_version"), result.get("result_schema_version"),
                    None if reproducible is None else int(bool(reproducible)),
                ),
            )
        return result_id

    def result(self, result_id: str) -> dict | None:
        with self.db() as db:
            row = db.execute("SELECT path FROM results WHERE id=?", (result_id,)).fetchone()
        if not row:
            return None
        result = json.loads((self.root / row[0]).read_text(encoding="utf-8"))
        return result

    def results(self, limit: int = 200, offset: int = 0, *, kind: str | None = None):
        where = ""
        arguments: list[Any] = []
        if kind:
            where = " WHERE kind=?"
            arguments.append(kind)
        arguments.extend([max(1, int(limit)), max(0, int(offset))])
        with self.db() as db:
            rows = [dict(r) for r in db.execute(
                "SELECT id,kind,symbol,created_at,title,protocol_version,result_schema_version,reproducible "
                f"FROM results{where} ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                arguments,
            )]
        for row in rows:
            supported = (
                row.get("protocol_version") is None
                or (row.get("protocol_version") == 5 and row.get("result_schema_version") == 2)
            )
            row["supported"] = supported
            row["reproducible"] = None if row.get("reproducible") is None else bool(row["reproducible"])
            if not supported:
                row["version_label"] = "不支持的历史版本"
        return rows

    def candidate_reference(self, result: dict, group: str, decision: str,
                            decision_index: int = 0, *, revision: int | None = None) -> dict | None:
        evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
        branch = (evidence.get(group) or {}).get(decision)
        if decision == "adaptive":
            if not isinstance(branch, list) or decision_index >= len(branch):
                raise LookupError("研究决策不存在")
            branch = branch[decision_index]
            if isinstance(branch, dict):
                field = {
                    "candidate_archives": "candidate_archive",
                    "trial_archives": "trial_archive",
                    "search_candidate_archives": "search_candidate_archive",
                }[group]
                branch = branch.get(field)
        if revision is not None:
            branch = self._revision_branch(result, decision, decision_index, int(revision))
        if not isinstance(branch, dict) or _CANDIDATE_REFERENCE not in branch:
            return None
        return branch

    def _revision_branch(self, result: dict, decision: str, decision_index: int,
                         revision: int) -> dict[str, Any]:
        """Resolve the archive a completion revision covers.

        A revision only covers the decision and decision index it was computed
        for.  Using it for another branch would show the pre-completion
        archive as if it had been revised, so the caller has to fail loudly.
        """
        metadata = self.completion_revision(result.get("id") or "", revision)
        if metadata is None:
            raise CompletionRevisionError(
                f"补算修订 {revision} 不存在",
                code="COMPLETION_REVISION_NOT_FOUND", revision=revision,
            )
        scope = {
            "decision": metadata.get("decision"),
            "decision_index": int(metadata.get("decision_index", 0) or 0),
        }
        if scope["decision"] != decision or scope["decision_index"] != int(decision_index):
            raise CompletionRevisionError(
                f"补算修订 {revision} 只覆盖决策 {scope['decision']} #{scope['decision_index']}，"
                f"不能用于 {decision} #{int(decision_index)}",
                code="COMPLETION_REVISION_OUT_OF_SCOPE", revision=revision, scope=scope,
            )
        merged = metadata.get("merged_candidate_archive")
        if not isinstance(merged, dict) or _CANDIDATE_REFERENCE not in merged:
            raise CompletionRevisionError(
                f"补算修订 {revision} 没有可用的候选档案",
                code="COMPLETION_REVISION_ARCHIVE_MISSING", revision=revision, scope=scope,
            )
        return merged

    def candidate_archive(self, result: dict, group: str, decision: str,
                          decision_index: int = 0, *, revision: int | None = None,
                          **query) -> dict:
        branch = self.candidate_reference(result, group, decision, decision_index, revision=revision)
        if branch is None:
            return {"rows": [], "total": 0, "offset": int(query.get("offset", 0)), "limit": query.get("limit")}
        return _candidate_rows(self.root, branch, **query)

    def candidate_total(self, result: dict, group: str, decision: str,
                        decision_index: int = 0, *, region_id: int | None = None,
                        filters: dict[str, Any] | None = None,
                        in_filters: dict[str, list[Any]] | None = None,
                        revision: int | None = None) -> int:
        """Count the rows an archive query would return, without materialising them."""
        reference = self.candidate_reference(result, group, decision, decision_index, revision=revision)
        if reference is None:
            return 0
        return _candidate_total(
            self.root, reference, region_id=region_id, filters=filters, in_filters=in_filters,
        )

    def merge_candidate_archives(self, base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        return _merge_candidate_archives(self.root, base, overlay)

    def candidate_axis_values(self, result: dict, group: str, decision: str,
                              decision_index: int = 0, *, names: list[str],
                              revision: int | None = None) -> dict[str, list[float]]:
        reference = self.candidate_reference(result, group, decision, decision_index, revision=revision)
        if reference is None:
            return {}
        return _candidate_axis_values(self.root, reference, names)

    def candidate_ordinals(self, result: dict, group: str, decision: str,
                           decision_index: int = 0, *, region_id: int | None = None,
                           sort_by: str = "ordinal", descending: bool = False,
                           revision: int | None = None) -> list[int]:
        reference = self.candidate_reference(result, group, decision, decision_index, revision=revision)
        if reference is None:
            return []
        return _candidate_ordinals(
            self.root, reference, region_id=region_id,
            sort_by=sort_by, descending=descending,
        )

    def candidate_stats(self, result: dict, group: str, decision: str,
                        decision_index: int = 0, *, region_id: int | None = None,
                        revision: int | None = None) -> dict[str, int]:
        reference = self.candidate_reference(result, group, decision, decision_index, revision=revision)
        if reference is None:
            return {"total": 0, "j_complete": 0}
        return _candidate_stats(self.root, reference, region_id=region_id)

    def iter_candidate_archive(self, result: dict, group: str, decision: str,
                               decision_index: int = 0, *, region_id: int | None = None,
                               sort_by: str = "ordinal", descending: bool = False,
                               revision: int | None = None):
        offset = 0
        page_size = 1000
        while True:
            page = self.candidate_archive(
                result, group, decision, decision_index,
                revision=revision,
                offset=offset, limit=page_size, sort_by=sort_by,
                descending=descending, region_id=region_id,
            )
            rows = page["rows"]
            if not rows:
                break
            yield from rows
            offset += len(rows)
            if offset >= page["total"]:
                break

    def iter_candidate_reference(self, reference: dict[str, Any], page_size: int = 1000):
        offset = 0
        while True:
            page = _candidate_rows(
                self.root, reference, offset=offset, limit=max(1, int(page_size)), sort_by="ordinal",
            )
            rows = page["rows"]
            if not rows:
                break
            yield from rows
            offset += len(rows)
            if offset >= page["total"]:
                break

    def completion_checkpoint(self, key: str) -> dict | None:
        path = self.root / "results" / ".completion_work" / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_completion_checkpoint(self, key: str, payload: dict) -> None:
        folder = self.root / "results" / ".completion_work"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{key}.json"
        # The read-modify-write cycle has to be serialised: two workers may
        # write the same checkpoint and the later write must not drop the
        # budget consumption or the processed points of the earlier one.
        with FileLock(str(target) + ".lock"):
            existing = self.completion_checkpoint(key) or {}
            merged = _merge_completion_checkpoint(existing, payload)
            temporary = folder / f".{key}.{uuid.uuid4().hex}.writing"
            temporary.write_text(dumps(merged), encoding="utf-8")
            temporary.replace(target)

    def completion_revisions(self, result_id: str) -> list[dict]:
        folder = self.root / "results" / f"{result_id}.completions"
        if not folder.is_dir():
            return []
        values = []
        for path in sorted(folder.glob("*.json")):
            try:
                values.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return values

    def completion_revision(self, result_id: str, revision: int) -> dict | None:
        return next(
            (item for item in self.completion_revisions(result_id) if int(item.get("revision", 0)) == int(revision)),
            None,
        )

    def save_completion_revision(self, result_id: str, payload: dict) -> dict:
        folder = self.root / "results" / f"{result_id}.completions"
        folder.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(folder / ".revision.lock"))
        with lock:
            revisions = [int(item.get("revision", 0)) for item in self.completion_revisions(result_id)]
            revision = max(revisions, default=0) + 1
            payload = {**payload, "revision": revision, "result_id": result_id, "created_at": now_iso()}
            target = folder / f"{revision:04d}.json"
            temporary = folder / f".{revision}.{uuid.uuid4().hex}.writing"
            temporary.write_text(dumps(payload), encoding="utf-8")
            temporary.replace(target)
        return payload

    def register_research(
        self,
        registry_id: str,
        protocol_hash: str,
        symbol: str,
        input_start: str | None,
        input_end: str | None,
        data_version: str | None,
        evaluation_intervals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Record a v5 study and count previously used evaluation dates.

        Intervals are stored under the protocol-neutral evaluation name used
        by current API payloads.
        """
        symbol = normalize_symbol(symbol)
        intervals = []
        seen = set()
        for item in evaluation_intervals:
            try:
                start = str(pd.Timestamp(item["start"]).date())
                end = str(pd.Timestamp(item["end"]).date())
                if end < start:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            intervals.append({"start": start, "end": end, **({"kind": item["kind"]} if item.get("kind") else {})})
        intervals.sort(key=lambda item: (item["start"], item["end"], item.get("kind", "")))
        encoded_intervals = dumps(intervals)
        with self.db() as db:
            # Serialize the read/overlap/insert sequence across concurrent jobs.
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM research_registry WHERE registry_id=?", (registry_id,)).fetchone()
            if existing:
                intervals = json.loads(existing["evaluation_intervals"] or "[]")
                overlap_count = int(existing["overlapping_prior_studies"])
                exact_count = int(existing["exact_repeat_prior_studies"])
                return {
                    "registry_id": existing["registry_id"],
                    "protocol_hash": existing["protocol_hash"],
                    "symbol": existing["symbol"],
                    "input_start": existing["input_start"],
                    "input_end": existing["input_end"],
                    "data_version": existing["data_version"],
                    "evaluation_intervals": intervals,
                    "overlapping_prior_studies": overlap_count,
                    "exact_repeat_prior_studies": exact_count,
                    "recorded_overlap_status": "overlap_detected" if overlap_count else "no_recorded_overlap",
                    "human_untouched_verified": False,
                    "idempotent_replay": True,
                }
            prior = db.execute(
                "SELECT registry_id,evaluation_intervals FROM research_registry WHERE symbol=?",
                (symbol,),
            ).fetchall()
            current_pairs = {(item["start"], item["end"]) for item in intervals}
            overlapping = set()
            exact = set()
            for row in prior:
                prior_intervals = json.loads(row["evaluation_intervals"] or "[]")
                if any(
                    pd.Timestamp(old["start"]) <= pd.Timestamp(new["end"])
                    and pd.Timestamp(new["start"]) <= pd.Timestamp(old["end"])
                    for old in prior_intervals
                    for new in intervals
                ):
                    overlapping.add(row["registry_id"])
                if current_pairs.intersection({(item["start"], item["end"]) for item in prior_intervals}):
                    exact.add(row["registry_id"])
            db.execute(
                "INSERT INTO research_registry VALUES(?,?,?,?,?,?,?,?,?,?)",
                (registry_id, protocol_hash, symbol, input_start, input_end, data_version, encoded_intervals, now_iso(), len(overlapping), len(exact)),
            )
        return {
            "registry_id": registry_id,
            "protocol_hash": protocol_hash,
            "symbol": symbol,
            "input_start": input_start,
            "input_end": input_end,
            "data_version": data_version,
            "evaluation_intervals": intervals,
            "overlapping_prior_studies": len(overlapping),
            "exact_repeat_prior_studies": len(exact),
            "recorded_overlap_status": "overlap_detected" if overlapping else "no_recorded_overlap",
            "human_untouched_verified": False,
            "idempotent_replay": False,
        }
