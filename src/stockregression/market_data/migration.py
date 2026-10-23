from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq
from filelock import FileLock

from .package_layout import PACKAGE_SCHEMA_VERSION

BASE_TIMEFRAMES = {"1day", "1min"}
CANONICAL_SYMBOL = re.compile(r"^\d{6}\.(SH|SZ)$")
CHUNK_SIZE = 4 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _connect(root: Path, *, readonly: bool = False) -> sqlite3.Connection:
    path = (root / "catalog.sqlite").resolve()
    if readonly:
        db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=60)
    else:
        db = sqlite3.connect(path, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=60000")
    return db


def _meta(db: sqlite3.Connection) -> dict[str, str]:
    return {str(row["key"]): str(row["value"]) for row in db.execute("SELECT key,value FROM parquet_meta")}


def _dependency_hash(symbol: str, timeframe: str, versions: dict, auxiliary: dict) -> str:
    suffix = ".SZ" if symbol.endswith(".SZ") else ".SH"
    calendar_path = f"auxiliary/calendars/{suffix.lstrip('.')}.parquet"
    actions_path = f"auxiliary/actions/{symbol}.parquet"
    payload = {
        "rules": 5,
        "calendar": (auxiliary.get(calendar_path) or {}).get("version"),
        "actions": (auxiliary.get(actions_path) or {}).get("version"),
        "daily": versions.get((symbol, "1day", "none")),
        "target": versions.get((symbol, timeframe, "none")),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _source_path(source: Path, raw: str) -> Path:
    path = Path(raw)
    path = path.resolve() if path.is_absolute() else (source / path).resolve()
    if not _inside(path, source):
        raise ValueError(f"分片引用逃出源目录：{raw}")
    if not path.is_file():
        raise FileNotFoundError(f"活跃分片缺失：{path}")
    return path


def _current_datasets(db: sqlite3.Connection) -> tuple[list[dict], list[dict]]:
    datasets = [dict(row) for row in db.execute(
        "SELECT * FROM parquet_datasets WHERE current_version>0 ORDER BY timeframe,symbol,adjustment"
    )]
    shards = []
    index_tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for dataset in datasets:
        identity = (
            dataset["symbol"], dataset["timeframe"], dataset["adjustment"],
            int(dataset["current_version"]),
        )
        version = db.execute(
            "SELECT * FROM parquet_versions WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?",
            identity,
        ).fetchone()
        if not version:
            raise ValueError(f"活跃版本记录缺失：{identity}")
        dataset["version_row"] = dict(version)
        if "parquet_shards" in index_tables:
            refs = [dict(row) for row in db.execute(
                """SELECT 'shard' ref_kind,period_key,period_start,period_end,path,sha256,
                          row_count,first_time,last_time,size,schema_version
                   FROM parquet_shards WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?
                   ORDER BY period_start,first_time,path""", identity,
            )]
        else:
            refs = [dict(row) for row in db.execute(
                """SELECT 'file' ref_kind,partition_key period_key,NULL period_start,NULL period_end,
                          path,sha256,row_count,first_time,last_time,size,schema_version
                   FROM parquet_files WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?
                   UNION ALL
                   SELECT 'increment',period_key,period_start,period_end,path,sha256,row_count,
                          first_time,last_time,size,schema_version
                   FROM parquet_increments WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?
                   ORDER BY period_start,first_time,path""",
                (*identity, *identity),
            )]
        if int(dataset["row_count"] or 0) and not refs:
            raise ValueError(f"活跃数据集没有分片引用：{identity}")
        if any(ref["ref_kind"] == "file" for ref in refs):
            raise ValueError(f"当前仍有旧式基础文件引用，停止迁移：{identity}")
        if dataset["timeframe"] not in BASE_TIMEFRAMES:
            continue
        if dataset["adjustment"] != "none":
            raise ValueError(f"发现不支持的复权口径：{identity}")
        if sum(int(ref["row_count"] or 0) for ref in refs) != int(dataset["row_count"] or 0):
            raise ValueError(f"分片行数与数据集总行数不符：{identity}")
        dataset["active_refs"] = refs
        for ref in refs:
            shards.append({
                "symbol": dataset["symbol"], "timeframe": dataset["timeframe"],
                "adjustment": dataset["adjustment"], **ref,
            })
    return [d for d in datasets if d["timeframe"] in BASE_TIMEFRAMES and d["adjustment"] == "none"], shards


def _unsupported(source: Path, active: set[Path], supported: set[str]) -> list[Path]:
    result = []
    root = source / "partitions"
    if not root.is_dir():
        return result
    for path in root.rglob("*.parquet"):
        if path.resolve() in active:
            continue
        symbol = next((part[7:] for part in path.parts if part.startswith("symbol=")), "")
        if symbol and (CANONICAL_SYMBOL.fullmatch(symbol) is None or symbol not in supported):
            result.append(path)
    return sorted(result)


def preflight(root: str | Path, source: str | Path, target: str | Path) -> dict:
    data_root = Path(root).expanduser().resolve()
    source_root = Path(source).expanduser().resolve()
    target_root = Path(target).expanduser().resolve()
    if not (data_root / "catalog.sqlite").is_file():
        raise ValueError(f"项目数据库不存在：{data_root / 'catalog.sqlite'}")
    if not source_root.is_dir() or source_root == target_root:
        raise ValueError("迁移源不存在，或源目录与目标目录相同")
    if not _inside(target_root, data_root / "dataset"):
        raise ValueError(f"目标目录必须位于项目 data/dataset 下：{target_root}")
    manifest_path = source_root / ".stockregression" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"迁移源缺少包清单：{manifest_path}")
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(old_manifest.get("schema_version") or 0) != 1:
        raise ValueError("迁移工具仅读取 schema 1 源包")
    db = _connect(data_root, readonly=True)
    try:
        meta = _meta(db)
        if meta.get("package_id") != old_manifest.get("package_id"):
            raise ValueError("数据库 package_id 与来源清单不一致")
        indexed_path = Path(meta.get("package_dir") or meta.get("source_package_dir") or "").resolve()
        if indexed_path != source_root:
            raise ValueError(f"数据库当前行情路径与迁移源不符：{indexed_path}")
        datasets, shards = _current_datasets(db)
        jobs = [dict(row) for row in db.execute(
            "SELECT id,kind,status,pid,created_at,updated_at FROM jobs "
            "WHERE status IN ('queued','running') ORDER BY created_at"
        )]
    finally:
        db.close()
    active_paths = {_source_path(source_root, row["path"]) for row in shards}
    supported = {str(row["symbol"]) for row in datasets}
    quarantine = _unsupported(source_root, active_paths, supported)
    periods = {}
    for timeframe in sorted(BASE_TIMEFRAMES):
        ds = [row for row in datasets if row["timeframe"] == timeframe]
        files = [row for row in shards if row["timeframe"] == timeframe]
        periods[timeframe] = {
            "datasets": len(ds), "shards": len(files),
            "rows": sum(int(row["row_count"] or 0) for row in ds),
            "bytes": sum(int(row["size"] or 0) for row in files),
        }
    return {
        "status": "preflight", "data_root": str(data_root),
        "source": str(source_root), "target": str(target_root),
        "package_id": meta["package_id"], "old_generation": meta.get("generation"),
        "dataset_count": len(datasets), "shard_count": len(shards),
        "shard_bytes": sum(int(row["size"] or 0) for row in shards),
        "quarantine_count": len(quarantine),
        "quarantine_bytes": sum(path.stat().st_size for path in quarantine),
        "auxiliary_count": len(old_manifest.get("auxiliary") or {}),
        "known_jobs": jobs, "by_timeframe": periods,
    }


def _digest(path: Path) -> tuple[str, int]:
    value = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            value.update(chunk)
            size += len(chunk)
    return value.hexdigest(), size


def _parquet_stats(path: Path) -> tuple[int, str | None, str | None]:
    parquet = pq.ParquetFile(path)
    rows = int(parquet.metadata.num_rows or 0)
    try:
        column = parquet.schema_arrow.names.index("timestamp")
    except ValueError as exc:
        raise ValueError(f"分片缺少 timestamp 列：{path}") from exc
    minima, maxima = [], []
    for group in range(parquet.metadata.num_row_groups):
        stat = parquet.metadata.row_group(group).column(column).statistics
        if stat is not None and stat.has_min_max:
            minima.append(pd.Timestamp(stat.min))
            maxima.append(pd.Timestamp(stat.max))
    first = min(minima).strftime("%Y-%m-%d %H:%M:%S") if minima else None
    last = max(maxima).strftime("%Y-%m-%d %H:%M:%S") if maxima else None
    return rows, first, last


def _copy(source: Path, target: Path) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename short for Windows installations where the
    # nested partition path plus a digest and UUID can hit MAX_PATH.
    temporary = target.with_name("." + uuid.uuid4().hex[:16] + ".writing")
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            while chunk := reader.read(CHUNK_SIZE):
                writer.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        shutil.copystat(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest(), size


def _stage_package(
    root: Path, source: Path, stage: Path, migration_id: str,
    progress: Callable[[str], None],
) -> dict:
    old_manifest = json.loads((source / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    db = _connect(root, readonly=True)
    try:
        datasets, shards = _current_datasets(db)
        for dataset in datasets:
            dataset["derived"] = db.execute(
                "SELECT * FROM parquet_derived WHERE symbol=? AND timeframe=? AND adjustment=?",
                (dataset["symbol"], dataset["timeframe"], dataset["adjustment"]),
            ).fetchone()
    finally:
        db.close()
    if stage.exists():
        raise FileExistsError(f"暂存目录已存在：{stage}")
    (stage / "partitions").mkdir(parents=True)
    for relative in (
        "datasets", "versions", "quality", "history", "staging",
        "transactions", "locks", "quarantine/unsupported-symbols",
    ):
        (stage / ".stockregression" / relative).mkdir(parents=True, exist_ok=True)
    for relative in ("calendars", "actions"):
        (stage / "auxiliary" / relative).mkdir(parents=True, exist_ok=True)

    copied: dict[str, dict] = {}
    total = len(shards)
    for index, row in enumerate(shards, start=1):
        source_file = _source_path(source, row["path"])
        digest, size = _digest(source_file)
        expected = str(row.get("sha256") or "")
        if expected and not expected.startswith("stat:") and expected != digest:
            raise ValueError(f"源分片摘要与索引不一致：{source_file}")
        if size != int(row["size"] or 0):
            raise ValueError(f"源分片大小与索引不一致：{source_file}")
        count, first, last = _parquet_stats(source_file)
        if count != int(row["row_count"] or 0):
            raise ValueError(f"Parquet 行数与索引不一致：{source_file}")
        if first and row.get("first_time") and pd.Timestamp(first) != pd.Timestamp(row["first_time"]):
            raise ValueError(f"Parquet 首条时间与索引不一致：{source_file}")
        if last and row.get("last_time") and pd.Timestamp(last) != pd.Timestamp(row["last_time"]):
            raise ValueError(f"Parquet 末条时间与索引不一致：{source_file}")
        relative = str(row["path"]).replace("\\", "/")
        target = stage / relative
        copied_digest, copied_size = _copy(source_file, target)
        if copied_digest != digest or copied_size != size:
            raise ValueError(f"分片复制校验失败：{target}")
        part = {
            "partition_key": row["period_key"],
            "period_start": row["period_start"], "period_end": row["period_end"],
            "path": relative, "sha256": digest, "size": size,
            "row_count": count, "first_time": first or row["first_time"],
            "last_time": last or row["last_time"],
            "schema_version": row["schema_version"] or "market-bar-v1",
        }
        copied[relative] = part
        if index % 5000 == 0 or index == total:
            progress(f"复制并校验行情分片 {index}/{total}")

    basic_files = {}
    for name in ("etf_basic_data.parquet", "stock_basic_data.parquet"):
        original = source / name
        if not original.is_file():
            if name == "etf_basic_data.parquet":
                raise FileNotFoundError(f"ETF 基本资料缺失：{original}")
            continue
        digest, size = _copy(original, stage / name)
        basic_files[name] = {
            "sha256": digest, "size": size,
            "mtime_ns": (stage / name).stat().st_mtime_ns, "kind": "basic",
        }

    auxiliary = {}
    for index, (relative, entry) in enumerate(sorted((old_manifest.get("auxiliary") or {}).items()), start=1):
        relative = relative.replace("\\", "/")
        original = _source_path(source, relative)
        digest, size = _copy(original, stage / relative)
        auxiliary[relative] = {
            "version": str(entry.get("version") or "migrated"),
            "verified": bool(entry.get("verified", False)),
            "sha256": digest, "size": size,
            "mtime_ns": (stage / relative).stat().st_mtime_ns,
        }
        if index % 500 == 0:
            progress(f"复制辅助资料 {index}/{len(old_manifest.get('auxiliary') or {})}")

    active_paths = {_source_path(source, row["path"]) for row in shards}
    supported = {row["symbol"] for row in datasets}
    quarantined = []
    for original in _unsupported(source, active_paths, supported):
        relative = original.relative_to(source / "partitions")
        timeframe = relative.parts[0] if relative.parts else "unknown"
        symbol_part = next((part[7:] for part in relative.parts if part.startswith("symbol=")), "unknown")
        period_part = next((part for part in relative.parts if part.startswith(("year=", "month="))), "unbucketed")
        destination_relative = (
            Path(".stockregression/quarantine/unsupported-symbols")
            / symbol_part / f"{timeframe}-{period_part}-{original.name}"
        )
        digest, size = _copy(original, stage / destination_relative)
        quarantined.append({
            "source_relative_path": str(original.relative_to(source)).replace("\\", "/"),
            "path": str(destination_relative).replace("\\", "/"),
            "sha256": digest, "size": size,
            "reason": "代码不符合正式六位 SH/SZ 规范，隔离且不登记",
        })

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in shards:
        grouped.setdefault((row["symbol"], row["timeframe"], row["adjustment"]), []).append(
            copied[str(row["path"]).replace("\\", "/")]
        )
    generation = uuid.uuid4().hex
    heads = {}
    for row in datasets:
        symbol, timeframe, adjustment = row["symbol"], row["timeframe"], row["adjustment"]
        key = (symbol, timeframe, adjustment)
        parts = sorted(grouped.get(key, []), key=lambda item: (item["period_start"], item["path"]))
        version_row = row["version_row"]
        try:
            preserved_report = json.loads(version_row["report"] or "{}")
        except (TypeError, ValueError):
            preserved_report = {}
        preserved_report["migration_quality_recheck"] = "pending"
        preserved_report["migration_id"] = migration_id
        payload = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "package_id": old_manifest["package_id"], "generation": generation,
            "symbol": symbol, "timeframe": timeframe, "adjustment": adjustment,
            "version": int(row["current_version"]), "row_count": int(row["row_count"] or 0),
            "raw_rows": int(row["raw_rows"] or 0),
            "synthetic_rows": int(row["synthetic_rows"] or 0),
            "pending_rows": int(row["pending_rows"] or 0),
            "first_time": row["first_time"], "last_time": row["last_time"],
            "completed_through": row["completed_through"],
            "status": row["status"], "reason": row["reason"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "source": version_row["source"], "created_at": _now(),
            "report": preserved_report,
            "shards": parts, "derived": None,
        }
        safe_symbol = "".join(c for c in symbol if c.isalnum() or c in ".-_")
        safe_adjustment = "".join(c for c in adjustment if c.isalnum() or c in "-_")
        version_number = int(row["current_version"])
        relative = f".stockregression/datasets/{timeframe}/{safe_symbol}/{safe_adjustment}/v{version_number}.json"
        _write_json(stage / relative, payload)
        size = sum(int(item["size"]) for item in parts)
        heads[f"{symbol}|{timeframe}|{adjustment}"] = {
            "version": version_number, "manifest": relative, "row_count": int(row["row_count"] or 0),
            "first_time": row["first_time"], "last_time": row["last_time"],
            "status": row["status"], "derived": None,
            "shard_count": len(parts), "shard_bytes": size,
        }
    top = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "package_id": old_manifest["package_id"], "generation": generation,
        "package_version": int(old_manifest.get("package_version") or 0) + 1,
        "created_at": _now(), "updated_at": _now(),
        "storage_layout": "time_partitioned", "partition_root": "partitions",
        "partition_scheme": {"1day": "year", "1min": "month"},
        "datasets": heads, "files": basic_files, "auxiliary": auxiliary,
        "shard_count": sum(head["shard_count"] for head in heads.values()),
        "shard_bytes": sum(head["shard_bytes"] for head in heads.values()),
        "quarantine": quarantined, "migration_id": migration_id,
    }
    _write_json(stage / ".stockregression" / "manifest.json", top)
    return {
        "package_id": top["package_id"], "generation": generation,
        "dataset_count": len(heads), "shard_count": top["shard_count"],
        "shard_bytes": top["shard_bytes"],
        "quarantine_count": len(quarantined),
        "quarantine_bytes": sum(row["size"] for row in quarantined),
        "auxiliary_count": len(auxiliary),
        "auxiliary_bytes": sum(int(row["size"]) for row in auxiliary.values()),
        "basic_files": basic_files,
    }


def _staged_package_summary(
    stage: Path, migration_id: str, baseline: dict,
) -> dict:
    """Validate a fully copied staging package before resuming its cutover.

    The migration report is only marked failed after an exception. Resumption
    is allowed only when the failed attempt retained its completed staging
    directory. The copy phase already checked every source and destination
    digest; this check confirms the retained manifest and all referenced files
    still match the frozen counts without repeating an 11+ GB hash pass.
    """
    manifest_path = stage / ".stockregression" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"已完成的暂存包清单缺失：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version") or 0) != PACKAGE_SCHEMA_VERSION:
        raise ValueError("恢复暂存包不是 schema 2")
    if manifest.get("storage_layout") != "time_partitioned":
        raise ValueError("恢复暂存包布局不是 time_partitioned")
    if manifest.get("migration_id") != migration_id:
        raise ValueError("暂存包 migration_id 与恢复请求不一致")
    if manifest.get("package_id") != baseline["package_id"]:
        raise ValueError("暂存包 package_id 与当前源包不一致")
    if not manifest.get("generation") or manifest.get("generation") == baseline.get("old_generation"):
        raise ValueError("暂存包 generation 无效")

    heads = manifest.get("datasets") or {}
    totals = {
        timeframe: {"datasets": 0, "shards": 0, "rows": 0, "bytes": 0}
        for timeframe in BASE_TIMEFRAMES
    }
    for key, head in heads.items():
        parts = str(key).split("|")
        if len(parts) != 3 or parts[1] not in BASE_TIMEFRAMES or parts[2] != "none":
            raise ValueError(f"暂存包包含非基础数据集：{key}")
        timeframe = parts[1]
        relative_manifest = str(head.get("manifest") or "")
        dataset_manifest_path = (stage / relative_manifest).resolve()
        if not relative_manifest or not _inside(dataset_manifest_path, stage) or not dataset_manifest_path.is_file():
            raise FileNotFoundError(f"暂存数据集版本清单缺失或越界：{relative_manifest}")
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if (
            dataset_manifest.get("package_id") != manifest.get("package_id")
            or dataset_manifest.get("generation") != manifest.get("generation")
            or f"{dataset_manifest.get('symbol')}|{dataset_manifest.get('timeframe')}|{dataset_manifest.get('adjustment')}" != key
            or int(dataset_manifest.get("version") or 0) != int(head.get("version") or 0)
        ):
            raise ValueError(f"暂存数据集版本身份不一致：{key}")
        shards = dataset_manifest.get("shards") or []
        if len(shards) != int(head.get("shard_count") or 0):
            raise ValueError(f"暂存数据集分片数不一致：{key}")
        dataset_bytes = 0
        dataset_rows = 0
        for shard in shards:
            relative = str(shard.get("path") or "")
            path = (stage / relative).resolve()
            if not relative or not _inside(path, stage) or not path.is_file():
                raise FileNotFoundError(f"暂存行情分片缺失或越界：{relative}")
            size = int(shard.get("size") or -1)
            if path.stat().st_size != size:
                raise ValueError(f"暂存行情分片大小不一致：{relative}")
            dataset_bytes += size
            dataset_rows += int(shard.get("row_count") or 0)
        if dataset_bytes != int(head.get("shard_bytes") or 0):
            raise ValueError(f"暂存数据集分片容量不一致：{key}")
        if dataset_rows != int(head.get("row_count") or 0):
            raise ValueError(f"暂存数据集行数不一致：{key}")
        totals[timeframe]["datasets"] += 1
        totals[timeframe]["shards"] += len(shards)
        totals[timeframe]["rows"] += dataset_rows
        totals[timeframe]["bytes"] += dataset_bytes

    if len(heads) != int(baseline["dataset_count"]):
        raise ValueError("暂存包数据集数与冻结基线不一致")
    if int(manifest.get("shard_count") or 0) != int(baseline["shard_count"]):
        raise ValueError("暂存包分片数与冻结基线不一致")
    if int(manifest.get("shard_bytes") or 0) != int(baseline["shard_bytes"]):
        raise ValueError("暂存包分片容量与冻结基线不一致")
    for timeframe, expected in baseline["by_timeframe"].items():
        if totals.get(timeframe) != expected:
            raise ValueError(f"暂存包 {timeframe} 汇总与冻结基线不一致")

    files = manifest.get("files") or {}
    auxiliary = manifest.get("auxiliary") or {}
    for relative, metadata in {**files, **auxiliary}.items():
        path = (stage / str(relative)).resolve()
        if not _inside(path, stage) or not path.is_file():
            raise FileNotFoundError(f"暂存辅助文件缺失或越界：{relative}")
        if path.stat().st_size != int(metadata.get("size") or -1):
            raise ValueError(f"暂存辅助文件大小不一致：{relative}")
    if len(auxiliary) != int(baseline["auxiliary_count"]):
        raise ValueError("暂存辅助资料数与冻结基线不一致")

    quarantine = manifest.get("quarantine") or []
    quarantine_bytes = 0
    for item in quarantine:
        relative = str(item.get("path") or "")
        path = (stage / relative).resolve()
        if not relative or not _inside(path, stage) or not path.is_file():
            raise FileNotFoundError(f"隔离分片缺失或越界：{relative}")
        size = int(item.get("size") or -1)
        if path.stat().st_size != size:
            raise ValueError(f"隔离分片大小不一致：{relative}")
        quarantine_bytes += size
    if len(quarantine) != int(baseline["quarantine_count"]):
        raise ValueError("隔离分片数与冻结基线不一致")
    if quarantine_bytes != int(baseline["quarantine_bytes"]):
        raise ValueError("隔离分片容量与冻结基线不一致")

    return {
        "package_id": manifest["package_id"], "generation": manifest["generation"],
        "dataset_count": len(heads), "shard_count": int(manifest["shard_count"]),
        "shard_bytes": int(manifest["shard_bytes"]),
        "quarantine_count": len(quarantine), "quarantine_bytes": quarantine_bytes,
        "auxiliary_count": len(auxiliary),
        "auxiliary_bytes": sum(int(row["size"]) for row in auxiliary.values()),
        "basic_files": files,
    }


def migrate_package(
    root: str | Path, source: str | Path, target: str | Path, *,
    apply: bool = False, resume_migration_id: str | None = None,
    progress: Callable[[str], None] = lambda _message: None,
) -> dict:
    data_root = Path(root).expanduser().resolve()
    source_root = Path(source).expanduser().resolve()
    target_root = Path(target).expanduser().resolve()
    baseline = preflight(data_root, source_root, target_root)
    if not apply:
        return baseline
    resuming = bool(resume_migration_id)
    migration_id = str(resume_migration_id) if resuming else (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    migration_dir = (data_root / "migrations" / migration_id).resolve()
    if not _inside(migration_dir, data_root / "migrations"):
        raise ValueError("migration_id 路径越界")
    if resuming:
        report_path = migration_dir / "migration.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"恢复迁移报告缺失：{report_path}")
        previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        if previous_report.get("status") != "failed" or not previous_report.get("rolled_back"):
            raise ValueError("只允许恢复已明确回退的失败迁移")
        if Path(previous_report.get("source", "")).resolve() != source_root:
            raise ValueError("恢复请求的源目录与失败迁移不一致")
        if Path(previous_report.get("target", "")).resolve() != target_root:
            raise ValueError("恢复请求的目标目录与失败迁移不一致")
        stage = Path(previous_report.get("stage_path") or "").resolve()
        expected_stage = (target_root.parent / f".{target_root.name}-{migration_id}.staging").resolve()
        if stage != expected_stage or not stage.is_dir():
            raise ValueError("失败迁移没有可恢复的完整暂存目录")
        for key in ("package_id", "old_generation", "dataset_count", "shard_count", "shard_bytes",
                    "auxiliary_count", "quarantine_count", "quarantine_bytes", "by_timeframe"):
            if previous_report.get(key) != baseline.get(key):
                raise ValueError(f"当前冻结基线与失败迁移不同：{key}")
        stage_manifest_path = stage / ".stockregression" / "manifest.json"
        if not stage_manifest_path.is_file():
            raise FileNotFoundError(f"恢复暂存清单缺失：{stage_manifest_path}")
        built = _staged_package_summary(stage, migration_id, baseline)
        backup_path = migration_dir / "catalog.before-resume.sqlite"
        if backup_path.exists():
            raise FileExistsError(f"恢复数据库备份已存在：{backup_path}")
    else:
        migration_dir.mkdir(parents=True, exist_ok=False)
        backup_path = migration_dir / "catalog.before.sqlite"
        stage = target_root.parent / f".{target_root.name}-{migration_id}.staging"
        previous_report = None
    replaced = migration_dir / "replaced-package"
    target_was_present = target_root.exists()
    marker = data_root / ".market-data-migration.lock"
    report_path = migration_dir / "migration.json"
    if resuming and not target_was_present:
        raise ValueError("恢复切换前的旧目标目录缺失")
    if resuming and replaced.exists():
        raise FileExistsError(f"恢复迁移的回退目录已存在：{replaced}")
    with FileLock(str(data_root / "migration.lock"), timeout=60):
        marker.write_text(migration_id, encoding="utf-8")
        try:
            # Apply the application's established orphan recovery before
            # freezing the catalog; live jobs cause a clean stop below.
            from ..jobs import JobManager
            from ..storage import Store

            JobManager(str(data_root))._recover_orphans(Store(data_root), startup_grace=0)
            baseline = preflight(data_root, source_root, target_root)
            if baseline["known_jobs"]:
                raise RuntimeError("仍有排队或运行中的任务；迁移停止且不会切换数据包")
            if resuming:
                for key in ("package_id", "old_generation", "dataset_count", "shard_count", "shard_bytes",
                            "auxiliary_count", "quarantine_count", "quarantine_bytes", "by_timeframe"):
                    if previous_report.get(key) != baseline.get(key):
                        raise ValueError(f"恢复时冻结基线发生变化：{key}")
            checkpoint = _connect(data_root)
            try:
                checkpoint.execute("PRAGMA wal_checkpoint(FULL)")
            finally:
                checkpoint.close()
            current = _connect(data_root, readonly=True)
            backup = sqlite3.connect(backup_path)
            try:
                current.backup(backup)
            finally:
                backup.close()
                current.close()
            if resuming:
                attempts = migration_dir / "attempts"
                attempts.mkdir(exist_ok=True)
                failed_attempt = attempts / "failed-before-resume.json"
                if failed_attempt.exists():
                    raise FileExistsError(f"失败尝试审计副本已存在：{failed_attempt}")
                shutil.copy2(report_path, failed_attempt)
                _write_json(report_path, {
                    **baseline, "status": "resuming", "migration_id": migration_id,
                    "database_backup": str(backup_path), "stage_path": str(stage),
                    "resumed_at": _now(), "previous_error": previous_report.get("error"),
                })
                progress(f"复用已完成暂存包 {migration_id}，跳过分片复制")
            else:
                _write_json(report_path, {
                    **baseline, "status": "copying", "migration_id": migration_id,
                    "database_backup": str(backup_path), "started_at": _now(),
                })
                progress(
                    f"基线冻结：{baseline['dataset_count']} 个数据集，"
                    f"{baseline['shard_count']} 个活跃分片"
                )
                built = _stage_package(data_root, source_root, stage, migration_id, progress)
            if built["dataset_count"] != baseline["dataset_count"] or built["shard_count"] != baseline["shard_count"]:
                raise ValueError("暂存包数量与冻结基线不一致")
            if target_root.exists():
                if replaced.exists():
                    raise FileExistsError(f"旧目标回退目录已存在：{replaced}")
                os.replace(target_root, replaced)
            try:
                os.replace(stage, target_root)
            except Exception:
                if replaced.exists() and not target_root.exists():
                    os.replace(replaced, target_root)
                raise

            db = _connect(data_root)
            try:
                db.execute("BEGIN IMMEDIATE")
                old_auxiliary = json.loads(
                    (source_root / ".stockregression" / "manifest.json").read_text(encoding="utf-8")
                ).get("auxiliary") or {}
                versions = {
                    (str(row["symbol"]), str(row["timeframe"]), str(row["adjustment"])):
                        int(row["current_version"])
                    for row in db.execute("SELECT * FROM parquet_datasets")
                }
                validations = db.execute(
                    "SELECT symbol,timeframe,adjustment,trade_date,rules_version,dependency_hash "
                    "FROM parquet_day_validation"
                ).fetchall()
                for record in validations:
                    expected = _dependency_hash(
                        str(record["symbol"]), str(record["timeframe"]), versions, old_auxiliary
                    )
                    if (
                        int(record["rules_version"] or 0) != 5
                        or record["dependency_hash"] != expected
                    ):
                        db.execute(
                            "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? "
                            "AND adjustment=? AND trade_date=?",
                            (record["symbol"], record["timeframe"], record["adjustment"], record["trade_date"]),
                        )
                coverages = db.execute(
                    "SELECT symbol,timeframe,adjustment,start_time,end_time,report FROM parquet_coverage"
                ).fetchall()
                for record in coverages:
                    try:
                        report = json.loads(record["report"] or "{}")
                    except ValueError:
                        report = {}
                    key = (str(record["symbol"]), str(record["timeframe"]), str(record["adjustment"]))
                    if report.get("package_id") != baseline["package_id"] or report.get("version") != versions.get(key):
                        db.execute(
                            "DELETE FROM parquet_coverage WHERE symbol=? AND timeframe=? AND adjustment=? "
                            "AND start_time=? AND end_time=?",
                            (record["symbol"], record["timeframe"], record["adjustment"], record["start_time"], record["end_time"]),
                        )
                index_tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for table in (
                    "parquet_shards", "parquet_increments", "parquet_source_archives",
                    "parquet_derived",
                    "parquet_files", "parquet_versions", "parquet_datasets",
                    "active_snapshots",
                ):
                    if table in index_tables:
                        db.execute(f"DELETE FROM {table}")
                for key, value in (
                    ("package_dir", str(target_root)),
                    ("source_package_dir", str(target_root)),
                    ("package_id", built["package_id"]),
                    ("generation", built["generation"]),
                ):
                    db.execute(
                        "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES(?,?)",
                        (key, value),
                    )
                db.commit()
            finally:
                db.close()

            from .repository import MarketDataRepository
            repository = MarketDataRepository(data_root, package_dir=target_root)
            index = repository.package_info()
            if index.get("index_status") != "ready":
                raise RuntimeError("包级清单已安装，但 SQLite 行情索引未能重建")
            index.update({
                "status": "completed", "datasets": built["dataset_count"],
                "versions": built["dataset_count"], "shards": built["shard_count"],
            })
            result = {
                **baseline, **built, "status": "applied", "migration_id": migration_id,
                "target": str(target_root),
                "replaced_package": str(replaced) if replaced.exists() else None,
                "database_backup": str(backup_path), "index": index,
                "completed_at": _now(), "verification": "required",
            }
            if resuming:
                result["resumed_from_failed_attempt"] = True
                result["previous_error"] = previous_report.get("error")
            _write_json(report_path, result)
            return result
        except Exception as exc:
            rollback_error = None
            failed_package = migration_dir / "failed-package"
            try:
                if replaced.exists():
                    if target_root.exists():
                        if failed_package.exists():
                            raise FileExistsError(f"失败包留存目录已存在：{failed_package}")
                        os.replace(target_root, failed_package)
                    os.replace(replaced, target_root)
                elif not target_was_present and target_root.exists():
                    if failed_package.exists():
                        raise FileExistsError(f"失败包留存目录已存在：{failed_package}")
                    os.replace(target_root, failed_package)
                if backup_path.is_file():
                    restored = data_root / f"catalog.{migration_id}.restoring"
                    shutil.copy2(backup_path, restored)
                    os.replace(restored, data_root / "catalog.sqlite")
                    for suffix in ("-wal", "-shm"):
                        Path(str(data_root / "catalog.sqlite") + suffix).unlink(missing_ok=True)
            except Exception as rollback_exc:
                rollback_error = str(rollback_exc)
            _write_json(report_path, {
                **baseline, "status": "failed", "migration_id": migration_id,
                "error": str(exc), "failed_at": _now(),
                "stage_path": str(stage) if stage.exists() else None,
                "replaced_package": str(replaced) if replaced.exists() else None,
                "target_exists": target_root.exists(),
                "rolled_back": rollback_error is None and backup_path.is_file(),
                "failed_package": str(failed_package) if failed_package.exists() else None,
                "rollback_error": rollback_error,
            })
            raise
        finally:
            marker.unlink(missing_ok=True)


def verify_package(
    root: str | Path, package_dir: str | Path, *, full: bool = False,
    progress: Callable[[str], None] = lambda _message: None,
) -> dict:
    data_root = Path(root).expanduser().resolve()
    package_root = Path(package_dir).expanduser().resolve()
    manifest_path = package_root / ".stockregression" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"数据包清单缺失：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version") or 0) != PACKAGE_SCHEMA_VERSION:
        raise ValueError("verify-package 仅接受 schema 2 分片包")
    if manifest.get("storage_layout") != "time_partitioned":
        raise ValueError("数据包存储布局不是 time_partitioned")
    db = _connect(data_root, readonly=True)
    try:
        meta = _meta(db)
        if Path(meta.get("package_dir", "")).resolve() != package_root:
            raise ValueError("SQLite package_dir 与目标路径不一致")
        if meta.get("package_id") != manifest.get("package_id"):
            raise ValueError("SQLite package_id 与包清单不一致")
        if meta.get("generation") != manifest.get("generation"):
            raise ValueError("SQLite generation 与包清单不一致")
        datasets, shards = _current_datasets(db)
        for row in datasets:
            key = f"{row['symbol']}|{row['timeframe']}|{row['adjustment']}"
            head = manifest.get("datasets", {}).get(key)
            if not head or int(head.get("version") or 0) != int(row["current_version"]):
                raise ValueError(f"索引版本与包级清单不一致：{key}")
    finally:
        db.close()
    expected_base_count = sum(
        1 for key in manifest.get("datasets", {})
        if len(key.split("|")) == 3 and key.split("|")[1] in BASE_TIMEFRAMES
    )
    if len(datasets) != expected_base_count:
        raise ValueError("基础数据集数与包级清单不一致")
    if len(shards) != int(manifest.get("shard_count") or 0):
        raise ValueError("基础分片数与包级清单不一致")

    failures = []
    total_bytes = 0
    sample_rows = []
    for index, row in enumerate(shards, start=1):
        path = (package_root / str(row["path"])).resolve()
        if not _inside(path, package_root) or not path.is_file():
            failures.append(f"分片缺失或路径越界：{path}")
            continue
        try:
            stat = path.stat()
            if stat.st_size != int(row["size"] or -1):
                raise ValueError("文件大小不一致")
            if full:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(CHUNK_SIZE):
                        digest.update(chunk)
                if digest.hexdigest() != str(row.get("sha256") or ""):
                    raise ValueError("SHA-256 不一致")
                count, first, last = _parquet_stats(path)
                if count != int(row["row_count"] or 0):
                    raise ValueError("Parquet 行数不一致")
                for field, actual in (("first_time", first), ("last_time", last)):
                    if row.get(field) and actual and pd.Timestamp(row[field]) != pd.Timestamp(actual):
                        raise ValueError(f"Parquet {field} 不一致")
            total_bytes += stat.st_size
            if index % 5000 == 0 or index == len(shards):
                progress(f"校验活跃分片 {index}/{len(shards)}")
        except Exception as exc:
            failures.append(f"{path}: {exc}")

    for relative, metadata in (manifest.get("files") or {}).items():
        path = (package_root / relative).resolve()
        if not _inside(path, package_root) or not path.is_file():
            failures.append(f"基本资料缺失或越界：{path}")
        elif full and hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
            failures.append(f"基本资料 SHA-256 不一致：{path}")
    for relative, metadata in (manifest.get("auxiliary") or {}).items():
        path = (package_root / relative).resolve()
        if not _inside(path, package_root) or not path.is_file():
            failures.append(f"辅助资料缺失或越界：{path}")
        elif full and hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
            failures.append(f"辅助资料 SHA-256 不一致：{path}")
    if failures:
        raise ValueError("数据包校验失败：" + "；".join(failures[:12]))

    result = {
        "status": "passed", "full": bool(full), "dataset_count": len(datasets),
        "shard_count": len(shards), "shard_bytes": total_bytes,
        "rows": sum(int(row["row_count"] or 0) for row in datasets),
        "package_id": manifest["package_id"], "generation": manifest["generation"],
        "package_dir": str(package_root), "checked_at": _now(),
        "quality_disposition": "旧代际校验未继承，需重新校验",
    }
    if full:
        from .repository import MarketDataRepository
        repository = MarketDataRepository(data_root, package_dir=package_root)
        sample = [row for row in datasets if row["timeframe"] in BASE_TIMEFRAMES]
        if len(sample) > 32:
            stride = max(1, len(sample) // 32)
            sample = sample[::stride][:32]
        for row in sample:
            data = repository.last_bars(
                str(row["symbol"]), str(row["timeframe"]), limit=3,
                fields=("timestamp", "open", "high", "low", "close", "volume"),
            )
            if data.empty or len(data) > 3:
                raise ValueError(f"分片查询抽验失败：{row['symbol']} {row['timeframe']}")
            sample_rows.append({
                "symbol": row["symbol"], "timeframe": row["timeframe"],
                "rows": len(data), "last_time": str(data.timestamp.iloc[-1]),
            })
        result["sample_queries"] = sample_rows
    migration_id = manifest.get("migration_id")
    if migration_id:
        report_path = data_root / "migrations" / str(migration_id) / "verify.json"
        _write_json(report_path, result)
    return result


def _service_smoke(root: Path, package_root: Path) -> dict:
    from ..models import BacktestConfig
    from ..storage import Store
    from .service import MarketDataService

    service = MarketDataService(Store(root), package_dir=package_root)
    info = service.package_info()
    if info.get("status") != "ready" or info.get("index_status") != "ready":
        raise ValueError(f"数据管理接口状态异常：{info}")
    listing = service.datasets(limit=8)
    if not listing.get("items"):
        raise ValueError("数据管理列表没有数据集")
    db = _connect(root, readonly=True)
    try:
        candidates = db.execute(
            """SELECT m.symbol,m.last_time FROM parquet_datasets m
               WHERE m.timeframe='1min' AND EXISTS(
                 SELECT 1 FROM parquet_datasets d WHERE d.symbol=m.symbol AND d.timeframe='1day'
               ) ORDER BY CASE WHEN m.symbol='510300.SH' THEN 0 ELSE 1 END,m.symbol LIMIT 64"""
        ).fetchall()
    finally:
        db.close()
    selected = None
    readiness = None
    for row in candidates:
        symbol = str(row["symbol"])
        end = pd.Timestamp(row["last_time"]).date()
        start = end - pd.Timedelta(days=20)
        config = BacktestConfig(
            schema_version=3, symbol=symbol, start_date=start, end_date=end,
            frequency="1m",
        )
        try:
            readiness = service.readiness(config)
        except Exception:
            continue
        if readiness.get("ready"):
            selected = (symbol, start, end)
            break
        # The migration retains only quality rows whose dependency fingerprint
        # and rule version still match. Revalidate this short smoke window when
        # its prior record was not eligible for inheritance.
        try:
            for timeframe in ("1day", "1min"):
                service.revalidate(
                    symbol, timeframe, start=start, end=end,
                    mode="incremental", progress=lambda *_: None,
                )
            readiness = service.readiness(config)
        except Exception:
            continue
        if readiness.get("ready"):
            selected = (symbol, start, end)
            break
    if selected is None:
        raise ValueError("没有找到可通过近期回测准备检查的基础行情数据集")
    symbol, start, end = selected
    minute = service.repository.last_bars(
        symbol, "1min", limit=32,
        fields=("timestamp", "open", "high", "low", "close", "volume"),
    )
    if minute.empty:
        raise ValueError("分钟行情读取冒烟检查失败")
    lower = pd.Timestamp(end) - pd.Timedelta(days=5)
    service.ensure_timeframe(symbol, "5min", start=lower, end=end)
    aggregated = service.repository.bars(
        symbol, "5min", start=lower, end=pd.Timestamp(end) + pd.Timedelta(days=1),
    )
    if aggregated.empty:
        raise ValueError("5 分钟派生聚合冒烟检查失败")
    return {
        "package": "ready", "inventory_items": len(listing["items"]),
        "symbol": symbol, "minute_rows": len(minute),
        "aggregation_rows": len(aggregated),
        "backtest_readiness": bool(readiness.get("ready")),
        "backtest_gaps": readiness.get("gaps", []),
        "range": [str(start), str(end)],
    }


def verify_source_independent(
    root: str | Path, package_dir: str | Path, *, full: bool = True,
    progress: Callable[[str], None] = lambda _message: None,
) -> dict:
    data_root = Path(root).expanduser().resolve()
    package_root = Path(package_dir).expanduser().resolve()
    manifest = json.loads((package_root / ".stockregression" / "manifest.json").read_text(encoding="utf-8"))
    migration_id = str(manifest.get("migration_id") or "")
    if not migration_id:
        raise ValueError("schema 2 清单缺少 migration_id，无法验证迁移来源隔离")
    migration_dir = (data_root / "migrations" / migration_id).resolve()
    report_path = migration_dir / "migration.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"迁移报告缺失：{report_path}")
    migration = json.loads(report_path.read_text(encoding="utf-8"))
    source = Path(migration["source"]).resolve()
    hold = migration_dir / "source-package-hold"
    if hold.exists():
        raise FileExistsError(f"来源隔离目录已存在：{hold}")
    if source.exists():
        if source == package_root or _inside(source, data_root):
            raise ValueError(f"来源隔离路径校验失败：{source}")
        os.replace(source, hold)
    try:
        result = verify_package(data_root, package_root, full=full, progress=progress)
        smoke = _service_smoke(data_root, package_root)
        result.update({
            "source_unavailable": not source.exists(),
            "source_hold_path": str(hold) if hold.exists() else None,
            "service_smoke": smoke,
        })
        _write_json(migration_dir / "verify.json", result)
        migration["source_unavailable_smoke"] = smoke
        migration["source_hold_path"] = str(hold) if hold.exists() else None
        _write_json(report_path, migration)
        return result
    except Exception:
        if hold.exists() and not source.exists():
            os.replace(hold, source)
        raise


def cleanup_verified(
    root: str | Path, package_dir: str | Path, migration_id: str, *,
    delete_source: bool = False, delete_replaced: bool = False,
) -> dict:
    data_root = Path(root).expanduser().resolve()
    package_root = Path(package_dir).expanduser().resolve()
    migration_dir = (data_root / "migrations" / migration_id).resolve()
    if not migration_dir.is_dir() or not _inside(migration_dir, data_root / "migrations"):
        raise ValueError("迁移报告目录无效")
    report_path = migration_dir / "migration.json"
    verify_path = migration_dir / "verify.json"
    if not report_path.is_file() or not verify_path.is_file():
        raise ValueError("清理需要 migration.json 和 verify.json")
    migration = json.loads(report_path.read_text(encoding="utf-8"))
    verified = json.loads(verify_path.read_text(encoding="utf-8"))
    if (
        migration.get("status") != "applied" or verified.get("status") != "passed"
        or not verified.get("full") or not verified.get("source_unavailable")
        or not verified.get("service_smoke", {}).get("backtest_readiness")
    ):
        raise ValueError("清理前必须通过全量校验、源目录隔离和服务冒烟检查")
    if Path(migration.get("target", "")).resolve() != package_root:
        raise ValueError("迁移报告目标路径与指定数据包不一致")
    db = _connect(data_root, readonly=True)
    try:
        meta = _meta(db)
        if Path(meta.get("package_dir", "")).resolve() != package_root:
            raise ValueError("SQLite 仍指向其他数据路径，拒绝清理")
        if meta.get("package_id") != verified.get("package_id") or meta.get("generation") != verified.get("generation"):
            raise ValueError("SQLite 身份与全量校验报告不一致，拒绝清理")
    finally:
        db.close()
    if not delete_source and not delete_replaced:
        raise ValueError("请指定 --delete-source 和/或 --delete-replaced")
    actions = []
    if delete_source:
        source = Path(migration["source"]).resolve()
        hold = Path(migration.get("source_hold_path") or (migration_dir / "source-package-hold")).resolve()
        if source.exists():
            if source == package_root or _inside(source, data_root):
                raise ValueError(f"源目录删除路径校验失败：{source}")
            shutil.rmtree(source)
            actions.append({"deleted": str(source)})
        if hold.exists():
            if hold != (migration_dir / "source-package-hold").resolve() or not _inside(hold, migration_dir):
                raise ValueError(f"隔离源目录删除路径校验失败：{hold}")
            shutil.rmtree(hold)
            actions.append({"deleted": str(hold)})
    if delete_replaced:
        replaced = (migration_dir / "replaced-package").resolve()
        if migration.get("replaced_package") and Path(migration["replaced_package"]).resolve() != replaced:
            raise ValueError("旧目标目录与迁移报告不一致")
        if replaced.exists():
            if not _inside(replaced, migration_dir):
                raise ValueError(f"旧目标包删除路径校验失败：{replaced}")
            shutil.rmtree(replaced)
            actions.append({"deleted": str(replaced)})
    migration["cleanup"] = {"actions": actions, "completed_at": _now()}
    _write_json(report_path, migration)
    return {"status": "completed", "migration_id": migration_id, "actions": actions}
