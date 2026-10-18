from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from ..storage import Store
from .cache import _cache
from .models import MarketDataError
from .repository import TIMEFRAMES, utc_now


STATE_KEY = "parquet-rebuild"
CACHE_PREFIXES = (
    "coverage:", "prepare:", "ready:", "data-source-spans:",
    "market-data:", "import-reference:", "catalog:",
)


def _rebuild_folder(root: Path) -> tuple[list[Path], list[str]]:
    targets: list[Path] = []
    notes: list[str] = []
    root = root.resolve()
    market = root / "market"
    if market.exists():
        for timeframe in TIMEFRAMES:
            folder = market / timeframe
            if not folder.exists():
                continue
            for suffix in ("bars.sqlite", "bars.sqlite-wal", "bars.sqlite-shm"):
                path = folder / suffix
                if path.exists():
                    targets.append(path)
            targets.extend(
                path for path in folder.glob("symbol=*") if path.is_dir()
            )
    for folder_name in ("parquet", "staging", "source_history"):
        folder = root / folder_name
        if folder.exists():
            targets.append(folder)
    manifests = root / "manifests"
    if manifests.exists():
        targets.extend(path for path in manifests.glob("*.json") if path.is_file())
    for target in targets:
        resolved = target.resolve()
        if not resolved.is_relative_to(root):
            raise MarketDataError(f"清理目标不在数据根目录内：{resolved}")
    notes.extend([
        "保留 catalog.sqlite 中的任务、结果、配置、自选和研究记录",
        "保留设置文件、导入目录和原始 ETF 下载目录",
        f"清理 {len(targets)} 个行情文件或目录",
    ])
    return targets, notes


def preview_reset(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    targets, notes = _rebuild_folder(root_path)
    with sqlite3.connect(root_path / "catalog.sqlite") as db:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        counts = {}
        for table in (
            "parquet_datasets", "parquet_versions", "parquet_files",
            "parquet_imports", "parquet_coverage", "parquet_derived",
            "parquet_auxiliary_history", "snapshots", "active_snapshots",
            "market_auxiliary", "sync_items",
        ):
            if table in tables:
                counts[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        active_jobs = (
            db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if "jobs" in tables else 0
        )
    return {
        "root": str(root_path), "targets": [str(path) for path in targets],
        "metrics": {
            "target_bytes": sum(
                path.stat().st_size if path.is_file() else sum(
                    item.stat().st_size for item in path.rglob("*") if item.is_file()
                ) for path in targets
            ),
            **counts, "active_jobs": active_jobs,
        },
        "notes": notes, "generation": _current_generation(root_path),
    }


def _current_generation(root: Path) -> str | None:
    path = root / "catalog.sqlite"
    if not path.exists():
        return None
    with sqlite3.connect(path) as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='parquet_meta'"
        ).fetchone()
        if not exists:
            return None
        row = db.execute(
            "SELECT value FROM parquet_meta WHERE key='generation'"
        ).fetchone()
        return row[0] if row else None


def reset_market_storage(
    root: str | Path, *, source_folder: str | Path, force_jobs: bool = False,
) -> dict:
    root_path = Path(root).resolve()
    source = Path(source_folder).expanduser().resolve()
    if not source.is_dir():
        raise MarketDataError("ETF 数据包目录不存在：" + str(source))
    if source == root_path or source.is_relative_to(root_path):
        raise MarketDataError("ETF 数据包目录不能位于数据根目录内")
    lock = FileLock(str(root_path / "market-reset.lock"), timeout=5)
    if not lock.acquire(blocking=False):
        raise MarketDataError("另一个行情重建任务正在执行")
    try:
        store = Store(root_path)
        active = [
            job for job in store.jobs(1000)
            if job["status"] in {"queued", "running"}
        ]
        if active and not force_jobs:
            raise MarketDataError(
                "仍有活动任务，停止计算和行情导入后再执行清空："
                + ", ".join(job["id"] for job in active[:8])
            )
        preview = preview_reset(root_path)
        for target in preview["targets"]:
            path = Path(target)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        with store.db() as db:
            for table in (
                "parquet_day_validation", "parquet_increments", "parquet_source_archives",
                "parquet_files", "parquet_versions", "parquet_datasets",
                "parquet_imports", "parquet_coverage", "parquet_derived",
                "parquet_auxiliary_history", "snapshots", "active_snapshots",
                "market_auxiliary", "sync_items",
            ):
                exists = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if exists:
                    db.execute(f"DELETE FROM {table}")
            db.execute(
                "DELETE FROM state WHERE key LIKE 'coverage:%' "
                "OR key LIKE 'prepare:%' OR key LIKE 'ready:%' "
                "OR key LIKE 'data-source-spans:%' OR key LIKE 'market-data:%' "
                "OR key LIKE 'import-reference:%' OR key LIKE 'catalog:%'"
            )
            generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            db.execute(
                "CREATE TABLE IF NOT EXISTS parquet_meta("
                "key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            db.execute(
                "INSERT OR REPLACE INTO parquet_meta(key,value) VALUES('generation',?)",
                (generation,),
            )
            db.execute(
                "DELETE FROM parquet_meta WHERE key IN ('source_package_dir')"
            )
            db.execute(
                "INSERT OR REPLACE INTO state(key,value) VALUES(?,?)",
                (
                    STATE_KEY,
                    json.dumps({
                        "generation": generation, "source": str(source),
                        "stage": "reset_complete", "reset_at": utc_now(),
                        "completed": [], "failed": [],
                    }, ensure_ascii=False),
                ),
            )
        _cache.clear()
        return {
            "status": "completed", "generation": generation,
            "source": str(source), "cleared": preview,
            "message": "旧行情已清理，数据包导入尚未开始",
        }
    finally:
        lock.release()


def rebuild_state(store: Store) -> dict | None:
    return store.state(STATE_KEY)


def update_rebuild_state(
    store: Store, *, stage: str, result: dict | None = None,
) -> dict:
    state = rebuild_state(store) or {}
    state.update({"stage": stage, "updated_at": utc_now()})
    if result is not None:
        state["completed"] = result.get("symbols", [])
        state["failed"] = result.get("failures", [])
        state["status"] = result.get("status")
        state["message"] = result.get("message")
    store.state(STATE_KEY, state)
    return state
