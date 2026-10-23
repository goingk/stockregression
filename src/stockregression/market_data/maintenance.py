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
from .package_layout import PackageLayout, is_package_root, resolve_package_root
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
            "parquet_datasets", "parquet_versions", "parquet_shards",
            "parquet_imports", "parquet_coverage", "parquet_coverage_days",
            "parquet_coverage_archives", "parquet_derived",
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
        # Capture the attached package before catalog cleanup.  A reset must not
        # silently revive its old market datasets on the next repository open.
        attached_packages: set[Path] = set()
        with sqlite3.connect(root_path / "catalog.sqlite") as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='parquet_meta'"
            ).fetchone()
            if exists:
                for key in ("package_dir", "source_package_dir"):
                    row = db.execute(
                        "SELECT value FROM parquet_meta WHERE key=?", (key,)
                    ).fetchone()
                    if row and row[0]:
                        attached_packages.add(Path(row[0]).resolve())
        active_count = store.active_job_count()
        if active_count and not force_jobs:
            raise MarketDataError(
                "仍有活动任务，停止计算和行情导入后再执行清空："
                + f"共 {active_count} 个任务"
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
                "parquet_day_validation", "parquet_coverage_days",
                "parquet_coverage_archives", "parquet_shards", "parquet_source_archives",
                "parquet_versions", "parquet_datasets",
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
                "DELETE FROM parquet_meta WHERE key IN "
                "('package_dir','source_package_dir')"
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
        candidate = resolve_package_root(root_path / "package")
        if is_package_root(candidate):
            package = PackageLayout(candidate)
            manifest = dict(package.manifest())
            # root/package is the repository's fallback package.  Leaving its
            # dataset heads in place would make the cleared catalog rebuild the
            # exact market data that reset was asked to remove.
            manifest["datasets"] = {}
            manifest["shard_count"] = 0
            manifest["shard_bytes"] = 0
            package.publish(manifest)
        # A package explicitly attached from elsewhere is not deleted.  It is
        # detached above, and its quality references are cleared so a later
        # explicit attachment cannot inherit results from the reset dataset.
        for package_path in sorted(attached_packages - {candidate}):
            if not is_package_root(package_path):
                continue
            package = PackageLayout(package_path)
            manifest = dict(package.manifest())
            changed = False
            for dataset in manifest.get("datasets", {}).values():
                if dataset.get("quality_archive") is not None:
                    dataset["quality_archive"] = None
                    changed = True
            if changed:
                package.publish(manifest)
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
