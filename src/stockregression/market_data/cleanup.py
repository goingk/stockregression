from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..storage import Store
from .models import MarketDataError


LEGACY_DIRECTORIES = ("parquet", "manifests", "raw", "source-probe", "imports")
LEGACY_STATE_PREFIXES = (
    "coverage:", "market-coverage:", "data-source-spans:", "ready:",
    "prepare:", "sync:", "import-reference:", "market-source-package",
)
PRESERVED_TABLES = (
    "results", "jobs", "configs", "instruments", "watchlist",
    "research_registry", "parquet_datasets", "parquet_versions",
    "parquet_files", "parquet_increments", "parquet_coverage",
    "parquet_day_validation", "parquet_auxiliary_history", "parquet_meta",
)


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _counts(root: Path) -> dict[str, int]:
    store = Store(root)
    with store.db() as db:
        names = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        output = {}
        for name in ("results", "jobs", "configs", "instruments", "watchlist", "snapshots"):
            output[name] = int(db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) if name in names else 0
    return output


def legacy_cleanup_plan(root: str | Path, *, current_package_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise MarketDataError("数据目录不存在：" + str(root))
    package = Path(current_package_dir).expanduser().resolve() if current_package_dir else None
    targets = []
    for name in LEGACY_DIRECTORIES:
        path = root / name
        if path.exists() and _inside(root, path):
            targets.append(path)
    empty_package = root / "package"
    if empty_package.exists() and _inside(root, empty_package) and empty_package != package:
        targets.append(empty_package)
    store = Store(root)
    with store.db() as db:
        names = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        snapshots = []
        if "snapshots" in names:
            for row in db.execute("SELECT version,symbol,manifest FROM snapshots").fetchall():
                try:
                    manifest = json.loads(row[2])
                except (TypeError, ValueError):
                    manifest = {}
                if manifest.get("market_data") is not True:
                    snapshots.append({"version": row[0], "symbol": row[1]})
    active = 0
    with store.db() as db:
        if "jobs" in {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }:
            active = int(db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0])
    return {
        "root": str(root),
        "targets": [str(path) for path in targets],
        "target_bytes": sum(_size(path) for path in targets),
        "legacy_snapshots": snapshots,
        "active_jobs": active,
        "preserves": list(PRESERVED_TABLES),
        "before_counts": _counts(root),
    }


def apply_legacy_cleanup(
    root: str | Path,
    *,
    current_package_dir: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    plan = legacy_cleanup_plan(root, current_package_dir=current_package_dir)
    if plan["active_jobs"]:
        raise MarketDataError(f"仍有 {plan['active_jobs']} 个排队或运行中的任务，停止任务后再清理")
    removed = []
    for value in plan["targets"]:
        path = Path(value).resolve()
        if not _inside(root, path) or path == root:
            raise MarketDataError("拒绝清理数据目录之外的路径：" + str(path))
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            removed.append(str(path))
        elif path.is_file():
            path.unlink()
            removed.append(str(path))

    store = Store(root)
    legacy_versions = {item["version"] for item in plan["legacy_snapshots"]}
    with store.db() as db:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "snapshots" in tables:
            for version in legacy_versions:
                db.execute("DELETE FROM snapshots WHERE version=?", (version,))
        if "active_snapshots" in tables:
            if legacy_versions:
                placeholders = ",".join("?" for _ in legacy_versions)
                db.execute(
                    "DELETE FROM active_snapshots WHERE version IN (" + placeholders + ")",
                    tuple(legacy_versions),
                )
        if "market_auxiliary" in tables:
            db.execute("DELETE FROM market_auxiliary")
        if "state" in tables:
            rows = db.execute("SELECT key FROM state").fetchall()
            stale = [
                row[0] for row in rows
                if any(str(row[0]).startswith(prefix) for prefix in LEGACY_STATE_PREFIXES)
            ]
            for key in stale:
                db.execute("DELETE FROM state WHERE key=?", (key,))
    after = _counts(root)
    report = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "removed_paths": removed,
        "removed_bytes": int(plan["target_bytes"]),
        "legacy_snapshot_count": len(legacy_versions),
        "before_counts": plan["before_counts"],
        "after_counts": after,
        "preserved_tables": list(PRESERVED_TABLES),
        "kept_current_package": str(current_package_dir) if current_package_dir else None,
    }
    if report_path:
        target = Path(report_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".writing")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    return report
