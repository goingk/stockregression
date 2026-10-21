"""Exact computation provenance used to gate result completion."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _digest(parts: list[bytes]) -> str:
    value = hashlib.sha256()
    for part in parts:
        value.update(part)
    return value.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.decode("utf-8", errors="replace").strip()


def _source_digest(root: Path) -> tuple[str, int]:
    package = root / "src" / "stockregression"
    files = sorted(
        path for path in package.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    ) if package.exists() else []
    parts: list[bytes] = []
    for path in files:
        parts.append(path.relative_to(root).as_posix().encode("utf-8"))
        parts.append(b"\0")
        parts.append(path.read_bytes())
        parts.append(b"\0")
    return _digest(parts), len(files)


def _lock_digest(root: Path) -> tuple[str | None, str | None]:
    for name in ("uv.lock", "poetry.lock", "pdm.lock", "Pipfile.lock", "requirements.lock", "requirements.txt"):
        path = root / name
        if path.is_file():
            return name, hashlib.sha256(path.read_bytes()).hexdigest()
    return None, None


def computation_provenance(root: str | Path, *, configuration: Any = None) -> dict[str, Any]:
    project = Path(root).resolve()
    source_hash, source_files = _source_digest(project)
    commit = _git(project, "rev-parse", "HEAD")
    status = _git(project, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _git(project, "diff", "HEAD", "--binary", "--", "src", "pyproject.toml", "uv.lock")
    lock_name, lock_hash = _lock_digest(project)
    packages = {}
    for name in ("numpy", "numba", "pandas", "scipy", "pyarrow", "pydantic", "stockregression"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    config_hash = _digest([
        json.dumps(configuration, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ]) if configuration is not None else None
    return {
        "schema_version": 1,
        "engine_version": packages.get("stockregression") or "0.1.0",
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_fingerprint": _digest([(status or "").encode("utf-8")]),
        "uncommitted_diff_fingerprint": _digest([(diff or "").encode("utf-8")]),
        "source_fingerprint": source_hash,
        "source_file_count": source_files,
        "dependency_lock_file": lock_name,
        "dependency_lock_hash": lock_hash,
        "configuration_fingerprint": config_hash,
        "runtime": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": packages,
        },
    }


def reproducibility_mismatches(original: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    if not isinstance(original, dict):
        return ["原始结果缺少计算来源指纹"]
    reasons: list[str] = []
    for field in (
        "engine_version", "source_fingerprint",
        "uncommitted_diff_fingerprint", "dependency_lock_hash", "configuration_fingerprint",
    ):
        if original.get(field) != current.get(field):
            reasons.append(f"计算来源字段不一致：{field}")
    old_packages = (original.get("runtime") or {}).get("packages") or {}
    new_packages = (current.get("runtime") or {}).get("packages") or {}
    if old_packages != new_packages:
        reasons.append("Python 数值依赖版本不一致")
    if (original.get("runtime") or {}).get("python") != (current.get("runtime") or {}).get("python"):
        reasons.append("Python 运行时版本不一致")
    return reasons
