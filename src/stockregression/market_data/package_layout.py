from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from .timeframes import normalize_timeframe


PACKAGE_SCHEMA_VERSION = 1
ETF_PREFIXES = {"15", "16", "50", "51", "52", "56", "58"}
MINUTE_TIMEFRAMES = {"1min", "5min", "15min", "30min", "60min"}
_MANIFEST_THREAD_LOCK = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def asset_type_for_symbol(symbol: str) -> str:
    code = str(symbol).split(".", 1)[0].zfill(6)
    return "etf" if code[:2] in ETF_PREFIXES else "stock"


def package_file_relative_path(symbol: str, timeframe: str) -> str:
    timeframe = normalize_timeframe(timeframe)
    asset = asset_type_for_symbol(symbol)
    if timeframe == "1day":
        return f"{asset}_daily.parquet"
    return f"{asset}_{timeframe}/{symbol}.parquet"


def auxiliary_relative_path(symbol: str, kind: str) -> str:
    if kind == "calendar":
        suffix = ".SZ" if str(symbol).endswith(".SZ") else ".SH"
        return f"auxiliary/calendars/{suffix.lstrip('.')}.parquet"
    return f"auxiliary/actions/{symbol}.parquet"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


class PackageLayout:
    """Stable on-disk identity and publication metadata for one data package."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self._root_text = os.path.normcase(os.path.normpath(str(self.root)))
        self.control = self.root / ".stockregression"
        self.versions_dir = self.control / "versions"
        self.quality_dir = self.control / "quality"
        self.history_dir = self.control / "history"
        self.archives_dir = self.control / "archives"
        self.bases_dir = self.control / "bases"
        self.increments_dir = self.control / "increments"
        self.staging_dir = self.control / "staging"
        self.transactions_dir = self.control / "transactions"
        self.locks_dir = self.control / "locks"
        self.manifest_path = self.control / "manifest.json"
        for path in (
            self.root,
            self.control,
            self.versions_dir,
            self.quality_dir,
            self.history_dir,
            self.archives_dir,
            self.bases_dir,
            self.increments_dir,
            self.staging_dir,
            self.transactions_dir,
            self.locks_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> dict[str, Any]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("package_id"):
                return value
        except (OSError, ValueError):
            pass
        return {}

    def _manifest_guard(self):
        lock = FileLock(str(self.control / "manifest.lock"), timeout=600)

        class Guard:
            def __enter__(inner):
                _MANIFEST_THREAD_LOCK.acquire()
                lock.acquire()
                return inner

            def __exit__(inner, exc_type, exc, traceback):
                try:
                    lock.release()
                finally:
                    _MANIFEST_THREAD_LOCK.release()
        return Guard()

    def _fresh_manifest(self, value: dict[str, Any] | None = None) -> dict[str, Any]:
        current = dict(value or {})
        now = utc_now()
        if not current or not current.get("package_id"):
            current = {
                "schema_version": PACKAGE_SCHEMA_VERSION,
                "package_id": "pkg-" + uuid.uuid4().hex[:24],
                "package_version": 0,
                "created_at": now,
                "updated_at": now,
                "files": {},
                "auxiliary": {},
            }
        current.setdefault("files", {})
        current.setdefault("auxiliary", {})
        current.setdefault("package_version", 0)
        return current

    def manifest(self) -> dict[str, Any]:
        with self._manifest_guard():
            raw = self._load_manifest()
            value = self._fresh_manifest(raw)
            if value != raw:
                value["updated_at"] = utc_now()
                _atomic_json(self.manifest_path, value)
            return value

    def publish(self, manifest: dict[str, Any]) -> dict[str, Any]:
        with self._manifest_guard():
            value = self._fresh_manifest(manifest)
            value["schema_version"] = PACKAGE_SCHEMA_VERSION
            value["updated_at"] = utc_now()
            _atomic_json(self.manifest_path, value)
            return value

    def record_file(
        self,
        relative_path: str,
        *,
        version: int | None = None,
        row_count: int | None = None,
        sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        relative_path = str(relative_path).replace("\\", "/")
        path = self.absolute(relative_path)
        with self._manifest_guard():
            manifest = self._fresh_manifest(self._load_manifest())
            manifest["package_version"] = int(manifest.get("package_version", 0)) + 1
            entry = {
                **dict(manifest["files"].get(relative_path) or {}),
                "size": path.stat().st_size if path.exists() else None,
                "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
                "sha256": sha256,
                "version": version,
                "row_count": row_count,
                "updated_at": utc_now(),
            }
            if metadata:
                entry.update(metadata)
            manifest["files"][relative_path] = entry
            manifest["index_only"] = False
            manifest["schema_version"] = PACKAGE_SCHEMA_VERSION
            manifest["updated_at"] = utc_now()
            _atomic_json(self.manifest_path, manifest)
            return manifest

    def record_auxiliary(self, relative_path: str, *, version: str, verified: bool) -> dict[str, Any]:
        relative_path = str(relative_path).replace("\\", "/")
        path = self.absolute(relative_path)
        with self._manifest_guard():
            manifest = self._fresh_manifest(self._load_manifest())
            manifest["package_version"] = int(manifest.get("package_version", 0)) + 1
            manifest["auxiliary"][relative_path] = {
                "version": version,
                "verified": bool(verified),
                "size": path.stat().st_size if path.exists() else None,
                "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
                "updated_at": utc_now(),
            }
            manifest["schema_version"] = PACKAGE_SCHEMA_VERSION
            manifest["updated_at"] = utc_now()
            _atomic_json(self.manifest_path, manifest)
            return manifest

    def write_quality(self, symbol: str, timeframe: str, payload: dict[str, Any]) -> None:
        digest = hashlib.sha256(
            f"{symbol}|{normalize_timeframe(timeframe)}".encode()
        ).hexdigest()[:16]
        _atomic_json(self.quality_dir / f"{symbol}-{digest}.json", payload)

    def write_version_manifest(self, version: str, payload: dict[str, Any]) -> None:
        safe = "".join(char for char in str(version) if char.isalnum() or char in "-_.")
        if safe:
            _atomic_json(self.versions_dir / f"{safe}.json", payload)

    def absolute(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else self.root / path

    def relative(self, path: str | Path) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            return str(candidate).replace("\\", "/")
        normalized = os.path.normpath(str(candidate))
        folded = os.path.normcase(normalized)
        if folded == self._root_text:
            return "."
        prefix = self._root_text + os.sep
        if folded.startswith(prefix):
            return normalized[len(str(self.root)) + 1:].replace("\\", "/")
        return normalized.replace("\\", "/")

    def lock_for(self, name: str) -> FileLock:
        digest = hashlib.sha256(name.encode()).hexdigest()[:24]
        return FileLock(str(self.locks_dir / f"{digest}.lock"), timeout=600)

    def set_file_metadata(self, relative_path: str, metadata: dict[str, Any]) -> None:
        relative_path = str(relative_path).replace("\\", "/")
        with self._manifest_guard():
            manifest = self._fresh_manifest(self._load_manifest())
            entry = dict(manifest["files"].get(relative_path) or {})
            entry.update(metadata)
            entry["updated_at"] = utc_now()
            manifest["files"][relative_path] = entry
            manifest["schema_version"] = PACKAGE_SCHEMA_VERSION
            manifest["updated_at"] = utc_now()
            _atomic_json(self.manifest_path, manifest)

    def file_metadata(self, relative_path: str) -> dict[str, Any]:
        return dict(self.manifest()["files"].get(str(relative_path).replace("\\", "/")) or {})

    def begin_transaction(self, payload: dict[str, Any]) -> tuple[str, Path]:
        transaction_id = uuid.uuid4().hex
        path = self.transactions_dir / f"{transaction_id}.json"
        _atomic_json(path, {"id": transaction_id, "status": "prepared", **payload})
        return transaction_id, path

    def update_transaction(self, path: Path, values: dict[str, Any]) -> None:
        payload = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {"id": path.stem}
        payload.update(values)
        _atomic_json(path, payload)

    def finish_transaction(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def pending_transactions(self) -> list[Path]:
        return sorted(self.transactions_dir.glob("*.json"))
