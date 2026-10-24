from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from .timeframes import normalize_timeframe


PACKAGE_SCHEMA_VERSION = 2
ETF_PREFIXES = {"15", "16", "50", "51", "52", "56", "58"}
_MANIFEST_THREAD_LOCK = threading.RLock()
_PUBLICATION_CHECKS = ContextVar("package_publication_checks", default={})


def is_package_root(path: str | Path) -> bool:
    """Return whether a directory contains a published schema-2 package."""
    root = Path(path)
    manifest_path = root / ".stockregression" / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        return False
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        int(value.get("schema_version") or 0) == PACKAGE_SCHEMA_VERSION
        and bool(value.get("package_id")) and bool(value.get("generation"))
        and isinstance(value.get("datasets"), dict)
    )


def resolve_package_root(value: str | Path) -> Path:
    """Resolve a dataset root such as ``data/dataset/etf``.

    ``data/dataset`` may group packages by asset.  A directory that already
    contains market files is used as-is; otherwise its immediate package
    children are considered.  Several package children require an explicit
    package path so one asset never silently hides another.
    """
    root = Path(value).expanduser().resolve()
    if is_package_root(root) or not root.is_dir():
        return root
    candidates = [
        child for child in sorted(root.iterdir())
        if child.is_dir() and is_package_root(child)
    ]
    if not candidates:
        return root
    if len(candidates) > 1:
        names = "、".join(str(child) for child in candidates)
        raise ValueError("数据包索引目录包含多个数据包，请指向具体包目录：" + names)
    return candidates[0]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def asset_type_for_symbol(symbol: str) -> str:
    code = str(symbol).split(".", 1)[0].zfill(6)
    return "etf" if code[:2] in ETF_PREFIXES else "stock"


def auxiliary_relative_path(symbol: str, kind: str) -> str:
    if kind == "calendar":
        suffix = ".SZ" if str(symbol).endswith(".SZ") else ".SH"
        return f"auxiliary/calendars/{suffix.lstrip('.')}.parquet"
    return f"auxiliary/actions/{symbol}.parquet"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".writing")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02 * 2 ** attempt)
    finally:
        temporary.unlink(missing_ok=True)


class PackageLayout:
    """Stable on-disk identity and publication metadata for one data package."""

    def __init__(self, root: str | Path, *, market_settings=None, read_only: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.read_only = read_only
        from ..settings import load_settings

        self.market_settings = market_settings or load_settings().market_data
        self.lock_timeout = int(self.market_settings.package_lock_timeout_seconds)
        self._root_text = os.path.normcase(os.path.normpath(str(self.root)))
        self.control = self.root / ".stockregression"
        self.versions_dir = self.control / "versions"
        self.datasets_dir = self.control / "datasets"
        self.quality_dir = self.control / "quality"
        self.history_dir = self.control / "history"
        self.partitions_dir = self.root / "partitions"
        self.staging_dir = self.control / "staging"
        self.transactions_dir = self.control / "transactions"
        self.locks_dir = self.control / "locks"
        self.manifest_path = self.control / "manifest.json"
        self._transaction_lock = FileLock(
            str(self.control / "transactions.lock"), timeout=self.lock_timeout,
            is_singleton=True,
        )
        self._manifest_cache: dict[str, Any] | None = None
        self._manifest_cache_key: tuple[int, int] | None = None
        for path in (() if read_only else (
            self.root,
            self.control,
            self.versions_dir,
            self.datasets_dir,
            self.quality_dir,
            self.history_dir,
            self.partitions_dir,
            self.staging_dir,
            self.transactions_dir,
            self.locks_dir,
        )):
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
        if self.read_only:
            return nullcontext()
        lock = FileLock(str(self.control / "manifest.lock"), timeout=self.lock_timeout)

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

    def _manifest_stat_key(self) -> tuple[int, int] | None:
        try:
            stat = self.manifest_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _cached_manifest(self) -> dict[str, Any] | None:
        if self._manifest_cache is not None and self._manifest_cache_key == self._manifest_stat_key():
            return self._manifest_cache
        return None

    def _remember_manifest(self, value: dict[str, Any]) -> dict[str, Any]:
        self._manifest_cache = value
        self._manifest_cache_key = self._manifest_stat_key()
        return value

    def _fresh_manifest(self, value: dict[str, Any] | None = None) -> dict[str, Any]:
        current = dict(value or {})
        now = utc_now()
        if not current or not current.get("package_id"):
            current = {
                "schema_version": PACKAGE_SCHEMA_VERSION,
                "package_id": "pkg-" + uuid.uuid4().hex[:24],
                "package_version": 0,
                "generation": uuid.uuid4().hex,
                "created_at": now,
                "updated_at": now,
                "files": {},
                "auxiliary": {},
                "datasets": {},
                "storage_layout": "time_partitioned",
                "partition_root": "partitions",
                "partition_scheme": {"1day": "year", "1min": "month"},
            }
        schema_version = int(current.get("schema_version", 0) or 0)
        if schema_version not in {PACKAGE_SCHEMA_VERSION}:
            raise ValueError(
                f"不支持的数据包清单版本 {schema_version}；请使用 data migrate-package 迁移到分片包"
            )
        current.setdefault("generation", uuid.uuid4().hex)
        current.setdefault("files", {})
        current.setdefault("auxiliary", {})
        current.setdefault("datasets", {})
        current.setdefault("package_version", 0)
        return current

    @staticmethod
    def dataset_key(symbol: str, timeframe: str, adjustment: str = "none") -> str:
        return f"{symbol}|{normalize_timeframe(timeframe)}|{adjustment}"

    def dataset_manifest_relative_path(
        self, symbol: str, timeframe: str, adjustment: str, version: int,
    ) -> str:
        safe_symbol = "".join(c for c in str(symbol) if c.isalnum() or c in ".-_ ")
        safe_adjustment = "".join(c for c in str(adjustment) if c.isalnum() or c in "-_ ")
        return (
            f".stockregression/datasets/{normalize_timeframe(timeframe)}/"
            f"{safe_symbol}/{safe_adjustment}/v{int(version)}.json"
        )

    def publish_dataset(
        self, payload: dict[str, Any], *, expected_version: int,
        pending_updates: list[dict[str, Any]] | None = None,
        dependent_pending_updates: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Publish one immutable dataset manifest, then atomically advance its package head."""
        self.check_publication()
        symbol = str(payload["symbol"])
        timeframe = normalize_timeframe(str(payload["timeframe"]))
        adjustment = str(payload.get("adjustment") or "none")
        version = int(payload["version"])
        relative = self.dataset_manifest_relative_path(
            symbol, timeframe, adjustment, version
        )
        manifest_path = self.absolute(relative)
        with self._manifest_guard():
            package = self._fresh_manifest(self._load_manifest())
            key = self.dataset_key(symbol, timeframe, adjustment)
            current = dict(package["datasets"].get(key) or {})
            current_version = int(current.get("version") or 0)
            if current_version != int(expected_version):
                if current_version == version and current.get("manifest") == relative:
                    return package
                raise ValueError(
                    f"数据集清单版本冲突：{symbol} {timeframe}，"
                    f"预期 {expected_version}，当前 {current_version}"
                )
            value = {
                **payload,
                "schema_version": PACKAGE_SCHEMA_VERSION,
                "package_id": package["package_id"],
                "generation": package["generation"],
                "updated_at": utc_now(),
            }
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                comparable_existing = {key: item for key, item in existing.items() if key != "updated_at"}
                comparable_value = {key: item for key, item in value.items() if key != "updated_at"}
                if comparable_existing != comparable_value:
                    raise ValueError(f"不可变数据集清单已存在但内容不同：{relative}")
                value = existing
            else:
                _atomic_json(manifest_path, value)
            base_timeframe = timeframe in {"1day", "1min"}
            old_bytes = int(current.get("shard_bytes") or 0) if base_timeframe else 0
            old_count = int(current.get("shard_count") or 0) if base_timeframe else 0
            shard_bytes = (
                sum(int(row.get("size") or 0) for row in value.get("shards", []))
                if base_timeframe else 0
            )
            shard_count = len(value.get("shards", [])) if base_timeframe else 0
            quality_reference = dict(current.get("quality_archive") or {})
            if quality_reference and pending_updates:
                existing_updates = list(quality_reference.get("pending_updates") or [])
                for update in pending_updates:
                    dates = sorted({str(value)[:10] for value in update.get("dates") or [] if value})
                    if not dates:
                        continue
                    existing_updates.append({
                        "dates": dates,
                        "reason": str(update.get("reason") or "行情修订"),
                    })
                quality_reference["pending_updates"] = existing_updates
            for dependent, updates in (dependent_pending_updates or {}).items():
                dependent_key = self.dataset_key(symbol, dependent, adjustment)
                dependent_head = dict(package["datasets"].get(dependent_key) or {})
                dependent_reference = dict(
                    dependent_head.get("quality_archive") or {}
                )
                if not dependent_reference:
                    continue
                dependent_first = str(dependent_head.get("first_time") or "")[:10]
                dependent_last = str(dependent_head.get("last_time") or "")[:10]
                existing_updates = list(
                    dependent_reference.get("pending_updates") or []
                )
                for update in updates:
                    dates = sorted({
                        str(value)[:10] for value in update.get("dates") or []
                        if value
                        and (not dependent_first or str(value)[:10] >= dependent_first)
                        and (not dependent_last or str(value)[:10] <= dependent_last)
                    })
                    if dates:
                        existing_updates.append({
                            "dates": dates,
                            "reason": str(
                                update.get("reason") or "日线依赖修订"
                            ),
                        })
                dependent_reference["pending_updates"] = existing_updates
                dependent_head["quality_archive"] = dependent_reference
                package["datasets"][dependent_key] = dependent_head
            package["datasets"][key] = {
                "version": version,
                "manifest": relative,
                "row_count": int(value.get("row_count") or 0),
                "first_time": value.get("first_time"),
                "last_time": value.get("last_time"),
                "status": value.get("status", "pending_validation"),
                "derived": value.get("derived"),
                "shard_count": shard_count,
                "shard_bytes": shard_bytes,
                # A data update keeps the last archive reference so unchanged
                # days remain reusable.  The archive's data_version makes the
                # revision visibly stale until changed days are revalidated.
                "quality_archive": quality_reference or None,
            }
            package["package_version"] = int(package.get("package_version", 0)) + 1
            package["shard_count"] = max(
                0, int(package.get("shard_count") or 0) - old_count + shard_count
            )
            package["shard_bytes"] = max(
                0, int(package.get("shard_bytes") or 0) - old_bytes + shard_bytes
            )
            package["storage_layout"] = "time_partitioned"
            package["partition_root"] = "partitions"
            package["schema_version"] = PACKAGE_SCHEMA_VERSION
            package["updated_at"] = utc_now()
            self.check_publication()
            _atomic_json(self.manifest_path, package)
            return self._remember_manifest(package)

    def publish_quality_archive(
        self,
        symbol: str,
        timeframe: str,
        adjustment: str,
        reference: dict[str, Any],
        *,
        expected_data_version: int,
        expected_revision: int,
    ) -> dict[str, Any]:
        """Publish a full coverage archive revision without changing market data.

        Market bars and the archive reference are reconciled through the same
        package head.  The monthly archive files themselves are immutable and
        content addressed; only this small reference advances.
        """
        return self.publish_quality_archives([{
            "symbol": symbol,
            "timeframe": timeframe,
            "adjustment": adjustment,
            "reference": reference,
            "expected_data_version": expected_data_version,
            "expected_revision": expected_revision,
        }])

    def publish_quality_archives(
        self, entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Publish archive references for several datasets as one head switch.

        Every entry has ``symbol``, ``timeframe``, ``adjustment``, ``reference``,
        ``expected_data_version`` and ``expected_revision``. Validation and
        immutable payload checks happen before any entry is applied, so a
        conflict cannot advance only part of a batch.
        """
        if not entries:
            return self.manifest()
        self.check_publication()
        with self._manifest_guard():
            package = self._fresh_manifest(self._load_manifest())
            prepared: dict[str, dict[str, Any]] = {}
            changed = False
            for entry in entries:
                symbol = str(entry["symbol"])
                timeframe = normalize_timeframe(str(entry["timeframe"]))
                adjustment = str(entry.get("adjustment") or "none")
                reference = dict(entry.get("reference") or {})
                expected_data_version = int(entry["expected_data_version"])
                expected_revision = int(entry["expected_revision"])
                key = self.dataset_key(symbol, timeframe, adjustment)
                if key in prepared:
                    raise ValueError(f"覆盖档案批次包含重复数据集：{key}")
                relative = str(reference.get("path") or "").replace("\\", "/")
                path = self.absolute(relative)
                if not path.is_file():
                    raise ValueError(f"覆盖档案文件不存在：{relative}")
                current = dict(package["datasets"].get(key) or {})
                if not current:
                    raise ValueError(f"覆盖档案没有对应的数据集：{key}")
                if int(current.get("version") or 0) != expected_data_version:
                    raise ValueError(
                        f"覆盖档案的数据版本冲突：{key}，预期 {expected_data_version}，"
                        f"当前 {current.get('version')}"
                    )
                previous = dict(current.get("quality_archive") or {})
                current_revision = int(previous.get("revision") or 0)
                if current_revision != expected_revision:
                    already_published = (
                        current_revision == int(reference.get("revision") or 0)
                        and previous.get("path") == reference.get("path")
                        and previous.get("sha256") == reference.get("sha256")
                        and int(previous.get("data_version") or 0)
                        == int(reference.get("data_version") or 0)
                    )
                    if already_published:
                        prepared[key] = current
                        continue
                    raise ValueError(
                        f"覆盖档案修订冲突：{key}，预期 {expected_revision}，"
                        f"当前 {current_revision}"
                    )
                prepared[key] = {
                    **current,
                    "quality_archive": {
                        **reference,
                        "updated_at": utc_now(),
                    },
                }
                changed = True
            if not changed:
                return package
            for key, current in prepared.items():
                package["datasets"][key] = current
            package["package_version"] = int(package.get("package_version", 0)) + 1
            package["schema_version"] = PACKAGE_SCHEMA_VERSION
            package["updated_at"] = utc_now()
            self.check_publication()
            _atomic_json(self.manifest_path, package)
            return self._remember_manifest(package)

    def load_dataset_manifest(self, symbol: str, timeframe: str, adjustment: str, version: int | None = None) -> dict[str, Any] | None:
        package = self.manifest()
        head = package.get("datasets", {}).get(
            self.dataset_key(symbol, timeframe, adjustment)
        )
        if not head or (version is not None and int(head.get("version", -1)) != int(version)):
            relative = self.dataset_manifest_relative_path(
                symbol, timeframe, adjustment, int(version or -1)
            )
            path = self.absolute(relative)
            if version is None or not path.is_file():
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("package_id") != package.get("package_id")
                or value.get("generation") != package.get("generation")
                or value.get("symbol") != symbol
                or normalize_timeframe(value.get("timeframe")) != normalize_timeframe(timeframe)
                or value.get("adjustment", "none") != adjustment
                or int(value.get("version", -1)) != int(version)
            ):
                raise ValueError(f"历史数据集清单身份或版本不匹配：{path}")
            return value
        path = self.absolute(str(head["manifest"]))
        if not path.is_file():
            raise ValueError(f"数据集清单丢失：{path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("package_id") != package.get("package_id")
            or value.get("generation") != package.get("generation")
            or int(value.get("version", -1)) != int(head.get("version", -2))
        ):
            raise ValueError(f"数据集清单身份或版本不匹配：{path}")
        return value

    def manifest(self) -> dict[str, Any]:
        with self._manifest_guard():
            cached = self._cached_manifest()
            if cached is not None:
                return cached
            raw = self._load_manifest()
            if self.read_only and not raw:
                raise ValueError("只读下载上下文要求已发布的数据包清单")
            value = self._fresh_manifest(raw)
            if value != raw and not self.read_only:
                value["updated_at"] = utc_now()
                self.check_publication()
                _atomic_json(self.manifest_path, value)
            return self._remember_manifest(value)

    def publish(self, manifest: dict[str, Any]) -> dict[str, Any]:
        with self._manifest_guard():
            value = self._fresh_manifest(manifest)
            value["schema_version"] = PACKAGE_SCHEMA_VERSION
            value["updated_at"] = utc_now()
            self.check_publication()
            _atomic_json(self.manifest_path, value)
            return self._remember_manifest(value)

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
            self.check_publication()
            _atomic_json(self.manifest_path, manifest)
            return self._remember_manifest(manifest)

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
            self.check_publication()
            _atomic_json(self.manifest_path, manifest)
            return self._remember_manifest(manifest)

    def write_quality(self, symbol: str, timeframe: str, payload: dict[str, Any]) -> None:
        self.check_publication()
        digest = hashlib.sha256(
            f"{symbol}|{normalize_timeframe(timeframe)}".encode()
        ).hexdigest()[:16]
        _atomic_json(self.quality_dir / f"{symbol}-{digest}.json", payload)

    def write_version_manifest(self, version: str, payload: dict[str, Any]) -> None:
        self.check_publication()
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
        return FileLock(str(self.locks_dir / f"{digest}.lock"), timeout=self.lock_timeout)

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
            self.check_publication()
            _atomic_json(self.manifest_path, manifest)
            self._remember_manifest(manifest)

    def file_metadata(self, relative_path: str) -> dict[str, Any]:
        return dict(self.manifest()["files"].get(str(relative_path).replace("\\", "/")) or {})

    def begin_transaction(self, payload: dict[str, Any]) -> tuple[str, Path]:
        self.check_publication()
        transaction_id = uuid.uuid4().hex
        path = self.transactions_dir / f"{transaction_id}.json"
        _atomic_json(path, {"id": transaction_id, "status": "prepared", **payload})
        return transaction_id, path

    def transaction_guard(self):
        """Serialize publication and recovery across threads and processes."""
        return self._transaction_lock

    @contextmanager
    def publication_fence(self, check):
        """Carry the attempt claim to each atomic head switch in this context."""
        checks = {**_PUBLICATION_CHECKS.get(), str(self.root): check}
        token = _PUBLICATION_CHECKS.set(checks)
        try:
            check()
            yield
        finally:
            _PUBLICATION_CHECKS.reset(token)

    def check_publication(self):
        if self.read_only:
            raise ValueError("只读下载上下文不能发布数据包")
        check = _PUBLICATION_CHECKS.get().get(str(self.root))
        if check is not None:
            check()

    def update_transaction(self, path: Path, values: dict[str, Any]) -> None:
        # A vanished transaction was completed elsewhere. Never resurrect it
        # without its dataset identity, even when the read fails transiently.
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("kind") != "dataset":
            raise ValueError(f"事务记录缺少数据集身份：{path.name}")
        payload.update(values)
        _atomic_json(path, payload)

    def finish_transaction(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def pending_transactions(self) -> list[Path]:
        return sorted(self.transactions_dir.glob("*.json"))
