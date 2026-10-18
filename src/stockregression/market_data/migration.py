from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd
from filelock import FileLock

from ..storage import Store, now_iso
from .repository import MarketDataRepository, PARSER_VERSION
from .service import MarketDataService, _params_hash


class LegacyMigration:
    """Explicit, resumable migration of active Parquet snapshots."""

    def __init__(self, store: Store):
        self.store = store
        self.repository = MarketDataRepository(store.root)
        self.service = MarketDataService(store)

    def run(self, *, dry_run=False, resume=True, progress=lambda *_: None, cancelled=lambda: False) -> dict:
        with self.store.db() as db:
            rows = db.execute(
                "SELECT s.manifest FROM snapshots s JOIN active_snapshots a USING(version) ORDER BY s.symbol"
            ).fetchall()
        manifests = [__import__("json").loads(row[0]) for row in rows]
        targets = [manifest for manifest in manifests if not manifest.get("market_data")]
        if not targets:
            return {"status": "completed", "targets": 0, "migrated": 0, "skipped": 0,
                    "message": "没有需要迁移的旧活动快照"}
        if dry_run:
            return {"status": "preview", "targets": len(targets),
                    "symbols": [manifest["symbol"] for manifest in targets],
                    "rows": {manifest["symbol"]: {
                        "minute": self._count(manifest, "minute"), "daily": self._count(manifest, "daily")
                    } for manifest in targets}, "message": "预检完成，未写入数据库"}
        backup = self._backup_catalog()
        results, failures, skipped = [], [], []
        for position, manifest in enumerate(targets, start=1):
            if cancelled():
                raise InterruptedError("迁移已取消")
            symbol = manifest["symbol"]
            symbol_lock = None
            try:
                migrated = []
                symbol_lock = FileLock(str(self.store.root / f"sync-{symbol}.lock"), timeout=120)
                symbol_lock.acquire()
                for kind, timeframe in (("minute", "1min"), ("daily", "1day")):
                    paths = manifest.get("files", {}).get(kind, [])
                    if not paths:
                        continue
                    fingerprint = hashlib.sha256(f"legacy|{manifest['version']}|{kind}".encode()).hexdigest()
                    seen = self.repository.fingerprint_seen(symbol, timeframe, "none", fingerprint, _params_hash(timeframe, "end", "none", "legacy", PARSER_VERSION))
                    if resume and seen and seen.get("status") == "completed":
                        skipped.append({"symbol": symbol, "timeframe": timeframe})
                        continue
                    frame = self.store.read_files(paths)
                    if frame.empty:
                        continue
                    if "timestamp" not in frame and "date" in frame:
                        frame["timestamp"] = pd.to_datetime(frame.date)
                    frame["timestamp"] = pd.to_datetime(frame.timestamp)
                    if kind == "daily":
                        frame["source_verified"] = True
                    elif "source_verified" not in frame:
                        profiles = manifest.get("profiles", {})
                        frame["source_verified"] = any(profile.get("verified") for profile in profiles.values())
                    frame["source_verified"] = frame.source_verified.map(bool)
                    frame["is_synthetic"] = frame.get("is_synthetic", False)
                    frame["is_auction_placeholder"] = frame.get("is_auction_placeholder", False)
                    frame["source"] = frame.get("source", "legacy")
                    report = {"status": "ready" if frame.source_verified.all() else "pending_validation",
                              "missing_rows": 0, "synthetic_rows": int(frame.is_synthetic.sum()),
                              "gap_count": 0, "pending_validation": [] if frame.source_verified.all() else ["旧快照来源标签未核验"]}
                    result = self.repository.commit_bars(
                        symbol, timeframe, frame, source="legacy", file_name=manifest["version"],
                        fingerprint=fingerprint,
                        params_hash=_params_hash(timeframe, "end", "none", "legacy", PARSER_VERSION),
                        report=report, metadata=manifest.get("metadata", {}),
                    )
                    self.repository.save_coverage(symbol, timeframe, "none", frame.timestamp.min(), frame.timestamp.max(), report)
                    if kind == "daily":
                        calendar_paths = manifest.get("files", {}).get("calendar", [])
                        if calendar_paths:
                            calendar_frame = self.store.read_files(calendar_paths)
                            if not calendar_frame.empty and "date" in calendar_frame:
                                self.repository.save_auxiliary(
                                    symbol, "calendar", calendar_frame[["date"]].to_dict("records"),
                                    version=manifest["version"], verified=True,
                                )
                        action_paths = manifest.get("files", {}).get("actions", [])
                        if action_paths:
                            action_frame = self.store.read_files(action_paths)
                            self.repository.save_auxiliary(
                                symbol, "actions", action_frame.to_dict("records"),
                                version=manifest["version"], verified=bool(manifest.get("metadata", {}).get("actions_verified")),
                            )
                    migrated.append({"timeframe": timeframe, **result})
                if migrated:
                    self.service.refresh_snapshot(symbol)
                    self.store.state(f"market-data:migrated:{symbol}", {"source_version": manifest["version"], "at": now_iso()})
                results.append({"symbol": symbol, "migrated": migrated})
                progress(position / len(targets), f"{position}/{len(targets)} · {symbol} 已迁移")
            except Exception as exc:
                failures.append({"symbol": symbol, "reason": str(exc)})
            finally:
                if symbol_lock is not None and symbol_lock.is_locked:
                    symbol_lock.release()
        return {"status": "completed" if not failures else "partial", "targets": len(targets),
                "migrated": len(results), "skipped": len(skipped), "failed": len(failures),
                "results": results, "failures": failures, "backup": str(backup),
                "message": f"迁移完成 {len(results)} 个标的，跳过 {len(skipped)}，失败 {len(failures)}"}

    def _backup_catalog(self) -> Path:
        folder = self.store.root / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"catalog-before-market-migration-{pd.Timestamp.now():%Y%m%d-%H%M%S}.sqlite"
        if self.store.db_path.exists():
            with sqlite3.connect(self.store.db_path) as source, sqlite3.connect(target) as destination:
                source.backup(destination)
        return target

    def _count(self, manifest, kind):
        paths = manifest.get("files", {}).get(kind, [])
        return len(self.store.read_files(paths)) if paths else 0
