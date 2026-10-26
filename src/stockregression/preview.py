"""Bounded, non-persistent previews; one latest pending input per page session."""
from collections import OrderedDict
import threading
import time
import numpy as np
from .data import DataService
from .engine import run_backtest

_REVISION_TOMBSTONE_TTL = 3600.0
_MAX_REVISION_SESSIONS = 1024


class PreviewManager:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopped = False
        self.records = OrderedDict()
        self.revisions = {}
        self.cancelled_revisions = {}
        self.revision_touched = {}
        self.bundles = OrderedDict()
        self.thread = None

    def submit(self, session, revision, config, version=None):
        with self.lock:
            self._prune_revisions_locked()
            previous = self.records.get(session)
            if revision <= max(self.revisions.get(session, -1), self.cancelled_revisions.get(session, -1)):
                if previous and previous["revision"] == revision:
                    return self.snapshot(session)
                return {"session_id": session, "revision": revision, "status": "cancelled", "progress": 0.0,
                        "message": "预览已取消", "version": None, "result": None}
            if previous:
                previous["cancel_flag"][0] = 1
            self.revisions[session] = revision
            self.revision_touched[session] = time.monotonic()
            self.records[session] = {"session_id": session, "revision": revision, "status": "queued", "progress": 0.0, "message": "等待检查行情数据", "config": config, "version": version, "cancel_flag": np.zeros(1, dtype=np.int32), "updated": time.monotonic()}
            self.records.move_to_end(session)
            while len(self.records) > 16:
                _, evicted = self.records.popitem(last=False)
                evicted["cancel_flag"][0] = 1
            self._prune_revisions_locked()
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, daemon=True, name="backtest-preview")
                self.thread.start()
            self.wake.set()
            return self.snapshot(session)

    def snapshot(self, session):
        with self.lock:
            record = self.records.get(session)
            if not record:
                return None
            return {key: value for key, value in record.items() if key not in {"config", "updated", "cancel_flag"}}

    def cancel(self, session, through_revision=None):
        with self.lock:
            self._prune_revisions_locked()
            through = self.revisions.get(session, -1) if through_revision is None else through_revision
            self.cancelled_revisions[session] = max(self.cancelled_revisions.get(session, -1), through)
            self.revision_touched[session] = time.monotonic()
            current = self.records.get(session)
            if current is None:
                self.revisions[session] = max(self.revisions.get(session, -1), through)
                self._prune_revisions_locked()
                return {"session_id": session, "revision": through, "status": "cancelled", "progress": 0.0,
                        "message": "预览已取消", "version": None, "result": None}
            if current["revision"] <= through:
                current["cancel_flag"][0] = 1
                current.update(status="cancelled", message="预览已取消")
            self.revisions[session] = max(self.revisions.get(session, -1), through)
            self._prune_revisions_locked()
            return self.snapshot(session)

    def _prune_revisions_locked(self):
        """Keep active fences; lazily expire inactive entries after 1h and cap them at 1024 sessions."""
        now = time.monotonic()
        active = {key for key, row in self.records.items() if row["status"] in {"queued", "running"}}
        stale = [key for key, touched in self.revision_touched.items()
                 if key not in active and now - touched > _REVISION_TOMBSTONE_TTL]
        for key in stale:
            self.revision_touched.pop(key, None)
            self.revisions.pop(key, None)
            self.cancelled_revisions.pop(key, None)
        inactive = sorted((touched, key) for key, touched in self.revision_touched.items()
                          if key not in active)
        while len(self.revision_touched) > _MAX_REVISION_SESSIONS and inactive:
            _, key = inactive.pop(0)
            self.revision_touched.pop(key, None)
            self.revisions.pop(key, None)
            self.cancelled_revisions.pop(key, None)

    @staticmethod
    def _data_key(config):
        reference = (config.dca.reference, config.dca.reference_symbol, config.dca.reference_version) if config.dca.mode != "fixed_shares" else None
        return (config.symbol, str(config.start_date), str(config.end_date), config.frequency, config.schema_version, reference)

    @staticmethod
    def _bundle_key(version, config):
        return (version, *PreviewManager._data_key(config))

    def _loop(self):
        while not self.stopped:
            self.wake.wait(.5)
            self.wake.clear()
            while not self.stopped:
                with self.lock:
                    entry = next(((key, dict(value)) for key, value in self.records.items() if value["status"] == "queued"), None)
                    if entry is None:
                        break
                    session, request = entry
                    self.records[session].update(status="running", message="检查本地行情并准备预览", progress=0.01)
                def cancelled():
                    return self.stopped or bool(request["cancel_flag"][0])
                def preparation_cancelled():
                    with self.lock:
                        current = self.records.get(session)
                        if self.stopped or not current or current["status"] == "cancelled":
                            return True
                        if current["revision"] != request["revision"]:
                            # A grid/strategy edit with the same data range can
                            # share this in-flight preparation. The stale
                            # calculation is still cancelled after preparation.
                            return self._data_key(current["config"]) != self._data_key(request["config"])
                        return False
                def update(**values):
                    with self.lock:
                        current = self.records.get(session)
                        if current and current["revision"] == request["revision"] and current["status"] == "running":
                            current.update(values)
                def report(fraction, message):
                    update(progress=max(0.0, min(1.0, float(fraction))), message=str(message))
                try:
                    config, version = request["config"], request["version"]
                    service = DataService(self.store)
                    active = self.store.snapshot(config.symbol, version)
                    lookup_version = version or (active["version"] if active else None)
                    cache_key = self._bundle_key(lookup_version, config) if lookup_version else None
                    with self.lock:
                        cached_bundle = self.bundles.get(cache_key) if cache_key else None
                        if cached_bundle is not None:
                            self.bundles.move_to_end(cache_key)
                    prepared_bundle = service.ensure_bundle(
                        config,
                        version,
                        lambda fraction, message: report(0.05 + 0.35 * max(0.0, min(1.0, fraction)), message),
                        preparation_cancelled,
                        cached_bundle=cached_bundle,
                    )
                    version = prepared_bundle.version
                    key = self._bundle_key(version, config)
                    with self.lock:
                        self.bundles[key] = prepared_bundle
                        self.bundles.move_to_end(key)
                        while len(self.bundles) > 2:
                            self.bundles.popitem(last=False)
                    if cancelled():
                        continue
                    update(version=version, progress=0.42, message="行情已检查并锁定；正在计算预览")
                    with self.lock:
                        bundle = self.bundles.get(key)
                        bundle = bundle or prepared_bundle
                        self.bundles.move_to_end(key)
                    result = run_backtest(config, bundle, lambda fraction, message: report(0.42 + 0.58 * max(0.0, min(1.0, fraction)), message), cancelled=cancelled, preview=True, cancel_flag=request["cancel_flag"])
                    with self.lock:
                        current = self.records.get(session)
                        if current and current["revision"] == request["revision"] and not cancelled():
                            current.update(status="completed", progress=1.0, message="预览完成", version=version, result=result)
                except InterruptedError:
                    pass
                except Exception as exc:
                    if not cancelled():
                        error = ({"status": exc.status_code, "detail": exc.detail}
                                 if hasattr(exc, "status_code") and hasattr(exc, "detail") else str(exc))
                        update(status="failed", message="预览失败", error=error)

    def stop(self):
        with self.lock:
            self.stopped = True
            for record in self.records.values():
                record["cancel_flag"][0] = 1
            self.wake.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
