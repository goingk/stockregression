"""Bounded, non-persistent previews; one latest pending input per page session."""
from collections import OrderedDict
import threading
import time
from .data import DataService
from .engine import run_backtest


class PreviewManager:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopped = False
        self.records = OrderedDict()
        self.bundles = OrderedDict()
        self.thread = None

    def submit(self, session, revision, config, version=None):
        with self.lock:
            previous = self.records.get(session)
            if previous and revision <= previous["revision"]:
                return self.snapshot(session)
            self.records[session] = {"session_id": session, "revision": revision, "status": "queued", "progress": 0.0, "message": "等待检查行情数据", "config": config, "version": version, "updated": time.monotonic()}
            self.records.move_to_end(session)
            while len(self.records) > 16:
                self.records.popitem(last=False)
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
            return {key: value for key, value in record.items() if key not in {"config", "updated"}}

    def cancel(self, session):
        with self.lock:
            if session in self.records:
                self.records[session]["status"] = "cancelled"

    @staticmethod
    def _data_key(config):
        return (config.symbol, str(config.start_date), str(config.end_date), config.frequency, config.schema_version)

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
                    with self.lock:
                        current = self.records.get(session)
                        return self.stopped or not current or current["revision"] != request["revision"] or current["status"] == "cancelled"
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
                    result = run_backtest(config, bundle, lambda fraction, message: report(0.42 + 0.58 * max(0.0, min(1.0, fraction)), message), cancelled=cancelled, preview=True)
                    with self.lock:
                        current = self.records.get(session)
                        if current and current["revision"] == request["revision"] and not cancelled():
                            current.update(status="completed", progress=1.0, message="预览完成", version=version, result=result)
                except InterruptedError:
                    pass
                except Exception as exc:
                    if not cancelled():
                        update(status="failed", message="预览失败", error=str(exc))

    def stop(self):
        self.stopped = True
        self.wake.set()
