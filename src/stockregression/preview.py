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

    def submit(self, session, revision, config, version):
        with self.lock:
            previous = self.records.get(session)
            if previous and revision <= previous["revision"]:
                return self.snapshot(session)
            self.records[session] = {"session_id": session, "revision": revision, "status": "queued", "config": config, "version": version, "updated": time.monotonic()}
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
                    self.records[session]["status"] = "running"
                def cancelled():
                    with self.lock:
                        current = self.records.get(session)
                        return self.stopped or not current or current["revision"] != request["revision"] or current["status"] == "cancelled"
                try:
                    config, version = request["config"], request["version"]
                    key = (version, config.symbol, config.start_date, config.end_date, config.frequency)
                    bundle = self.bundles.get(key)
                    if bundle is None:
                        bundle = DataService(self.store).load_bundle(config, version)
                        self.bundles[key] = bundle
                        while len(self.bundles) > 2:
                            self.bundles.popitem(last=False)
                    result = run_backtest(config, bundle, cancelled=cancelled, preview=True)
                    with self.lock:
                        if not cancelled():
                            self.records[session].update(status="completed", result=result)
                except InterruptedError:
                    pass
                except Exception as exc:
                    with self.lock:
                        if not cancelled():
                            self.records[session].update(status="failed", error=str(exc))

    def stop(self):
        self.stopped = True
        self.wake.set()
