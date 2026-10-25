"""Bounded latest-only background previews for research preflight estimates."""
from collections import OrderedDict
import threading
import time

import numpy as np

_REVISION_TOMBSTONE_TTL = 3600.0
_MAX_REVISION_SESSIONS = 1024


class ResearchPreviewManager:
    """Run one research preflight at a time, retaining only each session's latest request.

    ``callback(config, cancelled)`` receives a cancellation predicate and must
    check it between expensive stages. A returned dictionary is stored as the
    result for that revision.
    """

    def __init__(self, callback, max_sessions=16):
        self.callback = callback
        self.max_sessions = max(1, int(max_sessions))
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopped = False
        self.records = OrderedDict()
        self.revisions = {}
        self.cancelled_revisions = {}
        self.revision_touched = {}
        self.thread = None

    def submit(self, session_id, revision, config, estimate_key=None):
        with self.lock:
            self._prune_revisions_locked()
            previous = self.records.get(session_id)
            watermark = self.cancelled_revisions.get(session_id, -1)
            if revision <= max(watermark, self.revisions.get(session_id, -1)):
                if previous and revision == previous["revision"]:
                    return self._snapshot_locked(session_id)
                return {"session_id": session_id, "revision": revision, "status": "cancelled",
                        "progress": 0.0, "message": "研究预估已取消", "result": None,
                        "error": None, "estimate_key": estimate_key}
            if previous and revision <= previous["revision"]:
                return self._snapshot_locked(session_id)
            if previous:
                previous["cancel_flag"][0] = 1
            self.records[session_id] = {
                "session_id": session_id, "revision": revision,
                "status": "queued", "progress": 0.0,
                "message": "等待研究预估", "result": None, "error": None,
                "estimate_key": estimate_key, "config": config,
                "cancel_flag": np.zeros(1, dtype=np.int32),
                "updated": time.monotonic(),
            }
            self.revisions[session_id] = revision
            self.revision_touched[session_id] = time.monotonic()
            self.records.move_to_end(session_id)
            while len(self.records) > self.max_sessions:
                oldest, record = self.records.popitem(last=False)
                record["cancel_flag"][0] = 1
            self._prune_revisions_locked()
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, daemon=True, name="research-preview")
                self.thread.start()
            self.wake.set()
            return self._snapshot_locked(session_id)

    def _snapshot_locked(self, session_id):
        record = self.records.get(session_id)
        if not record:
            return None
        return {key: value for key, value in record.items()
                if key not in {"config", "updated", "cancel_flag"}}

    def snapshot(self, session_id):
        with self.lock:
            return self._snapshot_locked(session_id)

    def cancel(self, session_id, through_revision=None):
        with self.lock:
            self._prune_revisions_locked()
            record = self.records.get(session_id)
            through = (max(self.cancelled_revisions.get(session_id, -1),
                           record["revision"] if record else -1)
                       if through_revision is None else through_revision)
            self.cancelled_revisions[session_id] = max(self.cancelled_revisions.get(session_id, -1), through)
            self.revisions[session_id] = max(self.revisions.get(session_id, -1), through)
            self.revision_touched[session_id] = time.monotonic()
            if not record or record["revision"] > through:
                self._prune_revisions_locked()
                return self._snapshot_locked(session_id) or {
                    "session_id": session_id, "revision": through, "status": "cancelled",
                    "progress": 0.0, "message": "研究预估已取消", "result": None,
                    "error": None, "estimate_key": None,
                }
            record["cancel_flag"][0] = 1
            record.update(status="cancelled", message="研究预估已取消")
            self._prune_revisions_locked()
            return self._snapshot_locked(session_id)

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

    def _loop(self):
        while True:
            self.wake.wait(.5)
            self.wake.clear()
            while True:
                with self.lock:
                    if self.stopped:
                        return
                    entry = next(((sid, dict(row)) for sid, row in self.records.items()
                                  if row["status"] == "queued"), None)
                    if entry is None:
                        break
                    session_id, request = entry
                    current = self.records.get(session_id)
                    if current is None or current["revision"] != request["revision"]:
                        continue
                    current.update(status="running", message="正在预估研究规模")

                def cancelled():
                    return bool(request["cancel_flag"][0]) or self.stopped

                try:
                    result = self.callback(request["config"], cancelled)
                    with self.lock:
                        current = self.records.get(session_id)
                        if current and current["revision"] == request["revision"] and not cancelled():
                            current.update(status="completed", progress=1.0,
                                           message="研究预估完成", result=result)
                except InterruptedError:
                    with self.lock:
                        current = self.records.get(session_id)
                        if current and current["revision"] == request["revision"]:
                            current.update(status="cancelled", message="研究预估已取消")
                except Exception as exc:
                    with self.lock:
                        current = self.records.get(session_id)
                        if current and current["revision"] == request["revision"] and not cancelled():
                            error = ({"status": exc.status_code, "detail": exc.detail}
                                     if hasattr(exc, "status_code") and hasattr(exc, "detail") else str(exc))
                            current.update(status="failed", message="研究预估失败", error=error)

    def stop(self):
        with self.lock:
            self.stopped = True
            for record in self.records.values():
                if record["status"] in {"queued", "running"}:
                    record["cancel_flag"][0] = 1
            self.wake.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
