import time
import threading

from fastapi import HTTPException

from stockregression.preview import PreviewManager
from stockregression.research_preview import ResearchPreviewManager


def test_backtest_delete_tombstone_rejects_late_post_and_newer_revision_survives():
    manager = PreviewManager(None)
    deleted = manager.cancel("s", through_revision=5)
    assert deleted["status"] == "cancelled"
    assert manager.submit("s", 4, object())["status"] == "cancelled"
    assert manager.submit("s", 5, object())["status"] == "cancelled"
    assert manager.submit("s", 6, object())["revision"] == 6
    manager.stop()


def test_research_delete_tombstone_rejects_late_post_and_higher_through_revision():
    manager = ResearchPreviewManager(lambda config, cancelled: {})
    manager.submit("s", 2, {})
    state = manager.cancel("s", through_revision=5)
    assert state["revision"] == 2 and state["status"] == "cancelled"
    assert manager.submit("s", 4, {})["status"] == "cancelled"
    assert manager.submit("s", 5, {})["status"] == "cancelled"
    assert manager.submit("s", 6, {})["revision"] == 6
    manager.stop()


def test_research_preview_preserves_http_diagnostic_detail():
    def fail(_config, _cancelled):
        raise HTTPException(status_code=409, detail={"code": "DATA_NOT_READY", "source": "local"})

    manager = ResearchPreviewManager(fail)
    try:
        manager.submit("s", 1, {})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = manager.snapshot("s")
            if state["status"] == "failed":
                break
            time.sleep(.01)
        assert state["status"] == "failed"
        assert state["error"] == {"status": 409, "detail": {"code": "DATA_NOT_READY", "source": "local"}}
    finally:
        manager.stop()


def test_research_preview_processes_multiple_sessions():
    manager = ResearchPreviewManager(lambda config, _cancelled: {"value": config["value"]})
    try:
        manager.submit("a", 1, {"value": 10})
        manager.submit("b", 1, {"value": 20})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            states = manager.snapshot("a"), manager.snapshot("b")
            if all(state["status"] == "completed" for state in states):
                break
            time.sleep(.01)
        assert [state["result"] for state in states] == [{"value": 10}, {"value": 20}]
    finally:
        manager.stop()


def test_research_preview_keeps_only_first_running_and_latest_pending():
    started = threading.Event()
    release = threading.Event()
    calls = []

    def callback(config, cancelled):
        calls.append(config)
        if config == "first":
            started.set()
            assert release.wait(3)
            if cancelled():
                raise InterruptedError
        return {"value": config}

    manager = ResearchPreviewManager(callback)
    try:
        manager.submit("s", 1, "first")
        assert started.wait(3)
        manager.submit("s", 2, "middle")
        manager.submit("s", 3, "last")
        manager.submit("s", 4, "latest")
        release.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and manager.snapshot("s")["status"] != "completed":
            time.sleep(.01)
        state = manager.snapshot("s")
        assert calls == ["first", "latest"]
        assert state["revision"] == 4 and state["result"] == {"value": "latest"}
    finally:
        release.set()
        manager.stop()


def test_revision_tombstone_store_is_bounded():
    manager = ResearchPreviewManager(lambda _config, _cancelled: {})
    for index in range(1030):
        manager.cancel(f"inactive-{index}", through_revision=index)
    assert len(manager.cancelled_revisions) <= 1024
    assert len(manager.revisions) <= 1024
    manager.stop()
