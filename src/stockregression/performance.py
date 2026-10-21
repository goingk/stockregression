from __future__ import annotations

import contextlib
import contextvars
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock, Timeout


DEFAULT_CPU_FRACTION = 0.75
DEFAULT_DOWNLOAD_WORKERS = 4
DEFAULT_VALIDATION_WORKERS = 4
DEFAULT_BLAS_THREADS = 8
DEFAULT_WORKSPACE_MB = 8192


def logical_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def automatic_cpu_budget() -> int:
    return max(1, int(math.floor(logical_cpu_count() * DEFAULT_CPU_FRACTION)))


def effective_cpu_budget(performance: Any = None) -> int:
    """Resolve the process-wide compute budget from persisted settings."""
    configured = getattr(performance, "cpu_budget", None)
    mode = getattr(performance, "cpu_budget_mode", "auto")
    if mode == "custom" and configured:
        return max(1, min(logical_cpu_count(), int(configured)))
    return automatic_cpu_budget()


def effective_setting(performance: Any, name: str, default: int, maximum: int) -> int:
    value = getattr(performance, name, default) if performance is not None else default
    try:
        return max(1, min(int(maximum), int(value)))
    except (TypeError, ValueError):
        return max(1, min(int(maximum), int(default)))


@contextlib.contextmanager
def blas_thread_limit(limit: int) -> Iterator[None]:
    """Bound native BLAS/OpenMP pools for matrix-heavy phases."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - declared runtime dependency
        yield
        return
    with threadpool_limits(limits=max(1, int(limit))):
        yield


@dataclass
class PerformanceTrace:
    job_id: str | None = None
    started_at: float = field(default_factory=time.perf_counter)
    timings: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_seconds(self, stage: str, seconds: float) -> None:
        with self._lock:
            self.timings[stage] = round(self.timings.get(stage, 0.0) + max(0.0, seconds), 6)

    def add_count(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + int(value)

    def update(self, **values: Any) -> None:
        with self._lock:
            self.details.update(values)

    @contextlib.contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_seconds(stage, time.perf_counter() - started)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "total_seconds": round(time.perf_counter() - self.started_at, 6),
                "timings": dict(sorted(self.timings.items())),
                "counters": dict(sorted(self.counters.items())),
                **self.details,
            }


_TRACE: contextvars.ContextVar[PerformanceTrace | None] = contextvars.ContextVar(
    "stockregression_performance_trace", default=None
)
_CPU_BUDGET: contextvars.ContextVar[int] = contextvars.ContextVar(
    "stockregression_cpu_budget", default=automatic_cpu_budget()
)
_WORKSPACE_BYTES: contextvars.ContextVar[int] = contextvars.ContextVar(
    "stockregression_workspace_bytes", default=DEFAULT_WORKSPACE_MB * 1024 * 1024
)
_REBALANCE: contextvars.ContextVar = contextvars.ContextVar(
    "stockregression_resource_rebalance", default=None
)
_BLAS_THREADS: contextvars.ContextVar[int] = contextvars.ContextVar(
    "stockregression_blas_threads", default=DEFAULT_BLAS_THREADS
)


def activate_performance(
    *, cpu_budget: int, workspace_mb: int, trace: PerformanceTrace | None = None,
    rebalance=None, blas_threads: int = DEFAULT_BLAS_THREADS,
) -> tuple[contextvars.Token, ...]:
    return (
        _CPU_BUDGET.set(max(1, int(cpu_budget))),
        _WORKSPACE_BYTES.set(max(256, int(workspace_mb)) * 1024 * 1024),
        _TRACE.set(trace),
        _REBALANCE.set(rebalance),
        _BLAS_THREADS.set(max(1, int(blas_threads))),
    )


def reset_performance(tokens) -> None:
    for token in reversed(tokens):
        token.var.reset(token)


def current_cpu_budget() -> int:
    return max(1, int(_CPU_BUDGET.get()))


def rebalance_cpu_budget() -> int:
    callback = _REBALANCE.get()
    if callback is None:
        return current_cpu_budget()
    units = max(1, int(callback()))
    _CPU_BUDGET.set(units)
    return units


def current_workspace_bytes() -> int:
    return max(256 * 1024 * 1024, int(_WORKSPACE_BYTES.get()))


_WINDOWS_MEMORY_COUNTERS: Any = None


def _windows_rss_bytes() -> int:
    """Read ``WorkingSetSize`` for this process through ``psapi``."""
    import ctypes
    from ctypes import wintypes

    global _WINDOWS_MEMORY_COUNTERS
    if _WINDOWS_MEMORY_COUNTERS is None:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        _WINDOWS_MEMORY_COUNTERS = ProcessMemoryCounters
    counters = _WINDOWS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.GetCurrentProcess()
    call = None
    for name in ("K32GetProcessMemoryInfo", "GetProcessMemoryInfo"):
        candidate = getattr(kernel32, name, None)
        if candidate is not None:
            call = candidate
            break
    if call is None:  # pragma: no cover - legacy Windows without kernel32 export
        call = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    call.argtypes = [wintypes.HANDLE, ctypes.POINTER(_WINDOWS_MEMORY_COUNTERS), wintypes.DWORD]
    call.restype = wintypes.BOOL
    if not call(handle, ctypes.byref(counters), counters.cb):
        return 0
    return int(counters.WorkingSetSize)


def _posix_rss_bytes() -> int:
    """Read the kernel's maximum resident size for this process."""
    try:
        import resource
    except ImportError:  # pragma: no cover - POSIX without resource
        return 0
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux and the BSDs report kilobytes, macOS reports bytes.
    return usage if sys.platform == "darwin" else usage * 1024


def current_process_rss_bytes() -> int:
    """Resident set size of the current process in bytes.

    Windows reads the live working set through ``GetProcessMemoryInfo``;
    POSIX falls back to the kernel's peak resident size.  The sampler never
    raises: an unavailable counter reports ``0`` so a caller can skip the
    memory check instead of stopping the task.
    """
    try:
        if os.name == "nt":
            return max(0, _windows_rss_bytes())
        return max(0, _posix_rss_bytes())
    except Exception:  # pragma: no cover - defensive: never break a task
        return 0


def current_blas_threads() -> int:
    return max(1, min(logical_cpu_count(), int(_BLAS_THREADS.get())))


def current_trace() -> PerformanceTrace:
    return _TRACE.get() or PerformanceTrace()


def record_seconds(stage: str, seconds: float) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.add_seconds(stage, seconds)


def record_count(name: str, value: int = 1) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.add_count(name, value)


def record_details(**values: Any) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.update(**values)


class SharedSlots:
    """File-lock backed capacity shared by every process using one data root."""

    def __init__(self, root: str | Path, name: str, capacity: int):
        self.folder = Path(root) / ".resource-slots"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
        self.capacity = max(1, int(capacity))

    @contextlib.contextmanager
    def acquire(self, cancelled=lambda: False) -> Iterator[int]:
        while True:
            if cancelled():
                raise InterruptedError("任务已取消")
            for slot in range(self.capacity):
                lock = FileLock(str(self.folder / f"{self.name}-{slot}.lock"))
                try:
                    lock.acquire(timeout=0)
                except Timeout:
                    continue
                try:
                    yield slot
                finally:
                    lock.release()
                return
            time.sleep(0.05)
