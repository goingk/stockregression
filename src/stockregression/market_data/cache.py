from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable

import pandas as pd


DEFAULT_CACHE_MB = 256


def estimate_bytes(value: Any) -> int:
    """Estimate the memory retained by a cached monetary/market object."""
    if value is None:
        return 0
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(index=True, deep=True))
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return sum(estimate_bytes(key) + estimate_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(estimate_bytes(item) for item in value)
    return 1024


@dataclass
class _Entry:
    value: Any
    size: int


class ByteLRU:
    """Copy-on-read cache with a shared byte budget and single-flight loads."""

    def __init__(self, budget_mb: int = DEFAULT_CACHE_MB):
        self._lock = threading.RLock()
        self._entries: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._inflight: dict[Hashable, threading.Event] = {}
        self._budget = max(0, int(budget_mb)) * 1024 * 1024
        self._used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    def configure(self, budget_mb: int) -> None:
        with self._lock:
            self._budget = max(0, int(budget_mb)) * 1024 * 1024
            self._trim_locked(0)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._used = 0

    def get_or_load(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        if self._budget <= 0:
            return copy.deepcopy(loader())
        owner = False
        ready = None
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(entry.value)
            ready = self._inflight.get(key)
            if ready is None:
                ready = threading.Event()
                self._inflight[key] = ready
                owner = True
                self.misses += 1
        if not owner:
            ready.wait()
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return copy.deepcopy(entry.value)
            # The owner failed or the object did not fit. Retry independently.
            return copy.deepcopy(loader())
        try:
            value = loader()
            size = estimate_bytes(value)
            with self._lock:
                if 0 < size <= self._budget:
                    existing = self._entries.pop(key, None)
                    if existing:
                        self._used -= existing.size
                    self._trim_locked(size)
                    self._entries[key] = _Entry(copy.deepcopy(value), size)
                    self._used += size
            return copy.deepcopy(value)
        finally:
            with self._lock:
                ready.set()
                self._inflight.pop(key, None)

    def _trim_locked(self, incoming: int) -> None:
        if self._budget <= 0:
            while self._entries:
                _, entry = self._entries.popitem(last=False)
                self._used -= entry.size
            return
        while self._entries and self._used + incoming > self._budget:
            _, entry = self._entries.popitem(last=False)
            self._used -= entry.size
            self.evictions += 1


_cache = ByteLRU(DEFAULT_CACHE_MB)


def configure_cache(budget_mb: int) -> None:
    _cache.configure(budget_mb)


def cache_settings() -> dict[str, int]:
    return {
        "budget_bytes": _cache.budget_bytes,
        "used_bytes": _cache.used_bytes,
        "hits": _cache.hits,
        "misses": _cache.misses,
        "evictions": _cache.evictions,
    }


def cached(key: Hashable, loader: Callable[[], Any]) -> Any:
    return _cache.get_or_load(key, loader)
