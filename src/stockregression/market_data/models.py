from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


class MarketDataError(ValueError):
    """A user-visible import, validation or readiness failure."""


class MarketSyncCancelled(InterruptedError):
    """A stopped sync retains an honest report of its durable partial work."""

    def __init__(self, report: dict[str, Any], message: str = "任务已取消"):
        super().__init__(message)
        self.report = report


class DataNotReady(MarketDataError):
    """Local data exists but cannot satisfy the requested calculation."""

    code = "DATA_NOT_READY"

    def __init__(self, message: str, *, gaps: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.gaps = gaps or []


class DataVersionNotFound(MarketDataError):
    """An explicitly requested immutable market version is unavailable."""

    code = "DATA_VERSION_NOT_FOUND"


@dataclass(frozen=True)
class ImportRequest:
    symbol: str
    timeframe: str
    path: Path
    label_convention: str = "auto"
    adjustment: str = "none"
    source: str = "import"
    etf_category: str | None = None


@dataclass
class FileScanItem:
    path: Path
    relative_path: str
    symbol: str | None
    timeframe: str
    size: int
    duplicate: bool = False
    error: str | None = None
    content_hash: str | None = None


@dataclass
class FileImportResult:
    symbol: str
    timeframe: str
    status: str
    version: int | None = None
    rows_read: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    synthetic: int = 0
    gap_count: int = 0
    pending_validation: list[str] = field(default_factory=list)
    message: str = ""
    fingerprint: str | None = None
    import_mode: str = "merge"
    changed_dates: list[str] = field(default_factory=list)
    shard_count: int = 0
    layout: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "timeframe": self.timeframe, "status": self.status,
            "version": self.version, "rows_read": self.rows_read,
            "inserted": self.inserted, "updated": self.updated,
            "unchanged": self.unchanged, "synthetic": self.synthetic,
            "gap_count": self.gap_count, "pending_validation": self.pending_validation,
            "message": self.message, "fingerprint": self.fingerprint,
            "import_mode": self.import_mode, "changed_dates": self.changed_dates,
            "shard_count": self.shard_count, "layout": self.layout,
            "metrics": self.metrics,
        }
