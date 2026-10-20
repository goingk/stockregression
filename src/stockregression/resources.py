from __future__ import annotations

import os

from .performance import effective_cpu_budget
from .settings import PerformanceSettings
from .storage import Store, now_iso


CPU_JOB_KINDS = (
    "backtest", "optimization", "optimize", "batch", "batch_backtest",
    "market_validate", "market_prepare", "market_sync", "market_aggregate",
    "market_import", "market_import_batch", "market_import_package",
)


class ResourceLease:
    """Share the configured CPU budget between job processes.

    The lease is advisory with respect to the operating system. It prevents
    Numba, BLAS and validation workers launched by this application from
    multiplying their thread counts into an oversubscribed machine.
    """

    def __init__(self, store: Store, job_id: str | None, settings: PerformanceSettings | None = None):
        self.store = store
        self.job_id = job_id or f"process-{os.getpid()}"
        self.settings = settings or PerformanceSettings()
        self.budget = effective_cpu_budget(self.settings)
        self.units = 0

    @property
    def workspace_bytes(self) -> int:
        return int(self.settings.workspace_mb) * 1024 * 1024

    def _active_jobs(self, connection) -> int:
        placeholders = ",".join("?" for _ in CPU_JOB_KINDS)
        row = connection.execute(
            f"SELECT COUNT(DISTINCT id) FROM jobs WHERE status='running' AND kind IN ({placeholders})",
            CPU_JOB_KINDS,
        ).fetchone()
        return max(1, int(row[0] or 0))

    def acquire(self) -> int:
        with self.store.db() as db:
            active = self._active_jobs(db)
            self.units = max(1, min(self.budget, self.budget // active))
            db.execute(
                "INSERT OR REPLACE INTO resource_leases(job_id,pid,kind,units,updated_at) VALUES(?,?,?,?,?)",
                (self.job_id, os.getpid(), "cpu", self.units, now_iso()),
            )
        return self.units

    def rebalance(self) -> int:
        """Reduce this job's share when another compute job starts."""
        if not self.job_id:
            return max(1, self.budget)
        with self.store.db() as db:
            row = db.execute(
                "SELECT units FROM resource_leases WHERE job_id=?", (self.job_id,)
            ).fetchone()
            if row is None:
                return self.acquire()
            active = self._active_jobs(db)
            fair = max(1, min(self.budget, self.budget // active))
            units = min(int(row[0]), fair) if active > 1 else fair
            if units != int(row[0]):
                db.execute(
                    "UPDATE resource_leases SET units=?,pid=?,updated_at=? WHERE job_id=?",
                    (units, os.getpid(), now_iso(), self.job_id),
                )
            self.units = units
        return self.units

    def release(self) -> None:
        with self.store.db() as db:
            db.execute("DELETE FROM resource_leases WHERE job_id=? AND pid=?", (self.job_id, os.getpid()))
        self.units = 0


def clear_stale_leases(store: Store) -> int:
    """Drop rows whose application job is no longer running."""
    with store.db() as db:
        cursor = db.execute(
            """DELETE FROM resource_leases
               WHERE NOT EXISTS (
                   SELECT 1 FROM jobs j WHERE j.id=resource_leases.job_id AND j.status='running'
               )"""
        )
        return int(cursor.rowcount or 0)

