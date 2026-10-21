from __future__ import annotations

import os

from .performance import effective_cpu_budget
from .settings import PerformanceSettings
from .storage import Store, now_iso


CPU_JOB_KINDS = (
    "backtest", "optimization", "optimize", "batch", "batch_backtest", "research_completion",
    "market_validate", "market_prepare", "market_sync", "market_aggregate",
    "market_import", "market_import_batch", "market_import_package",
)


class ResourceLease:
    """Share the configured CPU budget between job processes.

    The lease is advisory with respect to the operating system. It prevents
    Numba, BLAS and validation workers launched by this application from
    multiplying their thread counts into an oversubscribed machine.

    The granted leases never sum above the configured budget: a job may only
    take what the other running compute jobs have left.  When more jobs are
    running than the budget can serve, the surplus jobs are refused with
    ``0`` units instead of being floored to one unit each.
    """

    def __init__(self, store: Store, job_id: str | None, settings: PerformanceSettings | None = None,
                 *, budget: int | None = None):
        self.store = store
        self.job_id = job_id or f"process-{os.getpid()}"
        self.settings = settings or PerformanceSettings()
        self.budget = (
            max(1, int(budget)) if budget is not None else effective_cpu_budget(self.settings)
        )
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
        return max(0, int(row[0] or 0))

    def _leased_units(self, connection, *, exclude: str | None = None) -> int:
        """Units already granted to the other running compute jobs."""
        row = connection.execute(
            "SELECT COALESCE(SUM(l.units), 0) FROM resource_leases l "
            "WHERE l.kind='cpu' AND l.job_id<>? AND EXISTS ("
            "SELECT 1 FROM jobs j WHERE j.id=l.job_id AND j.status='running')",
            (exclude or "",),
        ).fetchone()
        return max(0, int(row[0] or 0))

    def _share(self, connection) -> int:
        """Units this job may hold without pushing the leases above the budget.

        ``cap`` is the hard limit -- the budget minus what the other running
        compute jobs already hold.  ``fair`` is the even share among the
        running jobs.  When the budget cannot be split evenly the share falls
        back to a single unit, but only while the cap still allows it, so the
        granted units never sum above the configured budget.
        """
        active = max(1, self._active_jobs(connection))
        budget = max(1, int(self.budget))
        cap = max(0, budget - self._leased_units(connection, exclude=self.job_id))
        fair = budget // active
        if fair >= 1:
            return min(cap, fair)
        return min(cap, 1)

    def acquire(self) -> int:
        with self.store.db() as db:
            # Counting the running jobs and granting the units must be one
            # transaction: two jobs that start at the same time must not be
            # handed the same units.
            db.execute("BEGIN IMMEDIATE")
            units = self._share(db)
            db.execute(
                "INSERT OR REPLACE INTO resource_leases(job_id,pid,kind,units,updated_at) VALUES(?,?,?,?,?)",
                (self.job_id, os.getpid(), "cpu", units, now_iso()),
            )
        self.units = units
        return self.units

    def rebalance(self) -> int:
        """Reduce this job's share when another compute job starts."""
        if not self.job_id:
            return max(1, self.budget)
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT units FROM resource_leases WHERE job_id=?", (self.job_id,)
            ).fetchone()
            if row is None:
                missing = True
                current = 0
            else:
                missing = False
                current = max(0, int(row[0]))
                units = self._share(db)
                if current >= 1:
                    # The performance layer never runs a job below one unit,
                    # so the ledger must not claim less than what is used:
                    # a lease row of zero would hide the unit the job still
                    # occupies and let the next job oversubscribe the budget.
                    units = max(1, units)
                if units != int(row[0]):
                    db.execute(
                        "UPDATE resource_leases SET units=?,pid=?,updated_at=? WHERE job_id=?",
                        (units, os.getpid(), now_iso(), self.job_id),
                    )
                self.units = units
        if missing:
            # The lease row disappeared while the job was running.  Take a
            # fresh share and restore the row so the ledger keeps counting
            # the units this job really occupies.
            units = self.acquire()
            if units <= 0:
                units = 1
                with self.store.db() as db:
                    db.execute(
                        "UPDATE resource_leases SET units=?,pid=?,updated_at=? WHERE job_id=?",
                        (units, os.getpid(), now_iso(), self.job_id),
                    )
            self.units = units
            return units
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

