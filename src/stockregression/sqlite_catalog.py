"""Short-lived catalog connections with bounded setup retries and cleanup."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def catalog_connection(path: Path, timeout: int, *, synchronous: str | None = None):
    connection = None
    for attempt in range(3):
        connection = None
        try:
            connection = sqlite3.connect(path, timeout=timeout)
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout={timeout * 1000}")
            # Reading the current mode avoids requesting a mode change on every
            # connection while several workers are opening the same WAL file.
            if connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                connection.execute("PRAGMA journal_mode=WAL")
            if synchronous is not None:
                connection.execute(f"PRAGMA synchronous={synchronous}")
            break
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
            if attempt == 2 or code not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR}:
                exc.add_note(f"catalog={path}; SQLite={getattr(exc, 'sqlite_errorname', 'unknown')}")
                raise
            time.sleep(0.05 * 2 ** attempt)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
