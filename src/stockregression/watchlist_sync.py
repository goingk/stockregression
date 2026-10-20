"""Import new entries from the versioned watchlist without replacing local choices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from filelock import FileLock

from .models import normalize_symbol
from .rules import instrument_terms
from .storage import Store, dumps, now_iso


def import_watchlist_snapshot(store: Store, path: Path) -> int:
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return 0
    digest = hashlib.sha256(content).hexdigest()
    key = "watchlist_snapshot:" + str(path.resolve())
    # Serialize checks and writes across concurrent refreshes and app instances.
    with FileLock(str(store.root / "watchlist-import.lock"), timeout=10):
        with store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
            previous = json.loads(row[0]) if row else {}
            if previous.get("digest") == digest:
                return 0
            snapshot = json.loads(content.decode("utf-8-sig"))
            if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
                raise ValueError("自选快照必须使用 schema_version=1")
            items = snapshot.get("items")
            if not isinstance(items, list):
                raise ValueError("自选快照 items 必须为列表")
            entries = {}
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("symbol"), str):
                    raise ValueError("自选条目必须包含证券代码 symbol")
                symbol = normalize_symbol(item["symbol"])
                name = item.get("name", symbol)
                created_at = item.get("created_at") or now_iso()
                if not isinstance(name, str) or not isinstance(created_at, str):
                    raise ValueError("自选名称和加入时间必须为字符串")
                entries.setdefault(symbol, (name or symbol, created_at))
            seen = set(previous.get("symbols", []))
            added = 0
            for symbol, (name, created_at) in entries.items():
                if symbol in seen:
                    continue
                metadata = instrument_terms(symbol, {"name": name})
                db.execute(
                    "INSERT OR IGNORE INTO instruments(symbol,name,exchange,board,metadata) VALUES(?,?,?,?,?)",
                    (symbol, name, symbol.split(".")[1], metadata["board"], dumps(metadata)),
                )
                added += db.execute(
                    "INSERT OR IGNORE INTO watchlist(symbol,created_at) VALUES(?,?)",
                    (symbol, created_at),
                ).rowcount
            db.execute(
                "INSERT OR REPLACE INTO state(key,value) VALUES(?,?)",
                (key, dumps({"digest": digest, "symbols": list(entries)})),
            )
    return added
