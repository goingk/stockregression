from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stockregression.market_data import MarketDataService
from stockregression.market_data.sync import latest_closed_date
from stockregression.models import normalize_symbol
from stockregression.settings import load_settings
from stockregression.storage import Store


def _filter_basic_metadata(path: Path, excluded: set[str]) -> int:
    if not path.is_file() or not excluded:
        return 0
    table = pq.read_table(path)
    if "ts_code" not in table.column_names:
        return 0
    symbols = []
    for value in table["ts_code"].to_pylist():
        try:
            symbols.append(normalize_symbol(str(value)) if value else "")
        except (TypeError, ValueError):
            symbols.append("")
    keep = [symbol not in excluded for symbol in symbols]
    removed = len(keep) - sum(keep)
    if not removed:
        return 0
    temporary = path.with_name(path.name + ".purge-writing")
    try:
        pq.write_table(table.filter(pa.array(keep)), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Permanently exclude long-stagnant fund datasets.",
    )
    parser.add_argument("--older-than-days", type=int, default=365)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.older_than_days < 1:
        raise SystemExit("--older-than-days 必须为正数")

    settings = load_settings()
    store = Store(settings.data_dir, settings=settings)
    if store.active_job_count():
        raise SystemExit("仍有活动任务，停止更新后再清除长期停滞基金")
    service = MarketDataService(store, settings=settings)
    repository = service.repository
    cutoff = pd.Timestamp(latest_closed_date()).normalize()
    threshold = cutoff - pd.Timedelta(days=args.older_than_days)
    with repository._connect() as db:
        rows = db.execute(
            "SELECT d.symbol, d.last_time, i.name "
            "FROM parquet_datasets d "
            "LEFT JOIN instruments i ON i.symbol=d.symbol "
            "WHERE d.timeframe='1day' AND d.row_count>0 AND d.last_time<? "
            "ORDER BY d.symbol",
            (threshold.strftime("%Y-%m-%d"),),
        ).fetchall()
    candidates = [
        {
            "symbol": str(row["symbol"]),
            "name": str(row["name"] or row["symbol"]),
            "last_time": str(row["last_time"]),
        }
        for row in rows
        if str(row["symbol"]).split(".", 1)[0].startswith(("1", "5"))
    ]
    summary = {
        "rule": f"last daily bar older than {args.older_than_days} days",
        "cutoff": str(cutoff.date()),
        "threshold": str(threshold.date()),
        "candidates": len(candidates),
        "applied": bool(args.apply),
    }
    if not args.apply:
        print(json.dumps({**summary, "sample": candidates[:20]}, ensure_ascii=False, indent=2))
        return
    if not candidates:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    symbols = {item["symbol"] for item in candidates}
    for item in candidates:
        current = store.instrument(item["symbol"]) or {}
        store.put_instrument(
            item["symbol"], item["name"],
            asset_type="etf",
            delisting_date=str(pd.Timestamp(item["last_time"]).date()),
            permanent_exclusion="stale-fund-cleanup-v1",
            exclusion_reason=summary["rule"],
        )
    result = repository.purge_instruments(sorted(symbols))
    basic_removed = 0
    package_root = Path(repository.package_dir)
    for name in ("etf_basic_data.parquet", "stock_basic_data.parquet"):
        basic_removed += _filter_basic_metadata(package_root / name, symbols)
    store.state("catalog:excluded-instruments:v1", {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rule": summary["rule"],
        "cutoff": summary["cutoff"],
        "symbols": candidates,
    })
    report = {
        **summary,
        **result,
        "basic_metadata_removed": basic_removed,
    }
    output_dir = store.root.parent / ".local"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "stale-fund-cleanup-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
