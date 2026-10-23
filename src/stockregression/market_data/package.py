from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..models import normalize_symbol
from .models import MarketDataError
from .package_layout import resolve_package_root
from .service import MarketDataService
from .timeframes import normalize_timeframe


def scan_etf_package(folder: str | Path) -> dict:
    """Summarize a schema-2 package from its published catalog heads."""
    root = resolve_package_root(folder)
    manifest_path = root / ".stockregression" / "manifest.json"
    if not manifest_path.is_file():
        raise MarketDataError("目录缺少 schema 2 分片清单；请使用 data migrate-package 迁移")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MarketDataError(f"数据包清单不可读取：{exc}") from exc
    if int(manifest.get("schema_version") or 0) != 2:
        raise MarketDataError(
            "运行时只接受 schema 2 分片包；旧布局请先执行 data migrate-package"
        )
    if not manifest.get("package_id") or not manifest.get("generation"):
        raise MarketDataError("数据包清单缺少 package_id 或 generation")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise MarketDataError("schema 2 清单缺少 datasets 数据集引用")
    result = {
        "folder": str(root), "format": "partitioned-package", "files": {},
        "timeframes": {}, "errors": [], "bytes": 0, "rows": 0,
        "package_id": manifest["package_id"],
        "generation": manifest["generation"],
        "storage_layout": manifest.get("storage_layout", "time_partitioned"),
        "shard_count": 0, "shard_bytes": 0,
    }
    for asset in ("etf", "stock"):
        basic = root / f"{asset}_basic_data.parquet"
        result["files"][f"{asset}_basic"] = str(basic) if basic.is_file() else None
        if basic.is_file():
            try:
                metadata = pq.ParquetFile(basic).metadata
                result["files"][f"{asset}_basic_rows"] = int(metadata.num_rows or 0)
            except Exception as exc:
                result["errors"].append(f"{asset} 基本资料不可读：{exc}")

    grouped: dict[str, dict] = {}
    symbols_by_timeframe: dict[str, set[str]] = defaultdict(set)
    for dataset_key, head in datasets.items():
        try:
            symbol, timeframe, adjustment = str(dataset_key).split("|", 2)
            timeframe = normalize_timeframe(timeframe)
        except (ValueError, TypeError) as exc:
            result["errors"].append(f"数据集键无效：{dataset_key}")
            continue
        # Coarse timeframes are cached derivations. Only 1day and 1min count as
        # the package's authoritative source data.
        if timeframe not in {"1day", "1min"}:
            continue
        period = grouped.setdefault(timeframe, {
            "path": [str(root / "partitions" / timeframe)],
            "timeframe": timeframe, "file_count": 0, "bytes": 0,
            "rows": 0, "symbols": 0, "label_convention": "规范时间戳",
            "first_time": None, "last_time": None,
            "schema_errors": [], "schema_error_count": 0,
            "datasets": 0,
        })
        symbols_by_timeframe[timeframe].add(symbol)
        period["datasets"] += 1
        period["file_count"] += int(head.get("shard_count") or 0)
        period["bytes"] += int(head.get("shard_bytes") or 0)
        period["rows"] += int(head.get("row_count") or 0)
        period["first_time"] = min(
            filter(None, [period["first_time"], head.get("first_time")]), default=None
        )
        period["last_time"] = max(
            filter(None, [period["last_time"], head.get("last_time")]), default=None
        )
        result["rows"] += int(head.get("row_count") or 0)
        result["shard_count"] += int(head.get("shard_count") or 0)
        result["shard_bytes"] += int(head.get("shard_bytes") or 0)
    for timeframe, period in grouped.items():
        period["symbols"] = len(symbols_by_timeframe[timeframe])
        result["timeframes"][timeframe] = period
    from .package_layout import asset_type_for_symbol
    required_assets = {
        asset_type_for_symbol(symbol)
        for symbols in symbols_by_timeframe.values() for symbol in symbols
    }
    for asset in sorted(required_assets):
        if result["files"].get(f"{asset}_basic") is None:
            result["errors"].append(f"缺少 {asset}_basic_data.parquet")
    result["bytes"] = result["shard_bytes"]
    if not result["timeframes"]:
        result["errors"].append("schema 2 清单没有已发布的 1day 或 1min 基础行情")
    return result

def _basic_metadata(path: Path, asset_type: str = "etf") -> dict[str, dict]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if "ts_code" not in frame:
        return {}
    result = {}
    for row in frame.to_dict("records"):
        try:
            symbol = normalize_symbol(row.get("ts_code"))
        except (ValueError, TypeError):
            continue
        name = row.get("cname") or row.get("csname") or row.get("extname") or symbol
        metadata = {
            "name": str(name), "asset_type": asset_type, "listing_date": row.get("list_date"),
            "exchange": row.get("exchange"), "fund_manager": row.get("mgr_name"),
            "custodian": row.get("custod_name"), "management_fee": row.get("mgt_fee"),
            "etf_type": row.get("etf_type"), "source": f"{asset_type}-package",
        }
        result[symbol] = {key: value for key, value in metadata.items() if value is not None and not (isinstance(value, float) and np.isnan(value))}
    return result


def import_etf_package(
    service: MarketDataService,
    folder: str | Path,
    *,
    timeframes: list[str] | None = None,
    symbols: list[str] | None = None,
    label_convention: str = "auto",
    resume: bool = True,
    workers: int = 4,
    skip_invalid_rows: bool = False,
    progress=lambda *_: None,
    cancelled=lambda: False,
) -> dict:
    """Import/attach only a schema-2 package by rebuilding its manifest index."""
    return index_etf_package_direct(
        service, folder, timeframes=timeframes, symbols=symbols,
        progress=progress, cancelled=cancelled,
    )


def index_etf_package_direct(
    service: MarketDataService,
    folder: str | Path,
    *,
    timeframes: list[str] | tuple[str, ...] | None = None,
    symbols: list[str] | None = None,
    progress=lambda *_: None,
    cancelled=lambda: False,
) -> dict:
    """Rebuild the SQLite market index from a schema-2 package manifest."""
    root = resolve_package_root(folder)
    preview = scan_etf_package(root)
    if preview["errors"]:
        raise MarketDataError("数据包结构异常：" + "；".join(preview["errors"][:8]))
    wanted = sorted({normalize_symbol(value) for value in symbols}) if symbols else None
    selected = [normalize_timeframe(value) for value in timeframes] if timeframes else None
    if selected and any(value not in {"1day", "1min", "1week", "1month", "5min", "15min", "30min", "60min"} for value in selected):
        raise MarketDataError("只支持登记基础周期及其缓存派生周期")
    if cancelled():
        raise InterruptedError("数据包索引重建已取消")
    repository = service.repository
    try:
        previous_manifest = repository.package.manifest()
        same_generation = (
            previous_manifest.get("package_id") == preview["package_id"]
            and previous_manifest.get("generation") == preview["generation"]
        )
    except (OSError, ValueError):
        previous_manifest = {}
        same_generation = False
    with repository._connect() as db:
        previous_versions = {
            f"{row['symbol']}|{row['timeframe']}|{row['adjustment']}": int(row["current_version"])
            for row in db.execute("SELECT symbol,timeframe,adjustment,current_version FROM parquet_datasets")
        }
    repository.attach_package(root)
    basics = _basic_metadata(root / "etf_basic_data.parquet", "etf")
    stock_basic = root / "stock_basic_data.parquet"
    if stock_basic.is_file():
        basics.update(_basic_metadata(stock_basic, "stock"))
    for index, (symbol, metadata) in enumerate(sorted(basics.items()), start=1):
        if wanted is None or symbol in wanted:
            service.store.put_instrument(
                symbol, metadata.get("name", symbol),
                **{key: value for key, value in metadata.items() if key not in {"symbol", "name"}},
            )
        if index % 250 == 0:
            progress(0.05 + 0.15 * index / max(1, len(basics)), f"读取基本资料 {index}/{len(basics)}")
    result = repository.rebuild_index(symbols=wanted, timeframes=selected)
    heads = json.loads((root / ".stockregression" / "manifest.json").read_text(encoding="utf-8")).get("datasets", {})
    selected_heads = {}
    for key, head in heads.items():
        symbol, timeframe, _adjustment = key.split("|", 2)
        if wanted is not None and symbol not in wanted:
            continue
        if selected is not None and normalize_timeframe(timeframe) not in selected:
            continue
        selected_heads[key] = int(head.get("version") or 0)
    result["changed"] = sum(
        not same_generation or previous_versions.get(key) != version
        for key, version in selected_heads.items()
    )
    result["unchanged"] = len(selected_heads) - result["changed"]
    result["errors"] = []
    result.update({
        "folder": str(root), "format": "partitioned-package",
        "package_id": preview["package_id"], "generation": preview["generation"],
        "timeframes": selected or ["1day", "1min", "5min", "15min", "30min", "60min", "1week", "1month"],
        "rows": preview["rows"], "shard_count": preview["shard_count"],
        "shard_bytes": preview["shard_bytes"],
    })
    progress(1.0, f"已从清单重建 {result['datasets']} 个行情数据集")
    return result
