from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..models import normalize_symbol
from ..quality import minute_slots
from .models import MarketDataError
from .reader import (
    check_declared_adjustment, check_declared_identity, check_declared_timeframe,
    file_fingerprint, normalize_frame, read_file,
)
from .repository import PARSER_VERSION
from .service import MarketDataService, _params_hash
from .timeframes import normalize_timeframe, timeframe_spec


PACKAGE_TIMEFRAMES = ("1min", "5min", "15min", "30min", "60min")
PACKAGE_SOURCE = "etf-package"


def _reader():
    return duckdb.connect(":memory:")


def _source_path(folder: Path, timeframe: str) -> Path:
    return folder / ("etf_daily.parquet" if timeframe == "1day" else f"etf_{timeframe}")


def _inspect_minute_folder(folder: Path, timeframe: str) -> dict:
    files = sorted(folder.glob("*.parquet"))
    size = sum(path.stat().st_size for path in files)
    symbols = []
    rows = 0
    schema_errors = []
    def inspect(path):
        try:
            symbol = normalize_symbol(path.stem)
        except (ValueError, IndexError):
            return None, 0, f"{path.name}: 文件名没有唯一证券代码"
        try:
            metadata = pq.ParquetFile(path).metadata
            names = set(metadata.schema.names)
            missing = {"ts_code", "trade_time", "open", "high", "low", "close", "vol"} - names
            if missing:
                return symbol, int(metadata.num_rows or 0), f"{path.name}: 缺少 {', '.join(sorted(missing))}"
            return symbol, int(metadata.num_rows or 0), None
        except Exception as exc:
            return symbol, 0, f"{path.name}: {exc}"
    with ThreadPoolExecutor(max_workers=8) as executor:
        for symbol, count, error in executor.map(inspect, files):
            if symbol:
                symbols.append(symbol)
            rows += count
            if error:
                schema_errors.append(error)
    convention, first_time, last_time = _detect_label(folder, timeframe)
    return {
        "path": str(folder), "timeframe": timeframe, "file_count": len(files),
        "bytes": size, "rows": rows, "symbols": len(set(symbols)),
        "label_convention": convention, "first_time": first_time,
        "last_time": last_time, "schema_errors": schema_errors[:50],
        "schema_error_count": len(schema_errors),
    }


def _detect_label(folder: Path, timeframe: str) -> tuple[str, str | None, str | None]:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return "auto", None, None
    sample = max(files, key=lambda path: path.stat().st_size)
    try:
        with _reader() as db:
            first_time, last_time, first_clock, second_clock, last_clock = db.execute(
                """SELECT CAST(min(trade_time) AS VARCHAR),CAST(max(trade_time) AS VARCHAR),
                          strftime(min(trade_time),'%H:%M'),
                          (SELECT strftime(trade_time,'%H:%M') FROM read_parquet(?)
                           WHERE cast(trade_date AS DATE)=(
                               SELECT min(cast(trade_date AS DATE)) FROM read_parquet(?)
                           ) ORDER BY trade_time LIMIT 1 OFFSET 1),
                          strftime(max(trade_time),'%H:%M')
                   FROM read_parquet(?) """,
                [str(sample), str(sample), str(sample)],
            ).fetchone()
            first_day_count = db.execute(
                """SELECT count(*) FROM read_parquet(?)
                   WHERE cast(trade_date AS DATE)=(
                       SELECT min(cast(trade_date AS DATE)) FROM read_parquet(?)
                   )""",
                [str(sample), str(sample)],
            ).fetchone()[0]
    except Exception:
        return "auto", None, None
    minutes = timeframe_spec(timeframe).minutes or 1
    sample_day = pd.Timestamp(first_time).normalize()
    expected_slots = minute_slots(sample_day)
    canonical_end_count = len(expected_slots[(minutes - 1)::minutes])
    expected_start, expected_end = "09:30", "15:00"
    end_clock = (
        f"{9 + minutes // 60:02d}:{(30 + minutes % 60):02d}"
        if minutes < 60 else "10:30"
    )
    if first_clock == expected_start and last_clock == expected_end:
        # The package includes an explicit 09:30 opening-auction row followed
        # by end-labelled regular bars.  A start-labelled file ends at 14:59
        # for one-minute data rather than at the session close.
        if int(first_day_count) == canonical_end_count + 1:
            return "end", first_time, last_time
        return "start", first_time, last_time
    if first_clock == end_clock and last_clock == expected_end:
        return "end", first_time, last_time
    return "auto", first_time, last_time


def scan_etf_package(folder: str | Path) -> dict:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise MarketDataError("数据包目录不存在：" + str(root))
    assets = [
        asset for asset in ("etf", "stock")
        if (root / f"{asset}_basic_data.parquet").is_file()
        or (root / f"{asset}_daily.parquet").is_file()
        or any((root / f"{asset}_{timeframe}").is_dir() for timeframe in PACKAGE_TIMEFRAMES)
    ]
    if not assets:
        raise MarketDataError("数据包中没有 etf_* 或 stock_* 文件")
    result = {
        "folder": str(root), "format": "package", "files": {},
        "timeframes": {}, "errors": [], "bytes": 0, "rows": 0,
    }
    for asset in assets:
        basic = root / f"{asset}_basic_data.parquet"
        daily = root / f"{asset}_daily.parquet"
        result["files"][f"{asset}_basic"] = str(basic) if basic.exists() else None
        result["files"][f"{asset}_daily"] = str(daily) if daily.exists() else None
        if basic.exists():
            try:
                metadata = pq.ParquetFile(basic).metadata
                result["files"][f"{asset}_basic_rows"] = int(metadata.num_rows)
                result["bytes"] += basic.stat().st_size
            except Exception as exc:
                result["errors"].append(f"{asset} 基本资料不可读：{exc}")
        else:
            result["errors"].append(f"缺少 {basic.name}")
        if daily.exists():
            try:
                with _reader() as db:
                    rows, symbols, first_date, last_date = db.execute(
                        """SELECT count(*),count(DISTINCT ts_code),min(trade_date),max(trade_date)
                           FROM read_parquet(?)""",
                        [str(daily)],
                    ).fetchone()
                current = result["timeframes"].setdefault("1day", {
                    "path": [], "timeframe": "1day", "file_count": 0, "bytes": 0,
                    "rows": 0, "symbols": 0, "label_convention": "date",
                    "first_time": None, "last_time": None,
                    "schema_errors": [], "schema_error_count": 0,
                })
                current["path"].append(str(daily))
                current["file_count"] += 1
                current["bytes"] += daily.stat().st_size
                current["rows"] += int(rows)
                current["symbols"] += int(symbols)
                current["first_time"] = min(filter(None, [current["first_time"], str(first_date)]), default=None)
                current["last_time"] = max(filter(None, [current["last_time"], str(last_date)]), default=None)
                result["bytes"] += daily.stat().st_size
                result["rows"] += int(rows)
            except Exception as exc:
                result["errors"].append(f"{asset} 日线不可读：{exc}")
        for timeframe in PACKAGE_TIMEFRAMES:
            folder_path = root / f"{asset}_{timeframe}"
            if not folder_path.is_dir():
                if asset == "etf":
                    result["errors"].append(f"缺少 {folder_path.name} 目录")
                continue
            info = _inspect_minute_folder(folder_path, timeframe)
            current = result["timeframes"].setdefault(timeframe, {
                "path": [], "timeframe": timeframe, "file_count": 0, "bytes": 0,
                "rows": 0, "symbols": 0, "label_convention": info["label_convention"],
                "first_time": None, "last_time": None,
                "schema_errors": [], "schema_error_count": 0,
            })
            current["path"].append(info.get("path", str(folder_path)))
            current["file_count"] += info["file_count"]
            current["bytes"] += info["bytes"]
            current["rows"] += info["rows"]
            current["symbols"] += info["symbols"]
            current["schema_error_count"] += info["schema_error_count"]
            current["first_time"] = min(filter(None, [current["first_time"], info.get("first_time")]), default=None)
            current["last_time"] = max(filter(None, [current["last_time"], info.get("last_time")]), default=None)
            result["bytes"] += info["bytes"]
            result["rows"] += info["rows"]
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


def _daily_from_source(row: dict) -> dict:
    return {
        "timestamp": pd.to_datetime(row.get("trade_date"), errors="coerce"),
        "open": row.get("open"), "high": row.get("high"), "low": row.get("low"),
        # Tushare-style ETF daily volume is expressed in 100-share lots and
        # turnover in thousand yuan. The canonical store uses shares and yuan.
        "close": row.get("close"), "volume": float(row.get("vol") or 0) * 100.0,
        "turnover": float(row.get("amount") or 0) * 1000.0, "preclose": row.get("pre_close"),
        "tradestatus": 1 if pd.notna(row.get("close")) and float(row.get("close") or 0) > 0 else 0,
        "is_st": False, "adj_factor": row.get("adj_factor"),
        "source": PACKAGE_SOURCE, "source_verified": True,
        "is_synthetic": False, "is_auction_placeholder": False,
    }


def _daily_batches(path: Path, wanted: set[str]):
    with _reader() as db:
        db.execute("CREATE TEMP TABLE wanted(ts_code VARCHAR PRIMARY KEY)")
        db.executemany("INSERT INTO wanted VALUES(?)", [(symbol,) for symbol in sorted(wanted)])
        relation = db.execute(
            """SELECT d.ts_code,d.trade_date,d.open,d.high,d.low,d.close,d.pre_close,
                      d.vol,d.amount,d.adj_factor
               FROM read_parquet(?) d JOIN wanted w USING(ts_code)
               ORDER BY d.ts_code,d.trade_date""",
            [str(path)],
        )
        reader = relation.to_arrow_reader(65536)
        current_symbol = None
        parts = []
        for batch in reader:
            frame = batch.to_pandas()
            for symbol, part in frame.groupby("ts_code", sort=False):
                symbol = normalize_symbol(symbol)
                if current_symbol is not None and symbol != current_symbol:
                    yield current_symbol, pd.concat(parts, ignore_index=True)
                    parts = []
                current_symbol = symbol
                parts.append(part)
        if current_symbol is not None and parts:
            yield current_symbol, pd.concat(parts, ignore_index=True)


def _import_package_minute(
    service: MarketDataService, path: Path, timeframe: str, convention: str,
    *, skip_invalid_rows: bool, cancelled,
) -> dict:
    symbol = normalize_symbol(path.stem)
    if cancelled():
        raise InterruptedError("任务已取消")
    frame, fingerprint, encoding, batches = read_file(path)
    params_hash = _params_hash(timeframe, convention, "none", PACKAGE_SOURCE, PARSER_VERSION)
    old = service.repository.fingerprint_seen(
        symbol, timeframe, "none", fingerprint, params_hash,
    )
    if old and old.get("status") == "completed":
        return {
            "symbol": symbol, "timeframe": timeframe, "status": "skipped",
            "version": old.get("version"), "rows_read": old.get("rows_read", 0),
            "fingerprint": fingerprint,
        }
    check_declared_identity(frame, symbol)
    check_declared_timeframe(frame, timeframe)
    check_declared_adjustment(frame, "none")
    normalized, pending = normalize_frame(
        frame, timeframe, label_convention=convention,
        source=PACKAGE_SOURCE, skip_invalid_rows=skip_invalid_rows,
    )
    normalized, unit_pending = service._verify_volume_units(symbol, timeframe, normalized)
    if unit_pending:
        pending.append(unit_pending)
    report = {
        "status": "complete" if not pending else "partial",
        "expected_rows": len(normalized), "observed_rows": len(normalized),
        "missing_rows": 0, "synthetic_rows": int(normalized.is_synthetic.sum()),
        "gap_count": 0, "gaps": [], "unfillable": 0,
        "auction_placeholders": int(normalized.is_auction_placeholder.sum()),
        "pending_validation": pending,
        "reason": "; ".join(pending) or None,
    }
    metadata = {"source_file": path.name, "batches": batches, "encoding": encoding}
    if normalized.attrs.get("unit_verification"):
        metadata["unit_verification"] = normalized.attrs["unit_verification"]
    result = service.repository.commit_bars(
        symbol, timeframe, normalized, source=PACKAGE_SOURCE, adjustment="none",
        file_name=path.name, fingerprint=fingerprint, params_hash=params_hash,
        report=report, metadata=metadata,
    )
    service.repository.save_coverage(
        symbol, timeframe, "none", normalized.timestamp.min(), normalized.timestamp.max(), report,
    )
    service.revalidate(symbol, timeframe)
    service.refresh_snapshot(symbol)
    return {**result, "fingerprint": fingerprint}


def import_etf_package(
    service: MarketDataService,
    folder: str | Path,
    *,
    timeframes: list[str] | tuple[str, ...] | None = None,
    symbols: list[str] | None = None,
    label_convention: str = "auto",
    resume: bool = True,
    workers: int = 2,
    skip_invalid_rows: bool = False,
    progress=lambda *_: None,
    cancelled=lambda: False,
) -> dict:
    root = Path(folder).expanduser().resolve()
    preview = scan_etf_package(root)
    timeframes = [
        normalize_timeframe(value)
        for value in (timeframes or ("1day", "1min"))
    ]
    wanted = {normalize_symbol(value) for value in symbols} if symbols else None
    basics = _basic_metadata(root / "etf_basic_data.parquet", "etf")
    stock_basic = root / "stock_basic_data.parquet"
    if stock_basic.is_file():
        basics.update(_basic_metadata(stock_basic, "stock"))
    for symbol, metadata in basics.items():
        if wanted is None or symbol in wanted:
            service.store.put_instrument(symbol, metadata.get("name", symbol), **{key: value for key, value in metadata.items() if key not in {"symbol", "name"}})
    completed, skipped, failures = [], [], []
    if wanted is None:
        minute_symbols = set()
        for timeframe in timeframes:
            if timeframe == "1min":
                path = _source_path(root, timeframe)
                if path.is_dir():
                    minute_symbols.update(
                        normalize_symbol(item.stem) for item in path.glob("*.parquet")
                        if item.stem[:1].isdigit()
                    )
        wanted = set(basics) | minute_symbols
    total_map = preview.get("timeframes", {})
    units = sum(
        int(total_map.get(timeframe, {}).get("symbols") or 0)
        for timeframe in timeframes
    ) or len(wanted or [])
    completed_units = 0
    if cancelled():
        raise InterruptedError("任务已取消")
    if "1day" in timeframes:
        progress(0.02, "导入集中存放的 ETF 日线")
        daily_path = _source_path(root, "1day")
        if not daily_path.exists():
            failures.append({"symbol": "*", "timeframe": "1day", "reason": "缺少 etf_daily.parquet"})
        else:
            fingerprint, _ = file_fingerprint(daily_path)
            params_hash = _params_hash("1day", "end", "none", PACKAGE_SOURCE, PARSER_VERSION)
            for symbol, part in _daily_batches(daily_path, wanted):
                if cancelled():
                    raise InterruptedError("任务已取消")
                if resume and service.repository.fingerprint_seen(
                    symbol, "1day", "none", fingerprint, params_hash,
                ):
                    skipped.append({"symbol": symbol, "timeframe": "1day", "reason": "相同数据包内容已导入"})
                    completed_units += 1
                    continue
                try:
                    frame = pd.DataFrame([
                        _daily_from_source(row) for row in part.to_dict("records")
                    ])
                    frame = frame.dropna(subset=["timestamp"])
                    calendar_frame = pd.DataFrame({"date": pd.to_datetime(frame.timestamp).dt.normalize().unique()})
                    pending = []
                    if label_convention != "auto" and "start" in label_convention:
                        pending.append("日线按交易日期读取")
                    metadata = {
                        "name": basics.get(symbol, {}).get("name", symbol),
                        "etf_category": basics.get(symbol, {}).get("etf_type"),
                        "source_file": daily_path.name,
                        "tradestatus_derived": True,
                    }
                    result = service._commit_frame(
                        symbol, "1day", frame, fingerprint=fingerprint,
                        file_name=daily_path.name, params_hash=params_hash,
                        pending=pending, source=PACKAGE_SOURCE, adjustment="none",
                        etf_category=None, calendar_path=None, actions_path=None,
                        progress=lambda *_: None, cancelled=cancelled,
                        calendar_frame=calendar_frame,
                        source_metadata=metadata,
                    )
                    service.repository.save_auxiliary(
                        symbol, "calendar", calendar_frame.to_dict("records"),
                        version=f"pkg-{fingerprint[:16]}", verified=True,
                    )
                    service.refresh_snapshot(symbol)
                    completed.append(result)
                except Exception as exc:
                    failures.append({"symbol": symbol, "timeframe": "1day", "reason": str(exc)})
                completed_units += 1
                progress(0.02 + 0.15 * min(1, completed_units / max(1, units)), f"日线 {symbol}")
    minute_total = sum(
        len(list(_source_path(root, timeframe).glob("*.parquet")))
        for timeframe in timeframes if timeframe in PACKAGE_TIMEFRAMES
        and _source_path(root, timeframe).is_dir()
    )
    minute_done = 0
    for timeframe in timeframes:
        if timeframe not in PACKAGE_TIMEFRAMES:
            continue
        folder_path = _source_path(root, timeframe)
        if not folder_path.is_dir():
            failures.append({"symbol": "*", "timeframe": timeframe, "reason": f"缺少 {folder_path.name} 目录"})
            continue
        detected = _detect_label(folder_path, timeframe)[0]
        convention = label_convention if label_convention != "auto" else detected
        files = sorted(folder_path.glob("*.parquet"))
        if wanted:
            files = [
                path for path in files
                if path.stem.split(".")[0].isdigit() and normalize_symbol(path.stem) in wanted
            ]
        def one(path: Path):
            return _import_package_minute(
                service, path, timeframe, convention,
                skip_invalid_rows=skip_invalid_rows, cancelled=cancelled,
            )
        with ThreadPoolExecutor(max_workers=max(1, min(4, int(workers or 1)))) as executor:
            futures = {executor.submit(one, path): path for path in files}
            for future in as_completed(futures):
                if cancelled():
                    for pending in futures:
                        pending.cancel()
                    raise InterruptedError("任务已取消")
                path = futures[future]
                minute_done += 1
                try:
                    result = future.result()
                    if result.get("status") == "skipped":
                        skipped.append({"symbol": path.stem, "timeframe": timeframe, "reason": "相同内容已导入"})
                    else:
                        completed.append(result)
                except Exception as exc:
                    failures.append({
                        "symbol": path.stem, "timeframe": timeframe, "reason": str(exc),
                    })
                progress(
                    0.17 + 0.83 * minute_done / max(1, minute_total),
                    f"{timeframe} {minute_done}/{minute_total} · {path.stem}",
                )
    status = "completed" if not failures else "partial" if completed else "failed"
    return {
        "status": status, "format": "etf", "folder": str(root),
        "timeframes": timeframes, "completed": len(completed), "skipped": len(skipped),
        "failed": len(failures), "symbols": completed[:1000], "skipped_items": skipped,
        "failures": failures, "preview": preview,
        "message": f"ETF 数据包导入完成 {len(completed)} 项，跳过 {len(skipped)} 项，失败 {len(failures)} 项",
    }


def _parquet_source_stats(path: Path, timestamp_column: str) -> tuple[int, str | None, str | None]:
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    rows = int(metadata.num_rows or 0)
    column_index = metadata.schema.names.index(timestamp_column)
    minimum = maximum = None
    for group_index in range(metadata.num_row_groups):
        statistics = metadata.row_group(group_index).column(column_index).statistics
        if statistics is None:
            minimum = maximum = None
            break
        minimum = statistics.min if minimum is None else min(minimum, statistics.min)
        maximum = statistics.max if maximum is None else max(maximum, statistics.max)
    # Registration is metadata-only.  A file without timestamp statistics is
    # marked as an unknown range and is inspected lazily when that symbol is
    # first used; never scan a multi-million-row minute file just to register it.
    return (
        rows,
        str(minimum) if minimum is not None else None,
        str(maximum) if maximum is not None else None,
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
    """Idempotently index package files without deleting unrelated versions."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise MarketDataError("数据包目录不存在：" + str(root))
    wanted = {normalize_symbol(value) for value in symbols} if symbols else None
    selected = set(timeframes or ("1day", "1min"))
    assets = [
        asset for asset in ("etf", "stock")
        if (root / f"{asset}_daily.parquet").is_file()
        or any((root / f"{asset}_{timeframe}").is_dir() for timeframe in PACKAGE_TIMEFRAMES)
    ]
    if not assets:
        raise MarketDataError("数据包中没有 etf_* 或 stock_* 行情文件")
    basics = _basic_metadata(root / "etf_basic_data.parquet", "etf")
    stock_basic = root / "stock_basic_data.parquet"
    if stock_basic.is_file():
        basics.update(_basic_metadata(stock_basic, "stock"))
    for symbol, metadata in basics.items():
        if wanted is None or symbol in wanted:
            service.store.put_instrument(
                symbol, metadata.get("name", symbol),
                **{key: value for key, value in metadata.items() if key not in {"symbol", "name"}},
            )
    repository = service.repository
    repository.attach_package(root)
    now = pd.Timestamp.now(tz="UTC").isoformat()
    rows_count = 0
    files_count = 0
    errors = []
    unchanged_count = 0
    changed_count = 0
    manifest_files = {}
    label_cache = {}
    writer_connection = None
    progress(0.02, "登记 ETF 基本资料")

    @contextmanager
    def batch_db():
        if writer_connection is None:
            raise RuntimeError("数据包登记写连接尚未建立")
        yield writer_connection

    def upsert(symbol, timeframe, path, count, first, last, signature, schema, file_metadata):
        nonlocal rows_count, files_count, unchanged_count, changed_count
        relative = repository.package.relative(path)
        with batch_db() as db:
            existing = repository._dataset_row(db, symbol, timeframe, "none")
            current_version = int(existing["current_version"]) if existing else 0
            current_files = repository._current_files(
                db, symbol, timeframe, "none", current_version
            )
        same = bool(
            len(current_files) == 1
            and repository.package.relative(current_files[0].get("path", "")) == relative
            and str(current_files[0].get("sha256")) == signature
        )
        existing_metadata = (
            json.loads(existing["metadata"]) if existing and existing["metadata"] else {}
        )
        item_metadata = {
            **basics.get(symbol, {}),
            **existing_metadata,
            "source_package": str(root),
            "source_file": path.name,
            "package_id": repository.package_id,
        }
        if timeframe in PACKAGE_TIMEFRAMES:
            item_metadata["label_convention"] = file_metadata.get("label_convention", "auto")
        if same:
            with batch_db() as db:
                db.execute(
                    """UPDATE parquet_files SET path=?,sha256=?,size=?
                       WHERE symbol=? AND timeframe=? AND adjustment=? AND version=?""",
                    (
                        relative, signature, path.stat().st_size,
                        symbol, timeframe, "none", current_version,
                    ),
                )
            unchanged_count += 1
            rows_count += int(count)
            files_count += 1
            manifest_files[relative] = {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": signature,
                "version": current_version,
                "row_count": int(count),
                "schema_version": schema,
                **file_metadata,
                "updated_at": now,
            }
            return
        changed_count += 1
        version = current_version + 1 if existing else 1
        first_value = str(first) if first is not None else None
        last_value = str(last) if last is not None else None
        changed_validation_days = repository.changed_source_validation_days(
            symbol, timeframe,
            {
                "path": relative, "schema_version": schema,
                "first_time": first_value, "last_time": last_value,
            },
        )
        preserved_old_path = None
        source_files = [
            row for row in current_files if repository._is_package_source(row)
        ]
        if source_files:
            old_file = source_files[0]
            old_path = repository._absolute_source_path(old_file["path"])
            same_source_path = old_path.resolve() == path.resolve()
            with batch_db() as db:
                archive = db.execute(
                    "SELECT * FROM parquet_source_archives WHERE source_path=?",
                    (str(old_path.resolve()),),
                ).fetchone()
            if same_source_path:
                if not (
                    archive and int(archive["size"] or 0) == int(old_file["size"] or 0)
                    and repository.package.absolute(archive["archive_path"]).is_file()
                ):
                    raise MarketDataError(
                        "原文件已被覆盖且尚未归档，无法保留旧版本；请先取消覆盖并执行 data archive"
                    )
                preserved_old_path = archive["archive_path"]
            else:
                if archive and repository.package.absolute(archive["archive_path"]).is_file():
                    preserved_old_path = archive["archive_path"]
                elif old_path.is_file():
                    backup_folder = repository.package.history_dir / "files"
                    backup_folder.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256(
                        f"{old_path}|{old_path.stat().st_size}|{old_path.stat().st_mtime_ns}".encode()
                    ).hexdigest()[:20]
                    backup = backup_folder / f"{digest}.parquet"
                    if not backup.exists():
                        temporary = backup.with_name(backup.name + ".writing")
                        shutil.copy2(old_path, temporary)
                        os.replace(temporary, backup)
                    preserved_old_path = repository.package.relative(backup)
        status = "pending_validation"
        reason = "文件已登记，尚未完成覆盖和质量核验"
        source_name = f"{schema.split('-', 1)[0]}-package"
        with batch_db() as db:
            if preserved_old_path:
                preserved = repository.package.absolute(preserved_old_path)
                db.execute(
                    "UPDATE parquet_files SET path=?,sha256=?,size=? "
                    "WHERE symbol=? AND timeframe=? AND adjustment='none' AND version=?",
                    (
                        preserved_old_path,
                        f"stat:{preserved.stat().st_size}:{preserved.stat().st_mtime_ns}",
                        preserved.stat().st_size, symbol, timeframe, current_version,
                    ),
                )
            for start in range(0, len(changed_validation_days), 400):
                days = changed_validation_days[start:start + 400]
                placeholders = ",".join("?" for _ in days)
                db.execute(
                    "DELETE FROM parquet_day_validation WHERE symbol=? AND timeframe=? "
                    "AND adjustment='none' "
                    "AND trade_date IN (" + placeholders + ")",
                    (symbol, timeframe, *days),
                )
                if timeframe == "1day":
                    db.execute(
                        "DELETE FROM parquet_day_validation WHERE symbol=? AND adjustment='none' "
                        "AND timeframe IN ('1min','5min','15min','30min','60min') "
                        "AND trade_date IN (" + placeholders + ")",
                        (symbol, *days),
                    )
            db.execute(
                "UPDATE parquet_derived SET stale=1 WHERE symbol=? AND adjustment='none' "
                "AND source_timeframe=?",
                (symbol, timeframe),
            )
            db.execute(
                """INSERT INTO parquet_datasets(
                   symbol,timeframe,adjustment,current_version,first_time,last_time,
                   completed_through,row_count,raw_rows,synthetic_rows,pending_rows,
                   status,reason,metadata,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol,timeframe,adjustment) DO UPDATE SET
                   current_version=excluded.current_version,first_time=excluded.first_time,
                   last_time=excluded.last_time,completed_through=excluded.completed_through,
                   row_count=excluded.row_count,raw_rows=excluded.raw_rows,
                   status=excluded.status,reason=excluded.reason,metadata=excluded.metadata,
                   updated_at=excluded.updated_at""",
                (
                    symbol, timeframe, "none", version, first_value, last_value, last_value,
                    int(count), int(count), 0, 0, status, reason,
                    json.dumps(item_metadata, ensure_ascii=False), now, now,
                ),
            )
            db.execute(
                """INSERT INTO parquet_versions(
                   symbol,timeframe,adjustment,version,created_at,row_count,raw_rows,
                   synthetic_rows,pending_rows,first_time,last_time,status,source,
                   file_count,bytes,report
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, timeframe, "none", version, now, int(count), int(count),
                    0, 0, first_value, last_value, status, source_name, 1,
                    path.stat().st_size,
                    json.dumps({"storage": "source_package", "package_id": repository.package_id}),
                ),
            )
            db.execute(
                """INSERT OR REPLACE INTO parquet_files(
                   symbol,timeframe,adjustment,version,partition_key,path,sha256,
                   row_count,first_time,last_time,size,schema_version
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, timeframe, "none", version, f"source={schema}",
                    relative, signature, int(count), first_value, last_value,
                    path.stat().st_size, schema,
                ),
            )
        rows_count += int(count)
        files_count += 1
        manifest_files[relative] = {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": signature,
            "version": version,
            "row_count": int(count),
            "schema_version": schema,
            **file_metadata,
            "updated_at": now,
        }

    with repository._connect() as db:
        writer_connection = db
        if "1day" in selected:
            daily_path = _source_path(root, "1day")
            if "etf" in assets and not daily_path.is_file():
                errors.append("缺少 etf_daily.parquet")
            elif "etf" in assets:
                with _reader() as db_source:
                    stats = db_source.execute(
                        """SELECT ts_code,count(*),min(trade_date),max(trade_date)
                           FROM read_parquet(?) GROUP BY ts_code ORDER BY ts_code""",
                        [str(daily_path)],
                    ).fetchall()
                daily_signature = (
                    f"stat:{daily_path.stat().st_size}:{daily_path.stat().st_mtime_ns}"
                )
                size = daily_path.stat().st_size
                total_daily_rows = max(1, sum(int(row[1]) for row in stats))
                for position, (symbol, count, first, last) in enumerate(stats, start=1):
                    try:
                        symbol = normalize_symbol(symbol)
                    except (TypeError, ValueError):
                        continue
                    if wanted is not None and symbol not in wanted:
                        continue
                    metadata = basics.get(symbol, {})
                    reason = "; ".join(
                        item for item in (
                            None if metadata.get("name") else "缺少证券基本资料",
                        ) if item
                    ) or None
                    upsert(
                        symbol, "1day", daily_path, count, first, last,
                        daily_signature, "etf-source-daily",
                        {"label_convention": "end", "source_verified": True},
                    )
                    if position % 250 == 0:
                        progress(0.05 + 0.15 * position / max(1, len(stats)), f"登记日线 {position}/{len(stats)}")
            stock_daily_path = root / "stock_daily.parquet"
            if "stock" in assets and stock_daily_path.is_file():
                with _reader() as db_source:
                    stock_stats = db_source.execute(
                        """SELECT ts_code,count(*),min(trade_date),max(trade_date)
                           FROM read_parquet(?) GROUP BY ts_code ORDER BY ts_code""",
                        [str(stock_daily_path)],
                    ).fetchall()
                stock_signature = (
                    f"stat:{stock_daily_path.stat().st_size}:{stock_daily_path.stat().st_mtime_ns}"
                )
                for position, (symbol, count, first, last) in enumerate(stock_stats, start=1):
                    try:
                        symbol = normalize_symbol(symbol)
                    except (TypeError, ValueError):
                        continue
                    if wanted is not None and symbol not in wanted:
                        continue
                    upsert(
                        symbol, "1day", stock_daily_path, count, first, last,
                        stock_signature, "stock-source-daily",
                        {"label_convention": "end", "source_verified": True},
                    )
                    if position % 250 == 0:
                        progress(0.05 + 0.15 * position / max(1, len(stock_stats)), f"登记股票日线 {position}/{len(stock_stats)}")
        minute_items = []
        for asset in assets:
            for timeframe in PACKAGE_TIMEFRAMES:
                if timeframe not in selected:
                    continue
                folder_path = root / f"{asset}_{timeframe}"
                if not folder_path.is_dir():
                    if asset == "etf" and "etf" in assets:
                        errors.append(f"缺少 {folder_path.name} 目录")
                    continue
                minute_items.extend(
                    (asset, timeframe, path) for path in sorted(folder_path.glob("*.parquet"))
                )

        def inspect(item):
            asset, timeframe, path = item
            try:
                count, first, last = _parquet_source_stats(path, "trade_time")
                return asset, timeframe, path, count, first, last, None
            except Exception as exc:
                return asset, timeframe, path, 0, None, None, str(exc)

        inspected = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            for asset, timeframe, path, count, first, last, error in executor.map(inspect, minute_items):
                inspected += 1
                if inspected % 250 == 0 or inspected == len(minute_items):
                    progress(0.2 + 0.75 * inspected / max(1, len(minute_items)), f"登记行情文件 {inspected}/{len(minute_items)}")
                if error:
                    errors.append(f"{path.name} {timeframe}: {error}")
                else:
                    try:
                        symbol = normalize_symbol(path.stem)
                    except (ValueError, IndexError):
                        errors.append(f"{path.name}: 文件名没有唯一证券代码")
                        continue
                    if wanted is not None and symbol not in wanted:
                        continue
                    metadata = basics.get(symbol, {})
                    reason = "; ".join(
                        item for item in (
                            None if metadata.get("name") else "缺少证券基本资料",
                        ) if item
                    ) or None
                    signature = f"stat:{path.stat().st_size}:{path.stat().st_mtime_ns}"
                    label_key = (asset, timeframe)
                    if label_key not in label_cache:
                        label_cache[label_key], _, _ = _detect_label(path.parent, timeframe)
                    label = label_cache[label_key]
                    upsert(
                        symbol, timeframe, path, count, first, last, signature,
                        f"{asset}-source-minute",
                        {"label_convention": label, "source_verified": label in {"start", "end"}},
                    )
                if cancelled():
                    raise InterruptedError("任务已取消")
    if manifest_files:
        manifest = repository.package.manifest()
        manifest["files"].update(manifest_files)
        manifest["index_only"] = False
        manifest["package_version"] = int(manifest.get("package_version", 0)) + 1
        repository.package.publish(manifest)
    physical_files = len(manifest_files)
    progress(0.95, f"已登记 {physical_files} 个原始文件")
    service.store.state("market-source-package", {
        "folder": str(root), "rows": rows_count, "files": physical_files,
        "updated_at": now,
    })
    return {
        "status": "completed" if not errors else "partial",
        "mode": "direct_parquet", "folder": str(root),
        "files": physical_files, "records": files_count, "rows": rows_count, "errors": errors[:200],
        "unchanged": unchanged_count, "changed": changed_count,
        "package_id": repository.package_id,
        "failure_count": len(errors),
        "message": f"已在原目录登记 {physical_files} 个行情文件，共 {rows_count} 行；未复制 K 线",
    }
