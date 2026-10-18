from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from .preview import PreviewManager

from .data import DataService
from .data_diagnostics import derive_source_spans, project_data_diagnostics
from .quality import QUALITY_RULES_VERSION, _closing_auction_placeholders, minute_slots
from .jobs import JobManager
from .models import BacktestConfig, OptimizationConfig, normalize_symbol
from .market_data import MarketDataService
from .market_data.aggregation import AggregationService
from .market_data.models import DataVersionNotFound, MarketDataError
from .market_data.cache import cache_settings, configure_cache
from .market_data.package import scan_etf_package
from .market_data.preparation import MarketPreparationService
from .market_data.timeframes import normalize_timeframe, public_timeframe
from .rules import instrument_terms
from .research_preparation import preparation_config
from .schedule import configure_schedule
from .settings import PROJECT_ROOT, load_settings, save_settings
from .source_health import configured_sources, validate_source_settings
from .storage import Store, dumps, json_safe, now_iso


class SyncRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    mode: str = "history"
    scope: Literal["market", "watchlist", "selected"] | None = None
    resume: bool = False
    retry_failed: bool = False
    workers: int = Field(default=2, ge=1, le=4)


class MarketSyncRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    frequency: str | None = None
    timeframes: list[str] = Field(default_factory=list)
    mode: Literal["history", "incremental", "daily", "repair"] = "incremental"
    scope: Literal["market", "watchlist", "selected"] | None = None
    start_date: str | None = None
    end_date: str | None = None
    resume: bool = False
    retry_failed: bool = False
    initialize: bool = False
    workers: int = Field(default=2, ge=1, le=4)


class MarketImportRequest(BaseModel):
    symbol: str
    timeframe: str | None = None
    frequency: str | None = None
    path: str | None = None
    label_convention: Literal["auto", "start", "end"] = "auto"
    adjustment: str = "none"
    etf_category: str | None = None
    calendar_path: str | None = None
    actions_path: str | None = None
    skip_invalid_rows: bool = False
    import_mode: Literal["merge", "snapshot"] = "merge"


class MarketBatchRequest(BaseModel):
    folder: str | None = None
    timeframe: str | None = None
    frequency: str | None = None
    recursive: bool = False
    label_convention: Literal["auto", "start", "end"] = "auto"
    adjustment: str = "none"
    etf_category: str | None = None
    resume: bool = True
    calendar_path: str | None = None
    workers: int = Field(default=2, ge=1, le=4)
    skip_invalid_rows: bool = False
    import_mode: Literal["merge", "snapshot"] = "merge"


class MarketPackageRequest(BaseModel):
    folder: str
    format: Literal["etf", "stock", "package"] = "package"
    timeframes: list[str] | None = None
    symbols: list[str] | None = None
    label_convention: Literal["auto", "start", "end"] = "auto"
    resume: bool = True
    workers: int = Field(default=2, ge=1, le=4)
    skip_invalid_rows: bool = False


class MarketValidateRequest(BaseModel):
    symbol: str
    timeframe: str | None = None
    frequency: str | None = None
    label_convention: Literal["auto", "start", "end"] | None = None
    mode: Literal["incremental", "full"] = "incremental"
    start_date: str | None = None
    end_date: str | None = None


class MarketArchiveRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    frequency: str | None = None


class MarketAggregateRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    source_timeframe: str
    target_timeframe: str
    start_date: str | None = None
    end_date: str | None = None


class MarketPreparationRequest(BaseModel):
    config: BacktestConfig
    version: str | None = None


class MarketSourceProbeRequest(BaseModel):
    source: Literal["sina", "tdx", "baostock", "eastmoney"]
    capability: Literal["recent_minute", "etf_daily"]
    market: Literal["all", "stock", "etf"] = "all"


class RestartRequest(BaseModel):
    force: bool = False


def _safe_upload_path(name: str) -> Path:
    raw = str(name or "upload.csv").replace("\\", "/")
    pure = PurePosixPath(raw)
    parts = [part for part in pure.parts if part not in {"", "."}]
    if pure.is_absolute() or not parts or any(part == ".." or ":" in part for part in parts):
        raise ValueError("上传文件名包含不安全路径")
    if parts[-1].lower().rsplit(".", 1)[-1] not in {"csv", "parquet"}:
        raise ValueError("仅支持上传 CSV 和 Parquet 文件")
    return Path(*parts)


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                output.write(chunk)
                size += len(chunk)
    finally:
        await upload.close()
    if size == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("上传文件为空")
    return size


class PreviewRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    revision: int = Field(ge=1)
    config: BacktestConfig


class BatchRequest(BaseModel):
    configs: list[BacktestConfig] = Field(min_length=1, max_length=100)


_V2_ARCHIVE_FIELDS = {"candidate_archive", "trial_archive", "candidate_archives", "trial_archives"}


def _compact_v2_result(value, top_candidates: int, parent_key: str | None = None):
    """Remove research-only matrices/archives while retaining UI summaries."""
    if isinstance(value, dict):
        compact = {}
        for key, item in value.items():
            if key in _V2_ARCHIVE_FIELDS:
                continue
            if key == "candidate_ids":
                continue
            if parent_key == "diagnostics" and key == "costs":
                # This is the full [candidate][fold] matrix. Top candidates
                # keep their compact cost_scores, and diagnostics retain the
                # candidate/evaluation counts needed to interpret the result.
                continue
            if key == "candidates" and isinstance(item, list):
                compact[key] = [_compact_v2_result(row, top_candidates) for row in item[:top_candidates]]
            else:
                compact[key] = _compact_v2_result(item, top_candidates, key)
        return compact
    if isinstance(value, list):
        return [_compact_v2_result(item, top_candidates, parent_key) for item in value]
    if isinstance(value, tuple):
        return [_compact_v2_result(item, top_candidates, parent_key) for item in value]
    return value


def _quality_dates(result: dict) -> tuple[str | None, str | None]:
    quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
    days = [str(row["date"])[:10] for row in quality.get("days", [])
            if isinstance(row, dict) and row.get("date")]
    if days:
        return min(days), max(days)
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    base = config.get("base") if isinstance(config.get("base"), dict) else config
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    return (diagnostics.get("checked_start") or base.get("start_date"),
            diagnostics.get("checked_end") or result.get("effective_end_date") or base.get("end_date"))


def _result_data_metadata(store: Store, result: dict) -> dict | None:
    """Read provenance for the result's locked snapshot, without market fetches.

    Only a small projection of immutable Parquet columns is read once per
    version/date range. This does not re-run minute validation or replace the
    original result's quality report with today's active snapshot.
    """
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    base = config.get("base") if isinstance(config.get("base"), dict) else config
    symbol, version = result.get("symbol") or base.get("symbol"), result.get("data_version")
    if not symbol or not version:
        return None
    manifest = store.snapshot(symbol, version)
    if not manifest:
        return None
    start, end = _quality_dates(result)
    key = f"data-source-spans:v2:q{QUALITY_RULES_VERSION}:{version}:{start or ''}:{end or ''}"
    cached = store.state(key)
    if not isinstance(cached, dict) or not isinstance(cached.get("spans"), list):
        frames = []
        if manifest.get("market_data"):
            market_versions = manifest.get("market_versions", {})
            minute_name = next(
                (name for name in ("1min", "5min", "15min", "30min", "60min")
                 if name in market_versions), None,
            )
            if minute_name:
                frames.append(MarketDataService(store).repository.read_frame(
                    symbol, minute_name,
                    version=int(market_versions[minute_name]["version"]),
                    start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1) if end else None,
                    fields=("timestamp", "source", "source_verified", "is_synthetic", "close"),
                ))
        else:
            import pyarrow.parquet as pq
            for relative in manifest.get("files", {}).get("minute", []):
                path = store.root / relative
                names = set(pq.read_schema(path).names)
                columns = [
                    name for name in ("timestamp", "source", "source_verified", "is_synthetic", "close")
                    if name in names
                ]
                if "timestamp" in columns:
                    frames.append(pd.read_parquet(path, columns=columns))
        minute = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        spans = derive_source_spans(minute, manifest.get("profiles"), start, end)
        auction_by_day = {}
        if not minute.empty and "close" in minute:
            stamps = pd.to_datetime(minute.timestamp)
            if start:
                minute = minute[stamps >= pd.Timestamp(start)]
            if end:
                minute = minute[pd.to_datetime(minute.timestamp) < pd.Timestamp(end) + pd.Timedelta(days=1)]
            if "source_verified" not in minute and "source" in minute:
                minute = minute.assign(source_verified=minute.source.map(
                    lambda source: bool(manifest.get("profiles", {}).get(source, {}).get("verified", False))))
            for day, group in minute.groupby(pd.to_datetime(minute.timestamp).dt.normalize()):
                if len(group) >= 240:
                    continue
                indexed = group.drop_duplicates("timestamp", keep="last").set_index("timestamp").reindex(minute_slots(day))
                count = int(_closing_auction_placeholders(manifest["symbol"], day, indexed, indexed.close.isna().to_numpy()).sum())
                if count:
                    auction_by_day[str(day.date())] = count
        cached = {"spans": spans, "auction_placeholders_by_day": auction_by_day}
        store.state(key, cached)
    metadata = dict(manifest.get("metadata") or {})
    metadata["diagnostics"] = {**metadata.get("diagnostics", {}), "source_spans": cached["spans"]}
    metadata["_snapshot_profiles"] = manifest.get("profiles", {})
    metadata["_auction_placeholders_by_day"] = cached.get("auction_placeholders_by_day", {})
    return metadata


def _project_result_quality(result: dict, data_metadata: dict | None) -> dict:
    quality = result.get("quality")
    if not isinstance(quality, dict):
        return result
    start, end = _quality_dates(result)
    metadata = dict(data_metadata or {})
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    base = config.get("base") if isinstance(config.get("base"), dict) else config
    if "requested_frequency" not in quality and base.get("frequency"):
        quality = {**quality, "requested_frequency": base["frequency"]}
    # Old results remain immutable. Their display projection can distinguish
    # known auction grid slots using the exact locked raw snapshot, without
    # recomputing any historical fills, strategy metrics or parameter choices.
    auction_by_day = metadata.get("_auction_placeholders_by_day", {})
    if auction_by_day and isinstance(quality.get("days"), list):
        projected_days = []
        for row in quality["days"]:
            row = dict(row)
            count = int(auction_by_day.get(str(row.get("date"))[:10], 0))
            if count and row.get("mode") == "minute" and "missing_data_minutes" not in row:
                row["auction_placeholders"] = min(count, int(row.get("missing_minutes", 0)))
                row["missing_data_minutes"] = max(0, int(row.get("missing_minutes", 0)) - row["auction_placeholders"])
                if not row["missing_data_minutes"] and str(row.get("reason", "")).startswith("少量缺口"):
                    row["reason"] = "收盘集合竞价无逐分钟成交，占位保留"
            projected_days.append(row)
        quality = {**quality, "days": projected_days,
                   "auction_placeholders": sum(int(row.get("auction_placeholders", 0)) for row in projected_days if row.get("mode") == "minute")}
    own_diagnostics = result.get("diagnostics")
    if isinstance(own_diagnostics, dict) and own_diagnostics.get("schema_version") == 1:
        metadata["diagnostics"] = {**metadata.get("diagnostics", {}), **own_diagnostics}
    metadata["warnings"] = list(dict.fromkeys([*metadata.get("warnings", []), *result.get("warnings", [])]))
    metadata["source_events"] = [*metadata.get("source_events", []), *result.get("source_events", [])]
    metadata["requested_end_date"] = result.get("requested_end_date") or base.get("end_date")
    # Clean the quality warning list separately, so execution/parameter notices
    # do not become quality failures while archived provider errors disappear.
    quality_projection = project_data_diagnostics({}, quality, start, end)
    metadata["source_events"].extend(quality_projection["source_events"])
    quality = {**quality, "warnings": quality_projection["current_warnings"]}
    projection = project_data_diagnostics(metadata, quality, start, end, metadata.get("_snapshot_profiles"))
    return {**result, "quality": quality, "warnings": projection["current_warnings"],
            "diagnostics": projection["diagnostics"], "source_events": projection["source_events"]}


def public_result(result: dict, full: bool = False, data_metadata: dict | None = None) -> dict:
    """Send UI-sized results by default, preserving raw data for full requests."""
    if full:
        return result
    result = dict(result)
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    is_v2 = result.get("protocol_version") == 2 or config.get("protocol_version") == 2
    if is_v2:
        try:
            top_candidates = max(1, min(100, int(config.get("top_candidates", 20))))
        except (TypeError, ValueError):
            top_candidates = 20
        result = _compact_v2_result(result, top_candidates)
    result = _project_result_quality(result, data_metadata)
    if "strategies" in result:
        result["equity_frequency"] = "1d"
        strategies = {}
        for name, raw_strategy in result["strategies"].items():
            strategy = dict(raw_strategy)
            daily = {}
            for row in strategy.get("equity", []):
                daily[str(row.get("timestamp", ""))[:10]] = row
            strategy["equity"] = list(daily.values())
            strategy["equity_frequency"] = "1d"
            strategies[name] = strategy
        result["strategies"] = strategies
    for mode, key in [("historical", "best_result"), ("robust", "test_result")]:
        if result.get(mode) and result[mode].get(key):
            result[mode] = {**result[mode], key: public_result(result[mode][key], data_metadata=data_metadata)}
    for mode in ("fixed", "adaptive"):
        if result.get(mode) and result[mode].get("test_result"):
            result[mode] = {**result[mode], "test_result": public_result(result[mode]["test_result"], data_metadata=data_metadata)}
    return result


def _with_data_availability(store: Store, result: dict) -> dict:
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    base = config.get("base") if isinstance(config.get("base"), dict) else config
    symbol = result.get("symbol") or base.get("symbol")
    version = result.get("data_version")
    if not symbol or not version:
        return result
    available = store.snapshot(symbol, version) is not None
    return {
        **result, "data_version_available": available,
        "data_version_warning": None if available else "原数据版本已清理，可查看历史结果，但不能按原版本重算",
    }


def create_app(data_dir: str | None = None, start_jobs: bool = True):
    configured = load_settings()
    root = str(Path(data_dir or configured.data_dir).resolve())
    configure_cache(configured.market_cache_mb)
    store = Store(root)
    initial_package = configured.package_dir if data_dir is None else None
    initial_market = MarketDataService(store, package_dir=initial_package)
    package_dir = str(initial_market.repository.package_dir)
    manager = JobManager(root, configured.max_workers)
    previews = PreviewManager(store)
    instance_id = uuid.uuid4().hex

    @asynccontextmanager
    async def lifespan(app):
        if start_jobs:
            manager.start()
        yield
        previews.stop()
        if start_jobs:
            manager.stop()

    app = FastAPI(title="A 股及 ETF 行情与网格研究台", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.package_dir = package_dir
    app.state.previews = previews
    app.state.instance_id = instance_id
    app.state.restart_callback = None
    app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

    def submit(kind, payload):
        return {"job_id": store.new_job(kind, payload)}

    def snapshot_version(config):
        # Preserve a submission-time immutable input when the active snapshot
        # already covers this request. Missing or stale coverage is prepared by
        # the worker, so the API never blocks on a source download.
        return DataService(store).current_version_if_ready(config)

    def require_ready(config):
        manifest = store.snapshot(config.symbol)
        if manifest and manifest.get("market_data"):
            readiness = MarketDataService(store).readiness(config)
            if not readiness["ready"]:
                raise HTTPException(409, detail={
                    "code": "DATA_NOT_READY", **readiness,
                    "config": config.model_dump(mode="json"),
                })
            return manifest["version"]
        version = DataService(store).current_version_if_ready(config)
        if version is None:
            raise HTTPException(409, detail={
                "code": "DATA_NOT_READY", "ready": False, "symbol": config.symbol,
                "frequency": config.frequency,
                "config": config.model_dump(mode="json"),
                "gaps": [{"kind": "local", "message": "没有满足本次区间和周期的本地数据；请先在数据管理中准备"}],
                "preparation": {"symbol": config.symbol, "frequency": config.frequency,
                                  "start": str(config.start_date), "end": str(config.end_date)},
            })
        return version

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "ok", "data_dir": root, "package_dir": package_dir,
            "version": "0.1.0", "instance_id": instance_id,
            "restart_supported": callable(getattr(app.state, "restart_callback", None)),
        }

    @app.post("/api/v1/system/restart", status_code=202)
    def restart_project(payload: RestartRequest = RestartRequest()):
        callback = getattr(app.state, "restart_callback", None)
        if not callable(callback):
            raise HTTPException(409, detail={
                "code": "RESTART_UNAVAILABLE",
                "message": "当前进程不是由项目启动脚本托管，无法自动重启。请关闭后重新运行 start.cmd。",
            })
        active_jobs = sum(
            job.get("status") in {"queued", "running"}
            for job in store.jobs(1000)
        )
        if active_jobs and not payload.force:
            raise HTTPException(409, detail={
                "code": "RESTART_JOBS_ACTIVE",
                "message": f"当前有 {active_jobs} 个排队或运行中的任务；确认重启后运行任务会被中断。",
                "active_jobs": active_jobs,
            })
        try:
            callback(active_jobs)
        except Exception as exc:
            raise HTTPException(503, detail={
                "code": "RESTART_FAILED", "message": "无法启动重启助手：" + str(exc),
            }) from exc
        return {
            "status": "restarting", "active_jobs": active_jobs,
            "previous_instance_id": instance_id,
            "message": "服务将重启，运行中的任务会中断",
        }

    @app.get("/api/v1/instruments")
    def instruments(q: str = "", asset_type: Literal["stock", "etf"] | None = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
        return store.instruments(q, limit, asset_type, offset)

    @app.get("/api/v1/workbench/recent-symbols")
    def recent_workbench_symbols():
        items = []
        for submitted in store.recent_submitted_symbols(10):
            symbol = submitted["symbol"]
            stored = store.instrument(symbol)
            if stored is None:
                terms = instrument_terms(symbol, {"name": symbol})
                item = {
                    "symbol": symbol,
                    "name": symbol,
                    "exchange": symbol.rsplit(".", 1)[-1],
                    "board": terms["board"],
                    **terms,
                }
            else:
                item = stored
            items.append({**item, "last_submitted_at": submitted["last_submitted_at"]})
        return items

    @app.post("/api/v1/instruments/refresh")
    def refresh_instruments():
        return submit("instruments", {})

    @app.get("/api/v1/watchlist")
    def watches():
        return store.watches()

    @app.post("/api/v1/watchlist")
    def add_watch(payload: dict):
        try:
            symbol = normalize_symbol(payload["symbol"])
            store.add_watch(symbol, payload.get("name", ""))
            return {"symbol": symbol}
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/v1/watchlist/{symbol}")
    def delete_watch(symbol: str):
        with store.db() as db:
            db.execute("DELETE FROM watchlist WHERE symbol=?", (normalize_symbol(symbol),))
        return {"deleted": True}

    @app.get("/api/v1/data/frequencies")
    def data_frequencies():
        return {"timeframes": MarketDataService(store).frequency_catalog()}

    @app.get("/api/v1/data/package")
    def data_package():
        return json_safe(MarketDataService(store).package_info())

    @app.get("/api/v1/data/inventory")
    def data_inventory(
        q: str = "", asset_type: Literal["stock", "etf"] | None = None,
        timeframe: str | None = None, status: str | None = None,
        limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
    ):
        try:
            return json_safe(MarketDataService(store).inventory(
                query=q, asset_type=asset_type, timeframe=timeframe,
                status=status, limit=limit, offset=offset,
            ))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/import-package/scan")
    def import_package_scan(payload: MarketPackageRequest):
        try:
            return json_safe(scan_etf_package(payload.folder))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/import-package")
    def import_package(payload: MarketPackageRequest):
        nonlocal package_dir
        try:
            values = payload.model_dump()
            values["timeframes"] = [
                normalize_timeframe(item) for item in (payload.timeframes or [])
            ] or None
            values["symbols"] = [
                normalize_symbol(item) for item in (payload.symbols or [])
            ] or None
            scan_etf_package(payload.folder)
            market = MarketDataService(store)
            market.repository.attach_package(payload.folder)
            package_dir = str(market.repository.package_dir)
            return submit("market_index_package", values)
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/datasets")
    def data_datasets(timeframe: str | None = None, q: str = "", status: str | None = None, limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0)):
        try:
            return json_safe(MarketDataService(store).datasets(timeframe, q, status, limit, offset))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/coverage/{symbol}")
    def coverage(symbol: str, timeframe: str | None = None, frequency: str | None = None):
        try:
            requested = timeframe or frequency
            if requested:
                return json_safe(MarketDataService(store).coverage(normalize_symbol(symbol), normalize_timeframe(requested)))
            return json_safe(DataService(store).coverage(symbol))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/v1/data/sync")
    def sync_data(payload: MarketSyncRequest):
        try:
            requested = payload.timeframes or [payload.timeframe or payload.frequency]
            requested = [normalize_timeframe(value) for value in requested if value]
            if not requested:
                requested = ["1day", "1min"]
            invalid = [value for value in requested if value not in {"1day", "1min"}]
            if invalid:
                raise ValueError("线上更新只支持 1day 和 1min；粗周期由基础行情按需聚合")
            timeframes = sorted(set(requested), key=lambda value: 0 if value == "1day" else 1)
            values = payload.model_dump()
            values["timeframe"] = timeframes[0]
            values["timeframes"] = timeframes
            values["symbols"] = [normalize_symbol(symbol) for symbol in payload.symbols]
            if "1min" in timeframes and payload.mode == "history":
                raise ValueError("1 分钟历史行情只从本地数据包读取；请先登记数据包或导入文件，近期缺口使用缺口修补。")
            if "1min" in timeframes and payload.scope == "market" and payload.initialize:
                raise ValueError("1 分钟行情不支持全市场在线初始化；请登记本地数据包或导入文件。")
            if payload.scope == "market" and payload.symbols:
                raise ValueError("全市场范围不能同时指定代码")
            if payload.scope in ("market", "watchlist"):
                values["symbols"] = []
            return submit("market_sync", values)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/sources")
    def data_sources():
        return configured_sources(store)

    @app.post("/api/v1/data/sources/probe")
    def probe_data_source(payload: MarketSourceProbeRequest):
        try:
            values = payload.model_dump()
            if payload.capability == "recent_minute" and payload.source not in {"sina", "tdx"}:
                raise ValueError("近期 1 分钟探测来源应为 sina 或 tdx")
            if payload.capability == "etf_daily" and payload.source not in {"baostock", "eastmoney"}:
                raise ValueError("ETF 日线探测来源应为 baostock 或 eastmoney")
            if payload.capability == "etf_daily" and payload.market == "stock":
                raise ValueError("ETF 日线探测仅支持 etf 或 all")
            return submit("market_source_probe", values)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/sync-status")
    def sync_status(run_key: str | None = None, status: str | None = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
        return store.sync_status(run_key, status, limit, offset)

    @app.post("/api/v1/data/demo")
    def demo(payload: dict):
        try:
            payload["symbol"] = normalize_symbol(payload.get("symbol", "000001.SZ"))
            return submit("demo", payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/import")
    def import_data(payload: MarketImportRequest):
        try:
            values = payload.model_dump()
            values["symbol"] = normalize_symbol(payload.symbol)
            values["timeframe"] = normalize_timeframe(payload.timeframe or payload.frequency)
            if not payload.path:
                raise ValueError("必须提供本地 CSV 或 Parquet 文件路径")
            return submit("market_import", values)
        except (ValueError, KeyError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/upload")
    async def upload_import_data(
        market_file: UploadFile = File(...),
        symbol: str = Form(...),
        timeframe: str = Form(...),
        label_convention: str = Form("auto"),
        etf_category: str | None = Form(None),
        calendar_file: UploadFile | None = File(None),
        actions_file: UploadFile | None = File(None),
        skip_invalid_rows: bool = Form(False),
        import_mode: str = Form("merge"),
    ):
        try:
            timeframe = normalize_timeframe(timeframe)
            symbol = normalize_symbol(symbol)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if label_convention not in {"auto", "start", "end"}:
            raise HTTPException(422, "时间标签应为 auto/start/end")
        if import_mode not in {"merge", "snapshot"}:
            raise HTTPException(422, "导入模式应为 merge/snapshot")
        market = MarketDataService(store)
        stage = market.repository.package.staging_dir / "uploads" / timeframe / uuid.uuid4().hex
        metadata_stage = stage.parent / (stage.name + "-metadata")
        try:
            market_path = stage / _safe_upload_path(market_file.filename or "market.csv")
            size = await _save_upload(market_file, market_path)
            calendar_path = None
            if calendar_file is not None and calendar_file.filename:
                calendar_path = metadata_stage / _safe_upload_path(calendar_file.filename)
                await _save_upload(calendar_file, calendar_path)
            actions_path = None
            if actions_file is not None and actions_file.filename:
                actions_path = stage / "actions" / _safe_upload_path(actions_file.filename)
                await _save_upload(actions_file, actions_path)
            payload = {
                "symbol": symbol, "timeframe": timeframe, "path": str(market_path),
                "label_convention": label_convention, "adjustment": "none",
                "etf_category": etf_category or None,
                "calendar_path": str(calendar_path) if calendar_path else None,
                "actions_path": str(actions_path) if actions_path else None,
                "skip_invalid_rows": skip_invalid_rows,
                "import_mode": import_mode,
                "upload_dirs": [
                    str(stage), str(metadata_stage),
                ],
            }
            submitted = submit("market_import", payload)
            return {**submitted, "file_name": market_file.filename, "bytes": size, "timeframe": timeframe}
        except (ValueError, MarketDataError) as exc:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(metadata_stage, ignore_errors=True)
            raise HTTPException(422, str(exc)) from exc
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(metadata_stage, ignore_errors=True)
            raise

    @app.post("/api/v1/data/upload-batch")
    async def upload_import_batch(
        files: list[UploadFile] = File(...),
        timeframe: str = Form(...),
        label_convention: str = Form("auto"),
        etf_category: str | None = Form(None),
        calendar_file: UploadFile | None = File(None),
        skip_invalid_rows: bool = Form(False),
        import_mode: str = Form("merge"),
    ):
        try:
            timeframe = normalize_timeframe(timeframe)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if label_convention not in {"auto", "start", "end"}:
            raise HTTPException(422, "时间标签应为 auto/start/end")
        if import_mode not in {"merge", "snapshot"}:
            raise HTTPException(422, "导入模式应为 merge/snapshot")
        if not files:
            raise HTTPException(422, "请选择至少一个 CSV 或 Parquet 文件")
        market = MarketDataService(store)
        stage = market.repository.package.staging_dir / "uploads" / timeframe / uuid.uuid4().hex
        metadata_stage = stage.parent / (stage.name + "-metadata")
        count = 0
        total_bytes = 0
        try:
            for upload in files:
                relative = _safe_upload_path(upload.filename or "upload.csv")
                destination = stage / relative
                if destination.exists():
                    raise ValueError("上传文件存在重复相对路径：" + relative.as_posix())
                total_bytes += await _save_upload(upload, destination)
                count += 1
            calendar_path = None
            if calendar_file is not None and calendar_file.filename:
                calendar_path = metadata_stage / _safe_upload_path(calendar_file.filename)
                await _save_upload(calendar_file, calendar_path)
            payload = {
                "folder": str(stage), "timeframe": timeframe, "recursive": True,
                "label_convention": label_convention, "adjustment": "none",
                "etf_category": etf_category or None, "resume": True,
                "calendar_path": str(calendar_path) if calendar_path else None,
                "skip_invalid_rows": skip_invalid_rows,
                "import_mode": import_mode,
                "upload_dirs": [
                    str(stage), str(metadata_stage),
                ],
            }
            submitted = submit("market_import_batch", payload)
            return {**submitted, "files": count, "bytes": total_bytes, "timeframe": timeframe}
        except (ValueError, MarketDataError) as exc:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(metadata_stage, ignore_errors=True)
            raise HTTPException(422, str(exc)) from exc
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(metadata_stage, ignore_errors=True)
            raise

    @app.post("/api/v1/data/import-batch/scan")
    def import_batch_scan(payload: MarketBatchRequest):
        try:
            timeframe = normalize_timeframe(payload.timeframe or payload.frequency)
            folder = payload.folder or str(Path(root) / "imports" / timeframe)
            return json_safe(MarketDataService(store).preview_folder(folder, timeframe, recursive=payload.recursive))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/import-batch")
    def import_batch(payload: MarketBatchRequest):
        try:
            timeframe = normalize_timeframe(payload.timeframe or payload.frequency)
            values = payload.model_dump()
            values["timeframe"] = timeframe
            values["folder"] = payload.folder or str(Path(root) / "imports" / timeframe)
            return submit("market_import_batch", values)
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/validate")
    def validate_data(payload: MarketValidateRequest):
        try:
            timeframe = normalize_timeframe(payload.timeframe or payload.frequency)
            return submit("market_validate", {"symbol": normalize_symbol(payload.symbol), "timeframe": timeframe,
                                             "label_convention": payload.label_convention,
                                             "mode": payload.mode,
                                             "start_date": payload.start_date,
                                             "end_date": payload.end_date})
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/archive")
    def archive_data(payload: MarketArchiveRequest):
        try:
            timeframe = normalize_timeframe(payload.timeframe or payload.frequency)
            return submit("market_archive", {
                "symbols": [normalize_symbol(value) for value in payload.symbols],
                "timeframe": timeframe,
            })
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/compliance/{symbol}")
    def data_compliance(symbol: str):
        try:
            market = MarketDataService(store)
            normalized = normalize_symbol(symbol)
            return json_safe({
                timeframe: market.repository.validation_state(normalized, timeframe)
                for timeframe in ("1day", "1min")
                if market.repository.dataset(normalized, timeframe)
            })
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/aggregate")
    def aggregate_data(payload: MarketAggregateRequest):
        try:
            values = payload.model_dump()
            values["symbols"] = [normalize_symbol(symbol) for symbol in payload.symbols]
            values["source_timeframe"] = normalize_timeframe(payload.source_timeframe)
            values["target_timeframe"] = normalize_timeframe(payload.target_timeframe)
            return submit("market_aggregate", values)
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/readiness")
    def data_readiness(config: BacktestConfig, version: str | None = None):
        try:
            return json_safe(MarketDataService(store).readiness(config, version))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/preparation/plan")
    def preparation_plan(payload: MarketPreparationRequest):
        try:
            return json_safe(MarketPreparationService(MarketDataService(store)).plan(payload.config))
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/preparation")
    def preparation_submit(payload: MarketPreparationRequest):
        try:
            return submit("market_prepare", {
                "config": payload.config.model_dump(mode="json"),
                "version": payload.version,
            })
        except (ValueError, MarketDataError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/data/migrate-legacy")
    def migrate_legacy(payload: dict):
        return submit("market_migrate", {"dry_run": bool(payload.get("dry_run", False)), "resume": bool(payload.get("resume", True))})

    @app.get("/api/v1/data/bars/{symbol}")
    def bars(symbol: str, start_date: str | None = None, end_date: str | None = None, frequency: str = "1m", limit: int = Query(1500, ge=1, le=10000), version: str | None = None):
        normalized = normalize_timeframe(frequency)
        market = MarketDataService(store)
        manifest = store.snapshot(symbol, version)
        if version and manifest is None and not str(version).isdigit():
            raise HTTPException(404, {
                "code": "DATA_VERSION_NOT_FOUND",
                "message": f"原数据版本 {version} 已清理，不能按该版本读取",
            })
        if market.repository.dataset(normalize_symbol(symbol), normalized) or (manifest and manifest.get("market_data")):
            try:
                market_version = None
                if manifest and manifest.get("market_data"):
                    reference = (manifest.get("market_versions", {}).get(normalized) or {}).get("version")
                    if version and reference is None:
                        return []
                    market_version = int(reference) if reference is not None else None
                elif version and str(version).isdigit():
                    market_version = int(version)
                payload = market.bars(
                    symbol, normalized, start=pd.Timestamp(start_date) if start_date else None,
                    end=pd.Timestamp(end_date) if end_date else None,
                    version=market_version,
                    limit=limit,
                )
                return json_safe(payload["rows"])
            except DataVersionNotFound as exc:
                raise HTTPException(404, {"code": exc.code, "message": str(exc)}) from exc
            except MarketDataError:
                return []
        if not manifest:
            return []
        daily = store.read_files(manifest["files"]["daily"])
        if daily.empty:
            return []
        bundle = DataService(store).load_bundle(BacktestConfig(symbol=symbol, start_date=pd.Timestamp(start_date or daily.date.min()).date(), end_date=pd.Timestamp(end_date or daily.date.max()).date(), frequency=frequency), version=version)
        frame = bundle.bars
        if len(frame) > limit:
            # Uniformly sized contiguous OHLC buckets preserve all highs/lows.
            groups = pd.Series(range(len(frame))) // ((len(frame) + limit - 1) // limit)
            frame = frame.groupby(groups).agg(timestamp=("timestamp", "last"), open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        return json_safe(frame[[c for c in ("timestamp", "open", "high", "low", "close", "volume", "frequency", "is_synthetic") if c in frame]].to_dict("records"))

    @app.post("/api/v1/backtests")
    def backtest(config: BacktestConfig):
        return submit("backtest", {"config": config.model_dump(mode="json"), "version": require_ready(config)})

    @app.get("/api/v1/data/reference-price/{symbol}")
    def reference_price(symbol: str, start_date: str):
        try:
            return DataService(store).reference_price(normalize_symbol(symbol), pd.Timestamp(start_date).date())
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/market/bars/{symbol}")
    def market_bars(symbol: str, frequency: str = "1d", start_date: str | None = None, end_date: str | None = None, before: str | None = None, limit: int = Query(600, ge=1, le=5000), ma: str = "5,10,20,60"):
        try:
            timeframe = normalize_timeframe(frequency)
            market = MarketDataService(store)
            manifest = store.snapshot(symbol)
            if market.repository.dataset(normalize_symbol(symbol), timeframe) or (manifest and manifest.get("market_data")):
                return json_safe(market.market_bars(
                    symbol, timeframe, start=start_date, end=end_date, before=before,
                    limit=limit, ma_periods=[int(value) for value in ma.split(",") if value.strip()],
                ))
            return json_safe(DataService(store).market_bars(symbol, frequency, start_date, end_date, before, limit, [int(value) for value in ma.split(",") if value.strip()]))
        except MarketDataError as exc:
            return {"symbol": normalize_symbol(symbol), "frequency": normalize_timeframe(frequency),
                    "bars": [], "warnings": [str(exc)], "quality_flags": [str(exc)], "diagnostics": {}, "source_events": []}
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/backtests/preview")
    def preview(payload: PreviewRequest):
        if payload.config.schema_version != 2:
            raise HTTPException(422, "请先将旧配置转换为股数配置")
        version = require_ready(payload.config)
        # Preview is deliberately submitted without a synchronous coverage scan;
        # the worker reports preparation progress and pins the result version.
        return json_safe(previews.submit(payload.session_id, payload.revision, payload.config, version))

    @app.get("/api/v1/backtests/preview/{session_id}")
    def preview_status(session_id: str):
        status = previews.snapshot(session_id)
        if status is None:
            raise HTTPException(404, "预览已过期")
        return json_safe(status)

    @app.delete("/api/v1/backtests/preview/{session_id}")
    def cancel_preview(session_id: str):
        previews.cancel(session_id)
        return {"cancelled": True}

    @app.post("/api/v1/configs/convert")
    def convert_config(config: BacktestConfig):
        if config.schema_version == 2:
            return {"config": config.model_dump(mode="json"), "notice": "已载入股数配置"}
        try:
            service = DataService(store)
            bundle = service.ensure_bundle(config, check_minutes=False)
            context = service.reference_price(config.symbol, config.start_date, bundle.version)
            price = context["reference_price"]
            def shares(ratio):
                return int(config.initial_cash * ratio / price / 100) * 100
            def quantity(value):
                amount = value.value if value.mode == "shares" else value.value / price if value.mode == "cash" else config.initial_cash * value.value / price
                return max(200 if config.symbol.startswith(("688", "689")) else 100, int(amount / 100) * 100)
            values = {key: value for key, value in config.model_dump(mode="json").items() if key not in {"schema_version", "initial_cash", "initial_position", "min_position", "max_position", "buy_quantity", "sell_quantity", "dca_frequency"}}
            values.update(schema_version=2, initial_shares=shares(config.initial_position), min_shares=shares(config.min_position), max_shares=shares(config.max_position), buy_shares=quantity(config.buy_quantity), sell_shares=quantity(config.sell_quantity), reference_price_mode="manual" if config.reference_price else "auto", dca={"frequency": config.dca_frequency, "shares": 200 if config.symbol.startswith(("688", "689")) else 100})
            modern = BacktestConfig.model_validate(values)
            return {"config": modern.model_dump(mode="json"), "reference_date": context["reference_date"], "notice": "旧现金/比例配置已按历史参考价换算为股数草稿；资金按新规则重算，请检查后另存，原配置保持保留。"}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/backtests/batch")
    def batch(payload: BatchRequest):
        versions = [require_ready(config) for config in payload.configs]
        return {"job_ids": [submit("backtest", {"config": config.model_dump(mode="json"), "version": version})["job_id"] for config, version in zip(payload.configs, versions)]}

    @app.post("/api/v1/optimizations")
    def optimization(config: OptimizationConfig):
        if config.protocol_version == 2:
            prepared = preparation_config(store, config.base)
            version = require_ready(prepared)
        else:
            version = require_ready(config.base)
        return submit("optimization", {"config": config.model_dump(mode="json"), "version": version})

    @app.get("/api/v1/jobs")
    def jobs():
        return store.jobs()

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str):
        job = store.job(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return job

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = get_job(job_id)
        if job["status"] in ("queued", "running"):
            store.update_job(job_id, cancel_requested=1)
        return store.job(job_id)

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        get_job(job_id)
        async def events():
            previous = None
            while True:
                job = store.job(job_id)
                serialized = dumps(job)
                if serialized != previous:
                    yield f"data: {serialized}\n\n"
                    previous = serialized
                if not job or job["status"] in ("completed", "partial", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.5)
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/v1/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = get_job(job_id)
        if not job.get("result_id"):
            raise HTTPException(409, "任务尚无结果")
        result = store.result(job["result_id"])
        return _with_data_availability(
            store, public_result(result, data_metadata=_result_data_metadata(store, result))
        )

    @app.get("/api/v1/jobs/{job_id}/items")
    def job_items(job_id: str, section: str = "all", offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        job = get_job(job_id)
        if not job.get("result_id"):
            raise HTTPException(409, "任务尚无结果")
        result = store.result(job["result_id"])
        content = result.get("result", result)
        groups: list[dict] = []
        for name in ("results", "failures", "skipped_items", "symbols", "skipped"):
            if section not in ("all", name):
                continue
            values = content.get(name, []) if isinstance(content, dict) else []
            for item in values:
                groups.append({"section": name, **item} if isinstance(item, dict) else {"section": name, "value": item})
        if not groups and isinstance(content, dict):
            groups.append({"section": "result", **content})
        total = len(groups)
        return {"items": groups[offset:offset + limit], "total": total, "offset": offset, "limit": limit}

    @app.get("/api/v1/results")
    def results():
        return store.results()

    @app.get("/api/v1/results/{result_id}")
    def get_result(result_id: str, full: bool = False):
        result = store.result(result_id)
        if result is None:
            raise HTTPException(404, "结果不存在")
        public = public_result(
            result, full=full,
            data_metadata=None if full else _result_data_metadata(store, result),
        )
        return public if full else _with_data_availability(store, public)

    @app.get("/api/v1/results/{result_id}/export")
    def export(result_id: str, kind: str = "json"):
        result = get_result(result_id, full=True)
        if kind == "json":
            content, extension, media = dumps(result).encode("utf-8"), "json", "application/json"
        elif kind in ("equity", "trades"):
            records = []
            if result.get("protocol_version") == 2:
                runs = []
                for name in ("fixed", "adaptive"):
                    branch = result.get(name)
                    if isinstance(branch, dict):
                        runs.append((name, branch.get("test_result") or {}))
                for selection, run in runs:
                    for strategy, values in run.get("strategies", {}).items():
                        records.extend({"selection": selection, "strategy": strategy, **row} for row in values.get(kind, []))
            else:
                run = result if "strategies" in result else (result.get("historical", {}).get("best_result") or result.get("robust", {}).get("test_result") or {})
                for strategy, values in run.get("strategies", {}).items():
                    records.extend({"strategy": strategy, **row} for row in values.get(kind, []))
            content = pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig")
            extension, media = "csv", "text/csv"
        else:
            raise HTTPException(422, "导出类型应为 json/equity/trades")
        return Response(content=content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{result_id}-{kind}.{extension}"'})

    @app.get("/api/v1/configs")
    def configs():
        with store.db() as db:
            rows = db.execute("SELECT * FROM configs ORDER BY created_at DESC").fetchall()
        return [{**dict(row), "config": json.loads(row["config"])} for row in rows]

    @app.post("/api/v1/configs")
    def save_config(payload: dict):
        try:
            config = BacktestConfig.model_validate(payload["config"])
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        config_id = uuid.uuid4().hex
        with store.db() as db:
            db.execute("INSERT INTO configs VALUES(?,?,?,?)", (config_id, payload.get("name", "网格配置"), config.model_dump_json(), now_iso()))
        return {"id": config_id}

    @app.delete("/api/v1/configs/{config_id}")
    def delete_config(config_id: str):
        with store.db() as db:
            db.execute("DELETE FROM configs WHERE id=?", (config_id,))
        return {"deleted": True}

    @app.get("/api/v1/settings")
    def settings():
        current = load_settings()
        return {
            **current.model_dump(), "active_data_dir": root,
            "active_package_dir": package_dir,
            "package_dir": current.package_dir or package_dir,
            "restart_required": current.data_dir != root or bool(current.package_dir and current.package_dir != package_dir),
            "market_cache": cache_settings(),
        }

    @app.put("/api/v1/settings")
    def update_settings(payload: dict):
        try:
            validate_source_settings(store, payload)
            values = save_settings(payload)
            configure_cache(values.market_cache_mb)
            return {
                **values.model_dump(), "active_data_dir": root,
                "active_package_dir": package_dir,
                "restart_required": values.data_dir != root or bool(values.package_dir and values.package_dir != package_dir),
                "market_cache": cache_settings(),
            }
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/schedule/install")
    def install_schedule():
        try:
            return configure_schedule(True)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/v1/schedule/remove")
    def remove_schedule():
        try:
            return configure_schedule(False)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    frontend = PROJECT_ROOT / "frontend" / "dist"
    if (frontend / "assets").exists():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def index(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "接口不存在")
        if (frontend / "index.html").exists():
            return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})
        return {"message": "后端已启动。请在 frontend 目录运行 npm ci 和 npm run build，然后刷新", "docs": "/docs"}

    return app
