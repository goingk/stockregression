from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from .preview import PreviewManager

from .data import DataService
from .jobs import JobManager
from .models import BacktestConfig, OptimizationConfig, normalize_symbol
from .rules import instrument_terms
from .schedule import configure_schedule
from .settings import PROJECT_ROOT, load_settings, save_settings
from .storage import Store, dumps, json_safe, now_iso


class SyncRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    mode: str = "history"
    scope: Literal["market", "watchlist", "selected"] | None = None
    resume: bool = False
    retry_failed: bool = False
    workers: int = Field(default=2, ge=1, le=4)


class PreviewRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    revision: int = Field(ge=1)
    config: BacktestConfig


class BatchRequest(BaseModel):
    configs: list[BacktestConfig] = Field(min_length=1, max_length=100)


def public_result(result: dict) -> dict:
    """Keep full resolution on disk/export; send daily curves to the browser."""
    result = dict(result)
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
            result[mode] = {**result[mode], key: public_result(result[mode][key])}
    return result


def create_app(data_dir: str | None = None, start_jobs: bool = True):
    configured = load_settings()
    root = str(Path(data_dir or configured.data_dir).resolve())
    store = Store(root)
    manager = JobManager(root, configured.max_workers)
    previews = PreviewManager(store)

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
    app.state.previews = previews
    app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

    def submit(kind, payload):
        return {"job_id": store.new_job(kind, payload)}

    def snapshot_version(config):
        # Preserve a submission-time immutable input when the active snapshot
        # already covers this request. Missing or stale coverage is prepared by
        # the worker, so the API never blocks on a source download.
        return DataService(store).current_version_if_ready(config)

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "data_dir": root, "version": "0.1.0"}

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

    @app.get("/api/v1/data/coverage/{symbol}")
    def coverage(symbol: str):
        try:
            return json_safe(DataService(store).coverage(symbol))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/v1/data/sync")
    def sync_data(payload: SyncRequest):
        if payload.mode not in ("history", "daily", "repair"):
            raise HTTPException(422, "未知同步模式")
        try:
            values = payload.model_dump()
            values["symbols"] = [normalize_symbol(symbol) for symbol in payload.symbols]
            if payload.scope == "market" and payload.symbols:
                raise ValueError("全市场范围不能同时指定代码")
            return submit("sync", values)
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
    def import_data(payload: dict):
        try:
            payload["symbol"] = normalize_symbol(payload["symbol"])
            return submit("import", payload)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/data/bars/{symbol}")
    def bars(symbol: str, start_date: str | None = None, end_date: str | None = None, frequency: str = "1m", limit: int = Query(1500, ge=1, le=10000), version: str | None = None):
        manifest = store.snapshot(symbol, version)
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
        return submit("backtest", {"config": config.model_dump(mode="json"), "version": snapshot_version(config)})

    @app.get("/api/v1/data/reference-price/{symbol}")
    def reference_price(symbol: str, start_date: str):
        try:
            return DataService(store).reference_price(normalize_symbol(symbol), pd.Timestamp(start_date).date())
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/market/bars/{symbol}")
    def market_bars(symbol: str, frequency: str = "1d", start_date: str | None = None, end_date: str | None = None, before: str | None = None, limit: int = Query(600, ge=1, le=5000), ma: str = "5,10,20,60"):
        try:
            return json_safe(DataService(store).market_bars(symbol, frequency, start_date, end_date, before, limit, [int(value) for value in ma.split(",") if value.strip()]))
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/backtests/preview")
    def preview(payload: PreviewRequest):
        if payload.config.schema_version != 2:
            raise HTTPException(422, "请先将旧配置转换为股数配置")
        # Preview is deliberately submitted without a synchronous coverage scan;
        # the worker reports preparation progress and pins the result version.
        return json_safe(previews.submit(payload.session_id, payload.revision, payload.config, None))

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
        versions = [snapshot_version(config) for config in payload.configs]
        return {"job_ids": [submit("backtest", {"config": config.model_dump(mode="json"), "version": version})["job_id"] for config, version in zip(payload.configs, versions)]}

    @app.post("/api/v1/optimizations")
    def optimization(config: OptimizationConfig):
        return submit("optimization", {"config": config.model_dump(mode="json"), "version": snapshot_version(config.base)})

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
                if not job or job["status"] in ("completed", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.5)
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/v1/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = get_job(job_id)
        if not job.get("result_id"):
            raise HTTPException(409, "任务尚无结果")
        return public_result(store.result(job["result_id"]))

    @app.get("/api/v1/results")
    def results():
        return store.results()

    @app.get("/api/v1/results/{result_id}")
    def get_result(result_id: str, full: bool = False):
        result = store.result(result_id)
        if result is None:
            raise HTTPException(404, "结果不存在")
        return result if full else public_result(result)

    @app.get("/api/v1/results/{result_id}/export")
    def export(result_id: str, kind: str = "json"):
        result = get_result(result_id, full=True)
        if kind == "json":
            content, extension, media = dumps(result).encode("utf-8"), "json", "application/json"
        elif kind in ("equity", "trades"):
            records = []
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
        return {**load_settings().model_dump(), "active_data_dir": root, "restart_required": load_settings().data_dir != root}

    @app.put("/api/v1/settings")
    def update_settings(payload: dict):
        try:
            values = save_settings(payload)
            return {**values.model_dump(), "restart_required": values.data_dir != root}
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
