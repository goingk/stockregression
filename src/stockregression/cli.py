from __future__ import annotations

import multiprocessing
import os
import subprocess
import webbrowser
import json
from pathlib import Path
import shutil

import typer

from .jobs import execute_job
from .schedule import configure_schedule
from .settings import PROJECT_ROOT, load_settings
from .storage import Store
from .market_data.package import scan_etf_package
from .market_data.maintenance import preview_reset, rebuild_state, reset_market_storage

app = typer.Typer(help="本地 A 股及 ETF 行情与网格研究台", no_args_is_help=True)
schedule_app = typer.Typer(help="安装、移除或查看 Windows 补数计划")
data_app = typer.Typer(help="按周期管理本地行情数据库")
config_app = typer.Typer(help="查看并校验项目主配置")
app.add_typer(schedule_app, name="schedule")
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")


@app.command()
def serve(
    port: int | None = typer.Option(None, help="覆盖配置中的监听端口"),
    host: str | None = typer.Option(None, help="覆盖配置中的监听地址"),
    open_browser: bool = False,
):
    """启动本地网页和独立计算进程。"""
    import uvicorn
    from .api import create_app
    import threading

    settings = load_settings()
    host = host or settings.host
    port = int(port or settings.port)
    application = create_app()
    server = uvicorn.Server(uvicorn.Config(
        application, host=host, port=port, log_level="info",
    ))

    def restart_callback(_active_jobs: int):
        if os.name != "nt":
            raise RuntimeError("当前系统不支持由项目启动脚本自动重启")
        restart_script = PROJECT_ROOT / "restart.ps1"
        if not restart_script.is_file():
            raise RuntimeError("缺少项目重启脚本 restart.ps1")
        subprocess.Popen(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
                "-ExecutionPolicy", "Bypass", "-File", str(restart_script),
                "-ParentPid", str(os.getpid()), "-Port", str(port),
            ],
            cwd=str(PROJECT_ROOT),
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
        threading.Timer(0.8, lambda: setattr(server, "should_exit", True)).start()

    application.state.restart_callback = restart_callback
    if open_browser:
        timer = threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}"))
        timer.daemon = True
        timer.start()
    server.run()


@app.command()
def sync(mode: str = "daily", symbol: list[str] = typer.Option(None), scope: str | None = None, resume: bool = False, retry_failed: bool = False, workers: int = 4, dry_run: bool = False):
    """无网页时独立补数；mode 为 history/daily/repair。"""
    if mode not in ("history", "daily", "repair"):
        raise typer.BadParameter("mode 为 history/daily/repair")
    if scope not in (None, "market", "watchlist", "selected") or workers not in range(1, 5):
        raise typer.BadParameter("scope 为 market/watchlist/selected，workers 为 1–4")
    if scope == "market" and symbol:
        raise typer.BadParameter("全市场范围不能同时指定代码")
    settings = load_settings()
    root = Path(settings.data_dir)
    existing = root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    typer.echo(f"数据目录：{root}；可用空间：{shutil.disk_usage(existing).free / 1024 ** 3:.1f} GiB")
    typer.echo("全市场初始化会扩大证券范围；1 分钟历史只从本地数据包读取。" if scope == "market" else "仅更新指定范围；近期分钟来源按数据源设置使用")
    if dry_run:
        from .providers import fetch_instruments
        from .models import normalize_symbol
        if scope == "market":
            try:
                records = fetch_instruments(
                    network_timeout=settings.market_data.network_timeout_seconds,
                )
                symbols = sorted({item["symbol"] for item in records})
                typer.echo(f"目录目标：{len(symbols)} 个（股票 {sum(not code.startswith(('1', '5')) for code in symbols)}，ETF {sum(code.startswith(('1', '5')) for code in symbols)}）；不自动加入自选")
                for warning in sorted({warning for item in records for warning in item.get("catalog_warnings", [])}):
                    typer.echo(warning, err=True)
            except Exception as exc:
                raise typer.BadParameter(f"无法预检完整证券清单：{exc}") from exc
        else:
            symbols = [normalize_symbol(code) for code in symbol] if symbol else [item["symbol"] for item in Store(root).watches()]
            typer.echo(f"目标：{len(set(symbols))} 个标的")
        typer.echo("预检结束，未下载行情或创建同步任务。目录元数据可能写入来源缓存。")
        return
    timeframes = ["1day"] if mode == "history" else ["1day", "1min"]
    _run("market_sync", {
        "mode": "incremental" if mode == "daily" else mode,
        "symbols": symbol or [], "scope": scope, "resume": resume,
        "retry_failed": retry_failed, "workers": workers,
        "timeframe": "1day", "timeframes": timeframes,
    })


@data_app.command("import-file")
def data_import_file(
    symbol: str,
    frequency: str = typer.Option(..., "--frequency"),
    path: Path = typer.Option(..., "--path"),
    label_convention: str = typer.Option("auto", "--label-convention"),
    etf_category: str | None = typer.Option(None, "--etf-category"),
    calendar_path: Path | None = typer.Option(None, "--calendar-path"),
    actions_path: Path | None = typer.Option(None, "--actions-path"),
    skip_invalid_rows: bool = typer.Option(False, "--skip-invalid-rows"),
    import_mode: str = typer.Option("merge", "--import-mode"),
):
    """导入一个明确周期的 CSV 或 Parquet 文件。"""
    _run("market_import", {
        "symbol": symbol, "timeframe": frequency, "path": str(path),
        "label_convention": label_convention, "adjustment": "none",
        "etf_category": etf_category,
        "calendar_path": str(calendar_path) if calendar_path else None,
        "actions_path": str(actions_path) if actions_path else None,
        "skip_invalid_rows": skip_invalid_rows,
        "import_mode": import_mode,
    })


@data_app.command("import-folder")
def data_import_folder(
    frequency: str = typer.Option(..., "--frequency"),
    folder: Path | None = typer.Option(None, "--folder"),
    recursive: bool = typer.Option(False, "--recursive"),
    label_convention: str = typer.Option("auto", "--label-convention"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    resume: bool = typer.Option(True, "--resume/--fresh"),
    calendar_path: Path | None = typer.Option(None, "--calendar-path"),
    workers: int = typer.Option(4, "--workers", min=1, max=4),
    skip_invalid_rows: bool = typer.Option(False, "--skip-invalid-rows"),
    import_mode: str = typer.Option("merge", "--import-mode"),
):
    """扫描文件夹并通过文件名六位代码批量导入。"""
    settings = load_settings()
    target = folder or Path(settings.data_dir) / "imports" / frequency
    if dry_run:
        from .market_data import MarketDataService
        preview = MarketDataService(Store(settings.data_dir)).preview_folder(target, frequency, recursive=recursive)
        typer.echo(json.dumps(preview, ensure_ascii=False, indent=2, default=str))
        return
    _run("market_import_batch", {
        "folder": str(target), "timeframe": frequency, "recursive": recursive,
        "label_convention": label_convention, "adjustment": "none", "resume": resume,
        "calendar_path": str(calendar_path) if calendar_path else None,
        "workers": workers,
        "skip_invalid_rows": skip_invalid_rows,
        "import_mode": import_mode,
    })


@data_app.command("import-etf-package")
def data_import_etf_package(
    folder: Path = typer.Option(..., "--folder"),
    timeframe: list[str] = typer.Option(None, "--timeframe"),
    symbol: list[str] = typer.Option(None, "--symbol"),
    label_convention: str = typer.Option("auto", "--label-convention"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    resume: bool = typer.Option(True, "--resume/--fresh"),
    workers: int = typer.Option(4, "--workers", min=1, max=4),
    skip_invalid_rows: bool = typer.Option(False, "--skip-invalid-rows"),
):
    """扫描并导入 ETF 基本资料、日线和分钟 Parquet 数据包。"""
    preview = scan_etf_package(folder)
    if dry_run:
        typer.echo(json.dumps(preview, ensure_ascii=False, indent=2, default=str))
        return
    _run("market_index_package", {
        "folder": str(folder), "format": "etf",
        "timeframes": timeframe or None, "symbols": symbol or None,
        "label_convention": label_convention, "resume": resume,
        "workers": workers, "skip_invalid_rows": skip_invalid_rows,
    })


@data_app.command("rebuild-parquet")
def data_rebuild_parquet(
    folder: Path = typer.Option(..., "--folder"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    reset: bool = typer.Option(False, "--reset"),
    resume: bool = typer.Option(False, "--resume"),
    workers: int = typer.Option(4, "--workers", min=1, max=4),
):
    """清理旧行情并从 ETF 数据包建立新的 Parquet 行情库。"""
    settings = load_settings()
    root = Path(settings.data_dir)
    if dry_run:
        typer.echo(json.dumps({
            "reset": preview_reset(root), "package": scan_etf_package(folder),
        }, ensure_ascii=False, indent=2, default=str))
        return
    if not reset and not resume:
        raise typer.BadParameter("必须指定 --reset 或 --resume")
    if reset:
        if resume:
            raise typer.BadParameter("--reset 与 --resume 不能同时使用")
        reset_market_storage(root, source_folder=folder)
    else:
        state = rebuild_state(Store(root))
        if not state:
            raise typer.BadParameter("没有可恢复的 Parquet 重建，请先执行 --reset")
        folder = Path(state.get("source", folder))
    _run("market_index_package", {
        "folder": str(folder), "format": "etf", "timeframes": None,
        "symbols": None, "label_convention": "auto", "resume": True,
        "workers": workers, "skip_invalid_rows": False,
    })


@data_app.command("sync")
def data_sync(
    frequency: list[str] = typer.Option(None, "--frequency"),
    mode: str = typer.Option("incremental", "--mode"),
    symbol: list[str] = typer.Option(None, "--symbol"),
    scope: str | None = typer.Option(None, "--scope"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    resume: bool = typer.Option(False, "--resume"),
    retry_failed: bool = typer.Option(False, "--retry-failed"),
    initialize: bool = typer.Option(False, "--initialize"),
    workers: int = typer.Option(4, "--workers", min=1, max=4),
):
    """更新日线或 1 分钟基础行情；粗周期按需生成。"""
    timeframes = frequency or ["1day", "1min"]
    if any(value not in {"1day", "1min"} for value in timeframes):
        raise typer.BadParameter("线上更新只支持 1day 和 1min；粗周期来自基础行情聚合")
    if "1min" in timeframes and mode == "history":
        raise typer.BadParameter("1 分钟历史行情只从本地数据包读取；请登记数据包或导入文件，近期缺口使用 repair 模式。")
    if "1min" in timeframes and scope == "market" and initialize:
        raise typer.BadParameter("1 分钟行情不支持全市场在线初始化；请登记数据包或导入文件。")
    _run("market_sync", {
        "symbols": symbol or [], "timeframe": timeframes[0], "timeframes": timeframes,
        "mode": mode, "scope": scope,
        "start_date": start, "end_date": end, "resume": resume,
        "retry_failed": retry_failed,
        "initialize": initialize,
        "workers": workers,
    })


@data_app.command("sources")
def data_sources():
    """查看数据源选择、最近检测结果和失败冷却状态。"""
    from .source_health import configured_sources

    settings = load_settings()
    typer.echo(json.dumps(configured_sources(Store(settings.data_dir)), ensure_ascii=False, indent=2, default=str))


@data_app.command("probe")
def data_probe(
    source: str = typer.Option(..., "--source"),
    capability: str = typer.Option("recent_minute", "--capability"),
    market: str = typer.Option("all", "--market"),
):
    """只探测数据源，不写入行情；检测通过后可切换为该能力的可选来源。"""
    pairs = {"recent_minute": {"sina", "tdx"}, "etf_daily": {"baostock", "eastmoney"}}
    if capability not in pairs or source not in pairs[capability]:
        raise typer.BadParameter("capability 为 recent_minute/etf_daily，且 source 必须匹配该能力")
    if market not in {"all", "stock", "etf"}:
        raise typer.BadParameter("market 为 all/stock/etf")
    if capability == "etf_daily" and market == "stock":
        raise typer.BadParameter("ETF 日线探测仅支持 etf 或 all")
    _run("market_source_probe", {"source": source, "capability": capability, "market": market})


@data_app.command("validate")
def data_validate(
    symbol: str,
    frequency: str = typer.Option(..., "--frequency"),
    mode: str = typer.Option("incremental", "--mode"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
):
    """按交易日增量校验本地行情。"""
    if mode not in {"incremental", "full"}:
        raise typer.BadParameter("校验模式应为 incremental/full")
    _run("market_validate", {
        "symbol": symbol, "timeframe": frequency, "mode": mode,
        "start_date": start, "end_date": end,
    })


@data_app.command("archive")
def data_archive(
    symbol: list[str] = typer.Option(..., "--symbol"),
    frequency: str = typer.Option("1day", "--frequency"),
):
    """复制原始压缩文件，保护其历史内容不被后续覆盖。"""
    _run("market_archive", {"symbols": symbol, "timeframe": frequency})


@data_app.command("aggregate")
def data_aggregate(
    symbol: list[str] = typer.Option(None, "--symbol"),
    source: str = typer.Option(..., "--from"),
    target: str = typer.Option(..., "--to"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
):
    """从已有本地细周期显式生成粗周期。"""
    if not symbol:
        raise typer.BadParameter("至少指定一个 --symbol")
    _run("market_aggregate", {"symbols": symbol, "source_timeframe": source,
                             "target_timeframe": target, "start_date": start, "end_date": end})


@schedule_app.command("install")
def schedule_install():
    """以当前用户安装每日更新和周末核对任务。"""
    typer.echo(configure_schedule(True)["message"])


@schedule_app.command("remove")
def schedule_remove():
    """移除本应用注册的两个计划任务。"""
    typer.echo(configure_schedule(False)["message"])


@schedule_app.command("status")
def schedule_status():
    typer.echo(load_settings().schedule.model_dump_json(indent=2))


@app.command("set-data-dir")
def set_data_dir(path: Path):
    """修改数据目录，下次启动生效；原目录内容保留。"""
    from .configuration import current_configuration, patch_configuration

    document = current_configuration()
    patch_configuration(
        {"paths": {"data_dir": str(path.resolve())}},
        base_revision=document["config_revision"],
    )
    typer.echo(load_settings().data_dir)


@config_app.command("show")
def config_show():
    """输出当前主配置文件及其内容哈希。"""
    from .configuration import current_configuration

    typer.echo(json.dumps(current_configuration(), ensure_ascii=False, indent=2))


@config_app.command("validate")
def config_validate(path: Path | None = typer.Option(None, exists=True, dir_okay=False)):
    """校验主文件或指定 JSON 草稿，不写入任何配置。"""
    from .settings import read_configuration_document

    _, revision = read_configuration_document(path)
    typer.echo(f"配置有效；revision={revision}")


def _run(kind: str, payload: dict):
    import uuid
    settings = load_settings()
    store = Store(settings.data_dir)
    job_id = store.new_job(kind, payload)
    if not store.claim_job(job_id, uuid.uuid4().hex):
        raise typer.BadParameter("任务已被其他调度器领取")
    typer.echo(f"任务 {job_id}，数据目录 {store.root}")
    execute_job(str(store.root), job_id, lambda fraction, message: typer.echo(f"{fraction:6.1%} · {message}"))
    result = store.job(job_id)
    typer.echo(result.get("message", ""))
    if result.get("error"):
        typer.echo(result["error"], err=True)
        if not result.get("result_id"):
            raise typer.Exit(1)
    if result.get("result_id"):
        typer.echo(f"结果：{store.root / 'results' / (result['result_id'] + '.json')}")
        report = store.result(result["result_id"]) or {}
        if report.get("message"):
            typer.echo(report["message"])
        for failure in report.get("failures", []):
            typer.echo(f"失败 {failure['symbol']}：{failure['reason']}", err=True)
        warned = sum(bool(item.get("warnings")) for item in report.get("symbols", []) if isinstance(item, dict))
        if warned:
            typer.echo(f"{warned} 个标的有来源或覆盖警告，详见结果 JSON；任务完成不代表分钟历史完整。")
        if report.get("failures"):
            raise typer.Exit(1 if result.get("status") == "failed" else 2)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app()
