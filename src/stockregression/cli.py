from __future__ import annotations

import multiprocessing
import webbrowser
from pathlib import Path
import shutil

import typer

from .jobs import execute_job
from .schedule import configure_schedule
from .settings import load_settings, save_settings
from .storage import Store

app = typer.Typer(help="本地 A 股及 ETF 行情与网格研究台", no_args_is_help=True)
schedule_app = typer.Typer(help="安装、移除或查看 Windows 补数计划")
app.add_typer(schedule_app, name="schedule")


@app.command()
def serve(port: int = 8000, open_browser: bool = False):
    """启动本地网页和独立计算进程。"""
    import uvicorn
    from .api import create_app
    if open_browser:
        import threading
        timer = threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}"))
        timer.daemon = True
        timer.start()
    uvicorn.run(create_app(), host="127.0.0.1", port=port)


@app.command()
def sync(mode: str = "daily", symbol: list[str] = typer.Option(None), scope: str | None = None, resume: bool = False, retry_failed: bool = False, workers: int = 2, dry_run: bool = False):
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
    typer.echo("首次全市场历史分钟预计需下载数十 GB，缓存、分区及增量版本还需额外空间；实际覆盖与缺口见同步报告。" if scope == "market" else "仅更新指定范围")
    if dry_run:
        from .providers import estimate_history_bytes, fetch_instruments
        from .models import normalize_symbol
        if scope == "market":
            try:
                records = fetch_instruments(settings.hf_revision)
                symbols = sorted({item["symbol"] for item in records})
                typer.echo(f"目录目标：{len(symbols)} 个（股票 {sum(not code.startswith(('1', '5')) for code in symbols)}，ETF {sum(code.startswith(('1', '5')) for code in symbols)}）；不自动加入自选")
                for warning in sorted({warning for item in records for warning in item.get("catalog_warnings", [])}):
                    typer.echo(warning, err=True)
            except Exception as exc:
                raise typer.BadParameter(f"无法预检完整证券清单：{exc}") from exc
        else:
            symbols = [normalize_symbol(code) for code in symbol] if symbol else [item["symbol"] for item in Store(root).watches()]
            typer.echo(f"目标：{len(set(symbols))} 个标的")
        if mode == "history" and symbols:
            try:
                typer.echo(f"固定版本历史分钟源：{estimate_history_bytes(symbols, settings) / 1024 ** 3:.2f} GiB（不含额外来源、缓存和分区开销）")
            except Exception as exc:
                typer.echo(f"精确下载量暂不可得：{exc}", err=True)
        typer.echo("预检结束，未下载行情或创建同步任务。目录元数据可能写入来源缓存。")
        return
    _run("sync", {"mode": mode, "symbols": symbol or [], "scope": scope, "resume": resume, "retry_failed": retry_failed, "workers": workers})


@app.command()
def demo(symbol: str = "000001.SZ", days: int = 780):
    """生成明确标记的演示数据，含短缺口与按天降级示例。"""
    _run("demo", {"symbol": symbol, "days": days})


@app.command("import-data")
def import_data(symbol: str, daily_path: Path, minute_path: Path | None = None,
                actions_path: Path | None = None, calendar_path: Path | None = None,
                label_convention: str = "auto", etf_category: str | None = None):
    """导入本地未复权行情与公司行动 CSV/Parquet。"""
    payload = {"symbol": symbol, "daily_path": str(daily_path), "label_convention": label_convention}
    if etf_category is not None:
        if etf_category not in {"stock", "bond", "crossborder", "commodity", "money"}:
            raise typer.BadParameter("etf-category 为 stock/bond/crossborder/commodity/money")
        payload["etf_category"] = etf_category
    for key, value in [("minute_path", minute_path), ("actions_path", actions_path), ("calendar_path", calendar_path)]:
        if value:
            payload[key] = str(value)
    _run("import", payload)


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
    settings = save_settings({"data_dir": str(path.resolve())})
    typer.echo(settings.data_dir)


def _run(kind: str, payload: dict):
    settings = load_settings()
    store = Store(settings.data_dir)
    job_id = store.new_job(kind, payload)
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
