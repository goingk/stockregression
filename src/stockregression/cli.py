from __future__ import annotations

import multiprocessing
import webbrowser
from pathlib import Path

import typer

from .jobs import execute_job
from .schedule import configure_schedule
from .settings import load_settings, save_settings
from .storage import Store

app = typer.Typer(help="本地 A 股分钟级网格研究台", no_args_is_help=True)
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
def sync(mode: str = "daily", symbol: list[str] = typer.Option(None)):
    """无网页时独立补数；mode 为 history/daily/repair。"""
    if mode not in ("history", "daily", "repair"):
        raise typer.BadParameter("mode 为 history/daily/repair")
    _run("sync", {"mode": mode, "symbols": symbol or []})


@app.command()
def demo(symbol: str = "000001.SZ", days: int = 780):
    """生成明确标记的演示数据，含短缺口与按天降级示例。"""
    _run("demo", {"symbol": symbol, "days": days})


@app.command("import-data")
def import_data(symbol: str, daily_path: Path, minute_path: Path | None = None,
                actions_path: Path | None = None, calendar_path: Path | None = None,
                label_convention: str = "auto"):
    """导入本地未复权行情与公司行动 CSV/Parquet。"""
    payload = {"symbol": symbol, "daily_path": str(daily_path), "label_convention": label_convention}
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
    execute_job(str(store.root), job_id)
    result = store.job(job_id)
    typer.echo(result.get("message", ""))
    if result.get("error"):
        typer.echo(result["error"], err=True)
        raise typer.Exit(1)
    if result.get("result_id"):
        typer.echo(f"结果：{store.root / 'results' / (result['result_id'] + '.json')}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app()
