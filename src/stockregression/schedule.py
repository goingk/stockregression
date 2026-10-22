from __future__ import annotations

import base64
import os
import re
import subprocess
import sys

from filelock import FileLock

from .settings import PROJECT_ROOT, load_settings

TASK_NAMES = ["StockRegression-Daily", "StockRegression-Weekly"]


def _powershell_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def configure_schedule(install: bool) -> dict:
    if os.name != "nt":
        raise ValueError("Windows 计划任务仅可在 Windows 注册；其他系统请使用 CLI sync")
    settings = load_settings()
    for value in [settings.schedule.daily_time, settings.schedule.weekly_time]:
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("计划时间必须为 HH:MM")
    weekday = {"MON": "Monday", "TUE": "Tuesday", "WED": "Wednesday", "THU": "Thursday", "FRI": "Friday", "SAT": "Saturday", "SUN": "Sunday"}.get(settings.schedule.weekly_day.upper())
    if not weekday:
        raise ValueError("无效的周计划星期")
    commands = ["$ErrorActionPreference = 'Stop'"]
    if install:
        commands += [
            "$taskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24) -MultipleInstances IgnoreNew -Hidden",
            "$taskPrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited",
        ]
        for name, mode, schedule_time, days in [
            (TASK_NAMES[0], "daily", settings.schedule.daily_time, "Monday,Tuesday,Wednesday,Thursday,Friday"),
            (TASK_NAMES[1], "repair", settings.schedule.weekly_time, weekday),
        ]:
            sync_mode = "incremental" if mode == "daily" else "repair"
            frequencies = "--frequency 1day --frequency 1min" if mode == "daily" else "--frequency 1day"
            arguments = f"-m stockregression.cli data sync {frequencies} --mode {sync_mode} --scope market --resume"
            commands += [
                "$taskAction = New-ScheduledTaskAction -Execute " + _powershell_literal(sys.executable) + " -Argument " + _powershell_literal(arguments) + " -WorkingDirectory " + _powershell_literal(PROJECT_ROOT),
                "$taskTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek " + days + " -At " + _powershell_literal(schedule_time),
                "Register-ScheduledTask -TaskName " + _powershell_literal(name) + " -Action $taskAction -Trigger $taskTrigger -Settings $taskSettings -Principal $taskPrincipal -Force | Out-Null",
            ]
    else:
        for name in TASK_NAMES:
            commands.append("if (Get-ScheduledTask -TaskName " + _powershell_literal(name) + " -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName " + _powershell_literal(name) + " -Confirm:$false }")
    encoded = base64.b64encode("\n".join(commands).encode("utf-16le")).decode("ascii")
    lock_dir = PROJECT_ROOT / ".local"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_dir / "schedule.lock"), timeout=settings.schedule.lock_timeout_seconds):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=settings.schedule.subprocess_timeout_seconds,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    if completed.returncode:
        raise RuntimeError("计划任务注册失败：" + (completed.stderr or completed.stdout).strip())
    from .configuration import current_configuration, patch_configuration

    document = current_configuration()
    patch_configuration(
        {"schedule": {"enabled": install}},
        base_revision=document["config_revision"],
    )
    return {"enabled": install, "task_names": TASK_NAMES, "message": "已注册每日更新和周末核对；以当前登录用户运行，漏跑后补执行" if install else "计划任务已移除"}
