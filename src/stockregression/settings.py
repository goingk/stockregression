from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(os.environ.get("STOCKREGRESSION_HOME", Path(__file__).resolve().parents[2])).resolve()
SETTINGS_FILE = PROJECT_ROOT / ".local" / "settings.json"


class ScheduleSettings(BaseModel):
    enabled: bool = False
    daily_time: str = "18:30"
    weekly_day: str = "SAT"
    weekly_time: str = "10:00"


class Settings(BaseModel):
    data_dir: str = str(PROJECT_ROOT / "data")
    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings)
    host: str = "127.0.0.1"
    port: int = 8000
    max_workers: int = Field(default=2, ge=1, le=8)
    hf_revision: str = "ba589a11534825044fe5a6b84838f50ba8d8d188"


def load_settings() -> Settings:
    values = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) if SETTINGS_FILE.exists() else {}
    if os.environ.get("STOCKREGRESSION_DATA_DIR"):
        values["data_dir"] = os.environ["STOCKREGRESSION_DATA_DIR"]
    settings = Settings.model_validate(values)
    settings.data_dir = str(Path(settings.data_dir).expanduser().resolve())
    return settings


def save_settings(values: dict) -> Settings:
    previous = load_settings().model_dump()
    if "schedule" in values:
        values["schedule"] = {**previous["schedule"], **values["schedule"]}
    settings = Settings.model_validate({**previous, **values})
    settings.data_dir = str(Path(settings.data_dir).expanduser().resolve())
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS_FILE.with_suffix(".writing")
    temporary.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(SETTINGS_FILE)
    return settings
