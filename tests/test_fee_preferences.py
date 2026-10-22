import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from stockregression import settings
from stockregression.configuration import current_configuration, patch_configuration
from stockregression.models import BacktestConfig, FeeConfig


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "config" / "application.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", path)
    return path


def save_fees(fees: dict) -> None:
    document = current_configuration()
    patch_configuration(
        {"backtest": {"fees": fees}},
        base_revision=document["config_revision"],
    )


def test_fee_defaults_match_config_and_survive_partial_updates(settings_file):
    assert settings.load_settings().fee_defaults.model_dump() == {
        "commission_rate": 0.0005, "min_commission": 0.1,
    }
    assert FeeConfig().commission_rate == 0.0005
    assert FeeConfig().min_commission == 0.1
    config = BacktestConfig(symbol="600000.SH", start_date="2024-01-01", schema_version=2)
    assert config.dca.initial_shares == 0

    save_fees({"commission_rate": 0.0008})
    save_fees({"min_commission": 0.25})
    document = current_configuration()
    patch_configuration(
        {"schedule": {"daily_time": "19:10"}},
        base_revision=document["config_revision"],
    )
    restored = settings.load_settings()
    assert restored.fee_defaults.model_dump() == {
        "commission_rate": 0.0008, "min_commission": 0.25,
    }
    assert restored.schedule.daily_time == "19:10"
    assert json.loads(settings_file.read_text(encoding="utf-8"))["backtest"]["fees"] == restored.fee_defaults.model_dump()


def test_zero_fees_are_remembered(settings_file):
    save_fees({"commission_rate": 0, "min_commission": 0})
    assert settings.load_settings().fee_defaults.model_dump() == {
        "commission_rate": 0, "min_commission": 0,
    }


@pytest.mark.parametrize("patch", [
    {"commission_rate": -0.01}, {"commission_rate": 0.06},
    {"commission_rate": float("nan")}, {"min_commission": float("inf")},
    {"min_commission": -1}, {"min_commission": None},
    {"commission_rate": ""},
])
def test_invalid_fee_preferences_do_not_replace_saved_values(settings_file, patch):
    save_fees({"commission_rate": 0.0007, "min_commission": 0.2})
    original = settings_file.read_bytes()
    with pytest.raises(ValidationError):
        save_fees(patch)
    assert settings_file.read_bytes() == original


def test_fee_preferences_settings_api(settings_file, tmp_path):
    from stockregression.api import create_app

    with TestClient(create_app(str(tmp_path / "data"), start_jobs=False)) as client:
        initial = client.get("/api/v1/configuration").json()
        assert initial["config"]["backtest"]["fees"] == {
            "commission_rate": 0.0005, "min_commission": 0.1,
        }
        revision = initial["revision"]
        response = client.patch("/api/v1/configuration", json={
            "base_revision": revision,
            "patch": {"backtest": {"fees": {"commission_rate": 0}}},
        })
        assert response.status_code == 200
        assert response.json()["config"]["backtest"]["fees"] == {"commission_rate": 0, "min_commission": 0.1}
        revision = response.json()["revision"]
        assert client.patch("/api/v1/configuration", json={
            "base_revision": revision,
            "patch": {"backtest": {"fees": {"min_commission": -1}}},
        }).status_code == 422
        assert client.get("/api/v1/configuration").json()["config"]["backtest"]["fees"] == {
            "commission_rate": 0, "min_commission": 0.1,
        }
    with TestClient(create_app(str(tmp_path / "data"), start_jobs=False)) as client:
        assert client.get("/api/v1/configuration").json()["config"]["backtest"]["fees"]["commission_rate"] == 0
