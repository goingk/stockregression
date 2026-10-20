from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stockregression import settings as settings_module
from stockregression.api import create_app
from stockregression.configuration import (
    configuration_schema,
    current_configuration,
    patch_configuration,
    validate_configuration_patch,
)
from stockregression.settings import CONFIG_SCHEMA_VERSION, load_settings


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    target = tmp_path / "config" / "application.json"
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", target)
    monkeypatch.setattr(settings_module, "LEGACY_SETTINGS_FILE", tmp_path / ".local" / "settings.json")
    return target


def old_flat_settings() -> dict:
    return {
        "port": 8123,
        "data_dir": "data",
        "package_dir": "data/dataset/etf",
        "fee_defaults": {"commission_rate": .0007, "min_commission": .2},
        "workbench_defaults": {
            "grid": {"grid_mode": "geometric", "buy_step": .015, "sell_step": .02},
            "position": {"initial_shares": 10000, "min_shares": 0, "max_shares": 30000, "buy_shares": 1200, "sell_shares": 1000},
            "research": {
                "protocol_version": 4, "strategy_version": 1, "ranking_mode": "fixed_mean",
                "selection_target": "fixed", "max_drawdown": .3, "top_candidates": 100,
                "granularity": {
                    "buy_step": .0001, "sell_step": .0001, "sell_shares": 100,
                    "buy_shares": 100, "min_shares": 500, "max_shares": 500,
                },
                "ranges": {
                    "buy_step": {"low": .005, "high": .025, "step": .002},
                    "sell_step": {"low": .005, "high": .025, "step": .002},
                    "buy_shares": {"low": 800, "high": 1200, "step": 100},
                    "sell_shares": {"low": 800, "high": 1200, "step": 100},
                    "initial_shares": {"low": 10000, "high": 10000, "step": 1000},
                    "min_shares": {"low": 0, "high": 2000, "step": 1000},
                    "max_shares": {"low": 20000, "high": 22000, "step": 2000},
                },
            },
        },
    }


def test_flat_migration_materialises_one_file_and_archives_exact_legacy_defaults(config_file):
    legacy_file = settings_module.LEGACY_SETTINGS_FILE
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text(json.dumps(old_flat_settings()), encoding="utf-8")

    loaded = load_settings()
    document = current_configuration()
    assert config_file.exists()
    assert document["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert document["application"]["port"] == 8123
    assert document["backtest"]["fees"] == {"commission_rate": .0007, "min_commission": .2}
    archived = document["research"]["legacy"]["research_defaults"]
    assert archived["strategy_version"] == 1
    assert archived["protocol_version"] == 4
    assert len(archived["ranges"]) == 7
    assert archived["ranges"]["max_shares"] == {"low": 20000.0, "high": 22000.0, "step": 2000.0}
    assert document["research"]["defaults"]["strategy_version"] == 4
    assert loaded.workbench_defaults.research.strategy_version == 4
    assert document["market_data"]["network_timeout_seconds"] == 15


def test_patch_round_trips_zero_false_and_null_with_a_stable_revision(config_file):
    load_settings()
    document = current_configuration()
    updated = patch_configuration(
        {
            "backtest": {"fees": {"commission_rate": 0}},
            "schedule": {"enabled": False},
            "research": {"defaults": {"eta_hard_cap": None, "seed": 0}},
        },
        base_revision=document["config_revision"],
    )
    reread, revision = settings_module.read_configuration_document(config_file)
    assert reread == updated
    assert revision == updated["config_revision"]
    assert reread["backtest"]["fees"]["commission_rate"] == 0
    assert reread["schedule"]["enabled"] is False
    assert reread["research"]["defaults"]["eta_hard_cap"] is None
    assert reread["research"]["defaults"]["seed"] == 0


def test_invalid_patch_reports_path_and_does_not_touch_the_file(config_file):
    load_settings()
    before = config_file.read_bytes()
    document = current_configuration()
    with pytest.raises(ValueError, match="granularity|细粒度"):
        validate_configuration_patch(
            {"research": {"defaults": {"granularity": {"buy_step": 0}}}}, base_revision=document["config_revision"],
        )
    with pytest.raises(ValueError, match="granularity|细粒度"):
        patch_configuration(
            {"research": {"defaults": {"granularity": {"buy_step": 0}}}},
            base_revision=document["config_revision"],
        )
    assert config_file.read_bytes() == before


def test_invalid_manually_edited_file_is_not_replaced_with_defaults(config_file):
    config_file.parent.mkdir(parents=True)
    invalid = '{"config_schema_version": 4, "application": {"unknown_field": true}}'
    config_file.write_text(invalid, encoding="utf-8")
    before = config_file.read_bytes()
    with pytest.raises(ValueError):
        load_settings()
    assert config_file.read_bytes() == before


def test_configuration_api_conflict_and_legacy_write_gate(config_file, tmp_path):
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    with TestClient(app) as client:
        first = client.get("/api/v1/configuration").json()
        revision = first["revision"]
        saved = client.patch("/api/v1/configuration", json={
            "base_revision": revision,
            "patch": {"research": {"defaults": {"alpha": .35}}},
        })
        assert saved.status_code == 200, saved.text
        assert saved.json()["config"]["research"]["defaults"]["alpha"] == .35

        conflict = client.patch("/api/v1/configuration", json={
            "base_revision": revision,
            "patch": {"research": {"defaults": {"beta": 1.5}}},
        })
        assert conflict.status_code == 409
        legacy = client.put("/api/v1/settings", json={"fee_defaults": {"commission_rate": 0}})
        assert legacy.status_code == 428
        schema = client.get("/api/v1/configuration/schema")
        assert schema.status_code == 200
        assert "market_data" in schema.json()["json_schema"]["properties"]


def test_configuration_schema_exposes_a_complete_frontend_field_catalogue(config_file):
    load_settings()
    schema = configuration_schema()
    assert isinstance(schema["fields"], list)
    fields = {field["path"]: field for field in schema["fields"]}

    alpha = fields["research.defaults.alpha"]
    assert alpha["group"] == "research"
    assert alpha["default"] == pytest.approx(0.3)
    assert alpha["constraints"] == {"minimum": 0, "maximum": 1}
    assert alpha["unit"] == "权重或比例；界面显示百分比"
    assert alpha["method_impact"] is True

    training_max = fields["research.defaults.training_max_days"]
    assert training_max["default"] == 500
    assert training_max["unit"] == "交易日"
    assert training_max["effect"] == "下次任务提交"
    assert training_max["method_impact"] is True

    assert fields["market_data.network_timeout_seconds"]["default"] == pytest.approx(15)
    assert fields["runtime.workspace_mb"]["default"] == 8192
    assert fields["ui.polling_interval_ms"]["unit"] == "毫秒"
    assert fields["ui.polling_interval_ms"]["method_impact"] is False


def test_submitted_task_keeps_its_configuration_snapshot_after_file_edits(config_file, tmp_path):
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    with TestClient(app) as client:
        revision = client.get("/api/v1/configuration").json()["revision"]
        submitted = client.post("/api/v1/data/sync", json={
            "scope": "watchlist", "mode": "incremental", "configuration_revision": revision,
        })
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job_id"]
        frozen = client.get(f"/api/v1/jobs/{job_id}").json()["payload"]["configuration_snapshot"]
        assert frozen["active"]["data_dir"] == str((tmp_path / "data").resolve())
        assert frozen["settings"]["data_dir"] == frozen["active"]["data_dir"]
        changed = client.patch("/api/v1/configuration", json={
            "base_revision": revision,
            "patch": {"ui": {"polling_interval_ms": 9000}},
        })
        assert changed.status_code == 200
        after = client.get(f"/api/v1/jobs/{job_id}").json()["payload"]["configuration_snapshot"]
        assert after == frozen
        assert after["document"]["ui"]["polling_interval_ms"] != 9000


def test_stale_submission_revision_is_rejected_without_creating_a_job(config_file, tmp_path):
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    with TestClient(app) as client:
        revision = client.get("/api/v1/configuration").json()["revision"]
        saved = client.patch("/api/v1/configuration", json={
            "base_revision": revision, "patch": {"ui": {"page_size": 80}},
        })
        assert saved.status_code == 200
        stale = client.post("/api/v1/data/sync", json={
            "scope": "watchlist", "mode": "incremental", "configuration_revision": revision,
        })
        assert stale.status_code == 409
        assert stale.json()["detail"]["code"] == "CONFIG_REVISION_CONFLICT"


def test_legacy_partial_market_data_save_preserves_and_synchronises_other_fields(config_file):
    load_settings()
    saved = settings_module.save_settings({
        "market_data": {"daily_batch_size": 77},
        "recent_minute_source": "tdx",
    })
    document = current_configuration()
    assert saved.market_data.daily_batch_size == 77
    assert saved.market_data.network_timeout_seconds == 15
    assert saved.market_data.recent_minute_source == "tdx"
    assert saved.recent_minute_source == "tdx"
    assert document["market_data"]["daily_batch_size"] == 77
    assert document["market_data"]["network_timeout_seconds"] == 15
    assert document["market_data"]["recent_minute_source"] == "tdx"
