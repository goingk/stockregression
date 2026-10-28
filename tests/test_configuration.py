from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stockregression import configuration as configuration_module
from stockregression import settings as settings_module
from stockregression.api import create_app
from stockregression.configuration import (
    configuration_schema,
    current_configuration,
    patch_configuration,
    task_configuration_snapshot,
    validate_configuration_patch,
)
from stockregression.settings import CONFIG_SCHEMA_VERSION, load_settings


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    target = tmp_path / "config" / "application.json"
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", target)
    return target


def test_task_snapshot_derives_settings_from_its_document(config_file, tmp_path, monkeypatch):
    load_settings()
    original_read = settings_module.read_configuration_document
    autosaved = False
    saved_revision = None

    def read_then_autosave(path=None):
        nonlocal autosaved, saved_revision
        document, revision = original_read(path)
        if not autosaved:
            autosaved = True
            changed = deepcopy(document)
            changed["ui"]["polling_interval_ms"] = 9876
            changed["research"]["defaults"]["alpha"] = 0.91
            saved_document, _ = settings_module.save_configuration_document(
                changed, base_revision=revision,
            )
            saved_revision = saved_document["config_revision"]
        return document, revision

    monkeypatch.setattr(configuration_module, "read_configuration_document", read_then_autosave)
    active_data_dir = tmp_path / "active-data"
    active_package_dir = tmp_path / "active-package"

    snapshot = task_configuration_snapshot(
        active_data_dir=str(active_data_dir),
        active_package_dir=str(active_package_dir),
        active_host="0.0.0.0",
        active_port=8765,
    )

    assert snapshot["revision"] == snapshot["document"]["config_revision"]
    assert autosaved is True
    assert snapshot["revision"] != saved_revision
    assert snapshot["settings"]["ui"] == snapshot["document"]["ui"]
    assert snapshot["settings"]["workbench_defaults"]["research"]["alpha"] == (
        snapshot["document"]["research"]["defaults"]["alpha"]
    )
    assert snapshot["active"] == {
        "data_dir": str(active_data_dir),
        "package_dir": str(active_package_dir),
        "host": "0.0.0.0",
        "port": 8765,
    }
    assert snapshot["settings"]["data_dir"] == str(active_data_dir)
    assert snapshot["settings"]["package_dir"] == str(active_package_dir)
    assert snapshot["settings"]["host"] == "0.0.0.0"
    assert snapshot["settings"]["port"] == 8765


def test_task_snapshot_resolves_relative_paths_from_the_same_document(config_file):
    config_file.parent.mkdir(parents=True)
    document = settings_module.configuration_document(settings_module.Settings(
        data_dir="task-data",
        package_dir="task-packages",
    ))
    config_file.write_text(json.dumps(document), encoding="utf-8")

    snapshot = task_configuration_snapshot()

    assert snapshot["document"]["paths"] == {
        "data_dir": "task-data",
        "package_dir": "task-packages",
    }
    assert snapshot["settings"]["data_dir"] == str(
        (settings_module.PROJECT_ROOT / "task-data").resolve()
    )
    assert snapshot["settings"]["package_dir"] == str(
        (settings_module.PROJECT_ROOT / "task-packages").resolve()
    )


def test_only_the_current_grouped_configuration_is_materialised(config_file):
    loaded = load_settings()
    document = current_configuration()
    assert config_file.exists()
    assert document["config_schema_version"] == CONFIG_SCHEMA_VERSION == 5
    assert set(document["research"]) == {"defaults"}
    assert loaded.workbench_defaults.research.protocol_version == 5
    assert loaded.workbench_defaults.research.strategy_version == 5


def test_obsolete_flat_or_archived_configuration_is_rejected(config_file):
    config_file.parent.mkdir(parents=True)
    obsolete = {"port": 8123, "workbench_defaults": {"research": {"protocol_version": 4}}}
    config_file.write_text(json.dumps(obsolete), encoding="utf-8")
    before = config_file.read_bytes()
    with pytest.raises(ValueError):
        load_settings()
    assert config_file.read_bytes() == before

    current = settings_module.configuration_document(settings_module.Settings())
    current["research"]["legacy"] = {"protocol_version": 4}
    current.pop("config_revision", None)
    config_file.write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings()


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


def test_config_revision_metadata_is_ignored_in_patches(config_file):
    load_settings()
    document = current_configuration()
    patch = {
        "config_revision": "stale-browser-revision",
        "research": {"defaults": {"alpha": 0.31}},
    }

    validated = validate_configuration_patch(patch, base_revision=document["config_revision"])
    assert validated["research"]["defaults"]["alpha"] == 0.31
    assert validated["config_revision"] != "stale-browser-revision"

    unchanged = validate_configuration_patch(
        {"config_revision": "stale-browser-revision"},
        base_revision=document["config_revision"],
    )
    assert unchanged["config_revision"] == document["config_revision"]

    updated = patch_configuration(patch, base_revision=document["config_revision"])
    assert updated["research"]["defaults"]["alpha"] == 0.31
    assert updated["config_revision"] != "stale-browser-revision"


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


def test_configuration_api_conflict_and_removed_legacy_write_endpoint(config_file, tmp_path):
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
        assert legacy.status_code in {404, 405}
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
