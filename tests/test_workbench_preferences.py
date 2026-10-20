import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from stockregression import settings
from stockregression.api import create_app


def test_default_storage_paths_keep_asset_packages_under_data(settings_file):
    loaded = settings.load_settings()

    assert Path(loaded.data_dir) == settings.DEFAULT_DATA_DIR
    assert Path(loaded.package_dir) == settings.DEFAULT_DATASET_DIR / "etf"
    assert Path(loaded.package_dir).parent == settings.DEFAULT_DATASET_DIR
    assert settings.DEFAULT_STOCK_PACKAGE_DIR == settings.DEFAULT_DATASET_DIR / "stock"


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", path)
    monkeypatch.setattr(settings, "LEGACY_SETTINGS_FILE", tmp_path / ".local" / "settings.json")
    return path


def test_workbench_defaults_match_frontend_defaults_and_merge_partial_updates(settings_file):
    defaults = settings.load_settings().workbench_defaults.model_dump()
    assert defaults["grid"] == {
        "grid_mode": "trailing", "buy_step": 0.03, "sell_step": 0.03,
        "lower_price": None, "upper_price": None,
    }
    assert defaults["position"] == {
        "initial_shares": 1000, "min_shares": 0, "max_shares": 2000,
        "buy_shares": 100, "sell_shares": 100,
    }
    assert defaults["dca"] == {"frequency": "monthly", "shares": 100, "initial_shares": 0}
    assert defaults["research"]["mode"] == "both"
    assert defaults["research"]["n_trials"] == 1000
    assert defaults["research"]["protocol_version"] == 4
    assert defaults["research"]["strategy_version"] == 4
    assert defaults["research"]["ranking_mode"] == "fixed_mean"
    assert defaults["research"]["hypercube_edge"] == 3
    assert defaults["research"]["granularity"] == {
        "buy_step": 0.0001, "sell_step": 0.0001,
        "sell_shares": 100.0, "buy_shares": 100.0,
    }
    assert defaults["research"]["local_search_budget"] == 3000
    assert defaults["research"]["selection_target"] == "both"
    assert defaults["research"]["rebalance_days"] == 63
    assert defaults["research"]["bootstrap_reps"] == 10000
    assert defaults["research"]["ranges"] == {
        "buy_step": {"low": 0.005, "high": 0.1, "step": None},
        "sell_step": {"low": 0.005, "high": 0.1, "step": None},
        "buy_shares": {"low": 100.0, "high": 1000.0, "step": None},
        "sell_shares": {"low": 100.0, "high": 1000.0, "step": None},
    }

    settings.save_settings({
        "fee_defaults": {"commission_rate": 0.0008},
        "schedule": {"daily_time": "19:10"},
        "workbench_defaults": {
            "grid": {"buy_step": 0.05},
            "position": {"initial_shares": 1200},
            "research": {
                "n_trials": 75,
                "ranges": {"buy_shares": {"high": 2000}},
            },
        },
    })
    settings.save_settings({
        "workbench_defaults": {
            "research": {"ranges": {"buy_step": {"low": 0.01, "high": 0.2}}},
        },
    })

    saved = settings.load_settings()
    assert saved.workbench_defaults.grid.buy_step == 0.05
    assert saved.workbench_defaults.grid.sell_step == 0.03
    assert saved.workbench_defaults.position.initial_shares == 1200
    assert saved.workbench_defaults.position.max_shares == 2000
    assert saved.workbench_defaults.research.n_trials == 75
    assert saved.workbench_defaults.research.ranges["buy_shares"].model_dump() == {
        "low": 100.0, "high": 2000.0, "step": None,
    }
    assert saved.workbench_defaults.research.ranges["sell_step"].model_dump() == {
        "low": 0.005, "high": 0.1, "step": None,
    }
    assert saved.fee_defaults.commission_rate == 0.0008
    assert saved.fee_defaults.min_commission == 0.1
    assert saved.schedule.daily_time == "19:10"


def test_old_settings_gain_workbench_defaults_without_losing_existing_values(settings_file):
    settings_file.write_text(json.dumps({
        "port": 8001,
        "fee_defaults": {"commission_rate": 0.0007, "min_commission": 0.2},
        "schedule": {"enabled": True, "daily_time": "20:00", "weekly_day": "SAT", "weekly_time": "10:00"},
    }), encoding="utf-8")

    loaded = settings.load_settings()
    assert loaded.port == 8001
    assert loaded.fee_defaults.commission_rate == 0.0007
    assert loaded.schedule.enabled is True
    assert loaded.workbench_defaults.grid.grid_mode == "trailing"

    settings.save_settings({"workbench_defaults": {"dca": {"frequency": "weekly"}}})
    saved = settings.load_settings()
    assert saved.port == 8001
    assert saved.fee_defaults.model_dump() == {"commission_rate": 0.0007, "min_commission": 0.2}
    assert saved.schedule.daily_time == "20:00"
    assert saved.workbench_defaults.dca.frequency == "weekly"


@pytest.mark.parametrize("patch", [
    {"grid": {"buy_step": 0}},
    {"grid": {"lower_price": 12, "upper_price": 10}},
    {"position": {"initial_shares": 210}},
    {"position": {"min_shares": 1500}},
    {"position": {"initial_shares": 3000}},
    {"dca": {"shares": 0}},
    {"dca": {"initial_shares": -100}},
    {"research": {"n_trials": 0}},
    {"research": {"top_candidates": 101}},
    {"research": {"max_drawdown": 0}},
    {"research": {"min_history_days": 19}},
    {"research": {"min_minute_ratio": 1.1}},
    {"research": {"ranges": {"buy_shares": {"high": 99}}}},
    {"research": {"ranges": {"buy_step": {"low": 0.2, "high": 0.1}}}},
    {"research": {"granularity": {"buy_step": 0}}},
    {"research": {"ranges": {"unknown": {"low": 0, "high": 1, "step": 1}}}},
])
def test_invalid_workbench_preferences_do_not_replace_saved_settings(settings_file, patch):
    settings.save_settings({
        "fee_defaults": {"commission_rate": 0.0009, "min_commission": 0.3},
        "workbench_defaults": {"grid": {"buy_step": 0.04}},
    })
    original = settings_file.read_bytes()

    with pytest.raises(ValidationError):
        settings.save_settings({"workbench_defaults": patch})

    assert settings_file.read_bytes() == original
    restored = settings.load_settings()
    assert restored.fee_defaults.commission_rate == 0.0009
    assert restored.workbench_defaults.grid.buy_step == 0.04


def test_workbench_preferences_allow_valid_zero_values(settings_file):
    saved = settings.save_settings({
        "workbench_defaults": {
            "position": {"initial_shares": 0, "min_shares": 0, "max_shares": 0},
            "dca": {"initial_shares": 0},
            "research": {
                "min_trades": 0,
                "min_minute_ratio": 0,
                "ranges": {
                    "buy_step": {"low": 0.005, "high": 0.005},
                    "sell_step": {"low": 0.005, "high": 0.005},
                    "buy_shares": {"low": 100, "high": 100},
                    "sell_shares": {"low": 100, "high": 100},
                },
            },
        },
    })

    assert saved.workbench_defaults.position.initial_shares == 0
    assert saved.workbench_defaults.dca.initial_shares == 0
    assert saved.workbench_defaults.research.min_trades == 0
    assert saved.workbench_defaults.research.min_minute_ratio == 0
    assert saved.workbench_defaults.research.ranges["buy_shares"].low == 100


def test_settings_api_merges_workbench_defaults_and_keeps_fee_defaults(settings_file, tmp_path):
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    with TestClient(app) as client:
        initial = client.get("/api/v1/settings").json()
        assert initial["workbench_defaults"]["grid"]["grid_mode"] == "trailing"
        assert len(initial["workbench_defaults"]["research"]["ranges"]) == 4

        response = client.put("/api/v1/settings", json={
            "base_revision": initial["config_revision"],
            "fee_defaults": {"commission_rate": 0},
            "workbench_defaults": {
                "grid": {"grid_mode": "geometric", "buy_step": 0.05},
                "research": {"n_trials": 333},
            },
        })
        assert response.status_code == 200
        assert response.json()["workbench_defaults"]["grid"]["grid_mode"] == "geometric"
        assert response.json()["workbench_defaults"]["grid"]["sell_step"] == 0.03
        assert response.json()["workbench_defaults"]["research"]["n_trials"] == 333
        assert response.json()["fee_defaults"] == {"commission_rate": 0, "min_commission": 0.1}

        invalid = client.put("/api/v1/settings", json={
            "base_revision": response.json()["config_revision"],
            "workbench_defaults": {"position": {"initial_shares": 5000}},
        })
        assert invalid.status_code == 422
        after = client.get("/api/v1/settings").json()
        assert after["workbench_defaults"]["position"]["initial_shares"] == 1000
        assert after["fee_defaults"]["commission_rate"] == 0


def test_recent_symbols_reads_formal_jobs_deduplicates_and_excludes_other_jobs(settings_file, tmp_path):
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    store = app.state.store

    def add_job(kind, payload, created_at, status="queued"):
        job_id = store.new_job(kind, payload)
        with store.db() as db:
            db.execute("UPDATE jobs SET created_at=?,status=? WHERE id=?", (created_at, status, job_id))
        return job_id

    timestamps = {
        "backtest": "2026-01-01T00:01:00+00:00",
        "optimization": "2026-01-02T00:01:00+00:00",
        "optimize": "2026-01-03T00:01:00+00:00",
        "batch": "2026-01-04T00:01:00+00:00",
        "duplicate": "2026-01-05T00:01:00+00:00",
        "sync": "2026-01-09T00:01:00+00:00",
        "preview": "2026-01-10T00:01:00+00:00",
    }
    add_job("backtest", {"config": {"symbol": "600000.SH"}}, timestamps["backtest"])
    add_job("optimization", {"config": {"base": {"symbol": "000001.SZ"}}}, timestamps["optimization"])
    add_job("optimize", {"base": {"symbol": "510300.SH"}}, timestamps["optimize"], status="failed")
    add_job("batch_backtest", {"configs": [{"symbol": "600000.SH"}, {"symbol": "159915.SZ"}]}, timestamps["batch"], status="cancelled")
    add_job("backtest", {"config": {"symbol": "000001.SZ"}}, timestamps["duplicate"])
    add_job("sync", {"symbols": ["510050.SH"]}, timestamps["sync"])
    add_job("preview", {"config": {"symbol": "512000.SH"}}, timestamps["preview"])
    add_job("instruments", {}, "2026-01-11T00:01:00+00:00")
    store.put_instrument("000001.SZ", "平安银行")

    with TestClient(app) as client:
        response = client.get("/api/v1/workbench/recent-symbols")

    assert response.status_code == 200
    items = response.json()
    assert [item["symbol"] for item in items] == ["000001.SZ", "600000.SH", "159915.SZ", "510300.SH"]
    assert items[0]["name"] == "平安银行"
    assert items[0]["exchange"] == "SZ"
    assert items[0]["asset_type"] == "stock"
    assert items[0]["board"] == "main"
    assert items[0]["last_submitted_at"] == timestamps["duplicate"]
    assert items[1]["last_submitted_at"] == timestamps["batch"]
    assert items[2]["name"] == "159915.SZ"
    assert items[2]["exchange"] == "SZ"
    assert items[2]["asset_type"] == "etf"
    assert items[2]["board"] == "etf"
    assert items[3]["last_submitted_at"] == timestamps["optimize"]
    assert all(item["symbol"] not in {"510050.SH", "512000.SH"} for item in items)


def test_recent_symbols_is_limited_to_ten(settings_file, tmp_path):
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    store = app.state.store
    with store.db() as db:
        for index in range(12):
            code = 600000 + index
            db.execute(
                "INSERT INTO jobs(id,kind,status,progress,message,payload,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    f"job-{index:02}", "backtest", "failed", 0, "done",
                    json.dumps({"config": {"symbol": f"{code}.SH"}}),
                    f"2026-01-{index + 1:02}T00:00:00+00:00",
                    f"2026-01-{index + 1:02}T00:00:00+00:00",
                ),
            )

    with TestClient(app) as client:
        items = client.get("/api/v1/workbench/recent-symbols").json()

    assert len(items) == 10
    assert items[0]["symbol"] == "600011.SH"
    assert items[-1]["symbol"] == "600002.SH"
