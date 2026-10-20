import json

import pytest
from fastapi.testclient import TestClient

from stockregression import api
from stockregression.settings import Settings
from stockregression.storage import Store
from stockregression.watchlist_sync import import_watchlist_snapshot


def write_snapshot(path, *symbols):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "items": [{"symbol": symbol, "name": "快照名称", "created_at": f"2026-10-20T03:00:{index:02d}+00:00"}
                  for index, symbol in enumerate(symbols)],
    }, ensure_ascii=False), encoding="utf-8")


def test_import_preserves_local_metadata_dates_and_adds_new_entries(tmp_path):
    store = Store(tmp_path / "data")
    store.add_watch("159611.SZ", "本地名称")
    store.add_watch("600000.SH", "本地额外自选")
    with store.db() as db:
        before = db.execute("SELECT created_at FROM watchlist WHERE symbol='159611.SZ'").fetchone()[0]
    path = tmp_path / "watchlist.json"
    write_snapshot(path, "159611.SZ", "515330.SH", "512880.SH")

    assert import_watchlist_snapshot(store, path) == 2
    assert import_watchlist_snapshot(store, path) == 0
    assert {item["symbol"] for item in store.watches()} == {"159611.SZ", "600000.SH", "515330.SH", "512880.SH"}
    assert store.instrument("159611.SZ")["name"] == "本地名称"
    with store.db() as db:
        dates = dict(db.execute("SELECT symbol,created_at FROM watchlist"))
    assert dates["159611.SZ"] == before
    assert dates["515330.SH"] == "2026-10-20T03:00:01+00:00"


def test_new_snapshot_and_restart_respect_local_removals(tmp_path):
    root = tmp_path / "data"
    store = Store(root)
    path = tmp_path / "watchlist.json"
    write_snapshot(path, "159611.SZ", "515330.SH")
    assert import_watchlist_snapshot(store, path) == 2
    with store.db() as db:
        db.execute("DELETE FROM watchlist WHERE symbol='515330.SH'")

    store = Store(root)
    assert import_watchlist_snapshot(store, path) == 0
    write_snapshot(path, "159611.SZ", "515330.SH", "512880.SH")
    assert import_watchlist_snapshot(store, path) == 1
    assert {item["symbol"] for item in store.watches()} == {"159611.SZ", "512880.SH"}
    write_snapshot(path, "159611.SZ")
    assert import_watchlist_snapshot(store, path) == 0
    assert {item["symbol"] for item in store.watches()} == {"159611.SZ", "512880.SH"}
    write_snapshot(path, "159611.SZ", "515330.SH")
    assert import_watchlist_snapshot(store, path) == 1


@pytest.mark.parametrize("invalid", [
    '{',
    '{"schema_version": 2, "items": []}',
    '{"schema_version": 1, "items": [{"symbol": "515330.SH"}, {"symbol": "invalid"}]}',
])
def test_invalid_snapshot_leaves_database_unchanged(tmp_path, invalid):
    store = Store(tmp_path / "data")
    store.add_watch("159611.SZ")
    path = tmp_path / "watchlist.json"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError):
        import_watchlist_snapshot(store, path)
    assert [item["symbol"] for item in store.watches()] == ["159611.SZ"]
    assert store.state("watchlist_snapshot:" + str(path.resolve())) is None


def test_missing_snapshot_and_duplicate_symbols(tmp_path):
    store = Store(tmp_path / "data")
    path = tmp_path / "watchlist.json"
    assert import_watchlist_snapshot(store, path) == 0
    write_snapshot(path, "515330.SH", "515330.SH")
    assert import_watchlist_snapshot(store, path) == 1
    assert len(store.watches()) == 1


def test_app_imports_on_start_and_refresh_and_survives_bad_snapshot(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(api, "load_settings", lambda: Settings(data_dir=str(tmp_path / "data"), package_dir=None))
    path = tmp_path / "config" / "watchlist.json"
    write_snapshot(path, "159611.SZ")
    app = api.create_app(start_jobs=False)
    assert [item["symbol"] for item in app.state.store.watches()] == ["159611.SZ"]
    with TestClient(app) as client:
        write_snapshot(path, "159611.SZ", "515330.SH")
        assert {item["symbol"] for item in client.get("/api/v1/watchlist").json()} == {"159611.SZ", "515330.SH"}
        assert client.delete("/api/v1/watchlist/515330.SH").status_code == 200
        assert [item["symbol"] for item in client.get("/api/v1/watchlist").json()] == ["159611.SZ"]
        path.write_text("{", encoding="utf-8")
        response = client.get("/api/v1/watchlist")
        assert response.status_code == 200
        assert [item["symbol"] for item in response.json()] == ["159611.SZ"]
        assert "自选快照导入失败" in caplog.text


def test_explicit_isolated_data_directory_does_not_import_project_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    write_snapshot(tmp_path / "config" / "watchlist.json", "159611.SZ")
    app = api.create_app(str(tmp_path / "isolated"), start_jobs=False)
    assert app.state.store.watches() == []
