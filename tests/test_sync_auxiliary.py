from __future__ import annotations

import pandas as pd

from stockregression.market_data import MarketDataService
from stockregression.market_data.sync_auxiliary import persist_sync_auxiliary
from stockregression.settings import Settings
from stockregression.storage import Store


SYMBOL = "510300.SH"


def make_service(tmp_path):
    root = tmp_path / "data"
    settings = Settings(data_dir=str(root), package_dir=str(root / "package"))
    store = Store(root, settings=settings)
    store.put_instrument(SYMBOL, "沪深300ETF", asset_type="etf", etf_type="ETF")
    return MarketDataService(store, settings=settings)


def action(ex_date, cash, *, record_date=None):
    return {
        "ex_date": ex_date,
        "record_date": record_date,
        "cash_per_share": cash,
        "bonus_ratio": 0,
    }


def seed_actions(service, payload, *, verified=False, version="old-actions"):
    service.repository.save_auxiliary(
        SYMBOL, "actions", payload, version=version, verified=verified,
    )


def set_action_state(service, *, checked=(), unverified=(), verified=False):
    service.store.put_instrument(
        SYMBOL, "沪深300ETF", asset_type="etf", etf_type="ETF",
        actions_checked_years=list(checked),
        actions_unverified_years=list(unverified),
        actions_verified=verified,
    )


def test_calendar_subset_of_verified_exchange_calendar_skips_write_and_keeps_version(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    existing = [{"date": day} for day in ("2024-01-02", "2024-01-03", "2024-01-04")]
    service.repository.save_auxiliary(
        SYMBOL, "calendar", existing, version="calendar-v7", verified=True,
    )
    before = service.repository.auxiliary(SYMBOL, "calendar")
    calls = []
    original = service.repository.save_auxiliary

    def tracked(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(service.repository, "save_auxiliary", tracked)
    result = persist_sync_auxiliary(service, SYMBOL, {
        "calendar": pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-04"])}),
    })

    after = service.repository.auxiliary(SYMBOL, "calendar")
    assert calls == []
    assert after["version"] == before["version"] == "calendar-v7"
    assert after["payload"] == before["payload"]
    assert result["calendar_auxiliary_version"] == "calendar-v7"
    assert result["calendar_persisted"] is False


def test_incremental_actions_keep_older_years_and_merge_coverage(tmp_path):
    service = make_service(tmp_path)
    seed_actions(service, [action("2020-06-01", 0.2)], verified=True)
    set_action_state(service, checked=[2020], verified=True)

    result = persist_sync_auxiliary(service, SYMBOL, {
        "actions": pd.DataFrame([action("2026-03-01", 0.3)]),
        "actions_checked_years": [2026],
        "actions_unverified_years": [],
        "actions_verified": True,
    })

    saved = service.repository.auxiliary(SYMBOL, "actions")
    years = {_year for _year in (pd.Timestamp(row["ex_date"]).year for row in saved["payload"])}
    assert years == {2020, 2026}
    assert result["actions_checked_years"] == [2020, 2026]
    assert result["actions_verified"] is True
    assert saved["verified"] is True


def test_successfully_checked_year_replaces_and_can_revise_or_delete_rows(tmp_path):
    service = make_service(tmp_path)
    seed_actions(service, [
        action("2024-02-01", 0.1),
        action("2024-08-01", 0.2),
        action("2023-05-01", 0.3),
        action(None, 0.4),
    ])
    set_action_state(service, checked=[2023, 2024], unverified=[2024])

    persist_sync_auxiliary(service, SYMBOL, {
        # The second 2024 record is intentionally absent: successful year
        # refresh is authoritative for that year and can delete old rows.
        "actions": pd.DataFrame([action("2024-02-01", 0.15)]),
        "actions_checked_years": [2024],
        "actions_unverified_years": [],
        "actions_verified": True,
    })

    rows = service.repository.auxiliary(SYMBOL, "actions")["payload"]
    assert {(row.get("ex_date"), row["cash_per_share"]) for row in rows} == {
        ("2023-05-01", 0.3), ("2024-02-01", 0.15), (None, 0.4),
    }
    state = service.store.instrument(SYMBOL)
    assert state["actions_unverified_years"] == []
    assert state["actions_verified"] is True


def test_failed_year_retains_old_rows_and_stays_unverified(tmp_path):
    service = make_service(tmp_path)
    old = [action("2025-04-01", 0.5)]
    seed_actions(service, old, verified=True)
    set_action_state(service, checked=[2025], verified=True)

    result = persist_sync_auxiliary(service, SYMBOL, {
        "actions": pd.DataFrame([action("2025-04-01", 0.9)]),
        "actions_checked_years": [2025],
        "actions_unverified_years": [2025],
        "actions_verified": False,
    })

    saved = service.repository.auxiliary(SYMBOL, "actions")
    assert saved["payload"] == old
    assert saved["verified"] is False
    assert result["actions_verified"] is False
    assert result["actions_unverified_years"] == [2025]


def test_false_global_result_with_no_failed_year_detail_cannot_replace_rows(tmp_path):
    service = make_service(tmp_path)
    old = [action("2025-04-01", 0.5)]
    seed_actions(service, old, verified=True)
    set_action_state(service, checked=[2025], verified=True)

    result = persist_sync_auxiliary(service, SYMBOL, {
        "actions": pd.DataFrame([action("2025-04-01", 0.9)]),
        "actions_checked_years": [2025],
        "actions_unverified_years": [],
        "actions_verified": False,
    })

    saved = service.repository.auxiliary(SYMBOL, "actions")
    assert saved["payload"] == old
    assert saved["verified"] is False
    assert result["actions_unverified_years"] == [2025]


def test_partial_success_does_not_clear_other_unverified_year_or_promote_global_flag(tmp_path):
    service = make_service(tmp_path)
    seed_actions(service, [
        action("2020-01-01", 0.1),
        action("2024-01-01", 0.4),
    ])
    set_action_state(service, checked=[2020], unverified=[2024], verified=False)

    result = persist_sync_auxiliary(service, SYMBOL, {
        "actions": pd.DataFrame([action("2026-01-01", 0.6)]),
        "actions_checked_years": [2026],
        "actions_unverified_years": [],
        "actions_verified": True,
    })

    saved = service.repository.auxiliary(SYMBOL, "actions")
    years = {pd.Timestamp(row["ex_date"]).year for row in saved["payload"]}
    assert years == {2020, 2024, 2026}
    assert result["actions_unverified_years"] == [2024]
    assert result["actions_verified"] is False
    assert saved["verified"] is False


def test_missing_year_evidence_merges_and_deduplicates_without_replacing(tmp_path):
    service = make_service(tmp_path)
    seed_actions(service, [action("2024-01-01", 0.1)])
    set_action_state(service, checked=[2024], verified=False)

    result = persist_sync_auxiliary(service, SYMBOL, {
        "actions": pd.DataFrame([
            action("2024-01-01", 0.1),
            action("2025-01-01", 0.2),
        ]),
        "actions_verified": True,
    })

    rows = service.repository.auxiliary(SYMBOL, "actions")["payload"]
    assert {row["ex_date"] for row in rows} == {"2024-01-01", "2025-01-01"}
    assert result["actions_checked_years"] == [2024]
    assert result["actions_verified"] is False


def test_identical_actions_payload_and_verified_state_skip_auxiliary_write(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    payload = [action("2024-01-01", 0.1)]
    seed_actions(service, payload, verified=True, version="actions-v4")
    set_action_state(service, checked=[2024], verified=True)
    calls = []
    original = service.repository.save_auxiliary

    def tracked(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(service.repository, "save_auxiliary", tracked)
    result = persist_sync_auxiliary(service, SYMBOL, {
        "actions": pd.DataFrame(payload),
        "actions_checked_years": [2024],
        "actions_unverified_years": [],
        "actions_verified": True,
    })

    assert calls == []
    assert result["actions_persisted"] is False
    assert result["actions_auxiliary_version"] == "actions-v4"
