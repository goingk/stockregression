from __future__ import annotations

import json
import inspect
from pathlib import Path

import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.models import MarketDataError
from stockregression.market_data.package_layout import PackageLayout
from stockregression.market_data.sync import MarketDataSyncService
from stockregression.market_data import repository as repository_module
from stockregression.settings import Settings
from stockregression.storage import Store


RUN_KEY = "publication-fence-test"
SYMBOL = "600000.SH"


def _job(store: Store, job_id: str, claim_token: str) -> None:
    created = store.new_job("market_sync", {})
    with store.db() as db:
        db.execute("UPDATE jobs SET id=? WHERE id=?", (job_id, created))
    assert store.claim_job(job_id, claim_token)


def _sync_service(store: Store, service: MarketDataService, job_id: str, claim_token: str):
    sync = MarketDataSyncService(
        service, job_id=job_id, job_claim_token=claim_token, cutoff_date="2026-10-23",
    )
    sync._run_key = RUN_KEY
    return sync


def test_competing_jobs_cannot_claim_same_symbol_and_stale_owner_cannot_publish(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "catalog"), package_dir=str(tmp_path / "package"))
    store = Store(tmp_path / "catalog", settings=settings)
    service = MarketDataService(store, settings=settings)
    _job(store, "sync-a", "claim-a")
    _job(store, "sync-b", "claim-b")
    first = _sync_service(store, service, "sync-a", "claim-a")
    second = _sync_service(store, service, "sync-b", "claim-b")

    assert first._claim(RUN_KEY, SYMBOL, "history", "2020-01-01", "2026-10-23")
    assert store.claim_sync_item(
        RUN_KEY, SYMBOL, job_id="sync-b", job_claim_token="claim-b", attempt_token="attempt-b",
    ) is None

    assert store.update_job("sync-a", status="failed")
    assert second._claim(RUN_KEY, SYMBOL, "history", "2020-01-01", "2026-10-23")
    checkpoint = store.sync_item(RUN_KEY, SYMBOL)
    assert checkpoint["last_attempt_job_id"] == "sync-b"
    assert checkpoint["daily_through"] is None

    with pytest.raises(InterruptedError, match="所有权已失效"):
        with first._publication_guard(SYMBOL):
            pytest.fail("stale ownership must be rejected before publication")
    with pytest.raises(InterruptedError, match="所有权已失效"):
        first._finish(RUN_KEY, SYMBOL, status="completed", daily_through="2026-10-23")

    assert second._finish(RUN_KEY, SYMBOL, status="completed", daily_through="2026-10-23") is None
    assert store.sync_item(RUN_KEY, SYMBOL)["daily_through"] == "2026-10-23"


def test_cancelled_claim_cannot_finish_or_write_checkpoint_values(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "catalog"), package_dir=str(tmp_path / "package"))
    store = Store(tmp_path / "catalog", settings=settings)
    service = MarketDataService(store, settings=settings)
    _job(store, "sync-cancel", "claim-cancel")
    sync = _sync_service(store, service, "sync-cancel", "claim-cancel")
    assert sync._claim(RUN_KEY, SYMBOL, "incremental", "2026-10-20", "2026-10-23")

    assert store.update_job("sync-cancel", cancel_requested=True)
    with pytest.raises(InterruptedError, match="已取消或同步检查点所有权已失效"):
        with sync._publication_guard(SYMBOL):
            pytest.fail("a cancelled claim must not enter the publication section")
    with pytest.raises(InterruptedError, match="已取消或同步检查点所有权已失效"):
        sync._finish(RUN_KEY, SYMBOL, status="completed", daily_through="2026-10-23")

    checkpoint = store.sync_item(RUN_KEY, SYMBOL)
    assert checkpoint["status"] == "running"
    assert checkpoint["daily_through"] is None


@pytest.mark.parametrize("invalidate", ["owner", "cancel"])
def test_owner_loss_or_cancel_at_atomic_head_switch_keeps_previous_head(
    tmp_path, monkeypatch, invalidate,
):
    settings = Settings(data_dir=str(tmp_path / "catalog"), package_dir=str(tmp_path / "package"))
    store = Store(tmp_path / "catalog", settings=settings)
    service = MarketDataService(store, settings=settings)
    repository = service.repository
    frame = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-10-22")],
        "open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2],
        "volume": [24000.0], "turnover": [240000.0], "preclose": [10.0],
        "tradestatus": [1], "is_st": [False], "source": ["fixture"],
        "source_verified": [True], "is_synthetic": [False],
    })
    repository.commit_bars(SYMBOL, "1day", frame, source="fixture")
    dataset_key = repository.package.dataset_key(SYMBOL, "1day", "none")
    previous_head = repository.package.manifest()["datasets"][dataset_key]
    assert previous_head["version"] == 1

    _job(store, "sync-fenced", "claim-fenced")
    sync = _sync_service(store, service, "sync-fenced", "claim-fenced")
    assert sync._claim(RUN_KEY, SYMBOL, "repair", "2026-10-22", "2026-10-23")

    package = repository.package
    original_check = package.check_publication
    invalidated = False

    def invalidate_at_head_switch():
        nonlocal invalidated
        caller = inspect.currentframe().f_back
        at_head_switch = (
            caller.f_code.co_name == "publish_dataset"
            and caller.f_lineno > 300
        )
        if at_head_switch and not invalidated:
            invalidated = True
            if invalidate == "owner":
                assert store.update_job("sync-fenced", status="failed", claim_token=None)
            else:
                assert store.update_job("sync-fenced", cancel_requested=True)
        original_check()

    monkeypatch.setattr(package, "check_publication", invalidate_at_head_switch)
    revised = frame.assign(close=[11.2], high=[11.5], turnover=[268800.0])
    with pytest.raises(InterruptedError, match="已取消或同步检查点所有权已失效"):
        with sync._publication_guard(SYMBOL):
            repository.commit_bars(SYMBOL, "1day", revised, source="fixture-revision")

    current_head = package.manifest()["datasets"][dataset_key]
    assert current_head == previous_head, "a failed fence check must leave the published package head unchanged"
    assert invalidated
    with pytest.raises(InterruptedError, match="已取消或同步检查点所有权已失效"):
        sync._finish(RUN_KEY, SYMBOL, status="completed", daily_through="2026-10-23")
    checkpoint = store.sync_item(RUN_KEY, SYMBOL)
    assert checkpoint["status"] == "running"
    assert checkpoint["daily_through"] is None


def test_read_only_service_does_not_write_package_or_recover_existing_journal(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "catalog"), package_dir=str(tmp_path / "package"))
    store = Store(tmp_path / "catalog", settings=settings)
    writer = MarketDataService(store, settings=settings)
    package = writer.repository.package
    _, journal = package.begin_transaction({
        "kind": "dataset", "symbol": SYMBOL, "timeframe": "1day", "version": 1,
    })
    journal_before = journal.read_bytes()
    manifest_before = package.manifest_path.read_bytes()

    readonly = MarketDataService(
        store, package_dir=writer.repository.package_dir, settings=settings, read_only=True,
    )
    assert journal.is_file(), "read-only initialization must leave the pending journal untouched"
    assert journal.read_bytes() == journal_before

    with pytest.raises(MarketDataError, match="不能发布或恢复"):
        readonly.repository.commit_bars(SYMBOL, "1day", pd.DataFrame())

    assert journal.is_file()
    assert journal.read_bytes() == journal_before
    assert package.manifest_path.read_bytes() == manifest_before
    assert json.loads(journal.read_text(encoding="utf-8"))["status"] == "prepared"


def test_read_only_repository_rejects_archive_restore_without_touching_pending_journal(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "catalog"), package_dir=str(tmp_path / "package"))
    store = Store(tmp_path / "catalog", settings=settings)
    writer = MarketDataService(store, settings=settings)
    package = writer.repository.package
    _, journal = package.begin_transaction({
        "kind": "dataset", "symbol": SYMBOL, "timeframe": "1day", "version": 1,
    })
    journal_before = journal.read_bytes()
    readonly = MarketDataService(
        store, package_dir=writer.repository.package_dir, settings=settings, read_only=True,
    )

    with pytest.raises(MarketDataError, match="不能发布或恢复"):
        readonly.repository._restore_published_archives()

    assert journal.read_bytes() == journal_before


def test_nested_repository_snapshot_reenters_package_transaction_lock(tmp_path):
    # Mirrors refresh_snapshot -> Store.put_market_snapshot -> Store.snapshot,
    # where snapshot constructs another repository for the same package.
    from test_sync_resume_integration import seed

    service, days, _ = seed(tmp_path)
    job_id, claim_token = "sync-snapshot", "claim-snapshot"
    _job(service.store, job_id, claim_token)
    sync = _sync_service(service.store, service, job_id, claim_token)
    sync._run_key = RUN_KEY
    assert sync._claim(RUN_KEY, SYMBOL, "incremental", days[0], days[-1])

    with sync._publication_guard(SYMBOL):
        snapshot = service.refresh_snapshot(SYMBOL)
        assert service.store.snapshot(SYMBOL, snapshot["version"]) is not None

    assert snapshot is not None
    assert snapshot["market_data"] is True
    sync._finish(RUN_KEY, SYMBOL, status="completed", daily_through=str(days[-1].date()))


def test_refresh_snapshot_reuses_repository_and_still_checks_generation(tmp_path, monkeypatch):
    from test_sync_resume_integration import seed
    from stockregression.market_data.repository import MarketDataRepository

    service, _, _ = seed(tmp_path)
    def reject_constructor(*_args, **_kwargs):
        raise AssertionError("snapshot refresh must reuse its already reconciled repository")
    monkeypatch.setattr(MarketDataRepository, "__init__", reject_constructor)
    snapshot = service.refresh_snapshot(SYMBOL)
    assert snapshot is not None
    assert service.store.snapshot(SYMBOL, snapshot["version"], repository=service.repository)
    broken = {**snapshot, "generation": "wrong-generation", "version": "wrong-version"}
    assert service.store.put_market_snapshot(broken, SYMBOL, repository=service.repository) is None


def test_archive_recovery_batches_unarchived_dataset_cleanup(tmp_path, monkeypatch):
    from test_sync_resume_integration import seed
    service, _, _ = seed(tmp_path)
    repository = service.repository
    manifest = repository.package.manifest()
    manifest = {**manifest, "datasets": {
        f"{index:06d}.SZ|1day|none": {"version": 1}
        for index in range(100)
    }}
    monkeypatch.setattr(repository.package, "manifest", lambda: manifest)
    original = repository._connect
    calls = []
    def counted():
        calls.append(1)
        return original()
    monkeypatch.setattr(repository, "_connect", counted)
    assert repository._restore_published_archives() == 0
    assert len(calls) == 2


def test_snapshot_keeps_unverified_actions_unverified(tmp_path):
    from test_sync_resume_integration import seed
    service, _, _ = seed(tmp_path)
    service.repository.save_auxiliary(SYMBOL, "actions", [], verified=False)
    snapshot = service.refresh_snapshot(SYMBOL)
    assert snapshot["metadata"]["actions_verified"] is False


def test_publication_fence_is_visible_to_another_layout_for_same_root(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "catalog"), package_dir=str(tmp_path / "package"))
    store = Store(tmp_path / "catalog", settings=settings)
    service = MarketDataService(store, settings=settings)
    _job(store, "sync-cross-layout", "claim-cross-layout")
    sync = _sync_service(store, service, "sync-cross-layout", "claim-cross-layout")
    assert sync._claim(RUN_KEY, SYMBOL, "incremental", "2026-10-20", "2026-10-23")
    nested = PackageLayout(service.repository.package.root, market_settings=settings.market_data)

    invalidated = False

    def assert_attempt_current():
        if invalidated:
            raise InterruptedError("fence propagated")

    with sync.repository.package.publication_fence(assert_attempt_current):
        invalidated = True
        with pytest.raises(InterruptedError, match="fence propagated"):
            nested.check_publication()


def test_read_only_manifest_normalization_does_not_rewrite_manifest(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "catalog"), package_dir=str(tmp_path / "package"))
    store = Store(tmp_path / "catalog", settings=settings)
    service = MarketDataService(store, settings=settings)
    manifest_path = service.repository.package.manifest_path
    manifest_path.write_text(json.dumps({
        "schema_version": 2,
        "package_id": "readonly-fixture",
        "generation": "readonly-generation",
    }), encoding="utf-8")
    before = manifest_path.read_bytes()

    readonly = PackageLayout(service.repository.package.root, market_settings=settings.market_data, read_only=True)
    normalized = readonly.manifest()

    assert normalized["datasets"] == {}
    assert manifest_path.read_bytes() == before


def test_read_only_publish_dataset_does_not_create_immutable_manifest(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "catalog"), package_dir=str(tmp_path / "package"))
    store = Store(tmp_path / "catalog", settings=settings)
    service = MarketDataService(store, settings=settings)
    repository = service.repository
    frame = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-10-22")],
        "open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2],
        "volume": [24000.0], "turnover": [240000.0], "preclose": [10.0],
        "tradestatus": [1], "is_st": [False], "source": ["fixture"],
        "source_verified": [True], "is_synthetic": [False],
    })
    repository.commit_bars(SYMBOL, "1day", frame, source="fixture")
    package_root = repository.package.root
    package_manifest_before = repository.package.manifest_path.read_bytes()
    payload = repository.package.load_dataset_manifest(SYMBOL, "1day", "none")
    payload = {**payload, "version": 2}
    readonly = PackageLayout(package_root, market_settings=settings.market_data, read_only=True)
    next_manifest = readonly.absolute(readonly.dataset_manifest_relative_path(SYMBOL, "1day", "none", 2))

    with pytest.raises(ValueError, match="只读下载上下文"):
        readonly.publish_dataset(payload, expected_version=1)

    assert not next_manifest.exists(), "read-only publication must not leave an immutable manifest behind"
    assert repository.package.manifest_path.read_bytes() == package_manifest_before


@pytest.mark.parametrize("invalidate", ["owner", "cancel"])
@pytest.mark.parametrize("kind", ["calendar", "actions"])
def test_auxiliary_replace_then_revocation_restores_file_and_allows_new_owner(
    tmp_path, monkeypatch, invalidate, kind,
):
    from test_sync_resume_integration import seed

    service, days, _ = seed(tmp_path)
    store = service.store
    repository = service.repository
    old_job, old_claim = "sync-aux-old", "claim-aux-old"
    _job(store, old_job, old_claim)
    old_sync = _sync_service(store, service, old_job, old_claim)
    assert old_sync._claim(RUN_KEY, SYMBOL, "incremental", days[0], days[-1])

    relative = repository_module.auxiliary_relative_path(SYMBOL, kind)
    destination = repository.package.absolute(relative)
    existed_before = destination.is_file()
    old_bytes = destination.read_bytes() if existed_before else None
    manifest_before = repository.package.manifest_path.read_bytes()
    replacement = (
        [{"date": "2026-10-23"}] if kind == "calendar"
        else [{"ex_date": "2026-10-23", "cash_dividend": 0.2}]
    )

    original_replace = repository_module.os.replace
    replaced = False

    def replace_then_revoke(source, target):
        nonlocal replaced
        result = original_replace(source, target)
        if Path(target) == destination and not replaced:
            replaced = True
            if invalidate == "owner":
                assert store.update_job(old_job, status="failed", claim_token=None)
            else:
                assert store.update_job(old_job, cancel_requested=True)
        return result

    monkeypatch.setattr(repository_module.os, "replace", replace_then_revoke)
    with pytest.raises(InterruptedError, match="已取消或同步检查点所有权已失效"):
        with old_sync._publication_guard(SYMBOL):
            repository.save_auxiliary(
                SYMBOL, kind, replacement, version="revoked-attempt", verified=True,
            )

    assert replaced, "the cancellation/owner loss must be injected after destination replacement"
    assert repository.package.manifest_path.read_bytes() == manifest_before
    if existed_before:
        assert destination.read_bytes() == old_bytes, "the previous published bytes must be restored"
    else:
        assert not destination.exists(), "a failed first publication must remove its unindexed file"
    with pytest.raises(InterruptedError, match="已取消或同步检查点所有权已失效"):
        old_sync._finish(RUN_KEY, SYMBOL, status="completed", daily_through=str(days[-1].date()))

    new_job, new_claim = "sync-aux-new", "claim-aux-new"
    _job(store, new_job, new_claim)
    new_sync = _sync_service(store, service, new_job, new_claim)
    assert new_sync._claim(RUN_KEY, SYMBOL, "incremental", days[0], days[-1])
    with new_sync._publication_guard(SYMBOL):
        repository.save_auxiliary(
            SYMBOL, kind, replacement, version="new-attempt", verified=True,
        )
        new_sync._finish(RUN_KEY, SYMBOL, status="completed", daily_through=str(days[-1].date()))

    head = repository.package.manifest()["auxiliary"][relative]
    assert head["version"] == "new-attempt"
    assert store.sync_item(RUN_KEY, SYMBOL)["status"] == "completed"
