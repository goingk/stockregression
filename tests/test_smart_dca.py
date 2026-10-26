"""Smart DCA execution, causal reference signals and immutable submissions."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from stockregression import engine, settings
from stockregression.api import create_app
from stockregression.dca_reference import attach_dca_reference, freeze_dca_reference, freeze_research_dca_reference, prepare_index_reference
from stockregression.models import BacktestConfig, DcaConfig, OptimizationConfig, ResearchRequestV5
from stockregression.providers import normalize_actions
from stockregression.rules import instrument_terms
from stockregression.smart_dca import smart_dca_signals
from stockregression.storage import Store
from test_share_workflow import fixture, _share_action_market


@pytest.fixture(autouse=True)
def interpreted(monkeypatch):
    # Most cases exercise the ledger without recompiling Numba; parity has a
    # separate test below and existing share-workflow coverage.
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)


def case(mode="price_change", prices=(10, 9, 10.8, 9, 10), reference="index", **overrides):
    config, data = fixture(days=len(prices) - 1)
    for column in ("open", "high", "low", "close", "preclose"):
        data.daily[column] = 10.
        data.bars[column] = 10.
    config = config.model_copy(update={"schema_version": 3, "start_date": data.daily.date.iloc[2].date(),
                                       "max_shares": None, "buy_step": 1., "sell_step": 1.})
    symbol = "000300.SH" if reference == "index" else "510300.SH" if reference == "security" else config.symbol
    dca = DcaConfig(mode=mode, reference=reference, reference_symbol=symbol if reference != "target" else None,
                    frequency="daily", base_amount=10_000, ma_window=2, change_window=1, **overrides)
    config = config.model_copy(update={"dca": dca})
    daily = pd.DataFrame({"date": data.daily.date, "close": prices,
                          "preclose": np.r_[prices[0], prices[:-1]], "tradestatus": 1})
    if reference == "target":
        data.daily["close"] = daily.close
        data.daily["preclose"] = daily.preclose
    else:
        data.dca_reference = {"kind": reference, "symbol": symbol, "version": "ref-v1", "daily": daily,
                              "actions": normalize_actions(pd.DataFrame()), "metadata": {"price_tick": .001}}
    return config, data


def dca_result(config, data):
    return engine.run_backtest(config, data)["strategies"]["dca"]


def test_ordinary_default_is_identical_to_explicit_fixed_shares():
    config, data = fixture(days=4)
    raw = config.model_dump(mode="json")
    raw["dca"] = {"frequency": "daily", "shares": 100, "initial_shares": 0}
    implicit = BacktestConfig.model_validate(raw)
    assert implicit.dca.mode == "fixed_shares"
    explicit = implicit.model_copy(update={"dca": DcaConfig(**raw["dca"], mode="fixed_shares")})
    assert engine.run_backtest(implicit, data) == engine.run_backtest(explicit, data)


@pytest.mark.parametrize("mode,first_multiplier,second_multiplier", [
    ("price_change", 1.5, .5), ("ma", 1 + 5 / 19, 1 - 5 / 11),
])
def test_two_modes_increase_on_decline_decrease_on_rise_and_keep_cashflow_basis(mode, first_multiplier, second_multiplier):
    config, data = case(mode)
    result = dca_result(config, data)
    first, second = result["trades"][:2]
    assert first["dca_signal"]["multiplier"] == pytest.approx(first_multiplier)
    assert second["dca_signal"]["multiplier"] == pytest.approx(second_multiplier)
    assert first["quantity"] == int(10_000 * first_multiplier / 1000) * 100
    assert second["quantity"] == int(10_000 * second_multiplier / 1000) * 100
    assert all(trade["dca_signal"]["signal_date"] < trade["timestamp"][:10] for trade in result["trades"])
    invested = sum(trade["quantity"] * trade["price"] + trade["fees"] for trade in result["trades"])
    assert result["metrics"]["total_invested"] == pytest.approx(invested)
    assert result["return_basis"] == "cost"
    assert result["equity"][-1]["cumulative_profit"] == pytest.approx(result["equity"][-1]["equity"] - invested)
    assert result["dca_rule"] == "contrarian_linear_v1"


def test_multiplier_clamps_to_both_limits():
    config, data = case(prices=(10, 5, 20, 5, 10))
    result = dca_result(config, data)
    assert [row["dca_signal"]["multiplier"] for row in result["trades"][:2]] == [2, .5]


@pytest.mark.parametrize("mode", ["ma", "price_change"])
@pytest.mark.parametrize("reference", ["target", "security", "index"])
def test_execution_day_and_future_closes_cannot_change_prior_signals(mode, reference):
    config, data = case(mode, reference=reference)
    original_amounts, original = smart_dca_signals(config, data, data.calendar[1:])
    changed = deepcopy(data)
    execution_day = pd.Timestamp(config.start_date)
    frame = changed.daily if reference == "target" else changed.dca_reference["daily"]
    frame.loc[frame.date >= execution_day, "close"] = 98765.
    # Future unsupported actions must not poison a prior signal either.
    future_actions = normalize_actions(pd.DataFrame([{"record_date": execution_day,
        "ex_date": execution_day + pd.Timedelta(days=1), "stock_date": execution_day + pd.Timedelta(days=1),
        "cash_per_share": 0, "bonus_ratio": 1, "unsupported_reason": "future action"}]))
    if reference == "target":
        changed.actions = future_actions
    else:
        changed.dca_reference["actions"] = future_actions
    changed_amounts, changed_signals = smart_dca_signals(config, changed, changed.calendar[1:])
    assert changed_amounts[0] == original_amounts[0]
    assert changed_signals[str(execution_day.date())] == original[str(execution_day.date())]


def test_external_security_is_used_instead_of_flat_trading_target():
    config, data = case(reference="security")
    external = dca_result(config, data)
    target_data = deepcopy(data)
    target_data.dca_reference = None
    target = dca_result(config.model_copy(update={"dca": config.dca.model_copy(update={"reference": "target", "reference_symbol": None})}), target_data)
    assert external["trades"][0]["dca_signal"]["reference_symbol"] == "510300.SH"
    assert external["trades"][0]["quantity"] == 1500
    assert target["trades"][0]["quantity"] == 1000


def test_external_reference_never_falls_back_when_missing_or_wrong():
    config, data = case()
    data.dca_reference = None
    with pytest.raises(ValueError, match="不能使用交易标的替代"):
        dca_result(config, data)
    _, data = case()
    data.dca_reference["symbol"] = "000001.SH"
    with pytest.raises(ValueError, match="不匹配"):
        dca_result(config, data)


def test_missing_warmup_skips_without_injecting_principal_and_records_reason():
    config, data = case()
    config = config.model_copy(update={"dca": config.dca.model_copy(update={"mode": "ma", "ma_window": 250})})
    result = dca_result(config, data)
    assert not result["trades"]
    assert result["metrics"]["total_invested"] == 0
    assert result["metrics"]["skipped_periods"] == 3
    assert any(row["reason"] == "dca_insufficient_history" and row["message"] for row in result["rejections"])
    assert all(row["total_invested"] == 0 for row in result["equity"])


@pytest.mark.parametrize("drop,reason", [(1, "dca_reference_stale"), (0, "dca_reference_gap")])
def test_stale_or_gapped_reference_does_not_use_future_rows(drop, reason):
    config, data = case(mode="ma")
    # For a gap inside the selected window use three closes, retaining the
    # expected latest day so this is distinguishable from a stale reference.
    if drop == 0:
        config = config.model_copy(update={"start_date": data.daily.date.iloc[4].date(),
            "dca": config.dca.model_copy(update={"ma_window": 3})})
        data.dca_reference["daily"] = data.dca_reference["daily"].drop(index=1)
    else:
        data.dca_reference["daily"] = data.dca_reference["daily"].drop(index=drop)
    _, signals = smart_dca_signals(config, data, [pd.Timestamp(config.start_date)])
    assert signals[str(config.start_date)]["reason"] == reason


def test_suspended_reference_is_not_a_filled_forward_signal():
    config, data = case(reference="security")
    data.dca_reference["daily"].loc[1, "tradestatus"] = 0
    _, signals = smart_dca_signals(config, data, [pd.Timestamp(config.start_date)])
    assert signals[str(config.start_date)]["reason"] == "dca_reference_invalid"


@pytest.mark.parametrize("mode", ["ma", "price_change"])
def test_reference_split_is_not_a_market_decline(mode):
    config, data = case(mode, prices=(10, 5, 5, 5, 5), reference="security")
    daily = data.dca_reference["daily"]
    daily.loc[1, "preclose"] = 5.
    data.dca_reference["actions"] = normalize_actions(pd.DataFrame([{
        "record_date": daily.date.iloc[0], "ex_date": daily.date.iloc[1],
        "stock_date": daily.date.iloc[1], "cash_per_share": 0., "bonus_ratio": 1.,
    }]))
    result = dca_result(config, data)
    assert result["trades"][0]["dca_signal"]["signal"] == pytest.approx(0)
    assert result["trades"][0]["dca_signal"]["multiplier"] == pytest.approx(1)
    assert result["trades"][0]["quantity"] == 1000


def test_unexplained_reference_ex_rights_jump_skips():
    config, data = case(prices=(10, 5, 5, 5, 5), reference="security")
    data.dca_reference["daily"].loc[1, "preclose"] = 5.
    result = dca_result(config, data)
    assert result["rejections"][0]["reason"] == "dca_reference_action"


def test_amount_sizing_uses_slipped_price_lot_and_fees_count_as_contributions():
    config, data = case()
    config = config.model_copy(update={"fees": config.fees.model_copy(update={"slippage_bps": 10, "min_commission": 5}),
        "dca": config.dca.model_copy(update={"base_amount": 2000, "sensitivity": 1e-8})})
    first = dca_result(config, data)["trades"][0]
    assert first["quantity"] == 100
    assert first["price"] == pytest.approx(10.01)
    assert first["fees"] >= 5


def test_star_smart_dca_ignores_hidden_fixed_quantity_but_enforces_actual_lot():
    config, data = case()
    raw = config.model_dump(mode="json")
    raw.update(symbol="688001.SH", buy_shares=200)
    raw["dca"].update(shares=100, base_amount=1000)
    star = BacktestConfig.model_validate(raw)
    data.symbol = star.symbol
    data.metadata = instrument_terms(star.symbol, {})
    result = dca_result(star, data)
    assert not result["trades"]
    assert any(row["reason"] == "dca_amount_below_lot" for row in result["rejections"])
    raw["dca"]["mode"] = "fixed_shares"
    with pytest.raises(ValidationError, match="200"):
        BacktestConfig.model_validate(raw)


def test_smart_dca_preserves_target_corporate_action_settlement():
    config, data = fixture(days=3)
    _share_action_market(data, [(0, 1, 1, 1.)], [10, 10, 5, 5])
    # The exchange preclose on the split date is in the post-split basis.
    data.daily.loc[2, "preclose"] = 5.
    data.bars.loc[data.bars.date == data.daily.date.iloc[2], "preclose"] = 5.
    config = config.model_copy(update={"schema_version": 3, "max_shares": None,
        "dca": DcaConfig(mode="ma", ma_window=2, frequency="daily", base_amount=2000, initial_shares=100)})
    result = dca_result(config, data)
    # The 100 seed shares become 200; ex-date contribution is another 400
    # at 5, funded only when filled.
    ex_date = str(data.daily.date.iloc[2].date())
    ex_fill = next(row for row in result["trades"] if row["timestamp"][:10] == ex_date)
    # The first day has only one prior close, so its MA signal is skipped.
    assert ex_fill["quantity"] == 400
    assert ex_fill["shares"] == 600
    assert result["metrics"]["initial_equity"] == 1000


def test_smart_dca_changes_neither_grid_nor_hold_accounts():
    config, data = case()
    smart = engine.run_backtest(config, data)
    fixed = engine.run_backtest(config.model_copy(update={"dca": DcaConfig(frequency="daily")}), data)
    for key in ("grid", "buy_hold", "matched_funding_hold"):
        assert smart["strategies"][key] == fixed["strategies"][key]


@pytest.mark.parametrize("patch", [
    {"mode": "unknown"}, {"base_amount": 0}, {"base_amount": float("nan")},
    {"ma_window": 1}, {"ma_window": 2.5}, {"change_window": 0}, {"sensitivity": 0},
    {"min_multiplier": -1}, {"min_multiplier": 1.1}, {"max_multiplier": .9},
    {"max_multiplier": 21}, {"reference": "security"},
    {"reference": "index", "reference_symbol": "000001.SZ"},
    {"reference": "security", "reference_symbol": "000001.SH"}, {"surprise": True},
])
def test_parameter_validation(patch):
    with pytest.raises(ValidationError):
        DcaConfig.model_validate({"mode": "ma", **patch})


def test_reference_index_and_stock_namespaces_are_distinct():
    assert DcaConfig(mode="ma", reference="index", reference_symbol="000001.SH").reference_symbol == "000001.SH"
    assert DcaConfig(mode="ma", reference="security", reference_symbol="000001").reference_symbol == "000001.SZ"


def test_index_freeze_survives_latest_pointer_updates_and_missing_version_errors(tmp_path, monkeypatch):
    from stockregression import dca_reference as refs
    config, data = case()
    store = Store(tmp_path)
    frame = data.dca_reference["daily"]
    monkeypatch.setattr(refs, "fetch_index_daily", lambda *args: frame.copy())
    old = prepare_index_reference(store, config)
    frozen = freeze_dca_reference(store, config)
    service = SimpleNamespace(store=store)
    first = attach_dca_reference(service, frozen, deepcopy(data))
    frame = frame.assign(close=frame.close * 2)
    newer = prepare_index_reference(store, config)
    assert old["version"] != newer["version"]
    again = attach_dca_reference(service, frozen, deepcopy(data))
    pd.testing.assert_frame_equal(first.dca_reference["daily"], again.dca_reference["daily"])
    assert again.dca_reference["version"] == old["version"] == frozen.dca.reference_version
    store.state(f"smart_dca_index_v1:000300.SH:{old['version']}", None)
    with pytest.raises(ValueError, match="锁定版本不存在"):
        attach_dca_reference(service, frozen, data)
    with pytest.raises(ValueError, match="参考版本不存在"):
        freeze_dca_reference(store, frozen)
    with pytest.raises(ValueError, match="未锁定"):
        attach_dca_reference(service, config, data)


def test_external_security_freeze_uses_exact_daily_and_action_versions():
    config, data = case(reference="security")
    seen = []
    active = {"version": "security-v1", "market_versions": {"1day": {"version": 7}},
              "auxiliary_versions": {"actions": "actions-v1"}}
    snapshots = {"security-v1": deepcopy(active)}
    def snapshot(symbol, version=None):
        seen.append((symbol, version))
        return snapshots.get(version) if version else active
    action_calls = []
    service = SimpleNamespace(store=SimpleNamespace(snapshot=snapshot), repository=SimpleNamespace(
        bars=lambda symbol, timeframe, version: data.dca_reference["daily"].assign(source_version=version),
        auxiliary=lambda symbol, kind, version: action_calls.append(version) or {"payload": []}))
    frozen = freeze_dca_reference(service.store, config)
    active["version"] = "security-v2"
    active["market_versions"]["1day"]["version"] = 8
    attached = attach_dca_reference(service, frozen, data)
    assert seen[-1] == ("510300.SH", "security-v1")
    assert attached.dca_reference["version"] == "security-v1"
    assert attached.dca_reference["daily"].source_version.unique().tolist() == [7]
    assert action_calls == ["actions-v1"]


def test_target_reference_uses_target_snapshot_including_pre_start_warmup():
    config, data = case(reference="target")
    seen = []
    source = data.daily.copy()
    manifest = {"version": data.version, "market_versions": {"1day": {"version": 4}}, "auxiliary_versions": {}}
    service = SimpleNamespace(store=SimpleNamespace(snapshot=lambda symbol, version: seen.append((symbol, version)) or manifest),
        repository=SimpleNamespace(bars=lambda *args, **kwargs: source))
    attach_dca_reference(service, config, data)
    assert seen == [(config.symbol, data.version)]
    assert data.dca_reference["daily"].date.min().date() < config.start_date


def test_config_and_workbench_preferences_roundtrip_smart_dca(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "config" / "application.json")
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    config, _ = case(mode="ma", reference="security")
    client = TestClient(app)
    saved = client.post("/api/v1/configs", json={"name": "智能定投", "config": config.model_dump(mode="json")})
    assert saved.status_code == 200
    read = client.get("/api/v1/configs").json()
    assert read[0]["config"]["dca"] == config.dca.model_dump(mode="json")
    from stockregression.configuration import current_configuration, patch_configuration
    doc = current_configuration()
    patch_configuration({"backtest": {"dca": config.dca.model_dump(mode="json")}}, base_revision=doc["config_revision"])
    assert settings.load_settings().workbench_defaults.dca.model_dump() == config.dca.model_dump()


def test_index_prepare_api_and_submission_freeze(tmp_path, monkeypatch):
    from stockregression import dca_reference as refs
    from stockregression.market_data import MarketDataService
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "application.json")
    config, data = case()
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    store = app.state.store
    real_snapshot = store.snapshot
    monkeypatch.setattr(store, "snapshot", lambda symbol, version=None, **kwargs:
        {"version": "target-v1", "market_data": True} if symbol == config.symbol else real_snapshot(symbol, version, **kwargs))
    monkeypatch.setattr(MarketDataService, "readiness", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setattr(refs, "fetch_index_daily", lambda *args: data.dca_reference["daily"])
    client = TestClient(app)
    prepared = client.post("/api/v1/dca/reference/prepare", json=config.model_dump(mode="json"))
    assert prepared.status_code == 200
    submitted = client.post("/api/v1/backtests", json=config.model_dump(mode="json"))
    assert submitted.status_code == 200
    job = store.job(submitted.json()["job_id"])
    assert job["payload"]["config"]["dca"]["reference_version"] == prepared.json()["version"]
    assert job["payload"]["version"] == "target-v1"


def test_compiled_smart_dca_matches_interpreted(monkeypatch):
    config, data = case(mode="ma")
    interpreted_result = dca_result(config, data)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", True)
    compiled = dca_result(config, data)
    assert compiled == interpreted_result


@pytest.mark.parametrize("mode", ["ma", "price_change"])
@pytest.mark.parametrize("reference", ["security", "index"])
def test_research_reference_lock_survives_worker_validation_roundtrip(tmp_path, monkeypatch, mode, reference):
    from stockregression import dca_reference as refs
    config, data = case(mode=mode, reference=reference)
    research = OptimizationConfig(base=config)
    if reference == "index":
        store = Store(tmp_path)
        monkeypatch.setattr(refs, "fetch_index_daily", lambda *args: data.dca_reference["daily"])
        version = prepare_index_reference(store, config)["version"]
    else:
        version = "locked-security-v1"
        store = SimpleNamespace(snapshot=lambda *args, **kwargs:
            {"version": version, "market_versions": {"1day": {"version": 7}}})
    assert research.research_context.dca.reference_version is None
    frozen = freeze_research_dca_reference(store, research)
    assert frozen.base.dca.reference_version == version
    assert frozen.research_context.dca.reference_version == version
    # This is precisely the worker's model_validate(payload['config']) path.
    reloaded = OptimizationConfig.model_validate(frozen.model_dump(mode="json"))
    assert reloaded.base.dca.reference_version == version
    assert reloaded.research_context.dca.reference_version == version
    assert reloaded.base.dca == reloaded.research_context.dca
    # The public standalone request representation must retain the same lock.
    public = ResearchRequestV5(research_context=frozen.research_context, ranges=frozen.ranges)
    assert ResearchRequestV5.model_validate_json(public.model_dump_json()).to_optimization().base.dca.reference_version == version


@pytest.mark.parametrize("standalone", [False, True])
def test_research_api_enqueues_lock_in_base_and_context(tmp_path, monkeypatch, standalone):
    from stockregression import dca_reference as refs
    from stockregression.data import DataService
    from stockregression.market_data import MarketDataService
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "application.json")
    config, data = case(mode="ma")
    research = OptimizationConfig(base=config)
    app = create_app(str(tmp_path / "data"), start_jobs=False)
    store = app.state.store
    monkeypatch.setattr(refs, "fetch_index_daily", lambda *args: data.dca_reference["daily"])
    version = prepare_index_reference(store, config)["version"]
    manifest = {"version": "target-frozen", "market_data": True, "files": {"calendar": []}}
    monkeypatch.setattr(store, "snapshot", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(MarketDataService, "readiness", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setattr("stockregression.api.preparation_config", lambda store, base, **kwargs: base)
    bundle = SimpleNamespace(version="target-frozen", bars=pd.DataFrame(), daily=pd.DataFrame(), calendar=pd.DatetimeIndex([]))
    loaded_versions = []
    monkeypatch.setattr(DataService, "load_bundle", lambda self, base, *args, **kwargs:
        loaded_versions.append(base.dca.reference_version) or bundle)
    estimated_versions = []
    def estimate(cfg, *args, **kwargs):
        estimated_versions.append((cfg.base.dca.reference_version, cfg.research_context.dca.reference_version))
        return {"status": "ready", "effective_total_evaluation_budget": 100,
                "task_evaluation_upper_bound": 100, "max_runtime_seconds": 60,
                "estimated_task_memory_mb": 1, "workspace_budget_mb": 10}
    monkeypatch.setattr("stockregression.research.robust.estimate_robust_search", estimate)
    payload = research.model_dump(mode="json")
    if standalone:
        payload = ResearchRequestV5(research_context=research.research_context, ranges=research.ranges).model_dump(mode="json")
    client = TestClient(app)
    preview = client.post("/api/v1/optimizations/preview", json=payload)
    assert preview.status_code == 200, preview.text
    submitted = client.post("/api/v1/optimizations", json=payload)
    assert submitted.status_code == 200, submitted.text
    queued = store.job(submitted.json()["job_id"])["payload"]["config"]
    assert queued["base"]["dca"]["reference_version"] == version
    assert queued["research_context"]["dca"]["reference_version"] == version
    reloaded = OptimizationConfig.model_validate(queued)
    assert reloaded.base.dca.reference_version == reloaded.research_context.dca.reference_version == version
    assert loaded_versions and all(value == version for value in loaded_versions)
    assert estimated_versions and all(value == (version, version) for value in estimated_versions)
