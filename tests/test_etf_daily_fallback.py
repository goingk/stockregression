from __future__ import annotations

from contextlib import nullcontext
import pickle
import sys
import types

import pandas as pd
import pytest

from stockregression import providers
from stockregression.settings import MarketDataSettings, Settings
from stockregression.providers import EmptyUnresolvedError
from stockregression.quality import DataQualityError
from stockregression.source_health import (
    SourceFailure,
    SourceRouter,
    _qualification_key,
    classify_etf_product,
    fetch_auxiliary_routed,
    probe_source,
    record_success,
    source_qualification,
    source_state,
)
from stockregression.storage import Store


DATE = pd.Timestamp("2026-10-23")


def daily_frame(day=DATE, *, invalid=False):
    low, high = (11.0, 10.0) if invalid else (9.0, 11.0)
    return pd.DataFrame({
        "date": [pd.Timestamp(day)], "open": [10.0], "high": [high],
        "low": [low], "close": [10.5], "volume": [1000.0],
        "turnover": [10500.0], "preclose": [10.0],
    })


def make_store(tmp_path, *, source="baostock", enabled=True):
    root = tmp_path / "catalog"
    market = MarketDataSettings(
        etf_daily_source=source,
        etf_daily_fallback_enabled=enabled,
        request_retries=0,
    )
    settings = Settings(
        data_dir=str(root), package_dir=None, etf_daily_source=source,
        market_data=market,
    )
    return Store(root, settings=settings)


def add_instrument(store, symbol="510300.SH", *, name="沪深300ETF", etf_type="ETF"):
    store.put_instrument(
        symbol, name, asset_type="etf", etf_type=etf_type,
    )


def qualify(store, monkeypatch, product_type="etf", sample_symbol=None):
    symbol = sample_symbol or "159611.SZ"
    if sample_symbol:
        add_instrument(store, sample_symbol, name={"lof": "测试基金(LOF)", "reit": "测试REIT"}.get(product_type, "测试ETF"), etf_type=product_type.upper())
    elif product_type == "etf":
        add_instrument(store, "510300.SH")
    monkeypatch.setattr(providers, "latest_closed_date", lambda: DATE)
    monkeypatch.setattr(
        providers, "fetch_etf_daily",
        lambda *_args, **_kwargs: daily_frame(),
    )
    result = probe_source(
        store, "eastmoney", "etf_daily", "etf",
        product_type=product_type, sample_symbol=sample_symbol,
    )
    assert result["status"] == "completed"
    assert result["results"][0]["symbol"] == symbol
    return result["results"][0]["qualification"]


def patch_auxiliary_adapters(monkeypatch):
    monkeypatch.setattr(
        providers, "fetch_market_calendar",
        lambda *_args, **_kwargs: pd.DataFrame({"date": [DATE]}),
    )
    monkeypatch.setattr(
        providers, "fetch_market_actions",
        lambda *_args, **_kwargs: (pd.DataFrame(), {"actions_verified": True, "warnings": []}),
    )


def test_explicit_probe_writes_fresh_type_qualification_only(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    qualification = qualify(store, monkeypatch)

    assert qualification["status"] == "passed"
    assert qualification["product_type"] == "etf"
    assert qualification["sample_symbol"] == "159611.SZ"
    assert qualification["rows"] == 1
    assert source_qualification(store, "eastmoney", "etf")["status"] == "passed"

    store.state(_qualification_key("eastmoney", "etf"), {
        **qualification, "expires_at": "2000-01-01T00:00:00+00:00",
    })
    assert source_qualification(store, "eastmoney", "etf")["status"] == "expired"


def test_old_market_pass_does_not_qualify_fallback(tmp_path):
    store = make_store(tmp_path)
    record_success(store, "eastmoney", "etf_daily", "etf", rows=20)

    assert source_state(store, "eastmoney", "etf_daily", "etf")["status"] == "passed"
    assert source_qualification(store, "eastmoney", "etf")["status"] == "unknown"


def test_qualification_rejects_payload_from_another_policy_version(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    qualification = qualify(store, monkeypatch)
    legacy_version = "etf-daily-type-probe-v0"
    store.state(_qualification_key("eastmoney", "etf"), {
        **qualification, "policy_version": legacy_version,
    })

    result = source_qualification(store, "eastmoney", "etf")

    assert result["status"] == "unknown"
    assert result["policy_version"] == legacy_version
    assert result["qualification_error"] == "policy_version_mismatch"


@pytest.mark.parametrize("cache_kind", ["legacy", "expired"])
def test_legacy_or_expired_pass_still_allows_eastmoney_attempt(
    tmp_path, monkeypatch, cache_kind,
):
    store = make_store(tmp_path)
    add_instrument(store, "510300.SH")
    if cache_kind == "legacy":
        record_success(store, "eastmoney", "etf_daily", "etf", rows=20)
    else:
        qualification = qualify(store, monkeypatch)
        store.state(_qualification_key("eastmoney", "etf"), {
            **qualification, "expires_at": "2000-01-01T00:00:00+00:00",
        })
    patch_auxiliary_adapters(monkeypatch)
    calls = []

    def fetch(_symbol, _start, _end, source, *_args, **_kwargs):
        calls.append(source)
        if source == "baostock":
            raise EmptyUnresolvedError("BaoStock empty")
        return daily_frame()

    monkeypatch.setattr(providers, "fetch_etf_daily", fetch)
    _daily, _actions, _calendar, metadata = fetch_auxiliary_routed(store, "510300.SH", DATE, DATE)

    assert calls == ["baostock", "eastmoney"]
    assert metadata["etf_daily_route"]["fallback_attempted"] is True
    assert metadata["etf_daily_route"]["fallback_status"] == "succeeded"


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"asset_type": "etf", "name": "沪深300ETF"}, "etf"),
        ({"asset_type": "etf", "name": "指数增强LOF", "etf_type": "股票型"}, "lof"),
        ({"asset_type": "etf", "etf_type": "REITs"}, "reit"),
        ({"asset_type": "etf", "name": "某股票型基金", "etf_type": "股票型"}, "unknown"),
        ({"asset_type": "etf", "name": "510300.SH"}, "unknown"),
        ({"asset_type": "etf", "name": "LOF REIT"}, "unknown"),
    ],
)
def test_metadata_classifier_is_explicit_and_keeps_unknown(metadata, expected):
    assert classify_etf_product(metadata) == expected


@pytest.mark.parametrize("product_type", ["lof", "reit"])
def test_non_etf_product_requires_its_own_probe_and_then_can_fallback(
    tmp_path, monkeypatch, product_type,
):
    store = make_store(tmp_path)
    symbol = "160119.SZ" if product_type == "lof" else "508000.SH"
    add_instrument(
        store, symbol,
        name="测试基金(LOF)" if product_type == "lof" else "测试REIT",
        etf_type="LOF" if product_type == "lof" else "REITs",
    )
    assert source_qualification(store, "eastmoney", product_type)["status"] == "unknown"

    qualify(store, monkeypatch, product_type, symbol)
    patch_auxiliary_adapters(monkeypatch)
    calls = []

    def fetch(_symbol, _start, _end, source, *_args, **_kwargs):
        calls.append(source)
        if source == "baostock":
            raise EmptyUnresolvedError("BaoStock 空响应")
        return daily_frame()

    monkeypatch.setattr(providers, "fetch_etf_daily", fetch)
    daily, _actions, _calendar, metadata = fetch_auxiliary_routed(
        store, symbol, DATE, DATE,
    )

    assert calls == ["baostock", "eastmoney"]
    assert not daily.empty
    assert metadata["product_type"] == product_type
    assert metadata["etf_daily_route"]["fallback_status"] == "succeeded"
    assert metadata["etf_daily_route"]["source_attempts"][0]["error_category"] == "empty_unresolved"
    assert metadata["source_attempts"] == metadata["etf_daily_route"]["source_attempts"]


@pytest.mark.parametrize("product_type", ["unknown", "lof", "reit"])
def test_unqualified_or_unknown_metadata_still_attempts_eastmoney(
    tmp_path, monkeypatch, product_type,
):
    store = make_store(tmp_path)
    names = {"lof": "测试基金(LOF)", "reit": "测试REIT", "unknown": "股票型基金"}
    add_instrument(store, "510300.SH", name=names[product_type], etf_type=product_type.upper())
    patch_auxiliary_adapters(monkeypatch)
    calls = []

    def fetch(_symbol, _start, _end, source, *_args, **_kwargs):
        calls.append(source)
        raise EmptyUnresolvedError(f"{source} 空响应")

    monkeypatch.setattr(providers, "fetch_etf_daily", fetch)
    with pytest.raises(EmptyUnresolvedError) as caught:
        fetch_auxiliary_routed(store, "510300.SH", DATE, DATE)

    assert calls == ["baostock", "eastmoney"]
    assert caught.value.product_type == product_type
    assert caught.value.fallback_status == "failed"


def test_empty_both_sources_is_typed_and_does_not_cool_down_source_health(
    tmp_path, monkeypatch,
):
    store = make_store(tmp_path)
    qualify(store, monkeypatch)
    patch_auxiliary_adapters(monkeypatch)
    eastmoney_before = source_state(store, "eastmoney", "etf_daily", "etf")
    calls = []

    def fetch(_symbol, _start, _end, source, *_args, **_kwargs):
        calls.append(source)
        raise EmptyUnresolvedError(f"{source} 没有该区间数据")

    monkeypatch.setattr(providers, "fetch_etf_daily", fetch)
    with pytest.raises(EmptyUnresolvedError) as caught:
        fetch_auxiliary_routed(store, "510300.SH", DATE, DATE)

    assert calls == ["baostock", "eastmoney"]
    assert caught.value.category == "empty_unresolved"
    assert len(caught.value.source_attempts) == 2
    assert caught.value.etf_daily_route["fallback_attempted"] is True
    assert caught.value.etf_daily_route["fallback_status"] == "failed"
    assert len(caught.value.etf_daily_route["source_attempts"]) == 2
    assert caught.value.source_events[0]["kind"] == "source_fallback"
    assert source_state(store, "baostock", "etf_daily", "etf")["status"] == "unknown"
    eastmoney_after = source_state(store, "eastmoney", "etf_daily", "etf")
    assert eastmoney_after["status"] == eastmoney_before["status"] == "passed"
    assert eastmoney_after["last_failure_at"] == eastmoney_before["last_failure_at"]


def test_router_empty_dataframe_is_not_success_or_source_health_pass(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(SourceFailure) as caught:
        SourceRouter(store).call(
            "etf_daily", "etf", {"baostock": lambda: pd.DataFrame()},
            product_type="etf",
        )

    assert caught.value.category == "empty_unresolved"
    assert source_state(store, "baostock", "etf_daily", "etf")["status"] == "unknown"


def test_routed_error_metadata_survives_process_exception_serialization():
    error = SourceFailure(
        "baostock", "etf_daily", "data_quality", "bad bars",
        [{"source": "eastmoney", "status": "failed", "error_category": "data_quality"}],
        {"fallback_status": "failed"},
        [{"kind": "source_fallback"}],
    )

    restored = pickle.loads(pickle.dumps(error))

    assert restored.source_attempts == error.source_attempts
    assert restored.etf_daily_route == error.etf_daily_route
    assert restored.source_events == error.source_events


def test_probe_requires_matching_metadata_for_explicit_sample(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    add_instrument(store, "510300.SH", name="测试基金(LOF)", etf_type="LOF")
    calls = []
    monkeypatch.setattr(providers, "fetch_etf_daily", lambda *_a, **_k: calls.append(True) or daily_frame())

    with pytest.raises(ValueError, match="不能确认类型 etf"):
        probe_source(
            store, "eastmoney", "etf_daily", "etf",
            product_type="etf", sample_symbol="510300.SH",
        )
    assert calls == []


def test_lof_and_reit_probe_require_explicit_matching_sample(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    calls = []
    monkeypatch.setattr(providers, "fetch_etf_daily", lambda *_a, **_k: calls.append(True))

    for product_type in ("lof", "reit"):
        with pytest.raises(ValueError, match="必须提供.*sample_symbol"):
            probe_source(
                store, "eastmoney", "etf_daily", "etf", product_type=product_type,
            )

    assert calls == []


def test_probe_empty_response_does_not_write_market_wide_cooldown(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    monkeypatch.setattr(providers, "latest_closed_date", lambda: DATE)
    monkeypatch.setattr(providers, "fetch_etf_daily", lambda *_a, **_k: pd.DataFrame())

    result = probe_source(store, "eastmoney", "etf_daily", "etf")

    assert result["results"][0]["error_category"] == "empty_unresolved"
    assert result["results"][0]["qualification"]["status"] == "failed"
    assert source_state(store, "eastmoney", "etf_daily", "etf")["status"] == "unknown"


def test_invalid_eastmoney_fallback_is_rejected_before_success_is_recorded(
    tmp_path, monkeypatch,
):
    real_fetch = providers.fetch_etf_daily
    store = make_store(tmp_path)
    qualify(store, monkeypatch)
    patch_auxiliary_adapters(monkeypatch)
    monkeypatch.setattr(providers, "bounded_akshare_http", lambda *_a, **_k: nullcontext())
    monkeypatch.setitem(sys.modules, "akshare", types.SimpleNamespace(
        fund_etf_hist_em=lambda **_kwargs: pd.DataFrame({
            "date": [DATE], "open": [10.0], "high": [9.0], "low": [11.0],
            "close": [10.0], "volume": [10], "turnover": [100],
        }),
    ))
    with pytest.raises(DataQualityError):
        real_fetch("510300.SH", DATE, DATE, "eastmoney")

    def fetch(symbol, start, end, source="baostock", *_args, **kwargs):
        if source == "baostock":
            raise EmptyUnresolvedError("BaoStock 空响应")
        return real_fetch(symbol, start, end, source, *_args, **kwargs)

    monkeypatch.setattr(providers, "fetch_etf_daily", fetch)
    with pytest.raises(SourceFailure) as caught:
        fetch_auxiliary_routed(store, "510300.SH", DATE, DATE)

    assert caught.value.category == "data_quality"
    assert [attempt["source"] for attempt in caught.value.source_attempts] == ["baostock", "eastmoney"]
    assert caught.value.etf_daily_route["fallback_status"] == "failed"
    assert caught.value.etf_daily_route["source_attempts"] == caught.value.source_attempts
    assert caught.value.source_events[0]["kind"] == "source_fallback"
    assert source_state(store, "eastmoney", "etf_daily", "etf")["status"] == "passed"
    assert source_state(store, "eastmoney", "etf_daily", "etf")["last_failure_at"] is None


def test_market_data_settings_default_and_configuration_document_round_trip():
    from stockregression.settings import configuration_document, settings_from_document

    settings = Settings()
    document = configuration_document(settings)
    assert document["market_data"]["etf_daily_fallback_enabled"] is True
    restored = settings_from_document(document)
    assert restored.market_data.etf_daily_fallback_enabled is True
