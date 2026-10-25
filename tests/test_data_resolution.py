from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from stockregression.data_resolution import (
    POLICY, build_data_resolution, validate_data_resolution,
    resolution_intervals, resolution_sampling,
)
from stockregression.market_data.consumer import _load_resolved_bundle
from stockregression.market_data.repository import VALIDATION_RULES_VERSION
from stockregression.models import BacktestConfig
from stockregression.quality import minute_slots


SYMBOL = "510300.SH"
DAYS = pd.bdate_range("2026-01-05", periods=10)
PRIOR = pd.Timestamp("2026-01-02")


def _record(day, *, timeframe="1day", status="passed", **report):
    return {"trade_date": str(day.date()), "status": status, "scanned": True,
        "row_count": 240 if timeframe == "1min" else 1,
        "source_hash": f"source-{timeframe}-{day.date()}", "dependency_hash": "dependency",
        "rules_version": VALIDATION_RULES_VERSION, "report": report}


class Repository:
    def __init__(self):
        self.rows = {
            "1day": [_record(day) for day in [PRIOR, *DAYS]],
            "1min": [_record(day, timeframe="1min") for day in DAYS],
        }
        self.reads = []

    def day_validations(self, symbol, timeframe, *, start, end):
        return [row for row in self.rows[timeframe]
                if pd.Timestamp(start) <= pd.Timestamp(row["trade_date"]) <= pd.Timestamp(end)]

    def validation_state(self, symbol, timeframe, *, start, end, **kwargs):
        rows = self.day_validations(symbol, timeframe, start=start, end=end)
        return {"available": True, "verified_from": str(PRIOR.date()),
            "verified_through": str(DAYS[-1].date()),
            "failed": [row["trade_date"] for row in rows if row["status"] == "failed"],
            "pending": [row["trade_date"] for row in rows if row["status"] == "pending"]}

    def auxiliary(self, symbol, kind, version=None):
        return {"payload": [{"date": day} for day in [PRIOR, *DAYS]] if kind == "calendar" else [], "verified": True}

    def version_info(self, *args):
        return None

    def bars(self, symbol, timeframe, **kwargs):
        self.reads.append(timeframe)
        if timeframe == "1day":
            frame = pd.DataFrame({"timestamp": [PRIOR, *DAYS], "open": 10., "high": 10.2,
                "low": 9.8, "close": 10.1, "volume": 1000., "turnover": 10100.,
                "preclose": 10.1, "tradestatus": 1, "is_st": False})
            # The skipped day contains a poisoned source row. It must never
            # invalidate the retained sessions or become an artificial mark.
            for row in self.rows["1day"]:
                if row["status"] == "failed":
                    frame.loc[frame.timestamp.eq(pd.Timestamp(row["trade_date"])), "close"] = -1.
            return frame
        return pd.concat([pd.DataFrame({"timestamp": minute_slots(day), "open": 10., "high": 10.2,
            "low": 9.8, "close": 10.1, "volume": 1000 / 240, "turnover": 10100 / 240,
            "is_synthetic": False, "is_auction_placeholder": False, "source_verified": True})
            for day in DAYS], ignore_index=True)


def _harness():
    repository = Repository()
    service = SimpleNamespace(repository=repository)
    manifest = {"symbol": SYMBOL, "version": "frozen-test", "market_data": True,
        "metadata": {"asset_type": "etf", "price_tick": .001},
        "market_versions": {tf: {"version": 7, "first": str(PRIOR.date()), "last": str(DAYS[-1].date())}
                            for tf in ("1day", "1min")},
        "auxiliary_versions": {"calendar": "calendar-7", "actions": "actions-7"}}
    return service, manifest


def test_resolution_is_per_day_and_catalogue_only_without_any_ratio_gate():
    service, manifest = _harness()
    for row in service.repository.rows["1min"][:8]:
        row["status"] = "failed"
    service.repository.rows["1day"][5]["status"] = "failed"
    plan = build_data_resolution(service, SYMBOL, manifest, DAYS)
    assert service.repository.reads == []
    assert plan["counts"] == {"minute": 2, "daily": 7, "skip": 1, "suspended": 0, "no_trades": 0}
    assert plan["days"][4]["mode"] == "skip"
    assert plan["prior_daily_days"] == [str(PRIOR.date())]
    sampling = resolution_sampling(plan)
    assert sampling["valid_days"] == 2
    assert sampling["statistics_days"] == 2
    assert sampling["data_policy"] == POLICY
    assert len(resolution_intervals(plan, "1min")) == 1
    assert len(resolution_intervals(plan, "1day")) == 2


def test_short_filled_minutes_never_become_minute_statistics():
    service, manifest = _harness()
    service.repository.rows["1min"][0]["report"] = {"synthetic_rows": 1}
    service.repository.rows["1min"][1]["row_count"] = 238
    service.repository.rows["1min"][1]["report"] = {"auction_placeholders": 2}
    plan = build_data_resolution(service, SYMBOL, manifest, DAYS)
    assert plan["days"][0]["mode"] == "daily"
    assert plan["days"][1]["mode"] == "minute"


@pytest.mark.parametrize("field", ["days", "versions", "counts", "policy", "plan_hash"])
def test_frozen_resolution_rejects_tampering(field):
    service, manifest = _harness()
    manifest["data_resolution"] = build_data_resolution(service, SYMBOL, manifest, DAYS)
    assert validate_data_resolution(manifest)["counts"]["minute"] == 10
    corrupt = deepcopy(manifest)
    if field == "days":
        corrupt["data_resolution"][field][0]["mode"] = "daily"
    else:
        corrupt["data_resolution"][field] = "tampered"
    with pytest.raises(ValueError, match="DATA_RESOLUTION_MISMATCH"):
        validate_data_resolution(corrupt)
    corrupt = deepcopy(manifest)
    corrupt["market_versions"]["1day"]["version"] += 1
    with pytest.raises(ValueError, match="DATA_RESOLUTION_MISMATCH"):
        validate_data_resolution(corrupt)


def test_statistics_consumer_excludes_bad_days_but_keeps_full_calendar():
    service, manifest = _harness()
    service.repository.rows["1min"][1]["status"] = "failed"
    service.repository.rows["1day"][3]["status"] = "failed"
    manifest["data_resolution"] = build_data_resolution(service, SYMBOL, manifest, DAYS)
    config = BacktestConfig(symbol=SYMBOL, start_date=DAYS[0].date(), end_date=DAYS[-1].date(),
        schema_version=3, data_policy=POLICY)
    data = _load_resolved_bundle(service, config, manifest, statistics_only=True)
    observed_days = set(data.bars.timestamp.dt.normalize())
    assert DAYS[1] not in observed_days
    assert DAYS[2] not in observed_days
    assert len(data.calendar) == 10
    assert data.metadata["etf_sampling"]["statistics_days"] == 8
    assert data.quality["daily_days"] == 1
    assert data.quality["skipped_days"] == 1
    assert data.metadata["prior_closes"][str(PRIOR.date())] == 10.1
    # No mutable quality query is needed once the plan is frozen.
    service.repository.validation_state = lambda *a, **k: (_ for _ in ()).throw(AssertionError("live ledger read"))
    again = _load_resolved_bundle(service, config, manifest, statistics_only=True)
    pd.testing.assert_frame_equal(data.bars, again.bars)


def test_boundary_skip_keeps_requested_calendar_and_last_real_valuation_date():
    service, manifest = _harness()
    service.repository.rows["1day"][-1]["status"] = "failed"
    manifest["data_resolution"] = build_data_resolution(service, SYMBOL, manifest, DAYS)
    config = BacktestConfig(symbol=SYMBOL, start_date=DAYS[0].date(), end_date=DAYS[-1].date(),
        schema_version=3, data_policy=POLICY)
    data = _load_resolved_bundle(service, config, manifest, statistics_only=True)
    assert data.calendar[-1] == DAYS[-1]
    assert data.metadata["valuation_as_of"] == str(DAYS[-2].date())
    assert DAYS[-1] not in set(data.bars.timestamp.dt.normalize())


def test_requested_weekend_boundary_is_frozen_separately_from_sessions():
    from stockregression.data_resolution import _digest
    service, manifest = _harness()
    plan = build_data_resolution(service, SYMBOL, manifest, DAYS)
    plan["requested_start"] = "2026-01-03"
    plan["requested_end"] = str(DAYS[-1].date())
    plan["plan_hash"] = _digest({key: value for key, value in plan.items() if key != "plan_hash"})
    manifest["data_resolution"] = plan
    assert validate_data_resolution(manifest, "2026-01-03", DAYS[-1])["start"] == "2026-01-05"
    with pytest.raises(ValueError, match="DATA_RESOLUTION_MISMATCH"):
        validate_data_resolution(manifest, "2025-12-01", DAYS[-1])


def test_execution_consumer_uses_daily_and_skip_without_fake_price_bars():
    service, manifest = _harness()
    service.repository.rows["1min"][1]["status"] = "failed"
    service.repository.rows["1day"][3]["status"] = "failed"
    manifest["data_resolution"] = build_data_resolution(service, SYMBOL, manifest, DAYS)
    config = BacktestConfig(symbol=SYMBOL, start_date=DAYS[0].date(), end_date=DAYS[-1].date(),
        schema_version=3, data_policy=POLICY)
    data = _load_resolved_bundle(service, config, manifest)
    assert DAYS[2] not in set(data.bars.timestamp.dt.normalize())
    daily = data.bars[data.bars.timestamp.dt.normalize().eq(DAYS[1])]
    assert len(daily) == 1
    assert daily.iloc[0].frequency == "1d"
    assert data.bars[["open", "high", "low", "close"]].notna().all().all()
    assert len(data.calendar) == 10
    from stockregression.engine import run_backtest
    assert PRIOR in set(data.daily.date)
    result = run_backtest(config, data)
    for strategy in result["strategies"].values():
        assert all(str(DAYS[2].date()) not in str(point.get("timestamp", ""))
                   for point in strategy.get("equity", []))


def test_independently_certified_price_aggregate_discrepancy_is_visible():
    service, manifest = _harness()
    original = service.repository.bars
    def bars(symbol, timeframe, **kwargs):
        frame = original(symbol, timeframe, **kwargs)
        if timeframe == "1day":
            frame["high"] = 10.19
        return frame
    service.repository.bars = bars
    manifest["data_resolution"] = build_data_resolution(service, SYMBOL, manifest, DAYS)
    config = BacktestConfig(symbol=SYMBOL, start_date=DAYS[0].date(), end_date=DAYS[-1].date(),
        schema_version=3, data_policy=POLICY)
    data = _load_resolved_bundle(service, config, manifest)
    assert data.quality["minute_days"] == 10
    assert all(day["price_aggregation_error"] > 0 for day in data.quality["days"])
    assert any("独立校验凭证" in warning for warning in data.warnings)
