from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockregression.market_data import MarketDataService
from stockregression.market_data.models import DataNotReady
from stockregression.market_data.package import index_etf_package_direct
from stockregression.market_data.quality import validate_and_fill
from stockregression.market_data.service import _summary_from_anomalies
from stockregression.models import BacktestConfig
from stockregression.price_validation import details_to_rows, price_issue_details
from stockregression.quality import minute_slots
from stockregression.storage import Store


DAY = pd.Timestamp("2026-10-15")


def _frame(day=DAY) -> pd.DataFrame:
    slots = minute_slots(day)
    values = np.linspace(1.030, 1.040, len(slots))
    return pd.DataFrame({
        "timestamp": slots,
        "open": values,
        "high": values + 0.002,
        "low": values - 0.002,
        "close": values,
        "volume": np.full(len(slots), 100_000.0),
        "turnover": np.full(len(slots), 100_000.0),
    })


def test_price_issue_details_covers_zero_missing_and_inverted_prices():
    frame = _frame().iloc[:4].copy()
    frame.loc[0, "open"] = 0
    frame.loc[1, "close"] = np.nan
    frame.loc[2, "high"] = frame.loc[2, "low"] - 0.01
    frame.loc[3, "high"] = np.inf

    issues = price_issue_details(frame)

    assert len(issues) == 4
    assert all(value for value in issues.quality_fields)
    assert all(value for value in issues.quality_reason)


def test_zero_price_gap_is_reported_and_never_filled():
    frame = _frame()
    missing = frame.timestamp.iloc[100]
    frame = frame.drop(index=100).reset_index(drop=True)
    frame.attrs["price_anomaly_rows"] = details_to_rows(pd.DataFrame([{
        "timestamp": missing, "source": "package",
        "quality_fields": "open", "quality_reason": "open 非正",
    }]))
    frame.attrs["blocked_timestamps"] = [missing.isoformat()]
    frame["source"] = "package"
    frame["source_verified"] = True
    frame["is_synthetic"] = False
    frame["is_auction_placeholder"] = False

    result = validate_and_fill(
        frame, "1min", calendar=pd.DataFrame({"date": [DAY]}), fill=True,
        coverage_start=DAY, coverage_end=frame.timestamp.max(), symbol="510300.SH",
    )

    assert result.report["blocked_price_rows"] == 1
    assert result.report["synthetic_rows"] == 0
    assert result.report["unfillable"] >= 1
    assert not result.frame.timestamp.eq(missing).any()
    assert any("无效价格" in item for item in result.pending)


def test_zero_volume_day_is_not_reported_as_suspended():
    from stockregression.quality import prepare_bundle

    daily = pd.DataFrame({
        "date": [DAY], "open": [1.0], "high": [1.0], "low": [1.0],
        "close": [1.0], "volume": [0.0], "turnover": [0.0],
        "preclose": [1.0], "tradestatus": [1], "is_st": [False],
    })
    bundle = prepare_bundle(
        "510300.SH", _frame(), daily, pd.DataFrame(), pd.DataFrame({"date": [DAY]}),
        DAY, DAY, "1m", "v1", {"asset_type": "etf"}, {},
    )

    assert bundle.quality["suspended_days"] == 0
    assert bundle.quality["no_trade_days"] == 1
    assert bundle.quality["days"][0]["mode"] == "no_trades"
    assert bundle.bars.empty


def test_package_read_filters_invalid_prices_and_blocks_calculation(tmp_path):
    root = tmp_path / "package"
    minute_dir = root / "etf_1min"
    minute_dir.mkdir(parents=True)
    symbol = "510300.SH"
    basic = pd.DataFrame([{"ts_code": symbol, "cname": "测试ETF"}])
    daily = pd.DataFrame([{
        "ts_code": symbol, "trade_date": DAY,
        "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
        "pre_close": 1.0, "vol": 240_000.0, "amount": 240_000.0,
        "tradestatus": 1, "is_st": 0,
    }])
    minute = _frame().assign(
        ts_code=symbol, trade_date=DAY.strftime("%Y%m%d"),
        trade_time=lambda value: value.timestamp,
    ).rename(columns={"volume": "vol", "turnover": "amount"})
    minute.loc[[100, 101], ["open", "high", "low", "close"]] = 0
    basic.to_parquet(root / "etf_basic_data.parquet", index=False)
    daily.to_parquet(root / "etf_daily.parquet", index=False)
    minute.to_parquet(minute_dir / f"{symbol}.parquet", index=False)

    service = MarketDataService(Store(tmp_path / "data"), package_dir=root)
    index_etf_package_direct(service, root, timeframes=["1day", "1min"], symbols=[symbol])
    service.repository.save_auxiliary(symbol, "calendar", [{"date": DAY.isoformat()}], verified=True)
    service.repository.save_auxiliary(symbol, "actions", [], verified=True)
    service.refresh_snapshot(symbol)

    displayed = service.market_bars(
        symbol, "1min", start=DAY.isoformat(), end=DAY.isoformat(), limit=500,
    )
    assert len(displayed["bars"]) == 238
    assert displayed["price_quality"]["count"] == 2
    assert all(row["open"] > 0 for row in displayed["bars"])
    moving = [row.get("ma5") for row in displayed["bars"]]
    assert any(pd.isna(value) for value in moving)
    assert pd.notna(moving[-1])

    config = BacktestConfig(
        symbol=symbol, start_date=date(2026, 10, 15),
        end_date=date(2026, 10, 15), frequency="1m",
    )
    readiness = service.readiness(config)
    assert not readiness["ready"]
    assert any(gap["kind"] == "price_quality" for gap in readiness["gaps"])
    with pytest.raises(DataNotReady):
        service.load_backtest_bundle(config)


def test_anomaly_summary_round_trip_keeps_source_version_and_reason():
    issues = price_issue_details(_frame().iloc[:1].assign(open=0))
    summary = _summary_from_anomalies(issues, "510300.SH", "1min", 7)
    assert summary["count"] == 1
    assert summary["samples"][0]["symbol"] == "510300.SH"
    assert summary["samples"][0]["version"] == 7
    assert "非正" in summary["samples"][0]["reason"]
