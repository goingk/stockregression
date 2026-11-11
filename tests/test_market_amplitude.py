from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockregression.api import create_app
from stockregression.market_data import MarketDataService
from stockregression.market_data.amplitude import calculate_daily_amplitude
from stockregression.market_data.models import MarketDataError
from stockregression.storage import Store


def _one_day(high: float, low: float) -> pd.DataFrame:
    middle = (high + low) / 2.0
    return pd.DataFrame({
        "timestamp": ["2026-01-05"], "open": [middle], "high": [high],
        "low": [low], "close": [middle], "preclose": [100.0],
        "tradestatus": [1], "volume": [100.0],
    })


@pytest.mark.parametrize(
    "high,low,amplitude,upside,downside",
    [
        (110.0, 90.0, 0.20, 0.10, 0.10),
        (95.0, 90.0, 0.05, 0.00, 0.10),
        (110.0, 105.0, 0.05, 0.10, 0.00),
    ],
)
def test_daily_amplitude_uses_gap_aware_formulas(high, low, amplitude, upside, downside):
    result = calculate_daily_amplitude(_one_day(high, low))

    sample = result["samples"][0]
    assert sample["previous_close"] == pytest.approx(100.0)
    assert sample["amplitude"] == pytest.approx(amplitude)
    assert sample["upside"] == pytest.approx(upside)
    assert sample["downside"] == pytest.approx(downside)


@pytest.mark.parametrize(
    "amplitudes,mean,median",
    [
        ([0.02, 0.04, 0.06], 0.04, 0.04),
        ([0.02, 0.04, 0.06, 0.08], 0.05, 0.05),
    ],
)
def test_daily_amplitude_mean_and_median_for_odd_and_even_samples(amplitudes, mean, median):
    days = pd.bdate_range("2026-01-05", periods=len(amplitudes))
    frame = pd.DataFrame({
        "timestamp": days,
        "open": 100.0,
        "high": [100.0 + value * 50.0 for value in amplitudes],
        "low": [100.0 - value * 50.0 for value in amplitudes],
        "close": 100.0,
        "preclose": 100.0,
        "tradestatus": 1,
        "volume": 100.0,
    })

    result = calculate_daily_amplitude(frame)

    assert len(result["samples"]) == len(amplitudes)
    assert result["metrics"]["amplitude"]["mean"] == pytest.approx(mean)
    assert result["metrics"]["amplitude"]["median"] == pytest.approx(median)


def test_daily_amplitude_excludes_suspensions_and_invalid_rows_and_falls_back_to_prior_close():
    frame = pd.DataFrame([
        {"timestamp": "2026-01-05", "open": 100.0, "high": 110.0, "low": 90.0,
         "close": 100.0, "preclose": float("nan"), "tradestatus": 1, "volume": 100.0},
        {"timestamp": "2026-01-06", "open": 100.0, "high": 110.0, "low": 100.0,
         "close": 105.0, "preclose": 100.0, "tradestatus": 1, "volume": 0.0},
        {"timestamp": "2026-01-07", "open": 105.0, "high": 200.0, "low": 100.0,
         "close": 999.0, "preclose": 105.0, "tradestatus": 0, "volume": 100.0},
        {"timestamp": "2026-01-08", "open": 100.0, "high": 110.0, "low": 100.0,
         "close": 105.0, "preclose": float("nan"), "tradestatus": 1, "volume": 100.0},
        {"timestamp": "2026-01-09", "open": 100.0, "high": float("nan"), "low": 100.0,
         "close": 100.0, "preclose": 100.0, "tradestatus": 1, "volume": 100.0},
        {"timestamp": "2026-01-12", "open": 100.0, "high": 102.0, "low": 98.0,
         "close": 100.0, "preclose": float("nan"), "tradestatus": 1, "volume": 100.0},
        {"timestamp": "2026-01-13", "open": 100.0, "high": 90.0, "low": 105.0,
         "close": 100.0, "preclose": 100.0, "tradestatus": 1, "volume": 100.0},
        {"timestamp": "2026-01-14", "open": 100.0, "high": 102.0, "low": 98.0,
         "close": 100.0, "preclose": float("nan"), "tradestatus": 1, "volume": 100.0},
    ])

    result = calculate_daily_amplitude(frame)

    assert [sample["date"] for sample in result["samples"]] == [
        "2026-01-06", "2026-01-08", "2026-01-12", "2026-01-14",
    ]
    assert result["excluded_count"] == 4
    assert result["samples"][1]["previous_close"] == pytest.approx(105.0)
    assert result["samples"][2]["previous_close"] == pytest.approx(105.0)
    assert result["samples"][3]["previous_close"] == pytest.approx(100.0)


def test_daily_amplitude_rejects_open_or_close_outside_high_low_and_does_not_reuse_them():
    frame = pd.DataFrame([
        {"timestamp": "2026-01-05", "open": 100.0, "high": 102.0, "low": 98.0,
         "close": 100.0, "preclose": 100.0, "tradestatus": 1},
        {"timestamp": "2026-01-06", "open": 120.0, "high": 110.0, "low": 90.0,
         "close": 80.0, "preclose": 105.0, "tradestatus": 1},
        {"timestamp": "2026-01-07", "open": 100.0, "high": 102.0, "low": 98.0,
         "close": 100.0, "preclose": float("nan"), "tradestatus": 1},
    ])

    result = calculate_daily_amplitude(frame)

    assert [sample["date"] for sample in result["samples"]] == ["2026-01-05", "2026-01-07"]
    assert result["samples"][-1]["previous_close"] == pytest.approx(100.0)
    assert result["excluded_count"] == 1
    assert "超出最高价与最低价范围" in result["excluded_records"][0]["reason"]


def test_daily_amplitude_rejects_even_a_tiny_high_below_low():
    result = calculate_daily_amplitude(pd.DataFrame([{
        "timestamp": "2026-01-05", "open": 1.00000000025,
        "high": 1.0, "low": 1.0000000005, "close": 1.00000000025,
        "preclose": 1.0, "tradestatus": 1,
    }]))

    assert result["samples"] == []
    assert result["excluded_count"] == 1
    assert "最高价低于最低价" in result["excluded_records"][0]["reason"]


def _write_daily(
    path, days, *, closes=None, highs=None, lows=None, precloses=None,
    statuses=None,
):
    count = len(days)
    closes = closes or [100.0] * count
    highs = highs or [102.0] * count
    lows = lows or [98.0] * count
    precloses = precloses or [100.0] * count
    statuses = statuses or [1] * count
    pd.DataFrame({
        "date": days, "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": 1000.0, "turnover": 100000.0,
        "preclose": precloses, "tradestatus": statuses, "is_st": 0,
    }).to_csv(path, index=False)


def _service_with_daily(tmp_path, days, **write_options) -> MarketDataService:
    store = Store(tmp_path / "data")
    service = MarketDataService(store)
    path = tmp_path / "daily.csv"
    _write_daily(path, days, **write_options)
    service.import_file("600000.SH", "1day", path)
    return service


def test_market_amplitude_lookback_and_custom_range_override(tmp_path, monkeypatch):
    days = pd.bdate_range("2026-09-01", periods=5)
    service = _service_with_daily(tmp_path, days)
    monkeypatch.setattr(
        "stockregression.providers.latest_closed_date", lambda: pd.Timestamp(days[-1])
    )

    lookback = service.market_amplitude("600000.SH", lookback=2)
    ranged = service.market_amplitude(
        "600000.SH", lookback=1,
        start_date=days[1].date().isoformat(), end_date=days[3].date().isoformat(),
    )

    assert lookback["frequency"] == "1day"
    assert lookback["requested"] == {"lookback": 2, "start_date": None, "end_date": None}
    assert lookback["sample_count"] == 2
    assert lookback["actual"] == {
        "start_date": days[-2].date().isoformat(),
        "end_date": days[-1].date().isoformat(),
    }
    assert [sample["date"] for sample in lookback["samples"]] == [
        days[-2].date().isoformat(), days[-1].date().isoformat(),
    ]
    assert lookback["samples"][-1]["amplitude"] == pytest.approx(0.04)
    assert lookback["adjustment_anchor"] is None
    assert lookback["adjustment_version"] is None

    assert ranged["requested"] == {
        "lookback": 1,
        "start_date": days[1].date().isoformat(),
        "end_date": days[3].date().isoformat(),
    }
    assert ranged["sample_count"] == 3
    assert ranged["actual"] == {
        "start_date": days[1].date().isoformat(),
        "end_date": days[3].date().isoformat(),
    }


def test_market_amplitude_api_qfq_contract_and_error_mapping(tmp_path, monkeypatch):
    days = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"])
    root = tmp_path / "data"
    service = MarketDataService(Store(root))
    path = tmp_path / "daily.csv"
    _write_daily(
        path, days, closes=[100.0, 50.0, 50.0],
        highs=[110.0, 55.0, 52.0], lows=[90.0, 45.0, 48.0],
        precloses=[100.0, float("nan"), float("nan")],
    )
    service.import_file("600000.SH", "1day", path)
    service.repository.save_auxiliary(
        "600000.SH", "actions", [{
            "ex_date": "2026-09-02", "record_date": "2026-09-01",
            "cash_per_share": 0.0, "bonus_ratio": 1.0,
            "action_type": "split", "implemented": True,
        }], version="fixture", verified=True,
    )
    monkeypatch.setattr(
        "stockregression.providers.latest_closed_date", lambda: pd.Timestamp(days[-1])
    )

    with TestClient(create_app(str(root), start_jobs=False)) as client:
        response = client.get(
            "/api/v1/market/amplitude/600000.SH?lookback=3&adjustment=qfq"
        )
        assert response.status_code == 200, response.text
        payload = response.json()

        assert payload["symbol"] == "600000.SH"
        assert payload["frequency"] == "1day"
        assert payload["adjustment"] == "qfq"
        assert payload["adjustment_anchor"] == "2026-09-03"
        assert payload["adjustment_version"]
        assert payload["requested"] == {"lookback": 3, "start_date": None, "end_date": None}
        assert payload["actual"] == {"start_date": "2026-09-01", "end_date": "2026-09-03"}
        assert payload["sample_count"] == 3
        assert payload["excluded_count"] == 0
        assert set(payload["metrics"]) == {"amplitude", "upside", "downside"}
        assert all(
            set(values) == {"mean", "median"}
            for values in payload["metrics"].values()
        )
        assert [sample["date"] for sample in payload["samples"]] == [
            "2026-09-01", "2026-09-02", "2026-09-03",
        ]
        assert set(payload["samples"][0]) == {
            "date", "previous_close", "high", "low", "amplitude", "upside", "downside",
        }
        assert payload["samples"][1]["previous_close"] == pytest.approx(50.0)
        assert payload["samples"][1]["amplitude"] == pytest.approx(0.20)
        assert payload["metrics"]["amplitude"]["mean"] == pytest.approx(
            (0.20 + 0.20 + 0.08) / 3
        )
        assert isinstance(payload["warnings"], list)

        assert client.get(
            "/api/v1/market/amplitude/600000.SH?start_date=2026-09-01"
        ).status_code == 422
        assert client.get(
            "/api/v1/market/amplitude/600000.SH"
            "?start_date=2026-09-03&end_date=2026-09-01"
        ).status_code == 422

    with TestClient(create_app(str(tmp_path / "empty"), start_jobs=False)) as client:
        assert client.get(
            "/api/v1/market/amplitude/600000.SH"
        ).status_code == 409


def test_market_amplitude_custom_end_truncates_and_reports_warning(tmp_path, monkeypatch):
    days = pd.bdate_range("2026-09-01", periods=3)
    service = _service_with_daily(tmp_path, days)
    monkeypatch.setattr(
        "stockregression.providers.latest_closed_date", lambda: pd.Timestamp(days[-1])
    )

    result = service.market_amplitude(
        "600000.SH", start_date=days[0].date().isoformat(),
        end_date="2026-09-30",
    )

    assert result["actual"]["end_date"] == days[-1].date().isoformat()
    assert any("截断" in warning for warning in result["warnings"])


@pytest.mark.parametrize("adjustment", ["none", "qfq"])
def test_market_amplitude_keeps_excluded_tail_rows_after_qfq_anchor(
    tmp_path, monkeypatch, adjustment,
):
    days = pd.bdate_range("2026-09-01", periods=5)
    service = _service_with_daily(
        tmp_path, days, statuses=[1, 1, 1, 0, 0],
    )
    monkeypatch.setattr(
        "stockregression.providers.latest_closed_date", lambda: pd.Timestamp(days[-1])
    )

    result = service.market_amplitude(
        "600000.SH", start_date=days[0].date().isoformat(),
        end_date=days[-1].date().isoformat(), adjustment=adjustment,
    )

    assert result["sample_count"] == 3
    assert result["excluded_count"] == 2
    assert result["actual"] == {
        "start_date": days[0].date().isoformat(),
        "end_date": days[2].date().isoformat(),
    }
    assert any("已排除 2 条日线记录" in warning for warning in result["warnings"])


def test_market_amplitude_rejects_partial_dates_without_data(tmp_path):
    service = MarketDataService(Store(tmp_path / "data"))

    with pytest.raises(ValueError, match="同时提供"):
        service.market_amplitude("600000.SH", start_date="2026-09-01")
    with pytest.raises(MarketDataError):
        service.market_amplitude("600000.SH")
