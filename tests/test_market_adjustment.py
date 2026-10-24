import pandas as pd
import pytest

from stockregression.market_data.chart_adjustment import apply_chart_adjustment


@pytest.mark.parametrize("timeframe", ["1week", "1month"])
def test_chart_service_uses_local_anchor_and_rebuilds_partial_period_without_future_prices(tmp_path, timeframe):
    from stockregression.market_data import MarketDataService
    from stockregression.settings import Settings
    from stockregression.storage import Store

    settings = Settings(data_dir=str(tmp_path / "data"), package_dir=str(tmp_path / "package"))
    service = MarketDataService(Store(settings.data_dir, settings=settings), package_dir=settings.package_dir, settings=settings)
    days = pd.to_datetime(["2024-01-25", "2024-01-26", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
    closes = [100., 100., 100., 50., 50., 50., 600.]
    daily = pd.DataFrame({"date": days, "open": closes, "close": closes,
                          "high": [p + 1 for p in closes], "low": [p - 1 for p in closes],
                          "volume": 10., "turnover": 1000., "preclose": [100., 100., 100., 50., 50., 50., 50.],
                          "tradestatus": 1, "is_st": False})
    path = tmp_path / "daily.csv"
    daily.to_csv(path, index=False)
    service.import_file("159995.SZ", "1day", path)
    result = service.market_bars("159995.SZ", timeframe, end="2024-01-31", adjustment="qfq", limit=1, ma_periods=[2], materialize=False)
    assert result["adjustment_anchor"] == "2024-01-31"
    assert len(result["bars"]) == 1
    bar = result["bars"][0]
    assert pd.Timestamp(bar["timestamp"]) == pd.Timestamp("2024-01-31")
    assert bar["close"] == 50.
    assert bar["high"] <= 51.
    if timeframe == "1week":
        assert bar["ma2"] == 50.
        earlier = service.market_bars("159995.SZ", timeframe, before=result["next_before"], adjustment="qfq", adjustment_anchor=result["adjustment_anchor"], limit=1, ma_periods=[2], materialize=False)
        assert earlier["adjustment_version"] == result["adjustment_version"]
        assert earlier["adjustment_anchor"] == result["adjustment_anchor"]
        assert pd.Timestamp(earlier["bars"][-1]["timestamp"]) < pd.Timestamp(bar["timestamp"])
    weekend = service.market_bars("159995.SZ", "1day", end="2024-02-04", adjustment="qfq", materialize=False)
    assert weekend["adjustment_anchor"] == "2024-02-02"


class _Store:
    def instrument(self, symbol):
        return {"asset_type": "stock"}


def test_chart_adjustment_keeps_input_unchanged_and_preserves_turnover(monkeypatch):
    from stockregression.market_data import chart_adjustment

    class Repository:
        def dataset(self, symbol, timeframe):
            return {"current_version": 1}

        def active_rows(self, symbol, timeframe, **kwargs):
            return daily.copy()

        def auxiliary(self, *args, **kwargs):
            return None

    class Service:
        repository = Repository()
        store = _Store()

    daily = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [100., 50.], "high": [101., 51.], "low": [99., 49.],
        "close": [100., 50.], "preclose": [99., 50.], "volume": [1000., 1100.],
        "turnover": [100000., 55000.],
    })
    original = daily.copy(deep=True)
    incoming = daily.copy()
    adjusted, error, version = apply_chart_adjustment(
        Service(), "600000.SH", "1day", incoming,
        anchor=pd.Timestamp("2024-01-03"),
    )
    assert error is None
    assert version
    assert adjusted.loc[0, "close"] == 50
    assert adjusted.loc[1, "close"] == 50
    assert adjusted.turnover.tolist() == incoming.turnover.tolist()
    pd.testing.assert_frame_equal(daily, original)
    pd.testing.assert_frame_equal(incoming, daily)


def test_weekly_adjustment_rebuilds_from_daily(monkeypatch):
    from stockregression.market_data import chart_adjustment

    class Repository:
        def dataset(self, symbol, timeframe):
            return {"current_version": 2}

        def active_rows(self, symbol, timeframe, **kwargs):
            return daily.copy()

        def auxiliary(self, *args, **kwargs):
            return None

    class Service:
        repository = Repository()
        store = _Store()

    daily = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "open": [100., 100., 50.], "high": [102., 101., 52.],
        "low": [99., 98., 49.], "close": [100., 100., 50.],
        "preclose": [99., 100., 50.], "volume": [1., 2., 3.],
        "turnover": [100., 200., 150.],
    })
    result, error, _ = apply_chart_adjustment(
        Service(), "600000.SH", "1week", daily.iloc[[2]].copy(),
        anchor=pd.Timestamp("2024-01-03"),
    )
    assert error is None
    # Earlier sessions are adjusted independently before weekly OHLC aggregation.
    assert result.iloc[0].open == 50
    assert result.iloc[0].close == 50
    assert result.iloc[0].turnover == 450


def test_monthly_adjustment_applies_factors_per_day_across_event(monkeypatch):
    daily = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]),
        "open": [100., 90., 45.], "high": [101., 91., 46.],
        "low": [99., 89., 44.], "close": [100., 90., 45.],
        "preclose": [99., 90., 45.], "volume": [10., 20., 30.],
        "turnover": [1000., 1800., 1350.],
    })

    class Repository:
        daily_version = 7

        def __init__(self):
            self.rows = daily.copy()

        def dataset(self, symbol, timeframe):
            return {"current_version": self.daily_version}

        def active_rows(self, symbol, timeframe, **kwargs):
            end = kwargs.get("end")
            values = self.rows.copy()
            if end is not None:
                values = values[values.timestamp < pd.Timestamp(end).normalize() + pd.Timedelta(days=1)]
            return values

        def auxiliary(self, *args, **kwargs):
            return {"version": "actions-v2", "payload": []}

    class Service:
        repository = Repository()
        store = _Store()

    repo = Service.repository
    result, error, version = apply_chart_adjustment(
        Service(), "600000.SH", "1month", daily.copy(),
        anchor=pd.Timestamp("2024-02-01"), start="2024-02-01",
    )
    assert error is None
    assert len(result) == 2
    assert result.iloc[0].open == 45
    assert result.iloc[0].close == 45
    assert result.iloc[0].turnover == 2800
    assert result.iloc[1].close == 45
    assert result.iloc[1].turnover == 1350
    # The page start is not part of the factor/version identity.
    _, _, same_version = apply_chart_adjustment(
        Service(), "600000.SH", "1month", daily.copy(),
        anchor=pd.Timestamp("2024-02-01"), start="2024-01-31",
    )
    assert same_version == version
    repo.daily_version = 8
    repo.rows = pd.concat([daily, pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-02-02")], "open": [46.], "high": [47.],
        "low": [45.], "close": [46.], "preclose": [45.], "volume": [5.], "turnover": [230.],
    })], ignore_index=True)
    _, _, changed_version = apply_chart_adjustment(
        Service(), "600000.SH", "1month", daily.copy(),
        anchor=pd.Timestamp("2024-02-01"),
    )
    assert changed_version != version


def test_cash_dividend_and_consecutive_splits_compound(monkeypatch):
    from stockregression.market_data import chart_adjustment

    daily = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "open": [100., 90., 45., 60.], "high": [101., 91., 46., 61.],
        "low": [99., 89., 44., 59.], "close": [100., 90., 45., 60.],
        "preclose": [99., 90., 45., 60.], "volume": [10., 20., 30., 40.],
        "turnover": [1000., 1800., 1350., 2400.],
    })
    actions = pd.DataFrame([
        {"ex_date": "2024-01-03", "cash_per_share": 10., "bonus_ratio": 0.},
        {"ex_date": "2024-01-04", "cash_per_share": 0., "bonus_ratio": 1.},
        {"ex_date": "2024-01-05", "cash_per_share": 0., "bonus_ratio": 1/3},
    ])

    class Repository:
        def dataset(self, symbol, timeframe):
            return {"current_version": 1}

        def active_rows(self, symbol, timeframe, **kwargs):
            return daily.copy()

        def auxiliary(self, *args, **kwargs):
            return {"version": "events-1", "payload": actions.to_dict("records")}

    class Service:
        repository = Repository()
        store = _Store()

    adjusted, error, _ = apply_chart_adjustment(
        Service(), "600000.SH", "1day", daily.copy(),
        anchor=pd.Timestamp("2024-01-05"),
    )
    assert error is None
    # 0.9 cash factor, then 0.5 and 0.75 consecutive split factors.
    assert adjusted.close.tolist() == [33.75, 33.75, 33.75, 60.]
    assert adjusted.volume.tolist() == daily.volume.tolist()
    assert adjusted.turnover.tolist() == daily.turnover.tolist()


def test_reference_only_factor_is_not_rejected_by_action_issues(monkeypatch):
    from stockregression.market_data import chart_adjustment

    daily = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [100., 90.], "high": [101., 91.], "low": [99., 89.],
        "close": [100., 90.], "preclose": [99., 90.],
        "volume": [10., 10.], "turnover": [1000., 900.],
    })

    class Repository:
        def dataset(self, symbol, timeframe):
            return {"current_version": 1}

        def active_rows(self, symbol, timeframe, **kwargs):
            return daily.copy()

        def auxiliary(self, *args, **kwargs):
            return {"version": "empty", "payload": []}

    class Service:
        repository = Repository()
        store = _Store()

    result, error, _ = apply_chart_adjustment(
        Service(), "600000.SH", "1day", daily.copy(),
        anchor=pd.Timestamp("2024-01-03"), start="2024-01-02",
    )
    assert error is None
    assert result.close.tolist() == [90., 90.]
