from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stockregression import optimization
from stockregression.models import (
    BacktestConfig,
    DataBundle,
    OptimizationConfig,
    QuantityConfig,
    SearchRange,
)


def make_bundle(days: int = 40, *, minute_ratio: float = 1.0, holdout_shift: float = 0.0) -> DataBundle:
    calendar = pd.bdate_range("2024-01-02", periods=days)
    closes = [10.0 + index / 100 for index in range(days)]
    split_at = int(days * 0.8)
    for index in range(split_at, days):
        closes[index] += holdout_shift
    daily = pd.DataFrame({"date": calendar, "close": closes, "preclose": closes})
    bars = pd.DataFrame({
        "timestamp": calendar + pd.to_timedelta(15, unit="hours"),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": 1000,
        "turnover": 10000,
        "frequency": "1m",
        "tradable": True,
        "is_synthetic": False,
        "preclose": closes,
        "is_st": False,
        "limit_up": 0,
        "limit_down": 0,
    })
    validation_start = calendar[int(days * 0.6)]
    actions = pd.DataFrame({
        "record_date": [
            calendar[0].strftime("%Y%m%d"),
            validation_start.strftime("%Y%m%d"),
            calendar[-1].strftime("%Y-%m-%d"),
        ],
        "ex_date": [
            (calendar[1]).strftime("%Y%m%d"),
            (validation_start + pd.to_timedelta(1, unit="days")).strftime("%Y%m%d"),
            calendar[-1].strftime("%Y%m%d"),
        ],
        "cash_per_share": [0.1, 0.2, 0.3],
        "bonus_ratio": [0.0, 0.0, 0.0],
        "implemented": [True, True, True],
        "unsupported_reason": [None, None, None],
    })
    day_quality = [
        {"date": day.date().isoformat(), "mode": "minute", "missing_minutes": 0, "reason": None}
        for day in calendar
    ]
    quality = {
        "minute_ratio": minute_ratio,
        "minute_days": int(days * minute_ratio),
        "daily_days": days - int(days * minute_ratio),
        "suspended_days": 0,
        "synthetic_bars": 0,
        "total_days": days,
        "days": day_quality,
        "warnings": [],
    }
    return DataBundle(
        symbol="600000.SH",
        bars=bars,
        daily=daily,
        actions=actions,
        calendar=calendar,
        quality=quality,
        version="fixture-v1",
    )


def make_config(**overrides) -> OptimizationConfig:
    base = BacktestConfig(
        symbol="600000.SH",
        start_date=date(2024, 1, 2),
        grid_mode="trailing",
    )
    values = {"base": base, "mode": "historical", "n_trials": 4, "seed": 42}
    values.update(overrides)
    return OptimizationConfig(**values)


def fixed_legal_ranges():
    return {
        "buy_step": SearchRange(low=0.02, high=0.02),
        "sell_step": SearchRange(low=0.03, high=0.03),
        "buy_ratio": SearchRange(low=0.1, high=0.1),
        "sell_ratio": SearchRange(low=0.1, high=0.1),
        "initial_position": SearchRange(low=0.5, high=0.5),
        "min_position": SearchRange(low=0.2, high=0.2),
        "max_position": SearchRange(low=0.8, high=0.8),
    }


def fake_run_backtest(config, data, *, progress=None, cancelled=None, summary_only=False):
    if progress:
        progress(0.5, "fixture")
    close_mean = float(data.daily["close"].mean()) if not data.daily.empty else 0.0
    edge = float(config.buy_step if config.grid_mode == "trailing" else config.grid_step)
    metrics = {
        "total_return": 0.05 + edge,
        "cagr": 0.05 + edge,
        "max_drawdown": 0.10,
        "trade_count": 12,
        "total_fees": 1.0,
        "turnover": edge,
        "buy_count": 6,
        "sell_count": 6,
        "mean_close": close_mean,
    }
    return {
        "strategies": {
            "grid": {"metrics": metrics},
            "buy_hold": {"metrics": {"total_return": close_mean}},
            "dca": {"metrics": {"total_return": close_mean / 2}},
        },
        "summary_only": summary_only,
        "dates": [] if data.daily.empty else [str(data.daily["date"].min()), str(data.daily["date"].max())],
    }


def install_fake(monkeypatch):
    calls = []

    def fake(config, data, **kwargs):
        calls.append((config, data, kwargs))
        return fake_run_backtest(config, data, **kwargs)

    monkeypatch.setattr(optimization, "run_backtest", fake)
    return calls


def test_holdout_changes_do_not_change_robust_selection(monkeypatch):
    install_fake(monkeypatch)
    config = make_config(
        mode="robust",
        n_trials=8,
        top_candidates=4,
        min_history_days=20,
        min_trades=10,
    )

    first = optimization.optimize(config, make_bundle())
    second = optimization.optimize(config, make_bundle(holdout_shift=100.0))

    assert first["robust"]["status"] == "completed"
    assert second["robust"]["status"] == "completed"
    assert first["robust"]["selected_params"] == second["robust"]["selected_params"]
    assert first["robust"]["test_metrics"]["mean_close"] != second["robust"]["test_metrics"]["mean_close"]


def test_seed_is_deterministic_and_cash_quantity_becomes_ratio(monkeypatch):
    install_fake(monkeypatch)
    base = BacktestConfig(
        symbol="600000.SH",
        start_date=date(2024, 1, 2),
        buy_quantity=QuantityConfig(mode="cash", value=5000),
        sell_quantity=QuantityConfig(mode="cash", value=7000),
    )
    config = OptimizationConfig(base=base, mode="historical", n_trials=8, seed=7)

    first = optimization.optimize(config, make_bundle())
    second = optimization.optimize(config, make_bundle())
    first_params = [row["params"] for row in first["historical"]["candidates"]]
    second_params = [row["params"] for row in second["historical"]["candidates"]]

    assert first_params == second_params
    assert first["config"] == config.model_dump(mode="json")
    assert first["quality"]["total_days"] == 40
    assert first["data_version"] == "fixture-v1"
    assert first["is_demo"] is False
    assert all(row["buy_quantity"]["mode"] == "equity_ratio" for row in first_params)
    assert all(row["sell_quantity"]["mode"] == "equity_ratio" for row in first_params)
    assert first["historical"]["best_params"]["buy_quantity"]["value"] <= 1


def test_sampled_position_limits_remain_mergeable_and_legal(monkeypatch):
    install_fake(monkeypatch)
    config = make_config(n_trials=2, ranges=fixed_legal_ranges())

    result = optimization.optimize(config, make_bundle())

    assert result["historical"]["best_params"] is not None
    for candidate in result["historical"]["candidates"]:
        assert candidate["status"] == "completed"
        params = candidate["params"]
        assert params["min_position"] <= params["initial_position"] <= params["max_position"]
        assert params["buy_quantity"]["mode"] == "equity_ratio"
        assert params["sell_quantity"]["mode"] == "equity_ratio"


def test_fixed_grid_mode_searches_grid_step(monkeypatch):
    install_fake(monkeypatch)
    base = BacktestConfig(
        symbol="600000.SH",
        start_date=date(2024, 1, 2),
        grid_mode="arithmetic",
    )
    config = OptimizationConfig(
        base=base,
        mode="historical",
        n_trials=2,
        ranges={"grid_step": SearchRange(low=0.04, high=0.04)},
    )

    result = optimization.optimize(config, make_bundle())

    assert all(row["params"]["grid_step"] == 0.04 for row in result["historical"]["candidates"])
    assert all("buy_step" not in row["params"] and "sell_step" not in row["params"] for row in result["historical"]["candidates"])


def test_invalid_position_samples_are_skipped(monkeypatch):
    calls = install_fake(monkeypatch)
    config = make_config(
        n_trials=2,
        ranges={
            "buy_step": SearchRange(low=0.02, high=0.02),
            "sell_step": SearchRange(low=0.02, high=0.02),
            "buy_ratio": SearchRange(low=0.1, high=0.1),
            "sell_ratio": SearchRange(low=0.1, high=0.1),
            "initial_position": SearchRange(low=0.5, high=0.5),
            "min_position": SearchRange(low=0.6, high=0.6),
            "max_position": SearchRange(low=0.8, high=0.8),
        },
    )

    result = optimization.optimize(config, make_bundle())

    assert result["historical"]["best_params"] is None
    assert all(row["status"] == "invalid_constraints" for row in result["historical"]["candidates"])
    assert calls == []


def test_unsupported_range_is_rejected_before_backtests(monkeypatch):
    calls = install_fake(monkeypatch)
    config = make_config(ranges={"mystery_parameter": SearchRange(low=0.1, high=0.2)})

    with pytest.raises(ValueError, match="不支持的搜索参数"):
        optimization.optimize(config, make_bundle())

    assert calls == []


def test_minute_quality_gate_skips_robust_but_keeps_historical(monkeypatch):
    calls = install_fake(monkeypatch)
    config = make_config(
        mode="both",
        n_trials=2,
        min_history_days=20,
        min_minute_ratio=0.95,
        ranges=fixed_legal_ranges(),
    )

    result = optimization.optimize(config, make_bundle(minute_ratio=0.8))

    assert result["historical"]["best_params"] is not None
    assert result["robust"]["status"] == "unavailable"
    assert "保留率不足" in result["robust"]["reason"]
    assert len(calls) == 3  # two trials and one full historical result


def test_split_is_chronological_and_slice_filters_actions_and_quality_days():
    data = make_bundle()
    split = optimization._split_dates(data)

    assert split == {
        "train": {"start": "2024-01-02", "end": data.calendar[23].date().isoformat(), "days": 24},
        "validation": {"start": data.calendar[24].date().isoformat(), "end": data.calendar[31].date().isoformat(), "days": 8},
        "test": {"start": data.calendar[32].date().isoformat(), "end": data.calendar[-1].date().isoformat(), "days": 8},
    }
    validation_start = pd.Timestamp(split["validation"]["start"])
    validation_end = pd.Timestamp(split["validation"]["end"])
    sliced = optimization._slice_bundle(data, validation_start, validation_end)

    assert sliced.calendar[0] == validation_start
    assert sliced.calendar[-1] == validation_end
    assert len(sliced.daily) == 8
    assert sliced.quality["total_days"] == 8
    assert sliced.quality["minute_ratio"] == 1.0
    assert len(sliced.actions) == 1
    assert sliced.actions.iloc[0]["record_date"] == validation_start.strftime("%Y%m%d")


def test_short_history_keeps_historical_and_marks_robust_unavailable(monkeypatch):
    install_fake(monkeypatch)
    data = make_bundle(days=19)
    config = make_config(mode="both", n_trials=2, min_history_days=20)

    result = optimization.optimize(config, data)

    assert result["historical"]["status"] == "completed"
    assert result["robust"]["status"] == "unavailable"
    assert "历史交易日不足" in result["robust"]["reason"]


def test_cancellation_preserves_completed_trials(monkeypatch):
    install_fake(monkeypatch)
    config = make_config(n_trials=10, ranges=fixed_legal_ranges())
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3

    result = optimization.optimize(config, make_bundle(), cancelled=cancelled)

    assert result["historical"]["status"] == "cancelled"
    assert result["historical"]["best_params"] is not None
    assert len(result["historical"]["candidates"]) == 2
    assert all(row["status"] == "completed" for row in result["historical"]["candidates"])
    assert result["historical"]["best_result"] is None
