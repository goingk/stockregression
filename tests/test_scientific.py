from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockregression import scientific
from stockregression.models import BacktestConfig, DataBundle, OptimizationConfig, SearchRange


def make_bundle(days=504):
    calendar = pd.bdate_range("2022-01-03", periods=days)
    close = np.linspace(10.0, 12.0, days)
    daily = pd.DataFrame({"date": calendar, "close": close, "preclose": close})
    bars = pd.DataFrame({
        "timestamp": calendar + pd.Timedelta(hours=15),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1_000,
        "turnover": close * 1_000,
        "frequency": "1d",
        "tradable": True,
        "is_synthetic": False,
        "preclose": close,
    })
    return DataBundle(
        symbol="159611.SZ",
        bars=bars,
        daily=daily,
        calendar=calendar,
        quality={"minute_ratio": 1.0, "days": []},
    )


def make_config(**overrides):
    base = BacktestConfig(
        schema_version=2,
        symbol="159611.SZ",
        start_date=date(2023, 12, 1),
        initial_shares=1_000,
        min_shares=0,
        max_shares=2_000,
        buy_shares=100,
        sell_shares=100,
        frequency="1d",
    )
    values = {
        "base": base,
        "n_trials": 2,
        "seed": 4,
        "ranges": {
            "buy_step": SearchRange(low=0.02, high=0.04, step=0.01),
            "sell_step": SearchRange(low=0.03, high=0.03),
            "min_shares": SearchRange(low=0, high=1_000, step=100),
            "max_shares": SearchRange(low=1_000, high=2_000, step=100),
            "buy_shares": SearchRange(low=100, high=200, step=100),
            "sell_shares": SearchRange(low=100, high=200, step=100),
        },
    }
    values.update(overrides)
    return OptimizationConfig(**values)


def test_lookback_split_uses_generation_then_four_chronological_folds():
    dates = pd.bdate_range("2020-01-01", periods=504)
    generation, folds, split, reasons = scientific._folds_for_history(dates)

    assert split is True and reasons == []
    assert len(generation) == 252
    assert [len(pd.bdate_range(start, end)) for start, end in folds] == [63] * 4
    assert generation[-1] < folds[0][0]


def test_short_history_is_used_for_relative_selection_and_marks_insufficient_evidence():
    dates = pd.bdate_range("2024-01-01", periods=120)
    generation, folds, split, reasons = scientific._folds_for_history(dates)

    assert not split
    assert generation.equals(dates)
    assert folds == []
    assert any("不足504" in reason for reason in reasons)
    assert any("样本内" in reason for reason in reasons)


def test_middle_history_splits_first_half_for_generation_and_four_validation_folds():
    dates = pd.bdate_range("2024-01-01", periods=200)
    generation, folds, split, reasons = scientific._folds_for_history(dates)

    assert split
    assert len(generation) == 100
    assert len(folds) == 4
    assert generation[-1] < folds[0][0]
    assert any("不足504" in reason for reason in reasons)


def test_candidate_family_expands_each_dimension_and_keeps_fixed_initial_shares():
    config = make_config()
    base = config.base
    names = scientific._active_names(base)
    center = {
        "buy_step": 0.03,
        "sell_step": 0.03,
        "min_shares": 500,
        "max_shares": 1_500,
        "buy_shares": 100,
        "sell_shares": 100,
        "initial_shares": 1_000,
    }

    rows, neighborhoods = scientific._candidate_family([center], config, base, names)

    assert len(rows) > 1
    assert all(row["params"]["initial_shares"] == 1_000 for row in rows)
    assert all(row["params"]["min_shares"] <= 1_000 <= row["params"]["max_shares"] for row in rows)
    center_key = scientific._params_key(center, names)
    assert len(neighborhoods[center_key]) == 9
    assert all(row["params"]["buy_step"] in (0.02, 0.03, 0.04) for row in rows)


def test_window_score_is_annualized_growth_minus_drawdown():
    values = np.array([0.01, -0.05, 0.02])

    score = scientific._daily_window_score(values)
    annualized = np.prod(1 + values) ** (252 / len(values)) - 1
    equity = np.r_[1.0, np.cumprod(1 + values)]
    drawdown = np.max((np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity))

    assert score == pytest.approx(annualized - drawdown)


def test_v2_protocol_returns_fixed_adaptive_and_out_of_sample_evidence(monkeypatch):
    data = make_bundle(380)
    eval_start_index = 250
    base = make_config().base.model_copy(update={
        "start_date": data.calendar[eval_start_index].date(),
        "end_date": data.calendar[-1].date(),
    })
    config = make_config(
        base=base,
        protocol_version=2,
        selection_target="both",
        n_trials=2,
        rebalance_days=63,
        bootstrap_reps=12,
        min_trades=1,
        max_drawdown=0.5,
        ranges={
            "buy_step": SearchRange(low=0.02, high=0.04, step=0.01),
            "sell_step": SearchRange(low=0.03, high=0.03),
        },
    )
    calls = []
    progress_events = []

    def fake_batch(candidate_base, bundle, params, fold_ranges, cost_multiplier=1.0, cancelled=None):
        dates = pd.DatetimeIndex(bundle.calendar)
        fold_count = len(fold_ranges)
        matrix = np.column_stack([
            np.full(len(dates), 0.0005 + float(item["buy_step"]) * 0.01 - (cost_multiplier - 1) * 0.0001)
            for item in params
        ])
        folds = np.full((len(params), fold_count), 0.05)
        fold_metrics = [[{
            "max_drawdown": 0.05,
            "trade_count": 20,
            "buy_count": 10,
            "sell_count": 10,
            "total_fees": 1.0 * cost_multiplier,
            "turnover": 0.5,
        } for _ in range(fold_count)] for _ in params]
        calls.append((dates[0], dates[-1], list(params), list(fold_ranges), cost_multiplier))
        return {"dates": dates, "daily_returns": matrix, "fold_drawdowns": folds, "fold_metrics": fold_metrics,
                "metrics": [{} for _ in params]}

    def fake_run(candidate_base, bundle, progress=None, cancelled=None, parameter_schedule=None, **kwargs):
        dates = pd.DatetimeIndex(bundle.calendar)
        if progress:
            progress(0.2, "loaded")
            progress(1.0, "inner backtest completed")
        return {"strategies": {"grid": {"period_returns": [
            {"timestamp": (day + pd.Timedelta(hours=15)).isoformat(), "return": 0.001}
            for day in dates if day >= pd.Timestamp(candidate_base.start_date)
        ]}}, "parameter_schedule": parameter_schedule}

    monkeypatch.setattr(scientific, "_research_batch", fake_batch)
    monkeypatch.setattr(scientific, "_run_share_backtest", fake_run)
    monkeypatch.setattr(scientific, "_run_passive_backtest", lambda candidate_base, bundle, progress=None, cancelled=None: {
        "strategies": {"grid": {"period_returns": [
            {"timestamp": (day + pd.Timedelta(hours=15)).isoformat(), "return": 0.0}
            for day in pd.DatetimeIndex(bundle.calendar) if day >= pd.Timestamp(candidate_base.start_date)
        ]}}
    })
    monkeypatch.setattr(scientific, "_passive_returns", lambda candidate_base, bundle: (pd.DatetimeIndex(bundle.calendar[-130:]), np.zeros(min(130, len(bundle.calendar)))))

    result = scientific.optimize_scientific(config, data, progress=lambda fraction, message: progress_events.append((fraction, message)))

    assert result["status"] == "completed"
    assert [fraction for fraction, _ in progress_events] == sorted(fraction for fraction, _ in progress_events)
    assert [(fraction, message) for fraction, message in progress_events if fraction >= 1] == [(1.0, "科学参数选择完成")]
    assert result["protocol_version"] == 2
    assert result["fixed"]["params"]["initial_shares"] == base.initial_shares
    assert result["fixed"]["test_result"] is not None
    assert len(result["adaptive"]["schedule"]) == 3
    assert result["evidence"]["decisions"][0]["reused_from_fixed"] is True
    assert result["adaptive"]["active_params"] == result["adaptive"]["schedule"][-1]["params"]
    assert result["protocol"]["evaluation_start"] == data.calendar[eval_start_index].date().isoformat()
    assert result["adaptive"]["test_result"]["parameter_schedule"] == result["adaptive"]["schedule"]
    assert "same_account_passive" in result["adaptive"]["test_result"]["strategies"]
    assert "same_account_passive" in result["fixed"]["test_result"]["strategies"]
    assert result["fixed"]["current"]["as_of"] == data.calendar[-1].date().isoformat()
    assert result["adaptive"]["latest_params"] == result["fixed"]["current"]["params"]
    assert result["adaptive"]["latest_as_of"] == result["fixed"]["current"]["as_of"]
    assert result["evidence"]["final_126_day_check"]["days"] == 126
    assert set(result["evidence"]["final_126_day_check"]["bootstrap"]["sensitivity"]) == {"10", "20", "40"}
    assert result["computation"]["candidate_count"] >= len(result["adaptive"]["schedule"])
    assert any(call[-1] == 2.0 for call in calls)
    assert result["evidence"]["final_126_day_check"]["source"] == "actual_continuous_fixed_and_adaptive_oos_runs"
    assert set(result["evidence"]["final_126_day_check"]["bootstrap_family"]) == {"fixed", "adaptive"}
    tail = pd.DatetimeIndex(data.calendar)[-126:]
    assert not any(call[3] == [(tail[0].date().isoformat(), tail[-1].date().isoformat())] for call in calls)
    archives = result["evidence"]["candidate_archives"]
    unique_archives = [archives["fixed"], archives["current"], *[
        archive["candidate_archive"] for decision, archive in zip(result["evidence"]["decisions"], archives["adaptive"])
        if not decision["reused_from_fixed"]
    ]]
    assert result["computation"]["unique_selection_calls"] == 4
    assert result["computation"]["neighborhood_count"] == sum(not row["is_center"] for archive in unique_archives for row in archive)
    adaptive_only = scientific.optimize_scientific(config.model_copy(update={"selection_target": "adaptive"}), data)
    assert adaptive_only["computation"]["unique_selection_calls"] == len(adaptive_only["adaptive"]["schedule"]) + 1


def test_validation_cancel_preserves_trials_and_marks_original_cost_only_candidates_ineligible(monkeypatch):
    data = make_bundle(504)
    config = make_config(n_trials=2, min_trades=1, max_drawdown=0.5)
    calls = []

    def fake_batch(candidate_base, bundle, params, fold_ranges, cost_multiplier=1.0, cancelled=None):
        calls.append((cost_multiplier, len(fold_ranges)))
        if cost_multiplier == 2.0 and len(fold_ranges) == 4:
            raise InterruptedError("cancelled during stress validation")
        dates = pd.DatetimeIndex(bundle.calendar)
        matrix = np.full((len(dates), len(params)), 0.001)
        drawdowns = np.full((len(params), len(fold_ranges)), 0.05)
        fold_metrics = [[{
            "trade_count": 8, "buy_count": 4, "sell_count": 4,
            "total_fees": cost_multiplier, "turnover": 0.1,
        } for _ in fold_ranges] for _ in params]
        return {"dates": dates, "daily_returns": matrix, "fold_drawdowns": drawdowns, "fold_metrics": fold_metrics}

    monkeypatch.setattr(scientific, "_research_batch", fake_batch)
    selected = scientific._select_at(
        config, config.base, data, pd.Timestamp(data.calendar[-1]), reserve_cash=None,
        seed_offset=0, progress=None, cancelled=None,
    )

    assert calls[0] == (1.0, 1)  # generation is evaluated only on its own segment
    assert (1.0, 4) in calls and (2.0, 4) in calls
    assert selected["status"] == "cancelled"
    assert selected["trial_archive"]
    assert any(row["status"] == "completed" for row in selected["trial_archive"])
    partial = [row for row in selected["candidate_archive"] if row["evaluation_status"] == "cancelled_partial_cost"]
    assert partial
    assert all(not row["eligible"] for row in partial)
    assert all(row["folds"] and not row["stress_folds"] for row in partial)


def test_zero_oos_returns_structured_insufficient_result_without_using_future_dates():
    data = make_bundle(20)
    base = make_config().base.model_copy(update={
        "start_date": date(2020, 1, 1),
        "end_date": date(2020, 12, 31),
    })
    config = make_config(base=base, protocol_version=2, selection_target="both", n_trials=2)

    result = scientific.optimize_scientific(config, data)

    assert result["status"] == "partial"
    assert result["evidence"]["evaluation_days"] == 0
    assert result["evidence"]["evaluation_start"] == "2020-01-01"
    assert result["fixed"]["test_result"] is None
    assert result["adaptive"]["schedule"] == []
    assert result["evidence"]["final_126_day_check"]["status"] == "insufficient_history"
    assert "没有使用结束日期之后的行情" in result["fixed"]["reasons"][-1]


def test_current_family_bootstrap_uses_full_internal_candidate_family_and_labels_it_non_oos(monkeypatch):
    data = make_bundle(200)
    config = make_config(bootstrap_reps=20)
    dates = pd.DatetimeIndex(data.calendar)
    active_names = scientific._active_names(config.base)
    candidates = []
    internal_rows = []
    for index, step in enumerate((0.02, 0.03, 0.04), start=1):
        params = {name: getattr(config.base, name) for name in active_names}
        params["buy_step"] = step
        params["initial_shares"] = config.base.initial_shares
        candidates.append({
            "candidate_id": index, "params": params, "evaluation_status": "completed",
            "is_center": True, "eligible": True,
        })
        internal_rows.append({
            "candidate_id": index,
            "_current_validation_dates": dates,
            "_current_validation_returns": np.full(len(dates), 0.0002 * index),
        })
    folds = []
    for part in np.array_split(dates[100:], 4):
        folds.append({"start": part[0].date().isoformat(), "end": part[-1].date().isoformat()})
    current = {
        "as_of": dates[-1].date().isoformat(),
        "params": candidates[1]["params"],
        "diagnostics": {"selection_history_days": 200, "validation_folds": folds},
        "_family_rows": internal_rows,
    }
    monkeypatch.setattr(scientific, "_passive_returns", lambda _base, _data: (dates, np.zeros(len(dates))))

    result = scientific.evaluate_current_family(config, data, current, candidates, reserve_cash=None)

    assert result["status"] == "completed"
    assert result["family_count"] == 3
    assert result["sample_role"] == "internal_validation_used_for_selection"
    assert result["independent_oos"] is False
    assert result["matrix_source"] == "reused_original_cost_validation_matrix"
    assert result["bootstrap"]["primary"]["candidate_count"] == 3

