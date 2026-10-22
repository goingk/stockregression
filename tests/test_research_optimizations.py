"""Optimised research compute paths.

Three changes are pinned down here:

* the decision-level ledger count is the decision's own work.  One task
  budget is shared by the fixed, current and adaptive decisions, so a
  decision that already finished must keep reporting its own count instead of
  the task-wide total;
* the research batch simulates the grid ledger and its matched-funding
  baseline only.  Those two ledgers must stay numerically identical to the
  ordinary four-account backtest, including the mirrored contributions,
  corporate actions, fills and every reported statistic;
* the share input preparation derives the bar opening times and the daily
  price bands in bulk.  The vectorised results must be bit-identical to the
  per-bar and per-day rules they replaced, and the fills must not move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockregression import engine, share_engine
from stockregression.models import (
    BacktestConfig, DataBundle, FloorPenaltyConfig, OptimizationConfig,
    ResearchStrategySpec,
)
from stockregression.providers import normalize_actions
from stockregression.research import robust as robust_research
from stockregression.research.robust import EvaluationBudget, EvaluationSession
from stockregression.research.space import ROBUST_FIELDS, ParameterSpace
from stockregression.share_engine import research_batch, run_share_backtest
from stockregression.share_kernel import (
    ACCOUNT_MODE_FULL, ACCOUNT_MODE_RESEARCH, simulate,
)


# ---------------------------------------------------------------------------
# 1. Decision-level ledger counting
# ---------------------------------------------------------------------------


def _optimization_config(**overrides) -> OptimizationConfig:
    base = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date="2023-01-02", frequency="1d",
        initial_shares=1000, min_shares=0, max_shares=None, buy_shares=100, sell_shares=100,
        grid_mode="geometric", buy_step=0.01, sell_step=0.01,
    )
    strategy = {
        "strategy_version": 5, "search_method": "hierarchical", "ranking_mode": "robust_j",
        "granularity": {"buy_step": 0.005, "sell_step": 0.005, "buy_shares": 100, "sell_shares": 100},
        "hypercube_edge": 3, "training_budget": 1, "validation_budget": 3, "reporting_budget": 2,
        "recommendation_count": 1, "stress_region_count": 1, "stress_cost_multiplier": 2.0,
        "starts_per_candidate": 1, "start_min": 1, "start_max": 1,
        "training_min_days": 250, "training_max_days": 500,
        **overrides,
    }
    return OptimizationConfig(
        base=base, protocol_version=5,
        strategy=ResearchStrategySpec.model_validate(strategy),
        ranges={
            "buy_step": {"low": 0.005, "high": 0.015, "step": 0.005},
            "sell_step": {"low": 0.005, "high": 0.005, "step": 0.005},
            "buy_shares": {"low": 100, "high": 100, "step": 100},
            "sell_shares": {"low": 100, "high": 100, "step": 100},
        },
    )


class _CountingEvaluator:
    """Deterministic evaluator with a strictly monotonic return ladder."""

    chunk_size = 4

    def __init__(self, space: ParameterSpace):
        self.space = space
        self.calls: list[list[tuple[int, ...]]] = []
        self.stress_calls: list[tuple[list[tuple[int, ...]], float]] = []

    def _row(self, coordinate: tuple[int, ...]) -> dict:
        return {
            "key": coordinate, "coordinate": list(coordinate),
            "params": self.space.params(coordinate),
            "evaluation_status": "completed",
            "training_excess_return": 0.001 * sum(int(value) for value in coordinate),
            "max_principal_drawdown": 0.05, "trade_count": 20,
            "buy_count": 10, "sell_count": 10,
            "actual_max_pos_ratio": 1.0, "actual_min_pos_ratio": 0.5,
            "floor_episode_count": 0, "floor_episode_grids": [], "floor_penalty": 0.0,
        }

    def evaluate_training(self, chunk):
        self.calls.append([tuple(item) for item in chunk])
        return [self._row(tuple(item)) for item in chunk]

    def evaluate_training_stress(self, points, cost_multiplier=1.0):
        listed = [tuple(item) for item in points]
        self.stress_calls.append((listed, float(cost_multiplier)))
        return {
            point: {"training_excess_stress": 0.001 * sum(int(value) for value in point)
                    - 0.001 * float(cost_multiplier)}
            for point in listed
        }


def _session(budget: EvaluationBudget | None = None) -> EvaluationSession:
    config = _optimization_config()
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    return EvaluationSession(_CountingEvaluator(space), space, config, None, budget=budget)


def test_two_sessions_sharing_a_task_budget_count_their_own_ledger():
    budget = EvaluationBudget(total=6, max_runtime_seconds=600)
    first = _session(budget)
    second = _session(budget)
    coordinates = list(first.space.iterate())
    assert len(coordinates) == 3

    first.ensure(coordinates[:2], "training", 2)

    assert first.spent["training"] == 2
    assert first.ledger_evaluations == 4
    assert budget.consumed == 2

    second.ensure(coordinates[:1], "validation", 1)

    assert second.spent["validation"] == 1
    assert second.ledger_evaluations == 2
    # The finished decision keeps its own count.  A shared task budget must
    # never inflate the ledger of a decision that already ran.
    assert first.ledger_evaluations == 4
    assert first.ledger_evaluations != 2 * budget.consumed
    # The task total is still correct: it is the sum of both sessions.
    assert budget.consumed == 3
    assert first.ledger_evaluations + second.ledger_evaluations == 2 * budget.consumed


def test_ledger_count_covers_the_stress_batch_of_its_own_session_only():
    budget = EvaluationBudget(total=10, max_runtime_seconds=600)
    first = _session(budget)
    second = _session(budget)
    coordinates = list(first.space.iterate())

    first.ensure(coordinates[:1], "training", 1)
    second.ensure(coordinates[:2], "validation", 2)
    first.stress_spent += 3

    assert first.ledger_evaluations == 2 * (sum(first.spent.values()) + 3)
    assert second.ledger_evaluations == 2 * (sum(second.spent.values()) + 0)
    assert first.ledger_evaluations + second.ledger_evaluations == 2 * (budget.consumed + 3)


def _flat_market(days: int = 600) -> DataBundle:
    dates = pd.bdate_range("2020-01-02", periods=days)
    close = [10.0] * len(dates)
    daily = pd.DataFrame({
        "date": dates, "open": close, "high": [value + 0.5 for value in close],
        "low": [value - 0.5 for value in close], "close": close,
        "preclose": [close[0]] + close[:-1], "volume": 100000,
        "turnover": [value * 100000 for value in close], "tradestatus": 1, "is_st": False,
    })
    return DataBundle(
        symbol="600000.SH", bars=pd.DataFrame(), daily=daily,
        calendar=pd.DatetimeIndex(dates), version="flat-fixture",
    )


def test_decision_ledger_counts_stay_local_while_the_task_total_stays_global(monkeypatch):
    """Two decisions share one task budget; each reports its own work."""
    config = _optimization_config(total_evaluation_budget=40)
    base = config.base
    data = _flat_market()
    asof = pd.Timestamp(data.calendar[-1])
    space = ParameterSpace(config, field_order=ROBUST_FIELDS)
    monkeypatch.setattr(
        robust_research, "BatchEvaluator", lambda *args, **kwargs: _CountingEvaluator(space))

    budget = EvaluationBudget(total=40, max_runtime_seconds=600)
    first = robust_research.select_robust_at(config, base, data, asof, budget=budget)
    second = robust_research.select_robust_at(config, base, data, asof, budget=budget)

    for result in (first, second):
        diagnostics = result["diagnostics"]
        own = (
            diagnostics["training_unique_count"] + diagnostics["neighborhood_unique_count"]
            + diagnostics["reporting_unique_count"] + diagnostics["stress_unique_count"]
        )
        assert diagnostics["ledger_evaluation_count"] == 2 * own
    assert first["diagnostics"]["ledger_evaluation_count"] < 2 * budget.consumed
    assert (
        first["diagnostics"]["ledger_evaluation_count"]
        + second["diagnostics"]["ledger_evaluation_count"] == 2 * budget.consumed
    )


# ---------------------------------------------------------------------------
# 2. Research batch ledgers: grid and matched funding hold only
# ---------------------------------------------------------------------------


def _share_market(days: int = 6):
    """A v3 share account with intraday bars, a corporate action and a floor."""
    dates = pd.bdate_range("2024-01-02", periods=days)
    daily_prices = [10.0, 10.0, 9.0, 8.5, 10.5, 9.2, 11.0][:days]
    daily = pd.DataFrame({
        "date": dates, "open": daily_prices, "close": daily_prices,
        "high": [value + .01 for value in daily_prices],
        "low": [value - .01 for value in daily_prices],
        "preclose": np.r_[10.0, daily_prices[:-1]],
        "volume": 100000, "turnover": [value * 100000 for value in daily_prices],
        "tradestatus": 1, "is_st": False, "limit_up": 100.0, "limit_down": 0.01,
    })
    records = []
    for day, price in zip(dates, daily_prices):
        # The path sells into the morning high and buys back on the way down,
        # so the grid needs an external contribution and the matched account
        # mirrors it.
        for minute, factor in enumerate([1.0, 1.25, 0.7, 0.6]):
            value = price * factor
            records.append({
                "timestamp": day + pd.Timedelta(hours=9, minutes=31 + minute),
                "open": value, "high": value, "low": value, "close": value,
                "volume": 10000, "frequency": "1m", "tradable": True, "is_synthetic": False,
            })
    bars = pd.DataFrame(records)
    actions = normalize_actions(pd.DataFrame([{
        "record_date": str(dates[1].date()), "ex_date": str(dates[2].date()),
        "pay_date": str(dates[3].date()), "stock_date": str(dates[4].date()),
        "cash_per_share": .6, "bonus_ratio": .1, "implemented": True,
    }]))
    config = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date=dates[1].date(),
        end_date=dates[-1].date(), frequency="1m", initial_shares=1000,
        min_shares=0, max_shares=None, buy_shares=1000, sell_shares=100,
        grid_mode="geometric", buy_step=.03, sell_step=.03,
        reference_price_mode="manual", reference_price=10.0,
        fees={"commission_rate": 0, "min_commission": 0, "slippage_bps": 0},
        floor_penalty_config=FloorPenaltyConfig(eta_floor=.8, gamma_min=.4, floor_k_ref=10.0),
    )
    bundle = DataBundle(
        symbol=config.symbol, bars=bars, daily=daily, actions=actions,
        calendar=pd.DatetimeIndex(dates[1:]), version="share-fixture",
        metadata={"board": "main"}, quality={"minute_ratio": 1.0},
    )
    return config, bundle


def _fold_ranges(bundle) -> list[tuple[str, str]]:
    calendar = bundle.calendar
    return [
        (str(calendar[0].date()), str(calendar[2].date())),
        (str(calendar[3].date()), str(calendar[-1].date())),
    ]


def _kernel_output(config, bundle, account_mode, *, output_mode=2, folds=None):
    prepared = share_engine._inputs(config, bundle)
    money = share_engine.funding(config, bundle)
    options = share_engine._options(config, prepared, money, mode=output_mode)
    days = prepared["days"]
    fold_ids = np.full(len(days), -1, dtype=np.int64)
    for index, (start, end) in enumerate(folds or []):
        mask = (days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))
        fold_ids[mask] = index
    runner = simulate if engine._NUMBA_FASTPATH_ENABLED else getattr(simulate, "py_func", simulate)
    return runner(
        *prepared["arrays"], *options,
        np.zeros(0, dtype=np.int64), np.zeros((0, 7), dtype=np.float64),
        fold_ids, len(folds or []), False, False, account_mode,
    )


# Return tuple layout of the kernel: trades are 6..15, rejections 16..21.
TRADE_FIELDS = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
REJECTION_FIELDS = (16, 17, 18, 19, 20, 21)


def _ledger_trades(output):
    keep = np.isin(output[7], (0, 3))
    return tuple(output[index][keep] for index in TRADE_FIELDS)


def _ledger_rejections(output):
    keep = np.isin(output[18], (0, 3))
    return tuple(output[index][keep] for index in REJECTION_FIELDS)


def _assert_same_ledger(left, right):
    assert left[0].shape == right[0].shape
    # The grid row and the matched funding row must be bit-identical.
    np.testing.assert_array_equal(left[0][0], right[0][0])
    np.testing.assert_array_equal(left[0][3], right[0][3])
    # The funding statistics of the grid account.
    np.testing.assert_array_equal(left[-1][0], right[-1][0])
    # The fold statistics and the floor episodes.
    np.testing.assert_array_equal(left[-3], right[-3])
    np.testing.assert_array_equal(left[-2], right[-2])
    # The equity paths of both surviving ledgers.
    for index in (1, 2, 3, 4, 22):
        np.testing.assert_array_equal(left[index][0], right[index][0])
        np.testing.assert_array_equal(left[index][3], right[index][3])
    # Every fill and every rejection of the grid and the matched hold.
    for left_field, right_field in zip(_ledger_trades(left), _ledger_trades(right)):
        np.testing.assert_array_equal(left_field, right_field)
    for left_field, right_field in zip(_ledger_rejections(left), _ledger_rejections(right)):
        np.testing.assert_array_equal(left_field, right_field)


def test_research_ledgers_are_identical_to_the_four_account_ledger():
    config, bundle = _share_market()
    folds = _fold_ranges(bundle)

    full = _kernel_output(config, bundle, ACCOUNT_MODE_FULL, folds=folds)
    research = _kernel_output(config, bundle, ACCOUNT_MODE_RESEARCH, folds=folds)

    _assert_same_ledger(full, research)
    # The skipped ledgers keep their row and report nothing as if they ran.
    assert full[0].shape == research[0].shape == (4, 11)
    assert research[0][1].size and np.all(np.isnan(research[0][1]))
    assert np.all(np.isnan(research[0][2]))
    assert research[-1].shape == full[-1].shape == (4, 24)
    # The research path really produced fills on both surviving ledgers.
    assert len(_ledger_trades(research)[0]) > 0
    assert set(np.unique(research[7]).tolist()) <= {0, 3}


def test_research_ledgers_match_the_python_fallback(monkeypatch):
    config, bundle = _share_market()
    folds = _fold_ranges(bundle)

    compiled = _kernel_output(config, bundle, ACCOUNT_MODE_RESEARCH, folds=folds)
    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    interpreted = _kernel_output(config, bundle, ACCOUNT_MODE_RESEARCH, folds=folds)

    _assert_same_ledger(compiled, interpreted)


def test_research_batch_reports_the_same_numbers_as_the_full_ledger(monkeypatch):
    config, bundle = _share_market()
    folds = _fold_ranges(bundle)
    params = [{"buy_step": .03}, {"buy_step": .06, "sell_shares": 200}]

    research = research_batch(config, bundle, params, folds)
    full = research_batch(config, bundle, params, folds, account_mode=ACCOUNT_MODE_FULL)
    assert research["metrics"] == full["metrics"]
    assert research["fold_metrics"] == full["fold_metrics"]
    assert research["position_stats"] == full["position_stats"]
    np.testing.assert_array_equal(research["funding_stats"], full["funding_stats"])

    monkeypatch.setattr(engine, "_NUMBA_FASTPATH_ENABLED", False)
    interpreted = research_batch(config, bundle, params, folds)
    assert interpreted["metrics"] == research["metrics"]
    assert interpreted["fold_metrics"] == research["fold_metrics"]
    assert interpreted["position_stats"] == research["position_stats"]


def test_ordinary_backtest_still_runs_the_four_accounts():
    config, bundle = _share_market()
    run = run_share_backtest(config, bundle)
    assert set(run["strategies"]) == {"grid", "dca", "buy_hold", "matched_funding_hold"}
    assert run["strategies"]["dca"]["trades"]
    assert run["strategies"]["buy_hold"]["trades"] == []
    assert run["strategies"]["matched_funding_hold"]["trades"]


# ---------------------------------------------------------------------------
# 3. Vectorised input preparation
# ---------------------------------------------------------------------------


FREQUENCIES = ("1m", "5m", "15m", "30m", "60m", "1d")


def _session_ends(frequency: str) -> list[int]:
    """Minute-of-day labels of the bars of one trading session."""
    if frequency == "1d":
        return [15 * 60]
    minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60}[frequency]
    ends: list[int] = []
    for opening, closing in ((9 * 60 + 30, 11 * 60 + 30), (13 * 60, 15 * 60)):
        moment = opening
        while moment + minutes <= closing:
            ends.append(moment + minutes)
            moment += minutes
    return ends


def _frequency_market(frequency: str, days: int = 3):
    dates = pd.bdate_range("2024-01-02", periods=days + 1)
    ends = _session_ends(frequency)
    records = []
    for day_index, day in enumerate(dates[1:]):
        for bar_index, minute in enumerate(ends):
            price = 10.0 + ((day_index + bar_index) % 5 - 2) * 0.4
            records.append({
                "timestamp": day + pd.Timedelta(minutes=minute),
                "open": price, "high": price, "low": price, "close": price,
                "volume": 100000, "frequency": frequency, "tradable": True,
                "is_synthetic": False,
                # A per-bar band overrides the daily band on the first bar.
                "limit_up": 10.5 if bar_index == 0 else np.nan,
                "limit_down": 9.5 if bar_index == 0 else np.nan,
            })
    bars = pd.DataFrame(records)
    daily_prices = [10.0] * (days + 1)
    daily = pd.DataFrame({
        "date": dates, "open": daily_prices, "close": daily_prices,
        "high": [value + .01 for value in daily_prices],
        "low": [value - .01 for value in daily_prices],
        "preclose": np.r_[10.0, daily_prices[:-1]],
        "volume": 100000, "turnover": [value * 100000 for value in daily_prices],
        "tradestatus": 1, "is_st": False, "limit_up": 100.0, "limit_down": 0.01,
        # Every session after the first one has no price limit at all.
        "no_price_limit": [False] + [True] * days,
    })
    config = BacktestConfig(
        schema_version=3, symbol="600000.SH", start_date=dates[1].date(),
        end_date=dates[-1].date(), frequency=frequency, initial_shares=1000,
        min_shares=0, max_shares=None, buy_shares=100, sell_shares=100,
        grid_mode="geometric", buy_step=.03, sell_step=.03,
        reference_price_mode="manual", reference_price=10.0,
        fees={"commission_rate": 0, "min_commission": 0, "slippage_bps": 0},
    )
    bundle = DataBundle(
        symbol=config.symbol, bars=bars, daily=daily,
        calendar=pd.DatetimeIndex(dates[1:]), version="frequency-fixture",
        metadata={"board": "main"}, quality={"minute_ratio": 1.0},
    )
    return config, bundle


def _reference_open_timestamps(stamps: pd.DatetimeIndex, frequencies) -> np.ndarray:
    """The per-bar rule that the vectorised helper replaced."""
    return np.asarray(
        [pd.Timestamp(engine.open_timestamp(ts, str(frequency))).value
         for ts, frequency in zip(stamps, frequencies)],
        dtype=np.int64,
    )


def _reference_price_bands(config, bundle, prepared) -> tuple[np.ndarray, np.ndarray]:
    """The per-day mask assignment that the vectorised expansion replaced."""
    from stockregression.rules import (
        calculated_price_limits, etf_price_limit_rate, round_to_tick,
    )
    terms = prepared["terms"]
    days = prepared["days"]
    codes = prepared["arrays"][8]
    count = len(codes)
    start = pd.Timestamp(config.start_date)
    end = min(
        pd.Timestamp(config.end_date) if config.end_date else pd.Timestamp.max.normalize(),
        pd.Timestamp(bundle.calendar.max()) if len(bundle.calendar)
        else pd.Timestamp(bundle.daily.date.max()),
    )
    daily = bundle.daily[bundle.daily.date.between(start, end)].sort_values("date")
    lookup = {pd.Timestamp(row["date"]): row for row in daily.to_dict("records")}
    ups = np.full(count, np.nan)
    downs = np.full(count, np.nan)
    for day_id, day in enumerate(days):
        row = lookup.get(day, {})
        previous = float(row.get("preclose", np.nan))
        up, down = row.get("limit_up", np.nan), row.get("limit_down", np.nan)
        if not bool(row.get("no_price_limit", False)):
            if terms["asset_type"] == "etf":
                rate = etf_price_limit_rate(terms, day.date())
                calc = (
                    round_to_tick(previous * (1 + rate), terms["price_tick"]),
                    round_to_tick(previous * (1 - rate), terms["price_tick"]),
                ) if previous > 0 else (np.nan, np.nan)
            else:
                calc = calculated_price_limits(
                    previous if previous > 0 else None, terms["board"], day.date(),
                    bool(row.get("is_st", False)),
                )
            up, down = up if pd.notna(up) else calc[0], down if pd.notna(down) else calc[1]
            ups[codes == day_id], downs[codes == day_id] = up or np.nan, down or np.nan
    bars = bundle.bars.sort_values("timestamp").reset_index(drop=True)
    assert len(bars) == count
    for column, output in (("limit_up", ups), ("limit_down", downs)):
        if column in bars:
            values = pd.to_numeric(bars[column], errors="coerce").to_numpy()
            mask = np.isfinite(values) & np.isfinite(output)
            output[mask] = values[mask]
    return ups, downs


@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_open_timestamps_match_the_engine_rule_bit_for_bit(frequency):
    config, bundle = _frequency_market(frequency)
    prepared = share_engine._inputs(config, bundle)
    stamps = prepared["stamps"]
    frequencies = bundle.bars.sort_values("timestamp")["frequency"].astype(str).to_numpy()

    expected = _reference_open_timestamps(stamps, frequencies)
    np.testing.assert_array_equal(prepared["arrays"][7], expected)


@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_price_bands_match_the_per_day_assignment(frequency):
    config, bundle = _frequency_market(frequency)
    prepared = share_engine._inputs(config, bundle)
    ups, downs = _reference_price_bands(config, bundle, prepared)

    np.testing.assert_array_equal(prepared["arrays"][4], ups)
    np.testing.assert_array_equal(prepared["arrays"][5], downs)
    # A session without a price limit keeps its NaN bands.
    codes = prepared["arrays"][8]
    for day_id in range(1, len(prepared["days"])):
        mask = codes == day_id
        assert np.all(np.isnan(prepared["arrays"][4][mask]))
        assert np.all(np.isnan(prepared["arrays"][5][mask]))


@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_fills_do_not_move_with_the_vectorised_inputs(monkeypatch, frequency):
    config, bundle = _frequency_market(frequency)
    prepared = share_engine._inputs(config, bundle)
    reference_ups, reference_downs = _reference_price_bands(config, bundle, prepared)
    stamps = prepared["stamps"]
    frequencies = bundle.bars.sort_values("timestamp")["frequency"].astype(str).to_numpy()
    reference_open = _reference_open_timestamps(stamps, frequencies)

    vectorised = run_share_backtest(config, bundle)

    arrays = list(prepared["arrays"])
    arrays[4] = reference_ups
    arrays[5] = reference_downs
    arrays[7] = reference_open
    patched = {**prepared, "arrays": tuple(arrays)}
    monkeypatch.setattr(share_engine, "_inputs", lambda *args, **kwargs: patched)
    reference = run_share_backtest(config, bundle)

    for name in ("grid", "dca", "buy_hold", "matched_funding_hold"):
        assert (vectorised["strategies"][name]["trades"]
                == reference["strategies"][name]["trades"])
        assert (vectorised["strategies"][name]["rejections"]
                == reference["strategies"][name]["rejections"])
        assert (vectorised["strategies"][name]["metrics"]
                == reference["strategies"][name]["metrics"])


def test_open_timestamp_rule_covers_the_session_boundaries():
    day = pd.Timestamp("2024-01-02")
    stamps = pd.DatetimeIndex([
        day + pd.Timedelta(hours=9, minutes=30),
        day + pd.Timedelta(hours=9, minutes=31),
        day + pd.Timedelta(hours=10, minutes=0),
        day + pd.Timedelta(hours=11, minutes=30),
        day + pd.Timedelta(hours=11, minutes=31),
        day + pd.Timedelta(hours=13, minutes=0),
        day + pd.Timedelta(hours=13, minutes=1),
        day + pd.Timedelta(hours=15, minutes=0),
        day + pd.Timedelta(hours=23, minutes=59),
        day + pd.Timedelta(days=1),
    ])
    for frequency in FREQUENCIES:
        frequencies = np.full(len(stamps), frequency, dtype=object)
        expected = _reference_open_timestamps(stamps, frequencies)
        actual = share_engine._open_timestamps_ns(stamps, frequencies)
        np.testing.assert_array_equal(actual, expected)
