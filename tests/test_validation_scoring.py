from __future__ import annotations

import random

import pytest

from stockregression.validation_scoring import (
    common_decision_anchors,
    descriptive_summary,
    gss_screens,
    independent_account_summary,
    paired_forward_outcome,
    segment_principal_returns,
)


def _metrics(value: float) -> dict[str, float]:
    return {
        "rv_annualized": value,
        "path_redundancy_daily_mean": value,
        "path_redundancy_5d_mean": value,
        "variance_ratio_240m": value,
        "iovr": value,
        "zero_crossing_rate": value,
        "jump_ratio": value,
    }


def test_segment_returns_use_continuous_profit_levels_instead_of_summing_rates():
    equity = [
        {"timestamp": "2024-01-01T15:00:00", "equity": 1000,
         "cumulative_profit": 0, "total_invested": 1000},
        {"timestamp": "2024-01-10T15:00:00", "equity": 1100,
         "cumulative_profit": 100, "total_invested": 1000},
        {"timestamp": "2024-02-10T15:00:00", "equity": 2110,
         "cumulative_profit": 110, "total_invested": 2000},
    ]
    result = segment_principal_returns(equity, [
        {"start": "2024-01-02", "end": "2024-01-31", "days": 20},
        {"start": "2024-02-01", "end": "2024-02-29", "days": 20},
    ])
    assert result["rates_are_summed"] is False
    assert result["segments"][0]["principal_return"] == pytest.approx(.1)
    assert result["segments"][1]["principal_return"] == pytest.approx(10 / 2100)
    assert result["continuous_account_return"] == pytest.approx(.055)
    assert result["segments"][1]["profit_delta"] == pytest.approx(10)
    assert result["segments"][1]["segment_peak_principal"] == 2100


def test_first_block_uses_explicit_initial_equity_and_keeps_an_initial_loss():
    result = segment_principal_returns(
        [
            {"timestamp": "2024-01-03T09:31:00", "equity": 900,
             "total_invested": 1000},
            {"timestamp": "2024-01-03T15:00:00", "equity": 1000,
             "total_invested": 1000},
        ],
        [{"start": "2024-01-03", "end": "2024-01-03"}],
        initial_equity=1000,
    )
    segment = result["segments"][0]
    assert segment["initial_source"] == "explicit_initial_equity"
    assert segment["starting_equity"] == 1000
    assert segment["principal_return"] == 0
    assert segment["max_principal_drawdown"] == pytest.approx(.1)
    assert result["continuous_account_return"] == 0


def test_first_day_fees_are_included_before_the_first_equity_row():
    result = segment_principal_returns(
        [
            {"timestamp": "2024-01-03T09:31:00", "equity": 995,
             "total_invested": 1000},
            {"timestamp": "2024-01-03T15:00:00", "equity": 1000,
             "total_invested": 1000},
        ],
        [{"start": "2024-01-03", "end": "2024-01-03"}],
        initial_equity=1000,
    )
    segment = result["segments"][0]
    assert segment["principal_return"] == 0
    assert segment["max_principal_drawdown"] == pytest.approx(.005)


def test_first_block_intraday_principal_injection_raises_the_basis_only_once():
    result = segment_principal_returns(
        [
            {"timestamp": "2024-01-03T09:31:00", "equity": 1000,
             "total_invested": 1000},
            {"timestamp": "2024-01-03T11:30:00", "equity": 2000,
             "total_invested": 2000},
            {"timestamp": "2024-01-03T15:00:00", "equity": 2050,
             "total_invested": 2000},
        ],
        [{"start": "2024-01-03", "end": "2024-01-03"}],
        initial_equity=1000,
    )
    segment = result["segments"][0]
    assert segment["capital_added"] == 1000
    assert segment["segment_peak_principal"] == 2000
    assert segment["ending_profit"] == 50
    assert segment["principal_return"] == pytest.approx(.025)
    assert segment["max_principal_drawdown"] == 0


def test_paired_forward_outcome_forwards_both_initial_account_values():
    outcome = paired_forward_outcome(
        [
            {"timestamp": "2024-01-03T09:31:00", "equity": 990,
             "total_invested": 1000},
            {"timestamp": "2024-01-03T15:00:00", "equity": 1000,
             "total_invested": 1000},
        ],
        [
            {"timestamp": "2024-01-03T09:31:00", "equity": 1000,
             "total_invested": 1000},
            {"timestamp": "2024-01-03T15:00:00", "equity": 1000,
             "total_invested": 1000},
        ],
        start="2024-01-03", end="2024-01-03", days=1,
        grid_initial_equity=1000, matched_funding_initial_equity=1000,
    )
    assert outcome["grid_principal_return"] == 0
    assert outcome["matched_funding_return"] == 0
    assert outcome["paired_excess"] == 0
    assert outcome["max_principal_drawdown"] == pytest.approx(.01)


@pytest.mark.parametrize(("prior_equity", "basis"), [(1100.0, 2100.0), (900.0, 1900.0)])
def test_segment_rebuilds_assets_and_drawdown_after_prior_result_and_injection(
    prior_equity, basis,
):
    equity = [
        {"timestamp": "2024-01-01T15:00:00", "equity": 1000,
         "total_invested": 1000, "principal_drawdown": .8},
        {"timestamp": "2024-01-31T15:00:00", "equity": prior_equity,
         "total_invested": 1000, "principal_drawdown": .9},
        {"timestamp": "2024-02-01T15:00:00", "equity": basis,
         "total_invested": 2000, "principal_drawdown": .95},
        {"timestamp": "2024-02-10T15:00:00", "equity": basis + 50,
         "total_invested": 2000, "principal_drawdown": .97},
        {"timestamp": "2024-02-20T15:00:00", "equity": basis - 50,
         "total_invested": 2000, "principal_drawdown": .99},
        {"timestamp": "2024-02-29T15:00:00", "equity": basis + 100,
         "total_invested": 2000, "principal_drawdown": 1.0},
    ]
    segment = segment_principal_returns(equity, [
        {"start": "2024-02-01", "end": "2024-02-29"},
    ])["segments"][0]
    assert segment["starting_equity"] == pytest.approx(prior_equity)
    assert segment["ending_equity"] == pytest.approx(basis + 100)
    assert segment["capital_added"] == pytest.approx(1000)
    assert segment["segment_peak_principal"] == pytest.approx(basis)
    assert segment["principal_return"] == pytest.approx(100 / basis)
    assert segment["max_principal_drawdown"] == pytest.approx(100 / basis)


def test_descriptive_summary_has_no_significance_fields():
    summary = descriptive_summary([.1, None, .2, float("nan")])
    assert summary["count"] == 2
    assert summary["missing"] == 2
    assert summary["median"] == pytest.approx(.15)
    assert summary["evidence_kind"] == "descriptive"
    assert not set(summary) & {"p_value", "pvalue", "confidence_interval"}


def test_independent_account_summary_reads_qualification_strategy_metrics():
    result = independent_account_summary({
        "510300.SH": {
            "status": "completed",
            "strategy_metrics": {
                "peak_principal_return": .12,
                "max_principal_drawdown": -.04,
                "total_profit": 1200,
                "peak_invested": 10000,
            },
        },
    })
    assert result["accounts"]["510300.SH"] == {
        "peak_principal_return": .12,
        "max_principal_drawdown": -.04,
        "total_profit": 1200,
        "peak_invested": 10000,
    }
    assert result["portfolio_nav_constructed"] is False


def test_gss_four_screens_mark_shortfalls_and_do_not_fill_from_future():
    rows = [
        {"symbol": "A", "status": "completed", "metrics": _metrics(.1), "eligible": True},
        {"symbol": "B", "status": "completed", "metrics": _metrics(.2), "eligible": False},
        {"symbol": "C", "status": "completed", "metrics": _metrics(.3), "eligible": True},
        {"symbol": "D", "status": "completed", "metrics": {}, "eligible": False},
        {"symbol": "E", "status": "unavailable", "metrics": _metrics(9), "eligible": True},
        {"symbol": "F", "status": "completed", "metrics": _metrics(.4), "eligible": True},
        {"symbol": "G", "status": "completed", "metrics": _metrics(.05), "eligible": True},
    ]
    result = gss_screens(rows, top_n=3, seed=7, decision_date="2024-01-02")
    assert tuple(result["screens"]) == (
        "random_all", "score_only", "filter_only", "score_filter",
    )
    assert result["available_count"] == 7
    assert result["quality_cohort_size"] == 6
    assert result["unavailable_count"] == 1
    assert result["screens"]["random_all"]["count"] == 3
    assert "E" not in result["screens"]["random_all"]["symbols"]
    assert result["screens"]["score_only"]["symbols"] == ["F", "C", "B"]
    expected_filter = random.Random("7:2024-01-02:filter_only").sample(
        ["A", "C", "F", "G"], 3,
    )
    assert result["screens"]["filter_only"]["symbols"] == expected_filter
    assert result["screens"]["filter_only"]["rule"] == (
        "deterministic_random_from_filter_eligible"
    )
    assert result["screens"]["filter_only"]["sufficient"] is True
    assert result["screens"]["score_filter"]["symbols"] == ["F", "C", "A"]
    assert result["future_missing_policy"] == "retain_at_decision_no_future_imputation"
    assert result["historical_universe_verified"] is False


def test_one_symbol_cannot_support_three_member_any_screen():
    result = gss_screens([
        {"symbol": "ONLY", "status": "completed", "metrics": _metrics(.1), "eligible": True},
    ])
    assert all(not screen["sufficient"] for screen in result["screens"].values())
    assert result["screens"]["random_all"]["shortfall"] == 2
    assert result["screens"]["filter_only"]["shortfall"] == 2
    assert result["screens"]["score_only"]["shortfall"] == 3
    assert result["screens"]["score_filter"]["shortfall"] == 3
    assert result["score_cohort_size"] == 0


def test_common_anchors_use_evaluation_date_intersection_not_param_schedule_dates():
    result = common_decision_anchors({
        "A": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "B": ["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    }, horizon_days=2)
    assert result["common_dates"] == [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    ]
    assert result["anchors"] == [{
        "decision_date": "2024-01-02",
        "outcome_start": "2024-01-03",
        "outcome_end": "2024-01-04",
        "decision_index": 0,
        "outcome_observations": 2,
    }]

