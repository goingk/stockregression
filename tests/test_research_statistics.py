import numpy as np
import pytest

from stockregression.research_statistics import (
    _bootstrap_counts,
    paired_stationary_bootstrap,
    stationary_bootstrap_indices,
)


def test_stationary_bootstrap_indices_are_reproducible_and_keep_circular_blocks():
    first = stationary_bootstrap_indices(40, 5, np.random.default_rng(21))
    second = stationary_bootstrap_indices(40, 5, np.random.default_rng(21))

    assert np.array_equal(first, second)
    assert first.shape == (40,)
    assert np.all((0 <= first) & (first < 40))
    assert np.any(np.diff(first) % 40 == 1)


def test_compiled_bootstrap_counts_match_stationary_index_reference():
    rng = np.random.default_rng(21)
    starts = rng.integers(0, 40, size=40)
    restart = rng.random(40) < 0.2
    indices = np.empty(40, dtype=np.int64)
    indices[0] = starts[0]
    for day in range(1, 40):
        indices[day] = starts[day] if restart[day] else (indices[day - 1] + 1) % 40

    assert np.array_equal(
        _bootstrap_counts(starts, restart, 40),
        np.bincount(indices, minlength=40).astype(np.float64),
    )


def test_paired_bootstrap_uses_same_family_resamples_and_reports_sensitivity():
    rng = np.random.default_rng(12)
    passive = rng.normal(0.0001, 0.01, 126)
    candidates = np.column_stack((passive + 0.0005, passive + 0.0002, passive - 0.0001))

    first = paired_stationary_bootstrap(
        candidates, passive, selected_index=0, reps=250, block_lengths=(20, 10, 40), seed=7
    )
    second = paired_stationary_bootstrap(
        candidates, passive, selected_index=0, reps=250, block_lengths=(20, 10, 40), seed=7
    )

    assert first == second
    assert first["method"] == "paired_stationary_bootstrap_white_reality_check"
    assert set(first["sensitivity"]) == {"10", "20", "40"}
    assert first["primary"]["completed_reps"] == 250
    assert 0 <= first["primary"]["p_value"] <= 1
    assert np.isfinite(first["primary"]["selected_simultaneous_lower_log_excess"])
    assert "不是策略胜率" in first["primary"]["interpretation"]


def test_paired_bootstrap_validates_alignment_and_returns():
    with pytest.raises(ValueError, match="aligned"):
        paired_stationary_bootstrap(np.zeros((5, 2)), np.zeros(4), selected_index=0, reps=10)
    with pytest.raises(ValueError, match="greater than -100%"):
        paired_stationary_bootstrap(np.array([[-1.0]]), np.zeros(1), selected_index=0, reps=10)

