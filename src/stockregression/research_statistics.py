"""Paired, family-wise bootstrap summaries for strategy research.

The bootstrap resamples trading days once per replicate and applies the same
indices to every candidate.  It therefore preserves both serial dependence
within a candidate and the cross-candidate dependence needed for a Reality
Check style family correction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .performance import blas_thread_limit, current_blas_threads, record_count

try:
    from numba import njit
except ImportError:  # pragma: no cover - Numba is a runtime dependency
    njit = None


if njit is None:
    def _bootstrap_counts(starts, restart, n_days):
        indices = np.empty(n_days, dtype=np.int64)
        indices[0] = starts[0]
        for day in range(1, n_days):
            indices[day] = starts[day] if restart[day] else (indices[day - 1] + 1) % n_days
        return np.bincount(indices, minlength=n_days).astype(np.float64)
else:
    @njit(cache=True)
    def _bootstrap_counts(starts, restart, n_days):
        indices = np.empty(n_days, dtype=np.int64)
        counts = np.zeros(n_days, dtype=np.float64)
        indices[0] = starts[0]
        counts[indices[0]] += 1.0
        for day in range(1, n_days):
            if restart[day]:
                indices[day] = starts[day]
            else:
                indices[day] = (indices[day - 1] + 1) % n_days
            counts[indices[day]] += 1.0
        return counts


def _returns_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("daily returns must be a one- or two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("daily returns must contain days and candidates")
    if not np.isfinite(matrix).all() or np.any(matrix <= -1):
        raise ValueError("daily returns must be finite and greater than -100%")
    return matrix


def _paired_log_excess(candidate_returns: Any, passive_returns: Any) -> np.ndarray:
    candidates = _returns_matrix(candidate_returns)
    passive = np.asarray(passive_returns, dtype=np.float64).reshape(-1)
    if len(passive) != len(candidates):
        raise ValueError("passive and candidate returns must be aligned by day")
    if not np.isfinite(passive).all() or np.any(passive <= -1):
        raise ValueError("passive returns must be finite and greater than -100%")
    return np.log1p(candidates) - np.log1p(passive[:, None])


def stationary_bootstrap_indices(
    n_days: int,
    block_length: float = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw one circular stationary-bootstrap sample of length ``n_days``.

    A new block starts with probability 1 / block_length; otherwise the next
    observation follows the previous one, wrapping around the sample.  The
    output is one-dimensional so callers can share it across all candidates.
    """
    if n_days < 1:
        raise ValueError("n_days must be positive")
    if not np.isfinite(block_length) or block_length < 1:
        raise ValueError("block_length must be at least one day")
    generator = rng or np.random.default_rng()
    starts = generator.integers(0, n_days, size=n_days)
    restart = generator.random(n_days) < (1.0 / float(block_length))
    indices = np.empty(n_days, dtype=np.int64)
    indices[0] = starts[0]
    for day in range(1, n_days):
        indices[day] = starts[day] if restart[day] else (indices[day - 1] + 1) % n_days
    return indices


def _bootstrap_maxima(
    centered: np.ndarray,
    reps: int,
    block_length: float,
    rng: np.random.Generator,
    cancelled=None,
    progress=None,
) -> np.ndarray:
    n_days, _ = centered.shape
    maxima = np.empty(reps, dtype=np.float64)
    # Batch the expensive family matrix product while generating each sample
    # in replicate order.  Each row uses one shared index path for every
    # candidate, and this preserves the generator sequence across chunk sizes.
    max_chunk_bytes = 64 * 1024 * 1024
    bytes_per_replicate = np.dtype(np.float64).itemsize * (n_days + centered.shape[1])
    chunk_size = max(1, min(128, max_chunk_bytes // max(1, bytes_per_replicate)))
    for begin in range(0, reps, chunk_size):
        if cancelled and cancelled():
            return maxima[:begin]
        end = min(begin + chunk_size, reps)
        weights = np.empty((end - begin, n_days), dtype=np.float64)
        for offset in range(end - begin):
            starts = rng.integers(0, n_days, size=n_days)
            restart = rng.random(n_days) < (1.0 / float(block_length))
            weights[offset] = _bootstrap_counts(starts, restart, n_days)
        # Keep only block counts and the current product; never retain a
        # reps x days x candidates bootstrap tensor.
        with blas_thread_limit(current_blas_threads()):
            products = weights @ centered / n_days
        maxima[begin:end] = np.max(products, axis=1)
        record_count("bootstrap_replicates", end - begin)
        if progress:
            progress(end / max(1, reps))
    return maxima


def _one_bootstrap_summary(
    log_excess: np.ndarray,
    *,
    selected_index: int,
    reps: int,
    block_length: float,
    seed: int,
    alpha: float,
    cancelled=None,
    progress=None,
) -> dict[str, Any]:
    n_days, n_candidates = log_excess.shape
    if not 0 <= selected_index < n_candidates:
        raise ValueError("selected_index is outside the candidate family")
    means = log_excess.mean(axis=0)
    observed_best = float(np.max(means))
    observed_selected = float(means[selected_index])
    centered = log_excess - means[None, :]
    maxima = _bootstrap_maxima(
        centered,
        reps,
        block_length,
        np.random.default_rng(seed),
        cancelled,
        progress,
    )
    done = len(maxima)
    if done == 0:
        p_value = None
        max_quantile = None
        lower = None
    else:
        # The observed maximum is compared with the centered maximum family
        # distribution.  This is a family-level p-value, never a probability
        # that the selected strategy will win in the future.
        exceedances = int(np.count_nonzero(maxima >= observed_best))
        p_value = float((exceedances + 1) / (done + 1))
        max_quantile = float(np.quantile(maxima, 1.0 - alpha))
        lower = float(observed_selected - max_quantile)
    return {
        "block_length": float(block_length),
        "requested_reps": int(reps),
        "completed_reps": int(done),
        "alpha": float(alpha),
        "p_value": p_value,
        "family_max_mean_log_excess": observed_best,
        "selected_mean_log_excess": observed_selected,
        "selected_simultaneous_lower_log_excess": lower,
        "simultaneous_max_quantile": max_quantile,
        "days": int(n_days),
        "candidate_count": int(n_candidates),
        "interpretation": "配对、家族校正的零假设检验；p 值不是策略胜率。",
        "cancelled": bool(done < reps and cancelled is not None),
    }


def paired_stationary_bootstrap(
    candidate_returns: Any,
    passive_returns: Any,
    *,
    selected_index: int,
    reps: int = 10_000,
    block_lengths: tuple[int, ...] = (20, 10, 40),
    seed: int = 42,
    alpha: float = 0.05,
    cancelled=None,
    progress=None,
) -> dict[str, Any]:
    """Return a paired Reality Check and selected simultaneous lower bound.

    Inputs are simple daily returns.  They are converted to paired daily log
    excess returns against the same-cash passive account.  The null bootstrap
    recenters every candidate at zero and records the maximum statistic across
    the entire family for each shared resample.
    """
    if reps < 1:
        raise ValueError("reps must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not block_lengths or any(int(length) < 1 for length in block_lengths):
        raise ValueError("block_lengths must contain positive integers")
    log_excess = _paired_log_excess(candidate_returns, passive_returns)
    summaries = {}
    for offset, length in enumerate(block_lengths):
        if cancelled and cancelled():
            break
        summaries[str(int(length))] = _one_bootstrap_summary(
            log_excess,
            selected_index=selected_index,
            reps=reps,
            block_length=float(length),
            seed=int(seed) + offset,
            alpha=alpha,
            cancelled=cancelled,
            progress=(lambda fraction, offset=offset: progress((offset + fraction) / len(block_lengths))) if progress else None,
        )
        if progress:
            progress((offset + 1) / len(block_lengths))
    primary = summaries.get(str(int(block_lengths[0])))
    return {
        "method": "paired_stationary_bootstrap_white_reality_check",
        "return_basis": "daily_log_excess_vs_same_cash_passive",
        "selected_index": int(selected_index),
        "selected_candidate_mean_log_excess": float(log_excess.mean(axis=0)[selected_index]),
        "sensitivity": summaries,
        "primary": primary,
        "cancelled": bool(cancelled and any(item.get("cancelled") for item in summaries.values())),
    }


# A shorter alias is handy at call sites and preserves the explicit method
# name above for API users who want the statistical intent in code.
paired_reality_check = paired_stationary_bootstrap

