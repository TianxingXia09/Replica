from __future__ import annotations

import numpy as np


def gram_matrix(states: list[np.ndarray]) -> np.ndarray:
    """Hilbert--Schmidt Gram matrix for a finite state bank."""
    return np.asarray(
        [[float(np.real(np.trace(rho @ sigma))) for sigma in states] for rho in states],
        dtype=float,
    )


def block_temporal_dispersion(
    labels: np.ndarray,
    purity: np.ndarray,
    gram: np.ndarray,
    width: int,
) -> float:
    """Exact purity-minus-cross-overlap gap for non-overlapping windows.

    The weighting matches ``windowed_shadow_u_statistics``. Blocks containing
    fewer than two emissions are omitted.
    """
    labels = np.asarray(labels, dtype=int)
    if width < 2:
        raise ValueError("width must be at least two")
    block_ids = np.arange(len(labels), dtype=int) // width
    block_count = int(block_ids[-1]) + 1
    counts = np.zeros((block_count, len(purity)), dtype=float)
    np.add.at(counts, (block_ids, labels), 1.0)
    sizes = counts.sum(axis=1)
    valid = sizes >= 2
    if not np.any(valid):
        raise ValueError("no complete block with at least two emissions")
    counts = counts[valid]
    sizes = sizes[valid]
    purity_sum = counts @ purity
    ordered_overlap_sum = np.einsum("bi,ij,bj->b", counts, gram, counts) - purity_sum
    block_purity = purity_sum / sizes
    block_cross_overlap = ordered_overlap_sum / (sizes * (sizes - 1.0))
    return float(np.average(block_purity - block_cross_overlap, weights=sizes))


def synchronized_beats_temporal(
    dispersion: float,
    synchronized_variance: float,
    temporal_variance: float,
) -> bool:
    """Exact MSE boundary for an unbiased synchronized estimator.

    A temporal estimator unbiased for its within-window cross-overlap target
    has MSE ``temporal_variance + dispersion**2`` for instantaneous purity.
    """
    return dispersion**2 > synchronized_variance - temporal_variance


def safe_dispersion_gate(
    synchronized: np.ndarray,
    temporal: np.ndarray,
    synchronized_variance: np.ndarray,
    multiplier: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Observable-only conservative switch between two purity estimators.

    The rule uses no state labels or target values. It selects synchronization
    when the positive overlap gap exceeds a fixed multiple of the estimated
    synchronized standard error.
    """
    synchronized = np.asarray(synchronized, dtype=float)
    temporal = np.asarray(temporal, dtype=float)
    variance = np.maximum(np.asarray(synchronized_variance, dtype=float), 0.0)
    use_synchronized = np.maximum(0.0, synchronized - temporal) > (
        float(multiplier) * np.sqrt(variance)
    )
    return np.where(use_synchronized, synchronized, temporal), use_synchronized


def mismatch_bias(
    rho: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Bias when an overlap estimator is interpreted as purity of ``rho``."""
    return float(np.real(np.trace(rho @ sigma) - np.trace(rho @ rho)))


def mismatch_hs_bound(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Cauchy--Schwarz bound on absolute replication-mismatch bias."""
    delta = sigma - rho
    rho_hs = float(np.sqrt(np.real(np.trace(rho @ rho))))
    delta_hs = float(np.sqrt(np.real(np.trace(delta.conj().T @ delta))))
    return rho_hs * delta_hs
