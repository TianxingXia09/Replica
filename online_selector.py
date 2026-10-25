from __future__ import annotations

import numpy as np

from baseline_protocols import windowed_shadow_u_statistics
from phase3_theory import safe_dispersion_gate


def clip_purity(values: np.ndarray, subsets: list[tuple[int, ...]]) -> np.ndarray:
    lower = np.asarray([2.0 ** (-len(s)) for s in subsets], dtype=float)
    if np.ndim(values) == 2:
        lower = lower[:, None]
    return np.clip(values, lower, 1.0)


def fold_estimators(
    indices: np.ndarray,
    settings: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sync_samples: np.ndarray,
    subsets: list[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.sort(np.asarray(indices, dtype=int))
    count = len(indices)
    sync = sync_samples[indices].mean(axis=0)
    sync_variance = np.var(sync_samples[indices], axis=0, ddof=1) / count
    left_temporal = windowed_shadow_u_statistics(
        settings[indices], left[indices], subsets, [count]
    )[count]
    right_temporal = windowed_shadow_u_statistics(
        settings[indices], right[indices], subsets, [count]
    )[count]
    temporal = 0.5 * (left_temporal + right_temporal)
    pair = clip_purity(np.stack([sync, temporal], axis=1), subsets)
    return pair[:, 0], pair[:, 1], sync_variance


def crossfit_safe_selector(
    settings: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sync_samples: np.ndarray,
    subsets: list[tuple[int, ...]],
    rng: np.random.Generator,
    multiplier: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-fold selector with independent decision and estimation folds."""
    rounds = len(settings)
    permutation = rng.permutation(rounds)
    split = rounds // 2
    fold_a, fold_b = permutation[:split], permutation[split:]
    sync_a, temporal_a, variance_a = fold_estimators(
        fold_a, settings, left, right, sync_samples, subsets
    )
    sync_b, temporal_b, variance_b = fold_estimators(
        fold_b, settings, left, right, sync_samples, subsets
    )
    _, choose_sync_a = safe_dispersion_gate(
        sync_a, temporal_a, variance_a, multiplier
    )
    _, choose_sync_b = safe_dispersion_gate(
        sync_b, temporal_b, variance_b, multiplier
    )
    estimate_from_b = np.where(choose_sync_a, sync_b, temporal_b)
    estimate_from_a = np.where(choose_sync_b, sync_a, temporal_a)
    estimate = 0.5 * (estimate_from_a + estimate_from_b)
    selection_fraction = 0.5 * (
        choose_sync_a.astype(float) + choose_sync_b.astype(float)
    )
    return clip_purity(estimate, subsets), selection_fraction


def consensus_safe_selector(
    settings: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sync_samples: np.ndarray,
    subsets: list[tuple[int, ...]],
    rng: np.random.Generator,
    multiplier: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Select synchronization only when two independent folds agree."""
    rounds = len(settings)
    permutation = rng.permutation(rounds)
    split = rounds // 2
    fold_a, fold_b = permutation[:split], permutation[split:]
    sync_a, temporal_a, variance_a = fold_estimators(
        fold_a, settings, left, right, sync_samples, subsets
    )
    sync_b, temporal_b, variance_b = fold_estimators(
        fold_b, settings, left, right, sync_samples, subsets
    )
    _, choose_sync_a = safe_dispersion_gate(
        sync_a, temporal_a, variance_a, multiplier
    )
    _, choose_sync_b = safe_dispersion_gate(
        sync_b, temporal_b, variance_b, multiplier
    )
    consensus = choose_sync_a & choose_sync_b
    synchronized = 0.5 * (sync_a + sync_b)
    temporal = 0.5 * (temporal_a + temporal_b)
    estimate = np.where(consensus, synchronized, temporal)
    return clip_purity(estimate, subsets), consensus.astype(float)


def guarded_dispersion_gate(
    synchronized: np.ndarray,
    temporal: np.ndarray,
    subsystem_sizes: np.ndarray,
    rounds: int,
    base_threshold: float = 0.5,
    floor_threshold: float = 4.0,
    floor_band_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Purity-floor-aware observable gate with frozen analytic scaling."""
    synchronized = np.asarray(synchronized, dtype=float)
    temporal = np.asarray(temporal, dtype=float)
    sizes = np.asarray(subsystem_sizes, dtype=int)
    lower = 2.0 ** (-sizes)
    near_floor = temporal < lower + floor_band_fraction * (1.0 - lower)
    threshold = np.where(near_floor, floor_threshold, base_threshold)
    scale = np.sqrt((3.0**sizes - 1.0) / float(rounds))
    normalized_gap = (synchronized - temporal) / scale
    use_synchronized = normalized_gap > threshold
    return np.where(use_synchronized, synchronized, temporal), use_synchronized


def append_guarded_method(raw, rounds: int):
    """Append guarded estimates to a long-form benchmark table."""
    import pandas as pd

    keys = [
        "n_qubits",
        "schedule",
        "rep",
        "block",
        "subsystem",
        "size",
        "truth",
    ]
    cleaned = raw[raw.method != "observable_guarded_gate"].copy()
    synchronized = cleaned[cleaned.method == "sync"].copy()
    temporal = cleaned[cleaned.method == "temporal_pooled"][keys + ["estimate"]].rename(
        columns={"estimate": "temporal_estimate"}
    )
    guarded = synchronized.merge(temporal, on=keys, validate="one_to_one")
    estimate, selected = guarded_dispersion_gate(
        guarded.estimate.to_numpy(),
        guarded.temporal_estimate.to_numpy(),
        guarded["size"].to_numpy(),
        rounds,
    )
    guarded["method"] = "observable_guarded_gate"
    guarded["estimate"] = estimate
    guarded["error"] = guarded.estimate - guarded.truth
    guarded["sq_error"] = guarded.error**2
    guarded["selected_sync"] = selected.astype(float)
    guarded = guarded.drop(columns="temporal_estimate")
    return pd.concat([cleaned, guarded[cleaned.columns]], ignore_index=True)
