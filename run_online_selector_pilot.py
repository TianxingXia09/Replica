from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import pandas as pd

from baseline_protocols import windowed_shadow_u_statistics
from local_protocol import (
    all_subsystems,
    local_overlap_kernel,
    partial_trace,
    probability_table,
    random_density,
    sample_pair_with_setting,
    subsystem_purities,
)
from online_selector import consensus_safe_selector, crossfit_safe_selector


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "online_pilot_results"
N_QUBITS = [3, 4]
BLOCKS = 18
ROUNDS = 512
AUDIT_ROUNDS = 64
WINDOWS = [2, 16, 64, ROUNDS]
REPS = 6
SCHEDULES = ["stable_fast_stable", "fast_stable_fast", "recurring", "gradual"]
ETA = 40.0
SHARE = 0.08


def online_problem(n, seed=20261219):
    rng = np.random.default_rng(seed + n)
    dimension = 2**n
    zero = np.zeros(dimension, dtype=complex)
    one = np.zeros(dimension, dtype=complex)
    zero[0] = 1.0
    one[-1] = 1.0
    plus = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    ghz = (zero + one) / np.sqrt(2.0)
    bank = [
        np.outer(zero, zero.conj()),
        np.outer(one, one.conj()),
        np.outer(plus, plus.conj()),
        np.outer(ghz, ghz.conj()),
        random_density(dimension, rng, rank=max(2, dimension // 2)),
        np.eye(dimension, dtype=complex) / dimension,
    ]
    subsets = all_subsystems(n, max_size=2)
    tables = [probability_table(rho, n) for rho in bank]
    purities = np.stack([subsystem_purities(rho, subsets, n) for rho in bank])
    return bank, subsets, tables, purities


def local_gram(bank, subsets, n):
    grams = []
    for subset in subsets:
        reduced = [partial_trace(rho, subset, n) for rho in bank]
        grams.append(
            np.asarray(
                [[np.real(np.trace(a @ b)) for b in reduced] for a in reduced],
                dtype=float,
            )
        )
    return grams


def select_pair(grams):
    count = grams[0].shape[0]
    distances = np.zeros((count, count))
    for gram in grams:
        diagonal = np.diag(gram)
        distances += diagonal[:, None] + diagonal[None, :] - 2 * gram
    distances /= len(grams)
    high = np.unravel_index(np.argmax(distances), distances.shape)
    masked = distances + np.eye(count) * distances.max()
    positive = np.where(masked > 1e-10, masked, np.inf)
    low = np.unravel_index(np.argmin(positive), positive.shape)
    return tuple(map(int, high)), tuple(map(int, low))


def block_labels(schedule, block, high_pair, low_pair, rng):
    if schedule == "stable_fast_stable":
        if block < 6 or block >= 12:
            return np.full(ROUNDS, high_pair[0], dtype=int)
        return np.where(np.arange(ROUNDS) % 2, high_pair[0], high_pair[1])
    if schedule == "fast_stable_fast":
        if 6 <= block < 12:
            return np.full(ROUNDS, high_pair[0], dtype=int)
        return np.where(np.arange(ROUNDS) % 2, high_pair[0], high_pair[1])
    if schedule == "recurring":
        pair = high_pair if (block // 3) % 2 else low_pair
        if block % 3 == 0:
            return np.full(ROUNDS, pair[0], dtype=int)
        return rng.choice(pair, size=ROUNDS)
    if schedule == "gradual":
        probability = block / (BLOCKS - 1)
        pair = high_pair if block >= BLOCKS // 2 else low_pair
        return np.where(rng.random(ROUNDS) < probability, pair[1], pair[0])
    raise ValueError(schedule)


def clip_by_size(values, subsets):
    lower = np.asarray([2.0 ** (-len(s)) for s in subsets])
    if np.ndim(values) == 2:
        lower = lower[:, None]
    return np.clip(values, lower, 1.0)


def run_one(task):
    n, schedule, rep, seed = task
    rng = np.random.default_rng(seed)
    bank, subsets, tables, purities = online_problem(n)
    grams = local_gram(bank, subsets, n)
    high_pair, low_pair = select_pair(grams)
    expert_names = [
        "sync",
        *[
            "temporal_pooled" if width == ROUNDS else f"temporal_w{width}"
            for width in WINDOWS
        ],
    ]
    experts_count = len(expert_names)
    weights_share = np.full((len(subsets), experts_count), 1 / experts_count)
    weights_static = weights_share.copy()
    cumulative_loss = np.zeros_like(weights_share)
    rows = []

    for block in range(BLOCKS):
        labels = block_labels(schedule, block, high_pair, low_pair, rng)
        truth = purities[labels].mean(axis=0)
        settings = np.empty((ROUNDS, n), dtype=np.int8)
        left = np.empty((ROUNDS, n), dtype=np.int8)
        right = np.empty((ROUNDS, n), dtype=np.int8)
        sync_samples = np.empty((ROUNDS, len(subsets)))
        for t, label in enumerate(labels):
            settings[t], left[t], right[t] = sample_pair_with_setting(
                tables[int(label)], tables[int(label)], n, rng
            )
            sync_samples[t] = local_overlap_kernel(left[t], right[t], subsets)
        sync = sync_samples.mean(axis=0)
        temporal_left = windowed_shadow_u_statistics(settings, left, subsets, WINDOWS)
        temporal_right = windowed_shadow_u_statistics(settings, right, subsets, WINDOWS)
        candidates = [sync]
        for width in WINDOWS:
            candidates.append(0.5 * (temporal_left[width] + temporal_right[width]))
        candidates = clip_by_size(np.stack(candidates, axis=1), subsets)

        temporal_pooled = candidates[:, -1]
        dispersion_hat = np.maximum(0.0, candidates[:, 0] - temporal_pooled)
        sync_variance = np.var(sync_samples, axis=0, ddof=1) / ROUNDS
        crossover = np.sqrt(np.maximum(sync_variance, 0.0))
        gate_predictions = {}
        for multiplier in (0.5, 1.0, 1.5, 2.0):
            use_sync = dispersion_hat > multiplier * crossover
            gate_predictions[f"dispersion_gate_{multiplier:.1f}"] = np.where(
                use_sync, candidates[:, 0], temporal_pooled
            )
        crossfit_05, _ = crossfit_safe_selector(
            settings, left, right, sync_samples, subsets, rng, multiplier=0.5
        )
        crossfit_10, _ = crossfit_safe_selector(
            settings, left, right, sync_samples, subsets, rng, multiplier=1.0
        )
        consensus_05, _ = consensus_safe_selector(
            settings, left, right, sync_samples, subsets, rng, multiplier=0.5
        )

        audit_indices = rng.choice(ROUNDS, size=AUDIT_ROUNDS, replace=False)
        audit_sum = np.zeros(len(subsets))
        for t in audit_indices:
            _, audit_left, audit_right = sample_pair_with_setting(
                tables[int(labels[t])], tables[int(labels[t])], n, rng
            )
            audit_sum += local_overlap_kernel(audit_left, audit_right, subsets)
        audit = clip_by_size(audit_sum / AUDIT_ROUNDS, subsets)

        losses = (candidates - audit[:, None]) ** 2
        current_choice = np.argmin(losses, axis=1)
        prediction_audit_hard = candidates[np.arange(len(subsets)), current_choice]
        current_shifted = losses - losses.min(axis=1, keepdims=True)
        current_weights = np.exp(-ETA * current_shifted)
        current_weights /= current_weights.sum(axis=1, keepdims=True)
        prediction_audit_soft = np.sum(current_weights * candidates, axis=1)

        prediction_share = np.sum(weights_share * candidates, axis=1)
        prediction_static = np.sum(weights_static * candidates, axis=1)
        ftl_choice = np.argmin(cumulative_loss, axis=1)
        prediction_ftl = candidates[np.arange(len(subsets)), ftl_choice]
        uniform = candidates.mean(axis=1)

        shifted = losses - losses.min(axis=1, keepdims=True)
        weights_share *= np.exp(-ETA * shifted)
        weights_share /= weights_share.sum(axis=1, keepdims=True)
        weights_share = (1 - SHARE) * weights_share + SHARE / experts_count
        weights_static *= np.exp(-ETA * shifted)
        weights_static /= weights_static.sum(axis=1, keepdims=True)
        cumulative_loss += losses

        methods = {
            "observable_audit_selector": prediction_audit_hard,
            "observable_audit_soft": prediction_audit_soft,
            "online_fixed_share": prediction_share,
            "online_static_hedge": prediction_static,
            "online_follow_the_leader": prediction_ftl,
            "uniform_mixture": uniform,
            "crossfit_gate_0.5": crossfit_05,
            "crossfit_gate_1.0": crossfit_10,
            "consensus_gate_0.5": consensus_05,
            **gate_predictions,
        }
        methods.update({name: candidates[:, j] for j, name in enumerate(expert_names)})
        oracle_index = np.argmin((candidates - truth[:, None]) ** 2, axis=1)
        methods["block_oracle"] = candidates[np.arange(len(subsets)), oracle_index]

        for method, estimate in methods.items():
            for j, subset in enumerate(subsets):
                error = float(estimate[j] - truth[j])
                rows.append(
                    {
                        "n_qubits": n,
                        "schedule": schedule,
                        "rep": rep,
                        "block": block,
                        "method": method,
                        "subsystem": "-".join(map(str, subset)),
                        "size": len(subset),
                        "truth": float(truth[j]),
                        "estimate": float(estimate[j]),
                        "error": error,
                        "sq_error": error * error,
                        "audit_fraction": AUDIT_ROUNDS / ROUNDS,
                        "copy_overhead_fraction": AUDIT_ROUNDS / ROUNDS,
                    }
                )
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    master = np.random.default_rng(20261220)
    tasks = [
        (n, schedule, rep, int(master.integers(2**31 - 1)))
        for n in N_QUBITS
        for schedule in SCHEDULES
        for rep in range(REPS)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(run_one, tasks, chunksize=1):
            rows.extend(result)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "pilot_raw.csv", index=False)
    aggregate = raw.groupby(["n_qubits", "schedule", "size", "method"], as_index=False).agg(
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        mse=("sq_error", "mean"),
        records=("sq_error", "size"),
    )
    aggregate.to_csv(OUT / "pilot_aggregate.csv", index=False)

    online = aggregate[aggregate.method == "consensus_gate_0.5"]
    oracle = aggregate[aggregate.method == "block_oracle"]
    merged = online.merge(oracle, on=["n_qubits", "schedule", "size"], suffixes=("_online", "_oracle"))
    fixed_names = ["sync"] + [
        "temporal_pooled" if width == ROUNDS else f"temporal_w{width}"
        for width in WINDOWS
    ]
    fixed = aggregate[aggregate.method.isin(fixed_names)]
    best_fixed = fixed.loc[fixed.groupby(["n_qubits", "schedule", "size"]).mse.idxmin()]
    merged = merged.merge(best_fixed[["n_qubits", "schedule", "size", "method", "mse", "rmse"]], on=["n_qubits", "schedule", "size"])
    summary = {
        "tasks": len(tasks),
        "raw_rows": len(raw),
        "copy_overhead_fraction": AUDIT_ROUNDS / ROUNDS,
        "median_online_to_oracle_mse_ratio": float(np.median(merged.mse_online / merged.mse_oracle)),
        "median_online_to_best_fixed_mse_ratio": float(np.median(merged.mse_online / merged.mse)),
        "cells_online_beats_best_fixed": int((merged.mse_online < merged.mse).sum()),
        "cells": len(merged),
    }
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    merged.to_csv(OUT / "pilot_comparison.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
