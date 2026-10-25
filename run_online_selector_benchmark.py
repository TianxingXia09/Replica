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
    probability_table,
    random_density,
    sample_pair_with_setting,
    subsystem_purities,
)
from phase3_theory import safe_dispersion_gate
from online_selector import (
    append_guarded_method,
    consensus_safe_selector,
    crossfit_safe_selector,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "online_results"
N_QUBITS = [3, 4, 5]
BLOCKS = 20
ROUNDS = 512
WINDOWS = [2, 16, 64, ROUNDS]
REPS = 20
SCHEDULES = [
    "stationary",
    "markov_multiscale",
    "sinusoidal_mixture",
    "local_support_drift",
    "mixed_rank_drift",
    "bursty_prime",
]
FROZEN_MULTIPLIER = 0.5


def product_state(bits: list[int]) -> np.ndarray:
    index = 0
    for bit in bits:
        index = 2 * index + int(bit)
    vector = np.zeros(2 ** len(bits), dtype=complex)
    vector[index] = 1.0
    return np.outer(vector, vector.conj())


def online_problem(n: int):
    rng = np.random.default_rng(20261229 + n)
    dimension = 2**n
    zero = product_state([0] * n)
    one = product_state([1] * n)
    local_flip = product_state([0] * (n - 1) + [1])
    plus_vector = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    plus = np.outer(plus_vector, plus_vector.conj())
    mixed = random_density(dimension, rng, rank=max(2, dimension // 2))
    maximally_mixed = np.eye(dimension, dtype=complex) / dimension
    bank = [zero, one, local_flip, plus, mixed, maximally_mixed]
    subsets = all_subsystems(n, max_size=min(3, n))
    tables = [probability_table(rho, n) for rho in bank]
    purities = np.stack([subsystem_purities(rho, subsets, n) for rho in bank])
    return subsets, tables, purities


def markov_labels(rounds: int, pair: tuple[int, int], probability: float, rng):
    labels = np.empty(rounds, dtype=int)
    labels[0] = pair[0]
    for t in range(1, rounds):
        if rng.random() < probability:
            labels[t] = pair[1] if labels[t - 1] == pair[0] else pair[0]
        else:
            labels[t] = labels[t - 1]
    return labels


def schedule_labels(schedule: str, block: int, rng: np.random.Generator):
    if schedule == "stationary":
        return np.zeros(ROUNDS, dtype=int)
    if schedule == "markov_multiscale":
        probabilities = [0.01, 0.35, 0.03, 0.70]
        return markov_labels(ROUNDS, (0, 1), probabilities[block // 5], rng)
    if schedule == "sinusoidal_mixture":
        phase = 2 * np.pi * block / BLOCKS
        probability = 0.5 + 0.48 * np.sin(phase)
        return np.where(rng.random(ROUNDS) < probability, 1, 0)
    if schedule == "local_support_drift":
        switch_probability = 0.02 if block < 10 else 0.55
        return markov_labels(ROUNDS, (0, 2), switch_probability, rng)
    if schedule == "mixed_rank_drift":
        probability = 0.05 + 0.4 * (block / (BLOCKS - 1))
        return markov_labels(ROUNDS, (4, 5), probability, rng)
    if schedule == "bursty_prime":
        widths = [3, 7, 11, 17]
        width = widths[block % len(widths)]
        chunks = (ROUNDS + width - 1) // width
        states = rng.integers(0, 4, size=chunks)
        return np.repeat(states, width)[:ROUNDS]
    raise ValueError(schedule)


def clip_estimates(values: np.ndarray, subsets):
    lower = np.asarray([2.0 ** (-len(s)) for s in subsets])
    if values.ndim == 2:
        lower = lower[:, None]
    return np.clip(values, lower, 1.0)


def run_one(task):
    n, schedule, rep, seed = task
    rng = np.random.default_rng(seed)
    subsets, tables, purities = online_problem(n)
    rows = []
    for block in range(BLOCKS):
        labels = schedule_labels(schedule, block, rng)
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
        sync_variance = np.var(sync_samples, axis=0, ddof=1) / ROUNDS
        temporal_left = windowed_shadow_u_statistics(settings, left, subsets, WINDOWS)
        temporal_right = windowed_shadow_u_statistics(settings, right, subsets, WINDOWS)
        temporal = {
            width: 0.5 * (temporal_left[width] + temporal_right[width])
            for width in WINDOWS
        }
        names = ["sync"] + [
            "temporal_pooled" if width == ROUNDS else f"temporal_w{width}"
            for width in WINDOWS
        ]
        candidates = clip_estimates(
            np.stack([sync] + [temporal[width] for width in WINDOWS], axis=1),
            subsets,
        )
        pooled = candidates[:, -1]
        safe, selected = safe_dispersion_gate(
            candidates[:, 0], pooled, sync_variance, FROZEN_MULTIPLIER
        )
        conservative, _ = safe_dispersion_gate(
            candidates[:, 0], pooled, sync_variance, 1.0
        )
        naive, _ = safe_dispersion_gate(
            candidates[:, 0], pooled, sync_variance, 0.0
        )
        crossfit_safe, crossfit_selection = crossfit_safe_selector(
            settings,
            left,
            right,
            sync_samples,
            subsets,
            rng,
            multiplier=FROZEN_MULTIPLIER,
        )
        consensus_safe, consensus_selection = consensus_safe_selector(
            settings,
            left,
            right,
            sync_samples,
            subsets,
            rng,
            multiplier=FROZEN_MULTIPLIER,
        )
        half = ROUNDS // 2
        first = sync_samples[:half].mean(axis=0)
        second = sync_samples[half:].mean(axis=0)
        half_se = np.sqrt(
            np.var(sync_samples[:half], axis=0, ddof=1) / half
            + np.var(sync_samples[half:], axis=0, ddof=1) / half
        )
        change_detected = np.abs(first - second) > half_se
        change_point = np.where(change_detected, candidates[:, 0], pooled)
        oracle_index = np.argmin((candidates - truth[:, None]) ** 2, axis=1)
        oracle = candidates[np.arange(len(subsets)), oracle_index]

        methods = {name: candidates[:, j] for j, name in enumerate(names)}
        methods.update(
            {
                "observable_safe_gate": safe,
                "observable_crossfit_gate": crossfit_safe,
                "observable_consensus_gate": consensus_safe,
                "observable_gate_conservative": conservative,
                "observable_gate_naive": naive,
                "purity_change_detector": change_point,
                "block_oracle": oracle,
            }
        )
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
                        "selected_sync": (
                            float(crossfit_selection[j])
                            if method == "observable_crossfit_gate"
                            else (
                                float(consensus_selection[j])
                                if method == "observable_consensus_gate"
                                else (float(selected[j]) if method == "observable_safe_gate" else 0.0)
                            )
                        ),
                        "copy_overhead_fraction": 0.0,
                    }
                )
    return rows


def bootstrap_difference(per_rep: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    count = len(per_rep)
    indices = rng.integers(0, count, size=(2000, count))
    values = per_rep.difference.to_numpy()[indices].mean(axis=1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize(raw: pd.DataFrame):
    cell_keys = ["n_qubits", "schedule", "size"]
    aggregate = raw.groupby(cell_keys + ["method"], as_index=False).agg(
        mse=("sq_error", "mean"),
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        records=("sq_error", "size"),
    )
    expert_names = ["sync", "temporal_w2", "temporal_w16", "temporal_w64", "temporal_pooled"]
    experts = aggregate[aggregate.method.isin(expert_names)]
    best_fixed = experts.loc[experts.groupby(cell_keys).mse.idxmin()].rename(
        columns={"method": "best_fixed_method", "mse": "best_fixed_mse", "rmse": "best_fixed_rmse"}
    )
    safe = aggregate[aggregate.method == "observable_guarded_gate"].rename(
        columns={"mse": "safe_mse", "rmse": "safe_rmse"}
    )
    oracle = aggregate[aggregate.method == "block_oracle"].rename(
        columns={"mse": "oracle_mse", "rmse": "oracle_rmse"}
    )
    comparison = safe[cell_keys + ["safe_mse", "safe_rmse"]].merge(
        best_fixed[cell_keys + ["best_fixed_method", "best_fixed_mse", "best_fixed_rmse"]], on=cell_keys
    ).merge(oracle[cell_keys + ["oracle_mse", "oracle_rmse"]], on=cell_keys)
    comparison["relative_mse_reduction_vs_best_fixed"] = 1 - comparison.safe_mse / comparison.best_fixed_mse
    comparison["safe_to_oracle_mse_ratio"] = comparison.safe_mse / comparison.oracle_mse

    per_rep = raw.groupby(cell_keys + ["rep", "method"], as_index=False).sq_error.mean()
    intervals = []
    rng = np.random.default_rng(20261231)
    for key, group in per_rep.groupby(cell_keys):
        pivot = group.pivot(index="rep", columns="method", values="sq_error")
        fixed_method = comparison[
            (comparison.n_qubits == key[0])
            & (comparison.schedule == key[1])
            & (comparison["size"] == key[2])
        ].best_fixed_method.iloc[0]
        differences = pd.DataFrame(
            {"difference": pivot[fixed_method] - pivot["observable_guarded_gate"]}
        )
        low, high = bootstrap_difference(differences, int(rng.integers(2**31 - 1)))
        intervals.append(
            {
                **dict(zip(cell_keys, key)),
                "mse_advantage_ci_low": low,
                "mse_advantage_ci_high": high,
                "significant_win": low > 0,
                "significant_loss": high < 0,
            }
        )
    intervals = pd.DataFrame(intervals)
    comparison = comparison.merge(intervals, on=cell_keys)
    return aggregate, comparison


def main():
    OUT.mkdir(exist_ok=True)
    master = np.random.default_rng(20261230)
    tasks = [
        (n, schedule, rep, int(master.integers(2**31 - 1)))
        for n in N_QUBITS
        for schedule in SCHEDULES
        for rep in range(REPS)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for task_rows in executor.map(run_one, tasks, chunksize=1):
            rows.extend(task_rows)
    raw = append_guarded_method(pd.DataFrame(rows), ROUNDS)
    raw.to_csv(OUT / "raw.csv", index=False)
    aggregate, comparison = summarize(raw)
    aggregate.to_csv(OUT / "aggregate.csv", index=False)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    selection = raw[raw.method == "observable_guarded_gate"]
    summary = {
        "tasks": len(tasks),
        "raw_rows": len(raw),
        "blocks_per_task": BLOCKS,
        "rounds_per_block": ROUNDS,
        "frozen_multiplier": FROZEN_MULTIPLIER,
        "copy_overhead_fraction": 0.0,
        "cells": len(comparison),
        "cells_beating_best_fixed": int((comparison.safe_mse < comparison.best_fixed_mse).sum()),
        "significant_wins": int(comparison.significant_win.sum()),
        "significant_losses": int(comparison.significant_loss.sum()),
        "median_relative_mse_reduction_vs_best_fixed": float(
            comparison.relative_mse_reduction_vs_best_fixed.median()
        ),
        "median_safe_to_oracle_mse_ratio": float(comparison.safe_to_oracle_mse_ratio.median()),
        "fraction_round_cells_selecting_sync": float(selection.selected_sync.mean()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def finalize_existing():
    raw_path = OUT / "raw.csv"
    raw = append_guarded_method(pd.read_csv(raw_path), ROUNDS)
    raw.to_csv(raw_path, index=False)
    aggregate, comparison = summarize(raw)
    aggregate.to_csv(OUT / "aggregate.csv", index=False)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    selection = raw[raw.method == "observable_guarded_gate"]
    summary = {
        "tasks": int(raw[["n_qubits", "schedule", "rep"]].drop_duplicates().shape[0]),
        "raw_rows": len(raw),
        "blocks_per_task": BLOCKS,
        "rounds_per_block": ROUNDS,
        "frozen_rule": {
            "base_threshold": 0.5,
            "floor_threshold": 4.0,
            "floor_band_fraction": 0.1,
        },
        "copy_overhead_fraction": 0.0,
        "cells": len(comparison),
        "cells_beating_best_fixed": int((comparison.safe_mse < comparison.best_fixed_mse).sum()),
        "significant_wins": int(comparison.significant_win.sum()),
        "significant_losses": int(comparison.significant_loss.sum()),
        "median_relative_mse_reduction_vs_best_fixed": float(
            comparison.relative_mse_reduction_vs_best_fixed.median()
        ),
        "median_safe_to_oracle_mse_ratio": float(comparison.safe_to_oracle_mse_ratio.median()),
        "fraction_round_cells_selecting_sync": float(selection.selected_sync.mean()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys

    if "--finalize-existing" in sys.argv:
        finalize_existing()
    else:
        main()
