from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import pandas as pd

from baseline_protocols import windowed_shadow_u_statistics
from local_protocol import local_overlap_kernel, sample_pair_with_setting
from online_selector import clip_purity, guarded_dispersion_gate
from run_online_selector_benchmark import online_problem, markov_labels


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "online_large_results"
N_QUBITS = [3, 4, 5]
BLOCKS = 10
ROUNDS = 2000
WINDOWS = [2, 16, 64, ROUNDS]
REPS = 12
SCHEDULES = [
    "stationary_mixed",
    "telegraph_chirp",
    "burst_width_chirp",
    "local_then_global",
    "rank_cycle",
    "quasiperiodic",
]


def schedule_labels(schedule: str, block: int, rng: np.random.Generator):
    if schedule == "stationary_mixed":
        return np.full(ROUNDS, 4, dtype=int)
    if schedule == "telegraph_chirp":
        probabilities = [0.005, 0.015, 0.04, 0.10, 0.22, 0.38, 0.55, 0.70, 0.82, 0.92]
        return markov_labels(ROUNDS, (0, 1), probabilities[block], rng)
    if schedule == "burst_width_chirp":
        widths = [101, 71, 43, 29, 19, 13, 11, 7, 5, 3]
        width = widths[block]
        chunks = (ROUNDS + width - 1) // width
        labels = np.arange(chunks, dtype=int) % 2
        return np.repeat(labels, width)[:ROUNDS]
    if schedule == "local_then_global":
        pair = (0, 2) if block < BLOCKS // 2 else (0, 1)
        probability = 0.08 + 0.07 * block
        return markov_labels(ROUNDS, pair, min(probability, 0.85), rng)
    if schedule == "rank_cycle":
        probabilities = [0.01, 0.08, 0.25, 0.60, 0.15] * 2
        return markov_labels(ROUNDS, (4, 5), probabilities[block], rng)
    if schedule == "quasiperiodic":
        time = np.arange(ROUNDS)
        signal = np.sin(np.sqrt(2.0) * time + 0.7 * block) + 0.55 * np.sin(
            np.sqrt(5.0) * time - 0.3 * block
        )
        labels = np.zeros(ROUNDS, dtype=int)
        labels[signal > 0.45] = 1
        labels[signal < -0.45] = 3
        return labels
    raise ValueError(schedule)


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
        candidates = clip_purity(
            np.stack([sync] + [temporal[width] for width in WINDOWS], axis=1),
            subsets,
        )
        guarded, selected = guarded_dispersion_gate(
            candidates[:, 0],
            candidates[:, -1],
            np.asarray([len(s) for s in subsets]),
            ROUNDS,
        )
        oracle_index = np.argmin((candidates - truth[:, None]) ** 2, axis=1)
        oracle = candidates[np.arange(len(subsets)), oracle_index]
        methods = {name: candidates[:, j] for j, name in enumerate(names)}
        methods["observable_guarded_gate"] = guarded
        methods["block_oracle"] = oracle
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
                        "selected_sync": float(selected[j]) if method == "observable_guarded_gate" else 0.0,
                        "copy_overhead_fraction": 0.0,
                    }
                )
    return rows


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator):
    count = len(values)
    indices = rng.integers(0, count, size=(3000, count))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(raw: pd.DataFrame):
    cells = ["n_qubits", "schedule", "size"]
    aggregate = raw.groupby(cells + ["method"], as_index=False).agg(
        mse=("sq_error", "mean"),
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        records=("sq_error", "size"),
    )
    expert_names = ["sync", "temporal_w2", "temporal_w16", "temporal_w64", "temporal_pooled"]
    experts = aggregate[aggregate.method.isin(expert_names)]
    best = experts.loc[experts.groupby(cells).mse.idxmin()].rename(
        columns={"method": "best_fixed_method", "mse": "best_fixed_mse", "rmse": "best_fixed_rmse"}
    )
    gate = aggregate[aggregate.method == "observable_guarded_gate"].rename(
        columns={"mse": "gate_mse", "rmse": "gate_rmse"}
    )
    oracle = aggregate[aggregate.method == "block_oracle"].rename(
        columns={"mse": "oracle_mse", "rmse": "oracle_rmse"}
    )
    comparison = gate[cells + ["gate_mse", "gate_rmse"]].merge(
        best[cells + ["best_fixed_method", "best_fixed_mse", "best_fixed_rmse"]], on=cells
    ).merge(oracle[cells + ["oracle_mse", "oracle_rmse"]], on=cells)
    comparison["relative_mse_reduction_vs_best_fixed"] = 1 - comparison.gate_mse / comparison.best_fixed_mse
    comparison["gate_to_oracle_mse_ratio"] = comparison.gate_mse / comparison.oracle_mse

    per_rep = raw.groupby(cells + ["rep", "method"], as_index=False).sq_error.mean()
    rng = np.random.default_rng(20270102)
    intervals = []
    for key, group in per_rep.groupby(cells):
        pivot = group.pivot(index="rep", columns="method", values="sq_error")
        fixed_method = comparison[
            (comparison.n_qubits == key[0])
            & (comparison.schedule == key[1])
            & (comparison["size"] == key[2])
        ].best_fixed_method.iloc[0]
        difference = (pivot[fixed_method] - pivot["observable_guarded_gate"]).to_numpy()
        low, high = bootstrap_ci(difference, rng)
        intervals.append(
            {
                **dict(zip(cells, key)),
                "mse_advantage_ci_low": low,
                "mse_advantage_ci_high": high,
                "significant_win": low > 0,
                "significant_loss": high < 0,
            }
        )
    return aggregate, comparison.merge(pd.DataFrame(intervals), on=cells)


def main():
    OUT.mkdir(exist_ok=True)
    master = np.random.default_rng(20270101)
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
    raw = pd.DataFrame(rows)
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
        "repetitions": REPS,
        "frozen_rule": {
            "base_threshold": 0.5,
            "floor_threshold": 4.0,
            "floor_band_fraction": 0.1,
        },
        "copy_overhead_fraction": 0.0,
        "cells": len(comparison),
        "cells_beating_best_fixed": int((comparison.gate_mse < comparison.best_fixed_mse).sum()),
        "significant_wins": int(comparison.significant_win.sum()),
        "significant_losses": int(comparison.significant_loss.sum()),
        "median_relative_mse_reduction_vs_best_fixed": float(
            comparison.relative_mse_reduction_vs_best_fixed.median()
        ),
        "median_gate_to_oracle_mse_ratio": float(comparison.gate_to_oracle_mse_ratio.median()),
        "fraction_round_cells_selecting_sync": float(selection.selected_sync.mean()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
