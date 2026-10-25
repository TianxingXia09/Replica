from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from local_protocol import local_overlap_kernel, partial_trace, sample_pair
from phase3_theory import gram_matrix
from run_phase2_benchmarks import prepared_problem, trajectory


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "phase3_results"
N_QUBITS = [3, 4, 5]
ROUNDS = 2000
REGIMES = ["alternating", "iid_switch", "history_adaptive"]
MISMATCH_RATES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
REPS = 30


def task_list():
    rng = np.random.default_rng(20261201)
    return [
        (n, regime, rep, int(rng.integers(2**31 - 1)))
        for n in N_QUBITS
        for regime in REGIMES
        for rep in range(REPS)
    ]


def run_one(task):
    n, regime, rep, seed = task
    bank, subsets, tables, purities, _ = prepared_problem(n)
    seed_sequence = np.random.SeedSequence(seed)
    trajectory_seed, *measurement_seeds = seed_sequence.spawn(1 + len(MISMATCH_RATES))
    labels = trajectory(regime, ROUNDS, bank, np.random.default_rng(trajectory_seed))
    alternatives = (labels + 1) % len(bank)
    truth = purities[labels].mean(axis=0)

    overlap_banks = []
    hs_bounds = []
    for subset_index, subset in enumerate(subsets):
        reductions = [partial_trace(rho, subset, n) for rho in bank]
        gram = gram_matrix(reductions)
        overlap_banks.append(gram)
        distance = np.sqrt(
            np.maximum(
                0.0,
                np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram,
            )
        )
        hs_bounds.append(np.sqrt(np.diag(gram))[:, None] * distance)

    rows = []
    for alpha, measurement_seed in zip(MISMATCH_RATES, measurement_seeds):
        rng = np.random.default_rng(measurement_seed)
        estimate_sum = np.zeros(len(subsets))
        for label, alternative in zip(labels, alternatives):
            right_label = int(alternative if rng.random() < alpha else label)
            left_bits, right_bits = sample_pair(
                tables[int(label)], tables[right_label], n, rng
            )
            estimate_sum += local_overlap_kernel(left_bits, right_bits, subsets)
        estimate = estimate_sum / ROUNDS
        for j, subset in enumerate(subsets):
            gram = overlap_banks[j]
            exact_bias = float(
                alpha
                * np.mean(
                    gram[labels, alternatives] - gram[labels, labels]
                )
            )
            bound = float(alpha * np.mean(hs_bounds[j][labels, alternatives]))
            error = float(estimate[j] - truth[j])
            rows.append(
                {
                    "n_qubits": n,
                    "regime": regime,
                    "rep": rep,
                    "rounds": ROUNDS,
                    "mismatch_rate": alpha,
                    "subsystem": "-".join(map(str, subset)),
                    "size": len(subset),
                    "truth": float(truth[j]),
                    "estimate": float(estimate[j]),
                    "error": error,
                    "sq_error": error * error,
                    "exact_bias": exact_bias,
                    "hs_bias_bound": bound,
                }
            )
    return rows


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["n_qubits", "regime", "size", "mismatch_rate"]
    return raw.groupby(keys, as_index=False).agg(
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        empirical_bias=("error", "mean"),
        mean_exact_bias=("exact_bias", "mean"),
        mean_abs_exact_bias=("exact_bias", lambda x: float(np.mean(np.abs(x)))),
        mean_hs_bias_bound=("hs_bias_bound", "mean"),
        max_bound_violation=("exact_bias", lambda x: 0.0),
        records=("error", "size"),
    )


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for task_rows in executor.map(run_one, task_list(), chunksize=1):
            rows.extend(task_rows)
    raw = pd.DataFrame(rows)
    raw["bound_violation"] = np.abs(raw.exact_bias) - raw.hs_bias_bound
    raw.to_csv(OUT / "mismatch_raw.csv", index=False)
    summary_table = aggregate(raw)
    violations = raw.groupby(
        ["n_qubits", "regime", "size", "mismatch_rate"], as_index=False
    ).bound_violation.max()
    summary_table = summary_table.drop(columns="max_bound_violation").merge(
        violations.rename(columns={"bound_violation": "max_bound_violation"}),
        on=["n_qubits", "regime", "size", "mismatch_rate"],
    )
    zero = summary_table[summary_table.mismatch_rate == 0][
        ["n_qubits", "regime", "size", "rmse"]
    ].rename(columns={"rmse": "rmse_at_zero"})
    summary_table = summary_table.merge(zero, on=["n_qubits", "regime", "size"])
    summary_table["rmse_inflation"] = summary_table.rmse / summary_table.rmse_at_zero
    summary_table.to_csv(OUT / "mismatch_aggregate.csv", index=False)
    global_summary = {
        "tasks": len(task_list()),
        "raw_records": int(len(raw)),
        "rounds_per_condition": ROUNDS,
        "repetitions": REPS,
        "maximum_analytic_bound_violation": float(raw.bound_violation.max()),
        "median_rmse_inflation": {
            str(alpha): float(group.rmse_inflation.median())
            for alpha, group in summary_table.groupby("mismatch_rate")
        },
        "median_absolute_empirical_minus_exact_bias": float(
            np.median(np.abs(summary_table.empirical_bias - summary_table.mean_exact_bias))
        ),
    }
    (OUT / "mismatch_summary.json").write_text(json.dumps(global_summary, indent=2))
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()
