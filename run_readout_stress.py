from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd

from local_protocol import (
    all_subsystems,
    corrected_local_overlap_kernel,
    local_overlap_kernel,
    partial_trace,
    probability_table,
    random_density,
    sample_pair,
)


ROOT = Path(__file__).resolve().parent
N = 4
ROUNDS = [500, 2000]
ERROR_RATES = [0.0, 0.02, 0.05, 0.10]
REPS = 100


@lru_cache(maxsize=None)
def problem():
    rng = np.random.default_rng(771)
    d = 2**N
    bank = [random_density(d, rng, rank=r) for r in [1, 2, 4, 8, 16]]
    subsets = all_subsystems(N, max_size=3)
    tables = [probability_table(rho, N) for rho in bank]
    purities = []
    for rho in bank:
        vals = []
        for subset in subsets:
            red = partial_trace(rho, subset, N)
            vals.append(float(np.real(np.trace(red @ red))))
        purities.append(vals)
    return bank, subsets, tables, np.array(purities)


def run_one(task):
    rounds, error_rate, rep, seed = task
    rng = np.random.default_rng(seed)
    bank, subsets, tables, purities = problem()
    labels = rng.integers(0, len(bank), size=rounds)
    truth = purities[labels].mean(axis=0)
    raw = np.zeros(len(subsets))
    corrected = np.zeros(len(subsets))
    rates = np.full(N, error_rate)
    for label in labels:
        a, b = sample_pair(tables[int(label)], tables[int(label)], N, rng)
        if error_rate:
            a = np.bitwise_xor(a, rng.random(N) < error_rate)
            b = np.bitwise_xor(b, rng.random(N) < error_rate)
        raw += local_overlap_kernel(a, b, subsets)
        corrected += corrected_local_overlap_kernel(a, b, subsets, rates)
    rows = []
    for method, estimate in {
        "uncorrected": raw / rounds,
        "readout_corrected": corrected / rounds,
    }.items():
        for j, subset in enumerate(subsets):
            err = float(estimate[j] - truth[j])
            rows.append(
                {
                    "rounds": rounds,
                    "error_rate": error_rate,
                    "rep": rep,
                    "method": method,
                    "size": len(subset),
                    "subsystem": "".join(map(str, subset)),
                    "error": err,
                    "sq_error": err**2,
                }
            )
    return rows


def main():
    master = np.random.default_rng(20261024)
    tasks = [
        (rounds, rate, rep, int(master.integers(2**31 - 1)))
        for rounds in ROUNDS
        for rate in ERROR_RATES
        for rep in range(REPS)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for chunk in ex.map(run_one, tasks, chunksize=2):
            rows.extend(chunk)
    out = ROOT / "readout_results"
    out.mkdir(exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw.csv", index=False)
    agg = raw.groupby(["rounds", "error_rate", "size", "method"], as_index=False).agg(
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        bias=("error", "mean"),
    )
    agg.to_csv(out / "aggregate.csv", index=False)
    noisy = agg[agg.error_rate > 0]
    pivot = noisy.pivot_table(
        index=["rounds", "error_rate", "size"], columns="method", values=["rmse", "bias"]
    )
    summary = {
        "max_absolute_corrected_bias": float(noisy[noisy.method == "readout_corrected"].bias.abs().max()),
        "median_absolute_bias_reduction": float(
            1
            - pivot[("bias", "readout_corrected")].abs().median()
            / pivot[("bias", "uncorrected")].abs().median()
        ),
        "median_rmse_inflation_from_correction": float(
            (pivot[("rmse", "readout_corrected")] / pivot[("rmse", "uncorrected")]).median()
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
