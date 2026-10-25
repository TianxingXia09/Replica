from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd

from local_protocol import (
    all_subsystems,
    local_overlap_kernel,
    partial_trace,
    probability_table,
    random_density,
    sample_pair,
)


ROOT = Path(__file__).resolve().parent
N_QUBITS = [3, 4, 5]
ROUNDS = [500, 2000]
REPS = 60
WINDOWS = [2, 4, 8, 16, 64]
REGIMES = ["stationary", "alternating", "iid_switch", "burst_4", "burst_16"]


def make_bank(n: int, seed: int):
    rng = np.random.default_rng(seed)
    d = 2**n
    ranks = [1, 1, 2, max(2, d // 4), max(2, d // 2), d]
    return [random_density(d, rng, rank=r) for r in ranks]


@lru_cache(maxsize=None)
def prepared_problem(n: int):
    bank = make_bank(n, 9100 + n)
    subsets = all_subsystems(n, max_size=min(3, n))
    tables = [probability_table(rho, n) for rho in bank]
    reduced = [[partial_trace(rho, s, n) for s in subsets] for rho in bank]
    purities = np.array(
        [[float(np.real(np.trace(red @ red))) for red in row] for row in reduced]
    )
    return bank, subsets, tables, reduced, purities


def trajectory(regime: str, rounds: int, n_states: int, rng: np.random.Generator):
    if regime == "stationary":
        return np.zeros(rounds, dtype=int)
    if regime == "alternating":
        return np.arange(rounds) % 2
    if regime == "iid_switch":
        return rng.integers(0, n_states, size=rounds)
    if regime.startswith("burst_"):
        width = int(regime.split("_")[1])
        return np.repeat(rng.integers(0, n_states, size=(rounds + width - 1) // width), width)[:rounds]
    raise ValueError(regime)


def oracle_window(labels, reduced, subsets, width):
    rounds = len(labels)
    total = np.zeros(len(subsets))
    for start in range(0, rounds, width):
        block = labels[start : min(start + width, rounds)]
        weight = len(block) / rounds
        for j in range(len(subsets)):
            avg = sum(reduced[int(label)][j] for label in block) / len(block)
            total[j] += weight * float(np.real(np.trace(avg @ avg)))
    return total


def run_one(task):
    n, rounds, regime, rep, seed = task
    rng = np.random.default_rng(seed)
    bank, subsets, tables, reduced, purities = prepared_problem(n)
    labels = trajectory(regime, rounds, len(bank), rng)
    truth = purities[labels].mean(axis=0)

    accum = np.zeros(len(subsets))
    for label in labels:
        a, b = sample_pair(tables[int(label)], tables[int(label)], n, rng)
        accum += local_overlap_kernel(a, b, subsets)
    estimates = {"synchronized_local": accum / rounds}

    # These baselines are deliberately stronger than implementable estimators:
    # they know every block-averaged density matrix exactly and have no shot noise.
    estimates["pooled_oracle"] = oracle_window(labels, reduced, subsets, rounds)
    for width in WINDOWS:
        estimates[f"window_{width}_oracle"] = oracle_window(labels, reduced, subsets, width)

    rows = []
    for method, estimate in estimates.items():
        for j, subset in enumerate(subsets):
            error = float(estimate[j] - truth[j])
            rows.append(
                {
                    "n_qubits": n,
                    "rounds": rounds,
                    "regime": regime,
                    "rep": rep,
                    "method": method,
                    "subsystem": "".join(map(str, subset)),
                    "size": len(subset),
                    "truth": truth[j],
                    "estimate": estimate[j],
                    "error": error,
                    "sq_error": error**2,
                    "abs_error": abs(error),
                }
            )
    return rows


def main():
    master = np.random.default_rng(20261023)
    tasks = [
        (n, rounds, regime, rep, int(master.integers(2**31 - 1)))
        for n in N_QUBITS
        for rounds in ROUNDS
        for regime in REGIMES
        for rep in range(REPS)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for chunk in ex.map(run_one, tasks, chunksize=2):
            rows.extend(chunk)

    out = ROOT / "local_results"
    out.mkdir(exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw.csv", index=False)
    agg = raw.groupby(
        ["n_qubits", "rounds", "regime", "size", "method"], as_index=False
    ).agg(
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
    )
    agg.to_csv(out / "aggregate.csv", index=False)

    hard = agg[agg.regime != "stationary"]
    comparisons = []
    for keys, group in hard.groupby(["n_qubits", "rounds", "regime", "size"]):
        sync = float(group[group.method == "synchronized_local"].rmse.iloc[0])
        oracle = group[group.method != "synchronized_local"]
        best = oracle.loc[oracle.rmse.idxmin()]
        comparisons.append(
            {
                "n_qubits": keys[0],
                "rounds": keys[1],
                "regime": keys[2],
                "size": keys[3],
                "sync_rmse": sync,
                "best_oracle": best.method,
                "oracle_rmse": best.rmse,
                "relative_improvement": 1 - sync / best.rmse,
            }
        )
    comp = pd.DataFrame(comparisons)
    comp.to_csv(out / "comparisons.csv", index=False)

    # Scaling check: doubling accuracy by quadrupling rounds should roughly halve RMSE.
    sync = agg[(agg.method == "synchronized_local") & (agg.regime != "stationary")]
    pivot = sync.pivot_table(
        index=["n_qubits", "regime", "size"], columns="rounds", values="rmse"
    ).dropna()
    scaling_ratios = (pivot[2000] / pivot[500]).to_numpy()
    decision = {
        "median_rmse_ratio_N2000_over_N500": float(np.median(scaling_ratios)),
        "fraction_hard_cells_beating_zero_noise_oracle": float(
            np.mean(comp.relative_improvement > 0)
        ),
        "median_relative_improvement_when_positive": float(
            comp.loc[comp.relative_improvement > 0, "relative_improvement"].median()
        ),
        "note": "Oracle baselines know block-averaged states exactly and have zero shot noise.",
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    print(
        comp.groupby(["regime", "rounds"], as_index=False).agg(
            median_improvement=("relative_improvement", "median"),
            min_improvement=("relative_improvement", "min"),
            win_rate=("relative_improvement", lambda x: float(np.mean(x > 0))),
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
