from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from baseline_protocols import (
    bell_swap_probabilities,
    bell_swap_subset_kernel,
    global_rm_collision_from_purity,
    sample_bell_swap_signs,
    windowed_shadow_u_statistics,
)
from local_protocol import (
    all_subsystems,
    local_overlap_kernel,
    partial_trace,
    probability_table,
    random_density,
    sample_pair,
    sample_single,
)


ROOT = Path(__file__).resolve().parent
N_QUBITS = [3, 4, 5]
PAIRED_ROUNDS = [500, 2000]
REPS = 40
WINDOWS = [2, 4, 8, 16, 64]
REGIMES = [
    "stationary",
    "alternating",
    "iid_switch",
    "burst_4",
    "burst_16",
    "history_adaptive",
]


def make_bank(n: int, seed: int):
    rng = np.random.default_rng(seed)
    d = 2**n
    ranks = [1, 1, 2, max(2, d // 4), max(2, d // 2), d]
    return [random_density(d, rng, rank=r) for r in ranks]


def every_subset(n: int) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(n + 1)
        for subset in combinations(range(n), size)
    ]


@lru_cache(maxsize=None)
def prepared_problem(n: int):
    bank = make_bank(n, 9100 + n)
    declared = all_subsystems(n, max_size=min(3, n))
    all_sets = every_subset(n)
    tables = [probability_table(rho, n) for rho in bank]
    purity_rows = []
    bell_laws = []
    for rho in bank:
        purity_map = {(): 1.0}
        for subset in all_sets[1:]:
            reduced = partial_trace(rho, subset, n)
            purity_map[subset] = float(np.real(np.trace(reduced @ reduced)))
        purity_rows.append([purity_map[s] for s in declared])
        bell_laws.append(bell_swap_probabilities(purity_map, n))
    return bank, declared, tables, np.asarray(purity_rows), bell_laws


def trajectory(
    regime: str,
    rounds: int,
    bank: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    n_states = len(bank)
    if regime == "stationary":
        return np.zeros(rounds, dtype=int)
    if regime == "alternating":
        return np.arange(rounds, dtype=int) % 2
    if regime == "iid_switch":
        return rng.integers(0, n_states, size=rounds)
    if regime.startswith("burst_"):
        width = int(regime.split("_")[1])
        blocks = rng.integers(0, n_states, size=(rounds + width - 1) // width)
        return np.repeat(blocks, width)[:rounds]
    if regime == "history_adaptive":
        labels = np.empty(rounds, dtype=int)
        labels[0] = 0
        for t in range(1, rounds):
            previous = int(labels[t - 1])
            diagonal = np.real(np.diag(bank[previous]))
            first_qubit_one = float(diagonal[2 ** (int(np.log2(len(diagonal))) - 1) :].sum())
            feedback = int(rng.random() < first_qubit_one)
            labels[t] = (previous + 1 + feedback) % n_states
        return labels
    raise ValueError(regime)


def add_rows(
    rows: list[dict],
    *,
    n: int,
    paired_rounds: int,
    regime: str,
    rep: int,
    method: str,
    access_class: str,
    truth: np.ndarray,
    estimate: np.ndarray,
    subsets: list[tuple[int, ...]],
    copies_used: int,
    copies_for_all_targets: int,
    inter_copy_gates_per_round: int,
    setting_reuse: str,
):
    for index, subset in enumerate(subsets):
        error = float(estimate[index] - truth[index])
        rows.append(
            {
                "n_qubits": n,
                "paired_rounds": paired_rounds,
                "total_copy_budget": 2 * paired_rounds,
                "regime": regime,
                "rep": rep,
                "method": method,
                "access_class": access_class,
                "subsystem": "-".join(map(str, subset)),
                "size": len(subset),
                "truth": float(truth[index]),
                "estimate": float(estimate[index]),
                "error": error,
                "sq_error": error * error,
                "abs_error": abs(error),
                "copies_used": copies_used,
                "copies_for_all_targets": copies_for_all_targets,
                "inter_copy_gates_per_round": inter_copy_gates_per_round,
                "setting_reuse": setting_reuse,
            }
        )


def run_one(task):
    n, paired_rounds, regime, rep, seed = task
    started = time.perf_counter()
    seed_sequence = np.random.SeedSequence(seed)
    rng_pair, rng_one = [np.random.default_rng(s) for s in seed_sequence.spawn(2)]
    bank, subsets, tables, purities, bell_laws = prepared_problem(n)
    target_count = len(subsets)

    pair_labels = trajectory(regime, paired_rounds, bank, rng_pair)
    pair_truth = purities[pair_labels].mean(axis=0)
    sync_sum = np.zeros(target_count)
    bell_sum = np.zeros(target_count)
    rm_multi_sum = np.zeros(target_count)
    rm_single_sum = np.zeros(target_count)

    for label_value in pair_labels:
        label = int(label_value)
        left, right = sample_pair(tables[label], tables[label], n, rng_pair)
        sync_sum += local_overlap_kernel(left, right, subsets)

        sign_support, sign_probabilities = bell_laws[label]
        signs = sample_bell_swap_signs(sign_support, sign_probabilities, rng_pair)
        bell_sum += bell_swap_subset_kernel(signs, subsets)

        selected = int(rng_pair.integers(target_count))
        rm_value = global_rm_collision_from_purity(
            purities[label, selected], len(subsets[selected]), rng_pair
        )
        rm_multi_sum[selected] += target_count * rm_value

        dimensions = np.asarray([2 ** len(s) for s in subsets], dtype=float)
        probabilities = (1.0 + purities[label]) / (dimensions + 1.0)
        rm_single_sum += np.where(rng_pair.random(target_count) < probabilities, dimensions, -1.0)

    methods = [
        (
            "synchronized_local_pauli",
            "C2_product",
            sync_sum / paired_rounds,
            2 * paired_rounds,
            2 * paired_rounds,
            0,
            "all_subsystems",
        ),
        (
            "bell_swap_local_afrs_t2",
            "C3_intercopy_entangling",
            bell_sum / paired_rounds,
            2 * paired_rounds,
            2 * paired_rounds,
            n,
            "all_subsystems",
        ),
        (
            "global_rm_random_allocation",
            "C2_global_basis",
            rm_multi_sum / paired_rounds,
            2 * paired_rounds,
            2 * paired_rounds,
            0,
            "one_random_target_per_pair",
        ),
        (
            "global_rm_single_target",
            "C2_global_basis",
            rm_single_sum / paired_rounds,
            2 * paired_rounds,
            2 * paired_rounds * target_count,
            0,
            "separate_full_budget_per_target",
        ),
    ]

    rows: list[dict] = []
    for method, access, estimate, copies, all_copies, gates, reuse in methods:
        add_rows(
            rows,
            n=n,
            paired_rounds=paired_rounds,
            regime=regime,
            rep=rep,
            method=method,
            access_class=access,
            truth=pair_truth,
            estimate=estimate,
            subsets=subsets,
            copies_used=copies,
            copies_for_all_targets=all_copies,
            inter_copy_gates_per_round=gates,
            setting_reuse=reuse,
        )

    one_rounds = 2 * paired_rounds
    one_labels = trajectory(regime, one_rounds, bank, rng_one)
    one_truth = purities[one_labels].mean(axis=0)
    settings = np.empty((one_rounds, n), dtype=np.int8)
    bits = np.empty((one_rounds, n), dtype=np.int8)
    for t, label_value in enumerate(one_labels):
        settings[t], bits[t] = sample_single(tables[int(label_value)], n, rng_one)
    window_values = windowed_shadow_u_statistics(
        settings, bits, subsets, WINDOWS + [one_rounds]
    )
    for width in WINDOWS + [one_rounds]:
        method = "shadow_u_pooled" if width == one_rounds else f"shadow_u_window_{width}"
        add_rows(
            rows,
            n=n,
            paired_rounds=paired_rounds,
            regime=regime,
            rep=rep,
            method=method,
            access_class="C1_or_C4_temporal",
            truth=one_truth,
            estimate=window_values[width],
            subsets=subsets,
            copies_used=one_rounds,
            copies_for_all_targets=one_rounds,
            inter_copy_gates_per_round=0,
            setting_reuse="all_subsystems_cross_time",
        )
    elapsed = time.perf_counter() - started
    return rows, {
        "n_qubits": n,
        "paired_rounds": paired_rounds,
        "regime": regime,
        "rep": rep,
        "seconds": elapsed,
    }


def bootstrap_rmse_ci(per_rep_mse: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    count = len(per_rep_mse)
    samples = rng.integers(0, count, size=(2000, count))
    rmse = np.sqrt(per_rep_mse[samples].mean(axis=1))
    return tuple(float(x) for x in np.quantile(rmse, [0.025, 0.975]))


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["n_qubits", "paired_rounds", "regime", "size", "method", "access_class"]
    records = []
    seed_rng = np.random.default_rng(20261102)
    for key, group in raw.groupby(keys, sort=True):
        per_rep = group.groupby("rep", as_index=False).agg(
            mse=("sq_error", "mean"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
        )
        ci_low, ci_high = bootstrap_rmse_ci(
            per_rep.mse.to_numpy(), int(seed_rng.integers(2**31 - 1))
        )
        records.append(
            dict(
                zip(keys, key),
                rmse=float(np.sqrt(per_rep.mse.mean())),
                rmse_ci_low=ci_low,
                rmse_ci_high=ci_high,
                mae=float(per_rep.mae.mean()),
                bias=float(per_rep.bias.mean()),
                n_repetitions=int(len(per_rep)),
                copies_used=int(group.copies_used.iloc[0]),
                copies_for_all_targets=int(group.copies_for_all_targets.iloc[0]),
                inter_copy_gates_per_round=int(group.inter_copy_gates_per_round.iloc[0]),
                setting_reuse=str(group.setting_reuse.iloc[0]),
            )
        )
    return pd.DataFrame(records)


def comparison_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    proposed = aggregate[aggregate.method == "synchronized_local_pauli"]
    records = []
    for _, row in proposed.iterrows():
        peers = aggregate[
            (aggregate.n_qubits == row.n_qubits)
            & (aggregate.paired_rounds == row.paired_rounds)
            & (aggregate.regime == row.regime)
            & (aggregate["size"] == row["size"])
            & (aggregate.method != "synchronized_local_pauli")
        ]
        for _, peer in peers.iterrows():
            if float(peer.rmse) == 0.0:
                relative_reduction = np.nan
                rmse_ratio = np.inf if float(row.rmse) > 0.0 else 1.0
            else:
                relative_reduction = float(1.0 - row.rmse / peer.rmse)
                rmse_ratio = float(row.rmse / peer.rmse)
            records.append(
                {
                    "n_qubits": int(row.n_qubits),
                    "paired_rounds": int(row.paired_rounds),
                    "regime": row.regime,
                    "size": int(row["size"]),
                    "comparator": peer.method,
                    "proposed_rmse": float(row.rmse),
                    "comparator_rmse": float(peer.rmse),
                    "proposed_to_comparator_rmse_ratio": rmse_ratio,
                    "relative_rmse_reduction": relative_reduction,
                    "same_total_copy_budget_for_all_targets": bool(
                        row.copies_for_all_targets == peer.copies_for_all_targets
                    ),
                    "same_access_class": bool(row.access_class == peer.access_class),
                }
            )
    return pd.DataFrame(records)


def bootstrap_comparison_intervals(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["n_qubits", "paired_rounds", "regime", "size"]
    per_rep = raw.groupby(keys + ["method", "rep"], as_index=False).agg(
        mse=("sq_error", "mean")
    )
    proposed = per_rep[per_rep.method == "synchronized_local_pauli"]
    rng = np.random.default_rng(20261103)
    records = []
    for key, proposed_group in proposed.groupby(keys, sort=True):
        selector = np.ones(len(per_rep), dtype=bool)
        for column, value in zip(keys, key):
            selector &= per_rep[column].to_numpy() == value
        peers = per_rep[selector & (per_rep.method != "synchronized_local_pauli")]
        proposed_values = proposed_group.sort_values("rep").mse.to_numpy()
        for method, peer_group in peers.groupby("method", sort=True):
            peer_values = peer_group.sort_values("rep").mse.to_numpy()
            if len(peer_values) != len(proposed_values):
                raise ValueError("Comparator repetitions do not align.")
            count = len(proposed_values)
            indices = rng.integers(0, count, size=(2000, count))
            proposed_rmse = np.sqrt(proposed_values[indices].mean(axis=1))
            comparator_rmse = np.sqrt(peer_values[indices].mean(axis=1))
            valid = comparator_rmse > 0.0
            reductions = 1.0 - proposed_rmse[valid] / comparator_rmse[valid]
            if len(reductions):
                lower, upper = np.quantile(reductions, [0.025, 0.975])
                median = np.median(reductions)
            else:
                lower = upper = median = np.nan
            records.append(
                {
                    **dict(zip(keys, key)),
                    "comparator": method,
                    "median_bootstrap_reduction": float(median),
                    "reduction_ci_low": float(lower),
                    "reduction_ci_high": float(upper),
                    "significant_win": bool(lower > 0.0),
                    "significant_20pct_win": bool(lower >= 0.2),
                    "significant_loss": bool(upper < 0.0),
                    "valid_bootstrap_draws": int(len(reductions)),
                }
            )
    return pd.DataFrame(records)


def finalize_existing(output: Path):
    aggregate = pd.read_csv(output / "aggregate.csv")
    raw = pd.read_csv(output / "raw.csv")
    comparisons = comparison_table(aggregate)
    comparisons.to_csv(output / "comparisons.csv", index=False)
    significance = bootstrap_comparison_intervals(raw)
    significance.to_csv(output / "significance.csv", index=False)
    matched = comparisons[comparisons.same_total_copy_budget_for_all_targets]
    finite = matched[np.isfinite(matched.relative_rmse_reduction)]
    summary = {
        "configurations": int(
            aggregate[["n_qubits", "paired_rounds", "regime"]].drop_duplicates().shape[0]
            * REPS
        ),
        "repetitions_per_cell": REPS,
        "all_unit_of_comparison": "total prepared n-qubit systems",
        "bell_swap_note": "Statistically strong C3 baseline; local-AFRS t=2 moment specialization has the same SWAP-sign variance bound.",
        "global_rm_single_target_note": "Uses the full budget separately for every target and therefore costs M times more copies for all targets.",
        "zero_rmse_comparisons": int((matched.comparator_rmse == 0.0).sum()),
        "significant_matched_budget_wins": int(
            significance[
                significance.comparator != "global_rm_single_target"
            ].significant_win.sum()
        ),
        "median_relative_reduction_vs_matched_comparators": {
            str(name): float(group.relative_rmse_reduction.median())
            for name, group in finite.groupby("comparator")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def all_tasks():
    master = np.random.default_rng(20261101)
    return [
        (n, rounds, regime, rep, int(master.integers(2**31 - 1)))
        for n in N_QUBITS
        for rounds in PAIRED_ROUNDS
        for regime in REGIMES
        for rep in range(REPS)
    ]


def resume_existing(output: Path):
    raw_path = output / "raw.csv"
    timing_path = output / "timing.csv"
    raw = pd.read_csv(raw_path)
    timing = pd.read_csv(timing_path)
    expected_targets = {
        n: len(all_subsystems(n, max_size=min(3, n))) for n in N_QUBITS
    }
    task_columns = ["n_qubits", "paired_rounds", "regime", "rep"]
    counts = raw.groupby(task_columns).size().to_dict()
    tasks = all_tasks()
    incomplete = []
    incomplete_keys = set()
    for task in tasks:
        key = task[:4]
        expected_rows = 10 * expected_targets[int(task[0])]
        if int(counts.get(key, 0)) != expected_rows:
            incomplete.append(task)
            incomplete_keys.add(key)

    if incomplete_keys:
        keep_raw = [
            tuple(row) not in incomplete_keys
            for row in raw[task_columns].itertuples(index=False, name=None)
        ]
        raw = raw[np.asarray(keep_raw, dtype=bool)]
        keep_timing = [
            tuple(row) not in incomplete_keys
            for row in timing[task_columns].itertuples(index=False, name=None)
        ]
        timing = timing[np.asarray(keep_timing, dtype=bool)]

    new_rows = []
    new_timing = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for task_rows, task_timing in executor.map(run_one, incomplete, chunksize=1):
            new_rows.extend(task_rows)
            new_timing.append(task_timing)
    raw = pd.concat([raw, pd.DataFrame(new_rows)], ignore_index=True)
    timing = pd.concat([timing, pd.DataFrame(new_timing)], ignore_index=True)
    raw.to_csv(raw_path, index=False)
    timing.to_csv(timing_path, index=False)
    aggregate_results(raw).to_csv(output / "aggregate.csv", index=False)
    print(json.dumps({"resumed_tasks": len(incomplete), "raw_rows": len(raw)}, indent=2))
    finalize_existing(output)


def main():
    tasks = all_tasks()
    rows = []
    timing = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for task_rows, task_timing in executor.map(run_one, tasks, chunksize=1):
            rows.extend(task_rows)
            timing.append(task_timing)

    output = ROOT / "phase2_results"
    output.mkdir(exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "raw.csv", index=False)
    pd.DataFrame(timing).to_csv(output / "timing.csv", index=False)
    aggregate = aggregate_results(raw)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    comparisons = comparison_table(aggregate)
    comparisons.to_csv(output / "comparisons.csv", index=False)

    finalize_existing(output)


if __name__ == "__main__":
    if "--resume" in sys.argv:
        resume_existing(ROOT / "phase2_results")
    elif "--finalize-existing" in sys.argv:
        finalize_existing(ROOT / "phase2_results")
    else:
        main()
