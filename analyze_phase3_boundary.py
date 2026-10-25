from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from local_protocol import partial_trace
from phase3_theory import block_temporal_dispersion, gram_matrix
from run_phase2_benchmarks import WINDOWS, all_tasks, prepared_problem, trajectory


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "phase3_results"


def subsystem_name(subset: tuple[int, ...]) -> str:
    return "-".join(map(str, subset))


def reconstruct_dispersions() -> pd.DataFrame:
    rows: list[dict] = []
    cached_grams: dict[int, list[np.ndarray]] = {}
    for n, paired_rounds, regime, rep, seed in all_tasks():
        bank, subsets, _, purities, _ = prepared_problem(n)
        if n not in cached_grams:
            cached_grams[n] = []
            for subset in subsets:
                reductions = [partial_trace(rho, subset, n) for rho in bank]
                cached_grams[n].append(gram_matrix(reductions))
        spawned = np.random.SeedSequence(seed).spawn(2)
        rng_one = np.random.default_rng(spawned[1])
        one_rounds = 2 * paired_rounds
        labels = trajectory(regime, one_rounds, bank, rng_one)
        for subset_index, subset in enumerate(subsets):
            purity = purities[:, subset_index]
            gram = cached_grams[n][subset_index]
            for width in WINDOWS + [one_rounds]:
                method = "shadow_u_pooled" if width == one_rounds else f"shadow_u_window_{width}"
                rows.append(
                    {
                        "n_qubits": n,
                        "paired_rounds": paired_rounds,
                        "regime": regime,
                        "rep": rep,
                        "subsystem": subsystem_name(subset),
                        "size": len(subset),
                        "method": method,
                        "window": width,
                        "dispersion": block_temporal_dispersion(labels, purity, gram, width),
                    }
                )
    return pd.DataFrame(rows)


def calibration_boundary_margin(cells: pd.DataFrame) -> pd.Series:
    """Use only calibration data to select the strategy."""
    return cells.cal_dispersion_sq - (cells.cal_sync_noise_mse - cells.cal_temporal_noise_mse)


def summarize_boundary(raw: pd.DataFrame, dispersion: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    keys = ["n_qubits", "paired_rounds", "regime", "size"]
    sync = raw[raw.method == "synchronized_local_pauli"].copy()
    temporal = raw[raw.method.str.startswith("shadow_u_")].merge(
        dispersion,
        on=["n_qubits", "paired_rounds", "regime", "rep", "subsystem", "size", "method"],
        validate="one_to_one",
    )
    temporal["noise_error"] = temporal.error + temporal.dispersion
    temporal["dispersion_sq"] = temporal.dispersion**2

    calibration_sync = (
        sync[sync.rep < 20].groupby(keys, as_index=False).sq_error.mean()
        .rename(columns={"sq_error": "cal_sync_noise_mse"})
    )
    test_sync = (
        sync[sync.rep >= 20].groupby(keys, as_index=False).sq_error.mean()
        .rename(columns={"sq_error": "test_sync_mse"})
    )
    temporal_keys = keys + ["method", "window"]
    calibration_temporal = temporal[temporal.rep < 20].groupby(temporal_keys, as_index=False).agg(
        cal_temporal_noise_mse=("noise_error", lambda x: float(np.mean(np.square(x)))),
        cal_dispersion_sq=("dispersion_sq", "mean"),
    )
    test_temporal = temporal[temporal.rep >= 20].groupby(temporal_keys, as_index=False).agg(
        test_temporal_mse=("sq_error", "mean"),
        test_dispersion_sq=("dispersion_sq", "mean"),
        test_mean_dispersion=("dispersion", "mean"),
    )
    cells = calibration_temporal.merge(calibration_sync, on=keys).merge(
        test_temporal, on=temporal_keys
    ).merge(test_sync, on=keys)
    cells["threshold_margin"] = calibration_boundary_margin(cells)
    cells["predicted_sync_win"] = cells.threshold_margin > 0
    cells["actual_sync_win"] = cells.test_sync_mse < cells.test_temporal_mse
    cells["actual_mse_advantage"] = cells.test_temporal_mse - cells.test_sync_mse
    cells["correct_prediction"] = cells.predicted_sync_win == cells.actual_sync_win

    selection_score = cells.cal_temporal_noise_mse + cells.cal_dispersion_sq
    selected_indices = cells.assign(selection_score=selection_score).groupby(keys).selection_score.idxmin()
    selected = cells.loc[selected_indices].copy()
    selected_accuracy = float(selected.correct_prediction.mean())
    actual_advantage = selected.actual_mse_advantage.to_numpy()
    margin = selected.threshold_margin.to_numpy()
    correlation = float(np.corrcoef(margin, actual_advantage)[0, 1])

    summary = {
        "calibration_repetitions": 20,
        "held_out_repetitions": 20,
        "fixed_window_cells": int(len(cells)),
        "fixed_window_prediction_accuracy": float(cells.correct_prediction.mean()),
        "calibration_selected_window_cells": int(len(selected)),
        "selected_window_prediction_accuracy": selected_accuracy,
        "selected_margin_vs_heldout_advantage_correlation": correlation,
        "selected_predicted_wins": int(selected.predicted_sync_win.sum()),
        "selected_actual_wins": int(selected.actual_sync_win.sum()),
    }
    return cells, selected, summary


def main():
    OUT.mkdir(exist_ok=True)
    dispersion_path = OUT / "temporal_dispersion.csv"
    if dispersion_path.exists():
        dispersion = pd.read_csv(dispersion_path)
    else:
        dispersion = reconstruct_dispersions()
        dispersion.to_csv(dispersion_path, index=False)
    raw = pd.read_csv(ROOT / "phase2_results" / "raw.csv")
    cells, selected, summary = summarize_boundary(raw, dispersion)
    cells.to_csv(OUT / "boundary_fixed_windows.csv", index=False)
    selected.to_csv(OUT / "boundary_selected_windows.csv", index=False)
    (OUT / "boundary_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
