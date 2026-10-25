from pathlib import Path
import json

import numpy as np
import pandas as pd

from online_selector import append_guarded_method


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "online_pilot_results"
ROUNDS = 512


def main():
    raw_path = OUT / "pilot_raw.csv"
    raw = append_guarded_method(pd.read_csv(raw_path), ROUNDS)
    raw.to_csv(raw_path, index=False)
    keys = ["n_qubits", "schedule", "size", "method"]
    aggregate = raw.groupby(keys, as_index=False).agg(
        mse=("sq_error", "mean"),
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
    )
    aggregate.to_csv(OUT / "pilot_aggregate.csv", index=False)
    cells = ["n_qubits", "schedule", "size"]
    experts = aggregate[
        aggregate.method.isin(
            ["sync", "temporal_w2", "temporal_w16", "temporal_w64", "temporal_pooled"]
        )
    ]
    best = experts.loc[experts.groupby(cells).mse.idxmin()][cells + ["method", "mse"]]
    guarded = aggregate[aggregate.method == "observable_guarded_gate"][cells + ["mse"]]
    comparison = guarded.merge(best, on=cells, suffixes=("_guarded", "_best_fixed"))
    comparison["relative_mse_reduction"] = 1 - comparison.mse_guarded / comparison.mse_best_fixed
    comparison.to_csv(OUT / "guarded_development_comparison.csv", index=False)
    summary = {
        "development_only": True,
        "cells": len(comparison),
        "cells_beating_best_fixed": int((comparison.mse_guarded < comparison.mse_best_fixed).sum()),
        "median_relative_mse_reduction": float(comparison.relative_mse_reduction.median()),
        "rule_frozen_for_hidden_test": {
            "base_threshold": 0.5,
            "floor_threshold": 4.0,
            "floor_band_fraction": 0.1,
        },
    }
    (OUT / "guarded_development_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
