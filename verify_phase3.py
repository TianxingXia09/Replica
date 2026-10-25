from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "phase3_results"


def main():
    fixed = pd.read_csv(OUT / "boundary_fixed_windows.csv")
    selected = pd.read_csv(OUT / "boundary_selected_windows.csv")
    mismatch = pd.read_csv(OUT / "mismatch_raw.csv")
    aggregate = pd.read_csv(OUT / "mismatch_aggregate.csv")
    boundary_summary = json.loads((OUT / "boundary_summary.json").read_text())
    mismatch_summary = json.loads((OUT / "mismatch_summary.json").read_text())

    assert len(fixed) == 648
    assert len(selected) == 108
    assert len(mismatch) == 24840
    assert not fixed.isna().any().any()
    assert not selected.isna().any().any()
    assert not mismatch.isna().any().any()
    assert float((np.abs(mismatch.exact_bias) - mismatch.hs_bias_bound).max()) <= 1e-12
    expected_rates = {0.0, 0.01, 0.02, 0.05, 0.10, 0.20}
    assert set(np.round(mismatch.mismatch_rate.unique(), 2)) == expected_rates
    task_counts = mismatch.groupby(["n_qubits", "regime", "rep"]).size()
    target_counts = {3: 7, 4: 14, 5: 25}
    for (n, _, _), count in task_counts.items():
        assert count == 6 * target_counts[int(n)]
    assert boundary_summary["selected_window_prediction_accuracy"] > 0.93
    assert boundary_summary["selected_margin_vs_heldout_advantage_correlation"] > 0.99
    assert mismatch_summary["raw_records"] == 24840
    assert len(aggregate) == 162
    print("Phase 3 verification passed.")


if __name__ == "__main__":
    main()
