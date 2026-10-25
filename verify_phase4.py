from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent


def verify_result(folder, expected_tasks, expected_rows, expected_cells):
    path = ROOT / folder
    raw = pd.read_csv(path / "raw.csv")
    comparison = pd.read_csv(path / "comparison.csv")
    summary = json.loads((path / "summary.json").read_text())
    assert summary["tasks"] == expected_tasks
    assert len(raw) == expected_rows
    assert len(comparison) == expected_cells
    assert not raw.isna().any().any()
    assert not comparison.isna().any().any()
    assert (raw.copy_overhead_fraction == 0).all()
    assert summary["frozen_rule"] == {
        "base_threshold": 0.5,
        "floor_threshold": 4.0,
        "floor_band_fraction": 0.1,
    }
    return summary


def main():
    small = verify_result("online_results", 360, 1_214_400, 54)
    large = verify_result("online_large_results", 216, 231_840, 54)
    development = json.loads(
        (ROOT / "online_pilot_results" / "guarded_development_summary.json").read_text()
    )
    assert development["development_only"] is True
    assert development["rule_frozen_for_hidden_test"] == large["frozen_rule"]
    assert large["cells_beating_best_fixed"] == 8
    assert large["significant_wins"] == 8
    assert large["significant_losses"] == 4
    print("Phase 4 verification passed.")


if __name__ == "__main__":
    main()
