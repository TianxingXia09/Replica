from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "phase3_results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def main():
    boundary = pd.read_csv(OUT / "boundary_selected_windows.csv")
    mismatch = pd.read_csv(OUT / "mismatch_aggregate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    colors = np.where(boundary.correct_prediction, "#176b87", "#d1495b")
    ax.scatter(
        boundary.threshold_margin,
        boundary.actual_mse_advantage,
        c=colors,
        s=34,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axhline(0, color="0.25", lw=0.9)
    ax.axvline(0, color="0.25", lw=0.9)
    ax.set_xlabel(r"predicted boundary margin $D_w^2-(V_{sync}-V_w)$")
    ax.set_ylabel(r"held-out MSE advantage $\mathrm{MSE}_w-\mathrm{MSE}_{sync}$")
    ax.set_title("(a) Held-out phase-boundary test")

    ax = axes[1]
    grouped = mismatch.groupby("mismatch_rate", as_index=False).agg(
        median=("rmse_inflation", "median"),
        low=("rmse_inflation", lambda x: float(np.quantile(x, 0.1))),
        high=("rmse_inflation", lambda x: float(np.quantile(x, 0.9))),
    )
    x = 100 * grouped.mismatch_rate.to_numpy()
    ax.plot(x, grouped["median"], marker="o", color="#7a5195", lw=2)
    ax.fill_between(x, grouped.low, grouped.high, color="#7a5195", alpha=0.18)
    ax.axhline(1, color="0.25", lw=0.9)
    ax.set_xlabel("replication mismatch probability (%)")
    ax.set_ylabel("RMSE / matched-copy RMSE")
    ax.set_title("(b) Robustness to wrong-state pairing")

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"phase3_boundary_mismatch.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
