from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "phase2_results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def alternating_rmse_figure(aggregate: pd.DataFrame):
    selected = aggregate[
        (aggregate.paired_rounds == 2000)
        & (aggregate.regime == "alternating")
        & aggregate.method.isin(
            [
                "synchronized_local_pauli",
                "bell_swap_local_afrs_t2",
                "global_rm_random_allocation",
            ]
        )
    ].copy()
    temporal = aggregate[
        (aggregate.paired_rounds == 2000)
        & (aggregate.regime == "alternating")
        & aggregate.method.str.startswith("shadow_u_")
    ]
    temporal = temporal.loc[
        temporal.groupby(["n_qubits", "size"]).rmse.idxmin()
    ].copy()
    temporal["method"] = "best_temporal_u_statistic"
    selected = pd.concat([selected, temporal], ignore_index=True)

    labels = {
        "synchronized_local_pauli": "Synchronized local",
        "bell_swap_local_afrs_t2": "Bell-SWAP / AFRS-$t=2$",
        "global_rm_random_allocation": "Global RM allocation",
        "best_temporal_u_statistic": "Best temporal U-statistic",
    }
    colors = {
        "synchronized_local_pauli": "#1f77b4",
        "bell_swap_local_afrs_t2": "#2ca02c",
        "global_rm_random_allocation": "#d62728",
        "best_temporal_u_statistic": "#7f7f7f",
    }
    methods = list(labels)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25), sharey=True)
    x = np.arange(1, 4)
    for axis, n in zip(axes, [3, 4, 5]):
        panel = selected[selected.n_qubits == n]
        for method in methods:
            row = panel[panel.method == method].sort_values("size")
            axis.plot(
                row["size"],
                row.rmse,
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=labels[method],
            )
        axis.set_title(f"$n={n}$")
        axis.set_xticks(x)
        axis.set_xlabel("Subsystem size $k$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("RMSE")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    for extension in ["pdf", "png"]:
        fig.savefig(FIGURES / f"phase2_alternating_rmse.{extension}", dpi=220)
    plt.close(fig)


def temporal_advantage_heatmap(significance: pd.DataFrame):
    data = significance[
        (significance.paired_rounds == 2000)
        & significance.comparator.str.startswith("shadow_u_")
    ]
    worst = data.sort_values("median_bootstrap_reduction").groupby(
        ["regime", "n_qubits", "size"], as_index=False
    ).first()
    regimes = [
        "stationary",
        "alternating",
        "iid_switch",
        "burst_4",
        "burst_16",
        "history_adaptive",
    ]
    rows = [(regime, n) for regime in regimes for n in [3, 4, 5]]
    matrix = np.full((len(rows), 3), np.nan)
    lower = np.full_like(matrix, np.nan)
    for row_index, (regime, n) in enumerate(rows):
        for size in [1, 2, 3]:
            item = worst[
                (worst.regime == regime)
                & (worst.n_qubits == n)
                & (worst["size"] == size)
            ].iloc[0]
            matrix[row_index, size - 1] = item.median_bootstrap_reduction
            lower[row_index, size - 1] = item.reduction_ci_low

    fig, axis = plt.subplots(figsize=(6.0, 7.2))
    image = axis.imshow(matrix * 100, cmap="RdBu", vmin=-150, vmax=150, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            marker = "*" if lower[i, j] > 0 else ""
            axis.text(
                j,
                i,
                f"{100 * matrix[i, j]:.0f}{marker}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="black",
            )
    axis.set_xticks([0, 1, 2], ["$k=1$", "$k=2$", "$k=3$"])
    axis.set_yticks(
        np.arange(len(rows)), [f"{regime}, n={n}" for regime, n in rows], fontsize=7.5
    )
    axis.set_title("RMSE reduction vs strongest temporal baseline (%, $N=2000$)")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.75)
    colorbar.set_label("Positive favors synchronized local estimator")
    axis.text(
        0,
        len(rows) + 0.45,
        "* 95% cluster-bootstrap interval excludes zero",
        fontsize=8,
    )
    fig.tight_layout()
    for extension in ["pdf", "png"]:
        fig.savefig(FIGURES / f"phase2_temporal_advantage.{extension}", dpi=220)
    plt.close(fig)


def main():
    aggregate = pd.read_csv(RESULTS / "aggregate.csv")
    significance = pd.read_csv(RESULTS / "significance.csv")
    alternating_rmse_figure(aggregate)
    temporal_advantage_heatmap(significance)


if __name__ == "__main__":
    main()
