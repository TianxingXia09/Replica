from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def minimax_figure():
    df = pd.read_csv(ROOT / "minimax_results" / "matched_copy_bounds.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for k, group in df.groupby("subsystem_size"):
        line, = axes[0].loglog(
            group.total_copies,
            group.paired_local_worst_case_rmse_upper_bound,
            marker="o",
            label=f"k={k}",
        )
        axes[0].loglog(group.total_copies, group.paired_pure_product_exact_rmse,
                       color=line.get_color(), linestyle="--", alpha=0.75)
        axes[0].hlines(
            group.one_copy_minimax_abs_error_lower_bound.iloc[0],
            group.total_copies.min(),
            group.total_copies.max(),
            linestyles="dotted",
            alpha=0.45,
        )
    axes[0].set_xlabel("Total state copies (matched budget)")
    axes[0].set_ylabel("Error bound")
    axes[0].set_title("General-state bound and product-pure RMSE")
    axes[0].grid(True, which="both", alpha=0.25)
    colors = axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[0].add_artist(colors)
    axes[0].legend(handles=[Line2D([0],[0],color="black",label="General-state upper bound"),
                           Line2D([0],[0],color="black",linestyle="--",label="Exact product-pure RMSE"),
                           Line2D([0],[0],color="black",linestyle=":",label="One-copy absolute-error floor")],
                   fontsize=7,loc="lower left")

    cross = pd.read_csv(ROOT / "minimax_results" / "crossover.csv")
    axes[1].semilogy(
        cross.subsystem_size,
        cross.sufficient_total_copies_for_paired_upper_below_one_copy_lower,
        marker="o",
        color="#B24C63",
        label="General-state upper bound",
    )
    axes[1].semilogy(cross.subsystem_size,
        cross.pure_product_total_copies_below_one_copy_lower,
        marker="s",linestyle="--",color="#356C9B",label="Product-pure construction")
    axes[1].set_xlabel("Subsystem size k")
    axes[1].set_ylabel("Sufficient total copies")
    axes[1].set_title("Conservative crossover budget")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "minimax_separation.png", dpi=220)
    fig.savefig(OUT / "minimax_separation.pdf")
    plt.close(fig)


def readout_figure():
    df = pd.read_csv(ROOT / "readout_results" / "aggregate.csv")
    df = df[df.rounds == 2000]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    styles = {"uncorrected": ("--", "o"), "readout_corrected": ("-", "s")}
    for method, group in df.groupby("method"):
        for size, sub in group.groupby("size"):
            ls, marker = styles[method]
            axes[0].plot(
                sub.error_rate,
                sub.bias.abs(),
                linestyle=ls,
                marker=marker,
                label=f"{method}, k={size}",
            )
            axes[1].plot(
                sub.error_rate,
                sub.rmse,
                linestyle=ls,
                marker=marker,
                label=f"{method}, k={size}",
            )
    axes[0].set_title("Absolute bias")
    axes[1].set_title("RMSE")
    for ax in axes:
        ax.set_xlabel("Symmetric readout error")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Error")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "readout_tradeoff.png", dpi=220)
    fig.savefig(OUT / "readout_tradeoff.pdf")
    plt.close(fig)


if __name__ == "__main__":
    minimax_figure()
    readout_figure()
