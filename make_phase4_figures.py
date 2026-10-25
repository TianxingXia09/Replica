from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


def panel(ax, path, title):
    data = pd.read_csv(path)
    table = data.groupby(["schedule", "size"])[
        "relative_mse_reduction_vs_best_fixed"
    ].median().unstack()
    values = table.to_numpy()
    shown = np.clip(values, -1.0, 0.5)
    image = ax.imshow(shown, cmap="RdBu", vmin=-1.0, vmax=1.0, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if abs(shown[row, column]) > 0.55 else "black"
            ax.text(column, row, f"{100 * value:+.0f}%", ha="center", va="center", color=color, fontsize=7)
    ax.set_xticks(range(len(table.columns)), [str(x) for x in table.columns])
    ax.set_yticks(range(len(table.index)), [x.replace("_", " ") for x in table.index], fontsize=7)
    ax.set_xlabel("subsystem size")
    ax.set_title(title, fontsize=9)
    return image


def main():
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    image = panel(
        axes[0], ROOT / "online_results" / "comparison.csv", "(a) held out, 512 rounds"
    )
    panel(
        axes[1],
        ROOT / "online_large_results" / "comparison.csv",
        "(b) second held out, 2000 rounds",
    )
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("MSE reduction versus best fixed estimator")
    fig.savefig(FIGURES / "phase4_observable_gate.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "phase4_observable_gate.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
