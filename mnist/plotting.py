"""Shared plot style and the training-curve figure used by both training scripts."""

from pathlib import Path

import numpy as np

# Colour-blind-safe categorical palette (checked for CVD separation) on a white surface.
SERIES_COLORS = ("#257de0", "#f47230", "#00b894")  # blue, orange, teal
TRAIN_COLOR, VAL_COLOR = SERIES_COLORS[:2]
TEXT, TEXT_2, GRID = "#191919", "#666666", "#e6e6e6"


def plot_history(history: dict, path: Path, title: str) -> None:
    """Save loss and accuracy curves (train vs val) side by side, marking the best epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best = history["best_epoch"]
    panels = [
        ("Cross-entropy loss", "train_loss", "val_loss", "{:.3f}"),
        ("Accuracy", "train_acc", "val_acc", "{:.1%}"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (panel_title, train_key, val_key, fmt) in zip(axes, panels):
        for key, label, color in ((train_key, "train", TRAIN_COLOR), (val_key, "val", VAL_COLOR)):
            values = history[key]
            ax.plot(epochs, values, color=color, linewidth=2, label=label)
            # direct label at the line end, in text color (identity comes from the line beside it)
            ax.annotate(f"{label} {fmt.format(values[-1])}", (epochs[-1], values[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        color=TEXT, fontsize=9)
        ax.axvline(best, color=TEXT_2, linewidth=1, linestyle=":")
        ax.set_title(panel_title, loc="left", color=TEXT, fontsize=11)
        ax.set_xlabel("epoch", color=TEXT_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=9)
        ax.set_xlim(0.5, epochs[-1] + max(2, 0.22 * len(epochs)))  # room for end labels
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=None))
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=13)
    fig.text(0.01, 0.905, f"Dotted line: best validation epoch ({best}); final weights are "
             "restored from there.", ha="left", color=TEXT_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
