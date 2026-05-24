"""Visualize accuracy_matrix.json as a heatmap.

Rows = model sources (gm, lm1 … lm10)
Cols = test areas   (E1 … E10)

Color is column-normalised: within each area column the best model gets the
maximum shade, making it immediately clear which model wins on each area's
local test data.  The winning cell is also starred (*).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

MODEL_ORDER = ["gm"] + [f"lm{i}" for i in range(1, 11)]
AREA_ORDER  = [f"E{i}" for i in range(1, 11)]


def load_matrix(path: Path, config: str, metric: str, split: str) -> np.ndarray:
    """Return (n_models × n_areas) matrix averaged over all runs."""
    with open(path) as f:
        data = json.load(f)

    configs = list(data.keys())
    if config not in data:
        raise ValueError(
            f"Config '{config}' not found.\nAvailable:\n  " + "\n  ".join(configs)
        )

    runs = data[config]
    # sum values across runs, then average
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for run_data in runs.values():
        entries = run_data["hybrid_mse_avg"][split][metric]
        for key, val in entries.items():
            totals[key] = totals.get(key, 0.0) + val
            counts[key] = counts.get(key, 0) + 1

    mat = np.full((len(MODEL_ORDER), len(AREA_ORDER)), np.nan)
    for r, model in enumerate(MODEL_ORDER):
        for c, area in enumerate(AREA_ORDER):
            key = f"{model}->{area}"
            if key in totals:
                mat[r, c] = totals[key] / counts[key]
    return mat


def column_normalised(mat: np.ndarray) -> np.ndarray:
    """Normalise each column to [0, 1] relative to that column's max."""
    col_min = np.nanmin(mat, axis=0, keepdims=True)
    col_max = np.nanmax(mat, axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        normed = (mat - col_min) / np.where(col_max - col_min == 0, 1, col_max - col_min)
    return normed


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Heatmap of accuracy matrix")
    parser.add_argument(
        "--data", default="configs/accuracy_matrix.json",
        help="Path to accuracy_matrix.json"
    )
    parser.add_argument(
        "--config", default=None,
        help="Experiment config key (takes precedence over --alpha)"
    )
    parser.add_argument(
        "--alpha", default=None,
        help="Alpha value to filter configs (e.g., 0.1, 1000)"
    )
    parser.add_argument(
        "--metric", default="auc", choices=["auc", "tnr", "tpr"],
        help="Metric to visualise (default: auc)"
    )
    parser.add_argument(
        "--split", default="testing", choices=["testing", "validation"],
        help="Data split (default: testing)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Save figure to this path instead of showing it"
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    with open(data_path) as f:
        all_configs = list(json.load(f).keys())

    if args.config:
        config = args.config
    elif args.alpha:
        pattern = f"alpha{args.alpha}"
        matches = [c for c in all_configs if pattern in c]
        if not matches:
            # try without "alpha" prefix just in case
            matches = [c for c in all_configs if args.alpha in c]
        
        if not matches:
            print(f"Error: No config found matching alpha '{args.alpha}'.")
            print("Available configs:")
            for c in all_configs:
                print(f"  {c}")
            return
        config = matches[0]
        if len(matches) > 1:
            print(f"Note: Multiple matches for alpha '{args.alpha}', picking: {config}")
    else:
        config = all_configs[0]

    mat = load_matrix(data_path, config, args.metric, args.split)
    normed = column_normalised(mat)

    # best model index per column (for star annotation)
    best_row = np.nanargmax(mat, axis=0)  # shape: (n_areas,)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))

    cmap = mcolors.LinearSegmentedColormap.from_list("white_green", ["white", "#1a7d3a"])
    im = ax.imshow(normed, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # annotate each cell with the raw metric value
    for r in range(len(MODEL_ORDER)):
        for c in range(len(AREA_ORDER)):
            val = mat[r, c]
            if np.isnan(val):
                continue
            # choose text colour for readability
            bg = normed[r, c]
            text_color = "white" if bg > 0.7 else "black"
            label = f"{val:.4f}"
            is_best = (best_row[c] == r)
            ax.text(
                c, r, ("★ " if is_best else "") + label,
                ha="center", va="center",
                fontsize=7.5,
                color=text_color,
                fontweight="bold" if is_best else "normal",
            )

    ax.set_xticks(range(len(AREA_ORDER)))
    ax.set_xticklabels(AREA_ORDER, fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=10)
    ax.set_xlabel("Test Area", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    alpha_tag = config.replace("NonIID-10-Client_Data-", "")
    ax.set_title(
        f"Accuracy Matrix — {args.metric.upper()} ({args.split})\n"
        f"Config: {alpha_tag}  |  colour = column-normalised (★ = best per area)",
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relative performance (per area)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["worst", "mid", "best"])

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
