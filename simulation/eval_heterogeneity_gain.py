"""
Evaluate heterogeneity gain from local vs global IDS models across Dirichlet alpha values.
Plots AUC, TPR, and TNR across four analysis panels (3 rows × 4 cols).

Sampling mirrors environment._load_accuracy_matrix:
  - For each (scenario, run), randomly draw n_edges clients without replacement.
  - For each selected client c (acting as an edge):
      gm_val        = metric["gm->Ec"]
      lm_self_val   = metric["lmC->Ec"]
      lm_cross_val  = mean(metric["lmC'->Ec"] for c' != c in selected)
      mean_lms_val  = mean(metric["lmC'->Ec"] for c' in selected)
  - best_gain    = lm_self_val  - gm_val
  - avg_gain     = mean_lms_val - gm_val
  - self_vs_cross = lm_self_val - lm_cross_val

Usage:
    python eval_heterogeneity_gain.py
    python eval_heterogeneity_gain.py --n_edges 4 --n_samples 50 --out results/hetero
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS: List[Tuple[str, str]] = [
    ("auc", "AUC"),
    ("tpr", "TPR"),
    ("tnr", "TNR"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_alpha(key: str) -> float:
    m = re.search(r'alpha([\d.]+)', key)
    return float(m.group(1)) if m else 1.0


def _discover_clients(metric_map: Dict[str, float]) -> List[int]:
    clients: set[int] = set()
    for k in metric_map:
        m = re.search(r'->E(\d+)$', k)
        if m:
            clients.add(int(m.group(1)))
    return sorted(clients)


# ---------------------------------------------------------------------------
# Per-metric sampling (mirrors environment._load_accuracy_matrix)
# ---------------------------------------------------------------------------

def _sample_metric_once(
    metric_map: Dict[str, float],
    all_clients: List[int],
    n_edges: int,
    rng: np.random.Generator,
) -> Optional[Dict[str, float]]:
    """
    Draw n_edges clients, compute per-edge statistics for one metric.
    Returns None if no usable data found.
    """
    selected: List[int] = [
        int(c) for c in rng.choice(all_clients, size=n_edges, replace=False)
    ]

    has_gm = any(k.startswith("gm") for k in metric_map)
    has_lm = any(k.startswith("lm") for k in metric_map)

    gm_vals:       List[float] = []
    lm_self_vals:  List[float] = []
    lm_cross_vals: List[float] = []
    inter_vals:    List[float] = []
    best_gains:    List[float] = []
    avg_gains:     List[float] = []
    self_vs_cross: List[float] = []

    for c in selected:
        gm_val: Optional[float] = None
        if has_gm:
            gm_key = f"gm->E{c}"
            if gm_key in metric_map:
                gm_val = metric_map[gm_key]
                gm_vals.append(gm_val)

        lm_self_val: Optional[float] = None
        if has_lm:
            lm_self_key = f"lm{c}->E{c}"
            if lm_self_key in metric_map:
                lm_self_val = metric_map[lm_self_key]
                lm_self_vals.append(lm_self_val)

            all_lms_on_c = [
                metric_map[f"lm{c2}->E{c}"]
                for c2 in selected
                if f"lm{c2}->E{c}" in metric_map
            ]
            cross_lms_on_c = [
                metric_map[f"lm{c2}->E{c}"]
                for c2 in selected
                if c2 != c and f"lm{c2}->E{c}" in metric_map
            ]
            if all_lms_on_c:
                inter_vals.append(float(np.mean(all_lms_on_c)))
            if cross_lms_on_c:
                lm_cross_val = float(np.mean(cross_lms_on_c))
                lm_cross_vals.append(lm_cross_val)
                if lm_self_val is not None:
                    self_vs_cross.append(lm_self_val - lm_cross_val)
            if all_lms_on_c and gm_val is not None:
                avg_gains.append(float(np.mean(all_lms_on_c)) - gm_val)
            if lm_self_val is not None and gm_val is not None:
                best_gains.append(lm_self_val - gm_val)

    if not gm_vals and not lm_self_vals:
        return None

    result: Dict[str, float] = {}
    for vals, mk, sk in [
        (gm_vals,       "gm_mean",             "gm_std"),
        (lm_self_vals,  "lm_mean",             "lm_std"),
        (inter_vals,    "inter_model_mean",     "inter_model_std"),
        (lm_cross_vals, "lm_cross_mean",        "lm_cross_std"),
        (best_gains,    "best_gain_mean",       "best_gain_std"),
        (avg_gains,     "avg_gain_mean",        "avg_gain_std"),
        (self_vs_cross, "self_vs_cross_mean",   "self_vs_cross_std"),
    ]:
        if vals:
            result[mk] = float(np.mean(vals))
            result[sk] = float(np.std(vals))
    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

STAT_FIELDS: List[Tuple[str, str, str]] = [
    ("gm",           "gm_mean",           "gm_std"),
    ("lm",           "lm_mean",           "lm_std"),
    ("inter",        "inter_model_mean",  "inter_model_std"),
    ("lm_cross",     "lm_cross_mean",     "lm_cross_std"),
    ("best_gain",    "best_gain_mean",    "best_gain_std"),
    ("avg_gain",     "avg_gain_mean",     "avg_gain_std"),
    ("self_vs_cross","self_vs_cross_mean","self_vs_cross_std"),
]


def analyze(
    data: dict,
    n_edges: int,
    n_samples: int,
    seed: int,
) -> List[dict]:
    master_rng = np.random.default_rng(seed)
    alpha_results: List[dict] = []

    for scenario_key, runs in data.items():
        alpha = _extract_alpha(scenario_key)
        entry: dict = {"alpha": alpha, "scenario_key": scenario_key}

        # Accumulate per metric
        for metric_key, _ in METRICS:
            accum: Dict[str, List[float]] = {
                field: [] for field, _, _ in STAT_FIELDS
            }

            for run_content in runs.values():
                if "hybrid_mse_avg" not in run_content:
                    continue
                testing = run_content["hybrid_mse_avg"]["testing"]
                if metric_key not in testing:
                    continue
                metric_map: Dict[str, float] = testing[metric_key]
                all_clients = _discover_clients(metric_map)
                if not all_clients:
                    continue

                for _ in range(n_samples):
                    r = _sample_metric_once(metric_map, all_clients, n_edges, master_rng)
                    if r is None:
                        continue
                    for field, mk, _ in STAT_FIELDS:
                        if mk in r:
                            accum[field].append(r[mk])

            for field, mk, sk in STAT_FIELDS:
                vals = accum[field]
                if vals:
                    entry[f"{metric_key}_{mk}"] = float(np.mean(vals))
                    entry[f"{metric_key}_{sk}"] = float(np.std(vals))

        # Include only if at least one metric has data
        has_data = any(
            f"{mk}_{mn}" in entry
            for mk, _ in METRICS
            for mn in ("gm_mean", "lm_mean")
        )
        if has_data:
            alpha_results.append(entry)

    alpha_results.sort(key=lambda x: x["alpha"])
    return alpha_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _eb(ax, results: List[dict], mean_key: str, std_key: str, **kw) -> None:
    xs = [r["alpha"] for r in results if mean_key in r]
    ys = [r[mean_key] for r in results if mean_key in r]
    es = [r.get(std_key, 0.0) for r in results if mean_key in r]
    if xs:
        ax.errorbar(xs, ys, yerr=es, capsize=3, **kw)


def _style_ax(ax, ylabel: str, title: str, xscale: str = "log") -> None:
    ax.set_xscale(xscale)
    ax.set_xlabel("Alpha (lower = more heterogeneous)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", ls="-", alpha=0.2)


def plot(alpha_results: List[dict], out_prefix: str) -> None:
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(n_metrics, 4, figsize=(20, 5 * n_metrics))
    plt.subplots_adjust(hspace=0.42, wspace=0.30)

    for row, (mk, mlabel) in enumerate(METRICS):
        # Col 0: Average metric vs alpha (GM vs LM-self)
        ax = axes[row, 0]
        _eb(ax, alpha_results, f"{mk}_gm_mean", f"{mk}_gm_std",
            label="Global Model", marker='o')
        _eb(ax, alpha_results, f"{mk}_lm_mean", f"{mk}_lm_std",
            label="Local Model (Self)", marker='s')
        _style_ax(ax, f"Average {mlabel}", f"[{mlabel}] Avg vs Alpha")

        # Col 1: Inter-model spread (LM-self, avg-all-LMs, avg-cross-LMs)
        ax = axes[row, 1]
        _eb(ax, alpha_results, f"{mk}_lm_mean", f"{mk}_lm_std",
            label="LM Self", marker='s', color='steelblue')
        _eb(ax, alpha_results, f"{mk}_inter_model_mean", f"{mk}_inter_model_std",
            label="Avg LMs (incl. self)", marker='^', color='green')
        _eb(ax, alpha_results, f"{mk}_lm_cross_mean", f"{mk}_lm_cross_std",
            label="Avg Cross LMs", marker='v', color='darkorange', linestyle='--')
        _style_ax(ax, f"Mean {mlabel}", f"[{mlabel}] Inter-model Spread")

        # Col 2: Best gain (LM-self − GM) and self-vs-cross
        ax = axes[row, 2]
        _eb(ax, alpha_results, f"{mk}_best_gain_mean", f"{mk}_best_gain_std",
            label="LMi − GM on Ei", marker='D', color='red')
        _eb(ax, alpha_results, f"{mk}_self_vs_cross_mean", f"{mk}_self_vs_cross_std",
            label="LMi − avg LMj≠i on Ei", marker='P', color='sienna', linestyle='--')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        _style_ax(ax, f"{mlabel} Gain", f"[{mlabel}] Best Gain: Self vs GM / Cross")

        # Col 3: Average gain (mean(LMs) − GM)
        ax = axes[row, 3]
        _eb(ax, alpha_results, f"{mk}_avg_gain_mean", f"{mk}_avg_gain_std",
            label="mean(LMj) − GM on Ei", marker='x', color='purple')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        _style_ax(ax, f"{mlabel} Gain", f"[{mlabel}] Avg Gain: Avg Local vs GM")

    plt.tight_layout()
    png_path = Path(f"{out_prefix}.png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot to {png_path}")
    plt.close(fig)


def save_csv(alpha_results: List[dict], out_prefix: str) -> None:
    if not alpha_results:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for r in alpha_results:
        for k in r:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    csv_path = Path(f"{out_prefix}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(alpha_results)
    print(f"Saved CSV to {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Heterogeneity gain analysis (AUC/TPR/TNR)")
    parser.add_argument("--json", default="configs/accuracy_matrix.json")
    parser.add_argument("--n_edges", type=int, default=4,
                        help="Edge areas to sample per trial (default 4)")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Random draws per (scenario, run) (default 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="heterogeneity_gain_analysis",
                        help="Output prefix for .png and .csv")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    print(f"Loading {json_path} ...")
    data = json.loads(json_path.read_text())
    print(f"Scenarios: {list(data.keys())}")

    alpha_results = analyze(data, args.n_edges, args.n_samples, args.seed)

    print("\nSummary (AUC | TPR | TNR):")
    for r in alpha_results:
        parts = [f"alpha={r['alpha']:8.2f}"]
        for mk, mlabel in METRICS:
            gm = f"{r[f'{mk}_gm_mean']:.4f}" if f"{mk}_gm_mean" in r else "n/a  "
            lm = f"{r[f'{mk}_lm_mean']:.4f}" if f"{mk}_lm_mean" in r else "n/a  "
            bg = (f"BG={r[f'{mk}_best_gain_mean']:+.4f}"
                  if f"{mk}_best_gain_mean" in r else "")
            parts.append(f"{mlabel}[GM={gm} LM={lm} {bg}]")
        print("  " + "  ".join(parts))

    plot(alpha_results, args.out)
    save_csv(alpha_results, args.out)


if __name__ == "__main__":
    main()
