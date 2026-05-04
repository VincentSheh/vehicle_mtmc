"""
Evaluate baseline and proposed methods across different FL heterogeneity levels 
(Dirichlet alpha values) and plot results.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import yaml
from tqdm import tqdm

from environment import build_env_base
from eval_common import (
    BaseEvaluator, 
    METRICS, 
    PROPOSED_CONFIGS,
    OFFLOAD_DISPLAY_NAMES,
    MODEL_DISPLAY_NAMES
)

class HeterogeneityEvaluator(BaseEvaluator):
    def mutate_cfg(self, cfg: dict, alpha: float, offload_mode: str, acc_model: str) -> dict:
        cfg_mut = copy.deepcopy(cfg)
        cfg_mut["globals"]["attack_sampler"]["dirichlet_alpha"] = alpha
        cfg_mut["globals"]["offload_mode"] = offload_mode
        if "accuracy_matrix" in cfg_mut.get("globals", {}):
            cfg_mut["globals"]["accuracy_matrix"]["model"] = acc_model
        return cfg_mut

    def run(self):
        # Resolve Alphas
        if self.args.alphas:
            alphas = sorted(self.args.alphas)
        else:
            matrix_path = self.cfg_original["globals"]["accuracy_matrix"]["path"]
            with open(matrix_path) as f:
                matrix_data = json.load(f)
            alpha_set = set()
            for key in matrix_data:
                m = re.search(r'alpha([\d.]+)', key)
                if m: alpha_set.add(float(m.group(1)))
            alphas = sorted(list(alpha_set)) or [0.2, 0.5, 1.0, 10.0, 100.0, 1000.0]

        methods = self.get_methods()
        policies = self.build_policies(methods)
        
        cfg_offload_mode = self.cfg_original["globals"].get("offload_mode", "balance")
        cfg_acc_model = self.cfg_original["globals"].get("accuracy_matrix", {}).get("model", "gm")
        proposed_method = self.args.proposed_method
        BASELINE_OFFLOAD = "delay_workload"

        # Build runs: (policy_key, label, offload_mode, acc_model)
        runs = []
        for m in methods:
            if m not in policies: continue
            if proposed_method and m == proposed_method:
                for amod, om in PROPOSED_CONFIGS:
                    runs.append((m, self.make_proposed_display_label(m, amod, om), om, amod))
            else:
                runs.append((m, self.make_display_label(m, BASELINE_OFFLOAD, False), BASELINE_OFFLOAD, cfg_acc_model))

        methods_display = [r[1] for r in runs]
        proposed_label = self.args.proposed_label or (methods_display[-1] if proposed_method else None)

        csv_path = self.outdir / "heterogeneity.csv"

        if self.args.replot:
            print(f"[replot] Loading data from {csv_path}...")
            accumulated_means, _ = self._load_existing_results(csv_path, methods_display, alphas)
            plot_heterogeneity_bar(accumulated_means, methods_display, alphas, self.outdir / "heterogeneity_bar.png", proposed_label)
            plot_heterogeneity_line(accumulated_means, methods_display, alphas, self.outdir / "heterogeneity_line.png")
            return

        accumulated_means, done_alphas = self._load_existing_results(csv_path, methods_display, alphas)

        if len(done_alphas) == len(alphas):
            print(f"[replot] All {len(alphas)} alphas complete.")
        else:
            # Group by (om, acc_model)
            groups = {}
            for pkey, label, om, amod in runs:
                groups.setdefault((om, amod), []).append((pkey, label))

            for alpha in alphas:
                if alpha in done_alphas: continue
                print(f"\n>>> Evaluating: alpha={alpha}")
                
                # Load what we have for this alpha
                cell_means = accumulated_means.get(alpha, {})

                for (om, amod), pkey_label_pairs in groups.items():
                    # Skip env build if all methods in group are done with enough episodes
                    if all(lbl in cell_means and cell_means[lbl].get("n_episodes", 0) >= self.args.episodes 
                           for _, lbl in pkey_label_pairs):
                        continue

                    cfg = self.mutate_cfg(self.cfg_original, alpha, om, amod)
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                        yaml.dump(cfg, tmp)
                        tmp_path = tmp.name
                    try:
                        env = build_env_base(tmp_path)
                        for pkey, label in pkey_label_pairs:
                            if label in cell_means and cell_means[label].get("n_episodes", 0) >= self.args.episodes:
                                print(f"  [skip] {label} (already enough episodes in CSV)")
                                continue
                            res, _ = self.run_simulation(env, cfg, policies[pkey], label, cache_key=f"{label}_a{alpha}")
                            cell_means[label] = self.extract_metrics(res)
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                
                accumulated_means[alpha] = cell_means
                self._save_to_csv(csv_path, accumulated_means, methods_display, alphas)

        plot_heterogeneity_bar(accumulated_means, methods_display, alphas, self.outdir / "heterogeneity_bar.png", proposed_label)
        plot_heterogeneity_line(accumulated_means, methods_display, alphas, self.outdir / "heterogeneity_line.png")

    def _load_existing_results(self, path, methods_display, alphas):
        means, done = {}, set()
        if not path.exists(): return means, done
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    alpha, method, metric, value = float(row["alpha"]), row["method"], row["metric"], float(row["value"])
                    means.setdefault(alpha, {}).setdefault(method, {})[metric] = value
            for a in alphas:
                if a in means:
                    all_methods_done = True
                    for m in methods_display:
                        if m not in means[a]:
                            all_methods_done = False
                            break
                        # Check if all metrics are present AND if n_episodes matches
                        metrics_present = all(mk in means[a][m] for mk, _ in METRICS)
                        eps_match = means[a][m].get("n_episodes", 0) >= self.args.episodes
                        if not (metrics_present and eps_match):
                            all_methods_done = False
                            break
                    if all_methods_done:
                        done.add(a)
        except Exception as e: print(f"[warn] Failed to load CSV: {e}")
        return means, done

    def _save_to_csv(self, path, means, methods_display, alphas):
        rows = []
        for alpha in alphas:
            if alpha not in means: continue
            for m_label in methods_display:
                m_data = means[alpha].get(m_label, {})
                for mk, mt in METRICS:
                    rows.append({"alpha": alpha, "method": m_label, "metric": mk, "metric_label": mt, "value": m_data.get(mk, np.nan)})
                # Save n_episodes explicitly
                rows.append({"alpha": alpha, "method": m_label, "metric": "n_episodes", "metric_label": "Num Episodes", "value": m_data.get("n_episodes", self.args.episodes)})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["alpha", "method", "metric", "metric_label", "value"])
            writer.writeheader()
            writer.writerows(rows)

def plot_heterogeneity_bar(means, methods_display, alphas, outpath, proposed_label=None):
    alphas_plot = sorted(alphas)
    n_rows = len(METRICS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True, squeeze=False)
    colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(methods_display)}
    x, width = np.arange(len(alphas_plot)), 0.8 / len(methods_display)

    for row_i, (mk, mt) in enumerate(METRICS):
        ax = axes[row_i][0]
        for i, m_label in enumerate(methods_display):
            vals = [means.get(a, {}).get(m_label, {}).get(mk, np.nan) for a in alphas_plot]
            pos = x + (i - len(methods_display)/2 + 0.5) * width
            ax.bar(pos, vals, width, color=colors[m_label], label=m_label, alpha=0.8)
            
            if proposed_label and m_label != proposed_label:
                for j, alpha in enumerate(alphas_plot):
                    ref = means.get(alpha, {}).get(proposed_label, {}).get(mk, np.nan)
                    if not np.isnan(ref) and not np.isnan(vals[j]):
                        gap = vals[j] - ref
                        text = f"{gap:+.1%}" if mk in ("slo_vio", "atk_leak", "bcd") else f"{gap:+.1f}"
                        ax.text(pos[j], vals[j] + 0.01 * ax.get_ylim()[1], text, ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in alphas_plot])
        if mk in ("slo_vio", "atk_leak", "bcd"): ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(mt, fontsize=12, fontweight="bold")
        ax.grid(axis='y', linestyle="--", alpha=0.4)

    axes[-1][0].set_xlabel("Heterogeneity (Dirichlet α)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.02, 0.5), title="Method")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def plot_heterogeneity_line(means, methods_display, alphas, outpath):
    alphas_plot = sorted(alphas)
    n_rows = len(METRICS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True, squeeze=False)
    colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(methods_display)}
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "H"]

    for row_i, (mk, mt) in enumerate(METRICS):
        ax = axes[row_i][0]
        for i, m_label in enumerate(methods_display):
            vals = [means.get(a, {}).get(m_label, {}).get(mk, np.nan) for a in alphas_plot]
            ax.plot(alphas_plot, vals, label=m_label, color=colors[m_label], marker=markers[i % len(markers)], markersize=6, linewidth=2, alpha=0.8)
        
        ax.set_xscale("log")
        if mk in ("slo_vio", "atk_leak", "bcd"): ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(mt, fontsize=12, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[-1][0].set_xlabel("Heterogeneity (Dirichlet α) - Log Scale")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.02, 0.5), title="Method")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/simulation_ma_0.yaml")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--outdir", default="eval_out_hetero/")
    ap.add_argument("--ckpt", default="checkpoints/_singleedge/a4_sf20_atk3_a18_default_default/ckpt_best.pt")
    ap.add_argument("--tbsa_table", default="tbsa_table.npz")
    ap.add_argument("--ids_cpu_min", type=float, default=0.5)
    ap.add_argument("--scale_step", type=float, default=0.5)
    ap.add_argument("--decision_interval", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--offload_modes", nargs="+", default=None)
    ap.add_argument("--proposed_method", default="lstm_rl")
    ap.add_argument("--alphas", nargs="+", type=float, default=None)
    ap.add_argument("--proposed_label", default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    evaluator = HeterogeneityEvaluator(args)
    evaluator.run()

if __name__ == "__main__":
    main()
