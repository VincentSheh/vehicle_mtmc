"""
Evaluate baseline policies across 3 attack × 3 user levels and plot a 3×3 grid.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
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

LEVELS = ["low", "mid", "high"]
LEVEL_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}

class ScenarioGridEvaluator(BaseEvaluator):
    def mutate_cfg(self, cfg: dict, atk_lvl: str, user_lvl: str, offload_mode: str, acc_model: str) -> dict:
        cfg_mut = copy.deepcopy(cfg)
        cfg_mut["globals"]["attack_sampler"]["level"] = atk_lvl
        cfg_mut["globals"]["user_sampler"]["synthetic"]["level"] = user_lvl
        cfg_mut["globals"]["offload_mode"] = offload_mode
        if "accuracy_matrix" in cfg_mut.get("globals", {}):
            cfg_mut["globals"]["accuracy_matrix"]["model"] = acc_model
        return cfg_mut

    def run(self):
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
        csv_path = self.outdir / "scenario_grid.csv"

        if self.args.replot:
            print(f"[replot] Loading data from {csv_path}...")
            accumulated_means, _ = self._load_existing_results(csv_path, methods_display)
            
            # If a custom threshold is provided during replot, we must re-extract metrics from cache
            if self.args.slo_threshold is not None:
                print(f"[replot] Recalculating SLO violations with threshold={self.args.slo_threshold}...")
                for atk in LEVELS:
                    for user in LEVELS:
                        for pkey, label, om, amod in runs:
                            cache_key = f"{label}_{atk}_{user}"
                            # We need to reach into the cache to get the raw traces
                            safe_label = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in cache_key])
                            cache_path = self.cachedir / f"{safe_label}.npz"
                            if cache_path.exists():
                                with np.load(cache_path, allow_pickle=True) as data:
                                    arrays = {k: data[k] for k in data.files}
                                    accumulated_means[atk][user][label] = self.extract_metrics(arrays, slo_threshold=self.args.slo_threshold)

            plot_grid(accumulated_means, methods_display, self.outdir / "scenario_grid_by_user.png", x_dim="user")
            plot_grid(accumulated_means, methods_display, self.outdir / "scenario_grid_by_attack.png", x_dim="attack")
            return

        accumulated_means, done_cells = self._load_existing_results(csv_path, methods_display)

        if len(done_cells) == 9:
            print(f"[replot] All 9 cells complete.")
        else:
            # Group by (om, acc_model)
            groups = {}
            for pkey, label, om, amod in runs:
                groups.setdefault((om, amod), []).append((pkey, label))

            for atk in LEVELS:
                for user in LEVELS:
                    if (atk, user) in done_cells: continue
                    print(f"\n>>> Evaluating: atk={atk} user={user}")
                    
                    # Load what we have for this cell
                    cell_means = accumulated_means[atk][user]

                    for (om, amod), pkey_label_pairs in groups.items():
                        # Skip env build if all methods in group are done with enough episodes
                        if all(lbl in cell_means and cell_means[lbl].get("n_episodes", 0) >= self.args.episodes 
                               for _, lbl in pkey_label_pairs):
                            continue

                        cfg = self.mutate_cfg(self.cfg_original, atk, user, om, amod)
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                            yaml.dump(cfg, tmp)
                            tmp_path = tmp.name
                        try:
                            env = build_env_base(tmp_path)
                            for pkey, label in pkey_label_pairs:
                                if label in cell_means and cell_means[label].get("n_episodes", 0) >= self.args.episodes:
                                    print(f"  [skip] {label} (already enough episodes in CSV)")
                                    continue
                                res, _ = self.run_simulation(env, cfg, policies[pkey], label, cache_key=f"{label}_{atk}_{user}")
                                cell_means[label] = self.extract_metrics(res, slo_threshold=self.args.slo_threshold)
                        finally:
                            if os.path.exists(tmp_path): os.remove(tmp_path)
                    
                    accumulated_means[atk][user] = cell_means
                    self._save_to_csv(csv_path, accumulated_means, methods_display)

        plot_grid(accumulated_means, methods_display, self.outdir / "scenario_grid_by_user.png", x_dim="user")
        plot_grid(accumulated_means, methods_display, self.outdir / "scenario_grid_by_attack.png", x_dim="attack")

    def _load_existing_results(self, path, methods_display):
        means = {a: {u: {} for u in LEVELS} for a in LEVELS}
        done = set()
        if not path.exists(): return means, done
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    atk, user, method, metric, value = row["atk_level"], row["user_level"], row["method"], row["metric"], float(row["value"])
                    means[atk][user].setdefault(method, {})[metric] = value
            for a in LEVELS:
                for u in LEVELS:
                    all_methods_done = True
                    for m in methods_display:
                        if m not in means[a][u]:
                            all_methods_done = False
                            break
                        metrics_present = all(mk in means[a][u][m] for mk, _ in METRICS)
                        eps_match = means[a][u][m].get("n_episodes", 0) >= self.args.episodes
                        if not (metrics_present and eps_match):
                            all_methods_done = False
                            break
                    if all_methods_done:
                        done.add((a, u))
        except Exception as e: print(f"[warn] Failed to load CSV: {e}")
        return means, done

    def _save_to_csv(self, path, means, methods_display):
        rows = []
        for a in LEVELS:
            for u in LEVELS:
                if not means[a][u]: continue
                for m_label in methods_display:
                    m_data = means[a][u].get(m_label, {})
                    for mk, mt in METRICS:
                        rows.append({"atk_level": a, "user_level": u, "method": m_label, "metric": mk, "metric_label": mt, "value": m_data.get(mk, np.nan)})
                    rows.append({"atk_level": a, "user_level": u, "method": m_label, "metric": "n_episodes", "metric_label": "Num Episodes", "value": m_data.get("n_episodes", self.args.episodes)})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["atk_level", "user_level", "method", "metric", "metric_label", "value"])
            writer.writeheader()
            writer.writerows(rows)

def plot_grid(means, methods_display, outpath, x_dim):
    if x_dim == "user":
        row_levels, x_levels = LEVELS, LEVELS
        row_prefix, x_label  = "Atk:", "User Level"
        get_val = lambda m, r, x, mk: m.get(r, {}).get(x, {}).get(method_label, {}).get(mk, np.nan)
    else:
        row_levels, x_levels = LEVELS, LEVELS
        row_prefix, x_label  = "User:", "Attack Level"
        get_val = lambda m, r, x, mk: m.get(x, {}).get(r, {}).get(method_label, {}).get(mk, np.nan)

    n_rows, n_cols = len(row_levels), len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(methods_display)}
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(methods_display)}
    x_pos = np.arange(len(x_levels))

    for row_i, row_lvl in enumerate(row_levels):
        for col_j, (mk, mt) in enumerate(METRICS):
            ax = axes[row_i][col_j]
            for method_label in methods_display:
                vals = [get_val(means, row_lvl, xl, mk) for xl in x_levels]
                ax.plot(x_pos, vals, f"-{marker_map[method_label]}", color=colors[method_label], label=method_label, linewidth=1.5, markersize=6)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([LEVEL_LABELS[l] for l in x_levels])
            if mk in ("slo_vio", "atk_drop"): ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
            if row_i == 0: ax.set_title(mt, fontsize=12, fontweight="bold")
            if col_j == 0: ax.set_ylabel(f"{row_prefix} {LEVEL_LABELS[row_lvl]}", fontsize=11)
            if row_i == n_rows - 1: ax.set_xlabel(x_label, fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="center right", bbox_to_anchor=(1.0, 0.5), title="Method")
    plt.tight_layout(rect=[0, 0, 0.84, 1.0])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/simulation_ma_0.yaml")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--outdir", default="eval_out/scenario_grid")
    ap.add_argument("--ckpt", default="checkpoints/_singleedge/a4_sf20_atk3_a18_default_default/ckpt_best.pt")
    ap.add_argument("--tbsa_table", default="tbsa_table_15.npz")
    ap.add_argument("--ids_cpu_min", type=float, default=0.5)
    ap.add_argument("--scale_step", type=float, default=0.5)
    ap.add_argument("--decision_interval", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--offload_modes", nargs="+", default=None)
    ap.add_argument("--proposed_method", default=None)
    ap.add_argument("--slo_threshold", type=float, default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    evaluator = ScenarioGridEvaluator(args)
    evaluator.run()

if __name__ == "__main__":
    main()
