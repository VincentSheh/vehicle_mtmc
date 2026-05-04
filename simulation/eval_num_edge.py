"""
Evaluate baseline and proposed methods across different edge area scales (1 to 5)
and plot results as a grouped bar chart.
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

class NumEdgeEvaluator(BaseEvaluator):
    def mutate_cfg(self, cfg: dict, n_edge: int, offload_mode: str, acc_model: str) -> dict:
        cfg_mut = copy.deepcopy(cfg)
        
        # Scale Edge Areas
        base_area = cfg_mut["edge_areas"][0]
        cfg_mut["edge_areas"] = []
        for i in range(n_edge):
            area = copy.deepcopy(base_area)
            area["area_id"] = f"E{i+1}"
            cfg_mut["edge_areas"].append(area)
        
        # Update Delay Matrix (20ms default cross-edge delay)
        cfg_mut["globals"]["delay_ms"] = [[0.0 if i == j else 20.0 for j in range(n_edge)] for i in range(n_edge)]
        
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
        proposed_label = self.args.proposed_label or (methods_display[-1] if proposed_method else None)
        n_edges_list = sorted(self.args.n_edges)
        csv_path = self.outdir / "num_edge.csv"

        if self.args.replot:
            print(f"[replot] Loading data from {csv_path}...")
            accumulated_means, _ = self._load_existing_results(csv_path, methods_display, n_edges_list)
            plot_num_edge(accumulated_means, methods_display, n_edges_list, self.outdir / "num_edge.png", proposed_label)
            return

        accumulated_means, done_n_edges = self._load_existing_results(csv_path, methods_display, n_edges_list)

        if len(done_n_edges) == len(n_edges_list):
            print(f"[replot] All {len(n_edges_list)} scaling steps complete.")
        else:
            # Group by (om, acc_model)
            groups = {}
            for pkey, label, om, amod in runs:
                groups.setdefault((om, amod), []).append((pkey, label))

            for n_edge in n_edges_list:
                if n_edge in done_n_edges: continue
                print(f"\n>>> Evaluating: n_edge={n_edge}")
                
                # Load what we have for this n_edge
                cell_means = accumulated_means.get(n_edge, {})
                
                for (om, amod), pkey_label_pairs in groups.items():
                    # If all methods in this group (om, amod) are done with enough episodes, skip
                    if all(lbl in cell_means and cell_means[lbl].get("n_episodes", 0) >= self.args.episodes 
                           for _, lbl in pkey_label_pairs):
                        continue

                    cfg = self.mutate_cfg(self.cfg_original, n_edge, om, amod)
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                        yaml.dump(cfg, tmp)
                        tmp_path = tmp.name
                    try:
                        env = build_env_base(tmp_path)
                        for pkey, label in pkey_label_pairs:
                            if label in cell_means and cell_means[label].get("n_episodes", 0) >= self.args.episodes:
                                print(f"  [skip] {label} (already enough episodes in CSV)")
                                continue
                            res, _ = self.run_simulation(env, cfg, policies[pkey], label, cache_key=f"{label}_n{n_edge}")
                            cell_means[label] = self.extract_metrics(res)
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                
                accumulated_means[n_edge] = cell_means
                self._save_to_csv(csv_path, accumulated_means, methods_display, n_edges_list)

        plot_num_edge(accumulated_means, methods_display, n_edges_list, self.outdir / "num_edge.png", proposed_label)

    def _load_existing_results(self, path, methods_display, n_edges_list):
        means, done = {}, set()
        if not path.exists(): return means, done
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    n_edge, method, metric, value = int(row["n_edge"]), row["method"], row["metric"], float(row["value"])
                    means.setdefault(n_edge, {}).setdefault(method, {})[metric] = value
            for n in n_edges_list:
                if n in means:
                    all_methods_done = True
                    for m in methods_display:
                        if m not in means[n]:
                            all_methods_done = False
                            break
                        metrics_present = all(mk in means[n][m] for mk, _ in METRICS)
                        eps_match = means[n][m].get("n_episodes", 0) >= self.args.episodes
                        if not (metrics_present and eps_match):
                            all_methods_done = False
                            break
                    if all_methods_done:
                        done.add(n)
        except Exception as e: print(f"[warn] Failed to load CSV: {e}")
        return means, done

    def _save_to_csv(self, path, means, methods_display, n_edges_list):
        rows = []
        for n in n_edges_list:
            if n not in means: continue
            for m_label in methods_display:
                m_data = means[n].get(m_label, {})
                for mk, mt in METRICS:
                    rows.append({"n_edge": n, "method": m_label, "metric": mk, "metric_label": mt, "value": m_data.get(mk, np.nan)})
                rows.append({"n_edge": n, "method": m_label, "metric": "n_episodes", "metric_label": "Num Episodes", "value": m_data.get("n_episodes", self.args.episodes)})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["n_edge", "method", "metric", "metric_label", "value"])
            writer.writeheader()
            writer.writerows(rows)

def plot_num_edge(means, methods_display, n_edges_list, outpath, proposed_label=None):
    n_rows = len(METRICS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True, squeeze=False)
    colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(methods_display)}
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]

    for row_i, (mk, mt) in enumerate(METRICS):
        ax = axes[row_i][0]
        for i, m_label in enumerate(methods_display):
            vals = [means.get(n, {}).get(m_label, {}).get(mk, np.nan) for n in n_edges_list]
            
            lw = 3.0 if m_label == proposed_label else 1.5
            alpha = 1.0 if m_label == proposed_label else 0.7
            zorder = 5 if m_label == proposed_label else 3
            
            ax.plot(
                n_edges_list, 
                vals, 
                label=m_label, 
                color=colors[m_label], 
                marker=markers[i % len(markers)],
                markersize=8,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder
            )
        
        ax.set_xticks(n_edges_list)
        if mk in ("slo_vio", "atk_leak", "bcd"): 
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(mt, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1][0].set_xlabel("Number of Edge Areas")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.02, 0.5), title="Method")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/simulation_ma_0.yaml")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--outdir", default="eval_out_num_edge/")
    ap.add_argument("--ckpt", default="checkpoints/_singleedge/a4_sf20_atk3_a18_default_default/ckpt_best.pt")
    ap.add_argument("--tbsa_table", default="tbsa_table.npz")
    ap.add_argument("--ids_cpu_min", type=float, default=0.5)
    ap.add_argument("--scale_step", type=float, default=0.5)
    ap.add_argument("--decision_interval", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--offload_modes", nargs="+", default=None)
    ap.add_argument("--proposed_method", default="lstm_rl")
    ap.add_argument("--n_edges", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--proposed_label", default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    evaluator = NumEdgeEvaluator(args)
    evaluator.run()

if __name__ == "__main__":
    main()
