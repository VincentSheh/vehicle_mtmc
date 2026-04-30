"""
Evaluate baseline and proposed methods across different edge area scales (1 to 5)
and plot results as a grouped bar chart.

Usage:
    # Dummy mode
    python eval_num_edge.py --dummy

    # Real evaluation
    python eval_num_edge.py --cfg configs/simulation_ma_0.yaml --episodes 10
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
from typing import Dict, Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import yaml
from tqdm import tqdm

from method_policy import OFFLOAD_DISPLAY_NAMES, MODEL_DISPLAY_NAMES

METRICS = [
    ("slo_vio",  "SLO Violation Rate"),
    ("bcd",      "Benign Collateral Damage"),
    ("atk_leak", "Attack Leakage %"),
    ("realloc",  "Avg. Reallocations"),
]

DISPLAY_NAMES: Dict[str, str] = {
    "no_ids":          "No IDS",
    "static_low":      "Static Low",
    "static_balanced": "Static Balanced",
    "static_high":     "Static High",
    "autoscale_app":   "Autoscale (App)",
    "autoscale_def":   "Autoscale (Def)",
    "offline_optimal": "TBSA Optimal",
    "lstm_rl":         "LSTM RL",
    "ma_lstm_rl":      "MA LSTM RL",
    "gm":              "Global Model",
    "lm":              "Local Model",
}

DEFAULT_METHODS = [
    # "no_ids",
    # "static_low",
    # "static_high",
    # "autoscale_def",
    # "offline_optimal",
]

# (policy_key, offload_mode) pairs run for the proposed method
PROPOSED_CONFIGS = [
    ("gm", "delay_workload"),
    ("lm", "delay_workload"),
    ("lm", "cto"),
    ("lm", "cto_acc"),
]

# ---------------------------------------------------------------------------
# Metric extraction from raw arrays
# ---------------------------------------------------------------------------

def _extract_mean(arrays: Dict[str, np.ndarray], key: str, transform=None) -> float:
    vals = arrays.get(key, np.array([], dtype=np.float32))
    if transform is not None:
        vals = transform(vals)
    return float(np.mean(vals)) if vals.size > 0 else np.nan


def _arrays_to_means(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    atk_in  = arrays.get("attack_in_rate",   np.array([], dtype=np.float32))
    atk_drp = arrays.get("attack_drop_rate", np.array([], dtype=np.float32))
    if atk_in.size > 0 and atk_drp.size == atk_in.size:
        atk_pass = np.maximum(0.0, atk_in - atk_drp)
        lres = np.divide(atk_pass, atk_in, out=np.zeros_like(atk_pass), where=atk_in > 1e-6)
        attack_mask = atk_in > 1e-6
        # Leakage: fraction of attack traffic that passed through
        atk_leak_val = float(np.mean(lres[attack_mask])) if attack_mask.any() else 0.0
    else:
        atk_leak_val = np.nan

    return {
        "slo_vio":  _extract_mean(arrays, "qoe_vio_rate"),
        "bcd":      _extract_mean(arrays, "reward_benign_col_dmg"),
        "atk_leak": atk_leak_val,
        "realloc":  _extract_mean(arrays, "reallocations"),
    }

# ---------------------------------------------------------------------------
# Label Helpers
# ---------------------------------------------------------------------------

def _make_display_label(mname: str, offload_mode: str, show_offload: bool) -> str:
    base = DISPLAY_NAMES.get(mname, mname)
    if show_offload:
        ol = OFFLOAD_DISPLAY_NAMES.get(offload_mode, offload_mode)
        return f"{base} ({ol})"
    return base


def _make_proposed_display_label(model_key: str, offload_mode: str) -> str:
    base = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    ol   = OFFLOAD_DISPLAY_NAMES.get(offload_mode, offload_mode)
    return f"{base} ({ol})"

# ---------------------------------------------------------------------------
# CSV and State Helpers
# ---------------------------------------------------------------------------

def save_means_to_csv(means: dict, methods_display: list[str], n_edges_list: list[int], outpath: Path):
    rows = []
    for n_edge in n_edges_list:
        for method_label in methods_display:
            m = means.get(n_edge, {}).get(method_label, {})
            for metric_key, metric_title in METRICS:
                rows.append({
                    "n_edge":       n_edge,
                    "method":       method_label,
                    "metric":       metric_key,
                    "metric_label": metric_title,
                    "value":        m.get(metric_key, float("nan")),
                })
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n_edge", "method", "metric", "metric_label", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {outpath}")


def load_means_from_csv(path: Path) -> tuple[dict, list[str], list[int]]:
    means: dict = {}
    method_order: list[str] = []
    n_edges_found: set[int] = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n_edge, method, metric, value = (
                int(row["n_edge"]), row["method"],
                row["metric"], float(row["value"]),
            )
            if method not in method_order:
                method_order.append(method)
            n_edges_found.add(n_edge)
            means.setdefault(n_edge, {}).setdefault(method, {})[metric] = value
    return means, method_order, sorted(list(n_edges_found))


def _cell_is_complete(means: dict, n_edge: int, methods_display: list[str]) -> bool:
    cell = means.get(n_edge, {})
    return all(
        all(
            not np.isnan(cell.get(m, {}).get(mk, np.nan))
            for mk, _ in METRICS
        )
        for m in methods_display
    )

# ---------------------------------------------------------------------------
# Dummy Data
# ---------------------------------------------------------------------------

def _make_dummy_data(method_labels: list[str], n_edges_list: list[int]) -> dict:
    rng = np.random.default_rng(0)
    data: dict = {}
    for i, n_edge in enumerate(n_edges_list):
        data[n_edge] = {}
        for k, label in enumerate(method_labels):
            n = 50
            # Higher n_edge -> slightly better offloading potential -> lower metrics
            base_slo  = np.clip(0.15 - 0.01 * n_edge + 0.02 * k, 0.01, 1)
            base_bcd  = np.clip(0.08 - 0.005 * n_edge + 0.01 * k, 0.005, 1)
            base_leak = np.clip(0.30 - 0.02 * n_edge + 0.03 * k, 0.02, 1)
            
            data[n_edge][label] = {
                "qoe_vio_rate":          np.clip(rng.normal(base_slo,        0.02, n), 0, 1   ).astype(np.float32),
                "reward_benign_col_dmg": np.clip(rng.normal(base_bcd,        0.01, n), 0, None).astype(np.float32),
                "reward_lambda_res":     np.clip(rng.normal(base_leak,       0.03, n), 0, 1   ).astype(np.float32),
                "attack_in_rate":        np.ones(n, dtype=np.float32),
                "attack_drop_rate":      np.clip(rng.normal(1.0 - base_leak, 0.03, n), 0, 1   ).astype(np.float32),
                "reallocations":         np.array([rng.uniform(2, 8) for _ in range(5)], dtype=np.float32),
            }
    return data

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_data(args, n_edges_list: list[int], done_n_edges: set[int] | None = None, cell_callback=None) -> dict:
    from eval_baselines import run_episode
    from environment import build_env_base
    from train_sa_lstm import TorchRLEnvWrapper
    from method_policy import make_baseline_policy

    with open(args.cfg) as f:
        cfg_original = yaml.safe_load(f)

    base_seed = int(cfg_original["run"]["seed"])
    np.random.seed(base_seed)

    methods: list[str] = list(args.methods) if args.methods else list(DEFAULT_METHODS)
    proposed_method: Optional[str] = getattr(args, "proposed_method", None)
    if proposed_method is not None and proposed_method not in methods:
        methods.append(proposed_method)

    cfg_offload_mode = cfg_original["globals"].get("offload_mode", "balance")
    cfg_acc_model    = cfg_original["globals"].get("accuracy_matrix", {}).get("model", "gm")
    BASELINE_OFFLOAD = "delay_workload"

    if proposed_method is not None:
        runs = []
        for m in methods:
            if m == proposed_method:
                for acc_model, om in PROPOSED_CONFIGS:
                    label = _make_proposed_display_label(acc_model, om)
                    runs.append((om, acc_model, m, label))
            else:
                label = _make_display_label(m, BASELINE_OFFLOAD, False)
                runs.append((BASELINE_OFFLOAD, cfg_acc_model, m, label))
    else:
        offload_modes: list[str] = args.offload_modes if args.offload_modes else [cfg_offload_mode]
        multi_offload = len(offload_modes) > 1
        runs = [
            (om, cfg_acc_model, m, _make_display_label(m, om, multi_offload))
            for om in offload_modes
            for m in methods
        ]

    env_groups: dict = {}
    for om, acc_model, mname, label in runs:
        env_groups.setdefault((om, acc_model), []).append((mname, label))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_orig:
        yaml.dump(cfg_original, tmp_orig)
        tmp_orig_path = tmp_orig.name
    try:
        decision_interval = args.decision_interval or int(cfg_original["globals"]["decision_interval"])
        _wrapper     = TorchRLEnvWrapper(cfg_path=tmp_orig_path, decision_interval=decision_interval, device="cpu")
        obs_keys     = _wrapper.obs_keys
        reward_alpha = _wrapper.reward_alpha
        reward_beta  = _wrapper.reward_beta
        reward_gamma = _wrapper.reward_gamma
        reward_q_th  = _wrapper.reward_q_th
        n_actions    = _wrapper.n_actions
        del _wrapper
    finally:
        if os.path.exists(tmp_orig_path):
            os.remove(tmp_orig_path)

    all_policy_keys = list(dict.fromkeys(mname for _, _, mname, _ in runs))
    policies: dict = {}
    for mname in all_policy_keys:
        tbsa_path = str(args.tbsa_table) if mname in ("tbsa", "offline_optimal") else None
        ckpt      = args.ckpt            if mname in ("lstm_rl", "ma_lstm_rl") else None
        ok_keys   = obs_keys             if mname == "lstm_rl" else None
        try:
            policies[mname] = make_baseline_policy(
                mname,
                tbsa_table_path=tbsa_path,
                ckpt_path=ckpt,
                obs_keys=ok_keys,
                device=args.device,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"[warn] Skipping '{mname}': {exc}")

    runs = [(om, acc_model, mname, label) for om, acc_model, mname, label in runs if mname in policies]
    env_groups = {}
    for om, acc_model, mname, label in runs:
        env_groups.setdefault((om, acc_model), []).append((mname, label))

    data: dict = {}
    for n_edge in n_edges_list:
        if done_n_edges and n_edge in done_n_edges:
            print(f"\n[resume] Skipping n_edge={n_edge} (already complete)")
            data[n_edge] = {}
            continue

        print(f"\n{'='*70}")
        print(f" Evaluating: n_edge={n_edge}")
        print("="*70)

        if not runs:
            print("[warn] No valid methods, skipping.")
            data[n_edge] = {}
            continue

        cell: dict = {}
        last_ids_cpu: Dict[str, Optional[np.ndarray]] = {label: None for _, _, _, label in runs}

        for (offload_mode, acc_model), method_label_pairs in env_groups.items():
            cfg = copy.deepcopy(cfg_original)
            
            # Scale Edge Areas
            base_area = cfg["edge_areas"][0]
            cfg["edge_areas"] = []
            for i in range(n_edge):
                area = copy.deepcopy(base_area)
                area["area_id"] = f"E{i+1}"
                cfg["edge_areas"].append(area)
            
            # Update Delay Matrix (20ms default cross-edge delay)
            cfg["globals"]["delay_ms"] = [[0.0 if i == j else 20.0 for j in range(n_edge)] for i in range(n_edge)]
            
            cfg["globals"]["offload_mode"] = offload_mode
            if "accuracy_matrix" in cfg.get("globals", {}):
                cfg["globals"]["accuracy_matrix"]["model"] = acc_model

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                yaml.dump(cfg, tmp)
                tmp_path = tmp.name

            try:
                env = build_env_base(tmp_path)
                accumulated: Dict[str, Dict[str, np.ndarray]] = {label: {} for _, label in method_label_pairs}

                for ep in tqdm(range(args.episodes),
                               desc=f"episodes ({acc_model}/{offload_mode})"):
                    ep_seed = base_seed + (ep + 1) * 1000
                    for mname, label in method_label_pairs:
                        ep_result, final_ids = run_episode(
                            env=env,
                            cfg=cfg,
                            policy=policies[mname],
                            obs_keys=obs_keys,
                            decision_interval=decision_interval,
                            scale_step=args.scale_step,
                            n_actions=n_actions,
                            ids_cpu_min=args.ids_cpu_min,
                            seed=ep_seed,
                            reward_alpha=reward_alpha,
                            reward_beta=reward_beta,
                            reward_gamma=reward_gamma,
                            reward_q_th=reward_q_th,
                            initial_ids_cpu=last_ids_cpu[label],
                        )
                        last_ids_cpu[label] = final_ids

                        ep_ratios = ep_result["cpu_to_ids_ratio"]
                        realloc_count = 0
                        if ep_ratios.size > 1:
                            realloc_count = np.sum(np.abs(np.diff(ep_ratios)) > 1e-6)
                        ep_result["reallocations"] = np.array([realloc_count], dtype=np.float32)

                        for k, v in ep_result.items():
                            if k not in accumulated[label]:
                                accumulated[label][k] = v
                            else:
                                accumulated[label][k] = np.concatenate([accumulated[label][k], v], axis=0)

                for _, label in method_label_pairs:
                    cell[label] = accumulated[label]

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        data[n_edge] = cell
        if cell_callback is not None:
            cell_callback(n_edge, cell)

    return data

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_num_edge(
    means: dict,
    methods_display: list[str],
    n_edges_list: list[int],
    outpath: Path,
    proposed_label: str | None = None,
):
    n_rows = len(METRICS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True, squeeze=False)

    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods_display)}

    if not methods_display:
        print("[warn] plot_num_edge: no methods to plot, skipping.")
        return

    x = np.arange(len(n_edges_list))
    width = 0.8 / len(methods_display)

    for row_i, (metric_key, metric_title) in enumerate(METRICS):
        ax = axes[row_i][0]

        for i, method_label in enumerate(methods_display):
            vals = [means.get(n_edge, {}).get(method_label, {}).get(metric_key, np.nan) for n_edge in n_edges_list]
            pos = x + (i - len(methods_display)/2 + 0.5) * width
            rects = ax.bar(pos, vals, width, color=colors[method_label], label=method_label, alpha=0.8)

            # Annotate gap relative to proposed_label
            if proposed_label and method_label != proposed_label:
                for j, n_edge in enumerate(n_edges_list):
                    ref_val = means.get(n_edge, {}).get(proposed_label, {}).get(metric_key, np.nan)
                    val = vals[j]
                    if not np.isnan(ref_val) and not np.isnan(val):
                        gap = val - ref_val
                        if metric_key in ("slo_vio", "atk_leak", "bcd"):
                            text = f"{gap:+.1%}"
                        else:
                            text = f"{gap:+.1f}"
                        
                        ax.text(pos[j], val + 0.01 * ax.get_ylim()[1], text, 
                                ha='center', va='bottom', fontsize=8, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in n_edges_list])

        if metric_key in ("slo_vio", "atk_leak", "bcd"):
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
            )

        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.grid(axis='y', linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1][0].set_xlabel("Number of Edge Areas", fontsize=11)
    
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=10,
        title="Method",
        title_fontsize=10,
    )

    fig.suptitle("Performance vs Number of Edge Areas (Grouped Bar Chart)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg",               default="configs/simulation_ma_0.yaml")
    ap.add_argument("--methods",           nargs="+", default=None)
    ap.add_argument("--episodes",          type=int,   default=10)
    ap.add_argument("--outdir",            default="eval_out_num_edge/")
    ap.add_argument("--dummy",             action="store_true")
    ap.add_argument("--ckpt",              default="checkpoints/_singleedge/a4_sf20_atk3_a18_default_default/ckpt_best.pt")
    ap.add_argument("--tbsa_table",        default="tbsa_table.npz")
    ap.add_argument("--ids_cpu_min",       type=float, default=0.5)
    ap.add_argument("--scale_step",        type=float, default=0.5)
    ap.add_argument("--decision_interval", type=int,   default=None)
    ap.add_argument("--device",            default="cpu")
    ap.add_argument("--offload_modes",     nargs="+", default=None)
    ap.add_argument("--proposed_method",   default="lstm_rl")
    ap.add_argument("--n_edges",           nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--proposed_label",    default=None, help="The label of the proposed method for gap annotations")
    args = ap.parse_args()

    outdir  = Path(args.outdir)
    csv_path = outdir / "num_edge.csv"
    BASELINE_OFFLOAD = "delay_workload"

    with open(args.cfg) as f:
        cfg_orig = yaml.safe_load(f)

    n_edges_list = sorted(args.n_edges)

    def _build_methods_display(raw_methods: list[str], offload_modes: list[str]) -> list[str]:
        proposed = args.proposed_method
        if proposed is not None:
            labels = []
            for m in raw_methods:
                if m == proposed:
                    for model_key, om in PROPOSED_CONFIGS:
                        labels.append(_make_proposed_display_label(model_key, om))
                else:
                    labels.append(_make_display_label(m, BASELINE_OFFLOAD, False))
            return labels
        multi_offload = len(offload_modes) > 1
        return [
            _make_display_label(m, om, multi_offload)
            for om in offload_modes
            for m in raw_methods
        ]

    methods = list(args.methods) if args.methods else list(DEFAULT_METHODS)
    if args.proposed_method and args.proposed_method not in methods:
        methods.append(args.proposed_method)
    offload_modes = args.offload_modes or [cfg_orig["globals"].get("offload_mode", "balance")]
    methods_display = _build_methods_display(methods, offload_modes)

    if not methods_display:
        ap.error(
            "No methods to evaluate. Pass --methods and/or --proposed_method, "
            "or populate DEFAULT_METHODS in the script."
        )

    proposed_label = args.proposed_label
    if proposed_label is None and args.proposed_method is not None:
        proposed_label = methods_display[-1]

    if args.dummy:
        print("[dummy mode] Generating synthetic data...")
        data  = _make_dummy_data(methods_display, n_edges_list)
        means = {n: {label: _arrays_to_means(arrs) for label, arrs in data[n].items()} for n in n_edges_list}
        save_means_to_csv(means, methods_display, n_edges_list, csv_path)
    else:
        accumulated_means: dict = {}
        done_n_edges: set[int] = set()
        if csv_path.exists():
            try:
                accumulated_means, _, _ = load_means_from_csv(csv_path)
                for n in n_edges_list:
                    if _cell_is_complete(accumulated_means, n, methods_display):
                        done_n_edges.add(n)
                if len(done_n_edges) == len(n_edges_list):
                    print(f"[replot] All {len(n_edges_list)} scaling steps complete in {csv_path} — skipping simulation.")
                    plot_num_edge(accumulated_means, methods_display, n_edges_list, outdir / "num_edge.png", proposed_label=proposed_label)
                    return
                elif done_n_edges:
                    print(f"[resume] {len(done_n_edges)}/{len(n_edges_list)} scaling steps already complete, resuming...")
            except Exception as exc:
                print(f"[warn] Could not load {csv_path}: {exc} — starting fresh")
                accumulated_means = {}
                done_n_edges = set()

        def _on_cell_done(n_edge: int, cell_data: dict):
            accumulated_means[n_edge] = {
                label: _arrays_to_means(arrays)
                for label, arrays in cell_data.items()
            }
            save_means_to_csv(accumulated_means, methods_display, n_edges_list, csv_path)
            n_done = sum(_cell_is_complete(accumulated_means, n, methods_display) for n in n_edges_list)
            print(f"[checkpoint] {n_done}/{len(n_edges_list)} steps saved (n_edge={n_edge} done)")

        collect_data(args, n_edges_list, done_n_edges=done_n_edges, cell_callback=_on_cell_done)
        means = accumulated_means

    plot_num_edge(means, methods_display, n_edges_list, outdir / "num_edge.png", proposed_label=proposed_label)


if __name__ == "__main__":
    main()
