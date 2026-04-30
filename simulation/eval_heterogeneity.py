"""
Evaluate baseline and proposed methods across different FL heterogeneity levels 
(Dirichlet alpha values) and plot results.

Usage:
    # Dummy mode
    python eval_heterogeneity.py --dummy

    # Real evaluation
    python eval_heterogeneity.py --cfg configs/simulation_ma_0.yaml --episodes 10
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
    ("gm", "cto"),
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

def save_means_to_csv(means: dict, methods_display: list[str], alphas: list[float], outpath: Path):
    rows = []
    for alpha in alphas:
        for method_label in methods_display:
            m = means.get(alpha, {}).get(method_label, {})
            for metric_key, metric_title in METRICS:
                rows.append({
                    "alpha":        alpha,
                    "method":       method_label,
                    "metric":       metric_key,
                    "metric_label": metric_title,
                    "value":        m.get(metric_key, float("nan")),
                })
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["alpha", "method", "metric", "metric_label", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {outpath}")


def load_means_from_csv(path: Path) -> tuple[dict, list[str], list[float]]:
    means: dict = {}
    method_order: list[str] = []
    alphas_found: set[float] = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            alpha, method, metric, value = (
                float(row["alpha"]), row["method"],
                row["metric"], float(row["value"]),
            )
            if method not in method_order:
                method_order.append(method)
            alphas_found.add(alpha)
            means.setdefault(alpha, {}).setdefault(method, {})[metric] = value
    return means, method_order, sorted(list(alphas_found))


def _cell_is_complete(means: dict, alpha: float, methods_display: list[str]) -> bool:
    cell = means.get(alpha, {})
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

def _make_dummy_data(method_labels: list[str], alphas: list[float]) -> dict:
    rng = np.random.default_rng(0)
    data: dict = {}
    for i, alpha in enumerate(alphas):
        data[alpha] = {}
        # dummy values vary monotonically with log(alpha)
        log_alpha = np.log10(alpha)
        for k, label in enumerate(method_labels):
            n = 50
            # Higher alpha (more IID) -> better performance (lower metrics)
            base_slo  = np.clip(0.2 - 0.04 * log_alpha + 0.02 * k, 0.01, 1)
            base_bcd  = np.clip(0.1 - 0.02 * log_alpha + 0.01 * k, 0.005, 1)
            base_leak = np.clip(0.4 - 0.08 * log_alpha + 0.03 * k, 0.02, 1)
            
            data[alpha][label] = {
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

def collect_data(args, alphas: list[float], done_alphas: set[float] | None = None, cell_callback=None) -> dict:
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
    for alpha in alphas:
        if done_alphas and alpha in done_alphas:
            print(f"\n[resume] Skipping alpha={alpha} (already complete)")
            data[alpha] = {}
            continue

        print(f"\n{'='*70}")
        print(f" Evaluating: alpha={alpha}")
        print("="*70)

        if not runs:
            print("[warn] No valid methods, skipping.")
            data[alpha] = {}
            continue

        cell: dict = {}
        last_ids_cpu: Dict[str, Optional[np.ndarray]] = {label: None for _, _, _, label in runs}

        for (offload_mode, acc_model), method_label_pairs in env_groups.items():
            cfg = copy.deepcopy(cfg_original)
            cfg["globals"]["attack_sampler"]["dirichlet_alpha"] = alpha
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

        data[alpha] = cell
        if cell_callback is not None:
            cell_callback(alpha, cell)

    return data

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heterogeneity_bar(
    means: dict,
    methods_display: list[str],
    alphas: list[float],
    outpath: Path,
    proposed_label: str | None = None,
):
    # Reverse alpha order: highest alpha (lowest heterogeneity) first
    alphas_plot = sorted(alphas, reverse=True)
    n_rows = len(METRICS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True, squeeze=False)

    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods_display)}

    x = np.arange(len(alphas_plot))
    width = 0.8 / len(methods_display)

    for row_i, (metric_key, metric_title) in enumerate(METRICS):
        ax = axes[row_i][0]

        for i, method_label in enumerate(methods_display):
            vals = [means.get(a, {}).get(method_label, {}).get(metric_key, np.nan) for a in alphas_plot]
            pos = x + (i - len(methods_display)/2 + 0.5) * width
            rects = ax.bar(pos, vals, width, color=colors[method_label], label=method_label, alpha=0.8)

            # Annotate gap relative to proposed_label
            if proposed_label and method_label != proposed_label:
                for j, alpha in enumerate(alphas_plot):
                    ref_val = means.get(alpha, {}).get(proposed_label, {}).get(metric_key, np.nan)
                    val = vals[j]
                    if not np.isnan(ref_val) and not np.isnan(val):
                        gap = val - ref_val
                        # Absolute performance gap. For percentages, show as e.g. +5.2%
                        if metric_key in ("slo_vio", "atk_leak", "bcd"):
                            text = f"{gap:+.1%}"
                        else:
                            text = f"{gap:+.1f}"
                        
                        # Position text above the bar
                        ax.text(pos[j], val + 0.01 * ax.get_ylim()[1], text, 
                                ha='center', va='bottom', fontsize=8, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in alphas_plot])

        if metric_key in ("slo_vio", "atk_leak", "bcd"):
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
            )

        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.grid(axis='y', linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1][0].set_xlabel("Heterogeneity (Dirichlet α) — Reversed", fontsize=11)
    
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

    fig.suptitle("Performance vs FL Heterogeneity (Grouped Bar Chart)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


def plot_heterogeneity_line(
    means: dict,
    methods_display: list[str],
    alphas: list[float],
    outpath: Path,
):
    alphas_plot = sorted(alphas)
    n_rows = len(METRICS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True, squeeze=False)

    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods_display)}
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "H"]

    for row_i, (metric_key, metric_title) in enumerate(METRICS):
        ax = axes[row_i][0]
        for i, method_label in enumerate(methods_display):
            vals = [means.get(a, {}).get(method_label, {}).get(metric_key, np.nan) for a in alphas_plot]
            ax.plot(alphas_plot, vals, label=method_label, color=colors[method_label], 
                    marker=markers[i % len(markers)], markersize=6, linewidth=2, alpha=0.8)

        ax.set_xscale("log")
        if metric_key in ("slo_vio", "atk_leak", "bcd"):
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
            )

        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1][0].set_xlabel("Heterogeneity (Dirichlet α) - Log Scale", fontsize=11)
    
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

    fig.suptitle("Performance vs FL Heterogeneity (Line Chart)", fontsize=14, fontweight="bold", y=1.01)
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
    ap.add_argument("--outdir",            default="eval_out_hetero/")
    ap.add_argument("--dummy",             action="store_true")
    ap.add_argument("--ckpt",              default="checkpoints/_singleedge/a4_sf20_atk3_a18_default_default/ckpt_best.pt")
    ap.add_argument("--tbsa_table",        default="tbsa_table.npz")
    ap.add_argument("--ids_cpu_min",       type=float, default=0.5)
    ap.add_argument("--scale_step",        type=float, default=0.5)
    ap.add_argument("--decision_interval", type=int,   default=None)
    ap.add_argument("--device",            default="cpu")
    ap.add_argument("--offload_modes",     nargs="+", default=None)
    ap.add_argument("--proposed_method",   default="lstm_rl")
    ap.add_argument("--alphas",            nargs="+", type=float, default=None)
    ap.add_argument("--proposed_label",    default=None, help="The label of the proposed method for gap annotations")
    args = ap.parse_args()

    outdir  = Path(args.outdir)
    csv_path = outdir / "heterogeneity.csv"
    BASELINE_OFFLOAD = "delay_workload"

    with open(args.cfg) as f:
        cfg_orig = yaml.safe_load(f)

    # Resolve Alphas
    if args.alphas:
        alphas = sorted(args.alphas)
    else:
        matrix_path = cfg_orig["globals"]["accuracy_matrix"]["path"]
        with open(matrix_path) as f:
            matrix_data = json.load(f)
        alpha_set = set()
        for key in matrix_data:
            m = re.search(r'alpha([\d.]+)', key)
            if m:
                alpha_set.add(float(m.group(1)))
        alphas = sorted(list(alpha_set))
        if not alphas:
            alphas = [0.2, 0.5, 1.0, 10.0, 100.0, 1000.0]

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

    proposed_label = args.proposed_label
    if proposed_label is None and args.proposed_method is not None:
        # Default to the most advanced proposed config (usually last)
        proposed_label = methods_display[-1]

    if args.dummy:
        print("[dummy mode] Generating synthetic data...")
        data  = _make_dummy_data(methods_display, alphas)
        means = {a: {label: _arrays_to_means(arrs) for label, arrs in data[a].items()} for a in alphas}
        save_means_to_csv(means, methods_display, alphas, csv_path)
    else:
        accumulated_means: dict = {}
        done_alphas: set[float] = set()
        if csv_path.exists():
            try:
                accumulated_means, _, _ = load_means_from_csv(csv_path)
                for a in alphas:
                    if _cell_is_complete(accumulated_means, a, methods_display):
                        done_alphas.add(a)
                if len(done_alphas) == len(alphas):
                    print(f"[replot] All {len(alphas)} alphas complete in {csv_path} — skipping simulation.")
                    plot_heterogeneity_bar(accumulated_means, methods_display, alphas, outdir / "heterogeneity_bar.png", proposed_label=proposed_label)
                    plot_heterogeneity_line(accumulated_means, methods_display, alphas, outdir / "heterogeneity_line.png")
                    return
                elif done_alphas:
                    print(f"[resume] {len(done_alphas)}/{len(alphas)} alphas already complete, resuming...")
            except Exception as exc:
                print(f"[warn] Could not load {csv_path}: {exc} — starting fresh")
                accumulated_means = {}
                done_alphas = set()

        def _on_cell_done(alpha: float, cell_data: dict):
            accumulated_means[alpha] = {
                label: _arrays_to_means(arrays)
                for label, arrays in cell_data.items()
            }
            save_means_to_csv(accumulated_means, methods_display, alphas, csv_path)
            n_done = sum(_cell_is_complete(accumulated_means, a, methods_display) for a in alphas)
            print(f"[checkpoint] {n_done}/{len(alphas)} alphas saved (alpha={alpha} done)")

        collect_data(args, alphas, done_alphas=done_alphas, cell_callback=_on_cell_done)
        means = accumulated_means

    plot_heterogeneity_bar(means, methods_display, alphas, outdir / "heterogeneity_bar.png", proposed_label=proposed_label)
    plot_heterogeneity_line(means, methods_display, alphas, outdir / "heterogeneity_line.png")


if __name__ == "__main__":
    main()
