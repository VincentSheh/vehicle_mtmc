"""
Evaluate baseline policies across 3 attack × 3 user levels and plot a 3×3 grid.

Two plot versions are produced:
  scenario_grid_by_user.png   — rows = attack level, x-axis = user level
  scenario_grid_by_attack.png — rows = user level,   x-axis = attack level

If scenario_grid.csv already exists in --outdir the simulation is skipped and
the plots are regenerated from the saved values.

Usage:
    # Dummy mode — layout preview with random data
    python eval_scenario_grid.py --dummy

    # Real evaluation
    python eval_scenario_grid.py --cfg configs/simulation_0.yaml --episodes 10

    # Subset of methods
    python eval_scenario_grid.py --methods no_ids autoscale_def lstm_rl --episodes 5
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import yaml
from tqdm import tqdm

from method_policy import OFFLOAD_DISPLAY_NAMES

LEVELS = ["low", "mid", "high"]
LEVEL_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}

METRICS = [
    ("slo_vio",  "SLO Violation Rate"),
    ("bcd",      "Benign Collateral Damage"),
    ("atk_drop", "Attack Drop %"),
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
}


DEFAULT_METHODS = [
    "no_ids",
    "static_low",
    "static_high",
    "autoscale_def",
    "offline_optimal",
    "lstm_rl",
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
    # Custom handling for attack drop to exclude non-attack periods
    lres = arrays.get("reward_lambda_res", np.array([], dtype=np.float32))
    atk_in = arrays.get("attack_in_rate", np.array([], dtype=np.float32))
    if lres.size > 0 and atk_in.size == lres.size:
        mask = atk_in > 1e-6
        atk_drop_val = 1.0 - float(np.mean(lres[mask])) if mask.any() else 0.0
    else:
        atk_drop_val = np.nan

    return {
        "slo_vio":  _extract_mean(arrays, "qoe_vio_rate"),
        "bcd":      _extract_mean(arrays, "reward_benign_col_dmg"),
        "atk_drop": atk_drop_val,
        "realloc":  _extract_mean(arrays, "reallocations"),
    }


# ---------------------------------------------------------------------------
# Means dict helpers  (means[atk_lvl][user_lvl][method_label][metric_key])
# ---------------------------------------------------------------------------

def build_means_from_data(data: dict, methods_display: list[str]) -> dict:
    means: dict = {}
    for atk_lvl in LEVELS:
        means[atk_lvl] = {}
        for user_lvl in LEVELS:
            means[atk_lvl][user_lvl] = {}
            for method_label in methods_display:
                arrays = data.get(atk_lvl, {}).get(user_lvl, {}).get(method_label, {})
                means[atk_lvl][user_lvl][method_label] = _arrays_to_means(arrays)
    return means


def load_means_from_csv(path: Path) -> tuple[dict, list[str]]:
    """Return (means dict, ordered list of method labels found in CSV)."""
    means: dict = {a: {u: {} for u in LEVELS} for a in LEVELS}
    method_order: list[str] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            atk, user, method, metric, value = (
                row["atk_level"], row["user_level"],
                row["method"], row["metric"], float(row["value"]),
            )
            if method not in method_order:
                method_order.append(method)
            means[atk][user].setdefault(method, {})[metric] = value
    return means, method_order


def save_means_to_csv(means: dict, methods_display: list[str], outpath: Path):
    rows = []
    metric_label_map = dict(METRICS)
    for atk_lvl in LEVELS:
        for user_lvl in LEVELS:
            for method_label in methods_display:
                m = means.get(atk_lvl, {}).get(user_lvl, {}).get(method_label, {})
                for metric_key, metric_title in METRICS:
                    rows.append({
                        "atk_level":    atk_lvl,
                        "user_level":   user_lvl,
                        "method":       method_label,
                        "metric":       metric_key,
                        "metric_label": metric_title,
                        "value":        m.get(metric_key, float("nan")),
                    })
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["atk_level", "user_level", "method", "metric", "metric_label", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Dummy data generation
# ---------------------------------------------------------------------------

def _make_dummy_data(method_labels: list[str]) -> dict:
    """Returns data[atk_lvl][user_lvl][method_label] = {raw_key: array}.
    Accepts pre-built display labels (already include offload mode suffix when applicable).
    """
    rng = np.random.default_rng(0)
    data: dict = {}
    for i, atk_lvl in enumerate(LEVELS):
        data[atk_lvl] = {}
        for j, user_lvl in enumerate(LEVELS):
            data[atk_lvl][user_lvl] = {}
            for k, label in enumerate(method_labels):
                n = 50
                base_slo  = 0.05 + 0.15 * i + 0.05 * k + 0.03 * j
                base_bcd  = 0.02 + 0.08 * i + 0.02 * k - 0.01 * j
                base_drop = 0.90 - 0.10 * i - 0.05 * k + 0.03 * j
                data[atk_lvl][user_lvl][label] = {
                    "qoe_vio_rate":          np.clip(rng.normal(base_slo,        0.04, n), 0, 1   ).astype(np.float32),
                    "reward_benign_col_dmg": np.clip(rng.normal(base_bcd,        0.02, n), 0, None).astype(np.float32),
                    "reward_lambda_res":     np.clip(rng.normal(1.0 - base_drop, 0.04, n), 0, 1   ).astype(np.float32),
                    "attack_in_rate":        np.ones(n, dtype=np.float32),
                    "reallocations":         np.array([rng.uniform(2, 10) for _ in range(5)], dtype=np.float32),
                }
    return data


# ---------------------------------------------------------------------------
# Real evaluation
# ---------------------------------------------------------------------------

def _make_display_label(mname: str, offload_mode: str, multi_offload: bool) -> str:
    base = DISPLAY_NAMES.get(mname, mname)
    if multi_offload:
        ol = OFFLOAD_DISPLAY_NAMES.get(offload_mode, offload_mode)
        return f"{base} ({ol})"
    return base


def collect_data(args) -> dict:
    from eval_baselines import run_episode
    from environment import build_env_base
    from train_sa_lstm import TorchRLEnvWrapper
    from method_policy import make_baseline_policy

    with open(args.cfg) as f:
        cfg_original = yaml.safe_load(f)

    base_seed = int(cfg_original["run"]["seed"])
    np.random.seed(base_seed)

    methods: list[str] = args.methods if args.methods else list(DEFAULT_METHODS)

    cfg_offload_mode = cfg_original["globals"].get("offload_mode", "balance")
    offload_modes: list[str] = args.offload_modes if args.offload_modes else [cfg_offload_mode]
    multi_offload = len(offload_modes) > 1

    # Derive obs/reward params once from the original config
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

    # Build policies once (they are env-independent)
    policies: dict = {}
    for mname in methods:
        tbsa_path = str(args.tbsa_table) if mname in ("tbsa", "offline_optimal") else None
        ckpt      = args.ckpt            if mname == "lstm_rl" else None
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

    valid_methods = [m for m in methods if m in policies]

    data: dict = {}
    for atk_lvl in LEVELS:
        data[atk_lvl] = {}
        for user_lvl in LEVELS:
            print(f"\n{'='*70}")
            print(f" Evaluating: atk={atk_lvl}  user={user_lvl}")
            print("="*70)

            if not valid_methods:
                print(f"[warn] No valid methods, skipping.")
                data[atk_lvl][user_lvl] = {}
                continue

            cell: dict = {}

            for offload_mode in offload_modes:
                cfg = copy.deepcopy(cfg_original)
                cfg["globals"]["attack_sampler"]["level"] = atk_lvl
                cfg["globals"]["user_sampler"]["synthetic"]["level"] = user_lvl
                cfg["globals"]["offload_mode"] = offload_mode

                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                    yaml.dump(cfg, tmp)
                    tmp_path = tmp.name

                try:
                    env = build_env_base(tmp_path)

                    accumulated: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in valid_methods}
                    last_ids_cpu: Dict[str, Optional[np.ndarray]] = {m: None for m in valid_methods}

                    for ep in tqdm(range(args.episodes),
                                   desc=f"episodes (offload={offload_mode})"):
                        ep_seed = base_seed + (ep + 1) * 1000
                        for mname in valid_methods:
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
                                initial_ids_cpu=last_ids_cpu[mname],
                            )
                            last_ids_cpu[mname] = final_ids

                            ep_ratios = ep_result["cpu_to_ids_ratio"]
                            realloc_count = 0
                            if ep_ratios.size > 1:
                                realloc_count = np.sum(np.abs(np.diff(ep_ratios)) > 1e-6)
                            ep_result["reallocations"] = np.array([realloc_count], dtype=np.float32)

                            for k, v in ep_result.items():
                                if k not in accumulated[mname]:
                                    accumulated[mname][k] = v
                                else:
                                    accumulated[mname][k] = np.concatenate([accumulated[mname][k], v], axis=0)

                    for mname in valid_methods:
                        label = _make_display_label(mname, offload_mode, multi_offload)
                        cell[label] = accumulated[mname]

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            data[atk_lvl][user_lvl] = cell

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(
    means: dict,
    methods_display: list[str],
    outpath: Path,
    x_dim: str,          # "user" or "attack"
):
    """
    x_dim="user":   rows = attack levels, x-axis = user levels
    x_dim="attack": rows = user levels,   x-axis = attack levels
    """
    if x_dim == "user":
        row_levels, x_levels = LEVELS, LEVELS
        row_prefix, x_label  = "Atk:", "User Level"
        get_val = lambda means, row_lvl, x_lvl, mk: (
            means.get(row_lvl, {}).get(x_lvl, {}).get(method_label, {}).get(mk, np.nan)
        )
    else:
        row_levels, x_levels = LEVELS, LEVELS
        row_prefix, x_label  = "User:", "Attack Level"
        get_val = lambda means, row_lvl, x_lvl, mk: (
            means.get(x_lvl, {}).get(row_lvl, {}).get(method_label, {}).get(mk, np.nan)
        )

    n_rows, n_cols = len(row_levels), len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             sharex=False, squeeze=False)

    cmap = plt.get_cmap("tab10")
    colors    = {m: cmap(i % 10) for i, m in enumerate(methods_display)}
    markers   = ["o", "s", "^", "D", "v", "P", "*", "X"]
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(methods_display)}

    x_pos    = np.arange(len(x_levels))
    x_labels = [LEVEL_LABELS[l] for l in x_levels]

    for row_i, row_lvl in enumerate(row_levels):
        for col_j, (metric_key, metric_title) in enumerate(METRICS):
            ax = axes[row_i][col_j]

            for method_label in methods_display:
                vals = [get_val(means, row_lvl, x_lvl, metric_key) for x_lvl in x_levels]
                ax.plot(
                    x_pos, vals,
                    f"-{marker_map[method_label]}",
                    color=colors[method_label],
                    label=method_label,
                    linewidth=1.5,
                    markersize=6,
                )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.set_xlim(-0.4, len(x_levels) - 0.6)

            if metric_key in ("slo_vio", "atk_drop"):
                ax.yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
                )

            if row_i == 0:
                ax.set_title(metric_title, fontsize=12, fontweight="bold")
            if col_j == 0:
                ax.set_ylabel(f"{row_prefix} {LEVEL_LABELS[row_lvl]}", fontsize=11)
            if row_i == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=10)

            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        frameon=True,
        fontsize=10,
        title="Method",
        title_fontsize=10,
    )

    title = (
        "Performance vs User Level (per Attack Level)"
        if x_dim == "user"
        else "Performance vs Attack Level (per User Level)"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 0.84, 1.0])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg",               default="configs/simulation_0.yaml")
    ap.add_argument("--methods",           nargs="+", default=None)
    ap.add_argument("--episodes",          type=int,   default=10)
    ap.add_argument("--outdir",            default="eval_out/scenario_grid")
    ap.add_argument("--dummy",             action="store_true",
                    help="Use random data; skip simulation for layout verification")
    ap.add_argument("--ckpt",              default="checkpoints/singleedge/rew_32_netting_a4_rew/ckpt_best.pt")
    ap.add_argument("--tbsa_table",        default="tbsa_table.npz")
    ap.add_argument("--ids_cpu_min",       type=float, default=0.5)
    ap.add_argument("--scale_step",        type=float, default=0.5)
    ap.add_argument("--decision_interval", type=int,   default=None)
    ap.add_argument("--device",            default="cpu")
    ap.add_argument("--offload_modes",     nargs="+", default=None,
                    help="Offload modes to compare (e.g. none balance delay_workload full). "
                         "Defaults to the value in the config file.")
    args = ap.parse_args()

    outdir  = Path(args.outdir)
    csv_path = outdir / "scenario_grid.csv"

    def _build_methods_display(raw_methods: list[str], offload_modes: list[str]) -> list[str]:
        multi_offload = len(offload_modes) > 1
        labels = []
        for om in offload_modes:
            for m in raw_methods:
                labels.append(_make_display_label(m, om, multi_offload))
        return labels

    if args.dummy:
        print("[dummy mode] Generating synthetic data for layout preview...")
        methods = args.methods if args.methods else list(DEFAULT_METHODS)
        offload_modes = args.offload_modes or ["balance"]
        methods_display = _build_methods_display(methods, offload_modes)
        data  = _make_dummy_data(methods_display)
        means = build_means_from_data(data, methods_display)
        save_means_to_csv(means, methods_display, csv_path)

    elif csv_path.exists():
        print(f"[replot] Found existing {csv_path} — skipping simulation.")
        means, methods_display = load_means_from_csv(csv_path)

    else:
        methods = args.methods if args.methods else list(DEFAULT_METHODS)
        with open(args.cfg) as _f:
            _cfg_orig = yaml.safe_load(_f)
        offload_modes = args.offload_modes or [_cfg_orig["globals"].get("offload_mode", "balance")]
        methods_display = _build_methods_display(methods, offload_modes)
        data  = collect_data(args)
        means = build_means_from_data(data, methods_display)
        save_means_to_csv(means, methods_display, csv_path)

    plot_grid(means, methods_display, outdir / "scenario_grid_by_user.png",   x_dim="user")
    plot_grid(means, methods_display, outdir / "scenario_grid_by_attack.png", x_dim="attack")


if __name__ == "__main__":
    main()
