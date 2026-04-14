from __future__ import annotations

import os
import sys
import dataclasses
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment import build_env_base


T_MAX = 2500   # fixed episode length for the sweep


def _fix_attack_lambda(env, cfg_path: str):
    """Pin every AttackTypeSpec's lambda_base to the midpoint of the config
    range before reset() samples and instantiates the attacker, so all sweep
    points use the same fixed average attack intensity.
    AttackTypeSpec is frozen so dataclasses.replace is used."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    lb_range = cfg["globals"]["attack_sampler"]["lambda_base"]
    lambda_fixed = (lb_range[0] + lb_range[1]) / 2.0
    for edge in env.edge_areas:
        lib = edge.attack_type_library
        if lib is not None:
            lib._specs = [
                dataclasses.replace(s, lambda_base=lambda_fixed)
                for s in lib._specs
            ]


def _fix_users(env, mu0 = 40):
    """Scale mu_min/mu_max/sigma from the config's per-step intent to per-sec
    so that generate_req_trace (which multiplies by dt=slot_ms/1000) produces
    the originally intended per-step request rates."""
    dt = env.edge_areas[0].slot_ms / 1000.0   # 0.2 at fps=5
    scale = 1.0 / dt                            # 5.0
    for edge in env.edge_areas:
        for user in edge.users:
            if user.synth_cfg is not None:
                cfg = user.synth_cfg
                cfg["mu_min"] = mu0 * scale
                cfg["mu_max"] = mu0 * scale
                cfg["sigma"]  = 0


def _fix_attack(env, t_max: int):
    """Force every attacker to run for exactly t_max steps starting at t=0."""
    for edge in env.edge_areas:
        for a in edge.attackers:
            a.start      = 0
            a.active_len = t_max
            # _flows was generated with the original active_len (t_max//2 of config);
            # tile or slice to match the requested length.
            if hasattr(a, "_flows"):
                if len(a._flows) < t_max:
                    reps = -(-t_max // len(a._flows))   # ceiling division
                    a._flows     = np.tile(a._flows,     reps)[:t_max]
                    a._flows_ema = np.tile(a._flows_ema, reps)[:t_max]
                else:
                    a._flows     = a._flows[:t_max]
                    a._flows_ema = a._flows_ema[:t_max]


def run_sweep(cfg_path: str, plot: bool = False, out_dir: str = "./output/env_sweep"):
    os.makedirs(out_dir, exist_ok=True)

    # Determine CPU budget
    _probe = build_env_base(cfg_path)
    cpu_budget = _probe.edge_areas[0].budget.cpu
    ids_cpu_values = np.linspace(0.0, cpu_budget - 0.5, 48)

    records = []

    for ids_cpu_val in ids_cpu_values:
        env = build_env_base(cfg_path)
        env.t_max = T_MAX          # override episode length before reset
        _fix_users(env)            # scale mu to per-sec before reset generates traces
        _fix_attack_lambda(env, cfg_path)  # pin lambda_base to config midpoint
        env.reset(seed=1000)
        _fix_attack(env, T_MAX)    # pin attack to full T_MAX steps from t=0

        for _ in range(T_MAX):
            env.step([ids_cpu_val])

        df = pd.DataFrame([m.__dict__ for m in env.history])
        assert not df.empty,                              "No metrics produced"
        assert np.isfinite(df["qoe_mean"]).all(),         "Invalid QoE values"
        assert df["ids_coverage"].between(0, 1).all(),    "IDS coverage out of range"
        assert df["attack_cpu_frac"].between(0, 1).all(), "Attack CPU frac out of range"

        avg_qoe          = float(df["qoe_mean"].mean())
        avg_coverage     = float(df["ids_coverage"].mean())
        avg_atk_cpu_frac = float(df["attack_cpu_frac"].mean())

        records.append({
            "ids_cpu":         ids_cpu_val,
            "avg_qoe":         avg_qoe,
            "avg_coverage":    avg_coverage,
            "avg_atk_cpu_frac": avg_atk_cpu_frac,
        })

        print(
            f"ids_cpu={ids_cpu_val:.2f} | "
            f"QoE={avg_qoe:.4f} | "
            f"Coverage={avg_coverage:.4f} | "
            f"Atk CPU%={avg_atk_cpu_frac:.4f}"
        )

    result_df = pd.DataFrame(records)
    result_df.to_csv(os.path.join(out_dir, "ids_cpu_qoe_coverage_results.csv"), index=False)

    ids_cpu_arr        = result_df["ids_cpu"].to_numpy()
    qoe_arr            = result_df["avg_qoe"].to_numpy()
    coverage_arr       = result_df["avg_coverage"].to_numpy()
    atk_exhaustion_arr = result_df["avg_atk_cpu_frac"].to_numpy()

    # ------------------------------------------------------------------
    # Region classification
    # ------------------------------------------------------------------
    peak_qoe = float(qoe_arr.max()) if len(qoe_arr) > 0 else 1.0

    _region_meta = {
        1: ("tomato",       0.18, "R1: Severely Under-Defended"),
        2: ("orange",       0.15, "R2: Transition Onset"),
        3: ("limegreen",    0.18, "R3: Efficient Operating Region"),
        4: ("deepskyblue",  0.15, "R4: Safe but Over-Defended"),
        5: ("mediumpurple", 0.18, "R5: Excessively Over-Defended"),
    }

    def _classify(atk_frac: float, qoe: float) -> int:
        if atk_frac > 0.95:
            return 1
        if atk_frac < 0.1 and qoe < 0.70 * peak_qoe:
            return 5
        if qoe >= 0.90 * peak_qoe:
            return 3
        if atk_frac <= 0.2:
            return 4
        return 2

    region_labels = [_classify(v, q) for v, q in zip(atk_exhaustion_arr, qoe_arr)]

    # ------------------------------------------------------------------
    # Main sweep plot
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))

    # Region shading (contiguous spans)
    _seen = set()
    i = 0
    n = len(ids_cpu_arr)
    while i < n:
        reg = region_labels[i]
        j = i + 1
        while j < n and region_labels[j] == reg:
            j += 1
        x0 = (ids_cpu_arr[i - 1] + ids_cpu_arr[i]) / 2 if i > 0 else ids_cpu_arr[0]
        x1 = (ids_cpu_arr[j - 1] + ids_cpu_arr[j]) / 2 if j < n else ids_cpu_arr[-1]
        color, alpha, label = _region_meta[reg]
        ax1.axvspan(x0, x1, alpha=alpha, color=color,
                    label=label if reg not in _seen else None)
        _seen.add(reg)
        i = j

    line1, = ax1.plot(ids_cpu_arr, qoe_arr,      "o-",  markersize=3, lw=1.5,
                      label="Avg QoE")
    line2, = ax2.plot(ids_cpu_arr, coverage_arr,  "s--", markersize=3, lw=1.5,
                      color="tab:orange", label="Avg IDS Coverage")
    line3, = ax3.plot(ids_cpu_arr, atk_exhaustion_arr,    "^-.", markersize=3, lw=1.5,
                      color="tab:red", label="CPU Exhaustion by Attacker")

    ax1.set_xlabel("IDS CPU Allocation (cores)", fontsize=13)
    ax1.set_ylabel("Average QoE",                fontsize=13)
    ax2.set_ylabel("Average IDS Coverage",        fontsize=13)
    ax3.set_ylabel("CPU Exhaustion by Attacker", fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax3.set_ylim(0, 1.05)
    ax1.set_title("IDS CPU Allocation vs QoE, IDS Coverage & CPU Exhaustion", fontsize=14)
    ax1.grid(True, alpha=0.3)

    region_handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=_region_meta[r][0],
                       alpha=0.5)
        for r in sorted(_seen)
    ]
    region_labels_legend = [_region_meta[r][2] for r in sorted(_seen)]

    ax1.legend(
        [line1, line2, line3] + region_handles,
        ["Avg QoE", "Avg IDS Coverage", "CPU Exhaustion by Attacker"] + region_labels_legend,
        loc="best", fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ids_cpu_vs_qoe_coverage.png"), dpi=200)
    if plot:
        plt.show()
    else:
        plt.close()

    # ------------------------------------------------------------------
    # Time-series plots (from last ids_cpu sweep point)
    # ------------------------------------------------------------------

    print(f"\nSweep results saved to {out_dir}/")
    return result_df


if __name__ == "__main__":
    _default_cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "simulation_0.yaml")
    run_sweep(_default_cfg, plot=True)
