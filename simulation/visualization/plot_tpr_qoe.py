"""
Plot steady-state QoE vs IDS True Positive Rate (TPR) via full simulation
episodes, for one or more (mu0, lambda_base, ids_cpu) combinations.

All combinations are drawn as separate curves on a single plot.
Color encodes (mu0, lambda_base); linestyle encodes ids_cpu.

Usage
-----
    # single operating point
    python plot_tpr_qoe.py --mu0 10 --lambda_base 7000 --ids_cpu 2.0

    # sweep over user loads, attack intensities and IDS allocations
    python plot_tpr_qoe.py \\
        --mu0 5 20 \\
        --lambda_base 4000 14000 \\
        --ids_cpu 1.0 4.0 \\
        --episodes 3 --n_tpr 20
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment import build_env_base

T_MAX = 2500


# ---------------------------------------------------------------------------
# Environment patching helpers
# ---------------------------------------------------------------------------

def _fix_users(env, mu0: float):
    dt = env.edge_areas[0].slot_ms / 1000.0
    rate_per_sec = mu0 / dt
    for edge in env.edge_areas:
        for user in edge.users:
            if user.synth_cfg is not None:
                cfg = user.synth_cfg
                cfg["mu_min"] = rate_per_sec
                cfg["mu_max"] = rate_per_sec
                cfg["sigma"]  = 0.0


def _fix_attack_lambda(env, lambda_base: float):
    for edge in env.edge_areas:
        lib = edge.attack_type_library
        if lib is not None:
            lib._specs = [
                dataclasses.replace(s, lambda_base=float(lambda_base))
                for s in lib._specs
            ]


def _fix_attack(env, t_max: int):
    for edge in env.edge_areas:
        for atk in edge.attackers:
            atk.start      = 0
            atk.active_len = t_max
            if hasattr(atk, "_flows"):
                if len(atk._flows) < t_max:
                    reps           = -(-t_max // len(atk._flows))
                    atk._flows     = np.tile(atk._flows,     reps)[:t_max]
                    atk._flows_ema = np.tile(atk._flows_ema, reps)[:t_max]
                else:
                    atk._flows     = atk._flows[:t_max]
                    atk._flows_ema = atk._flows_ema[:t_max]


def _fix_attack_constant(env, lambda_base: float, t_max: int):
    """Replace every attacker's flow trace with a flat constant rate.

    Calls _fix_attack_lambda (sets the spec) then _fix_attack (sets
    start/active_len), then overwrites _flows with a constant so the
    sinus/pw/expo envelope has no effect.
    """
    _fix_attack_lambda(env, lambda_base)
    _fix_attack(env, t_max)
    for edge in env.edge_areas:
        dt = edge.slot_ms / 1000.0
        flows_per_step = float(lambda_base) * dt
        for atk in edge.attackers:
            atk._flows     = np.full(t_max, flows_per_step, dtype=np.float32)
            atk._flows_ema = np.full(t_max, flows_per_step, dtype=np.float32)
            atk.scaling    = 1.0   # load_at multiplies _flows by scaling; pin to 1


def _fix_tpr(env, tpr: float, fpr: float):
    for edge in env.edge_areas:
        for k in list(edge.ids.acc_tpr_fpr.keys()):
            edge.ids.acc_tpr_fpr[k] = (float(tpr), float(fpr))


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, ids_cpu: float, tpr: float, fpr: float,
                lambda_base: float, mu0: float, seed: int) -> float:
    _fix_users(env, mu0)
    _fix_attack_lambda(env, lambda_base)   # must be before reset so spec is sampled correctly
    env.reset(seed=seed)
    _fix_attack_constant(env, lambda_base, T_MAX)  # overwrite trace with flat constant
    _fix_tpr(env, tpr, fpr)
    for _ in range(T_MAX):
        env.step([ids_cpu])
    df = pd.DataFrame([m.__dict__ for m in env.history])
    return float(df["qoe_mean"].mean()) if not df.empty else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot QoE vs TPR via simulation")
    ap.add_argument("--cfg",         default="../configs/simulation_0.yaml")
    ap.add_argument("--mu0",         type=float, nargs="+", required=True,
                    help="User requests per slot (one or more values)")
    ap.add_argument("--lambda_base", type=float, nargs="+", required=True,
                    help="Attack flow rate in flows/sec (one or more values)")
    ap.add_argument("--ids_cpu",     type=float, nargs="+", default=[3.0],
                    help="IDS CPU core allocations (one or more values)")
    ap.add_argument("--fpr",         type=float, default=0.01)
    ap.add_argument("--n_tpr",       type=int,   default=40)
    ap.add_argument("--episodes",    type=int,   default=5)
    ap.add_argument("--outfile",     default="visualization/output/env_sweep/qoe_vs_tpr.png")
    args = ap.parse_args()

    cfg_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), args.cfg)
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)

    cfg        = yaml.safe_load(open(cfg_path))
    slot_ms    = 1000.0 / float(cfg["globals"].get("fps", 5.0))
    fps        = 1000.0 / slot_ms
    budget_cpu = float(cfg["edge_areas"][0]["budget"]["cpu"])

    mu0_list     = sorted(args.mu0)
    lb_list      = sorted(args.lambda_base)
    ids_cpu_list = sorted(c for c in args.ids_cpu if c < budget_cpu)
    for skipped in set(args.ids_cpu) - set(ids_cpu_list):
        print(f"[warn] ids_cpu={skipped} >= budget_cpu={budget_cpu}, skipping")

    tpr_values = np.linspace(0.75, 1.0, args.n_tpr)

    # Color per (mu0, lambda_base) pair; linestyle per ids_cpu
    load_pairs  = [(mu0, lb) for mu0 in mu0_list for lb in lb_list]
    pair_colors = plt.cm.tab10(np.linspace(0, 1, max(len(load_pairs), 1)))
    linestyles  = ["-", "--", "-.", ":"]

    # One env per ids_cpu, reused across all (mu0, lambda_base, tpr, ep)
    envs = {c: build_env_base(cfg_path) for c in ids_cpu_list}
    for e in envs.values():
        e.t_max = T_MAX

    env_ceil = build_env_base(cfg_path)
    env_ceil.t_max = T_MAX

    total = len(ids_cpu_list) * len(load_pairs) * args.n_tpr * args.episodes
    done  = 0

    fig, ax = plt.subplots(figsize=(9, 5))

    seen_ceiling_mu0 = set()

    for (mu0, lb), pair_color in zip(load_pairs, pair_colors):
        # No-attack ceiling (once per unique mu0)
        if mu0 not in seen_ceiling_mu0:
            seen_ceiling_mu0.add(mu0)
            ceil_runs = []
            for ep in range(max(1, args.episodes)):
                _fix_users(env_ceil, mu0)
                _fix_attack_lambda(env_ceil, 0.0)
                env_ceil.reset(seed=1000 + ep * 97)
                _fix_attack_constant(env_ceil, 0.0, T_MAX)
                _fix_tpr(env_ceil, tpr=1.0, fpr=0.0)
                for _ in range(T_MAX):
                    env_ceil.step([0.0])
                df = pd.DataFrame([m.__dict__ for m in env_ceil.history])
                ceil_runs.append(float(df["qoe_mean"].mean()) if not df.empty else 0.0)
            qoe_ceil = float(np.mean(ceil_runs))
            ax.axhline(
                qoe_ceil, color=pair_color, linestyle=":", linewidth=0.9, alpha=0.5,
                label=f"ceiling  μ₀={mu0:.0f} ({qoe_ceil:.2f})",
            )

        for ids_cpu, ls in zip(ids_cpu_list, linestyles):
            env    = envs[ids_cpu]
            qoe_curve = np.zeros(args.n_tpr, dtype=np.float32)

            for ti, tpr in enumerate(tpr_values):
                ep_qoes = []
                for ep in range(args.episodes):
                    qoe = run_episode(
                        env, ids_cpu=ids_cpu, tpr=tpr, fpr=args.fpr,
                        lambda_base=lb, mu0=mu0, seed=1000 + ep * 97,
                    )
                    ep_qoes.append(qoe)
                    done += 1
                    print(
                        f"[{done:5d}/{total}]  μ₀={mu0:.0f}  "
                        f"λ={lb:.0f}  ids={ids_cpu:.1f}  "
                        f"tpr={tpr:.3f}  ep={ep}  QoE={qoe:.4f}"
                    )
                qoe_curve[ti] = float(np.mean(ep_qoes))

            ax.plot(
                tpr_values, qoe_curve,
                linestyle=ls, marker="o", markersize=3, lw=1.6,
                color=pair_color,
                label=f"μ₀={mu0:.0f}  λ={lb:.0f}  IDS={ids_cpu:.1f}",
            )

    ax.set_xlabel("TPR (True Positive Rate)", fontsize=12)
    ax.set_ylabel("Mean QoE", fontsize=12)
    ax.set_title(
        f"QoE vs TPR  —  {args.episodes} ep × {T_MAX} steps   FPR={args.fpr}   fps={fps:.0f}",
        fontsize=11,
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, loc="upper left", ncol=max(1, len(load_pairs) // 4))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.outfile, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.outfile}")


if __name__ == "__main__":
    main()
