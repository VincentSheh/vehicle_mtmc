"""
Sweep offload weights across modes and report average QoE + benign_col_dmg.

Weight scaling rationale for cto_acc:
  score = (FNR[s,s]-FNR[s,r])*w1 + (FPR[s,s]-FPR[s,r])*w2 - d_prop*w3

  FPR diff magnitude : |0.0 - 0.1| = 0.1   (with heterogeneous accuracy matrix)
  FNR diff magnitude : 0.0                  (all FNR=0 in current config)
  Delay range        : 20–35 ms

  Equal-scale condition: 0.1*w2 ≈ 20*w3  →  w3 ≈ 0.005*w2
  Grid anchors from that: w3 in [0.001, 0.005, 0.01, 0.025, 0.05]
    - w3 << 0.005*w2 : accuracy dominates, topology ignored
    - w3 >> 0.005*w2 : delay dominates, degenerates to plain cto
"""
from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from environment import build_env_from_cfg

CFG_PATH        = "./configs/simulation_ma_0.yaml"
N_EPISODES      = 3
SEED_BASE       = 1000
DECISION_INTERVAL = 300
MAX_PROP_DELAY_MS = 50.0

# ------------------------------------------------------------------
# cto_acc weight grid  (w1=FNR, w2=FPR, w3=delay)
# w1 kept at 0 because FNR=0 everywhere in current config
# ------------------------------------------------------------------
W1_VALUES = [0.0]                            # FNR weight  (no FNR signal in config)
W2_VALUES = [0.0, 0.5, 1.0, 2.0]            # FPR weight
W3_VALUES = [0.001, 0.005, 0.01, 0.025, 0.05]  # delay penalty (calibrated to 20-35 ms range)


# ------------------------------------------------------------------
def _run_episodes(env, n_episodes: int, seed_base: int) -> tuple[list, list]:
    qoes, bcds = [], []
    for ep in range(n_episodes):
        env.reset(seed_base + ep * 1000)
        ids_cpu = np.array([e.ids_cpu for e in env.edge_areas], dtype=np.float32)
        t = 0
        while t < env.t_max:
            for _ in range(DECISION_INTERVAL):
                env.step(ids_cpu.tolist())
                t += 1
                if t >= env.t_max:
                    break
        df = pd.DataFrame([m.__dict__ for m in env.history])
        qoes.append(float(df["qoe_mean"].mean()))
        bcds.append(float(df["benign_col_dmg"].mean()))
    return qoes, bcds


def run_baseline(base_cfg: dict, mode: str) -> dict:
    """Run a weight-free baseline (balance, cto)."""
    cfg = copy.deepcopy(base_cfg)
    cfg["globals"]["offload_mode"] = mode
    env = build_env_from_cfg(cfg)
    qoes, bcds = _run_episodes(env, N_EPISODES, SEED_BASE)
    return {
        "mode": mode, "w1": "-", "w2": "-", "w3": "-",
        "qoe_mean": float(np.mean(qoes)),
        "qoe_std":  float(np.std(qoes)),
        "bcd_mean": float(np.mean(bcds)),
    }


def run_cto_acc(base_cfg: dict, w1: float, w2: float, w3: float) -> dict:
    cfg = copy.deepcopy(base_cfg)
    g = cfg["globals"]
    g["offload_mode"]    = "cto_acc"
    g["max_prop_delay_ms"] = MAX_PROP_DELAY_MS
    # w4 slot kept for interface compatibility; unused by cto_acc
    g["offload_weights"] = {"w1": w1, "w2": w2, "w3": w3, "w4": 0.0}
    env = build_env_from_cfg(cfg)
    qoes, bcds = _run_episodes(env, N_EPISODES, SEED_BASE)
    return {
        "mode": "cto_acc", "w1": w1, "w2": w2, "w3": w3,
        "qoe_mean": float(np.mean(qoes)),
        "qoe_std":  float(np.std(qoes)),
        "bcd_mean": float(np.mean(bcds)),
    }


def main():
    base_cfg = yaml.safe_load(Path(CFG_PATH).read_text())
    rows = []

    # --- Baselines ---
    for mode in ("balance", "cto"):
        print(f"Running baseline: {mode} ...", flush=True)
        r = run_baseline(base_cfg, mode)
        rows.append(r)
        print(f"  {mode:<8} → QoE={r['qoe_mean']:.4f} ± {r['qoe_std']:.4f}  BCD={r['bcd_mean']:.4f}")

    # --- cto_acc sweep ---
    combos = list(itertools.product(W1_VALUES, W2_VALUES, W3_VALUES))
    total  = len(combos)
    print(f"\nSweeping cto_acc over {total} combinations ...", flush=True)
    for idx, (w1, w2, w3) in enumerate(combos, 1):
        print(f"[{idx:02d}/{total}] cto_acc  w1={w1}  w2={w2}  w3={w3} ...", flush=True)
        r = run_cto_acc(base_cfg, w1, w2, w3)
        rows.append(r)
        print(f"          → QoE={r['qoe_mean']:.4f} ± {r['qoe_std']:.4f}  BCD={r['bcd_mean']:.4f}")

    df = pd.DataFrame(rows)
    Path("logs").mkdir(exist_ok=True)
    out = "logs/weight_sweep.csv"
    df.to_csv(out, index=False)

    print("\n=== Top 10 by QoE ===")
    print(df.sort_values("qoe_mean", ascending=False).head(10).to_string(index=False))

    # Show best cto_acc vs baselines
    print("\n=== Best cto_acc vs baselines ===")
    baselines = df[df["mode"].isin(["balance", "cto"])]
    best_cto_acc = df[df["mode"] == "cto_acc"].sort_values("qoe_mean", ascending=False).head(3)
    print(pd.concat([baselines, best_cto_acc]).to_string(index=False))

    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
