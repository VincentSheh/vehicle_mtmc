"""
Sweep offload weights (w1=0 fixed, vary w2/w3/w4) across offload modes and
report average QoE + benign_col_dmg per configuration.
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

CFG_PATH = "./configs/simulation_ma_0.yaml"
N_EPISODES = 3
SEED_BASE   = 1000

# ------------------------------------------------------------------
# Experiment grid
# ------------------------------------------------------------------
W2_VALUES = [0.0, 1.0, 1.5]        # FPR improvement weight
W3_VALUES = [0.1, 0.25, 0.5, 1.0]  # delay penalty weight
W4_VALUES = [0.0, 0.15]            # IDS util balance weight

MODES = ["balance", "full"]         # "balance" ignores all weights

MAX_PROP_DELAY_MS = 50.0            # tightened to actual range


# ------------------------------------------------------------------
def run_config(base_cfg: dict, mode: str, w2: float, w3: float, w4: float) -> dict:
    cfg = copy.deepcopy(base_cfg)
    g = cfg["globals"]

    g["offload_mode"] = mode
    g["max_prop_delay_ms"] = MAX_PROP_DELAY_MS
    g["offload_weights"] = {"w1": 0.0, "w2": w2, "w3": w3, "w4": w4}

    qoes, bcds = [], []
    for ep in range(N_EPISODES):
        env = build_env_from_cfg(cfg)
        env.reset(SEED_BASE + ep * 1000)

        n_edges = len(env.edge_areas)
        ids_cpu = np.array([e.ids_cpu for e in env.edge_areas], dtype=np.float32)

        t = 0
        decision_interval = 300
        while t < env.t_max:
            for _ in range(decision_interval):
                env.step(ids_cpu.tolist())
                t += 1
                if t >= env.t_max:
                    break

        df = pd.DataFrame([m.__dict__ for m in env.history])
        qoes.append(float(df["qoe_mean"].mean()))
        bcds.append(float(df["benign_col_dmg"].mean()))

    return {
        "mode": mode,
        "w2": w2,
        "w3": w3,
        "w4": w4,
        "qoe_mean": float(np.mean(qoes)),
        "qoe_std":  float(np.std(qoes)),
        "bcd_mean": float(np.mean(bcds)),
    }


def main():
    base_cfg = yaml.safe_load(Path(CFG_PATH).read_text())

    rows = []

    # balance mode: weights have no effect, run once as baseline
    print("Running baseline: balance mode ...", flush=True)
    r = run_config(base_cfg, mode="balance", w2=0.0, w3=0.0, w4=0.0)
    rows.append(r)
    print(f"  balance  → QoE={r['qoe_mean']:.4f} ± {r['qoe_std']:.4f}  BCD={r['bcd_mean']:.4f}")

    # full mode: sweep (w2, w3, w4)
    combos = list(itertools.product(W2_VALUES, W3_VALUES, W4_VALUES))
    total = len(combos)
    for idx, (w2, w3, w4) in enumerate(combos, 1):
        print(f"[{idx:02d}/{total}] full  w2={w2}  w3={w3}  w4={w4} ...", flush=True)
        r = run_config(base_cfg, mode="full", w2=w2, w3=w3, w4=w4)
        rows.append(r)
        print(f"         → QoE={r['qoe_mean']:.4f} ± {r['qoe_std']:.4f}  BCD={r['bcd_mean']:.4f}")

    df = pd.DataFrame(rows)
    out = "logs/weight_sweep.csv"
    Path("logs").mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print("\n=== Top 10 by QoE ===")
    print(df.sort_values("qoe_mean", ascending=False).head(10).to_string(index=False))
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
