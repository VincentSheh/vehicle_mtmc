#!/usr/bin/env python3
"""
Offline TBSA (Table-Based Static Allocation) search.

Sweeps constant IDS CPU allocations over multiple episodes and decision windows.
For each window, records (attack_in_rate, local_num_req, ids_cpu, qoe).
Builds a 2D lookup table:
    (attack_rate_bin, req_rate_bin)  →  best IDS CPU allocation (highest mean QoE)

Output: tbsa_table.npz  +  tbsa_table_raw.csv (for inspection)

Usage:
    python tbsa_offline.py --cfg configs/simulation_0.yaml
    python tbsa_offline.py --cfg configs/simulation_0.yaml --n_episodes 20 --out tbsa_table.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from environment import build_env_base


class TBSAPolicy:
    """
    Table-Based Static Allocation (TBSA) policy.

    Loads an offline lookup table built by ``tbsa_offline.py`` and, at every
    decision interval, maps the current (avg_attack_in_rate, avg_local_num_req)
    to the IDS CPU allocation that historically achieved the best QoE in that
    operating region.
    """

    def __init__(self, table_path: str, default_cpu: float = 4.0):
        data = np.load(table_path)
        self.best_cpu: np.ndarray = data["best_cpu"]         # (n_attack_bins, n_req_bins)
        self.attack_edges: np.ndarray = data["attack_edges"]  # percentile bin edges
        self.req_edges: np.ndarray = data["req_edges"]        # percentile bin edges
        self.default_cpu = default_cpu

    def select_ids_cpu(self, avg_attack_rate: float, avg_req_rate: float) -> float:
        """Return the pre-computed best IDS CPU for the given operating point."""
        n_attack = self.best_cpu.shape[0]
        n_req = self.best_cpu.shape[1]

        # digitize against interior edges (same convention as tbsa_offline.py)
        a_idx = int(np.clip(
            np.digitize(avg_attack_rate, self.attack_edges[1:-1]),
            0, n_attack - 1,
        ))
        r_idx = int(np.clip(
            np.digitize(avg_req_rate, self.req_edges[1:-1]),
            0, n_req - 1,
        ))
        return float(self.best_cpu[a_idx, r_idx])

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data(
    cfg_path: str,
    ids_cpu_values: list,
    n_episodes: int,
    decision_interval: int,
) -> pd.DataFrame:
    """
    Run episodes with each constant IDS CPU value and record per-window metrics.

    Returns a DataFrame with columns:
        ids_cpu, attack_in_rate, local_num_req, qoe, episode
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg["run"]["seed"])
    t_max = int(cfg["run"]["t_max"])

    env = build_env_base(cfg_path)
    n_edges = len(env.edge_areas)
    ids_cpu_max_arr = np.array(
        [e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32
    )

    rows = []

    for ids_cpu_val in tqdm(ids_cpu_values, desc="IDS CPU sweep"):
        ids_cpu = np.clip(
            np.full(n_edges, float(ids_cpu_val), dtype=np.float32),
            0.5,
            ids_cpu_max_arr,
        )

        for ep in range(n_episodes):
            # Use a seed that varies by both episode and IDS CPU level so
            # different CPU values see genuinely different attack sequences.
            seed = base_seed + ep * 1000 + int(round(ids_cpu_val * 100))
            env.reset(seed)

            t = 0
            while t < t_max:
                for _ in range(decision_interval):
                    env.step(ids_cpu.tolist())
                    t += 1
                    if t >= t_max:
                        break

                # Only record once we have a full window
                if len(env.history) < decision_interval * n_edges:
                    continue

                block = env.history[-decision_interval * n_edges :]
                df_block = pd.DataFrame([m.__dict__ for m in block])

                avg_attack = float(df_block["attack_in_rate"].mean()) # During Offline Search, use the ground truth
                avg_req = float(df_block["local_num_req"].mean())
                avg_qoe = float(df_block["qoe_mean"].mean())

                rows.append(
                    {
                        "ids_cpu": float(ids_cpu_val),
                        "attack_drop_rate": avg_attack,
                        "local_num_req": avg_req,
                        "qoe": avg_qoe,
                        "episode": ep,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lookup table construction
# ---------------------------------------------------------------------------

def _fill_nearest(grid: np.ndarray, has_data: np.ndarray) -> np.ndarray:
    """Fill empty cells in `grid` using nearest non-empty cell (BFS)."""
    if has_data.all():
        return grid

    filled = grid.copy()
    rows_idx, cols_idx = np.indices(grid.shape)

    # Collect coordinates of cells with data
    src_r = rows_idx[has_data].ravel()
    src_c = cols_idx[has_data].ravel()

    tgt_r = rows_idx[~has_data].ravel()
    tgt_c = cols_idx[~has_data].ravel()

    for r, c in zip(tgt_r, tgt_c):
        dists = (src_r - r) ** 2 + (src_c - c) ** 2
        nearest = int(np.argmin(dists))
        filled[r, c] = grid[src_r[nearest], src_c[nearest]]

    return filled


def build_lookup_table(
    df: pd.DataFrame,
    n_attack_bins: int,
    n_req_bins: int,
) -> dict:
    """
    Build a 2D best-CPU table from collected (attack_rate, req_rate, ids_cpu, qoe) data.

    Bin edges are derived from observed data percentiles so the grid covers
    the distribution of values seen in practice.

    Returns dict with keys: best_cpu, attack_edges, req_edges
    """
    # Bin edges from percentiles of observed data
    attack_edges = np.percentile(
        df["attack_drop_rate"].values,
        np.linspace(0, 100, n_attack_bins + 1),
    )
    req_edges = np.percentile(
        df["local_num_req"].values,
        np.linspace(0, 100, n_req_bins + 1),
    )

    df = df.copy()
    # np.digitize returns 1-based bucket; interior edges only so we don't need
    # the outer bounds
    df["attack_bin"] = np.clip(
        np.digitize(df["attack_drop_rate"].values, attack_edges[1:-1]),
        0,
        n_attack_bins - 1,
    )
    df["req_bin"] = np.clip(
        np.digitize(df["local_num_req"].values, req_edges[1:-1]),
        0,
        n_req_bins - 1,
    )

    # Mean QoE per (attack_bin, req_bin, ids_cpu)
    grouped = (
        df.groupby(["attack_bin", "req_bin", "ids_cpu"])["qoe"]
        .mean()
        .reset_index()
    )

    # Best ids_cpu per (attack_bin, req_bin)
    best_idx = grouped.groupby(["attack_bin", "req_bin"])["qoe"].idxmax()
    best = grouped.loc[best_idx].set_index(["attack_bin", "req_bin"])

    # Default fallback: median of all swept values
    all_ids_cpu = df["ids_cpu"].unique()
    default_cpu = float(np.median(all_ids_cpu))

    best_cpu = np.full((n_attack_bins, n_req_bins), default_cpu, dtype=np.float32)
    has_data = np.zeros((n_attack_bins, n_req_bins), dtype=bool)

    for (ab, rb), row in best.iterrows():
        best_cpu[int(ab), int(rb)] = float(row["ids_cpu"])
        has_data[int(ab), int(rb)] = True

    # Fill empty cells with nearest-neighbor
    best_cpu = _fill_nearest(best_cpu, has_data)

    return {
        "best_cpu": best_cpu,
        "attack_edges": attack_edges,
        "req_edges": req_edges,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build TBSA offline lookup table"
    )
    ap.add_argument("--cfg", default="./configs/simulation_0.yaml")
    ap.add_argument("--out", default="tbsa_table.npz",
                    help="Output .npz path for the lookup table")
    ap.add_argument("--n_episodes", type=int, default=10,
                    help="Episodes per IDS CPU value")
    ap.add_argument("--decision_interval", type=int, default=300,
                    help="Ticks between decisions (should match eval)")
    ap.add_argument("--n_attack_bins", type=int, default=10)
    ap.add_argument("--n_req_bins", type=int, default=10)
    ap.add_argument(
        "--ids_cpu_values",
        nargs="+",
        type=float,
        default=list(np.arange(0.5, 8.0, 0.5)),
        help="Constant IDS CPU allocations to sweep",
    )
    args = ap.parse_args()

    print(
        f"Sweeping {len(args.ids_cpu_values)} IDS CPU values × "
        f"{args.n_episodes} episodes each"
    )
    print(f"IDS CPU values: {[round(v, 2) for v in args.ids_cpu_values]}")

    df = collect_data(
        cfg_path=args.cfg,
        ids_cpu_values=args.ids_cpu_values,
        n_episodes=args.n_episodes,
        decision_interval=args.decision_interval,
    )

    if df.empty:
        raise RuntimeError("No data collected — check config and decision_interval.")

    print(f"\nCollected {len(df)} data points")
    print(
        f"Attack drop rate : [{df['attack_drop_rate'].min():.2f}, "
        f"{df['attack_drop_rate'].max():.2f}]"
    )
    print(
        f"Request rate : [{df['local_num_req'].min():.2f}, "
        f"{df['local_num_req'].max():.2f}]"
    )
    print(f"QoE range    : [{df['qoe'].min():.3f}, {df['qoe'].max():.3f}]")

    table = build_lookup_table(df, args.n_attack_bins, args.n_req_bins)

    out_path = Path(args.out)
    np.savez(
        out_path,
        best_cpu=table["best_cpu"],
        attack_edges=table["attack_edges"],
        req_edges=table["req_edges"],
    )
    print(f"\nSaved lookup table → {out_path}")
    print(f"Table shape : {table['best_cpu'].shape}")
    print(
        f"Best CPU    : [{table['best_cpu'].min():.2f}, "
        f"{table['best_cpu'].max():.2f}]"
    )
    print("\nBest-CPU table (attack_bin × req_bin):")
    print(pd.DataFrame(table["best_cpu"]).to_string())

    raw_path = out_path.with_name(out_path.stem + "_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data → {raw_path}")


if __name__ == "__main__":
    main()
