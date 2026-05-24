"""
Sweep CTO+Acc weights (w2) for different Dirichlet alpha values.
FNR weight (w1) is kept at 0.
Allocation: constant_3.0, Edge Areas: 3, Model: lm.
Comparison: cto with lm and gm.
Propagation delay: randomized 10-50ms range.
"""
from __future__ import annotations

import argparse
import copy
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import pandas as pd
from tqdm import tqdm

from environment import build_env_base
from method_policy import ConstantPolicy, make_baseline_policy
from eval_common import run_episode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
W2_RANGE = [0.5, 1.5, 2.5, 3.5, 4.0, 4.5, 5.5, 6.5]
DIRICHLET_ALPHAS = [0.1, 1.0, 100.0]
EPISODES = 10
BASE_CFG = "configs/simulation_ma_0.yaml"
OUTDIR   = "eval_out/weight_sweep_refined"

def randomize_delays(n_edges: int, min_ms: float = 10.0, max_ms: float = 50.0) -> List[List[float]]:
    """Generate a symmetric matrix with random delays in [min_ms, max_ms]."""
    mat = np.zeros((n_edges, n_edges))
    for i in range(n_edges):
        for j in range(i + 1, n_edges):
            d = np.random.uniform(min_ms, max_ms)
            mat[i, j] = mat[j, i] = d
    return mat.tolist()

def _build_env(cfg: dict, offload_mode: str, acc_model: str, w1: float, w2: float, w3: float, delays: List[List[float]]) -> object:
    patched = copy.deepcopy(cfg)
    patched["globals"]["offload_mode"] = offload_mode
    patched["globals"]["accuracy_matrix"]["model"] = acc_model
    patched["globals"]["offload_weights"] = {"w1": w1, "w2": w2, "w3": w3}
    patched["globals"]["delay_ms"] = delays
    
    # Ensure 3 edge areas
    if len(patched["edge_areas"]) > 3:
        patched["edge_areas"] = patched["edge_areas"][:3]
    elif len(patched["edge_areas"]) < 3:
        while len(patched["edge_areas"]) < 3:
            patched["edge_areas"].append(copy.deepcopy(patched["edge_areas"][-1]))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(patched, f)
        path = f.name
    try:
        return build_env_base(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

def main():
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    
    with open(BASE_CFG) as f:
        cfg_base = yaml.safe_load(f)
    
    base_seed = int(cfg_base["run"]["seed"])
    decision_interval = int(cfg_base["globals"]["decision_interval"])

    # Extract common eval params (using fixed config baseline for rewards)
    _reward_cfg = cfg_base["globals"].get("reward", {})
    reward_alpha_inv = float(_reward_cfg.get("alpha_inv", 10.0))
    reward_beta      = float(_reward_cfg.get("beta_inv",  5.0))
    reward_gamma     = float(_reward_cfg.get("gamma_inv", 8.33))
    reward_q_th      = float(_reward_cfg.get("q_th", 0.20))
    n_actions        = 9 
    scale_step       = 0.5 
    ids_cpu_min      = 0.5 
    
    sweep_results: Dict[float, Dict[str, List[float]]] = {a: {} for a in DIRICHLET_ALPHAS}
    
    baselines = [
        ("cto", "lm", "CTO (LM)"),
        ("cto", "gm", "CTO (GM)"),
    ]
    
    policy = ConstantPolicy(3.0)
    
    print(f"Sweeping FPR weights for Dirichlet Alphas: {DIRICHLET_ALPHAS}")
    print(f"W2 Range: {W2_RANGE}\n")

    # 1. Run Baselines
    for mode, model, label in baselines:
        print(f"Running Baseline: {label}")
        rewards_per_alpha = {a: [] for a in DIRICHLET_ALPHAS}
        
        for ep in tqdm(range(EPISODES), desc=f"  {label}"):
            seed = base_seed + (ep + 1) * 1000
            np.random.seed(seed)
            delays = randomize_delays(3)
            
            for alpha in DIRICHLET_ALPHAS:
                cfg_ep = copy.deepcopy(cfg_base)
                # Correct: patch dirichlet_alpha, NOT reward alpha
                cfg_ep["globals"]["attack_sampler"]["dirichlet_alpha"] = alpha
                
                env = _build_env(cfg_ep, mode, model, 1.0, 1.0, 0.01, delays)
                
                ep_res, _ = run_episode(
                    env=env,
                    cfg=cfg_ep,
                    policy=policy,
                    obs_keys=[],
                    decision_interval=decision_interval,
                    scale_step=scale_step,
                    n_actions=n_actions,
                    ids_cpu_min=ids_cpu_min,
                    seed=seed,
                    reward_alpha=reward_alpha_inv,
                    reward_beta=reward_beta,
                    reward_gamma=reward_gamma,
                    reward_q_th=reward_q_th,
                )
                rewards_per_alpha[alpha].append(np.mean(ep_res["reward"]))

        for alpha in DIRICHLET_ALPHAS:
            sweep_results[alpha][label] = rewards_per_alpha[alpha]

    # 2. Sweep W2 for CTO+Acc (LM)
    for w2 in W2_RANGE:
        label = f"CTO+Acc (w2={w2})"
        print(f"Running Sweep: {label}")
        rewards_per_alpha = {a: [] for a in DIRICHLET_ALPHAS}
        
        for ep in tqdm(range(EPISODES), desc=f"  {label}"):
            seed = base_seed + (ep + 1) * 1000
            np.random.seed(seed)
            delays = randomize_delays(3)
            
            for alpha in DIRICHLET_ALPHAS:
                cfg_ep = copy.deepcopy(cfg_base)
                cfg_ep["globals"]["attack_sampler"]["dirichlet_alpha"] = alpha
                
                # Use CTO+Acc, model LM, w1=0, w3=0.01
                env = _build_env(cfg_ep, "cto_acc", "lm", 0.0, w2, 0.01, delays)
                
                ep_res, _ = run_episode(
                    env=env,
                    cfg=cfg_ep,
                    policy=policy,
                    obs_keys=[],
                    decision_interval=decision_interval,
                    scale_step=scale_step,
                    n_actions=n_actions,
                    ids_cpu_min=ids_cpu_min,
                    seed=seed,
                    reward_alpha=reward_alpha_inv,
                    reward_beta=reward_beta,
                    reward_gamma=reward_gamma,
                    reward_q_th=reward_q_th,
                )
                rewards_per_alpha[alpha].append(np.mean(ep_res["reward"]))

        for alpha in DIRICHLET_ALPHAS:
            sweep_results[alpha][label] = rewards_per_alpha[alpha]

    # 3. Analyze and Print Results
    print("\n" + "="*80)
    print(f"{'D_Alpha':<10} {'Config':<20} {'Mean Reward':<15} {'Best?'}")
    print("-" * 80)
    
    summary_data = []

    for alpha in DIRICHLET_ALPHAS:
        alpha_data = sweep_results[alpha]
        alpha_rows = []
        for label, rewards in alpha_data.items():
            mean_r = np.mean(rewards)
            alpha_rows.append({"alpha": alpha, "label": label, "mean_reward": mean_r})
            summary_data.append(alpha_rows[-1])
        
        alpha_rows.sort(key=lambda x: x["mean_reward"], reverse=True)
        
        for i, row in enumerate(alpha_rows):
            is_best = "*" if i == 0 else ""
            print(f"{row['alpha']:<10} {row['label']:<20} {row['mean_reward']:>15.4f} {is_best}")
        print("-" * 80)

    df = pd.DataFrame(summary_data)
    df.to_csv(Path(OUTDIR) / "sweep_results_dirichlet.csv", index=False)
    print(f"\nResults saved to {OUTDIR}/sweep_results_dirichlet.csv")

if __name__ == "__main__":
    main()
