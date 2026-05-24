
import os
import yaml
import json
import numpy as np
from environment import build_env_from_cfg
from method_policy import make_baseline_policy

def run_eval(offload_mode, acc_model, episodes=3):
    with open("configs/simulation_ma_eval_v1.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    cfg["globals"]["offload_mode"] = offload_mode
    cfg["globals"]["accuracy_matrix"]["model"] = acc_model
    cfg["eval"]["episode"] = episodes
    cfg["run"]["t_max"] = 2000 # Short run for speed
    
    # Ensure logs are fresh
    if os.path.exists("pd_bto_instrumentation.log"):
        os.remove("pd_bto_instrumentation.log")
        
    print(f"\nEvaluating {offload_mode} with {acc_model}...")
    env = build_env_from_cfg(cfg)
    
    # Action: fixed CPU allocation for all edges
    n_edges = len(env.edge_areas)
    ids_cpus = [2.0] * n_edges
    
    total_qoe = []
    
    for ep in range(episodes):
        env.reset(seed=cfg["run"]["seed"] + ep)
        done = False
        while not done:
            env.step(ids_cpus)
            if env.t >= env.t_max:
                done = True
        
        # Calculate mean QoE from history for this episode
        ep_history = env.history[-env.t_max * n_edges:]
        ep_qoe = np.mean([float(m.qoe_mean) for m in ep_history])
        total_qoe.append(ep_qoe)
        print(f"  Episode {ep}: mean_qoe = {ep_qoe:.4f}")
    
    avg_qoe = np.mean(total_qoe)
    print(f"Average QoE: {avg_qoe:.4f}")
    
    stats_summary = {}
    if os.path.exists("pd_bto_instrumentation.log"):
        with open("pd_bto_instrumentation.log", "r") as f:
            lines = f.readlines()
            for line in lines:
                entry = json.loads(line)
                for k, v in entry.items():
                    stats_summary[k] = stats_summary.get(k, 0) + v
        
        # Average per Stage B call (assuming one Stage B call per step? No, once per decision interval)
        n_calls = len(lines)
        if n_calls > 0:
            for k in stats_summary:
                stats_summary[k] /= n_calls
            print(f"PD_BTO Stats (per call): {json.dumps(stats_summary, indent=2)}")
    
    return avg_reward, stats_summary

if __name__ == "__main__":
    r_cto, _ = run_eval("cto_acc", "gm", episodes=2)
    r_pd, s_pd = run_eval("pd_bto", "gm", episodes=2)
    
    print("\n" + "="*40)
    print(f"CTO_ACC Reward: {r_cto:.2f}")
    print(f"PD_BTO Reward:  {r_pd:.2f}")
    print(f"Gap:            {r_cto - r_pd:.2f}")
    print("="*40)
