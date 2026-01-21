# --- add near imports ---
import os
import wandb
import pandas as pd
import torch
import numpy as np


def wandb_init(cfg: dict, cfg_path: str):
    run = wandb.init(
        entity="asture123-national-taiwan-university",
        project="edgeids",
        name=f"ppo_{os.path.splitext(os.path.basename(cfg_path))[0]}",
        config={
            "cfg_path": cfg_path,
            "decision_interval": cfg.get("run", {}).get("decision_interval", None),
            "t_max": cfg["run"]["t_max"],
            "seed": cfg["run"]["seed"],
        },
    )
    wandb.define_metric("iter")
    wandb.define_metric("loss/*", step_metric="iter")
    wandb.define_metric("reward/*", step_metric="iter")

    wandb.define_metric("ts_step")
    wandb.define_metric("ts/*", step_metric="ts_step")
    return run

def wandb_log_env_block(env, step: int, decision_interval: int):
    # log last decision block metrics aggregated per edge
    if not getattr(env, "history", None):
        return

    records = env.history[-decision_interval * len(env.edge_areas):]
    if not records:
        return

    df = pd.DataFrame([m.__dict__ for m in records])

    # global rollups
    wandb.log(
        {
            "qoe/mean": float(df["qoe_mean"].mean()),
            "qoe/min": float(df["qoe_min"].min()),
            "ids/coverage_mean": float(df["ids_coverage"].mean()),
            "attack/in_rate_mean": float(df["attack_in_rate"].mean()),
            "attack/drop_rate_mean": float(df["attack_drop_rate"].mean()),
            "user/drop_rate_mean": float(df["user_drop_rate"].mean()),
            "cpu/ids_util_mean": float(df["ids_cpu_utilization"].mean()),
            "cpu/va_util_mean": float(df["va_cpu_utilization"].mean()),
            "bw/util_mean": float(df["bw_utilization"].mean()),
            "offload/I_net_mean": float(df["I_net"].mean()),
        },
        step=step,
    )

    # per-edge rollups
    for area_id, g in df.groupby("area_id"):
        wandb.log(
            {
                f"{area_id}/qoe_mean": float(g["qoe_mean"].mean()),
                f"{area_id}/ids_coverage": float(g["ids_coverage"].mean()),
                f"{area_id}/attack_in_rate": float(g["attack_in_rate"].mean()),
                f"{area_id}/cpu_to_ids_ratio": float(g["cpu_to_ids_ratio"].iloc[-1]),
                f"{area_id}/va_cpu_util": float(g["va_cpu_utilization"].mean()),
                f"{area_id}/ids_cpu_util": float(g["ids_cpu_utilization"].mean()),
                f"{area_id}/bw_util": float(g["bw_utilization"].mean()),
                f"{area_id}/I_net": float(g["I_net"].mean()),
                f"{area_id}/num_objects": float(g["num_objects"].mean()),
            },
            step=step,
        )

def wandb_save_plots_from_history(env, out_dir="logs/wandb_plots"):
    os.makedirs(out_dir, exist_ok=True)
    if not getattr(env, "history", None):
        return

    df = pd.DataFrame([m.__dict__ for m in env.history])
    if df.empty:
        return

    paths = []

    def save_pivot(title, col, fname):
        fig = (
            df.pivot(index="t", columns="area_id", values=col)
            .plot(figsize=(10, 4), title=title)
            .get_figure()
        )
        p = os.path.join(out_dir, fname)
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)

    save_pivot("QoE over time", "qoe_mean", "qoe_over_time.png")
    save_pivot("IDS coverage", "ids_coverage", "ids_coverage.png")
    save_pivot("Attack In Rate", "attack_in_rate", "attack_in_rate.png")
    save_pivot("Post-offload tracking load", "num_objects", "num_objects.png")
    save_pivot("Available uplink", "uplink_available", "uplink_available.png")

    # log as images + artifact
    for p in paths:
        wandb.log({os.path.basename(p): wandb.Image(p)})

    art = wandb.Artifact("plots", type="evaluation")
    for p in paths:
        art.add_file(p)
    wandb.log_artifact(art)
    
    
    # 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
    
def wandb_log_iteration_timeseries(it: int, batch, obs_ts: dict):
    """
    obs_ts: dict of numpy or torch arrays shaped [T] or [T, n_edges]
      keys like:
        "local_num_objects", "attack_drop_rate", "cpu_to_ids_ratio",
        "va_cpu_utilization", "ids_cpu_utilization", "bw_utilization", "I_net"
    """
    T = batch.batch_size[0]  # decision timesteps in this iteration
    actions = batch["action"].detach().cpu().numpy()  # shape [T] or [T, n_edges]

    table = wandb.Table(columns=[
        "iter", "t", "global_step", "edge",
        "action",
        "local_num_objects",
        "attack_drop_rate",
        "cpu_to_ids_ratio",
        "va_cpu_utilization",
        "ids_cpu_utilization",
        "bw_utilization",
        "I_net",
    ])

    n_edges = actions.shape[1] if actions.ndim == 2 else 1
    
    print(obs_ts)

    for t in range(T):
        for e in range(n_edges):
            a = actions[t, e] if actions.ndim == 2 else actions[t]

            def pick(x):
                if x is None:
                    return None
                x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
                return float(x[t, e]) if (np.ndim(x) == 2) else float(x[t])

            row = [
                it,
                t,
                it * T + t,
                e,
                float(a),
                pick(obs_ts.get("local_num_objects")),
                pick(obs_ts.get("attack_drop_rate")),
                pick(obs_ts.get("cpu_to_ids_ratio")),
                pick(obs_ts.get("va_cpu_utilization")),
                pick(obs_ts.get("ids_cpu_utilization")),
                pick(obs_ts.get("bw_utilization")),
                pick(obs_ts.get("I_net")),
            ]
            table.add_data(*row)

    wandb.log({"timeseries/step_table": table}, step=it)    
    
def wandb_log_batch_transitions(batch, obs_keys, step, tag="batch"):
    # batch keys typically: "observation", "action", ("next","reward"), ("next","done"), ...
    obs = batch["observation"].detach().cpu()          # [T, n_edges, obs_dim] or [T, B, n_edges, obs_dim]
    act = batch["action"].detach().cpu()               # [T] or [T, B]
    rew = batch["next", "reward"].detach().cpu()       # [T, 1] or [T, B, 1]
    done = batch["next", "done"].detach().cpu()        # [T, 1] or [T, B, 1]
    term = batch["next", "terminated"].detach().cpu()  # [T, 1] or [T, B, 1]

    # Squeeze batch dim if B=1, keep code robust
    if obs.ndim == 4:  # [T, B, E, D]
        obs = obs[:, 0]
    if act.ndim == 2:
        act = act[:, 0]
    if rew.ndim == 3:
        rew = rew[:, 0]
    if done.ndim == 3:
        done = done[:, 0]
    if term.ndim == 3:
        term = term[:, 0]

    T, E, D = obs.shape

    columns = ["iter", "t", "edge", "action", "reward", "done", "terminated"] + obs_keys
    table = wandb.Table(columns=columns)

    for t in range(T):
        a = int(act[t].item())
        r = float(rew[t].squeeze(-1).item())
        d = bool(done[t].squeeze(-1).item())
        te = bool(term[t].squeeze(-1).item())

        for e in range(E):
            row = [int(step), int(t), int(e), a, r, d, te]
            row += [float(obs[t, e, j].item()) for j in range(D)]
            table.add_data(*row)

    wandb.log({f"{tag}/transitions": table}, step=step)    