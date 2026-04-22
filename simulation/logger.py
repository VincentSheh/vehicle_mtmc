import os
import json
import wandb
import pandas as pd
import torch
import numpy as np


def wandb_init(env_cfg: dict, train_cfg: dict):
    logger_cfg    = train_cfg.get("logger", {}) or {}
    collector_cfg = train_cfg.get("collector", {}) or {}
    optim_cfg     = train_cfg.get("optim", {}) or {}
    loss_cfg      = train_cfg.get("loss", {}) or {}
    model_cfg     = train_cfg.get("model", {}) or {}
    obs_norm_cfg  = train_cfg.get("observation_norm", {}) or {}

    globals_cfg  = env_cfg.get("globals", {}) or {}
    sampler_cfg  = globals_cfg.get("attack_sampler", {}) or {}
    reward_cfg   = globals_cfg.get("reward", {}) or {}

    exp_name = logger_cfg.get("exp_name", None)
    cfg_path = train_cfg.get("cfg_path", None)
    if exp_name is None:
        exp_name = f"ppo_{os.path.splitext(os.path.basename(cfg_path))[0]}" if cfg_path else "ppo_run"

    run = wandb.init(
        entity=logger_cfg.get("entity", "asture123-national-taiwan-university"),
        project=logger_cfg.get("project_name", "multiedgeids"),
        name=exp_name,
        config={
            "env_cfg": json.dumps(env_cfg),

            "env/t_max":             env_cfg.get("run", {}).get("t_max", None),
            "env/seed":              env_cfg.get("run", {}).get("seed", None),
            "env/decision_interval": globals_cfg.get("decision_interval", None),
            "env/num_envs":          int(collector_cfg.get("num_envs", 1)),

            "reward/alpha_inv": reward_cfg.get("alpha_inv", 0.10),
            "reward/beta_inv":  reward_cfg.get("beta_inv",  0.20),
            "reward/gamma_inv": reward_cfg.get("gamma_inv", 0.12),
            "reward/q_th":      reward_cfg.get("q_th",      0.20),

            "attack/pattern_types": str(sampler_cfg.get("pattern_types", None)),
            "attack/lambda_base":   str(sampler_cfg.get("lambda_base",   None)),
            "attack/noise_std":     str(sampler_cfg.get("noise_std",     None)),
            "attack/t_cycle_min":   str(sampler_cfg.get("t_cycle_min",   None)),
            "attack/t_cycle_delta": str(sampler_cfg.get("t_cycle_delta", None)),

            "optim/lr":            optim_cfg.get("lr", None),
            "optim/eps":           optim_cfg.get("eps", None),
            "optim/weight_decay":  optim_cfg.get("weight_decay", None),
            "optim/max_grad_norm": optim_cfg.get("max_grad_norm", None),
            "optim/anneal_lr":     optim_cfg.get("anneal_lr", None),

            "loss/gamma":               loss_cfg.get("gamma", None),
            "loss/gae_lambda":          loss_cfg.get("gae_lambda", None),
            "loss/ppo_epochs":          loss_cfg.get("ppo_epochs", None),
            "loss/mini_batch_size":     loss_cfg.get("mini_batch_size", None),
            "loss/clip_epsilon":        loss_cfg.get("clip_epsilon", None),
            "loss/anneal_clip_epsilon": loss_cfg.get("anneal_clip_epsilon", None),
            "loss/entropy_coeff":       loss_cfg.get("entropy_coeff", None),
            "loss/critic_coeff":        loss_cfg.get("critic_coeff", None),
            "loss/loss_critic_type":    loss_cfg.get("loss_critic_type", None),
            "loss/seq_len":             loss_cfg.get("seq_len", None),
            "loss/normalize_advantage": loss_cfg.get("normalize_advantage", None),

            "collector/frames_per_batch": collector_cfg.get("frames_per_batch", None),
            "collector/total_frames":     collector_cfg.get("total_frames", None),
            "collector/trust_policy":     collector_cfg.get("trust_policy", None),

            "model/hidden_dim": model_cfg.get("hidden_dim", None),
            "model/n_actions":  model_cfg.get("n_actions", None),

            "obs_norm/standard_normal": obs_norm_cfg.get("standard_normal", None),
        },
    )

    wandb.define_metric("iter")
    wandb.define_metric("loss/*",   step_metric="iter")
    wandb.define_metric("reward/*", step_metric="iter")
    wandb.define_metric("qoe/*",    step_metric="iter")
    wandb.define_metric("attack/*", step_metric="iter")
    wandb.define_metric("decision_step")
    wandb.define_metric("obs/*",    step_metric="decision_step")
    wandb.define_metric("ts_step")
    wandb.define_metric("ts/*",     step_metric="ts_step")
    return run


def wandb_log_obs_steps(
    obs,
    obs_keys: list,
    keep_keys,
    global_step_start: int,
) -> int:
    """
    Log per-decision-step observations filtered to keep_keys.

    obs: Tensor or ndarray [B,T,E,D] or [T,E,D].
         B (env) dimension is averaged before logging.
    Returns the updated global_step (caller should assign back).
    """
    if torch.is_tensor(obs):
        obs = obs.detach().cpu().numpy()
    if obs.ndim == 4:
        obs = obs.mean(axis=0)   # [T, E, D]
    if obs.ndim != 3:
        raise ValueError(f"Expected obs [T,E,D] after averaging, got {obs.shape}")

    keep = set(keep_keys) if not isinstance(keep_keys, set) else keep_keys
    key_indices = [(j, name) for j, name in enumerate(obs_keys) if name in keep]

    T, E, _ = obs.shape
    for t in range(T):
        step_log = {"decision_step": global_step_start + t}
        for j, name in key_indices:
            for e in range(E):
                step_log[f"obs/edge_{e}/{name}"] = float(obs[t, e, j])
            step_log[f"obs/{name}"] = float(obs[t, :, j].mean())
        wandb.log(step_log)

    return global_step_start + T


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

    save_pivot("QoE over time",              "qoe_mean",         "qoe_over_time.png")
    save_pivot("IDS coverage",               "ids_coverage",     "ids_coverage.png")
    save_pivot("Attack In Rate",             "attack_in_rate",   "attack_in_rate.png")
    save_pivot("Post-offload tracking load", "num_objects",      "num_objects.png")
    save_pivot("Available uplink",           "uplink_available", "uplink_available.png")

    for p in paths:
        wandb.log({os.path.basename(p): wandb.Image(p)})

    art = wandb.Artifact("plots", type="evaluation")
    for p in paths:
        art.add_file(p)
    wandb.log_artifact(art)
