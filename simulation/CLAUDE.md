# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**
```bash
conda env create -f environment.yml && conda activate edgeids
# or
pip install -r requirements.txt
```

**Training:**
```bash
python train.py                          # PPO + LSTM (main)
python train_lstm.py                     # explicit LSTM variant
python train_mlp.py                      # MLP-only variant
python train.py --resume_ckpt checkpoints/<run>/ckpt_iter_XXXXXX.pt
```

**Evaluation:**
```bash
python policy.py --cfg configs/simulation_0.yaml --outdir eval_out --episodes 10
```

**Smoke-test the environment (reactive policy, 1 episode):**
```bash
python environment.py
```

**Snapshot/restore unit test:**
```bash
python test_snapshot.py
```

## Architecture

The simulation models a single edge node (or multi-edge network) that splits CPU between a **Video Analytics (VA)** pipeline and an **IDS**. An RL agent learns to adjust the IDS CPU allocation in response to DDoS attacks while preserving application QoE.

### Data flow

```
train.py
└── TorchRLEnvWrapper (environment.py)
    └── Environment (environment.py)
        └── EdgeArea (edgearea.py)
            ├── IDS (service.py)          — detects/drops attack flows
            ├── VideoPipeline (service.py) — tracks objects, returns mAP-based QoE
            ├── User (request.py)          — benign request arrival (trace or synthetic OU)
            └── Attacker (request.py)      — attack flow arrival (trace or synthetic: yoyo/pw/sinus)
```

**`Environment`** owns the simulation clock and history. Its `step(ids_cpus)` runs two passes per tick: one with attacks disabled (ideal baseline for reward shaping) and one real pass, then appends a `StepMetrics` record per edge.

**`TorchRLEnvWrapper`** bridges `Environment` to TorchRL. Each RL *action* (scale-down / hold / scale-up) covers `decision_interval` simulation ticks. Observation is a `(n_edges × obs_dim)` tensor aggregated over the last decision window.

**`policy.py`** is evaluation-only. It contains `MLPPolicy` / `LSTMPolicy` wrappers that load checkpoints, plus `run_episode()` which supports methods: `"reactive"`, `"random"`, `"constant_X"`, `"rl"`.

### Reward

Reward per edge per step:
```
- α * λ_res        (attack residual rate)
- β * benign_col_dmg  (QoE loss vs ideal)
- γ * qoe_shortfall   (QoE below SLO threshold)
```
Final episode QoE is penalised by `exp(-β * violation_rate)` for SLO violations.

### Key configuration knobs (`configs/simulation_0.yaml`)

- `globals.decision_interval` — ticks between RL decisions (default 500)
- `globals.fps` — simulation slot rate (default 5 → slot_ms = 200 ms)
- `edge_areas[].budget.cpu` — total CPU cores available (default 8)
- `edge_areas[].attackers` — which attack profiles are active (comment/uncomment)
- Attack `pattern_type`: `"trace"` | `"yoyo"` | `"pw"` (pulse-wave) | `"sinus"`

### Checkpoints

Saved under `checkpoints/<exp_name>/ckpt_iter_XXXXXX.pt`. Each checkpoint stores `policy`, `critic`, `obsnorm`, and `train_cfg` keys.
