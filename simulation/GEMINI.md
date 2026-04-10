# EdgeIDS: Edge Computing Simulation for IDS and Task Offloading

This project is a high-fidelity simulation environment designed to study the trade-offs between application performance (Video Analytics) and security (Intrusion Detection Systems) in edge computing nodes. It employs Reinforcement Learning (RL) to optimize resource allocation in the presence of various DDoS attack patterns.

## Project Overview

- **Core Purpose:** Simulate a single or multi-edge node environment where a fixed CPU budget must be split between processing benign video analytics workloads and defending against network attacks via an IDS.
- **Main Technologies:**
    - **Language:** Python 3.x
    - **RL Framework:** `torchrl`, `tensordict` (PyTorch ecosystem)
    - **Data Handling:** `pandas`, `numpy`
    - **Logging:** `wandb` (Weights & Biases)
    - **Configuration:** YAML-based parameters

## Architecture

The simulation is built on a modular hierarchy:

1.  **`Environment` (`environment.py`):** Manages the global simulation clock, history, and state transitions. It provides a `TorchRLEnvWrapper` to interface with the `torchrl` library.
2.  **`EdgeArea` (`edgearea.py`):** The primary unit of simulation. It handles:
    - **Resource Management:** Splitting the CPU budget between IDS and VA.
    - **VA Pipeline:** Dynamic selection of object detection models (e.g., NanoDet) and resolutions based on available cycles.
    - **Offloading:** Cooperative task offloading between nodes (if configured).
3.  **`IDS` (`service.py`):** A cycle-based model that filters incoming traffic. Its effectiveness (coverage) depends on the assigned CPU and the incoming packet rate.
4.  **`VideoPipeline` (`service.py`):** Models the computation cost (cycles) and accuracy (mAP) of various detection and ReID tasks.
5.  **`Request` (`request.py`):** Models benign `User` traffic (synthetic or trace-based) and `Attacker` traffic with various patterns (`sinus`, `pulse`, `yoyo`, `trace`).

### Data Flow
```
RL Action (IDS CPU change) -> Environment Step -> EdgeArea Resource Split 
-> IDS Filtering -> VA Workload Processing -> Reward Calculation (QoE + Security)
```

## Building and Running

### Environment Setup
```bash
# Using Conda
conda env create -f environment.yml && conda activate edgeids

# Using Pip
pip install -r requirements.txt
```

### Training
The project uses PPO with optional LSTM support for temporal state.
```bash
python train_lstm.py   # Main training script (LSTM variant)
python train_mlp.py    # MLP-only variant
```

### Evaluation & Testing
```bash
# Evaluate a trained policy
python policy.py --cfg configs/simulation_0.yaml --episodes 10

# Smoke-test the environment logic (reactive policy)
python environment.py

# Run unit tests for state snapshot/restore
python test_snapshot.py
```

## Development Conventions

- **Simulation Steps:** Discrete time slots (default `slot_ms = 200ms`, derived from `fps=5`).
- **Decision Interval:** RL agent makes decisions every `N` steps (default 300-500).
- **Observations:** Local requests, attack rate, IDS utilization, and current CPU allocation ratio.
- **Actions:** Discrete shifts in CPU allocation (e.g., `{-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}`).
- **Reward Function:** A weighted sum of:
    - `qoe_penalty`: Penalty for dropping benign frames or low mAP.
    - `lambda_res`: Penalty for allowed attack traffic (residual rate).
    - `benign_col_dmg`: Penalty for loss of QoE compared to an ideal (attack-free) scenario.
- **SLO Enforcement:** Final rewards are often penalized by a violation rate if QoE falls below a threshold.

## Key Configuration Files

- `configs/simulation_0.yaml`: Global simulation parameters, hardware specs, and attack profiles.
- `configs/train.yaml`: RL hyperparameters, optimizer settings, and logging configuration.
