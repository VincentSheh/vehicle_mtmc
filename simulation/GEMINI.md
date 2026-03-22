# Edge Computing Simulation for IDS and Task Offloading

This project is a high-fidelity simulation environment for edge computing scenarios, specifically focusing on the interaction between application performance (Video Analytics), security (Intrusion Detection Systems - IDS), and resource management (Task Offloading). It utilizes Reinforcement Learning (RL) to optimize resource allocation and mitigation strategies under varying attack traces.

## Project Overview

- **Core Purpose:** Simulate an edge computing node or network where resources (CPU, Memory, Uplink) must be split between processing benign application workloads (like video analytics) and defending against DDoS attacks using an IDS.
- **Main Technologies:**
    - **Language:** Python
    - **RL Framework:** `torchrl`, `tensordict` (PyTorch ecosystem)
    - **Data Handling:** `pandas`, `numpy`
    - **Logging:** `wandb` (Weights & Biases)
    - **Simulation Components:** Custom environments for edge areas, attackers, users, and video pipelines.

## Architecture

The project is structured into several modular components:
- `environment.py`: Defines the `TorchRLEnvWrapper` and the core simulation logic.
- `edgearea.py`: Manages the resource budget, application/IDS split, and cooperative offloading logic.
- `request.py`: Models benign `User` requests and various `Attacker` patterns (trace-based or synthetic like sinus/pulse).
- `service.py`: Implements the `IDS` and `VideoPipeline` models, calculating processing latencies and accuracy (mAP, TPR/FPR).
- `policy.py`: Contains policy wrappers for MLP and LSTM models to be used during evaluation.
- `train.py`, `train_lstm.py`, `train_mlp.py`: Entry points for training RL agents.

## Getting Started

### Environment Setup

You can set up the environment using the provided `environment.yml` or `requirements.txt`.

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate edgeids
```

**Using Pip:**
```bash
pip install -r requirements.txt
```

### Configuration

Configurations are managed via YAML files in the `configs/` directory:
- `configs/simulation_0.yaml`: Defines global parameters like CPU clock, IDS latency, attack types, and edge area budgets.
- `configs/train.yaml`: Contains RL hyperparameters for PPO, optimizer settings, and logger configuration.

### Running the Project

**Training:**
To start training an agent (e.g., using the default PPO setup):
```bash
python train.py
```
*Note: Ensure you have adjusted the `exp_name` and `project_name` in `configs/train.yaml` if using wandb.*

**Evaluation/Visualization:**
- Use `visualize.ipynb` for analyzing results.
- `policy.py` can be used to load checkpoints and evaluate them against specific scenarios.

## Development Conventions

- **Simulation Steps:** The environment operates in discrete time slots (e.g., `slot_ms`).
- **Observations:** Includes metrics like CPU utilization, IDS coverage, current QoE, and attack intensity estimates.
- **Actions:** Typically discrete adjustments to resource allocation or mitigation levels (e.g., scale up/down/hold IDS CPU).
- **Rewards:** Calculated based on a combination of application QoE (Quality of Experience) and successful attack mitigation, often with penalties for SLO violations.

## Key Files Summary

| File | Description |
| --- | --- |
| `environment.py` | The main TorchRL environment wrapper. |
| `edgearea.py` | Logic for resource management and offloading. |
| `request.py` | User and Attacker workload generation. |
| `service.py` | Models for IDS and Video Analytics. |
| `train.py` | Main training script using PPO. |
| `configs/` | Directory containing simulation and training parameters. |
| `input/` | Contains trace data for benign and attack traffic. |
| `checkpoints/` | Directory where trained models are saved. |
