# Drone Hunter

An autonomous drone defense system using reinforcement learning. The AI agent learns to identify and shoot down incoming drones, prioritizing kamikaze threats that fly directly toward the camera.

**Key Achievement**: The trained agent achieves +91.66 reward, exceeding both human-level performance (~60 points) and the oracle baseline (+80.9).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Core Concepts](#core-concepts)
4. [Environment Design](#environment-design)
5. [Development Journey](#development-journey)
6. [Codebase Structure](#codebase-structure)
7. [Usage Guide](#usage-guide)
8. [Key Takeaways](#key-takeaways)

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run quick demo (AI agent playing)
uv run python main.py

# Play yourself
uv run python -m drone_hunter.scripts.play

# Evaluate best model
uv run python -m drone_hunter.scripts.evaluate \
  runs/detector_v2/best_model/best_model.zip \
  --single-target --detector-mode --episodes 5 --fps 15
```

---

## Project Overview

### The Problem

Drones appear on screen and move toward the camera. Some are harmless (just flying around), but **kamikaze drones** fly directly at you. If a kamikaze reaches the camera, you lose. Your goal: survive 1000 frames by shooting threats before they reach you.

### The Solution

Train a reinforcement learning agent using **PPO (Proximal Policy Optimization)** to:
1. Observe drone positions and velocities
2. Decide which grid cell to shoot (or wait)
3. Prioritize kamikaze threats over normal drones
4. Manage ammo (10 shots per clip, 30-frame reload)

### Why This Is Interesting

The agent must operate with **realistic sensor data**. Real object detectors only provide 2D bounding boxes, not depth or velocity. We use a **Kalman filter** to estimate these hidden values from noisy observations. Remarkably, the agent trained on these estimates **outperforms** the one trained with perfect information.

---

## Core Concepts

### Reinforcement Learning (RL)

RL is learning by trial and error. An **agent** takes **actions** in an **environment**, receives **rewards**, and learns a **policy** (strategy) to maximize total reward.

```
┌─────────────────────────────────────────────┐
│                                             │
│   Agent ──action──> Environment             │
│     ^                    │                  │
│     └──reward, state─────┘                  │
│                                             │
└─────────────────────────────────────────────┘
```

In Drone Hunter:
- **Agent**: Neural network that decides what to shoot
- **Environment**: Game with drones, ammo, and scoring
- **Actions**: Shoot at one of 64 grid cells, or wait
- **Rewards**: +1 for normal kill, +2 for kamikaze kill, -5 for getting hit

### PPO (Proximal Policy Optimization)

PPO is a popular RL algorithm that balances exploration (trying new things) with exploitation (doing what works). It's stable, sample-efficient, and works well with neural networks.

**Key hyperparameters used:**

| Parameter | Value | What it does |
|-----------|-------|--------------|
| Learning rate | 3e-4 | How fast the network updates |
| Rollout steps | 2048 | Frames collected before each update |
| Epochs | 10 | Training passes per update |
| Clip range | 0.2 | Limits how much policy can change |
| GAE Lambda | 0.95 | Balances bias/variance in advantage estimation |
| Gamma | 0.99 | How much to value future rewards |

**Network architecture**: Two hidden layers of 256 neurons each, using ReLU activation.

### Kalman Filter

The Kalman filter estimates hidden state from noisy measurements. Think of it as "smart averaging" that understands physics.

**The problem**: Object detectors give us bounding boxes (x, y, width, height), but we need depth (z) and velocity (vz) to prioritize threats.

**The solution**:
1. Estimate depth from bounding box size (bigger = closer)
2. Track depth over time to estimate velocity
3. Use physics model (constant velocity) to smooth estimates

```
State: [z, vz]  (depth and depth-velocity)

Prediction step:
  z_predicted = z + vz * dt
  vz_predicted = vz

Update step:
  z_measured = reference_z * (reference_height / bbox_height)
  z_new = weighted_average(z_predicted, z_measured)
```

The Kalman filter's key insight: combine predictions (what physics says should happen) with measurements (what we observe) based on their respective uncertainties.

### Hungarian Algorithm

When tracking multiple drones across frames, we need to match "which detection in frame N+1 corresponds to which detection in frame N." The Hungarian algorithm finds the optimal assignment by minimizing total cost (1 - IoU for each pair).

### Object Detection (NanoDet)

NanoDet is a lightweight object detector designed for mobile/edge deployment. We fine-tuned it on drone images for real-world detection.

- **Architecture**: ShuffleNetV2 backbone + GhostPAN neck
- **Input**: 320x320 pixels
- **Output**: Bounding boxes with confidence scores
- **Export**: ONNX format for deployment

---

## Environment Design

### Game Mechanics

**Drone Types:**

| Type | Spawn Rate | Behavior | Points |
|------|------------|----------|--------|
| Normal | 70% | Random movement, slight approach | +1 |
| Kamikaze | 20% | Flies directly at camera center | +2 |
| Erratic | 10% | Changes direction every 15 frames | +1 |

**Ammo System:**
- Clip size: 10 shots
- Reload time: 30 frames (~1 second)
- Cannot fire while reloading

**Win/Lose:**
- Win: Survive 1000 frames (+3 bonus)
- Lose: Kamikaze reaches camera (z < 0.08) (-5 penalty)

### Action Space

The screen is divided into an 8x8 grid. Each action fires at one cell or waits:

```
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 8 │ 9 │...│   │   │   │   │15 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
│   │   │   │   │   │   │   │   │
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│56 │57 │...│   │   │   │   │63 │
└───┴───┴───┴───┴───┴───┴───┴───┘

Action 0: Wait (don't fire)
Actions 1-64: Fire at grid cell (action - 1)
Total: 65 actions
```

### Observation Space

The agent sees the **most urgent target** plus game state:

**Target Features (19 values):**

| Feature | Description |
|---------|-------------|
| z | Depth (0=camera, 1=far) |
| vz * 10 | Velocity scaled for gradient flow |
| urgency | 1 / (1 + frames_to_impact/50) |
| grid_x one-hot | 8 values indicating x position |
| grid_y one-hot | 8 values indicating y position |

**Game State (5 values):**

| Feature | Description |
|---------|-------------|
| ammo_fraction | Current ammo / 10 |
| reload_fraction | Reload progress (0 if not reloading) |
| frame_fraction | Current frame / 1000 |
| threat_level | Max threat from approaching drones |
| has_target | 1 if target exists, 0 otherwise |

**Total: 24 features** (simple enough for a small MLP)

### Reward Function

```
Hit normal drone:     +1.0
Hit kamikaze:         +2.0
Miss (fire, no hit):  -0.1
Survival per frame:   +0.01
Empty clip + threat:  -0.5
Kamikaze impact:      -5.0
Episode complete:     +3.0
```

---

## Development Journey

This project evolved through systematic ablation studies. Each experiment tested a specific hypothesis.

### Experiment 1: Multi-Target Failure

**Hypothesis**: Give the agent all drone information (10 drones x 4 frames x 22 features = 885 inputs).

**Result**: -7.17 reward (complete failure)

**Lesson**: MLPs cannot solve attention-like problems. The agent couldn't figure out which drone to focus on among 40 candidates.

### Experiment 2: Single-Target Success

**Hypothesis**: Pre-select the most urgent target using a simple heuristic (closest approaching drone).

**Result**: +82.2 reward (human-level performance!)

**Lesson**: Move complexity from the policy to the observation. The agent only needs to decide *where* to shoot, not *which drone* to track.

```
Observation reduction: 885 → 25 features
Performance:           -7.17 → +82.2 reward
```

### Experiment 3: Removing Ground Truth Labels

**Hypothesis**: The agent shouldn't need the `is_kamikaze` label. It can infer threat from velocity (vz < 0 means approaching).

**Version 1**: Removed from observation, kept in threat_level computation → +62.2

**Version 2**: Fully removed from all computations → +80.9

**Lesson**: Ground truth labels can be harmful! The agent learns better when it must infer threat from observable behavior, matching real-world deployment conditions.

### Experiment 4: Kalman Filter Integration

**Hypothesis**: Train with estimated (z, vz) from Kalman filter instead of oracle ground truth.

**detector_v1** (500k steps): +67.9 (84% of oracle)

**detector_v2** (700k steps): **+91.66** (beat the oracle!)

**Key Finding**: Extended training with Kalman-filtered estimates produces a *better* policy than training with perfect information. The temporal smoothing from the Kalman filter may help the agent learn more robust behavior.

### Results Summary

| Experiment | Reward | Key Finding |
|------------|--------|-------------|
| oracle_multitarget | -7.17 | Multi-target fails with MLPs |
| oracle_single_target_v1 | +82.2 | Single-target works |
| oracle_no_kamikaze_v2 | +80.9 | No ground truth needed |
| detector_v1 | +67.9 | Kalman filter works |
| **detector_v2** | **+91.66** | Extended training beats oracle! |

---

## Codebase Structure

```
drone_hunter/
├── src/drone_hunter/              # Main training codebase
│   ├── envs/                      # Gymnasium environment
│   │   ├── drone_hunter_env.py    # RL environment wrapper
│   │   ├── game_state.py          # Core game logic
│   │   └── difficulty.py          # Visual presets
│   ├── tracking/                  # State estimation
│   │   └── kalman_tracker.py      # Kalman filter + Hungarian
│   ├── detection/                 # Object detection
│   │   └── nanodet_onnx.py        # NanoDet inference wrapper
│   └── scripts/                   # Entry points
│       ├── train.py               # PPO training
│       ├── evaluate.py            # Model evaluation
│       ├── play.py                # Human play mode
│       └── generate_dataset.py    # Dataset generation
│
├── drone_hunter_edge/             # Edge deployment code
│   ├── core/                      # Optimized game logic
│   ├── inference/                 # ONNX/NCNN backends
│   │   ├── policy.py              # Policy inference
│   │   ├── nanodet.py             # Detector inference
│   │   ├── onnx_backend.py        # ONNX Runtime
│   │   └── ncnn_backend.py        # NCNN (mobile)
│   └── termux/                    # Android deployment
│       ├── run_live.py            # Live camera
│       └── run_simulation.py      # Simulation
│
├── runs/                          # Trained models (407MB)
│   ├── oracle_single_target_v1/   # Baseline
│   ├── oracle_no_kamikaze_v2/     # No ground truth
│   ├── detector_v1/               # First Kalman
│   └── detector_v2/               # Best model!
│
├── models/                        # Pre-trained detectors
│   ├── nanodet.onnx               # Base model
│   ├── nanodet_drone.onnx         # Fine-tuned v1
│   └── nanodet_drone_v2.onnx      # Fine-tuned v2
│
├── data/                          # Training datasets
│   ├── drone_detection/           # Dataset v1
│   └── drone_detection_v2/        # Dataset v2
│
├── configs/                       # Configuration files
│   └── nanodet_drone_finetune.yml # Detector training config
│
├── docs/                          # Documentation
│   └── training_results.md        # Detailed experiment logs
│
├── main.py                        # Quick demo
└── pyproject.toml                 # Package configuration
```

---

## Usage Guide

### Installation

```bash
cd drone_hunter
uv sync
```

### Running the Demo

```bash
# Watch the best AI agent play
uv run python main.py
```

### Playing Yourself

```bash
uv run python -m drone_hunter.scripts.play
```

Controls:
- **Mouse**: Aim at grid cell
- **Left Click**: Fire
- **R**: Manual reload

### Evaluating Models

```bash
# Evaluate detector_v2 (best model)
uv run python -m drone_hunter.scripts.evaluate \
  runs/detector_v2/best_model/best_model.zip \
  --single-target --detector-mode --episodes 5 --fps 15

# Evaluate oracle model
uv run python -m drone_hunter.scripts.evaluate \
  runs/oracle_single_target_v1/final_model/model.zip \
  --single-target --episodes 5 --fps 15
```

### Training Your Own Model

```bash
# Basic training (detector mode, recommended)
uv run python -m drone_hunter.scripts.train \
  --run-name my_experiment \
  --timesteps 700000 \
  --detector-mode

# Oracle mode (for comparison)
uv run python -m drone_hunter.scripts.train \
  --run-name my_oracle \
  --timesteps 500000

# With TensorBoard monitoring
tensorboard --logdir runs/my_experiment/tensorboard
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-name` | required | Experiment name |
| `--timesteps` | 1000000 | Total training steps |
| `--detector-mode` | False | Use Kalman filter estimates |
| `--single-target` | True | Single-target observation |
| `--n-envs` | 4 | Parallel environments |
| `--learning-rate` | 3e-4 | PPO learning rate |

---

## Key Takeaways

1. **Simplify the observation space**: MLPs work better with 24 features than 885. Move complexity to preprocessing (heuristic target selection) rather than expecting the policy to learn attention.

2. **Remove ground truth labels**: Training without oracle information produces deployable models. The agent can infer threat from observable behavior (velocity).

3. **State estimation can help**: The Kalman filter's temporal smoothing appears to produce more robust policies than raw oracle values. Noise may act as regularization.

4. **Extended training matters**: Detector mode took longer to converge (400k vs 250k steps) but ultimately achieved higher performance with extended training.

5. **Systematic ablation**: Each experiment tested one hypothesis. This methodical approach revealed that the multi-target failure was due to observation complexity, not network capacity.

---

## References

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - PPO implementation
- [Gymnasium](https://gymnasium.farama.org/) - RL environment framework
- [NanoDet](https://github.com/RangiLyu/nanodet) - Lightweight object detector
- [Kalman Filter](https://www.kalmanfilter.net/) - State estimation tutorial
