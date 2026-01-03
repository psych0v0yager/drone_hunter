# Training Results Log

## oracle_single_target_v1 (Baseline Oracle)

**Date**: 2026-01-02
**Status**: SUCCESS - Baseline for ablation studies

### Configuration

| Parameter | Value |
|-----------|-------|
| Mode | Single-target oracle |
| Observation normalization | Disabled |
| Network architecture | 64x64 MLP (ReLU) |
| Total timesteps | 500,000 |
| Parallel envs | 8 |
| Learning rate | 3e-4 |
| Grid size | 8x8 |
| Max frames | 1000 |

### Observation Space

| Feature | Count |
|---------|-------|
| z (depth) | 1 |
| is_kamikaze | 1 |
| vz (velocity, scaled) | 1 |
| urgency | 1 |
| grid_x one-hot | 8 |
| grid_y one-hot | 8 |
| game_state (ammo, reload, frame, threat, has_target) | 5 |
| **Total** | **25** |

### Results

| Metric | Value |
|--------|-------|
| Best eval reward | +82.2 (at 430k steps) |
| Final eval reward | +70.8 ± 24.7 |
| Mean episode length | 919 frames |
| Hit rate | 55.1% |
| Total hits (10 ep eval) | 520 |
| Total misses (10 ep eval) | 424 |
| Convergence | ~150k steps |

### Training Progression

| Steps | Eval Reward | Episode Length |
|-------|-------------|----------------|
| 10k | -14.7 | 112 |
| 100k | -12.9 | 216 |
| 200k | +27.5 | 821 |
| 270k | +46.4 | 882 |
| 330k | +55.5 | 861 |
| 370k | +60.8 | 970 |
| 430k | +82.2 | 1000 (full) |
| 500k | +70.8 | 919 |

### Key Fixes (vs failed oracle_2M_v1)

1. **Single-target mode**: Reduced input from 885 to 25 features
2. **Disabled VecNormalize obs**: Preserved one-hot encoding structure
3. **Priority sorting**: Kamikazes first, then by z-depth

### Model Location

```
runs/oracle_single_target_v1/
├── final_model/
│   ├── model.zip
│   └── vec_normalize.pkl
├── best_model/
├── checkpoints/
├── tensorboard/
└── eval_logs/
```

### Visualization Command

```bash
uv run python -m drone_hunter.scripts.evaluate \
  runs/oracle_single_target_v1/final_model/model.zip \
  --single-target --episodes 5 --fps 15
```

### Notes

- Agent exceeds human-level performance (~60 points)
- "Pulsing" fire pattern due to 10-round clip + 30-frame reload
- 55% hit rate with high rewards suggests aggressive but strategic firing
- This is the baseline for detector integration ablations

---

## oracle_multitarget_nonorm_v1 (Ablation)

**Date**: 2026-01-02
**Status**: FAILED - Confirms single-target was necessary

### Purpose

Ablation to test if `norm_obs=False` alone was sufficient, or if single-target was also required.

### Configuration

| Parameter | Value |
|-----------|-------|
| Mode | Multi-target (10 drones × 4 frames) |
| Observation normalization | Disabled |
| Network architecture | 256x256 MLP (ReLU) |
| Total timesteps | 500,000 |
| Input features | 885 |

### Results

| Metric | Value |
|--------|-------|
| Best eval reward | -7.17 (at 460k) |
| Final eval reward | -10.3 |
| Episode length | 251 frames |

### Comparison

| Metric | Single-target | Multi-target |
|--------|---------------|--------------|
| Best reward | +82.2 | -7.17 |
| Final reward | +70.8 | -10.3 |
| Converged | Yes | No |

### Conclusion

Multi-target mode fails even with `norm_obs=False`. The single-target simplification was necessary because:
1. MLPs cannot handle the attention-like target selection problem
2. 885 features (40 drone slots) is too complex for policy learning
3. Single-target with heuristic priority sorting is the correct design

---

## oracle_no_kamikaze_v1 (No Ground Truth)

**Date**: 2026-01-03
**Status**: SUCCESS - Proves agent can infer threat from velocity

### Purpose

Validate system works without ground truth `is_kamikaze` label. Agent must infer threat from observable behavior (vz velocity).

### Configuration

| Parameter | Value |
|-----------|-------|
| Mode | Single-target |
| Observation normalization | Disabled |
| Network architecture | 64x64 MLP (ReLU) |
| Total timesteps | 500,000 |
| Input features | 24 (was 25) |
| is_kamikaze | **REMOVED** |

### Changes from Baseline

1. Removed `is_kamikaze` feature (25 → 24 features)
2. Updated urgency computation to use `vz < 0` (approaching)
3. Updated sorting to use computed urgency instead of ground truth type

### Results

| Metric | Value |
|--------|-------|
| Best eval reward | +62.2 (at 500k) |
| Final eval reward | +62.2 |
| Episode length | 910 frames |
| Convergence | ~350k steps |

### Comparison

| Run | Best Reward | is_kamikaze |
|-----|-------------|-------------|
| oracle_single_target_v1 | +82.2 | Yes (ground truth) |
| oracle_no_kamikaze_v1 | +62.2 | **No** (removed) |

### Conclusion

Agent successfully learned to infer threat from observable behavior:
- **vz < 0** (approaching) = potential threat
- Agent learns correlation through reward signal (-5 for kamikaze impact)
- ~20 point drop is expected overhead for learning vs being told

This model is **deployment-ready** - no ground truth labels required.

---

## oracle_no_kamikaze_v2 (Full Cleanup)

**Date**: 2026-01-03
**Status**: SUCCESS - Best performance yet without ground truth!

### Purpose

Complete removal of `is_kamikaze` from all agent-visible features, including:
- Target observation (already done in v1)
- `threat_level` computation (now uses `vz < 0`)
- Multi-drone observation (22 → 21 features)

### Configuration

| Parameter | Value |
|-----------|-------|
| Mode | Single-target |
| Observation normalization | Disabled |
| Network architecture | 64x64 MLP (ReLU) |
| Total timesteps | 500,000 |
| Parallel envs | 4 |
| Input features | 19 (target) + 5 (game_state) = 24 |
| is_kamikaze | **FULLY REMOVED** |

### Changes from v1

1. `threat_level` now uses `vz < 0` instead of `is_kamikaze`
2. Multi-drone observation also updated (for consistency)
3. Added `has_approaching_threat` property

### Results

| Metric | Value |
|--------|-------|
| Best eval reward | **+80.9** (at 460k steps) |
| Final eval reward | +77.2 ± 26.4 |
| Episode length | 916 frames (1000 at peak) |
| Convergence | ~250k steps |

### Training Progression

| Steps | Eval Reward | Episode Length |
|-------|-------------|----------------|
| 100k | -4.3 | 232 |
| 150k | +5.8 | 450 |
| 190k | +26.0 | 712 |
| 250k | +40.5 | 810 |
| 270k | +52.5 | 850 |
| 330k | +55.7 | 892 |
| 380k | +59.7 | 858 |
| 420k | +72.5 | 946 |
| 460k | **+80.9** | 1000 (full!) |
| 500k | +77.2 | 916 |

### Key Finding: is_kamikaze May Have Been Harmful

| Run | Best Reward | is_kamikaze | threat_level |
|-----|-------------|-------------|--------------|
| oracle_single_target_v1 | +82.2 | Yes | Uses is_kamikaze |
| oracle_no_kamikaze_v1 | +62.2 | Removed from obs | Still uses is_kamikaze |
| oracle_no_kamikaze_v2 | **+80.9** | **Fully removed** | Uses vz < 0 |

**Conclusion**: The `is_kamikaze` ground truth label may have actually been a *confounding signal*. When we fully removed it (including from `threat_level`), performance returned to baseline levels. The agent learns threat purely from velocity, which is the correct inductive bias for real-world deployment.

### Model Location

```
runs/oracle_no_kamikaze_v2/
├── final_model/
├── best_model/
├── checkpoints/
├── tensorboard/
└── eval_logs/
```

### Visualization Command

```bash
uv run python -m drone_hunter.scripts.evaluate \
  runs/oracle_no_kamikaze_v2/final_model/model.zip \
  --single-target --episodes 5 --fps 15
```

---

## Future Experiments

### Planned Ablations
- [ ] With detector (NanoDet) instead of oracle
- [ ] With Kalman filter for tracking
- [ ] With frame stacking for temporal context
- [ ] With queued target system for planning

### Queued Target System (TODO)
Concept: Pass N most urgent targets instead of just 1, allowing agent to:
- Predict next target after current kill
- Plan ammo usage across multiple threats
- Combined with Kalman filter for trajectory prediction
