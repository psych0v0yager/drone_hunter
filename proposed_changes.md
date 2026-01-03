# Proposed Changes for Next Training Run

## Current Status
Training `oracle_improved_v1` in progress with:
- 256x256 ReLU network
- 22 features per drone (including one-hot grid cells)
- 885 total input features

## Observations from Current Training
- One-hot grid encoding dramatically improved learning speed
- Agent hit -13 reward at 50k steps (previous best was -5.44 at 2M)
- x,y coordinates are redundant with one-hot encoding

---

## Proposed Changes

### 1. Observation Space Cleanup

**Remove redundant features:**
| Feature | Current | Proposed | Reason |
|---------|---------|----------|--------|
| x, y | Included | Remove | Redundant with one-hot grid |
| z (depth) | Included | Keep | Urgency calculation |
| is_kamikaze | Included | Keep | Priority flag |
| vz | Included | Keep | Approach speed |
| urgency | Included | Keep | Precomputed priority |
| grid_x one-hot | Included | Keep | Action hint |
| grid_y one-hot | Included | Keep | Action hint |

**Per-drone features:** 22 → 20

### 2. Network Architecture

TODO: Evaluate current run results first

### 3. Reward Structure

TODO: Analyze agent behavior from current run

### 4. Detector Integration Prep

TODO: Plan Kalman filter + NanoDet pipeline

---

## Questions to Answer
1. Does the current improved agent achieve positive rewards?
2. What's the hit rate? Is it actually hitting drones now?
3. Does it prioritize kamikazes correctly?
4. How does it perform compared to human play (~60 points)?

---

## Implementation Priority
1. [ ] Evaluate current training run
2. [ ] Simplify observation (remove x,y)
3. [ ] Tune reward structure if needed
4. [ ] Add detector integration
5. [ ] Add Kalman filter for tracking
