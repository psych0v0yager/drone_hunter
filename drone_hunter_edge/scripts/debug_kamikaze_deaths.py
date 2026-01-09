#!/usr/bin/env python3
"""Debug script to analyze kamikaze deaths in Drone Hunter.

Captures detailed telemetry for the last N frames before each kamikaze death
and identifies patterns that lead to deaths.

Usage:
    uv run python scripts/debug_kamikaze_deaths.py --episodes 100
"""

import sys
import argparse
from pathlib import Path
from collections import deque
from typing import List, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.game_state import GameState
from core.kalman_tracker import KalmanTracker
from core.observation import build_tracker_observation, compute_max_urgency
from core.adaptive_scheduler import AdaptiveScheduler
from core.tiny_detector import TinyDetector
from core.renderer import Renderer
from inference.nanodet import create_detector
from inference.policy import SimplePolicyInference


def find_kamikaze_track(tracks, drone, threshold=0.15):
    """Find the track that matches the kamikaze drone by position."""
    for track in tracks:
        dist = np.sqrt((track.x - drone.x)**2 + (track.y - drone.y)**2)
        if dist < threshold:
            return track
    return None


def compute_frames_to_impact(z: float, vz: float) -> float:
    """Compute frames until impact (z <= 0.08)."""
    if vz >= 0:
        return float('inf')
    impact_z = 0.08
    remaining_z = z - impact_z
    if remaining_z <= 0:
        return 0
    return remaining_z / abs(vz)


def run_debug(
    detector_path: str,
    policy_path: str,
    tiny_path: str,
    num_episodes: int = 100,
    grid_size: int = 8,
):
    """Run simulation and capture death event telemetry."""

    print("Loading models...")
    detector = create_detector(detector_path, conf_threshold=0.60)
    policy = SimplePolicyInference(policy_path, grid_size=grid_size)
    tiny_detector = TinyDetector(tiny_path) if Path(tiny_path).exists() else None

    # Aggregated stats
    deaths = []
    survivals = 0

    # Cause counters
    causes = {
        "never_tracked": 0,
        "track_lost": 0,
        "never_fired": 0,
        "wrong_grid": 0,
        "reloading": 0,
        "low_urgency": 0,
        "late_detection": 0,
    }

    # Error aggregation
    all_z_errors = []
    all_vz_errors = []
    all_urgencies_at_death = []
    tier_counts = {0: 0, 1: 0, 2: 0}

    print(f"Running {num_episodes} episodes...")

    for ep in range(num_episodes):
        # Reset
        game = GameState(max_frames=1000)
        renderer = Renderer(320, 320)
        tracker = KalmanTracker(max_age=15)
        scheduler = AdaptiveScheduler(base_skip=1, tiny_detector=tiny_detector)
        renderer.reset_background()

        # Per-kamikaze history: drone_id -> deque of frame data
        kamikaze_history = {}

        while not game.game_over:
            frame = renderer.render(game)

            # Get kamikazes
            kamikazes = [d for d in game.drones if d.is_kamikaze]

            # Detection
            unc = tracker.get_max_uncertainty()
            max_urg = compute_max_urgency(tracker)
            tier = scheduler.get_detection_tier(frame, unc, max_urg)
            tier_counts[tier] += 1

            if tier == 2:
                dets = detector.detect(frame)
            elif tier == 1 and tiny_detector:
                dets = []
                for trk in tracker.get_tracks_for_observation():
                    px, py = trk.center
                    td = tiny_detector.detect_at_roi(frame, px, py)
                    for d in td:
                        d.w = float(np.clip(d.w, trk.bbox_size[0]*0.7, trk.bbox_size[0]*1.4))
                        d.h = float(np.clip(d.h, trk.bbox_size[1]*0.7, trk.bbox_size[1]*1.4))
                    dets.extend(td)
            else:
                dets = None

            # Update tracker
            if dets is None:
                tracks = tracker.predict_only()
            else:
                tracks = tracker.update(dets if dets else [])

            if tier > 0:
                scheduler.mark_detection_complete(len(tracks))

            # Build observation
            obs = build_tracker_observation(tracker, grid_size=grid_size)
            obs["game_state"][0] = game.ammo_fraction
            obs["game_state"][1] = game.reload_fraction
            obs["game_state"][2] = game.frame_fraction

            # Policy action
            action_idx, grid_coords = policy.predict(obs)

            # Record per-kamikaze
            all_tracks = tracker.get_tracks_for_observation()
            for kam in kamikazes:
                if kam.track_id not in kamikaze_history:
                    kamikaze_history[kam.track_id] = deque(maxlen=60)

                matching_track = find_kamikaze_track(all_tracks, kam)

                gt_grid_x = min(grid_size-1, max(0, int(kam.x * grid_size)))
                gt_grid_y = min(grid_size-1, max(0, int(kam.y * grid_size)))

                frame_data = {
                    "frame": game.frame_count,
                    "tier": tier,
                    "unc": unc,
                    # Ground truth
                    "gt_x": kam.x, "gt_y": kam.y, "gt_z": kam.z, "gt_vz": kam.vz,
                    "gt_grid": (gt_grid_x, gt_grid_y),
                    "gt_fti": compute_frames_to_impact(kam.z, kam.vz),
                    # Track
                    "tracked": matching_track is not None,
                    "trk_z": matching_track.z if matching_track else None,
                    "trk_vz": matching_track.vz if matching_track else None,
                    "trk_grid": (int(matching_track.x * grid_size), int(matching_track.y * grid_size)) if matching_track else None,
                    # Errors
                    "z_err": (matching_track.z - kam.z) if matching_track else None,
                    "vz_err": (matching_track.vz - kam.vz) if matching_track else None,
                    # Observation
                    "obs_z": obs["target"][0],
                    "obs_vz": obs["target"][1],
                    "obs_urg": obs["target"][2],
                    # Action
                    "action": action_idx,
                    "action_grid": grid_coords,
                    "would_hit": (grid_coords == (gt_grid_x, gt_grid_y)) if grid_coords else False,
                    # Ammo
                    "ammo": game.ammo,
                    "reloading": game.is_reloading,
                }
                kamikaze_history[kam.track_id].append(frame_data)

            # Execute action
            if action_idx > 0 and grid_coords:
                game.fire(grid_coords[0], grid_coords[1], grid_size)

            game.step()

        # Episode ended
        if "Kamikaze" in game.game_over_reason:
            # Find the kamikaze that killed us
            for drone in game.drones:
                if drone.is_kamikaze and drone.z <= 0.08:
                    hist = list(kamikaze_history.get(drone.track_id, []))
                    if not hist:
                        continue

                    # Analyze this death
                    frames_tracked = sum(1 for f in hist if f["tracked"])
                    frames_fired = sum(1 for f in hist if f["action"] > 0)
                    frames_correct = sum(1 for f in hist if f["would_hit"])
                    frames_reload = sum(1 for f in hist if f["reloading"])
                    max_urgency = max(f["obs_urg"] for f in hist)

                    # Collect errors
                    for f in hist:
                        if f["z_err"] is not None:
                            all_z_errors.append(f["z_err"])
                            all_vz_errors.append(f["vz_err"])

                    all_urgencies_at_death.append(hist[-1]["obs_urg"])

                    # Determine cause(s)
                    if frames_tracked == 0:
                        causes["never_tracked"] += 1
                    elif frames_tracked < len(hist) * 0.5:
                        causes["track_lost"] += 1

                    if frames_fired == 0:
                        causes["never_fired"] += 1
                    elif frames_correct == 0:
                        causes["wrong_grid"] += 1

                    if frames_reload > len(hist) * 0.3:
                        causes["reloading"] += 1

                    if max_urgency < 0.5:
                        causes["low_urgency"] += 1

                    # Late detection: first tracked frame
                    first_tracked = None
                    for f in hist:
                        if f["tracked"]:
                            first_tracked = f["frame"]
                            break
                    if first_tracked is None or (game.frame_count - first_tracked) < 25:
                        causes["late_detection"] += 1

                    deaths.append({
                        "episode": ep,
                        "frame": game.frame_count,
                        "history_len": len(hist),
                        "frames_tracked": frames_tracked,
                        "frames_fired": frames_fired,
                        "frames_correct": frames_correct,
                        "max_urgency": max_urgency,
                    })
                    break
        else:
            survivals += 1

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}: {survivals} survivals, {len(deaths)} deaths")

    # Print report
    total = num_episodes
    death_count = len(deaths)

    print("\n" + "=" * 60)
    print("KAMIKAZE DEATH ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nSurvival: {survivals}/{total} ({survivals/total*100:.1f}%)")
    print(f"Deaths: {death_count}/{total} ({death_count/total*100:.1f}%)")

    print("\n--- CAUSE BREAKDOWN ---")
    for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
        pct = count / max(1, death_count) * 100
        print(f"  {cause:18s}: {count:3d} ({pct:5.1f}%)")

    if all_z_errors:
        print("\n--- DEPTH (z) ESTIMATION ---")
        print(f"  Mean z error:  {np.mean(all_z_errors):+.4f}")
        print(f"  Std z error:   {np.std(all_z_errors):.4f}")
        print(f"  Mean |z| err:  {np.mean(np.abs(all_z_errors)):.4f}")

    if all_vz_errors:
        print("\n--- VELOCITY (vz) ESTIMATION ---")
        print(f"  Mean vz error: {np.mean(all_vz_errors):+.5f}")
        print(f"  Std vz error:  {np.std(all_vz_errors):.5f}")
        wrong_sign = sum(1 for e in all_vz_errors if e > 0) / len(all_vz_errors)
        print(f"  Wrong sign:    {wrong_sign*100:.1f}%")

    if all_urgencies_at_death:
        print("\n--- URGENCY AT DEATH ---")
        print(f"  Mean: {np.mean(all_urgencies_at_death):.3f}")
        print(f"  Min:  {np.min(all_urgencies_at_death):.3f}")
        print(f"  Max:  {np.max(all_urgencies_at_death):.3f}")

    print("\n--- TIER DISTRIBUTION (all frames) ---")
    total_frames = sum(tier_counts.values())
    for t, c in tier_counts.items():
        print(f"  Tier {t}: {c:6d} ({c/total_frames*100:5.1f}%)")

    print("\n" + "=" * 60)

    return deaths, causes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--detector", default="../models/nanodet_drone_opset21.onnx")
    parser.add_argument("--policy", default="models/policy_detector_real.onnx")
    parser.add_argument("--tiny-detector", default="models/tiny_drone_16k_medium.onnx")
    args = parser.parse_args()

    run_debug(
        detector_path=args.detector,
        policy_path=args.policy,
        tiny_path=args.tiny_detector,
        num_episodes=args.episodes,
    )
