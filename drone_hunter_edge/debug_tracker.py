#!/usr/bin/env python3
"""Debug script to compare tracker vz estimates with ground truth."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from core.game_state import GameState
from core.renderer import Renderer
from core.kalman_tracker import KalmanTracker
from core.detection import Detection

# Set seed for reproducibility
import random
random.seed(42)
np.random.seed(42)


def main():
    # Check for detector model - use drone-specific model
    detector = None
    model_path = Path(__file__).parent.parent / "models" / "nanodet_drone.onnx"
    if model_path.exists():
        try:
            from inference.nanodet import create_detector
            detector = create_detector(
                str(model_path),
                backend_type="onnx",
                conf_threshold=0.60,
                iou_threshold=0.5,
            )
            print(f"Using NanoDet drone detector: {model_path}")
        except Exception as e:
            print(f"Failed to load detector: {e}")
    else:
        print("No detector model found, using oracle detections")

    game_state = GameState(max_frames=200)
    renderer = Renderer(width=320, height=320)
    tracker = KalmanTracker()

    game_state.reset()
    renderer.reset_background()
    tracker.reset()

    print("\nFrame | GT_z   GT_vz  | Est_z  Est_vz | vz_err | grid_x grid_y | urgency")
    print("-" * 80)

    for step in range(100):
        # Render frame
        frame = renderer.render(game_state)

        # Get ground truth drones
        gt_drones = [d for d in game_state.drones if d.is_on_screen()]

        # Get detections from detector or oracle
        if detector is not None:
            detections = detector.detect(frame)
            det_source = "detector"
        else:
            # Oracle detections using d.size
            detections = [
                Detection(x=d.x, y=d.y, w=d.size, h=d.size, confidence=1.0)
                for d in gt_drones
            ]
            det_source = "oracle"

        # Update tracker
        tracker.update(detections)

        # Get tracks
        tracks = tracker.get_tracks_for_observation()

        # Debug: print detection bbox heights and estimated z
        if detections and gt_drones:
            det = detections[0]
            gt = gt_drones[0]
            z_est_raw = 0.5 * (0.06 / det.h) if det.h > 0.001 else 1.0
            z_est_clamped = max(0.1, min(1.0, z_est_raw))
            print(f"  [{det_source}] det_h={det.h:.4f} gt_size={gt.size:.4f} ratio={det.h/gt.size:.2f} | z_raw={z_est_raw:.3f}")

        # Compare tracker estimates with ground truth
        if tracks and gt_drones:
            track = tracks[0]

            # Find matching ground truth drone (closest position)
            best_match = None
            best_dist = float('inf')
            for drone in gt_drones:
                dist = abs(drone.x - track.center[0]) + abs(drone.y - track.center[1])
                if dist < best_dist:
                    best_dist = dist
                    best_match = drone

            if best_match:
                # Compute urgency from tracker
                if track.vz < 0:
                    frames_to_impact = track.z / max(0.001, abs(track.vz))
                    urgency = 1.0 / (1.0 + frames_to_impact / 50.0)
                else:
                    urgency = 0.1

                grid_x = min(7, max(0, int(track.center[0] * 8)))
                grid_y = min(7, max(0, int(track.center[1] * 8)))

                vz_error = track.vz - best_match.vz

                print(f"{step:5d} | {best_match.z:.3f}  {best_match.vz:.4f} | "
                      f"{track.z:.3f}  {track.vz:.4f} | {vz_error:+.4f} | "
                      f"{grid_x:6d} {grid_y:6d} | {urgency:.3f}"
                      f" {'KAMIKAZE' if best_match.is_kamikaze else ''}")
        elif gt_drones:
            d = gt_drones[0]
            print(f"{step:5d} | {d.z:.3f}  {d.vz:.4f} | NO TRACK (min_hits not reached)")

        # Step game
        game_state.step()

        if game_state.game_over:
            print(f"\nGame over: {game_state.game_over_reason}")
            break


if __name__ == "__main__":
    main()
