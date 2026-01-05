#!/usr/bin/env python3
"""Test different confidence thresholds to find optimal value."""

import sys
from pathlib import Path
import random
import numpy as np

# Set seeds
random.seed(42)
np.random.seed(42)

sys.path.insert(0, str(Path(__file__).parent))

from core.game_state import GameState
from core.renderer import Renderer
from inference.nanodet import create_detector

def test_threshold(threshold: float, num_frames: int = 100) -> dict:
    random.seed(42)
    np.random.seed(42)

    detector = create_detector(
        str(Path(__file__).parent / "models" / "nanodet.onnx"),
        backend_type="onnx",
        conf_threshold=threshold,
        iou_threshold=0.5,
    )

    game_state = GameState(max_frames=500)
    renderer = Renderer(width=320, height=320)

    game_state.reset()
    renderer.reset_background()

    total_drones = 0
    total_detections = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for _ in range(num_frames):
        frame = renderer.render(game_state)
        detections = detector.detect(frame)
        gt_drones = [d for d in game_state.drones if d.is_on_screen()]

        total_drones += len(gt_drones)
        total_detections += len(detections)

        # Simple matching: for each GT drone, check if any detection is close
        for drone in gt_drones:
            matched = False
            for det in detections:
                if abs(det.x - drone.x) < 0.1 and abs(det.y - drone.y) < 0.1:
                    matched = True
                    true_positives += 1
                    break
            if not matched:
                false_negatives += 1

        # Count false positives: detections not near any drone
        for det in detections:
            is_fp = True
            for drone in gt_drones:
                if abs(det.x - drone.x) < 0.1 and abs(det.y - drone.y) < 0.1:
                    is_fp = False
                    break
            if is_fp:
                false_positives += 1

        game_state.step()
        if game_state.game_over:
            break

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    return {
        "threshold": threshold,
        "total_gt": total_drones,
        "total_det": total_detections,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def main():
    print("Threshold | GT | Det | TP | FP | FN | Precision | Recall |  F1")
    print("-" * 70)

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        result = test_threshold(threshold)
        print(f"  {result['threshold']:.2f}   | {result['total_gt']:3d} | {result['total_det']:3d} | "
              f"{result['tp']:3d} | {result['fp']:3d} | {result['fn']:3d} | "
              f"{result['precision']:.3f}     | {result['recall']:.3f}  | {result['f1']:.3f}")

if __name__ == "__main__":
    main()
