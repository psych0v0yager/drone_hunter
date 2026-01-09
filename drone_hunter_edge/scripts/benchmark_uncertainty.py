#!/usr/bin/env python3
"""Profile Kalman uncertainty across different detection skip rates.

This benchmark helps calibrate the uncertainty thresholds for the adaptive
scheduler. It runs episodes at various skip intervals and tracks how
uncertainty grows between detections.

Usage:
    python benchmark_uncertainty.py [--episodes 5] [--max-frames 1000]
    python benchmark_uncertainty.py --detector models/nanodet_drone.onnx

Output includes:
- Mean, std, p95, max uncertainty for each skip rate
- Uncertainty growth rate per skipped frame
- Recommended threshold values
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.game_state import GameState
from core.renderer import Renderer
from core.kalman_tracker import KalmanTracker
from core.detection import Detection


def run_uncertainty_profile(
    detector_model: Optional[str] = None,
    skip_intervals: List[int] = [1, 2, 3, 5, 8, 10],
    num_episodes: int = 5,
    max_frames: int = 1000,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """Profile uncertainty at different skip rates.

    Args:
        detector_model: Path to detector model (None for oracle mode).
        skip_intervals: List of skip intervals to test.
        num_episodes: Number of episodes per skip interval.
        max_frames: Maximum frames per episode.
        verbose: Print progress.

    Returns:
        Dict mapping skip_interval -> uncertainty statistics.
    """
    # Load detector if provided
    detector = None
    oracle_mode = detector_model is None
    if detector_model:
        try:
            from inference.nanodet import create_detector
            detector = create_detector(
                detector_model,
                backend_type="onnx",
                conf_threshold=0.60,
                iou_threshold=0.5,
            )
            if verbose:
                print(f"Loaded detector: {detector_model}")
        except Exception as e:
            print(f"Warning: Failed to load detector: {e}")
            print("Using oracle mode")
            oracle_mode = True

    results = {}

    for skip in skip_intervals:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Testing skip interval: {skip}")
            print(f"{'='*50}")

        all_uncertainties = []
        all_uncertainty_growth = []

        for episode in range(num_episodes):
            game_state = GameState(max_frames=max_frames)
            renderer = Renderer(width=320, height=320)
            tracker = KalmanTracker()

            step = 0
            last_detection_uncertainty = 0.0

            while not game_state.game_over:
                # Render frame
                frame = renderer.render(game_state)

                # Decide whether to detect
                run_detection = (step % skip == 0)

                if run_detection:
                    if oracle_mode:
                        detections = [
                            Detection(
                                x=d.x,
                                y=d.y,
                                w=d.size,
                                h=d.size,
                                confidence=1.0,
                            )
                            for d in game_state.drones
                            if d.is_on_screen()
                        ]
                    else:
                        detections = detector.detect(frame)
                else:
                    detections = []

                # Update tracker
                tracker.update(detections)

                # Get uncertainty
                uncertainty = tracker.get_max_uncertainty()
                all_uncertainties.append(uncertainty)

                # Track uncertainty growth since last detection
                if run_detection:
                    if step > 0:
                        growth = uncertainty - last_detection_uncertainty
                        all_uncertainty_growth.append(growth)
                    last_detection_uncertainty = uncertainty

                game_state.step()
                step += 1

            if verbose:
                print(f"  Episode {episode+1}/{num_episodes}: "
                      f"steps={step}, mean_unc={np.mean(all_uncertainties[-step:]):.4f}")

        # Compute statistics
        uncertainties = np.array(all_uncertainties)
        results[skip] = {
            "mean": float(np.mean(uncertainties)),
            "std": float(np.std(uncertainties)),
            "p50": float(np.percentile(uncertainties, 50)),
            "p95": float(np.percentile(uncertainties, 95)),
            "p99": float(np.percentile(uncertainties, 99)),
            "max": float(np.max(uncertainties)),
            "min": float(np.min(uncertainties)),
        }

        if all_uncertainty_growth:
            growth = np.array(all_uncertainty_growth)
            results[skip]["growth_mean"] = float(np.mean(growth))
            results[skip]["growth_std"] = float(np.std(growth))

    return results


def print_results_table(results: Dict[int, Dict[str, float]]):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("UNCERTAINTY PROFILE RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Skip':<6} {'Mean':<10} {'Std':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'Max':<10}")
    print("-" * 80)

    for skip in sorted(results.keys()):
        stats = results[skip]
        print(f"{skip:<6} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
              f"{stats['p50']:<10.4f} {stats['p95']:<10.4f} "
              f"{stats['p99']:<10.4f} {stats['max']:<10.4f}")

    print("-" * 80)

    # Uncertainty growth per frame
    print("\nUNCERTAINTY GROWTH (per detection cycle)")
    print("-" * 40)
    for skip in sorted(results.keys()):
        stats = results[skip]
        if "growth_mean" in stats:
            per_frame = stats["growth_mean"] / skip if skip > 0 else 0
            print(f"Skip {skip}: growth/cycle={stats['growth_mean']:.4f} "
                  f"(~{per_frame:.4f}/frame)")

    # Threshold recommendations
    print("\n" + "=" * 80)
    print("THRESHOLD RECOMMENDATIONS")
    print("=" * 80)

    # Use skip=1 as baseline for "low" uncertainty
    if 1 in results:
        baseline_p95 = results[1]["p95"]
        print(f"UNCERTAINTY_LOW  = {baseline_p95:.3f}  (p95 at skip=1, coast on prediction)")
    else:
        print("UNCERTAINTY_LOW  = 0.1  (default, no skip=1 data)")

    # Use skip=5 as target for "high" uncertainty
    if 5 in results:
        target_p50 = results[5]["p50"]
        print(f"UNCERTAINTY_HIGH = {target_p50:.3f}  (p50 at skip=5, trigger full detection)")
    elif 3 in results:
        target_p50 = results[3]["p50"]
        print(f"UNCERTAINTY_HIGH = {target_p50:.3f}  (p50 at skip=3, trigger full detection)")
    else:
        print("UNCERTAINTY_HIGH = 0.3  (default, no skip=5 data)")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Profile Kalman uncertainty at different skip rates"
    )
    parser.add_argument(
        "--detector", type=str, default=None,
        help="Path to detector model (default: oracle mode)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes per skip interval"
    )
    parser.add_argument(
        "--max-frames", type=int, default=1000,
        help="Maximum frames per episode"
    )
    parser.add_argument(
        "--skips", type=str, default="1,2,3,5,8,10",
        help="Comma-separated skip intervals to test"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    skip_intervals = [int(x) for x in args.skips.split(",")]

    print("=" * 80)
    print("KALMAN UNCERTAINTY PROFILER")
    print("=" * 80)
    print(f"Mode: {'Detector' if args.detector else 'Oracle'}")
    print(f"Skip intervals: {skip_intervals}")
    print(f"Episodes: {args.episodes}")
    print(f"Max frames: {args.max_frames}")

    start_time = time.time()

    results = run_uncertainty_profile(
        detector_model=args.detector,
        skip_intervals=skip_intervals,
        num_episodes=args.episodes,
        max_frames=args.max_frames,
        verbose=not args.quiet,
    )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    print_results_table(results)


if __name__ == "__main__":
    main()
