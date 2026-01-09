#!/usr/bin/env python3
"""Ablation study for uncertainty threshold tuning.

Tests different LOW/HIGH threshold combinations and reports metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.game_state import GameState
from core.renderer import Renderer
from core.kalman_tracker import KalmanTracker
from core.detection import Detection
from core.adaptive_scheduler import AdaptiveScheduler


def run_episode(low_thresh, high_thresh, max_frames=500, base_skip=3):
    """Run single episode with given thresholds."""
    game_state = GameState(max_frames=max_frames)
    renderer = Renderer(width=320, height=320)
    tracker = KalmanTracker()
    
    scheduler = AdaptiveScheduler(
        base_skip=base_skip,
        uncertainty_low=low_thresh,
        uncertainty_high=high_thresh,
    )
    
    tier_counts = {0: 0, 1: 0, 2: 0}
    uncertainties = []
    
    while not game_state.game_over:
        frame = renderer.render(game_state)
        uncertainty = tracker.get_max_uncertainty()
        uncertainties.append(uncertainty)
        
        tier = scheduler.get_detection_tier(frame, uncertainty)
        tier_counts[tier] += 1
        
        if tier == 2:  # Full detection
            # Oracle detections
            detections = [
                Detection(x=d.x, y=d.y, w=d.size, h=d.size, confidence=1.0)
                for d in game_state.drones if d.is_on_screen()
            ]
            tracker.update(detections)
            scheduler.mark_detection_complete()
        else:
            tracker.update([])  # Predict only
        
        game_state.step()
    
    total = sum(tier_counts.values())
    detect_rate = tier_counts[2] / total * 100
    skip_rate = tier_counts[0] / total * 100
    
    return {
        "detect_rate": detect_rate,
        "skip_rate": skip_rate,
        "tier_counts": tier_counts,
        "mean_uncertainty": np.mean(uncertainties),
        "p95_uncertainty": np.percentile(uncertainties, 95),
        "frames": total,
    }


def run_ablation():
    """Run ablation across threshold combinations."""
    # Based on observed data:
    # - Post-detection uncertainty: mean=2.47, p75=2.91
    # - Post-skip uncertainty: mean=5.37, p75=8.01
    
    low_values = [2.0, 2.5, 3.0, 3.5, 4.0]
    high_values = [4.0, 5.0, 6.0, 7.0, 8.0]
    
    print("=" * 80)
    print("UNCERTAINTY THRESHOLD ABLATION")
    print("=" * 80)
    print(f"Testing LOW: {low_values}")
    print(f"Testing HIGH: {high_values}")
    print(f"Episodes: 3 per combination, 500 frames each")
    print()
    
    results = []
    
    for low in low_values:
        for high in high_values:
            if low >= high:
                continue  # Skip invalid combinations
            
            episode_results = []
            for _ in range(3):
                r = run_episode(low, high)
                episode_results.append(r)
            
            # Average across episodes
            avg_detect = np.mean([r["detect_rate"] for r in episode_results])
            avg_skip = np.mean([r["skip_rate"] for r in episode_results])
            avg_unc = np.mean([r["mean_uncertainty"] for r in episode_results])
            
            results.append({
                "low": low,
                "high": high,
                "detect_rate": avg_detect,
                "skip_rate": avg_skip,
                "mean_unc": avg_unc,
            })
    
    # Print results table
    print(f"{'LOW':<6} {'HIGH':<6} {'Detect%':<10} {'Skip%':<10} {'Mean Unc':<10}")
    print("-" * 50)
    
    for r in sorted(results, key=lambda x: (x["low"], x["high"])):
        print(f"{r['low']:<6.1f} {r['high']:<6.1f} {r['detect_rate']:<10.1f} "
              f"{r['skip_rate']:<10.1f} {r['mean_unc']:<10.2f}")
    
    # Find best configurations
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Target: ~30-40% detection rate for good balance
    target_rate = 35
    best = min(results, key=lambda x: abs(x["detect_rate"] - target_rate))
    print(f"Best for ~{target_rate}% detection: LOW={best['low']}, HIGH={best['high']} "
          f"({best['detect_rate']:.1f}% detect, {best['skip_rate']:.1f}% skip)")
    
    # Most aggressive skip
    most_skip = max(results, key=lambda x: x["skip_rate"])
    print(f"Most aggressive skip: LOW={most_skip['low']}, HIGH={most_skip['high']} "
          f"({most_skip['detect_rate']:.1f}% detect, {most_skip['skip_rate']:.1f}% skip)")


if __name__ == "__main__":
    run_ablation()
