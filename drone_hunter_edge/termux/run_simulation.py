#!/usr/bin/env python3
"""Main simulation loop for Drone Hunter edge inference.

Runs the trained agent with NanoDet detection on Termux or desktop.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.game_state import GameState
from core.renderer import Renderer
from core.kalman_tracker import KalmanTracker
from core.detection import Detection
from core.observation import (
    ObservationNormalizer,
    build_tracker_observation,
    build_oracle_observation,
    compute_max_urgency,
)
from core.adaptive_scheduler import AdaptiveScheduler, calibrate_device
from core.tiny_detector import TinyDetector


def run_simulation(
    policy_model: Optional[str] = None,
    detector_model: Optional[str] = None,
    normalization_stats: Optional[str] = None,
    backend_type: str = "onnx",
    oracle_mode: bool = False,
    num_episodes: int = 5,
    max_frames: int = 1000,
    grid_size: int = 8,
    render_output: bool = True,
    output_dir: Optional[str] = None,
    save_interval: int = 50,
    verbose: bool = True,
    detect_interval: int = 1,
    provider_priority: str = "auto",
    adaptive_mode: bool = False,
    adaptive_base_skip: Optional[int] = None,
    tiny_detector_path: Optional[str] = None,
) -> dict:
    """Run the drone hunter simulation.

    Args:
        policy_model: Path to ONNX policy model (None for random actions).
        detector_model: Path to detector model (None for oracle mode).
        normalization_stats: Path to normalization.json.
        backend_type: Inference backend ("onnx" or "ncnn").
        oracle_mode: Use ground truth instead of detector.
        num_episodes: Number of episodes to run.
        max_frames: Maximum frames per episode.
        grid_size: Size of firing grid.
        render_output: Save rendered frames.
        output_dir: Directory for output frames.
        save_interval: Save frame every N steps.
        verbose: Print progress information.
        detect_interval: Run detector every N frames (1=every frame, ignored if adaptive).
        provider_priority: Execution provider selection.
        adaptive_mode: Use adaptive scheduler instead of fixed detect_interval.
        adaptive_base_skip: Override base_skip for adaptive mode (None=auto-calibrate).

    Returns:
        Dictionary with episode statistics.
    """
    # Initialize components
    game_state = GameState(max_frames=max_frames)
    renderer = Renderer(width=320, height=320)
    # Adaptive mode needs higher max_age since detection is less frequent
    # and Tier 1 may miss drones outside its ROI
    tracker_max_age = 15 if adaptive_mode else 5
    tracker = KalmanTracker(max_age=tracker_max_age)

    # Load detector if provided
    detector = None
    if detector_model and not oracle_mode:
        try:
            from inference.nanodet import create_detector
            detector = create_detector(
                detector_model,
                backend_type=backend_type,
                conf_threshold=0.60,
                iou_threshold=0.5,
                provider_priority=provider_priority,
            )
            if verbose:
                print(f"Loaded detector: {detector_model} ({backend_type})")
                if hasattr(detector.backend, 'active_provider'):
                    print(f"  Provider: {detector.backend.active_provider}")
        except Exception as e:
            print(f"Warning: Failed to load detector: {e}")
            print("Falling back to oracle mode")
            oracle_mode = True

    # Load policy if provided
    policy = None
    if policy_model:
        try:
            from inference.policy import SimplePolicyInference
            policy = SimplePolicyInference(
                policy_model,
                grid_size=grid_size,
                deterministic=True,
                provider_priority=provider_priority,
            )
            if verbose:
                print(f"Loaded policy: {policy_model}")
                print(f"  Provider: {policy.active_provider}")
        except Exception as e:
            print(f"Warning: Failed to load policy: {e}")
            print("Using random actions")

    # Load normalizer if provided
    normalizer = None
    if normalization_stats:
        try:
            normalizer = ObservationNormalizer(normalization_stats)
            if verbose:
                print(f"Loaded normalization stats: {normalization_stats}")
        except Exception as e:
            print(f"Warning: Failed to load normalizer: {e}")

    # Initialize adaptive scheduler if enabled
    scheduler = None
    tiny_detector = None
    if adaptive_mode and not oracle_mode:
        # Load tiny detector if provided
        if tiny_detector_path:
            tiny_detector = TinyDetector(tiny_detector_path)
            if tiny_detector.enabled:
                if verbose:
                    print(f"Loaded tiny detector: {tiny_detector_path}")
            else:
                tiny_detector = None
                if verbose:
                    print(f"Warning: Failed to load tiny detector: {tiny_detector_path}")

        if adaptive_base_skip is not None:
            scheduler = AdaptiveScheduler(base_skip=adaptive_base_skip, tiny_detector=tiny_detector)
            if verbose:
                print(f"Adaptive mode enabled (base_skip={adaptive_base_skip})")
        elif detector is not None:
            # Auto-calibrate based on device speed
            base_skip, latency_ms = calibrate_device(detector)
            scheduler = AdaptiveScheduler(base_skip=base_skip, tiny_detector=tiny_detector)
            if verbose:
                print(f"Adaptive mode enabled (auto-calibrated: {latency_ms:.1f}ms -> base_skip={base_skip})")
        else:
            scheduler = AdaptiveScheduler(base_skip=3, tiny_detector=tiny_detector)
            if verbose:
                print("Adaptive mode enabled (default base_skip=3)")

    # Setup output directory
    if render_output and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Statistics
    all_stats = {
        "rewards": [],
        "scores": [],
        "hits": [],
        "misses": [],
        "lengths": [],
        "timing": {
            "render": [],
            "detect": [],
            "track": [],
            "policy": [],
        },
        "adaptive": {
            "tier_counts": {0: 0, 1: 0, 2: 0},
            "uncertainties": [],
        },
    }

    if verbose and detect_interval > 1 and scheduler is None:
        print(f"Detection interval: every {detect_interval} frames (frame skipping enabled)")

    for episode in range(num_episodes):
        # Reset for new episode
        game_state.reset()
        renderer.reset_background()
        tracker.reset()
        if scheduler is not None:
            scheduler.reset()

        episode_reward = 0.0
        step = 0

        if verbose:
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        while not game_state.game_over:
            # Timing instrumentation
            t0 = time.time()

            # Render frame
            frame = renderer.render(game_state)
            t_render = time.time() - t0

            # Get detections (with optional frame skipping or adaptive scheduling)
            t0 = time.time()

            if scheduler is not None:
                # Adaptive mode: use scheduler to decide detection tier
                kalman_uncertainty = tracker.get_max_uncertainty()
                max_urg = compute_max_urgency(tracker)
                detection_tier = scheduler.get_detection_tier(frame, kalman_uncertainty, max_urg)
            else:
                # Fixed interval mode
                run_detection = (step % detect_interval == 0) or oracle_mode or detector is None
                detection_tier = 2 if run_detection else 0

            if detection_tier == 2:
                # Tier 2: Full detection
                if oracle_mode or detector is None:
                    # Oracle mode: use ground truth
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
                    # Detector mode: run NanoDet
                    detections = detector.detect(frame)
            elif detection_tier == 1 and tiny_detector is not None:
                # Tier 1: Run tiny detector at each predicted track location
                detections = []
                for track in tracker.get_tracks_for_observation():
                    pred_x, pred_y = track.center
                    track_detections = tiny_detector.detect_at_roi(frame, pred_x, pred_y)

                    # Clamp detection bbox to be within 2x of track's bbox
                    # This prevents IoU failure from tiny detector's different bbox scale
                    # (tiny detector outputs bbox sizes up to 3x different from NanoDet)
                    for det in track_detections:
                        min_scale, max_scale = 0.7, 1.4  # Tightened to reduce T1 bbox divergence
                        det.w = float(np.clip(det.w, track.bbox_size[0] * min_scale,
                                                     track.bbox_size[0] * max_scale))
                        det.h = float(np.clip(det.h, track.bbox_size[1] * min_scale,
                                                     track.bbox_size[1] * max_scale))

                    detections.extend(track_detections)
            else:
                # Tier 0: Skip detection entirely
                detections = None  # Sentinel to indicate no detection ran
            t_detect = time.time() - t0

            # Update tracker
            t0 = time.time()
            if detections is None and scheduler is not None:
                # Adaptive Tier 0: Just advance predictions, don't penalize tracks
                confirmed_tracks = tracker.predict_only()
            else:
                # Tier 1/2, or skip-N mode: Full update with detection verification
                # Note: skip-N passes [] which penalizes tracks, but the policy
                # was trained expecting this behavior (tracks invisible on skip frames)
                confirmed_tracks = tracker.update(detections if detections else [])
            t_track = time.time() - t0

            # Update scheduler with track count (for has_active_tracks state)
            if scheduler is not None and detection_tier > 0:
                scheduler.mark_detection_complete(len(confirmed_tracks))

            # Build observation
            if oracle_mode:
                obs = build_oracle_observation(
                    drones=game_state.drones,
                    ammo_fraction=game_state.ammo_fraction,
                    reload_fraction=game_state.reload_fraction,
                    frame_fraction=game_state.frame_fraction,
                    threat_level=game_state.threat_level,
                    grid_size=grid_size,
                )
            else:
                obs = build_tracker_observation(tracker, grid_size=grid_size)
                # Fill in game state values
                obs["game_state"][0] = game_state.ammo_fraction
                obs["game_state"][1] = game_state.reload_fraction
                obs["game_state"][2] = game_state.frame_fraction

            # Normalize observation
            if normalizer:
                obs = normalizer.normalize(obs)

            # Get action
            # Discrete(65): 0 = wait, 1-64 = fire at grid position
            t0 = time.time()
            if policy:
                action_idx, grid_coords = policy.predict(obs)
            else:
                # Random action: 20% wait, 80% fire at random cell
                if np.random.random() < 0.2:
                    action_idx = 0
                    grid_coords = None
                else:
                    action_idx = np.random.randint(1, grid_size * grid_size + 1)
                    cell_idx = action_idx - 1
                    grid_coords = (cell_idx % grid_size, cell_idx // grid_size)
            t_policy = time.time() - t0

            # Collect timing stats
            all_stats["timing"]["render"].append(t_render)
            all_stats["timing"]["detect"].append(t_detect)
            all_stats["timing"]["track"].append(t_track)
            all_stats["timing"]["policy"].append(t_policy)

            # Collect adaptive stats
            if scheduler is not None:
                all_stats["adaptive"]["tier_counts"][detection_tier] += 1
                all_stats["adaptive"]["uncertainties"].append(kalman_uncertainty)

            # Execute action
            reward = 0.01  # Survival reward per frame
            grid_x, grid_y = None, None

            if action_idx > 0 and grid_coords is not None:
                # Fire action
                grid_x, grid_y = grid_coords
                hit, drone = game_state.fire(grid_x, grid_y, grid_size)
                if hit:
                    reward += 2.0 if drone and drone.is_kamikaze else 1.0
                else:
                    reward -= 0.1  # Match original (was 0.3)

            # Step game
            game_state.step()

            # Game over rewards/penalties
            if game_state.game_over:
                if game_state.game_over_reason == "Kamikaze impact!":
                    reward -= 5.0
                elif game_state.game_over_reason == "Episode complete!":
                    reward += 3.0

            episode_reward += reward
            step += 1

            # Save frame periodically
            if render_output and output_dir and step % save_interval == 0:
                overlay_frame = renderer.render_with_overlay(
                    game_state,
                    grid_size=grid_size,
                    highlight_cell=(grid_x, grid_y) if action_idx > 0 else None,
                    detections=detections,
                )
                img = Image.fromarray(overlay_frame)
                img.save(output_path / f"ep{episode:02d}_frame{step:04d}.png")

        # Episode complete
        all_stats["rewards"].append(episode_reward)
        all_stats["scores"].append(game_state.score)
        all_stats["hits"].append(game_state.hits)
        all_stats["misses"].append(game_state.misses)
        all_stats["lengths"].append(step)

        if verbose:
            hit_rate = game_state.hits / max(1, game_state.hits + game_state.misses)
            print(f"Reward: {episode_reward:.2f} | Score: {game_state.score} | "
                  f"Hits: {game_state.hits} | Misses: {game_state.misses} | "
                  f"Hit Rate: {hit_rate:.1%} | Reason: {game_state.game_over_reason}")

    # Summary
    if verbose:
        print(f"\n{'=' * 50}")
        print("SIMULATION SUMMARY")
        print(f"{'=' * 50}")
        print(f"Episodes: {num_episodes}")
        print(f"Mean Reward: {np.mean(all_stats['rewards']):.2f} +/- {np.std(all_stats['rewards']):.2f}")
        print(f"Mean Score: {np.mean(all_stats['scores']):.1f}")
        print(f"Mean Hits: {np.mean(all_stats['hits']):.1f}")
        print(f"Mean Misses: {np.mean(all_stats['misses']):.1f}")
        total_hits = sum(all_stats["hits"])
        total_misses = sum(all_stats["misses"])
        print(f"Overall Hit Rate: {total_hits / max(1, total_hits + total_misses):.1%}")

        # Timing summary
        if all_stats["timing"]["render"]:
            print(f"\n{'=' * 50}")
            print("TIMING BREAKDOWN (ms)")
            print(f"{'=' * 50}")
            print(f"Render:  {np.mean(all_stats['timing']['render'])*1000:6.2f} ms (avg)")
            print(f"Detect:  {np.mean(all_stats['timing']['detect'])*1000:6.2f} ms (avg)")
            print(f"Track:   {np.mean(all_stats['timing']['track'])*1000:6.2f} ms (avg)")
            print(f"Policy:  {np.mean(all_stats['timing']['policy'])*1000:6.2f} ms (avg)")
            total_time = (
                np.mean(all_stats['timing']['render']) +
                np.mean(all_stats['timing']['detect']) +
                np.mean(all_stats['timing']['track']) +
                np.mean(all_stats['timing']['policy'])
            )
            print(f"Total:   {total_time*1000:6.2f} ms/frame")
            print(f"Est FPS: {1.0/total_time:.1f}")
            if detect_interval > 1 and scheduler is None:
                print(f"\n(Detection running every {detect_interval} frames)")

        # Adaptive scheduler stats
        if scheduler is not None and all_stats["adaptive"]["uncertainties"]:
            print(f"\n{'=' * 50}")
            print("ADAPTIVE SCHEDULER STATS")
            print(f"{'=' * 50}")
            tier_counts = all_stats["adaptive"]["tier_counts"]
            total_frames = sum(tier_counts.values())
            print(f"Tier 0 (skip):  {tier_counts[0]:5d} ({tier_counts[0]/total_frames*100:5.1f}%)")
            print(f"Tier 1 (tiny):  {tier_counts[1]:5d} ({tier_counts[1]/total_frames*100:5.1f}%)")
            print(f"Tier 2 (full):  {tier_counts[2]:5d} ({tier_counts[2]/total_frames*100:5.1f}%)")
            detection_rate = (tier_counts[1] + tier_counts[2]) / total_frames * 100
            print(f"Detection rate: {detection_rate:.1f}%")

            # Tier 0 and Tier 2 reason breakdowns
            stats = scheduler.get_stats()

            tier0_reasons = stats["tier0_reasons"]
            tier0_total = tier_counts[0]
            if tier0_total > 0:
                print(f"\nTier 0 Breakdown:")
                for reason, count in tier0_reasons.items():
                    if count > 0:
                        pct = count / tier0_total * 100
                        print(f"  {reason:20s}: {count:5d} ({pct:5.1f}%)")

            tier2_reasons = stats["tier2_reasons"]
            tier2_total = tier_counts[2]
            if tier2_total > 0:
                print(f"\nTier 2 Breakdown:")
                for reason, count in tier2_reasons.items():
                    if count > 0:
                        pct = count / tier2_total * 100
                        print(f"  {reason:20s}: {count:5d} ({pct:5.1f}%)")

            uncertainties = all_stats["adaptive"]["uncertainties"]
            print(f"\nUncertainty: mean={np.mean(uncertainties):.3f} "
                  f"std={np.std(uncertainties):.3f} "
                  f"max={np.max(uncertainties):.3f}")

        print(f"{'=' * 50}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Run Drone Hunter simulation on edge device"
    )
    parser.add_argument(
        "--policy", type=str, default=None,
        help="Path to ONNX policy model"
    )
    parser.add_argument(
        "--detector", type=str, default=None,
        help="Path to detector model (ONNX or NCNN)"
    )
    parser.add_argument(
        "--normalization", type=str, default=None,
        help="Path to normalization.json"
    )
    parser.add_argument(
        "--backend", type=str, default="onnx",
        choices=["onnx", "ncnn"],
        help="Inference backend"
    )
    parser.add_argument(
        "--oracle", action="store_true",
        help="Use oracle mode (ground truth instead of detector)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes"
    )
    parser.add_argument(
        "--max-frames", type=int, default=1000,
        help="Maximum frames per episode"
    )
    parser.add_argument(
        "--grid-size", type=int, default=8,
        help="Firing grid size"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save rendered frames"
    )
    parser.add_argument(
        "--save-interval", type=int, default=50,
        help="Save frame every N steps"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )
    parser.add_argument(
        "--detect-interval", type=int, default=1,
        help="Run detector every N frames (1=every frame, 3=skip 2, etc). "
             "Higher values improve FPS but reduce detection accuracy."
    )
    parser.add_argument(
        "--provider", type=str, default="auto",
        choices=["auto", "desktop", "mobile", "mobile-npu", "nnapi", "xnnpack", "cuda", "cpu"],
        help="Execution provider: auto (detect platform), desktop (CUDA/CPU), "
             "mobile (XNNPACK/CPU for budget phones), mobile-npu (NNAPI/XNNPACK/CPU for flagships), "
             "or specific provider"
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Use adaptive scheduler instead of fixed detect-interval. "
             "Dynamically adjusts detection based on Kalman uncertainty and motion."
    )
    parser.add_argument(
        "--adaptive-base-skip", type=int, default=None,
        help="Override base_skip for adaptive mode (default: auto-calibrate based on device speed)"
    )
    parser.add_argument(
        "--tiny-detector", type=str, default=None,
        help="Path to tiny detector ONNX model for Tier 1 detection in adaptive mode"
    )

    args = parser.parse_args()

    start_time = time.time()

    stats = run_simulation(
        policy_model=args.policy,
        detector_model=args.detector,
        normalization_stats=args.normalization,
        backend_type=args.backend,
        oracle_mode=args.oracle,
        num_episodes=args.episodes,
        max_frames=args.max_frames,
        grid_size=args.grid_size,
        render_output=args.output_dir is not None,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        verbose=not args.quiet,
        detect_interval=args.detect_interval,
        provider_priority=args.provider,
        adaptive_mode=args.adaptive,
        adaptive_base_skip=args.adaptive_base_skip,
        tiny_detector_path=args.tiny_detector,
    )

    elapsed = time.time() - start_time
    if not args.quiet:
        print(f"\nTotal time: {elapsed:.1f}s")
        total_frames = sum(stats["lengths"])
        print(f"Average FPS: {total_frames / elapsed:.1f}")


if __name__ == "__main__":
    main()
