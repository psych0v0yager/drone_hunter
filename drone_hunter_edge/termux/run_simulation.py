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
)


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
        detect_interval: Run detector every N frames (1=every frame).
        provider_priority: Execution provider selection.

    Returns:
        Dictionary with episode statistics.
    """
    # Initialize components
    game_state = GameState(max_frames=max_frames)
    renderer = Renderer(width=320, height=320)
    tracker = KalmanTracker()

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
    }

    if verbose and detect_interval > 1:
        print(f"Detection interval: every {detect_interval} frames (frame skipping enabled)")

    for episode in range(num_episodes):
        # Reset for new episode
        game_state.reset()
        renderer.reset_background()
        tracker.reset()

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

            # Get detections (with optional frame skipping)
            t0 = time.time()
            run_detection = (step % detect_interval == 0) or oracle_mode or detector is None

            if run_detection:
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
            else:
                # Skip detection, tracker will predict
                detections = []
            t_detect = time.time() - t0

            # Update tracker
            t0 = time.time()
            tracker.update(detections)
            t_track = time.time() - t0

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
            if detect_interval > 1:
                print(f"\n(Detection running every {detect_interval} frames)")
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
    )

    elapsed = time.time() - start_time
    if not args.quiet:
        print(f"\nTotal time: {elapsed:.1f}s")
        total_frames = sum(stats["lengths"])
        print(f"Average FPS: {total_frames / elapsed:.1f}")


if __name__ == "__main__":
    main()
