#!/usr/bin/env python3
"""Live display simulation for Drone Hunter edge inference.

Runs the simulation with real-time pygame display for Termux X11.
"""

import argparse
import sys
import time
import threading
from pathlib import Path
from typing import Optional, List

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for pygame
try:
    import pygame
except ImportError:
    print("pygame not installed. Install with: pip install pygame")
    sys.exit(1)

from PIL import Image

from core.game_state import GameState
from core.renderer import Renderer
from core.kalman_tracker import KalmanTracker
from core.detection import Detection
from core.observation import (
    ObservationNormalizer,
    build_tracker_observation,
    build_oracle_observation,
)


class AsyncDetector:
    """Background thread detector for non-blocking inference."""

    def __init__(self, detector):
        self.detector = detector
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_detections: List = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.detection_count = 0

    def start(self):
        """Start the background detection thread."""
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the background detection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _worker(self):
        """Background worker that continuously runs detection."""
        while self.running:
            # Grab latest frame
            with self.lock:
                frame = self.latest_frame

            if frame is not None:
                # Run detection (this is the slow part)
                detections = self.detector.detect(frame)

                # Store results
                with self.lock:
                    self.latest_detections = detections
                    self.detection_count += 1
            else:
                # No frame yet, wait a bit
                time.sleep(0.01)

    def update_frame(self, frame: np.ndarray):
        """Update the frame for detection (called from main thread)."""
        with self.lock:
            self.latest_frame = frame

    def get_detections(self) -> List:
        """Get latest detections (called from main thread, non-blocking)."""
        with self.lock:
            return self.latest_detections


def run_live(
    policy_model: Optional[str] = None,
    detector_model: Optional[str] = None,
    normalization_stats: Optional[str] = None,
    backend_type: str = "onnx",
    oracle_mode: bool = False,
    max_frames: int = 1000,
    grid_size: int = 8,
    display_scale: int = 2,
    target_fps: int = 30,
    detect_interval: int = 1,
    async_detect: bool = False,
) -> None:
    """Run the drone hunter simulation with live display.

    Args:
        policy_model: Path to ONNX policy model (None for random actions).
        detector_model: Path to detector model (None for oracle mode).
        normalization_stats: Path to normalization.json.
        backend_type: Inference backend ("onnx" or "ncnn").
        oracle_mode: Use ground truth instead of detector.
        max_frames: Maximum frames per episode.
        grid_size: Size of firing grid.
        display_scale: Scale factor for display window.
        target_fps: Target frames per second.
        detect_interval: Run detector every N frames (1=every frame, 2=skip one, etc).
        async_detect: Run detector in background thread (non-blocking).
    """
    # Initialize components
    game_state = GameState(max_frames=max_frames)
    renderer = Renderer(width=320, height=320)
    tracker = KalmanTracker()

    # Load detector if provided
    detector = None
    async_detector = None
    if detector_model and not oracle_mode:
        try:
            from inference.nanodet import create_detector
            detector = create_detector(
                detector_model,
                backend_type=backend_type,
                conf_threshold=0.60,
                iou_threshold=0.5,
            )
            print(f"Loaded detector: {detector_model} ({backend_type})")

            # Wrap in async detector if enabled
            if async_detect:
                async_detector = AsyncDetector(detector)
                async_detector.start()
                print("Async detection enabled (background thread)")
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
            )
            print(f"Loaded policy: {policy_model}")
        except Exception as e:
            print(f"Warning: Failed to load policy: {e}")
            print("Using random actions")

    # Load normalizer if provided
    normalizer = None
    if normalization_stats:
        try:
            normalizer = ObservationNormalizer(normalization_stats)
            print(f"Loaded normalization stats: {normalization_stats}")
        except Exception as e:
            print(f"Warning: Failed to load normalizer: {e}")

    # Initialize pygame
    pygame.init()
    display_width = 320 * display_scale
    display_height = 320 * display_scale
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Drone Hunter Edge")
    clock = pygame.time.Clock()

    # Font for HUD
    try:
        font = pygame.font.Font(None, 24 * display_scale // 2)
    except:
        font = pygame.font.SysFont("monospace", 16 * display_scale // 2)

    running = True
    episode = 0
    frame_time = 1.0 / target_fps

    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset episode")
    print("  Q/ESC - Quit")
    print(f"\nTarget FPS: {target_fps}")
    if detect_interval > 1:
        print(f"Detection interval: every {detect_interval} frames (frame skipping enabled)")

    while running:
        episode += 1
        print(f"\n--- Episode {episode} ---")

        # Reset for new episode
        game_state.reset()
        renderer.reset_background()
        tracker.reset()

        episode_reward = 0.0
        step = 0
        paused = False

        while not game_state.game_over and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("PAUSED" if paused else "RESUMED")
                    elif event.key == pygame.K_r:
                        game_state.game_over = True  # Force reset
                        print("Resetting...")

            if paused:
                clock.tick(10)
                continue

            start_time = time.time()

            # Timing instrumentation
            t0 = time.perf_counter()

            # Render frame
            frame = renderer.render(game_state)
            t_render = time.perf_counter() - t0

            # Get detections
            t0 = time.perf_counter()

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
            elif async_detector is not None:
                # Async mode: update frame and get latest results (non-blocking)
                async_detector.update_frame(frame)
                detections = async_detector.get_detections()
            else:
                # Sync mode with optional frame skipping
                run_detection = (step % detect_interval == 0)
                if run_detection:
                    detections = detector.detect(frame)
                else:
                    detections = []
            t_detect = time.perf_counter() - t0

            # Update tracker
            t0 = time.perf_counter()
            tracker.update(detections)
            t_track = time.perf_counter() - t0

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
                obs["game_state"][0] = game_state.ammo_fraction
                obs["game_state"][1] = game_state.reload_fraction
                obs["game_state"][2] = game_state.frame_fraction

            # Normalize observation
            if normalizer:
                obs = normalizer.normalize(obs)

            # Get action
            t0 = time.perf_counter()
            if policy:
                action_idx, grid_coords = policy.predict(obs)
            else:
                if np.random.random() < 0.2:
                    action_idx = 0
                    grid_coords = None
                else:
                    action_idx = np.random.randint(1, grid_size * grid_size + 1)
                    cell_idx = action_idx - 1
                    grid_coords = (cell_idx % grid_size, cell_idx // grid_size)
            t_policy = time.perf_counter() - t0

            # Execute action
            reward = 0.01
            grid_x, grid_y = None, None

            if action_idx > 0 and grid_coords is not None:
                grid_x, grid_y = grid_coords
                hit, drone = game_state.fire(grid_x, grid_y, grid_size)
                if hit:
                    reward += 2.0 if drone and drone.is_kamikaze else 1.0
                else:
                    reward -= 0.1

            # Step game
            game_state.step()

            if game_state.game_over:
                if game_state.game_over_reason == "Kamikaze impact!":
                    reward -= 5.0
                elif game_state.game_over_reason == "Episode complete!":
                    reward += 3.0

            episode_reward += reward
            step += 1

            # Render with overlay (reuse pre-rendered frame)
            t0 = time.perf_counter()
            overlay_frame = renderer.render_with_overlay(
                game_state,
                grid_size=grid_size,
                highlight_cell=(grid_x, grid_y) if action_idx > 0 else None,
                detections=detections,
                frame=frame,  # Reuse frame from detection step
            )

            # Convert to pygame surface (direct numpy -> pygame, skip PIL)
            # pygame.surfarray expects (width, height, channels), numpy is (height, width, channels)
            pygame_surface = pygame.surfarray.make_surface(overlay_frame.swapaxes(0, 1))
            if display_scale != 1:
                pygame_surface = pygame.transform.scale(pygame_surface, (display_width, display_height))
            screen.blit(pygame_surface, (0, 0))

            # Draw additional HUD
            mode_text = "ORACLE" if oracle_mode else "DETECTOR"
            hud_lines = [
                f"Episode: {episode}  Frame: {step}/{max_frames}",
                f"Mode: {mode_text}  Reward: {episode_reward:.1f}",
            ]

            y_offset = 5
            for line in hud_lines:
                text_surface = font.render(line, True, (255, 255, 0))
                # Draw with shadow for visibility
                shadow = font.render(line, True, (0, 0, 0))
                screen.blit(shadow, (display_width - text_surface.get_width() - 4, y_offset + 1))
                screen.blit(text_surface, (display_width - text_surface.get_width() - 5, y_offset))
                y_offset += text_surface.get_height() + 2

            pygame.display.flip()
            t_display = time.perf_counter() - t0

            # Print timing every 30 frames
            if step % 30 == 0:
                total = t_render + t_detect + t_track + t_policy + t_display
                fps_est = 1000.0 / (total * 1000) if total > 0 else 0
                print(f"[{step:4d}] render={t_render*1000:5.1f}ms detect={t_detect*1000:5.1f}ms "
                      f"track={t_track*1000:4.1f}ms policy={t_policy*1000:4.1f}ms "
                      f"display={t_display*1000:5.1f}ms | total={total*1000:5.1f}ms ({fps_est:.0f} FPS)")

            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            clock.tick(target_fps * 2)  # Allow some headroom

        # Episode complete
        if running:
            hit_rate = game_state.hits / max(1, game_state.hits + game_state.misses)
            print(f"Reward: {episode_reward:.2f} | Score: {game_state.score} | "
                  f"Hits: {game_state.hits} | Misses: {game_state.misses} | "
                  f"Hit Rate: {hit_rate:.1%} | Reason: {game_state.game_over_reason}")

            # Brief pause between episodes
            time.sleep(0.5)

    # Cleanup
    if async_detector is not None:
        async_detector.stop()
        print(f"Async detector ran {async_detector.detection_count} detections")

    pygame.quit()
    print("\nSimulation ended.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Drone Hunter with live pygame display"
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
        "--max-frames", type=int, default=1000,
        help="Maximum frames per episode"
    )
    parser.add_argument(
        "--grid-size", type=int, default=8,
        help="Firing grid size"
    )
    parser.add_argument(
        "--scale", type=int, default=2,
        help="Display scale factor (default: 2x)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Target frames per second"
    )
    parser.add_argument(
        "--detect-interval", type=int, default=1,
        help="Run detector every N frames (1=every frame, 3=skip 2, etc). "
             "Higher values improve FPS but reduce detection accuracy."
    )
    parser.add_argument(
        "--async-detect", action="store_true",
        help="Run detector in background thread (non-blocking). "
             "Enables smooth display FPS independent of detection speed."
    )

    args = parser.parse_args()

    run_live(
        policy_model=args.policy,
        detector_model=args.detector,
        normalization_stats=args.normalization,
        backend_type=args.backend,
        oracle_mode=args.oracle,
        max_frames=args.max_frames,
        grid_size=args.grid_size,
        display_scale=args.scale,
        target_fps=args.fps,
        detect_interval=args.detect_interval,
        async_detect=args.async_detect,
    )


if __name__ == "__main__":
    main()
