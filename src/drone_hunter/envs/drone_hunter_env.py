"""Drone Hunter Gymnasium environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_hunter.envs.difficulty import DifficultyConfig
from drone_hunter.envs.game_state import GameState
from drone_hunter.envs.renderer import SpriteRenderer
from drone_hunter.tracking import Detection, KalmanTracker


class DroneHunterEnv(gym.Env):
    """Gymnasium environment for Drone Hunter.

    An RL environment where an agent must detect and shoot down drones.
    Supports both oracle mode (ground truth positions) and detection mode
    (using detector outputs).

    Action Space:
        Discrete(grid_size^2 + 1): Fire at grid cell or wait (action 0)

    Observation Space (Oracle Mode):
        Dict with:
        - "drones": Box(-1, 1, (4, max_drones, 6)) - stacked drone detections
        - "game_state": Box(0, 1, (4,)) - ammo, reload, frame, threat

    Rewards:
        +1.0 for hitting normal drone
        +2.0 for hitting kamikaze drone
        -0.1 for missing a shot
        -5.0 for kamikaze impact (game over)
        +0.01 per frame survived
        -0.5 for empty clip with active threat
        +3.0 for completing episode
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        grid_size: int = 8,
        max_drones: int = 10,
        frame_stack: int = 4,
        width: int = 320,
        height: int = 320,
        max_frames: int = 1000,
        oracle_mode: bool = True,
        single_target_mode: bool = False,
        detection_noise: float = 0.02,
        detection_dropout: float = 0.05,
        assets_dir: Path | str | None = None,
        detector_model: Path | str | None = None,
        difficulty: str | DifficultyConfig | None = None,
    ):
        """Initialize Drone Hunter environment.

        Args:
            render_mode: "human" for display, "rgb_array" for training.
            grid_size: Size of the firing grid (grid_size x grid_size).
            max_drones: Maximum drones to track per frame.
            frame_stack: Number of frames to stack for temporal context.
            width: Render width in pixels.
            height: Render height in pixels.
            max_frames: Maximum frames per episode.
            oracle_mode: If True, use ground truth positions (no detector).
            single_target_mode: If True, only pass most urgent drone (simplified).
            detection_noise: Noise std for simulated detector (position/size jitter).
            detection_dropout: Probability of missing a detection per frame.
            assets_dir: Directory containing assets (backgrounds, drones).
            detector_model: Path to ONNX detector model. If provided, uses real
                detector instead of simulated one (only in detector mode).
            difficulty: Visual difficulty preset or DifficultyConfig instance.
                Options: "easy" (default), "medium", "hard", "forest", "urban".
                Only affects visuals/detector training, not core gameplay.
        """
        super().__init__()

        # Parse difficulty config
        if difficulty is None:
            self.difficulty = DifficultyConfig.easy()
        elif isinstance(difficulty, str):
            self.difficulty = DifficultyConfig.from_name(difficulty)
        else:
            self.difficulty = difficulty

        self.render_mode = render_mode
        self.grid_size = grid_size
        self.max_drones = max_drones
        self.frame_stack = frame_stack
        self.width = width
        self.height = height
        self.oracle_mode = oracle_mode
        self.single_target_mode = single_target_mode
        self.detection_noise = detection_noise
        self.detection_dropout = detection_dropout

        # Kalman tracker for detector mode
        self.tracker = KalmanTracker(max_age=5, min_hits=2)

        # Initialize game state
        self.game_state = GameState(max_frames=max_frames)

        # Configure distractor settings from difficulty
        self.game_state.distractors_enabled = self.difficulty.distractors_enabled
        self.game_state.distractor_spawn_rate = self.difficulty.distractor_spawn_rate
        self.game_state.distractor_types = list(self.difficulty.distractor_types)
        self.game_state.static_obstacles_enabled = self.difficulty.static_obstacles_enabled
        self.game_state.static_obstacle_types = list(self.difficulty.static_obstacle_types)

        # Initialize renderer
        assets_path = Path(assets_dir) if assets_dir else None
        backgrounds_dir = assets_path / "backgrounds" if assets_path else None
        drones_dir = assets_path / "drones" if assets_path else None

        self.renderer = SpriteRenderer(
            width=width,
            height=height,
            backgrounds_dir=backgrounds_dir,
            drones_dir=drones_dir,
            difficulty_config=self.difficulty,
        )

        # Initialize real detector if model path provided
        self.detector = None
        if detector_model is not None:
            from drone_hunter.detection import NanoDetONNX
            self.detector = NanoDetONNX(
                str(detector_model),
                input_size=(height, width),
                conf_threshold=0.55,  # Higher threshold for single-class models
                iou_threshold=0.5,
            )

        # Frame buffer for stacking (only used in multi-drone mode)
        self._frame_buffer: list[np.ndarray] = []

        # Action space: 0 = wait, 1 to grid_size^2 = fire at cell
        self.action_space = spaces.Discrete(grid_size * grid_size + 1)

        # Observation space depends on mode
        if single_target_mode:
            # Single target mode: just one drone (the most urgent)
            # Features: z, vz, urgency, grid_x_onehot[8], grid_y_onehot[8]
            # Note: is_kamikaze removed - agent infers threat from vz
            features_per_target = 3 + grid_size * 2  # 19 for 8x8 grid
            self.observation_space = spaces.Dict({
                "target": spaces.Box(
                    low=-1.0,
                    high=2.0,
                    shape=(features_per_target,),
                    dtype=np.float32,
                ),
                "game_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(5,),  # ammo, reload, frame, threat, has_target
                    dtype=np.float32,
                ),
            })
        else:
            # Multi-drone mode (original)
            # Features: x, y, z, vz, urgency, grid_x_onehot[8], grid_y_onehot[8]
            # Note: is_kamikaze removed - agent infers threat from vz
            features_per_drone = 5 + grid_size * 2  # 21 for 8x8 grid
            self.observation_space = spaces.Dict({
                "detections": spaces.Box(
                    low=-1.0,
                    high=2.0,
                    shape=(frame_stack, max_drones, features_per_drone),
                    dtype=np.float32,
                ),
                "game_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(5,),  # ammo, reload, frame, threat, n_drones
                    dtype=np.float32,
                ),
            })

        # For rendering
        self._last_action: int | None = None
        self._last_hit: bool = False

    def _generate_detections(self) -> list[Detection]:
        """Generate detections using real detector or simulation.

        If a real detector model was provided, renders the current frame
        and runs inference. Otherwise, generates simulated detections
        from ground truth with noise.

        Returns:
            List of Detection objects.
        """
        if self.detector is not None:
            # Use real detector on rendered frame
            frame = self.renderer.render(self.game_state)
            return self.detector.detect(frame)
        else:
            # Use simulated detector
            return self._generate_simulated_detections()

    def _generate_simulated_detections(self) -> list[Detection]:
        """Generate simulated detections from ground truth drones.

        Adds noise to positions and sizes to simulate real detector behavior.
        Also randomly drops some detections to simulate missed detections.

        Returns:
            List of Detection objects with noisy measurements.
        """
        detections = []
        rng = np.random.default_rng()

        for drone in self.game_state.drones:
            # Random dropout (simulate missed detections)
            if rng.random() < self.detection_dropout:
                continue

            # Get ground truth bbox
            x1, y1, x2, y2 = drone.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Add noise to position and size
            cx_noisy = cx + rng.normal(0, self.detection_noise)
            cy_noisy = cy + rng.normal(0, self.detection_noise)
            w_noisy = w * (1 + rng.normal(0, self.detection_noise * 2))
            h_noisy = h * (1 + rng.normal(0, self.detection_noise * 2))

            # Clamp to valid ranges
            cx_noisy = np.clip(cx_noisy, 0, 1)
            cy_noisy = np.clip(cy_noisy, 0, 1)
            w_noisy = max(0.01, w_noisy)
            h_noisy = max(0.01, h_noisy)

            # Confidence based on size (larger = more confident)
            confidence = min(1.0, 0.5 + h_noisy * 2)

            detections.append(Detection(
                x=cx_noisy,
                y=cy_noisy,
                w=w_noisy,
                h=h_noisy,
                confidence=confidence,
            ))

        return detections

    def _get_tracker_observation(self) -> dict[str, np.ndarray]:
        """Get observation from Kalman tracker (detector mode).

        Uses tracker-estimated z and vz instead of ground truth.

        Returns:
            Observation dict with tracker-based features.
        """
        # Get active tracks sorted by urgency
        tracks = self.tracker.get_tracks_for_observation()

        # Features: z, vz, urgency, grid_x_onehot[8], grid_y_onehot[8]
        features_per_target = 3 + self.grid_size * 2  # 19 for grid_size=8
        target_obs = np.zeros(features_per_target, dtype=np.float32)

        has_target = 0.0

        if tracks:
            # Get the #1 most urgent track
            track = tracks[0]
            has_target = 1.0

            # Compute grid cell from track position
            cx, cy = track.center
            grid_x = min(self.grid_size - 1, max(0, int(cx * self.grid_size)))
            grid_y = min(self.grid_size - 1, max(0, int(cy * self.grid_size)))

            # Compute urgency from tracker estimates
            z = track.z
            vz = track.vz

            if vz < 0:  # Approaching
                frames_to_impact = z / max(0.001, abs(vz))
                urgency = 1.0 / (1.0 + frames_to_impact / 50.0)
            else:
                urgency = 0.1

            # Fill features
            target_obs[0] = z
            target_obs[1] = vz * 10  # Scale for gradient
            target_obs[2] = urgency

            # One-hot grid_x (positions 3-10)
            target_obs[3 + grid_x] = 1.0

            # One-hot grid_y (positions 11-18)
            target_obs[3 + self.grid_size + grid_y] = 1.0

        # Game state (threat_level computed from tracker)
        max_threat = 0.0
        for track in tracks:
            if track.vz < 0:
                threat = 1.0 - track.z
                max_threat = max(max_threat, threat)

        game_state_obs = np.array([
            self.game_state.ammo_fraction,
            self.game_state.reload_fraction,
            self.game_state.frame_fraction,
            max_threat,  # Tracker-based threat level
            has_target,
        ], dtype=np.float32)

        return {
            "target": target_obs,
            "game_state": game_state_obs,
        }

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation."""
        if self.single_target_mode:
            if self.oracle_mode:
                # Oracle mode: use ground truth
                return self.game_state.get_single_target_observation(self.grid_size)
            else:
                # Detector mode: use Kalman tracker estimates
                return self._get_tracker_observation()
        else:
            # Multi-drone mode with frame stacking
            while len(self._frame_buffer) < self.frame_stack:
                obs = self.game_state.get_oracle_observation(self.max_drones, self.grid_size)
                self._frame_buffer.append(obs["detections"])

            stacked = np.array(self._frame_buffer[-self.frame_stack:], dtype=np.float32)

            return {
                "detections": stacked,
                "game_state": self.game_state.get_oracle_observation(self.max_drones, self.grid_size)["game_state"],
            }

    def _get_info(self) -> dict[str, Any]:
        """Get auxiliary info about current state."""
        info = {
            "score": self.game_state.score,
            "hits": self.game_state.hits,
            "misses": self.game_state.misses,
            "frame": self.game_state.frame_count,
            "ammo": self.game_state.ammo,
            "num_drones": len(self.game_state.drones),
            "threat_level": self.game_state.threat_level,
            "game_over_reason": self.game_state.game_over_reason,
        }

        # Add tracker info in detector mode
        if not self.oracle_mode:
            confirmed_tracks = [t for t in self.tracker.tracks if t.hits >= 2]
            info["num_tracks"] = len(confirmed_tracks)
            info["tracker_frame"] = self.tracker.frame_count

        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        # Reset game state
        self.game_state.reset()

        # Reset frame buffer
        self._frame_buffer.clear()

        # Reset Kalman tracker
        self.tracker.reset()

        # Reset renderer background
        self.renderer.reset_background()

        # Reset tracking
        self._last_action = None
        self._last_hit = False

        # Initial tracker update if in detector mode
        if not self.oracle_mode:
            detections = self._generate_detections()
            self.tracker.update(detections)

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: 0 = wait, 1-N = fire at grid cell (N-1)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        reward = 0.0
        self._last_action = action
        self._last_hit = False

        # Process action
        if action > 0:
            # Fire at grid cell
            cell_idx = action - 1
            grid_x = cell_idx % self.grid_size
            grid_y = cell_idx // self.grid_size

            hit, drone = self.game_state.fire(grid_x, grid_y, self.grid_size)

            if hit:
                self._last_hit = True
                # Reward based on drone type
                if drone and drone.is_kamikaze:
                    reward += 2.0
                else:
                    reward += 1.0
            else:
                # Miss penalty
                reward -= 0.1

        # Check for empty clip during threat
        if (self.game_state.ammo == 0 and
            not self.game_state.is_reloading and
            self.game_state.has_kamikaze_threat):
            reward -= 0.5

        # Step game state
        self.game_state.step()

        # Survival reward
        reward += 0.01

        # Update tracker with new detections (detector mode only)
        if not self.oracle_mode:
            detections = self._generate_detections()
            self.tracker.update(detections)

        # Update frame buffer (only in multi-drone mode)
        if not self.single_target_mode:
            obs = self.game_state.get_oracle_observation(self.max_drones, self.grid_size)
            self._frame_buffer.append(obs["detections"])
            if len(self._frame_buffer) > self.frame_stack:
                self._frame_buffer.pop(0)

        # Check termination
        terminated = self.game_state.game_over
        truncated = False

        if terminated:
            if self.game_state.game_over_reason == "Kamikaze impact!":
                reward -= 5.0
            elif self.game_state.game_over_reason == "Episode complete!":
                reward += 3.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current state.

        Returns:
            RGB array if render_mode is "rgb_array", None for "human".
        """
        # Determine highlight cell from last action
        highlight_cell = None
        if self._last_action is not None and self._last_action > 0:
            cell_idx = self._last_action - 1
            grid_x = cell_idx % self.grid_size
            grid_y = cell_idx // self.grid_size
            highlight_cell = (grid_x, grid_y)

        frame = self.renderer.render_with_overlay(
            self.game_state,
            grid_size=self.grid_size,
            highlight_cell=highlight_cell,
        )

        if self.render_mode == "human":
            # Display using OpenCV or matplotlib
            try:
                import cv2
                cv2.imshow("Drone Hunter", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            except ImportError:
                pass  # No display available
            return None
        else:
            return frame

    def close(self) -> None:
        """Clean up resources."""
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass

    def action_to_grid(self, action: int) -> tuple[int, int] | None:
        """Convert action index to grid coordinates.

        Args:
            action: Action index (0 = wait, 1+ = fire)

        Returns:
            (grid_x, grid_y) or None if action is wait.
        """
        if action == 0:
            return None
        cell_idx = action - 1
        return (cell_idx % self.grid_size, cell_idx // self.grid_size)

    def grid_to_action(self, grid_x: int, grid_y: int) -> int:
        """Convert grid coordinates to action index.

        Args:
            grid_x: Grid column.
            grid_y: Grid row.

        Returns:
            Action index.
        """
        return grid_y * self.grid_size + grid_x + 1


# Register the environment with Gymnasium
gym.register(
    id="DroneHunter-v0",
    entry_point="drone_hunter.envs:DroneHunterEnv",
    max_episode_steps=1000,
)
