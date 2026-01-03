"""Drone Hunter Gymnasium environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_hunter.envs.game_state import GameState
from drone_hunter.envs.renderer import SpriteRenderer


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
        assets_dir: Path | str | None = None,
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
            assets_dir: Directory containing assets (backgrounds, drones).
        """
        super().__init__()

        self.render_mode = render_mode
        self.grid_size = grid_size
        self.max_drones = max_drones
        self.frame_stack = frame_stack
        self.width = width
        self.height = height
        self.oracle_mode = oracle_mode
        self.single_target_mode = single_target_mode

        # Initialize game state
        self.game_state = GameState(max_frames=max_frames)

        # Initialize renderer
        assets_path = Path(assets_dir) if assets_dir else None
        backgrounds_dir = assets_path / "backgrounds" if assets_path else None
        drones_dir = assets_path / "drones" if assets_path else None

        self.renderer = SpriteRenderer(
            width=width,
            height=height,
            backgrounds_dir=backgrounds_dir,
            drones_dir=drones_dir,
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
            features_per_drone = 6 + grid_size * 2  # 22 for 8x8 grid
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

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation."""
        if self.single_target_mode:
            # Simple single-target observation
            return self.game_state.get_single_target_observation(self.grid_size)
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
        return {
            "score": self.game_state.score,
            "hits": self.game_state.hits,
            "misses": self.game_state.misses,
            "frame": self.game_state.frame_count,
            "ammo": self.game_state.ammo,
            "num_drones": len(self.game_state.drones),
            "threat_level": self.game_state.threat_level,
            "game_over_reason": self.game_state.game_over_reason,
        }

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

        # Reset renderer background
        self.renderer.reset_background()

        # Reset tracking
        self._last_action = None
        self._last_hit = False

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
