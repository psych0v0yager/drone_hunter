"""Human-playable mode for Drone Hunter.

Controls:
    - Mouse click: Fire at clicked location
    - Space: Wait (do nothing)
    - R: Reset episode
    - Q/Escape: Quit
    - G: Toggle grid overlay
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from drone_hunter.envs import DroneHunterEnv


class HumanPlayer:
    """Human-playable interface for Drone Hunter."""

    def __init__(
        self,
        grid_size: int = 8,
        width: int = 640,
        height: int = 640,
        max_frames: int = 1000,
        fps: int = 30,
    ):
        """Initialize human player interface.

        Args:
            grid_size: Size of the firing grid.
            width: Display width.
            height: Display height.
            max_frames: Maximum frames per episode.
            fps: Target frames per second.
        """
        if not HAS_OPENCV:
            raise ImportError(
                "OpenCV is required for human play mode. "
                "Install with: uv add opencv-python"
            )

        self.grid_size = grid_size
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_delay = int(1000 / fps)

        # Create environment
        self.env = DroneHunterEnv(
            render_mode="rgb_array",
            grid_size=grid_size,
            width=width,
            height=height,
            max_frames=max_frames,
            oracle_mode=True,
        )

        # State
        self.show_grid = True
        self.pending_action: int | None = None
        self.running = True

        # Window
        self.window_name = "Drone Hunter - Human Play"

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert click to grid cell
            cell_w = self.width / self.grid_size
            cell_h = self.height / self.grid_size

            grid_x = int(x / cell_w)
            grid_y = int(y / cell_h)

            # Clamp to valid range
            grid_x = max(0, min(self.grid_size - 1, grid_x))
            grid_y = max(0, min(self.grid_size - 1, grid_y))

            # Convert to action
            self.pending_action = self.env.grid_to_action(grid_x, grid_y)

    def run(self) -> None:
        """Run the human play loop."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        obs, info = self.env.reset()
        total_reward = 0.0
        episode = 1

        print("\n" + "=" * 50)
        print("DRONE HUNTER - Human Play Mode")
        print("=" * 50)
        print("Controls:")
        print("  Left Click - Fire at location")
        print("  Space      - Wait (skip turn)")
        print("  R          - Reset episode")
        print("  G          - Toggle grid overlay")
        print("  Q/Escape   - Quit")
        print("=" * 50 + "\n")

        while self.running:
            # Render
            frame = self.env.render()

            # Optionally hide grid
            if not self.show_grid:
                frame = self.env.renderer.render(self.env.game_state)

            # Convert RGB to BGR for OpenCV
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Show frame
            cv2.imshow(self.window_name, display_frame)

            # Handle input
            key = cv2.waitKey(self.frame_delay) & 0xFF

            # Process key
            action = 0  # Default: wait

            if key == ord('q') or key == 27:  # Q or Escape
                self.running = False
                continue
            elif key == ord('r'):  # Reset
                obs, info = self.env.reset()
                total_reward = 0.0
                episode += 1
                print(f"\n--- Episode {episode} started ---")
                continue
            elif key == ord('g'):  # Toggle grid
                self.show_grid = not self.show_grid
                continue
            elif key == ord(' '):  # Space = wait
                action = 0
            elif self.pending_action is not None:
                action = self.pending_action
                self.pending_action = None

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # Print action feedback
            if action > 0:
                grid_coords = self.env.action_to_grid(action)
                hit = "HIT!" if self.env._last_hit else "miss"
                print(f"Fire at {grid_coords} - {hit} | "
                      f"Reward: {reward:+.2f} | Total: {total_reward:.2f}")

            # Handle episode end
            if terminated or truncated:
                print(f"\n{'=' * 40}")
                print(f"Episode {episode} ended: {info['game_over_reason']}")
                print(f"Final Score: {info['score']}")
                print(f"Hits: {info['hits']} | Misses: {info['misses']}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"{'=' * 40}")
                print("Press R to restart or Q to quit...")

                # Wait for input
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('r'):
                        obs, info = self.env.reset()
                        total_reward = 0.0
                        episode += 1
                        print(f"\n--- Episode {episode} started ---")
                        break
                    elif key == ord('q') or key == 27:
                        self.running = False
                        break

        # Cleanup
        self.env.close()
        cv2.destroyAllWindows()


def main() -> None:
    """Entry point for human play mode."""
    parser = argparse.ArgumentParser(description="Play Drone Hunter as a human")
    parser.add_argument("--grid-size", type=int, default=8, help="Firing grid size")
    parser.add_argument("--width", type=int, default=640, help="Display width")
    parser.add_argument("--height", type=int, default=640, help="Display height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--max-frames", type=int, default=1000, help="Max frames per episode")

    args = parser.parse_args()

    try:
        player = HumanPlayer(
            grid_size=args.grid_size,
            width=args.width,
            height=args.height,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        player.run()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo play in human mode, install OpenCV:")
        print("  uv add opencv-python")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nQuitting...")


if __name__ == "__main__":
    main()
