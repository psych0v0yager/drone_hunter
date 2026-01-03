"""Abstract base class for renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drone_hunter.envs.game_state import GameState


class BaseRenderer(ABC):
    """Abstract renderer interface for Drone Hunter.

    Renderers convert game state to visual frames.
    Different implementations can provide 2D sprites, 3D rendering, etc.
    """

    def __init__(self, width: int = 320, height: int = 320):
        """Initialize renderer.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self.width = width
        self.height = height

    @abstractmethod
    def render(self, game_state: GameState) -> np.ndarray:
        """Render the current game state to an RGB frame.

        Args:
            game_state: Current game state to render.

        Returns:
            RGB frame as numpy array of shape (height, width, 3), dtype uint8.
        """
        pass

    @abstractmethod
    def render_with_overlay(
        self,
        game_state: GameState,
        grid_size: int | None = None,
        highlight_cell: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Render game state with optional debug overlay.

        Args:
            game_state: Current game state to render.
            grid_size: If provided, draw firing grid overlay.
            highlight_cell: If provided, highlight this grid cell.

        Returns:
            RGB frame with overlays.
        """
        pass

    def screen_to_normalized(self, x: int, y: int) -> tuple[float, float]:
        """Convert screen coordinates to normalized (0-1) coordinates."""
        return x / self.width, y / self.height

    def normalized_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert normalized (0-1) coordinates to screen coordinates."""
        return int(x * self.width), int(y * self.height)
