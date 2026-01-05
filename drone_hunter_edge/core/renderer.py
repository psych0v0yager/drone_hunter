"""Pillow-based sprite renderer for edge deployment.

Simplified version without OpenCV, augmentations, or weather effects.
All rendering is procedural - no asset files required.
"""

from typing import List, Tuple, Optional
import random
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core.game_state import GameState, Drone


class Renderer:
    """Pillow-based 2D renderer for edge deployment."""

    def __init__(
        self,
        width: int = 320,
        height: int = 320,
    ):
        """Initialize renderer.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self.width = width
        self.height = height
        self._current_bg: Optional[np.ndarray] = None

    def reset_background(self) -> None:
        """Generate a new random background for the episode."""
        self._current_bg = self._generate_sky_background()

        # Apply random brightness variation
        factor = random.uniform(0.85, 1.15)
        self._current_bg = np.clip(self._current_bg * factor, 0, 255).astype(np.uint8)

    def _generate_sky_background(self) -> np.ndarray:
        """Generate a procedural sky background with clouds."""
        img = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(img)

        # Sky gradient (top = darker blue, bottom = lighter)
        for y in range(self.height):
            ratio = y / self.height
            r = int(135 + ratio * 50)
            g = int(206 + ratio * 30)
            b = int(235 + ratio * 20)
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))

        # Add clouds
        for _ in range(random.randint(2, 5)):
            cx = random.randint(0, self.width)
            cy = random.randint(0, self.height // 2)
            w = random.randint(40, 100)
            h = random.randint(20, 40)

            cloud_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            cloud_draw = ImageDraw.Draw(cloud_layer)
            cloud_draw.ellipse(
                [cx - w, cy - h, cx + w, cy + h],
                fill=(255, 255, 255, 180)
            )
            cloud_layer = cloud_layer.filter(ImageFilter.GaussianBlur(radius=10))

            img = Image.alpha_composite(img.convert("RGBA"), cloud_layer).convert("RGB")

        return np.array(img)

    def _generate_placeholder_drone(self, size_px: int, is_kamikaze: bool) -> np.ndarray:
        """Generate a placeholder drone shape (RGBA numpy array)."""
        img = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if is_kamikaze:
            body_color = (200, 50, 50, 255)
            arm_color = (150, 30, 30, 255)
        else:
            body_color = (80, 80, 80, 255)
            arm_color = (50, 50, 50, 255)

        center = size_px // 2
        body_radius = size_px // 6
        arm_length = size_px // 3
        arm_width = max(2, size_px // 20)
        rotor_radius = max(3, size_px // 10)

        # Draw arms (X pattern)
        for angle in [45, 135, 225, 315]:
            rad = angle * math.pi / 180
            x2 = int(center + arm_length * math.cos(rad))
            y2 = int(center + arm_length * math.sin(rad))
            draw.line([(center, center), (x2, y2)], fill=arm_color, width=arm_width)
            draw.ellipse([
                x2 - rotor_radius, y2 - rotor_radius,
                x2 + rotor_radius, y2 + rotor_radius
            ], fill=arm_color, outline=(30, 30, 30, 255))

        # Center body
        draw.ellipse([
            center - body_radius, center - body_radius,
            center + body_radius, center + body_radius
        ], fill=body_color, outline=(30, 30, 30, 255))

        return np.array(img)

    def _resize_sprite(self, sprite: np.ndarray, size_px: int) -> np.ndarray:
        """Resize sprite to target size."""
        pil_img = Image.fromarray(sprite)
        pil_img = pil_img.resize((size_px, size_px), Image.Resampling.LANCZOS)
        return np.array(pil_img)

    def _composite_sprite(self, frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
        """Composite RGBA sprite onto RGB frame at position (x, y)."""
        h, w = sprite.shape[:2]
        fh, fw = frame.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        if x1 >= x2 or y1 >= y2:
            return

        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

        sprite_region = sprite[sy1:sy2, sx1:sx2]
        frame_region = frame[y1:y2, x1:x2]

        if sprite_region.shape[2] == 4:
            alpha = sprite_region[:, :, 3:4].astype(np.float32) / 255.0
            rgb = sprite_region[:, :, :3]
            blended = (rgb.astype(np.float32) * alpha +
                      frame_region.astype(np.float32) * (1 - alpha))
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = sprite_region[:, :, :3]

    def normalized_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized [0,1] coordinates to screen pixels."""
        return (int(x * self.width), int(y * self.height))

    def _render_drone(self, drone: Drone, frame: np.ndarray) -> None:
        """Render a single drone onto the frame."""
        screen_x, screen_y = self.normalized_to_screen(drone.x, drone.y)
        size_px = int(drone.size * self.width)
        size_px = max(8, size_px)

        sprite = self._generate_placeholder_drone(size_px, drone.is_kamikaze)

        # Random slight rotation for visual variety
        angle = random.uniform(-15, 15)
        if abs(angle) > 1:
            pil_img = Image.fromarray(sprite)
            pil_img = pil_img.rotate(angle, expand=False, resample=Image.Resampling.BILINEAR)
            sprite = np.array(pil_img)

        paste_x = screen_x - sprite.shape[1] // 2
        paste_y = screen_y - sprite.shape[0] // 2

        self._composite_sprite(frame, sprite, paste_x, paste_y)

    def render(self, game_state: GameState) -> np.ndarray:
        """Render the current game state to an RGB frame.

        Args:
            game_state: Current game state with drones

        Returns:
            RGB numpy array of shape (height, width, 3)
        """
        if self._current_bg is None:
            self.reset_background()

        frame = self._current_bg.copy()

        # Render drones sorted by depth (far to near)
        for drone in sorted(game_state.drones, key=lambda d: -d.z):
            if drone.is_on_screen():
                self._render_drone(drone, frame)

        return frame

    def render_with_overlay(
        self,
        game_state: GameState,
        grid_size: int = 8,
        highlight_cell: Optional[Tuple[int, int]] = None,
        detections: Optional[List] = None,
    ) -> np.ndarray:
        """Render game state with debug overlay.

        Args:
            game_state: Current game state
            grid_size: Size of the firing grid
            highlight_cell: Grid cell to highlight (gx, gy)
            detections: Optional list of Detection objects to draw

        Returns:
            RGB numpy array with overlays
        """
        frame = self.render(game_state)
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        cell_w = self.width / grid_size
        cell_h = self.height / grid_size

        # Grid lines
        for i in range(1, grid_size):
            x = int(i * cell_w)
            y = int(i * cell_h)
            draw.line([(x, 0), (x, self.height)], fill=(100, 100, 100), width=1)
            draw.line([(0, y), (self.width, y)], fill=(100, 100, 100), width=1)

        # Highlight cell
        if highlight_cell is not None:
            gx, gy = highlight_cell
            x1 = int(gx * cell_w)
            y1 = int(gy * cell_h)
            x2 = int((gx + 1) * cell_w)
            y2 = int((gy + 1) * cell_h)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)

        # Drone bounding boxes (ground truth)
        for drone in game_state.drones:
            if drone.is_on_screen():
                x1, y1, x2, y2 = drone.bbox
                sx1 = int(x1 * self.width)
                sy1 = int(y1 * self.height)
                sx2 = int(x2 * self.width)
                sy2 = int(y2 * self.height)
                color = (255, 0, 0) if drone.is_kamikaze else (0, 255, 0)
                draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=2)

        # Detection boxes (from detector)
        if detections:
            for det in detections:
                x1 = int((det.x - det.w / 2) * self.width)
                y1 = int((det.y - det.h / 2) * self.height)
                x2 = int((det.x + det.w / 2) * self.width)
                y2 = int((det.y + det.h / 2) * self.height)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 255), width=1)

        # HUD
        ammo_text = f"AMMO: {game_state.ammo}/{game_state.clip_size}"
        if game_state.is_reloading:
            ammo_text = f"RELOAD: {game_state.reload_timer}/{game_state.reload_time}"
        draw.text((10, self.height - 25), ammo_text, fill=(255, 255, 255))
        draw.text((10, 10), f"SCORE: {game_state.score}", fill=(255, 255, 255))
        draw.text((self.width - 80, 10), f"H:{game_state.hits} M:{game_state.misses}", fill=(255, 255, 255))
        draw.text((self.width - 80, self.height - 25),
                 f"{game_state.frame_count}/{game_state.max_frames}", fill=(255, 255, 255))

        if game_state.threat_level > 0.5:
            draw.text((self.width // 2 - 30, 10), "THREAT!", fill=(255, 0, 0))

        return np.array(pil_img)
