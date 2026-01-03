"""2D sprite-based renderer with OpenCV/Pillow dual backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import random

import numpy as np

# Try OpenCV first, fall back to Pillow
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from drone_hunter.envs.renderer.base_renderer import BaseRenderer

if TYPE_CHECKING:
    from drone_hunter.envs.game_state import GameState, Drone


class SpriteRenderer(BaseRenderer):
    """Sprite-based 2D renderer with OpenCV/Pillow dual backend.

    Uses OpenCV when available for better performance,
    falls back to Pillow for compatibility (e.g., Termux).
    """

    def __init__(
        self,
        width: int = 320,
        height: int = 320,
        backgrounds_dir: Path | str | None = None,
        drones_dir: Path | str | None = None,
        use_placeholders: bool = True,
        force_backend: str | None = None,
    ):
        """Initialize sprite renderer.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            backgrounds_dir: Directory containing background images.
            drones_dir: Directory containing drone sprite images.
            use_placeholders: If True, use colored shapes when assets missing.
            force_backend: Force "opencv" or "pillow", or None for auto-detect.
        """
        super().__init__(width, height)

        # Select backend
        if force_backend == "opencv" and not HAS_OPENCV:
            raise ImportError("OpenCV requested but not available")
        elif force_backend == "pillow":
            self.use_opencv = False
        else:
            self.use_opencv = HAS_OPENCV

        self.backgrounds_dir = Path(backgrounds_dir) if backgrounds_dir else None
        self.drones_dir = Path(drones_dir) if drones_dir else None
        self.use_placeholders = use_placeholders

        # Load assets (stored as numpy arrays for OpenCV compatibility)
        self.backgrounds: list[np.ndarray] = []
        self.drone_sprites: list[np.ndarray] = []  # BGRA format

        if self.backgrounds_dir and self.backgrounds_dir.exists():
            self._load_backgrounds()
        if self.drones_dir and self.drones_dir.exists():
            self._load_drone_sprites()

        # Current background (changes between episodes)
        self._current_bg: np.ndarray | None = None

        # Augmentation settings
        self.augment_brightness = True
        self.augment_contrast = True
        self.motion_blur = True

    @property
    def backend_name(self) -> str:
        """Return current backend name."""
        return "opencv" if self.use_opencv else "pillow"

    def _load_backgrounds(self) -> None:
        """Load background images from directory."""
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for path in self.backgrounds_dir.glob(ext):
                try:
                    if self.use_opencv:
                        img = cv2.imread(str(path))
                        img = cv2.resize(img, (self.width, self.height))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        pil_img = Image.open(path).convert("RGB")
                        pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
                        img = np.array(pil_img)
                    self.backgrounds.append(img)
                except Exception:
                    pass

    def _load_drone_sprites(self) -> None:
        """Load drone sprite images from directory."""
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for path in self.drones_dir.glob(ext):
                try:
                    if self.use_opencv:
                        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                        if img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    else:
                        pil_img = Image.open(path).convert("RGBA")
                        img = np.array(pil_img)
                    self.drone_sprites.append(img)
                except Exception:
                    pass

    def _generate_sky_background(self) -> np.ndarray:
        """Generate a procedural sky background."""
        if self.use_opencv:
            return self._generate_sky_background_cv()
        else:
            return self._generate_sky_background_pil()

    def _generate_sky_background_cv(self) -> np.ndarray:
        """Generate sky background using OpenCV."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Sky gradient
        for y in range(self.height):
            ratio = y / self.height
            r = int(135 + ratio * 50)
            g = int(206 + ratio * 30)
            b = int(235 + ratio * 20)
            img[y, :] = [r, g, b]

        # Add clouds
        for _ in range(random.randint(2, 5)):
            cx = random.randint(0, self.width)
            cy = random.randint(0, self.height // 2)
            axes = (random.randint(40, 100), random.randint(20, 40))

            overlay = img.copy()
            cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, (255, 255, 255), -1)
            overlay = cv2.GaussianBlur(overlay, (21, 21), 10)
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        return img

    def _generate_sky_background_pil(self) -> np.ndarray:
        """Generate sky background using Pillow."""
        img = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(img)

        # Sky gradient
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
            cloud_draw.ellipse([cx - w, cy - h, cx + w, cy + h], fill=(255, 255, 255, 180))
            cloud_layer = cloud_layer.filter(ImageFilter.GaussianBlur(radius=10))

            img = Image.alpha_composite(img.convert("RGBA"), cloud_layer).convert("RGB")

        return np.array(img)

    def _generate_placeholder_drone(self, size_px: int, is_kamikaze: bool) -> np.ndarray:
        """Generate a placeholder drone shape (RGBA numpy array)."""
        if self.use_opencv:
            return self._generate_placeholder_drone_cv(size_px, is_kamikaze)
        else:
            return self._generate_placeholder_drone_pil(size_px, is_kamikaze)

    def _generate_placeholder_drone_cv(self, size_px: int, is_kamikaze: bool) -> np.ndarray:
        """Generate placeholder drone using OpenCV."""
        img = np.zeros((size_px, size_px, 4), dtype=np.uint8)

        if is_kamikaze:
            body_color = (50, 50, 200, 255)  # BGR + A
            arm_color = (30, 30, 150, 255)
        else:
            body_color = (80, 80, 80, 255)
            arm_color = (50, 50, 50, 255)

        center = size_px // 2
        body_radius = size_px // 6
        arm_length = size_px // 3
        arm_width = max(2, size_px // 20)
        rotor_radius = max(3, size_px // 10)

        # Arms (X pattern)
        for angle in [45, 135, 225, 315]:
            rad = angle * np.pi / 180
            x2 = int(center + arm_length * np.cos(rad))
            y2 = int(center + arm_length * np.sin(rad))
            cv2.line(img, (center, center), (x2, y2), arm_color, arm_width)
            cv2.circle(img, (x2, y2), rotor_radius, arm_color, -1)
            cv2.circle(img, (x2, y2), rotor_radius, (30, 30, 30, 255), 1)

        # Center body
        cv2.circle(img, (center, center), body_radius, body_color, -1)
        cv2.circle(img, (center, center), body_radius, (30, 30, 30, 255), 1)

        return img

    def _generate_placeholder_drone_pil(self, size_px: int, is_kamikaze: bool) -> np.ndarray:
        """Generate placeholder drone using Pillow."""
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

        # Arms (X pattern)
        for angle in [45, 135, 225, 315]:
            rad = angle * np.pi / 180
            x2 = int(center + arm_length * np.cos(rad))
            y2 = int(center + arm_length * np.sin(rad))
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

    def _apply_motion_blur(self, sprite: np.ndarray, vx: float, vy: float) -> np.ndarray:
        """Apply motion blur based on velocity."""
        if not self.motion_blur:
            return sprite

        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed < 0.005:
            return sprite

        blur_radius = min(5, int(speed * 200))
        if blur_radius < 1:
            return sprite

        if self.use_opencv:
            ksize = blur_radius * 2 + 1
            return cv2.GaussianBlur(sprite, (ksize, ksize), blur_radius)
        else:
            pil_img = Image.fromarray(sprite)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            return np.array(pil_img)

    def _resize_sprite(self, sprite: np.ndarray, size_px: int) -> np.ndarray:
        """Resize sprite to target size."""
        if self.use_opencv:
            return cv2.resize(sprite, (size_px, size_px), interpolation=cv2.INTER_LINEAR)
        else:
            pil_img = Image.fromarray(sprite)
            pil_img = pil_img.resize((size_px, size_px), Image.Resampling.LANCZOS)
            return np.array(pil_img)

    def _rotate_sprite(self, sprite: np.ndarray, angle: float) -> np.ndarray:
        """Rotate sprite by angle degrees."""
        if abs(angle) < 1:
            return sprite

        if self.use_opencv:
            h, w = sprite.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(sprite, matrix, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))
        else:
            pil_img = Image.fromarray(sprite)
            pil_img = pil_img.rotate(angle, expand=False, resample=Image.Resampling.BILINEAR)
            return np.array(pil_img)

    def _composite_sprite(self, frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
        """Composite RGBA sprite onto RGB frame at position (x, y)."""
        h, w = sprite.shape[:2]
        fh, fw = frame.shape[:2]

        # Calculate valid region (handle clipping at edges)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        if x1 >= x2 or y1 >= y2:
            return  # Completely off-screen

        # Sprite region
        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

        sprite_region = sprite[sy1:sy2, sx1:sx2]
        frame_region = frame[y1:y2, x1:x2]

        if sprite_region.shape[2] == 4:
            # Alpha blending
            alpha = sprite_region[:, :, 3:4].astype(np.float32) / 255.0
            rgb = sprite_region[:, :, :3]

            if self.use_opencv:
                # OpenCV stores as BGR, but we're working in RGB
                blended = (rgb.astype(np.float32) * alpha +
                          frame_region.astype(np.float32) * (1 - alpha))
            else:
                blended = (rgb.astype(np.float32) * alpha +
                          frame_region.astype(np.float32) * (1 - alpha))

            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = sprite_region[:, :, :3]

    def reset_background(self) -> None:
        """Select a new random background for the episode."""
        if self.backgrounds:
            self._current_bg = self.backgrounds[random.randint(0, len(self.backgrounds) - 1)].copy()
        else:
            self._current_bg = self._generate_sky_background()

        # Apply random augmentations
        if self.augment_brightness:
            factor = random.uniform(0.8, 1.2)
            self._current_bg = np.clip(self._current_bg * factor, 0, 255).astype(np.uint8)

        if self.augment_contrast:
            factor = random.uniform(0.9, 1.1)
            mean = self._current_bg.mean()
            self._current_bg = np.clip((self._current_bg - mean) * factor + mean, 0, 255).astype(np.uint8)

    def _render_drone(self, drone: Drone, frame: np.ndarray) -> None:
        """Render a single drone onto the frame."""
        screen_x, screen_y = self.normalized_to_screen(drone.x, drone.y)
        size_px = int(drone.size * self.width)
        size_px = max(8, size_px)

        # Get or generate drone sprite
        if self.drone_sprites:
            sprite = self.drone_sprites[random.randint(0, len(self.drone_sprites) - 1)].copy()
            sprite = self._resize_sprite(sprite, size_px)
        else:
            sprite = self._generate_placeholder_drone(size_px, drone.is_kamikaze)

        # Apply motion blur
        sprite = self._apply_motion_blur(sprite, drone.vx, drone.vy)

        # Random rotation
        angle = random.uniform(-15, 15)
        sprite = self._rotate_sprite(sprite, angle)

        # Calculate paste position (center sprite on drone position)
        paste_x = screen_x - sprite.shape[1] // 2
        paste_y = screen_y - sprite.shape[0] // 2

        # Composite onto frame
        self._composite_sprite(frame, sprite, paste_x, paste_y)

    def render(self, game_state: GameState) -> np.ndarray:
        """Render the current game state to an RGB frame."""
        if self._current_bg is None:
            self.reset_background()

        frame = self._current_bg.copy()

        for drone in game_state.drones:
            if drone.is_on_screen():
                self._render_drone(drone, frame)

        return frame

    def render_with_overlay(
        self,
        game_state: GameState,
        grid_size: int | None = None,
        highlight_cell: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Render game state with debug overlay."""
        if self._current_bg is None:
            self.reset_background()

        frame = self._current_bg.copy()

        for drone in game_state.drones:
            if drone.is_on_screen():
                self._render_drone(drone, frame)

        # Draw overlays
        if self.use_opencv:
            self._draw_overlays_cv(frame, game_state, grid_size, highlight_cell)
        else:
            frame = self._draw_overlays_pil(frame, game_state, grid_size, highlight_cell)

        return frame

    def _draw_overlays_cv(
        self,
        frame: np.ndarray,
        game_state: GameState,
        grid_size: int | None,
        highlight_cell: tuple[int, int] | None,
    ) -> None:
        """Draw overlays using OpenCV."""
        # Grid
        if grid_size is not None:
            cell_w = self.width // grid_size
            cell_h = self.height // grid_size

            for i in range(1, grid_size):
                x = i * cell_w
                y = i * cell_h
                cv2.line(frame, (x, 0), (x, self.height), (100, 100, 100), 1)
                cv2.line(frame, (0, y), (self.width, y), (100, 100, 100), 1)

            if highlight_cell is not None:
                gx, gy = highlight_cell
                x1, y1 = gx * cell_w, gy * cell_h
                x2, y2 = (gx + 1) * cell_w, (gy + 1) * cell_h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Drone bounding boxes
        for drone in game_state.drones:
            if drone.is_on_screen():
                x1, y1, x2, y2 = drone.bbox
                sx1 = int(x1 * self.width)
                sy1 = int(y1 * self.height)
                sx2 = int(x2 * self.width)
                sy2 = int(y2 * self.height)
                color = (0, 0, 255) if drone.is_kamikaze else (0, 255, 0)
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)

        # HUD
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thickness = 1

        # Ammo
        ammo_text = f"AMMO: {game_state.ammo}/{game_state.clip_size}"
        if game_state.is_reloading:
            ammo_text = f"RELOAD: {game_state.reload_timer}/{game_state.reload_time}"
        cv2.putText(frame, ammo_text, (10, self.height - 10), font, scale, (255, 255, 255), thickness)

        # Score
        cv2.putText(frame, f"SCORE: {game_state.score}", (10, 20), font, scale, (255, 255, 255), thickness)

        # Hits/Misses
        cv2.putText(frame, f"H:{game_state.hits} M:{game_state.misses}",
                   (self.width - 70, 20), font, scale, (255, 255, 255), thickness)

        # Frame
        cv2.putText(frame, f"{game_state.frame_count}/{game_state.max_frames}",
                   (self.width - 70, self.height - 10), font, scale, (255, 255, 255), thickness)

        # Threat
        if game_state.threat_level > 0.5:
            cv2.putText(frame, "THREAT!", (self.width // 2 - 30, 20),
                       font, scale, (0, 0, 255), thickness + 1)

    def _draw_overlays_pil(
        self,
        frame: np.ndarray,
        game_state: GameState,
        grid_size: int | None,
        highlight_cell: tuple[int, int] | None,
    ) -> np.ndarray:
        """Draw overlays using Pillow."""
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # Grid
        if grid_size is not None:
            cell_w = self.width / grid_size
            cell_h = self.height / grid_size

            for i in range(1, grid_size):
                x = int(i * cell_w)
                y = int(i * cell_h)
                draw.line([(x, 0), (x, self.height)], fill=(100, 100, 100), width=1)
                draw.line([(0, y), (self.width, y)], fill=(100, 100, 100), width=1)

            if highlight_cell is not None:
                gx, gy = highlight_cell
                x1 = int(gx * cell_w)
                y1 = int(gy * cell_h)
                x2 = int((gx + 1) * cell_w)
                y2 = int((gy + 1) * cell_h)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)

        # Drone bounding boxes
        for drone in game_state.drones:
            if drone.is_on_screen():
                x1, y1, x2, y2 = drone.bbox
                sx1 = int(x1 * self.width)
                sy1 = int(y1 * self.height)
                sx2 = int(x2 * self.width)
                sy2 = int(y2 * self.height)
                color = (255, 0, 0) if drone.is_kamikaze else (0, 255, 0)
                draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=2)

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
