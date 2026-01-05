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
    from drone_hunter.envs.game_state import (
        GameState, Drone, Distractor, StaticObstacle,
        DistractorType, StaticObstacleType
    )
    from drone_hunter.envs.difficulty import DifficultyConfig


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
        difficulty_config: DifficultyConfig | None = None,
    ):
        """Initialize sprite renderer.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            backgrounds_dir: Directory containing background images.
            drones_dir: Directory containing drone sprite images.
            use_placeholders: If True, use colored shapes when assets missing.
            force_backend: Force "opencv" or "pillow", or None for auto-detect.
            difficulty_config: Visual difficulty settings for augmentation.
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

        # Difficulty configuration (import here to avoid circular imports)
        if difficulty_config is None:
            from drone_hunter.envs.difficulty import DifficultyConfig
            difficulty_config = DifficultyConfig()
        self.difficulty_config = difficulty_config

        # Load assets (stored as numpy arrays for OpenCV compatibility)
        self.backgrounds: list[np.ndarray] = []
        self.drone_sprites: list[np.ndarray] = []  # BGRA format

        if self.backgrounds_dir and self.backgrounds_dir.exists():
            self._load_backgrounds()
        if self.drones_dir and self.drones_dir.exists():
            self._load_drone_sprites()

        # Current background (changes between episodes)
        self._current_bg: np.ndarray | None = None

        # Augmentation settings (legacy - now controlled by difficulty_config)
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

    def _generate_sky_for_lighting(self) -> np.ndarray:
        """Generate sky background based on lighting mode from difficulty config."""
        mode = self.difficulty_config.lighting_mode

        if mode == "random":
            mode = random.choice(["day", "golden_hour", "overcast", "dusk", "night"])

        if mode == "day":
            return self._generate_sky_background()

        elif mode == "golden_hour":
            return self._generate_golden_hour_sky()

        elif mode == "overcast":
            return self._generate_overcast_sky()

        elif mode == "dusk":
            return self._generate_dusk_sky()

        elif mode == "night":
            return self._generate_night_sky()

        elif mode == "thermal":
            return self._generate_thermal_background()

        elif mode == "night_vision":
            return self._generate_night_vision_background()

        # Fallback to day
        return self._generate_sky_background()

    def _generate_golden_hour_sky(self) -> np.ndarray:
        """Generate golden hour / sunset sky."""
        if self.use_opencv:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                ratio = y / self.height
                r = 255
                g = int(150 + ratio * 50)
                b = int(80 + ratio * 40)
                img[y, :] = [r, g, b]
            # Add golden-tinted clouds
            return self._add_clouds_cv(img, tint=(255, 200, 150))
        else:
            img = Image.new("RGB", (self.width, self.height))
            draw = ImageDraw.Draw(img)
            for y in range(self.height):
                ratio = y / self.height
                r = 255
                g = int(150 + ratio * 50)
                b = int(80 + ratio * 40)
                draw.line([(0, y), (self.width, y)], fill=(r, g, b))
            return self._add_clouds_pil(np.array(img), tint=(255, 200, 150))

    def _generate_overcast_sky(self) -> np.ndarray:
        """Generate flat gray overcast sky."""
        gray = random.randint(160, 200)
        return np.full((self.height, self.width, 3), gray, dtype=np.uint8)

    def _generate_dusk_sky(self) -> np.ndarray:
        """Generate dusk/twilight sky."""
        if self.use_opencv:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                ratio = y / self.height
                r = int(30 + ratio * 20)
                g = int(40 + ratio * 30)
                b = int(80 + ratio * 40)
                img[y, :] = [r, g, b]
            return img
        else:
            img = Image.new("RGB", (self.width, self.height))
            draw = ImageDraw.Draw(img)
            for y in range(self.height):
                ratio = y / self.height
                r = int(30 + ratio * 20)
                g = int(40 + ratio * 30)
                b = int(80 + ratio * 40)
                draw.line([(0, y), (self.width, y)], fill=(r, g, b))
            return np.array(img)

    def _generate_night_sky(self) -> np.ndarray:
        """Generate night sky with stars."""
        img = np.full((self.height, self.width, 3), 15, dtype=np.uint8)
        # Add stars in upper portion
        for _ in range(50):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height // 2)
            brightness = random.randint(150, 255)
            img[y, x] = [brightness, brightness, brightness]
        return img

    def _generate_thermal_background(self) -> np.ndarray:
        """Generate thermal imaging background (dark grayscale)."""
        return np.full((self.height, self.width, 3), 40, dtype=np.uint8)

    def _generate_night_vision_background(self) -> np.ndarray:
        """Generate night vision green background."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :, 1] = 30  # Green channel
        return img

    def _add_clouds_cv(self, img: np.ndarray, tint: tuple = (255, 255, 255)) -> np.ndarray:
        """Add clouds using OpenCV with optional color tint."""
        for _ in range(random.randint(2, 5)):
            cx = random.randint(0, self.width)
            cy = random.randint(0, self.height // 2)
            axes = (random.randint(40, 100), random.randint(20, 40))

            overlay = img.copy()
            cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, tint, -1)
            overlay = cv2.GaussianBlur(overlay, (21, 21), 10)
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        return img

    def _add_clouds_pil(self, img: np.ndarray, tint: tuple = (255, 255, 255)) -> np.ndarray:
        """Add clouds using Pillow with optional color tint."""
        pil_img = Image.fromarray(img)
        for _ in range(random.randint(2, 5)):
            cx = random.randint(0, self.width)
            cy = random.randint(0, self.height // 2)
            w = random.randint(40, 100)
            h = random.randint(20, 40)

            cloud_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            cloud_draw = ImageDraw.Draw(cloud_layer)
            cloud_color = (*tint[:3], 180) if len(tint) == 3 else tint
            cloud_draw.ellipse([cx - w, cy - h, cx + w, cy + h], fill=cloud_color)
            cloud_layer = cloud_layer.filter(ImageFilter.GaussianBlur(radius=10))

            pil_img = Image.alpha_composite(pil_img.convert("RGBA"), cloud_layer).convert("RGB")
        return np.array(pil_img)

    # ===== Scene Type Generation =====

    def _generate_scene_background(self) -> np.ndarray:
        """Generate background based on scene type from difficulty config."""
        scene = self.difficulty_config.scene_type

        if scene == "random":
            scene = random.choice(["sky", "forest", "urban"])

        if scene == "sky":
            return self._generate_sky_for_lighting()
        elif scene == "forest":
            return self._generate_forest_background()
        elif scene == "urban":
            return self._generate_urban_background()

        # Fallback to sky
        return self._generate_sky_for_lighting()

    def _generate_forest_background(self) -> np.ndarray:
        """Generate forest/treeline background."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Sky peek at top (use lighting mode colors)
        mode = self.difficulty_config.lighting_mode
        if mode == "random":
            mode = random.choice(["day", "golden_hour", "overcast", "dusk"])

        if mode == "day":
            sky_colors = [(100, 150, 180), (150, 200, 210)]
        elif mode == "golden_hour":
            sky_colors = [(200, 140, 100), (230, 170, 120)]
        elif mode == "overcast":
            sky_colors = [(160, 160, 160), (180, 180, 180)]
        elif mode == "dusk":
            sky_colors = [(30, 40, 60), (50, 60, 100)]
        elif mode == "night":
            sky_colors = [(15, 15, 20), (20, 20, 30)]
        else:
            sky_colors = [(100, 150, 180), (150, 200, 210)]

        # Draw sky gradient in top third
        for y in range(self.height // 3):
            ratio = y / (self.height // 3)
            r = int(sky_colors[0][0] + ratio * (sky_colors[1][0] - sky_colors[0][0]))
            g = int(sky_colors[0][1] + ratio * (sky_colors[1][1] - sky_colors[0][1]))
            b = int(sky_colors[0][2] + ratio * (sky_colors[1][2] - sky_colors[0][2]))
            img[y, :] = [r, g, b]

        # Far treeline (darker, smaller)
        self._draw_treeline(img, y_start=self.height // 3, height=self.height // 4,
                           base_color=(20, 60, 20), variation=10)

        # Near treeline (lighter, taller)
        self._draw_treeline(img, y_start=self.height // 2, height=self.height // 2,
                           base_color=(30, 80, 30), variation=15)

        return img

    def _draw_treeline(self, img: np.ndarray, y_start: int, height: int,
                      base_color: tuple, variation: int = 10) -> None:
        """Draw a treeline at specified position."""
        # Create wavy treeline profile
        num_trees = random.randint(8, 15)

        for i in range(num_trees):
            x = int(i * self.width / num_trees + random.randint(-10, 10))
            tree_height = height + random.randint(-height // 4, height // 4)
            tree_width = random.randint(self.width // 15, self.width // 8)

            # Randomize color slightly
            r = max(0, min(255, base_color[0] + random.randint(-variation, variation)))
            g = max(0, min(255, base_color[1] + random.randint(-variation, variation)))
            b = max(0, min(255, base_color[2] + random.randint(-variation, variation)))
            color = (r, g, b)

            # Draw triangular tree shape
            if self.use_opencv:
                pts = np.array([
                    [x, y_start],
                    [x - tree_width // 2, y_start + tree_height],
                    [x + tree_width // 2, y_start + tree_height]
                ], np.int32)
                cv2.fillPoly(img, [pts], color)
            else:
                # Fill area below treeline
                for ty in range(y_start, min(y_start + tree_height, self.height)):
                    progress = (ty - y_start) / tree_height if tree_height > 0 else 0
                    half_width = int((tree_width // 2) * progress)
                    x_start = max(0, x - half_width)
                    x_end = min(self.width, x + half_width)
                    img[ty, x_start:x_end] = color

    def _generate_urban_background(self) -> np.ndarray:
        """Generate urban/cityscape background with buildings."""
        # Start with sky
        img = self._generate_sky_for_lighting()

        # Add building silhouettes at bottom
        for _ in range(random.randint(5, 10)):
            x = random.randint(0, self.width - 30)
            w = random.randint(20, 60)
            h = random.randint(self.height // 4, self.height // 2)
            y = self.height - h

            # Building color (dark gray)
            building_color = (40 + random.randint(0, 20),
                            40 + random.randint(0, 20),
                            50 + random.randint(0, 20))

            if self.use_opencv:
                cv2.rectangle(img, (x, y), (x + w, self.height), building_color, -1)

                # Add windows
                for wy in range(y + 5, self.height - 5, 10):
                    for wx in range(x + 3, x + w - 3, 8):
                        if random.random() < 0.3:
                            window_color = (200, 200, 100)  # Yellow-ish lit window
                            cv2.rectangle(img, (wx, wy), (wx + 4, wy + 6), window_color, -1)
            else:
                # Manual rectangle fill for Pillow backend
                img[y:self.height, x:x+w] = building_color

                # Add windows
                for wy in range(y + 5, self.height - 5, 10):
                    for wx in range(x + 3, min(x + w - 3, self.width - 4), 8):
                        if random.random() < 0.3:
                            img[wy:wy+6, wx:wx+4] = (200, 200, 100)

        return img

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
            # Use scene type and lighting mode from difficulty config
            self._current_bg = self._generate_scene_background()

        # Apply random brightness/contrast augmentations
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

    def _render_distractor(self, distractor: Distractor, frame: np.ndarray) -> None:
        """Render a flying distractor onto the frame."""
        from drone_hunter.envs.game_state import DistractorType

        screen_x, screen_y = self.normalized_to_screen(distractor.x, distractor.y)
        size_px = int(distractor.size * self.width)
        size_px = max(4, size_px)

        if distractor.distractor_type == DistractorType.BIRD:
            self._draw_bird(frame, screen_x, screen_y, size_px, distractor.age)
        elif distractor.distractor_type == DistractorType.DEBRIS:
            self._draw_debris(frame, screen_x, screen_y, size_px)
        elif distractor.distractor_type == DistractorType.BALLOON:
            self._draw_balloon(frame, screen_x, screen_y, size_px)
        elif distractor.distractor_type == DistractorType.PLANE:
            self._draw_plane(frame, screen_x, screen_y, size_px)

    def _draw_bird(self, frame: np.ndarray, x: int, y: int, size: int, age: int) -> None:
        """Draw a bird silhouette with flapping wings."""
        # Wing flap animation
        flap = (age // 4) % 2
        wing_angle = 30 if flap else -20

        color = (30, 30, 30)  # Dark silhouette

        if self.use_opencv:
            # Body
            cv2.ellipse(frame, (x, y), (size // 3, size // 6), 0, 0, 360, color, -1)
            # Wings
            wing_len = size // 2
            if flap:
                # Wings up
                cv2.line(frame, (x, y), (x - wing_len, y - wing_len // 2), color, 2)
                cv2.line(frame, (x, y), (x + wing_len, y - wing_len // 2), color, 2)
            else:
                # Wings down
                cv2.line(frame, (x, y), (x - wing_len, y + wing_len // 3), color, 2)
                cv2.line(frame, (x, y), (x + wing_len, y + wing_len // 3), color, 2)
        else:
            # Simple ellipse for body
            half = size // 6
            frame[max(0, y-half):min(frame.shape[0], y+half),
                  max(0, x-size//3):min(frame.shape[1], x+size//3)] = color

    def _draw_debris(self, frame: np.ndarray, x: int, y: int, size: int) -> None:
        """Draw small debris speck."""
        color = (50, 50, 50)
        radius = max(2, size // 4)

        if self.use_opencv:
            cv2.circle(frame, (x, y), radius, color, -1)
        else:
            # Simple square
            frame[max(0, y-radius):min(frame.shape[0], y+radius),
                  max(0, x-radius):min(frame.shape[1], x+radius)] = color

    def _draw_balloon(self, frame: np.ndarray, x: int, y: int, size: int) -> None:
        """Draw a balloon shape."""
        balloon_color = (random.randint(150, 255), random.randint(50, 150), random.randint(50, 150))
        string_color = (80, 80, 80)

        if self.use_opencv:
            # Balloon body
            cv2.ellipse(frame, (x, y), (size // 2, size // 2 + size // 4), 0, 0, 360, balloon_color, -1)
            # String
            cv2.line(frame, (x, y + size // 2), (x, y + size), string_color, 1)
        else:
            half = size // 2
            frame[max(0, y-half):min(frame.shape[0], y+half),
                  max(0, x-half):min(frame.shape[1], x+half)] = balloon_color

    def _draw_plane(self, frame: np.ndarray, x: int, y: int, size: int) -> None:
        """Draw a fixed-wing aircraft silhouette."""
        color = (60, 60, 60)

        if self.use_opencv:
            # Fuselage
            cv2.ellipse(frame, (x, y), (size // 2, size // 8), 0, 0, 360, color, -1)
            # Wings
            pts = np.array([
                [x - size // 6, y],
                [x - size // 3, y - size // 3],
                [x + size // 3, y - size // 3],
                [x + size // 6, y]
            ], np.int32)
            cv2.fillPoly(frame, [pts], color)
            # Tail
            cv2.line(frame, (x - size // 2, y), (x - size // 2, y - size // 4), color, 2)
        else:
            half = size // 4
            frame[max(0, y-half):min(frame.shape[0], y+half),
                  max(0, x-size//2):min(frame.shape[1], x+size//2)] = color

    def _render_static_obstacle(self, obstacle: StaticObstacle, frame: np.ndarray) -> None:
        """Render a static obstacle onto the frame."""
        from drone_hunter.envs.game_state import StaticObstacleType

        if obstacle.obstacle_type == StaticObstacleType.TREE:
            self._draw_tree_obstacle(frame, obstacle)
        elif obstacle.obstacle_type == StaticObstacleType.POWERLINE:
            self._draw_powerline_obstacle(frame, obstacle)

    def _draw_tree_obstacle(self, frame: np.ndarray, obstacle: StaticObstacle) -> None:
        """Draw a tree silhouette obstacle."""
        cx = int((obstacle.x + obstacle.sway_offset) * self.width)
        bottom = int(obstacle.y * self.height)
        w = int(obstacle.width * self.width)
        h = int(obstacle.height * self.height)

        trunk_color = (30, 20, 10)
        canopy_color = (20, 50, 20)

        if self.use_opencv:
            # Trunk
            trunk_w = w // 5
            cv2.rectangle(frame,
                         (cx - trunk_w // 2, bottom - h // 3),
                         (cx + trunk_w // 2, bottom),
                         trunk_color, -1)
            # Canopy (triangle)
            pts = np.array([
                [cx, bottom - h],
                [cx - w // 2, bottom - h // 3],
                [cx + w // 2, bottom - h // 3]
            ], np.int32)
            cv2.fillPoly(frame, [pts], canopy_color)
        else:
            # Simple rectangle for trunk and triangle for canopy
            trunk_w = w // 5
            frame[bottom - h // 3:bottom,
                  max(0, cx - trunk_w // 2):min(frame.shape[1], cx + trunk_w // 2)] = trunk_color

    def _draw_powerline_obstacle(self, frame: np.ndarray, obstacle: StaticObstacle) -> None:
        """Draw horizontal powerline with poles."""
        y_px = int(obstacle.y * self.height)
        wire_color = (30, 30, 30)
        pole_color = (50, 40, 30)

        if self.use_opencv:
            # Main wire
            cv2.line(frame, (0, y_px), (self.width, y_px), wire_color, 2)
            # Second wire (slight offset)
            cv2.line(frame, (0, y_px + 4), (self.width, y_px + 4), wire_color, 1)
            # Poles at edges
            cv2.line(frame, (15, y_px - 25), (15, y_px + 10), pole_color, 4)
            cv2.line(frame, (self.width - 15, y_px - 25), (self.width - 15, y_px + 10), pole_color, 4)
        else:
            # Simple horizontal lines
            frame[y_px:y_px + 2, :] = wire_color
            frame[y_px + 4:y_px + 5, :] = wire_color

    def render(self, game_state: GameState) -> np.ndarray:
        """Render the current game state to an RGB frame."""
        if self._current_bg is None:
            self.reset_background()

        frame = self._current_bg.copy()

        # 1. Render static obstacles first (background layer)
        for obstacle in game_state.static_obstacles:
            self._render_static_obstacle(obstacle, frame)

        # 2. Render flying distractors (sorted by depth, far to near)
        for distractor in sorted(game_state.distractors, key=lambda d: -d.z):
            if distractor.is_on_screen():
                self._render_distractor(distractor, frame)

        # 3. Render drones (sorted by depth, far to near)
        for drone in sorted(game_state.drones, key=lambda d: -d.z):
            if drone.is_on_screen():
                self._render_drone(drone, frame)

        # Apply post-processing augmentations based on difficulty
        frame = self._apply_augmentations(frame)

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

        # 1. Render static obstacles first (background layer)
        for obstacle in game_state.static_obstacles:
            self._render_static_obstacle(obstacle, frame)

        # 2. Render flying distractors (sorted by depth, far to near)
        for distractor in sorted(game_state.distractors, key=lambda d: -d.z):
            if distractor.is_on_screen():
                self._render_distractor(distractor, frame)

        # 3. Render drones (sorted by depth, far to near)
        for drone in sorted(game_state.drones, key=lambda d: -d.z):
            if drone.is_on_screen():
                self._render_drone(drone, frame)

        # Apply post-processing augmentations BEFORE overlays (so HUD is readable)
        frame = self._apply_augmentations(frame)

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

    def _apply_augmentations(self, frame: np.ndarray) -> np.ndarray:
        """Apply post-processing augmentations based on difficulty config.

        Args:
            frame: RGB frame as numpy array.

        Returns:
            Augmented frame.
        """
        config = self.difficulty_config

        # Apply weather effects first (they affect the whole scene)
        if config.rain_intensity > 0:
            from drone_hunter.envs.renderer.effects import apply_rain
            frame = apply_rain(frame, config.rain_intensity)

        if config.fog_density > 0:
            from drone_hunter.envs.renderer.effects import apply_fog_gradient
            frame = apply_fog_gradient(frame, config.fog_density)

        # Apply noise
        if config.noise_level > 0:
            frame = self._apply_noise(frame, config.noise_level)

        # Apply hue shift
        if config.hue_shift_range > 0:
            frame = self._apply_hue_shift(frame, config.hue_shift_range)

        # Apply JPEG compression artifacts
        if config.jpeg_quality < 100:
            frame = self._apply_jpeg_compression(frame, config.jpeg_quality)

        return frame

    def _apply_noise(self, frame: np.ndarray, noise_level: float) -> np.ndarray:
        """Apply Gaussian noise to simulate sensor noise.

        Args:
            frame: RGB frame.
            noise_level: Standard deviation of noise (0.0-0.3).

        Returns:
            Noisy frame.
        """
        noise = np.random.normal(0, noise_level * 255, frame.shape).astype(np.float32)
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _apply_hue_shift(self, frame: np.ndarray, shift_range: float) -> np.ndarray:
        """Apply random hue shift to simulate color variation.

        Args:
            frame: RGB frame.
            shift_range: Maximum hue shift (0.0-0.15).

        Returns:
            Hue-shifted frame.
        """
        shift = random.uniform(-shift_range, shift_range)

        if self.use_opencv:
            # Convert RGB to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            # Hue is 0-180 in OpenCV
            hsv[:, :, 0] = (hsv[:, :, 0] + shift * 180) % 180
            hsv = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            # Use Pillow
            pil_img = Image.fromarray(frame)
            hsv = pil_img.convert("HSV")
            h, s, v = hsv.split()
            # Pillow hue is 0-255
            h_arr = np.array(h, dtype=np.float32)
            h_arr = (h_arr + shift * 255) % 255
            h = Image.fromarray(h_arr.astype(np.uint8))
            hsv = Image.merge("HSV", (h, s, v))
            return np.array(hsv.convert("RGB"))

    def _apply_jpeg_compression(self, frame: np.ndarray, quality: int) -> np.ndarray:
        """Apply JPEG compression artifacts.

        Args:
            frame: RGB frame.
            quality: JPEG quality (50-100).

        Returns:
            Compressed frame with artifacts.
        """
        if self.use_opencv:
            # Encode then decode to simulate compression
            _, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        else:
            # Use Pillow
            from io import BytesIO
            pil_img = Image.fromarray(frame)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed = Image.open(buffer)
            return np.array(compressed.convert("RGB"))
