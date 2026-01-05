"""Post-processing weather effects for visual difficulty."""

from __future__ import annotations

import random

import numpy as np

# Try OpenCV first, fall back to pure numpy
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def apply_rain(frame: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Add rain streaks to the frame.

    Args:
        frame: RGB frame as numpy array.
        intensity: Rain intensity (0.0-1.0).

    Returns:
        Frame with rain effect applied.
    """
    if intensity <= 0:
        return frame

    overlay = frame.copy()
    num_drops = int(intensity * 100)

    for _ in range(num_drops):
        x = random.randint(0, frame.shape[1] - 1)
        y1 = random.randint(0, frame.shape[0] - 20)
        length = random.randint(10, 20)
        # Angled rain streaks (wind effect)
        x_offset = random.randint(-3, 0)

        if HAS_OPENCV:
            cv2.line(overlay, (x, y1), (x + x_offset, y1 + length),
                    (200, 200, 200), 1, cv2.LINE_AA)
        else:
            # Simple numpy line (vertical only)
            y2 = min(y1 + length, frame.shape[0] - 1)
            overlay[y1:y2, x] = [200, 200, 200]

    # Blend with original
    alpha = 0.3
    if HAS_OPENCV:
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    else:
        return (frame * (1 - alpha) + overlay * alpha).astype(np.uint8)


def apply_fog(frame: np.ndarray, density: float = 0.3) -> np.ndarray:
    """Add distance-based fog effect.

    Args:
        frame: RGB frame as numpy array.
        density: Fog density (0.0-0.5).

    Returns:
        Frame with fog effect applied.
    """
    if density <= 0:
        return frame

    # Create fog color (light gray/white)
    fog_color = np.full_like(frame, 200)

    # Blend uniformly (could be enhanced with depth-based gradient)
    if HAS_OPENCV:
        return cv2.addWeighted(frame, 1 - density, fog_color, density, 0)
    else:
        return (frame * (1 - density) + fog_color * density).astype(np.uint8)


def apply_fog_gradient(frame: np.ndarray, density: float = 0.3) -> np.ndarray:
    """Add distance-based fog effect with vertical gradient.

    More fog at horizon (top), less near camera (bottom).

    Args:
        frame: RGB frame as numpy array.
        density: Maximum fog density at horizon (0.0-0.5).

    Returns:
        Frame with gradient fog effect applied.
    """
    if density <= 0:
        return frame

    result = frame.copy().astype(np.float32)
    height = frame.shape[0]

    for y in range(height):
        # More fog at top (horizon), less at bottom (near)
        ratio = 1.0 - (y / height)  # 1.0 at top, 0.0 at bottom
        local_density = density * ratio

        if local_density > 0:
            result[y] = result[y] * (1 - local_density) + 200 * local_density

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_mist(frame: np.ndarray, intensity: float = 0.2) -> np.ndarray:
    """Add misty/hazy atmosphere effect.

    Args:
        frame: RGB frame as numpy array.
        intensity: Mist intensity (0.0-0.5).

    Returns:
        Frame with mist effect applied.
    """
    if intensity <= 0:
        return frame

    # Reduce contrast
    mean = frame.mean()
    result = frame.astype(np.float32)
    result = (result - mean) * (1 - intensity) + mean

    # Add slight white overlay
    result = result * (1 - intensity * 0.3) + 220 * (intensity * 0.3)

    return np.clip(result, 0, 255).astype(np.uint8)
