"""Motion detection for frame differencing gate.

Provides cv2.absdiff when available, falls back to numpy for edge deployment.
Used by adaptive scheduler to skip detection on static scenes.
"""

from typing import Tuple
import numpy as np

# Try to import cv2, fall back to pure numpy if unavailable
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def frame_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute mean absolute difference between two frames.

    Args:
        frame1: First frame (H, W, 3) uint8 RGB.
        frame2: Second frame (H, W, 3) uint8 RGB.

    Returns:
        Mean absolute difference (0-255 scale).
    """
    if HAS_CV2:
        # cv2.absdiff is optimized and handles uint8 overflow correctly
        return float(cv2.absdiff(frame1, frame2).mean())
    else:
        # Pure numpy fallback - cast to int16 to handle overflow
        diff = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))
        return float(diff.mean())


class MotionDetector:
    """Simple frame differencing motion detector.

    Compares current frame to previous frame to detect scene changes.
    Used as a gate to skip detection when nothing is moving.
    """

    def __init__(self, threshold: float = 0.3):
        """Initialize motion detector.

        Args:
            threshold: Mean pixel difference threshold for motion detection.
                Default 0.3 tuned for drone sprites (~0.4 diff when moving).
                Lower = more sensitive, higher = less sensitive.
        """
        self.prev_frame = None
        self.threshold = threshold

    def reset(self) -> None:
        """Reset detector state."""
        self.prev_frame = None

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Detect motion by comparing to previous frame.

        Args:
            frame: Current frame (H, W, 3) uint8 RGB.

        Returns:
            Tuple of (has_motion, motion_score):
            - has_motion: True if motion detected above threshold
            - motion_score: Mean absolute difference (0-255)
        """
        if self.prev_frame is None:
            # First frame - assume motion (need initial detection)
            self.prev_frame = frame.copy()
            return True, 255.0

        # Compute frame difference
        score = frame_diff(frame, self.prev_frame)

        # Update previous frame
        self.prev_frame = frame.copy()

        return score > self.threshold, score

    def detect_motion_roi(
        self,
        frame: np.ndarray,
        x: float,
        y: float,
        roi_size: float = 0.2,
    ) -> Tuple[bool, float]:
        """Detect motion in a specific region of interest.

        Args:
            frame: Current frame (H, W, 3) uint8 RGB.
            x: ROI center x (normalized 0-1).
            y: ROI center y (normalized 0-1).
            roi_size: ROI size as fraction of frame (default 0.2 = 20%).

        Returns:
            Tuple of (has_motion, motion_score) for the ROI.
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return True, 255.0

        h, w = frame.shape[:2]

        # Compute ROI bounds
        half_size = roi_size / 2
        x_min = int(max(0, (x - half_size) * w))
        x_max = int(min(w, (x + half_size) * w))
        y_min = int(max(0, (y - half_size) * h))
        y_max = int(min(h, (y + half_size) * h))

        # Extract ROIs
        roi_curr = frame[y_min:y_max, x_min:x_max]
        roi_prev = self.prev_frame[y_min:y_max, x_min:x_max]

        # Compute difference on ROI only
        score = frame_diff(roi_curr, roi_prev)

        # Update previous frame (full frame for next call)
        self.prev_frame = frame.copy()

        return score > self.threshold, score
