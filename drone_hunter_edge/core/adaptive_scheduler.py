"""Adaptive detection scheduler for hierarchical compute allocation.

Core insight: spend detection budget where uncertainty is high, coast on
Kalman prediction where it's low.

Detection Tiers:
- Tier 0: Skip (Kalman predict only) - ~0ms
- Tier 1: Tiny detector at predicted ROI - ~2-5ms (when available)
- Tier 2: Full NanoDet (320x320) - ~30ms

Decision flow:
1. Frame diff gate - skip if no motion
2. Check Kalman uncertainty - full detect if high
3. Check frames since detection - skip if under budget
4. Otherwise tiny detect (if available) or full detect
"""

from typing import Optional, Tuple
import time
import numpy as np

from core.motion import MotionDetector
from core.tiny_detector import TinyDetector


# Uncertainty thresholds (tuned empirically via benchmark_uncertainty.py)
# These are trace(P[:3,:3]) values from the Kalman filter
UNCERTAINTY_LOW = 0.1    # Below this: coast on prediction
UNCERTAINTY_HIGH = 0.3   # Above this: must run full detection


class AdaptiveScheduler:
    """Adaptive scheduler for hierarchical detection.

    Decides which detection tier to use based on:
    - Motion detection (frame differencing)
    - Kalman filter uncertainty
    - Time since last detection (staleness)
    - Hardware calibration (base_skip)
    """

    def __init__(
        self,
        base_skip: int = 3,
        tiny_detector: Optional[TinyDetector] = None,
        motion_threshold: float = 5.0,
        uncertainty_low: float = UNCERTAINTY_LOW,
        uncertainty_high: float = UNCERTAINTY_HIGH,
    ):
        """Initialize adaptive scheduler.

        Args:
            base_skip: Minimum frames between detections (hardware dependent).
                Set via calibrate_device() or manually based on device speed.
            tiny_detector: Optional TinyDetector for Tier 1 detection.
            motion_threshold: Threshold for motion detection (default 5.0).
            uncertainty_low: Below this, prefer Tier 1 or skip.
            uncertainty_high: Above this, always use Tier 2 (full detection).
        """
        self.base_skip = base_skip
        self.tiny_detector = tiny_detector
        self.motion_detector = MotionDetector(threshold=motion_threshold)

        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high

        # State
        self.frames_since_detection = 0
        self.last_tier = 0

        # Statistics
        self.tier_counts = {0: 0, 1: 0, 2: 0}
        self.total_frames = 0

    def reset(self) -> None:
        """Reset scheduler state for new episode."""
        self.frames_since_detection = 0
        self.last_tier = 0
        self.motion_detector.reset()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.tier_counts = {0: 0, 1: 0, 2: 0}
        self.total_frames = 0

    def get_detection_tier(
        self,
        frame: np.ndarray,
        kalman_uncertainty: float,
    ) -> int:
        """Decide which detection tier to use.

        Args:
            frame: Current frame (H, W, 3) uint8 RGB.
            kalman_uncertainty: Scalar uncertainty from tracker.get_max_uncertainty().

        Returns:
            Detection tier:
            - 0: Skip (use Kalman prediction only)
            - 1: Tiny detector at predicted ROI
            - 2: Full NanoDet detection
        """
        self.total_frames += 1
        self.frames_since_detection += 1

        # Check motion
        has_motion, motion_score = self.motion_detector.detect_motion(frame)

        # Decision logic
        tier = self._decide_tier(has_motion, kalman_uncertainty)

        # Update state if we're detecting
        if tier > 0:
            self.frames_since_detection = 0

        self.last_tier = tier
        self.tier_counts[tier] += 1

        return tier

    def _decide_tier(self, has_motion: bool, uncertainty: float) -> int:
        """Core tier decision logic.

        Args:
            has_motion: True if motion detected in frame.
            uncertainty: Kalman uncertainty value.

        Returns:
            Detection tier (0, 1, or 2).
        """
        # Rule 1: No motion + recent detection = skip
        # (but don't skip too long - staleness check)
        if not has_motion and self.frames_since_detection < self.base_skip * 2:
            return 0

        # Rule 2: High uncertainty = must detect (full)
        if uncertainty > self.uncertainty_high:
            return 2

        # Rule 3: Under budget = skip
        if self.frames_since_detection < self.base_skip:
            return 0

        # Rule 4: Staleness check - must detect if too long since last
        if self.frames_since_detection > self.base_skip * 3:
            return 2

        # Rule 5: Low uncertainty + motion = tiny detector (if available)
        if uncertainty < self.uncertainty_low:
            if self.tiny_detector and self.tiny_detector.enabled:
                return 1

        # Default: full detection
        return 2

    def mark_detection_complete(self) -> None:
        """Call after successful detection to reset counter."""
        self.frames_since_detection = 0

    def get_stats(self) -> dict:
        """Get scheduling statistics.

        Returns:
            Dict with tier counts, percentages, and detection rate.
        """
        total = max(1, self.total_frames)
        return {
            "total_frames": self.total_frames,
            "tier_counts": self.tier_counts.copy(),
            "tier_percentages": {
                k: v / total * 100 for k, v in self.tier_counts.items()
            },
            "detection_rate": (self.tier_counts[1] + self.tier_counts[2]) / total * 100,
            "skip_rate": self.tier_counts[0] / total * 100,
        }


def calibrate_device(
    detector,
    num_iterations: int = 10,
) -> Tuple[int, float]:
    """Calibrate base_skip based on device detection latency.

    Runs detector on dummy frames and measures average latency.
    Returns recommended base_skip for this device.

    Args:
        detector: NanoDetDetector instance.
        num_iterations: Number of warmup/timing iterations.

    Returns:
        Tuple of (base_skip, detection_latency_ms).
    """
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        detector.detect(dummy_frame)

    # Timed runs
    times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        detector.detect(dummy_frame)
        times.append(time.perf_counter() - t0)

    detection_latency = np.mean(times)
    detection_latency_ms = detection_latency * 1000

    # Determine base_skip from latency
    # Goal: ~30 FPS total, so detection should be ~33ms budget
    # If detection takes 30ms, we can detect every frame
    # If detection takes 100ms, we need to skip 2-3 frames
    if detection_latency < 0.020:  # 20ms - fast device
        base_skip = 1
    elif detection_latency < 0.035:  # 35ms - medium-fast
        base_skip = 2
    elif detection_latency < 0.050:  # 50ms - medium
        base_skip = 3
    elif detection_latency < 0.080:  # 80ms - medium-slow
        base_skip = 4
    else:  # > 80ms - slow device
        base_skip = 5

    return base_skip, detection_latency_ms


def create_adaptive_scheduler(
    detector=None,
    tiny_detector_path: Optional[str] = None,
    auto_calibrate: bool = True,
    base_skip: Optional[int] = None,
) -> AdaptiveScheduler:
    """Factory function to create configured AdaptiveScheduler.

    Args:
        detector: NanoDetDetector for calibration (optional).
        tiny_detector_path: Path to tiny detector ONNX (optional).
        auto_calibrate: If True and detector provided, calibrate base_skip.
        base_skip: Manual base_skip override.

    Returns:
        Configured AdaptiveScheduler instance.
    """
    # Determine base_skip
    if base_skip is not None:
        skip = base_skip
        latency_ms = None
    elif auto_calibrate and detector is not None:
        skip, latency_ms = calibrate_device(detector)
        print(f"Device calibration: {latency_ms:.1f}ms/detection -> base_skip={skip}")
    else:
        skip = 3  # Safe default

    # Load tiny detector if provided
    tiny = None
    if tiny_detector_path:
        tiny = TinyDetector(tiny_detector_path)
        if tiny.enabled:
            print(f"Loaded tiny detector: {tiny_detector_path}")
        else:
            tiny = None

    return AdaptiveScheduler(base_skip=skip, tiny_detector=tiny)
