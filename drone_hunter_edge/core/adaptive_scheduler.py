"""Adaptive detection scheduler for hierarchical compute allocation.

Core insight: spend detection budget where uncertainty is high, coast on
Kalman prediction where it's low.

Detection Tiers:
- Tier 0: Skip (Kalman predict only) - ~0ms
- Tier 1: Tiny detector at predicted ROI - ~2-5ms (when available)
- Tier 2: Full NanoDet (320x320) - ~30ms

Key design decisions (v2 - principled thresholds):
1. Uncertainty metric is x,y only (not z) - that's what ROI cares about
2. Thresholds are based on ROI geometry, not empirical tuning
3. Missed associations boost uncertainty (prevents "confident but lost")
4. Guaranteed discovery scans every N frames (like skip-N's retry)
"""

from typing import Optional, Tuple
import time
import numpy as np

from core.tiny_detector import TinyDetector


# =============================================================================
# ROI Geometry Constants
# =============================================================================
# Tier 1 uses a 40x40 ROI on a 320x320 frame
TIER1_ROI_SIZE = 40 / 320           # 0.125 normalized
TIER1_ROI_RADIUS = TIER1_ROI_SIZE / 2  # 0.0625 - half the ROI width

# Size limits for Tier 1 viability
# NanoDet bboxes are typically 0.10-0.20 (max of w,h)
# The 40x40 ROI (0.125 normalized) will partially crop larger drones,
# but the tiny detector can still work if enough of the drone is visible.
# Setting threshold at 0.18 to allow Tier 1 for most drones.
TIER1_MAX_DRONE_SIZE = 0.18   # Drone too big for 40x40 ROI (typically 0.10-0.20)
TIER1_MIN_DRONE_SIZE = 0.015  # Drone too small for tiny detector accuracy

# =============================================================================
# Uncertainty Thresholds (calibrated from empirical distribution)
# =============================================================================
# The uncertainty metric is sqrt(P[x,x] + P[y,y]) - the x,y position standard
# deviation from the Kalman covariance matrix.
#
# Empirical distribution (from simulation with 70% detection rate):
#   25th percentile: 0.12 (just after Kalman update)
#   50th percentile: 0.13
#   75th percentile: 0.22 (after a few prediction-only frames)
#   95th percentile: 0.37 (after many consecutive skips)
#
# Thresholds chosen to match distribution percentiles:
UNCERTAINTY_TIER0_MAX = 0.20   # Confident, safe to skip
UNCERTAINTY_TIER1_MAX = 0.35   # Moderate uncertainty, use tiny detector if drone fits
# Above TIER1_MAX -> use Tier 2 (high uncertainty, need full detection)

# Guaranteed discovery interval (like skip-N's unconditional retry)
DISCOVERY_INTERVAL = 10  # Max frames between Tier 2 scans


class AdaptiveScheduler:
    """Adaptive scheduler for hierarchical detection.

    Decides which detection tier to use based on:
    - Kalman filter x,y prediction uncertainty
    - Drone size (too big/small for Tier 1 ROI)
    - Guaranteed discovery interval (periodic Tier 2 scans)

    v2 Changes:
    - Removed motion detector (uncertainty metric now captures drift)
    - Thresholds based on ROI geometry, not empirical tuning
    - Guaranteed discovery scans prevent indefinite skipping
    """

    def __init__(
        self,
        tiny_detector: Optional[TinyDetector] = None,
        discovery_interval: int = DISCOVERY_INTERVAL,
        uncertainty_tier0_max: float = UNCERTAINTY_TIER0_MAX,
        uncertainty_tier1_max: float = UNCERTAINTY_TIER1_MAX,
    ):
        """Initialize adaptive scheduler.

        Args:
            tiny_detector: Optional TinyDetector for Tier 1 detection.
            discovery_interval: Max frames between Tier 2 scans (default 10).
            uncertainty_tier0_max: Max uncertainty for Tier 0 (default ~0.019).
            uncertainty_tier1_max: Max uncertainty for Tier 1 (default ~0.050).
        """
        self.tiny_detector = tiny_detector
        self.discovery_interval = discovery_interval
        self.uncertainty_tier0_max = uncertainty_tier0_max
        self.uncertainty_tier1_max = uncertainty_tier1_max

        # State
        self.frames_since_tier2 = 0
        self.last_tier = 0

        # Statistics
        self.tier_counts = {0: 0, 1: 0, 2: 0}
        self.tier_reasons = {
            # Tier 2 reasons
            "periodic_discovery": 0,
            "no_tracks": 0,
            "size_exceeds_roi": 0,
            "size_too_small": 0,
            "uncertainty_exceeds_roi": 0,
            "no_tiny_detector": 0,
            # Tier 1 reasons
            "moderate_uncertainty": 0,
            # Tier 0 reasons
            "confident_prediction": 0,
        }
        self.total_frames = 0

    def reset(self) -> None:
        """Reset scheduler state for new episode."""
        self.frames_since_tier2 = 0
        self.last_tier = 0

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.tier_counts = {0: 0, 1: 0, 2: 0}
        self.tier_reasons = {
            "periodic_discovery": 0,
            "no_tracks": 0,
            "size_exceeds_roi": 0,
            "size_too_small": 0,
            "uncertainty_exceeds_roi": 0,
            "no_tiny_detector": 0,
            "moderate_uncertainty": 0,
            "confident_prediction": 0,
        }
        self.total_frames = 0

    def get_detection_tier(
        self,
        xy_uncertainty: float,
        num_tracks: int,
        max_size: float = 0.0,
    ) -> int:
        """Decide which detection tier to use.

        Args:
            xy_uncertainty: X,Y prediction uncertainty from tracker.get_max_uncertainty().
                This is sqrt(x_var + y_var) in normalized screen units.
            num_tracks: Number of active tracks.
            max_size: Largest drone size (normalized bbox dimension).

        Returns:
            Detection tier:
            - 0: Skip (use Kalman prediction only)
            - 1: Tiny detector at predicted ROI
            - 2: Full NanoDet detection
        """
        self.total_frames += 1
        self.frames_since_tier2 += 1

        # Core decision logic
        tier, reason = self._decide_tier(xy_uncertainty, num_tracks, max_size)

        # Only Tier 2 resets discovery counter
        if tier == 2:
            self.frames_since_tier2 = 0

        self.last_tier = tier
        self.tier_counts[tier] += 1
        self.tier_reasons[reason] += 1

        return tier

    def _decide_tier(
        self,
        xy_uncertainty: float,
        num_tracks: int,
        max_size: float,
    ) -> Tuple[int, str]:
        """Core tier decision logic.

        Principled decision tree based on ROI geometry:

        | Condition                      | Tier | Reason                  |
        |--------------------------------|------|-------------------------|
        | frames_since_tier2 >= interval | 2    | periodic_discovery      |
        | num_tracks == 0                | 2    | no_tracks               |
        | max_size > TIER1_MAX           | 2    | size_exceeds_roi        |
        | max_size < TIER1_MIN           | 2    | size_too_small          |
        | uncertainty > TIER1_MAX        | 2    | uncertainty_exceeds_roi |
        | uncertainty > TIER0_MAX        | 1    | moderate_uncertainty    |
        | otherwise                      | 0    | confident_prediction    |

        Args:
            xy_uncertainty: X,Y prediction uncertainty (normalized screen units).
            num_tracks: Number of active tracks.
            max_size: Largest drone size (normalized).

        Returns:
            Tuple of (tier, reason_string).
        """
        has_tiny = self.tiny_detector is not None and self.tiny_detector.enabled

        # Rule 0: UNCONDITIONAL periodic discovery (like skip-N's retry guarantee)
        # This breaks the feedback loop that caused adaptive mode to fail
        if self.frames_since_tier2 >= self.discovery_interval:
            return 2, "periodic_discovery"

        # Rule 1: No tracks - must discover new drones
        # Tier 1 can only confirm existing tracks, not find new ones
        if num_tracks == 0:
            return 2, "no_tracks"

        # Rule 2: Large drones exceed Tier 1 ROI (40x40 can't capture them)
        if max_size > TIER1_MAX_DRONE_SIZE:
            return 2, "size_exceeds_roi"

        # Rule 3: Tiny drones need full resolution for accurate detection
        if 0 < max_size < TIER1_MIN_DRONE_SIZE:
            return 2, "size_too_small"

        # Rule 4: High uncertainty - drone likely outside ROI
        if xy_uncertainty > self.uncertainty_tier1_max:
            return 2, "uncertainty_exceeds_roi"

        # Rule 5: Medium uncertainty - use Tier 1 if available
        if xy_uncertainty > self.uncertainty_tier0_max:
            if has_tiny:
                return 1, "moderate_uncertainty"
            else:
                return 2, "no_tiny_detector"

        # Rule 6: Low uncertainty - safe to skip, predictions are reliable
        return 0, "confident_prediction"

    def get_stats(self) -> dict:
        """Get scheduling statistics.

        Returns:
            Dict with tier counts, percentages, and reason breakdown.
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
            "tier_reasons": self.tier_reasons.copy(),
            "tier_reason_percentages": {
                k: v / max(1, v) * 100 for k, v in self.tier_reasons.items()
            },
            "discovery_interval": self.discovery_interval,
            "uncertainty_thresholds": {
                "tier0_max": self.uncertainty_tier0_max,
                "tier1_max": self.uncertainty_tier1_max,
            },
        }


def calibrate_device(
    detector,
    num_iterations: int = 10,
) -> Tuple[int, float]:
    """Calibrate discovery_interval based on device detection latency.

    Runs detector on dummy frames and measures average latency.
    Returns recommended discovery_interval for this device.

    Args:
        detector: NanoDetDetector instance.
        num_iterations: Number of warmup/timing iterations.

    Returns:
        Tuple of (discovery_interval, detection_latency_ms).
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

    # Determine discovery_interval from latency
    # Faster devices can do more frequent discovery scans
    if detection_latency < 0.020:  # 20ms - fast device
        discovery_interval = 5
    elif detection_latency < 0.035:  # 35ms - medium-fast
        discovery_interval = 8
    elif detection_latency < 0.050:  # 50ms - medium
        discovery_interval = 10
    elif detection_latency < 0.080:  # 80ms - medium-slow
        discovery_interval = 12
    else:  # > 80ms - slow device
        discovery_interval = 15

    return discovery_interval, detection_latency_ms


def create_adaptive_scheduler(
    detector=None,
    tiny_detector_path: Optional[str] = None,
    auto_calibrate: bool = True,
    discovery_interval: Optional[int] = None,
) -> AdaptiveScheduler:
    """Factory function to create configured AdaptiveScheduler.

    Args:
        detector: NanoDetDetector for calibration (optional).
        tiny_detector_path: Path to tiny detector ONNX (optional).
        auto_calibrate: If True and detector provided, calibrate discovery_interval.
        discovery_interval: Manual discovery_interval override.

    Returns:
        Configured AdaptiveScheduler instance.
    """
    # Determine discovery_interval
    if discovery_interval is not None:
        interval = discovery_interval
    elif auto_calibrate and detector is not None:
        interval, latency_ms = calibrate_device(detector)
        print(f"Device calibration: {latency_ms:.1f}ms/detection -> discovery_interval={interval}")
    else:
        interval = DISCOVERY_INTERVAL  # Default from constants

    # Load tiny detector if provided
    tiny = None
    if tiny_detector_path:
        tiny = TinyDetector(tiny_detector_path)
        if tiny.enabled:
            print(f"Loaded tiny detector: {tiny_detector_path}")
        else:
            tiny = None

    return AdaptiveScheduler(tiny_detector=tiny, discovery_interval=interval)
