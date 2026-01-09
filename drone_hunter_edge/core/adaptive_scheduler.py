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


# Uncertainty thresholds (tuned via ablation_thresholds.py)
# These are trace(P[:3,:3]) values from the Kalman filter
# Post-detection uncertainty: ~2.9, post-skip: ~5-8
#
# Calibrated from actual simulation data (50 episodes):
#   50th percentile: 1.56
#   75th percentile: 11.73
#   95th percentile: 33.31
#   max: 103.48
#
# Decision table (see _decide_tier for full logic):
# | Uncertainty Range         | No Motion    | Has Motion   |
# |---------------------------|--------------|--------------|
# | < VERY_LOW (0.1)          | Tier 0 skip  | Tier 0 skip  |
# | VERY_LOW - LOW (0.1-1)    | Tier 0 skip  | Tier 1 tiny  |
# | LOW - MEDIUM (1-15)       | Tier 1 tiny  | Tier 1 tiny  |
# | MEDIUM - HIGH (15-40)     | Tier 1 tiny  | Tier 2 full  |
# | > HIGH (40+)              | Tier 2 full  | Tier 2 full  |
UNCERTAINTY_VERY_LOW = 0.1
UNCERTAINTY_LOW = 0.5      # Reduced: more Tier 1 instead of Tier 0
UNCERTAINTY_MEDIUM = 8.0   # Reduced: more Tier 1 instead of skip
UNCERTAINTY_HIGH = 40.0    # ~95th percentile


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
        uncertainty_very_low: float = UNCERTAINTY_VERY_LOW,
        uncertainty_low: float = UNCERTAINTY_LOW,
        uncertainty_medium: float = UNCERTAINTY_MEDIUM,
        uncertainty_high: float = UNCERTAINTY_HIGH,
    ):
        """Initialize adaptive scheduler.

        Args:
            base_skip: Minimum frames between detections (hardware dependent).
                Set via calibrate_device() or manually based on device speed.
            tiny_detector: Optional TinyDetector for Tier 1 detection.
            motion_threshold: Threshold for motion detection (default 5.0).
            uncertainty_very_low: Below this, skip even with motion.
            uncertainty_low: Below this, use tiny detector (if motion) or skip.
            uncertainty_medium: Below this with motion, use Tier 1 instead of Tier 2.
            uncertainty_high: Above this, always use Tier 2 (full detection).
        """
        self.base_skip = base_skip
        self.tiny_detector = tiny_detector
        self.motion_detector = MotionDetector(threshold=motion_threshold)

        self.uncertainty_very_low = uncertainty_very_low
        self.uncertainty_low = uncertainty_low
        self.uncertainty_medium = uncertainty_medium
        self.uncertainty_high = uncertainty_high

        # State
        self.frames_since_detection = 0
        self.last_tier = 0
        self.has_active_tracks = False  # Must do full detect to discover drones

        # Statistics
        self.tier_counts = {0: 0, 1: 0, 2: 0}
        self.tier0_reasons = {
            "no_tracks_budget": 0,    # Rule 0: No tracks, under budget
            "no_tracks_no_motion": 0, # Rule 0: No tracks, no motion
            "under_budget": 0,        # Rule 3: Under budget
            "very_low_uncert": 0,     # Rule 4: Very low uncertainty
            "low_uncert_no_motion": 0,# Rule 5: Low uncertainty, no motion
        }
        self.tier2_reasons = {
            "no_tracks": 0,      # Rule 0: No active tracks, need to discover
            "high_uncertainty": 0,  # Rule 1: Uncertainty > 8.0
            "staleness": 0,      # Rule 2: Too long since last detection
            "motion_medium": 0,  # Rule 6: Medium uncertainty + motion
            "fallback": 0,       # Tiny detector unavailable fallback
        }
        self.total_frames = 0

    def reset(self) -> None:
        """Reset scheduler state for new episode."""
        self.frames_since_detection = 0
        self.last_tier = 0
        self.motion_detector.reset()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.tier_counts = {0: 0, 1: 0, 2: 0}
        self.tier0_reasons = {
            "no_tracks_budget": 0,
            "no_tracks_no_motion": 0,
            "under_budget": 0,
            "very_low_uncert": 0,
            "low_uncert_no_motion": 0,
        }
        self.tier2_reasons = {
            "no_tracks": 0,
            "high_uncertainty": 0,
            "staleness": 0,
            "motion_medium": 0,
            "fallback": 0,
        }
        self.total_frames = 0

    def get_detection_tier(
        self,
        frame: np.ndarray,
        kalman_uncertainty: float,
        max_urgency: float = 0.0,
    ) -> int:
        """Decide which detection tier to use.

        Args:
            frame: Current frame (H, W, 3) uint8 RGB.
            kalman_uncertainty: Scalar uncertainty from tracker.get_max_uncertainty().
            max_urgency: Maximum urgency across tracks (0-1). High urgency = close diving drone.

        Returns:
            Detection tier:
            - 0: Skip (use Kalman prediction only)
            - 1: Tiny detector at predicted ROI
            - 2: Full NanoDet detection
        """
        self.total_frames += 1
        self.frames_since_detection += 1

        # CRITICAL: Force T2 when urgency is high (close diving drones)
        # T1's small 40x40 ROI can't capture large close targets
        if max_urgency > 0.6:
            self.tier_counts[2] += 1
            self.tier2_reasons["high_urgency"] = self.tier2_reasons.get("high_urgency", 0) + 1
            self.frames_since_detection = 0
            self.last_tier = 2
            return 2

        # Check motion
        has_motion, motion_score = self.motion_detector.detect_motion(frame)

        # Decision logic
        tier = self._decide_tier(has_motion, kalman_uncertainty)

        # Update state if we're doing full detection
        # Only Tier 2 can discover NEW drones, so only Tier 2 resets staleness counter
        # Tier 1 only looks at existing track ROIs, can't find new threats
        if tier == 2:
            self.frames_since_detection = 0

        self.last_tier = tier
        self.tier_counts[tier] += 1

        return tier

    def _decide_tier(self, has_motion: bool, uncertainty: float) -> int:
        """Core tier decision logic.

        New tiered logic (v2):
        | Uncertainty     | No Motion      | Has Motion     |
        |-----------------|----------------|----------------|
        | < 2.0 (v.low)   | Tier 0 (skip)  | Tier 0 (skip)  |
        | 2.0-5.0 (low)   | Tier 0 (skip)  | Tier 1 (tiny)  |
        | 5.0-8.0 (med)   | Tier 1 (tiny)  | Tier 2 (full)  |
        | > 8.0 (high)    | Tier 2 (full)  | Tier 2 (full)  |

        Special case: No active tracks → must use Tier 2 to discover drones
        (Tier 1 needs predicted ROI location from existing tracks)

        Args:
            has_motion: True if motion detected in frame.
            uncertainty: Kalman uncertainty value.

        Returns:
            Detection tier (0, 1, or 2).
        """
        has_tiny = self.tiny_detector and self.tiny_detector.enabled

        # Rule 0: No tracks = need full detection to discover drones
        # (Tier 1 can only confirm existing tracks, not find new ones)
        if not self.has_active_tracks:
            # Still respect budget to avoid spamming when scene is empty
            if self.frames_since_detection < self.base_skip:
                self.tier0_reasons["no_tracks_budget"] += 1
                return 0
            # Check for new drones if motion or been a while
            if has_motion or self.frames_since_detection > self.base_skip * 3:
                self.tier2_reasons["no_tracks"] += 1
                return 2
            self.tier0_reasons["no_tracks_no_motion"] += 1
            return 0  # No motion, no tracks = nothing happening

        # Rule 1: High uncertainty - use Tier 1 if available to refresh tracks
        # Only use Tier 2 if no tiny detector
        if uncertainty > self.uncertainty_high:
            has_tiny = self.tiny_detector and self.tiny_detector.enabled
            if has_tiny:
                # Tier 1 can refresh existing tracks cheaply
                return 1
            self.tier2_reasons["high_uncertainty"] += 1
            return 2

        # Rule 2: Staleness check - periodic full detect to catch new drones
        if self.frames_since_detection > self.base_skip * 5:
            self.tier2_reasons["staleness"] += 1
            return 2

        # Rule 3: Under budget - skip unless uncertainty warrants Tier 1
        if self.frames_since_detection < self.base_skip:
            # Allow Tier 1 even under budget if uncertainty is medium+
            # This helps on slow devices with high base_skip
            if uncertainty > self.uncertainty_low and has_tiny:
                return 1
            self.tier0_reasons["under_budget"] += 1
            return 0

        # Rule 4: Very low uncertainty = skip (track is solid)
        if uncertainty < self.uncertainty_very_low:
            self.tier0_reasons["very_low_uncert"] += 1
            return 0

        # Rule 5: Low uncertainty (0.1-1.0)
        if uncertainty < self.uncertainty_low:
            if has_motion:
                if has_tiny:
                    return 1
                else:
                    self.tier2_reasons["fallback"] += 1
                    return 2
            else:
                self.tier0_reasons["low_uncert_no_motion"] += 1
                return 0  # No motion, skip

        # Rule 6: Medium uncertainty (LOW to HIGH)
        # Split: below MEDIUM (25) use Tier 1 even with motion
        #        above MEDIUM use Tier 2 with motion
        if has_motion:
            if uncertainty < self.uncertainty_medium:
                # Lower-medium: Tier 1 is sufficient
                if has_tiny:
                    return 1
                else:
                    self.tier2_reasons["fallback"] += 1
                    return 2
            else:
                # Upper-medium: need full detection
                self.tier2_reasons["motion_medium"] += 1
                return 2
        else:
            if has_tiny:
                return 1
            else:
                self.tier2_reasons["fallback"] += 1
                return 2

    def mark_detection_complete(self, num_tracks: int = 0) -> None:
        """Update track state after detection.

        NOTE: Does NOT reset frames_since_detection - that's now handled
        internally in get_detection_tier() to only reset for Tier 2
        (full detection that can discover new drones).

        Args:
            num_tracks: Number of active tracks after this detection.
        """
        # Counter reset removed - now only Tier 2 resets it (in get_detection_tier)
        self.has_active_tracks = num_tracks > 0

    def get_stats(self) -> dict:
        """Get scheduling statistics.

        Returns:
            Dict with tier counts, percentages, and detection rate.
        """
        total = max(1, self.total_frames)
        tier0_total = max(1, self.tier_counts[0])
        tier2_total = max(1, self.tier_counts[2])
        return {
            "total_frames": self.total_frames,
            "tier_counts": self.tier_counts.copy(),
            "tier_percentages": {
                k: v / total * 100 for k, v in self.tier_counts.items()
            },
            "detection_rate": (self.tier_counts[1] + self.tier_counts[2]) / total * 100,
            "skip_rate": self.tier_counts[0] / total * 100,
            "tier0_reasons": self.tier0_reasons.copy(),
            "tier0_reason_percentages": {
                k: v / tier0_total * 100 for k, v in self.tier0_reasons.items()
            },
            "tier2_reasons": self.tier2_reasons.copy(),
            "tier2_reason_percentages": {
                k: v / tier2_total * 100 for k, v in self.tier2_reasons.items()
            },
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
