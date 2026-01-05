"""Kalman filter-based multi-object tracker for drone depth estimation.

When using a detector instead of oracle mode, we don't have ground truth z or vz.
This module estimates depth (z) from bounding box size and velocity (vz) from
changes in z over time using Kalman filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    """A single detection from the object detector."""
    x: float           # Center x (normalized 0-1)
    y: float           # Center y (normalized 0-1)
    w: float           # Width (normalized)
    h: float           # Height (normalized)
    confidence: float  # Detection confidence
    class_id: int = 0  # Class ID (for multi-class detectors)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class DroneTrack:
    """Single drone track with Kalman filter state for depth estimation.

    State vector: [z, vz] where z is depth and vz is depth velocity.
    Measurement: z estimated from bounding box size.
    """
    track_id: int
    x: np.ndarray              # State [z, vz]
    P: np.ndarray              # Covariance matrix (2x2)
    bbox: Tuple[float, float, float, float]  # Last (x, y, w, h)
    age: int = 0               # Frames since creation
    hits: int = 1              # Consecutive detections
    misses: int = 0            # Consecutive missed detections
    confidence: float = 0.5    # Track confidence

    # Kalman filter matrices (shared across all tracks)
    # State transition: constant velocity model
    F: np.ndarray = field(default_factory=lambda: np.array([
        [1, 1],  # z_new = z + vz * dt (dt=1 frame)
        [0, 1],  # vz_new = vz
    ], dtype=np.float32))

    # Measurement matrix: we observe z only
    H: np.ndarray = field(default_factory=lambda: np.array([
        [1, 0],  # z_measured
    ], dtype=np.float32))

    # Process noise (model uncertainty)
    Q: np.ndarray = field(default_factory=lambda: np.array([
        [0.01, 0],
        [0, 0.001],
    ], dtype=np.float32))

    # Measurement noise (sensor uncertainty)
    R: np.ndarray = field(default_factory=lambda: np.array([
        [0.05],  # z measurement variance
    ], dtype=np.float32))

    @property
    def z(self) -> float:
        """Estimated depth."""
        return float(self.x[0])

    @property
    def vz(self) -> float:
        """Estimated depth velocity."""
        return float(self.x[1])

    @property
    def center(self) -> Tuple[float, float]:
        """Last known center position."""
        return (self.bbox[0], self.bbox[1])

    def predict(self) -> np.ndarray:
        """Predict next state using constant velocity model.

        Returns:
            Predicted state [z, vz]
        """
        # State prediction: x' = F @ x
        self.x = self.F @ self.x

        # Covariance prediction: P' = F @ P @ F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        return self.x

    def update(self, z_measured: float, detection: Detection) -> None:
        """Update state with new measurement.

        Args:
            z_measured: Depth estimated from bounding box
            detection: The detection used for measurement
        """
        # Measurement
        z_meas = np.array([[z_measured]], dtype=np.float32)

        # Innovation (measurement residual): y = z - H @ x
        y = z_meas - self.H @ self.x

        # Innovation covariance: S = H @ P @ H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P @ H^T @ S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K @ y
        self.x = self.x + (K @ y).flatten()

        # Covariance update: P = (I - K @ H) @ P
        I = np.eye(2, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        # Update bbox and counters
        self.bbox = (detection.x, detection.y, detection.w, detection.h)
        self.hits += 1
        self.misses = 0
        self.confidence = min(1.0, self.confidence + 0.1)

    def mark_missed(self) -> None:
        """Mark this track as having missed a detection."""
        self.misses += 1
        self.hits = 0
        self.confidence = max(0.0, self.confidence - 0.2)


class KalmanTracker:
    """Multi-object tracker using Kalman filters for depth estimation.

    Associates detections to existing tracks using Hungarian algorithm,
    maintains track state with Kalman filtering, and estimates z/vz.
    """

    # Depth estimation calibration
    # Reference: at z=0.5, a drone has bbox_height ≈ 0.06 (based on base_size=0.06)
    # Formula: size = base_size * (0.5 / z), so at z=0.5, size = base_size
    REFERENCE_HEIGHT: float = 0.06
    REFERENCE_Z: float = 0.5

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 2,
        iou_threshold: float = 0.2,
    ):
        """Initialize tracker.

        Args:
            max_age: Maximum frames a track can be unmatched before deletion
            min_hits: Minimum hits before track is considered confirmed
            iou_threshold: Minimum IOU for detection-track association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[DroneTrack] = []
        self.next_id = 0
        self.frame_count = 0

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0

    def estimate_depth(self, bbox_height: float) -> float:
        """Estimate depth from bounding box height.

        Larger bbox = closer to camera = smaller z.
        Uses inverse relationship calibrated to reference values.

        Args:
            bbox_height: Normalized bounding box height (0-1)

        Returns:
            Estimated depth z (clamped to 0.1-1.0)
        """
        if bbox_height <= 0.001:
            return 1.0  # Very small = far away

        # z is inversely proportional to bbox height
        # z = reference_z * (reference_height / bbox_height)
        z = self.REFERENCE_Z * (self.REFERENCE_HEIGHT / bbox_height)

        # Clamp to valid range
        return float(np.clip(z, 0.1, 1.0))

    def _compute_iou(self, box1: Tuple[float, float, float, float],
                     box2: Tuple[float, float, float, float]) -> float:
        """Compute IoU between two boxes (x, y, w, h format).

        Args:
            box1: First box (center_x, center_y, width, height)
            box2: Second box (center_x, center_y, width, height)

        Returns:
            Intersection over Union (0-1)
        """
        # Convert to corner format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _associate_detections(
        self, detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using Hungarian algorithm.

        Args:
            detections: List of current frame detections

        Returns:
            Tuple of:
            - matches: List of (track_idx, detection_idx) pairs
            - unmatched_tracks: List of track indices without matches
            - unmatched_detections: List of detection indices without matches
        """
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))

        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []

        # Build cost matrix (negative IOU for minimization)
        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                det_box = (det.x, det.y, det.w, det.h)
                iou = self._compute_iou(track.bbox, det_box)
                cost_matrix[t_idx, d_idx] = 1 - iou  # Cost = 1 - IOU

        # Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))

        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] <= (1 - self.iou_threshold):
                matches.append((t_idx, d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(d_idx)

        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections: List[Detection]) -> List[DroneTrack]:
        """Update tracker with new detections.

        Args:
            detections: List of detections from current frame

        Returns:
            List of active (confirmed) tracks
        """
        self.frame_count += 1

        # Predict all tracks forward
        for track in self.tracks:
            track.predict()

        # Associate detections to tracks
        matches, unmatched_tracks, unmatched_dets = self._associate_detections(detections)

        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]

            # Estimate z from bbox
            z_measured = self.estimate_depth(det.h)

            # Update track with measurement
            track.update(z_measured, det)

        # Mark unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]

            # Initial state
            z_init = self.estimate_depth(det.h)
            vz_init = 0.0  # No velocity estimate yet

            # Initial covariance (high uncertainty)
            P_init = np.array([
                [0.1, 0],
                [0, 0.1],
            ], dtype=np.float32)

            track = DroneTrack(
                track_id=self.next_id,
                x=np.array([z_init, vz_init], dtype=np.float32),
                P=P_init,
                bbox=(det.x, det.y, det.w, det.h),
            )
            self.tracks.append(track)
            self.next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        # Return confirmed tracks only
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def get_tracks_for_observation(self) -> List[DroneTrack]:
        """Get active tracks suitable for agent observation.

        Returns tracks sorted by urgency (approaching tracks first).
        """
        confirmed = [t for t in self.tracks if t.hits >= self.min_hits]

        # Sort by urgency: approaching (vz < 0) and close (low z) first
        def urgency(track: DroneTrack) -> float:
            if track.vz < 0:  # Approaching
                frames_to_impact = track.z / max(0.001, abs(track.vz))
                return 1.0 / (1.0 + frames_to_impact / 50.0)
            else:
                return 0.1

        return sorted(confirmed, key=lambda t: -urgency(t))
