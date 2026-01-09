"""Kalman filter-based multi-object tracker for drone depth estimation.

Pure Python implementation without scipy dependency for edge deployment.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from core.detection import Detection


def hungarian_algorithm(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python Hungarian algorithm for optimal assignment.

    Simple O(n^3) implementation suitable for small matrices (<20x20).

    Args:
        cost_matrix: NxM cost matrix (rows=tracks, cols=detections)

    Returns:
        Tuple of (row_indices, col_indices) for optimal assignment
    """
    n_rows, n_cols = cost_matrix.shape

    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Pad to square matrix
    n = max(n_rows, n_cols)
    padded = np.full((n, n), cost_matrix.max() + 1, dtype=np.float32)
    padded[:n_rows, :n_cols] = cost_matrix

    # Step 1: Subtract row minimum
    padded -= padded.min(axis=1, keepdims=True)

    # Step 2: Subtract column minimum
    padded -= padded.min(axis=0, keepdims=True)

    # Iterative assignment
    max_iter = n * 2
    for _ in range(max_iter):
        # Find zeros and try to assign
        row_assigned = np.full(n, -1, dtype=np.int32)
        col_assigned = np.full(n, -1, dtype=np.int32)

        for i in range(n):
            for j in range(n):
                if padded[i, j] == 0 and row_assigned[i] == -1 and col_assigned[j] == -1:
                    row_assigned[i] = j
                    col_assigned[j] = i

        # Count assignments
        n_assigned = np.sum(row_assigned >= 0)
        if n_assigned == n:
            break

        # Mark rows without assignment
        row_covered = row_assigned >= 0
        col_covered = np.zeros(n, dtype=bool)

        # Cover columns with zeros in uncovered rows
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not row_covered[i]:
                    for j in range(n):
                        if padded[i, j] == 0 and not col_covered[j]:
                            col_covered[j] = True
                            changed = True

            for j in range(n):
                if col_covered[j]:
                    for i in range(n):
                        if row_assigned[i] == j and row_covered[i]:
                            row_covered[i] = False
                            changed = True

        # Find minimum uncovered value
        min_val = float('inf')
        for i in range(n):
            for j in range(n):
                if not row_covered[i] and not col_covered[j]:
                    min_val = min(min_val, padded[i, j])

        if min_val == float('inf'):
            break

        # Subtract from uncovered, add to doubly covered
        for i in range(n):
            for j in range(n):
                if not row_covered[i] and not col_covered[j]:
                    padded[i, j] -= min_val
                elif row_covered[i] and col_covered[j]:
                    padded[i, j] += min_val

    # Extract valid assignments (within original matrix bounds)
    rows = []
    cols = []
    for i in range(n_rows):
        if row_assigned[i] >= 0 and row_assigned[i] < n_cols:
            rows.append(i)
            cols.append(row_assigned[i])

    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)


@dataclass
class DroneTrack:
    """Single drone track with Kalman filter state for position and depth.

    State vector: [x, y, z, vx, vy, vz]
    - x, y: normalized screen position (0-1)
    - z: depth (0.1-1.0)
    - vx, vy, vz: velocities (per frame)

    Measurements: x, y from bbox center, z from bbox height.
    """
    track_id: int
    state: np.ndarray          # State [x, y, z, vx, vy, vz]
    P: np.ndarray              # Covariance matrix (6x6)
    bbox_size: Tuple[float, float]  # Last (w, h) for IoU calculation
    age: int = 0
    hits: int = 1
    misses: int = 0
    confidence: float = 0.5

    # Kalman filter matrices (6-state constant velocity model)
    F: np.ndarray = field(default_factory=lambda: np.array([
        [1, 0, 0, 1, 0, 0],  # x_new = x + vx
        [0, 1, 0, 0, 1, 0],  # y_new = y + vy
        [0, 0, 1, 0, 0, 1],  # z_new = z + vz
        [0, 0, 0, 1, 0, 0],  # vx_new = vx
        [0, 0, 0, 0, 1, 0],  # vy_new = vy
        [0, 0, 0, 0, 0, 1],  # vz_new = vz
    ], dtype=np.float32))

    # Measurement matrix: we observe x, y, z
    H: np.ndarray = field(default_factory=lambda: np.array([
        [1, 0, 0, 0, 0, 0],  # measure x
        [0, 1, 0, 0, 0, 0],  # measure y
        [0, 0, 1, 0, 0, 0],  # measure z
    ], dtype=np.float32))

    # Process noise (model uncertainty)
    Q: np.ndarray = field(default_factory=lambda: np.array([
        [0.001, 0, 0, 0, 0, 0],      # x position noise
        [0, 0.001, 0, 0, 0, 0],      # y position noise
        [0, 0, 0.01, 0, 0, 0],       # z position noise (depth less certain)
        [0, 0, 0, 0.005, 0, 0],      # vx noise
        [0, 0, 0, 0, 0.005, 0],      # vy noise
        [0, 0, 0, 0, 0, 0.001],      # vz noise
    ], dtype=np.float32))

    # Measurement noise (sensor uncertainty)
    R: np.ndarray = field(default_factory=lambda: np.array([
        [0.01, 0, 0],     # x measurement variance
        [0, 0.01, 0],     # y measurement variance
        [0, 0, 0.05],     # z measurement variance (depth estimation)
    ], dtype=np.float32))

    @property
    def x(self) -> float:
        """Estimated x position (normalized 0-1)."""
        return float(self.state[0])

    @property
    def y(self) -> float:
        """Estimated y position (normalized 0-1)."""
        return float(self.state[1])

    @property
    def z(self) -> float:
        """Estimated depth."""
        return float(self.state[2])

    @property
    def vx(self) -> float:
        """Estimated x velocity."""
        return float(self.state[3])

    @property
    def vy(self) -> float:
        """Estimated y velocity."""
        return float(self.state[4])

    @property
    def vz(self) -> float:
        """Estimated depth velocity."""
        return float(self.state[5])

    @property
    def center(self) -> Tuple[float, float]:
        """Predicted center position (x, y)."""
        return (self.x, self.y)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Construct bbox from predicted position and last known size."""
        return (self.x, self.y, self.bbox_size[0], self.bbox_size[1])

    def predict(self) -> np.ndarray:
        """Predict next state using constant velocity model."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        return self.state

    def update(self, z_measured: float, detection: Detection) -> None:
        """Update state with new measurement."""
        # Measurement vector: [x, y, z]
        measurement = np.array([
            [detection.x],
            [detection.y],
            [z_measured],
        ], dtype=np.float32)

        # Innovation (measurement residual)
        y = measurement - self.H @ self.state.reshape(-1, 1)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + (K @ y).flatten()

        # Covariance update
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        # Update bbox size for IoU calculation
        self.bbox_size = (detection.w, detection.h)
        self.hits += 1
        self.misses = 0
        self.confidence = min(1.0, self.confidence + 0.1)

    def mark_missed(self) -> None:
        """Mark this track as having missed a detection."""
        self.misses += 1
        self.hits = 0
        self.confidence = max(0.0, self.confidence - 0.2)

    def get_position_uncertainty(self) -> float:
        """Scalar uncertainty from position covariance.

        Returns the trace of the position submatrix of P (x, y, z variances).
        Higher values indicate less certainty about the track's position.
        Used by adaptive scheduler to decide when to run detection.
        """
        return float(np.trace(self.P[:3, :3]))


class KalmanTracker:
    """Multi-object tracker using Kalman filters for depth estimation."""

    # Adjusted for edge: smaller reference to get better z resolution for small drones
    # Original: 0.06 @ z=0.5 means bbox=0.03 gives z=1.0 (clamped)
    # New: 0.03 @ z=0.5 means bbox=0.03 gives z=0.5, bbox=0.06 gives z=0.25
    REFERENCE_HEIGHT: float = 0.03
    REFERENCE_Z: float = 0.5

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 1,  # Reduced from 2 - detections are intermittent
        iou_threshold: float = 0.2,
    ):
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
        """Estimate depth from bounding box height."""
        if bbox_height <= 0.001:
            return 1.0
        z = self.REFERENCE_Z * (self.REFERENCE_HEIGHT / bbox_height)
        return float(np.clip(z, 0.1, 1.0))

    def _compute_iou(self, box1: Tuple[float, float, float, float],
                     box2: Tuple[float, float, float, float]) -> float:
        """Compute IoU between two boxes (x, y, w, h format)."""
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _associate_detections(
        self, detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using Hungarian algorithm."""
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))

        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []

        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                det_box = (det.x, det.y, det.w, det.h)
                iou = self._compute_iou(track.bbox, det_box)
                cost_matrix[t_idx, d_idx] = 1 - iou

        track_indices, det_indices = hungarian_algorithm(cost_matrix)

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))

        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] <= (1 - self.iou_threshold):
                matches.append((t_idx, d_idx))
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_detections:
                    unmatched_detections.remove(d_idx)

        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections: List[Detection]) -> List[DroneTrack]:
        """Update tracker with new detections."""
        self.frame_count += 1

        for track in self.tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_dets = self._associate_detections(detections)

        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            z_measured = self.estimate_depth(det.h)
            track.update(z_measured, det)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_dets:
            det = detections[det_idx]

            # Initial state: [x, y, z, vx, vy, vz]
            z_init = self.estimate_depth(det.h)
            state_init = np.array([
                det.x,    # x position from detection
                det.y,    # y position from detection
                z_init,   # z estimated from bbox height
                0.0,      # vx unknown
                0.0,      # vy unknown
                0.0,      # vz unknown
            ], dtype=np.float32)

            # Initial covariance (6x6, high uncertainty for velocities)
            P_init = np.diag([
                0.01,   # x position variance (pretty certain)
                0.01,   # y position variance (pretty certain)
                0.1,    # z variance (less certain from height)
                0.1,    # vx variance (unknown)
                0.1,    # vy variance (unknown)
                0.1,    # vz variance (unknown)
            ]).astype(np.float32)

            track = DroneTrack(
                track_id=self.next_id,
                state=state_init,
                P=P_init,
                bbox_size=(det.w, det.h),
            )
            self.tracks.append(track)
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def get_tracks_for_observation(self) -> List[DroneTrack]:
        """Get active tracks sorted by urgency (approaching tracks first)."""
        confirmed = [t for t in self.tracks if t.hits >= self.min_hits]

        def urgency(track: DroneTrack) -> float:
            if track.vz < 0:
                frames_to_impact = track.z / max(0.001, abs(track.vz))
                return 1.0 / (1.0 + frames_to_impact / 50.0)
            else:
                return 0.1

        return sorted(confirmed, key=lambda t: -urgency(t))

    def get_max_uncertainty(self) -> float:
        """Get highest uncertainty across all tracks.

        Returns 1.0 if no tracks exist (high uncertainty = need detection).
        Used by adaptive scheduler to trigger detection when prediction
        confidence is low.
        """
        if not self.tracks:
            return 1.0  # No tracks = high uncertainty, need detection
        return max(t.get_position_uncertainty() for t in self.tracks)

    def get_mean_uncertainty(self) -> float:
        """Get mean uncertainty across all tracks.

        Returns 1.0 if no tracks exist.
        Alternative to get_max_uncertainty() for smoother scheduling.
        """
        if not self.tracks:
            return 1.0
        return sum(t.get_position_uncertainty() for t in self.tracks) / len(self.tracks)
