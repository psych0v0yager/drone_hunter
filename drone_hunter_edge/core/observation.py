"""Observation building and normalization for edge inference."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.kalman_tracker import KalmanTracker, DroneTrack


@dataclass
class NormalizationStats:
    """Statistics for normalizing observations."""
    mean: np.ndarray
    std: np.ndarray
    clip: float


class ObservationNormalizer:
    """Normalizes observations using pre-computed VecNormalize statistics."""

    def __init__(self, stats_path: Path | str):
        """Load normalization statistics from JSON.

        Args:
            stats_path: Path to normalization.json
        """
        with open(stats_path, 'r') as f:
            raw_stats = json.load(f)

        self.stats: Dict[str, NormalizationStats] = {}

        for key, data in raw_stats.items():
            mean = np.array(data["mean"], dtype=np.float32)
            var = np.array(data["var"], dtype=np.float32)
            std = np.sqrt(var + 1e-8)

            self.stats[key] = NormalizationStats(
                mean=mean,
                std=std,
                clip=data.get("clip", 10.0),
            )

    def normalize(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize observation dictionary.

        Args:
            obs: Dict with observation arrays

        Returns:
            Normalized observation dict
        """
        normalized = {}

        for key, value in obs.items():
            if key in self.stats:
                stats = self.stats[key]
                norm_value = (value - stats.mean) / stats.std
                norm_value = np.clip(norm_value, -stats.clip, stats.clip)
                normalized[key] = norm_value.astype(np.float32)
            else:
                normalized[key] = value.astype(np.float32)

        return normalized


def compute_track_urgency(z: float, vz: float) -> float:
    """Compute urgency for a single track.

    Urgency measures how soon a drone will impact (low z + diving = high urgency).

    Args:
        z: Estimated depth (0=impact, 1=far)
        vz: Estimated vertical velocity (negative = approaching)

    Returns:
        Urgency value in [0.1, 1.0]
    """
    if vz < 0:
        frames_to_impact = z / max(0.001, abs(vz))
        return 1.0 / (1.0 + frames_to_impact / 50.0)
    return 0.1


def compute_max_urgency(tracker: KalmanTracker, min_hits: int = 5) -> float:
    """Compute maximum urgency across all mature tracks.

    Used by adaptive scheduler to force T2 when drones are close/diving.

    Args:
        tracker: KalmanTracker with active tracks
        min_hits: Minimum track hits for maturity filter

    Returns:
        Maximum urgency in [0.0, 1.0], or 0.0 if no mature tracks
    """
    tracks = tracker.get_tracks_for_observation()
    tracks = [t for t in tracks if t.hits >= min_hits]

    if not tracks:
        return 0.0

    max_urg = 0.0
    for track in tracks:
        urg = compute_track_urgency(track.z, track.vz)
        max_urg = max(max_urg, urg)

    return max_urg


def build_tracker_observation(
    tracker: KalmanTracker,
    grid_size: int = 8,
    min_hits: int = 5,
) -> Dict[str, np.ndarray]:
    """Build observation from Kalman tracker (detector mode).

    Uses tracker-estimated z and vz instead of ground truth.

    Args:
        tracker: KalmanTracker with active tracks
        grid_size: Size of action grid
        min_hits: Minimum track hits before including in observation.
            Prevents snap firing on new/immature tracks.

    Returns:
        Dict with "target" and "game_state" arrays
    """
    tracks = tracker.get_tracks_for_observation()
    # Filter immature tracks - prevent snap firing on new detections
    tracks = [t for t in tracks if t.hits >= min_hits]

    features_per_target = 3 + grid_size * 2  # 19 for grid_size=8
    target_obs = np.zeros(features_per_target, dtype=np.float32)

    has_target = 0.0
    max_threat = 0.0

    if tracks:
        track = tracks[0]
        has_target = 1.0

        cx, cy = track.center
        grid_x = min(grid_size - 1, max(0, int(cx * grid_size)))
        grid_y = min(grid_size - 1, max(0, int(cy * grid_size)))

        z = track.z
        vz = track.vz

        if vz < 0:
            frames_to_impact = z / max(0.001, abs(vz))
            urgency = 1.0 / (1.0 + frames_to_impact / 50.0)
        else:
            urgency = 0.1

        target_obs[0] = z
        target_obs[1] = vz * 10
        target_obs[2] = urgency
        target_obs[3 + grid_x] = 1.0
        target_obs[3 + grid_size + grid_y] = 1.0

    # Compute threat level from all tracks
    for track in tracks:
        if track.vz < 0:
            threat = 1.0 - track.z
            max_threat = max(max_threat, threat)

    # Note: game_state needs to be filled by caller with actual ammo/reload info
    # This returns a placeholder that should be overwritten
    game_state_obs = np.array([
        0.5,  # ammo_fraction (placeholder)
        0.0,  # reload_fraction (placeholder)
        0.5,  # frame_fraction (placeholder)
        max_threat,
        has_target,
    ], dtype=np.float32)

    return {
        "target": target_obs,
        "game_state": game_state_obs,
    }


def build_oracle_observation(
    drones: List,  # List[Drone] but avoid circular import
    ammo_fraction: float,
    reload_fraction: float,
    frame_fraction: float,
    threat_level: float,
    grid_size: int = 8,
) -> Dict[str, np.ndarray]:
    """Build observation from ground truth drones (oracle mode).

    Args:
        drones: List of Drone objects
        ammo_fraction: Current ammo / clip_size
        reload_fraction: Reload progress (0 if not reloading)
        frame_fraction: Current frame / max_frames
        threat_level: Threat level (0-1)
        grid_size: Size of action grid

    Returns:
        Dict with "target" and "game_state" arrays
    """
    def compute_urgency(drone) -> float:
        if drone.vz < 0:
            frames_to_impact = drone.z / max(0.001, abs(drone.vz))
            return 1.0 / (1.0 + frames_to_impact / 50.0)
        else:
            return 0.1

    features_per_target = 3 + grid_size * 2
    target_obs = np.zeros(features_per_target, dtype=np.float32)

    has_target = 0.0

    if drones:
        sorted_drones = sorted(drones, key=lambda d: -compute_urgency(d))
        drone = sorted_drones[0]
        has_target = 1.0

        grid_x = min(grid_size - 1, max(0, int(drone.x * grid_size)))
        grid_y = min(grid_size - 1, max(0, int(drone.y * grid_size)))
        urgency = compute_urgency(drone)

        target_obs[0] = drone.z
        target_obs[1] = drone.vz * 10
        target_obs[2] = urgency
        target_obs[3 + grid_x] = 1.0
        target_obs[3 + grid_size + grid_y] = 1.0

    game_state_obs = np.array([
        ammo_fraction,
        reload_fraction,
        frame_fraction,
        threat_level,
        has_target,
    ], dtype=np.float32)

    return {
        "target": target_obs,
        "game_state": game_state_obs,
    }
