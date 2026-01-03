"""Game state dataclasses for Drone Hunter environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List
import random
import math


class DroneType(Enum):
    """Types of drones with different behaviors."""
    NORMAL = auto()    # Random trajectory, exits screen
    KAMIKAZE = auto()  # Flies toward camera center
    ERRATIC = auto()   # Zigzag pattern


@dataclass
class Drone:
    """Represents a single drone in the game.

    All positions are normalized to [0, 1] range.
    (0, 0) is top-left, (1, 1) is bottom-right.

    Depth (z) simulates 3D perspective:
    - z = 1.0: Far away (small appearance)
    - z = 0.0: At camera (large appearance, impact for kamikaze)
    """
    x: float                    # Horizontal position (0-1)
    y: float                    # Vertical position (0-1)
    z: float                    # Depth (1.0 = far, 0.0 = close/impact)
    vx: float                   # Horizontal velocity (per frame)
    vy: float                   # Vertical velocity (per frame)
    vz: float                   # Depth velocity (negative = approaching)
    base_size: float            # Base size at z=0.5 (reference distance)
    drone_type: DroneType       # Behavior type
    health: int = 1             # Hits required to destroy
    age: int = 0                # Frames since spawn

    # For tracking (used by Kalman filter later)
    track_id: int = -1          # Unique ID for tracking

    # Size scaling parameters
    MIN_SIZE_SCALE: float = 0.3   # Size multiplier at z=1.0 (far)
    MAX_SIZE_SCALE: float = 4.0   # Size multiplier at z=0.0 (close)

    @property
    def is_kamikaze(self) -> bool:
        """Check if this drone is a kamikaze type."""
        return self.drone_type == DroneType.KAMIKAZE

    @property
    def size(self) -> float:
        """Get apparent size based on depth (z).

        Closer drones (low z) appear larger.
        Uses inverse relationship: size scales as 1/z
        """
        # Clamp z to avoid division issues
        z_clamped = max(0.1, min(1.0, self.z))

        # Scale factor: closer = larger
        # At z=1.0: scale = MIN_SIZE_SCALE (0.3x)
        # At z=0.5: scale = 1.0x (reference)
        # At z=0.1: scale = MAX_SIZE_SCALE (4.0x)
        scale = 0.5 / z_clamped  # Inverse relationship
        scale = max(self.MIN_SIZE_SCALE, min(self.MAX_SIZE_SCALE, scale))

        return self.base_size * scale

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Get bounding box as (x1, y1, x2, y2) normalized."""
        apparent_size = self.size
        half_size = apparent_size / 2
        return (
            self.x - half_size,
            self.y - half_size,
            self.x + half_size,
            self.y + half_size,
        )

    @property
    def center(self) -> tuple[float, float]:
        """Get center position."""
        return (self.x, self.y)

    def distance_to_center(self) -> float:
        """Distance from drone to screen center (0.5, 0.5)."""
        dx = self.x - 0.5
        dy = self.y - 0.5
        return math.sqrt(dx * dx + dy * dy)

    def distance_3d_to_camera(self) -> float:
        """3D distance considering depth (for kamikaze impact)."""
        dx = self.x - 0.5
        dy = self.y - 0.5
        # z represents distance to camera
        return math.sqrt(dx * dx + dy * dy + self.z * self.z)

    def is_on_screen(self, margin: float = 0.1) -> bool:
        """Check if drone is within screen bounds (with margin)."""
        return (
            -margin <= self.x <= 1 + margin and
            -margin <= self.y <= 1 + margin and
            self.z > 0  # Not past the camera
        )

    def update(self) -> None:
        """Update drone position based on velocity."""
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        self.age += 1

        # Erratic drones change direction periodically
        if self.drone_type == DroneType.ERRATIC and self.age % 15 == 0:
            self.vx += random.uniform(-0.005, 0.005)
            self.vy += random.uniform(-0.005, 0.005)
            self.vz += random.uniform(-0.001, 0.001)


@dataclass
class GameState:
    """Complete state of the Drone Hunter game."""

    # Drones currently in play
    drones: List[Drone] = field(default_factory=list)

    # Ammo system
    ammo: int = 10
    clip_size: int = 10
    is_reloading: bool = False
    reload_timer: int = 0
    reload_time: int = 30  # Frames to reload (1 second at 30 FPS)

    # Scoring
    score: int = 0
    hits: int = 0
    misses: int = 0

    # Timing
    frame_count: int = 0
    max_frames: int = 1000  # Episode length

    # Spawn control
    base_spawn_rate: float = 0.05  # Base probability per frame
    spawn_rate_increase: float = 0.00005  # Increase per frame

    # Drone type probabilities
    normal_prob: float = 0.70
    kamikaze_prob: float = 0.20
    erratic_prob: float = 0.10

    # Track ID counter
    _next_track_id: int = 0

    # Game over flag
    game_over: bool = False
    game_over_reason: str = ""

    @property
    def current_spawn_rate(self) -> float:
        """Get current spawn rate (increases over time)."""
        return self.base_spawn_rate + self.spawn_rate_increase * self.frame_count

    @property
    def ammo_fraction(self) -> float:
        """Ammo as fraction of clip size."""
        return self.ammo / self.clip_size

    @property
    def reload_fraction(self) -> float:
        """Reload progress as fraction (0 if not reloading)."""
        if not self.is_reloading:
            return 0.0
        return self.reload_timer / self.reload_time

    @property
    def frame_fraction(self) -> float:
        """Episode progress as fraction."""
        return self.frame_count / self.max_frames

    @property
    def threat_level(self) -> float:
        """Threat level based on approaching drone proximity (0-1).

        Uses depth (z) and velocity (vz) - approaching drones (vz < 0) that are
        closer to camera = higher threat. Agent infers threat from observable
        behavior, not ground truth labels.
        """
        if not self.drones:
            return 0.0

        max_threat = 0.0
        for drone in self.drones:
            if drone.vz < 0:  # Approaching camera
                # Threat based on depth (z): lower z = closer = higher threat
                # z=1.0 (far) -> threat=0, z=0.1 (impact) -> threat=1.0
                threat = 1.0 - drone.z
                max_threat = max(max_threat, threat)

        return max(0.0, min(1.0, max_threat))

    @property
    def has_approaching_threat(self) -> bool:
        """Check if there's an approaching drone (potential threat)."""
        return any(d.vz < 0 for d in self.drones)

    @property
    def has_kamikaze_threat(self) -> bool:
        """Check if there's an active kamikaze drone (internal use only)."""
        return any(d.is_kamikaze for d in self.drones)

    def spawn_drone(self) -> Drone | None:
        """Attempt to spawn a new drone based on current spawn rate."""
        if random.random() > self.current_spawn_rate:
            return None

        # Determine drone type
        roll = random.random()
        if roll < self.normal_prob:
            drone_type = DroneType.NORMAL
        elif roll < self.normal_prob + self.kamikaze_prob:
            drone_type = DroneType.KAMIKAZE
        else:
            drone_type = DroneType.ERRATIC

        # Spawn position (from edges)
        edge = random.randint(0, 3)  # 0=top, 1=right, 2=bottom, 3=left

        if edge == 0:  # Top
            x = random.uniform(0.1, 0.9)
            y = -0.05
        elif edge == 1:  # Right
            x = 1.05
            y = random.uniform(0.1, 0.9)
        elif edge == 2:  # Bottom
            x = random.uniform(0.1, 0.9)
            y = 1.05
        else:  # Left
            x = -0.05
            y = random.uniform(0.1, 0.9)

        # Initial depth (z) - all drones start far away
        z = random.uniform(0.8, 1.0)

        # Velocity based on type
        speed = random.uniform(0.005, 0.015)

        if drone_type == DroneType.KAMIKAZE:
            # Kamikaze must reach (0.5, 0.5, 0) - center of screen at camera
            # Calculate velocities so drone arrives at all three coordinates together
            dx = 0.5 - x
            dy = 0.5 - y
            dz = 0 - z  # Target z=0 (at camera)

            # 3D distance to target
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Speed determines how many frames to reach target
            # Higher speed = fewer frames = more urgent threat
            speed_3d = random.uniform(0.012, 0.025)

            # Set velocities proportionally so all dimensions reach target together
            vx = (dx / dist_3d) * speed_3d
            vy = (dy / dist_3d) * speed_3d
            vz = (dz / dist_3d) * speed_3d  # Will be negative (approaching)
        else:
            # Random direction (biased toward center)
            target_x = random.uniform(0.2, 0.8)
            target_y = random.uniform(0.2, 0.8)
            dx = target_x - x
            dy = target_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            vx = (dx / dist) * speed
            vy = (dy / dist) * speed

            # Normal/erratic drones: random depth movement
            # Some approach, some recede, some stay same depth
            vz = random.uniform(-0.005, 0.003)  # Slight bias toward approaching

        # Base size (actual size will be computed from z)
        base_size = random.uniform(0.04, 0.08)

        drone = Drone(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            base_size=base_size,
            drone_type=drone_type,
            track_id=self._next_track_id,
        )
        self._next_track_id += 1

        self.drones.append(drone)
        return drone

    def update_drones(self) -> List[Drone]:
        """Update all drones and remove off-screen ones.

        Returns:
            List of drones that went off-screen (for scoring).
        """
        off_screen = []
        remaining = []

        for drone in self.drones:
            drone.update()

            # Check for kamikaze impact (reached camera)
            if drone.is_kamikaze:
                # Impact when drone reaches camera (z near 0)
                # Since trajectory is synchronized, z=0 means it's at center
                if drone.z <= 0.08:
                    self.game_over = True
                    self.game_over_reason = "Kamikaze impact!"
                    remaining.append(drone)  # Keep for visualization
                    continue

            # Check if off-screen (includes z > 0 check)
            if not drone.is_on_screen():
                off_screen.append(drone)
            else:
                remaining.append(drone)

        self.drones = remaining
        return off_screen

    def update_reload(self) -> None:
        """Update reload timer if reloading."""
        if self.is_reloading:
            self.reload_timer += 1
            if self.reload_timer >= self.reload_time:
                self.ammo = self.clip_size
                self.is_reloading = False
                self.reload_timer = 0

    def fire(self, grid_x: int, grid_y: int, grid_size: int) -> tuple[bool, Drone | None]:
        """Attempt to fire at a grid cell.

        Args:
            grid_x: Grid column (0 to grid_size-1)
            grid_y: Grid row (0 to grid_size-1)
            grid_size: Size of the firing grid

        Returns:
            Tuple of (hit_something, drone_hit_or_none)
        """
        # Can't fire while reloading or empty
        if self.is_reloading or self.ammo <= 0:
            return False, None

        # Consume ammo
        self.ammo -= 1

        # Start reload if empty
        if self.ammo <= 0:
            self.is_reloading = True
            self.reload_timer = 0

        # Calculate grid cell bounds
        cell_width = 1.0 / grid_size
        cell_height = 1.0 / grid_size

        cell_x1 = grid_x * cell_width
        cell_y1 = grid_y * cell_height
        cell_x2 = cell_x1 + cell_width
        cell_y2 = cell_y1 + cell_height

        # Check for hits (bounding box overlap with cell)
        # Larger (closer) drones have bigger hitboxes = easier to hit
        for drone in self.drones:
            drone_x1, drone_y1, drone_x2, drone_y2 = drone.bbox

            # Check if drone bbox overlaps with cell
            # Two rectangles overlap if they overlap on both axes
            x_overlap = cell_x1 < drone_x2 and cell_x2 > drone_x1
            y_overlap = cell_y1 < drone_y2 and cell_y2 > drone_y1

            if x_overlap and y_overlap:
                # Hit!
                drone.health -= 1
                if drone.health <= 0:
                    self.drones.remove(drone)
                    self.score += 2 if drone.is_kamikaze else 1
                    self.hits += 1
                    return True, drone

        # Miss
        self.misses += 1
        return False, None

    def step(self) -> None:
        """Advance game state by one frame."""
        self.frame_count += 1
        self.update_reload()
        self.update_drones()
        self.spawn_drone()

        # Check for episode end
        if self.frame_count >= self.max_frames:
            self.game_over = True
            self.game_over_reason = "Episode complete!"

    def reset(self) -> None:
        """Reset game state for new episode."""
        self.drones.clear()
        self.ammo = self.clip_size
        self.is_reloading = False
        self.reload_timer = 0
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.frame_count = 0
        self._next_track_id = 0
        self.game_over = False
        self.game_over_reason = ""

    def get_oracle_observation(self, max_drones: int = 10, grid_size: int = 8) -> dict:
        """Get ground-truth observation for oracle mode training.

        Returns observation dict with perfect drone positions AND grid cell info.
        Each drone: (x, y, z, vz, urgency, grid_x_onehot[8], grid_y_onehot[8])

        Note: is_kamikaze removed - agent infers threat from vz (approaching = negative)

        Key insight: Include grid cell as one-hot so agent doesn't need to learn
        the coordinate-to-action mapping from scratch.
        """
        import numpy as np

        # Features per drone: 5 scalar + 8 grid_x one-hot + 8 grid_y one-hot = 21
        features_per_drone = 5 + grid_size * 2
        drone_obs = np.zeros((max_drones, features_per_drone), dtype=np.float32)

        for i, drone in enumerate(self.drones[:max_drones]):
            # Compute grid cell
            grid_x = min(grid_size - 1, max(0, int(drone.x * grid_size)))
            grid_y = min(grid_size - 1, max(0, int(drone.y * grid_size)))

            # Urgency: how soon will this drone impact? (for approaching drones)
            # Higher urgency = closer to impact (lower z, negative vz)
            # Agent infers threat from vz, not ground truth labels
            if drone.vz < 0:  # Approaching
                frames_to_impact = drone.z / max(0.001, abs(drone.vz))
                urgency = 1.0 / (1.0 + frames_to_impact / 50.0)  # Sigmoid-ish
            else:
                urgency = 0.1  # Low urgency for non-approaching

            # Scalar features (no is_kamikaze - agent learns from vz + rewards)
            drone_obs[i, 0] = drone.x
            drone_obs[i, 1] = drone.y
            drone_obs[i, 2] = drone.z
            drone_obs[i, 3] = drone.vz * 10  # Scale for gradient
            drone_obs[i, 4] = urgency

            # One-hot grid_x (positions 5-12)
            drone_obs[i, 5 + grid_x] = 1.0

            # One-hot grid_y (positions 13-20)
            drone_obs[i, 5 + grid_size + grid_y] = 1.0

        # Game state: (ammo_frac, reload_frac, frame_frac, threat_level, n_drones_normalized)
        game_state_obs = np.array([
            self.ammo_fraction,
            self.reload_fraction,
            self.frame_fraction,
            self.threat_level,
            len(self.drones) / max_drones,  # How crowded is the screen?
        ], dtype=np.float32)

        return {
            "detections": drone_obs,
            "game_state": game_state_obs,
        }

    def get_single_target_observation(self, grid_size: int = 8) -> dict:
        """Get simplified observation with only the most urgent target.

        This fixes the multi-drone selection problem by pre-computing which
        drone the agent should shoot. The agent just needs to learn:
        grid_one_hot → action (trivially simple).

        Returns:
            Dict with:
            - "target": (19,) array - single drone features (or zeros if no drones)
            - "game_state": (5,) array - ammo, reload, frame, threat, has_target
        """
        import numpy as np

        def compute_urgency(drone) -> float:
            """Compute urgency from observable features (no ground truth)."""
            if drone.vz < 0:  # Approaching
                frames_to_impact = drone.z / max(0.001, abs(drone.vz))
                return 1.0 / (1.0 + frames_to_impact / 50.0)
            else:  # Moving away or stationary
                return 0.1

        # Features: z, vz, urgency, grid_x_onehot[8], grid_y_onehot[8]
        # Note: is_kamikaze removed - agent infers threat from vz
        features_per_target = 3 + grid_size * 2  # 19 for grid_size=8
        target_obs = np.zeros(features_per_target, dtype=np.float32)

        has_target = 0.0

        if self.drones:
            # Sort drones by computed urgency (highest first)
            sorted_drones = sorted(self.drones, key=lambda d: -compute_urgency(d))

            # Get the #1 most urgent target
            drone = sorted_drones[0]
            has_target = 1.0

            # Compute grid cell
            grid_x = min(grid_size - 1, max(0, int(drone.x * grid_size)))
            grid_y = min(grid_size - 1, max(0, int(drone.y * grid_size)))

            # Compute urgency from observable features
            urgency = compute_urgency(drone)

            # Fill features (no is_kamikaze - agent learns from vz + rewards)
            target_obs[0] = drone.z
            target_obs[1] = drone.vz * 10  # Scale for gradient
            target_obs[2] = urgency

            # One-hot grid_x (positions 3-10)
            target_obs[3 + grid_x] = 1.0

            # One-hot grid_y (positions 11-18)
            target_obs[3 + grid_size + grid_y] = 1.0

        # Game state
        game_state_obs = np.array([
            self.ammo_fraction,
            self.reload_fraction,
            self.frame_fraction,
            self.threat_level,
            has_target,  # 1.0 if there's a drone to shoot, 0.0 otherwise
        ], dtype=np.float32)

        return {
            "target": target_obs,
            "game_state": game_state_obs,
        }
