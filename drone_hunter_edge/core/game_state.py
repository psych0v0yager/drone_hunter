"""Simplified game state for Drone Hunter edge inference.

Stripped down version without distractors/obstacles for MVP.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional
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
    Depth (z) simulates 3D perspective: z=1.0 far, z=0.0 at camera.
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
    track_id: int = -1          # Unique ID for tracking

    MIN_SIZE_SCALE: float = 0.3
    MAX_SIZE_SCALE: float = 4.0

    @property
    def is_kamikaze(self) -> bool:
        """Check if this drone is a kamikaze type."""
        return self.drone_type == DroneType.KAMIKAZE

    @property
    def size(self) -> float:
        """Get apparent size based on depth (z)."""
        z_clamped = max(0.1, min(1.0, self.z))
        scale = 0.5 / z_clamped
        scale = max(self.MIN_SIZE_SCALE, min(self.MAX_SIZE_SCALE, scale))
        return self.base_size * scale

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
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
    def center(self) -> Tuple[float, float]:
        """Get center position."""
        return (self.x, self.y)

    def is_on_screen(self, margin: float = 0.1) -> bool:
        """Check if drone is within screen bounds (with margin)."""
        return (
            -margin <= self.x <= 1 + margin and
            -margin <= self.y <= 1 + margin and
            self.z > 0
        )

    def update(self) -> None:
        """Update drone position based on velocity."""
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        self.age += 1

        if self.drone_type == DroneType.ERRATIC and self.age % 15 == 0:
            self.vx += random.uniform(-0.005, 0.005)
            self.vy += random.uniform(-0.005, 0.005)
            self.vz += random.uniform(-0.001, 0.001)


@dataclass
class GameState:
    """Complete state of the Drone Hunter game (simplified for edge)."""

    drones: List[Drone] = field(default_factory=list)

    # Ammo system
    ammo: int = 10
    clip_size: int = 10
    is_reloading: bool = False
    reload_timer: int = 0
    reload_time: int = 30

    # Scoring
    score: int = 0
    hits: int = 0
    misses: int = 0

    # Timing
    frame_count: int = 0
    max_frames: int = 1000

    # Spawn control
    base_spawn_rate: float = 0.05
    spawn_rate_increase: float = 0.00005

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
        """Threat level based on approaching drone proximity (0-1)."""
        if not self.drones:
            return 0.0

        max_threat = 0.0
        for drone in self.drones:
            if drone.vz < 0:
                threat = 1.0 - drone.z
                max_threat = max(max_threat, threat)

        return max(0.0, min(1.0, max_threat))

    @property
    def has_approaching_threat(self) -> bool:
        """Check if there's an approaching drone."""
        return any(d.vz < 0 for d in self.drones)

    @property
    def has_kamikaze_threat(self) -> bool:
        """Check if there's an active kamikaze drone."""
        return any(d.is_kamikaze for d in self.drones)

    def spawn_drone(self) -> Optional[Drone]:
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
        edge = random.randint(0, 3)

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

        z = random.uniform(0.8, 1.0)
        speed = random.uniform(0.005, 0.015)

        if drone_type == DroneType.KAMIKAZE:
            dx = 0.5 - x
            dy = 0.5 - y
            dz = 0 - z
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            speed_3d = random.uniform(0.012, 0.025)
            vx = (dx / dist_3d) * speed_3d
            vy = (dy / dist_3d) * speed_3d
            vz = (dz / dist_3d) * speed_3d
        else:
            target_x = random.uniform(0.2, 0.8)
            target_y = random.uniform(0.2, 0.8)
            dx = target_x - x
            dy = target_y - y
            dist = math.sqrt(dx * dx + dy * dy)
            vx = (dx / dist) * speed
            vy = (dy / dist) * speed
            vz = random.uniform(-0.005, 0.003)

        base_size = random.uniform(0.04, 0.08)

        drone = Drone(
            x=x, y=y, z=z,
            vx=vx, vy=vy, vz=vz,
            base_size=base_size,
            drone_type=drone_type,
            track_id=self._next_track_id,
        )
        self._next_track_id += 1
        self.drones.append(drone)
        return drone

    def update_drones(self) -> List[Drone]:
        """Update all drones and remove off-screen ones."""
        off_screen = []
        remaining = []

        for drone in self.drones:
            drone.update()

            if drone.is_kamikaze and drone.z <= 0.08:
                self.game_over = True
                self.game_over_reason = "Kamikaze impact!"
                remaining.append(drone)
                continue

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

    def fire(self, grid_x: int, grid_y: int, grid_size: int) -> Tuple[bool, Optional[Drone]]:
        """Attempt to fire at a grid cell."""
        if self.is_reloading or self.ammo <= 0:
            return False, None

        self.ammo -= 1
        if self.ammo <= 0:
            self.is_reloading = True
            self.reload_timer = 0

        cell_width = 1.0 / grid_size
        cell_height = 1.0 / grid_size
        cell_x1 = grid_x * cell_width
        cell_y1 = grid_y * cell_height
        cell_x2 = cell_x1 + cell_width
        cell_y2 = cell_y1 + cell_height

        for drone in self.drones:
            drone_x1, drone_y1, drone_x2, drone_y2 = drone.bbox
            x_overlap = cell_x1 < drone_x2 and cell_x2 > drone_x1
            y_overlap = cell_y1 < drone_y2 and cell_y2 > drone_y1

            if x_overlap and y_overlap:
                drone.health -= 1
                if drone.health <= 0:
                    self.drones.remove(drone)
                    self.score += 2 if drone.is_kamikaze else 1
                    self.hits += 1
                    return True, drone

        self.misses += 1
        return False, None

    def step(self) -> None:
        """Advance game state by one frame."""
        self.frame_count += 1
        self.update_reload()
        self.update_drones()
        self.spawn_drone()

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
