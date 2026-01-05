"""Difficulty configuration for Drone Hunter environment.

Visual difficulty settings that only affect the detector, not the agent.
Core mechanics (drones, scoring, ammo) remain unchanged regardless of difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DifficultyConfig:
    """Visual difficulty settings for detector training.

    Core gameplay mechanics (drones, scoring, ammo) are unchanged.
    These settings only affect visual complexity for training a more robust detector.

    Attributes:
        scene_type: Background scene - sky, forest, urban, or random.
        noise_level: Gaussian noise intensity (0.0-0.3).
        hue_shift_range: Random hue shift range (0.0-0.15).
        jpeg_quality: JPEG compression quality (50-100).
        lighting_mode: Lighting preset for sky generation.
        distractors_enabled: Enable flying distractors (birds, debris, etc.).
        distractor_spawn_rate: Probability of spawning distractor per frame.
        distractor_types: List of distractor types to spawn.
        static_obstacles_enabled: Enable static obstacles (trees, powerlines).
        static_obstacle_types: List of static obstacle types.
        rain_intensity: Rain effect intensity (0.0-1.0).
        fog_density: Fog effect density (0.0-0.5).
    """

    # Scene type
    scene_type: str = "sky"  # sky | forest | urban | random

    # Augmentation
    noise_level: float = 0.0  # 0.0-0.3 sensor noise
    hue_shift_range: float = 0.0  # 0.0-0.15 color shift
    jpeg_quality: int = 100  # 50-100 compression quality

    # Lighting
    lighting_mode: str = "day"  # day | golden_hour | overcast | dusk | night | thermal | night_vision | random

    # Flying distractors (cause detector false positives)
    distractors_enabled: bool = False
    distractor_spawn_rate: float = 0.02
    distractor_types: list[str] = field(
        default_factory=lambda: ["bird", "debris", "balloon", "plane"]
    )

    # Static obstacles (occlusion, visual noise)
    static_obstacles_enabled: bool = False
    static_obstacle_types: list[str] = field(
        default_factory=lambda: ["tree", "powerline"]
    )

    # Weather effects
    rain_intensity: float = 0.0  # 0.0-1.0
    fog_density: float = 0.0  # 0.0-0.5

    @classmethod
    def easy(cls) -> DifficultyConfig:
        """Current behavior - clear sky, no distractors.

        Identical to the original game. Use this for baseline training.
        """
        return cls()

    @classmethod
    def medium(cls) -> DifficultyConfig:
        """Mixed lighting, some distractors.

        Adds visual variety without extreme conditions.
        Good for initial detector robustness training.
        """
        return cls(
            lighting_mode="random",
            noise_level=0.05,
            distractors_enabled=True,
            distractor_types=["bird", "debris"],
        )

    @classmethod
    def hard(cls) -> DifficultyConfig:
        """Full complexity - all distractors, obstacles, weather.

        Maximum visual challenge for detector training.
        Includes all scene types, obstacles, and weather effects.
        """
        return cls(
            scene_type="random",
            lighting_mode="random",
            noise_level=0.1,
            hue_shift_range=0.1,
            distractors_enabled=True,
            distractor_types=["bird", "debris", "balloon", "plane"],
            static_obstacles_enabled=True,
            static_obstacle_types=["tree", "powerline"],
            rain_intensity=0.2,
            fog_density=0.15,
        )

    @classmethod
    def forest(cls) -> DifficultyConfig:
        """Forest/woodland scene with trees.

        Simulates detection in a forested environment.
        High visual complexity from treeline backgrounds.
        """
        return cls(
            scene_type="forest",
            lighting_mode="random",
            static_obstacles_enabled=True,
            static_obstacle_types=["tree"],
            distractors_enabled=True,
            distractor_types=["bird"],
        )

    @classmethod
    def urban(cls) -> DifficultyConfig:
        """Urban scene with powerlines and buildings.

        Simulates detection in an urban/suburban environment.
        Powerlines create confusing linear patterns.
        """
        return cls(
            scene_type="urban",
            lighting_mode="random",
            static_obstacles_enabled=True,
            static_obstacle_types=["powerline"],
            distractors_enabled=True,
            distractor_types=["bird", "plane"],
        )

    @classmethod
    def from_name(cls, name: str) -> DifficultyConfig:
        """Create a DifficultyConfig from a preset name.

        Args:
            name: Preset name (easy, medium, hard, forest, urban).

        Returns:
            DifficultyConfig instance.

        Raises:
            ValueError: If name is not a valid preset.
        """
        presets = {
            "easy": cls.easy,
            "medium": cls.medium,
            "hard": cls.hard,
            "forest": cls.forest,
            "urban": cls.urban,
        }
        if name not in presets:
            valid = ", ".join(presets.keys())
            raise ValueError(f"Unknown difficulty preset: {name}. Valid options: {valid}")
        return presets[name]()
