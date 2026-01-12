#!/usr/bin/env python3
"""Generate 40x40 ROI training data for tiny detector.

Creates .npz files with cropped regions around drone centers for training
the lightweight ROI detector used in Tier 1 of the adaptive scheduler.

Usage:
    python scripts/generate_tiny_dataset.py --difficulty medium --num-samples 15000
    python scripts/generate_tiny_dataset.py --difficulty hard --num-samples 5000 --append
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from drone_hunter.envs.difficulty import DifficultyConfig
from drone_hunter.envs.game_state import Drone, DroneType, GameState
from drone_hunter.envs.renderer import SpriteRenderer


def create_random_drone(
    x_range: Tuple[float, float] = (0.1, 0.9),
    y_range: Tuple[float, float] = (0.1, 0.9),
    z_range: Tuple[float, float] = (0.1, 0.9),  # Include close drones (up to ~100px)
) -> Drone:
    """Create a drone with random position."""
    drone_type = random.choice([DroneType.NORMAL, DroneType.KAMIKAZE, DroneType.ERRATIC])
    return Drone(
        x=random.uniform(*x_range),
        y=random.uniform(*y_range),
        z=random.uniform(*z_range),
        vx=random.uniform(-0.02, 0.02),
        vy=random.uniform(-0.02, 0.02),
        vz=random.uniform(-0.005, 0.005),
        base_size=0.06,
        drone_type=drone_type,
    )


def crop_roi(
    frame: np.ndarray,
    cx: float,
    cy: float,
    roi_size: int,
    jitter_px: int = 5,
) -> Tuple[np.ndarray, float, float]:
    """Crop ROI around center with optional jitter.

    Args:
        frame: Full frame (H, W, 3) uint8 RGB.
        cx: Center x (normalized 0-1).
        cy: Center y (normalized 0-1).
        roi_size: Size of square ROI.
        jitter_px: Random jitter in pixels (simulates Kalman prediction error).

    Returns:
        Tuple of (roi, new_cx, new_cy) where new_cx/new_cy are the
        drone center relative to the ROI (0-1 normalized).
    """
    h, w = frame.shape[:2]

    # Convert to pixels
    cx_px = int(cx * w)
    cy_px = int(cy * h)

    # Add jitter (simulates Kalman prediction error)
    jx = random.randint(-jitter_px, jitter_px)
    jy = random.randint(-jitter_px, jitter_px)
    crop_cx = cx_px + jx
    crop_cy = cy_px + jy

    # Compute ROI bounds
    half = roi_size // 2
    x1 = crop_cx - half
    y1 = crop_cy - half
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # Handle edge cases
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Crop
    roi = frame[y1:y2, x1:x2].copy()

    # Pad if necessary
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
        padded[pad_top:roi_size-pad_bottom, pad_left:roi_size-pad_right] = roi
        roi = padded

    # Compute drone center relative to ROI
    # The original drone center (before jitter) relative to crop center
    new_cx = 0.5 - jx / roi_size
    new_cy = 0.5 - jy / roi_size

    return roi, new_cx, new_cy


def generate_positive_sample(
    renderer: SpriteRenderer,
    roi_size: int,
    jitter_px: int = 0,  # Deprecated, kept for API compatibility
    disable_motion_blur: bool = True,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Generate a positive sample (ROI with drone at various positions).

    The drone can appear anywhere within the ROI (0.05-0.95), including edges
    and corners. This teaches the model to detect drones that aren't perfectly
    centered, handling occlusion cases where part of the drone is outside ROI.

    Args:
        renderer: SpriteRenderer instance.
        roi_size: Size of ROI crop.
        jitter_px: Deprecated, ignored. Position variance is now built-in.
        disable_motion_blur: If True, disable motion blur for crisp drones.

    Returns:
        Tuple of (roi, target, has_drone) where:
        - roi: (roi_size, roi_size, 3) uint8 RGB
        - target: (4,) float32 [cx, cy, w, h] - drone center within ROI (0-1)
        - has_drone: True
    """
    # Disable motion blur for crisp drone silhouettes
    original_motion_blur = renderer.motion_blur
    if disable_motion_blur:
        renderer.motion_blur = False

    # Create game state with one drone
    game_state = GameState(max_frames=1)
    drone = create_random_drone()
    game_state.drones = [drone]

    # Render
    frame = renderer.render(game_state)
    frame_h, frame_w = frame.shape[:2]

    # Restore original motion blur setting
    renderer.motion_blur = original_motion_blur

    # Convert drone position to pixels
    drone_x_px = drone.x * frame_w
    drone_y_px = drone.y * frame_h

    # Pick target position for drone within ROI (0.05-0.95)
    # This gives full coverage including edges and corners
    target_cx = random.uniform(0.05, 0.95)
    target_cy = random.uniform(0.05, 0.95)

    # Calculate ROI crop center that places drone at target position
    # offset = how far to shift the crop from drone center
    # If target_cx=0.5, crop is centered on drone (offset=0)
    # If target_cx=0.2, drone is left of center, so crop right of drone (offset>0)
    offset_x_px = (0.5 - target_cx) * roi_size
    offset_y_px = (0.5 - target_cy) * roi_size

    crop_cx_px = int(drone_x_px + offset_x_px)
    crop_cy_px = int(drone_y_px + offset_y_px)

    # Compute ROI bounds
    half = roi_size // 2
    x1 = crop_cx_px - half
    y1 = crop_cy_px - half
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # Clamp ROI to frame bounds (no black padding - matches inference)
    if x1 < 0:
        x1, x2 = 0, roi_size
    if y1 < 0:
        y1, y2 = 0, roi_size
    if x2 > frame_w:
        x1, x2 = frame_w - roi_size, frame_w
    if y2 > frame_h:
        y1, y2 = frame_h - roi_size, frame_h

    # Recalculate target position based on clamped ROI
    target_cx = (drone_x_px - x1) / roi_size
    target_cy = (drone_y_px - y1) / roi_size

    # Crop (guaranteed to be full size, no padding needed)
    roi = frame[y1:y2, x1:x2].copy()

    # Compute drone size relative to ROI
    drone_size_px = drone.size * frame_w
    w = drone_size_px / roi_size
    h = drone_size_px / roi_size

    target = np.array([target_cx, target_cy, w, h], dtype=np.float32)

    return roi, target, True


def generate_negative_sample(
    renderer: SpriteRenderer,
    roi_size: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Generate a negative sample (ROI without drone).

    Returns:
        Tuple of (roi, target, has_drone) where:
        - roi: (roi_size, roi_size, 3) uint8 RGB
        - target: (4,) zeros (no drone)
        - has_drone: False
    """
    # Create game state with no drones
    game_state = GameState(max_frames=1)
    game_state.drones = []

    # Render
    frame = renderer.render(game_state)

    # Random crop location
    cx = random.uniform(0.1, 0.9)
    cy = random.uniform(0.1, 0.9)
    roi, _, _ = crop_roi(frame, cx, cy, roi_size, jitter_px=0)

    target = np.zeros(4, dtype=np.float32)

    return roi, target, False


def generate_hard_negative_sample(
    renderer: SpriteRenderer,
    roi_size: int,
    min_distance: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Generate a hard negative (ROI near but not on drone).

    This helps the model learn to distinguish between drone and background
    when the prediction is slightly off.
    """
    # Create game state with one drone
    game_state = GameState(max_frames=1)
    drone = create_random_drone()
    game_state.drones = [drone]

    # Render
    frame = renderer.render(game_state)

    # Random crop NOT on the drone (at least min_distance away)
    for _ in range(10):  # Try a few times
        cx = random.uniform(0.1, 0.9)
        cy = random.uniform(0.1, 0.9)
        dist = ((cx - drone.x) ** 2 + (cy - drone.y) ** 2) ** 0.5
        if dist > min_distance:
            break

    roi, _, _ = crop_roi(frame, cx, cy, roi_size, jitter_px=0)
    target = np.zeros(4, dtype=np.float32)

    return roi, target, False


def generate_multi_drone_sample(
    renderer: SpriteRenderer,
    roi_size: int,
    num_drones: int = 2,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Generate a positive sample with multiple drones in the ROI.

    This creates harder training examples where the model must detect
    drones even when multiple are present in the same region.

    Args:
        renderer: SpriteRenderer instance.
        roi_size: Size of ROI crop.
        num_drones: Number of drones to place (2-4).

    Returns:
        Tuple of (roi, target, has_drone) where target is for the primary drone.
    """
    # Disable motion blur for crisp drone silhouettes
    original_motion_blur = renderer.motion_blur
    renderer.motion_blur = False

    # Create game state with multiple drones
    game_state = GameState(max_frames=1)
    drones = [create_random_drone() for _ in range(num_drones)]
    game_state.drones = drones

    # Render
    frame = renderer.render(game_state)
    frame_h, frame_w = frame.shape[:2]

    renderer.motion_blur = original_motion_blur

    # Pick a random drone as the "primary" one we're tracking
    primary = random.choice(drones)
    drone_x_px = primary.x * frame_w
    drone_y_px = primary.y * frame_h

    # Pick target position for primary drone within ROI
    target_cx = random.uniform(0.15, 0.85)
    target_cy = random.uniform(0.15, 0.85)

    # Calculate ROI crop position
    offset_x_px = (0.5 - target_cx) * roi_size
    offset_y_px = (0.5 - target_cy) * roi_size

    crop_cx_px = int(drone_x_px + offset_x_px)
    crop_cy_px = int(drone_y_px + offset_y_px)

    half = roi_size // 2
    x1 = crop_cx_px - half
    y1 = crop_cy_px - half
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # Clamp to frame bounds
    if x1 < 0:
        x1, x2 = 0, roi_size
    if y1 < 0:
        y1, y2 = 0, roi_size
    if x2 > frame_w:
        x1, x2 = frame_w - roi_size, frame_w
    if y2 > frame_h:
        y1, y2 = frame_h - roi_size, frame_h

    # Recalculate target position
    target_cx = (drone_x_px - x1) / roi_size
    target_cy = (drone_y_px - y1) / roi_size

    roi = frame[y1:y2, x1:x2].copy()

    # Primary drone size
    drone_size_px = primary.size * frame_w
    w = drone_size_px / roi_size
    h = drone_size_px / roi_size

    target = np.array([target_cx, target_cy, w, h], dtype=np.float32)

    return roi, target, True


def generate_cloud_edge_negative_sample(
    renderer: SpriteRenderer,
    roi_size: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Generate hard negative from cloud transition regions.

    Samples ROIs from soft gradient zones between clouds and sky,
    which are the main source of false positives. These areas have:
    - Soft blurry transitions (not hard edges)
    - Intermediate grey values in the gradient
    - Similar appearance to blurry drone silhouettes

    Returns:
        Tuple of (roi, target, has_drone) where:
        - roi: (roi_size, roi_size, 3) uint8 RGB
        - target: (4,) zeros (no drone)
        - has_drone: False
    """
    # Create game state with no drones
    game_state = GameState(max_frames=1)
    game_state.drones = []

    # Render frame with clouds
    frame = renderer.render(game_state)
    h, w = frame.shape[:2]

    # Convert to grayscale
    gray = np.mean(frame, axis=2)

    # Find areas with local gradient (soft transitions)
    # Use local standard deviation to find gradient regions
    from scipy.ndimage import uniform_filter

    # Compute local mean and variance
    window_size = 15
    local_mean = uniform_filter(gray, size=window_size)
    local_sq_mean = uniform_filter(gray**2, size=window_size)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))

    # Target regions with:
    # 1. Some local variance (gradient regions) - std between 2-20
    # 2. DARKER areas that could be confused for drones (mean 120-200)
    #    Drones are dark grey/black (~50-150), so negatives should be similar darkness
    # 3. Away from frame edges (avoid padding artifacts)
    margin = roi_size // 2 + 5

    gradient_mask = (
        (local_std > 2) & (local_std < 20) &  # Soft gradient regions
        (local_mean > 120) & (local_mean < 200) &  # Darker grey similar to drone brightness
        (np.arange(h)[:, None] > margin) & (np.arange(h)[:, None] < h - margin) &
        (np.arange(w)[None, :] > margin) & (np.arange(w)[None, :] < w - margin)
    )

    # Find candidate positions
    gradient_positions = np.argwhere(gradient_mask)

    if len(gradient_positions) > 0:
        # Sample from gradient region
        idx = random.randint(0, len(gradient_positions) - 1)
        cy_px, cx_px = gradient_positions[idx]
        cx = cx_px / w
        cy = cy_px / h
    else:
        # Fallback: sample from upper half where clouds usually are
        # but avoid edges
        cx = random.uniform(0.15, 0.85)
        cy = random.uniform(0.15, 0.5)

    roi, _, _ = crop_roi(frame, cx, cy, roi_size, jitter_px=0)
    target = np.zeros(4, dtype=np.float32)

    return roi, target, False


def generate_dataset(
    output_dir: str,
    difficulty: str = "medium",
    num_samples: int = 15000,
    positive_ratio: float = 0.5,  # 50/50 split for balanced training
    val_split: float = 0.1,
    roi_size: int = 40,
    jitter_px: int = 0,
    cloud_edge_negatives: bool = True,
    multi_drone_ratio: float = 0.2,  # 20% of positives have multiple drones
    append: bool = False,
    seed: int = 42,
):
    """Generate training dataset.

    Args:
        output_dir: Output directory for .npz files.
        difficulty: Difficulty preset (easy, medium, hard, forest, urban).
        num_samples: Total number of samples to generate.
        positive_ratio: Fraction of positive samples (default 0.5 for balanced).
        val_split: Fraction for validation set.
        roi_size: Size of square ROI crops.
        jitter_px: Jitter for positive samples (default 0, no jitter).
        cloud_edge_negatives: Use cloud-edge sampling for hard negatives (default True).
        multi_drone_ratio: Fraction of positives with multiple drones (default 0.2).
        append: Append to existing dataset.
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create renderer with difficulty
    difficulty_config = DifficultyConfig.from_name(difficulty)
    renderer = SpriteRenderer(width=320, height=320, difficulty_config=difficulty_config)

    # Calculate sample counts
    num_positive = int(num_samples * positive_ratio)
    num_negative = num_samples - num_positive
    num_single_drone = int(num_positive * (1 - multi_drone_ratio))
    num_multi_drone = num_positive - num_single_drone
    num_hard_negative = num_negative // 2
    num_easy_negative = num_negative - num_hard_negative

    hard_neg_type = "cloud-edge" if cloud_edge_negatives else "near-drone"
    print(f"Generating dataset:")
    print(f"  Difficulty: {difficulty}")
    print(f"  Total samples: {num_samples}")
    print(f"  Positive: {num_positive} ({num_single_drone} single, {num_multi_drone} multi-drone)")
    print(f"  Hard negative: {num_hard_negative} ({hard_neg_type})")
    print(f"  Easy negative: {num_easy_negative}")
    print(f"  ROI size: {roi_size}x{roi_size}")

    # Generate samples
    images = []
    targets = []
    has_drone = []

    # Single-drone positive samples
    print("\nGenerating single-drone positive samples...")
    for _ in tqdm(range(num_single_drone), desc="Single drone"):
        renderer.reset_background()
        roi, target, label = generate_positive_sample(renderer, roi_size, jitter_px)
        images.append(roi)
        targets.append(target)
        has_drone.append(label)

    # Multi-drone positive samples
    print("Generating multi-drone positive samples...")
    for _ in tqdm(range(num_multi_drone), desc="Multi drone"):
        renderer.reset_background()
        num_drones = random.choice([2, 2, 2, 3, 3, 4])  # Weighted toward 2-3
        roi, target, label = generate_multi_drone_sample(renderer, roi_size, num_drones)
        images.append(roi)
        targets.append(target)
        has_drone.append(label)

    # Hard negative samples (cloud-edge or near-drone based on setting)
    if cloud_edge_negatives:
        print("Generating cloud-edge hard negative samples...")
        for _ in tqdm(range(num_hard_negative), desc="Cloud edge neg"):
            renderer.reset_background()
            roi, target, label = generate_cloud_edge_negative_sample(renderer, roi_size)
            images.append(roi)
            targets.append(target)
            has_drone.append(label)
    else:
        print("Generating near-drone hard negative samples...")
        for _ in tqdm(range(num_hard_negative), desc="Hard neg"):
            renderer.reset_background()
            roi, target, label = generate_hard_negative_sample(renderer, roi_size)
            images.append(roi)
            targets.append(target)
            has_drone.append(label)

    # Easy negative samples (no drones)
    print("Generating easy negative samples...")
    for _ in tqdm(range(num_easy_negative), desc="Easy neg"):
        renderer.reset_background()
        roi, target, label = generate_negative_sample(renderer, roi_size)
        images.append(roi)
        targets.append(target)
        has_drone.append(label)

    # Convert to arrays
    images = np.array(images, dtype=np.uint8)
    targets = np.array(targets, dtype=np.float32)
    has_drone = np.array(has_drone, dtype=bool)

    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    targets = targets[indices]
    has_drone = has_drone[indices]

    # Split into train/val
    split_idx = int(len(images) * (1 - val_split))
    train_images = images[:split_idx]
    train_targets = targets[:split_idx]
    train_has_drone = has_drone[:split_idx]
    val_images = images[split_idx:]
    val_targets = targets[split_idx:]
    val_has_drone = has_drone[split_idx:]

    # Load existing data if appending
    if append:
        train_path = output_dir / "train.npz"
        val_path = output_dir / "val.npz"
        if train_path.exists():
            print(f"\nAppending to existing dataset...")
            existing = np.load(train_path)
            train_images = np.concatenate([existing["images"], train_images])
            train_targets = np.concatenate([existing["targets"], train_targets])
            train_has_drone = np.concatenate([existing["has_drone"], train_has_drone])
        if val_path.exists():
            existing = np.load(val_path)
            val_images = np.concatenate([existing["images"], val_images])
            val_targets = np.concatenate([existing["targets"], val_targets])
            val_has_drone = np.concatenate([existing["has_drone"], val_has_drone])

    # Save
    print(f"\nSaving to {output_dir}...")
    np.savez_compressed(
        output_dir / "train.npz",
        images=train_images,
        targets=train_targets,
        has_drone=train_has_drone,
    )
    np.savez_compressed(
        output_dir / "val.npz",
        images=val_images,
        targets=val_targets,
        has_drone=val_has_drone,
    )

    print(f"Train: {len(train_images)} samples ({train_has_drone.sum()} positive)")
    print(f"Val: {len(val_images)} samples ({val_has_drone.sum()} positive)")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Generate tiny detector training data")
    parser.add_argument("--output", type=str, default="data/tiny_drone",
                        help="Output directory")
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard", "forest", "urban"],
                        help="Difficulty preset")
    parser.add_argument("--num-samples", type=int, default=15000,
                        help="Total number of samples")
    parser.add_argument("--positive-ratio", type=float, default=0.5,
                        help="Fraction of positive samples (default 0.5 for balanced)")
    parser.add_argument("--multi-drone-ratio", type=float, default=0.2,
                        help="Fraction of positives with multiple drones (default 0.2)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction for validation")
    parser.add_argument("--roi-size", type=int, default=40,
                        help="ROI size (square)")
    parser.add_argument("--jitter", type=int, default=0,
                        help="Position jitter in pixels (default 0, no jitter)")
    parser.add_argument("--cloud-edge", action="store_true", default=True,
                        help="Use cloud-edge sampling for hard negatives (default: True)")
    parser.add_argument("--no-cloud-edge", action="store_false", dest="cloud_edge",
                        help="Use near-drone sampling for hard negatives instead of cloud-edge")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        difficulty=args.difficulty,
        num_samples=args.num_samples,
        positive_ratio=args.positive_ratio,
        multi_drone_ratio=args.multi_drone_ratio,
        val_split=args.val_split,
        roi_size=args.roi_size,
        jitter_px=args.jitter,
        cloud_edge_negatives=args.cloud_edge,
        append=args.append,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
