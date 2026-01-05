"""Generate synthetic training data for NanoDet drone detection.

Creates a COCO-format dataset with rendered drone images and ground truth
bounding box annotations from the simulation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from drone_hunter.envs.game_state import Drone, DroneType, GameState
from drone_hunter.envs.renderer import SpriteRenderer

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    from PIL import Image
    HAS_OPENCV = False


def create_random_drone(
    x_range: tuple[float, float] = (0.1, 0.9),
    y_range: tuple[float, float] = (0.1, 0.9),
    z_range: tuple[float, float] = (0.2, 0.9),
    velocity_range: float = 0.02,
) -> Drone:
    """Create a drone with random position and velocity.

    Args:
        x_range: Min/max horizontal position.
        y_range: Min/max vertical position.
        z_range: Min/max depth (affects apparent size).
        velocity_range: Max velocity magnitude.

    Returns:
        Drone with randomized properties.
    """
    # Random type (affects color in placeholder renderer)
    drone_type = random.choice([DroneType.NORMAL, DroneType.KAMIKAZE, DroneType.ERRATIC])

    # Random position
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    z = random.uniform(*z_range)

    # Random velocity (affects motion blur)
    vx = random.uniform(-velocity_range, velocity_range)
    vy = random.uniform(-velocity_range, velocity_range)
    vz = random.uniform(-0.005, 0.005)  # Slower depth change

    # Base size (standard drone)
    base_size = 0.06

    return Drone(
        x=x, y=y, z=z,
        vx=vx, vy=vy, vz=vz,
        base_size=base_size,
        drone_type=drone_type,
    )


def bbox_to_coco(
    bbox: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> tuple[list[float], float]:
    """Convert normalized bbox to COCO format.

    Args:
        bbox: Normalized (x1, y1, x2, y2) in [0, 1].
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Tuple of (coco_bbox, area) where coco_bbox is [x, y, width, height].
    """
    x1, y1, x2, y2 = bbox

    # Convert to pixel coordinates
    px1 = x1 * img_width
    py1 = y1 * img_height
    px2 = x2 * img_width
    py2 = y2 * img_height

    # Clamp to image bounds
    px1 = max(0, min(img_width, px1))
    py1 = max(0, min(img_height, py1))
    px2 = max(0, min(img_width, px2))
    py2 = max(0, min(img_height, py2))

    # COCO format: [x, y, width, height]
    w = px2 - px1
    h = py2 - py1
    area = w * h

    return [px1, py1, w, h], area


def generate_frame(
    renderer: SpriteRenderer,
    num_drones: int,
    game_state: GameState,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate a single frame with random drones.

    Args:
        renderer: Sprite renderer instance.
        num_drones: Number of drones to render.
        game_state: Game state to populate with drones.

    Returns:
        Tuple of (frame_rgb, annotations) where annotations is list of
        dicts with 'bbox' and 'area' keys.
    """
    # Clear existing drones
    game_state.drones = []

    # Create random drones
    for _ in range(num_drones):
        drone = create_random_drone()
        game_state.drones.append(drone)

    # Render frame
    frame = renderer.render(game_state)

    # Extract annotations
    annotations = []
    for drone in game_state.drones:
        # Skip drones that are off-screen or too small
        if not drone.is_on_screen(margin=0.0):
            continue

        bbox, area = bbox_to_coco(
            drone.bbox,
            renderer.width,
            renderer.height,
        )

        # Skip tiny boxes (less than 4x4 pixels)
        if bbox[2] < 4 or bbox[3] < 4:
            continue

        annotations.append({
            "bbox": bbox,
            "area": area,
        })

    return frame, annotations


def save_image(frame: np.ndarray, path: Path) -> None:
    """Save frame to disk."""
    if HAS_OPENCV:
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    else:
        img = Image.fromarray(frame)
        img.save(path)


def generate_dataset(
    output_dir: Path,
    num_images: int = 1000,
    val_split: float = 0.1,
    negative_ratio: float = 0.1,
    image_size: tuple[int, int] = (320, 320),
    max_drones: int = 5,
    min_drones: int = 1,
    background_change_freq: int = 50,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate COCO-format dataset from simulation.

    Args:
        output_dir: Output directory for dataset.
        num_images: Total number of images to generate.
        val_split: Fraction of images for validation set.
        negative_ratio: Fraction of images with no drones.
        image_size: Image dimensions (width, height).
        max_drones: Maximum drones per frame.
        min_drones: Minimum drones per frame (for non-negative samples).
        background_change_freq: Regenerate background every N frames.
        seed: Random seed for reproducibility.

    Returns:
        Stats dictionary with generation summary.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir = Path(output_dir)

    # Create directory structure
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    ann_dir = output_dir / "annotations"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Initialize renderer and game state
    width, height = image_size
    renderer = SpriteRenderer(width=width, height=height)
    game_state = GameState()

    # COCO annotation structures
    categories = [
        {"id": 1, "name": "drone", "supercategory": "vehicle"}
    ]

    train_data = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    val_data = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    # Counters
    image_id = 0
    annotation_id = 0
    num_train = int(num_images * (1 - val_split))
    num_negatives = int(num_images * negative_ratio)

    # Determine which images are negatives (randomly distributed)
    negative_indices = set(random.sample(range(num_images), num_negatives))

    # Stats
    total_annotations = 0
    train_annotations = 0
    val_annotations = 0

    print(f"Generating {num_images} images ({num_train} train, {num_images - num_train} val)")
    print(f"Including {num_negatives} negative samples ({negative_ratio:.0%})")

    for i in tqdm(range(num_images), desc="Generating"):
        # Regenerate background periodically
        if i % background_change_freq == 0:
            renderer.reset_background()

        # Determine number of drones
        if i in negative_indices:
            num_drones = 0
        else:
            num_drones = random.randint(min_drones, max_drones)

        # Generate frame and annotations
        frame, frame_annotations = generate_frame(renderer, num_drones, game_state)

        # Determine train/val split
        is_train = i < num_train
        img_dir = train_img_dir if is_train else val_img_dir
        data = train_data if is_train else val_data

        # Save image
        filename = f"{i:06d}.png"
        save_image(frame, img_dir / filename)

        # Add image entry
        data["images"].append({
            "id": image_id,
            "file_name": filename,
            "height": height,
            "width": width,
        })

        # Add annotations
        for ann in frame_annotations:
            data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # drone
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": 0,
            })
            annotation_id += 1
            total_annotations += 1
            if is_train:
                train_annotations += 1
            else:
                val_annotations += 1

        image_id += 1

    # Save annotations
    with open(ann_dir / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(ann_dir / "val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    # Summary stats
    stats = {
        "total_images": num_images,
        "train_images": num_train,
        "val_images": num_images - num_train,
        "negative_images": num_negatives,
        "total_annotations": total_annotations,
        "train_annotations": train_annotations,
        "val_annotations": val_annotations,
        "avg_annotations_per_image": total_annotations / num_images,
    }

    print(f"\nDataset generated at: {output_dir}")
    print(f"  Train images: {stats['train_images']}")
    print(f"  Val images: {stats['val_images']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Avg per image: {stats['avg_annotations_per_image']:.1f}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for NanoDet drone detection"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/drone_detection",
        help="Output directory (default: data/drone_detection)",
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=1000,
        help="Number of images to generate (default: 1000)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.1,
        help="Fraction of images with no drones (default: 0.1)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Image size (square, default: 320)",
    )
    parser.add_argument(
        "--max-drones",
        type=int,
        default=5,
        help="Maximum drones per frame (default: 5)",
    )
    parser.add_argument(
        "--min-drones",
        type=int,
        default=1,
        help="Minimum drones per frame (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    generate_dataset(
        output_dir=Path(args.output),
        num_images=args.num_images,
        val_split=args.val_split,
        negative_ratio=args.negative_ratio,
        image_size=(args.image_size, args.image_size),
        max_drones=args.max_drones,
        min_drones=args.min_drones,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
