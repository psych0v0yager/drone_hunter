#!/usr/bin/env python3
"""Compare rendered frames between edge and original to spot differences."""

import sys
from pathlib import Path
import random
import numpy as np
from PIL import Image

# Set seeds
random.seed(42)
np.random.seed(42)

sys.path.insert(0, str(Path(__file__).parent))

from core.game_state import GameState
from core.renderer import Renderer

# Reset seeds after imports
random.seed(42)
np.random.seed(42)

def main():
    # Create game state and renderer
    game_state = GameState(max_frames=200)
    renderer = Renderer(width=320, height=320)

    game_state.reset()
    renderer.reset_background()

    # Step until we have a drone on screen
    for _ in range(20):
        game_state.step()

    # Render frame
    frame = renderer.render(game_state)

    # Save frame
    out_path = Path(__file__).parent / "test_frame_edge.png"
    Image.fromarray(frame).save(out_path)
    print(f"Saved edge frame to: {out_path}")

    # Print drone info
    for i, drone in enumerate(game_state.drones):
        if drone.is_on_screen():
            print(f"Drone {i}: pos=({drone.x:.3f}, {drone.y:.3f}), z={drone.z:.3f}, "
                  f"size={drone.size:.4f}, size_px={int(drone.size * 320)}")

    # Also run detector on the frame
    model_path = Path(__file__).parent / "models" / "nanodet.onnx"
    if model_path.exists():
        from inference.nanodet import create_detector
        detector = create_detector(
            str(model_path),
            backend_type="onnx",
            conf_threshold=0.55,
            iou_threshold=0.5,
        )
        detections = detector.detect(frame)
        print(f"\nDetector found {len(detections)} detection(s):")
        for det in detections:
            print(f"  x={det.x:.3f}, y={det.y:.3f}, w={det.w:.4f}, h={det.h:.4f}, conf={det.confidence:.3f}")


if __name__ == "__main__":
    main()
