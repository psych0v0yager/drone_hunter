"""Detection dataclass for object detector outputs."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    """A single detection from the object detector.

    All coordinates are normalized to [0, 1] range.
    """
    x: float           # Center x (normalized 0-1)
    y: float           # Center y (normalized 0-1)
    w: float           # Width (normalized)
    h: float           # Height (normalized)
    confidence: float  # Detection confidence
    class_id: int = 0  # Class ID (for multi-class detectors)

    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates."""
        return (self.x, self.y)

    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.w * self.h

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """Get bounding box as (x1, y1, x2, y2)."""
        half_w = self.w / 2
        half_h = self.h / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )
