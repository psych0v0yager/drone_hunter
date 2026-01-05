"""Inference backends for Drone Hunter edge deployment."""

from inference.backend import InferenceBackend
from inference.nanodet import NanoDetDetector, create_detector

__all__ = [
    "InferenceBackend",
    "NanoDetDetector",
    "create_detector",
]
