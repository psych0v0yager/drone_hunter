"""Tiny detector module for fast ROI-based drone detection.

This module provides training code for a lightweight 40x40 CNN that runs
on cropped regions around Kalman-predicted locations. The trained model
is deployed to drone_hunter_edge for Tier 1 detection.

Components:
- model.py: TinyDroneNet architecture (configurable for ablations)
- dataset.py: PyTorch Dataset for loading 40x40 crops
- train.py: Training loop with TensorBoard logging
"""

from drone_hunter.tiny_detector.model import TinyDroneNet, create_tiny_model

__all__ = ["TinyDroneNet", "create_tiny_model"]
