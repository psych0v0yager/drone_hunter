"""Abstract inference backend interface.

Defines a common interface for different inference backends (ONNX, NCNN).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model from file.

        Args:
            model_path: Path to model file.
        """
        pass

    @abstractmethod
    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference on input tensor.

        Args:
            input_tensor: Input tensor of shape (1, C, H, W).

        Returns:
            Output tensor from model.
        """
        pass

    @abstractmethod
    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape (N, C, H, W)."""
        pass

    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @property
    def name(self) -> str:
        """Backend name for logging."""
        return self.__class__.__name__
