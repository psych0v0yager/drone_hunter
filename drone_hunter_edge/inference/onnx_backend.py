"""ONNX Runtime inference backend.

Supports CPU execution on Termux, with optional GPU acceleration on desktop.
"""

from typing import Tuple, Optional
import numpy as np

from inference.backend import InferenceBackend

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


class ONNXBackend(InferenceBackend):
    """ONNX Runtime inference backend.

    Automatically selects best available execution provider:
    - CUDAExecutionProvider (NVIDIA GPU)
    - CPUExecutionProvider (fallback)
    """

    def __init__(self, use_gpu: bool = True):
        """Initialize ONNX backend.

        Args:
            use_gpu: Whether to attempt GPU acceleration.
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime not available. Install with: pip install onnxruntime"
            )

        self.use_gpu = use_gpu
        self.session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_shape: Optional[Tuple[int, ...]] = None

    def load(self, model_path: str) -> None:
        """Load ONNX model.

        Args:
            model_path: Path to .onnx model file.
        """
        providers = []

        if self.use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")

        # Session options for mobile optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

        # Cache input/output info
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]

        self._input_name = input_info.name
        self._output_name = output_info.name
        self._input_shape = tuple(input_info.shape)
        self._output_shape = tuple(output_info.shape)

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            input_tensor: Input of shape (1, C, H, W), float32.

        Returns:
            Model output tensor.
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        outputs = self.session.run(
            [self._output_name],
            {self._input_name: input_tensor.astype(np.float32)},
        )

        return outputs[0]

    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get input shape (N, C, H, W)."""
        if self._input_shape is None:
            raise RuntimeError("Model not loaded.")
        return self._input_shape

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape."""
        if self._output_shape is None:
            raise RuntimeError("Model not loaded.")
        return self._output_shape

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.session is not None

    @property
    def active_provider(self) -> str:
        """Get the active execution provider."""
        if self.session is None:
            return "None"
        return self.session.get_providers()[0]
