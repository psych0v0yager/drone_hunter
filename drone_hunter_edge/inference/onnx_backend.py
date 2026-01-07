"""ONNX Runtime inference backend.

Supports CPU execution on Termux, with optional GPU acceleration on desktop.
Includes mobile execution providers (NNAPI, XNNPACK) for Android devices.
"""

import platform
import sys
from typing import List, Optional, Tuple, Union

import numpy as np

from inference.backend import InferenceBackend

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


# Provider priority presets
PROVIDER_PRESETS = {
    "desktop": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "mobile": ["XnnpackExecutionProvider", "CPUExecutionProvider"],
    "mobile-npu": ["NnapiExecutionProvider", "XnnpackExecutionProvider", "CPUExecutionProvider"],
}


def get_execution_providers(
    priority: Union[str, List[str]] = "auto"
) -> List[str]:
    """Get ordered list of available execution providers.

    Args:
        priority: Provider selection strategy:
            - "auto": Detect platform (desktop vs mobile)
            - "desktop": CUDA -> CPU
            - "mobile": XNNPACK -> CPU (best for budget phones)
            - "mobile-npu": NNAPI -> XNNPACK -> CPU (for flagships with NPU)
            - "nnapi", "xnnpack", "cuda", "cpu": Specific provider + CPU fallback
            - List of providers: Custom order (e.g., ["XnnpackExecutionProvider", "CPUExecutionProvider"])

    Returns:
        List of available providers in priority order, always ending with CPUExecutionProvider.
    """
    if not ONNX_AVAILABLE:
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()

    # Handle custom list
    if isinstance(priority, list):
        requested = priority
    # Handle preset names
    elif priority == "auto":
        # Detect platform: Android/ARM = mobile, else desktop
        is_mobile = (
            "android" in sys.platform.lower()
            or "TERMUX_VERSION" in __import__("os").environ
            or platform.machine().lower() in ("aarch64", "arm64", "armv7l")
        )
        preset_name = "mobile" if is_mobile else "desktop"
        requested = PROVIDER_PRESETS[preset_name]
    elif priority in PROVIDER_PRESETS:
        requested = PROVIDER_PRESETS[priority]
    elif priority.lower() in ("nnapi", "xnnpack", "cuda", "cpu"):
        # Map shorthand to full provider name
        provider_map = {
            "nnapi": "NnapiExecutionProvider",
            "xnnpack": "XnnpackExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }
        requested = [provider_map[priority.lower()], "CPUExecutionProvider"]
    else:
        # Unknown preset, fall back to auto
        requested = PROVIDER_PRESETS["desktop"]

    # Filter to only available providers, always ensure CPU fallback
    providers = [p for p in requested if p in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    return providers


class ONNXBackend(InferenceBackend):
    """ONNX Runtime inference backend.

    Automatically selects best available execution provider:
    - Desktop: CUDAExecutionProvider -> CPUExecutionProvider
    - Mobile: XnnpackExecutionProvider -> CPUExecutionProvider
    - Mobile NPU: NnapiExecutionProvider -> XnnpackExecutionProvider -> CPUExecutionProvider
    """

    def __init__(
        self,
        use_gpu: bool = True,
        provider_priority: Union[str, List[str]] = "auto",
    ):
        """Initialize ONNX backend.

        Args:
            use_gpu: Whether to attempt GPU acceleration (legacy, use provider_priority instead).
            provider_priority: Provider selection strategy:
                - "auto": Detect platform (desktop vs mobile)
                - "desktop": CUDA -> CPU
                - "mobile": XNNPACK -> CPU (best for budget phones)
                - "mobile-npu": NNAPI -> XNNPACK -> CPU (for flagships with NPU)
                - "nnapi", "xnnpack", "cuda", "cpu": Specific provider + CPU fallback
                - List of providers: Custom order
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime not available. Install with: pip install onnxruntime"
            )

        self.use_gpu = use_gpu
        self.provider_priority = provider_priority
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
        providers = get_execution_providers(self.provider_priority)

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
