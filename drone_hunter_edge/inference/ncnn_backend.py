"""NCNN inference backend for mobile deployment.

NCNN is optimized for ARM processors and supports Vulkan GPU acceleration.
"""

from typing import Tuple, Optional
import numpy as np

from inference.backend import InferenceBackend

try:
    import ncnn
    NCNN_AVAILABLE = True
except ImportError:
    ncnn = None
    NCNN_AVAILABLE = False


class NCNNBackend(InferenceBackend):
    """NCNN inference backend.

    Optimized for mobile ARM processors with optional Vulkan GPU.

    NCNN models require two files:
    - model.param: Network architecture
    - model.bin: Weights

    The model_path should point to the .param file, and .bin is auto-detected.
    """

    def __init__(
        self,
        use_vulkan: bool = True,
        num_threads: int = 4,
        input_name: str = "input.1",
        output_name: str = "output",
    ):
        """Initialize NCNN backend.

        Args:
            use_vulkan: Whether to use Vulkan GPU acceleration.
            num_threads: Number of CPU threads.
            input_name: Name of input blob (check with ncnnoptimize).
            output_name: Name of output blob.
        """
        if not NCNN_AVAILABLE:
            raise ImportError(
                "ncnn not available. Install with: pip install ncnn"
            )

        self.use_vulkan = use_vulkan
        self.num_threads = num_threads
        self.input_name = input_name
        self.output_name = output_name

        self.net: Optional[ncnn.Net] = None
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_shape: Optional[Tuple[int, ...]] = None

    def load(self, model_path: str) -> None:
        """Load NCNN model.

        Args:
            model_path: Path to .param file (.bin auto-detected).
        """
        # Auto-detect .bin file
        if model_path.endswith(".param"):
            bin_path = model_path.replace(".param", ".bin")
        else:
            bin_path = model_path + ".bin"
            model_path = model_path + ".param"

        self.net = ncnn.Net()

        # Configure options
        if self.use_vulkan and ncnn.get_gpu_count() > 0:
            self.net.opt.use_vulkan_compute = True
        else:
            self.net.opt.use_vulkan_compute = False

        self.net.opt.num_threads = self.num_threads

        # Load model
        self.net.load_param(model_path)
        self.net.load_model(bin_path)

        # Input shape for NanoDet-Plus 320x320
        # Note: actual shape determined by model, this is default
        self._input_shape = (1, 3, 320, 320)

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            input_tensor: Input of shape (1, C, H, W), float32.

        Returns:
            Model output tensor.
        """
        if self.net is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Remove batch dimension for NCNN (expects CHW)
        input_data = input_tensor[0] if input_tensor.ndim == 4 else input_tensor

        # Create NCNN mat from numpy array
        mat_in = ncnn.Mat(input_data)

        # Create extractor
        ex = self.net.create_extractor()

        # Set input
        ex.input(self.input_name, mat_in)

        # Get output
        ret, mat_out = ex.extract(self.output_name)

        if ret != 0:
            raise RuntimeError(f"NCNN extraction failed with code {ret}")

        # Convert to numpy and add batch dimension
        output = np.array(mat_out)
        if output.ndim == 2:
            output = output[np.newaxis, ...]

        return output

    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get input shape (N, C, H, W)."""
        if self._input_shape is None:
            raise RuntimeError("Model not loaded.")
        return self._input_shape

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape."""
        if self._output_shape is None:
            # For NanoDet, typical output is (1, num_anchors, num_classes + 32)
            return (1, 2125, 33)  # Single-class NanoDet default
        return self._output_shape

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.net is not None

    @property
    def vulkan_enabled(self) -> bool:
        """Check if Vulkan is actually being used."""
        if self.net is None:
            return False
        return self.net.opt.use_vulkan_compute
