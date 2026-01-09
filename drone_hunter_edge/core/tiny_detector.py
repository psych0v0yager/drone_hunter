"""Tiny detector for fast ROI-based track confirmation.

This is scaffolding for a small (40x40 or 64x64) detector that runs
on cropped regions around Kalman-predicted locations. Much faster than
full-frame detection for track confirmation.

The actual model will be provided separately as an ONNX file.
"""

from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

from core.detection import Detection


class TinyDetector:
    """40x40 ROI detector for fast track confirmation.

    This detector runs on small cropped regions around predicted drone
    locations. It's ~10x faster than full NanoDet since it processes
    ~1/64th the pixels (40x40 vs 320x320).

    Returns full Detection objects (like NanoDet) for Kalman update.

    Usage:
        tiny = TinyDetector("tiny_drone.onnx", roi_size=40)
        detections = tiny.detect_at_roi(frame, predicted_x=0.5, predicted_y=0.5)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        roi_size: int = 40,
        conf_threshold: float = 0.5,
    ):
        """Initialize tiny detector.

        Args:
            model_path: Path to ONNX model (optional, can load later).
            roi_size: Size of square ROI to crop (default 40x40).
            conf_threshold: Minimum confidence threshold.
        """
        self.model_path = model_path
        self.roi_size = roi_size
        self.conf_threshold = conf_threshold
        self.session = None
        self.enabled = False

        if model_path:
            self.load(model_path)

    def load(self, model_path: str) -> bool:
        """Load tiny detector ONNX model.

        Args:
            model_path: Path to ONNX model file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Try XNNPACK first (fast on mobile), fall back to CPU
            providers = ["XnnpackExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
            )

            self.model_path = model_path
            self.enabled = True

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            return True

        except Exception as e:
            print(f"Warning: Failed to load tiny detector: {e}")
            self.enabled = False
            return False

    def _crop_roi(
        self,
        frame: np.ndarray,
        center_x: float,
        center_y: float,
    ) -> Tuple[np.ndarray, int, int]:
        """Crop ROI from frame around predicted center.

        Args:
            frame: Full frame (H, W, 3) uint8 RGB.
            center_x: ROI center x (normalized 0-1).
            center_y: ROI center y (normalized 0-1).

        Returns:
            Tuple of (cropped_roi, x_offset, y_offset) where offsets
            are in pixels for converting back to full-frame coords.
        """
        h, w = frame.shape[:2]

        # Convert normalized coords to pixels
        cx_px = int(center_x * w)
        cy_px = int(center_y * h)

        # Compute ROI bounds (centered on prediction)
        half_size = self.roi_size // 2
        x_min = max(0, cx_px - half_size)
        y_min = max(0, cy_px - half_size)
        x_max = min(w, cx_px + half_size)
        y_max = min(h, cy_px + half_size)

        # Crop ROI
        roi = frame[y_min:y_max, x_min:x_max]

        # Pad if necessary (when near edges)
        if roi.shape[0] < self.roi_size or roi.shape[1] < self.roi_size:
            padded = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
            ph, pw = roi.shape[:2]
            padded[:ph, :pw] = roi
            roi = padded

        return roi, x_min, y_min

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess ROI for inference.

        Args:
            roi: Cropped ROI (roi_size, roi_size, 3) uint8 RGB.

        Returns:
            Preprocessed tensor (1, 3, roi_size, roi_size) float32.
        """
        # Resize to model input size if needed
        if roi.shape[0] != self.roi_size or roi.shape[1] != self.roi_size:
            pil_img = Image.fromarray(roi)
            pil_img = pil_img.resize((self.roi_size, self.roi_size), Image.Resampling.BILINEAR)
            roi = np.array(pil_img)

        # Convert RGB to BGR (NanoDet convention)
        bgr = roi[..., ::-1].copy()

        # Normalize (ImageNet stats)
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        normalized = (bgr.astype(np.float32) - mean) / std

        # NCHW format with batch dimension
        return normalized.transpose(2, 0, 1)[None, ...]

    def _postprocess(
        self,
        outputs: np.ndarray,
        x_offset: int,
        y_offset: int,
        frame_width: int,
        frame_height: int,
    ) -> List[Detection]:
        """Postprocess model outputs to detections in full-frame coords.

        Model output format: [cx, cy, w, h, conf] (5 values)
        - cx, cy: drone center within ROI (0-1 normalized to ROI)
        - w, h: drone size relative to ROI (can be > 1 for clipped drones)
        - conf: confidence score (0-1)

        Args:
            outputs: Raw model output of shape (1, 5).
            x_offset: ROI x offset in pixels.
            y_offset: ROI y offset in pixels.
            frame_width: Full frame width.
            frame_height: Full frame height.

        Returns:
            List of Detection objects with full-frame normalized coords.
        """
        # outputs shape: (1, 5) -> [cx, cy, w, h, conf]
        cx, cy, w, h, conf = outputs[0]

        if conf < self.conf_threshold:
            return []

        # Convert ROI-local coords to full-frame normalized coords
        # cx, cy are 0-1 within ROI
        full_x = (x_offset + cx * self.roi_size) / frame_width
        full_y = (y_offset + cy * self.roi_size) / frame_height
        full_w = (w * self.roi_size) / frame_width
        full_h = (h * self.roi_size) / frame_height

        return [Detection(
            x=float(full_x),
            y=float(full_y),
            w=float(full_w),
            h=float(full_h),
            confidence=float(conf),
        )]

    def detect_at_roi(
        self,
        frame: np.ndarray,
        predicted_x: float,
        predicted_y: float,
    ) -> List[Detection]:
        """Run detection on cropped ROI around predicted location.

        Args:
            frame: Full frame (H, W, 3) uint8 RGB.
            predicted_x: Kalman-predicted center x (normalized 0-1).
            predicted_y: Kalman-predicted center y (normalized 0-1).

        Returns:
            List of Detection objects with coordinates in full-frame space.
            Returns empty list if model not loaded or detection fails.
        """
        if not self.enabled or self.session is None:
            return []

        try:
            h, w = frame.shape[:2]

            # Crop ROI around predicted location
            roi, x_offset, y_offset = self._crop_roi(frame, predicted_x, predicted_y)

            # Preprocess
            input_tensor = self._preprocess(roi)

            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

            # Postprocess and convert to full-frame coords
            detections = self._postprocess(outputs, x_offset, y_offset, w, h)

            return detections

        except Exception as e:
            print(f"Warning: Tiny detector inference failed: {e}")
            return []

    def detect_at_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Tuple[float, float]],
    ) -> List[Detection]:
        """Run detection at multiple predicted track locations.

        Args:
            frame: Full frame (H, W, 3) uint8 RGB.
            tracks: List of (predicted_x, predicted_y) tuples.

        Returns:
            Combined list of detections from all ROIs.
        """
        all_detections = []
        for predicted_x, predicted_y in tracks:
            detections = self.detect_at_roi(frame, predicted_x, predicted_y)
            all_detections.extend(detections)

        # TODO: NMS across ROIs if there's overlap
        return all_detections
