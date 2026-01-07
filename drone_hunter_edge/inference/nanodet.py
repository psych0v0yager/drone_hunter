"""NanoDet object detection with pluggable backends.

Backend-agnostic implementation using Pillow for preprocessing.
Supports ONNX Runtime and NCNN backends.
"""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

from core.detection import Detection
from inference.backend import InferenceBackend


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _generate_anchors(
    featmap_sizes: List[Tuple[int, int]],
    strides: List[int],
) -> np.ndarray:
    """Generate anchor points for NanoDet.

    Args:
        featmap_sizes: List of (height, width) for each feature map level.
        strides: List of strides for each level.

    Returns:
        Anchors array of shape (1, num_anchors, 4) with [x, y, stride, stride].
    """
    anchors_list = []
    for i, stride in enumerate(strides):
        h, w = featmap_sizes[i]
        x_range = np.arange(w) * stride
        y_range = np.arange(h) * stride
        y, x = np.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        stride_arr = np.ones_like(x) * stride
        anchors = np.stack([y, x, stride_arr, stride_arr], axis=-1)
        anchors = np.expand_dims(anchors, axis=0)
        anchors_list.append(anchors)
    return np.concatenate(anchors_list, axis=1)


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Non-maximum suppression (pure numpy).

    Args:
        boxes: Array of shape (N, 4) with [x_center, y_center, width, height].
        scores: Array of shape (N,) with confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Array of indices to keep.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    # Convert to corner format
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


class NanoDetDetector:
    """NanoDet-Plus object detector with pluggable backends.

    This class handles preprocessing and postprocessing, delegating
    the actual inference to a backend (ONNX or NCNN).
    """

    # NanoDet-Plus constants
    DEFAULT_STRIDES = [8, 16, 32, 64]
    REG_MAX = 7
    BOX_PARAMS = 32  # 4 directions * (reg_max + 1) = 4 * 8 = 32

    # ImageNet normalization (BGR format)
    MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)

    def __init__(
        self,
        backend: InferenceBackend,
        input_size: Tuple[int, int] = (320, 320),
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        num_classes: Optional[int] = None,
    ):
        """Initialize NanoDet detector.

        Args:
            backend: Inference backend (ONNXBackend or NCNNBackend).
            input_size: Model input size (height, width).
            conf_threshold: Minimum confidence threshold.
            iou_threshold: IoU threshold for NMS.
            num_classes: Number of classes. Auto-detected from ONNX, required for NCNN.
        """
        self.backend = backend
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Auto-detect or use provided num_classes
        if num_classes is not None:
            self.num_classes = num_classes
        elif backend.is_loaded:
            # Try to detect from output shape
            output_shape = backend.get_output_shape()
            self.num_classes = output_shape[-1] - self.BOX_PARAMS
        else:
            self.num_classes = 1  # Default for custom drone model

        # Pre-compute anchors
        h, w = input_size
        self.featmap_sizes = [
            (int(np.ceil(h / s)), int(np.ceil(w / s)))
            for s in self.DEFAULT_STRIDES
        ]
        self.anchors = _generate_anchors(self.featmap_sizes, self.DEFAULT_STRIDES)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference using Pillow.

        Args:
            frame: RGB image of shape (H, W, 3), uint8.

        Returns:
            Preprocessed tensor of shape (1, 3, H, W), float32.
        """
        h, w = self.input_size

        # Use Pillow for resize (no cv2 dependency)
        pil_img = Image.fromarray(frame)
        pil_img = pil_img.resize((w, h), Image.Resampling.BILINEAR)
        resized = np.array(pil_img)

        # Convert RGB to BGR (NanoDet uses BGR)
        bgr = resized[..., ::-1].copy()

        # Normalize
        normalized = (bgr.astype(np.float32) - self.MEAN) / self.STD

        # NCHW format
        transposed = normalized.transpose(2, 0, 1)

        # Add batch dimension
        return transposed[None, ...]

    def _postprocess(
        self,
        outputs: np.ndarray,
    ) -> List[Detection]:
        """Postprocess model outputs to detections.

        Args:
            outputs: Raw model output of shape (1, num_anchors, num_classes + 32).

        Returns:
            List of Detection objects with normalized coordinates.
        """
        model_h, model_w = self.input_size

        # Split classes and boxes
        classes = outputs[..., :self.num_classes]
        boxes_raw = outputs[..., self.num_classes:]

        # Apply sigmoid to class scores
        classes = 1 / (1 + np.exp(-classes))

        # Decode boxes using distance-to-box conversion
        batch = boxes_raw.shape[0]
        x = boxes_raw.reshape(batch, -1, 4, self.REG_MAX + 1)
        x = _softmax(x, axis=-1)
        x = np.matmul(x, np.arange(0, self.REG_MAX + 1, dtype=np.float32))
        x = x.reshape(batch, -1, 4)

        # Scale by stride
        distances = x * self.anchors[..., 2:3]

        # Convert distances to boxes [x_center, y_center, width, height]
        w = distances[..., 0:1] + distances[..., 2:3]
        h = distances[..., 1:2] + distances[..., 3:4]
        x_c = self.anchors[..., 0:1] - distances[..., 0:1] + w / 2
        y_c = self.anchors[..., 1:2] - distances[..., 1:2] + h / 2
        boxes = np.concatenate([x_c, y_c, w, h], axis=-1)

        # Remove batch dimension
        boxes = boxes[0]
        classes = classes[0]

        # Get max class score and class ID for each box
        max_scores = np.max(classes, axis=-1)
        class_ids = np.argmax(classes, axis=-1)

        # Filter by confidence
        mask = max_scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Apply NMS
        keep_indices = _nms(boxes, scores, self.iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]

        # Normalize coordinates to [0, 1]
        boxes[:, 0] /= model_w  # x_center
        boxes[:, 1] /= model_h  # y_center
        boxes[:, 2] /= model_w  # width
        boxes[:, 3] /= model_h  # height

        # Clip to valid range
        boxes = np.clip(boxes, 0, 1)

        # Create Detection objects
        detections = []
        for i in range(len(boxes)):
            det = Detection(
                x=float(boxes[i, 0]),
                y=float(boxes[i, 1]),
                w=float(boxes[i, 2]),
                h=float(boxes[i, 3]),
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
            )
            detections.append(det)

        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run object detection on a frame.

        Args:
            frame: RGB image of shape (H, W, 3), uint8.

        Returns:
            List of Detection objects with normalized [0, 1] coordinates.
        """
        # Preprocess
        input_tensor = self._preprocess(frame)

        # Run inference via backend
        outputs = self.backend.run(input_tensor)

        # Postprocess
        detections = self._postprocess(outputs)

        return detections


def create_detector(
    model_path: str,
    backend_type: str = "onnx",
    input_size: Tuple[int, int] = (320, 320),
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    num_classes: int = 1,
    **backend_kwargs,
) -> NanoDetDetector:
    """Factory function to create NanoDet detector with specified backend.

    Args:
        model_path: Path to model file (.onnx or .param).
        backend_type: Backend type ("onnx" or "ncnn").
        input_size: Model input size (height, width).
        conf_threshold: Minimum confidence threshold.
        iou_threshold: IoU threshold for NMS.
        num_classes: Number of classes (default 1 for drone detection).
        **backend_kwargs: Additional arguments for backend constructor.
            For ONNX backend:
                - provider_priority: "auto", "desktop", "mobile", "mobile-npu",
                  or specific provider like "xnnpack", "nnapi", "cuda", "cpu"

    Returns:
        Configured NanoDetDetector instance.
    """
    if backend_type.lower() == "onnx":
        from inference.onnx_backend import ONNXBackend
        backend = ONNXBackend(**backend_kwargs)
    elif backend_type.lower() == "ncnn":
        from inference.ncnn_backend import NCNNBackend
        backend = NCNNBackend(**backend_kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    backend.load(model_path)

    return NanoDetDetector(
        backend=backend,
        input_size=input_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        num_classes=num_classes,
    )
