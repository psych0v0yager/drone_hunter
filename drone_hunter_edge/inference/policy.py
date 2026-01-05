"""Policy inference for trained PPO agents.

Supports ONNX-exported policies from Stable Baselines3.
"""

from typing import Dict, Tuple, Optional
import numpy as np

from inference.backend import InferenceBackend


class PolicyInference:
    """Run inference on exported PPO policy.

    Expects a policy exported to ONNX format with:
    - Input: flattened observation vector
    - Output: action logits or action values

    For discrete action spaces (like DroneHunter's MultiDiscrete),
    the output should be action indices or logits.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        action_dims: Tuple[int, int, int] = (9, 8, 8),  # fire, grid_x, grid_y
        deterministic: bool = True,
    ):
        """Initialize policy inference.

        Args:
            backend: Inference backend (ONNX or NCNN).
            action_dims: Dimensions of MultiDiscrete action space.
            deterministic: If True, take argmax; if False, sample from logits.
        """
        self.backend = backend
        self.action_dims = action_dims
        self.deterministic = deterministic
        self.total_actions = sum(action_dims)

    def _flatten_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten dict observation to vector.

        Args:
            obs: Dictionary with "target" and "game_state" arrays.

        Returns:
            Flattened observation vector.
        """
        # Concatenate observation components in expected order
        parts = []
        if "target" in obs:
            parts.append(obs["target"].flatten())
        if "game_state" in obs:
            parts.append(obs["game_state"].flatten())

        return np.concatenate(parts).astype(np.float32)

    def _decode_action(self, logits: np.ndarray) -> np.ndarray:
        """Decode action from model output.

        Args:
            logits: Raw model output (action logits or values).

        Returns:
            Action array [fire, grid_x, grid_y].
        """
        # Split logits by action dimension
        actions = []
        offset = 0

        for dim in self.action_dims:
            action_logits = logits[offset:offset + dim]

            if self.deterministic:
                action = np.argmax(action_logits)
            else:
                # Sample from softmax
                probs = np.exp(action_logits - np.max(action_logits))
                probs = probs / np.sum(probs)
                action = np.random.choice(dim, p=probs)

            actions.append(action)
            offset += dim

        return np.array(actions, dtype=np.int32)

    def predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get action from observation.

        Args:
            obs: Dictionary observation with "target" and "game_state".

        Returns:
            Action array [fire, grid_x, grid_y].
        """
        # Flatten observation
        flat_obs = self._flatten_observation(obs)

        # Add batch dimension: (obs_dim,) -> (1, obs_dim)
        input_tensor = flat_obs[np.newaxis, :]

        # Run inference
        output = self.backend.run(input_tensor)

        # Remove batch dimension and decode
        logits = output[0] if output.ndim > 1 else output

        return self._decode_action(logits)


class SimplePolicyInference:
    """Simplified policy inference using direct ONNX loading.

    Alternative to backend-based approach for simpler deployment.
    """

    def __init__(
        self,
        model_path: str,
        action_dims: Tuple[int, int, int] = (9, 8, 8),
        deterministic: bool = True,
    ):
        """Initialize with ONNX model path.

        Args:
            model_path: Path to ONNX policy model.
            action_dims: Dimensions of MultiDiscrete action space.
            deterministic: Take argmax if True.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required for SimplePolicyInference")

        self.action_dims = action_dims
        self.deterministic = deterministic

        # Load model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get action from observation.

        Args:
            obs: Dictionary observation.

        Returns:
            Action array [fire, grid_x, grid_y].
        """
        # Flatten and batch
        parts = []
        if "target" in obs:
            parts.append(obs["target"].flatten())
        if "game_state" in obs:
            parts.append(obs["game_state"].flatten())

        flat_obs = np.concatenate(parts).astype(np.float32)
        input_tensor = flat_obs[np.newaxis, :]

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )

        logits = outputs[0][0]

        # Decode action
        actions = []
        offset = 0
        for dim in self.action_dims:
            action_logits = logits[offset:offset + dim]
            if self.deterministic:
                action = np.argmax(action_logits)
            else:
                probs = np.exp(action_logits - np.max(action_logits))
                probs = probs / np.sum(probs)
                action = np.random.choice(dim, p=probs)
            actions.append(action)
            offset += dim

        return np.array(actions, dtype=np.int32)
