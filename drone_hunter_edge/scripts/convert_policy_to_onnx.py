#!/usr/bin/env python3
"""Convert trained SB3 PPO policy to ONNX format for edge deployment.

This script exports the policy network from a trained PPO model to ONNX,
along with the VecNormalize statistics as JSON.

Usage:
    python convert_policy_to_onnx.py runs/detector_v2/best_model.zip \
        --vec-normalize runs/detector_v2/vec_normalize.pkl \
        --output models/policy.onnx
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def load_sb3_model(model_path: str):
    """Load a Stable Baselines3 PPO model."""
    from stable_baselines3 import PPO
    return PPO.load(model_path)


def export_vec_normalize_stats(
    vec_normalize_path: str,
    output_path: str,
) -> None:
    """Export VecNormalize statistics to JSON.

    Args:
        vec_normalize_path: Path to VecNormalize .pkl file.
        output_path: Path for output JSON file.
    """
    import pickle

    with open(vec_normalize_path, "rb") as f:
        vec_normalize = pickle.load(f)

    stats = {}

    # Extract observation normalization stats
    if hasattr(vec_normalize, "obs_rms"):
        obs_rms = vec_normalize.obs_rms

        # Handle dict observations
        if isinstance(obs_rms, dict):
            for key, rms in obs_rms.items():
                stats[key] = {
                    "mean": rms.mean.tolist(),
                    "var": rms.var.tolist(),
                    "clip": vec_normalize.clip_obs,
                }
        else:
            # Single observation space
            stats["obs"] = {
                "mean": obs_rms.mean.tolist(),
                "var": obs_rms.var.tolist(),
                "clip": vec_normalize.clip_obs,
            }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved normalization stats to: {output_path}")


class PolicyWrapper(nn.Module):
    """Wrapper to export SB3 policy to ONNX.

    Handles the dict observation space by flattening inputs.
    """

    def __init__(self, policy, obs_keys=("target", "game_state")):
        super().__init__()
        self.policy = policy
        self.obs_keys = obs_keys

    def forward(self, x):
        """Forward pass with flattened observation.

        Args:
            x: Flattened observation tensor (batch, obs_dim).

        Returns:
            Action logits tensor.
        """
        # The policy's mlp_extractor and action_net handle the forward pass
        features = self.policy.mlp_extractor(x)[0]  # Get policy features
        action_logits = self.policy.action_net(features)
        return action_logits


def export_policy_to_onnx(
    model_path: str,
    output_path: str,
    obs_dim: int = 24,  # 19 (target) + 5 (game_state) for single_target mode
    opset_version: int = 14,
) -> None:
    """Export SB3 PPO policy to ONNX.

    Args:
        model_path: Path to SB3 model .zip file.
        output_path: Path for output ONNX file.
        obs_dim: Flattened observation dimension.
        opset_version: ONNX opset version.
    """
    # Load model
    model = load_sb3_model(model_path)

    # Get policy network
    policy = model.policy

    # Create wrapper
    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"Exported policy to: {output_path}")

    # Verify the exported model
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")

    # Test inference
    session = ort.InferenceSession(output_path)
    test_input = np.random.randn(1, obs_dim).astype(np.float32)
    outputs = session.run(None, {"observation": test_input})
    print(f"Test inference output shape: {outputs[0].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SB3 PPO model to ONNX for edge deployment"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to SB3 model .zip file"
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to VecNormalize .pkl file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="policy.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--normalization-output",
        type=str,
        default=None,
        help="Output JSON file for normalization stats"
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=24,
        help="Flattened observation dimension (19 + 5 for single_target mode)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version"
    )

    args = parser.parse_args()

    # Export policy
    export_policy_to_onnx(
        args.model_path,
        args.output,
        obs_dim=args.obs_dim,
        opset_version=args.opset,
    )

    # Export normalization stats if provided
    if args.vec_normalize:
        norm_output = args.normalization_output or args.output.replace(".onnx", "_normalization.json")
        export_vec_normalize_stats(args.vec_normalize, norm_output)


if __name__ == "__main__":
    main()
