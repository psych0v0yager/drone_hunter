#!/usr/bin/env python3
"""Export trained TinyDroneNet to ONNX format.

Usage:
    python scripts/export_tiny_onnx.py \
        --checkpoint runs/tiny_detector/best.pt \
        --output models/tiny_drone.onnx
"""

import argparse
from pathlib import Path

import torch
import torch.onnx

from drone_hunter.tiny_detector.model import TinyDroneNet


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 21,
    fp16: bool = False,
):
    """Export model to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        output_path: Output .onnx path.
        opset_version: ONNX opset version.
        fp16: Export with FP16 weights.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    channels = config.get("channels", [16, 32, 64, 64])
    head_dim = config.get("head_dim", 32)
    roi_size = config.get("roi_size", 40)

    print(f"Model config: channels={channels}, head_dim={head_dim}, roi_size={roi_size}")

    # Create model
    model = TinyDroneNet(channels=channels, head_dim=head_dim, roi_size=roi_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Count parameters
    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,}")

    # Dummy input
    dummy_input = torch.randn(1, 3, roi_size, roi_size)

    # Export
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes=None,  # Fixed batch size
        do_constant_folding=True,
    )

    # Convert external data to internal (single file)
    import onnx
    onnx_model = onnx.load(str(output_path))

    # Check if external data was created and convert to single file
    data_file = Path(str(output_path) + ".data")
    if data_file.exists():
        print("Converting external data to single file...")
        from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
        load_external_data_for_model(onnx_model, str(output_path.parent))
        # Save with all tensors embedded
        onnx.save(
            onnx_model,
            str(output_path),
            save_as_external_data=False,
        )
        # Remove the .data file
        data_file.unlink()
        print(f"Removed external data file: {data_file}")

    # Verify
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully!")

    # Get file size
    file_size = output_path.stat().st_size
    print(f"Output: {output_path} ({file_size / 1024:.1f} KB)")

    # FP16 conversion
    if fp16:
        fp16_path = output_path.with_suffix(".fp16.onnx")
        try:
            from onnxconverter_common import float16
            fp16_model = float16.convert_float_to_float16(onnx_model)
            onnx.save(fp16_model, str(fp16_path))
            fp16_size = fp16_path.stat().st_size
            print(f"FP16 output: {fp16_path} ({fp16_size / 1024:.1f} KB)")
        except ImportError:
            print("Warning: onnxconverter-common not installed, skipping FP16 conversion")

    # Test inference
    print("\nTesting inference...")
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    test_input = dummy_input.numpy()
    outputs = session.run([output_name], {input_name: test_input})[0]
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output: cx={outputs[0, 0]:.3f}, cy={outputs[0, 1]:.3f}, "
          f"w={outputs[0, 2]:.3f}, h={outputs[0, 3]:.3f}, conf={outputs[0, 4]:.3f}")

    print("\nExport complete!")


def main():
    parser = argparse.ArgumentParser(description="Export TinyDroneNet to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="models/tiny_drone.onnx",
                        help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=21,
                        help="ONNX opset version")
    parser.add_argument("--fp16", action="store_true",
                        help="Also export FP16 version")

    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
