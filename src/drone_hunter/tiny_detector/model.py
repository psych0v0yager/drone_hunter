"""TinyDroneNet: Lightweight CNN for 40x40 ROI drone detection.

This model runs on small cropped regions around Kalman-predicted locations.
Much faster than full NanoDet since it processes ~1/64th the pixels.

Output format: [cx, cy, w, h, conf]
- cx, cy: drone center within ROI (0-1 normalized)
- w, h: drone size relative to ROI (can be > 1 for clipped drones)
- conf: confidence score (sigmoid)
"""

from typing import List, Optional
import torch
import torch.nn as nn


class TinyDroneNet(nn.Module):
    """Minimal CNN for 40x40 drone detection.

    Architecture is configurable for ablation studies:
    - channels: Number of channels at each conv layer
    - head_dim: Hidden dimension in FC head
    - roi_size: Input size (must match training data)
    """

    def __init__(
        self,
        channels: List[int] = [16, 32, 64, 64],
        head_dim: int = 32,
        roi_size: int = 40,
    ):
        """Initialize TinyDroneNet.

        Args:
            channels: List of channel sizes for conv layers.
                Default [16, 32, 64, 64] gives ~50k params.
                Use [8, 16, 32, 32] for ~15k params (tiny variant).
            head_dim: Hidden dimension for FC head.
            roi_size: Input image size (square).
        """
        super().__init__()
        self.roi_size = roi_size
        self.channels = channels
        self.head_dim = head_dim

        # Build feature extractor
        layers = []
        in_channels = 3
        for i, out_channels in enumerate(channels):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels

        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)

        # Compute feature size after pooling
        self.feature_size = channels[-1]

        # FC head for bbox + confidence
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, 5),  # [cx, cy, w, h, conf]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, roi_size, roi_size).

        Returns:
            Output tensor of shape (B, 5) with [cx, cy, w, h, conf].
            - cx, cy: sigmoid activated (0-1)
            - w, h: exp activated (positive, unbounded)
            - conf: sigmoid activated (0-1)
        """
        x = self.features(x)
        x = self.head(x)

        # Apply activations
        cx = torch.sigmoid(x[:, 0:1])
        cy = torch.sigmoid(x[:, 1:2])
        wh = torch.exp(x[:, 2:4])  # Positive, can be > 1 for clipped drones
        conf = torch.sigmoid(x[:, 4:5])

        return torch.cat([cx, cy, wh, conf], dim=1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tiny_model(
    variant: str = "small",
    roi_size: int = 40,
    channels: Optional[List[int]] = None,
    head_dim: Optional[int] = None,
) -> TinyDroneNet:
    """Factory function to create TinyDroneNet with preset or custom config.

    Args:
        variant: Preset variant name ("tiny", "small", "medium").
        roi_size: Input size (square).
        channels: Override channel sizes (if provided, ignores variant).
        head_dim: Override head dimension (if provided, ignores variant).

    Returns:
        Configured TinyDroneNet instance.
    """
    # Preset configurations
    presets = {
        "tiny": {"channels": [8, 16, 32, 32], "head_dim": 16},
        "small": {"channels": [16, 32, 64, 64], "head_dim": 32},
        "medium": {"channels": [32, 64, 128, 128], "head_dim": 64},
    }

    if channels is None or head_dim is None:
        if variant not in presets:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(presets.keys())}")
        preset = presets[variant]
        channels = channels or preset["channels"]
        head_dim = head_dim or preset["head_dim"]

    model = TinyDroneNet(channels=channels, head_dim=head_dim, roi_size=roi_size)

    return model


if __name__ == "__main__":
    # Quick test
    for variant in ["tiny", "small", "medium"]:
        model = create_tiny_model(variant=variant)
        params = model.count_parameters()
        print(f"{variant}: {params:,} params ({params * 2 / 1024:.1f} KB fp16)")

        # Test forward pass
        x = torch.randn(1, 3, 40, 40)
        y = model(x)
        print(f"  Output shape: {y.shape}")
        print(f"  cx={y[0, 0]:.3f}, cy={y[0, 1]:.3f}, w={y[0, 2]:.3f}, h={y[0, 3]:.3f}, conf={y[0, 4]:.3f}")
        print()
