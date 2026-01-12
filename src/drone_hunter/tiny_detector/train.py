"""Training script for TinyDroneNet.

Features:
- TensorBoard logging for loss curves, metrics, and sample visualizations
- Early stopping on validation loss
- Cosine annealing LR schedule
- Configurable architecture for ablations
- Gradient clipping

Usage:
    python -m drone_hunter.tiny_detector.train \
        --data data/tiny_drone/ \
        --epochs 100 \
        --channels 16,32,64,64 \
        --tensorboard
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from drone_hunter.tiny_detector.model import TinyDroneNet, create_tiny_model
from drone_hunter.tiny_detector.dataset import create_dataloaders


def compute_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    has_drone: torch.Tensor,
    conf_weight: float = 2.0,  # Reduced from 50 to allow bbox learning
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute combined bbox and confidence loss.

    Args:
        outputs: (B, 5) model outputs [cx, cy, w, h, conf].
        targets: (B, 4) ground truth [cx, cy, w, h].
        has_drone: (B,) bool mask for positive samples.
        conf_weight: Weight for confidence loss to balance with bbox loss.
            Default 50.0 to make conf loss ~equal to bbox loss magnitude.

    Returns:
        Tuple of (total_loss, loss_dict).
    """
    # Split outputs
    pred_bbox = outputs[:, :4]  # [cx, cy, w, h]
    pred_conf = outputs[:, 4]  # confidence

    # Confidence loss (all samples) with class weighting
    conf_target = has_drone.float()
    # Handle class imbalance: negatives are minority, weight them higher
    num_pos = has_drone.sum().float().clamp(min=1)
    num_neg = (~has_drone).sum().float().clamp(min=1)
    neg_weight = num_pos / num_neg  # weight negatives higher since fewer of them

    # BCEWithLogitsLoss would be better but we already have sigmoid in model
    # Instead, manually weight the loss per sample
    per_sample_loss = F.binary_cross_entropy(pred_conf, conf_target, reduction='none')
    # Weight negative samples higher (since model tends to predict all positive)
    sample_weights = torch.where(has_drone, torch.ones_like(per_sample_loss),
                                  neg_weight * torch.ones_like(per_sample_loss))
    conf_loss = (per_sample_loss * sample_weights).mean()

    # Bbox loss (positive samples only)
    if has_drone.any():
        pos_mask = has_drone
        bbox_loss = F.smooth_l1_loss(pred_bbox[pos_mask], targets[pos_mask])
    else:
        bbox_loss = torch.tensor(0.0, device=outputs.device)

    # Total loss - weight conf_loss to be comparable to bbox_loss
    total_loss = bbox_loss + conf_weight * conf_loss

    return total_loss, {
        "total": total_loss.item(),
        "bbox": bbox_loss.item(),
        "conf": conf_loss.item(),
    }


def compute_metrics(
    outputs: torch.Tensor,
    has_drone: torch.Tensor,
    conf_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        outputs: (B, 5) model outputs.
        has_drone: (B,) bool ground truth.
        conf_threshold: Confidence threshold for predictions.

    Returns:
        Dict with precision, recall, f1, accuracy.
    """
    pred_conf = outputs[:, 4]
    pred_positive = pred_conf > conf_threshold
    true_positive = has_drone

    tp = (pred_positive & true_positive).sum().item()
    fp = (pred_positive & ~true_positive).sum().item()
    fn = (~pred_positive & true_positive).sum().item()
    tn = (~pred_positive & ~true_positive).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch.

    Returns:
        Dict with average losses.
    """
    model.train()
    total_losses = {"total": 0.0, "bbox": 0.0, "conf": 0.0}
    num_batches = 0

    for images, targets, has_drone in loader:
        images = images.to(device)
        targets = targets.to(device)
        has_drone = has_drone.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss, loss_dict = compute_loss(outputs, targets, has_drone)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] += v
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Validate model.

    Returns:
        Tuple of (loss_dict, metrics_dict).
    """
    model.eval()
    total_losses = {"total": 0.0, "bbox": 0.0, "conf": 0.0}
    all_outputs = []
    all_has_drone = []
    num_batches = 0

    for images, targets, has_drone in loader:
        images = images.to(device)
        targets = targets.to(device)
        has_drone = has_drone.to(device)

        outputs = model(images)
        _, loss_dict = compute_loss(outputs, targets, has_drone)

        for k, v in loss_dict.items():
            total_losses[k] += v
        num_batches += 1

        all_outputs.append(outputs.cpu())
        all_has_drone.append(has_drone.cpu())

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    # Compute metrics on all samples
    all_outputs = torch.cat(all_outputs, dim=0)
    all_has_drone = torch.cat(all_has_drone, dim=0)
    metrics = compute_metrics(all_outputs, all_has_drone)

    return avg_losses, metrics


def train(
    data_path: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    channels: List[int] = [16, 32, 64, 64],
    head_dim: int = 32,
    roi_size: int = 40,
    patience: int = 15,
    tensorboard: bool = True,
    device: Optional[str] = None,
) -> Dict:
    """Train TinyDroneNet.

    Args:
        data_path: Path to dataset directory.
        output_dir: Path to save checkpoints and logs.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Initial learning rate.
        weight_decay: Weight decay for AdamW.
        channels: Conv channel sizes for model.
        head_dim: FC head hidden dimension.
        roi_size: Input size (must match dataset).
        patience: Early stopping patience.
        tensorboard: Enable TensorBoard logging.
        device: Device to use (None for auto-detect).

    Returns:
        Dict with training results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create model
    model = TinyDroneNet(channels=channels, head_dim=head_dim, roi_size=roi_size)
    model = model.to(device)
    print(f"Model: {model.count_parameters():,} parameters")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_path, batch_size=batch_size, roi_size=roi_size
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # TensorBoard
    writer = None
    if tensorboard:
        writer = SummaryWriter(output_dir / "tensorboard")
        # Log model config
        writer.add_text("config/channels", str(channels))
        writer.add_text("config/head_dim", str(head_dim))
        writer.add_text("config/roi_size", str(roi_size))
        writer.add_scalar("config/parameters", model.count_parameters())

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_losses, val_metrics = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        # Log
        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_losses["total"])
        history["val_metrics"].append(val_metrics)

        if writer:
            writer.add_scalar("loss/train", train_losses["total"], epoch)
            writer.add_scalar("loss/train_bbox", train_losses["bbox"], epoch)
            writer.add_scalar("loss/train_conf", train_losses["conf"], epoch)
            writer.add_scalar("loss/val", val_losses["total"], epoch)
            writer.add_scalar("loss/val_bbox", val_losses["bbox"], epoch)
            writer.add_scalar("loss/val_conf", val_losses["conf"], epoch)
            writer.add_scalar("metrics/precision", val_metrics["precision"], epoch)
            writer.add_scalar("metrics/recall", val_metrics["recall"], epoch)
            writer.add_scalar("metrics/f1", val_metrics["f1"], epoch)
            writer.add_scalar("metrics/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_losses['total']:.4f} | "
              f"Val: {val_losses['total']:.4f} | "
              f"F1: {val_metrics['f1']:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # Early stopping
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "config": {
                    "channels": channels,
                    "head_dim": head_dim,
                    "roi_size": roi_size,
                },
            }, output_dir / "best.pt")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
                break

    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "channels": channels,
            "head_dim": head_dim,
            "roi_size": roi_size,
        },
    }, output_dir / "final.pt")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()

    return {
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
        "total_time": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Train TinyDroneNet")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="runs/tiny_detector", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--channels", type=str, default="16,32,64,64",
                        help="Conv channel sizes (comma-separated)")
    parser.add_argument("--head-dim", type=int, default=32, help="FC head hidden dim")
    parser.add_argument("--roi-size", type=int, default=40, help="Input size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Parse channels
    channels = [int(x) for x in args.channels.split(",")]

    # Add timestamp to output dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"

    print("=" * 60)
    print("TINY DRONE DETECTOR TRAINING")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Channels: {channels}")
    print(f"Head dim: {args.head_dim}")
    print(f"ROI size: {args.roi_size}")
    print("=" * 60)

    results = train(
        data_path=args.data,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        channels=channels,
        head_dim=args.head_dim,
        roi_size=args.roi_size,
        patience=args.patience,
        tensorboard=args.tensorboard,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Epochs trained: {results['epochs_trained']}")
    print(f"Total time: {results['total_time']/60:.1f} minutes")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
