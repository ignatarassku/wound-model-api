"""
train_tissue.py — Training loop for multi-class tissue type segmentation.

Strategy
--------
1. Load TissueUNet with encoder bootstrapped from the pretrained binary UNet.
2. Freeze the encoder for the first WARMUP_EPOCHS, then unfreeze and fine-tune
   with a very small LR.
3. Optimize with TissueLoss (Focal + multi-class Dice).
4. Evaluate per-class IoU and macro-IoU each epoch.
5. Save the best checkpoint (by macro-IoU on the validation set).

Prerequisite
------------
• Binary UNet checkpoint must exist at CFG.BEST_MODEL_PATH.
• Tissue masks must be placed in CFG.TISSUE_MASKS_DIR with RGB colour coding
  as defined in CFG.TISSUE_COLOURS.

Run
---
    python -m src.train_tissue
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import CFG, ensure_dirs
from src.dataset import get_tissue_dataloaders
from src.losses import TissueLoss
from src.model import TissueUNet

WARMUP_EPOCHS = 5    # encoder frozen during first N epochs
IGNORE_INDEX  = 255


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_per_class_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> List[float]:
    """
    Compute IoU per tissue class, ignoring IGNORE_INDEX pixels.

    Args:
        preds    : [B, C, H, W] logits
        targets  : [B, H, W] class indices
        num_classes : number of tissue classes

    Returns:
        List of IoU values, one per class (NaN if class absent in batch).
    """
    pred_cls = preds.argmax(dim=1)    # [B, H, W]
    valid    = targets != ignore_index

    ious: List[float] = []
    for c in range(num_classes):
        pred_c = (pred_cls == c) & valid
        gt_c   = (targets == c) & valid
        inter  = (pred_c & gt_c).sum().item()
        union  = (pred_c | gt_c).sum().item()
        ious.append(float("nan") if union == 0 else inter / union)
    return ious


def nanmean(values: List[float]) -> float:
    valid = [v for v in values if not np.isnan(v)]
    return float("nan") if not valid else sum(valid) / len(valid)


# ── One epoch ─────────────────────────────────────────────────────────────

def _run_epoch(
    model:       TissueUNet,
    loader:      torch.utils.data.DataLoader,
    criterion:   nn.Module,
    optimizer:   Optional[torch.optim.Optimizer],
    device:      torch.device,
    num_classes: int,
    train:       bool,
) -> Tuple[float, float, List[float]]:
    """
    Run one epoch (train or val).

    Returns:
        mean_loss, macro_iou, per_class_ious
    """
    model.train(train)
    total_loss   = 0.0
    all_ious: List[List[float]] = []

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)                            # [B, C, H, W]
            loss   = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            batch_ious  = compute_per_class_iou(logits, labels, num_classes)
            all_ious.append(batch_ious)

    mean_loss      = total_loss / max(len(loader), 1)
    per_class_ious = [nanmean([b[c] for b in all_ious]) for c in range(num_classes)]
    macro_iou      = nanmean(per_class_ious)
    return mean_loss, macro_iou, per_class_ious


# ── Training loop ─────────────────────────────────────────────────────────

def train_tissue(
    epochs:          int   = CFG.TISSUE_EPOCHS,
    lr:              float = CFG.TISSUE_LR,
    seg_checkpoint:  Optional[str] = str(CFG.BEST_MODEL_PATH),
    save_path:       Path  = CFG.TISSUE_MODEL_PATH,
    metrics_csv:     Path  = CFG.TISSUE_METRICS_CSV,
) -> None:
    """
    Full training loop for tissue type segmentation.

    Args:
        epochs         : Total training epochs (encoder frozen for first WARMUP_EPOCHS).
        lr             : Initial learning rate for the decoder.
        seg_checkpoint : Path to binary UNet checkpoint (encoder bootstrap).
        save_path      : Where to save the best tissue model.
        metrics_csv    : Where to write per-epoch metrics.
    """
    ensure_dirs()
    device      = CFG.DEVICE
    num_classes = CFG.TISSUE_CLASSES

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[train_tissue] Loading tissue dataloaders …")
    train_loader, val_loader, _ = get_tissue_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────
    print("[train_tissue] Building TissueUNet …")
    model = TissueUNet(num_classes=num_classes, freeze_encoder=True).to(device)

    if seg_checkpoint and Path(seg_checkpoint).exists():
        model.load_encoder_weights(seg_checkpoint)
    else:
        print(
            f"[train_tissue] WARNING: binary checkpoint not found at {seg_checkpoint}. "
            "Training from ImageNet weights only."
        )

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = TissueLoss(
        gamma        = CFG.TISSUE_FOCAL_GAMMA,
        num_classes  = num_classes,
        ignore_index = IGNORE_INDEX,
    )

    # ── Optimiser: only decoder params initially ───────────────────────────
    decoder_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad
    ]
    optimizer  = AdamW(decoder_params, lr=lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── CSV logging ───────────────────────────────────────────────────────
    metrics_csv = Path(metrics_csv)
    fieldnames  = (
        ["epoch", "train_loss", "val_loss", "macro_iou"] +
        [f"iou_{cls}" for cls in CFG.TISSUE_NAMES]
    )
    csv_file    = open(metrics_csv, "w", newline="")
    writer      = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_val_iou  = -1.0
    class_labels  = CFG.TISSUE_NAMES

    print(f"\n[train_tissue] Starting training for {epochs} epochs …\n")
    print(f"  Warmup (encoder frozen): first {WARMUP_EPOCHS} epochs")
    print(f"  Classes: {class_labels}")
    print(f"  Device:  {device}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Unfreeze encoder after warmup
        if epoch == WARMUP_EPOCHS + 1:
            print(f"\n[train_tissue] Epoch {epoch}: Unfreezing encoder, lowering LR.\n")
            for p in model.parameters():
                p.requires_grad = True
            optimizer.add_param_group(
                {"params": [p for p in model.parameters() if not any(
                    p is q for q in decoder_params
                )], "lr": lr * 0.1}
            )

        train_loss, train_iou, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, num_classes, train=True
        )
        val_loss, val_iou, per_class = _run_epoch(
            model, val_loader, criterion, None, device, num_classes, train=False
        )

        scheduler.step()
        elapsed = time.time() - t0

        row: Dict = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "macro_iou": val_iou}
        for cls, iou in zip(class_labels, per_class):
            row[f"iou_{cls}"] = round(iou, 4) if not np.isnan(iou) else None
        writer.writerow(row)
        csv_file.flush()

        class_str = "  ".join(
            f"{cls}={iou:.3f}" if not np.isnan(iou) else f"{cls}=N/A"
            for cls, iou in zip(class_labels, per_class)
        )
        print(
            f"Epoch [{epoch:>3}/{epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"macro_IoU={val_iou:.4f}  [{elapsed:.1f}s]"
        )
        print(f"         per-class: {class_str}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_path)
            print(f"         ✓ Best model saved → {save_path}")

    csv_file.close()
    print(f"\n[train_tissue] Done. Best macro-IoU: {best_val_iou:.4f}")
    print(f"[train_tissue] Metrics saved  → {metrics_csv}")
    print(f"[train_tissue] Best model     → {save_path}")


if __name__ == "__main__":
    train_tissue()
