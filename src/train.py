"""
train.py — Main training script.

Run from the project root:
    python -m src.train

What this does:
    1. Seeds everything for reproducibility
    2. Creates data loaders (train / val / test)
    3. Initialises U-Net with pretrained ResNet34 encoder
    4. Trains for CFG.EPOCHS with BCE+Dice loss
    5. Validates each epoch — prints ALL metrics from PDF
    6. Saves best checkpoint (by IoU) and last checkpoint
    7. Saves metrics history to CSV
"""

import csv
import random
import numpy as np
import torch
from pathlib import Path

from src.config import CFG, ensure_dirs
from src.dataset import get_dataloaders
from src.model import get_model
from src.losses import BCEDiceLoss
from src.metrics import compute_metrics, print_metrics, collect_predictions


# ── Reproducibility ───────────────────────────────────────────────────────

def set_seeds(seed: int = CFG.RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── CSV logger ────────────────────────────────────────────────────────────

def init_csv(path: Path) -> None:
    """Create metrics CSV with header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss",
            "IoU", "Dice", "F1", "Precision", "Recall", "Specificity",
            "Accuracy", "MCC", "ROC_AUC", "PR_AUC", "FPR", "FNR",
            "Kappa", "TP", "TN", "FP", "FN",
        ])


def append_csv(path: Path, epoch: int, train_loss: float, metrics: dict) -> None:
    """Append one epoch of results to the CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, round(train_loss, 6),
            metrics["IoU"],    metrics["Dice"],    metrics["F1"],
            metrics["Precision"], metrics["Recall"], metrics["Specificity"],
            metrics["Accuracy"], metrics["MCC"],   metrics["ROC_AUC"],
            metrics["PR_AUC"], metrics["FPR"],     metrics["FNR"],
            metrics["Kappa"],  metrics["TP"],      metrics["TN"],
            metrics["FP"],     metrics["FN"],
        ])


# ── Training loop ─────────────────────────────────────────────────────────

def train() -> None:
    """Full training pipeline."""

    set_seeds()
    ensure_dirs()

    print("\n" + "="*58)
    print("  WOUND SEGMENTATION — U-NET TRAINING")
    print("="*58)
    print(f"  Device   : {CFG.DEVICE}")
    print(f"  Epochs   : {CFG.EPOCHS}")
    print(f"  Batch    : {CFG.BATCH_SIZE}")
    print(f"  LR       : {CFG.LEARNING_RATE}")
    print(f"  Img size : {CFG.IMG_SIZE}×{CFG.IMG_SIZE}")
    print("="*58 + "\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── Model, loss, optimiser, scheduler ─────────────────────────────────
    model     = get_model()
    criterion = BCEDiceLoss(alpha=0.5, beta=0.5)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CFG.LEARNING_RATE,
                                 weight_decay=CFG.WEIGHT_DECAY)
                                 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=CFG.PATIENCE
    )

    init_csv(CFG.METRICS_CSV)

    best_iou           = 0.0
    early_stop_counter = 0

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, CFG.EPOCHS + 1):

        # ---- Training phase ----
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(CFG.DEVICE)
            masks  = masks.to(CFG.DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch}/{CFG.EPOCHS}  "
                      f"Batch {batch_idx+1}/{len(train_loader)}  "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # ---- Validation phase ----
        all_masks, all_probs = collect_predictions(model, val_loader)
        metrics = compute_metrics(all_masks, all_probs)
        print_metrics(metrics, epoch=epoch)
        print(f"  Train Loss: {avg_train_loss:.4f}")

        # ---- Scheduler step (maximise IoU) ----
        scheduler.step(metrics["IoU"])

        # ---- Save metrics to CSV ----
        append_csv(CFG.METRICS_CSV, epoch, avg_train_loss, metrics)

        # ---- Save best checkpoint ----
        if metrics["IoU"] > best_iou:
            best_iou           = metrics["IoU"]
            early_stop_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_iou":    best_iou,
                "metrics":     metrics,
            }, CFG.BEST_MODEL_PATH)
            print(f"\n  ✅ New best model saved! IoU = {best_iou:.4f}\n")
        else:
            early_stop_counter += 1
            if early_stop_counter >= CFG.PATIENCE * 2:
                print(f"\n  ⏹  Early stopping after {epoch} epochs (no improvement for {early_stop_counter} epochs)")
                break

        # ---- Save last checkpoint ----
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "metrics":     metrics,
        }, CFG.LAST_MODEL_PATH)

    # ── Final evaluation on TEST set ──────────────────────────────────────
    print("\n" + "="*58)
    print("  FINAL TEST SET EVALUATION")
    print("="*58)

    checkpoint = torch.load(CFG.BEST_MODEL_PATH, map_location=CFG.DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    test_masks, test_probs = collect_predictions(model, test_loader)
    test_metrics = compute_metrics(test_masks, test_probs)
    print_metrics(test_metrics, epoch=None)

    print(f"\n  Best validation IoU : {best_iou:.4f}")
    print(f"  Test IoU            : {test_metrics['IoU']:.4f}")
    print(f"  Test Dice/F1        : {test_metrics['Dice']:.4f}")
    print(f"  Metrics saved to    : {CFG.METRICS_CSV}")
    print(f"  Best model saved to : {CFG.BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
