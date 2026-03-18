"""
train_classifier.py — Training script for WoundClassifier (Feature 4).

Strategy
--------
1. Load images from data/wound_types/<class_name>/*.jpg|.png
2. Freeze backbone for WARMUP_EPOCHS, then unfreeze for fine-tuning
3. Cross-entropy loss with label smoothing (handles noisy clinical labels)
4. Heavy augmentation: random flips, rotation, colour jitter, cutout
5. Save best checkpoint by validation accuracy

Dataset structure expected:
    data/wound_types/
    ├── diabetic_foot_ulcer/   ← place JPEGs/PNGs here
    ├── venous_leg_ulcer/
    ├── pressure_injury/
    ├── surgical/
    └── burn/

Run:
    python -m src.train_classifier
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

from src.config import CFG, ensure_dirs
from src.classifier import WoundClassifier, get_classifier

WARMUP_EPOCHS = 3
IMG_SIZE      = 224


# ── Dataset ───────────────────────────────────────────────────────────────────

class WoundTypeDataset(Dataset):
    """
    ImageFolder-style dataset for wound type classification.

    Reads images from data/wound_types/<class_name>/ directories.
    Class index is determined by CFG.WOUND_TYPES order.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, root: Path, transform=None) -> None:
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for cls_idx, cls_name in enumerate(CFG.WOUND_TYPES):
            cls_dir = root / cls_name
            if not cls_dir.exists():
                print(f"[WoundTypeDataset] WARNING: class dir not found: {cls_dir}")
                continue
            found = [
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in self.EXTENSIONS
            ]
            self.samples.extend((p, cls_idx) for p in found)
            print(f"[WoundTypeDataset]   {cls_name}: {len(found)} images")

        print(f"[WoundTypeDataset] Total: {len(self.samples)} images | "
              f"{len(CFG.WOUND_TYPES)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(
    batch_size: int = 16,
    val_split:  float = 0.15,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders with augmentation on train split."""

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.10)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD),
    ])

    # Load full dataset with val transforms first to get indices, then wrap
    full = WoundTypeDataset(CFG.WOUND_TYPES_DIR, transform=None)
    if len(full) == 0:
        raise RuntimeError(
            f"No images found in {CFG.WOUND_TYPES_DIR}. "
            "Place wound images in data/wound_types/<class_name>/ subdirectories."
        )

    n_val  = max(1, int(len(full) * val_split))
    n_test = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val - n_test

    gen = torch.Generator().manual_seed(CFG.RANDOM_SEED)
    train_sub, val_sub, test_sub = random_split(full, [n_train, n_val, n_test], generator=gen)

    # Apply transforms per split using a wrapper
    class _TransformWrap(Dataset):
        def __init__(self, subset, tf):
            self.subset = subset
            self.tf     = tf
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, i):
            path, label = self.subset.dataset.samples[self.subset.indices[i]]
            img = Image.open(path).convert("RGB")
            return self.tf(img), label

    train_ds = _TransformWrap(train_sub, train_tf)
    val_ds   = _TransformWrap(val_sub,   val_tf)
    test_ds  = _TransformWrap(test_sub,  val_tf)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def _run_epoch(
    model:     WoundClassifier,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    train:     bool,
) -> Tuple[float, float]:
    """Run one epoch. Returns (mean_loss, accuracy)."""
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)

            if train and optimizer:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_classifier(
    epochs:     int   = CFG.CLASSIFIER_EPOCHS,
    lr:         float = CFG.CLASSIFIER_LR,
    batch_size: int   = 16,
    save_path:  Path  = CFG.CLASSIFIER_PATH,
    metrics_csv: Path = CFG.CLASSIFIER_METRICS,
) -> None:
    ensure_dirs()
    device = CFG.DEVICE

    print("\n[train_classifier] Loading dataset …")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    print("[train_classifier] Building WoundClassifier (ResNet50) …")
    model     = get_classifier(num_classes=len(CFG.WOUND_TYPES))
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Only train FC head during warmup
    decoder_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = AdamW(decoder_params, lr=lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    csv_file   = open(metrics_csv, "w", newline="")
    writer     = csv.DictWriter(csv_file,
                                fieldnames=["epoch", "train_loss", "val_loss",
                                            "train_acc", "val_acc"])
    writer.writeheader()

    best_val_acc = -1.0
    print(f"\n[train_classifier] Training {epochs} epochs | device={device}\n"
          f"  Warmup (backbone frozen): first {WARMUP_EPOCHS} epochs\n"
          f"  Classes: {CFG.WOUND_TYPES}\n")

    for epoch in range(1, epochs + 1):
        if epoch == WARMUP_EPOCHS + 1:
            print(f"\n[train_classifier] Epoch {epoch}: Unfreezing backbone.\n")
            model.unfreeze_all()
            optimizer.add_param_group({
                "params": [p for p in model.parameters()
                           if not any(p is q for q in decoder_params)],
                "lr": lr * 0.1,
            })

        t0 = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion,
                                           optimizer, device, train=True)
        val_loss,   val_acc   = _run_epoch(model, val_loader,   criterion,
                                           None,      device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        writer.writerow({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
        })
        csv_file.flush()

        print(f"Epoch [{epoch:>3}/{epochs}]  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  [{elapsed:.1f}s]")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"         ✓ Best model saved → {save_path}")

    csv_file.close()

    # ── Test set evaluation ────────────────────────────────────────────────
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_loss, test_acc = _run_epoch(model, test_loader, criterion,
                                     None, device, train=False)
    print(f"\n[train_classifier] Done.")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Metrics saved     → {metrics_csv}")
    print(f"  Best model        → {save_path}")


if __name__ == "__main__":
    train_classifier()
