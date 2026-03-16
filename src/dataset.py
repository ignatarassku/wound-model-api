"""
dataset.py — PyTorch Dataset for wound image segmentation.

Handles:
- Loading image / mask pairs from disk
- Train augmentations (albumentations)
- Validation / test transforms (resize + normalize only)
- Automatic train/val/test splitting

Usage:
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import CFG


# ── Augmentation pipelines ────────────────────────────────────────────────

def get_train_transforms() -> A.Compose:
    """
    Augmentations applied ONLY to the training set.
    Albumentations applies the same transform to both image and mask automatically.
    """
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),          # simulate different lighting/cameras
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.GridDistortion(p=0.2),               # elastic deformation for wound shapes
        A.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """
    Transforms applied to validation and test sets — NO augmentation, only resize + normalize.
    """
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD),
        ToTensorV2(),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────

class WoundDataset(Dataset):
    """
    PyTorch Dataset for wound segmentation.

    Expects:
        images_dir/  → RGB wound photos  (.jpg or .png)
        masks_dir/   → Binary masks       (.png): white=wound(255), black=background(0)

    Filename pairing rule:
        Image  "data/images/001.jpg"  →  Mask "data/masks/001.png"
        (same file stem, mask always .png)

    Args:
        images_dir  : Path to folder containing wound images.
        masks_dir   : Path to folder containing binary mask images.
        transform   : Albumentations transform pipeline to apply.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.transform  = transform

        # Collect all image filenames (support .jpg and .png)
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.image_files) == 0:
            raise FileNotFoundError(
                f"No images found in {self.images_dir}. "
                "Make sure you placed wound photos there."
            )

        print(f"[Dataset] Found {len(self.image_files)} images in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image : Float tensor [3, H, W] — normalized RGB image.
            mask  : Float tensor [1, H, W] — binary mask (0.0 or 1.0).
        """
        img_filename  = self.image_files[idx]
        stem          = Path(img_filename).stem          # filename without extension
        mask_filename = stem + ".png"                    # masks are always .png

        # Load image as RGB numpy array
        img_path = self.images_dir / img_filename
        image    = np.array(Image.open(img_path).convert("RGB"))

        # Load mask as grayscale numpy array
        mask_path = self.masks_dir / mask_filename
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: {mask_path}\n"
                f"Expected mask for image: {img_filename}"
            )
        mask = np.array(Image.open(mask_path).convert("L"))  # grayscale [H, W]

        # Binarize: any non-zero pixel → wound (1)
        mask = (mask > 127).astype(np.float32)

        # Apply transforms (albumentations handles image+mask together)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]                      # tensor [3, H, W]
            mask  = augmented["mask"].unsqueeze(0)          # tensor [1, H, W]

        return image, mask


# ── DataLoader factory ────────────────────────────────────────────────────

def get_dataloaders(
    images_dir: Path = CFG.IMAGES_DIR,
    masks_dir:  Path = CFG.MASKS_DIR,
    batch_size: int  = CFG.BATCH_SIZE,
    seed:       int  = CFG.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train / val / test DataLoaders with proper transforms.

    Split: 70% train, 15% val, 15% test (from config.py).

    Args:
        images_dir : Path to wound images.
        masks_dir  : Path to binary masks.
        batch_size : Samples per batch.
        seed       : Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Full dataset (no transforms yet — assigned per subset below)
    full_dataset = WoundDataset(images_dir, masks_dir, transform=None)
    n            = len(full_dataset)

    # Reproducible shuffle
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    # Split indices
    n_train = int(CFG.TRAIN_SPLIT * n)
    n_val   = int(CFG.VAL_SPLIT   * n)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    print(f"[DataLoader] Split → Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Assign transforms per subset via wrapper
    train_set = _TransformSubset(full_dataset, train_idx, get_train_transforms())
    val_set   = _TransformSubset(full_dataset, val_idx,   get_val_transforms())
    test_set  = _TransformSubset(full_dataset, test_idx,  get_val_transforms())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader


class _TransformSubset(Dataset):
    """Internal helper: applies a specific transform to a subset of indices."""

    def __init__(self, dataset: WoundDataset, indices: list, transform: A.Compose) -> None:
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Temporarily swap transform, get item, restore
        original_transform      = self.dataset.transform
        self.dataset.transform  = self.transform
        sample                  = self.dataset[self.indices[idx]]
        self.dataset.transform  = original_transform
        return sample
