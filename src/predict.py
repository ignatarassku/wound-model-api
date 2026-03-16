"""
predict.py — Run inference on a single new wound image.

Usage:
    python -m src.predict --image path/to/wound_photo.jpg
    python -m src.predict --image path/to/wound_photo.jpg --threshold 0.4
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import CFG
from src.model import get_model


def load_and_preprocess(image_path: Path) -> tuple:
    """
    Load and preprocess a single image for inference.

    Returns:
        tensor     : Preprocessed tensor [1, 3, H, W]
        orig_image : Original PIL image (for visualisation)
    """
    transform = A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD),
        ToTensorV2(),
    ])

    orig_image = Image.open(image_path).convert("RGB")
    image_np   = np.array(orig_image)
    augmented  = transform(image=image_np)
    tensor     = augmented["image"].unsqueeze(0)  # [1, 3, H, W]

    return tensor, orig_image


def predict_single(image_path: str, threshold: float = CFG.THRESHOLD) -> np.ndarray:
    """
    Predict wound segmentation mask for a single image.

    Args:
        image_path : Path to the wound photo.
        threshold  : Decision boundary (0.5 default).

    Returns:
        Binary mask as numpy array [H, W] — 1 = wound, 0 = background.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load model
    model      = get_model()
    checkpoint = torch.load(CFG.BEST_MODEL_PATH, map_location=CFG.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Preprocess
    tensor, orig_image = load_and_preprocess(image_path)
    tensor = tensor.to(CFG.DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()  # [H, W]

    binary_mask = (prob >= threshold).astype(np.uint8)

    # ── Visualise ─────────────────────────────────────────────────────────
    mean = np.array(CFG.NORM_MEAN)
    std  = np.array(CFG.NORM_STD)

    orig_resized = np.array(orig_image.resize((CFG.IMG_SIZE, CFG.IMG_SIZE))) / 255.0

    # Red overlay on wound region
    overlay = orig_resized.copy()
    overlay[binary_mask == 1, 0] = np.clip(overlay[binary_mask == 1, 0] + 0.45, 0, 1)
    overlay[binary_mask == 1, 1] = overlay[binary_mask == 1, 1] * 0.6
    overlay[binary_mask == 1, 2] = overlay[binary_mask == 1, 2] * 0.6

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(orig_resized)
    axes[0].set_title("Original Image", fontsize=12)

    axes[1].imshow(prob, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"Probability Map (threshold={threshold})", fontsize=12)
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Wound Overlay", fontsize=12)

    for ax in axes:
        ax.axis("off")

    wound_pixels = binary_mask.sum()
    total_pixels = binary_mask.size
    wound_pct    = 100 * wound_pixels / total_pixels

    plt.suptitle(
        f"Wound segmentation  |  {wound_pixels:,} wound pixels ({wound_pct:.1f}% of image)",
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    out_path = CFG.PREDICTIONS_DIR / f"{image_path.stem}_prediction.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Predict] Wound area: {wound_pixels:,} px ({wound_pct:.1f}%)")
    print(f"[Predict] Saved to  : {out_path}")

    return binary_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wound segmentation inference")
    parser.add_argument("--image",     type=str,   required=True, help="Path to wound photo")
    parser.add_argument("--threshold", type=float, default=CFG.THRESHOLD, help="Decision threshold")
    args = parser.parse_args()

    predict_single(args.image, args.threshold)
