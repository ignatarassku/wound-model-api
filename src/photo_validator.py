"""
photo_validator.py — Photo quality validation before wound analysis.

Validates each uploaded patient photo across four clinical checks:
  1. Resolution  — image must be large enough to contain useful detail
  2. Blur        — Laplacian variance detects camera shake / out-of-focus
  3. Brightness  — rejects photos that are too dark or overexposed
  4. Wound presence — quick segmentation check; rejects if no wound visible

Returns a PhotoQuality result that the API uses to either:
  - Accept the image → proceed to full analysis
  - Reject the image → HTTP 422 with a patient-friendly guidance message

Usage:
    from src.photo_validator import validate_photo, PhotoQuality
    quality = validate_photo(pil_image, model=loaded_unet)
    if not quality.passed:
        raise HTTPException(422, quality.guidance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ── Thresholds ─────────────────────────────────────────────────────────────────

MIN_SHORT_SIDE_PX:  int   = 200    # shortest image dimension in pixels
BLUR_LAPLACIAN_MIN: float = 80.0   # Laplacian variance below this = too blurry
BRIGHTNESS_MIN:     float = 35.0   # mean pixel value (0–255) below this = too dark
BRIGHTNESS_MAX:     float = 225.0  # mean pixel value above this = overexposed
MIN_WOUND_AREA_PCT: float = 0.003  # wound must be at least 0.3% of the image area


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PhotoQuality:
    """Result of photo quality validation."""
    passed:    bool
    issues:    list[str]            = field(default_factory=list)
    guidance:  str                  = ""
    blur_score: Optional[float]     = None   # Laplacian variance (higher = sharper)
    brightness: Optional[float]     = None   # mean pixel value 0–255
    resolution: Optional[str]       = None   # "WxH" string
    wound_area_pct: Optional[float] = None   # fraction of image that is wound


# ── Individual check functions ─────────────────────────────────────────────────

def _check_resolution(image: Image.Image) -> tuple[bool, str, str]:
    """Returns (ok, issue_key, resolution_str)."""
    w, h = image.size
    short = min(w, h)
    res_str = f"{w}×{h}"
    if short < MIN_SHORT_SIDE_PX:
        return False, "too_low_resolution", res_str
    return True, "", res_str


def _check_blur(image: Image.Image) -> tuple[bool, str, float]:
    """Returns (ok, issue_key, laplacian_variance)."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if variance < BLUR_LAPLACIAN_MIN:
        return False, "too_blurry", variance
    return True, "", variance


def _check_brightness(image: Image.Image) -> tuple[bool, str, float]:
    """Returns (ok, issue_key, mean_brightness)."""
    gray = np.array(image.convert("L"), dtype=np.float32)
    mean = float(gray.mean())
    if mean < BRIGHTNESS_MIN:
        return False, "too_dark", mean
    if mean > BRIGHTNESS_MAX:
        return False, "too_bright", mean
    return True, "", mean


def _check_wound_presence(
    image: Image.Image,
    model,                  # UNet — optional, skipped if None
    threshold: float = 0.5,
) -> tuple[bool, str, Optional[float]]:
    """
    Returns (ok, issue_key, wound_area_pct).
    Runs a quick forward pass to confirm at least MIN_WOUND_AREA_PCT of pixels
    are classified as wound. If model is None, this check is skipped (passes).
    """
    if model is None:
        return True, "", None

    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from src.config import CFG

    transform = A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD),
        ToTensorV2(),
    ])
    tensor = transform(image=np.array(image.convert("RGB")))["image"].unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze().numpy()

    mask      = (probs >= threshold).astype(np.uint8)
    area_pct  = float(mask.sum() / mask.size)

    if area_pct < MIN_WOUND_AREA_PCT:
        return False, "wound_not_visible", area_pct
    return True, "", area_pct


# ── Patient-facing guidance messages ──────────────────────────────────────────

_GUIDANCE: dict[str, str] = {
    "too_low_resolution": (
        "The photo resolution is too low. Please take a closer photo "
        "or use a higher quality camera setting."
    ),
    "too_blurry": (
        "The photo is blurry. Hold your phone steady, tap the screen to focus "
        "on the wound, and retake the photo."
    ),
    "too_dark": (
        "The photo is too dark. Move to a brighter area or turn on a light "
        "and retake the photo."
    ),
    "too_bright": (
        "The photo is overexposed. Avoid direct flash or bright sunlight "
        "and retake the photo."
    ),
    "wound_not_visible": (
        "No wound was detected in the photo. Make sure the wound is clearly "
        "visible and centred in the frame, then retake the photo."
    ),
}


# ── Public API ─────────────────────────────────────────────────────────────────

def validate_photo(
    image:     Image.Image,
    model      = None,
    threshold: float = 0.5,
) -> PhotoQuality:
    """
    Run all quality checks on a PIL image.

    Args:
        image     : PIL RGB image (original resolution, before any resize).
        model     : Optional loaded UNet for wound-presence check. If None,
                    the wound-presence check is skipped.
        threshold : Segmentation threshold for wound-presence check.

    Returns:
        PhotoQuality with passed=True if all checks pass, or passed=False
        with a list of issues and a patient-friendly guidance string.
    """
    issues:     list[str]       = []
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    resolution: Optional[str]   = None
    wound_pct:  Optional[float] = None

    # 1. Resolution
    ok, issue, resolution = _check_resolution(image)
    if not ok:
        issues.append(issue)

    # 2. Blur
    ok, issue, blur_score = _check_blur(image)
    if not ok:
        issues.append(issue)

    # 3. Brightness
    ok, issue, brightness = _check_brightness(image)
    if not ok:
        issues.append(issue)

    # 4. Wound presence (only if no other issues yet — avoids running model on garbage)
    if not issues:
        ok, issue, wound_pct = _check_wound_presence(image, model, threshold)
        if not ok:
            issues.append(issue)

    passed = len(issues) == 0

    # Build a single combined guidance message
    if issues:
        guidance = " ".join(_GUIDANCE.get(i, i) for i in issues)
    else:
        guidance = "Photo quality is good."

    return PhotoQuality(
        passed=passed,
        issues=issues,
        guidance=guidance,
        blur_score=round(blur_score, 1) if blur_score is not None else None,
        brightness=round(brightness, 1) if brightness is not None else None,
        resolution=resolution,
        wound_area_pct=round(wound_pct, 4) if wound_pct is not None else None,
    )
