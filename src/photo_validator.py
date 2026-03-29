"""
photo_validator.py — Photo quality validation before wound analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

MIN_SHORT_SIDE_PX: int = 200
BLUR_LAPLACIAN_MIN: float = 30.0
BRIGHTNESS_MIN: float = 35.0
BRIGHTNESS_MAX: float = 225.0
MIN_WOUND_AREA_PCT: float = 0.003


@dataclass
class PhotoQuality:
    passed: bool
    issues: list[str] = field(default_factory=list)
    guidance: str = ""
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    resolution: Optional[str] = None
    wound_area_pct: Optional[float] = None


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


def _check_resolution(image: Image.Image) -> tuple[bool, str, str]:
    w, h = image.size
    short = min(w, h)
    res_str = f"{w}×{h}"
    if short < MIN_SHORT_SIDE_PX:
        return False, "too_low_resolution", res_str
    return True, "", res_str


def _check_blur(image: Image.Image) -> tuple[bool, str, float]:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if variance < BLUR_LAPLACIAN_MIN:
        return False, "too_blurry", variance
    return True, "", variance


def _check_brightness(image: Image.Image) -> tuple[bool, str, float]:
    gray = np.array(image.convert("L"), dtype=np.float32)
    mean = float(gray.mean())
    if mean < BRIGHTNESS_MIN:
        return False, "too_dark", mean
    if mean > BRIGHTNESS_MAX:
        return False, "too_bright", mean
    return True, "", mean


def validate_photo_basic(image: Image.Image) -> PhotoQuality:
    """
    Resolution, blur, and brightness only — no model forward pass.
    """
    issues: list[str] = []
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    resolution: Optional[str] = None

    ok, issue, resolution = _check_resolution(image)
    if not ok:
        issues.append(issue)

    ok, issue, blur_score = _check_blur(image)
    if not ok:
        issues.append(issue)

    ok, issue, brightness = _check_brightness(image)
    if not ok:
        issues.append(issue)

    passed = len(issues) == 0
    guidance = " ".join(_GUIDANCE.get(i, i) for i in issues) if issues else "Photo quality is good."

    return PhotoQuality(
        passed=passed,
        issues=issues,
        guidance=guidance,
        blur_score=round(blur_score, 1) if blur_score is not None else None,
        brightness=round(brightness, 1) if brightness is not None else None,
        resolution=resolution,
        wound_area_pct=None,
    )


def validate_wound_from_probs(probs: np.ndarray, threshold: float = 0.5) -> PhotoQuality:
    """
    Wound-presence check using an existing probability map (same U-Net forward pass as full analysis).
    """
    mask = (probs >= threshold).astype(np.uint8)
    area_pct = float(mask.sum() / mask.size)

    if area_pct < MIN_WOUND_AREA_PCT:
        return PhotoQuality(
            passed=False,
            issues=["wound_not_visible"],
            guidance=_GUIDANCE["wound_not_visible"],
            wound_area_pct=round(area_pct, 4),
        )

    return PhotoQuality(
        passed=True,
        issues=[],
        guidance="Wound visible.",
        wound_area_pct=round(area_pct, 4),
    )


def validate_photo(
    image: Image.Image,
    model=None,
    threshold: float = 0.5,
) -> PhotoQuality:
    """
    Legacy path: basic checks + optional extra model forward pass for wound presence.
    Prefer validate_photo_basic + single U-Net inference + validate_wound_from_probs in main.py.
    """
    basic = validate_photo_basic(image)
    if not basic.passed:
        return basic

    if model is None:
        return basic

    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from src.config import CFG

    transform = A.Compose(
        [
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD),
            ToTensorV2(),
        ]
    )
    tensor = transform(image=np.array(image.convert("RGB")))["image"].unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze().numpy()

    wq = validate_wound_from_probs(probs, threshold)
    if not wq.passed:
        return PhotoQuality(
            passed=False,
            issues=basic.issues + wq.issues,
            guidance=wq.guidance,
            blur_score=basic.blur_score,
            brightness=basic.brightness,
            resolution=basic.resolution,
            wound_area_pct=wq.wound_area_pct,
        )

    return PhotoQuality(
        passed=True,
        issues=[],
        guidance="Photo quality is good.",
        blur_score=basic.blur_score,
        brightness=basic.brightness,
        resolution=basic.resolution,
        wound_area_pct=wq.wound_area_pct,
    )
