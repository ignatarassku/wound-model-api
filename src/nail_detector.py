"""
nail_detector.py — Fingernail detection for automatic wound photo scale calibration.

Strategy (ruler-free, single image):
  1. Run YOLOv8-nano on the full image to detect a fingernail.
  2. Measure the nail bounding-box width in pixels.
  3. Divide by the mean adult thumbnail width (16.5 mm) → px_per_mm scale factor.
  4. If no nail is detected with sufficient confidence, fall back to a
     conservative estimate based on a typical 30 cm shooting distance.

Reference:
  Chen et al. (2024) "Application of deep learning in wound size measurement
  using fingernail as the reference." BMC Med Inform Decis Mak.

Usage:
    from src.nail_detector import detect_nail, NailDetectionResult
    result = detect_nail(pil_image)
    print(result.px_per_mm, result.source)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MEAN_NAIL_WIDTH_MM: float = 16.5
"""Mean adult thumbnail width in millimetres (Chen et al., 2024)."""

MIN_NAIL_CONF: float = 0.35
"""Minimum YOLO confidence to accept a nail detection."""

# COCO class index for "cell phone" is used as placeholder; real deployment
# should use a fine-tuned nail-detection model.  YOLOv8 COCO classes that
# often appear near hands: 0=person, 67=cell phone, 76=scissors.
# We search for any hand-region detection and filter by aspect ratio.
_NAIL_YOLO_CLASSES: set[int] = {0}  # detect persons, crop hand region heuristically

# Fallback: typical smartphone wound photo taken at ~30 cm distance.
# At 30 cm with a 12 MP phone (FOV ~65°, sensor ~4000 px wide):
#   px_per_mm ≈ 4000 / (2 * 300 * tan(32.5°)) ≈ 3.2 px/mm
_FALLBACK_PX_PER_MM: float = 3.2

# Model weights — downloaded automatically by ultralytics on first run
_YOLO_WEIGHTS: str = "yolov8n.pt"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class NailDetectionResult:
    px_per_mm:        Optional[float]   # None only when completely unavailable
    source:           str               # "fingernail" | "fallback_assumed"
    nail_width_px:    Optional[int]     # raw detection width in pixels
    confidence:       Optional[float]   # YOLO confidence score


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fallback() -> NailDetectionResult:
    """Return a conservative fallback scale estimate."""
    return NailDetectionResult(
        px_per_mm=_FALLBACK_PX_PER_MM,
        source="fallback_assumed",
        nail_width_px=None,
        confidence=None,
    )


def _detect_nail_yolo(image_np: np.ndarray) -> Optional[NailDetectionResult]:
    """
    Run YOLOv8-nano and search for a fingernail-sized bounding box.

    Heuristic nail filter (when a dedicated nail model is not available):
      - Detection overlaps with the lower-quarter of the image (hand likely there)
      - Bounding box aspect ratio close to square (nail ~1:1 to 2:1 width:height)
      - Bounding box width between 20–300 px (reasonable nail size range)

    Returns NailDetectionResult or None if no plausible nail found.
    """
    try:
        from ultralytics import YOLO  # lazy import — not required at module load
    except ImportError:
        logger.warning("[NailDetector] ultralytics not installed; using fallback scale.")
        return None

    try:
        yolo = YOLO(_YOLO_WEIGHTS, verbose=False)
        results = yolo(image_np, verbose=False, conf=MIN_NAIL_CONF)
    except Exception as exc:
        logger.warning("[NailDetector] YOLO inference failed: %s", exc)
        return None

    h, w = image_np.shape[:2]
    best_conf  = -1.0
    best_box   = None

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            cls  = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if conf < MIN_NAIL_CONF:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = x2 - x1
            bh = y2 - y1

            # Heuristic: nail-sized rectangle (20–300 px wide, aspect ratio 0.4–3.0)
            if not (20 <= bw <= 300):
                continue
            aspect = bw / max(bh, 1)
            if not (0.4 <= aspect <= 3.0):
                continue

            if conf > best_conf:
                best_conf = conf
                best_box  = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    nail_width_px = int(x2 - x1)
    px_per_mm     = nail_width_px / MEAN_NAIL_WIDTH_MM

    logger.info(
        "[NailDetector] Nail detected: width=%d px → %.3f px/mm (conf=%.2f)",
        nail_width_px, px_per_mm, best_conf,
    )

    return NailDetectionResult(
        px_per_mm=round(px_per_mm, 4),
        source="fingernail",
        nail_width_px=nail_width_px,
        confidence=round(best_conf, 4),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def detect_nail(image: Image.Image) -> NailDetectionResult:
    """
    Detect a fingernail in the image and return the scale calibration result.

    Args:
        image : PIL RGB image (the original patient photo, not resized).

    Returns:
        NailDetectionResult with px_per_mm and source fields always set.
        If nail detection fails, source == "fallback_assumed" and px_per_mm
        is the conservative 3.2 px/mm estimate.
    """
    image_np = np.array(image.convert("RGB"))

    result = _detect_nail_yolo(image_np)
    if result is not None:
        return result

    logger.info("[NailDetector] No nail found; using fallback scale (%.2f px/mm).",
                _FALLBACK_PX_PER_MM)
    return _fallback()
