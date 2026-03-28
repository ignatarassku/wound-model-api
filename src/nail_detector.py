"""
nail_detector.py — Fingernail detection for automatic wound photo scale calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MEAN_NAIL_WIDTH_MM: float = 16.5
MIN_NAIL_CONF: float = 0.35
_NAIL_YOLO_CLASSES: set[int] = {0}
_FALLBACK_PX_PER_MM: float = 3.2
_YOLO_WEIGHTS: str = "yolov8n.pt"

_yolo_model = None


def get_yolo_model():
    """Load YOLO once per process; reused across all nail-detection calls."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("[NailDetector] ultralytics not installed; nail YOLO disabled.")
        return None
    _yolo_model = YOLO(_YOLO_WEIGHTS, verbose=False)
    return _yolo_model


@dataclass
class NailDetectionResult:
    px_per_mm: Optional[float]
    source: str
    nail_width_px: Optional[int]
    confidence: Optional[float]


def _fallback() -> NailDetectionResult:
    return NailDetectionResult(
        px_per_mm=_FALLBACK_PX_PER_MM,
        source="fallback_assumed",
        nail_width_px=None,
        confidence=None,
    )


def _detect_nail_yolo(image_np: np.ndarray) -> Optional[NailDetectionResult]:
    yolo = get_yolo_model()
    if yolo is None:
        return None

    try:
        results = yolo(image_np, verbose=False, conf=MIN_NAIL_CONF)
    except Exception as exc:
        logger.warning("[NailDetector] YOLO inference failed: %s", exc)
        return None

    h, w = image_np.shape[:2]
    best_conf = -1.0
    best_box = None

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            conf = float(box.conf[0].item())
            if conf < MIN_NAIL_CONF:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = x2 - x1
            bh = y2 - y1

            if not (20 <= bw <= 300):
                continue
            aspect = bw / max(bh, 1)
            if not (0.4 <= aspect <= 3.0):
                continue

            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    nail_width_px = int(x2 - x1)
    px_per_mm = nail_width_px / MEAN_NAIL_WIDTH_MM

    logger.info(
        "[NailDetector] Nail detected: width=%d px → %.3f px/mm (conf=%.2f)",
        nail_width_px,
        px_per_mm,
        best_conf,
    )

    return NailDetectionResult(
        px_per_mm=round(px_per_mm, 4),
        source="fingernail",
        nail_width_px=nail_width_px,
        confidence=round(best_conf, 4),
    )


def detect_nail(image: Image.Image) -> NailDetectionResult:
    image_np = np.array(image.convert("RGB"))

    result = _detect_nail_yolo(image_np)
    if result is not None:
        return result

    logger.info(
        "[NailDetector] No nail found; using fallback scale (%.2f px/mm).",
        _FALLBACK_PX_PER_MM,
    )
    return _fallback()
