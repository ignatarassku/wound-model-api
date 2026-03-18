"""
measure.py — Wound geometry measurements from a binary segmentation mask.

Computes:
  - Bounding box width and height (px and cm)
  - Wound area (px and cm²)
  - Wound perimeter (px and cm) via largest contour
  - Tissue breakdown percentages (when a tissue class map is provided)

All real-world conversions require a px_per_mm scale factor produced by
nail_detector.py. If scale is unknown, cm values are returned as None.

Usage:
    from src.measure import compute_measurements, compute_tissue_breakdown
    result = compute_measurements(binary_mask, px_per_mm=12.3)
    breakdown = compute_tissue_breakdown(tissue_class_map, binary_mask)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class WoundMeasurements:
    """All geometric measurements of a wound mask."""

    # Pixel-space (always available)
    wound_area_px:       int
    wound_width_px:      int
    wound_height_px:     int
    wound_perimeter_px:  float

    # Bounding box in pixel coordinates (always available, 0 when mask empty)
    bbox_x_min:          int
    bbox_y_min:          int
    bbox_x_max:          int
    bbox_y_max:          int

    # Real-world (None when scale is unknown)
    wound_area_cm2:      Optional[float]
    wound_width_cm:      Optional[float]
    wound_height_cm:     Optional[float]
    wound_perimeter_cm:  Optional[float]

    # Scale metadata
    scale_px_per_mm:     Optional[float]
    scale_source:        str             # "fingernail" | "fallback_assumed" | "unknown"

    # Tissue breakdown (populated when a TissueUNet prediction is available)
    # keys: tissue class names (e.g. "granulation"), values: fraction [0–1]
    tissue_breakdown:    Dict[str, float] = field(default_factory=dict)


def compute_measurements(
    binary_mask: np.ndarray,
    px_per_mm:   Optional[float] = None,
    scale_source: str = "unknown",
) -> WoundMeasurements:
    """
    Derive geometric measurements from a binary segmentation mask.

    Args:
        binary_mask  : 2-D uint8 array with values 0 or 1 (shape [H, W]).
        px_per_mm    : Pixels per millimetre from nail_detector (or None).
        scale_source : Where the scale came from ("fingernail", "fallback_assumed",
                       or "unknown").

    Returns:
        WoundMeasurements dataclass with px and cm fields.
    """
    mask_u8 = (binary_mask * 255).astype(np.uint8)

    # ── Area ──────────────────────────────────────────────────────────────────
    wound_area_px = int(binary_mask.sum())

    # ── Bounding box via moments / non-zero coords ────────────────────────────
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        # Empty mask — return zeros
        return WoundMeasurements(
            wound_area_px=0,
            wound_width_px=0,
            wound_height_px=0,
            wound_perimeter_px=0.0,
            bbox_x_min=0,
            bbox_y_min=0,
            bbox_x_max=0,
            bbox_y_max=0,
            wound_area_cm2=None,
            wound_width_cm=None,
            wound_height_cm=None,
            wound_perimeter_cm=None,
            scale_px_per_mm=px_per_mm,
            scale_source=scale_source,
        )

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    wound_width_px  = x_max - x_min + 1
    wound_height_px = y_max - y_min + 1

    # ── Perimeter via largest external contour ────────────────────────────────
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        wound_perimeter_px = float(cv2.arcLength(largest, closed=True))
    else:
        wound_perimeter_px = 0.0

    # ── Real-world conversion ─────────────────────────────────────────────────
    if px_per_mm is not None and px_per_mm > 0:
        mm_per_px      = 1.0 / px_per_mm
        cm_per_px      = mm_per_px / 10.0
        cm2_per_px2    = cm_per_px ** 2

        wound_area_cm2     = round(wound_area_px  * cm2_per_px2,  2)
        wound_width_cm     = round(wound_width_px  * cm_per_px,   2)
        wound_height_cm    = round(wound_height_px * cm_per_px,   2)
        wound_perimeter_cm = round(wound_perimeter_px * cm_per_px, 2)
    else:
        wound_area_cm2     = None
        wound_width_cm     = None
        wound_height_cm    = None
        wound_perimeter_cm = None

    return WoundMeasurements(
        wound_area_px=wound_area_px,
        wound_width_px=wound_width_px,
        wound_height_px=wound_height_px,
        wound_perimeter_px=round(wound_perimeter_px, 1),
        bbox_x_min=x_min,
        bbox_y_min=y_min,
        bbox_x_max=x_max,
        bbox_y_max=y_max,
        wound_area_cm2=wound_area_cm2,
        wound_width_cm=wound_width_cm,
        wound_height_cm=wound_height_cm,
        wound_perimeter_cm=wound_perimeter_cm,
        scale_px_per_mm=round(px_per_mm, 4) if px_per_mm is not None else None,
        scale_source=scale_source,
    )


def compute_tissue_breakdown(
    tissue_class_map: np.ndarray,
    wound_mask:       Optional[np.ndarray] = None,
    class_names:      Optional[List[str]]  = None,
) -> Dict[str, float]:
    """
    Compute the fractional area of each tissue type within the wound.

    Args:
        tissue_class_map : 2-D int array [H, W] — class index per pixel (0–N).
                           Values equal to IGNORE_INDEX (255) are excluded.
        wound_mask       : Optional binary 2-D array [H, W]. If supplied, only
                           pixels inside the wound are counted; otherwise all
                           non-ignored pixels are used.
        class_names      : Ordered list of tissue class names.
                           Defaults to ["granulation", "slough", "eschar",
                           "epithelialisation"].

    Returns:
        Dict mapping class name → fraction of wound area [0.0 – 1.0].
        If the wound area is zero, all fractions are 0.0.
    """
    if class_names is None:
        class_names = ["granulation", "slough", "eschar", "epithelialisation"]

    ignore_val  = 255
    valid_mask  = tissue_class_map != ignore_val

    if wound_mask is not None:
        valid_mask = valid_mask & (wound_mask > 0)

    total_valid = valid_mask.sum()
    breakdown: Dict[str, float] = {}

    for cls_idx, name in enumerate(class_names):
        if total_valid == 0:
            breakdown[name] = 0.0
        else:
            count = int(((tissue_class_map == cls_idx) & valid_mask).sum())
            breakdown[name] = round(count / total_valid, 4)

    return breakdown
