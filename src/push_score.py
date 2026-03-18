"""
push_score.py — NPUAP PUSH Tool (Pressure Ulcer Scale for Healing).

PUSH score measures wound healing progress on a 0–17 scale (0 = healed).
It is the clinical standard for pressure ulcer documentation worldwide.

Three sub-scores:
    A  Surface area  (auto, from wound measurement)     0–10
    B  Exudate level (manual, entered by doctor)        0–3
    C  Tissue type   (auto, from tissue classification) 0–4

    Total PUSH = A + B + C

Reference: Thomas et al. (1997), Adv Wound Care 10(5):96–101.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ── NPUAP area lookup table ────────────────────────────────────────────────────
# Boundaries are inclusive on the lower end, exclusive on the upper end.
# (0, 0.1) → 0 pts  means area == 0 → 0 pts

_AREA_BREAKPOINTS: list[tuple[float, float, int]] = [
    (0.0,   0.0,   0),   # closed wound
    (0.0,   0.3,   1),
    (0.3,   0.6,   2),
    (0.6,   1.0,   3),
    (1.0,   2.0,   4),
    (2.0,   3.0,   5),
    (3.0,   4.0,   6),
    (4.0,   8.0,   7),
    (8.0,  12.0,   8),
    (12.0, 24.0,   9),
]
_AREA_MAX_SCORE = 10   # > 24 cm²

# Exudate level → score mapping (for validation / display)
EXUDATE_LEVELS: Dict[str, int] = {
    "none":     0,
    "light":    1,
    "moderate": 2,
    "heavy":    3,
}

# PUSH total → clinical interpretation
_PUSH_INTERPRETATION: list[tuple[int, int, str]] = [
    (0,  0,  "Healed — wound is closed."),
    (1,  5,  "Healing well — continue current treatment protocol."),
    (6,  10, "Moderate wound — monitor closely, review treatment plan."),
    (11, 14, "Significant wound — consider specialist review."),
    (15, 17, "Severe wound — urgent clinical intervention recommended."),
]


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class PushResult:
    """Full PUSH score result with all sub-scores and interpretation."""
    area_score:     int             # Sub-score A (0–10)
    exudate_score:  int             # Sub-score B (0–3)
    tissue_score:   int             # Sub-score C (0–4)
    total_score:    int             # Sum (0–17)
    interpretation: str             # Plain-language clinical summary
    wound_area_cm2: Optional[float] # The area used for scoring
    tissue_dominant: Optional[str]  # Dominant tissue class name (for display)


# ── Sub-score A: Surface area ──────────────────────────────────────────────────

def compute_push_area_score(wound_area_cm2: Optional[float]) -> int:
    """
    Return PUSH area sub-score (0–10) from wound area in cm².

    Args:
        wound_area_cm2: Wound area from geometric measurement. None → score 0.

    Returns:
        Integer score 0–10.
    """
    if wound_area_cm2 is None or wound_area_cm2 <= 0.0:
        return 0
    if wound_area_cm2 > 24.0:
        return _AREA_MAX_SCORE
    for lo, hi, pts in reversed(_AREA_BREAKPOINTS):
        if wound_area_cm2 >= lo and (hi == 0.0 or wound_area_cm2 < hi or hi == 24.0):
            # Special-case: exact 0 → 0
            if wound_area_cm2 == 0.0:
                return 0
            return pts
    return 0


def compute_push_area_score_v2(wound_area_cm2: Optional[float]) -> int:
    """
    Clean lookup — simpler implementation used internally.
    Kept separate so original docstring stays readable.
    """
    if wound_area_cm2 is None or wound_area_cm2 <= 0.0:
        return 0
    a = wound_area_cm2
    if a > 24.0:  return 10
    if a > 12.0:  return 9
    if a > 8.0:   return 8
    if a > 4.0:   return 7
    if a > 3.0:   return 6
    if a > 2.0:   return 5
    if a > 1.0:   return 4
    if a > 0.6:   return 3
    if a > 0.3:   return 2
    if a > 0.0:   return 1
    return 0


# ── Sub-score B: Exudate ──────────────────────────────────────────────────────

def compute_push_exudate_score(exudate_level: str) -> int:
    """
    Return PUSH exudate sub-score (0–3) from a string level.

    Args:
        exudate_level: One of "none", "light", "moderate", "heavy"
                       (case-insensitive). Unknown values → 0.

    Returns:
        Integer score 0–3.
    """
    return EXUDATE_LEVELS.get(exudate_level.strip().lower(), 0)


# ── Sub-score C: Tissue type ──────────────────────────────────────────────────

def compute_push_tissue_score(tissue_breakdown: Optional[Dict[str, float]]) -> tuple[int, Optional[str]]:
    """
    Return PUSH tissue sub-score (0–4) from tissue classification output.

    NPUAP tissue scoring uses the *worst* (highest-score) tissue present
    with a significant fraction (≥ 5%). This is the standard clinical rule:
    even a small amount of necrotic tissue drives the score up.

    Score mapping:
        4 — eschar          (black/hard necrosis — worst)
        3 — slough          (yellow necrosis)
        2 — granulation     (healthy red tissue)
        1 — epithelialisation (healing skin)
        0 — closed / no tissue detected

    Args:
        tissue_breakdown: Dict of {class_name: fraction [0–1]} from
                          compute_tissue_breakdown(). None → score 0.

    Returns:
        (tissue_score: int, dominant_tissue: str | None)
    """
    if not tissue_breakdown:
        return 0, None

    THRESHOLD = 0.05   # class must cover at least 5% of wound to count

    # Ordered worst → best; first match above threshold sets the score
    priority: list[tuple[str, int]] = [
        ("eschar",            4),
        ("slough",            3),
        ("granulation",       2),
        ("epithelialisation", 1),
    ]

    for cls_name, score in priority:
        if tissue_breakdown.get(cls_name, 0.0) >= THRESHOLD:
            return score, cls_name

    return 0, None


# ── Total PUSH ────────────────────────────────────────────────────────────────

def _interpret(total: int) -> str:
    for lo, hi, text in _PUSH_INTERPRETATION:
        if lo <= total <= hi:
            return text
    return "Score out of expected range."


def compute_push_total(
    wound_area_cm2:  Optional[float],
    exudate_level:   str,
    tissue_breakdown: Optional[Dict[str, float]] = None,
) -> PushResult:
    """
    Compute the full PUSH score from all three inputs.

    Args:
        wound_area_cm2:   Wound area (cm²) from /analyze — used for sub-score A.
        exudate_level:    Doctor-entered exudate level string for sub-score B.
        tissue_breakdown: Tissue fractions dict from /analyze for sub-score C.
                          If None, sub-score C = 0.

    Returns:
        PushResult dataclass with all sub-scores, total, and interpretation.
    """
    area_score    = compute_push_area_score_v2(wound_area_cm2)
    exudate_score = compute_push_exudate_score(exudate_level)
    tissue_score, dominant = compute_push_tissue_score(tissue_breakdown)
    total         = area_score + exudate_score + tissue_score

    return PushResult(
        area_score     = area_score,
        exudate_score  = exudate_score,
        tissue_score   = tissue_score,
        total_score    = total,
        interpretation = _interpret(total),
        wound_area_cm2 = wound_area_cm2,
        tissue_dominant = dominant,
    )
