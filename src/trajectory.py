"""
trajectory.py — Healing Trajectory Prediction (Feature 3).

Takes a time-series of wound visit measurements and predicts:
  • Trend label  : "healing" | "stable" | "worsening" | "insufficient_data"
  • Closure date : estimated date the wound area reaches 0 cm²  (or None)
  • Weekly rate  : mean area change per week in cm²
  • Confidence   : R² of the linear fit (0–1)

Algorithm
---------
1. Require at least 2 visits.
2. Compute elapsed days from the first visit date for each record.
3. Fit a linear regression: area ~ days  (numpy least-squares).
4. Convert slope to weekly rate (cm²/week).
5. Classify trend based on percentage-change thresholds:
     slope_pct < -HEALING_THRESHOLD    → "healing"
     slope_pct >  WORSENING_THRESHOLD  → "worsening"
     else                               → "stable"
   where slope_pct = weekly_rate / mean_area
6. Extrapolate linear fit to area = 0 → estimated closure date.
   Only when trend == "healing" and the extrapolated date is in the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np


# ── Thresholds (mirror CFG to avoid circular import) ──────────────────────────
HEALING_THRESHOLD   = 0.05   # 5% area decrease per week → healing
WORSENING_THRESHOLD = 0.05   # 5% area increase per week → worsening
MIN_VISITS          = 2      # minimum data points required


# ── Data schemas ──────────────────────────────────────────────────────────────

@dataclass
class VisitRecord:
    """
    One wound assessment visit.

    Fields:
        visit_date       : ISO-8601 date string or datetime.date
        wound_area_cm2   : Wound area in cm² from /analyze (None → excluded)
        tissue_breakdown : Optional tissue fractions dict from /analyze
        push_score       : Optional total PUSH score from /push-score
        notes            : Free-text clinical notes (not used in computation)
    """
    visit_date:       str | date
    wound_area_cm2:   Optional[float]
    tissue_breakdown: Optional[Dict[str, float]] = field(default_factory=dict)
    push_score:       Optional[int]              = None
    notes:            Optional[str]              = None

    def date_obj(self) -> date:
        if isinstance(self.visit_date, date):
            return self.visit_date
        return date.fromisoformat(str(self.visit_date))


@dataclass
class TrajectoryResult:
    """Full healing trajectory prediction output."""
    trend:                str               # "healing" | "stable" | "worsening" | "insufficient_data"
    weekly_rate_cm2:      Optional[float]   # cm² change per week (negative = shrinking)
    weekly_rate_pct:      Optional[float]   # % change relative to mean area per week
    estimated_closure:    Optional[str]     # ISO date of predicted full closure, or None
    weeks_to_closure:     Optional[float]   # float weeks until area = 0, or None
    r_squared:            Optional[float]   # goodness-of-fit (0–1)
    visits_used:          int               # number of valid visits in regression
    first_area_cm2:       Optional[float]   # area at first visit
    latest_area_cm2:      Optional[float]   # area at most recent visit
    total_change_cm2:     Optional[float]   # absolute change from first to last
    total_change_pct:     Optional[float]   # % change from first to last
    interpretation:       str               # plain-language summary for doctor


# ── Core computation ──────────────────────────────────────────────────────────

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def _interpret(
    trend:            str,
    weekly_rate_cm2:  Optional[float],
    weekly_rate_pct:  Optional[float],
    weeks_to_closure: Optional[float],
    first_area:       Optional[float],
    latest_area:      Optional[float],
) -> str:
    if trend == "insufficient_data":
        return "Not enough visits to predict trend. Record at least 2 measurements."
    rate_str = f"{abs(weekly_rate_cm2):.2f} cm²/week" if weekly_rate_cm2 is not None else "unknown rate"
    pct_str  = f"({abs(weekly_rate_pct)*100:.1f}%/week)" if weekly_rate_pct is not None else ""
    if trend == "healing":
        closure_str = (
            f" At this rate, estimated closure in {weeks_to_closure:.1f} weeks."
            if weeks_to_closure and weeks_to_closure > 0 else ""
        )
        return (
            f"Wound is healing — shrinking at {rate_str} {pct_str}.{closure_str} "
            "Continue current treatment."
        )
    if trend == "worsening":
        return (
            f"Wound is worsening — growing at {rate_str} {pct_str}. "
            "Review treatment plan and consider specialist referral."
        )
    return (
        f"Wound is stable — minimal change ({rate_str} {pct_str}). "
        "Monitor and reassess at next visit."
    )


def compute_trend(visits: List[VisitRecord]) -> TrajectoryResult:
    """
    Compute the healing trajectory from a list of visit records.

    Args:
        visits: List of VisitRecord objects, ordered by date (oldest first).
                Records with wound_area_cm2 = None are silently skipped.

    Returns:
        TrajectoryResult with trend label, rates, closure estimate, and
        plain-language interpretation.
    """
    # Filter to records with a valid area measurement, sort by date
    valid = sorted(
        [v for v in visits if v.wound_area_cm2 is not None and v.wound_area_cm2 >= 0],
        key=lambda v: v.date_obj(),
    )

    if len(valid) < MIN_VISITS:
        return TrajectoryResult(
            trend             = "insufficient_data",
            weekly_rate_cm2   = None,
            weekly_rate_pct   = None,
            estimated_closure = None,
            weeks_to_closure  = None,
            r_squared         = None,
            visits_used       = len(valid),
            first_area_cm2    = valid[0].wound_area_cm2 if valid else None,
            latest_area_cm2   = valid[-1].wound_area_cm2 if valid else None,
            total_change_cm2  = None,
            total_change_pct  = None,
            interpretation    = _interpret("insufficient_data", None, None, None, None, None),
        )

    # Build arrays: x = days since first visit, y = area
    t0     = valid[0].date_obj()
    days   = np.array([(v.date_obj() - t0).days for v in valid], dtype=float)
    areas  = np.array([v.wound_area_cm2 for v in valid], dtype=float)

    # Linear regression: area = slope * days + intercept
    A      = np.column_stack([days, np.ones(len(days))])
    result = np.linalg.lstsq(A, areas, rcond=None)
    slope_per_day, intercept = result[0]

    y_pred    = slope_per_day * days + intercept
    r2        = _r_squared(areas, y_pred)

    # Convert slope to per-week
    weekly_rate = slope_per_day * 7.0
    mean_area   = float(areas.mean())
    weekly_pct  = weekly_rate / mean_area if mean_area > 0 else 0.0

    # Classify trend
    if weekly_pct < -HEALING_THRESHOLD:
        trend = "healing"
    elif weekly_pct > WORSENING_THRESHOLD:
        trend = "worsening"
    else:
        trend = "stable"

    # Estimate closure date (only meaningful when healing)
    estimated_closure = None
    weeks_to_closure  = None
    latest_day        = days[-1]
    latest_area       = areas[-1]

    if trend == "healing" and slope_per_day < 0:
        # days_to_zero: solve intercept + slope * d = 0
        days_to_zero = -intercept / slope_per_day
        days_remaining = days_to_zero - latest_day
        if days_remaining > 0:
            closure_date      = valid[-1].date_obj() + timedelta(days=int(days_remaining))
            estimated_closure = closure_date.isoformat()
            weeks_to_closure  = days_remaining / 7.0

    first_area       = float(areas[0])
    total_change     = float(areas[-1]) - first_area
    total_change_pct = total_change / first_area if first_area > 0 else None

    return TrajectoryResult(
        trend             = trend,
        weekly_rate_cm2   = round(weekly_rate, 4),
        weekly_rate_pct   = round(weekly_pct,  4),
        estimated_closure = estimated_closure,
        weeks_to_closure  = round(weeks_to_closure, 2) if weeks_to_closure else None,
        r_squared         = round(r2, 4),
        visits_used       = len(valid),
        first_area_cm2    = round(first_area, 4),
        latest_area_cm2   = round(float(areas[-1]), 4),
        total_change_cm2  = round(total_change, 4),
        total_change_pct  = round(total_change_pct, 4) if total_change_pct is not None else None,
        interpretation    = _interpret(trend, weekly_rate, weekly_pct,
                                       weeks_to_closure, first_area, float(areas[-1])),
    )
