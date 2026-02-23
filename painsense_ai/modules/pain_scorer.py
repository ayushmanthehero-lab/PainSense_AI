"""
modules/pain_scorer.py
──────────────────────────────────────────────────────────────────────────────
Movement Abnormality Score (MAS) — Deterministic biomechanical assessment.

This module detects MOVEMENT ABNORMALITY, not pain directly.

Pain cannot be reliably inferred from pose data alone without labeled clinical
data and supervised training.  Instead, this module measures WHAT IS OBJECTIVE:
  • Range of Motion reduction
  • Bilateral asymmetry
  • Protective guarding posture
  • Trunk forward lean (antalgic posture)
  • Movement velocity reduction
  • Facial expression signals

MedGemma then interprets these findings clinically — it does not score them.

Scoring rubric (max 120 pts, capped at 100):
┌──────────────────────────┬─────────┬──────────────────────────────────┐
│ Factor                   │ Max pts │ Rationale                         │
├──────────────────────────┼─────────┼──────────────────────────────────┤
│ ROM Deficit              │  40     │ Primary biomechanical indicator   │
│ Movement Asymmetry       │  25     │ Side-to-side functional limit     │
│ Trunk Forward Lean       │  20     │ Antalgic / guarding posture       │
│ Guarding Behavior        │  15     │ Protective movement pattern       │
│ Velocity Reduction       │  10     │ Antalgic movement slowing         │
│ Facial Expression Cues   │  10     │ Supplementary behavioural signal  │
└──────────────────────────┴─────────┴──────────────────────────────────┘

MAS Restriction Levels:
   0–24  → Normal           (movement within expected range)
  25–49  → Mild Restriction  (early functional limitation; monitor)
  50–74  → Moderate Restriction (notable impairment; clinical assessment advised)
  75–100 → Severe Restriction  (significant movement abnormality detected)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from modules.feature_extractor import ClinicalFeatureVector
from config import NORMAL_ROM


# Result dataclass

@dataclass
class MASResult:
    """Structured output of the deterministic movement abnormality assessment."""
    mas_score:            int             # 0–100 deterministic score
    restriction_level:    str             # Normal / Mild Restriction / Moderate Restriction / Severe Restriction
    movement_grade:       str             # Normal / Mildly Restricted / Moderately Restricted / Severely Restricted
    contributing_factors: List[str] = field(default_factory=list)
    region_contributions: Dict[str, int] = field(default_factory=dict)

    @property
    def confidence(self) -> str:
        """Rule-based scores always carry High confidence."""
        return "High"

    def summary_line(self) -> str:
        """Single-line summary for embedding in MedGemma prompts."""
        fctrs = (
            "; ".join(self.contributing_factors[:3])
            if self.contributing_factors
            else "No significant movement abnormalities detected"
        )
        return (
            f"MAS={self.mas_score}/100 ({self.restriction_level} — {self.movement_grade}). "
            f"Findings: {fctrs}."
        )


# Backward-compat alias (used by older code if any)
BPIResult = MASResult


# Internal helpers

def _restriction_level(score: int) -> str:
    if score >= 75:
        return "Severe Restriction"
    if score >= 50:
        return "Moderate Restriction"
    if score >= 25:
        return "Mild Restriction"
    return "Normal"


def _movement_grade(score: int) -> str:
    if score >= 75:
        return "Severely Restricted"
    if score >= 50:
        return "Moderately Restricted"
    if score >= 25:
        return "Mildly Restricted"
    return "Normal"


# Main scoring function

def compute_mas(fv: ClinicalFeatureVector) -> MASResult:
    """
    Compute the Movement Abnormality Score (MAS) from a ClinicalFeatureVector.

    Fully deterministic — every point awarded traces to a specific threshold crossing.
    MedGemma is NOT involved here.
    """
    score = 0
    factors: List[str] = []
    region_pts: Dict[str, int] = {}

    # ── 1. ROM Deficit  (0–40 pts) ─────────────────────────────────────────
    rom = fv.rom_deficit_pct
    if rom >= 60:
        pts = 40
        factors.append(f"Severe ROM reduction {rom:.0f}% below normal range")
    elif rom >= 40:
        pts = 30
        factors.append(f"Moderate ROM reduction {rom:.0f}% below normal range")
    elif rom >= 20:
        pts = 20
        factors.append(f"Mild ROM reduction {rom:.0f}% below normal range")
    elif rom >= 10:
        pts = 10
        factors.append(f"Borderline ROM reduction {rom:.0f}%")
    else:
        pts = 0
    score += pts
    if pts:
        region_pts["general"] = pts

    # ── 2. Movement Asymmetry  (0–25 pts) ──────────────────────────────────
    asym = fv.movement_asymmetry_pct
    if asym >= 50:
        pts = 25
        factors.append(f"Severe bilateral asymmetry {asym:.0f}%")
    elif asym >= 30:
        pts = 20
        factors.append(f"Marked bilateral asymmetry {asym:.0f}%")
    elif asym >= 15:
        pts = 12
        factors.append(f"Mild bilateral asymmetry {asym:.0f}%")
    else:
        pts = 0
    score += pts

    # ── 3. Trunk Forward Lean — Antalgic Posture  (0–20 pts) ───────────────
    lean      = fv.trunk_forward_lean_deg
    lean_norm = NORMAL_ROM.get("lumbar_forward_lean", 12)
    if lean > lean_norm + 15:           # > 27° = significant antalgic posture
        pts = 20
        factors.append(f"Significant antalgic forward lean {lean:.1f}° (protective posture)")
        region_pts["lumbar"] = region_pts.get("lumbar", 0) + 20
    elif lean > lean_norm:              # 12–27° = lumbar guarding
        pts = 10
        factors.append(f"Guarding posture: trunk forward lean {lean:.1f}°")
        region_pts["lumbar"] = region_pts.get("lumbar", 0) + 10
    else:
        pts = 0
    score += pts

    # ── 4. Guarding Behavior  (0–15 pts) ───────────────────────────────────
    if fv.guarding_detected:
        score += 15
        factors.append("Protective guarding movement pattern detected")
        region_pts["lumbar"] = region_pts.get("lumbar", 0) + 8

    # ── 5. Velocity Reduction — Movement Slowing  (0–10 pts) ───────────────
    vel = fv.velocity_reduction_pct
    if vel >= 50:
        pts = 10
        factors.append(f"Significant movement slowing {vel:.0f}% below baseline")
    elif vel >= 30:
        pts = 5
        factors.append(f"Reduced movement velocity {vel:.0f}% below baseline")
    else:
        pts = 0
    score += pts

    # ── 6. Facial Expression Signals  (0–10 pts) ───────────────────────────
    face = fv.face_pain_score
    if face >= 60:
        pts = 10
        factors.append(f"Strong facial pain-expression signals {face:.0f}/100")
    elif face >= 35:
        pts = 5
        factors.append(f"Mild facial expression signals {face:.0f}/100")
    else:
        pts = 0
    score += pts

    score = min(100, score)

    return MASResult(
        mas_score=score,
        restriction_level=_restriction_level(score),
        movement_grade=_movement_grade(score),
        contributing_factors=factors,
        region_contributions=region_pts,
    )


# Backward-compat alias
compute_bpi = compute_mas
