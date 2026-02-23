"""
modules/movement_classifier.py
────────────────────────────────
Determines which body region was primarily tested in a recording.

This drives region-locked prompting: MedGemma only receives data
relevant to the detected region, preventing hallucination across
unrelated anatomical areas.

Region labels
─────────────
  "shoulder"    – shoulder abduction / flexion test
  "elbow_wrist" – elbow/forearm/wrist test
  "hip_knee"    – lower limb mechanics test
  "lumbar"      – lumbar / lower back assessment
  "cervical"    – neck / cervical assessment
  "full_body"   – multiple regions simultaneously active
  "unknown"     – no movement detected
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

__all__ = [
    "RegionClassification",
    "classify_movement_region",
    "REGION_SHOULDER",
    "REGION_ELBOW",
    "REGION_HIP_KNEE",
    "REGION_LUMBAR",
    "REGION_CERVICAL",
    "REGION_FULL",
    "REGION_UNKNOWN",
]

REGION_SHOULDER = "shoulder"
REGION_ELBOW    = "elbow_wrist"
REGION_HIP_KNEE = "hip_knee"
REGION_LUMBAR   = "lumbar"
REGION_CERVICAL = "cervical"
REGION_FULL     = "full_body"
REGION_UNKNOWN  = "unknown"


@dataclass
class RegionClassification:
    region:        str         # primary region label (use REGION_* constants)
    confidence:    float       # 0.0 – 1.0
    active_joints: List[str]   # names of joints that had real movement data
    reason:        str         # human-readable explanation for logging / UI


# Sentinel checks
def _real(v: float) -> bool:
    """True if a joint value is real data: >1° (not -1 sentinel, 999 sentinel, or default 0)."""
    return 1.0 < v < 999.0


def classify_movement_region(fv) -> RegionClassification:
    """
    Determine which body region was primarily exercised in the recording.

    Scoring is additive per region; the region with the highest score wins.
    If ≥3 regions score above 0, the result is classified as full_body.

    Parameters
    ----------
    fv : ClinicalFeatureVector

    Returns
    -------
    RegionClassification
    """
    scores: dict[str, float] = {
        REGION_LUMBAR:   0.0,
        REGION_HIP_KNEE: 0.0,
        REGION_SHOULDER: 0.0,
        REGION_ELBOW:    0.0,
        REGION_CERVICAL: 0.0,
    }
    active: List[str] = []

    # ── Lumbar signals ────────────────────────────────────────────────────────
    _FWD_BASE = 8.0   # anything above this starts contributing
    if fv.trunk_forward_lean_deg > _FWD_BASE:
        # Scales from 0 at 8° up to 60 pts at 20°+
        scores[REGION_LUMBAR] += min(60.0,
            (fv.trunk_forward_lean_deg - _FWD_BASE) / 12.0 * 60)
        active.append("trunk_forward_lean")
    if fv.trunk_lateral_lean_deg > 8.0:
        scores[REGION_LUMBAR] += min(30.0, fv.trunk_lateral_lean_deg / 15.0 * 30)
        active.append("trunk_lateral_lean")
    if fv.guarding_detected:
        scores[REGION_LUMBAR] += 25.0
        if "guarding" not in active:
            active.append("guarding")

    # ── Hip / Knee signals ────────────────────────────────────────────────────
    hip_detected  = any(_real(v) for v in [
        fv.hip_flexion_left, fv.hip_flexion_right,
        fv.hip_abduction_left, fv.hip_abduction_right,
    ])
    knee_detected = any(_real(v) for v in [
        fv.knee_flexion_left, fv.knee_flexion_right,
    ])
    ankle_detected = any(1.0 < v < 999 for v in [
        fv.ankle_dorsiflexion_left, fv.ankle_dorsiflexion_right,
    ])
    if hip_detected:
        scores[REGION_HIP_KNEE] += 45.0
        active.append("hip")
    if knee_detected:
        scores[REGION_HIP_KNEE] += 40.0
        active.append("knee")
    if ankle_detected:
        scores[REGION_HIP_KNEE] += 20.0
        active.append("ankle")

    # ── Shoulder signals ──────────────────────────────────────────────────────
    sh_detected = any(_real(v) for v in [
        fv.shoulder_abduction_left, fv.shoulder_abduction_right,
        fv.shoulder_flexion_left,   fv.shoulder_flexion_right,
    ])
    el_detected = any(_real(v) for v in [
        fv.elbow_flexion_left, fv.elbow_flexion_right,
    ])
    if sh_detected:
        scores[REGION_SHOULDER] += 45.0
        active.append("shoulder")
        if el_detected:
            # Elbow present alongside shoulder → upper limb test (shoulder is primary)
            scores[REGION_SHOULDER] += 15.0

    # ── Elbow / Wrist signals (only when shoulder is NOT the dominant upper limb signal) ─
    wr_detected = any(1.0 < v < 999 for v in [
        fv.wrist_flexion_left, fv.wrist_flexion_right,
    ])
    if el_detected and not sh_detected:
        scores[REGION_ELBOW] += 50.0
        active.append("elbow")
    if wr_detected:
        scores[REGION_ELBOW] += 20.0
        if "wrist" not in active:
            active.append("wrist")

    # ── Cervical signals ──────────────────────────────────────────────────────
    if fv.neck_lateral_flexion_deg > 20.0 or fv.head_tilt_deg > 15.0:
        scores[REGION_CERVICAL] += 40.0
        active.append("cervical")
    if fv.head_nod_deg > 25.0 or fv.head_turn_deg > 25.0:
        scores[REGION_CERVICAL] += 20.0
        if "head_motion" not in active:
            active.append("head_motion")

    # ── Determine winner ──────────────────────────────────────────────────────
    if not active:
        return RegionClassification(
            region=REGION_UNKNOWN, confidence=0.0,
            active_joints=[],
            reason="No movement detected in any joint.",
        )

    ranked = sorted(
        [(r, s) for r, s in scores.items() if s > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    # Three or more distinct regions scoring > 0 → full body
    if len(ranked) >= 3:
        return RegionClassification(
            region=REGION_FULL,
            confidence=0.5,
            active_joints=list(dict.fromkeys(active)),   # preserve order, dedupe
            reason=f"Multiple regions active: {', '.join(r for r, _ in ranked[:3])}",
        )

    primary, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    # Confidence: how much the winner dominates
    conf = min(0.95, 0.5 + (top_score - second_score) / (top_score + 1.0) * 0.5)

    return RegionClassification(
        region=primary,
        confidence=round(conf, 2),
        active_joints=list(dict.fromkeys(active)),
        reason=f"Primary region: {primary} (score {top_score:.0f} vs next {second_score:.0f})",
    )
