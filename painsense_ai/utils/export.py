"""
utils/export.py
────────────────
Report export utilities for PainSense AI.

Supports:
- JSON  : machine-readable structured report
- Markdown / text : human-readable clinical report (saveable as .md or .txt)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.feature_extractor import ClinicalFeatureVector
    from modules.clinical_reasoning import ClinicalAssessment
    from modules.safety_layer import SafetyReport
    from modules.documentation import ClinicalDocumentation


# JSON export

def build_json_report(
    features:   "ClinicalFeatureVector",
    assessment: "ClinicalAssessment",
    safety:     "SafetyReport",
    docs:       "ClinicalDocumentation",
) -> dict:
    """Return a fully structured dict representing the analysis session."""
    return {
        "painsense_ai_report": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "version": "1.0",
        },
        "biomechanics": {
            "frames_analyzed":           features.frames_analyzed,
            "frames_with_detection":     features.frames_with_detection,
            "shoulder_abduction_left":   round(features.shoulder_abduction_left, 1),
            "shoulder_abduction_right":  round(features.shoulder_abduction_right, 1),
            "elbow_flexion_left":        round(features.elbow_flexion_left, 1),
            "elbow_flexion_right":       round(features.elbow_flexion_right, 1),
            "hip_flexion_left":          round(features.hip_flexion_left, 1),
            "hip_flexion_right":         round(features.hip_flexion_right, 1),
            "knee_flexion_left":         round(features.knee_flexion_left, 1),
            "knee_flexion_right":        round(features.knee_flexion_right, 1),
            "rom_deficit_pct":           round(features.rom_deficit_pct, 1),
            "movement_asymmetry_pct":    round(features.movement_asymmetry_pct, 1),
            "velocity_reduction_pct":    round(features.velocity_reduction_pct, 1),
            "guarding_detected":         features.guarding_detected,
            "facial_strain_detected":    features.facial_strain_detected,
            "heuristic_pain_score":      round(features.heuristic_pain_score, 1),
            "affected_side":             features.affected_side,
            "most_restricted_joint":     features.most_restricted_joint,
        },
        "head_face": {
            "head_tilt_deg":             round(features.head_tilt_deg, 1),
            "head_nod_deg":              round(features.head_nod_deg, 1),
            "head_turn_deg":             round(features.head_turn_deg, 1),
            "eye_squeeze_detected":      features.eye_squeeze_detected,
            "mouth_tension_detected":    features.mouth_tension_detected,
            "face_pain_score":           round(features.face_pain_score, 1),
        },
        "clinical_assessment": {
            "mas_score":                 assessment.mas_score,
            "confidence":                assessment.confidence,
            "restriction_level":         assessment.restriction_level,
            "clinical_reasoning":        assessment.clinical_reasoning,
            "differential_diagnoses":    assessment.differential_diagnoses,
            "affected_structures":       assessment.affected_structures,
            "urgent_eval_required":      assessment.urgent_eval_required,
        },
        "safety": {
            "risk_level":                safety.risk_level,
            "urgent":                    safety.urgent,
            "red_flags":                 safety.red_flags,
            "recommendation":            safety.recommendation,
            "refer_immediately":         safety.refer_immediately,
        },
        "documentation": {
            "soap_note":                 docs.soap_note,
            "patient_explanation":       docs.patient_explanation,
            "rehab_suggestions":         docs.rehab_suggestions,
        },
    }


def save_json_report(report: dict) -> str:
    """Write report dict to a temp JSON file and return the path."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_painsense_{ts}.json",
        delete=False,
        encoding="utf-8",
    )
    json.dump(report, tmp, indent=2, ensure_ascii=False)
    tmp.close()
    return tmp.name


# Markdown / plain-text report

def build_text_report(
    features:   "ClinicalFeatureVector",
    assessment: "ClinicalAssessment",
    safety:     "SafetyReport",
    docs:       "ClinicalDocumentation",
) -> str:
    """Return a rich Markdown/plain-text clinical report."""
    ts   = datetime.now().strftime("%d %b %Y, %H:%M")
    dds  = "\n".join(f"  - {d}" for d in assessment.differential_diagnoses) or "  - Not determined"
    flags = "\n".join(f"  - {f}" for f in safety.red_flags) or "  - None identified"
    structs = ", ".join(assessment.affected_structures) or "Not specified"

    return f"""# MoveSense AI — Clinical Movement Report
Generated: {ts}

---

## Movement Assessment Summary

| Parameter             | Value                          |
|-----------------------|--------------------------------|
| MAS Score             | **{assessment.mas_score}/100**      |
| Restriction Level     | **{assessment.restriction_level}**  |
| Confidence            | {assessment.confidence}             |
| Affected Area         | {features.most_restricted_joint} ({features.affected_side} side) |
| Affected Structures   | {structs}                           |

**Clinical Interpretation:**
{assessment.clinical_reasoning}

### Differential Diagnoses
{dds}

---

## Biomechanical Measurements

| Joint                      | Left (°) | Right (°) | Normal (°) |
|----------------------------|----------|-----------|------------|
| Shoulder Abduction         | {features.shoulder_abduction_left:.1f}     | {features.shoulder_abduction_right:.1f}      | 180        |
| Elbow Flexion              | {features.elbow_flexion_left:.1f}     | {features.elbow_flexion_right:.1f}      | 145        |
| Hip Flexion                | {features.hip_flexion_left:.1f}     | {features.hip_flexion_right:.1f}      | 120        |
| Knee Flexion               | {features.knee_flexion_left:.1f}     | {features.knee_flexion_right:.1f}      | 135        |

| Clinical Signal           | Value                   |
|---------------------------|-------------------------|
| ROM Deficit               | {features.rom_deficit_pct:.1f}%               |
| Movement Asymmetry        | {features.movement_asymmetry_pct:.1f}%               |
| Velocity Reduction        | {features.velocity_reduction_pct:.1f}%               |
| Guarding Detected         | {"Yes" if features.guarding_detected else "No"}  |
| Facial Strain Proxy       | {"Yes" if features.facial_strain_detected else "No"} |
| Heuristic Score           | {features.heuristic_pain_score:.0f}/100           |
| Frames Analyzed           | {features.frames_analyzed}                |

---

## Safety Screening

**Risk Level: {safety.risk_level}**{"  ⚠️ URGENT" if safety.urgent else ""}

Red Flags:
{flags}

**Recommendation:** {safety.recommendation}

---

## SOAP Note

{docs.soap_note or "_Not generated._"}

---

## Patient Explanation

{docs.patient_explanation or "_Not generated._"}

---

## Rehabilitation Plan

{docs.rehab_suggestions or "_Not generated._"}

---

> ⚠️ **Disclaimer:** PainSense AI is a research/demonstration tool and does not replace
> professional clinical diagnosis. Always consult a qualified healthcare provider.
"""


def save_text_report(text: str) -> str:
    """Write text report to a temp Markdown file and return the path."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_painsense_{ts}.md",
        delete=False,
        encoding="utf-8",
    )
    tmp.write(text)
    tmp.close()
    return tmp.name
