"""
modules/safety_layer.py
────────────────────────
Red-flag detection using MedGemma.

Determines whether the movement data suggests a potentially
serious condition requiring urgent evaluation (fracture,
ligament tear, severe inflammatory process, etc.).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional
from modules.clinical_reasoning import _extract_json_obj

from modules.feature_extractor import ClinicalFeatureVector
from modules.clinical_reasoning import ClinicalAssessment
from modules.medgemma_engine import MedGemmaEngine


@dataclass
class SafetyReport:
    """Output of the red-flag safety check."""
    urgent:            bool        = False
    red_flags:         List[str]   = field(default_factory=list)
    risk_level:        str         = "Low"    # Low / Moderate / High / Critical
    recommendation:    str         = ""
    refer_immediately: bool        = False
    raw_response:      str         = ""


_SAFETY_PROMPT_TEMPLATE = """\
You are a clinical safety specialist reviewing a musculoskeletal movement assessment.

ASSESSMENT SUMMARY:
- Movement Abnormality Score (MAS): {mas_score}/100
- Restriction Level:       {restriction_level}
- Primary affected joint:  {most_restricted_joint}
- Affected side:           {affected_side}
- ROM deficit:             {rom_deficit_pct}%
- Asymmetry:               {movement_asymmetry_pct}%
- Guarding:                {guarding_detected}
- Velocity reduction:      {velocity_reduction_pct}%

Clinical diagnoses considered:
{differential_diagnoses}

SAFETY SCREENING TASK:
Evaluate if any of the following red flags are present:
1. Possible fracture (sudden onset, point tenderness pattern, marked velocity reduction)
2. Ligament or tendon tear (severe ROM loss >70%, acute asymmetry)
3. Severe inflammatory condition (bilateral involvement, marked guarding)
4. Neurovascular compromise (extreme ROM deficit with guarding)
5. Malignancy or systemic condition indicator

Respond ONLY in this exact JSON format, no other text:
{{
  "urgent": <true|false>,
  "red_flags": ["<flag 1>", "<flag 2>"],
  "risk_level": "<Low|Moderate|High|Critical>",
  "recommendation": "<1-2 sentences on immediate action>",
  "refer_immediately": <true|false>
}}
"""


class SafetyChecker:
    """
    Runs MedGemma red-flag safety analysis.

    Usage
    -----
    checker = SafetyChecker(engine)
    report  = checker.check(features, assessment)
    """

    def __init__(self, engine: Optional[MedGemmaEngine] = None):
        self._engine = engine

    def _get_engine(self) -> MedGemmaEngine:
        if self._engine is None:
            self._engine = MedGemmaEngine()
        return self._engine

    def check(
        self,
        features:   ClinicalFeatureVector,
        assessment: ClinicalAssessment,
    ) -> SafetyReport:
        """Run red-flag detection and return a SafetyReport."""
        diag_str = "\n".join(
            f"  - {d}" for d in assessment.differential_diagnoses
        ) or "  - Not determined"

        prompt = _SAFETY_PROMPT_TEMPLATE.format(
            mas_score             = assessment.mas_score,
            restriction_level     = assessment.restriction_level,
            most_restricted_joint = features.most_restricted_joint,
            affected_side         = features.affected_side,
            rom_deficit_pct       = round(features.rom_deficit_pct, 1),
            movement_asymmetry_pct= round(features.movement_asymmetry_pct, 1),
            guarding_detected     = features.guarding_detected,
            velocity_reduction_pct= round(features.velocity_reduction_pct, 1),
            differential_diagnoses= diag_str,
        )

        _SAFETY_SYSTEM = (
            "You are a clinical safety specialist. "
            "Always respond with valid JSON only — no preamble, no markdown."
        )
        raw = self._get_engine().generate(
            prompt,
            max_new_tokens=300,
            temperature=0.1,
            system_prompt=_SAFETY_SYSTEM,
        )
        return self._parse(raw, assessment)

    def _parse(self, raw: str, assessment: ClinicalAssessment) -> SafetyReport:
        report = SafetyReport(raw_response=raw)
        print(f"[SafetyChecker] Raw response ({len(raw)} chars): {raw[:200]!r}")

        data = _extract_json_obj(raw)
        if data is not None:
            try:
                report.urgent           = bool(data.get("urgent", False))
                report.red_flags        = list(data.get("red_flags", []))
                report.risk_level       = str(data.get("risk_level", "Low"))
                report.recommendation   = str(data.get("recommendation", ""))
                report.refer_immediately= bool(data.get("refer_immediately", False))
                print("[SafetyChecker] JSON parsed successfully.")
                return report
            except (ValueError, TypeError) as e:
                print(f"[SafetyChecker] JSON field error: {e}")

        # Fallback – derive from assessment
        report.urgent = assessment.urgent_eval_required or assessment.mas_score >= 75
        report.risk_level = (
            "High"     if assessment.mas_score >= 75 else
            "Moderate" if assessment.mas_score >= 40 else
            "Low"
        )
        report.recommendation = (
            "Recommend immediate clinical evaluation." if report.urgent
            else "Monitor and follow-up if symptoms persist."
        )
        return report
