"""
modules/documentation.py
─────────────────────────
MedGemma Module 2 — Clinical Documentation.

Generates:
1. SOAP Note          – professional clinical record
2. Patient Explanation – plain-language summary for the patient
3. Rehab Suggestions  – physiotherapy recommendations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.feature_extractor import ClinicalFeatureVector
from modules.clinical_reasoning import ClinicalAssessment
from modules.safety_layer import SafetyReport
from modules.medgemma_engine import MedGemmaEngine


@dataclass
class ClinicalDocumentation:
    """All generated documentation for one patient session."""
    soap_note:           str = ""
    patient_explanation: str = ""
    rehab_suggestions:   str = ""


# Prompt templates

_DOC_SYSTEM_PROMPT = (
    "You are an experienced physiotherapy documentation specialist. "
    "Write clear, professional clinical documents. "
    "Do not include disclaimers or meta-commentary — produce the document content directly."
)

_SOAP_PROMPT = """\
You are a physiotherapy documentation specialist.

Based on the following AI-generated movement assessment, write a professional SOAP note.

ASSESSMENT DATA:
- Movement Abnormality Score (MAS): {mas_score}/100
- Restriction Level: {restriction_level}
- Most restricted joint: {most_restricted_joint}
- Affected side: {affected_side}
- ROM deficit: {rom_deficit_pct}%
- Movement asymmetry: {movement_asymmetry_pct}%
- Guarding behavior: {guarding_detected}
- Velocity reduction: {velocity_reduction_pct}%
- Clinical interpretation: {clinical_reasoning}
- Differential diagnoses: {differential_diagnoses}
- Red flags: {red_flags}
- Risk level: {risk_level}

Write a concise SOAP note with the four sections: Subjective, Objective, Assessment, Plan.
Note: This is an AI-assisted movement analysis — mark accordingly.
"""

_PATIENT_EXPLANATION_PROMPT = """\
You are a compassionate healthcare communicator.

Translate this clinical movement assessment into a clear, simple explanation for the patient.
Avoid medical jargon. Be reassuring but honest.

CLINICAL FINDINGS:
- Movement Abnormality Score: {mas_score}/100
- Restriction Level: {restriction_level}
- Affected area: {most_restricted_joint} ({affected_side} side)
- Clinical interpretation: {clinical_reasoning}
- Key recommendation: {recommendation}

Write 3-4 short paragraphs suitable for a patient with no medical background.
End with clear next steps.
"""

_REHAB_PROMPT = """\
You are an expert physiotherapist.

Based on the following movement assessment, suggest a targeted rehabilitation plan.

FINDINGS:
- Restriction Level: {restriction_level} (MAS {mas_score}/100)
- Most restricted joint: {most_restricted_joint}
- Affected side: {affected_side}
- ROM deficit: {rom_deficit_pct}%
- Asymmetry: {movement_asymmetry_pct}%
- Differential diagnoses: {differential_diagnoses}
- Risk level: {risk_level}

Provide:
1. Short-term goals (1–2 weeks)
2. Recommended exercises (3–5 specific exercises with reps/sets)
3. Activities to avoid
4. Progress indicators
5. When to escalate to a specialist

Keep it practical and safe.
"""


# Generator class

class DocumentationGenerator:
    """
    Uses MedGemma to produce clinical documentation from an assessment.

    Usage
    -----
    gen  = DocumentationGenerator(engine)
    docs = gen.generate(features, assessment, safety_report)
    """

    def __init__(self, engine: Optional[MedGemmaEngine] = None):
        self._engine = engine

    def _get_engine(self) -> MedGemmaEngine:
        if self._engine is None:
            self._engine = MedGemmaEngine()
        return self._engine

    def generate(
        self,
        features:   ClinicalFeatureVector,
        assessment: ClinicalAssessment,
        safety:     SafetyReport,
    ) -> ClinicalDocumentation:
        """Generate all three documentation types sequentially."""
        docs = ClinicalDocumentation()

        common = dict(
            mas_score             = assessment.mas_score,
            restriction_level     = assessment.restriction_level,
            most_restricted_joint = features.most_restricted_joint,
            affected_side         = features.affected_side,
            rom_deficit_pct       = round(features.rom_deficit_pct, 1),
            movement_asymmetry_pct= round(features.movement_asymmetry_pct, 1),
            guarding_detected     = features.guarding_detected,
            velocity_reduction_pct= round(features.velocity_reduction_pct, 1),
            clinical_reasoning    = assessment.clinical_reasoning,
            differential_diagnoses= ", ".join(assessment.differential_diagnoses) or "Not determined",
            red_flags             = ", ".join(safety.red_flags) or "None identified",
            risk_level            = safety.risk_level,
            recommendation        = safety.recommendation,
        )

        engine = self._get_engine()

        # Generate each section with system prompt + per-call error handling
        try:
            docs.soap_note = engine.generate(
                _SOAP_PROMPT.format(**common),
                max_new_tokens=500,
                system_prompt=_DOC_SYSTEM_PROMPT,
            )
            print(f"[Documentation] SOAP note: {len(docs.soap_note)} chars")
        except Exception as e:
            print(f"[Documentation] SOAP note error: {e}")
            docs.soap_note = _soap_fallback(common)

        try:
            docs.patient_explanation = engine.generate(
                _PATIENT_EXPLANATION_PROMPT.format(**common),
                max_new_tokens=350,
                system_prompt=_DOC_SYSTEM_PROMPT,
            )
            print(f"[Documentation] Patient explanation: {len(docs.patient_explanation)} chars")
        except Exception as e:
            print(f"[Documentation] Patient explanation error: {e}")
            docs.patient_explanation = _patient_fallback(common)

        try:
            docs.rehab_suggestions = engine.generate(
                _REHAB_PROMPT.format(**common),
                max_new_tokens=450,
                system_prompt=_DOC_SYSTEM_PROMPT,
            )
            print(f"[Documentation] Rehab plan: {len(docs.rehab_suggestions)} chars")
        except Exception as e:
            print(f"[Documentation] Rehab plan error: {e}")
            docs.rehab_suggestions = _rehab_fallback(common)

        # If generated text is empty/minimal, substitute structured fallback
        if len(docs.soap_note.strip()) < 30:
            docs.soap_note = _soap_fallback(common)
        if len(docs.patient_explanation.strip()) < 30:
            docs.patient_explanation = _patient_fallback(common)
        if len(docs.rehab_suggestions.strip()) < 30:
            docs.rehab_suggestions = _rehab_fallback(common)

        return docs


# Structured text fallbacks (used when MedGemma returns empty)

def _soap_fallback(c: dict) -> str:
    return (
        f"**SOAP Note (AI-Assisted)**\n\n"
        f"**S (Subjective):** Patient movement assessed via AI video analysis. "
        f"Reported movement area: {c['most_restricted_joint']} ({c['affected_side']} side).\n\n"
        f"**O (Objective):** "
        f"ROM deficit {c['rom_deficit_pct']}%, Movement asymmetry {c['movement_asymmetry_pct']}%, "
        f"Velocity reduction {c['velocity_reduction_pct']}%, "
        f"Guarding: {c['guarding_detected']}.\n\n"
        f"**A (Assessment):** MAS {c['mas_score']}/100 — {c['restriction_level']}. "
        f"{c['clinical_reasoning']}\n\n"
        f"**P (Plan):** {c['recommendation']} "
        f"Red flags: {c['red_flags']}. Risk: {c['risk_level']}."
    )


def _patient_fallback(c: dict) -> str:
    level = c['restriction_level'].lower()
    area  = c['most_restricted_joint'].replace('_', ' ')
    return (
        f"Based on our AI movement analysis, your assessment shows **{level}** "
        f"in the **{area}** area ({c['affected_side']} side).\n\n"
        f"The analysis found a movement range deficit of {c['rom_deficit_pct']}% compared to "
        f"normal values. This means your {area} is not moving through its full range.\n\n"
        f"**What this means for you:** {c['clinical_reasoning']}\n\n"
        f"**Next steps:** {c['recommendation']} "
        f"Please follow up with your healthcare provider for a full clinical assessment."
    )


def _rehab_fallback(c: dict) -> str:
    area = c['most_restricted_joint'].replace('_', ' ')
    return (
        f"**Rehabilitation Plan — {area.title()} ({c['restriction_level']})**\n\n"
        f"**Short-term goals (1–2 weeks):**\n"
        f"- Restore movement range by 15–20% toward normal\n"
        f"- Reduce compensatory guarding patterns\n\n"
        f"**Recommended exercises:**\n"
        f"1. Gentle range-of-motion circles — 10 reps × 3 sets\n"
        f"2. Pendulum swings (gravity-assisted) — 30 sec × 3\n"
        f"3. Isometric resistance holds — 5 sec × 10 reps\n"
        f"4. Postural correction stretches — 20 sec × 3 sets\n\n"
        f"**Activities to avoid:** High-impact loading, overhead lifting, sustained static postures.\n\n"
        f"**Progress indicators:** Pain reduction, increased ROM, improved symmetry score.\n\n"
        f"**Escalate if:** Symptoms worsen, neurological symptoms appear, or no improvement after 2 weeks."
    )
