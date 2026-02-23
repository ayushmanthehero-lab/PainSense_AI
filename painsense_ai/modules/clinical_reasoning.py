"""
modules/clinical_reasoning.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MedGemma Module 1 â€” Pain Probability + Differential Diagnosis.

Builds a structured clinical prompt from the feature vector,
calls MedGemma, and parses the structured response into a
ClinicalAssessment dataclass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from modules.feature_extractor import ClinicalFeatureVector
from modules.medgemma_engine import MedGemmaEngine
from modules.pain_scorer import compute_mas, MASResult
from modules.movement_classifier import (
    classify_movement_region,
    REGION_SHOULDER, REGION_ELBOW, REGION_HIP_KNEE,
    REGION_LUMBAR, REGION_CERVICAL, REGION_FULL, REGION_UNKNOWN,
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Shared JSON extraction helper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _extract_json_obj(text: str) -> Optional[dict]:
    """
    Robustly extract the outermost JSON object from model output.
    Handles:
      - Markdown code fences (```json ... ```)
      - Python True/False/None booleans
      - Trailing commas
      - Extra text before/after the JSON
    """
    # Strip markdown fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    cleaned = cleaned.replace('```', '')

    # Normalise Python-style literals to JSON
    cleaned = re.sub(r'\bTrue\b',  'true',  cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNone\b',  'null',  cleaned)
    # Remove trailing commas before } or ]
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    # Find outermost { ... } using balanced-brace scan
    start = cleaned.find('{')
    if start == -1:
        return None
    depth = 0
    in_str  = False
    escaped = False
    for i, ch in enumerate(cleaned[start:], start):
        if escaped:
            escaped = False
            continue
        if ch == '\\' and in_str:
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    break
    return None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Output dataclass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class ClinicalAssessment:
    """Structured output from MedGemma Module 1."""
    mas_score:           int    = 0       # 0-100 Movement Abnormality Score
    confidence:          str    = "High" # always High (rule-based)
    restriction_level:   str    = "Normal" # Normal / Mild Restriction / Moderate Restriction / Severe Restriction
    region:              str    = "unknown"  # detected/reported anatomical region
    clinical_reasoning:  str    = ""
    differential_diagnoses: List[str] = field(default_factory=list)
    affected_structures: List[str] = field(default_factory=list)
    urgent_eval_required: bool   = False
    raw_response:        str    = ""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Shared system prompt
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_CLINICAL_SYSTEM_PROMPT = (
    "You are an expert musculoskeletal physiotherapist providing clinical interpretation of "
    "objective biomechanical movement data. Always respond with valid JSON only â€” no preamble, no markdown, no "
    "explanation outside the JSON object."
)

_EXPLANATION_SCHEMA = """The Movement Abnormality Score (MAS) above has been OBJECTIVELY DETERMINED by biomechanical rules.
Do NOT re-score, override, or question the MAS score.
Your task: clinically interpret the movement abnormalities found and suggest possible causes.

Respond in this exact JSON format, no other text:
{
  "clinical_reasoning": "<2-4 sentences explaining the biomechanical findings and their clinical significance>",
  "differential_diagnoses": ["<most likely diagnosis>", "<second possibility>", "<third possibility>"],
  "affected_structures": ["<anatomical structure 1>", "<anatomical structure 2>"],
  "urgent_eval_required": <true|false>
}
HARD RULES:
- Explain ONLY the region specified in this prompt. Do NOT infer other regions.
- If data is absent/N/A for a joint, ignore that joint completely.
- Do NOT suggest pathology in anatomical regions not covered by the data above.
- Your interpretation must be consistent with the MAS score and restriction level given."""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Region-locked prompt builders
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _na(v: float, invert: bool = False) -> str:
    """Format float value or 'N/A' for sentinels."""
    if v < 0 or v >= 999:
        return "N/A"
    return f"{v:.1f}Â°"


def _asym(l: float, r: float) -> str:
    if l < 0 or r < 0 or l >= 999 or r >= 999:
        return "N/A"
    avg = (l + r) / 2.0
    if avg < 1e-3:
        return "N/A"
    return f"{abs(l - r) / avg * 100:.1f}%"


def _prompt_shoulder(fv: ClinicalFeatureVector, mas: "MASResult") -> str:
    return f"""\
MOVEMENT ABNORMALITY SCORE (RULE-BASED): {mas.mas_score}/100
Movement Grade: {mas.movement_grade}  |  Restriction Level: {mas.restriction_level}
Key findings: {", ".join(mas.contributing_factors) if mas.contributing_factors else "None identified"}

MOVEMENT TEST: Shoulder abduction and flexion assessment.
You are evaluating a SHOULDER movement test ONLY.
Only explain shoulder, rotator cuff, or upper arm conditions.
Do NOT suggest spine, hip, knee, or lower-body pathology.

SHOULDER DATA (vertical-reference angles; 0Â°=arm at side, 180Â°=fully overhead):
- Left shoulder abduction:   {_na(fv.shoulder_abduction_left)}  (normal max: 180Â°)
- Right shoulder abduction:  {_na(fv.shoulder_abduction_right)} (normal max: 180Â°)
- Left shoulder flexion:     {_na(fv.shoulder_flexion_left)}    (normal max: 180Â°)
- Right shoulder flexion:    {_na(fv.shoulder_flexion_right)}   (normal max: 180Â°)
- Left elbow flexion:        {_na(fv.elbow_flexion_left)}       (normal max: 145Â°)
- Right elbow flexion:       {_na(fv.elbow_flexion_right)}      (normal max: 145Â°)
- Left-right shoulder asymmetry: {_asym(fv.shoulder_abduction_left, fv.shoulder_abduction_right)}

GUARDING / MOVEMENT QUALITY:
- Guarding detected:         {fv.guarding_detected}
- Velocity reduction:        {fv.velocity_reduction_pct:.1f}%
- ROM deficit (shoulder):    {fv.rom_deficit_pct:.1f}%

HARD RULE: If both shoulders > 120° AND asymmetry < 20%, clinical_reasoning must note near-normal range.

{_EXPLANATION_SCHEMA}"""


def _prompt_elbow_wrist(fv: ClinicalFeatureVector, mas: "MASResult") -> str:
    return f"""\
MOVEMENT ABNORMALITY SCORE (RULE-BASED): {mas.mas_score}/100
Movement Grade: {mas.movement_grade}  |  Restriction Level: {mas.restriction_level}
Key findings: {", ".join(mas.contributing_factors) if mas.contributing_factors else "None identified"}

MOVEMENT TEST: Elbow and wrist range-of-motion assessment.
You are evaluating an ELBOW / WRIST movement test ONLY.
Only explain elbow, forearm, or wrist conditions.
Do NOT suggest shoulder, spine, or lower-body pathology.

ELBOW / WRIST DATA:
- Left elbow flexion (anatomic): {_na(fv.elbow_flexion_left)}   (normal max: 145Â°)
- Right elbow flexion (anatomic):{_na(fv.elbow_flexion_right)}  (normal max: 145Â°)
- Left wrist flexion (geometric):{_na(fv.wrist_flexion_left)}   (normal min: 100Â°; lower = more range)
- Right wrist flexion (geometric):{_na(fv.wrist_flexion_right)} (normal min: 100Â°)
- Elbow asymmetry:               {_asym(fv.elbow_flexion_left, fv.elbow_flexion_right)}

MOVEMENT QUALITY:
- ROM deficit:        {fv.rom_deficit_pct:.1f}%
- Asymmetry:          {fv.movement_asymmetry_pct:.1f}%
- Velocity reduction: {fv.velocity_reduction_pct:.1f}%

HARD RULE: If elbow > 100° both sides AND asymmetry < 20%, clinical_reasoning must note near-normal range.

{_EXPLANATION_SCHEMA}"""


def _prompt_hip_knee(fv: ClinicalFeatureVector, mas: "MASResult") -> str:
    return f"""\
MOVEMENT ABNORMALITY SCORE (RULE-BASED): {mas.mas_score}/100
Movement Grade: {mas.movement_grade}  |  Restriction Level: {mas.restriction_level}
Key findings: {", ".join(mas.contributing_factors) if mas.contributing_factors else "None identified"}

MOVEMENT TEST: Lower limb (hip and knee) range-of-motion assessment.
You are evaluating a HIP / KNEE movement test ONLY.
Only explain hip, knee, or lower-limb conditions.
Do NOT suggest lumbar spine, shoulder, or cervical pathology.

LOWER LIMB DATA:
- Left hip flexion (anatomic):   {_na(fv.hip_flexion_left)}     (normal max: 120Â°)
- Right hip flexion (anatomic):  {_na(fv.hip_flexion_right)}    (normal max: 120Â°)
- Left hip abduction:            {_na(fv.hip_abduction_left)}   (normal max: 40Â°; leg angle from vertical)
- Right hip abduction:           {_na(fv.hip_abduction_right)}  (normal max: 40Â°)
- Left knee flexion (anatomic):  {_na(fv.knee_flexion_left)}    (normal max: 135Â°)
- Right knee flexion (anatomic): {_na(fv.knee_flexion_right)}   (normal max: 135Â°)
- Hip asymmetry:                 {_asym(fv.hip_flexion_left, fv.hip_flexion_right)}
- Knee asymmetry:                {_asym(fv.knee_flexion_left, fv.knee_flexion_right)}

MOVEMENT QUALITY:
- ROM deficit:        {fv.rom_deficit_pct:.1f}%
- Asymmetry:          {fv.movement_asymmetry_pct:.1f}%
- Guarding detected:  {fv.guarding_detected}
- Velocity reduction: {fv.velocity_reduction_pct:.1f}%

HARD RULE: If knee > 90Â° both sides AND hip > 80Â° both sides AND asymmetry < 20%,
clinical_reasoning must note near-normal range.

{_EXPLANATION_SCHEMA}"""


def _prompt_lumbar(fv: ClinicalFeatureVector, mas: "MASResult") -> str:
    return f"""\
MOVEMENT ABNORMALITY SCORE (RULE-BASED): {mas.mas_score}/100
Movement Grade: {mas.movement_grade}  |  Restriction Level: {mas.restriction_level}
Key findings: {", ".join(mas.contributing_factors) if mas.contributing_factors else "None identified"}

MOVEMENT TEST: Lumbar spine / lower back assessment.
You are evaluating a LUMBAR / LOWER BACK movement pattern ONLY.
Only explain lumbar, sacroiliac, or lower back conditions.
Do NOT suggest shoulder, knee, or cervical pathology unless clearly stated below.

SPINAL / POSTURAL DATA:
- Trunk forward lean (sagittal): {fv.trunk_forward_lean_deg:.1f}Â°
  (normal <12Â°; 12â€“20Â° = mild guarding; >20Â° = significant antalgic/guarding posture)
- Trunk lateral lean:            {fv.trunk_lateral_lean_deg:.1f}Â°
  (normal <8Â°; >15Â° = lateral scoliotic lean or pain-avoidance)

ASSOCIATED LOWER LIMB (context only):
- Hip flexion left/right:  {_na(fv.hip_flexion_left)} / {_na(fv.hip_flexion_right)}
- Knee flexion left/right: {_na(fv.knee_flexion_left)} / {_na(fv.knee_flexion_right)}

MOVEMENT QUALITY:
- Guarding detected:  {fv.guarding_detected}
- Velocity reduction: {fv.velocity_reduction_pct:.1f}%
- ROM deficit:        {fv.rom_deficit_pct:.1f}%

FACE / PAIN BEHAVIOUR:
- Face pain score:    {fv.face_pain_score:.0f}/100
- Facial strain:      {fv.facial_strain_detected}

HARD RULE: If forward lean < 12Â° AND lateral lean < 8Â° AND no guarding,
clinical_reasoning must confirm posture within normal limits.

{_EXPLANATION_SCHEMA}"""


def _prompt_cervical(fv: ClinicalFeatureVector, mas: "MASResult") -> str:
    return f"""\
MOVEMENT ABNORMALITY SCORE (RULE-BASED): {mas.mas_score}/100
Movement Grade: {mas.movement_grade}  |  Restriction Level: {mas.restriction_level}
Key findings: {", ".join(mas.contributing_factors) if mas.contributing_factors else "None identified"}

MOVEMENT TEST: Cervical spine / neck assessment.
You are evaluating a CERVICAL / NECK movement test ONLY.
Only explain cervical, neck, or upper thoracic conditions.
Do NOT suggest lumbar, shoulder, or lower-body pathology.

CERVICAL / HEAD DATA:
- Neck lateral flexion:     {fv.neck_lateral_flexion_deg:.1f}Â°  (normal <25Â°; ear-to-ear axis tilt)
- Head lateral tilt:        {fv.head_tilt_deg:.1f}Â°             (normal <5Â°)
- Head forward nod ROM:     {fv.head_nod_deg:.1f}Â°              (normal <18Â°; elevated = guarding)
- Head rotation ROM:        {fv.head_turn_deg:.1f}Â°             (normal <20Â°; elevated = cervical restriction)
- Trunk forward lean:       {fv.trunk_forward_lean_deg:.1f}Â°    (context; normal <12Â°)

PAIN BEHAVIOUR:
- Eye squint / grimace:     {fv.eye_squeeze_detected}
- Mouth tension:            {fv.mouth_tension_detected}
- Face pain score:          {fv.face_pain_score:.0f}/100

HARD RULE: If head tilt < 10Â° AND nod ROM < 20Â° AND neck flex < 20Â°,
clinical_reasoning must confirm cervical range within normal limits.

{_EXPLANATION_SCHEMA}"""


def _prompt_full_body(fv: ClinicalFeatureVector, mas: MASResult) -> str:
    """Full-body prompt â€” used only when â‰¥3 regions are simultaneously active."""
    return f"""\
MOVEMENT TEST: Full-body / multi-region assessment.
Multiple body regions show simultaneous movement. Assess the PRIMARY pain source.
Report only the most clinically significant region.

UPPER LIMB:
- Shoulder abd L/R:  {_na(fv.shoulder_abduction_left)} / {_na(fv.shoulder_abduction_right)} (normal 180Â°)
- Shoulder flex L/R: {_na(fv.shoulder_flexion_left)} / {_na(fv.shoulder_flexion_right)}
- Elbow flex L/R:    {_na(fv.elbow_flexion_left)} / {_na(fv.elbow_flexion_right)} (normal 145Â°)

LOWER LIMB:
- Hip flex L/R:      {_na(fv.hip_flexion_left)} / {_na(fv.hip_flexion_right)} (normal 120Â°)
- Knee flex L/R:     {_na(fv.knee_flexion_left)} / {_na(fv.knee_flexion_right)} (normal 135Â°)

SPINE / POSTURE:
- Trunk forward lean:{fv.trunk_forward_lean_deg:.1f}Â°  (normal <12Â°; key guarding signal)
- Trunk lateral lean:{fv.trunk_lateral_lean_deg:.1f}Â°  (normal <8Â°)

CLINICAL SUMMARY:
- ROM deficit:       {fv.rom_deficit_pct:.1f}%
- Asymmetry:         {fv.movement_asymmetry_pct:.1f}%
- Guarding:          {fv.guarding_detected}
- Most restricted:   {fv.most_restricted_joint}
- Face pain score:   {fv.face_pain_score:.0f}/100

HARD RULE: Do NOT diagnose a region unless its data shows clear abnormality.
If data is absent (N/A) for a region, exclude that region from diagnosis.

{_EXPLANATION_SCHEMA}"""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Reasoner
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ClinicalReasoner:
    """
    Hybrid pain assessment engine.

    Pipeline
    --------
    1. Deterministic MAS  — rule-based movement abnormality score (no LLM)
    2. If MAS < 15        — return Normal result immediately (no LLM call)
    3. Region classify    — route to correct body region
    4. Region-locked prompt with MAS as GIVEN FACT -> MedGemma explains only
    5. Parse explanation; force mas_score from deterministic MAS (MedGemma cannot override)
    """

    def __init__(self, engine=None):
        self._engine = engine

    def _get_engine(self):
        if self._engine is None:
            self._engine = MedGemmaEngine()
        return self._engine

    # -- Region-locked prompt builder ----------------------------------------
    @staticmethod
    def _build_prompt(fv: ClinicalFeatureVector, region: str, mas) -> str:
        if region == REGION_SHOULDER:
            return _prompt_shoulder(fv, mas)
        if region == REGION_ELBOW:
            return _prompt_elbow_wrist(fv, mas)
        if region == REGION_HIP_KNEE:
            return _prompt_hip_knee(fv, mas)
        if region == REGION_LUMBAR:
            return _prompt_lumbar(fv, mas)
        if region == REGION_CERVICAL:
            return _prompt_cervical(fv, mas)
        return _prompt_full_body(fv, mas)

    # -- Main assess entry point ----------------------------------------------
    def assess(self, features: ClinicalFeatureVector, symptom_region: str = None) -> "ClinicalAssessment":
        """
        Hybrid pipeline: rule-based MAS determines movement restriction level,
        MedGemma only explains and documents.
        """
        # -- Step 1: Compute Biomechanical Pain Index (deterministic) ---------
        mas = compute_mas(features)
        print(f"[ClinicalReasoner] MAS={mas.mas_score}/100 -- {mas.restriction_level} ({mas.movement_grade})")
        if mas.contributing_factors:
            print(f"[ClinicalReasoner] MAS findings: {'; '.join(mas.contributing_factors)}")

        # -- Step 2: Low MAS -> skip LLM entirely ----------------------------
        if mas.mas_score < 15:
            print("[ClinicalReasoner] MAS < 15 -- movement within normal limits, skipping LLM.")
            return ClinicalAssessment(
                mas_score=mas.mas_score,
                confidence="High",
                restriction_level="Normal",
                region="unknown",
                clinical_reasoning=(
                    f"[MAS={mas.mas_score}/100 -- {mas.restriction_level}] "
                    f"All measured movements are within normal biomechanical range. "
                    f"ROM deficit {features.rom_deficit_pct:.0f}%, asymmetry "
                    f"{features.movement_asymmetry_pct:.0f}%, no guarding, trunk posture normal."
                ),
                differential_diagnoses=[],
                affected_structures=[],
                urgent_eval_required=False,
                raw_response="[mas_gate]",
            )

        # -- Step 3: Classify which region was tested -------------------------
        if symptom_region and symptom_region != 'auto':
            from modules.movement_classifier import RegionClassification
            region_info = RegionClassification(
                region=symptom_region,
                confidence=1.0,
                active_joints=[],
                reason=f'Patient-reported region: {symptom_region}',
            )
            print(f'[ClinicalReasoner] Using patient-reported region: {symptom_region}')
        else:
            region_info = classify_movement_region(features)
            print(f"[ClinicalReasoner] Region: {region_info.region} "
                  f"(conf={region_info.confidence:.2f}) -- {region_info.reason}")

        if features.most_restricted_joint in ("none_detected", "unknown"):
            features.most_restricted_joint = region_info.region

        # -- Step 4: Build region-locked prompt with MAS as GIVEN FACT ------
        prompt = ClinicalReasoner._build_prompt(features, region_info.region, mas)

        # -- Step 5: MedGemma -> explanation only -----------------------------
        engine   = self._get_engine()
        raw_resp = engine.generate(
            prompt,
            max_new_tokens=512,
            temperature=0.1,
            system_prompt=_CLINICAL_SYSTEM_PROMPT,
        )

        result = self._parse_response(raw_resp, features, mas)

        # ALWAYS force score from deterministic MAS -- MedGemma cannot override
        result.mas_score          = mas.mas_score
        result.restriction_level  = mas.restriction_level
        result.confidence         = "High"
        result.region             = region_info.region
        result.clinical_reasoning = (
            f"[MAS={mas.mas_score}/100 -- {mas.restriction_level}] "
            f"[Region: {region_info.region}] "
            + result.clinical_reasoning
        )
        return result

    # -- Response parser (explanation only) ----------------------------------
    def _parse_response(
        self,
        raw: str,
        features: ClinicalFeatureVector,
        mas,
    ) -> "ClinicalAssessment":
        """
        Parse MedGemma explanation JSON.
        mas_score / restriction_level / confidence are NOT extracted from MedGemma --
        they are overwritten by the caller from the deterministic MAS.
        """
        assessment = ClinicalAssessment(
            mas_score=mas.mas_score,
            confidence="High",
            restriction_level=mas.restriction_level,
            region="unknown",
            raw_response=raw,
        )

        print(f"[ClinicalReasoner] Raw response ({len(raw)} chars): {raw[:200]!r}")
        data = _extract_json_obj(raw)
        if data is not None:
            try:
                reasoning = str(data.get("clinical_reasoning", "")).strip()
                if reasoning:
                    assessment.clinical_reasoning     = reasoning
                    assessment.differential_diagnoses = list(data.get("differential_diagnoses", []))
                    assessment.affected_structures    = list(data.get("affected_structures", []))
                    assessment.urgent_eval_required   = bool(data.get("urgent_eval_required", False))
                    print("[ClinicalReasoner] Explanation JSON parsed successfully.")
                    return assessment
            except (ValueError, TypeError) as e:
                print(f"[ClinicalReasoner] JSON field error: {e}")

        # -- Fallback: build explanation from MAS findings -------------------
        print("[ClinicalReasoner] JSON parse failed -- generating explanation from MAS findings.")
        factors_str = (
            "; ".join(mas.contributing_factors)
            if mas.contributing_factors
            else "no significant movement abnormalities identified"
        )
        assessment.clinical_reasoning = (
            f"Movement assessment -- MAS {mas.mas_score}/100 ({mas.restriction_level}): "
            f"{factors_str}. "
            f"ROM deficit {features.rom_deficit_pct:.0f}%, movement asymmetry "
            f"{features.movement_asymmetry_pct:.0f}%."
        )
        return assessment
    # -- Muscle-level anatomy explanation ------------------------------------
    def explain_muscles(
        self,
        features:   "ClinicalFeatureVector",
        region:     str,
        mas,                            # MASResult
        muscles:    list,
    ) -> str:
        """
        Ask MedGemma to explain which muscles are likely involved and why,
        given the region and MAS score.  Used by the Anatomy Map tab.
        """
        if not muscles:
            return _muscle_fallback(region, [], mas)

        muscles_str = "\n".join(f"  - {m}" for m in muscles)
        prompt = (
            f"Movement Test Analysis:\n"
            f"Region: {region.replace('_', ' ').title()}\n"
            f"Movement Abnormality Score (MAS): {mas.mas_score}/100 ({mas.restriction_level})\n"
            f"ROM Deficit: {features.rom_deficit_pct:.0f}%\n"
            f"Bilateral Asymmetry: {features.movement_asymmetry_pct:.0f}%\n"
            f"Guarding Detected: {features.guarding_detected}\n\n"
            f"Muscles potentially involved in this region:\n{muscles_str}\n\n"
            "As an expert musculoskeletal physiotherapist, provide a brief clinical explanation:\n"
            "1. Which specific muscles are most likely contributing to this restriction and why\n"
            "2. Most probable cause given the MAS score and findings\n"
            "3. One clear clinical recommendation\n\n"
            "Be concise (3-4 sentences per point). Use clinical but accessible language."
        )
        try:
            engine = self._get_engine()
            text   = engine.generate(
                prompt,
                max_new_tokens=400,
                temperature=0.15,
                system_prompt=(
                    "You are an expert musculoskeletal physiotherapist. "
                    "Provide concise, clinically accurate muscle-level explanations. "
                    "Do not include disclaimers or meta-commentary."
                ),
            )
            if text and len(text.strip()) > 30:
                return text
        except Exception as e:
            print(f"[ClinicalReasoner] explain_muscles error: {e}")

        return _muscle_fallback(region, muscles, mas)


# Fallback for muscle explanation (no LLM)

def _muscle_fallback(region: str, muscles: list, mas) -> str:
    muscles_list = "\n".join(f"- {m}" for m in muscles) if muscles else "- Not determined"
    area = region.replace("_", " ").title()
    return (
        f"**Likely Muscles Involved — {area}**\n\n"
        f"{muscles_list}\n\n"
        f"**MAS {mas.mas_score}/100 — {mas.restriction_level}:**  \n"
        f"The observed movement restriction in the {area} region suggests increased "
        f"load or biomechanical stress in the above muscles. "
        f"A ROM deficit of {mas.mas_score}% and restriction pattern typical of "
        f"compensatory guarding are consistent with these muscle groups being involved.  \n\n"
        f"**Recommendation:** Clinical evaluation of the {area} region is advised "
        f"to confirm which specific structures are contributing to the restriction."
    )