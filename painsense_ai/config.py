"""
PainSense AI – Global Configuration
"""
import os

# ── Model paths ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(os.path.dirname(BASE_DIR), "med gemma")  # sibling folder

# ── MedGemma inference settings ───────────────────────────────────────────────
MEDGEMMA_LOAD_IN_4BIT  = True        # 4-bit quant for GTX 1650 (4 GB VRAM)
MEDGEMMA_MAX_NEW_TOKENS = 512
MEDGEMMA_TEMPERATURE    = 0.2        # low temp → deterministic clinical output
DEVICE                  = "cuda"     # change to "cpu" if no GPU

# ── MediaPipe pose settings ───────────────────────────────────────────────────
POSE_MODEL_COMPLEXITY = 1           # 0=Lite, 1=Full, 2=Heavy
POSE_MIN_DETECTION_CONFIDENCE = 0.6
POSE_MIN_TRACKING_CONFIDENCE  = 0.6

# ── Normal ROM (Range of Motion) reference values in degrees ──────────────────
# Source: Standard physiotherapy reference values
# NOTE: wrist_flexion and ankle_dorsiflexion use the MINIMUM angle achieved
#       (elbow→wrist→index and knee→ankle→foot_index respectively).
#       For all other joints the value is the MAXIMUM angle achievable.
NORMAL_ROM = {
    # Upper limb
    "shoulder_abduction":   180,   # max hip→shoulder→wrist (lateral/frontal elevation)
    "shoulder_flexion":     180,   # max hip→shoulder→elbow (forward sagittal elevation)
    "elbow_flexion":        145,   # max shoulder→elbow→wrist
    "wrist_flexion":        100,   # MIN elbow→wrist→index geometric angle (80° clinical ROM → 180-80=100° vertex)
    # Lower limb
    "hip_flexion":          120,   # max anatomic flex (180 − min_geometric)
    "hip_abduction":         40,   # max lateral angle of hip→knee vector from vertical
    "knee_flexion":         135,   # max anatomic flex (180 − min_geometric)
    "ankle_dorsiflexion":   120,   # MIN knee→ankle→foot_index geometric angle (≈neutral dorsiflexed)
    # Spine / neck (reference only — not in main ROM deficit loop)
    "cervical_flexion":      45,
    "lumbar_flexion":        90,
    "trunk_lateral_lean":    15,   # degrees of lateral spine lean before flagging
    "neck_lateral_flexion":  25,   # max normal ear-to-shoulder lateral deviation
    "lumbar_forward_lean":   12,   # degrees of forward trunk lean before flagging (>12° = guarding)
}

# ── Movement Abnormality Score (MAS) thresholds ──────────────────────────────
# MAS is computed deterministically — these thresholds drive UI color only
PAIN_HIGH_THRESHOLD   = 50   # MAS >= 50 → red  (Moderate / Severe Restriction)
PAIN_MEDIUM_THRESHOLD = 25   # MAS >= 25 → amber (Mild Restriction)

# ── Feature weights (used in heuristic pre-score sent to MedGemma) ────────────
FEATURE_WEIGHTS = {
    "rom_deficit_pct":          0.35,
    "movement_asymmetry_pct":   0.25,
    "guarding_detected":        0.20,
    "velocity_reduction_pct":   0.15,
    "facial_strain_detected":   0.05,
}

# ── Gradio UI ─────────────────────────────────────────────────────────────────
UI_TITLE       = "MoveSense AI — Musculoskeletal Movement Assessment"
UI_DESCRIPTION = (
    "AI-assisted musculoskeletal movement assessment using MediaPipe biomechanical analysis "
    "and MedGemma clinical interpretation. Upload a short movement clip — the system scores "
    "movement restriction objectively and provides clinical explanations."
)
UI_SERVER_PORT = 7860
