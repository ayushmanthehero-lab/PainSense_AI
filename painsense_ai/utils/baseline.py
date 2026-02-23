"""
utils/baseline.py
─────────────────
Per-patient baseline comparison for MoveSense AI.

Workflow
────────
1. Patient performs their "normal" movement → click "Record Baseline"
2. `save_baseline(fv)` serialises joint angles to data/baseline.json
3. On subsequent sessions, `load_baseline()` + `compute_deviations(fv, baseline)`
   returns per-joint % deviation so clinicians can track improvement/regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Avoid circular import: import the dataclass type lazily in function signatures
# so this module can be imported before modules.feature_extractor is fully loaded.

_BASELINE_PATH = Path(__file__).parent.parent / "data" / "baseline.json"

# All joint scalars we compare (must match ClinicalFeatureVector field names)
_JOINT_KEYS: list[str] = [
    "shoulder_abduction_left",
    "shoulder_abduction_right",
    "shoulder_flexion_left",
    "shoulder_flexion_right",
    "elbow_flexion_left",
    "elbow_flexion_right",
    "hip_flexion_left",
    "hip_flexion_right",
    "hip_abduction_left",
    "hip_abduction_right",
    "knee_flexion_left",
    "knee_flexion_right",
    "trunk_forward_lean_deg",
    "trunk_lateral_lean_deg",
    "neck_lateral_flexion_deg",
]

# Human-readable labels for the deviation table
_JOINT_LABELS: dict[str, str] = {
    "shoulder_abduction_left":   "Shoulder Abduction — Left",
    "shoulder_abduction_right":  "Shoulder Abduction — Right",
    "shoulder_flexion_left":     "Shoulder Flexion — Left",
    "shoulder_flexion_right":    "Shoulder Flexion — Right",
    "elbow_flexion_left":        "Elbow Flexion — Left",
    "elbow_flexion_right":       "Elbow Flexion — Right",
    "hip_flexion_left":          "Hip Flexion — Left",
    "hip_flexion_right":         "Hip Flexion — Right",
    "hip_abduction_left":        "Hip Abduction — Left",
    "hip_abduction_right":       "Hip Abduction — Right",
    "knee_flexion_left":         "Knee Flexion — Left",
    "knee_flexion_right":        "Knee Flexion — Right",
    "trunk_forward_lean_deg":    "Trunk Forward Lean",
    "trunk_lateral_lean_deg":    "Trunk Lateral Lean",
    "neck_lateral_flexion_deg":  "Neck Lateral Flexion",
}


# Save / load

def save_baseline(fv) -> str:
    """
    Persist joint angles from *fv* (ClinicalFeatureVector) to disk.

    Parameters
    ----------
    fv : ClinicalFeatureVector

    Returns
    -------
    Absolute path to the saved JSON file (for UI confirmation).
    """
    _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    for k in _JOINT_KEYS:
        val = getattr(fv, k, -1.0)
        # Only store detected values (sentinel -1 or 999 = not visible in frame)
        if val > 0 and val < 500:
            data[k] = round(float(val), 2)
    with open(_BASELINE_PATH, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
    return str(_BASELINE_PATH)


def load_baseline() -> Optional[dict]:
    """
    Load persisted baseline from disk.

    Returns
    -------
    dict of {joint_key: degrees_float} or None if no baseline saved yet.
    """
    if not _BASELINE_PATH.exists():
        return None
    with open(_BASELINE_PATH, encoding="utf-8") as fp:
        return json.load(fp)


def baseline_summary() -> str:
    """One-line human summary of the stored baseline, for UI display."""
    bl = load_baseline()
    if bl is None:
        return "No baseline recorded yet."
    n = len(bl)
    path = _BASELINE_PATH
    return f"Baseline stored ({n} joints) — {path.name}"


# Deviation computation

def compute_deviations(fv, baseline: dict) -> dict[str, float]:
    """
    Compare *fv* against *baseline* and return per-joint % deviation.

    Only joints that have a valid baseline value AND were measured in the
    current session are included.

    Parameters
    ----------
    fv       : ClinicalFeatureVector
    baseline : dict from load_baseline()

    Returns
    -------
    dict of {joint_key: deviation_%}   (positive = worse than baseline)
    """
    devs: dict[str, float] = {}
    for k in _JOINT_KEYS:
        base_val = baseline.get(k)
        if base_val is None or base_val <= 0:
            continue
        curr_val = getattr(fv, k, -1.0)
        if curr_val <= 0 or curr_val >= 500:
            continue
        # For lower-limb "lower is better" (wrist/ankle stored as geometric min)
        # we still want absolute deviation irrespective of direction:
        dev_pct = abs(curr_val - base_val) / max(base_val, 1.0) * 100.0
        devs[k] = round(dev_pct, 1)
    return devs


def format_deviation_table(fv, baseline: Optional[dict]) -> str:
    """
    Produce a Markdown table comparing current session to baseline.

    Parameters
    ----------
    fv       : ClinicalFeatureVector
    baseline : dict from load_baseline(), or None

    Returns
    -------
    Markdown string (empty table-hint when no baseline).
    """
    if not baseline:
        return (
            "> ℹ️ **No baseline recorded.** Click **📐 Record Baseline** after a "
            "representative session to enable session-over-session comparison."
        )

    devs = compute_deviations(fv, baseline)
    if not devs:
        return (
            "> ℹ️ Baseline exists but no joints were detected in both sessions. "
            "Ensure full body is visible."
        )

    lines = [
        "### 📐 Deviation from Personal Baseline",
        "",
        "| Joint | Baseline | Current | Deviation |",
        "|---|---|---|---|",
    ]
    for k, dev in sorted(devs.items()):
        label    = _JOINT_LABELS.get(k, k.replace("_", " ").title())
        base_val = baseline.get(k, 0)
        curr_val = getattr(fv, k, 0)
        flag     = "  ⚠️" if dev > 20 else ("  ✅" if dev <= 10 else "")
        lines.append(
            f"| {label} | {base_val:.1f}° | {curr_val:.1f}° | **{dev:.1f}%**{flag} |"
        )

    # Summary sentence
    mean_dev = sum(devs.values()) / len(devs) if devs else 0.0
    status   = "Improved" if mean_dev < 10 else ("Stable" if mean_dev < 20 else "Regressed")
    lines += ["", f"> **Overall**: {mean_dev:.1f}% mean deviation — {status}"]

    return "\n".join(lines)
