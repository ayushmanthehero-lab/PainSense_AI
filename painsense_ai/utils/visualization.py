"""
utils/visualization.py
────────────────────────
Helpers for rendering charts and annotated frames used in the UI.
"""

from __future__ import annotations

import io
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from typing import Optional

from modules.feature_extractor import ClinicalFeatureVector
from modules.clinical_reasoning import ClinicalAssessment
from config import PAIN_HIGH_THRESHOLD, PAIN_MEDIUM_THRESHOLD



def pain_gauge_chart(pain_probability: int) -> Image.Image:
    """
    Draw a semicircular gauge showing the pain probability.

    Returns
    -------
    PIL Image (RGB)
    """
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)

    # Background arc
    theta = np.linspace(0, np.pi, 200)
    ax.plot(theta, [1] * 200, color="#e0e0e0", linewidth=20, solid_capstyle="round")

    # Value arc
    pain_theta = np.linspace(0, np.pi * pain_probability / 100, 200)
    color = (
        "#e74c3c" if pain_probability >= PAIN_HIGH_THRESHOLD else
        "#f39c12" if pain_probability >= PAIN_MEDIUM_THRESHOLD else
        "#2ecc71"
    )
    ax.plot(pain_theta, [1] * len(pain_theta), color=color, linewidth=20, solid_capstyle="round")

    ax.set_ylim(0, 1.5)
    ax.set_axis_off()
    ax.text(
        np.pi / 2, 0.3,
        f"{pain_probability}%",
        ha="center", va="center",
        fontsize=28, fontweight="bold", color=color,
    )
    ax.text(np.pi / 2, -0.15, "Pain Likelihood", ha="center", va="center", fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")



def rom_bar_chart(features: ClinicalFeatureVector) -> Image.Image:
    """
    Bar chart comparing measured ROM vs normal ROM for all joints.
    Joints with sentinel values (-1 = not detected, 999 = not detected for
    lower-is-better joints) are omitted so they don't show as misleading 0-bars.
    For "lower is better" joints (wrist, ankle) values are converted to
    'range-from-straight' = 180 - min_angle, so higher bar = more range = better.
    """
    SENTINEL_NEG = -1.0    # landmark never seen (most joints)
    SENTINEL_INF = 999.0   # landmark never seen (wrist / ankle lower-is-better)

    def _ra(val, lower_is_better=False):
        """
        Return (displayable_value, detected) tuple.
        detected=False means the joint was not measured — skip it.
        """
        if lower_is_better:
            if val >= SENTINEL_INF - 1:
                return None, False          # not detected
            return max(0.0, 180.0 - val), True
        else:
            if val < 0:
                return None, False          # not detected
            return max(0.0, val), True

    wf_norm_range = 180 - 80   # = 100° expected range for wrist
    ad_norm_range = 180 - 80   # = 100° expected range for ankle

    def _entry(label, val, normal, lower_is_better=False):
        v, det = _ra(val, lower_is_better)
        if not det:
            return None
        return (label, v, normal)

    upper_raw = [
        _entry("Shldr Abd\n(L)",    features.shoulder_abduction_left,   180),
        _entry("Shldr Abd\n(R)",    features.shoulder_abduction_right,  180),
        _entry("Shldr Flex\n(L)",   features.shoulder_flexion_left,     180),
        _entry("Shldr Flex\n(R)",   features.shoulder_flexion_right,    180),
        _entry("Elbow Flex\n(L)",   features.elbow_flexion_left,        145),
        _entry("Elbow Flex\n(R)",   features.elbow_flexion_right,       145),
        _entry("Wrist Flex\n(L)*",  features.wrist_flexion_left,   wf_norm_range, True),
        _entry("Wrist Flex\n(R)*",  features.wrist_flexion_right,  wf_norm_range, True),
    ]
    lower_raw = [
        _entry("Hip Flex\n(L)",    features.hip_flexion_left,          120),
        _entry("Hip Flex\n(R)",    features.hip_flexion_right,         120),
        _entry("Hip Abd\n(L)",     features.hip_abduction_left,         40),
        _entry("Hip Abd\n(R)",     features.hip_abduction_right,        40),
        _entry("Knee Flex\n(L)",   features.knee_flexion_left,         135),
        _entry("Knee Flex\n(R)",   features.knee_flexion_right,        135),
        _entry("Ankle DF\n(L)*",   features.ankle_dorsiflexion_left,  ad_norm_range, True),
        _entry("Ankle DF\n(R)*",   features.ankle_dorsiflexion_right, ad_norm_range, True),
    ]

    upper_joints = [e for e in upper_raw if e is not None]
    lower_joints = [e for e in lower_raw if e is not None]

    def _draw_panel(ax, joints, title):
        if not joints:
            ax.text(0.5, 0.5, "No data detected for this panel\n"
                    "(ensure full body is visible in video)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9,
                    color="#888888")
            ax.set_title(title, fontsize=9)
            ax.set_axis_off()
            return
        labels   = [j[0] for j in joints]
        measured = [j[1] for j in joints]
        normals  = [j[2] for j in joints]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, normals,  w, label="Normal ROM", color="#3498db", alpha=0.5)
        ax.bar(x + w/2, measured, w, label="Measured ROM", color="#e74c3c", alpha=0.8)
        ax.set_ylabel("Degrees (°)")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 200)
        ax.legend(fontsize=7)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    _draw_panel(ax1, upper_joints, "Upper Limb ROM  (* = 180−min_angle; higher = more range)")
    _draw_panel(ax2, lower_joints, "Lower Limb ROM  (* = 180−min_angle; higher = more range)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")



def feature_radar_chart(features: ClinicalFeatureVector) -> Image.Image:
    """Radar chart covering all 9 clinical signal dimensions derived from all 33 landmarks."""

    # ── Upper limb score: best shoulder + elbow + wrist deficit ──────────────
    sa_best = max(features.shoulder_abduction_left, features.shoulder_abduction_right)
    sf_best = max(features.shoulder_flexion_left,   features.shoulder_flexion_right)
    ef_best = max(features.elbow_flexion_left,       features.elbow_flexion_right)
    wf_best = min(features.wrist_flexion_left,       features.wrist_flexion_right)  # lower=better
    upper_scores = []
    if sa_best > 0:  upper_scores.append(max(0, (180 - sa_best) / 180 * 100))
    if sf_best > 0:  upper_scores.append(max(0, (180 - sf_best) / 180 * 100))
    if ef_best > 0:  upper_scores.append(max(0, (145 - ef_best) / 145 * 100))
    if wf_best < 999: upper_scores.append(max(0, (wf_best - 80) / 80 * 100))
    upper_deficit = float(np.mean(upper_scores)) if upper_scores else 0.0

    # ── Lower limb score: hip + knee + ankle deficit ──────────────────────────
    hf_best = max(features.hip_flexion_left,   features.hip_flexion_right)
    ha_best = max(features.hip_abduction_left, features.hip_abduction_right)
    kf_best = max(features.knee_flexion_left,  features.knee_flexion_right)
    ad_best = min(features.ankle_dorsiflexion_left, features.ankle_dorsiflexion_right)
    lower_scores = []
    if hf_best > 0:  lower_scores.append(max(0, (120 - hf_best) / 120 * 100))
    if ha_best > 0:  lower_scores.append(max(0, (40  - ha_best) / 40  * 100))
    if kf_best > 0:  lower_scores.append(max(0, (135 - kf_best) / 135 * 100))
    if ad_best < 999: lower_scores.append(max(0, (ad_best - 80) / 80  * 100))
    lower_deficit = float(np.mean(lower_scores)) if lower_scores else 0.0

    # ── Trunk & neck score ────────────────────────────────────────────────────
    trunk_score = min(features.trunk_lateral_lean_deg   / 20.0 * 100, 100)
    neck_score  = min(features.neck_lateral_flexion_deg / 35.0 * 100, 100)
    posture_score = (trunk_score + neck_score) / 2

    categories = [
        "ROM Deficit",
        "Asymmetry",
        "Velocity ↓",
        "Guarding",
        "Upper Limb",
        "Lower Limb",
        "Trunk/Neck",
        "Head Tilt",
        "Face Pain",
    ]
    values = [
        min(features.rom_deficit_pct, 100),
        min(features.movement_asymmetry_pct, 100),
        min(features.velocity_reduction_pct, 100),
        100 if features.guarding_detected else 0,
        min(upper_deficit, 100),
        min(lower_deficit, 100),
        min(posture_score, 100),
        min(features.head_tilt_deg / 15.0 * 100, 100),
        min(features.face_pain_score, 100),
    ]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.plot(angles, values, "o-", linewidth=2, color="#e74c3c")
    ax.fill(angles, values, alpha=0.25, color="#e74c3c")
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("Clinical Signal Radar (All 33 Landmarks)", pad=20, fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")



def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to PIL Image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
