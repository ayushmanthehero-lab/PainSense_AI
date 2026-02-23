"""
modules/feature_extractor.py
─────────────────────────────
Converts raw MediaPipe landmarks (list of PoseFrame) into a structured
clinical feature vector used by MedGemma.

Key signals extracted
─────────────────────
1. Joint angles  – shoulder abduction/flexion, elbow, hip, knee
2. ROM deficit   – deviation from normal reference values
3. Asymmetry     – left-vs-right joint angle difference
4. Velocity      – temporal change rate (proxy for movement speed)
5. Guarding      – postural collapse / self-protective posture
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NORMAL_ROM, FEATURE_WEIGHTS
from modules.pose_estimator import PoseFrame


# Data structure

@dataclass
class ClinicalFeatureVector:
    """Structured biomechanical output fed to MedGemma."""

    # ── Upper-limb joint angles (degrees, peak ROM during recording) ──────────
    shoulder_abduction_left:  float = 0.0   # hip→shoulder→wrist  (lateral elevation)
    shoulder_abduction_right: float = 0.0
    shoulder_flexion_left:    float = 0.0   # hip→shoulder→elbow  (forward elevation)
    shoulder_flexion_right:   float = 0.0
    elbow_flexion_left:       float = 0.0   # shoulder→elbow→wrist
    elbow_flexion_right:      float = 0.0
    wrist_flexion_left:       float = 0.0   # MIN elbow→wrist→index  (lower = better)
    wrist_flexion_right:      float = 0.0

    # ── Lower-limb joint angles ───────────────────────────────────────────────
    hip_flexion_left:         float = 0.0   # shoulder→hip→knee
    hip_flexion_right:        float = 0.0
    hip_abduction_left:       float = 0.0   # lateral leg angle from vertical
    hip_abduction_right:      float = 0.0
    knee_flexion_left:        float = 0.0   # hip→knee→ankle
    knee_flexion_right:       float = 0.0
    ankle_dorsiflexion_left:  float = 0.0   # MIN knee→ankle→foot_index (lower = better)
    ankle_dorsiflexion_right: float = 0.0

    # ── Spine / postural scalars ──────────────────────────────────────────────
    trunk_lateral_lean_deg:   float = 0.0   # lateral lean of shoulder_mid vs hip_mid
    trunk_forward_lean_deg:   float = 0.0   # forward (sagittal) lean — key lumbar guarding signal
    neck_lateral_flexion_deg: float = 0.0   # ear-to-ear axis tilt from horizontal

    # ── Derived clinical signals ──────────────────────────────────────────────
    rom_deficit_pct:          float = 0.0   # % reduction from normal ROM
    movement_asymmetry_pct:   float = 0.0   # % left-right difference
    velocity_reduction_pct:   float = 0.0   # % speed below baseline
    guarding_detected:        bool  = False  # postural guarding flag
    facial_strain_detected:   bool  = False  # head-tilt proxy flag

    # ── Head / Face pose signals ──────────────────────────────────────────────
    head_tilt_deg:            float = 0.0
    head_nod_deg:             float = 0.0
    head_turn_deg:            float = 0.0
    eye_squeeze_detected:     bool  = False
    mouth_tension_detected:   bool  = False
    face_pain_score:          float = 0.0

    # Pre-calculated heuristic pain score (0–100)
    heuristic_pain_score:     float = 0.0

    # Frame statistics
    frames_analyzed:          int   = 0
    frames_with_detection:    int   = 0

    # Dominant body side
    affected_side:            str   = "unknown"
    most_restricted_joint:    str   = "unknown"

    def to_prompt_dict(self) -> Dict:
        """Return a clean dict suitable for embedding in the MedGemma prompt."""
        return {
            # Upper limb
            "shoulder_abduction_left_deg":    round(self.shoulder_abduction_left, 1),
            "shoulder_abduction_right_deg":   round(self.shoulder_abduction_right, 1),
            "shoulder_flexion_left_deg":      round(self.shoulder_flexion_left, 1),
            "shoulder_flexion_right_deg":     round(self.shoulder_flexion_right, 1),
            "elbow_flexion_left_deg":         round(self.elbow_flexion_left, 1),
            "elbow_flexion_right_deg":        round(self.elbow_flexion_right, 1),
            "wrist_flexion_left_deg":         round(self.wrist_flexion_left, 1),
            "wrist_flexion_right_deg":        round(self.wrist_flexion_right, 1),
            # Lower limb
            "hip_flexion_left_deg":           round(self.hip_flexion_left, 1),
            "hip_flexion_right_deg":          round(self.hip_flexion_right, 1),
            "hip_abduction_left_deg":         round(self.hip_abduction_left, 1),
            "hip_abduction_right_deg":        round(self.hip_abduction_right, 1),
            "knee_flexion_left_deg":          round(self.knee_flexion_left, 1),
            "knee_flexion_right_deg":         round(self.knee_flexion_right, 1),
            "ankle_dorsiflexion_left_deg":    round(self.ankle_dorsiflexion_left, 1),
            "ankle_dorsiflexion_right_deg":   round(self.ankle_dorsiflexion_right, 1),
            # Spine/posture
            "trunk_lateral_lean_deg":         round(self.trunk_lateral_lean_deg, 1),
            "trunk_forward_lean_deg":         round(self.trunk_forward_lean_deg, 1),
            "neck_lateral_flexion_deg":       round(self.neck_lateral_flexion_deg, 1),
            # Clinical summary
            "rom_deficit_pct":                round(self.rom_deficit_pct, 1),
            "movement_asymmetry_pct":         round(self.movement_asymmetry_pct, 1),
            "velocity_reduction_pct":         round(self.velocity_reduction_pct, 1),
            "guarding_detected":              self.guarding_detected,
            "facial_strain_detected":         self.facial_strain_detected,
            # Head / face
            "head_tilt_deg":                  round(self.head_tilt_deg, 1),
            "head_nod_deg":                   round(self.head_nod_deg, 1),
            "head_turn_deg":                  round(self.head_turn_deg, 1),
            "eye_squeeze_detected":           self.eye_squeeze_detected,
            "mouth_tension_detected":         self.mouth_tension_detected,
            "face_pain_score":                round(self.face_pain_score, 1),
            "heuristic_pain_score":           round(self.heuristic_pain_score, 1),
            "affected_side":                  self.affected_side,
            "most_restricted_joint":          self.most_restricted_joint,
            "frames_analyzed":                self.frames_analyzed,
        }


# Geometry helpers

def _vec3(lm: Optional[Dict]) -> Optional[np.ndarray]:
    """Return (x, y, z) numpy vector, or None if landmark is missing / low visibility."""
    if lm is None:
        return None
    if lm.get("visibility", 0) < 0.4:
        return None
    return np.array([lm["x"], lm["y"], lm["z"]])


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at point B formed by vectors B→A and B→C.
    Returns angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _safe_angle(frame: PoseFrame, name_a: str, name_b: str, name_c: str) -> Optional[float]:
    a = _vec3(frame.get_lm(name_a))
    b = _vec3(frame.get_lm(name_b))
    c = _vec3(frame.get_lm(name_c))
    if a is None or b is None or c is None:
        return None
    return _angle_3pts(a, b, c)


def _smooth_median(lst: List[float], window: int = 5) -> List[float]:
    """
    Rolling-median filter to suppress per-frame landmark noise.

    A window=5 smoothing removes single-frame spikes (e.g. 82°→90°→85°)
    without significantly attenuating genuine peak values, because the
    median always returns an observed measurement (it never averages two
    adjacent frames into a non-physical middle value).
    """
    if len(lst) < window:
        return lst[:]
    half = window // 2
    out: List[float] = []
    for i in range(len(lst)):
        lo = max(0, i - half)
        hi = min(len(lst), i + half + 1)
        out.append(float(np.median(lst[lo:hi])))
    return out


# Main extractor

class FeatureExtractor:
    """
    Converts a list of PoseFrame objects into a ClinicalFeatureVector.

    Usage
    -----
    extractor = FeatureExtractor()
    features  = extractor.extract(pose_frames)
    """

    # Adaptive velocity: if observed velocity > this fraction of baseline, treat as normal
    _VELOCITY_BASELINE = 0.018   # normalised units per frame
    # Guarding: spine (shoulder→hip midpoint) forward-lean threshold (degrees from vertical)
    _GUARDING_LEAN_THRESHOLD_DEG = 15.0

    def extract(self, pose_frames: List[PoseFrame]) -> ClinicalFeatureVector:
        """
        Main entry point.

        Parameters
        ----------
        pose_frames : output of PoseEstimator.process_video()

        Returns
        -------
        ClinicalFeatureVector
        """
        fv = ClinicalFeatureVector()
        fv.frames_analyzed = len(pose_frames)

        if not pose_frames:
            return fv

        # ── 1. Collect per-frame joint angles (all 33 landmarks) ─────────────
        angles: Dict[str, List[float]] = {
            # Upper limb
            "sa_l": [], "sa_r": [],   # shoulder abduction  (hip→shoulder→wrist)
            "sf_l": [], "sf_r": [],   # shoulder flexion    (hip→shoulder→elbow)
            "ef_l": [], "ef_r": [],   # elbow flexion       (shoulder→elbow→wrist)
            "wf_l": [], "wf_r": [],   # wrist flexion       (elbow→wrist→index) — MIN
            # Lower limb
            "hf_l": [], "hf_r": [],   # hip flexion         (shoulder→hip→knee)
            "ha_l": [], "ha_r": [],   # hip abduction       (leg angle from vertical)
            "kf_l": [], "kf_r": [],   # knee flexion        (hip→knee→ankle)
            "ad_l": [], "ad_r": [],   # ankle dorsiflexion  (knee→ankle→foot_index) — MIN
        }

        positions: Dict[str, List[np.ndarray]] = {
            "ls": [], "rs": [],   # shoulders (for velocity)
            "lh": [], "rh": [],   # hips
        }

        trunk_lean_list: List[float] = []    # lateral lean of spine (°)
        trunk_fwd_list:  List[float] = []    # forward (sagittal) lean of spine (°)
        neck_flex_list:  List[float] = []    # ear-to-ear tilt (°)

        for pf in pose_frames:
            # ── Upper limb ────────────────────────────────────────────────
            # Shoulder abduction / flexion: angle of arm vector from downward vertical.
            # Using vertical reference (not hip→shoulder→wrist) so the measurement
            # works even when the hip is off-screen (e.g. seated or cropped video).
            #   0°  = arm hanging at side
            #   90° = arm horizontal
            #   180° = arm fully overhead
            _down = np.array([0.0, 1.0, 0.0])   # y increases downward in MediaPipe
            for _sh_name, _wr_name, _el_name, _sa_k, _sf_k in [
                ("left_shoulder",  "left_wrist",  "left_elbow",  "sa_l", "sf_l"),
                ("right_shoulder", "right_wrist", "right_elbow", "sa_r", "sf_r"),
            ]:
                _sh = _vec3(pf.get_lm(_sh_name))
                if _sh is None:
                    continue
                _wr = _vec3(pf.get_lm(_wr_name))
                if _wr is not None:
                    _arm = _wr - _sh
                    _ca  = np.dot(_arm, _down) / (np.linalg.norm(_arm) + 1e-9)
                    angles[_sa_k].append(math.degrees(math.acos(np.clip(_ca, -1.0, 1.0))))
                _el = _vec3(pf.get_lm(_el_name))
                if _el is not None:
                    _ua  = _el - _sh
                    _ca  = np.dot(_ua, _down) / (np.linalg.norm(_ua) + 1e-9)
                    angles[_sf_k].append(math.degrees(math.acos(np.clip(_ca, -1.0, 1.0))))

            # Elbow flexion: shoulder → elbow → wrist
            v = _safe_angle(pf, "left_shoulder",  "left_elbow",  "left_wrist")
            if v is not None: angles["ef_l"].append(v)
            v = _safe_angle(pf, "right_shoulder", "right_elbow", "right_wrist")
            if v is not None: angles["ef_r"].append(v)

            # Wrist flexion: elbow → wrist → index finger  (track MINIMUM)
            v = _safe_angle(pf, "left_elbow",  "left_wrist",  "left_index")
            if v is not None: angles["wf_l"].append(v)
            v = _safe_angle(pf, "right_elbow", "right_wrist", "right_index")
            if v is not None: angles["wf_r"].append(v)

            # ── Lower limb ────────────────────────────────────────────────
            # Hip flexion: shoulder → hip → knee
            v = _safe_angle(pf, "left_shoulder",  "left_hip",  "left_knee")
            if v is not None: angles["hf_l"].append(v)
            v = _safe_angle(pf, "right_shoulder", "right_hip", "right_knee")
            if v is not None: angles["hf_r"].append(v)

            # Hip abduction: angle of hip→knee vector from downward vertical
            # 0° = leg straight down, 30° = 30° of abduction
            lh_v = _vec3(pf.get_lm("left_hip"));  lk_v = _vec3(pf.get_lm("left_knee"))
            rh_v = _vec3(pf.get_lm("right_hip")); rk_v = _vec3(pf.get_lm("right_knee"))
            down = np.array([0.0, 1.0, 0.0])   # y increases downward in MediaPipe
            if lh_v is not None and lk_v is not None:
                leg = lk_v - lh_v
                cos_a = np.dot(leg, down) / (np.linalg.norm(leg) + 1e-9)
                angles["ha_l"].append(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))
            if rh_v is not None and rk_v is not None:
                leg = rk_v - rh_v
                cos_a = np.dot(leg, down) / (np.linalg.norm(leg) + 1e-9)
                angles["ha_r"].append(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))

            # Knee flexion: hip → knee → ankle
            v = _safe_angle(pf, "left_hip",  "left_knee",  "left_ankle")
            if v is not None: angles["kf_l"].append(v)
            v = _safe_angle(pf, "right_hip", "right_knee", "right_ankle")
            if v is not None: angles["kf_r"].append(v)

            # Ankle dorsiflexion: knee → ankle → foot_index  (track MINIMUM)
            v = _safe_angle(pf, "left_knee",  "left_ankle",  "left_foot_index")
            if v is not None: angles["ad_l"].append(v)
            v = _safe_angle(pf, "right_knee", "right_ankle", "right_foot_index")
            if v is not None: angles["ad_r"].append(v)

            # ── Spine / posture scalars ────────────────────────────────────
            ls_v = _vec3(pf.get_lm("left_shoulder"))
            rs_v = _vec3(pf.get_lm("right_shoulder"))
            lh_p = _vec3(pf.get_lm("left_hip"))
            rh_p = _vec3(pf.get_lm("right_hip"))
            if all(x is not None for x in [ls_v, rs_v, lh_p, rh_p]):
                sh_mid  = (ls_v + rs_v) / 2
                hip_mid = (lh_p + rh_p) / 2
                spine   = sh_mid - hip_mid      # should point upward (−y in MediaPipe)
                # Trunk lateral lean: atan2(lateral/vertical component)
                lat_lean = math.degrees(math.atan2(abs(spine[0]),
                                                   abs(spine[1]) + 1e-9))
                trunk_lean_list.append(lat_lean)

                # Trunk forward lean (sagittal): atan2(z-depth / vertical)
                # In MediaPipe, z < 0 means landmark is closer to camera.
                # When a person bends forward the shoulder moves toward the camera
                # relative to the hip, increasing |spine_z|.
                fwd_lean = math.degrees(math.atan2(abs(spine[2]),
                                                   abs(spine[1]) + 1e-9))
                trunk_fwd_list.append(min(fwd_lean, 60.0))

                # Neck lateral flexion: tilt of ear-to-ear axis from horizontal
                # 0° = level head, 30° = significant lateral tilt
                l_ear_v = _vec3(pf.get_lm("left_ear"))
                r_ear_v = _vec3(pf.get_lm("right_ear"))
                if l_ear_v is not None and r_ear_v is not None:
                    ear_axis = r_ear_v - l_ear_v   # left→right ear vector
                    # y increases downward in MediaPipe; for level head ear_axis is horizontal
                    tilt = abs(math.degrees(
                        math.atan2(abs(ear_axis[1]), abs(ear_axis[0]) + 1e-9)
                    ))
                    neck_flex_list.append(min(tilt, 60.0))

            # Landmark positions for velocity
            for lm_name, key in [("left_shoulder","ls"),("right_shoulder","rs"),
                                  ("left_hip","lh"),("right_hip","rh")]:
                p = _vec3(pf.get_lm(lm_name))
                if p is not None:
                    positions[key].append(p)

        fv.frames_with_detection = sum(1 for k in angles.values() if len(k) > 0)

        # ── 1b. Temporal smoothing — rolling-median (window=5) to remove ───────
        # per-frame landmark noise before peak/min aggregation.
        # This prevents single-frame tracking blips from inflating ROM values.
        angles = {k: _smooth_median(v, window=5) for k, v in angles.items()}

        # ── 2. Aggregate joint values ─────────────────────────────────────────
        # -1.0 sentinel = landmark never detected / joint not in frame
        # 999.0 sentinel = used for _LOWER_IS_BETTER joints (wrist/ankle)
        def _max_or_neg1(lst): return float(np.max(lst)) if lst else -1.0
        def _min_or_inf(lst):  return float(np.min(lst)) if lst else 999.0
        def _range_of(lst):    return float(np.max(lst) - np.min(lst)) if len(lst) >= 2 else 0.0

        def _anatomic_flex(lst):
            """
            Convert geometric vertex angles to maximum achieved anatomic flexion ROM.

            Geometric: 180° = straight, ~45° = maximally bent.
            Anatomic:    0° = neutral,  135° = full knee flex, etc.
            Conversion: anatomic = 180 − min_geometric

            Returns -1.0 if no data (landmark off-screen).
            """
            if not lst:
                return -1.0   # sentinel: not detected
            min_geo = float(np.min(lst))
            if min_geo < 5.0:     # degenerate / noise
                return -1.0
            return max(0.0, 180.0 - min_geo)

        # Upper limb
        # Shoulder abduction/flexion: vertical-reference angle (0°=at side, 180°=overhead)
        fv.shoulder_abduction_left  = _max_or_neg1(angles["sa_l"])
        fv.shoulder_abduction_right = _max_or_neg1(angles["sa_r"])
        fv.shoulder_flexion_left    = _max_or_neg1(angles["sf_l"])
        fv.shoulder_flexion_right   = _max_or_neg1(angles["sf_r"])
        # Elbow flexion: geometric 180°=straight → anatomic = 180 − min_geometric
        fv.elbow_flexion_left       = _anatomic_flex(angles["ef_l"])
        fv.elbow_flexion_right      = _anatomic_flex(angles["ef_r"])
        # Wrist flexion — use MIN geometric angle (lower = more flexion)
        fv.wrist_flexion_left       = _min_or_inf(angles["wf_l"])
        fv.wrist_flexion_right      = _min_or_inf(angles["wf_r"])

        # Lower limb
        # Hip flexion: geometric 180°=standing → anatomic = 180 − min_geometric
        fv.hip_flexion_left         = _anatomic_flex(angles["hf_l"])
        fv.hip_flexion_right        = _anatomic_flex(angles["hf_r"])
        # Hip abduction: angle from vertical — cap at 89° (>90° = sitting/legs inverted)
        _ha_l = _max_or_neg1(angles["ha_l"])
        _ha_r = _max_or_neg1(angles["ha_r"])
        fv.hip_abduction_left       = min(_ha_l, 89.0) if _ha_l >= 0 else -1.0
        fv.hip_abduction_right      = min(_ha_r, 89.0) if _ha_r >= 0 else -1.0
        # Knee flexion: geometric 180°=straight → anatomic = 180 − min_geometric
        fv.knee_flexion_left        = _anatomic_flex(angles["kf_l"])
        fv.knee_flexion_right       = _anatomic_flex(angles["kf_r"])
        # Ankle dorsiflexion — use MIN geometric angle (lower = more dorsiflexion)
        fv.ankle_dorsiflexion_left  = _min_or_inf(angles["ad_l"])
        fv.ankle_dorsiflexion_right = _min_or_inf(angles["ad_r"])

        # Spine / posture scalars — use MEDIAN (stable central estimate)
        if trunk_lean_list:
            fv.trunk_lateral_lean_deg   = float(np.median(trunk_lean_list))
        if trunk_fwd_list:
            fv.trunk_forward_lean_deg   = float(np.median(trunk_fwd_list))
        if neck_flex_list:
            fv.neck_lateral_flexion_deg = float(np.median(neck_flex_list))

        # Per-joint range of motion (max−min across recording, used for MOVE_THR)
        _rom_range = {k: _range_of(v) for k, v in angles.items()}
        _MOVE_THR      = 18.0   # °: minimum excursion to consider a joint "actively used"
        _MOVE_THR_WRIST  = 10.0  # wrist has smaller expected excursion
        _MOVE_THR_ANKLE  = 12.0  # ankle too

        # For wrist/ankle "lower is better" joints, _MOVE_THR still uses range
        # (max-min) which is symmetric regardless of direction

        # Joints where LOWER measured value = better performance
        # → deficit = how much measured EXCEEDS expected minimum
        _LOWER_IS_BETTER = {"wrist_flexion", "ankle_dorsiflexion"}

        # ── 3. ROM deficit ────────────────────────────────────────────────────
        # Structure: (joint_name, l_val, r_val, l_rng, r_rng, move_thr)
        rom_comparisons = [
            ("shoulder_abduction", fv.shoulder_abduction_left,  fv.shoulder_abduction_right,
             _rom_range["sa_l"],   _rom_range["sa_r"],   _MOVE_THR),
            ("shoulder_flexion",   fv.shoulder_flexion_left,    fv.shoulder_flexion_right,
             _rom_range["sf_l"],   _rom_range["sf_r"],   _MOVE_THR),
            ("elbow_flexion",      fv.elbow_flexion_left,       fv.elbow_flexion_right,
             _rom_range["ef_l"],   _rom_range["ef_r"],   _MOVE_THR),
            ("wrist_flexion",      fv.wrist_flexion_left,       fv.wrist_flexion_right,
             _rom_range["wf_l"],   _rom_range["wf_r"],   _MOVE_THR_WRIST),
            ("hip_flexion",        fv.hip_flexion_left,         fv.hip_flexion_right,
             _rom_range["hf_l"],   _rom_range["hf_r"],   _MOVE_THR),
            ("hip_abduction",      fv.hip_abduction_left,       fv.hip_abduction_right,
             _rom_range["ha_l"],   _rom_range["ha_r"],   _MOVE_THR),
            ("knee_flexion",       fv.knee_flexion_left,        fv.knee_flexion_right,
             _rom_range["kf_l"],   _rom_range["kf_r"],   _MOVE_THR),
            ("ankle_dorsiflexion", fv.ankle_dorsiflexion_left,  fv.ankle_dorsiflexion_right,
             _rom_range["ad_l"],   _rom_range["ad_r"],   _MOVE_THR_ANKLE),
        ]

        deficits: List[float] = []
        worst_deficit = 0.0
        worst_joint   = "none_detected"

        for joint_name, l_val, r_val, l_rng, r_rng, thr in rom_comparisons:
            if max(l_rng, r_rng) < thr:
                deficits.append(0.0)
                continue
            norm = NORMAL_ROM.get(joint_name, 90)
            if joint_name in _LOWER_IS_BETTER:
                # Deficit = how much the min angle EXCEEDS the expected minimum
                best = min(l_val, r_val)   # lower = more range
                d = max(0.0, (best - norm) / norm * 100) if best < 999 else 0.0
            else:
                best = max(l_val, r_val)   # higher = more range
                if best < 0:               # both sides are -1 sentinel (not detected)
                    deficits.append(0.0)
                    continue
                d = max(0.0, (norm - best) / norm * 100)
            deficits.append(d)
            if d > worst_deficit:
                worst_deficit = d
                worst_joint   = joint_name

        # ── Lumbar forward lean: inject as deficit when significant (lower back guarding) ──
        _fwd_norm = NORMAL_ROM.get("lumbar_forward_lean", 12)
        if fv.trunk_forward_lean_deg > _fwd_norm:
            fwd_deficit = min(100.0,
                              (fv.trunk_forward_lean_deg - _fwd_norm) / _fwd_norm * 100)
            deficits.append(fwd_deficit)
            if fwd_deficit > worst_deficit:
                worst_deficit = fwd_deficit
                worst_joint   = "lumbar_spine"

        fv.rom_deficit_pct       = float(np.mean(deficits)) if deficits else 0.0
        fv.most_restricted_joint = worst_joint if worst_deficit > 15.0 else "none_detected"

        # ── 4. Asymmetry ──────────────────────────────────────────────────────
        asym_vals: List[float] = []
        side_deficits: Dict[str, float] = {"left": 0.0, "right": 0.0}

        for joint_name, l_val, r_val, l_rng, r_rng, thr in rom_comparisons:
            if l_rng < thr or r_rng < thr:
                continue
            norm = NORMAL_ROM.get(joint_name, 90)
            if joint_name in _LOWER_IS_BETTER:
                if l_val < 999 and r_val < 999:
                    avg = (l_val + r_val) / 2.0
                    asym_vals.append(abs(l_val - r_val) / (avg + 1e-6) * 100)
                if l_val < 999:
                    side_deficits["left"]  += max(0.0, (l_val - norm) / norm * 100)
                if r_val < 999:
                    side_deficits["right"] += max(0.0, (r_val - norm) / norm * 100)
            else:
                if l_val > 0 and r_val > 0:
                    avg = (l_val + r_val) / 2.0
                    asym_vals.append(abs(l_val - r_val) / avg * 100)
                if l_val > 0:
                    side_deficits["left"]  += max(0.0, (norm - l_val) / norm * 100)
                if r_val > 0:
                    side_deficits["right"] += max(0.0, (norm - r_val) / norm * 100)

        fv.movement_asymmetry_pct = float(np.mean(asym_vals)) if asym_vals else 0.0

        # Determine affected side using per-side ROM deficit totals
        if fv.movement_asymmetry_pct > 10:
            fv.affected_side = "right" if side_deficits["right"] > side_deficits["left"] else "left"
        else:
            fv.affected_side = "bilateral"

        # ── 5. Velocity reduction ─────────────────────────────────────────
        all_velocities: List[float] = []
        for key in ["ls", "rs", "lh", "rh"]:
            pts = positions[key]
            for i in range(1, len(pts)):
                all_velocities.append(float(np.linalg.norm(pts[i] - pts[i-1])))

        if all_velocities:
            avg_vel     = float(np.mean(all_velocities))
            median_vel  = float(np.median(all_velocities))
            # Use adaptive baseline: 75th percentile of observed velocities as
            # "expected normal" for this recording, then compare mean to it.
            p75_vel = float(np.percentile(all_velocities, 75)) if len(all_velocities) >= 4 else self._VELOCITY_BASELINE
            effective_baseline = max(p75_vel * 1.15, self._VELOCITY_BASELINE)
            fv.velocity_reduction_pct = max(
                0.0,
                (effective_baseline - avg_vel) / effective_baseline * 100
            )
        else:
            fv.velocity_reduction_pct = 0.0

        # ── 6. Guarding detection (spine forward-lean angle) ──────────────
        # Measure the angle between the vertical axis and the vector from
        # hip-midpoint → shoulder-midpoint.  Consistent lean > threshold = guarding.
        lean_angles: List[float] = []
        for pf in pose_frames:
            ls = _vec3(pf.get_lm("left_shoulder"))
            rs = _vec3(pf.get_lm("right_shoulder"))
            lh = _vec3(pf.get_lm("left_hip"))
            rh = _vec3(pf.get_lm("right_hip"))
            if ls is not None and rs is not None and lh is not None and rh is not None:
                shoulder_mid = (ls + rs) / 2
                hip_mid      = (lh + rh) / 2
                spine_vec    = shoulder_mid - hip_mid   # points upward
                # Angle from vertical (y-axis in normalised coords: y increases downward)
                # Vertical in image coords points toward -y (up)
                vertical = np.array([0.0, -1.0, 0.0])
                cos_a = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-9)
                cos_a = np.clip(cos_a, -1.0, 1.0)
                lean_angles.append(math.degrees(math.acos(cos_a)))

        if lean_angles:
            # Guarding: median lean > threshold in > 40% of frames
            lean_arr = np.array(lean_angles)
            fv.guarding_detected = float(np.median(lean_arr)) > self._GUARDING_LEAN_THRESHOLD_DEG
        else:
            # Legacy fallback: nose-above-shoulder heuristic
            nose_guarding: List[bool] = []
            for pf in pose_frames:
                nose = _vec3(pf.get_lm("nose"))
                ls_  = _vec3(pf.get_lm("left_shoulder"))
                rs_  = _vec3(pf.get_lm("right_shoulder"))
                if nose is not None and ls_ is not None and rs_ is not None:
                    mid_y = (ls_[1] + rs_[1]) / 2
                    nose_guarding.append(nose[1] < mid_y - 0.05)
            fv.guarding_detected = (
                len(nose_guarding) > 0 and
                sum(nose_guarding) / len(nose_guarding) > 0.4
            )

        # ── 6b. Head pose & face pain signals ────────────────────────────────
        # All computed from MediaPipe face landmarks already in PoseFrame:
        #   nose(0), left_eye(2), right_eye(5), left_ear(7), right_ear(8)
        #   mouth_left(9), mouth_right(10)
        #
        # Coordinate system: x=right, y=down, z=depth (toward camera = negative)

        # Collect per-frame raw ratios (not yet degrees) for ROM-based aggregation
        tilt_devs:   List[float] = []  # true deviation from horizontal (°)
        nod_raws:    List[float] = []  # nose-y relative to ear-mid-y, normalised
        turn_raws:   List[float] = []  # nose-x relative to ear-mid-x, normalised (signed)
        eye_scores:  List[float] = []  # eye-squint proxy (normalised)
        mouth_scores:List[float] = []  # mouth-tension proxy  (normalised)

        for pf in pose_frames:
            nose   = _vec3(pf.get_lm("nose"))
            l_eye  = _vec3(pf.get_lm("left_eye"))
            r_eye  = _vec3(pf.get_lm("right_eye"))
            l_ear  = _vec3(pf.get_lm("left_ear"))
            r_ear  = _vec3(pf.get_lm("right_ear"))
            m_l    = _vec3(pf.get_lm("mouth_left"))
            m_r    = _vec3(pf.get_lm("mouth_right"))
            l_ei   = _vec3(pf.get_lm("left_eye_inner"))
            r_ei   = _vec3(pf.get_lm("right_eye_inner"))
            l_eo   = _vec3(pf.get_lm("left_eye_outer"))
            r_eo   = _vec3(pf.get_lm("right_eye_outer"))

            # ── Head tilt (roll): true deviation of eye-line from horizontal ─
            # BUG FIX: In MediaPipe coords, right_eye (person's right) appears on
            # LEFT side of image → dx is NEGATIVE → atan2(~0, negative) ≈ 180°.
            # Fix: map raw abs-angle to deviation from horizontal via
            #      tilt_dev = angle if angle ≤ 90 else (180 - angle)
            # So 176° → 4°, 10° → 10°, 90° → 90° (head completely sideways).
            if l_eye is not None and r_eye is not None:
                dy = r_eye[1] - l_eye[1]
                dx = r_eye[0] - l_eye[0] + 1e-9
                raw_abs = abs(math.degrees(math.atan2(dy, dx)))
                tilt_dev = raw_abs if raw_abs <= 90.0 else 180.0 - raw_abs
                tilt_devs.append(tilt_dev)

            # ── Head nod (pitch): range-of-motion across frames ──────────
            # Normalise by inter-eye distance (much more stable than inter-ear-x
            # which becomes near-zero when the face is front-on, amplifying noise).
            # Store signed ratio; temporal IQR captures actual nod motion.
            if nose is not None and l_eye is not None and r_eye is not None:
                eye_mid_y     = (l_eye[1] + r_eye[1]) / 2
                inter_eye_x   = abs(l_eye[0] - r_eye[0]) + 1e-6
                nod_raws.append((nose[1] - eye_mid_y) / inter_eye_x)

            # ── Head turn (yaw): range-of-motion across frames ────────────
            # Use ear-midpoint for X reference but normalise by inter-eye width.
            if nose is not None and l_ear is not None and r_ear is not None and \
               l_eye is not None and r_eye is not None:
                ear_mid_x   = (l_ear[0] + r_ear[0]) / 2
                inter_eye_x = abs(l_eye[0] - r_eye[0]) + 1e-6
                turn_raws.append((nose[0] - ear_mid_x) / inter_eye_x)

            # ── Eye squint proxy: inner/outer eye landmark vertical span ──
            if l_ei is not None and l_eo is not None and r_ei is not None and r_eo is not None:
                l_span = abs(l_ei[1] - l_eo[1])
                r_span = abs(r_ei[1] - r_eo[1])
                eye_width = abs(l_eye[0] - r_eye[0]) + 1e-6 if l_eye is not None and r_eye is not None else 0.1
                squint_norm = (l_span + r_span) / (eye_width + 1e-6)
                eye_scores.append(squint_norm)

            # ── Mouth tension proxy: mouth-width vs eye-span ──────────────
            if m_l is not None and m_r is not None and l_eye is not None and r_eye is not None:
                mouth_w = abs(m_l[0] - m_r[0])
                eye_w   = abs(l_eye[0] - r_eye[0]) + 1e-6
                mouth_scores.append(mouth_w / eye_w)

        # ── Aggregate: head tilt = median deviation from horizontal ──────────
        if tilt_devs:
            fv.head_tilt_deg = float(np.median(tilt_devs))

        # ── Aggregate: nod & turn = interquartile range-of-motion ────────────
        # Calibrated for inter-eye normalization:
        #   still head  → IQR ≈ 0.01-0.03 → × 60 → 0.6–1.8°
        #   real 15° nod → IQR ≈ 0.25     → × 60 → 15°
        if nod_raws:
            nod_iqr = float(np.percentile(nod_raws, 75) - np.percentile(nod_raws, 25))
            fv.head_nod_deg = float(np.clip(nod_iqr * 60.0, 0.0, 90.0))

        if turn_raws:
            turn_iqr = float(np.percentile(turn_raws, 75) - np.percentile(turn_raws, 25))
            fv.head_turn_deg = float(np.clip(turn_iqr * 60.0, 0.0, 90.0))

        # ── Eye squint ────────────────────────────────────────────────────────
        if eye_scores:
            fv.eye_squeeze_detected = float(np.median(eye_scores)) < 0.08

        # ── Mouth tension ─────────────────────────────────────────────────────
        if mouth_scores:
            fv.mouth_tension_detected = float(np.median(mouth_scores)) > 1.2

        # ── Facial strain: conservative thresholds to avoid false positives ──
        #   tilt > 12° = clearly visible lateral ear-drop
        #   nod  > 18° = clearly visible forward chin nod  (IQR ≈ 0.30 × 60)
        #   turn > 20° = clearly visible left/right rotation (IQR ≈ 0.33 × 60)
        fv.facial_strain_detected = (
            fv.head_tilt_deg > 12.0 or
            fv.head_nod_deg  > 18.0 or
            fv.head_turn_deg > 20.0 or
            fv.eye_squeeze_detected or
            fv.mouth_tension_detected
        )

        # ── Face pain score (0–100) ───────────────────────────────────────────
        # Scale references: tilt 15° = full (30 pts), nod 25° = full (25 pts),
        #                   turn 30° = full (15 pts).
        # Still head: ~ 0–3 pts total.
        face_score = 0.0
        if tilt_devs:
            face_score += min(fv.head_tilt_deg / 15.0, 1.0) * 30   # tilt → 30 pts
        if nod_raws:
            face_score += min(fv.head_nod_deg  / 25.0, 1.0) * 25   # nod  → 25 pts
        if turn_raws:
            face_score += min(fv.head_turn_deg / 30.0, 1.0) * 15   # turn → 15 pts
        if fv.eye_squeeze_detected:
            face_score += 20   # grimace → 20 pts
        if fv.mouth_tension_detected:
            face_score += 10   # tension → 10 pts
        fv.face_pain_score = min(face_score, 100.0)

        # ── 7. Heuristic pain score (0–100) ───────────────────────────────
        body_score = (
            FEATURE_WEIGHTS["rom_deficit_pct"]        * min(fv.rom_deficit_pct, 100) +
            FEATURE_WEIGHTS["movement_asymmetry_pct"] * min(fv.movement_asymmetry_pct, 100) +
            FEATURE_WEIGHTS["velocity_reduction_pct"] * min(fv.velocity_reduction_pct, 100) +
            FEATURE_WEIGHTS["guarding_detected"]      * (100 if fv.guarding_detected else 0) +
            FEATURE_WEIGHTS["facial_strain_detected"] * (100 if fv.facial_strain_detected else 0)
        )
        # Blend body score (80%) + face pain score (20%)
        fv.heuristic_pain_score = min(
            body_score * 0.80 + fv.face_pain_score * 0.20,
            100.0
        )

        return fv
