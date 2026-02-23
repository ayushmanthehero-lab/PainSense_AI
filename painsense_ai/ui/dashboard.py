"""
ui/dashboard.py
────────────────
Gradio-based MoveSense AI dashboard.

Workflow
────────
1. User uploads a short movement video clip
2. Pose estimation runs (MediaPipe)
3. Feature extraction produces the clinical vector
4. MedGemma Module 1 → Pain probability + differential diagnosis
5. MedGemma Safety Layer → Red flag check
6. MedGemma Module 2 → SOAP note + patient explanation + rehab
7. All outputs displayed in tabbed interface
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UI_TITLE, UI_DESCRIPTION, UI_SERVER_PORT, PAIN_HIGH_THRESHOLD, PAIN_MEDIUM_THRESHOLD
from modules.pose_estimator import PoseEstimator
from modules.feature_extractor import FeatureExtractor
from modules.medgemma_engine import MedGemmaEngine
from modules.clinical_reasoning import ClinicalReasoner
from modules.safety_layer import SafetyChecker
from modules.documentation import DocumentationGenerator
from modules.movement_classifier import (
    REGION_SHOULDER, REGION_ELBOW, REGION_HIP_KNEE,
    REGION_LUMBAR, REGION_CERVICAL,
)
from modules.pain_scorer import compute_mas
from utils.visualization import pain_gauge_chart, rom_bar_chart, feature_radar_chart, bgr_to_pil
from utils.export import build_json_report, save_json_report, build_text_report, save_text_report
from utils.anatomy_map import (
    draw_region_highlight, draw_zoom_view, get_muscle_list,
    draw_pose_muscle_overlay, draw_pose_zoom_view,
    REGION_FULL, REGION_UNKNOWN,
)
from utils.baseline import (
    save_baseline, load_baseline,
    format_deviation_table, baseline_summary,
)

# Mapping from UI dropdown label -> region constant used by ClinicalReasoner
_REGION_MAP = {
    "Auto-detect":          "auto",
    "Shoulder / Upper Arm": REGION_SHOULDER,
    "Elbow / Wrist":        REGION_ELBOW,
    "Hip / Knee":           REGION_HIP_KNEE,
    "Lower Back / Lumbar":  REGION_LUMBAR,
    "Neck / Cervical":      REGION_CERVICAL,
}

# Human-readable region labels for all UI display surfaces
_REGION_LABELS = {
    REGION_SHOULDER: "Shoulder / Upper Arm",
    REGION_ELBOW:    "Elbow / Wrist",
    REGION_HIP_KNEE: "Hip / Knee",
    REGION_LUMBAR:   "Lower Back / Lumbar Spine",
    REGION_CERVICAL: "Neck / Cervical Spine",
    REGION_FULL:     "Full Body",
    REGION_UNKNOWN:  "Full Body (Auto-detected)",
}

# Fallback differentials when MedGemma returns an empty list
_FALLBACK_DIFFERENTIALS = {
    REGION_LUMBAR:   [
        "Lumbar muscle strain / myofascial pain",
        "Lumbar facet joint dysfunction",
        "Lumbar disc herniation with radiculopathy (early presentation)",
    ],
    REGION_SHOULDER: [
        "Rotator cuff tendinopathy",
        "Glenohumeral impingement syndrome",
        "Adhesive capsulitis (frozen shoulder)",
    ],
    REGION_ELBOW:    [
        "Lateral epicondylalgia (tennis elbow)",
        "Medial epicondylalgia (golfer's elbow)",
        "Cubital tunnel syndrome",
    ],
    REGION_HIP_KNEE: [
        "Patellofemoral pain syndrome",
        "Hip flexor strain / iliopsoas tendinopathy",
        "Greater trochanteric pain syndrome",
    ],
    REGION_CERVICAL: [
        "Cervical facet joint dysfunction",
        "Cervical myofascial pain syndrome",
        "Cervicogenic headache",
    ],
    REGION_FULL:     [
        "Generalised myofascial pain syndrome",
        "Postural deconditioning",
        "Musculoskeletal overuse syndrome",
    ],
    REGION_UNKNOWN:  [
        "Generalised myofascial pain syndrome",
        "Postural deconditioning",
        "Musculoskeletal overuse syndrome",
    ],
}


# Module-level cache — stores last analysis objects for export
_last_results: dict = {}


# Initialise pipeline components (lazy – model loads on first use)
pose_estimator   = PoseEstimator()
feature_extractor= FeatureExtractor()


def _get_engine() -> MedGemmaEngine:
    return MedGemmaEngine()   # singleton


def _pain_color(prob: int) -> str:
    if prob >= PAIN_HIGH_THRESHOLD:
        return "#e74c3c"
    elif prob >= PAIN_MEDIUM_THRESHOLD:
        return "#f39c12"
    return "#2ecc71"


# Core analysis function

def _blank_image(text: str = "Awaiting analysis...", w: int = 400, h: int = 250) -> Image.Image:
    """Return a dark placeholder PIL image shown before analysis runs."""
    img  = Image.new("RGB", (w, h), color=(30, 30, 40))
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, h // 2), text, fill=(120, 120, 140), anchor="mm")
    return img

_BLANK = _blank_image()

# Temp files created by transcode — tracked for cleanup
_TRANSCODE_TMP: list[str] = []


def _ensure_h264(video_path: str) -> str:
    """
    Transcode video to browser-safe H.264 MP4 using ffmpeg.
    Returns path to transcoded file (or original if ffmpeg fails).
    Required because phone/screen-recording videos often use HEVC or
    other codecs that browsers cannot play natively.
    """
    # Already transcoded — skip
    if video_path.endswith("_h264.mp4"):
        return video_path
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix="_h264.mp4", delete=False,
            dir=tempfile.gettempdir()
        )
        out_path = tmp.name
        tmp.close()

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-c:v", "libx264",   # H.264 — universally browser-playable
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",  # stream-optimised MP4
            "-loglevel", "error",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0:
            _TRANSCODE_TMP.append(out_path)
            print(f"[PainSense] Transcoded video → {out_path}")
            return out_path
        else:
            print(f"[PainSense] ffmpeg transcode failed: {result.stderr.decode()[:200]}")
            return video_path
    except Exception as e:
        print(f"[PainSense] Transcode skipped ({e})")
        return video_path


def analyze_video(upload_path: str, camera_path: str, symptom_region: str = "Auto-detect"):
    """
    Full pipeline: video → clinical report.
    Accepts either an uploaded file or a camera recording (whichever is provided).

    Yields intermediate status messages, then final results.
    """
    video_path = upload_path or camera_path

    if video_path is None:
        yield (
            "No video provided. Please upload a file or record from camera first.",
            None, None, None,
            "", "",
            None, None, "",
            "",
            "", "", "",
        )
        return

    # Transcode to H.264 so browser can play it back and OpenCV reads cleanly
    video_path = _ensure_h264(video_path)

    status_log = []

    def status(msg):
        status_log.append(msg)
        try:
            print(f"[PainSense] {msg}")
        except UnicodeEncodeError:
            safe = msg.encode("ascii", "replace").decode("ascii")
            print(f"[PainSense] {safe}")

    # ── Step 1: Pose estimation ────────────────────────────────────────────
    status("Extracting pose landmarks from video ...")
    try:
        pose_frames, thumbnail = pose_estimator.process_video(video_path, max_frames=300)
    except Exception as e:
        yield (f"Pose estimation failed: {e}", _BLANK, _BLANK, _BLANK, "", "", None, None, "", "", "", "", "")
        return

    if not pose_frames:
        yield ("No pose detected in video. Please ensure the patient is fully visible.", _BLANK, _BLANK, _BLANK, "", "", None, None, "", "", "", "", "")
        return

    status(f"Detected pose in {len(pose_frames)} frames.")

    # Cache video frame + landmarks for pose-overlay anatomy map
    _last_results["thumbnail_bgr"] = thumbnail
    _last_results["pose_frames"]   = pose_frames

    # ── Step 2: Feature extraction ────────────────────────────────────────
    status("Extracting biomechanical features ...")
    features = feature_extractor.extract(pose_frames)
    status(f"Features extracted ({features.frames_analyzed} frames).")

    # ── Step 3: MedGemma clinical reasoning ───────────────────────────────
    status("Clinical reasoning (Module 1) ...")
    engine   = _get_engine()
    reasoner = ClinicalReasoner(engine)
    region_code = _REGION_MAP.get(symptom_region or "Auto-detect", "auto")
    try:
        assessment = reasoner.assess(features, symptom_region=region_code)
    except Exception as e:
        yield (f"Clinical reasoning failed: {e}", _BLANK, _BLANK, _BLANK, "", "", None, None, "", "", "", "", "")
        return

    status(f"MAS: {assessment.mas_score}/100 — {assessment.restriction_level} ({assessment.confidence} confidence)")

    # ── Step 3b: Pose-frame muscle overlay (falls back to anatomy art) ──────
    anatomy_region  = assessment.region if assessment.region != "unknown" else REGION_FULL
    _pf_list        = _last_results.get("pose_frames", [])
    # Use the LAST frame of the video — most representative end-position image
    _last_frame     = _pf_list[-1] if _pf_list else None
    _last_frame_bgr = _last_frame.annotated_image if _last_frame is not None else _last_results.get("thumbnail_bgr")
    _last_landmarks = _last_frame.landmarks if _last_frame is not None else None
    try:
        if _last_frame_bgr is not None and _last_landmarks is not None:
            anatomy_img = draw_pose_muscle_overlay(
                _last_frame_bgr, _last_landmarks, anatomy_region, assessment.mas_score
            )
        else:
            anatomy_img = draw_region_highlight(anatomy_region, assessment.mas_score)
        # Zoom view uses real Gray's anatomy image with muscle highlights + legend
        zoom_img = draw_zoom_view(anatomy_region, assessment.mas_score)
    except Exception as e:
        anatomy_img = None
        zoom_img    = draw_zoom_view(anatomy_region, assessment.mas_score)
        status(f"Anatomy map error: {e}")
    status(f"Anatomy overlay rendered for region: {anatomy_region}")

    # ── Step 4: Safety check ──────────────────────────────────────────────
    status("Red-flag safety check ...")
    checker = SafetyChecker(engine)
    try:
        safety = checker.check(features, assessment)
    except Exception as e:
        from modules.safety_layer import SafetyReport
        safety = SafetyReport()    # fallback to empty report
        status(f"Safety check error: {e}")

    status(f"Risk level: {safety.risk_level}")

    # ── Step 5: Documentation ─────────────────────────────────────────────
    status("Generating documentation (Module 2) ...")
    doc_gen = DocumentationGenerator(engine)
    try:
        docs = doc_gen.generate(features, assessment, safety)
    except Exception as e:
        from modules.documentation import ClinicalDocumentation
        docs = ClinicalDocumentation()
        status(f"Documentation error: {e}")

    status("Documentation complete.")

    # ── Step 5b: Muscle-level MedGemma explanation ────────────────────────────
    status("Muscle-level explanation ...")
    muscles = get_muscle_list(anatomy_region)
    try:
        muscle_md_text = reasoner.explain_muscles(features, anatomy_region, compute_mas(features), muscles)
    except Exception as e:
        muscle_md_text = (
            f"**Muscles involved in {anatomy_region.replace('_',' ').title()} region:**\n\n"
            + "\n".join(f"- {m}" for m in muscles)
            + f"\n\n*Explanation unavailable: {e}*"
        )
    status("Muscle explanation ready.")

    # ── Build outputs ─────────────────────────────────────────────────────
    gauge_img  = pain_gauge_chart(assessment.mas_score)
    rom_img    = rom_bar_chart(features)
    radar_img  = feature_radar_chart(features)
    # Cache raw objects for export buttons
    _last_results["features"]   = features
    _last_results["assessment"] = assessment
    _last_results["safety"]     = safety
    _last_results["docs"]       = docs
    # Clinical summary text
    region_label = _REGION_LABELS.get(
        anatomy_region, anatomy_region.replace("_", " ").title()
    )
    _joint_display = (
        "No significant movement detected in this recording"
        if features.most_restricted_joint == "none_detected"
        else f"{features.most_restricted_joint} ({features.affected_side} side)"
    )
    def _j(val: float, suffix: str = "°", warn_below: float = None, warn_above: float = None) -> str:
        """Format a joint value; returns N/A if sentinel 999 or -1 (not detected)."""
        if val >= 999 or val < 0:
            return "N/A"
        txt = f"{val:.1f}{suffix}"
        if warn_below is not None and val < warn_below:
            txt += "  [Restricted]"
        elif warn_above is not None and val > warn_above:
            txt += "  [Elevated]"
        return txt

    pain_prob_text = (
        f"## MAS: {assessment.mas_score}/100 — {assessment.restriction_level}\n\n"
        f"**Region:** {region_label}  |  "
        f"**Confidence:** {assessment.confidence}  |  "
        f"**Primary Joint:** {_joint_display}\n\n"
        f"**Clinical Interpretation:**\n{assessment.clinical_reasoning}\n\n"
        f"---\n"
        f"### Upper Limb ROM\n"
        f"| Joint | Left | Right | Normal |\n"
        f"|---|---|---|---|\n"
        f"| Shoulder Abduction | {_j(features.shoulder_abduction_left, warn_below=30)} "
        f"| {_j(features.shoulder_abduction_right, warn_below=30)} | max 180 deg |\n"
        f"| Shoulder Flexion | {_j(features.shoulder_flexion_left, warn_below=30)} "
        f"| {_j(features.shoulder_flexion_right, warn_below=30)} | max 180 deg |\n"
        f"| Elbow Flexion | {_j(features.elbow_flexion_left, warn_below=90)} "
        f"| {_j(features.elbow_flexion_right, warn_below=90)} | max 145 deg |\n"
        f"| Wrist Flexion (geo) | {_j(features.wrist_flexion_left, warn_above=100)} "
        f"| {_j(features.wrist_flexion_right, warn_above=100)} | ≤100 deg geo |\n"
        f"\n"
        f"### Lower Limb ROM\n"
        f"| Joint | Left | Right | Normal |\n"
        f"|---|---|---|---|\n"
        f"| Hip Flexion | {_j(features.hip_flexion_left, warn_below=80)} "
        f"| {_j(features.hip_flexion_right, warn_below=80)} | ≤120 deg |\n"
        f"| Hip Abduction | {_j(features.hip_abduction_left, warn_below=20)} "
        f"| {_j(features.hip_abduction_right, warn_below=20)} | ≤40 deg |\n"
        f"| Knee Flexion | {_j(features.knee_flexion_left, warn_below=90)} "
        f"| {_j(features.knee_flexion_right, warn_below=90)} | ≤135 deg |\n"
        f"| Ankle Dorsiflexion (geo) | {_j(features.ankle_dorsiflexion_left, warn_above=120)} "
        f"| {_j(features.ankle_dorsiflexion_right, warn_above=120)} | ≤120 deg geo |\n"
        f"> Note: N/A = joint landmark not visible in frame. Record the full body "
        f"(head-to-toe) to get lower-limb measurements.\n"
        f"\n"
        f"### Spine & Posture\n"
        f"| Signal | Value | Normal | Interpretation |\n"
        f"|---|---|---|---|\n"
        f"| Trunk Forward Lean | {_j(features.trunk_forward_lean_deg, warn_above=12)} | ≤12 deg | Lumbar guarding/antalgic posture |\n"
        f"| Trunk Lateral Lean | {_j(features.trunk_lateral_lean_deg, warn_above=15)} | ≤15 deg | Lateral scoliotic lean |\n"
        f"| Neck Lateral Flexion | {_j(features.neck_lateral_flexion_deg, warn_above=25)} | ≤25 deg | Head tilt |\n"
        f"\n"
        f"---\n"
        f"### Head & Face Signals\n"
        f"| Signal | Value |\n"
        f"|---|---|\n"
        f"| Head Tilt | {_j(features.head_tilt_deg, warn_above=12)} |\n"
        f"| Head Nod ROM | {_j(features.head_nod_deg, warn_above=18)} |\n"
        f"| Head Turn ROM | {_j(features.head_turn_deg, warn_above=20)} |\n"
        f"| Eye Squeeze | {'Yes' if features.eye_squeeze_detected else 'No'} |\n"
        f"| Mouth Tension | {'Yes' if features.mouth_tension_detected else 'No'} |\n"
        f"| Face Pain Score | {'No tension detected (0/100)' if features.face_pain_score < 5 else f'{features.face_pain_score:.0f}/100'} |\n"
    )

    _ddx = assessment.differential_diagnoses or _FALLBACK_DIFFERENTIALS.get(
        anatomy_region, _FALLBACK_DIFFERENTIALS[REGION_FULL]
    )
    diff_diag_text = (
        f"## Differential Diagnoses — {region_label}\n\n"
        + "\n".join(f"- {d}" for d in _ddx)
    )

    # ── Baseline deviation ─────────────────────────────────────────────────
    _baseline = load_baseline()
    _deviation_md = format_deviation_table(features, _baseline)
    pain_prob_text += f"\n\n---\n\n{_deviation_md}"

    # Safety alert text
    safety_text = (
        f"## Risk Level: {safety.risk_level}\n\n"
        + ("**URGENT EVALUATION REQUIRED**\n\n" if safety.urgent else "")
        + ("**Red Flags Identified:**\n" + "\n".join(f"- {f}" for f in safety.red_flags) + "\n\n"
           if safety.red_flags else "No critical red flags detected.\n\n")
        + f"**Recommendation:** {safety.recommendation}"
    )

    progress_text = (
        f"**Analysis complete**\n\n"
        f"- **Region assessed:** {region_label}\n"
        f"- **MAS score:** {assessment.mas_score}/100 — {assessment.restriction_level}\n"
        f"- **Risk level:** {safety.risk_level}\n"
        f"- **Frames analysed:** {len(pose_frames)}\n"
        f"- **Confidence:** {assessment.confidence}"
    )

    yield (
        progress_text,
        gauge_img,
        rom_img,
        radar_img,
        pain_prob_text,
        diff_diag_text,
        anatomy_img,
        zoom_img,
        muscle_md_text,
        safety_text,
        docs.soap_note,
        docs.patient_explanation,
        docs.rehab_suggestions,
    )


# Build Gradio UI

_CSS = """
.pain-high   { color: #e74c3c; font-weight: bold; }
.pain-medium { color: #f39c12; font-weight: bold; }
.pain-low    { color: #2ecc71; font-weight: bold; }
#title       { text-align: center; }
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(title=UI_TITLE) as demo:

        gr.Markdown(f"# {UI_TITLE}", elem_id="title")
        gr.Markdown(UI_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Video Input")

                with gr.Tabs() as input_tabs:
                    with gr.TabItem("Upload File", id="tab_upload"):
                        upload_input = gr.Video(
                            label="Upload Patient Movement Clip",
                            sources=["upload"],
                            height=260,
                            format="mp4",
                        )

                    with gr.TabItem("Record from Camera", id="tab_camera"):
                        camera_input = gr.Video(
                            label="Record Patient Movement",
                            sources=["webcam"],
                            height=260,
                            include_audio=False,
                        )
                        gr.Markdown(
                            "**Tip:** Position the patient so their full body is visible. "
                            "Record 5-15 seconds of the movement being assessed. "
                            "Click Stop to finish recording, then click Analyse."
                        )

                analyze_btn = gr.Button("Analyse Movement", variant="primary", size="lg")
                symptom_region_dd = gr.Dropdown(
                    choices=[
                        "Auto-detect",
                        "Shoulder / Upper Arm",
                        "Elbow / Wrist",
                        "Hip / Knee",
                        "Lower Back / Lumbar",
                        "Neck / Cervical",
                    ],
                    value="Auto-detect",
                    label="Reported Symptom Region (optional)",
                    info="Select the body region the patient reported symptoms in. Overrides auto-detection.",
                )
                with gr.Row():
                    clear_btn     = gr.Button("Clear",            size="sm", variant="secondary")
                    baseline_btn  = gr.Button("Record Baseline",  size="sm", variant="secondary")
                baseline_status = gr.Markdown(
                    value=f"_{baseline_summary()}_",
                    label="Baseline Status",
                )
                status_box  = gr.Markdown(
                    label="Pipeline Status",
                    value="Upload a video or record from camera, then click Analyse."
                )

            with gr.Column(scale=2):
                with gr.Tabs():

                    # ── Tab 1: Movement Score ────────────────────────────────
                    with gr.TabItem("Movement Score"):
                        with gr.Row():
                            gauge_out = gr.Image(label="MAS Gauge", show_label=True)
                            radar_out = gr.Image(label="Signal Radar", show_label=True)
                        pain_md   = gr.Markdown()

                    # ── Tab 2: Anatomy Map ────────────────────────────────
                    with gr.TabItem("Anatomy Map"):
                        gr.Markdown(
                            "### Detected Region Highlighted on Anatomical Map\n"
                            "Colour = restriction severity: "
                            "Normal | Mild | Moderate | Severe"
                        )
                        with gr.Row():
                            anatomy_map_out = gr.Image(
                                label="Full Body — Affected Region",
                                show_label=True,
                            )
                            zoom_out = gr.Image(
                                label="Zoom View — Involved Muscles",
                                show_label=True,
                            )
                        muscle_md = gr.Markdown(label="MedGemma Muscle Explanation")

                    # ── Tab 2: Biomechanics ──────────────────────────────
                    with gr.TabItem("Biomechanics"):
                        rom_out   = gr.Image(label="Range of Motion Chart", show_label=True)
                        diag_md   = gr.Markdown()

                    # ── Tab 3: Safety ────────────────────────────────────
                    with gr.TabItem("Safety"):
                        safety_md = gr.Markdown()

                    # ── Tab 4: SOAP Note ─────────────────────────────────
                    with gr.TabItem("SOAP Note"):
                        soap_out  = gr.Markdown()

                    # ── Tab 5: Patient Info ──────────────────────────────
                    with gr.TabItem("Patient Info"):
                        patient_out = gr.Markdown()

                    # ── Tab 6: Rehab Plan ────────────────────────────────
                    with gr.TabItem("Rehab Plan"):
                        rehab_out = gr.Markdown()
        # ── Export section ─────────────────────────────────────────────────
        with gr.Row():
            export_json_btn = gr.Button("Export JSON Report", size="sm", variant="secondary")
            export_md_btn   = gr.Button("Export Markdown Report", size="sm", variant="secondary")
        with gr.Row():
            export_file_out = gr.File(label="Download Report", visible=False)
        # ── Wiring ───────────────────────────────────────────────────────
        _all_outputs = [
            status_box,
            gauge_out, rom_out, radar_out,
            pain_md, diag_md,
            anatomy_map_out, zoom_out, muscle_md,
            safety_md,
            soap_out, patient_out, rehab_out,
        ]

        # Transcode immediately on upload so the browser can preview the video
        def _transcode_upload(video_path):
            if video_path is None:
                return None
            return _ensure_h264(video_path)

        upload_input.upload(
            fn=_transcode_upload,
            inputs=[upload_input],
            outputs=[upload_input],
        )

        analyze_btn.click(
            fn=analyze_video,
            inputs=[upload_input, camera_input, symptom_region_dd],
            outputs=_all_outputs,
        )

        # ── Record-Baseline handler ────────────────────────────────────────────
        def _record_baseline():
            if "features" not in _last_results:
                return "_Run **Analyse** first, then click Record Baseline._"
            path = save_baseline(_last_results["features"])
            return f"Baseline saved ({path.split('data/')[-1]}) — re-run Analyse to see deviations."

        baseline_btn.click(
            fn=_record_baseline,
            inputs=[],
            outputs=[baseline_status],
        )

        def _clear_all():
            """Reset both video inputs and all output panels."""
            _last_results.clear()
            empty = (
                "Upload a video or record from camera, then click Analyse.",
                None, None, None,   # gauge, rom, radar
                "", "",             # pain_md, diag_md
                None, None, "",     # anatomy_map, zoom, muscle_md
                "",                 # safety_md
                "", "", "",         # soap, patient, rehab
            )
            # Keep baseline_status reflecting persisted baseline (not cleared on Reset)
            return (None, None) + empty + (f"_{baseline_summary()}_",)

        clear_btn.click(
            fn=_clear_all,
            inputs=[],
            outputs=[upload_input, camera_input] + _all_outputs + [baseline_status],
        )

        # ── Export handlers ──────────────────────────────────────────────
        def _export_json():
            if not _last_results:
                return gr.update(visible=False, value=None)
            report = build_json_report(
                _last_results["features"], _last_results["assessment"],
                _last_results["safety"],   _last_results["docs"],
            )
            path = save_json_report(report)
            return gr.update(visible=True, value=path)

        def _export_md():
            if not _last_results:
                return gr.update(visible=False, value=None)
            text = build_text_report(
                _last_results["features"], _last_results["assessment"],
                _last_results["safety"],   _last_results["docs"],
            )
            path = save_text_report(text)
            return gr.update(visible=True, value=path)

        export_json_btn.click(fn=_export_json, inputs=[], outputs=[export_file_out])
        export_md_btn.click(fn=_export_md,   inputs=[], outputs=[export_file_out])

        gr.Markdown(
            "> ⚠️ **Disclaimer:** MoveSense AI is a research/demonstration tool. "
            "It does not replace professional clinical diagnosis. Always consult a "
            "qualified healthcare provider for medical decisions."
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=UI_SERVER_PORT, share=False, inbrowser=True, theme=gr.themes.Soft())
