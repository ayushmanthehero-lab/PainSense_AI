"""
main.py – PainSense AI Entry Point
────────────────────────────────────
Usage:
    # Launch the full Gradio web dashboard
    python main.py

    # CLI single-video analysis (headless / script mode)
    python main.py --video path/to/clip.mp4

    # CLI mode – skip documentation (faster)
    python main.py --video path/to/clip.mp4 --no-docs
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ── Force UTF-8 output so emoji in print() don't crash on Windows cp1252 ──────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Resolve paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Ensure ffmpeg (installed via winget) is on PATH ───────────────────────────
_FFMPEG_CANDIDATES = [
    # winget default install location
    r"C:\Users\Kumar Deblin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin",
    # common manual installs
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
]
for _p in _FFMPEG_CANDIDATES:
    if os.path.isfile(os.path.join(_p, "ffmpeg.exe")) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
        print(f"[PainSense] ffmpeg found and added to PATH: {_p}")
        break


def _cli_analyze(video_path: str, generate_docs: bool = True):
    """
    Headless CLI pipeline for a single video file.
    Prints a structured JSON report to stdout.
    """
    from modules.pose_estimator   import PoseEstimator
    from modules.feature_extractor import FeatureExtractor
    from modules.medgemma_engine  import MedGemmaEngine
    from modules.clinical_reasoning import ClinicalReasoner
    from modules.safety_layer     import SafetyChecker
    from modules.documentation    import DocumentationGenerator

    print(f"\n{'═'*60}")
    print(f"  PainSense AI  –  Video Analysis")
    print(f"{'═'*60}")
    print(f"  Video : {video_path}")
    print(f"{'─'*60}\n")

    # ── Pose Estimation ──────────────────────────────────────────────────────
    print("Step 1/5  Pose estimation …")
    estimator = PoseEstimator()
    pose_frames, thumbnail = estimator.process_video(video_path)
    if not pose_frames:
        print("ERROR: No pose detected in video.")
        sys.exit(1)
    print(f"         → {len(pose_frames)} frames with pose detected.")

    # ── Feature Extraction ───────────────────────────────────────────────────
    print("Step 2/5  Feature extraction …")
    extractor = FeatureExtractor()
    features  = extractor.extract(pose_frames)
    print(f"         → Heuristic score: {features.heuristic_pain_score:.1f}/100")

    # ── MedGemma Clinical Reasoning ──────────────────────────────────────────
    print("Step 3/5  MedGemma clinical reasoning …")
    engine    = MedGemmaEngine()
    reasoner  = ClinicalReasoner(engine)
    assessment= reasoner.assess(features)
    print(f"         → MAS: {assessment.mas_score}/100  ({assessment.restriction_level})")

    # ── Safety Check ─────────────────────────────────────────────────────────
    print("Step 4/5  Red-flag safety check …")
    checker = SafetyChecker(engine)
    safety  = checker.check(features, assessment)
    print(f"         → Risk level: {safety.risk_level}")

    # ── Documentation (optional) ─────────────────────────────────────────────
    docs = None
    if generate_docs:
        print("Step 5/5  Generating clinical documentation …")
        gen  = DocumentationGenerator(engine)
        docs = gen.generate(features, assessment, safety)
        print("         → Done.")
    else:
        print("Step 5/5  Documentation skipped (--no-docs).")

    # ── Print Report ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  CLINICAL REPORT")
    print(f"{'═'*60}")

    report = {
        "movement_assessment": {
            "mas_score":              assessment.mas_score,
            "confidence":             assessment.confidence,
            "restriction_level":      assessment.restriction_level,
            "clinical_reasoning":     assessment.clinical_reasoning,
            "differential_diagnoses": assessment.differential_diagnoses,
            "affected_structures":    assessment.affected_structures,
        },
        "biomechanical_features": features.to_prompt_dict(),
        "safety": {
            "urgent":             safety.urgent,
            "risk_level":         safety.risk_level,
            "red_flags":          safety.red_flags,
            "recommendation":     safety.recommendation,
            "refer_immediately":  safety.refer_immediately,
        },
    }
    if docs:
        report["documentation"] = {
            "soap_note":           docs.soap_note,
            "patient_explanation": docs.patient_explanation,
            "rehab_suggestions":   docs.rehab_suggestions,
        }

    print(json.dumps(report, indent=2))

    # Save to file
    out_path = os.path.splitext(video_path)[0] + "_painsense_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved → {out_path}\n")


def _launch_dashboard():
    """Launch the Gradio web dashboard."""
    import gradio as gr
    from ui.dashboard import build_ui
    from config import UI_SERVER_PORT
    demo = build_ui()
    print(f"\n  PainSense AI Dashboard starting on http://localhost:{UI_SERVER_PORT}\n")
    demo.launch(server_port=UI_SERVER_PORT, share=False, inbrowser=True, theme=gr.themes.Soft())


def main():
    parser = argparse.ArgumentParser(
        prog="PainSense AI",
        description="Offline AI-powered pain assessment from patient movement.",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a video file for CLI analysis mode."
    )
    parser.add_argument(
        "--no-docs", dest="no_docs", action="store_true",
        help="Skip SOAP note / documentation generation (CLI mode only)."
    )
    args = parser.parse_args()

    if args.video:
        if not os.path.isfile(args.video):
            print(f"ERROR: Video file not found: {args.video}")
            sys.exit(1)
        _cli_analyze(args.video, generate_docs=not args.no_docs)
    else:
        _launch_dashboard()


if __name__ == "__main__":
    main()
