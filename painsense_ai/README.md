# PainSense AI -- Musculoskeletal Movement Assessment

AI-assisted musculoskeletal movement assessment using **MediaPipe BlazePose** for biomechanical
analysis and **MedGemma** (Google, 4-bit) for clinical interpretation.
Upload a short patient movement clip -- the system scores movement restriction objectively
and generates clinical explanations, differential diagnoses, anatomy highlights, SOAP notes,
and a personalised rehab plan -- all running fully offline on a consumer GPU.

---

## Pipeline


Video Upload / Camera Recording
        |
        v  ffmpeg H.264 transcode (universal browser compatibility)
        |
        v  MediaPipe BlazePose Full -- 33-landmark pose estimation
        |
        v  Feature Extractor -- joint angles, ROM, asymmetry, velocity, guarding, face pain
        |
        v  Movement Region Classifier -- auto-detects: shoulder / elbow / hip-knee / lumbar / cervical / full-body
        |
        v  MAS (Movement Abnormality Score) -- deterministic rule-based 0-100 score
        |
        +- MAS < 15  ->  Normal (no LLM call)
        |
        v  MedGemma 4B-IT (4-bit NF4) -- region-locked clinical reasoning
        |   +-- Clinical reasoning and differential diagnoses
        |   +-- Safety layer -- red-flag detection and risk level
        |   +-- Anatomy map -- muscle-level explanation
        |   +-- SOAP note + patient explanation + rehab plan
        |
        v  Gradio Dashboard (7 tabs) + JSON / Markdown export


---

## Project Structure


painsense_ai/
│
├── main.py                      # Entry point – launches Gradio dashboard
├── config.py                    # Global settings: model paths, ROM norms, MAS thresholds, UI configs
├── requirements.txt
├── pose_landmarker_full.task    # MediaPipe BlazePose Full model (bundled)
│
├── modules/
│   ├── pose_estimator.py        # MediaPipe wrapper + per-frame landmark extraction
│   ├── feature_extractor.py     # ClinicalFeatureVector:
│   │                            #   - Joint angles
│   │                            #   - ROM deficit
│   │                            #   - Asymmetry
│   │                            #   - Velocity reduction
│   │                            #   - Guarding detection
│   │                            #   - Head/trunk posture
│   │                            #   - Facial pain indicators
│   │
│   ├── movement_classifier.py   # Auto-detect active body region
│   ├── pain_scorer.py           # Deterministic MAS engine (rule-based)
│   ├── medgemma_engine.py       # MedGemma 4B-IT loader (4-bit NF4, lazy singleton)
│   ├── clinical_reasoning.py    # Region-locked prompts → MedGemma → ClinicalAssessment
│   ├── safety_layer.py          # Red-flag detection + risk stratification
│   └── documentation.py         # SOAP note + patient explanation + rehab plan
│
├── ui/
│   └── dashboard.py             # Gradio dashboard (7 tabs, H.264 transcode on upload)
│
├── utils/
│   ├── visualization.py         # MAS gauge, ROM bar chart, signal radar chart
│   ├── anatomy_map.py           # Anatomy overlay – region highlight + muscle zoom
│   ├── baseline.py              # Per-patient baseline recording + deviation comparison
│   └── export.py                # JSON and Markdown report builder
│
└── assets/
    └── anatomy/                 # Gray anatomical reference images (7 body regions)


---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Movement Score** | MAS gauge (0-100), signal radar chart, clinical reasoning, differential diagnoses |
| **Anatomy Map** | Full-body region highlight + anatomy zoom with muscle overlay and legend |
| **Biomechanics** | ROM bar chart, joint angles table, asymmetry and velocity metrics |
| **Safety** | Red-flag checklist, risk level, urgency recommendation |
| **SOAP Note** | Structured Subjective / Objective / Assessment / Plan |
| **Patient Info** | Plain-language patient explanation |
| **Rehab Plan** | Personalised exercise and rehabilitation recommendations |

---

## Movement Abnormality Score (MAS)

The MAS is computed **entirely by deterministic biomechanical rules** -- MedGemma cannot
override or re-score it. MedGemma only provides the clinical interpretation.

| MAS Range | Label |
|-----------|-------|
| 0 - 14 | Normal |
| 15 - 24 | Mild Restriction |
| 25 - 49 | Moderate Restriction |
| 50 - 100 | Severe Restriction |

---

## Clinical Signals Extracted

| Signal | Description |
|--------|-------------|
| Shoulder abduction / flexion (L/R) | 3-point joint angle -- hip->shoulder->wrist / elbow |
| Elbow flexion (L/R) | shoulder->elbow->wrist |
| Wrist flexion (L/R) | geometric elbow->wrist->index angle |
| Hip flexion / abduction (L/R) | shoulder->hip->knee; lateral leg angle from vertical |
| Knee flexion (L/R) | hip->knee->ankle |
| Trunk forward lean | sagittal spine angle (>12 deg = guarding signal) |
| Trunk lateral lean | coronal spine angle (>8 deg = scoliotic lean) |
| Neck lateral flexion / head tilt | cervical region assessment |
| ROM deficit % | deviation from standard reference values |
| Bilateral asymmetry % | left vs right angle difference |
| Velocity reduction % | temporal landmark displacement |
| Guarding | postural heuristic (trunk lean + velocity collapse) |
| Face pain score | facial landmark strain composite |
| Rolling median smoothing | 5-frame temporal smoothing on all joint angles |

---

## Requirements

- Python **3.12**
- NVIDIA GPU with 4 GB+ VRAM (GTX 1650 tested)
- CUDA **11.8+**
- ffmpeg in PATH (for H.264 video transcode)

---

## Installation

`powershell
# 1. Create and activate virtual environment
python -m venv .venv312
.venv312\Scripts\Activate.ps1

# 2. Install PyTorch for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install all other dependencies
pip install -r requirements.txt

# 4. Install ffmpeg -- https://www.gyan.dev/ffmpeg/builds/ -- add bin/ to PATH
`

---

## Model Setup

MedGemma is loaded from a **local directory** (not downloaded at runtime).
Place the model files in a sibling folder called med gemma/ next to painsense_ai/:

`
Desktop/
+-- med gemma/            <- root workspace folder
    +-- med gemma/        <- MedGemma weights: config.json, *.safetensors, tokenizer.*
    +-- painsense_ai/     <- this repo
`

Model: google/medgemma-4b-it loaded at 4-bit NF4 quantisation via itsandbytes.
The model path is configured in config.py -> MODEL_DIR.

---

## Usage

`powershell
cd painsense_ai
python main.py
`

Opens Gradio dashboard at http://localhost:7860.

1. Upload or record a video (any format -- automatically transcoded to H.264 on upload)
2. Optionally select the reported symptom region from the dropdown
3. Click **Analyse Movement**
4. Results populate across all 7 tabs in ~7-13 minutes

---

## Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1650 4 GB | RTX 3060+ |
| RAM | 16 GB | 32 GB |
| Storage | 15 GB (model + env) | SSD |
| OS | Windows 10+ | Windows 11 |

---

## Disclaimer

PainSense AI is a **research and demonstration tool**.
It does **not** constitute medical advice and must **not** replace professional clinical diagnosis.
Always consult a qualified healthcare provider for medical decisions.
