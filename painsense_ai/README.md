# PainSense AI

**Offline AI-powered pain assessment from patient movement**

---

## Architecture

```
Camera / Video Clip
        ↓
Pose Estimation  (MediaPipe BlazePose Full)
        ↓
Biomechanical Feature Extraction
        ↓
Structured Clinical Feature Vector
        ↓
MedGemma Module 1:  Pain Probability + Differential Diagnosis
        ↓
MedGemma Safety Layer:  Red-Flag Detection
        ↓
MedGemma Module 2:  SOAP Note + Patient Explanation + Rehab Plan
        ↓
Gradio Dashboard  /  JSON Report
```

---

## Project Structure

```
painsense_ai/
├── main.py                      # Entry point (dashboard or CLI)
├── config.py                    # All settings & paths
├── requirements.txt
├── modules/
│   ├── pose_estimator.py        # MediaPipe BlazePose wrapper
│   ├── feature_extractor.py     # Joint angles, ROM, asymmetry, velocity, guarding
│   ├── medgemma_engine.py       # MedGemma model loader (4-bit quantised)
│   ├── clinical_reasoning.py   # MedGemma Module 1 – pain probability
│   ├── safety_layer.py          # MedGemma red-flag detection
│   └── documentation.py         # MedGemma Module 2 – SOAP + rehab
├── ui/
│   └── dashboard.py             # Gradio web dashboard (6 tabs)
└── utils/
    └── visualization.py         # Pain gauge, ROM bars, radar charts
```

---

## Requirements

- Python 3.10+
- NVIDIA GTX 1650 (4 GB VRAM) — runs MedGemma at 4-bit quantisation
- CUDA 11.8+ / cuDNN

---

## Installation

```powershell
# 1. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install PyTorch with CUDA (adjust cu118/cu121 to match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install all other dependencies
pip install -r requirements.txt
```

---

## Usage

### Launch Gradio Dashboard

```powershell
python main.py
```

Opens at `http://localhost:7860`

### CLI – single video analysis

```powershell
python main.py --video path\to\patient_clip.mp4
```

Saves a JSON report alongside the video file.

```powershell
# Skip SOAP/rehab generation (faster)
python main.py --video clip.mp4 --no-docs
```

---

## Clinical Signals Extracted

| Signal | Method |
|--------|--------|
| Shoulder abduction (L/R) | 3-point joint angle: hip→shoulder→wrist |
| Elbow flexion (L/R) | shoulder→elbow→wrist |
| Hip flexion (L/R) | shoulder→hip→knee |
| Knee flexion (L/R) | hip→knee→ankle |
| ROM deficit % | Deviation from normal reference values |
| Movement asymmetry % | Left vs right angle difference |
| Velocity reduction % | Temporal landmark displacement |
| Guarding | Postural collapse heuristic |

---

## MedGemma Usage (Two Modules)

| Module | Input | Output |
|--------|-------|--------|
| **Module 1** | Clinical feature vector | Pain probability, confidence, differential diagnoses, red flags |
| **Safety Layer** | Assessment summary | Red-flag detection, urgency, risk level |
| **Module 2** | Full assessment | SOAP note, patient explanation (plain language), rehab plan |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1650 (4 GB) | RTX 3060+ |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB (model) | SSD |
| OS | Windows 10+ | Windows 11 |

---

## Disclaimer

PainSense AI is a **research and demonstration tool**. It does not constitute medical advice
and must not replace professional clinical diagnosis.

---
