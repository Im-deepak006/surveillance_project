# Moment-DETR Baseline for Attribute-Aware Surveillance Retrieval

## Project Overview

This repository contains a verified working baseline implementation of **Moment-DETR**, prepared as the foundation for our final-year research project.

The objective of this repository is to provide a reproducible environment where the original Moment-DETR model can run successfully before extending it with attribute-aware surveillance retrieval.

---

## Base Paper

**Moment-DETR: End-to-End Natural Language Video Moment Retrieval**

This repository uses the official Moment-DETR implementation as the baseline.

---

## Project Status

✅ Repository cloned

✅ Environment configured

✅ Python 3.10 virtual environment

✅ Dependencies installed

✅ FFmpeg configured

✅ Pretrained checkpoint loaded

✅ Demo inference successfully verified

---

## Repository Structure

```
moment_detr/
│
├── data/
├── moment_detr/
├── run_on_video/
├── standalone_eval/
├── utils/
├── requirements.txt
├── SETUP.md
├── README.md
└── LICENSE
```

---

## Environment

- Python 3.10
- PyTorch 2.13 (CPU)
- FFmpeg
- CLIP
- Moment-DETR

---

## Run Demo

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
$env:PYTHONPATH="."
python run_on_video\run.py
```

---

## Verified Output

The baseline successfully predicts temporal moments from a natural language query.

Example:

Query:

```
Chef makes pizza and cuts it up.
```

Ground Truth:

```
[106,122]
```

Predicted:

```
[108.7863,125.417]
Confidence: 0.9851
```

---

## Future Work

This repository will be extended with:

- YOLO Object Detection
- Person Attribute Extraction
- Color Detection
- Attribute Matching
- Surveillance Clip Retrieval
- User Interface

---

## Team

Final Year Project Team

2026