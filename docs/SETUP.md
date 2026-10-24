# Moment-DETR Baseline Setup Guide

## 1. Clone the Repository

```bash
git clone <repository-url>
cd moment_detr
```

---

## 2. Install Python

Recommended version:

```
Python 3.10.x
```

---

## 3. Create Virtual Environment

Windows PowerShell

```powershell
py -3.10 -m venv .venv
```

Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 5. Install FFmpeg

Download FFmpeg.

Add

```
<ffmpeg>\bin
```

to Windows PATH.

Verify

```powershell
ffmpeg -version
ffprobe -version
```

---

## 6. Run Demo

```powershell
$env:PYTHONPATH="."
python run_on_video\run.py
```

Expected output:

```
Build models...
Loading feature extractors...
Loading trained Moment-DETR model...
Run prediction...
Predicted moments...
```