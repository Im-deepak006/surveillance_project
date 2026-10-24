# CHANGELOG

## Version 1.0 - Baseline Setup

Date: August 2026

### Environment

- Installed Python 3.10
- Created isolated virtual environment (.venv)
- Installed all required Python dependencies
- Installed FFmpeg
- Configured FFmpeg in Windows PATH

---

### Compatibility Fixes

#### 1. Device Selection

Original:

```python
device="cuda"
```

Modified:

```python
device="cuda" if torch.cuda.is_available() else "cpu"
```

Reason:

Allows the project to run on systems without an NVIDIA GPU.

---

#### 2. PyTorch 2.x Checkpoint Loading

Original:

```python
torch.load(ckpt_path, map_location="cpu")
```

Modified:

```python
torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False
)
```

Reason:

PyTorch 2.6+ changed the default checkpoint loading behavior.

---

### Verification

Successfully executed:

```
python run_on_video/run.py
```

Verified:

- CLIP loaded
- Feature extractor loaded
- Moment-DETR checkpoint loaded
- Demo inference completed successfully

---

### Next Development Phase

- Run inference on custom surveillance videos
- Study Moment-DETR architecture
- Integrate YOLO
- Build attribute-aware retrieval