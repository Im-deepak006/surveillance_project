# Project Audit

Read-only inventory of the Moment-DETR based video retrieval project. **No scripts were run for this audit**, so execution commands below are documented entry points, not verified run instructions. Historical claims of successful execution are identified as documentation claims.

## 1. Folder tree

Bulky/generated content is omitted (virtual environment, `outputs/`, model files, video files, and annotation/feature data contents).

```text
moment-detr-baseline/
├── app.py
├── download_vocab.py
├── docs/                 setup, status, project notes, changelog
├── extension/            full_pipeline.py, attribute_matching.py
├── moment_detr/          model, training, inference, dataset and utility modules
│   └── scripts/          train.sh, pretrain.sh, inference.sh
├── res/                  model overview image
├── run_on_video/         raw-video inference and feature extraction
│   ├── clip/             bundled CLIP implementation/tokenizer
│   ├── example/          example query file (video omitted)
│   └── moment_detr_ckpt/ checkpoint and training/evaluation artifacts
├── standalone_eval/      evaluator and sample predictions/metrics
├── utils/                shared utilities
├── data/                 QVHighlights annotations and data documentation (omitted)
├── README.md
└── requirements.txt
```

## 2. Python scripts and purposes

- `app.py` — Streamlit upload UI; calls video inference, attribute reranking, validation, and clip extraction.
- `download_vocab.py` — helper to download the vocabulary/tokenizer asset.
- `extension/full_pipeline.py` — CLI-style composition of Moment-DETR prediction and attribute reranking.
- `extension/attribute_matching.py` — YOLO person detection/cropping and CLIP attribute scoring to rerank candidate moments.
- `run_on_video/run.py` — raw video and text-query Moment-DETR inference entry point (`MomentDETRPredictor`).
- `run_on_video/clip_extractor.py` — video duration, candidate-window validation, timestamp formatting and clip extraction helpers.
- `run_on_video/data_utils.py`, `run_on_video/model_utils.py` — video feature/data preparation and inference model setup.
- `moment_detr/train.py`, `moment_detr/inference.py` — model training and feature-based split inference.
- `moment_detr/model.py`, `transformer.py`, `matcher.py`, `position_encoding.py`, `text_encoder.py`, `postprocessing_moment_detr.py`, `span_utils.py`, `start_end_dataset.py`, `config.py`, `misc.py` — model architecture, matching, encoding, dataset/configuration, and postprocessing components.
- `standalone_eval/eval.py`, `standalone_eval/utils.py` — prediction evaluation and metric utilities.
- `utils/basic_utils.py`, `model_utils.py`, `temporal_nms.py`, `tensor_utils.py`, `windows_utils.py` — shared data, model, tensor, temporal-NMS, and window utilities.
- `run_on_video/clip/clip.py`, `model.py`, `simple_tokenizer.py` — bundled CLIP model and tokenizer code.

## 3. Entry points and documented commands

The interactive application entry point is `app.py`. The model demo entry point is `run_on_video/run.py`; training and split inference are launched through shell scripts.

From the project root (`moment-detr-baseline`), in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="."
streamlit run app.py
```

Other documented commands:

```powershell
$env:PYTHONPATH="."
python run_on_video\run.py
bash moment_detr/scripts/train.sh
bash moment_detr/scripts/inference.sh <checkpoint-path> val
bash standalone_eval/eval_sample.sh
```

The shell-script commands require a Bash environment. The repository’s README describes Unix-style `PYTHONPATH` commands. The setup/changelog documentation says the demo was successfully run historically; this audit did not rerun it.

## 4. Dependencies and versions

`requirements.txt` pins these principal packages: `torch==2.13.0`, `torchvision==0.28.0`, `torchaudio==2.11.0`, `numpy==2.2.6`, `pandas==2.3.3`, `scikit-learn==1.7.2`, `scipy==1.15.3`, `easydict==1.13`, `ffmpeg-python==0.2.0`, `ftfy==6.3.1`, `regex==2026.7.19`, `tqdm==4.70.0`, `tensorboard==2.21.0`, `tabulate==0.10.0`, `ipython==8.39.0`, and `pillow==12.3.0`. It also pins supporting packages (including `filelock`, `fsspec`, `Jinja2`, `networkx`, `protobuf`, `sympy`, `Werkzeug`, and `typing_extensions`); see `requirements.txt` for the complete list.

`streamlit>=1.32` is the only unpinned direct requirement. The extension imports `open_clip`, `ultralytics`, `cv2` (OpenCV), and `PIL`, but `open_clip` and `ultralytics` are not declared in this requirements file. FFmpeg/ffprobe executables are also needed for documented video processing; they are system tools, not Python package pins. The upstream README specifies Python 3.7/PyTorch 1.9, while the project setup guide specifies Python 3.10; this is a documentation/version compatibility discrepancy, not a tested compatibility statement.

## 5. Models and local weight files

| Model/use | Weight filename | Size |
|---|---|---:|
| Moment-DETR checkpoint for raw-video inference | `run_on_video/moment_detr_ckpt/model_best.ckpt` | 16,952,721 bytes (16.16 MiB) |
| YOLO11n person detector | `yolo11n.pt` | 5,613,764 bytes (5.35 MiB) |
| CLIP used by raw video pipeline | `ViT-B/32` | No local weight file identified; code expects/downloads model assets |
| OpenCLIP attribute scoring | `ViT-B-32`, pretrained=`openai` | No local weight file identified; loaded via `open_clip` |

Weights were not copied or loaded during this audit; sizes are filesystem metadata.

## 6. Dataset and annotation formats

The bundled dataset is **QVHighlights**, from the upstream project release. Its README points to the QVHighlights raw-video archive and describes query-level moment annotations in JSON Lines (`.jsonl`) format. Fields include query/video IDs, query text, duration, relevant time windows, relevant 2-second clip IDs, and three annotator saliency ratings per clip. Test release annotations omit ground truth; a separate `highlight_test_with_gt.jsonl` is present locally.

| File | Rows | Split/content |
|---|---:|---|
| `data/highlight_train_release.jsonl` | 7,218 | Train with moment/saliency labels |
| `data/highlight_val_release.jsonl` | 1,550 | Validation with labels |
| `data/highlight_test_release.jsonl` | 1,542 | Test release without labels |
| `data/highlight_test_with_gt.jsonl` | 1,542 | Test annotations with labels |
| `data/subs_train.jsonl` | 235,878 | ASR/subtitle pretraining records |
| `data/moment_detr_with_attributes.jsonl` | 301 | Attribute-annotated subset; label fields plus `attributes` |

Row counts were obtained by reading nonempty JSONL records. The documentation points to the upstream QVHighlights video download; the project’s own demo video and generated session videos are excluded from this inventory.

## 7. Existing training/evaluation metrics

These are **pre-existing artifacts**, not results produced or independently verified by this audit.

- `run_on_video/moment_detr_ckpt/train.log.txt` contains logged training epochs through epoch 200. For example, its final recorded training loss is `loss_overall 1.5462` at epoch 200. It also includes validation metrics alongside some epochs.
- `run_on_video/moment_detr_ckpt/inference_hl_val_test_code_preds_metrics.json` records validation metrics for the supplied checkpoint: moment retrieval full-split mAP `30.58`, R1@0.5 `53.23`, R1@0.7 `34.00`; highlight mAP/Hit@1 are reported separately at Fair/Good/VeryGood thresholds.
- `standalone_eval/sample_val_preds_metrics_raw.json` contains a separate sample validation metric record (full mAP `32.20`, R1@0.5 `53.94`, R1@0.7 `34.84`).
- `run_on_video/moment_detr_ckpt/eval.log.txt` has logged epoch evaluations. Metric definitions and prediction format are documented in `standalone_eval/README.md`.

The differently named metrics files contain different values; they should not be treated as one experiment without checking their provenance/configuration.

## 8. Existing outputs and implemented pipeline stages

- The repository has a trained Moment-DETR checkpoint, training/evaluation logs, validation predictions, and sample evaluator outputs.
- `outputs/` currently contains 12 session directories with 100 extracted clips and 12 uploaded originals (112 MP4 files total). Session/file names are omitted to avoid exposing any user video metadata.
- Code is present for: uploaded-video handling in the UI; video duration checks; Moment-DETR temporal localization; YOLO person detection/cropping; CLIP attribute scoring and candidate reranking; candidate bounds/padding validation; clip extraction; displaying/downloading results in the UI.
- This describes code/artifacts found, not verified runtime behavior. The docs say the pretrained baseline demo was run successfully in an earlier setup; they mark surveillance-video trials, extension integration/attribute matching, evaluation, UI, report, and final presentation as incomplete in `docs/PROJECT_STATUS.md`. Generated UI clips show prior clip-generation activity, but do not by themselves validate the full pipeline’s correctness.

## 9. Git status and recent history

Git repository root is `FYP/` (the project is in its `moment-detr-baseline/` subdirectory). At audit time, `git status --short` reported no tracked modifications. Untracked paths include `moment-detr-baseline/download_vocab.py`, `moment-detr-baseline/venv/`, this audit file, and nine generated output session directories. Twelve output session directories are present overall; three older sessions are tracked in Git. The repository’s ownership required a per-command `safe.directory` override to read Git metadata; no Git configuration was changed.

Recent commits:

```text
c25f96b Second review
a4fecc8 First extension run
3f72786 Fix - project repository structure
a38a9d4 Add FYP project files
d138721 first commit
```

## 10. Extended code, model, and integration audit

### Repository/environment facts

- Repository root: `FYP/`; application/project root: `FYP/moment-detr-baseline/`.
- Runtime examined: project-local `venv`; Python 3.10.8. The selected environment is a Windows virtual environment.
- Executables `ffmpeg.exe` and `ffprobe.exe` are discoverable on this machine. Their availability is machine-specific; the code calls them through `ffmpeg-python`.
- The code uses relative paths such as `run_on_video/moment_detr_ckpt/model_best.ckpt` and `yolo11n.pt`; commands therefore assume the current directory is the project root. `extension/attribute_matching.py` also contains a machine-specific example path (`D:\FYP\sample.mp4`), but it is only in its `__main__` example block.
- No ActPrompt references or tracker implementation were found in the project source. No tracking IDs, track association, person re-identification, or representative track-crop stage is implemented.
- The checkpoint's saved `opt.json` names `annotations/highlight_train.jsonl`, `annotations/highlight_val.jsonl`, and `features/clip_features`; the present repository instead has `data/highlight_train_release.jsonl`, `data/highlight_val_release.jsonl`, and no checked-in `features/` tree. These paths impede reproduction of training from the saved configuration. The checkpoint is sufficient evidence of an available model artifact, not evidence that this checkout can recreate its training run.

### Actual application call path

```text
app.py
  ├─ load_predictor()
  │    └─ run_on_video/run.py: MomentDETRPredictor
  │         ├─ run_on_video/data_utils.py: sample video at 0.5 fps and encode with bundled OpenAI CLIP ViT-B/32
  │         └─ run_on_video/model_utils.py: load model_best.ckpt and build Moment-DETR
  └─ run_pipeline()
       ├─ run_on_video/clip_extractor.py: duration check
       ├─ predictor.localize_moment(): ranked temporal windows + saliency scores
       ├─ extension/attribute_matching.py: sample frames, YOLO person crops, OpenCLIP text/image similarity
       ├─ run_on_video/clip_extractor.py: normalize/rank windows and FFmpeg clip extraction
       └─ app.py: render and offer clips/ZIP downloads
```

`extension/full_pipeline.py` is a separate script-level composition of inference and reranking. It is **not** what the Streamlit app calls, and it prints ranked results without extracting clips or displaying a UI. The Streamlit app calls the same component functions directly.

### Important source files and data contracts

| Path | Main functions/classes and contract | Findings |
|---|---|---|
| `app.py` | `load_predictor()`, `run_pipeline()`; accepts uploaded video bytes + one text query; produces session files and rendered MP4 clips | Relative checkpoint/output paths assume project-root working directory. A session saves the uploaded original video and clips under `outputs/`; no cleanup/retention policy is present. The app limits duration to about 152 seconds, while the model itself asserts at most 75 two-second segments (150 seconds). |
| `run_on_video/run.py` | `MomentDETRPredictor.localize_moment(video_path, query_list)`; returns per-query dicts containing ranked `[start,end,score]` windows and saliency list | Model samples at 0.5 fps and represents times in seconds; no query ID is returned for local app use. Max length is 150 seconds. This is temporal moment retrieval, not a person detector or tracker. |
| `run_on_video/data_utils.py` | `ClipFeatureExtractor`, `VideoLoader`; video path/text list → CLIP features | Uses FFmpeg/ffprobe and bundled `run_on_video/clip`; video decode failure returns a dict sentinel that later code expects to be a tensor, so the failure may surface downstream unclearly. |
| `run_on_video/model_utils.py` | `build_inference_model(ckpt_path)`; checkpoint → Moment-DETR module | Uses `torch.load(..., weights_only=False)`, which loads a pickle-based checkpoint. Only load trusted local checkpoint files. Saved options determine dimensions/architecture. |
| `extension/attribute_matching.py` | module-level YOLO/OpenCLIP initialization; `extract_frames`, `detect_person_crops`, `score_attributes`, `rerank_with_attributes` | The input `attribute_phrases` is expected to be phrase(s), but the app passes the entire natural-language query as one phrase. No parser separates clothing/object attributes from action/location. YOLO detects COCO `person` class only; it does not detect “backpack” or identify/track a person across moments. Reranking is a fixed 0.5 temporal + 0.5 attribute weighted average with no tuning/evaluation evidence. If no person crop is found, attribute score remains 0. Frames without detections therefore get a systematic attribute-score penalty. |
| `extension/full_pipeline.py` | `run_moment_detr`, `full_pipeline`; video + query + checkpoint → printed ranked windows | It passes the full query as the attribute phrase, just like the UI path. No clip extraction, tracking, or UI in this script. |
| `run_on_video/clip_extractor.py` | `get_video_duration`, `normalize_and_validate[_scored]`, `extract_clip`; window dictionaries → validated bounds and FFmpeg MP4s | Extraction writes new files. This module was statically inspected only; the audit did not invoke FFmpeg. |
| `moment_detr/train.py`, `moment_detr/inference.py`, `moment_detr/config.py` | Feature-based training/evaluation entry points and options | The shell scripts expect feature directories under `features/`. Training writes checkpoints/logs/results. Not run during this audit. |
| `standalone_eval/eval.py`, `standalone_eval/utils.py` | JSONL predictions + annotation JSONL → moment-retrieval/highlight metrics | Evaluator supports split annotation paths, but there is no current saved test-split prediction/metric pair establishing held-out performance. |

The remaining files under `moment_detr/` implement the Moment-DETR model (transformer, matching, positional encodings, span/dataset utilities); `utils/` contains shared JSON, tensor, NMS, window and model helpers; `run_on_video/clip/` is the bundled OpenAI CLIP implementation. Calls into them come through the entry points above. No notebooks were found in the relevant project tree.

### Model and training provenance

- **Moment-DETR**: the local `model_best.ckpt` is loaded by `app.py` and `run_on_video/run.py`. `opt.json` specifies a 10-query Moment-DETR with 512-dimensional CLIP video/text features, `video_tef`, `max_v_l=75`, 2-second clips, and a 200-epoch run. `run_on_video/moment_detr_ckpt/README.md` describes the checkpoint as trained from scratch on CLIP image/text features without ASR pretraining; this is repository documentation. The checkpoint/log/config and matching val prediction provide supporting artifacts, but there is no independently verifiable dataset feature cache or training rerun here. Do not call this surveillance-domain fine-tuning: no such run or dataset is present.
- **OpenAI CLIP `ViT-B/32`**: used by the baseline feature extractor for full-frame video and text. It is loaded from the bundled implementation and may fetch weights if absent from cache. No local CLIP weight file was identified in the project tree.
- **YOLO11n**: `yolo11n.pt` (5,613,764 bytes) is configured in the attribute module and the code selects COCO class 0 (`person`). The code does not record a training run or custom surveillance dataset. Treat it as a supplied/pretrained detector asset; project-specific fine-tuning is not evidenced.
- **OpenCLIP `ViT-B-32`, pretrained `openai`**: used for crop/text attribute similarity. Weights are resolved by `open_clip` and were not found as a local project file. The call may need a cached or downloaded external model asset.
- No tracker/re-identification model or ActPrompt model/checkpoint is present. No custom detector-training labels, class map, training CSV, or surveillance training configuration was found.

The app imports `extension.attribute_matching` at module load. That module constructs YOLO and OpenCLIP globally, so simply starting/importing the UI may initialize those models and OpenCLIP may attempt to obtain pretrained weights. This was not attempted because it can involve a download. The project-local environment has metadata/import specs for `torch 2.13.0`, `torchvision 0.28.0`, `torchaudio 2.11.0`, `streamlit 1.63.0`, `open_clip 3.3.0`, `ultralytics 8.4.145`, `opencv-python 5.0.0.93`, and the pinned core packages. Package availability does not establish GPU/CUDA compatibility or successful application startup.

## 11. Dataset, output, and evaluation validity

- QVHighlights train/val/test row counts are reported in section 6. `highlight_test_release.jsonl` has 1,542 queries without GT; `highlight_test_with_gt.jsonl` has 1,542 queries with GT. The local presence of test labels makes offline test evaluation possible in principle, subject to the dataset license and responsible use. However, no matching test predictions/metrics are present in the saved artifacts inspected here. Therefore held-out test performance is **not established**.
- `run_on_video/moment_detr_ckpt/inference_hl_val_test_code_preds.jsonl` parses as 1,550 prediction records, matching the validation split size; despite `val_test` in its filename, its record count and companion metrics indicate validation, not evidence of test evaluation. `standalone_eval/sample_val_preds.jsonl` also parses as 1,550 records. Both use the expected prediction keys (`qid`, `query`, `vid`, `pred_relevant_windows`, `pred_saliency_scores`).
- The checkpoint metric JSON reports full-split MR mAP averaged across IoU thresholds 0.5–0.95 = 30.58, MR R1@0.5 = 53.23, R1@0.7 = 34.00, plus highlight mAP and Hit@1 at Fair/Good/VeryGood thresholds. The sample metric JSON reports different values (full MR mAP 32.20; R1@0.5 53.94; R1@0.7 34.84). Their provenance/configurations are not demonstrated to be identical; do not treat the difference as an improvement or compare as one experiment.
- Missing evaluation: YOLO precision/recall/mAP on surveillance data; person tracking metrics (IDF1/HOTA/MOTA or equivalent); person-attribute retrieval accuracy; end-to-end query-to-moment retrieval on surveillance data; held-out surveillance temporal localization; and ablation/validation of the 0.5/0.5 reranking weights. Existing QVHighlights temporal/highlight metrics do not measure those tasks.
- Existing `outputs/` contains 12 session directories, 100 files in `clips/`, and 12 originals (112 MP4s total); all checked files are nonzero bytes. This is only a filesystem check, not media decoding or proof the clips correspond to correct moments. Three session directories are committed, and nine are untracked. No video content, names, or metadata are reproduced in this report.

## 12. Read-only validation performed and blockers

- **Static syntax validation:** parsed 32 project Python files with Python 3.10.8 using `ast.parse`; 0 syntax errors. This does not validate imports or runtime behavior.
- **Dependency availability:** checked package import specifications and installed distribution metadata in the existing `venv` without importing the application or loading models. Required packages for the extension are installed in that environment even though `open_clip`, `ultralytics`, and OpenCV are absent from `requirements.txt`.
- **Data/artifact structure:** counted JSONL rows, parsed the two saved prediction JSONL files (1,550 records each, zero parse errors), read checkpoint option JSON and metric JSON, checked output MP4 file sizes, and checked Git status/history. No private video metadata or record contents were included.
- **Not run:** app startup, any model imports/weight loading, inference, clip extraction, training, evaluation, video decoding, CUDA checks. These can download/cache weights, use substantial compute, or write files. As a result, no runtime or end-to-end success claim is made here.

### Practical readiness assessment

The repository contains a usable-looking baseline UI path for **short video temporal retrieval plus an experimental person-crop reranker and clip output**, with existing generated clips and validation artifacts. The source does not implement the intended long-surveillance-person tracking workflow. Current hard limit is about 150 seconds, person identity is not maintained across frames, natural-language attribute parsing is absent, there is no surveillance-domain training/evaluation, and held-out test performance is unreported. These are the main scope gaps to address before presenting the system as the full project goal.
