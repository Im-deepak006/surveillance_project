# End-to-End QA Report

**Final status: PARTIAL PASS**

The real-video processing function completed successfully with the configured models and produced validated clips. The Streamlit server also started and passed its health check. The browser-based upload-and-click flow was not exercised because the available browser automation could not select a local file; that UI interaction remains unverified.

## Environment and models

- Project: `D:\jegan_repo\FYP\moment-detr-baseline`
- OS/shell: Windows, PowerShell
- Python: 3.10.8 in the project virtual environment
- PyTorch: 2.13.0+cpu; CUDA unavailable, so inference ran on CPU
- Streamlit: 1.63.0
- OpenCLIP: 3.3.0
- Ultralytics: 8.4.145
- OpenCV: 5.0.0
- FFmpeg/ffprobe: available on PATH
- Moment-DETR checkpoint: `run_on_video/moment_detr_ckpt/model_best.ckpt` (16,952,721 bytes)
- YOLO checkpoint: `yolo11n.pt` (5,613,764 bytes)
- OpenAI CLIP `ViT-B/32` and OpenCLIP `ViT-B-32` weights were already cached locally. The end-to-end process was run with Hugging Face offline mode enabled; no weights were downloaded.

The real run used the actual Moment-DETR predictor, its CLIP video/text feature extractor, YOLO person detector, OpenCLIP scorer, and FFmpeg extraction. No model was mocked or fine-tuned. The status widget alone was replaced with a no-op collector because the shared processing function was invoked outside a Streamlit script context.

## Commands executed

From the project folder:

```powershell
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Result: **24 tests passed**.

To verify the bundled video's frame count at Moment-DETR's 0.5 FPS sampling rate:

```powershell
ffmpeg -hide_banner -v info -i run_on_video/example/RoripwjYFp8_60.0_210.0.mp4 -vf fps=0.5 -f null NUL
```

Result: 75 sampled frames.

Streamlit startup and health check:

```powershell
.\venv\Scripts\streamlit.exe run app.py --server.headless true --server.address 127.0.0.1 --server.port 8502
Invoke-WebRequest -Uri http://127.0.0.1:8502/_stcore/health -TimeoutSec 20
```

Result: startup log reported the app URL and the health endpoint returned `HTTP 200: ok`.

For the real-video run, a PowerShell here-string piped a Python harness to `python -B -`. It imported `app.run_pipeline`, set a unique `outputs/qa_end_to_end_<id>` output root, set `YOLO_CONFIG_DIR` inside that new root, enabled `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and passed the bundled MP4 and query below. The harness did not replace inference, scoring, ranking, or extraction. Its JSON result is saved at `outputs/qa_end_to_end_8f7d0a1f17/test_summary.json`.

## Real-video result

- Input: `run_on_video/example/RoripwjYFp8_60.0_210.0.mp4`, 10,404,801 bytes, 150.0165 seconds according to ffprobe.
- Query: “Find the person wearing a white shirt who makes pizza and cuts it up.”
- Parsed appearance text: “person wearing a white shirt”; the full original query was passed to Moment-DETR.
- Runtime: 122.21 seconds on CPU.
- Temporal candidates: 10 produced, all 10 valid and within the source duration.
- Attribute matching: all 10 candidates received a non-null appearance similarity from actual YOLO person crops and OpenCLIP scoring.
- Extraction: **10 of 10** clips exist, have nonzero size, and can be probed by ffprobe. Each clip's duration agrees with its selected interval within ordinary container/encoding tolerance.
- Highest-ranked interval: `[106.6285, 130.3430]`, temporal confidence `0.9884`, appearance similarity `0.2695`.
- The bundled annotation marks the pizza-making moment as `[106, 122]` seconds. The top result has temporal IoU about `0.632` with that annotation. This is a single-example overlap check, not an evaluation metric or evidence of a performance improvement.
- Output directory: `outputs/qa_end_to_end_8f7d0a1f17/session_b01504a41c/`; clips are named `clips/clip_001_*.mp4` through `clips/clip_010_*.mp4`. The summary JSON records each exact path, size, interval, score, and probed duration.

## Automated coverage

The 24-test `unittest` suite covers:

- appearance-only, action-only, combined, ambiguous, empty, and unsupported queries;
- crop scoring, no detections, absent appearance evidence, malformed score output, ranking ties, and temporal-score precedence;
- timestamp formatting, window clamping and boundaries, invalid windows, missing video input, and actual FFmpeg extraction from the bundled video;
- missing checkpoint failure and the 150-second model-duration limit, including the 150.0165-second container metadata case.

No separate project test suite or pytest configuration was found. The automated suite does not simulate an absent Python package; required packages were present and used during the real run. Missing/corrupt video handling was exercised with a missing input path; unsupported codec behavior was not separately fixture-tested.

## Bug found and fix

The first app-pipeline attempt rejected the bundled sample because its container duration is 150.0165 seconds, slightly above the checkpoint's nominal 150-second limit. FFmpeg sampling showed the file actually yields 75 two-second frames, which is within the model's positional-embedding limit. The app now allows up to 0.1 seconds of finite metadata rounding via `is_within_duration_limit`; the predictor's own 75-frame assertion remains in place. Regression tests cover the boundary and non-finite durations.

Files changed during this QA pass:

- `app.py` — use the duration-limit helper when checking the checkpoint's supported input duration.
- `run_on_video/clip_extractor.py` — add the finite duration/tolerance helper.
- `tests/test_clip_extractor.py` — add boundary, timestamp, missing-input, and real extraction checks.
- `tests/test_model_failures.py` — check that a missing checkpoint fails explicitly.
- `tests/test_query_logic.py` — add unsupported-query and crop-score input cases.

The existing implementation changes in `extension/query_logic.py`, `extension/attribute_matching.py`, `extension/full_pipeline.py`, and ranking logic in `run_on_video/clip_extractor.py` were reviewed. The complete processing path described above was exercised without changing the model or training configuration. `git diff --check` passed.

## Streamlit UI and limitations

Streamlit startup and its health endpoint were verified. The underlying `app.run_pipeline()` function called by the Analyze button was run end to end with a real uploaded-file object and actual local models. Local-file selection through the browser uploader and the rendered results/download buttons were not verified, so the complete UI interaction is not claimed as tested.

The parser is intentionally narrow and does not understand arbitrary natural language. Attribute matching is person detection and appearance similarity scoring, not person tracking. This project run did not use ActPrompt, train or fine-tune a model, or measure a retrieval benchmark beyond the one annotated example overlap. Existing training/evaluation metrics remain those recorded in the project's audit and evaluation artifacts; no new aggregate metrics were generated.

The run's output is separate from existing sessions. Failed preliminary attempts also left their new QA directories and copied input videos in place; they were not deleted. A dependency inspection also generated the untracked `Ultralytics/settings.json` under the project after the default user settings directory returned an access error; that file was retained. No pre-existing source, checkpoint, dataset, or output was overwritten.
