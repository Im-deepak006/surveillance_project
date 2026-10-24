# Action and Attribute Retrieval Evaluation

## Status

**PARTIAL PASS.** The actual local model pipeline processed one annotated local video, scored action evidence, generated ranked candidate timestamps, and extracted clips that were verified as readable and within the source duration. Regression tests pass. Streamlit startup, health endpoint, and rendered controls were verified, but the browser file chooser could not be automated, so upload-and-Analyze remains unverified. This report does not claim a general retrieval improvement.

## Environment and command record

Project: `D:\jegan_repo\FYP\moment-detr-baseline` on Windows PowerShell. The project Python is 3.10.8, PyTorch 2.13.0+cpu, Streamlit 1.63.0, OpenCLIP 3.3.0, Ultralytics 8.4.145, and OpenCV 5.0.0. CUDA was unavailable; inference ran on CPU. FFmpeg/ffprobe were available. The configured Moment-DETR checkpoint and YOLO checkpoint were local; cached CLIP/OpenCLIP weights were used. The evaluation reports `weights_downloaded: false`.

Exact successful verification commands:

```powershell
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\venv\Scripts\python.exe -B -m py_compile app.py run_on_video/model_utils.py extension/query_logic.py extension/attribute_matching.py extension/full_pipeline.py run_on_video/clip_extractor.py standalone_eval/action_attribute_metrics.py standalone_eval/run_local_action_attribute_evaluation.py
.\venv\Scripts\python.exe -B standalone_eval\run_local_action_attribute_evaluation.py
git diff --check
```

The unit test command passed **33 tests**. Python compilation passed. `git diff --check` passed. The evaluation command used actual configured models and the same `app.run_pipeline()` processing function used by the Streamlit Analyze Video flow; it did not mock the models. Its artifacts are in `outputs/action_attribute_eval_bf906f3158/`. Two earlier evaluation attempts failed before pipeline startup due to a Python import-path collision; their empty output folders were left intact and not used for results.

The UI command was:

```powershell
.\venv\Scripts\streamlit.exe run app.py --server.headless true --server.address 127.0.0.1 --server.port 8503
```

The first in-sandbox attempt was inaccessible. Relaunching outside the sandbox produced `Uvicorn server started on 127.0.0.1:8503`; `Invoke-WebRequest -Uri http://127.0.0.1:8503/_stcore/health -UseBasicParsing` returned `ok`, and browser inspection showed the app title, ranking control, file-upload widget, query field, and Analyze button. The browser's supported filechooser event timed out for the Streamlit upload button, so no upload or UI-triggered inference was completed. The upload-and-click flow is **unverified**, not a pipeline failure. Launch command for a local interactive session:

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

## Data and protocol

The only suitable local annotated example found was `run_on_video/example/RoripwjYFp8_60.0_210.0.mp4` (10,404,801 bytes, 150.016533 seconds) and `run_on_video/example/queries.jsonl`. It contains one query, “Chef makes pizza and cuts it up.”, with the annotated interval `[106, 122]` seconds. The query yields an action cue and no appearance cue. The dataset has no appearance labels, so temporal+appearance and all-component evaluation cannot establish the utility of appearance ranking.

One shared set of 10 Moment-DETR proposals was scored for each ablation. Component scores were normalized to within-query descending tied rank percentiles and averaged without tuned weights. Missing component evidence was omitted from that candidate's fusion denominator. Metrics are top-1 Recall at IoU 0.5 and 0.7 and mean top-1 IoU. These four ablations are descriptive for one query only; no confidence interval or generalization claim is warranted.

| Ranking mode | R@1 IoU .5 | R@1 IoU .7 | Mean IoU@1 | Top proposal | Top IoU |
|---|---:|---:|---:|---|---:|
| Temporal | 0.0 | 0.0 | 0.0000 | 51.0595–65.5663 s | 0.0000 |
| Temporal + appearance | 0.0 | 0.0 | 0.0000 | 51.0595–65.5663 s | 0.0000 |
| Temporal + action | 1.0 | 0.0 | 0.5298 | 111.5118–125.7947 s | 0.5298 |
| Temporal + appearance + action | 1.0 | 0.0 | 0.5298 | 111.5118–125.7947 s | 0.5298 |

In this sample, action rank fusion changed the top-ranked proposal to one overlapping the annotation above 0.5 IoU. With one query and no appearance cue, this is an observed example, not a demonstrated performance improvement. The action score is static-frame OpenCLIP similarity, not motion recognition.

## Pipeline and output checks

| Stage | Result |
|---|---|
| Input decoding and duration probe | Passed; source duration 150.016533 s |
| Moment-DETR inference | Passed with local checkpoint; 10 candidate intervals |
| Action/appearance query parsing | Parsed action `makes pizza and cuts it up`; appearance absent; original query retained for temporal grounding |
| Semantic scoring | Passed; 10 candidates received frame-level action scores; no appearance scoring was applicable |
| Candidate ranking | All four ablations emitted ranked records with raw scores, component ranks, and fused score |
| Timestamp validation | Passed; all candidate times were within the source video's bounds |
| Clip extraction | Passed; 10 clips generated, each non-empty and probeable |
| Repeatability check | Passed; a repeated inference produced identical proposals after loading the model in evaluation mode |
| Streamlit UI | Not verified; local app endpoint was unreachable in this sandbox attempt |

End-to-end pipeline time was **16.003 s** on CPU: video probe 0.114 s, model loading 1.743 s, temporal inference 2.734 s, semantic scoring 7.159 s, ranking/validation under 0.001 s, and clip extraction 4.244 s. A separate repeated inference took 2.732 s. Timings are for this 150-second source and current machine only.

All 10 extracted files exist, have nonzero sizes, and were opened/probed successfully. Their measured durations correspond to the selected intervals within normal container/encoding rounding. The top fused proposal's clip is `outputs/action_attribute_eval_bf906f3158/pipeline/session_b5c9e234d7/clips/clip_001_6d6495.mp4` (929,882 bytes, 14.3143 s, interval 111.5118–125.7947 s). Full per-clip paths, durations, byte sizes, raw scores, rankings, and metrics are recorded in the JSON results file.

Machine-readable result: `outputs/action_attribute_eval_bf906f3158/ACTION_ATTRIBUTE_EVALUATION_RESULTS.json`.

## Implementation changes and limits

- `extension/query_logic.py` adds bounded lexical cue parsing, action scoring aggregation, and tied rank-percentile fusion. It is not a general natural-language parser.
- `extension/attribute_matching.py` reuses existing OpenCLIP to score sampled full frames for action cues, and existing person-crop appearance scoring where an appearance cue exists.
- `app.py` exposes the four ablations while preserving the original query for Moment-DETR and reports component scores/timings.
- `extension/full_pipeline.py` and `run_on_video/clip_extractor.py` carry the component scores and fused ranking through extraction.
- `run_on_video/model_utils.py` sets the loaded Moment-DETR model to evaluation mode; the previous training-mode dropout caused non-repeatable predictions.
- `standalone_eval/action_attribute_metrics.py` and `standalone_eval/run_local_action_attribute_evaluation.py` implement the local annotated ablation protocol and JSON output.
- `tests/test_query_logic.py`, `tests/test_model_failures.py`, and `tests/test_action_attribute_metrics.py` cover parser/ranking edge cases and metric/failure behavior.

There was no training or fine-tuning, no new detector, no person tracking, and no ActPrompt implementation. No weights were downloaded. Existing modifications and untracked files were preserved; this evaluation did not clean up prior user outputs. The working tree includes unrelated/pre-existing audit documents, QA outputs, `Ultralytics/`, `download_vocab.py`, and `venv/`; inspect `git status --short` before staging anything.

## Final assessment

The local inference-to-clip pipeline and its regression tests pass for the bundled annotated sample. The full request is **PARTIAL PASS** because the Streamlit upload UI was not exercised and the available evidence is one action-only query. Appearance contribution, robustness across videos, and general retrieval metrics remain unestablished.
