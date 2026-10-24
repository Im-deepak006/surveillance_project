# Soft-Biometric Retrieval Evaluation

## Outcome

**PARTIAL PASS.** A real local video was processed end to end with the configured Moment-DETR, YOLO11n, OpenCLIP, and FFmpeg components. Ten candidate clips were extracted and validated. The parser, person-crop OpenCLIP path, and YOLO object-category path were exercised. Attribute-related ranking effectiveness remains unevaluated because the only raw-video query has no appearance or carried-object annotation and the 301-row attribute subset has no matching raw videos in this checkout.

## Data coverage

| Source | Available records | What they support |
|---|---:|---|
| `run_on_video/example/RoripwjYFp8_60.0_210.0.mp4` | 1 local raw video, 150.016533 s | Real model pipeline and clip validation |
| `run_on_video/example/queries.jsonl` | 1 query and 1 relevant interval `[106, 122]` | Action-containing temporal evaluation only |
| `data/moment_detr_with_attributes.jsonl` | 301 rows with structured attribute labels | Annotation inventory and cue-coverage audit only; associated raw videos are unavailable locally |

The bundled query is “Chef makes pizza and cuts it up.” A bounded lexical scan of the 301 annotation queries recognizes appearance cues in 30, action cues in 104, both in 8, and carried-object cues in 3. These are parser coverage counts, not a labeled evaluation subset or model accuracy figures. The structured annotations include `shirt_color`, `pant_color`, `shoe_color`, `bag`, `cap`, `object`, and `action`; these values are ground-truth labels in the file, not predictions made by this application. They also contain gender and age-group fields, which this system does not infer or use.

Although these annotation rows contain useful labels, no corresponding raw source videos were found under `run_on_video/`. They cannot be used to evaluate visual appearance or object evidence in the real-video pipeline without those videos. The single raw example has no appearance or carried-object label.

## A–E ablation results

The five methods use the same 10 Moment-DETR proposals for the one annotated query. Component values are normalized within the query as tied rank percentiles and averaged unweighted over available evidence. No fusion weights were tuned. Metrics are top-1 Recall at IoU 0.5 and 0.7 and mean top-1 IoU.

| Method | R@1 IoU .5 | R@1 IoU .7 | Mean IoU@1 | Query count |
|---|---:|---:|---:|---:|
| A. Temporal | 0.0 | 0.0 | 0.0000 | 1 |
| B. Temporal + appearance | 0.0 | 0.0 | 0.0000 | 1 |
| C. Temporal + action | 1.0 | 0.0 | 0.5298 | 1 |
| D. Temporal + appearance + action | 1.0 | 0.0 | 0.5298 | 1 |
| E. Temporal + appearance + action + detected object | 1.0 | 0.0 | 0.5298 | 1 |

For this query, the appearance cue is absent, so B behaves like A. It also has no object cue, so E behaves like D. The action rank changes the top proposal from `[51.0595, 65.5663]` (IoU 0.0) to `[111.5118, 125.7947]` (IoU 0.5298). This repeats the previous one-query descriptive observation; it does not establish a general accuracy improvement. Appearance-containing, combined, and carried-object query subsets each have **zero** locally video-grounded evaluation queries, so their subgroup metrics are reported as unavailable (`null`) in the results JSON.

## Real-model checks and resources

- **Input:** 10,404,801-byte bundled video, 150.016533 seconds.
- **Models:** local Moment-DETR checkpoint, local YOLO11n (`yolo11n.pt`), cached CLIP/OpenCLIP; no weights were downloaded. YOLO model names were read at runtime and matched the configured COCO class IDs for backpack, umbrella, handbag, suitcase, bottle, cup, laptop, cell phone, and book.
- **Device:** CPU; CUDA unavailable in this environment.
- **Pipeline runtime:** 16.732 s: probe 0.079 s, model load 1.914 s, temporal inference 2.554 s, semantic scoring 7.855 s, ranking/validation 0.0002 s, extraction 4.315 s. A repeat Moment-DETR inference took 2.662 s and reproduced the proposal set exactly; the model was in evaluation mode.
- **Clips:** 10 of 10 files were nonempty and probeable. All timestamps were within the source duration, and measured clip durations were consistent with the selected intervals.
- **Appearance diagnostic:** actual YOLO person crops and OpenCLIP scoring ran for three sampled frames on the top fused candidate. The raw OpenCLIP cosine for the unlabeled diagnostic phrase “person wearing a blue shirt” was `0.250262`. This is not a probability, label, or accuracy measurement.
- **Object diagnostic:** the same three frames were checked with actual YOLO11n for a backpack. No backpack was detected; the recorded state is `no_matching_detection`, with no score. This is unknown visual evidence, not a confirmed mismatch and not a labeled false negative.
- **Streamlit:** the app started on local port 8504, rendered the upload/query controls, and exposed all five ranking choices including E. Browser file selection was not automatable, so upload-and-Analyze in the browser remains unverified. The underlying `app.run_pipeline()` path was exercised directly with the actual video and model pipeline.

The top fused clip is `outputs/soft_biometric_eval_20f4aa90c0/pipeline/session_1c62d3e044/clips/clip_001_d217e2.mp4` (929,882 bytes; 14.3143 s). Per-clip paths, byte sizes, durations, and timestamps appear in the JSON file. An initial complete E-mode run was also preserved at `outputs/soft_biometric_eval_c4f348f079/`; the second unique run added the appearance-crop and supported-class diagnostic. Neither run overwrote existing project outputs.

## Automated checks

Exact commands run:

```powershell
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\venv\Scripts\python.exe -B -m py_compile app.py extension/query_logic.py extension/attribute_matching.py extension/full_pipeline.py run_on_video/clip_extractor.py standalone_eval/action_attribute_metrics.py standalone_eval/run_local_action_attribute_evaluation.py tests/test_query_logic.py tests/test_action_attribute_metrics.py
.\venv\Scripts\python.exe -B standalone_eval\run_local_action_attribute_evaluation.py --output-prefix soft_biometric_eval
git diff --check
```

The complete suite passed **48 tests**. Compilation and `git diff --check` passed. Tests cover supported and unsupported cue parsing, carried-object synonyms, appearance-only/action-only/combined queries, invalid inputs, empty crops, detector errors, absent evidence, tied/missing component ranking, metadata preservation through clip validation, metric subgroups, timestamp boundaries, clip extraction, and deterministic model evaluation mode.

The updated Streamlit launch command is:

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

## Machine-readable results and remaining data request

Detailed per-query predictions, all five ablations, component scores/ranks, query subsets, stage timings, clip validation, and the unlabeled object/person-crop diagnostic are in [SOFT_BIOMETRIC_EVALUATION_RESULTS.json](outputs/soft_biometric_eval_20f4aa90c0/SOFT_BIOMETRIC_EVALUATION_RESULTS.json).

To evaluate the soft-biometric contribution, provide the raw videos corresponding to the attribute-annotated queries, their ground-truth time intervals, and person-linked appearance/object labels for each relevant interval. Carrying labels should identify the person-object relationship rather than merely object presence. Until those data are available, appearance/object impact and cross-video generalization remain unproven.
