# Soft-Biometric and Attribute-Aware Retrieval: Implementation Report

## Status

**PARTIAL: the implementation and local model paths are verified; attribute-related retrieval quality cannot be evaluated from the available raw-video annotations.** Appearance and detected-object evidence now have separate fields, availability statuses, and rank-fusion components. The actual person-crop OpenCLIP path and local YOLO path both ran on the bundled video. The only raw-video query has no appearance or carried-object ground truth, so no appearance or object accuracy claim is supported.

## Findings before this implementation

- The Streamlit app runs `app.run_pipeline()`: save uploaded video → probe duration → pass the original complete query to Moment-DETR → parse cues → sample candidate frames → optional YOLO person crops/OpenCLIP appearance and action scoring → rank fusion → interval validation → FFmpeg extraction.
- `extension/full_pipeline.py` is a separate CLI composition. The UI calls the same semantic scorer and rank helper directly; it does not call `full_pipeline.py`.
- Before this work the parser could recognize some clothing and carrying wording, but clothing and carried-object cues shared one appearance string. The YOLO path detected only people. No person-crop attributes were extracted as structured labels, and there was no object-category score in ranking.
- Person-crop OpenCLIP cosine similarity and frame/action OpenCLIP cosine similarity were already separate raw scores. They were converted to within-query rank percentiles by the A–D fusion modes; they are not calibrated probabilities. The previous evaluation had one action-only annotated query.
- The repository has 301 rows in `data/moment_detr_with_attributes.jsonl` with structured annotation fields including `shirt_color`, `pant_color`, `shoe_color`, `bag`, `cap`, `object`, and `action`. These are labels in the annotation file, not model predictions. The raw videos for this attribute subset are not present in `run_on_video/`; the only raw video there is the bundled 150-second example.
- The source annotation for that example has one interval, `[106, 122]` seconds, for “Chef makes pizza and cuts it up.” It contains neither an appearance cue nor a carried-object label.

## Attribute support and limits

| Evidence type | Query parsing | Visual evidence | Compared with query | Can affect ranking | Limits |
|---|---|---|---|---|---|
| Clothing / general appearance | Bounded clothing vocabulary; preserves the complete query and leaves unknown descriptions unset | YOLO person boxes; usable crops are at least 16×16 pixels | Existing OpenCLIP text/image cosine on detected person crops; strongest finite crop similarity per candidate | Yes, B and D/E | One whole-person text comparison; no explicit upper/lower garment classifier, structured clothing labels, calibrated probability, or quality/occlusion correction |
| Carried-object category | Supported object names are extracted only when a bounded cue such as “carrying,” “holding,” “with,” or “wearing” links the phrase to a supported category | Local YOLO11n COCO class detections in sampled candidate frames | Exact requested category match; mean of the strongest detector confidence for each category observed | Yes, E | Category co-occurrence is not proof that a detected person carried the object; no box-to-person association or temporal tracking |
| Action | Existing bounded action phrase parser | Existing sampled full-frame OpenCLIP similarity | Mean finite frame/text cosine | Yes, C–E | Static-image similarity is not motion recognition |
| Structured labels / demographics / identity | Not inferred | None | None | No | Existing attribute labels are annotations only. The application does not infer gender, age, identity, or person tracking/re-identification. |

The local YOLO11n class map was checked during the real run. Supported object categories are `backpack`, `umbrella`, `handbag`, `suitcase`, `bottle`, `cup`, `laptop`, `cell phone`, and `book`. Synonyms such as `rucksack`, `purse`, `mug`, and `mobile phone` map to those COCO categories. Ambiguous generic `bag` and unsupported objects are not assigned a guessed detector class.

`carried_object_score` is the mean raw YOLO confidence over requested categories that were actually detected. If a requested category is not detected, it is recorded as unavailable evidence (`no_matching_detection`) and omitted from that candidate’s fusion denominator. Detector errors, empty frames, missing person crops, and invalid appearance scores are represented by explicit status strings. An absent detection is not assigned a zero-valued mismatch score.

All selected ranking modes use the existing transparent tied-rank-percentile fusion: descending component scores are converted to per-query ranks in `[0,1]`, then averaged unweighted over available evidence. Raw component scores are retained. No weights were tuned against these evaluation examples.

## Previously present versus added

**Previously present:** Moment-DETR temporal proposals; original-query temporal grounding; YOLO person detection/crops; OpenCLIP raw person-crop appearance cosine; sampled-frame action cosine; A–D temporal/appearance/action ablations; bounds checking and FFmpeg clip extraction; the evaluation utilities for one local annotated example.

**Added here:** separate `carried_object_texts`; clothing versus carried-category cue separation; YOLO11n COCO object-category detections alongside person crops in one detector pass per sampled frame; positive-only carried-object candidate scores and explicit missing-evidence states; E ranking mode; object score/rank/status preservation through clip validation and results UI; A–E evaluation output and appearance/action/combined/object query-subset metrics; regression tests.

No new model, model download, training, fine-tuning, demographic classifier, person tracker, or identity recognition was added.

## Verification

The local E-mode evaluation ran through the same `app.run_pipeline()` function called by the Analyze Video button. It used the local Moment-DETR checkpoint, local YOLO11n weights, cached CLIP/OpenCLIP models, and FFmpeg; inference ran on CPU. No weights were downloaded. The original temporal proposal set was repeated and identical after the existing evaluation-mode fix.

On a three-frame, unlabeled diagnostic interval selected from the run’s top fused proposal, actual YOLO11n and person-crop OpenCLIP scoring ran. OpenCLIP returned raw cosine `0.250262` for the diagnostic text “person wearing a blue shirt”; this text is not a ground-truth label or accuracy result. YOLO returned no backpack detection (`no_matching_detection`). The diagnostic used no label and did not affect evaluation metrics.

The Streamlit app started and rendered the new E selector. Its five options were visible in the browser. The local file chooser could not be automated, so the browser upload-and-Analyze flow remains unverified; the underlying UI processing function was exercised directly with the real video.

Exact commands:

```powershell
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\venv\Scripts\python.exe -B -m py_compile app.py extension/query_logic.py extension/attribute_matching.py extension/full_pipeline.py run_on_video/clip_extractor.py standalone_eval/action_attribute_metrics.py standalone_eval/run_local_action_attribute_evaluation.py tests/test_query_logic.py tests/test_action_attribute_metrics.py
.\venv\Scripts\python.exe -B standalone_eval\run_local_action_attribute_evaluation.py --output-prefix soft_biometric_eval
git diff --check
```

The final complete suite passed **48 tests**. Compilation and `git diff --check` passed. Two uniquely named evaluation directories were created for the first E-mode run and the follow-up appearance/object diagnostics; no existing outputs, datasets, checkpoints, or worktree changes were deleted or overwritten.

## Changed files

- `extension/query_logic.py` — bounded clothing/carried-object parsing, evidence aggregation/status, carried-object rank component and E ablation.
- `extension/attribute_matching.py` — shared YOLO person/object detection pass, raw crop/action/object scoring, object-category evidence, and explicit statuses.
- `app.py` — E selection and display of object/appearance evidence; keeps the original query for Moment-DETR.
- `extension/full_pipeline.py` — passes parsed carried-object cues into the shared scorer and reports the new component.
- `run_on_video/clip_extractor.py` — preserves object scores, ranks, detection details, and evidence status through interval validation.
- `standalone_eval/action_attribute_metrics.py`, `standalone_eval/run_local_action_attribute_evaluation.py` — five ablations, subgroup metrics, real-model diagnostic, and machine-readable output.
- `tests/test_query_logic.py`, `tests/test_action_attribute_metrics.py` — cue separation, supported synonyms, invalid inputs, missing detector evidence, rank fusion, and subgroup metrics.
- `SOFT_BIOMETRIC_IMPLEMENTATION_REPORT.md`, `SOFT_BIOMETRIC_EVALUATION.md` — this audit and results report.

`git status --short` also includes pre-existing project changes, audit documents, earlier QA/evaluation outputs, `Ultralytics/`, `download_vocab.py`, and `venv/`. They were preserved and are not included above as newly created changes.

## Remaining work

The code path is implemented and testable, but evidence for retrieval effectiveness is missing. To evaluate appearance and carried-object contribution, provide raw video files corresponding to attribute-annotated queries, query-aligned ground-truth intervals, and explicit appearance/object labels for the relevant person or moment. In particular, per-person clothing and object-presence labels are needed to distinguish object co-occurrence from carrying. The existing 301-row annotation file alone is insufficient because its raw videos are unavailable in this checkout.
