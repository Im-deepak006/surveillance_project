# Single-video ablation report

## Status: PASS for the five-query pilot; limited evaluation

The earlier “model load blocked” diagnosis was incorrect. The checkpoint and `MomentDETRPredictor` load successfully, and Moment-DETR returns candidate intervals for the selected local video. The previous runner printed a model-loading line but had no subsequent stage or per-query progress logs; CPU semantic scoring could then run silently for a long time. A prior run’s exact exit code was not captured, so its termination cause cannot be reconstructed. No checkpoint, architecture, or ranking formula was changed.

The runner now logs stages and each query, writes progress/results, accepts a query subset, and allows its existing scorer frame count to be specified. The five-query pilot used one frame per candidate for both appearance and action scoring. The same Moment-DETR proposals were used for A, B, and C. The full run returned exit code 0.

The parser/evaluation audit corrected coordinated object extraction and preservation of unsupported object mentions. It also removes carried-object phrases from appearance scoring, keeps raw object phrases for audit, and excludes a carry verb from action scoring when a later walking action is present. A fresh offline run using this corrected parser took 122.1 seconds on CPU and exited 0. Aggregate metrics were unchanged, while q2 B/C's top proposal changed.

Parser defect details: the old q2 carried-object regex required a carry verb close to each class noun and allowed at most two intervening words; after matching `gray backpack`, the coordinated `and a bright yellow shoulder bag` had no repeated verb and fell outside that pattern. The old q3 parser only mapped detector-supported aliases, so although its broad `object_texts` contained `bag`, `shopping bag` never appeared in `carried_object_texts`. Its action field also duplicated `carries a bright green shopping bag` with the object request. The audit fix maps q2's shoulder bag to the supported `handbag` category and preserves its full mention; q3's shopping bag remains explicitly unresolved and out of detector scoring.

Field comparison: q1's old appearance string included a dangling `as they`, and its detector category `backpack` discarded the requested red color; q2's old appearance string included the yellow bag plus dangling `as they`, while the carried-object field lost that bag. Corrected outputs preserve raw `red backpack`, `gray backpack`, and `bright yellow shoulder bag` mentions separately, though YOLO category scores do not test those colors. q3's former action/object overlap is removed. q4 and q5 appearance and action phrases were already separated correctly and remain unchanged.

## Environment and model-load diagnosis

- Python 3.10.8: project venv at `D:\jegan_repo\FYP\moment-detr-baseline\venv\Scripts\python.exe`.
- PyTorch 2.13.0+cpu; CUDA unavailable. CPU inference used one PyTorch thread in the runner.
- Video: `D:\yolo_enhance\video\MOT17-04-SDP-raw.webm`, 5,094,187 bytes; 35.0 s, 30 fps, 960×540, 1,050 frames. OpenCV/FFMPEG decoding and sampled reads succeeded.
- Moment-DETR checkpoint: `run_on_video/moment_detr_ckpt/model_best.ckpt`, 16,952,721 bytes.
- CLIP weights: local cache `C:\Users\deepa\.cache\clip\ViT-B-32.pt`, 353,976,522 bytes.
- YOLO11 weights: `yolo11n.pt`, 5,613,764 bytes. No downloads were attempted.

The isolated loader trace captured successful completion and exit code 0 at every step:

| Step after the log line | Result | Time |
|---|---|---:|
| `torch.load(..., map_location="cpu", weights_only=False)` | Loaded checkpoint with `model` and `opt`; 92 state-dict entries | 0.020 s |
| `build_transformer(args)` | Complete | 0.079 s |
| `build_position_encoding(args)` | Complete | 0.002 s |
| `MomentDETR(...)` construction | Complete | 0.017 s |
| `model.load_state_dict(ckpt["model"])`, then `eval()` | Complete | 0.013 s |
| Exact `MomentDETRPredictor` initialization with cached CLIP weights and CPU | Complete | 5.204 s |

The checkpoint’s saved `opt.device` says CUDA, but inference explicitly places the model on CPU; this did not cause a compatibility error. A one-query `localize_moment` call returned ten proposals (8.971 s in the isolated run). The final five-query batched localization took 4.808 s.

The previous failure symptom came from insufficient progress reporting, not a reproducible model-loading failure. The old run’s exit status is unknown. The full pilot’s semantic stage took 102.426 CPU seconds with one sample frame per candidate (individual query scoring took 13.726–44.626 s), which explains why an unlogged run could appear stalled. The default five-frame setting would take longer. The smallest corrective change was runner observability/configurability; the retrieval algorithm and model were untouched.

## Query results

Intervals below are seconds. Each A/B/C top interval came from the same ten Moment-DETR proposals for that query. Raw CLIP appearance scores are cosine similarities, not calibrated probabilities; only B/C use the appearance rank in ordering. “Person/attribute” labels were manually adjudicated by checking whether the annotated target and requested visual cues appear within the top interval; this is separate from temporal IoU and is not an identity model metric.

| Query | Ground truth | A: temporal-only top1 (IoU) | B: temporal + appearance top1 (IoU) | C: + carried-object top1 (IoU) | Manual target/attribute visibility A / B / C |
|---|---|---|---|---|---|
| q1 red backpack | 23.50–35.00 | 10.3764–17.4982 (0.0000) | 16.4884–25.1523 (0.0893) | 16.4884–25.1523 (0.0893) | no / partial / partial |
| q2 gray backpack + yellow bag | 23.50–35.00 | 10.9382–18.9563 (0.0000) | 9.5940–16.8307 (0.0000) | 9.5940–16.8307 (0.0000) | no / no / no |
| q3 green shopping bag | 30.25–32.00 | 14.3047–25.8424 (0.0000) | 14.3047–25.8424 (0.0000) | not applicable | no / no / n/a |
| q4 white shirt | 6.00–22.50 | 13.6716–22.4799 (0.5338) | 13.6716–22.4799 (0.5338) | not applicable | yes / yes / n/a |
| q5 pale pink shirt | 6.00–22.50 | 13.9417–21.1365 (0.4360) | 13.9417–21.1365 (0.4360) | not applicable | yes / yes / n/a |

Corrected parser decomposition (raw object mentions are retained separately from detector category names):

| Query | Appearance passed to person crop scorer | Action passed to frame scorer | Detector-supported category text | Unresolved requested object |
|---|---|---|---|---|
| q1 | person in a dark jacket and tan trousers | walk away from the camera | backpack (`red backpack`) | none |
| q2 | person in a green jacket and blue trousers | walk away from the camera | backpack (`gray backpack`), handbag (`bright yellow shoulder bag`) | none |
| q3 | person in a white shirt and dark trousers | while walking across the lower-left foreground | none | bright green shopping bag |
| q4 | person in a white shirt and dark trousers | walks toward the camera along the left side | none | none |
| q5 | person in a pale pink shirt and dark trousers | walks beside the white-shirted person toward the camera | none | none |

Top1 scores, in temporal / appearance-cosine order:

| Query | A temporal score (appearance diagnostic) | B temporal score + appearance score | C carried-object evidence at top1 |
|---|---|---|---|
| q1 | 0.9364 (0.2869) | 0.2327 + 0.3128 | Unknown: no matching detection |
| q2 | 0.8790 (0.2765) | 0.0528 + 0.2913 | Unknown: no matching detection |
| q3 | 0.8314 (0.3153) | 0.8314 + 0.3153 | Not run: shopping bag is unsupported |
| q4 | 0.8424 (0.3050) | 0.8424 + 0.3050 | Not applicable |
| q5 | 0.9626 (0.3380) | 0.9626 + 0.3380 | Not applicable |

Appearance evidence status was `scored` for every candidate: YOLO11 produced person crops and OpenCLIP crop/text similarities. This does not associate a crop with the manually annotated target identity. The carried-object detector is restricted in this pipeline to the configured COCO categories: backpack (24), umbrella (25), handbag (26), suitcase (28), bottle (39), cup (41), laptop (63), cell phone (67), and book (73), plus person (0) for person crops. A supported class with no detection is unknown/unobserved, not an unsupported class or confirmed absence. The corrected q2 request includes both backpack and handbag; no matching detection was returned in the candidate windows. Category detection does not verify color or person-object ownership. The q3 shopping bag is unsupported and was excluded from object reranking while retained as an unresolved phrase. In q1, a non-top proposal ([15.2211, 22.2804]) had a backpack-category detection (0.2682); this does not identify its owner.

## Aggregate metrics

| Condition | Queries in denominator | Recall@1, IoU ≥ 0.5 | Mean top1 temporal IoU |
|---|---:|---:|---:|
| A. Temporal only | 5 | 1/5 = 0.200 | 0.1940 |
| B. Temporal + appearance | 5 | 1/5 = 0.200 | 0.2118 |
| C. Temporal + appearance + carried-object evidence | 2 (q1, q2 only) | 0/2 = 0.000 | 0.0446 |

Separate manual person/attribute visibility counts: A = 2 full, 0 partial, 3 no match (5 queries); B = 2 full, 1 partial, 2 no match (5 queries); C = 0 full, 1 partial, 1 no match (2 queries). These counts are not learned retrieval metrics. This small single-video pilot does not support a general improvement claim. Temporal localization and person/attribute visibility are distinct outcomes.

All three conditions rank the same ten Moment-DETR proposal intervals for each query. The scorer preserves this candidate set; A ranks temporal scores, B ranks temporal plus appearance, and C ranks the same candidates with action evidence disabled. The fresh result records `same_moment_detr_candidate_intervals_for_conditions: true`. A/B use all five queries. C is eligible only for q1 and q2: their requested and annotated carried objects map to supported detector classes. q3 is excluded because shopping bag is unsupported; q4/q5 do not request a carried object. The JSON records eligible query IDs per metric.

Manual person/attribute visibility labels are human visual adjudications separate from temporal IoU. They do not come from YOLO scores. q2 remains `no_match`: its corrected B/C top interval ends at 16.8307 s, before the target annotation begins at 23.5 s. No person identity tracking or object ownership inference is claimed.

## Tests and real-video checks

**Real-video run:**

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\venv\Scripts\python.exe -u standalone_eval\run_mot17_single_video_ablation.py --video 'D:\yolo_enhance\video\MOT17-04-SDP-raw.webm' --output-dir 'outputs\single_video_mot17_04_all5_20261023' --appearance-frames 1 --action-frames 1
```

Result: exit code 0; five queries, ten Moment-DETR candidates each; all five appearance-scoring stages completed; candidate intervals were identical across A/B/C. One-frame sampling was selected to keep this CPU pilot bounded. No clip-extraction stage is part of this ablation runner, and no Streamlit UI upload test was performed.

**Automated tests:**

```powershell
.\venv\Scripts\python.exe -m compileall -q extension\query_logic.py standalone_eval\run_mot17_single_video_ablation.py tests\test_query_logic.py tests\test_single_video_ablation_runner.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result after audit regressions: 54 passed, 0 failed. Unit tests and real-video evidence are reported separately; the latter is not inferred from the tests.

The corrected pilot was rerun offline to a new directory, preserving the original run log and output:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\venv\Scripts\python.exe -u standalone_eval\run_mot17_single_video_ablation.py --video 'D:\yolo_enhance\video\MOT17-04-SDP-raw.webm' --output-dir 'outputs\single_video_mot17_04_parser_audit_20261023' --appearance-frames 1 --action-frames 1
```

Result: exit code 0, five queries, ten candidates per query, 122.1 seconds for inference and semantic scoring, and identical aggregate metrics to the original pilot.

## Artifacts

- [Ground-truth annotations](evaluation_data/MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json)
- [Updated machine-readable results](outputs/single_video_mot17_04_all5_20261023/results.json)
- [Fresh corrected run output](outputs/single_video_mot17_04_parser_audit_20261023/results.json)
- Actual per-stage and per-query run log: `outputs/single_video_mot17_04_all5_20261023/run.log`
- [Isolated model-load trace](outputs/single_video_mot17_04_all5_20261023/model_load_trace.txt)
- [Runner](standalone_eval/run_mot17_single_video_ablation.py)
- [Runner regression test](tests/test_single_video_ablation_runner.py)

The pre-existing failed-run record was replaced at the requested results path with the completed five-query results. The separate all-five run directory retains the source run log and full result payload. Unrelated working-tree changes were preserved.
