# Contribution and Results

## Contribution

This project keeps Moment-DETR's temporal proposals and adds a transparent reranking experiment for person-appearance queries. The tested candidate score replaces one-frame/max-over-all-crops appearance evidence with the median of each sampled frame's strongest person-crop similarity. It requires usable appearance evidence in at least two of three evenly spaced frames. The implementation does not link crop indices across frames or claim identity tracking.

A separate object-category component accepts a YOLO category only after positive detections in at least two of three sampled frames. Missing or non-repeated detections remain unknown; unsupported object phrases remain unsupported. This is category persistence, not proof that a person carries the detected object.

The experiment reuses the existing proposal pool and cached OpenCLIP crop similarities. It does not modify the Moment-DETR checkpoint, tune weights, or train a model.

## Five-query pilot results

All conditions use the same five queries and proposal sets; denominator is 5 for every row. These are temporal localization metrics only.

| Condition | Recall@.3 | Recall@.5 | Recall@.7 | Mean IoU |
|---|---:|---:|---:|---:|
| A: temporal baseline | 0.400 | 0.200 | 0.000 | 0.1940 |
| B: saved single-frame appearance | 0.400 | 0.200 | 0.000 | 0.2118 |
| D: saved person-object fusion | 0.800 | 0.400 | 0.000 | 0.4020 |
| E: median multi-frame appearance | 0.600 | 0.400 | 0.000 | 0.3150 |
| F: E plus repeated positive object categories | 0.600 | 0.400 | 0.000 | 0.3150 |
| G: single-frame appearance plus repeated object categories | 0.400 | 0.200 | 0.000 | 0.2118 |

| Query | A IoU | B IoU | D IoU | E IoU |
|---|---:|---:|---:|---:|
| q1 red backpack | 0.0000 | 0.0893 | 0.5712 | 0.0893 |
| q2 gray backpack + yellow shoulder bag | 0.0000 | 0.0000 | 0.4691 | 0.5158 |
| q3 green shopping bag | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q4 white shirt | 0.5338 | 0.5338 | 0.5338 | 0.5338 |
| q5 pale pink shirt | 0.4360 | 0.4360 | 0.4360 | 0.4360 |

The median method improves this pilot's mean IoU and Recall@.5 relative to A and B, driven mainly by q2. It is below D. F equals E because no candidate had repeated positive detections for a requested supported object category in two sampled frames. G equals B for the same reason. This is an exploratory result from five queries in one video; it does not establish general improvement. No confidence interval is reported for this non-independent, single-video sample.

The proposal oracle remains a ceiling of mean max IoU 0.5175, Recall@.3 1.0, Recall@.5 0.6, and Recall@.7 0.0. q3 and q5 therefore retain proposal-coverage limits that reranking alone cannot remove.

## Person and object verification limits

The automated crop report shows q1's appearance phrase selects crop 3 while its red-backpack phrase selects crop 4 in the D representative frame. q2's appearance, gray-backpack, and yellow-shoulder-bag phrases all select crop 4. Crop numbers are frame-local; neither result verifies persistent identity. No person-association accuracy or object-ownership accuracy is claimed. The q2 sampled frame has no YOLO backpack/handbag box; that is a detector non-observation, not evidence the bags are absent. YOLO does not support the requested shopping-bag class in q3.

## Runtime, tests, and artifacts

- Automated crop diagnostic generation: 611.2 seconds, including YOLO/OpenCLIP scoring and report generation.
- Consensus reranking and YOLO-only multi-frame category pass: 37.2 seconds; Moment-DETR and CLIP were not rerun.
- Training: none. Five queries from one video are inadequate for valid grouped training/validation/test splits.
- Full test suite: 72 tests passed.

The results, per-query CSV, config, technical report, comparison HTML, and local crop diagnostics are in `outputs/attribute_aware_consensus_20261023/`. The earlier detailed crop-level report is also preserved at `outputs/automated_visual_diagnostics_mot17_04_verified_20261023/`.

## Reproduction

From the repository root, use a new output directory because the runner refuses to overwrite an existing one:

```powershell
.\venv\Scripts\python.exe standalone_eval\run_automated_visual_diagnostics.py --source-results outputs\proposal_diagnosis_mot17_04_verified_20261023\results.json --annotations evaluation_data\MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json --video D:\yolo_enhance\video\MOT17-04-SDP-raw.webm --output-dir outputs\automated_visual_diagnostics_repro_20261023 --frames 3
.\venv\Scripts\python.exe standalone_eval\run_temporal_consensus_ablation.py --source-results outputs\proposal_diagnosis_mot17_04_verified_20261023\results.json --cached-scores outputs\automated_visual_diagnostics_repro_20261023\results.json --video D:\yolo_enhance\video\MOT17-04-SDP-raw.webm --output-dir outputs\attribute_aware_consensus_repro_20261023 --frames 3 --repeat-positive-frames 2
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Next steps, ordered by impact

1. Collect an independently sampled multi-video evaluation set with temporal intervals, per-frame target-person boxes or IDs, and explicit person-object ownership annotations.
2. Validate the median rule and category persistence on held-out videos, with video-grouped validation and no test-set weight tuning.
3. Improve proposal coverage for short events such as q3 before attributing those errors to reranking.
4. Revisit object evidence only with person-object association evidence or a separately validated association method.
5. Consider learned fusion only after enough independent labeled videos exist.

## Panel-ready summary

Moment-DETR proposes candidate moments, and our reranker checks clothing cues across three frames instead of trusting one crop. In this five-query pilot, median aggregation raised Recall@0.5 from 0.2 to 0.4 and mean IoU from 0.194 to 0.315 compared with temporal-only baseline. The gain is exploratory, mostly from one query, and remains below existing D; one video is not enough to claim general improvement. Repeated object detection added no usable evidence in this pilot, and person identity/object ownership are still unverified.

