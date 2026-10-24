# Proposal Diagnosis and Evidence-Aware Reranking

**Status:** implementation and five-query controlled pilot completed; general improvement is not established.

## What was measured

The final run used the existing Moment-DETR checkpoint and its ten proposals for each of the same five manually annotated MOT17-04 queries. A, B, C, and D rerank those same intervals. A is temporal rank only; B adds person-crop appearance; C adds supported YOLO object-category evidence to A's appearance condition; D adds person-crop text scores and supported YOLO categories. Components with unavailable scores are omitted from a candidate's unweighted mean rank. No weights were tuned and no model was retrained.

Single-frame scoring now samples the proposal midpoint. The first diagnostic run used proposal starts because `linspace(start, end, 1)` returns `start`; its output is retained at `outputs/proposal_diagnosis_mot17_04_20261023/` for audit only. The final results below are from the midpoint-corrected run at `outputs/proposal_diagnosis_mot17_04_verified_20261023/`. An interrupted midpoint run is retained at `outputs/proposal_diagnosis_mot17_04_midpoint_20261023/`.

## Proposal and per-query results

Intervals are seconds. “Oracle rank” is the rank of the single proposal with maximum temporal IoU under that condition. Proposal counts show how many of the ten generated intervals meet each IoU threshold. A dash means the condition is ineligible.

| Query | Ground truth | A top-1 (IoU) | B top-1 (IoU) | C top-1 (IoU) | D top-1 (IoU) | Best proposal (IoU; temporal rank) | Proposals ≥.3 / ≥.5 / ≥.7 |
|---|---|---|---|---|---|---|---|
| q1 red backpack | 23.50–35.00 | 10.3764–17.4982 (0.0000) | 16.4884–25.1523 (0.0893) | 16.4884–25.1523 (0.0893) | 19.0469–32.6118 (0.5712) | 21.6925–31.4410 (0.5967; 3) | 4 / 3 / 0 |
| q2 gray backpack + yellow shoulder bag | 23.50–35.00 | 10.9382–18.9563 (0.0000) | 10.9382–18.9563 (0.0000) | 10.9382–18.9563 (0.0000) | 18.5610–31.2121 (0.4691) | 26.5813–33.2894 (0.5833; 8) | 4 / 2 / 0 |
| q3 green shopping bag | 30.25–32.00 | 14.3047–25.8424 (0.0000) | 9.3146–17.5783 (0.0000) | — | 9.3146–17.5783 (0.0000) | 29.7159–35.0000 (0.3312; 8) | 1 / 0 / 0 |
| q4 white shirt | 6.00–22.50 | 13.6716–22.4799 (0.5338) | 13.6716–22.4799 (0.5338) | — | 13.6716–22.4799 (0.5338) | 7.7984–17.7829 (0.6051; 5) | 6 / 3 / 0 |
| q5 pale pink shirt | 6.00–22.50 | 13.9417–21.1365 (0.4360) | 13.9417–21.1365 (0.4360) | — | 13.9417–21.1365 (0.4360) | 8.4765–16.2471 (0.4709; 5) | 6 / 0 / 0 |

| Query | D top-1 scores: temporal / appearance / action diagnostic / YOLO object / person-crop object | D oracle proposal rank | Midpoint person crops from YOLO | Diagnosis |
|---|---|---:|---:|---|
| q1 | 0.0008 / 0.3137 / 0.2262 / unavailable / 0.2867 | 2 | 12 | Proposals cover the target interval, but A/B rank it poorly. D selects a proposal at IoU 0.571. The highest red-backpack phrase score came from crop 4, visually a green-jacket/yellow-bag person, rather than the red-backpack crop (crop 0); D's visual evidence is therefore not reliably target-associated. |
| q2 | 0.0021 / 0.2945 / 0.2223 / unavailable / 0.2948 | 7 | 12 | Proposals cover the interval; temporal and appearance top-1 miss it. D moves to IoU 0.469. The selected object-text crop matches the green jacket/yellow shoulder bag visually, but does not prove the gray backpack or persistent identity. |
| q3 | 0.7680 / 0.3133 / 0.2474 / not requested / 0.2908 | 2 | 15 | Proposal-limited: maximum IoU is 0.331, with no proposal at 0.5. Shopping bag is unsupported by YOLO. The D top interval also precedes the annotated target interval. |
| q4 | 0.8424 / 0.2986 / 0.2573 / not requested / not requested | 3 | 19 | Existing temporal top-1 already reaches IoU 0.534; appearance leaves it unchanged. |
| q5 | 0.9626 / 0.3190 / 0.2671 / not requested / not requested | 4 | 23 | Proposal ceiling: the best available interval reaches only IoU 0.471. A, B, and D select IoU 0.436. |

The action values above are diagnostic raw image-text cosine scores; A–D do not include the action component. The complete raw scores, evidence statuses, object detections, per-category states, and per-mention person-crop indices are in `results.json` for all ten proposals per query.

### Root-cause summary

- **Missing/weak proposals:** q3 has no proposal at IoU 0.5 or higher; q5's oracle is 0.471. Reranking cannot fix these coverage limits. The q3 request is a short 1.75-second moment while its best proposal is 5.28 seconds and extends to the video end.
- **Ranking:** q1 and q2 each have proposals above 0.5 IoU, but temporal ranks are 3 and 8. B ranks those oracle proposals 8 and 10; appearance-only ranking does not solve the misses. q4 already has a good temporal top-1.
- **Visual detections:** q1 has one backpack detection in a non-top candidate at 14.1065–19.2600 s, before its annotated target interval starts at 23.5 s; it is not evidence for the target's backpack. The q1 and q2 D top candidates have no YOLO object boxes, and q2 has no matching backpack or handbag box in any candidate. These are detector non-observations, not evidence that the bags are absent. The category detector cannot test requested colors or bag ownership.
- **Person association:** YOLO produced multiple person crops per sampled frame. Text scores are computed on individual crops, and multiple requested object phrases must score on the same crop. The scorer then takes the best crop in that frame without a track or target-person identity. Crop ordering is not a stable person ID. The q1 example shows this can choose another visible person. No ground-truth per-frame person boxes/IDs are available to measure target-crop recall or ID switches.

## Parser audit

| Query | Appearance phrase | Action/location phrase | Detector categories | Full object mentions retained |
|---|---|---|---|---|
| q1 | person in a dark jacket and tan trousers | walk away from the camera | backpack | red backpack |
| q2 | person in a green jacket and blue trousers | walk away from the camera | backpack, handbag | gray backpack; bright yellow shoulder bag |
| q3 | person in a white shirt and dark trousers | while walking across the lower-left foreground | none | bright green shopping bag (unresolved for YOLO) |
| q4 | person in a white shirt and dark trousers | walks toward the camera along the left side | none | none |
| q5 | person in a pale pink shirt and dark trousers | walks beside the white-shirted person toward the camera | none | none |

The original query remains the Moment-DETR input. For the semantic fields, no clothing, color, bag, or location phrase is lost in these five parser cases. q1/q2 retain clothing colors separately from their full colored bag mentions; detector category names intentionally normalize those mentions. q2's coordinated second bag is now parsed without requiring a repeated “wears” verb. q3's shopping-bag phrase is retained, while the carry phrase is not duplicated as an action cue. q4 and q5 preserve their directional/relative-person clauses as action text. q3's parser output does not create a shopping-bag detector class.

The YOLO11 checkpoint's actual `model.names` in this run contains person (0), backpack (24), umbrella (25), handbag (26), suitcase (28), bottle (39), cup (41), laptop (63), cell phone (67), and book (73). The carried-object path supports the nine non-person categories in that list. “Shopping bag” has no class mapping here. A supported category without a returned box has an `observed_negative` state meaning only “the sampled detector pass returned no box”; it contributes no negative score. Detector failure or unavailable frames remain unknown/unavailable.

## Controlled ablation metrics

Recall uses each condition's eligible top-1 queries as its denominator; the JSON records those qids. Mean IoU uses the same denominator. C includes q1 and q2 only because both request a YOLO-supported class with an annotated carried-object label. A, B, and D use all five queries.

| Condition | n / eligible qids | Recall@1 IoU≥.3 | Recall@1 IoU≥.5 | Recall@1 IoU≥.7 | Mean top-1 IoU |
|---|---|---:|---:|---:|---:|
| A temporal baseline | 5 / q1–q5 | 2/5 = 0.400 | 1/5 = 0.200 | 0/5 = 0.000 | 0.1940 |
| B + appearance | 5 / q1–q5 | 2/5 = 0.400 | 1/5 = 0.200 | 0/5 = 0.000 | 0.2118 |
| C + supported YOLO object category | 2 / q1–q2 | 0/2 = 0.000 | 0/2 = 0.000 | 0/2 = 0.000 | 0.0446 |
| D + person-crop object phrase evidence | 5 / q1–q5 | 4/5 = 0.800 | 2/5 = 0.400 | 0/5 = 0.000 | 0.4020 |
| Proposal oracle (max IoU per query) | 5 / q1–q5 | 5/5 = 1.000 | 3/5 = 0.600 | 0/5 = 0.000 | 0.5175 mean max IoU |

D's +1/5 top-1 hit at IoU≥0.5 and higher mean IoU than A/B are an observed result on these five queries. The q1 crop mismatch, q2's sub-0.5 result, the q3 coverage ceiling, the single video, and the small sample prevent a claim of validated or general improvement. C's two-query result is not directly comparable with the five-query metrics. Confidence intervals are not reported: five queries from one video do not support a useful uncertainty estimate, and the query outcomes are not independent video samples.

## Person visibility, latency, and limitations

At D's actual one-frame midpoint sample, manual contact-sheet review found the q1 red-backpack crop and q2 green-jacket/yellow-bag crop among the YOLO person crops. The q1 object-text maximum nevertheless selected crop index 4, which was not the red-backpack target. q2's two object phrases shared crop index 4. For q3, the selected D proposal's midpoint is outside its annotated target interval. q4 and q5's sampled frames show the requested white and pale-pink shirts. These are visual checks of the sampled frame, reported separately from temporal IoU; they do not establish persistent identity or ownership. See the saved `inspection/` contact sheets.

The final offline CPU run took 295.6 seconds wall time: 9.3 seconds Moment-DETR inference and 269.0 seconds attribute scoring. Peak process RSS sampled every 0.2 seconds was 1,652.6 MiB. No larger multi-video evaluation was run: this checkout has the five-query MOT17-04 pilot annotation/video, while the 301-row QVHighlights attribute annotation set lacks its corresponding raw videos in this checkout (see `MISSING_VIDEO_DATA_AUDIT.md`). No tracker or ID-switch evaluation was run because person-track ground truth is unavailable.

## Changes completed and follow-up

**Completed:** preserve raw object mentions apart from normalized detector classes; explicitly record supported-positive, observed-no-box, unsupported, and unavailable states; rank only positive detector evidence; keep appearance/action/object scores separate; score object phrases on one person crop at a time; sample one-frame proposals at their midpoint; expose D as a research ranking option in the app; preserve evidence fields through prediction normalization; and add regressions for the five query decompositions, unknown/unsupported object evidence, multiple objects, same-crop association, frame sampling, condition denominators, shared proposals, and proposal oracle metrics.

**Experimentally verified:** the full local test suite passes (63 tests). The final real-video run completed all five queries with ten identical proposals per query across A–D. It records per-query top intervals, proposal oracle values/ranks, exact metric denominators, condition definitions, raw scores, sampled crop counts, runtime, and sampled memory.

**Future work:** use annotated target-person boxes or tracks to measure whether the correct target crop is present and stable; associate objects with a person using box geometry or tracking evidence rather than frame co-occurrence; improve q3/q5 proposal coverage; validate any scoring changes on validation videos; and evaluate a larger multi-video test set before making an improvement claim. Weights were not tuned on this pilot.

## Reproduction

Run from the repository root in PowerShell. Use a new output directory because the runner refuses to overwrite existing results.

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\venv\Scripts\python.exe -u standalone_eval\run_mot17_single_video_ablation.py --video 'D:\yolo_enhance\video\MOT17-04-SDP-raw.webm' --output-dir 'outputs\proposal_diagnosis_mot17_04_repro_20261023' --appearance-frames 1 --action-frames 1
```

Run compile and tests:

```powershell
.\venv\Scripts\python.exe -m compileall -q app.py extension\query_logic.py extension\attribute_matching.py run_on_video\clip_extractor.py standalone_eval\run_mot17_single_video_ablation.py tests
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```
