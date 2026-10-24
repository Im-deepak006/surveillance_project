# Action and Attribute Retrieval Design

## Findings before implementation

- The application path is `app.run_pipeline()` → `MomentDETRPredictor.localize_moment()` → narrow query parsing → optional YOLO person crops/OpenCLIP appearance scores → candidate ranking → interval validation and FFmpeg extraction. The full original query is passed to Moment-DETR.
- The parser recognizes some explicit clothing/accessory descriptions and relative `who`/`that` clauses. It does not extract action phrases for scoring. The full query's action semantics can influence Moment-DETR's temporal predictions, but action evidence is not currently a separate ranking component.
- Moment-DETR confidence is its foreground softmax probability. OpenCLIP returns a cosine similarity. The observed sample scores had different distributions (temporal approximately 0.0025–0.9884; appearance approximately 0.2546–0.3058), so direct weighted score addition is not justified without calibration data.
- The only local raw video with a usable annotation found for this task is `run_on_video/example/RoripwjYFp8_60.0_210.0.mp4`. `queries.jsonl` contains one query, “Chef makes pizza and cuts it up.”, with one relevant interval `[106, 122]`. This annotation has action wording but no appearance label. The existing QVHighlights feature/annotation files are not accompanied by the raw videos needed by this raw-video inference pipeline.
- Existing behavior ranks by temporal confidence, using appearance only for exact temporal-score ties. Previous QA successfully ran this sample, scored crops, extracted clips, and found that the top interval overlapped the annotated moment. Those results are documented in `END_TO_END_TEST_REPORT.md`; they do not establish an ablation result.
- A repeated real inference initially produced different proposals because the loaded Moment-DETR model remained in training mode. Its configured dropout layers were active. The predictor now switches the loaded model to evaluation mode; repeated inference on the same input produced identical proposals in the final evaluation run.

## Method to implement

1. Keep the original query intact for Moment-DETR and result display. Add a deliberately bounded lexical parser for person/entity reference, supported appearance phrase, and supported action phrase. Unknown or ambiguous spans stay unset and fall back to the original temporal baseline.
2. For each temporal candidate, sample a configurable small number of frames uniformly across its interval. Use the existing local OpenCLIP image/text encoder to score full frames against the action phrase; aggregate finite scores by their mean. This is static-frame image-text similarity, not video motion understanding.
3. Keep `temporal_score`, `appearance_score`, and `action_score` separate. Preserve a temporal-only baseline mode. For semantic modes, use within-query descending component ranks normalized to `[0,1]`, then take the unweighted mean of the selected, available component rank scores. Missing component evidence is omitted from that candidate's denominator rather than assigned a zero. Store raw scores, normalized ranks, component availability, and final fusion score.
4. The four ablations select components explicitly: temporal; temporal+appearance; temporal+action; all three. No score weights are tuned on the evaluation example. The rank-fusion rule is configurable by component selection and sampling count; it is descriptive and intended for an unweighted comparison, not presented as calibrated probabilities.
5. Evaluate identical Moment-DETR candidate intervals for each ablation. Use the one exact local annotation, report top-1 Recall at IoU 0.5 and 0.7 plus mean top-1 IoU. Because it has no appearance label, the appearance-only ablation has no appearance evidence and must match temporal-only; this is not a valid test of appearance benefit. Results are a one-query demonstration, not generalization evidence.

## Relation to ActPrompt

ActPrompt introduces Action-Cue Injection with video-guided and verb-guided prompts into a VLM image encoder, Context-aware Temporal Prompt Learning over neighboring frames, and training objectives for in-domain feature adaptation. This repository will not implement those modules or training. It will compare a separate frame-level OpenCLIP similarity signal against the existing Moment-DETR temporal ranking, without replacing Moment-DETR's predicted intervals.

Reference: Wang et al., [ActPrompt: In-Domain Feature Adaptation via Action Cues for Video Temporal Grounding](https://arxiv.org/abs/2408.06622) (2024). The paper describes its action-prompt injection, temporal prompts, and pretext training objectives in the method section; this project uses none of those mechanisms.

## Implemented evaluation

See `ACTION_ATTRIBUTE_EVALUATION.md` and the machine-readable `outputs/action_attribute_eval_bf906f3158/ACTION_ATTRIBUTE_EVALUATION_RESULTS.json` for the exact run, scores, stage timings, clip checks, and limits. In this single query, temporal-only top-1 IoU was 0.0; temporal+action and the all-components mode ranked a proposal with IoU 0.5298 first. The only query has no appearance description, so the appearance modes reduce to temporal-only. This observation is not evidence of a general performance gain.
