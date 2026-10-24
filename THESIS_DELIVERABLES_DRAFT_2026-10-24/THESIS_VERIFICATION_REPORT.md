# Thesis Verification Report

Generated 24 October 2026. Repository D:\jegan_repo\FYP\moment-detr-baseline. Existing working-tree changes preserved; source and model files not changed.

## Files inspected
app.py; extension/attribute_matching.py, full_pipeline.py, query_logic.py, temporal_evidence.py, ui_person_visualization.py; run_on_video/run.py, clip_extractor.py, model_utils.py, data_utils.py, moment_detr_ckpt/opt.json; moment_detr/config.py, model.py, transformer.py; standalone_eval/run_temporal_consensus_ablation.py; outputs/attribute_aware_consensus_20261023 results.json, report, CSV and config; verified proposal/diagnostic JSON, HTML and images; evaluation_data/MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json; README, requirements, data docs/license, existing thesis draft and DOCX; contribution, audit, single-video, proposal, soft-biometric, action/attribute, end-to-end, missing-video and dataset-plan reports.

## Verified implementation
2-second sampling, max 75 segments (~150 s), CLIP ViT-B/32, feature dimensions 512, hidden 256, two encoder/decoder layers, eight heads, ten moment queries. Method D is inference-time mean of available within-set tied-rank percentiles. Saved D uses temporal, appearance, person-object phrase, and category evidence; model and intervals remain fixed. YOLO/OpenCLIP do not track identity or prove object ownership. E is a separate median multi-frame condition.

## Metrics checked against saved JSON and report
| Condition | Mean IoU | R@.3 | R@.5 | R@.7 |
|---|---:|---:|---:|---:|
| A | .1940 | .400 | .200 | .000 |
| B | .2118 | .400 | .200 | .000 |
| D | .4020 | .800 | .400 | .000 |
| E | .3150 | .600 | .400 | .000 |
| F | .3150 | .600 | .400 | .000 |
| G | .2118 | .400 | .200 | .000 |

Each denominator is five; D-A mean difference +.2080. Saved result says proposals unchanged, no model training/weight tuning. This verifies saved figures, not a fresh replay. Video is external at D:\yolo_enhance\video\MOT17-04-SDP-raw.webm.

## References checked
Metadata checked against NeurIPS, ECCV, and PMLR primary proceedings for Moment-DETR/QVHighlights, DETR, and CLIP; official software links included. Pilot's exact Ultralytics/OpenCLIP versions were not preserved.

## Unverified or missing
Student/supervisor/institution and official forms; video license and annotation protocol; hardware and pilot software versions; authentic UI screenshots for initial form, complete comparison, expanded result; no fresh experiment or source tests were run; university formatting requirements.

## Created
Final_Year_Project_Thesis_DRAFT.docx; Project_Architecture_and_Results.docx; figures/ with four code-based diagrams, two metric plots, and real saved diagnostic frames; this report; submission checklist. PDF is produced by the document renderer if available. Update Word TOC after final edits.
