# Attribute-Aware Video Moment Retrieval for Person-Centric Search

## Thesis draft

> Replace bracketed items with your institution, department, student, supervisor, and submission details. This draft is based on the implementation and recorded experiments in this repository. Keep the stated limitations and do not present pilot results as general performance claims.

## Title page

**ATTRIBUTE-AWARE VIDEO MOMENT RETRIEVAL FOR PERSON-CENTRIC SEARCH**

A project thesis submitted in partial fulfilment of the requirements for the award of the degree of **[Degree Name]**

Submitted by: **[Student Name(s) and Register Number(s)]**  
Under the guidance of: **[Supervisor Name and Designation]**

**[Department]**  
**[Institution Name]**  
**[City, Country]**  
**[Month and Year]**

## Declaration

We declare that the project titled **“Attribute-Aware Video Moment Retrieval for Person-Centric Search”** is the work carried out by us under the guidance of **[Supervisor Name]**. The results and limitations reported in this thesis are based on the implementation and experiments identified in the project repository. Sources and third-party software used in the work are acknowledged in the references.

## Acknowledgements

We thank **[Supervisor Name]**, **[Department/Institution]**, and our project reviewers for their guidance and feedback. We also acknowledge the authors of Moment-DETR, QVHighlights, CLIP, OpenCLIP, Ultralytics YOLO, PyTorch, and FFmpeg, whose published work and software support this project. Add any other individual or organization that directly contributed to the work.

## Abstract

Searching for a person or event in a long video by manually reviewing the entire recording is time-consuming. This project develops a prototype that accepts a short video and a natural-language query, proposes relevant time intervals, optionally reranks those intervals using person-appearance cues, and extracts the highest-ranked intervals as video clips. The temporal localization component uses a pretrained Moment-DETR checkpoint. The extension detects person regions with YOLO11n and compares person crops with query text using OpenCLIP. A separate experimental reranking study aggregates appearance evidence over three sampled frames and investigates whether repeated object-category detections can contribute additional evidence.

The application integrates video upload, query processing, temporal proposal generation, optional semantic reranking, interval validation, and FFmpeg-based clip extraction. A five-query pilot on one video compared temporal-only and appearance-aware ranking conditions. The temporal baseline achieved Recall@0.5 of 0.20 and mean IoU of 0.194. Median multi-frame appearance aggregation achieved Recall@0.5 of 0.40 and mean IoU of 0.315. The best tested person-object fusion condition achieved mean IoU of 0.402 and Recall@0.5 of 0.40. These results are exploratory: they use five queries from one video, are influenced by individual queries, and do not establish general improvement. Object persistence added no measurable gain in this pilot. The project also does not implement person tracking or prove that a detected object belongs to a particular person. The prototype demonstrates an integrated retrieval workflow while identifying the annotated multi-video evaluation required for stronger conclusions.

**Keywords:** video moment retrieval, temporal localization, Moment-DETR, person appearance, OpenCLIP, YOLO, video search, reranking.

## Suggested table of contents

1. Introduction
2. Literature Review and Technical Background
3. System Analysis and Design
4. Implementation
5. Experiments and Results
6. Conclusion and Future Work
References
Appendices

## List of abbreviations

| Abbreviation | Meaning |
|---|---|
| AI | Artificial Intelligence |
| CLIP | Contrastive Language–Image Pretraining |
| IoU | Intersection over Union |
| MR | Moment Retrieval |
| NMS | Non-Maximum Suppression |
| QVHighlights | Query-based Video Highlights dataset |
| R@1 | Recall at rank 1 |
| UI | User Interface |
| YOLO | You Only Look Once |

# Chapter 1: Introduction

## 1.1 Background

Video is widely used to record activities in public, institutional, and operational settings. Finding a particular event in a recording often requires a person to watch many minutes of footage. Text-based video retrieval aims to reduce this effort by allowing a user to describe an event and automatically identify relevant portions of the recording.

This project studies a prototype for natural-language moment retrieval with an optional person-appearance reranking stage. The query can describe an event and may also contain visible appearance information, such as clothing colour. The system first estimates relevant temporal intervals. It can then use person crops and text-image similarity as additional ranking evidence. The ranked intervals are validated against the video duration and can be extracted for review.

The project builds on Moment-DETR, an end-to-end model for localizing moments in video from natural-language queries. Its added appearance experiment uses person detections and OpenCLIP similarity. The system is a short-video prototype and experimental retrieval aid. It is not a person-tracking system, an identity recognition system, or a validated surveillance analytics product.

## 1.2 Problem statement

Manual review of a video is inefficient when the reviewer needs to locate a short event described in natural language. A temporal retrieval model can propose intervals, but its proposals may not use person-appearance details in the query. This project asks whether combining temporal relevance with person-crop appearance similarity can help rank candidate intervals, and whether evidence aggregated across several sampled frames is a useful experiment.

The repository contains a working implementation path and limited pilot measurements, but the available appearance/object annotations are not paired with a sufficient set of raw videos. Therefore, the project must distinguish software integration from measured retrieval effectiveness and must report the data limitations explicitly.

## 1.3 Aim

To implement and evaluate a prototype that retrieves candidate moments from a short video using a natural-language query, with an experimental person-appearance reranking component.

## 1.4 Objectives

1. Study the Moment-DETR temporal moment retrieval pipeline and its input/output format.
2. Integrate the pretrained temporal model into a video-query application.
3. Detect person regions in selected frames and calculate text-image similarity for appearance-aware ranking experiments.
4. Validate proposed time boundaries and extract ranked video clips.
5. Compare recorded temporal and appearance-aware ranking conditions on the available pilot queries.
6. Document evaluation limits, including the absence of person identity tracking and person-object ownership verification.

## 1.5 Scope

The implemented application processes short videos of approximately 150 seconds or less, subject to the model's 75 two-second-segment input limit. It accepts a video and a text query, generates Moment-DETR temporal proposals, supports person-crop scoring through YOLO and OpenCLIP, validates intervals, and extracts selected clips. The separate multi-frame ablation uses three sampled frames and evaluates a five-query, single-video pilot.

The scope excludes persistent person tracking, cross-camera re-identification, demographic inference, custom detector training, and validated proof of object ownership. The system does not establish performance on a representative surveillance-video dataset. The limited pilot results must not be generalized to other videos or populations.

## 1.6 Research questions

1. Can a pretrained temporal moment retrieval model be integrated into a prototype that returns timestamped clips for a text query?
2. How does the tested multi-frame appearance reranking condition compare with temporal-only ranking on the available pilot queries?
3. What evaluation data and system capabilities are missing before appearance retrieval can be assessed reliably?

## 1.7 Contributions

The project contribution is an integrated prototype and a documented experimental analysis. The prototype connects Moment-DETR proposals, person detection, OpenCLIP crop-text scoring, interval checks, and clip extraction. The ablation compares the same five query/proposal sets under several ranking conditions, including median appearance aggregation across sampled frames. The project also records failure boundaries: the sample is too small for a general claim, person crops are not linked into identities across frames, and detected object categories are not evidence of person-object ownership.

## 1.8 Organization of the thesis

Chapter 2 reviews temporal moment retrieval and vision-language/person-detection components. Chapter 3 presents requirements, architecture, and data flow. Chapter 4 describes the implementation. Chapter 5 reports available experiments and their limitations. Chapter 6 summarizes the work and proposes a data-driven path for future evaluation.

# Chapter 2: Literature Review and Technical Background

## 2.1 Natural-language video moment retrieval

Natural-language moment retrieval maps a text query to one or more temporal intervals in a video. A model typically represents video frames or segments and the query in a shared or interacting feature space, then predicts candidate boundaries and relevance scores. Retrieval quality depends on both the model's ranking and whether its candidate intervals cover the annotated event.

## 2.2 Moment-DETR

Moment-DETR formulates moment retrieval as direct set prediction. It uses a transformer-based architecture to predict a set of temporal windows and associated scores. The upstream work evaluates the method on QVHighlights and also studies highlight detection. This repository includes a pretrained checkpoint and inference code that extracts CLIP video/text features and returns ranked temporal windows. In the current application, Moment-DETR supplies candidate intervals; the experimental reranker may reorder those candidates but does not change the trained checkpoint.

## 2.3 CLIP and OpenCLIP similarity

CLIP-style models encode image and text inputs into comparable representations. The cosine similarity between a text embedding and an image embedding can be used as a ranking signal. It is not a calibrated probability and, without labeled data, does not establish that a description is true. This project uses an OpenCLIP image-text similarity score between detected person crops and appearance phrases as auxiliary reranking evidence.

## 2.4 YOLO person detection

YOLO is a family of object detectors that predicts object categories and locations in images. The project configures YOLO11n to detect the COCO person category and crops detected people for visual similarity scoring. The selected model is a supplied pretrained detector; the project does not report custom training on surveillance footage or detection precision/recall on a project-specific annotated dataset.

## 2.5 Multi-frame evidence and its limits

Aggregating appearance scores from multiple sampled frames can reduce dependence on a single frame. In this project, each frame contributes its strongest detected-person crop similarity, and the tested consensus score is the median across sampled frames, subject to a minimum evidence rule. Crop indices are frame-local. No crop association, track identity, or person re-identification is computed. Thus, multi-frame aggregation here means temporal score aggregation and must not be described as tracking a persistent person.

Repeated object-category detections can indicate that a supported category was detected in multiple sampled frames. A category detector does not establish which person, if any, carries the object. A missing detection is treated as unknown evidence rather than proof of absence.

## 2.6 Evaluation metrics

For a predicted interval and a ground-truth interval, temporal Intersection over Union is the length of their temporal intersection divided by the length of their temporal union. Recall@IoU threshold measures the share of queries for which a relevant prediction reaches the specified overlap threshold. Mean IoU averages the selected predicted interval's overlap with ground truth across queries. The thesis must state whether metrics use top-1 predictions or any prediction in the candidate set; the pilot table in Chapter 5 reports temporal localization scores as defined in the saved project ablation report.

## 2.7 Summary and research gap

Existing temporal retrieval provides a useful candidate generator, while crop-text similarity offers a simple auxiliary ranking signal for appearance phrases. The project explores their combination in software but lacks the independent, video-grounded annotations needed to determine whether appearance-aware ranking improves general retrieval. In particular, a defensible evaluation requires multiple videos, temporal intervals, target-person annotations, and explicit person-object links for any ownership claim.

# Chapter 3: System Analysis and Design

## 3.1 Existing process and motivation

The baseline retrieval process uses a text query to generate temporal proposals. A user otherwise has to inspect the video manually and identify useful timestamps. The project adds an experimental scoring path so that person-appearance descriptions can be considered during candidate ranking, then extracts the selected windows for viewing.

## 3.2 Requirements

### Functional requirements

- Accept a supported video file and a non-empty natural-language query.
- Generate timestamped candidate moments using the available Moment-DETR checkpoint.
- Optionally score detected person crops against appearance text.
- Rank candidates and validate interval boundaries against video duration.
- Extract valid intervals into playable clips and show or provide them to the user.

### Non-functional requirements

- Reject invalid or unsupported inputs with an understandable error.
- Keep all returned timestamps within source-video bounds.
- Preserve the original candidate metadata needed to interpret ranking output.
- Avoid describing similarity scores as probabilities or detector non-observations as confirmed absence.
- Make evaluation provenance and limitations visible in reports.

## 3.3 System architecture

The implementation is a Python application with a Streamlit user interface. The application calls the Moment-DETR inference wrapper for temporal proposals. Its attribute-matching component samples video frames, detects person boxes using YOLO, crops the detected regions, and scores the crops against text using OpenCLIP. Candidate windows are normalized and validated before FFmpeg extracts clips. The user interface presents the resulting ranked clips.

**Figure 3.1 (to prepare):** Architecture diagram showing user input, video/query validation, Moment-DETR proposal generation, optional YOLO/OpenCLIP scoring, ranking, interval validation, FFmpeg extraction, and results display.

## 3.4 Data flow

1. The user supplies a video and a text query.
2. The application checks the input and video duration.
3. The inference wrapper samples video content and obtains temporal windows and scores from Moment-DETR.
4. In the attribute-aware path, sampled frames are passed through YOLO; person boxes yield image crops, which OpenCLIP compares with the text cue.
5. The configured ranking method combines available temporal and semantic evidence. Missing evidence must remain distinguishable from a genuine low similarity score.
6. Candidate times are normalized and constrained to the source duration.
7. FFmpeg extracts valid clips, which the interface displays or makes available for download.

## 3.5 Ranking experiment design

The recorded multi-frame ablation reuses the same candidate proposals and cached OpenCLIP crop similarities. It compares temporal-only ranking, saved single-frame appearance scoring, a saved person-object fusion condition, median multi-frame appearance, and variants that require repeated positive object-category evidence. The experiment does not retrain the model or tune weights. Repeated category evidence is accepted only for supported categories detected in at least two of three sampled frames. A non-detection remains unknown. This design allows a controlled pilot comparison over fixed proposals but does not validate the proposed score on an independent test set.

## 3.6 Ethical and responsible-use considerations

The prototype processes visual recordings that may contain identifiable people. Any deployment should have a lawful purpose, appropriate authorization, access controls, retention limits, and human review. Appearance similarity is uncertain and can be affected by lighting, occlusion, camera quality, and model bias. The current project does not infer identity, age, or gender and does not provide reliable person tracking. Outputs should be treated as candidate clips for review, not as verified evidence about an individual.

# Chapter 4: Implementation

## 4.1 Development environment

The repository documents a Windows Python environment with PyTorch, Streamlit, Ultralytics, OpenCLIP, OpenCV, and FFmpeg integration. The application uses a local Moment-DETR checkpoint. Runtime environment and model weight availability can vary by machine; report the exact machine, operating system, package versions, CPU/GPU, and model checkpoint used for the final demonstration in **Table 4.1**. Do not imply that the checkpoint was trained on the project's surveillance videos.

## 4.2 Temporal localization

The Moment-DETR wrapper loads the configured checkpoint and extracts CLIP video features from the input recording. It accepts the query text and returns a ranked set of windows with start/end times and scores. The present model configuration supports at most 75 two-second segments, resulting in an approximate input limit of 150 seconds. The model proposes moments; it does not detect or track a named individual.

## 4.3 Person-crop attribute scoring

The appearance extension samples frames from candidate intervals and runs YOLO person detection. Detected boxes are cropped and compared with a text phrase using OpenCLIP. The resulting cosine similarity is used as a relative ranking signal. The application does not parse arbitrary natural language perfectly into a normalized set of attributes, and passing a full action sentence as an appearance phrase may yield misleading similarity. Score interpretation therefore depends on the chosen query and the crop quality.

## 4.4 Multi-frame consensus experiment

The pilot consensus method samples three frames evenly across a candidate. For each frame, it retains the strongest available person-crop similarity. The candidate score is the median of the per-frame evidence when enough frames provide usable evidence. This provides a simple way to reduce reliance on one sampled frame. Because person boxes are not associated across frames, the method can aggregate evidence from different detected people and is not an identity-consistent score.

## 4.5 Object-category evidence

The experimental object stage checks supported COCO object categories with YOLO. A positive category is considered repeated only when it is detected in at least two of the three sampled frames. The detector output is not linked to a person box, so the system cannot claim that a person carries or owns the object. Unsupported categories and missing detections are recorded as unavailable/unknown evidence.

## 4.6 Interval validation and clip generation

Candidate intervals are normalized and checked against source duration before extraction. FFmpeg writes each selected time window into an MP4 clip. Validation prevents negative or out-of-range timestamps from being used directly as extraction bounds. The repository's prior end-to-end report records successful clip probing for one supplied example; include the exact run and output only if it is used in the submitted demonstration.

## 4.7 User interface

The Streamlit application provides controls for video input and query entry and presents ranked clip results. The underlying pipeline has been exercised directly. A prior evaluation report records that the UI rendered its upload/query controls, while browser file selection and a complete browser-based upload-and-analyze interaction were not verified in that run. In the final submission, repeat a complete demonstration and document the exact tested path before claiming full UI verification.

# Chapter 5: Experiments and Results

## 5.1 Available data and evaluation boundary

The principal multi-frame reranking comparison covers five queries from one video, with the same proposals in each condition. It is a small exploratory pilot, not a statistically representative dataset. No confidence interval is reported. The results are temporal localization measurements and do not measure person identity accuracy, person tracking, object ownership, or generalization to new videos.

The repository also contains a separate one-query action/attribute evaluation report and QVHighlights validation artifacts. These records use different data and evaluation contexts and must not be merged into one result or treated as independent confirmation of the five-query pilot. QVHighlights validation metrics describe the supplied temporal/highlight checkpoint artifacts, not surveillance-domain appearance retrieval.

## 5.2 Five-query ablation results

All conditions use the same five queries and proposal sets. The denominator is five for every row. The table reports the temporal localization metrics recorded in the project's contribution report.

| Condition | Recall@0.3 | Recall@0.5 | Recall@0.7 | Mean IoU |
|---|---:|---:|---:|---:|
| A: Temporal baseline | 0.400 | 0.200 | 0.000 | 0.1940 |
| B: Saved single-frame appearance | 0.400 | 0.200 | 0.000 | 0.2118 |
| D: Saved person-object fusion | 0.800 | 0.400 | 0.000 | 0.4020 |
| E: Median multi-frame appearance | 0.600 | 0.400 | 0.000 | 0.3150 |
| F: E plus repeated positive object categories | 0.600 | 0.400 | 0.000 | 0.3150 |
| G: Single-frame appearance plus repeated object categories | 0.400 | 0.200 | 0.000 | 0.2118 |

The median multi-frame condition increased Recall@0.5 from 0.20 to 0.40 and mean IoU from 0.1940 to 0.3150 compared with the temporal baseline. The saved person-object fusion condition had the highest mean IoU in this pilot (0.4020). Results are strongly affected by individual examples, particularly the second query. The median condition did not exceed the person-object fusion condition. No tested condition achieved Recall@0.7. These values describe this five-query sample only.

## 5.3 Per-query observations

The project report records that the second query improved from IoU 0.0000 under the temporal baseline to 0.5158 under median multi-frame appearance. The first query improved under saved person-object fusion, while the third query remained at IoU 0.0000 across the reported conditions. The fourth and fifth queries had the same IoU across baseline and median conditions. These observations illustrate that aggregate improvement does not mean every query improved.

## 5.4 Object persistence findings

Conditions F and E produced identical metrics; condition G matched B. The report attributes these results to the absence of repeated positive detections for a supported requested object category in two sampled frames. This is an absence of usable positive detector evidence in the tested frames, not proof that the objects were absent from the video. The project has no validated person-object association measurement.

## 5.5 Proposal coverage

The reported proposal oracle has mean maximum IoU 0.5175, Recall@0.3 of 1.0, Recall@0.5 of 0.6, and Recall@0.7 of 0.0. The oracle is an upper bound based on selecting the best available proposal for each query. It shows that reranking is constrained by proposal coverage: if no candidate sufficiently overlaps the target interval, changing the ranking cannot recover it. It is not a deployable retrieval score.

## 5.6 Runtime and verification evidence

The recorded diagnostic generation took 611.2 seconds, including YOLO/OpenCLIP scoring and report generation. The consensus reranking and YOLO-only category pass took 37.2 seconds and reused prior temporal proposals and cached crop scores; Moment-DETR and CLIP were not rerun in that stage. The repository reports 72 tests passed for the broader project state at the time of the contribution report. These timings depend on hardware and cached artifacts and should not be presented as a clean, full end-to-end latency measurement.

## 5.7 Limitations and threats to validity

- Five queries from one video are insufficient to support general performance claims or robust statistical inference.
- The query sample is not independent across videos because it comes from one video.
- Person crops are selected independently per frame; no stable identity or track is established.
- YOLO category detections do not establish which person carries an object.
- Detector non-observation is unknown evidence and may reflect occlusion, scale, or detector error.
- The available annotated attribute subset lacks corresponding raw videos for a grounded appearance evaluation.
- The fixed proposal pool imposes an upper bound on reranking performance.
- The temporal baseline, saved fusion artifacts, and exploratory ablation have different provenance; comparisons must retain their stated setup.
- The system's short-video duration limit and dependence on local model assets constrain deployment.

# Chapter 6: Conclusion and Future Work

## 6.1 Conclusion

This project implemented a short-video natural-language retrieval prototype by integrating Moment-DETR temporal proposals, YOLO person crops, OpenCLIP similarity, interval validation, and FFmpeg clip extraction. It also compared multi-frame appearance aggregation and repeated object-category evidence against saved ranking conditions on a five-query, single-video pilot. Median multi-frame appearance improved Recall@0.5 and mean IoU over the temporal baseline in that sample, while the saved person-object fusion result was higher on mean IoU. Repeated object-category evidence did not change the reported pilot metrics.

The findings are preliminary and do not demonstrate general improvement. The project does not maintain person identity across frames, verify object ownership, or establish appearance-retrieval accuracy on a held-out multi-video dataset. Its principal outcome is an integrated prototype and an explicit account of the experiments needed to validate it.

## 6.2 Future work

1. Collect or obtain multiple raw videos with independently annotated temporal intervals and target-person boxes or IDs.
2. Add explicit appearance labels and person-object ownership annotations for each relevant interval.
3. Split data by video into training, validation, and held-out test partitions to avoid leakage.
4. Compare temporal-only, single-frame, and multi-frame appearance methods using prespecified metrics and confidence intervals.
5. Measure detector, tracking/association, and retrieval performance separately; do not infer one from another.
6. Improve proposal coverage for short events before attributing errors solely to reranking.
7. Evaluate query parsing and score calibration, including how action words and appearance phrases are separated.
8. Measure end-to-end runtime on a documented hardware setup and test the complete user-interface workflow.
9. Establish privacy, access, retention, and human-review procedures before any real-world use.

# References

Use your institution's required citation style and verify bibliographic details before submission.

1. Lei, J., Berg, T. L., and Bansal, M. “Detecting Moments and Highlights in Videos via Natural Language Queries.” *Advances in Neural Information Processing Systems*, 2021. arXiv:2107.09609. https://arxiv.org/abs/2107.09609
2. Radford, A., et al. “Learning Transferable Visual Models From Natural Language Supervision.” *Proceedings of the International Conference on Machine Learning*, 2021.
3. Ultralytics. “YOLO Documentation.” https://docs.ultralytics.com/ (record the version used).
4. OpenCLIP contributors. “OpenCLIP.” https://github.com/mlfoundations/open_clip (record the version and pretrained model used).
5. PyTorch contributors. “PyTorch Documentation.” https://pytorch.org/docs/ (record the version used).
6. FFmpeg contributors. “FFmpeg Documentation.” https://ffmpeg.org/documentation.html (record the version used).
7. QVHighlights dataset and Moment-DETR project materials. Cite the dataset paper and repository according to the institution's required style and the license terms of the material used.

# Appendices to complete

- **Appendix A:** User manual with screenshots of the tested application flow.
- **Appendix B:** System setup and reproducible commands, adapted from the project README.
- **Appendix C:** Full per-query metrics and experiment configuration from `outputs/attribute_aware_consensus_20261023/`.
- **Appendix D:** Test inventory and verification report, with test date and repository revision.
- **Appendix E:** Sample query format and example output, with any identifying video material removed or permission-cleared.

