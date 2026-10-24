# Evaluation Dataset Plan: Action and Soft-Biometric Retrieval

**Project:** `moment-detr-baseline`  
**Status:** planning only; no raw video was downloaded or extracted, no annotation or output was changed, no ranking code was changed, and no metrics were produced.  
**Prepared:** 2026-10-23

## Executive finding

The 301-row `data/moment_detr_with_attributes.jsonl` file is an attribute-augmented subset of QVHighlights validation data: all 301 rows match `data/highlight_val_release.jsonl` on the original fields (`qid`, `query`, `vid`, duration, temporal windows, clip IDs, and saliency scores). It contains 289 distinct 150-second segment IDs. The attribute values are not documented in project material or Git history beyond being added in commit `3f72786`; they have no person IDs, boxes, tracks, or person-object relation labels.

The official Moment-DETR README points to a raw-video `.tar.gz` archive, but its exact availability and size could not be verified from this environment. A metadata-only HEAD attempt failed with `No such host is known` for `nlp.cs.unc.edu`; the browser fetcher also could not open the archive URL. These failures describe this environment’s access, not proof that the archive is unavailable everywhere. No official archive size was returned, so the exact expected download size is **unknown**. The 8 GB archive described in the Moment-DETR README is **pre-extracted model features**, not the raw videos, and is not a size estimate for the raw archive.

A third-party Hugging Face dataset card identifies a mirror of all 25,124 QVHighlights videos and reports **135.3 GB** in its description (the hosting page displays **145 GB** total). Treat this as a full-mirror scale estimate only: it is not an official archive size, and this audit did not verify the selected files, their per-file sizes, or underlying video rights. Do not start a full mirror download. Selective retrieval of individual files may be possible according to that card, but should be approved and license-reviewed before use.

The smallest practical **candidate** set found in the current attribute file is **three query rows across two unique QVHighlights segments**: an appearance-focused query and an action-focused query share one 150-second segment; a combined query uses a second segment. The segment MP4s are not in this project. Their media duration, decodability, and archive presence remain unverified. The table below is a metadata mapping, not a validated video manifest.

## Verified source and access / license conditions

- **Dataset/source:** QVHighlights, the Query-based Video Highlights dataset distributed with the official [Moment-DETR repository](https://github.com/jayleicn/moment_detr). The official [data README](https://github.com/jayleicn/moment_detr/blob/main/data/README.md) documents JSONL annotations, `vid` formatting as `{youtube_id}_{start_time}_{end_time}`, relevant windows in seconds, and how the YouTube ID plus start/end identify a source segment.
- **Official raw-video link in the checked-in/upstream README:** [`qvhilights_videos.tar.gz`](https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz) (the published filename uses the spelling `qvhilights`). A separate recent paper supplement also writes `qvhighlights_videos.tar.gz`; that spelling variation was not independently verified against the server. Use the link attached to the official repository README as the authoritative published link, and check with the maintainer if it remains unavailable.
- **Official instructions:** the data README says raw videos can be downloaded from its link. It does not specify the raw archive size, selective per-ID retrieval, archive directory layout, or a complete extraction command. After approval and successful access, inspect the archive listing first; extract only what is needed if the archive format permits it. Do not assume an extraction directory or path before inspecting the archive.
- **Size/availability checks performed:** one HTTP `HEAD` request to the official URL failed locally at DNS resolution; the browser source fetch also returned inaccessible. No response headers or `Content-Length` were obtained. Hence: official archive size unknown, archive availability unresolved, no download made.
- **Third-party mirror:** the [Hugging Face `ayushsdev/qvhighlights-videos` dataset card](https://huggingface.co/datasets/ayushsdev/qvhighlights-videos) describes 25,124 videos, states a total of 135.3 GB, reports an overall page size of 145 GB, gives a `{first-character}/{vid}.mp4` layout, and documents per-file access. This is **not the official source** and its size/license statements are mirror metadata. It is only a potential selective-access fallback, pending a per-file existence/size check and rights review. The apparent size difference may reflect display units/metadata; do not treat either number as the raw UNC archive size.
- **License:** upstream Moment-DETR documentation says the **annotation files** are CC BY-NC-SA 4.0 and code is MIT; see the checked-in [`data/LICENSE`](data/LICENSE). This does not establish a license for the underlying YouTube videos. The third-party mirror labels its collection CC BY-NC-SA 4.0, but that mirror label does not by itself establish rights granted by the original video owners. Obtain the original media terms/permission and follow applicable platform terms before redistributing or using video files. The current evidence does not establish that raw-video use is prohibited; it also does not establish blanket permission.

## Selected candidate subset and manifest fields

Query strata below are assigned from the query wording for dataset design only; they are not ground-truth labels for a person. `declared_duration` is copied from the JSONL annotation. `measured_duration` is `NOT VERIFIED` because no exact MP4 is available locally. Current local video path is reported as absent; no path has been invented. If a source archive is obtained, record its actual extracted path after inspecting its layout.

| qid | vid | Query | Ground-truth interval(s), seconds | Declared duration | Current exact local video path | Measured duration | Validation status |
|---:|---|---|---|---:|---|---|---|
| 548 | `lwNho_1tKrc_60.0_210.0` | “Man in striped shirt is wearing a orange lei.” | `[2, 28]`; `[30, 48]` | 150 s | **MISSING** — no exact `lwNho_1tKrc_60.0_210.0.mp4` in project or outputs | NOT VERIFIED | Annotation matches QVHighlights val; video file and attributes unverified; candidate appearance-focused query |
| 1918 | `lwNho_1tKrc_60.0_210.0` | “The three men are given leis and dance while they are greeted by musicians playing.” | `[0, 26]` | 150 s | **MISSING** — same segment as qid 548 | NOT VERIFIED | Annotation matches QVHighlights val; video file and attributes unverified; candidate action-focused query |
| 5071 | `NUsG9BgSes0_60.0_210.0` | “A woman sitting in front of a desk wearing headphones and using her laptop” | `[118, 136]` | 150 s | **MISSING** — no exact `NUsG9BgSes0_60.0_210.0.mp4` in project or outputs | NOT VERIFIED | Annotation matches QVHighlights val; video file and attributes unverified; candidate combined query |

**Candidate file stems:** `lwNho_1tKrc_60.0_210.0.mp4` and `NUsG9BgSes0_60.0_210.0.mp4`. These are exact QVHighlights segment-ID-derived names; actual archive paths/layout and file presence are not confirmed. A full local filename scan found only one exact segment match among the 289 IDs: `RoripwjYFp8_60.0_210.0.mp4`. That video is not part of this candidate subset and its attribute row was flagged in `MISSING_VIDEO_DATA_AUDIT.md` as unsuitable without manual relabeling.

### Why this is the smallest practical candidate set

Two unique video segments cover three query forms: appearance-focused and action-focused rows share the `lwNho...` segment, and one combined row uses `NUsG9...`. This keeps video retrieval/annotation limited to two clips while allowing the same source segment to support two query types. It is a **pilot/dev set only**, not enough for a reliable general performance claim. Before results, expand to a predeclared set with multiple distinct source videos and enough queries in each stratum. Use the same exact videos, query instances, temporal candidates, and evaluation measure across temporal-only, temporal+appearance, temporal+action, and combined runs.

## Audit of custom `attributes`

The attribute JSONL has 301 rows and twelve categorical fields under `attributes`: `gender`, `age_group`, `shirt_color`, `pant_color`, `shoe_color`, `bag`, `cap`, `mask`, `helmet`, `glasses`, `object`, and `action`. The row values are filled, but the project contains no source document, annotation instructions, annotator records, person reference, confidence values, provenance/license statement for the additions, or review/correction history. Git shows the 301-row file first introduced in commit `3f72786` (“Fix - project repository structure”); history does not explain how those labels were produced.

The JSONL schema has no `person_id`, `bbox`, `frame_id`, `track_id`, person-specific interval, or person-object relation. Therefore **all custom attribute labels must be considered unverified** before they are used as person-level evaluation ground truth. The values `none`, `yes`, and `no` are categories, not proof that an attribute is visually absent or that a person was checked. `object: X` means only that the row carries that category; it does not encode who interacted with X or whether X was held/carried.

Examples to review manually; these are **flags**, not automatic corrections:

- qid 1918 query describes three men dancing, while its row-level attribute says `gender: female`, `action: cooking`, `object: laptop`, `helmet: yes`. Without person-level IDs, it is not possible to tell whether this is a wrong label, an unrelated person, or a schema/provenance problem. It must not be scored as a person-label match.
- qid 5071 query mentions a woman sitting and using a laptop; the row says `object: shopping_bag` and `action: talking`. These could be additional co-occurring content, but the schema cannot identify the target person or support the relation. Verify against video; do not silently change labels from query wording alone.
- qid 548 describes a striped shirt and orange lei, whereas the row offers color/object categories but no pattern or lei category. `shirt_color: pink` and `object: cup` need visual review; query text alone cannot establish a correction.

The existing exact local example qid 8056 similarly has query “Chef makes pizza and cuts it up.” and labels `action: entering`, `object: book`; it is a provenance/review warning, not a safe substitute evaluation item.

## Manual annotation protocol for selected videos

Perform this only after legal access to the exact two segments is established and each MP4 has been checked against its full `vid` stem. Keep the existing QVHighlights temporal ground truth separate from added person attributes.

1. **Video identity / QC:** record source URL/archive reference, full `vid`, `qid`, local path, SHA-256, byte size, codec, frame rate, decoded duration, and validation outcome. Require full segment-ID agreement; compare measured duration with the annotation’s 150 s and investigate any difference. Never link a file based only on similar content or YouTube ID prefix.
2. **Temporal action table (query/video level):** for each query, annotate the action/event label(s), start/end seconds within the segment, annotator, and confidence/evidence note. Review the complete QVH relevant interval plus context; retain original `relevant_windows` separately. Do not copy `attributes.action` into this table. If a specific person is visibly responsible, record an optional `actor_person_id` with evidence; otherwise leave it unknown.
3. **Person frame table:** sample three frames within each QVH relevant interval (near start, midpoint, near end; avoid boundary ambiguity). Add a frame if visibility changes or a carried-object claim needs evidence. For each visible person record timestamp, stable within-video `person_id`, pixel `bbox_xyxy`, visibility/occlusion, and whether the person is the query target with confidence. Use `unknown` when uncertain. Reuse a person ID only with a clear visual match; this is annotation identity within one clip, not real-world identity.
4. **Appearance table:** attach each label to `(vid, frame/time, person_id)`, not the whole query row. Annotate directly visible clothing/accessories (top color/pattern, lower-body clothing color, footwear color, headwear, eyewear, face covering, bag type) with values `present`, `absent`, `unobservable`, or `uncertain` and annotator confidence. Avoid guessing hidden attributes. Age/gender are subjective/sensitive; omit them from the primary benchmark or separately justify categories, consent/ethics, uncertainty, and `not assessable` handling before use.
5. **Object and relation table:** annotate object category and its own box separately from person appearance. Add `person_id`, relation (`held`, `worn`, `carried`, `interacting`, `nearby`, `unknown`), evidence frame/time, and confidence only when visible evidence supports it. Mere co-occurrence or proximity must be `nearby`/`unknown`, never `carried`.
6. **Independent review:** two annotators independently label the pilot; adjudicate disagreements and preserve both original judgments plus a resolution. Freeze the protocol and manifest before scoring. Do not use model predictions to write labels.

Suggested files/columns for a later annotation package:

- `video_manifest.csv`: `qid,vid,query,source_ref,video_path,sha256,size_bytes,annotation_duration_s,measured_duration_s,decode_ok,status`.
- `person_frames.csv`: `vid,qid,timestamp_s,person_id,x1,y1,x2,y2,visibility,target_match,annotator,confidence`.
- `person_attributes.csv`: `vid,qid,timestamp_s,person_id,attribute,value,confidence,evidence_note,annotator`.
- `objects_relations.csv`: `vid,qid,timestamp_s,object_id,category,x1,y1,x2,y2,person_id,relation,confidence,evidence_note,annotator`.
- `actions.csv`: `vid,qid,action_label,start_s,end_s,actor_person_id_or_unknown,confidence,annotator`.

## Estimated effort and next steps

Planning estimate, not a measured labor result: retrieve and QC two 150-second segments; inspect three query intervals; double-annotate sampled person boxes/attributes and action intervals; resolve disagreements. Budget roughly **2–4 annotator-hours per reviewer** for this two-video pilot, plus access/download/QC time and adjudication. Multi-person scenes, occlusion, or repeated objects can increase that estimate. A later statistically useful evaluation requires more segments and a separately justified annotation budget.

1. **Resolve access:** retry the official archive URL from a network that can resolve the UNC host; request/record `Content-Length` (or archive listing size) before downloading. If the archive is large, stop and obtain explicit approval before download/extraction. Ask the dataset maintainer if the link or filename spelling is unresolved.
2. **If official access is unavailable:** request approval to verify the exact two files and per-file sizes in the third-party mirror, and conduct license/provenance review before selectively obtaining them. Do not treat the mirror’s CC label as proof of rights in source footage.
3. **If video rights/access cannot be established:** replace these candidates with a clearly labelled **new, manually annotated pilot set** from user-provided or institutionally approved footage with permission/consent; do not present it as QVHighlights or mix it into the QVH test split. Preserve a source/consent record and an explicit train/dev/test separation.
4. **After approved retrieval:** inventory archive contents without bulk extraction if possible; match only full `vid` stems; create `video_manifest.csv`; measure each clip with ffprobe/OpenCV and decode-check; update status to valid only after the file’s content, duration, and interval coordinate system are verified.
5. **Audit labels:** compare the custom attributes visually against the selected frames, document corrections in a separate adjudicated annotation file (never overwrite source JSONL), and add explicit person/object associations as above.
6. **Freeze and evaluate:** select a larger predeclared set after the pilot protocol works; run all ranking conditions on identical examples and candidate windows. Report metrics only after data and labels pass QC. No improvement is asserted by this plan.

## Current status summary

| Item | Status |
|---|---|
| QVHighlights annotation identity | Verified against local validation release (301/301 base rows) |
| Official raw archive link | Documented upstream; archive reachability unresolved from this environment |
| Official raw archive size | Unknown; no size header returned |
| Full-mirror indicative scale | Third-party metadata reports 135.3 GB / 145 GB; not official and not downloaded |
| Candidate query rows | 3 (`qid` 548, 1918, 5071), 2 unique `vid`s |
| Candidate exact videos locally present | 0 of 2 |
| Candidate measured video durations | 0 of 2; 150 s is annotation metadata only |
| Custom attribute provenance / person linkage | Not documented / absent |
| Manual labels or metrics created | None |
| Ranking algorithm changed | No |

