# Missing Video Data Audit

**Project:** Natural Language Video Moment Retrieval (Moment-DETR baseline)  
**Audit scope:** map `data/moment_detr_with_attributes.jsonl` to its source videos; no retrieval code, data, checkpoint, or output was changed. This report is the only file created by this audit.

## Findings

The annotations are **QVHighlights validation-set** annotations plus a project-specific `attributes` object. This is not a case of the 301-row file being wholly disconnected from QVHighlights: every row's `qid`, `query`, `vid`, `duration`, `relevant_windows`, `relevant_clip_ids`, and `saliency_scores` exactly matches a row in `data/highlight_val_release.jsonl` (**301/301** checked). The attribute file has **301 rows**, **289 unique QVHighlights segments**, and **93 YouTube video IDs**.

The checkout has the annotations but not the corresponding QVHighlights raw-video collection. The upstream project documents raw video as a separately downloaded archive; its repo README also documents pre-extracted features as a separate 8 GB archive. This repository tracks annotation files, but no QVHighlights feature directory is present in the project data tree. Evidence supports **raw-video assets were not acquired/included in this checkout**, not a bad annotation-to-path setting: this application does not load these 301 rows to locate local files, and the training dataset loader expects feature files named from `vid` instead. The docs do not expose the raw archive's internal folder layout, so a more specific archive destination cannot be asserted from the checked-out documentation.

One exact video-ID file is present: `run_on_video/example/RoripwjYFp8_60.0_210.0.mp4` (**10,404,801 bytes**). Its segment ID occurs in one attribute row (qid 8056). Its query is “Chef makes pizza and cuts it up.”, ground-truth interval `[106, 122]`, while the added row says `action: entering` and `object: book`. This is a reliable **video/segment filename match**, but the query and these custom labels do not provide a credible appearance/action supervision example. No other project video filename exactly matches any of the 289 attribute-segment IDs. There are 181 video files when generated/cached `outputs/` are included; output clip names and `original_video.mp4` copies lack a verifiable `vid` manifest, and therefore cannot be assigned to the remaining annotations by content or filename from the evidence available. They were not treated as interchangeable.

## Dataset identity, source, and documented access

- **Dataset:** QVHighlights (Query-based Video Highlights), validation split.
- **Original/source reference:** the [official Moment-DETR repository](https://github.com/jayleicn/moment_detr) identifies the QVHighlights data and links the raw video archive. The upstream [data README](https://github.com/jayleicn/moment_detr/blob/main/data/README.md) defines segment IDs and annotation fields.
- **Raw-video archive URL documented upstream:** <https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz> (the published filename spells “highlights” as `qvhilights`). No archive request/download was made in this audit.
- **Documented access procedure:** follow the raw-video archive link from the official README, obtain the archive, then inspect/extract it locally. The README does not document selective per-ID downloads, an internal archive directory schema, or a complete extraction command for this video archive. Do not use the documented 8 GB `moment_detr_features.tar.gz` as a substitute for raw MP4s: those are pre-extracted model features. The code README's generic archive extraction example applies to that features archive.
- **Expected ID convention:** `vid` is `{youtube_id}_{start_time}_{end_time}`; upstream documents using the YouTube ID and start/end to identify the corresponding segment. The bundled raw demo's name confirms a matching filename convention: `{vid}.mp4`. The complete list of 289 IDs represented in this attribute file is below. Annotation membership does not guarantee that each source video is currently available from YouTube or the original archive.

### Exact segment IDs (filename stem expected by this checkout's demo convention)

```text
-_s0sXOfS3w_210.0_360.0
-_s0sXOfS3w_360.0_510.0
-_s0sXOfS3w_510.0_660.0
-_s0sXOfS3w_60.0_210.0
2iW3ei-5fpE_210.0_360.0
2iW3ei-5fpE_360.0_510.0
2iW3ei-5fpE_60.0_210.0
6Mzo1y19iTU_210.0_360.0
6Mzo1y19iTU_360.0_510.0
6Mzo1y19iTU_60.0_210.0
6OV8g7JkMvg_210.0_360.0
6OV8g7JkMvg_360.0_510.0
6OV8g7JkMvg_60.0_210.0
6aFvTqmnrJQ_210.0_360.0
6aFvTqmnrJQ_360.0_510.0
6aFvTqmnrJQ_510.0_660.0
6aFvTqmnrJQ_60.0_210.0
74jtgDnsEBU_210.0_360.0
74jtgDnsEBU_360.0_510.0
74jtgDnsEBU_60.0_210.0
8yHwJ3f7Fi4_210.0_360.0
8yHwJ3f7Fi4_360.0_510.0
8yHwJ3f7Fi4_510.0_660.0
8yHwJ3f7Fi4_60.0_210.0
AVErftnrqwY_210.0_360.0
AVErftnrqwY_510.0_660.0
AVErftnrqwY_60.0_210.0
AVErftnrqwY_660.0_810.0
A_MFAuOwK5k_210.0_360.0
A_MFAuOwK5k_360.0_510.0
A_MFAuOwK5k_510.0_660.0
A_MFAuOwK5k_60.0_210.0
B7bTx5l-N8I_60.0_210.0
Ci4jrBQXtTI_210.0_360.0
Ci4jrBQXtTI_360.0_510.0
Ci4jrBQXtTI_60.0_210.0
DTInxNfWXVc_210.0_360.0
DTInxNfWXVc_360.0_510.0
DTInxNfWXVc_60.0_210.0
DTrwYYi25Dc_210.0_360.0
DTrwYYi25Dc_360.0_510.0
DTrwYYi25Dc_60.0_210.0
DTrwYYi25Dc_660.0_810.0
E6yFTWlChmE_210.0_360.0
E6yFTWlChmE_360.0_510.0
E6yFTWlChmE_510.0_660.0
E6yFTWlChmE_60.0_210.0
E6yFTWlChmE_660.0_810.0
Ehpo8gXg27I_210.0_360.0
Ehpo8gXg27I_360.0_510.0
Ehpo8gXg27I_510.0_660.0
Ehpo8gXg27I_60.0_210.0
Ehpo8gXg27I_660.0_810.0
FL0Cos34RjU_210.0_360.0
FL0Cos34RjU_360.0_510.0
FL0Cos34RjU_60.0_210.0
FdMI99a1TZE_210.0_360.0
FdMI99a1TZE_360.0_510.0
FdMI99a1TZE_60.0_210.0
GAUdBAL0K5A_210.0_360.0
GAUdBAL0K5A_360.0_510.0
GAUdBAL0K5A_60.0_210.0
GOzEeIvLZO4_210.0_360.0
GOzEeIvLZO4_360.0_510.0
GOzEeIvLZO4_510.0_660.0
GOzEeIvLZO4_60.0_210.0
GOzEeIvLZO4_660.0_810.0
Gwdizqh16JY_210.0_360.0
Gwdizqh16JY_360.0_510.0
Gwdizqh16JY_60.0_210.0
H15bUc-1LJo_210.0_360.0
H15bUc-1LJo_360.0_510.0
H15bUc-1LJo_510.0_660.0
H15bUc-1LJo_60.0_210.0
H15bUc-1LJo_660.0_810.0
IAbAn-MkMH8_210.0_360.0
IAbAn-MkMH8_360.0_510.0
IAbAn-MkMH8_510.0_660.0
IAbAn-MkMH8_60.0_210.0
IdpHJK90VDw_210.0_360.0
IdpHJK90VDw_360.0_510.0
IdpHJK90VDw_510.0_660.0
IdpHJK90VDw_60.0_210.0
IdpHJK90VDw_660.0_810.0
J_6fDCo1REI_60.0_210.0
Jz1Cszaqck0_210.0_360.0
Jz1Cszaqck0_360.0_510.0
Jz1Cszaqck0_510.0_660.0
Jz1Cszaqck0_60.0_210.0
Jz1Cszaqck0_660.0_810.0
KQyJtq52Jcw_210.0_360.0
KQyJtq52Jcw_360.0_510.0
KQyJtq52Jcw_510.0_660.0
KQyJtq52Jcw_60.0_210.0
KQyJtq52Jcw_660.0_810.0
KRHEqswrh9I_210.0_360.0
KRHEqswrh9I_360.0_510.0
KRHEqswrh9I_510.0_660.0
KRHEqswrh9I_60.0_210.0
KRHEqswrh9I_660.0_810.0
KWBfQjuwp4E_210.0_360.0
KWBfQjuwp4E_60.0_210.0
L9cUEhaXnK4_210.0_360.0
L9cUEhaXnK4_360.0_510.0
L9cUEhaXnK4_510.0_660.0
L9cUEhaXnK4_60.0_210.0
L9cUEhaXnK4_660.0_810.0
MmGtg0qgceA_210.0_360.0
MmGtg0qgceA_360.0_510.0
MmGtg0qgceA_60.0_210.0
NUsG9BgSes0_210.0_360.0
NUsG9BgSes0_360.0_510.0
NUsG9BgSes0_510.0_660.0
NUsG9BgSes0_60.0_210.0
NUsG9BgSes0_660.0_810.0
NcRo-rLqXp4_210.0_360.0
NcRo-rLqXp4_360.0_510.0
NcRo-rLqXp4_60.0_210.0
Nhelnhenn1w_210.0_360.0
Nhelnhenn1w_360.0_510.0
Nhelnhenn1w_510.0_660.0
Nhelnhenn1w_60.0_210.0
Nhelnhenn1w_660.0_810.0
O5RKn-qeIVI_210.0_360.0
O5RKn-qeIVI_360.0_510.0
O5RKn-qeIVI_60.0_210.0
ObBiYbyai0Y_210.0_360.0
ObBiYbyai0Y_360.0_510.0
ObBiYbyai0Y_60.0_210.0
OckIALHg9Uc_60.0_210.0
PAnfDGUj0JI_60.0_210.0
PS5rtWdxn3A_210.0_360.0
PS5rtWdxn3A_360.0_510.0
PS5rtWdxn3A_510.0_660.0
PS5rtWdxn3A_60.0_210.0
PS5rtWdxn3A_660.0_810.0
PXj6QOKJ_5Q_60.0_210.0
RVOg4rk7MRg_210.0_360.0
RVOg4rk7MRg_360.0_510.0
RVOg4rk7MRg_60.0_210.0
RoripwjYFp8_210.0_360.0
RoripwjYFp8_360.0_510.0
RoripwjYFp8_60.0_210.0
T60_zAvnAsU_210.0_360.0
T60_zAvnAsU_360.0_510.0
T60_zAvnAsU_60.0_210.0
UFWQKrcbhjI_210.0_360.0
UFWQKrcbhjI_360.0_510.0
UFWQKrcbhjI_60.0_210.0
Vrsl2ZU1LWg_60.0_210.0
WBkLObX2Fdo_210.0_360.0
WBkLObX2Fdo_360.0_510.0
WBkLObX2Fdo_60.0_210.0
WEpnzfufxJQ_210.0_360.0
WEpnzfufxJQ_360.0_510.0
WEpnzfufxJQ_60.0_210.0
WEpnzfufxJQ_660.0_810.0
XpVt6Z1Gjjo_210.0_360.0
XpVt6Z1Gjjo_360.0_510.0
XpVt6Z1Gjjo_510.0_660.0
XpVt6Z1Gjjo_60.0_210.0
XpVt6Z1Gjjo_660.0_810.0
YIUaJzjNPuo_210.0_360.0
YIUaJzjNPuo_360.0_510.0
YIUaJzjNPuo_60.0_210.0
Z-L9RjWBTHg_210.0_360.0
Z-L9RjWBTHg_360.0_510.0
Z-L9RjWBTHg_60.0_210.0
Z3-IZ3HAmIA_210.0_360.0
Z3-IZ3HAmIA_360.0_510.0
Z3-IZ3HAmIA_60.0_210.0
ZWldnViJ4iY_210.0_360.0
ZWldnViJ4iY_360.0_510.0
ZWldnViJ4iY_60.0_210.0
_9vleIcuQd0_210.0_360.0
_9vleIcuQd0_360.0_510.0
_9vleIcuQd0_60.0_210.0
_NznSSfDxdo_210.0_360.0
_NznSSfDxdo_60.0_210.0
_i9qWLsZToY_210.0_360.0
_i9qWLsZToY_360.0_510.0
_i9qWLsZToY_60.0_210.0
_i9qWLsZToY_660.0_810.0
_t21xx94wZw_210.0_360.0
_t21xx94wZw_360.0_510.0
_t21xx94wZw_60.0_210.0
_ufoIeYZYjw_210.0_360.0
_ufoIeYZYjw_360.0_510.0
_ufoIeYZYjw_60.0_210.0
_ufoIeYZYjw_660.0_810.0
aG3duH53XDs_210.0_360.0
aG3duH53XDs_360.0_510.0
aG3duH53XDs_60.0_210.0
atDm_Gamcqo_210.0_360.0
atDm_Gamcqo_360.0_510.0
atDm_Gamcqo_60.0_210.0
bP5KfdFJzC4_210.0_360.0
bP5KfdFJzC4_360.0_510.0
bP5KfdFJzC4_510.0_660.0
bP5KfdFJzC4_60.0_210.0
bP5KfdFJzC4_660.0_810.0
cJ8kzdeoevg_210.0_360.0
cJ8kzdeoevg_360.0_510.0
cJ8kzdeoevg_60.0_210.0
eGBKPpkSoM0_210.0_360.0
eGBKPpkSoM0_60.0_210.0
fOU0-2bBRnU_210.0_360.0
fOU0-2bBRnU_360.0_510.0
fOU0-2bBRnU_60.0_210.0
geklhsKfw7I_210.0_360.0
geklhsKfw7I_60.0_210.0
gpDpN44mky0_210.0_360.0
gpDpN44mky0_360.0_510.0
gpDpN44mky0_60.0_210.0
h6QKDqomIPk_210.0_360.0
h6QKDqomIPk_360.0_510.0
h6QKDqomIPk_60.0_210.0
izeyQalOwGg_210.0_360.0
izeyQalOwGg_60.0_210.0
j9NNAmqweOA_210.0_360.0
j9NNAmqweOA_360.0_510.0
j9NNAmqweOA_510.0_660.0
j9NNAmqweOA_60.0_210.0
j9NNAmqweOA_660.0_810.0
jv7033VUyHE_60.0_210.0
jyHSYDfqPnU_210.0_360.0
jyHSYDfqPnU_360.0_510.0
jyHSYDfqPnU_60.0_210.0
kgwvIEF1A9E_60.0_210.0
kiSm0Nuqomg_60.0_210.0
lwNho_1tKrc_210.0_360.0
lwNho_1tKrc_60.0_210.0
nY42UppPhhg_210.0_360.0
nY42UppPhhg_360.0_510.0
nY42UppPhhg_60.0_210.0
n_NdBRz6Z1E_210.0_360.0
n_NdBRz6Z1E_360.0_510.0
n_NdBRz6Z1E_60.0_210.0
opM6YIx3gA8_60.0_210.0
pA6Z-qYhSNg_210.0_360.0
pA6Z-qYhSNg_360.0_510.0
pA6Z-qYhSNg_60.0_210.0
pu5p_ojK6BI_210.0_360.0
pu5p_ojK6BI_360.0_510.0
pu5p_ojK6BI_60.0_210.0
qBe1PUrRHUo_210.0_360.0
qBe1PUrRHUo_360.0_510.0
qBe1PUrRHUo_510.0_660.0
qBe1PUrRHUo_60.0_210.0
r7A-cfBq2Xw_210.0_360.0
r7A-cfBq2Xw_360.0_510.0
r7A-cfBq2Xw_60.0_210.0
rNPSRSs3reQ_60.0_210.0
rrTIeJRVGjg_210.0_360.0
rrTIeJRVGjg_360.0_510.0
rrTIeJRVGjg_60.0_210.0
tlQPRLVDyH4_60.0_210.0
uGPMVvhFSCg_60.0_210.0
uoVRb7a58GU_210.0_360.0
uoVRb7a58GU_360.0_510.0
uoVRb7a58GU_60.0_210.0
wUgPzvcKK5c_210.0_360.0
wUgPzvcKK5c_360.0_510.0
wUgPzvcKK5c_60.0_210.0
x4kKIm8WWOA_210.0_360.0
x4kKIm8WWOA_360.0_510.0
x4kKIm8WWOA_510.0_660.0
x4kKIm8WWOA_60.0_210.0
x4kKIm8WWOA_660.0_810.0
xUeRLu1o0v4_210.0_360.0
xUeRLu1o0v4_360.0_510.0
xUeRLu1o0v4_60.0_210.0
xtaVwcSitCo_210.0_360.0
xtaVwcSitCo_360.0_510.0
xtaVwcSitCo_60.0_210.0
yId2wIocTys_210.0_360.0
yId2wIocTys_360.0_510.0
yId2wIocTys_60.0_210.0
yeCaAL4uM-U_210.0_360.0
yeCaAL4uM-U_360.0_510.0
yeCaAL4uM-U_60.0_210.0
yeCaAL4uM-U_660.0_810.0
yutpAXhjMlI_60.0_210.0
zVwsEVwS8Kw_210.0_360.0
zVwsEVwS8Kw_360.0_510.0
zVwsEVwS8Kw_60.0_210.0
zY6XnLhTp7g_210.0_360.0
zY6XnLhTp7g_360.0_510.0
zY6XnLhTp7g_60.0_210.0
```

## Annotation schema and association limits

Each JSONL line is a dict. The inherited QVHighlights fields mean:

| Field | Meaning for mapping/evaluation |
|---|---|
| `qid` | Unique query/annotation ID; all 301 IDs were found in the validation release. |
| `vid` | Source YouTube segment ID `{youtube_id}_{start}_{end}`; determines exact source segment identity. |
| `query` | Natural-language query text. |
| `duration` | Segment duration in seconds; these attribute subset rows report 150 seconds. |
| `relevant_windows` | Ground-truth query-relevant interval(s), in seconds relative to the segment. |
| `relevant_clip_ids` | Relevant 2-second clip indices. |
| `saliency_scores` | Three annotator saliency values per listed relevant clip. |

The project-specific `attributes` dict contains `gender`, `age_group`, `shirt_color`, `pant_color`, `shoe_color`, `bag`, `cap`, `mask`, `helmet`, `glasses`, `object`, and `action`. Values are row-level categorical values (including `none` and yes/no labels), not a person table. There is **no person ID, bounding box, track ID, per-person time interval, or person-to-object relation field**. Thus these fields identify an annotated row's attribute categories, but do not establish which detected person has them or whether an object is carried by that person. The exact provenance/annotation protocol for these 12 extra fields is not documented in the project README or the commit that introduced this JSONL.

## Repository evidence and paths checked

- `data/README.md`, `data/LICENSE`: upstream dataset description, `vid` convention, intervals, saliency and license.
- `README.md`: upstream feature archive (8 GB), raw-video archive link, and demo filename/query.
- `data/moment_detr_with_attributes.jsonl`: tracked 301-row file.
- `data/highlight_val_release.jsonl`: all 301 base records match exactly on inherited annotation fields.
- `moment_detr/start_end_dataset.py`: training/evaluation data loader reads feature files from configured feature directories as `{vid}.npz`; it does not resolve raw MP4 paths.
- `run_on_video/example/`: contains the one exact matching raw/demo MP4 above and its `queries.jsonl`.
- `outputs/`: cached/generated videos inspected by filename; no remaining exact ID-stem match or source-ID manifest was found.
- Project-wide video-extension scan excluded `venv/`, included generated `outputs/`, and found **181** files total. Exact-stem join against annotation `vid` values produced **1** result(s).
- Git root: `D:\jegan_repo\FYP`. `git log --follow` shows the attribute JSONL introduced in commit `3f72786` (“Fix - project repository structure”, 2026-09-06). No prior history or commit description records attribute-label provenance.

### Git state observed (preserved)

`git status --short` already showed unrelated/pre-existing modified and untracked work, including edits under `app.py`, `extension/`, and `run_on_video/`, reports/design docs, tests, generated `outputs/` sessions, `Ultralytics/`, and `venv/`. These were not changed or cleaned by this audit. Recent commits:

```text

```

## Smallest valid evaluation subset for four ranking conditions

For a **mechanically comparable** temporal-only, temporal+appearance, temporal+action, and combined ranking run, the same query/video instances and temporal candidate windows must be used in all four conditions. Minimum data requirements per instance are:

1. the exact raw segment file for its `vid`, plus a validated mapping and duration;
2. a query and at least one annotated relevant temporal interval, and a candidate list containing both relevant and non-relevant/less-relevant windows so ranking can change;
3. an appearance description/label with a defensible association to the target person or person crop;
4. an action description/label aligned to the query and interval.

The strict toy minimum is one combined query instance with both kinds of supervision and at least two distinct ranked candidates; this is only a pipeline/debug subset, not a defensible aggregate evaluation. To exercise the three query strata, use at least one appearance-only, one action-only, and one combined instance, all with matched video, interval, and relevant person-level labels. A meaningful reported comparison needs a larger, predeclared query set and ranking metrics; choose its size from the intended statistical precision rather than presenting a one-row result as evidence. In the current data the sole matched segment's custom action/object labels conflict with its query and no person-level association exists, so **there is no valid four-condition subset yet**.

## Next required action

Obtain, with approval, the official QVHighlights raw-video archive (or a legally authorized copy of the exact required segment files), then inventory the archive before extraction/copying and join files only by the full `vid` segment ID and verified interval. Preserve a manifest with `qid`, `vid`, archive path, extracted path, byte size, duration, and verification status. Separately document how the custom `attributes` labels were produced and add person-level association/grounding if the evaluation claims person-attribute retrieval. Do not run evaluation on unmatched `outputs/` files or treat the one demo match as representative.

## Conclusion

**Diagnosis:** expected QVHighlights raw-video assets are absent from the checkout; annotations are present and correctly aligned to the validation release. One bundled demo segment matches exactly; the other 288 annotated segment IDs have no exact local video match. The attribute extension's provenance and person-level grounding remain undocumented. No dataset download, annotation edit, algorithm change, or output modification was performed.
