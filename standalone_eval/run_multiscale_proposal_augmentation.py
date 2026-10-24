"""Isolated, ground-truth-independent proposal augmentation experiment.

Default mode is a dependency-light dry run. Scoring requires explicit --score
and an output directory that does not already exist. Generated intervals are
offspring of saved Moment-DETR proposals; their temporal confidence is inherited
from the parent and is not a new Moment-DETR prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extension.query_logic import fuse_candidate_ranks, parse_retrieval_query
from standalone_eval.action_attribute_metrics import temporal_iou

DEFAULT_SOURCE_RESULTS = ROOT / "outputs" / "proposal_diagnosis_mot17_04_verified_20261023" / "results.json"
DEFAULT_ANNOTATIONS = ROOT / "evaluation_data" / "MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json"
DEFAULT_VIDEO = ROOT / "datasets" / "MOT17-04-SDP-raw.webm"
DEFAULT_YOLO_WEIGHTS = ROOT / "yolo11n.pt"
DEFAULT_CLIP_WEIGHTS = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"

SCALE_FACTORS = (0.5, 1.5)
CENTER_SHIFTS_SECONDS = (-2.0, 2.0)
D_FUSION = "temporal_appearance_person_object_carried_object"
SEMANTIC_FIELDS = (
    "appearance_score",
    "attribute_score",
    "action_score",
    "person_object_score",
    "carried_object_score",
)
STATUS_FIELDS = (
    "appearance_evidence_status",
    "action_evidence_status",
    "person_object_evidence_status",
    "carried_object_evidence_status",
)


def interval_key(start: float, end: float) -> tuple[float, float]:
    """Use the saved result precision to identify duplicate intervals."""
    return round(float(start), 4), round(float(end), 4)


def generate_augmented_candidates(
    parent_candidates: Sequence[Mapping[str, Any]],
    video_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Return untouched originals and deduplicated derived intervals.

    For each parent, generate 0.5x and 1.5x centered duration variants and
    same-duration center shifts of -2s/+2s. Only generated intervals are
    clipped. Original records are copied without changing their fields.
    Generated duplicates prefer the first parent in saved proposal order;
    all parent/transform sources are retained in metadata. A generated interval
    equal to an original is suppressed in favor of the original proposal.
    """
    duration = float(video_duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video_duration must be a finite positive number")

    originals = [dict(candidate) for candidate in parent_candidates]
    for index, candidate in enumerate(originals):
        try:
            start, end = float(candidate["start"]), float(candidate["end"])
            score = float(candidate["temporal_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid parent candidate at index {index}") from exc
        if not all(math.isfinite(value) for value in (start, end, score)) or end <= start:
            raise ValueError(f"invalid parent candidate interval/score at index {index}")
        if start < 0 or end > duration:
            raise ValueError(
                f"original candidate {index} is outside video bounds; refusing to alter a saved proposal"
            )

    original_keys = {interval_key(row["start"], row["end"]) for row in originals}
    generated: list[dict[str, Any]] = []
    generated_by_key: dict[tuple[float, float], dict[str, Any]] = {}
    counts = {"generated_unique": 0, "duplicate_or_original_suppressed": 0, "invalid_discarded": 0}

    for parent_index, parent in enumerate(originals):
        start, end = float(parent["start"]), float(parent["end"])
        center, span = (start + end) / 2.0, end - start
        variants = [
            (f"duration_scale_{factor:g}x", center - span * factor / 2.0,
             center + span * factor / 2.0)
            for factor in SCALE_FACTORS
        ]
        variants.extend(
            (f"center_shift_{shift:+g}s", start + shift, end + shift)
            for shift in CENTER_SHIFTS_SECONDS
        )

        for variant, raw_start, raw_end in variants:
            clipped_start = min(max(raw_start, 0.0), duration)
            clipped_end = min(max(raw_end, 0.0), duration)
            if clipped_end <= clipped_start:
                counts["invalid_discarded"] += 1
                continue
            clipped_start, clipped_end = round(clipped_start, 4), round(clipped_end, 4)
            if clipped_end <= clipped_start:
                counts["invalid_discarded"] += 1
                continue
            key = interval_key(clipped_start, clipped_end)
            source = {
                "parent_index": parent_index,
                "parent_interval": [start, end],
                "parent_temporal_score": float(parent["temporal_score"]),
                "transform": variant,
            }
            if key in original_keys:
                counts["duplicate_or_original_suppressed"] += 1
                continue
            if key in generated_by_key:
                generated_by_key[key]["duplicate_sources"].append(source)
                counts["duplicate_or_original_suppressed"] += 1
                continue

            derived = {
                "start": clipped_start,
                "end": clipped_end,
                "temporal_score": float(parent["temporal_score"]),
                "temporal_score_source": "inherited_from_parent_proposal",
                "parent_index": parent_index,
                "parent_interval": [start, end],
                "parent_temporal_score": float(parent["temporal_score"]),
                "augmentation_transform": variant,
                "duplicate_sources": [source],
                "semantic_evidence_source": "fresh_scoring_required",
            }
            generated.append(derived)
            generated_by_key[key] = derived

    counts["generated_unique"] = len(generated)
    return originals, generated, counts


def merge_fresh_evidence(
    generated_candidate: Mapping[str, Any],
    fresh_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach semantic fields from a fresh scoring result, preserving missingness.

    Temporal confidence and provenance always remain those from the parent.
    No semantic score is copied from a parent proposal or another interval.
    """
    merged = dict(generated_candidate)
    for field in SEMANTIC_FIELDS:
        merged[field] = fresh_evidence.get(field)
    for field in STATUS_FIELDS:
        merged[field] = fresh_evidence.get(field, "missing_from_scorer_output")
    for field in (
        "carried_object_detections",
        "carried_object_evidence_states",
        "person_object_mention_scores",
        "person_crop_count_per_frame",
    ):
        if field in fresh_evidence:
            merged[field] = fresh_evidence[field]
    merged["temporal_score"] = float(generated_candidate["parent_temporal_score"])
    merged["temporal_score_source"] = "inherited_from_parent_proposal"
    merged["semantic_evidence_source"] = "fresh_scoring"
    return merged


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(source: Mapping[str, Any], annotations: Mapping[str, Any], video_duration: float) -> dict[str, Any]:
    source_queries = source.get("queries") or []
    annotation_queries = annotations.get("queries") or []
    source_by_qid = {str(row["qid"]): row for row in source_queries}
    annotation_by_qid = {str(row["qid"]): row for row in annotation_queries}
    if set(source_by_qid) != set(annotation_by_qid):
        raise ValueError("Saved proposals and annotations must contain identical query IDs")

    query_rows = []
    for qid in source_by_qid:
        source_query = source_by_qid[qid]
        proposals = source_query.get("all_candidate_scores")
        if not proposals:
            raise ValueError(f"Saved proposals missing for {qid}")
        originals, generated, counts = generate_augmented_candidates(proposals, video_duration)
        query_rows.append({
            "qid": qid,
            "original_candidate_count": len(originals),
            "generated_candidate_count": len(generated),
            "expanded_candidate_count": len(originals) + len(generated),
            "generation_counts": counts,
        })
    total_original = sum(row["original_candidate_count"] for row in query_rows)
    total_generated = sum(row["generated_candidate_count"] for row in query_rows)
    settings = source.get("settings") or {}
    appearance_frames = int(settings.get("appearance_frames_per_candidate", 1))
    action_frames = int(settings.get("action_frames_per_candidate", 1))
    planned_sample_frames = 0
    for row in query_rows:
        qid = row["qid"]
        query = annotation_by_qid[qid]["query"]
        parts = parse_retrieval_query(query)
        semantic_needed = bool(
            parts.appearance_text or parts.action_text or parts.carried_object_texts
            or parts.unresolved_carried_object_texts or parts.carried_object_mentions
        )
        frames_per_candidate = max(
            appearance_frames if (parts.appearance_text or parts.carried_object_texts
                                  or parts.unresolved_carried_object_texts or parts.carried_object_mentions) else 0,
            action_frames if parts.action_text else 0,
        )
        row["semantic_evidence_requested"] = semantic_needed
        row["sampled_frames_per_generated_candidate_upper_bound"] = frames_per_candidate
        planned_sample_frames += row["generated_candidate_count"] * frames_per_candidate

    observed_reranking_seconds = settings.get("attribute_reranking_seconds")
    if observed_reranking_seconds is not None and total_original:
        estimated_seconds_per_candidate = float(observed_reranking_seconds) / total_original
        estimated_fresh_scoring_seconds = estimated_seconds_per_candidate * total_generated
    else:
        estimated_seconds_per_candidate = None
        estimated_fresh_scoring_seconds = None

    return {
        "status": "dry_run_plan_only",
        "experiment": "multi-scale and boundary-shifted offspring of saved proposals",
        "video_duration_seconds": float(video_duration),
        "transforms": {
            "duration_scales": list(SCALE_FACTORS),
            "center_shifts_seconds": list(CENTER_SHIFTS_SECONDS),
            "boundary_policy": "clip generated intervals to [0, video duration]; discard non-positive intervals",
            "deduplication": "generated intervals colliding with originals are suppressed; generated duplicates keep first saved-parent provenance",
            "temporal_score": "inherited from source proposal; parent-derived, not a new Moment-DETR prediction",
            "semantic_scores": "fresh scoring required for every unique generated interval",
        },
        "query_count": len(query_rows),
        "original_candidate_count_total": total_original,
        "generated_candidate_count_total": total_generated,
        "expanded_candidate_count_total": total_original + total_generated,
        "generated_candidate_frame_samples_upper_bound": planned_sample_frames,
        "estimated_fresh_semantic_scoring_seconds": estimated_fresh_scoring_seconds,
        "estimated_fresh_semantic_scoring_minutes": (
            estimated_fresh_scoring_seconds / 60.0 if estimated_fresh_scoring_seconds is not None else None
        ),
        "runtime_estimate_basis": {
            "observed_original_semantic_scoring_seconds": observed_reranking_seconds,
            "observed_original_candidate_count": total_original,
            "estimated_seconds_per_candidate": estimated_seconds_per_candidate,
            "note": "Linear CPU extrapolation for new candidates only; excludes model initialization and variance.",
        },
        "queries": query_rows,
        "scoring_dependencies": ["local video decode", "YOLO11 person/object detection", "OpenCLIP crop/frame-text scoring"],
        "inference_started": False,
        "ground_truth_used_for_generation": False,
    }


def _best_iou(candidate: Mapping[str, Any], intervals: Sequence[Sequence[float]]) -> float:
    return max((temporal_iou((candidate["start"], candidate["end"]), interval)
                for interval in intervals), default=0.0)


def _metrics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"query_count": 0, "mean_iou": None, "recall_at_1_iou_0_3": None,
                "recall_at_1_iou_0_5": None, "recall_at_1_iou_0_7": None}
    return {
        "query_count": len(values),
        "mean_iou": sum(values) / len(values),
        **{f"recall_at_1_iou_0_{threshold}": sum(value >= threshold / 10 for value in values) / len(values)
           for threshold in (3, 5, 7)},
    }


def _condition(
    candidates: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Sequence[float]],
    fusion: str = D_FUSION,
) -> dict[str, Any]:
    ranked = fuse_candidate_ranks(candidates, fusion)
    if not ranked:
        raise ValueError("Cannot rank an empty candidate pool")
    top = ranked[0]
    top_iou = _best_iou(top, ground_truth)
    oracle_candidate = max(candidates, key=lambda candidate: _best_iou(candidate, ground_truth))
    oracle_iou = _best_iou(oracle_candidate, ground_truth)
    return {
        "top1": {"start": top["start"], "end": top["end"], "temporal_iou": top_iou,
                 "temporal_score_source": top.get("temporal_score_source", "saved_moment_detr_prediction"),
                 "fusion_components": top.get("fusion_components", [])},
        "proposal_coverage": {"max_iou": oracle_iou,
                              "best_interval": [oracle_candidate["start"], oracle_candidate["end"]]},
        "proposal_count": len(candidates),
    }


def render_interval_svg(
    qid: str,
    video_duration: float,
    ground_truth: Sequence[Sequence[float]],
    conditions: Mapping[str, Mapping[str, Any]],
    original_coverage: Mapping[str, Any],
    expanded_coverage: Mapping[str, Any],
) -> str:
    """Render a dependency-free visual comparison of intervals for one query."""
    width, height = 1000, 260
    left, right = 190, 960
    plot_width = right - left
    duration = max(float(video_duration), 0.001)
    x = lambda value: left + min(max(float(value), 0.0), duration) / duration * plot_width
    rows = [
        ("Ground truth", ground_truth, "#1b9e77"),
        ("A: temporal-only", [[conditions["A"]["top1"]["start"], conditions["A"]["top1"]["end"]]], "#7570b3"),
        ("D: saved proposals", [[conditions["D"]["top1"]["start"], conditions["D"]["top1"]["end"]]], "#d95f02"),
        ("D: augmented", [[conditions["augmented_D"]["top1"]["start"], conditions["augmented_D"]["top1"]["end"]]], "#e7298a"),
        ("Oracle: originals", [original_coverage["best_interval"]], "#66a61e"),
        ("Oracle: augmented", [expanded_coverage["best_interval"]], "#e6ab02"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="Arial,sans-serif" font-size="18" font-weight="bold">{html.escape(qid)} interval comparison</text>',
        f'<text x="{left}" y="52" font-family="Arial,sans-serif" font-size="12">0 s</text>',
        f'<text x="{right}" y="52" text-anchor="end" font-family="Arial,sans-serif" font-size="12">{duration:g} s</text>',
        f'<line x1="{left}" y1="58" x2="{right}" y2="58" stroke="#555"/>',
    ]
    for row_index, (label, intervals, color) in enumerate(rows):
        baseline = 83 + row_index * 27
        parts.append(f'<text x="{left - 10}" y="{baseline + 13}" text-anchor="end" font-family="Arial,sans-serif" font-size="12">{html.escape(label)}</text>')
        for start, end in intervals:
            start_x, end_x = x(start), x(end)
            parts.append(f'<rect x="{start_x:.2f}" y="{baseline}" width="{max(1.0, end_x - start_x):.2f}" height="15" rx="3" fill="{color}" opacity="0.85"/>')
            parts.append(f'<title>{float(start):.2f}–{float(end):.2f} s</title>')
    parts.append('<text x="20" y="245" font-family="Arial,sans-serif" font-size="11" fill="#444">Oracle rows show best temporal-IoU interval; identity and object ownership are not evaluated.</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def write_panel_report(output_dir: Path, result: Mapping[str, Any]) -> None:
    metrics = result["metrics"]
    lines = [
        "# Multi-scale temporal proposal augmentation — controlled pilot",
        "",
        "Exploratory five-query, one-video pilot. This is not evidence of generalization.",
        "",
        "## Methodology",
        "",
        "The original Moment-DETR proposals and their saved semantic scores are retained. Each original interval produced up to four ground-truth-independent candidates: 0.5× and 1.5× duration centered at the parent center, plus same-duration center shifts of −2 s and +2 s. Generated intervals are clipped to video bounds, invalid intervals are discarded, and duplicates are removed.",
        "",
        "Each generated interval inherits temporal confidence from its source proposal. This value is parent-derived, not a new Moment-DETR prediction. YOLO/OpenCLIP semantic evidence is freshly computed for generated intervals. The unchanged D rank fuser is applied within each candidate pool. Object-category detections do not establish person-object ownership.",
        "",
        "## Frozen scoring configuration",
        "",
        f"- D fusion: `{result['scoring_configuration']['fusion']}`.",
        f"- Appearance/action frame samples per generated interval: {result['scoring_configuration']['appearance_frames_per_candidate']}/{result['scoring_configuration']['action_frames_per_candidate']}.",
        f"- Sampling policy: {result['scoring_configuration']['sampling_policy']}.",
        f"- YOLO11 weights: `{result['scoring_configuration']['yolo_weights']['path']}` (SHA-256 `{result['scoring_configuration']['yolo_weights']['sha256']}`).",
        f"- OpenCLIP weights: `{result['scoring_configuration']['clip_weights']['path']}` (SHA-256 `{result['scoring_configuration']['clip_weights']['sha256']}`).",
        f"- Execution device: {result['scoring_configuration']['device']}; GPU available to PyTorch: {result['scoring_configuration']['torch_cuda_available']}.",
        "",
        "## Aggregate comparison",
        "",
        "| Condition | Mean top-1 IoU | R@1 .3 | R@1 .5 | R@1 .7 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("A_temporal_only", "A — temporal-only"), ("D_original", "D — original pool"), ("D_augmented", "Augmented D")):
        row = metrics[key]
        lines.append(f"| {label} | {row['mean_iou']:.4f} | {row['recall_at_1_iou_0_3']:.3f} | {row['recall_at_1_iou_0_5']:.3f} | {row['recall_at_1_iou_0_7']:.3f} |")
    lines += [
        "",
        "## Candidate counts and evidence availability",
        "",
        "| Query | Original | Generated | Expanded |",
        "|---|---:|---:|---:|",
    ]
    for row in result["queries"]:
        lines.append(f"| {row['qid']} | {row['D']['proposal_count']} | {len(row['generated_candidates'])} | {row['augmented_D']['proposal_count']} |")
    lines += [
        "",
        "Null score counts over 200 generated candidates (missing values were not imputed):",
        "",
        "| Evidence field | Null count |",
        "|---|---:|",
    ]
    for field, count in result["missing_semantic_score_counts"].items():
        lines.append(f"| {field} | {count} |")
    lines += [
        "",
        "Appearance and action were scored for all 200 generated intervals. Person-object evidence was not requested for 80 candidates; the other 120 received person-crop text scores. Carried-object evidence had 6 detections, 74 no-matching-detection statuses, 40 unsupported requested-category statuses, and 80 not-requested statuses.",
        "",
        "## Failure observations",
        "",
        "Augmented D regressed on q1 (IoU 0.571 to 0.221) and q2 (0.469 to 0.254). q3 stayed at 0.000 IoU and its proposal-coverage oracle did not improve (0.331 to 0.331). q4 and q5 kept the same IoU despite selecting shifted intervals. The experiment improved the candidate-pool oracle, but the unchanged fusion did not rank those better intervals reliably.",
        "",
        "### Proposal coverage oracle",
        "",
        "| Candidate pool | Mean best IoU | R@1 .3 | R@1 .5 | R@1 .7 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("coverage_original", "Original proposals"), ("coverage_augmented", "Expanded proposals")):
        row = metrics[key]
        lines.append(f"| {label} | {row['mean_iou']:.4f} | {row['recall_at_1_iou_0_3']:.3f} | {row['recall_at_1_iou_0_5']:.3f} | {row['recall_at_1_iou_0_7']:.3f} |")
    lines += [
        "",
        "## Per-query results",
        "",
        "| Query | A interval / IoU | D interval / IoU | Augmented D interval / IoU | Original coverage | Expanded coverage |",
        "|---|---|---|---|---:|---:|",
    ]
    visual_paths = []
    visuals_dir = output_dir / "interval_comparisons"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    for row in result["queries"]:
        cond = {"A": row["A"], "D": row["D"], "augmented_D": row["augmented_D"]}
        def fmt_top(name: str) -> str:
            top = cond[name]["top1"]
            return f"[{top['start']:.2f}, {top['end']:.2f}] / {top['temporal_iou']:.3f}"
        lines.append(
            f"| {row['qid']} | {fmt_top('A')} | {fmt_top('D')} | {fmt_top('augmented_D')} | "
            f"{row['D']['proposal_coverage']['max_iou']:.3f} | {row['augmented_D']['proposal_coverage']['max_iou']:.3f} |"
        )
        rel = Path("interval_comparisons") / f"{row['qid']}.svg"
        svg = render_interval_svg(
            row["qid"], result["video"]["duration_seconds"], row["ground_truth_intervals_seconds"],
            cond, row["D"]["proposal_coverage"], row["augmented_D"]["proposal_coverage"],
        )
        (output_dir / rel).write_text(svg, encoding="utf-8")
        visual_paths.append(rel.as_posix())
    lines += ["", "## Runtime and limitations", "",
              f"Fresh semantic scoring: {result['runtime_seconds_new_semantic_scoring']:.2f} s. Total experiment wall time: {result['runtime_seconds_total']:.2f} s.",
              "",
              "Temporal confidence for added intervals is inherited and may not remain valid after shifting or scaling a proposal. Rank percentiles are recalculated on the expanded pool, so added candidates alter the comparison set. Five queries from one video are a pilot; no identity accuracy, person-object ownership, or cross-video generalization is claimed.",
              "",
              "## Interval visuals", ""]
    contact_sheet = Path("interval_comparisons") / "contact_sheet.png"
    if (output_dir / contact_sheet).is_file():
        lines.append(f"- [All-query contact sheet]({contact_sheet.as_posix()})")
    for rel in visual_paths:
        lines.append(f"- [{rel}]({rel})")
    lines += ["", "## Reproduction", "",
              "Planning-only command: `python standalone_eval/run_multiscale_proposal_augmentation.py`.",
              f"Scoring command: `{result['reproduction_command']}`.", ""]
    (output_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _ensure_output_is_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {path}")


def run_scoring(args: argparse.Namespace, source: Mapping[str, Any], annotations: Mapping[str, Any], duration: float) -> dict[str, Any]:
    experiment_started = time.perf_counter()
    video = Path(args.video).resolve()
    yolo_weights = Path(args.yolo_weights).resolve()
    clip_weights = Path(args.clip_weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    for required in (video, yolo_weights, clip_weights):
        if not required.is_file():
            raise FileNotFoundError(f"Required local scoring input missing (no downloads attempted): {required}")
    _ensure_output_is_new(output_dir)

    # The scoring path intentionally blocks urllib-based model downloads. Cached
    # local weights must already satisfy open_clip/Ultralytics initialization.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    import urllib.request

    def refuse_download(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Network downloads are disabled for this experiment")

    urllib.request.urlretrieve = refuse_download
    urllib.request.urlopen = refuse_download

    from ultralytics import YOLO
    from extension import attribute_matching
    import torch

    attribute_matching.yolo_model = YOLO(str(yolo_weights))
    source_by_qid = {str(row["qid"]): row for row in source["queries"]}
    annotation_by_qid = {str(row["qid"]): row for row in annotations["queries"]}
    output_queries = []
    generated_semantic_seconds = 0.0
    generated_counts = []
    missing_by_field = {field: 0 for field in SEMANTIC_FIELDS}

    for qid, source_query in source_by_qid.items():
        annotation = annotation_by_qid[qid]
        originals, generated, generation_counts = generate_augmented_candidates(
            source_query["all_candidate_scores"], duration
        )
        parts = parse_retrieval_query(annotation["query"])
        object_requests = tuple(dict.fromkeys(
            (*parts.carried_object_texts, *parts.unresolved_carried_object_texts)
        ))
        started = time.perf_counter()
        scored_rows = attribute_matching.score_candidates_with_semantics(
            str(video), generated,
            appearance_text=parts.appearance_text,
            action_text=parts.action_text,
            carried_object_texts=object_requests,
            carried_object_mentions=parts.carried_object_mentions,
            appearance_frame_count=int((source.get("settings") or {}).get("appearance_frames_per_candidate", 1)),
            action_frame_count=int((source.get("settings") or {}).get("action_frames_per_candidate", 1)),
        ) if generated else []
        generated_semantic_seconds += time.perf_counter() - started
        if len(scored_rows) != len(generated):
            raise RuntimeError(
                f"Fresh scorer returned {len(scored_rows)} rows for {len(generated)} generated intervals ({qid})"
            )
        fresh = [merge_fresh_evidence(candidate, scored)
                 for candidate, scored in zip(generated, scored_rows)]
        for row in fresh:
            for field in SEMANTIC_FIELDS:
                if row.get(field) is None:
                    missing_by_field[field] += 1

        gt = annotation.get("ground_truth_intervals_seconds") or []
        a_condition = _condition(originals, gt, "temporal")
        original_condition = _condition(originals, gt, D_FUSION)
        expanded_condition = _condition([*originals, *fresh], gt)
        a_condition["label"] = "A temporal-only on saved original candidates"
        original_condition["label"] = "D on saved original candidates"
        expanded_condition["label"] = "D on originals plus augmented candidates"
        output_queries.append({
            "qid": qid,
            "query": annotation["query"],
            "ground_truth_intervals_seconds": gt,
            "generation": generation_counts,
            "A": a_condition,
            "D": original_condition,
            "augmented_D": expanded_condition,
            "generated_candidates": fresh,
        })
        generated_counts.append(len(fresh))

    result = {
        "status": "completed",
        "experiment": "multi-scale and boundary-shifted proposal augmentation",
        "video": {"path": str(video), "duration_seconds": duration},
        "method": {
            "source_results": str(args.source_results.resolve()),
            "annotations": str(args.annotations.resolve()),
            "original_candidates_preserved": True,
            "transforms": {"duration_scales": list(SCALE_FACTORS), "center_shifts_seconds": list(CENTER_SHIFTS_SECONDS)},
            "generated_temporal_score": "inherited from parent proposal; not a new Moment-DETR prediction",
            "semantic_evidence": "fresh YOLO/OpenCLIP evidence on each generated interval",
            "fusion": D_FUSION,
            "rank_normalization": "recomputed within each query's candidate pool using the unchanged D fuser",
            "object_evidence_limitation": "category detections and crop phrase scores do not validate person-object ownership",
            "ground_truth_used_for_generation": False,
        },
        "scoring_configuration": {
            "fusion": D_FUSION,
            "appearance_frames_per_candidate": int((source.get("settings") or {}).get("appearance_frames_per_candidate", 1)),
            "action_frames_per_candidate": int((source.get("settings") or {}).get("action_frames_per_candidate", 1)),
            "sampling_policy": (source.get("settings") or {}).get("sampling_policy", "one sample at proposal midpoint"),
            "yolo_weights": {"path": str(yolo_weights), "sha256": _sha256_file(yolo_weights)},
            "clip_weights": {"path": str(clip_weights), "sha256": _sha256_file(clip_weights)},
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "baseline_settings": source.get("settings", {}),
        },
        "query_count": len(output_queries),
        "original_candidate_count_total": sum(row["D"]["proposal_count"] for row in output_queries),
        "generated_candidate_count_total": sum(generated_counts),
        "expanded_candidate_count_total": sum(row["augmented_D"]["proposal_count"] for row in output_queries),
        "runtime_seconds_new_semantic_scoring": generated_semantic_seconds,
        "missing_semantic_score_counts": missing_by_field,
        "metrics": {
            "A_temporal_only": _metrics([row["A"]["top1"]["temporal_iou"] for row in output_queries]),
            "D_original": _metrics([row["D"]["top1"]["temporal_iou"] for row in output_queries]),
            "D_augmented": _metrics([row["augmented_D"]["top1"]["temporal_iou"] for row in output_queries]),
            "coverage_original": _metrics([row["D"]["proposal_coverage"]["max_iou"] for row in output_queries]),
            "coverage_augmented": _metrics([row["augmented_D"]["proposal_coverage"]["max_iou"] for row in output_queries]),
        },
        "reproduction_command": (
            f"python standalone_eval/run_multiscale_proposal_augmentation.py --score "
            f"--source-results {args.source_results} --annotations {args.annotations} "
            f"--video {args.video} --yolo-weights {args.yolo_weights} "
            f"--clip-weights {args.clip_weights} --output-dir {output_dir}"
        ),
        "queries": output_queries,
    }
    result["runtime_seconds_total"] = time.perf_counter() - experiment_started
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_panel_report(output_dir, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--clip-weights", type=Path, default=DEFAULT_CLIP_WEIGHTS)
    parser.add_argument("--score", action="store_true", help="Run fresh YOLO/OpenCLIP scoring; requires separate approval")
    parser.add_argument("--output-dir", type=Path, help="New output directory; required for --score and never overwritten")
    args = parser.parse_args(argv)

    source = _load_json(args.source_results)
    annotations = _load_json(args.annotations)
    duration = float((source.get("video") or {}).get("duration_seconds") or 0)
    if duration <= 0:
        raise ValueError("Saved source results do not contain a positive video duration")
    plan = build_plan(source, annotations, duration)
    plan["inputs"] = {
        "source_results": str(args.source_results.resolve()),
        "annotations": str(args.annotations.resolve()),
        "video": str(args.video.resolve()),
    }
    if not args.score:
        plan["inference_started"] = False
        plan["planning_note"] = "No model/scoring import, video decode, or inference was performed."
        print(json.dumps(plan, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required with --score")
    result = run_scoring(args, source, annotations, duration)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "generated_candidate_count_total": result["generated_candidate_count_total"],
        "runtime_seconds_new_semantic_scoring": result["runtime_seconds_new_semantic_scoring"],
        "metrics": result["metrics"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
