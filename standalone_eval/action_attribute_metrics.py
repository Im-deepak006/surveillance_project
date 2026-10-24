"""Small-data metrics and component-rank ablation helpers."""

import math
import statistics
from typing import Mapping, Sequence

from extension.query_logic import ABLATION_COMPONENTS, fuse_candidate_ranks


def temporal_iou(predicted: Sequence[float], ground_truth: Sequence[float]) -> float:
    """Compute temporal intersection over union for [start, end] intervals."""
    p_start, p_end = map(float, predicted)
    g_start, g_end = map(float, ground_truth)
    if p_end <= p_start or g_end <= g_start:
        return 0.0
    intersection = max(0.0, min(p_end, g_end) - max(p_start, g_start))
    union = max(p_end, g_end) - min(p_start, g_start)
    return intersection / union if union > 0 else 0.0


def _best_iou(candidate: Mapping, ground_truth: Sequence[Sequence[float]]) -> float:
    if not ground_truth:
        return 0.0
    return max(temporal_iou((candidate["start"], candidate["end"]), interval) for interval in ground_truth)


def _metrics(rows):
    if not rows:
        return {"query_count": 0, "recall_at_1_iou_0_5": None, "recall_at_1_iou_0_7": None, "mean_iou_at_1": None}
    ious = [row["top1_iou"] for row in rows]
    return {
        "query_count": len(rows),
        "recall_at_1_iou_0_5": sum(iou >= 0.5 for iou in ious) / len(ious),
        "recall_at_1_iou_0_7": sum(iou >= 0.7 for iou in ious) / len(ious),
        "mean_iou_at_1": statistics.mean(ious),
    }


def score_distributions(candidates):
    distributions = {}
    fields = ("temporal_score", "appearance_score", "action_score", "carried_object_score")
    for field in fields:
        values = []
        for candidate in candidates:
            raw = candidate.get(field)
            if raw is None and field == "appearance_score":
                raw = candidate.get("attribute_score")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        distributions[field] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
        }
    return distributions


def evaluate_ablation(
    queries: Sequence[Mapping],
    ablation: str,
) -> dict:
    """Evaluate top-1 proposals against all GT windows for each query."""
    if ablation not in ABLATION_COMPONENTS:
        raise ValueError(f"Unknown ablation: {ablation}")
    details = []
    action_details = []
    appearance_details = []
    combined_details = []
    carried_object_details = []
    for query in queries:
        candidates = fuse_candidate_ranks(query["candidates"], ablation)
        ground_truth = query.get("ground_truth_intervals", [])
        ranked = []
        for rank, candidate in enumerate(candidates, start=1):
            ranked.append({
                "rank": rank,
                "start": candidate["start"],
                "end": candidate["end"],
                "temporal_score": candidate.get("temporal_score"),
                "appearance_score": candidate.get("appearance_score", candidate.get("attribute_score")),
                "action_score": candidate.get("action_score"),
                "carried_object_score": candidate.get("carried_object_score"),
                "temporal_rank_score": candidate.get("temporal_rank_score"),
                "appearance_rank_score": candidate.get("appearance_rank_score"),
                "action_rank_score": candidate.get("action_rank_score"),
                "carried_object_rank_score": candidate.get("carried_object_rank_score"),
                "fused_score": candidate.get("fused_score"),
                "fusion_components": candidate.get("fusion_components", []),
                "max_iou_to_ground_truth": _best_iou(candidate, ground_truth),
            })
        top1_iou = ranked[0]["max_iou_to_ground_truth"] if ranked else 0.0
        row = {
            "qid": query.get("qid"),
            "video": query.get("video"),
            "query": query.get("query"),
            "action_query": bool(query.get("action_text")),
            "appearance_query": bool(query.get("appearance_text")),
            "carried_object_query": bool(query.get("carried_object_texts")),
            "ground_truth_intervals": ground_truth,
            "top1_iou": top1_iou,
            "recall_at_1_iou_0_5": top1_iou >= 0.5,
            "recall_at_1_iou_0_7": top1_iou >= 0.7,
            "ranked_predictions": ranked,
        }
        details.append(row)
        if row["action_query"]:
            action_details.append(row)
        if row["appearance_query"]:
            appearance_details.append(row)
        if row["action_query"] and row["appearance_query"]:
            combined_details.append(row)
        if row["carried_object_query"]:
            carried_object_details.append(row)

    return {
        "components": list(ABLATION_COMPONENTS[ablation]),
        "metrics": _metrics(details),
        "action_query_metrics": _metrics(action_details),
        "appearance_query_metrics": _metrics(appearance_details),
        "combined_query_metrics": _metrics(combined_details),
        "carried_object_query_metrics": _metrics(carried_object_details),
        "queries": details,
    }
