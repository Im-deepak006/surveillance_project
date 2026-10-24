"""Pure helpers for conservative cross-frame visual evidence aggregation."""

import math
from statistics import median


def aggregate_frame_appearance(frame_crop_scores, min_observed_frames=2):
    """Median of per-frame max crop similarity; require repeated frame evidence.

    Crop indices are intentionally not joined across frames. Missing crops/score
    frames are unavailable, not zero-valued evidence.
    """
    if not isinstance(min_observed_frames, int) or min_observed_frames < 1:
        raise ValueError("min_observed_frames must be a positive integer")
    frame_maxima = []
    for frame_scores in frame_crop_scores or ():
        values = []
        for raw in frame_scores or ():
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        frame_maxima.append(max(values) if values else None)
    observed = [value for value in frame_maxima if value is not None]
    if len(observed) < min_observed_frames:
        return {
            "score": None,
            "status": "unknown_insufficient_frame_evidence",
            "observed_frame_count": len(observed),
            "frame_maxima": frame_maxima,
        }
    return {
        "score": float(median(observed)),
        "status": "multi_frame_consensus",
        "observed_frame_count": len(observed),
        "frame_maxima": frame_maxima,
    }


def aggregate_repeated_object_evidence(
    per_frame_detections,
    requested_categories,
    supported_categories,
    min_positive_frames=2,
):
    """Score categories only when positive detections repeat across frames.

    Non-detections and one-frame positives remain unknown for ranking. No
    person-object association is inferred from the category detections.
    """
    if not isinstance(min_positive_frames, int) or min_positive_frames < 1:
        raise ValueError("min_positive_frames must be a positive integer")
    supported = set(supported_categories)
    requested = tuple(dict.fromkeys(str(x).strip().lower() for x in (requested_categories or ()) if str(x).strip()))
    if not requested:
        return {"score": None, "status": "not_requested", "states": {}, "categories": {}}
    states = {}
    category_evidence = {}
    for category in requested:
        if category not in supported:
            states[category] = "unsupported"
            continue
        per_frame_confidence = []
        for detections in per_frame_detections or ():
            valid = []
            for detection in detections or ():
                if str(detection.get("class_name", "")).lower() != category:
                    continue
                try:
                    confidence = float(detection.get("confidence"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if math.isfinite(confidence):
                    valid.append(confidence)
            per_frame_confidence.append(max(valid) if valid else None)
        positives = [value for value in per_frame_confidence if value is not None]
        if len(positives) >= min_positive_frames:
            states[category] = "supported_positive_repeated"
            category_evidence[category] = {
                "score": float(median(positives)),
                "positive_frame_count": len(positives),
                "per_frame_confidence": per_frame_confidence,
            }
        else:
            states[category] = "unknown_insufficient_repeat_positive"
    category_scores = [item["score"] for item in category_evidence.values()]
    return {
        "score": float(sum(category_scores) / len(category_scores)) if category_scores else None,
        "status": "repeated_positive_category_evidence" if category_scores else (
            "unsupported_requested_category" if requested and all(x not in supported for x in requested)
            else "unknown_no_repeated_positive"
        ),
        "states": states,
        "categories": category_evidence,
    }
