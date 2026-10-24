"""Track-consistent OpenCLIP ranking feasibility experiment.

The isolated runner supports official Ultralytics ByteTrack and a separately
identified SciPy Hungarian IoU baseline. It does not modify the production
reranker.
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extension.query_logic import fuse_candidate_ranks, parse_retrieval_query
from standalone_eval.action_attribute_metrics import temporal_iou

DEFAULT_SOURCE = ROOT / "outputs/proposal_diagnosis_mot17_04_verified_20261023/results.json"
DEFAULT_CACHE = ROOT / "outputs/automated_visual_diagnostics_mot17_04_verified_20261023/results.json"
DEFAULT_ANNOTATIONS = ROOT / "evaluation_data/MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json"
DEFAULT_VIDEO = Path(r"D:\yolo_enhance\video\MOT17-04-SDP-raw.webm")
DEFAULT_YOLO = ROOT / "yolo11n.pt"


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _linear_assignment_max(scores):
    if not scores or not scores[0]:
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(1.0 - np.asarray(scores, dtype=float))
    return list(zip(rows.tolist(), cols.tolist()))


class IoUTrackAssociator:
    """One-video chronological IoU tracker; not ByteTrack or ReID."""

    def __init__(self, iou_threshold=0.3, max_age_frames=30):
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if not isinstance(max_age_frames, int) or max_age_frames < 0:
            raise ValueError("max_age_frames must be a nonnegative integer")
        self.iou_threshold = float(iou_threshold)
        self.max_age_frames = max_age_frames
        self.last_frame_index = -1
        self.next_track_id = 1
        self.active = {}
        self.tracks = {}
        self.matched_pairs = 0

    def update(self, frame_index, timestamp_seconds, detections):
        """Assign IDs for a frame, requiring strictly chronological input."""
        if frame_index <= self.last_frame_index:
            raise ValueError("Frames must be processed in strictly increasing order")
        self.last_frame_index = frame_index
        boxes = [d["box_xyxy"] for d in detections]
        active_ids = [tid for tid, state in self.active.items()
                      if frame_index - state["last_frame_index"] <= self.max_age_frames]
        scores = [[box_iou(self.active[tid]["box_xyxy"], box) for box in boxes]
                  for tid in active_ids]
        matches = {}
        used_tracks = set()
        for row, col in _linear_assignment_max(scores):
            if scores[row][col] >= self.iou_threshold:
                tid = active_ids[row]
                matches[col] = tid
                used_tracks.add(tid)
                self.matched_pairs += 1
        assigned = []
        for det_index, detection in enumerate(detections):
            tid = matches.get(det_index)
            if tid is None:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {"track_id": tid, "observations": []}
            observation = {
                "frame_index": int(frame_index),
                "timestamp_seconds": float(timestamp_seconds),
                "box_xyxy": [float(x) for x in detection["box_xyxy"]],
                "confidence": float(detection["confidence"]),
            }
            self.tracks[tid]["observations"].append(observation)
            self.active[tid] = {"box_xyxy": observation["box_xyxy"],
                                "last_frame_index": int(frame_index)}
            assigned.append({**detection, "track_id": tid})
        self.active = {tid: state for tid, state in self.active.items()
                       if frame_index - state["last_frame_index"] <= self.max_age_frames}
        return assigned


def match_cached_crops(cached_crops, detections, min_iou=0.5):
    """Match cached crop rows to full-video detections by bbox IoU, not order."""
    overlaps = [[box_iou(crop["box_xyxy"], detection["box_xyxy"])
                 for detection in detections] for crop in cached_crops]
    matches = {}
    used_detections = set()
    for crop_index, detection_index in _linear_assignment_max(overlaps):
        overlap = overlaps[crop_index][detection_index]
        if overlap >= min_iou:
            matches[crop_index] = {"detection_index": detection_index,
                                   "bbox_iou": float(overlap),
                                   "track_id": detections[detection_index]["track_id"]}
            used_detections.add(detection_index)
    return {"matches": matches,
            "unmatched_cached_crop_indices": [i for i in range(len(cached_crops)) if i not in matches],
            "unmatched_detection_indices": [i for i in range(len(detections)) if i not in used_detections]}


def tracked_detections_from_result(result):
    """Read detections and IDs from an Ultralytics tracking Results object."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    ids = getattr(boxes, "id", None)
    if ids is None:
        raise RuntimeError("YOLO tracking returned detections without track IDs")
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    track_ids = ids.cpu().numpy()
    if not (len(xyxy) == len(confidences) == len(track_ids)):
        raise RuntimeError("YOLO tracking box, confidence, and ID counts differ")
    return [{"box_xyxy": [float(x) for x in box],
             "confidence": float(confidence), "track_id": int(track_id)}
            for box, confidence, track_id in zip(xyxy, confidences, track_ids)]


def run_bytetrack_frame(model, frame):
    """Use Ultralytics' default ByteTrack YAML and persistent video state."""
    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          classes=[0], verbose=False, device="cpu")
    if not results:
        raise RuntimeError("YOLO.track returned no Results object")
    return tracked_detections_from_result(results[0])


def finite_score(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def aggregate_track_phrase_evidence(observations, min_observations=2):
    """Median each phrase per track over distinct frames; preserve missingness."""
    if not isinstance(min_observations, int) or min_observations < 1:
        raise ValueError("min_observations must be a positive integer")
    by_track = {}
    for observation in observations:
        tid = observation.get("track_id")
        if tid is None:
            continue
        row = by_track.setdefault(tid, {"appearance": {}, "action": {}})
        frame_index = observation["frame_index"]
        for phrase in ("appearance", "action"):
            score = finite_score(observation.get(f"{phrase}_score"))
            if score is not None:
                row[phrase][frame_index] = score
    result = {}
    for phrase in ("appearance", "action"):
        per_track = {}
        for tid, components in by_track.items():
            values = components[phrase]
            if len(values) >= min_observations:
                per_track[tid] = {"score": float(statistics.median(values.values())),
                                  "observation_count": len(values),
                                  "frame_indices": sorted(values)}
        best_tid = max(per_track, key=lambda tid: (per_track[tid]["score"], -int(tid))) if per_track else None
        result[phrase] = {
            "score": per_track[best_tid]["score"] if best_tid is not None else None,
            "best_track_id": best_tid,
            "status": "scored" if best_tid is not None else "unknown_insufficient_track_evidence",
            "eligible_track_count": len(per_track),
            "tracks": per_track,
        }
    return result


def build_track_candidate(original_candidate, observations, min_observations=2):
    """Keep track appearance/action separate and explicitly exclude ownership."""
    evidence = aggregate_track_phrase_evidence(observations, min_observations)
    row = dict(original_candidate)
    row["saved_single_frame_appearance_score"] = original_candidate.get("appearance_score")
    row["saved_proposal_action_score_not_track_specific"] = original_candidate.get("action_score")
    row["saved_person_object_score"] = original_candidate.get("person_object_score")
    row["saved_carried_object_score"] = original_candidate.get("carried_object_score")
    row["appearance_score"] = evidence["appearance"]["score"]
    row["action_score"] = evidence["action"]["score"]
    row["track_phrase_evidence"] = evidence
    row["person_object_score"] = None
    row["carried_object_score"] = None
    row["person_object_evidence_status"] = "excluded_unverified_ownership"
    row["carried_object_evidence_status"] = "excluded_unverified_ownership"
    return row


def rank_conditions(original_candidates, track_candidates):
    """Return baseline A/D and new T rankings; D input is never mutated."""
    a = fuse_candidate_ranks(original_candidates, "temporal")
    d = fuse_candidate_ranks(original_candidates, "temporal_appearance_person_object_carried_object")
    t = fuse_candidate_ranks(track_candidates, "temporal_appearance_action")
    return {"A": a, "D": d, "T": t}


def _score_actions_for_sampled_detections(video_path, fps, frame_count, frame_detections,
                                          qids_by_frame_detection, action_phrases, batch_size=32):
    """Batch OpenCLIP phrase scoring for matched person crops at cache timestamps."""
    if not qids_by_frame_detection:
        return {}, {"status": "no_matched_detections", "crop_count": 0, "runtime_seconds": 0.0}
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from extension import attribute_matching as am

    started = time.perf_counter()
    qids = list(action_phrases)
    valid_qids = [qid for qid in qids if isinstance(action_phrases[qid], str) and action_phrases[qid].strip()]
    if not valid_qids:
        return {}, {"status": "no_action_phrases", "crop_count": 0,
                    "runtime_seconds": time.perf_counter() - started}
    text_tokens = am.clip_tokenizer([action_phrases[qid] for qid in valid_qids]).to(am.device)
    with torch.no_grad():
        text_features = am.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    needed_by_frame = {}
    for frame_index, detection_index, qid in qids_by_frame_detection:
        if qid in valid_qids:
            needed_by_frame.setdefault(frame_index, {}).setdefault(detection_index, set()).add(qid)
    wall_started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video for track action phrase scoring: {video_path}")
    score_lookup = {}
    batch_images, batch_keys = [], []
    encoded_count = 0

    def flush():
        nonlocal encoded_count, batch_images, batch_keys
        if not batch_images:
            return
        image_batch = torch.stack(batch_images).to(am.device)
        with torch.no_grad():
            features = am.clip_model.encode_image(image_batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            similarities = (features @ text_features.T).cpu().numpy()
        for row_index, (frame_index, detection_index, track_id, requested_qids) in enumerate(batch_keys):
            for qid in requested_qids:
                phrase_index = valid_qids.index(qid)
                value = float(similarities[row_index, phrase_index])
                score_lookup[(qid, frame_index, track_id)] = value if math.isfinite(value) else None
        encoded_count += len(batch_images)
        batch_images, batch_keys = [], []

    current_frame_index = -1
    ok = True
    while ok and current_frame_index < max(needed_by_frame, default=-1):
        ok, frame = capture.read()
        if not ok:
            break
        current_frame_index += 1
        rows = needed_by_frame.get(current_frame_index)
        if not rows:
            continue
        for detection_index, requested_qids in rows.items():
            if detection_index >= len(frame_detections[current_frame_index]):
                continue
            detection = frame_detections[current_frame_index][detection_index]
            x1, y1, x2, y2 = [int(round(x)) for x in detection["box_xyxy"]]
            height, width = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            tensor = am.clip_preprocess(image)
            batch_images.append(tensor)
            batch_keys.append((current_frame_index, detection_index,
                               detection["track_id"], sorted(requested_qids)))
            if len(batch_images) >= batch_size:
                flush()
    capture.release()
    flush()
    return score_lookup, {"status": "scored", "crop_count": encoded_count,
                          "runtime_seconds": time.perf_counter() - started,
                          "text_phrase_count": len(valid_qids), "device": am.device}


def summarize(values):
    return {"query_count": len(values),
            "recall_at_1_iou_0_3": sum(v >= .3 for v in values) / len(values),
            "recall_at_1_iou_0_5": sum(v >= .5 for v in values) / len(values),
            "recall_at_1_iou_0_7": sum(v >= .7 for v in values) / len(values),
            "mean_iou": sum(values) / len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", default=str(DEFAULT_SOURCE))
    parser.add_argument("--cached-scores", default=str(DEFAULT_CACHE))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO))
    parser.add_argument("--output-dir", required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--tracker-backend", choices=("bytetrack", "hungarian_iou"),
                        default="hungarian_iou",
                        help="Select official Ultralytics ByteTrack or the prior custom IoU baseline")
    parser.add_argument("--association-iou", type=float, default=.30)
    parser.add_argument("--max-age-frames", type=int, default=30)
    parser.add_argument("--cache-match-iou", type=float, default=.50)
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.min_observations < 1 or args.clip_batch_size < 1:
        raise ValueError("min-observations and clip-batch-size must be positive")
    source_path, cache_path, ann_path, video_path, weights = [Path(x).resolve() for x in (
        args.source_results, args.cached_scores, args.annotations, args.video, args.yolo_weights)]
    clip_weights = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"
    for path in (source_path, cache_path, ann_path, video_path, weights, clip_weights):
        if not path.is_file():
            raise FileNotFoundError(f"Required local input/weight missing (no download attempted): {path}")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    source, cached, annotations = [json.loads(path.read_text(encoding="utf-8"))
                                   for path in (source_path, cache_path, ann_path)]
    source_by_qid = {q["qid"]: q for q in source["queries"]}
    cached_by_qid = {q["qid"]: q for q in cached["queries"]}
    gt_by_qid = {q["qid"]: q["ground_truth_intervals_seconds"] for q in annotations["queries"]}
    if set(source_by_qid) != set(cached_by_qid) or set(source_by_qid) != set(gt_by_qid):
        raise ValueError("Saved proposals, cached scores, and annotations must contain identical query IDs")

    # Parse only saved cache timestamps to associate later; no GT timestamps enter inference.
    query_parts = {qid: parse_retrieval_query(source_by_qid[qid]["query"]) for qid in source_by_qid}
    cached_candidate_by_qid_interval = {}
    qids_by_frame = {}
    cached_frames_by_qid_frame = {}
    for qid, cached_query in cached_by_qid.items():
        for candidate in cached_query["all_experimental_candidates"]:
            key_interval = (candidate["start"], candidate["end"])
            cached_candidate_by_qid_interval[(qid, key_interval)] = candidate
            for frame in candidate["appearance_crop_frame_audit"]:
                timestamp = float(frame["timestamp_seconds"])
                qids_by_frame.setdefault(timestamp, set()).add(qid)
                cached_frames_by_qid_frame.setdefault((qid, round(timestamp, 4)), []).append(frame)

    # Full-video, chronological inference is needed: the cached 147 irregular
    # samples are for score matching and are not enough to update a video tracker.
    import cv2
    from ultralytics import YOLO
    import scipy
    import ultralytics

    wall_started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or expected_frames <= 0:
        capture.release()
        raise RuntimeError("Video metadata did not provide valid FPS and frame count")
    model = YOLO(str(weights))
    tracker = (IoUTrackAssociator(args.association_iou, args.max_age_frames)
               if args.tracker_backend == "hungarian_iou" else None)
    frame_detections = []
    frame_rows = []
    track_records = {}
    seen_track_ids = set()
    matched_existing_track_detections = 0
    frames_with_track_ids = 0
    inference_started = time.perf_counter()
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / fps
        if args.tracker_backend == "bytetrack":
            assigned = run_bytetrack_frame(model, frame)
        else:
            prediction = model(frame, classes=[0], conf=.25, verbose=False, device="cpu")[0]
            detections = []
            boxes = prediction.boxes
            if boxes is not None:
                for box, confidence in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                    detections.append({"box_xyxy": [float(x) for x in box],
                                       "confidence": float(confidence)})
            assigned = tracker.update(frame_index, timestamp, detections)
        if assigned:
            frames_with_track_ids += 1
        for detection in assigned:
            tid = int(detection["track_id"])
            if tid in seen_track_ids:
                matched_existing_track_detections += 1
            seen_track_ids.add(tid)
            record = track_records.setdefault(tid, {"track_id": tid, "observations": []})
            record["observations"].append({"frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "box_xyxy": [float(x) for x in detection["box_xyxy"]],
                "confidence": float(detection["confidence"])})
        frame_detections.append(assigned)
        frame_rows.append({"frame_index": frame_index, "timestamp_seconds": timestamp,
                           "person_detection_count": len(assigned)})
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"YOLO full-video inference: {frame_index}/{expected_frames} frames", flush=True)
    capture.release()
    inference_runtime = time.perf_counter() - inference_started
    if frame_index != expected_frames:
        print(f"Video decoder returned {frame_index} frames; metadata reported {expected_frames}", flush=True)
    frames_by_index = {row["frame_index"]: row for row in frame_rows}

    # Match cached OpenCLIP crop scores to tracked detections by timestamp-derived
    # frame index and Hungarian bbox IoU; never rely on crop list ordering.
    match_lookup = {}
    cache_match_rows = []
    cache_crop_count = 0
    cache_crop_matched = 0
    bbox_ious = []
    qids_by_frame_detection = set()
    missing_cache_frames = 0
    for (qid, rounded_time), repeated_frames in cached_frames_by_qid_frame.items():
        timestamp = float(repeated_frames[0]["timestamp_seconds"])
        frame_idx = int(timestamp * fps)
        detections = frame_detections[frame_idx] if frame_idx < len(frame_detections) else []
        # Duplicates arise when overlapping proposals sample the same frame.
        cached_crops = repeated_frames[0].get("person_crops", [])
        if any(r.get("person_crops", []) != cached_crops for r in repeated_frames[1:]):
            raise AssertionError("Repeated qid/timestamp audits contain inconsistent cached crops")
        matching = match_cached_crops(cached_crops, detections, args.cache_match_iou)
        cache_crop_count += len(cached_crops)
        cache_crop_matched += len(matching["matches"])
        bbox_ious.extend(m["bbox_iou"] for m in matching["matches"].values())
        if not detections and cached_crops:
            missing_cache_frames += 1
        match_lookup[(qid, rounded_time)] = matching
        for crop_index, match in matching["matches"].items():
            det_index = match["detection_index"]
            qids_by_frame_detection.add((frame_idx, det_index, qid))
        cache_match_rows.append({"qid": qid, "timestamp_seconds": timestamp,
                                 "frame_index": frame_idx,
                                 "cached_person_crops": len(cached_crops),
                                 "matched_crops": len(matching["matches"]),
                                 "unmatched_cached_crop_indices": matching["unmatched_cached_crop_indices"]})

    action_phrases = {qid: query_parts[qid].action_text for qid in query_parts}
    action_scores, action_runtime = _score_actions_for_sampled_detections(
        video_path, fps, frame_index, frame_detections,
        sorted(qids_by_frame_detection), action_phrases, args.clip_batch_size)

    # Build same-proposal candidate rows and compute track-level phrase summaries.
    query_rows = []
    metric_ious = {name: [] for name in ("A", "D", "T")}
    baseline_checks = []
    proposal_missing_appearance = proposal_missing_action = 0
    proposal_count = 0
    for qid, old_query in source_by_qid.items():
        original_candidates = [dict(c) for c in old_query["all_candidate_scores"]]
        track_candidates, observation_audits = [], []
        for original in original_candidates:
            interval = (original["start"], original["end"])
            cached_candidate = cached_candidate_by_qid_interval[(qid, interval)]
            observations, frame_audit = [], []
            for sampled_frame in cached_candidate["appearance_crop_frame_audit"]:
                timestamp = float(sampled_frame["timestamp_seconds"])
                rounded_time = round(timestamp, 4)
                frame_idx = int(timestamp * fps)
                detections = frame_detections[frame_idx] if frame_idx < len(frame_detections) else []
                matching = match_lookup.get((qid, rounded_time), {"matches": {},
                    "unmatched_cached_crop_indices": list(range(len(sampled_frame.get("person_crops", []))))})
                crops = sampled_frame.get("person_crops", [])
                for crop_index, match in matching["matches"].items():
                    detection = detections[match["detection_index"]]
                    crop = crops[crop_index]
                    track_id = detection["track_id"]
                    observations.append({"track_id": track_id, "frame_index": frame_idx,
                        "timestamp_seconds": timestamp,
                        "appearance_score": crop.get("appearance_similarity"),
                        "action_score": action_scores.get((qid, frame_idx, track_id)),
                        "bbox_iou": match["bbox_iou"],
                        "detection_confidence": detection["confidence"]})
                    qids_by_frame_detection.add((frame_idx, match["detection_index"], qid))
                frame_audit.append({"timestamp_seconds": timestamp, "frame_index": frame_idx,
                    "status": sampled_frame.get("status"),
                    "cached_person_crops": len(crops),
                    "matched_person_crops": len(matching["matches"]),
                    "unmatched_cached_crop_indices": matching["unmatched_cached_crop_indices"]})
            track_candidate = build_track_candidate(original, observations, args.min_observations)
            track_candidate["track_observation_audit"] = observations
            track_candidate["sampled_frame_matching_audit"] = frame_audit
            track_candidates.append(track_candidate)
            observation_audits.append({"interval_seconds": [original["start"], original["end"]],
                "appearance_status": track_candidate["track_phrase_evidence"]["appearance"]["status"],
                "action_status": track_candidate["track_phrase_evidence"]["action"]["status"]})
            proposal_count += 1
            proposal_missing_appearance += track_candidate["appearance_score"] is None
            proposal_missing_action += track_candidate["action_score"] is None

        ranked = rank_conditions(original_candidates, track_candidates)
        saved_d = old_query["conditions"]["D_person_aware_evidence_fusion"]
        computed_d = ranked["D"][0]
        baseline_same = (computed_d["start"], computed_d["end"]) == (saved_d["start"], saved_d["end"])
        baseline_checks.append({"qid": qid, "saved_D_interval": [saved_d["start"], saved_d["end"]],
                                "recomputed_D_interval": [computed_d["start"], computed_d["end"]],
                                "same": baseline_same})
        gt = gt_by_qid[qid]
        conditions = {}
        for method in ("A", "D", "T"):
            top = ranked[method][0]
            value = max(temporal_iou((top["start"], top["end"]), interval) for interval in gt)
            metric_ious[method].append(value)
            conditions[method] = {"start": top["start"], "end": top["end"],
                "temporal_iou": value, "temporal_score": top.get("temporal_score"),
                "appearance_score": top.get("appearance_score"),
                "action_score": top.get("action_score"),
                "track_phrase_evidence": top.get("track_phrase_evidence"),
                "fusion_components": top.get("fusion_components", [])}
        query_rows.append({"qid": qid, "query": old_query["query"],
            "ground_truth_intervals_seconds": gt,
            "query_phrases": {"appearance": query_parts[qid].appearance_text,
                              "action": query_parts[qid].action_text},
            "conditions": conditions,
            "same_saved_proposals": len(original_candidates) == len(track_candidates),
            "all_candidate_scores": track_candidates})
    if not all(row["same"] for row in baseline_checks):
        raise AssertionError(f"Recomputed D did not reproduce saved top intervals: {baseline_checks}")

    track_lengths = [len(track["observations"]) for track in track_records.values()]
    durations = [track["observations"][-1]["timestamp_seconds"] -
                 track["observations"][0]["timestamp_seconds"] for track in track_records.values()]
    metrics = {name: summarize(values) for name, values in metric_ious.items()}
    total_runtime = time.perf_counter() - inference_started
    result = {
        "status": "completed", "experiment": "track-consistent OpenCLIP evidence ranking",
        "video": str(video_path), "query_count": len(query_rows),
        "proposal_count": proposal_count, "same_saved_proposals": True,
        "tracker": {"implementation": ("Ultralytics YOLO.track with bytetrack.yaml and persist=True"
                           if args.tracker_backend == "bytetrack" else
                           "custom SciPy Hungarian assignment over YOLO box IoU; not ByteTrack"),
            "backend": args.tracker_backend,
            "ultralytics_version": ultralytics.__version__, "scipy_version": scipy.__version__,
            "yolo_weights": str(weights), "tracker_yaml_available_but_unused": args.tracker_backend != "bytetrack",
            "tracker_yaml": (str(Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml")
                             if args.tracker_backend == "bytetrack" else None),
            "lap_version": (__import__("importlib.metadata", fromlist=["version"]).version("lap")
                            if args.tracker_backend == "bytetrack" else None),
            "tracker_api_blocker": None if args.tracker_backend == "bytetrack" else
                "ByteTrack API was previously unavailable because optional lap was missing; the explicit custom IoU backend is selected.",
            "association_iou_threshold": args.association_iou if tracker is not None else None,
            "max_age_frames": args.max_age_frames if tracker is not None else None,
            "bytetrack_default_low_confidence_and_buffer_config": ({"track_low_thresh": .1,
                "track_buffer": 30, "tracker_cfg": "bytetrack.yaml"}
                if args.tracker_backend == "bytetrack" else None),
            "person_detection_confidence_threshold": 0.1 if args.tracker_backend == "bytetrack" else 0.25,
            "chronological_frame_count": frame_index,
            "frames_with_track_ids": frames_with_track_ids,
            "video_fps": fps, "video_width": width, "video_height": height,
            "unique_track_count": len(track_records),
            "detections_using_previously_seen_track_id": (tracker.matched_pairs if tracker is not None
                else matched_existing_track_detections),
            "track_ids_returned_and_verified": bool(track_records),
            "median_track_observations": statistics.median(track_lengths) if track_lengths else None,
            "p90_track_observations": sorted(track_lengths)[min(len(track_lengths)-1, int(.9*len(track_lengths)))] if track_lengths else None,
            "median_track_duration_seconds": statistics.median(durations) if durations else None,
            "track_observations": [track for _, track in sorted(track_records.items())]},
        "settings": {"minimum_track_observations": args.min_observations,
            "cached_score_match": "timestamp via floor(timestamp*fps) plus Hungarian bbox-IoU match",
            "cached_score_min_bbox_iou": args.cache_match_iou,
            "track_phrase_aggregation": "median of finite per-observation OpenCLIP cosine similarities per track; proposal component takes the strongest eligible track",
            "action_score_source": "new OpenCLIP similarity on the matched person crop; appearance scores reused from cache",
            "score_normalization": "existing per-query tied-rank percentiles; T is unweighted mean of temporal, appearance, and action component ranks",
            "object_evidence": "excluded from T because category boxes do not establish ownership; no object proximity association",
            "identity_accuracy": "unmeasured; no ground-truth person tracks or identity labels"},
        "inference_and_runtime": {"full_video_yolo_frames": frame_index,
            "yolo_inference_seconds": inference_runtime,
            "track_action_clip_crop_count": action_runtime["crop_count"],
            "track_action_clip_runtime_seconds": action_runtime["runtime_seconds"],
            "track_action_clip_device": action_runtime.get("device"),
            "total_experiment_runtime_seconds": total_runtime,
            "wall_runtime_seconds_including_setup": time.perf_counter() - wall_started,
            "moment_detr_inference_repeated": False, "yolo_training": False,
            "moment_detr_training": False,
            "tracker_dependencies_installed_before_run": (["lap==0.5.13"]
                if args.tracker_backend == "bytetrack" else []),
            "dependencies_installed_during_run": []},
        "cache_matching": {"cached_person_crop_count": cache_crop_count,
            "matched_cached_person_crop_count": cache_crop_matched,
            "match_fraction": cache_crop_matched/cache_crop_count if cache_crop_count else None,
            "minimum_bbox_iou": args.cache_match_iou,
            "median_matched_bbox_iou": statistics.median(bbox_ious) if bbox_ious else None,
            "cached_frames_without_person_detections": missing_cache_frames,
            "appearance_or_action_evidence_missing_proposals": {
                "appearance": proposal_missing_appearance, "action": proposal_missing_action,
                "denominator": proposal_count},
            "per_query_frame_matches": cache_match_rows},
        "metrics_temporal_iou_only": metrics,
        "queries": query_rows,
        "baseline_reproduction_checks": baseline_checks,
        "limitations": ["ByteTrack IDs can still switch at crossings and fragment during occlusion; no annotated tracks exist to measure identity accuracy or ID switches.",
            "Identity accuracy and ID switches cannot be measured without annotated person tracks.",
            "OpenCLIP crop-text similarity is not motion recognition; action phrase scores use still person crops.",
            "The strongest appearance and action evidence can come from different tracks; both best track IDs are exposed.",
            "Object-category evidence is excluded because no validated person-object ownership association is available.",
            "The pilot remains five queries from one video and supports no generalization claim."],
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = ["# Track-consensus ablation", "",
        "## Method and feasibility", "",
        f"The existing cache contains irregular proposal samples, not consecutive video detections; full-video YOLO inference was therefore run on {frame_index} chronological frames. The installed Ultralytics version is {ultralytics.__version__}. Tracker backend: {args.tracker_backend}." + (" The supported `YOLO.track(..., persist=True, tracker='bytetrack.yaml')` API ran successfully and returned IDs." if args.tracker_backend == "bytetrack" else " The supported ByteTrack API requires `lap`; the selected backend is the explicitly labeled custom Hungarian IoU baseline, not ByteTrack."), "",
        f"The runner loads the existing YOLO11 checkpoint `{weights.name}` and processes one video chronologically with a persistent tracker. Cached appearance scores are matched to full-video detections by exact sampled frame index plus bbox IoU ≥ {args.cache_match_iou:.2f}; crop list order is not used. Track appearance and action phrase similarities are separately aggregated by median across distinct observations, requiring at least {args.min_observations} observations. The proposal component uses the strongest eligible track for each phrase.", "",
        "T normalizes temporal, appearance, and action components using the existing within-query tied-rank percentiles and takes their unweighted mean. Appearance similarities are reused from cache; action phrase similarities are newly scored by OpenCLIP on the matched person crop. Still-crop action similarity is not motion recognition. Object-category and person-object scores are excluded from T because the data does not validate ownership; saved D is reproduced unchanged.", "",
        "## Results", "", "These are temporal localization metrics only.", "",
        "| Method | Mean IoU | Recall@1 .3 | Recall@1 .5 | Recall@1 .7 |", "|---|---:|---:|---:|---:|"]
    for name in ("A", "D", "T"):
        m = metrics[name]
        report.append(f"| {name} | {m['mean_iou']:.4f} | {m['recall_at_1_iou_0_3']:.3f} | {m['recall_at_1_iou_0_5']:.3f} | {m['recall_at_1_iou_0_7']:.3f} |")
    report += ["", "| Query | A interval / IoU | D interval / IoU | T interval / IoU | T − D IoU |", "|---|---|---|---|---:|"]
    for query in query_rows:
        c = query["conditions"]
        report.append(f"| {query['qid']} | [{c['A']['start']:.2f}, {c['A']['end']:.2f}] / {c['A']['temporal_iou']:.3f} | [{c['D']['start']:.2f}, {c['D']['end']:.2f}] / {c['D']['temporal_iou']:.3f} | [{c['T']['start']:.2f}, {c['T']['end']:.2f}] / {c['T']['temporal_iou']:.3f} | {c['T']['temporal_iou']-c['D']['temporal_iou']:+.3f} |")
    matched_pct = 100 * result["cache_matching"]["match_fraction"] if result["cache_matching"]["match_fraction"] is not None else 0
    missing = result["cache_matching"]["appearance_or_action_evidence_missing_proposals"]
    report += ["", "## Tracks, evidence coverage, and runtime", "",
        f"- Full-video tracker generated {len(track_records)} IDs across {frame_index} frames and returned IDs on {frames_with_track_ids} frames; {tracker.matched_pairs if tracker is not None else matched_existing_track_detections} detections reused an existing track ID. Median track length was {statistics.median(track_lengths):.1f} observations ({statistics.median(durations):.2f} s); p90 length was {sorted(track_lengths)[min(len(track_lengths)-1, int(.9*len(track_lengths)))]} observations.",
        f"- Cached crop-to-detection matches: {cache_crop_matched}/{cache_crop_count} ({matched_pct:.1f}%), with median bbox IoU {statistics.median(bbox_ious):.3f}.",
        f"- Proposals lacking minimum repeated appearance evidence: {missing['appearance']}/{missing['denominator']}; lacking action evidence: {missing['action']}/{missing['denominator']}. Missing evidence remains `None` and is omitted from rank fusion.",
        f"- Runtime: YOLO {inference_runtime:.1f} s; OpenCLIP track action crop scoring {action_runtime['runtime_seconds']:.1f} s on {action_runtime.get('device')}; total {total_runtime:.1f} s. Moment-DETR was not rerun; no model trained. `lap==0.5.13` was installed before the run to enable the existing ByteTrack API; no dependency was installed during the run.", "",
        "## Interpretation and limitations", "",
        f"Recomputed A and D top intervals match the saved baselines: {all(row['same'] for row in baseline_checks)}. T {'improves' if metrics['T']['mean_iou'] > metrics['D']['mean_iou'] else 'does not improve'} on D by mean IoU ({metrics['T']['mean_iou']:.4f} vs {metrics['D']['mean_iou']:.4f}); D remains the best method in this pilot. Identity accuracy and ID switches are unmeasured because there are no annotated person tracks. ByteTrack IDs can switch at crossings or fragment through occlusions, and strongest appearance and action evidence can come from different tracks. Object ownership is unverified and excluded from T. This five-query, one-video pilot does not support a generalization claim.", "",
        "Recommended next priority: annotate a small set of person tracks and person-object ownership across multiple videos. This would test ID consistency and backpack ownership directly; the present result gives no evidence that switching trackers improves retrieval ranking.", "",
        "## Reproduction and tests", "",
        f"`venv\\Scripts\\python.exe standalone_eval\\run_tracking_consensus_ablation.py --source-results outputs\\proposal_diagnosis_mot17_04_verified_20261023\\results.json --cached-scores outputs\\automated_visual_diagnostics_mot17_04_verified_20261023\\results.json --annotations evaluation_data\\MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json --video D:\\yolo_enhance\\video\\MOT17-04-SDP-raw.webm --yolo-weights yolo11n.pt --output-dir outputs\\tracking_consensus_bytetrack_repro_20261023 --tracker-backend {args.tracker_backend}" + (f" --association-iou {args.association_iou} --max-age-frames {args.max_age_frames}" if args.tracker_backend == "hungarian_iou" else "") + f" --cache-match-iou {args.cache_match_iou} --min-observations {args.min_observations}`", "",
        "`venv\\Scripts\\python.exe -m unittest discover -s tests -p test_tracking_consensus_ablation.py -v`", "",
        "`venv\\Scripts\\python.exe -m unittest discover -s tests -v`", ""]
    (output / "EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "metrics": metrics,
        "runtime_seconds": total_runtime,
        "cache_match_fraction": result["cache_matching"]["match_fraction"],
        "tracks": len(track_records), "proposals_missing_appearance": proposal_missing_appearance,
        "proposals_missing_action": proposal_missing_action}, indent=2), flush=True)


if __name__ == "__main__":
    main()
