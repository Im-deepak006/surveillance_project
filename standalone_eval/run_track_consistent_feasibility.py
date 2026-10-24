"""Offline IoU association and track-consistent appearance ranking baseline."""
import argparse
import json
import statistics
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.query_logic import fuse_candidate_ranks
from standalone_eval.action_attribute_metrics import temporal_iou

DEFAULT_SOURCE = ROOT / "outputs/proposal_diagnosis_mot17_04_verified_20261023/results.json"
DEFAULT_CACHE = ROOT / "outputs/automated_visual_diagnostics_mot17_04_verified_20261023/results.json"
DEFAULT_ANNOTATIONS = ROOT / "evaluation_data/MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json"


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def associate_detections(frames, iou_threshold=0.3, max_gap_seconds=0.5):
    """Greedy highest-IoU matching to active tracks; not ByteTrack/ReID.

    Every detection is assigned a local track ID. Tracks can reconnect only
    within max_gap_seconds; no appearance or object evidence is used for IDs.
    """
    next_id = 0
    active = {}
    tracks = {}
    assignments = {}
    match_count = 0
    for frame in frames:
        timestamp = float(frame["timestamp_seconds"])
        crops = frame.get("person_crops", [])
        boxes = [crop["box_xyxy"] for crop in crops]
        eligible_tracks = {tid: state for tid, state in active.items()
                           if 0 <= timestamp-state["timestamp"] <= max_gap_seconds}
        pairs = []
        for tid, state in eligible_tracks.items():
            for det_index, box in enumerate(boxes):
                score = box_iou(state["box"], box)
                if score >= iou_threshold:
                    pairs.append((score, tid, det_index))
        pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
        used_tracks, used_detections = set(), set()
        assigned = {}
        for score, tid, det_index in pairs:
            if tid in used_tracks or det_index in used_detections:
                continue
            assigned[det_index] = tid
            used_tracks.add(tid)
            used_detections.add(det_index)
            match_count += 1
        for det_index, box in enumerate(boxes):
            tid = assigned.get(det_index)
            if tid is None:
                tid = next_id
                next_id += 1
                tracks[tid] = {"first_timestamp": timestamp, "last_timestamp": timestamp,
                               "observations": 0, "observation_timestamps": []}
            active[tid] = {"box": box, "timestamp": timestamp}
            record = tracks[tid]
            record["last_timestamp"] = timestamp
            record["observations"] += 1
            record["observation_timestamps"].append(timestamp)
            assignments[(round(timestamp, 4), det_index)] = tid
    return assignments, tracks, {"matched_detections": match_count, "track_count": len(tracks)}


def aggregate_track_scores(candidate, assignments, min_observed_frames=2):
    """For each track, median its per-frame max crop similarity; choose best eligible track."""
    by_track = {}
    for frame in candidate.get("appearance_crop_frame_audit", []):
        timestamp = round(float(frame["timestamp_seconds"]), 4)
        for crop_index, crop in enumerate(frame.get("person_crops", [])):
            tid = assignments.get((timestamp, crop_index))
            if tid is None:
                continue
            try:
                score = float(crop.get("appearance_similarity"))
            except (TypeError, ValueError):
                continue
            if score != score or abs(score) == float("inf"):
                continue
            by_track.setdefault(tid, {}).setdefault(timestamp, []).append(score)
    eligible = {}
    for tid, frame_values in by_track.items():
        per_frame = [max(values) for values in frame_values.values() if values]
        if len(per_frame) >= min_observed_frames:
            eligible[tid] = float(statistics.median(per_frame))
    return {"score": max(eligible.values()) if eligible else None,
            "eligible_track_count": len(eligible), "scores_by_track": eligible}


def summarize(ious):
    return {"query_count": len(ious),
            "recall_at_1_iou_0_3": sum(x >= .3 for x in ious) / len(ious),
            "recall_at_1_iou_0_5": sum(x >= .5 for x in ious) / len(ious),
            "recall_at_1_iou_0_7": sum(x >= .7 for x in ious) / len(ious),
            "mean_iou": sum(ious) / len(ious)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", default=str(DEFAULT_SOURCE))
    parser.add_argument("--cached-detections", default=str(DEFAULT_CACHE))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=.3)
    parser.add_argument("--max-gap-seconds", type=float, default=.5)
    parser.add_argument("--min-track-frames", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    paths = [Path(x) for x in (args.source_results, args.cached_detections, args.annotations)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source, cached, annotation = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    cached_by_qid = {q["qid"]: q for q in cached["queries"]}
    gt_by_qid = {q["qid"]: q["ground_truth_intervals_seconds"] for q in annotation["queries"]}

    # Same video timestamps are reused across queries/proposals. Deduplicate and
    # verify coordinates before associating the common detection sequence.
    frames_by_time = {}
    duplicate_frame_conflicts = 0
    for query in cached["queries"]:
        for candidate in query["all_experimental_candidates"]:
            for frame in candidate["appearance_crop_frame_audit"]:
                key = round(float(frame["timestamp_seconds"]), 4)
                prior = frames_by_time.get(key)
                if prior is None:
                    frames_by_time[key] = frame
                elif [x.get("box_xyxy") for x in prior.get("person_crops", [])] != [x.get("box_xyxy") for x in frame.get("person_crops", [])]:
                    duplicate_frame_conflicts += 1
    frames = [frames_by_time[t] for t in sorted(frames_by_time)]
    assignments, tracks, association_stats = associate_detections(
        frames, args.iou_threshold, args.max_gap_seconds)

    methods = {"A_temporal_only": [], "D_saved_person_object_fusion": [], "track_consistent_appearance": []}
    per_query = []
    candidate_missing = 0
    candidate_total = 0
    for old in source["queries"]:
        qid = old["qid"]
        original_candidates = [dict(c) for c in old["all_candidate_scores"]]
        cache_candidates = {(c["start"], c["end"]): c for c in cached_by_qid[qid]["all_experimental_candidates"]}
        tracked_candidates = []
        for cand in original_candidates:
            cached_cand = cache_candidates[(cand["start"], cand["end"])]
            evidence = aggregate_track_scores(cached_cand, assignments, args.min_track_frames)
            candidate_total += 1
            candidate_missing += evidence["score"] is None
            tracked_candidates.append({**cand, "appearance_score": evidence["score"],
                                       "track_consistent_evidence": evidence})
        a_top = fuse_candidate_ranks(original_candidates, "temporal")[0]
        track_top = fuse_candidate_ranks(tracked_candidates, "temporal_appearance")[0]
        d_saved = old["conditions"]["D_person_aware_evidence_fusion"]
        d_cand = next(c for c in original_candidates if c["start"] == d_saved["start"] and c["end"] == d_saved["end"])
        gt = gt_by_qid[qid]
        def top_row(c):
            iou = max(temporal_iou((c["start"], c["end"]), interval) for interval in gt)
            return {"start": c["start"], "end": c["end"], "temporal_iou": iou}
        rows = {"A_temporal_only": top_row(a_top),
                "D_saved_person_object_fusion": top_row(d_cand),
                "track_consistent_appearance": top_row(track_top)}
        for name in methods:
            methods[name].append(rows[name]["temporal_iou"])
        per_query.append({"qid": qid, "ground_truth_intervals_seconds": gt, "methods": rows,
                          "track_evidence_at_track_ranked_top1": track_top.get("track_consistent_evidence")})
    runtime = time.perf_counter() - started
    track_obs = [t["observations"] for t in tracks.values()]
    track_durations = [t["last_timestamp"]-t["first_timestamp"] for t in tracks.values()]
    result = {
        "status": "completed", "experiment": "greedy IoU association with track-consistent appearance evidence",
        "query_count": len(per_query), "proposals_per_query": 10, "same_saved_proposals": True,
        "settings": {"association": "greedy highest-IoU one-to-one matching between active tracks and frame detections",
                     "iou_threshold": args.iou_threshold, "max_gap_seconds": args.max_gap_seconds,
                     "minimum_observed_frames_per_track_for_evidence": args.min_track_frames,
                     "appearance": "median of per-frame maximum crop similarities for each associated track; candidate uses highest eligible track score",
                     "object_category_evidence_used_in_new_ranker": False,
                     "identity_accuracy": "unmeasured; no ground-truth tracks or per-frame identity labels",
                     "rank_fusion": "existing unweighted mean of available within-query temporal and appearance rank percentiles"},
        "frame_and_association_audit": {"unique_timestamps": len(frames),
            "timestamp_span_seconds": [frames[0]["timestamp_seconds"], frames[-1]["timestamp_seconds"]],
            "unique_person_detections": sum(len(f.get("person_crops", [])) for f in frames),
            "duplicate_timestamp_detection_conflicts": duplicate_frame_conflicts,
            "matched_detections_across_adjacent_samples": association_stats["matched_detections"],
            "track_count": association_stats["track_count"],
            "single_observation_tracks": sum(n == 1 for n in track_obs),
            "single_observation_track_fraction": sum(n == 1 for n in track_obs)/len(track_obs) if track_obs else None,
            "median_track_observations": statistics.median(track_obs) if track_obs else None,
            "median_track_duration_seconds": statistics.median(track_durations) if track_durations else None,
            "candidate_proposals_without_eligible_track_evidence": candidate_missing,
            "candidate_proposal_count": candidate_total},
        "metrics_temporal_iou_only": {name: summarize(vals) for name, vals in methods.items()},
        "per_query": per_query, "runtime_seconds": runtime,
        "dependencies_added": [], "model_training_or_inference": "none; cached person boxes and similarities reused",
        "limitations": ["IoU IDs are associations, not verified identities.",
            "Large occlusion, crossings, scale changes, or gaps can fragment tracks or switch IDs.",
            "Track-consistent appearance ranking evaluates temporal IoU only; identity and clothing correctness are unmeasured.",
            "Object-category scores are excluded; proximity does not establish ownership."],
    }
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = result["metrics_temporal_iou_only"]
    report = ["# Track-consistent evidence ranking feasibility experiment", "",
        "## Feasibility and method", "",
        f"The saved visual diagnostics contained {len(frames)} unique timestamped frames and {result['frame_and_association_audit']['unique_person_detections']} person detections. Detection boxes and per-crop appearance similarities were cached; no new model inference or training was required. The environment has Ultralytics installed (with tracker support), while BoxMOT, Supervision, Norfair, and FilterPy are absent. This experiment uses a simple documented greedy IoU baseline, not ByteTrack.", "",
        f"Person detections were associated by highest-IoU one-to-one greedy matching (IoU ≥ {args.iou_threshold:.2f}) between active tracks, allowing a maximum {args.max_gap_seconds:.2f} second gap. Appearance was aggregated per track as the median of its per-frame maximum crop similarities; a track needed at least {args.min_track_frames} scored frames. Candidate appearance evidence is the best eligible track score. Object-category evidence was kept out of the new ranker.", "",
        "## Results", "", "Metrics measure temporal localization only. Identity accuracy is unmeasured because the data has no ground-truth tracks.", "",
        "| Method | Recall@1 .3 | Recall@1 .5 | Recall@1 .7 | Mean IoU |", "|---|---:|---:|---:|---:|"]
    for key, m in metrics.items():
        report.append(f"| {key} | {m['recall_at_1_iou_0_3']:.3f} | {m['recall_at_1_iou_0_5']:.3f} | {m['recall_at_1_iou_0_7']:.3f} | {m['mean_iou']:.4f} |")
    report += ["", "| Query | A IoU | D IoU | Track-consistent IoU | Change vs D |", "|---|---:|---:|---:|---:|"]
    for q in per_query:
        m=q["methods"]; diff=m["track_consistent_appearance"]["temporal_iou"]-m["D_saved_person_object_fusion"]["temporal_iou"]
        report.append(f"| {q['qid']} | {m['A_temporal_only']['temporal_iou']:.4f} | {m['D_saved_person_object_fusion']['temporal_iou']:.4f} | {m['track_consistent_appearance']['temporal_iou']:.4f} | {diff:+.4f} |")
    fa=result["frame_and_association_audit"]
    report += ["", "## Association diagnostics", "",
        f"- Greedy matches across adjacent sampled frames: {fa['matched_detections_across_adjacent_samples']}; generated track IDs: {fa['track_count']}.",
        f"- Single-observation tracks: {fa['single_observation_tracks']} ({fa['single_observation_track_fraction']:.1%}); median track length {fa['median_track_observations']:.1f} observations / {fa['median_track_duration_seconds']:.2f} seconds.",
        f"- Proposals without any track with enough appearance observations: {fa['candidate_proposals_without_eligible_track_evidence']}/{fa['candidate_proposal_count']}. These statistics describe association continuity, not tracking accuracy or true fragmentation.",
        f"- Runtime: {runtime:.2f} seconds. Added dependencies: none.", "",
        "## Interpretation and next step", "",
        "The IoU baseline provides enough cached detections to run the comparison, but IDs may switch at crossings and split at occlusions or sparse gaps. No person identity, attribute correctness, or object ownership result is supported. If this ranking does not beat D, keep D as the pilot best and prioritize collecting multiple annotated videos with person tracks and object ownership labels; those labels are needed to assess association before integrating tracking.", "",
        "## Reproduction", "",
        f"`venv\\Scripts\\python.exe standalone_eval\\run_track_consistent_feasibility.py --source-results outputs\\proposal_diagnosis_mot17_04_verified_20261023\\results.json --cached-detections outputs\\automated_visual_diagnostics_mot17_04_verified_20261023\\results.json --annotations evaluation_data\\MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json --output-dir {output} --iou-threshold {args.iou_threshold} --max-gap-seconds {args.max_gap_seconds} --min-track-frames {args.min_track_frames}`", ""]
    (output / "EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output.resolve()), "runtime_seconds": runtime,
                      "metrics": metrics, "association_diagnostics": fa}, indent=2))

if __name__ == "__main__":
    main()
