"""Run one isolated frame-evidence boundary refinement pilot (no reranking)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone_eval.boundary_refinement import consensus_boundary_shift, refine_interval
from standalone_eval.action_attribute_metrics import temporal_iou

BASELINE = ROOT / "outputs/proposal_diagnosis_mot17_04_verified_20261023/results.json"
DIAGNOSTICS = ROOT / "outputs/automated_visual_diagnostics_mot17_04_verified_20261023/results.json"
VIDEO = ROOT / "datasets/MOT17-04-SDP-raw.webm"
ANNOTATIONS = ROOT / "evaluation_data/MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json"
YOLO_WEIGHTS = ROOT / "yolo11n.pt"
CLIP_WEIGHTS = Path.home() / ".cache/clip/ViT-B-32.pt"


def dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metrics(values: list[float]) -> dict:
    return {
        "sample_count": len(values),
        "mean_iou": sum(values) / len(values) if values else None,
        **{f"recall_at_1_iou_{str(t).replace('.', '_')}":
           (sum(v >= t for v in values) / len(values) if values else None)
           for t in (0.3, 0.5, 0.7)},
    }


def matches_saved(values: list[float], saved: dict) -> bool:
    actual = metrics(values)
    pairs = (("mean_iou", "mean_top1_iou"),
             ("recall_at_1_iou_0_3", "recall_at_1_iou_0_3"),
             ("recall_at_1_iou_0_5", "recall_at_1_iou_0_5"),
             ("recall_at_1_iou_0_7", "recall_at_1_iou_0_7"))
    return all(abs(actual[new] - saved[old]) < 1e-12 for new, old in pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--baseline", default=str(BASELINE))
    parser.add_argument("--diagnostics", default=str(DIAGNOSTICS))
    parser.add_argument("--annotations", default=str(ANNOTATIONS))
    parser.add_argument("--yolo-weights", default=str(YOLO_WEIGHTS))
    parser.add_argument("--clip-weights", default=str(CLIP_WEIGHTS))
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    paths = {"video": Path(args.video), "baseline": Path(args.baseline),
             "diagnostics": Path(args.diagnostics), "annotations": Path(args.annotations),
             "yolo": Path(args.yolo_weights), "clip": Path(args.clip_weights)}
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {out}")
    for name, path in paths.items():
        if name != "annotations" and not path.is_file():
            raise FileNotFoundError(f"Required {name} input missing: {path}")
    if not paths["annotations"].is_file():
        raise FileNotFoundError(f"Evaluation annotation missing: {paths['annotations']}")
    out.mkdir(parents=True, exist_ok=False)
    log = out / "run.log"
    started = time.perf_counter()

    def say(message: str) -> None:
        line = f"[{time.perf_counter()-started:.2f}s] {message}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    # This stage reads only saved prediction/query fields. Ground-truth keys are
    # not referenced; annotation JSON is first opened after predictions.json exists.
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    diagnostic = json.loads(paths["diagnostics"].read_text(encoding="utf-8"))
    d_key = "D_person_aware_evidence_fusion"
    d_by_id = {q["qid"]: q for q in baseline["queries"]}
    diag_by_id = {q["qid"]: q for q in diagnostic["queries"]}
    if set(d_by_id) != set(diag_by_id) or len(d_by_id) != 5:
        raise ValueError("Saved baseline and diagnostic caches must contain the same five queries")
    metadata = {"fps": None, "frame_count": None, "duration_seconds": None}
    cap = cv2.VideoCapture(str(paths["video"]))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {paths['video']}")
    metadata = {"fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    metadata["duration_seconds"] = metadata["frame_count"] / metadata["fps"]
    if not math.isfinite(metadata["fps"]) or metadata["fps"] <= 0 or metadata["duration_seconds"] <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {metadata}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Reuse the exact project's YOLO crop and OpenCLIP scorers. Import initializes
    # its local cached CLIP weights; offline flags prohibit fetching replacements.
    from ultralytics import YOLO
    from extension import attribute_matching as scoring
    scoring.yolo_model = YOLO(str(paths["yolo"]))

    records = []
    frame_model_evaluations = {"decoded_probe_frames": 0, "yolo_frame_calls": 0,
                               "appearance_crop_clip_calls": 0, "action_frame_clip_calls": 0}
    for qid, q in d_by_id.items():
        d = q["conditions"][d_key]
        start, end = float(d["start"]), float(d["end"])
        if not 0 <= start < end <= metadata["duration_seconds"]:
            raise ValueError(f"Invalid saved D interval for {qid}: {[start, end]}")
        query = q.get("query_decomposition") or {}
        appearance_text, action_text = query.get("appearance_text"), query.get("action_text")
        if not appearance_text or not action_text:
            raise ValueError(f"Missing required appearance/action phrase for {qid}")
        probe_times = {"start_outside": start - .5, "start_inside": start + .5,
                       "end_inside": end - .5, "end_outside": end + .5}
        probes = {}
        shifts = {}
        statuses = {}
        for label, timestamp in probe_times.items():
            result = {"requested_timestamp_seconds": timestamp, "available": False,
                      "appearance_score": None, "appearance_status": "not_scored",
                      "action_score": None, "action_status": "not_scored", "person_crop_count": None}
            if not 0 <= timestamp < metadata["duration_seconds"]:
                result["appearance_status"] = result["action_status"] = "frame_outside_video"
                probes[label] = result
                continue
            frame_index = int(timestamp * metadata["fps"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                result["appearance_status"] = result["action_status"] = "frame_unavailable"
                probes[label] = result
                continue
            frame_model_evaluations["decoded_probe_frames"] += 1
            result.update({"available": True, "frame_index": frame_index,
                           "decoded_timestamp_seconds": frame_index / metadata["fps"]})
            crops = None
            try:
                crops = scoring.detect_person_crops(frame)
                frame_model_evaluations["yolo_frame_calls"] += 1
                result["person_crop_count"] = len(crops)
                if crops:
                    crop_scores = [scoring.score_attributes(crop, appearance_text) for crop in crops]
                    frame_model_evaluations["appearance_crop_clip_calls"] += len(crops)
                    finite = [float(v) for v in crop_scores if v is not None and math.isfinite(float(v))]
                    if finite:
                        result["appearance_score"] = max(finite)
                        result["appearance_status"] = "scored_max_over_person_crops"
                    else:
                        result["appearance_status"] = "no_valid_crop_similarity"
                else:
                    result["appearance_status"] = "no_person_detections"
            except Exception as exc:
                result["appearance_status"] = "scoring_error"
                result["appearance_error"] = f"{type(exc).__name__}: {exc}"
            try:
                action = scoring.score_action_frame(frame, action_text)
                frame_model_evaluations["action_frame_clip_calls"] += 1
                if action is not None and math.isfinite(float(action)):
                    result["action_score"] = float(action)
                    result["action_status"] = "scored_image_text_similarity_not_action_recognition"
                else:
                    result["action_status"] = "no_valid_similarity"
            except Exception as exc:
                result["action_status"] = "scoring_error"
                result["action_error"] = f"{type(exc).__name__}: {exc}"
            probes[label] = result

        # For each boundary both cues must rank the same side strictly higher.
        ss, start_status = consensus_boundary_shift(
            "start", probes["start_inside"]["appearance_score"], probes["start_outside"]["appearance_score"],
            probes["start_inside"]["action_score"], probes["start_outside"]["action_score"])
        es, end_status = consensus_boundary_shift(
            "end", probes["end_inside"]["appearance_score"], probes["end_outside"]["appearance_score"],
            probes["end_inside"]["action_score"], probes["end_outside"]["action_score"])
        new_start, new_end, interval_status = refine_interval(start, end, ss, es, metadata["duration_seconds"])
        records.append({"qid": qid, "query": q["query"],
                        "appearance_phrase": appearance_text, "action_phrase": action_text,
                        "original_interval_seconds": [start, end],
                        "original_method_d_fused_score": d["fused_score"],
                        "original_temporal_confidence": d["temporal_score"],
                        "refined_interval_seconds": [new_start, new_end],
                        "start_shift_seconds": new_start-start, "end_shift_seconds": new_end-end,
                        "start_decision": start_status, "end_decision": end_status,
                        "interval_status": interval_status, "probes": probes,
                        "identity_note": "Person crops are independently max-scored; identity continuity is unverified."})
        say(f"Scored {qid}: start={start_status}, end={end_status}")
    cap.release()
    runtime_before_eval = time.perf_counter() - started
    # Mandatory stage boundary: persist all refinement evidence/predictions before
    # opening annotations. If evaluation fails, the prediction artifact remains.
    predictions = {"status": "predictions_saved_before_ground_truth_evaluation",
                   "video": str(paths["video"]), "video_metadata": metadata,
                   "method": "strict appearance plus frame-image/action-text consensus; 0.5s shifts",
                   "cache_reuse": "none; required timestamps differ from saved appearance frame timestamps; action cache is not per-frame",
                   "frame_model_evaluations": frame_model_evaluations,
                   "scoring_runtime_seconds": runtime_before_eval, "queries": records}
    dump(out / "predictions.json", predictions)
    say("Saved predictions.json; opening ground truth now for evaluation")

    try:
        annotations = json.loads(paths["annotations"].read_text(encoding="utf-8"))
        gt_by_id = {q["qid"]: q["ground_truth_intervals_seconds"] for q in annotations["queries"]}
        if set(gt_by_id) != set(d_by_id):
            raise ValueError("Annotation qids differ from the frozen five-query baseline")
        rows = []
        for record in records:
            intervals = gt_by_id[record["qid"]]
            original_iou = max((temporal_iou(record["original_interval_seconds"], gt) for gt in intervals), default=0.0)
            refined_iou = max((temporal_iou(record["refined_interval_seconds"], gt) for gt in intervals), default=0.0)
            rows.append({**record, "ground_truth_intervals_seconds": intervals,
                         "original_temporal_iou": original_iou, "refined_temporal_iou": refined_iou})
        # Recompute A and D top-1 from the saved proposal rows after predictions exist.
        from extension.query_logic import fuse_candidate_ranks
        a_ious, d_ious = [], []
        for q in baseline["queries"]:
            gt = gt_by_id[q["qid"]]
            for ablation, bag in (("temporal", a_ious),
                                  ("temporal_appearance_person_object_carried_object", d_ious)):
                ranked = fuse_candidate_ranks(q["all_candidate_scores"], ablation)
                top = ranked[0]
                bag.append(max((temporal_iou((top["start"], top["end"]), g) for g in gt), default=0.0))
        stored_a = baseline["metrics"]["A_temporal_only"]
        stored_d = baseline["metrics"][d_key]
        baseline_check = {"A_recomputed": metrics(a_ious), "A_saved": stored_a,
                          "D_recomputed": metrics(d_ious), "D_saved": stored_d,
                          "A_exact_match": matches_saved(a_ious, stored_a),
                          "D_exact_match": matches_saved(d_ious, stored_d)}
        if not baseline_check["D_exact_match"]:
            raise RuntimeError("Could not reproduce saved D metrics exactly; results not interpreted")
        original_values = [r["original_temporal_iou"] for r in rows]
        refined_values = [r["refined_temporal_iou"] for r in rows]
        boundaries = [r[k] for r in records for k in ("start_decision", "end_decision")]
        evaluation = {"status": "PASS" if baseline_check["D_exact_match"] else "FAIL",
                      "baseline_reproduction": baseline_check,
                      "metrics": {"A": metrics(a_ious), "D": metrics(original_values),
                                 "refined_D": metrics(refined_values)},
                      "boundary_counts": {
                          "moved": sum(s.startswith("moved_") for s in boundaries),
                          "unchanged": sum(s == "unchanged_tie" for s in boundaries),
                          "rejected_missing_or_conflicting": sum(s in {"rejected_missing_evidence", "rejected_conflicting_evidence"} for s in boundaries),
                          "other_or_fallback": sum(not (s.startswith("moved_") or s == "unchanged_tie" or s.startswith("rejected_")) for s in boundaries)},
                      "runtime_seconds_total": time.perf_counter()-started,
                      "runtime_seconds_scoring_before_evaluation": runtime_before_eval,
                      "frame_model_evaluations": frame_model_evaluations,
                      "queries": rows,
                      "interpretation": "Five queries from one video are a feasibility pilot only. Refinement never reranks proposals or changes Method D fused scores."}
        dump(out / "results.json", evaluation)
        report(evaluation, out)
        say(f"Evaluation saved; total runtime {evaluation['runtime_seconds_total']:.2f}s")
    except Exception:
        (out / "evaluation_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        say("Evaluation stage failed; predictions preserved. See evaluation_error.txt")
        raise


def report(result: dict, out: Path) -> None:
    d, r, a = result["metrics"]["D"], result["metrics"]["refined_D"], result["metrics"]["A"]
    lines = ["# Query-aware boundary refinement pilot", "",
             "## Method", "",
             "Fresh scores were computed at exactly s−0.5, s+0.5, e−0.5, e+0.5 seconds. Appearance is the maximum OpenCLIP similarity over YOLO person crops in each single frame. The action phrase score is full-frame OpenCLIP image/text similarity, not action recognition. Both cues must strictly prefer the same side for a 0.5-second boundary move. Ties, missing scores/frames, or conflicts retain the original boundary. No object detections, ground truth, reranking, or score changes are used in refinement. Crops across frames do not establish identity.", "",
             "Ground truth was read only after `predictions.json` was written. Cached evidence was not reused because the required probe timestamps did not match the cache and cached action scores were not per-frame.", "",
             "## Results", "", "| Method | Mean IoU | R@1 .3 | R@1 .5 | R@1 .7 |", "|---|---:|---:|---:|---:|",
             f"| A | {a['mean_iou']:.4f} | {a['recall_at_1_iou_0_3']:.3f} | {a['recall_at_1_iou_0_5']:.3f} | {a['recall_at_1_iou_0_7']:.3f} |",
             f"| D (saved interval) | {d['mean_iou']:.4f} | {d['recall_at_1_iou_0_3']:.3f} | {d['recall_at_1_iou_0_5']:.3f} | {d['recall_at_1_iou_0_7']:.3f} |",
             f"| D + boundary refinement | {r['mean_iou']:.4f} | {r['recall_at_1_iou_0_3']:.3f} | {r['recall_at_1_iou_0_5']:.3f} | {r['recall_at_1_iou_0_7']:.3f} |", "",
             "| Query | Original interval | Refined interval | IoU D | IoU refined | Start decision | End decision | D score |",
             "|---|---:|---:|---:|---:|---|---|---:|"]
    for q in result["queries"]:
        lines.append(f"| {q['qid']} | {q['original_interval_seconds']} | {q['refined_interval_seconds']} | {q['original_temporal_iou']:.4f} | {q['refined_temporal_iou']:.4f} | {q['start_decision']} | {q['end_decision']} | {q['original_method_d_fused_score']:.6f} |")
    counts = result["boundary_counts"]
    lines += ["", f"Boundaries: {counts['moved']} moved, {counts['unchanged']} tied/unchanged, {counts['rejected_missing_or_conflicting']} rejected for missing/conflicting evidence.",
              f"Runtime: {result['runtime_seconds_total']:.2f}s total; scoring portion {result['runtime_seconds_scoring_before_evaluation']:.2f}s. Frame/model evaluations: `{json.dumps(result['frame_model_evaluations'])}`.", "",
              "## Limitations and decision", "", "Only five queries in one video were evaluated. Appearance uses a per-frame maximum across detections and does not track identity; a different person can dominate a probe. The action cue measures static image/text compatibility, not movement. Boundary candidates are limited to a fixed half-second and evidence can be noisy near occlusion, scene edges, and camera perspective changes. Identity accuracy is unmeasured.", "",
              ("**GO (feasibility only):** refinement improved the measured mean IoU or recall on this pilot; validate on separately annotated videos before considering broader use." if r["mean_iou"] > d["mean_iou"] or any(r[k] > d[k] for k in ("recall_at_1_iou_0_3", "recall_at_1_iou_0_5", "recall_at_1_iou_0_7")) else "**NO-GO:** the pilot did not improve mean IoU or any reported recall threshold; retain Method D as best. Prioritize additional annotated videos and identity-aware validation before another boundary variant."), "",
              "## Files", "", f"- `{out / 'predictions.json'}`", f"- `{out / 'results.json'}`", f"- `{out / 'run.log'}`"]
    (out / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
