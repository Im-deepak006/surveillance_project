"""Run a small, offline single-video retrieval ablation on local MOT17-04 footage.

Uses the project's actual Moment-DETR predictor, YOLO/OpenCLIP evidence scorer,
and rank fuser. It does not download weights or modify the ranking algorithm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import torch
try:
    import psutil
except ImportError:  # Optional on environments without psutil.
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.query_logic import fuse_candidate_ranks, parse_retrieval_query
from standalone_eval.action_attribute_metrics import temporal_iou


DEFAULT_ANNOTATIONS = ROOT / "evaluation_data" / "MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json"
DEFAULT_CHECKPOINT = ROOT / "run_on_video" / "moment_detr_ckpt" / "model_best.ckpt"
DEFAULT_CLIP_WEIGHTS = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"
DEFAULT_YOLO_WEIGHTS = ROOT / "yolo11n.pt"


def metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    readable = 0
    for second in range(int(frames / fps) + 1 if fps else 0):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        readable += int(cap.read()[0])
    cap.release()
    return {"fps": fps, "frame_count": frames, "width": width, "height": height,
            "duration_seconds": frames / fps if fps else None,
            "whole_second_sample_reads": readable}


def evaluate_rank(candidates, intervals, ablation):
    ranked = fuse_candidate_ranks(candidates, ablation)
    if not ranked:
        return None, 0.0
    top = ranked[0]
    iou = max((temporal_iou((top["start"], top["end"]), gt) for gt in intervals), default=0.0)
    return top, iou


def summarize_top1(rows, condition_key):
    eligible = [row for row in rows if row.get("conditions", {}).get(condition_key) is not None]
    values = [row["conditions"][condition_key]["temporal_iou"] for row in eligible]
    summary = {
        "sample_count": len(values),
        "eligible_qids": [row["qid"] for row in eligible],
        "mean_top1_iou": sum(values) / len(values) if values else None,
    }
    for threshold in (.3, .5, .7):
        summary[f"recall_at_1_iou_{str(threshold).replace('.', '_')}"] = (
            sum(value >= threshold for value in values) / len(values) if values else None
        )
    return summary


def proposal_oracle(candidates, intervals):
    ranked_temporal = sorted(candidates, key=lambda candidate: -float(candidate["temporal_score"]))
    values = [max((temporal_iou((c["start"], c["end"]), gt) for gt in intervals), default=0.0) for c in ranked_temporal]
    best_index = max(range(len(values)), key=lambda index: (values[index], -index)) if values else None
    best = ranked_temporal[best_index] if best_index is not None else None
    return {
        "proposal_count": len(candidates),
        "max_iou": values[best_index] if best_index is not None else None,
        "best_proposal_interval": [best["start"], best["end"]] if best else None,
        "best_proposal_temporal_rank_1_based": best_index + 1 if best_index is not None else None,
        "proposal_count_at_iou": {
            str(threshold): sum(value >= threshold for value in values)
            for threshold in (.3, .5, .7)
        },
        "any_proposal_reaches_iou_0_5": any(value >= .5 for value in values),
    }


def condition_rank_of_interval(candidates, interval, ablation):
    ranked = fuse_candidate_ranks(candidates, ablation)
    return next((index + 1 for index, candidate in enumerate(ranked)
                 if candidate["start"] == interval[0] and candidate["end"] == interval[1]), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-dir", required=True, help="Must not already exist")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--clip-weights", default=str(DEFAULT_CLIP_WEIGHTS))
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--qids", nargs="*", help="Optional subset of annotation qids; defaults to all")
    parser.add_argument("--appearance-frames", type=int, default=5)
    parser.add_argument("--action-frames", type=int, default=5)
    args = parser.parse_args()

    video = Path(args.video).resolve()
    annotations_path = Path(args.annotations).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    clip_weights = Path(args.clip_weights).resolve()
    yolo_weights = Path(args.yolo_weights).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    for required in (annotations_path, checkpoint, clip_weights, yolo_weights):
        if not required.is_file():
            raise FileNotFoundError(f"Required local input/weight missing (no download attempted): {required}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    queries = annotations["queries"]
    if args.qids is not None:
        wanted = set(args.qids)
        queries = [entry for entry in queries if entry["qid"] in wanted]
        missing = wanted - {entry["qid"] for entry in queries}
        if missing:
            raise ValueError(f"Unknown qid(s): {sorted(missing)}")
    if not queries:
        raise ValueError("No queries selected")
    if args.appearance_frames < 1 or args.action_frames < 1:
        raise ValueError("Frame sample counts must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "run.log"
    run_started = time.perf_counter()
    memory_stop = threading.Event()
    rss_samples = []
    if psutil is not None:
        process = psutil.Process()

        def sample_rss():
            while not memory_stop.wait(.2):
                try:
                    rss_samples.append(process.memory_info().rss)
                except Exception:
                    break

        memory_thread = threading.Thread(target=sample_rss, daemon=True)
        memory_thread.start()
    else:
        memory_thread = None

    def log(message):
        message = str(message)
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    video_meta = metadata(video)
    if not video_meta["whole_second_sample_reads"]:
        raise RuntimeError("No sampled video frames decoded")
    video_duration = video_meta["duration_seconds"]

    # Some installed dependencies expose a top-level ``utils`` module. This
    # repository uses a namespace-style ``utils/`` directory, so ensure its
    # local modules win when Moment-DETR imports ``utils.*``.
    loaded_utils = sys.modules.get("utils")
    local_utils = str(ROOT / "utils")
    if loaded_utils is not None and not hasattr(loaded_utils, "__path__"):
        for name in [name for name in sys.modules if name == "utils" or name.startswith("utils.")]:
            del sys.modules[name]
        loaded_utils = None
    if loaded_utils is None:
        import types
        loaded_utils = types.ModuleType("utils")
        loaded_utils.__path__ = [local_utils]
        sys.modules["utils"] = loaded_utils
    elif local_utils not in loaded_utils.__path__:
        loaded_utils.__path__.insert(0, local_utils)

    from run_on_video.run import MomentDETRPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Large default thread pools can make this small transformer unusually
        # slow during initialization/inference on desktop CPUs.
        torch.set_num_threads(1)
    log(f"MODEL_INIT_BEGIN device={device} checkpoint={checkpoint}")
    try:
        predictor = MomentDETRPredictor(
            ckpt_path=str(checkpoint),
            clip_model_name_or_path=str(clip_weights),
            device=device,
        )
    except BaseException as exc:
        (output_dir / "results.json").write_text(json.dumps({
            "status": "FAIL", "stage": "model_initialization",
            "exception_type": type(exc).__name__, "exception": str(exc),
        }, indent=2), encoding="utf-8")
        raise
    all_query_text = [entry["query"] for entry in queries]
    started = time.perf_counter()
    log(f"MOMENT_DETR_INFERENCE_BEGIN query_count={len(queries)}")
    predictions = predictor.localize_moment(str(video), all_query_text)
    inference_seconds = time.perf_counter() - started
    log(f"MOMENT_DETR_INFERENCE_COMPLETE seconds={inference_seconds:.3f}")
    # Moment-DETR and attribute_matching each own a CLIP model. Avoid holding
    # both large encoders in memory at once on this CPU-only pilot.
    del predictor
    import gc
    gc.collect()

    from extension import attribute_matching

    from ultralytics import YOLO
    attribute_matching.yolo_model = YOLO(str(yolo_weights))
    log(f"ATTRIBUTE_MODELS_READY yolo_weights={yolo_weights}")

    comparison_rows = []
    shared_candidate_check = True
    total_semantic_seconds = 0.0

    def persist_progress():
        (output_dir / "results.json").write_text(json.dumps({
            "status": "RUNNING",
            "video": {"path": str(video), "duration_seconds": video_duration},
            "settings": {"annotation_file": str(annotations_path),
                         "appearance_frames_per_candidate": args.appearance_frames,
                         "action_frames_per_candidate": args.action_frames,
                         "same_moment_detr_candidate_intervals_for_conditions": shared_candidate_check},
            "queries_completed": comparison_rows,
        }, indent=2), encoding="utf-8")

    for entry, prediction in zip(queries, predictions):
        parts = parse_retrieval_query(entry["query"])
        detector_object_requests = tuple(dict.fromkeys(
            (*parts.carried_object_texts, *parts.unresolved_carried_object_texts)
        ))
        candidates = []
        for start, end, score in prediction["pred_relevant_windows"]:
            start = max(0.0, min(float(start), video_duration))
            end = max(0.0, min(float(end), video_duration))
            if end > start:
                candidates.append({"start": start, "end": end, "temporal_score": float(score)})
        if not candidates:
            raise RuntimeError(f"Moment-DETR produced no valid intervals for {entry['qid']}")

        log(f"QUERY_SCORING_BEGIN qid={entry['qid']} candidates={len(candidates)}")
        sem_start = time.perf_counter()
        try:
            scored = attribute_matching.score_candidates_with_semantics(
                str(video), candidates,
                appearance_text=parts.appearance_text,
                action_text=parts.action_text,
                carried_object_texts=detector_object_requests,
                carried_object_mentions=parts.carried_object_mentions,
                appearance_frame_count=args.appearance_frames,
                action_frame_count=args.action_frames,
            )
        except Exception as exc:
            total_semantic_seconds += time.perf_counter() - sem_start
            comparison_rows.append({"qid": entry["qid"], "query": entry["query"],
                                    "status": "scoring_failed", "exception_type": type(exc).__name__,
                                    "exception": str(exc), "conditions": {"A_temporal_only": None,
                                    "B_temporal_plus_appearance": None,
                                    "C_temporal_appearance_plus_carried_object": None}})
            log(f"QUERY_SCORING_FAILED qid={entry['qid']} type={type(exc).__name__} error={exc}")
            persist_progress()
            continue
        total_semantic_seconds += time.perf_counter() - sem_start
        a_top, a_iou = evaluate_rank(scored, entry["ground_truth_intervals_seconds"], "temporal")
        b_top, b_iou = evaluate_rank(scored, entry["ground_truth_intervals_seconds"], "temporal_appearance")
        gt_object = str(entry.get("attributes", {}).get("carried_object", "")).lower()
        has_supported_object_gt = bool(parts.carried_object_texts) and gt_object not in {"", "unknown", "none"}
        if has_supported_object_gt:
            # C is eligible only when the query requests a supported detector
            # category and the annotation has a carried-object label.
            # Use the existing configured fusion method; action evidence is absent in C.
            c_ranked = fuse_candidate_ranks(scored, "temporal_appearance_carried_object")
            c_top = c_ranked[0]
            c_iou = max(temporal_iou((c_top["start"], c_top["end"]), gt)
                        for gt in entry["ground_truth_intervals_seconds"])
        else:
            c_top, c_iou = None, None
        d_top, d_iou = evaluate_rank(
            scored,
            entry["ground_truth_intervals_seconds"],
            "temporal_appearance_person_object_carried_object",
        )
        oracle = proposal_oracle(candidates, entry["ground_truth_intervals_seconds"])
        oracle_interval = oracle["best_proposal_interval"]

        def pack(top, iou):
            if top is None:
                return None
            return {"start": top["start"], "end": top["end"], "temporal_score": top["temporal_score"],
                    "appearance_score": top.get("appearance_score", top.get("attribute_score")),
                    "action_score": top.get("action_score"),
                    "carried_object_score": top.get("carried_object_score"),
                    "person_object_score": top.get("person_object_score"),
                    "fused_score": top.get("fused_score"),
                    "temporal_iou": iou, "fusion_components": top.get("fusion_components", [])}

        comparison_rows.append({
            "qid": entry["qid"], "query": entry["query"],
            "ground_truth_intervals_seconds": entry["ground_truth_intervals_seconds"],
            "target_person_id_within_video": entry["target_person_id_within_video"],
            "attributes": entry["attributes"],
            "query_decomposition": {"appearance_text": parts.appearance_text, "action_text": parts.action_text,
                                    "carried_object_texts": list(parts.carried_object_texts),
                                    "unresolved_carried_object_texts": list(parts.unresolved_carried_object_texts),
                                    "carried_object_mentions": list(parts.carried_object_mentions)},
            "carried_object_evaluation_status": (
                "eligible_supported_detector_category" if has_supported_object_gt
                else "unsupported_requested_category" if parts.unresolved_carried_object_texts
                else "not_requested"
            ),
            "candidate_count": len(candidates),
            "proposal_oracle": oracle,
            "conditions": {"A_temporal_only": pack(a_top, a_iou),
                           "B_temporal_plus_appearance": pack(b_top, b_iou),
                           "C_temporal_appearance_plus_carried_object": pack(c_top, c_iou),
                           "D_person_aware_evidence_fusion": pack(d_top, d_iou)},
            "oracle_proposal_rank_by_condition": {
                "A_temporal_only": condition_rank_of_interval(scored, oracle_interval, "temporal"),
                "B_temporal_plus_appearance": condition_rank_of_interval(scored, oracle_interval, "temporal_appearance"),
                "C_temporal_appearance_plus_carried_object": condition_rank_of_interval(
                    scored, oracle_interval, "temporal_appearance_carried_object") if has_supported_object_gt else None,
                "D_person_aware_evidence_fusion": condition_rank_of_interval(scored, oracle_interval,
                    "temporal_appearance_person_object_carried_object"),
            },
            "appearance_evidence_statuses": sorted({x["appearance_evidence_status"] for x in scored}),
            "object_evidence_statuses": sorted({x["carried_object_evidence_status"] for x in scored}),
            "all_candidate_scores": scored,
        })
        shared_candidate_check &= all(
            (x["start"], x["end"]) == (y["start"], y["end"])
            for x, y in zip(scored, candidates)
        ) and len(scored) == len(candidates)
        log(f"QUERY_SCORING_COMPLETE qid={entry['qid']} seconds={time.perf_counter() - sem_start:.3f} candidates={len(scored)}")
        persist_progress()

    successful_rows = [row for row in comparison_rows if "conditions" in row and row.get("candidate_count")]
    memory_stop.set()
    if memory_thread is not None:
        memory_thread.join(timeout=1)
    peak_rss_mb = max(rss_samples) / (1024 * 1024) if rss_samples else None
    output = {
        "status": "PASS" if len(successful_rows) == len(queries) else "PARTIAL PASS",
        "evaluation_completed": len(successful_rows) > 0,
        "queries_requested": len(queries),
        "queries_evaluated": len(successful_rows),
        "query_failures": [row for row in comparison_rows if row not in successful_rows],
        "video": {"path": str(video), "size_bytes": video.stat().st_size,
                  "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), **video_meta},
        "models": {"moment_detr_checkpoint": str(checkpoint), "moment_detr_checkpoint_size_bytes": checkpoint.stat().st_size,
                   "clip_weights": str(clip_weights), "clip_weights_size_bytes": clip_weights.stat().st_size,
                   "yolo_weights": str(yolo_weights), "yolo_weights_size_bytes": yolo_weights.stat().st_size,
                   "device": device, "downloads_disabled": True},
        "visual_evidence": {
            "person_class_id": 0,
            "supported_carried_object_classes": attribute_matching.COCO_CARRIED_OBJECT_IDS,
            "person_object_policy": "All distinct requested object phrases must score on one crop; the best crop in a sampled frame is used, without identity tracking or ownership verification.",
            "missing_object_box_policy": "An observed detector non-detection contributes no negative score; unavailable and unsupported evidence remain scoreless.",
        },
        "settings": {"annotation_file": str(annotations_path), "appearance_frames_per_candidate": args.appearance_frames,
                     "action_frames_per_candidate": args.action_frames,
                     "sampling_policy": "one sample at proposal midpoint; multiple samples evenly include both endpoints",
                     "same_moment_detr_candidate_intervals_for_conditions": shared_candidate_check,
                     "moment_detr_inference_seconds": inference_seconds,
                     "attribute_reranking_seconds": total_semantic_seconds,
                     "total_wall_seconds": time.perf_counter() - run_started,
                     "peak_process_rss_mb_sampled": peak_rss_mb,
                     "rss_sampling_interval_seconds": .2 if psutil is not None else None},
        "condition_definitions": {
            "A_temporal_only": ["temporal"],
            "B_temporal_plus_appearance": ["temporal", "appearance"],
            "C_temporal_appearance_plus_carried_object": ["temporal", "appearance", "carried_object"],
            "D_person_aware_evidence_fusion": ["temporal", "appearance", "person_object", "carried_object"],
            "note": "All conditions rerank the same Moment-DETR proposal intervals; unavailable component evidence is omitted from that candidate's mean rank score.",
        },
        "metrics": {"A_temporal_only": summarize_top1(comparison_rows, "A_temporal_only"),
                    "B_temporal_plus_appearance": summarize_top1(comparison_rows, "B_temporal_plus_appearance"),
                    "C_temporal_appearance_plus_carried_object": summarize_top1(comparison_rows, "C_temporal_appearance_plus_carried_object"),
                    "D_person_aware_evidence_fusion": summarize_top1(comparison_rows, "D_person_aware_evidence_fusion")},
        "proposal_oracle_metrics": {
            "sample_count": len(successful_rows),
            "eligible_qids": [row["qid"] for row in successful_rows],
            "recall_at_iou": {
                str(threshold): sum(row["proposal_oracle"]["max_iou"] >= threshold for row in successful_rows) / len(successful_rows)
                if successful_rows else None
                for threshold in (.3, .5, .7)
            },
            "mean_max_iou": sum(row["proposal_oracle"]["max_iou"] for row in successful_rows) / len(successful_rows)
            if successful_rows else None,
        },
        "queries": comparison_rows,
        "interpretation": "Temporal IoU measures localization only. Person/attribute matches require separate manual visual adjudication; no match is inferred from a detector category score.",
    }
    (output_dir / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    log("RUN_COMPLETE")
    print(json.dumps({"output": str(output_dir / "results.json"), "metrics": output["metrics"],
                      "runtime_seconds": inference_seconds + total_semantic_seconds,
                      "total_wall_seconds": output["settings"]["total_wall_seconds"],
                      "peak_process_rss_mb_sampled": peak_rss_mb,
                      "proposal_oracle_metrics": output["proposal_oracle_metrics"],
                      "same_candidates": shared_candidate_check, "device": device}, indent=2))


if __name__ == "__main__":
    main()
