"""Run all five temporal/action/appearance/object ablations on the example.

Invoke from the project root with the project virtual environment. The script
uses local model caches in offline mode and always creates a new output folder.
"""

import argparse
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", help="New output directory; must not already exist")
    parser.add_argument("--output-prefix", default="action_attribute_eval")
    args = parser.parse_args()

    project = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path
        if Path(entry or ".").resolve() != script_dir
    ]
    sys.path.insert(0, str(project))
    video_path = project / "run_on_video" / "example" / "RoripwjYFp8_60.0_210.0.mp4"
    annotation_path = project / "run_on_video" / "example" / "queries.jsonl"
    if not video_path.is_file() or not annotation_path.is_file():
        raise SystemExit("The bundled example video or annotation file is missing")

    run_root = Path(args.output_root) if args.output_root else (
        project / "outputs" / f"{args.output_prefix}_{uuid.uuid4().hex[:10]}"
    )
    run_root = run_root.resolve()
    if run_root.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {run_root}")
    run_root.mkdir(parents=True)

    os.environ["YOLO_CONFIG_DIR"] = str(run_root / "ultralytics_config")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logging.disable(logging.CRITICAL)
    # Importing app.py constructs its Streamlit UI at module scope. Mute those
    # bare-mode notices while keeping the real processing code and models.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import app
    from extension.query_logic import ABLATION_COMPONENTS, parse_retrieval_query
    from run_on_video.clip_extractor import get_video_duration
    from standalone_eval.action_attribute_metrics import evaluate_ablation, score_distributions

    status_log = []

    class Status:
        def write(self, *items, **kwargs):
            status_log.append({"type": "write", "text": " ".join(map(str, items))})

        def update(self, **kwargs):
            status_log.append({"type": "update", **kwargs})

    app.st.status = lambda *items, **kwargs: Status()
    app.OUTPUT_ROOT = str(run_root / "pipeline")

    annotation = json.loads(annotation_path.read_text(encoding="utf-8").splitlines()[0])
    query = annotation["query"]
    query_parts = parse_retrieval_query(query)

    class UploadedVideo:
        name = video_path.name

        def getbuffer(self):
            return memoryview(video_path.read_bytes())

    started = time.perf_counter()
    result = app.run_pipeline(
        UploadedVideo(),
        query,
        padding=0.0,
        ranking_mode="temporal_appearance_action_carried_object",
    )
    elapsed = time.perf_counter() - started
    if result is None:
        failure_path = run_root / "pipeline_failure.json"
        failure_path.write_text(json.dumps({"status": status_log}, indent=2), encoding="utf-8")
        raise SystemExit(f"App pipeline failed; stage log saved at {failure_path}")

    predictor, _ = app.load_predictor()
    repeat_started = time.perf_counter()
    repeated_prediction = predictor.localize_moment(
        video_path=result["video_path"], query_list=[query]
    )[0]["pred_relevant_windows"]
    repeat_seconds = time.perf_counter() - repeat_started
    first_proposals = result["temporal_candidates"]
    identical_proposals = len(first_proposals) == len(repeated_prediction) and all(
        abs(float(first["start"]) - float(second[0])) < 1e-6
        and abs(float(first["end"]) - float(second[1])) < 1e-6
        and abs(float(first["temporal_score"]) - float(second[2])) < 1e-6
        for first, second in zip(first_proposals, repeated_prediction)
    )

    query_record = {
        "qid": annotation.get("qid"),
        "video": video_path.name,
        "query": query,
        "action_text": query_parts.action_text,
        "appearance_text": query_parts.appearance_text,
        "carried_object_texts": query_parts.carried_object_texts,
        "ground_truth_intervals": annotation.get("relevant_windows", []),
        "candidates": result["scored_candidates"],
    }
    ablations = {
        name: evaluate_ablation([query_record], name)
        for name in ABLATION_COMPONENTS
    }
    clips = []
    for prediction in result["predictions"]:
        clip_path = prediction.get("clip_path")
        clip = {
            "rank": prediction["index"],
            "start": prediction["start"],
            "end": prediction["end"],
            "temporal_score": prediction["score"],
            "appearance_score": prediction.get("appearance_score"),
            "action_score": prediction.get("action_score"),
            "carried_object_score": prediction.get("carried_object_score"),
            "carried_object_detections": prediction.get("carried_object_detections", []),
            "carried_object_evidence_status": prediction.get("carried_object_evidence_status"),
            "path": clip_path,
            "exists_nonempty": bool(clip_path and Path(clip_path).is_file() and Path(clip_path).stat().st_size > 0),
        }
        if clip["exists_nonempty"]:
            clip["size_bytes"] = Path(clip_path).stat().st_size
            clip["duration_seconds"] = get_video_duration(clip_path)
        clips.append(clip)

    report = {
        "protocol": {
            "dataset": "Bundled annotated example only",
            "video_count": 1,
            "query_interval_annotation_count": len(query_record["ground_truth_intervals"]),
            "action_query_count": int(bool(query_parts.action_text)),
            "appearance_annotated_query_count": int(bool(query_parts.appearance_text)),
            "carried_object_query_count": int(bool(query_parts.carried_object_texts)),
            "carried_object_ground_truth_count": 0,
            "candidate_source": "One shared Moment-DETR prediction set per query",
            "fusion": "Within-query tied rank percentiles; unweighted mean of available components",
            "weights_tuned": False,
            "limitations": [
                "One query is descriptive only and cannot establish generalization or statistical improvement.",
                "The source annotation contains no appearance label; appearance ablations omit that component and match temporal-only when evidence is absent.",
                "Action similarity is static-frame OpenCLIP image-text similarity, not video motion recognition.",
                "The bundled query has no appearance or carried-object cue or labels; E has no object evidence to evaluate.",
                "Object-category detections are not evidence of a person-object carrying relation.",
            ],
        },
        "runtime": {
            "seconds": round(elapsed, 3),
            "stage_timings_seconds": result["stage_timings"],
            "repeat_inference_seconds": round(repeat_seconds, 3),
            "repeat_proposals_identical": identical_proposals,
            "model_in_eval_mode": result["model_eval_mode"],
            "device": result["device"],
            "actual_model_pipeline": True,
            "weights_downloaded": False,
            "query": query,
            "parsed_action": query_parts.action_text,
            "parsed_appearance": query_parts.appearance_text,
            "parsed_person": query_parts.person_text,
            "parsed_objects": list(query_parts.object_texts),
            "parsed_carried_objects": list(query_parts.carried_object_texts),
        },
        "score_distributions": score_distributions(query_record["candidates"]),
        "ablations": ablations,
        "failed_examples": [],
        "clip_validation": {
            "source_duration_seconds": result["video_duration"],
            "clip_count": len(clips),
            "nonempty_probeable_clip_count": sum(clip["exists_nonempty"] and clip.get("duration_seconds", 0) > 0 for clip in clips),
            "all_candidate_timestamps_in_source": all(
                0 <= prediction["start"] < prediction["end"] <= result["video_duration"]
                for prediction in result["predictions"] if prediction["valid"]
            ),
            "all_clip_durations_consistent_with_intervals": all(
                not clip.get("exists_nonempty") or abs(
                    clip.get("duration_seconds", 0) - (clip["end"] - clip["start"])
                ) <= max(0.75, (clip["end"] - clip["start"]) * 0.05)
                for clip in clips
            ),
            "clips": clips,
        },
        "output": {
            "pipeline_session_dir": result["session_dir"],
            "evaluation_results_file": str(run_root / "SOFT_BIOMETRIC_EVALUATION_RESULTS.json"),
        },
    }

    # The annotated sample has no carried-object query. Run a one-candidate
    # detector-path diagnostic without treating its result as labeled accuracy.
    detector_diagnostic = {"status": "not_run", "ground_truth": None}
    if result["temporal_candidates"]:
        diagnostic_started = time.perf_counter()
        try:
            from extension import attribute_matching

            diagnostic = attribute_matching.score_candidates_with_semantics(
                result["video_path"],
                [{
                    "start": result["predictions"][0]["start"],
                    "end": result["predictions"][0]["end"],
                    "temporal_score": result["predictions"][0]["temporal_score"],
                }],
                appearance_text="person wearing a blue shirt",
                carried_object_texts=("backpack",),
                appearance_frame_count=3,
            )[0]
            detector_diagnostic = {
                "query": "person carrying a backpack (diagnostic only; no annotation)",
                "target_categories": ["backpack"],
                "candidate_interval": [diagnostic["start"], diagnostic["end"]],
                "score": diagnostic["carried_object_score"],
                "detections": diagnostic["carried_object_detections"],
                "evidence_status": diagnostic["carried_object_evidence_status"],
                "appearance_query": "person wearing a blue shirt (diagnostic only; no annotation)",
                "appearance_score": diagnostic["appearance_score"],
                "appearance_evidence_status": diagnostic["appearance_evidence_status"],
                "openclip_cosine_computed": diagnostic["appearance_score"] is not None,
                "supported_yolo_class_names": {
                    class_id: attribute_matching.yolo_model.names.get(class_id)
                    for class_id in attribute_matching.COCO_CARRIED_OBJECT_IDS.values()
                },
                "runtime_seconds": round(time.perf_counter() - diagnostic_started, 3),
                "actual_yolo_model": True,
                "used_as_retrieval_metric": False,
                "ground_truth": None,
            }
        except Exception as exc:
            detector_diagnostic = {
                "query": "person carrying a backpack (diagnostic only; no annotation)",
                "target_categories": ["backpack"],
                "runtime_seconds": round(time.perf_counter() - diagnostic_started, 3),
                "actual_yolo_model": False,
                "error": f"{type(exc).__name__}: {exc}",
                "used_as_retrieval_metric": False,
                "ground_truth": None,
            }
    report["unannotated_object_detector_diagnostic"] = detector_diagnostic

    if not identical_proposals or not result["model_eval_mode"]:
        raise SystemExit("Inference reproducibility check failed; no evaluation results were written")
    out_path = run_root / "SOFT_BIOMETRIC_EVALUATION_RESULTS.json"
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing results file: {out_path}")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "results_file": str(out_path),
        "runtime_seconds": report["runtime"]["seconds"],
        "device": report["runtime"]["device"],
        "ablations": {name: metrics["metrics"] for name, metrics in ablations.items()},
        "clip_validation": {
            key: report["clip_validation"][key]
            for key in ("clip_count", "nonempty_probeable_clip_count", "all_candidate_timestamps_in_source")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
