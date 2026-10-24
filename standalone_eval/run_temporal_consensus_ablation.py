"""Test median frame-consensus appearance and repeated category evidence.

Moment-DETR proposals and cached CLIP crop scores are reused from the prior
verified pilot. Only YOLO category detections are recomputed on the same three
sample timestamps. No model training or rank-weight tuning is performed.
"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.temporal_evidence import aggregate_frame_appearance, aggregate_repeated_object_evidence
from extension.query_logic import fuse_candidate_ranks, SUPPORTED_CARRIED_OBJECT_CATEGORIES
from standalone_eval.action_attribute_metrics import temporal_iou

DEFAULT_SOURCE = ROOT / "outputs" / "proposal_diagnosis_mot17_04_verified_20261023" / "results.json"
DEFAULT_CACHE = ROOT / "outputs" / "automated_visual_diagnostics_mot17_04_verified_20261023" / "results.json"
DEFAULT_ANNOTATIONS = ROOT / "evaluation_data" / "MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json"
DEFAULT_VIDEO = Path(r"D:\yolo_enhance\video\MOT17-04-SDP-raw.webm")
DEFAULT_OUTPUT = ROOT / "outputs" / "attribute_aware_consensus_20261023"


def summarize(rows, condition):
    eligible = [row for row in rows if condition in row["conditions"]]
    vals = [row["conditions"][condition]["temporal_iou"] for row in eligible]
    return {
        "sample_count": len(vals), "denominator": len(eligible),
        "eligible_qids": [row["qid"] for row in eligible],
        "recall_at_1_iou_0_3": sum(x >= .3 for x in vals) / len(vals) if vals else None,
        "recall_at_1_iou_0_5": sum(x >= .5 for x in vals) / len(vals) if vals else None,
        "recall_at_1_iou_0_7": sum(x >= .7 for x in vals) / len(vals) if vals else None,
        "mean_top1_iou": sum(vals) / len(vals) if vals else None,
    }


def build_html(result):
    lines = ["<!doctype html><meta charset='utf-8'><title>Temporal-consensus ablation</title>",
             "<style>body{font:15px system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #aaa;padding:.4rem}th{background:#eee}.note{background:#fff2d4;padding:.8rem}</style>",
             "<h1>Attribute-aware temporal-consensus ablation</h1>",
             "<p class='note'>This is an exploratory five-query, one-video reranking comparison. No person identity or object ownership accuracy is measured. Missing detections are unknown evidence.</p>",
             "<h2>Temporal metrics</h2><table><tr><th>Condition</th><th>n / denominator</th><th>R@.3</th><th>R@.5</th><th>R@.7</th><th>Mean IoU</th></tr>"]
    for key, metric in result["metrics"].items():
        lines.append(f"<tr><td>{html.escape(key)}</td><td>{metric['sample_count']}/{metric['denominator']}</td><td>{metric['recall_at_1_iou_0_3']:.3f}</td><td>{metric['recall_at_1_iou_0_5']:.3f}</td><td>{metric['recall_at_1_iou_0_7']:.3f}</td><td>{metric['mean_top1_iou']:.4f}</td></tr>")
    lines.append("</table><h2>Per-query top results</h2><table><tr><th>Query</th><th>GT</th>" +
                 "".join(f"<th>{html.escape(key)}</th>" for key in result["condition_order"]) + "</tr>")
    for row in result["queries"]:
        gt = ", ".join(f"{a:.2f}–{b:.2f}" for a,b in row["ground_truth_intervals_seconds"])
        lines.append(f"<tr><td>{html.escape(row['qid'])}</td><td>{gt}</td>" +
                     "".join((f"<td>{v['start']:.2f}–{v['end']:.2f}<br>IoU {v['temporal_iou']:.3f}</td>" if (v:=row['conditions'].get(key)) else "<td>—</td>") for key in result["condition_order"]) + "</tr>")
    lines.append("</table><h2>Evidence diagnostics</h2><p>Appearance is the median of per-frame maximum crop similarities. Person crop indices are not linked across frames. Object category evidence requires positive detections in at least two sampled frames. This is category persistence only, not person-object association.</p><p><a href='visual_diagnostics/index.html'>Open crop-level visual diagnostics</a> · <a href='results.json'>Machine-readable results</a> · <a href='per_query.csv'>CSV</a></p>")
    return "\n".join(lines)


def build_report(result):
    lines = [
        "# Attribute-aware temporal-consensus experiment",
        "",
        "## Method",
        "",
        "The experiment reuses the same 10 saved Moment-DETR proposals for each of the same five queries. It reuses cached three-frame OpenCLIP person-crop similarities. YOLO11 category detections are run on those same three timestamp samples.",
        "",
        "Appearance per candidate is the median of per-frame maxima across available person crops. A candidate needs valid crop scores in at least two frames. Crop indices are not associated across frames. For each requested supported object category, category evidence is used only when YOLO returns a positive box in at least two of three sampled frames; zero or one positive frame stays unknown and contributes no score. Unsupported categories remain unsupported. This does not establish which person carries an object.",
        "",
        "Conditions: A is temporal baseline; B is saved single-frame appearance; D is the saved person-object fusion; E is temporal plus median multi-frame appearance; F adds repeated positive supported-object category evidence to E; G uses saved single-frame appearance plus repeated object evidence. Rank fusion remains the existing unweighted mean of normalized component ranks. No weights were tuned.",
        "",
        "## Metrics",
        "",
        "All table rows use identical five-query denominators. Metrics evaluate temporal IoU only.",
        "",
        "| Condition | n/denominator | R@.3 | R@.5 | R@.7 | Mean IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, m in result["metrics"].items():
        lines.append(f"| {key} | {m['sample_count']}/{m['denominator']} | {m['recall_at_1_iou_0_3']:.3f} | {m['recall_at_1_iou_0_5']:.3f} | {m['recall_at_1_iou_0_7']:.3f} | {m['mean_top1_iou']:.4f} |")
    lines += ["", "## Per-query results", "", "| Query | Ground truth | A | B | D | E median appearance | F appearance + repeated object | G single appearance + repeated object |", "|---|---|---|---|---|---|---|---|"]
    for row in result["queries"]:
        gt = ", ".join(f"{a:.2f}–{b:.2f}" for a,b in row["ground_truth_intervals_seconds"])
        values = []
        for key in result["condition_order"]:
            v = row["conditions"].get(key)
            values.append(f"{v['start']:.2f}–{v['end']:.2f} ({v['temporal_iou']:.3f})" if v else "—")
        lines.append(f"| {row['qid']} | {gt} | " + " | ".join(values) + " |")
    lines += [
        "", "## Interpretation and limitations", "",
        "This experiment tests a robust aggregation rule, not a trained model. In the five-query pilot, any metric change is exploratory and cannot establish general improvement. Person/attribute correctness and ownership correctness were not measured because no frame-level target boxes, tracks, or proposal-level ownership labels are available. The repeated detector rule may reduce false one-frame positives but may miss small or occluded objects; that is a sensitivity trade-off, not evidence of absence.",
        "",
        f"YOLO-only reranking time: {result['settings']['runtime_seconds_yolo_and_reranking']:.1f} s. No Moment-DETR or CLIP inference was repeated; no model was trained. Proposal intervals unchanged: {result['proposal_set_verified']}.",
        "",
        "## Reproduction",
        "",
        "Run from repository root after producing cached scores: `venv\\Scripts\\python.exe standalone_eval\\run_automated_visual_diagnostics.py --source-results outputs\\proposal_diagnosis_mot17_04_verified_20261023\\results.json --annotations evaluation_data\\MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json --video D:\\yolo_enhance\\video\\MOT17-04-SDP-raw.webm --output-dir outputs\\automated_visual_diagnostics_repro_20261023 --frames 3`, then `venv\\Scripts\\python.exe standalone_eval\\run_temporal_consensus_ablation.py --source-results outputs\\proposal_diagnosis_mot17_04_verified_20261023\\results.json --cached-scores outputs\\automated_visual_diagnostics_repro_20261023\\results.json --video D:\\yolo_enhance\\video\\MOT17-04-SDP-raw.webm --output-dir outputs\\attribute_aware_consensus_repro_20261023 --frames 3 --repeat-positive-frames 2`.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-results", default=str(DEFAULT_SOURCE))
    p.add_argument("--cached-scores", default=str(DEFAULT_CACHE))
    p.add_argument("--annotations", default=str(ROOT / "evaluation_data" / "MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json"))
    p.add_argument("--video", default=str(DEFAULT_VIDEO))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--repeat-positive-frames", type=int, default=2)
    args = p.parse_args()
    if args.frames < args.repeat_positive_frames or args.repeat_positive_frames < 1:
        raise ValueError("Frame count must be >= the positive-frame quorum >= 1")
    source_path, cache_path, annotation_path, video_path, output = [Path(v).resolve() for v in (
        args.source_results, args.cached_scores, args.annotations, args.video, args.output_dir)]
    for file in (source_path, cache_path, annotation_path, video_path):
        if not file.is_file():
            raise FileNotFoundError(file)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    gt_by_qid = {item["qid"]: item for item in annotations["queries"]}
    cache_by_qid = {item["qid"]: item for item in cached["queries"]}
    output.mkdir(parents=True)
    visual_source = cache_path.parent
    visual_target = output / "visual_diagnostics"
    visual_target.mkdir()
    for name in ("index.html", "results.json"):
        shutil.copy2(visual_source / name, visual_target / name)
    for name in ("images", "crops"):
        shutil.copytree(visual_source / name, visual_target / name)

    from ultralytics import YOLO
    model = YOLO(str(ROOT / "yolo11n.pt"))
    category_ids = {name: int(class_id) for class_id, name in model.names.items()
                    if name in SUPPORTED_CARRIED_OBJECT_CATEGORIES}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not decode video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    supported = tuple(SUPPORTED_CARRIED_OBJECT_CATEGORIES)
    start_time = time.perf_counter()
    queries = []
    proposal_set_verified = True
    condition_order = ["A_temporal_only", "B_saved_single_frame_appearance", "D_saved_person_object_fusion",
                       "E_median_multiframe_appearance", "F_median_appearance_plus_repeated_object",
                       "G_single_frame_appearance_plus_repeated_object"]

    def detections_at(timestamp, requested):
        frame_idx = max(0, int(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return {"status": "frame_unavailable", "detections": []}
        ids = [category_ids[x] for x in requested if x in category_ids]
        classes = [0, *ids]
        result = model(frame, classes=classes, verbose=False)[0]
        found = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            names = getattr(result, "names", model.names)
            for class_value, confidence in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                class_id = int(class_value)
                name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                if name != "person" and name in requested:
                    found.append({"class_name": name, "confidence": float(confidence)})
        return {"status": "detector_pass_completed", "detections": found, "frame_index": frame_idx}

    for old in source["queries"]:
        qid = old["qid"]
        cached_query = cache_by_qid[qid]
        cached_by_interval = {(c["start"], c["end"]): c for c in cached_query["all_experimental_candidates"]}
        candidates = [dict(c) for c in old["all_candidate_scores"]]
        single_frame_candidates = [dict(c) for c in candidates]
        experiment = []
        decomposition = old["query_decomposition"]
        requested = list(dict.fromkeys((*decomposition.get("carried_object_texts", []),
                                        *decomposition.get("unresolved_carried_object_texts", []))))
        for candidate in candidates:
            cache_row = cached_by_interval[(candidate["start"], candidate["end"])]
            frame_audit = cache_row["appearance_crop_frame_audit"][:args.frames]
            if len(frame_audit) != args.frames:
                raise AssertionError(f"Expected {args.frames} cached frames for {qid}")
            frame_crop_scores = [[crop.get("appearance_similarity") for crop in frame["person_crops"]]
                                 for frame in frame_audit if frame["status"] == "scored"]
            appearance = aggregate_frame_appearance(frame_crop_scores, min_observed_frames=2)
            frame_object = []
            for frame in frame_audit:
                frame_object.append(detections_at(frame["timestamp_seconds"], requested))
            object_result = aggregate_repeated_object_evidence(
                [row["detections"] for row in frame_object], requested, supported,
                min_positive_frames=args.repeat_positive_frames,
            )
            experiment.append({
                **candidate,
                "single_frame_appearance_score": candidate.get("appearance_score"),
                "appearance_score": appearance["score"],
                "appearance_consensus": appearance,
                "single_frame_carried_object_score": candidate.get("carried_object_score"),
                "carried_object_score": object_result["score"],
                "repeated_object_evidence": object_result,
                "object_frame_detection_audit": frame_object,
            })
        original_set = [(c["start"], c["end"]) for c in candidates]
        if [(c["start"], c["end"]) for c in experiment] != original_set:
            proposal_set_verified = False
            raise AssertionError("Proposal interval changed")
        ranks = {
            "A_temporal_only": fuse_candidate_ranks(experiment, "temporal"),
            "B_saved_single_frame_appearance": fuse_candidate_ranks(single_frame_candidates, "temporal_appearance"),
            "D_saved_person_object_fusion": fuse_candidate_ranks(single_frame_candidates, "temporal_appearance_person_object_carried_object"),
            "E_median_multiframe_appearance": fuse_candidate_ranks(experiment, "temporal_appearance"),
            "F_median_appearance_plus_repeated_object": fuse_candidate_ranks(experiment, "temporal_appearance_carried_object"),
            "G_single_frame_appearance_plus_repeated_object": None,
        }
        single_obj = [{**c, "appearance_score": c.get("single_frame_appearance_score")} for c in experiment]
        ranks["G_single_frame_appearance_plus_repeated_object"] = fuse_candidate_ranks(single_obj, "temporal_appearance_carried_object")
        gt = gt_by_qid[qid]["ground_truth_intervals_seconds"]
        conditions = {}
        for name, values in ranks.items():
            top = values[0]
            iou = max(temporal_iou((top["start"], top["end"]), interval) for interval in gt)
            conditions[name] = {"start": top["start"], "end": top["end"], "temporal_iou": iou,
                                "appearance_score": top.get("appearance_score"),
                                "carried_object_score": top.get("carried_object_score"),
                                "person_object_score": top.get("person_object_score"),
                                "fusion_components": top.get("fusion_components", [])}
        queries.append({
            "qid": qid, "query": old["query"], "ground_truth_intervals_seconds": gt,
            "query_decomposition": decomposition,
            "target_person_identity_status": "unverified; no per-frame identity boxes/tracks",
            "object_ownership_status": "unverified; repeated category detection is not ownership evidence",
            "proposal_count": len(candidates), "proposal_intervals_unchanged": True,
            "object_eligible_requested_categories": [x for x in requested if x in supported],
            "unsupported_requested_categories": [x for x in requested if x not in supported],
            "conditions": conditions,
            "proposal_oracle": old["proposal_oracle"],
            "all_candidates": experiment,
        })
    cap.release()
    runtime = time.perf_counter() - start_time
    metrics = {condition: summarize(queries, condition) for condition in condition_order}
    result = {
        "status": "completed", "experiment": "median frame-consensus appearance + repeated positive object category evidence",
        "proposal_source": str(source_path), "cached_clip_scores": str(cache_path),
        "video": str(video_path), "query_count": len(queries), "queries": queries,
        "condition_order": condition_order, "metrics": metrics,
        "proposal_set_verified": proposal_set_verified,
        "proposal_oracle_metrics": source["proposal_oracle_metrics"],
        "settings": {"sampled_frames_per_candidate": args.frames,
                     "appearance_aggregation": "median of per-frame maxima across detected person crops; min 2 valid frames",
                     "object_positive_quorum": f"at least {args.repeat_positive_frames} positive detector frames",
                     "missing_or_nonrepeated_object_policy": "unknown; excluded from ranking, not negative",
                     "unsupported_object_policy": "unsupported; excluded from ranking",
                     "rank_fusion": "existing unweighted mean of normalized component ranks",
                     "weights_tuned": False, "training_performed": False,
                     "same_query_denominator": True, "proposal_set_identical": proposal_set_verified,
                     "runtime_seconds_yolo_and_reranking": runtime,
                     "person_association_metric": None,
                     "reason_no_learned_fusion": "five queries from one video cannot support independent training and validation folds"},
        "interpretation": "All reported accuracy metrics are temporal localization. No person identity, attribute correctness, or object ownership metric is claimed.",
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "per_query.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["qid", "condition", "start", "end", "temporal_iou", "denominator"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in queries:
            for name in condition_order:
                value = row["conditions"][name]
                writer.writerow({"qid": row["qid"], "condition": name,
                                 "start": value["start"], "end": value["end"],
                                 "temporal_iou": value["temporal_iou"], "denominator": metrics[name]["denominator"]})
    (output / "index.html").write_text(build_html(result), encoding="utf-8")
    (output / "EXPERIMENT_REPORT.md").write_text(build_report(result), encoding="utf-8")
    (output / "config.json").write_text(json.dumps({k:v for k,v in result["settings"].items()}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "runtime_seconds": runtime,
                      "proposal_set_verified": proposal_set_verified, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
