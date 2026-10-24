"""Offline comparison of cached per-frame appearance aggregation rules."""
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

def finite_frame_maxima(candidate, cached_candidate, frame_count):
    maxima = []
    audit = cached_candidate.get("appearance_crop_frame_audit", [])[:frame_count]
    for frame in audit:
        values = []
        for crop in frame.get("person_crops", []):
            try:
                value = float(crop.get("appearance_similarity"))
            except (TypeError, ValueError):
                continue
            if value == value and abs(value) != float("inf"):
                values.append(value)
        if values:
            maxima.append(max(values))
    return maxima

def metrics(values):
    return {
        "n": len(values),
        "recall_at_1_iou_0_3": sum(x >= .3 for x in values) / len(values),
        "recall_at_1_iou_0_5": sum(x >= .5 for x in values) / len(values),
        "recall_at_1_iou_0_7": sum(x >= .7 for x in values) / len(values),
        "mean_iou": sum(values) / len(values),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-results", default=str(ROOT / "outputs/proposal_diagnosis_mot17_04_verified_20261023/results.json"))
    p.add_argument("--cached-scores", default=str(ROOT / "outputs/automated_visual_diagnostics_mot17_04_verified_20261023/results.json"))
    p.add_argument("--annotations", default=str(ROOT / "evaluation_data/MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--frames", type=int, default=3)
    args = p.parse_args()
    started = time.perf_counter()
    source_path, cache_path, ann_path, out = map(Path, (args.source_results, args.cached_scores, args.annotations, args.output_dir))
    for path in (source_path, cache_path, ann_path):
        if not path.is_file(): raise FileNotFoundError(path)
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    annotations = json.loads(ann_path.read_text(encoding="utf-8"))
    cache_queries = {q["qid"]: q for q in cached["queries"]}
    ground_truth = {q["qid"]: q["ground_truth_intervals_seconds"] for q in annotations["queries"]}
    methods = ["A_temporal_only", "D_saved_person_object_fusion", "median", "mean", "top2_mean"]
    per_query, ious = [], {method: [] for method in methods}
    for old in source["queries"]:
        qid = old["qid"]
        candidates = [dict(c) for c in old["all_candidate_scores"]]
        cached_by_interval = {(c["start"], c["end"]): c for c in cache_queries[qid]["all_experimental_candidates"]}
        variants = {"median": [], "mean": [], "top2_mean": []}
        for candidate in candidates:
            maxima = finite_frame_maxima(candidate, cached_by_interval[(candidate["start"], candidate["end"])], args.frames)
            if len(maxima) < 2:
                aggregate = {name: None for name in variants}
            else:
                aggregate = {
                    "median": statistics.median(maxima),
                    "mean": statistics.mean(maxima),
                    "top2_mean": statistics.mean(sorted(maxima, reverse=True)[:2]),
                }
            for name in variants:
                variants[name].append({**candidate, "appearance_score": aggregate[name]})
        ranksets = {
            "A_temporal_only": fuse_candidate_ranks(candidates, "temporal"),
            "D_saved_person_object_fusion": fuse_candidate_ranks(candidates, "temporal_appearance_person_object_carried_object"),
            **{name: fuse_candidate_ranks(rows, "temporal_appearance") for name, rows in variants.items()},
        }
        row = {"qid": qid, "ground_truth_intervals_seconds": ground_truth[qid], "methods": {}}
        for name, ranked in ranksets.items():
            top = ranked[0]
            iou = max(temporal_iou((top["start"], top["end"]), interval) for interval in ground_truth[qid])
            ious[name].append(iou)
            row["methods"][name] = {"start": top["start"], "end": top["end"], "temporal_iou": iou}
        per_query.append(row)
    output_metrics = {name: metrics(values) for name, values in ious.items()}
    runtime = time.perf_counter() - started
    result = {
        "status": "completed", "experiment": "cached per-frame appearance aggregation comparison",
        "query_count": len(per_query), "proposal_count_per_query": [q["candidate_count"] for q in source["queries"]],
        "same_proposals": True, "frames_used": args.frames,
        "methods": {
            "A_temporal_only": "existing temporal score rank only",
            "D_saved_person_object_fusion": "existing saved D ranking; temporal, appearance, person-object, carried-object rank percentiles",
            "median": "median of each sampled frame's maximum person-crop similarity; temporal plus appearance ranks",
            "mean": "arithmetic mean of each sampled frame's maximum person-crop similarity; temporal plus appearance ranks",
            "top2_mean": "mean of the two highest available frame maxima; temporal plus appearance ranks",
        },
        "appearance_minimum_observed_frames": 2,
        "metrics": output_metrics, "per_query": per_query, "runtime_seconds": runtime,
        "identity_attribute_ownership_metrics": "not measured; temporal IoU only",
        "training_or_model_inference": "none; cached scores and saved proposals reused",
    }
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = ["# Cached appearance aggregation comparison", "", "Offline five-query, one-video experiment. The same ten saved Moment-DETR proposals per query and cached three-frame OpenCLIP crop similarities were used. No model inference or training was run.", "", "A uses temporal ranking only. D uses the saved person-object fusion ranking. The three new variants rank candidates by the unweighted mean of normalized temporal and appearance ranks, where appearance is aggregated from the per-frame maximum crop similarity.", "", "| Method | R@.3 | R@.5 | R@.7 | Mean IoU |", "|---|---:|---:|---:|---:|"]
    for name, m in output_metrics.items(): report.append(f"| {name} | {m['recall_at_1_iou_0_3']:.3f} | {m['recall_at_1_iou_0_5']:.3f} | {m['recall_at_1_iou_0_7']:.3f} | {m['mean_iou']:.4f} |")
    report += ["", "| Query | A IoU | D IoU | Median IoU | Mean IoU | Top-two mean IoU |", "|---|---:|---:|---:|---:|---:|"]
    for row in per_query:
        m = row["methods"]; report.append("| " + row["qid"] + " | " + " | ".join(f"{m[k]['temporal_iou']:.4f}" for k in methods) + " |")
    report += ["", f"Offline execution: {runtime:.3f} seconds. Temporal IoU does not measure target identity, clothing correctness, or object ownership. All aggregate methods use rank normalization within each query; raw CLIP cosine values are not interpreted as probabilities.", "", "## Reproduction", "", f"`venv\\Scripts\\python.exe standalone_eval\\run_cached_aggregation_comparison.py --output-dir {out}`", ""]
    (out / "EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "runtime_seconds": runtime, "metrics": output_metrics}, indent=2))

if __name__ == "__main__": main()
