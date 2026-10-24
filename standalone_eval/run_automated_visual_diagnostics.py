"""Create crop-level visual diagnostics and a fixed-proposal multiframe experiment.

The experiment reuses saved Moment-DETR proposals and the existing unweighted
rank fusion. Only appearance evidence is changed: maximum CLIP similarity over
three evenly spaced frames and all detected person crops. D's object evidence
is reused from the saved one-frame run so this isolates the appearance sampling
change and avoids treating frame-local crop indices as persistent identities.
"""
import argparse
import html
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.query_logic import aggregate_similarity_scores, fuse_candidate_ranks, parse_retrieval_query
from standalone_eval.action_attribute_metrics import temporal_iou

DEFAULT_SOURCE = ROOT / "outputs" / "proposal_diagnosis_mot17_04_verified_20261023" / "results.json"
DEFAULT_ANNOTATIONS = ROOT / "evaluation_data" / "MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json"
DEFAULT_VIDEO = Path(r"D:\yolo_enhance\video\MOT17-04-SDP-raw.webm")
DEFAULT_OUTPUT = ROOT / "outputs" / "automated_visual_diagnostics_mot17_04_verified_20261023"


def summarize(values):
    vals = [float(value) for value in values if value is not None]
    return {
        "sample_count": len(vals), "eligible_qids_count": len(vals),
        "recall_at_1_iou_0_3": sum(v >= .3 for v in vals) / len(vals) if vals else None,
        "recall_at_1_iou_0_5": sum(v >= .5 for v in vals) / len(vals) if vals else None,
        "recall_at_1_iou_0_7": sum(v >= .7 for v in vals) / len(vals) if vals else None,
        "mean_top1_iou": sum(vals) / len(vals) if vals else None,
    }


def get_frame(cap, timestamp, fps):
    # Match extract_frames() in attribute_matching.py exactly (floor via int).
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(timestamp * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", default=str(DEFAULT_SOURCE))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError("The experimental aggregation requires at least two frames")
    source_path, annotation_path, video_path, output = map(
        lambda p: Path(p).resolve(),
        (args.source_results, args.annotations, args.video, args.output_dir),
    )
    for path in (source_path, annotation_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation_by_qid = {row["qid"]: row for row in annotations["queries"]}
    output.mkdir(parents=True)
    image_root = output / "images"
    crop_root = output / "crops"
    image_root.mkdir()
    crop_root.mkdir()

    # Respect local-only project dependencies and model files.
    from extension import attribute_matching as am
    from ultralytics import YOLO
    am.yolo_model = YOLO(str(ROOT / "yolo11n.pt"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    device = am.device
    started = time.perf_counter()
    rows = []
    reports = []

    def detector_frame(frame, query_parts):
        requested = tuple(dict.fromkeys((*query_parts.carried_object_texts,
                                         *query_parts.unresolved_carried_object_texts)))
        classes = [0] + [am.COCO_CARRIED_OBJECT_IDS[x] for x in requested
                          if x in am.COCO_CARRIED_OBJECT_IDS]
        result = am.yolo_model(frame, classes=classes, verbose=False)[0]
        people, objects = [], []
        annotated = frame.copy()
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box, class_value, confidence in zip(
                boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()
            ):
                class_id = int(class_value)
                names = getattr(result, "names", am.yolo_model.names)
                name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                x1, y1, x2, y2 = map(int, box)
                record = {"class_name": name, "class_id": class_id,
                          "confidence": float(confidence),
                          "box_xyxy": [float(v) for v in box]}
                if name == "person":
                    crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                    if crop.size and crop.shape[0] >= 16 and crop.shape[1] >= 16:
                        record["crop"] = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        people.append(record)
                    color = (30, 210, 30)
                else:
                    objects.append(record)
                    color = (20, 140, 240)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated, f"{name} {confidence:.2f}", (x1,max(18,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
        return people, objects, annotated

    def score_crops(people, phrases):
        prompts = list(dict.fromkeys(phrases))
        output_scores = []
        for person in people:
            vals = am.score_image_text_many(person["crop"], prompts) if prompts else []
            output_scores.append({phrase: (vals[i] if i < len(vals) else None)
                                  for i, phrase in enumerate(prompts)})
        return prompts, output_scores

    def write_diagnostic(qid, condition, item, annotation, parts):
        start, end = float(item["start"]), float(item["end"])
        timestamp = (start + end) / 2
        frame = get_frame(cap, timestamp, fps)
        if frame is None:
            return {"status": "frame_unavailable", "timestamp_seconds": timestamp}
        people, objects, annotated_frame = detector_frame(frame, parts)
        phrases = []
        if parts.appearance_text:
            phrases.append(parts.appearance_text)
        object_prompts = [f"person carrying {mention}" for mention in parts.carried_object_mentions]
        phrases.extend(object_prompts)
        unique_phrases, scores = score_crops(people, phrases)
        frame_name = f"{qid}_{condition}_t{timestamp:.2f}.jpg"
        cv2.imwrite(str(image_root / frame_name), annotated_frame)
        saved_crops = []
        selected = set()
        for m in item.get("person_object_mention_scores", []):
            index = m.get("person_crop_index")
            if isinstance(index, int):
                selected.add(index)
        crop_details = []
        for index, person in enumerate(people):
            crop_name = f"{qid}_{condition}_crop{index}.jpg"
            person["crop"].save(crop_root / crop_name, quality=90)
            saved_crops.append(crop_name)
            crop_details.append({
                "crop_index_frame_local": index,
                "box_xyxy": person["box_xyxy"],
                "detector_confidence": person["confidence"],
                "selected_by_saved_person_object_scorer": index in selected,
                "similarities": scores[index],
            })
        phrase_best_crop = {}
        for phrase in unique_phrases:
            valid = [(entry["crop_index_frame_local"], entry["similarities"].get(phrase))
                     for entry in crop_details if entry["similarities"].get(phrase) is not None]
            phrase_best_crop[phrase] = (
                {"crop_index_frame_local": max(valid, key=lambda pair: pair[1])[0],
                 "similarity": max(pair[1] for pair in valid)} if valid else None
            )
        object_details = [{k:v for k,v in item_obj.items() if k != "crop"} for item_obj in objects]
        category_map = am.COCO_CARRIED_OBJECT_IDS
        requested_categories = list(parts.carried_object_texts)
        detected_names = {obj["class_name"] for obj in object_details}
        category_status = {
            phrase: (("supported_positive_detection" if phrase in detected_names
                      else "supported_category_no_box_in_sample; not evidence of absence") if phrase in category_map
                     else "unsupported_by_YOLO; category detection unavailable")
            for phrase in requested_categories
        }
        for unsupported in parts.unresolved_carried_object_texts:
            category_status[unsupported] = "unsupported_by_YOLO; retained as text evidence only"
        return {
            "status": "ok", "condition": condition, "interval_seconds": [start,end],
            "temporal_iou": item.get("temporal_iou"), "representative_timestamp_seconds": timestamp,
            "frame_image": f"images/{frame_name}", "person_crops": saved_crops,
            "appearance_phrase": parts.appearance_text,
            "action_phrase": parts.action_text,
            "saved_full_frame_action_similarity": item.get("action_score"),
            "object_phrases_scored": object_prompts,
            "person_crop_scores": crop_details,
            "highest_scoring_crop_by_phrase": phrase_best_crop,
            "saved_scorer_selected_crop_indices_frame_local": sorted(selected),
            "object_detections": object_details,
            "object_evidence_status": category_status,
            "ownership_status": "unverified; frame co-occurrence and person-crop text similarity do not establish ownership",
        }

    for old_row in source["queries"]:
        annotation = annotation_by_qid[old_row["qid"]]
        parts = parse_retrieval_query(old_row["query"])
        candidates = [dict(candidate) for candidate in old_row["all_candidate_scores"]]
        proposal_set = [(float(c["start"]), float(c["end"])) for c in candidates]
        experiment_candidates = []
        # Record max-pool appearance over N frames and every per-frame person crop.
        for candidate in candidates:
            timestamps = np.linspace(candidate["start"], candidate["end"], args.frames).tolist()
            raw_scores, frame_audit = [], []
            phrases = [parts.appearance_text] if parts.appearance_text else []
            for timestamp in timestamps:
                frame = get_frame(cap, timestamp, fps)
                if frame is None:
                    frame_audit.append({"timestamp_seconds": timestamp, "status": "frame_unavailable",
                                        "person_crops": []})
                    continue
                people, objects, _ = detector_frame(frame, parts)
                _, crop_scores = score_crops(people, phrases)
                score_values = [row.get(parts.appearance_text) for row in crop_scores] if phrases else []
                raw_scores.extend(score_values)
                frame_audit.append({
                    "timestamp_seconds": timestamp,
                    "status": "scored" if score_values else "no_usable_person_crop",
                    "person_crops": [{"crop_index_frame_local": i,
                                      "detector_confidence": people[i]["confidence"],
                                      "box_xyxy": people[i]["box_xyxy"],
                                      "appearance_similarity": score_values[i]}
                                     for i in range(len(people))],
                })
            experiment_candidates.append({
                **candidate,
                "single_frame_appearance_score": candidate.get("appearance_score"),
                "appearance_score": aggregate_similarity_scores(raw_scores),
                "appearance_aggregation": "maximum finite CLIP cosine over all detected person crops and evenly spaced sampled frames",
                "appearance_sample_count": args.frames,
                "appearance_crop_frame_audit": frame_audit,
            })

        exp_intervals = [(float(c["start"]), float(c["end"])) for c in experiment_candidates]
        if exp_intervals != proposal_set or len(experiment_candidates) != len(candidates):
            raise AssertionError("Experimental condition changed the saved proposal set")
        ranked = {
            "A_temporal_only": fuse_candidate_ranks(experiment_candidates, "temporal"),
            "B_multiframe_appearance": fuse_candidate_ranks(experiment_candidates, "temporal_appearance"),
            "D_multiframe_appearance_plus_saved_person_object": fuse_candidate_ranks(
                experiment_candidates, "temporal_appearance_person_object_carried_object"),
        }
        gt = annotation["ground_truth_intervals_seconds"]
        cond_rows = {}
        for condition, values in ranked.items():
            top = values[0]
            top_iou = max(temporal_iou((top["start"], top["end"]), interval) for interval in gt)
            cond_rows[condition] = {"start": top["start"], "end": top["end"],
                                    "temporal_iou": top_iou,
                                    "appearance_score": top.get("appearance_score"),
                                    "person_object_score": top.get("person_object_score"),
                                    "carried_object_score": top.get("carried_object_score"),
                                    "fusion_components": top.get("fusion_components", [])}
        condition_mapping = {
            "A": old_row["conditions"]["A_temporal_only"],
            "B": old_row["conditions"]["B_temporal_plus_appearance"],
            "D": old_row["conditions"]["D_person_aware_evidence_fusion"],
        }
        diagnostic = {}
        for condition in ("A", "B", "D"):
            packed = condition_mapping[condition]
            item = next(candidate for candidate in candidates
                        if candidate["start"] == packed["start"]
                        and candidate["end"] == packed["end"])
            item = {**item, "temporal_iou": packed["temporal_iou"]}
            diagnostic[condition] = write_diagnostic(old_row["qid"], condition, item, annotation, parts)
        qid = old_row["qid"]
        per_query = {
            "qid": qid, "query": old_row["query"],
            "ground_truth_intervals_seconds": gt,
            "target_person_id_annotation": annotation.get("target_person_id_within_video"),
            "identity_verification": "unverified; annotations contain no frame-level person boxes or track IDs",
            "query_phrases": {"appearance": parts.appearance_text, "action": parts.action_text,
                              "object_mentions": list(parts.carried_object_mentions),
                              "detector_categories": list(parts.carried_object_texts),
                              "unsupported_objects": list(parts.unresolved_carried_object_texts)},
            "saved_A_B_D_diagnostics": diagnostic,
            "multiframe_variant_top1": cond_rows,
            "proposal_count": len(candidates),
            "same_proposal_intervals_across_A_B_D_and_experiment": True,
            "all_experimental_candidates": experiment_candidates,
        }
        rows.append(per_query)

    cap.release()
    qids = [row["qid"] for row in rows]
    metric_sources = {
        "A_saved": [row["saved_A_B_D_diagnostics"]["A"]["temporal_iou"] for row in rows],
        "B_saved": [row["saved_A_B_D_diagnostics"]["B"]["temporal_iou"] for row in rows],
        "D_saved": [row["saved_A_B_D_diagnostics"]["D"]["temporal_iou"] for row in rows],
        "B_multiframe_appearance": [row["multiframe_variant_top1"]["B_multiframe_appearance"]["temporal_iou"] for row in rows],
        "D_multiframe_appearance_plus_saved_person_object": [row["multiframe_variant_top1"]["D_multiframe_appearance_plus_saved_person_object"]["temporal_iou"] for row in rows],
    }
    result = {
        "status": "completed", "source_results": str(source_path), "video": str(video_path),
        "proposal_source": "saved final five-query A-D results; no new Moment-DETR inference",
        "proposal_set_verified": True, "query_count": len(rows), "eligible_qids": qids,
        "metrics_temporal_localization_only": {key: summarize(values) for key, values in metric_sources.items()},
        "person_attribute_correctness": "not measured; no frame-level target-person boxes/tracks or crop identity labels",
        "object_ownership_correctness": "not measured; visual annotations describe ownership, but detector boxes/crop similarities do not establish per-candidate ownership",
        "experimental_variant": {"name": "three-frame max-pooled appearance", "frame_count": args.frames,
                                 "aggregation": "maximum finite CLIP cosine over every detected person crop in evenly spaced proposal samples",
                                 "proposal_intervals_unchanged": True,
                                 "D_object_evidence": "reused from saved one-frame results to isolate appearance aggregation",
                                 "weights_tuned": False},
        "runtime_seconds": time.perf_counter() - started,
        "queries": rows,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    html_text = build_html(result)
    (output / "index.html").write_text(html_text, encoding="utf-8")
    (output / "AUTOMATED_VISUAL_DIAGNOSTICS.md").write_text(build_markdown(result), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "html": str(output / "index.html"),
                      "results": str(output / "results.json"), "runtime_seconds": result["runtime_seconds"],
                      "metrics": result["metrics_temporal_localization_only"]}, indent=2))


def fmt(value):
    return "—" if value is None else f"{float(value):.3f}"


def build_html(data):
    pieces = ["<!doctype html><html><head><meta charset='utf-8'><title>Automated visual diagnostics</title>",
              "<style>body{font:15px system-ui;max-width:1300px;margin:2rem auto;padding:0 1rem;color:#202124}section{border-top:2px solid #bbb;padding:1rem 0} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:1rem}.card{border:1px solid #ccc;border-radius:8px;padding:.8rem}img.full{width:100%;height:auto}img.crop{height:180px;max-width:150px;object-fit:contain;background:#eee}.cropgrid{display:flex;flex-wrap:wrap;gap:.5rem}.muted{color:#555}.warn{color:#8a3b00;background:#fff3dc;padding:.5rem}.scores{font:12px ui-monospace,monospace;white-space:pre-wrap}</style></head><body>",
              "<h1>Automated visual diagnostics</h1><p>Temporal localization is evaluated separately from person/attribute correctness. Frame-local YOLO crop indices are not identities. Object detections and CLIP similarities do not establish ownership.</p>",
              "<h2>Temporal metrics</h2><table border='1' cellpadding='6'><tr><th>Condition</th><th>n / denominator</th><th>R@.3</th><th>R@.5</th><th>R@.7</th><th>Mean IoU</th></tr>"]
    for key, metrics in data["metrics_temporal_localization_only"].items():
        pieces.append(f"<tr><td>{html.escape(key)}</td><td>{metrics['sample_count']} / {metrics['sample_count']}</td><td>{fmt(metrics['recall_at_1_iou_0_3'])}</td><td>{fmt(metrics['recall_at_1_iou_0_5'])}</td><td>{fmt(metrics['recall_at_1_iou_0_7'])}</td><td>{fmt(metrics['mean_top1_iou'])}</td></tr>")
    pieces.append("</table><p class='warn'>Person identity and object ownership are unverified. The annotations do not provide frame-level person boxes or tracks.</p>")
    for query in data["queries"]:
        pieces.append(f"<section><h2>{html.escape(query['qid'])}</h2><p><b>Query:</b> {html.escape(query['query'])}</p><p><b>Ground truth:</b> {html.escape(str(query['ground_truth_intervals_seconds']))} seconds. <b>Identity:</b> unverified.</p>")
        pieces.append("<div class='grid'>")
        for cond, diag in query["saved_A_B_D_diagnostics"].items():
            if diag.get("status") != "ok":
                pieces.append(f"<div class='card'><h3>{cond}</h3><p>Frame unavailable</p></div>")
                continue
            pieces.append(f"<div class='card'><h3>Saved {cond}: {diag['interval_seconds'][0]:.2f}–{diag['interval_seconds'][1]:.2f}s; IoU {fmt(diag['temporal_iou'])}</h3><p>Representative frame {diag['representative_timestamp_seconds']:.2f}s</p><img class='full' src='{html.escape(diag['frame_image'])}'><p>Green boxes: person. Orange boxes: requested supported object detections.</p>")
            phrase_best = html.escape(json.dumps(diag.get('highest_scoring_crop_by_phrase', {})))
            pieces.append(f"<p><b>Appearance phrase:</b> {html.escape(str(diag['appearance_phrase']))}<br><b>Action phrase (full-frame score):</b> {html.escape(str(diag.get('action_phrase')))} ({fmt(diag.get('saved_full_frame_action_similarity'))})<br><b>Scored object prompts:</b> {html.escape(', '.join(diag['object_phrases_scored']))}<br><b>Highest crop per phrase:</b> {phrase_best}<br><b>Saved scorer selected crop(s):</b> {html.escape(str(diag['saved_scorer_selected_crop_indices_frame_local']))}</p>")
            pieces.append("<div class='cropgrid'>")
            for crop, details in zip(diag["person_crops"], diag["person_crop_scores"]):
                marker = " — saved scorer selected" if details["selected_by_saved_person_object_scorer"] else ""
                score_text = "<br>".join(f"{html.escape(k)}: {fmt(v)}" for k,v in details["similarities"].items()) or "No text score"
                pieces.append(f"<div><img class='crop' src='crops/{html.escape(crop)}'><br>frame-local crop {details['crop_index_frame_local']}{marker}<div class='scores'>{score_text}</div></div>")
            pieces.append("</div><p><b>Object detections:</b> " + html.escape(json.dumps(diag['object_detections'])) + "</p><p><b>Object evidence status:</b> " + html.escape(json.dumps(diag['object_evidence_status'])) + "</p><p>Ownership unverified.</p></div>")
        pieces.append("</div><h3>Three-frame max-pooled appearance experiment</h3><table border='1' cellpadding='5'><tr><th>Condition</th><th>Top interval</th><th>IoU</th><th>Appearance</th><th>Person-object</th></tr>")
        for key, val in query["multiframe_variant_top1"].items():
            pieces.append(f"<tr><td>{html.escape(key)}</td><td>{val['start']:.2f}–{val['end']:.2f}s</td><td>{fmt(val['temporal_iou'])}</td><td>{fmt(val.get('appearance_score'))}</td><td>{fmt(val.get('person_object_score'))}</td></tr>")
        pieces.append("</table><p>Appearance aggregate is max finite CLIP similarity over detected crops in three evenly spaced frames. Crop indices are frame-local. D reuses its saved one-frame object evidence.</p></section>")
    pieces.append("<footer><p>Machine-readable details: <a href='results.json'>results.json</a></p></footer></body></html>")
    return "\n".join(pieces)


def build_markdown(data):
    lines = [
        "# Automated Visual Diagnostics",
        "",
        "This report uses the saved five-query Moment-DETR proposals and local MOT17-04 video. It does not infer person identity or object ownership from model predictions.",
        "",
        "## Temporal localization metrics",
        "",
        "Recall denominators are the eligible queries shown per condition; all current conditions include the same five query IDs.",
        "",
        "| Condition | n / denominator | Recall@.3 | Recall@.5 | Recall@.7 | Mean IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, metric in data["metrics_temporal_localization_only"].items():
        lines.append(f"| {key} | {metric['sample_count']}/{metric['eligible_qids_count']} | {fmt(metric['recall_at_1_iou_0_3'])} | {fmt(metric['recall_at_1_iou_0_5'])} | {fmt(metric['recall_at_1_iou_0_7'])} | {fmt(metric['mean_top1_iou'])} |")
    lines += [
        "",
        "## Per-query top-1 results",
        "",
        "| Query | GT interval | Saved A (IoU) | Saved B (IoU) | Saved D (IoU) | B three-frame (IoU) | D three-frame appearance (IoU) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in data["queries"]:
        gt = ", ".join(f"{a:.2f}–{b:.2f}" for a,b in row["ground_truth_intervals_seconds"])
        saved = row["saved_A_B_D_diagnostics"]
        vals = []
        for key in ("A", "B", "D"):
            d = saved[key]
            vals.append(f"{d['interval_seconds'][0]:.2f}–{d['interval_seconds'][1]:.2f} ({fmt(d['temporal_iou'])})")
        exp = row["multiframe_variant_top1"]
        b = exp["B_multiframe_appearance"]
        d = exp["D_multiframe_appearance_plus_saved_person_object"]
        lines.append(f"| {row['qid']} | {gt} | " + " | ".join(vals) + f" | {b['start']:.2f}–{b['end']:.2f} ({fmt(b['temporal_iou'])}) | {d['start']:.2f}–{d['end']:.2f} ({fmt(d['temporal_iou'])}) |")
    lines += [
        "",
        "## What the scores establish",
        "",
        "- Temporal IoU and Recall@1 measure moment localization only.",
        "- Crop-level CLIP values are raw cosine similarities to the exact phrases listed in the HTML and JSON; they are not calibrated correctness probabilities.",
        "- q1/q2 list every detected person crop in the representative saved D frame, phrase-specific scores, and the crop selected by the saved object-phrase scorer.",
        "- Action similarity is the saved full-frame score for its exact parsed action phrase; appearance and object phrase similarities are per-person-crop scores.",
        "- YOLO boxes and confidences are category detections only. Missing detections are unknown/non-observation; unsupported categories are shown explicitly.",
        "- Person identity and object ownership are not evaluated. The annotations contain no per-frame target boxes or track IDs, and frame co-occurrence is not ownership evidence.",
        "- The multiframe variant max-pools appearance over three evenly spaced samples and every person crop. It reuses the exact saved proposal intervals; D also reuses saved one-frame object evidence. No weights were tuned.",
        "",
        "## Artifacts and reproducibility",
        "",
        "- Browse `index.html`; all diagnostic images and crops are local relative files.",
        "- `results.json` contains per-query, per-proposal, per-frame, and per-crop scores.",
        "- Run from repository root: `venv\\Scripts\\python.exe standalone_eval\\run_automated_visual_diagnostics.py --source-results outputs\\proposal_diagnosis_mot17_04_verified_20261023\\results.json --video D:\\yolo_enhance\\video\\MOT17-04-SDP-raw.webm --output-dir outputs\\automated_visual_diagnostics_mot17_04_verified_20261023 --frames 3`.",
        f"- Elapsed scoring/report time: {data['runtime_seconds']:.1f} seconds.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
